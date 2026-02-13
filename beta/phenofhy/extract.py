import pandas as pd
# dxdata/dxpy may be used transitively by your environment/tools; keep imports if needed.
import dxdata  # noqa: F401
import dxpy    # noqa: F401
import os
import logging
import subprocess
import json
import shutil
import sys
import tempfile
from pathlib import Path

# --- CONFIG ---
from .utils import find_latest_dx_file_id

CONFIG_FILENAME = "config.json"
CONFIG_FILE_ID = find_latest_dx_file_id(name_pattern=CONFIG_FILENAME)
HELPERS_DIR = Path("/mnt/project/helpers")

# --- LOGGING ---
logging.basicConfig(level=logging.INFO)

# --- FILE LOADER WITH LOCAL FALLBACK ---
def load_or_download_file(file_path: Path, file_id: str, description: str = "", validate_json=False):
    """Load a file locally or download it from DNAnexus.

    Args:
        file_path: Expected local path.
        file_id: DNAnexus file id to download if missing.
        description: Human-readable description for logging.
        validate_json: If True, require file to be valid JSON.

    Returns:
        Path to the resolved local file.

    Raises:
        FileNotFoundError: If download fails.
    """
    def is_valid_json(path: Path):
        try:
            with open(path, "r") as f:
                json.load(f)
            return True
        except Exception as e:
            if not isinstance(e, FileNotFoundError):
                logging.warning(f"Invalid JSON in {path}: {e}")
            return False

    if file_path.exists() and (not validate_json or is_valid_json(file_path)):
        logging.info(f"Using existing {description}: {file_path}")
        return file_path

    project_root = Path("/mnt/project").resolve()
    try:
        relative_path = file_path.resolve().relative_to(project_root)
    except ValueError:
        relative_path = file_path.name

    fallback_path = Path(".") / relative_path

    if fallback_path.exists() and (not validate_json or is_valid_json(fallback_path)):
        logging.info(f"Using existing local fallback: {fallback_path}")
        return fallback_path

    fallback_path.parent.mkdir(parents=True, exist_ok=True)
    logging.info(f"{description} not found or invalid. Downloading to {fallback_path}...")

    result = os.system(f"dx download {file_id} -o {fallback_path} --overwrite")
    if result != 0 or not fallback_path.exists():
        raise FileNotFoundError(f"Failed to download {description} ({file_id})")

    logging.info(f"Downloaded {description} to {fallback_path}")
    return fallback_path

# --- HELPER TO RESOLVE FILE PATHS FROM CONFIG ---
def resolve_path(config, file_info):
    """Resolve a file path from config metadata.

    Args:
        config: Parsed config dictionary.
        file_info: File info dict with BASE and FILENAME keys.

    Returns:
        Resolved Path for the file.
    """
    base_key = file_info["BASE"]
    parts = base_key.split(".")
    base_path = config["BASE_PATHS"]
    for part in parts:
        base_path = base_path[part]
    return Path(config["PROJECT_DIR_PATH"]) / base_path / file_info["FILENAME"]

def get_file(file_info, config):
    """Resolve and load a file described in config.

    Args:
        file_info: File metadata from config.
        config: Parsed config dictionary.

    Returns:
        Path to the resolved local file.
    """
    file_path = resolve_path(config, file_info)
    return load_or_download_file(file_path, file_info["ID"], file_info["FILENAME"])

# --- EXTRACT FIELDS USING DX ---
def _extract_fields(dataset_id, field_list_path: Path, output_file: str, sql_only: bool = False):
    """Extract fields from a DNAnexus dataset or write SQL.

    Args:
        dataset_id: DNAnexus dataset id.
        field_list_path: CSV path with columns ['entity', 'name'].
        output_file: Output file path for CSV or SQL.
        sql_only: If True, output SQL instead of extracting data.
    """
    try:
        df = pd.read_csv(field_list_path)

        # Standardize expected columns
        if "name" not in df.columns and "coding_name" in df.columns:
            logging.warning("'name' column missing — deriving from 'coding_name'")
            df["name"] = df["coding_name"]

        if not {"entity", "name"}.issubset(df.columns):
            raise ValueError("Input file must contain 'entity' and 'name' columns.")

        # Trim, drop empties, dedupe
        df["entity"] = df["entity"].astype(str).str.strip()
        df["name"] = df["name"].astype(str).str.strip()
        df = df.replace({"": pd.NA}).dropna(subset=["entity", "name"])
        df = df.drop_duplicates(subset=["entity", "name"])

        # Build field list for dx
        field_names = ",".join(f"{e}.{n}" for e, n in zip(df["entity"], df["name"]))

        if sql_only:
            sql_file = "extracted_query.sql"
            cmd = [
                "dx", "extract_dataset",
                dataset_id,
                "--fields", field_names,
                "--output", sql_file,
                "--sql",
            ]
            subprocess.check_call(cmd)
            logging.info("SQL query generated and saved to %s", sql_file)

            Path(output_file).parent.mkdir(parents=True, exist_ok=True)
            shutil.move(sql_file, output_file)
            logging.info("SQL file moved to %s", output_file)
            return

        temp_output = "temp_extracted_data.csv"
        cmd_extract = [
            "dx", "extract_dataset",
            dataset_id,
            "--fields", field_names,
            "--delimiter", ",",
            "--output", temp_output,
        ]
        subprocess.check_call(cmd_extract)

        pd.read_csv(temp_output).to_csv(output_file, index=False)
        os.remove(temp_output)
        logging.info("Dataset extracted and saved to %s", output_file)

    except Exception as e:
        logging.error(f"Failed to extract dataset: {e}")
        raise

# ---------- SQL → pandas / Spark ----------

def _read_sql_string(sql_or_path: str | Path) -> str:
    """Read SQL from a file or return the string as-is.

    Args:
        sql_or_path: SQL string or path to a .sql file.

    Returns:
        SQL string without trailing semicolon.
    """
    sql = Path(sql_or_path).read_text(encoding="utf-8") if Path(str(sql_or_path)).exists() else str(sql_or_path)
    sql = sql.strip()
    if sql.endswith(";"):
        sql = sql[:-1].rstrip()
    return sql

def pandas_from_sql(sql_or_path, *, downcast_floats=True, downcast_ints=True):
    """Execute SQL in Spark and collect in partitions to pandas.

    Args:
        sql_or_path: SQL string or .sql file path.
        downcast_floats: If True, downcast float columns.
        downcast_ints: If True, downcast integer columns.

    Returns:
        Pandas DataFrame containing the SQL results.
    """
    from pyspark.sql import SparkSession, functions as F

    sql = _read_sql_string(sql_or_path)
    spark = SparkSession.builder.getOrCreate()
    spark.conf.set("spark.sql.execution.arrow.pyspark.enabled", "true")

    sdf = spark.sql(sql)

    # Use current partitioning; DO NOT repartition (no shuffle)
    pid_col = F.spark_partition_id()
    nparts = sdf.rdd.getNumPartitions()

    def _collect_one(df_spark):
        pdf = df_spark.toPandas()
        if downcast_ints:
            for c in pdf.select_dtypes(include=["int64", "int32"]).columns:
                pdf[c] = pd.to_numeric(pdf[c], downcast="integer")
        if downcast_floats:
            for c in pdf.select_dtypes(include=["float64"]).columns:
                pdf[c] = pd.to_numeric(pdf[c], downcast="float")
        for c in pdf.select_dtypes(include=["bool"]).columns:
            pdf[c] = pdf[c].astype("boolean")
        return pdf

    chunks = []
    for pid in range(nparts):
        part = sdf.where(pid_col == pid)
        chunks.append(_collect_one(part))

    return pd.concat(chunks, ignore_index=True)

def sql_to_pandas(sql_or_path, *, try_fast_first=True):
    """Execute SQL via Spark and return a pandas DataFrame.

    Args:
        sql_or_path: SQL string or .sql file path.
        try_fast_first: If True, try direct toPandas() first.

    Returns:
        Pandas DataFrame with SQL results.
    """
    from pyspark.sql import SparkSession
    sql = _read_sql_string(sql_or_path)

    spark = SparkSession.builder.getOrCreate()
    spark.conf.set("spark.sql.execution.arrow.pyspark.enabled", "true")
    sdf = spark.sql(sql)

    if try_fast_first:
        try:
            return sdf.toPandas()  # fastest when it fits
        except Exception:
            pass  # fall back on any Arrow/OOM error

    # Fallback without shuffle
    return pandas_from_sql(sql_or_path)

def sql_to_spark(sql_or_path: str | Path):
    """Return a Spark DataFrame for the given SQL.

    Args:
        sql_or_path: SQL string or .sql file path.

    Returns:
        Spark DataFrame for the SQL query.
    """
    from pyspark.sql import SparkSession
    sql = _read_sql_string(sql_or_path)
    spark = SparkSession.builder.getOrCreate()
    return spark.sql(sql)

# --- MAIN EXECUTION LOGIC ---
def run_extraction(
    output_path: Path,
    input_file=None,
    *,
    fields: list[str] | None = None,
    fields_dict: dict[str, str] | None = None,
    dataset_id_override=None,
    sql_only: bool = False,
    cohort_key: str = "TEST_COHORT_ID",
):
    """Run extraction using a file list or inline field definitions.

    Args:
        output_path: Output path for CSV or SQL.
        input_file: Path to list file or config key.
        fields: List of "entity.field" strings.
        fields_dict: Dict of entity -> field.
        dataset_id_override: Optional dataset id override.
        sql_only: If True, output SQL instead of extracting data.
        cohort_key: Config cohort key to resolve dataset id.

    Raises:
        ValueError: If input_file and fields/fields_dict are both provided.
        KeyError: If required config keys are missing.
    """
    # prevent ambiguous inputs
    if (fields or fields_dict) and input_file is not None:
        raise ValueError("Pass either fields/fields_dict OR input_file, not both.")

    config_path = load_or_download_file(HELPERS_DIR / CONFIG_FILENAME, CONFIG_FILE_ID, "Config file", validate_json=True)
    with open(config_path, "r") as f:
        config = json.load(f)

    tmp_csv_path = None
    try:
        # Build or resolve phenotype list file
        if fields or fields_dict:
            rows = []
            if fields:
                for item in fields:
                    if "." not in item:
                        raise ValueError(f"Expected 'entity.field' format, got: {item}")
                    entity, field = item.split(".", 1)
                    rows.append({"entity": entity, "name": field, "coding_name": field})
            if fields_dict:
                for entity, field in fields_dict.items():
                    rows.append({"entity": entity, "name": field, "coding_name": field})
            df = pd.DataFrame(rows, columns=["entity", "name", "coding_name"])

            # Normalize to CSV for _extract_fields
            tmp = tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False, newline="")
            df.to_csv(tmp.name, index=False)
            tmp.close()
            tmp_csv_path = tmp.name
            pheno_list_file = Path(tmp_csv_path)

        else:
            # Existing file-handling logic; normalize non-CSV to CSV for _extract_fields
            if isinstance(input_file, str) and input_file.endswith((".csv", ".tsv", ".xlsx")):
                in_path = Path(input_file)
                if in_path.suffix.lower() == ".csv":
                    pheno_list_file = in_path
                else:
                    if in_path.suffix.lower() == ".tsv":
                        df = pd.read_csv(in_path, sep="\t")
                    else:  # .xlsx
                        df = pd.read_excel(in_path)
                    tmp = tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False, newline="")
                    df.to_csv(tmp.name, index=False)
                    tmp.close()
                    tmp_csv_path = tmp.name
                    pheno_list_file = Path(tmp_csv_path)
            else:
                pheno_info = config["FILES"]["PHENOTYPE_FILES"].get(input_file)
                if not pheno_info:
                    available = ", ".join(config["FILES"]["PHENOTYPE_FILES"].keys())
                    raise KeyError(f"phenotype_key '{input_file}' not found in config. Available: {available}")
                pheno_path = resolve_path(config, pheno_info)
                pheno_list_file = load_or_download_file(pheno_path, pheno_info["ID"], pheno_info["FILENAME"])

        # Resolve dataset id
        if dataset_id_override:
            dataset_id = dataset_id_override
        else:
            try:
                dataset_id = config["COHORTS"][cohort_key]
            except KeyError:
                available = ", ".join(config["COHORTS"].keys())
                raise KeyError(f"Cohort key '{cohort_key}' not found. Available keys: {available}")

        # Ensure output directory
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Clearer sql_only warning (suffix-insensitive)
        if sql_only and not str(output_path).lower().endswith(".sql"):
            logging.warning("sql_only=True but output_file does not end with .sql; you may be writing SQL into a non-SQL file.")

        # Execute extraction
        _extract_fields(
            dataset_id,
            pheno_list_file,
            str(output_path),
            sql_only=sql_only,
        )
        logging.info(f"Output saved to {output_path.resolve()}")

    finally:
        # Best-effort cleanup of temp CSV (if created)
        if tmp_csv_path:
            try:
                Path(tmp_csv_path).unlink(missing_ok=True)
            except Exception:
                logging.debug(f"Could not remove temp file: {tmp_csv_path}")

# --- PUBLIC ENTRYPOINT FOR NOTEBOOK OR IMPORT ---
def fields(
    output_file: str = "outputs/pilot_phenotypes_raw_values.csv",
    input_file: str | None = "PILOT_PHENOTYPES",
    cohort_key: str = "TEST_COHORT_ID",
    dataset_id: str | None = None,
    sql_only: bool = False,
    *,
    fields: list[str] | None = None,
    fields_dict: dict[str, str] | None = None,
):
    """Extract phenotype values or generate SQL queries.

    Args:
        output_file: Output file path for CSV or SQL.
        input_file: Path to list file or config key.
        cohort_key: Config cohort key to resolve dataset id.
        dataset_id: Optional dataset id override.
        sql_only: If True, output SQL instead of extracting data.
        fields: List of "entity.field" strings.
        fields_dict: Dict of entity -> field.
    """
    run_extraction(
        Path(output_file),
        input_file=input_file,
        fields=fields,
        fields_dict=fields_dict,
        dataset_id_override=dataset_id,
        sql_only=sql_only,
        cohort_key=cohort_key,
    )

# --- CLI ENTRY POINT ---
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(
        description="Extract phenotype values or SQL from DNAnexus using config-defined field list or inline fields."
    )
    parser.add_argument("--input", type=str, default="PILOT_PHENOTYPES",
                        help="Config key under FILES.PHENOTYPE_FILES or a path to a .csv/.tsv/.xlsx file.")
    parser.add_argument("--output", type=str, default="outputs/raw/pilot_phenotypes_raw_values.csv")
    parser.add_argument("--cohort", type=str, default="TEST_COHORT_ID")
    parser.add_argument("--dataset", type=str, default=None)
    parser.add_argument("--sql-only", action="store_true")
    # New inputs:
    parser.add_argument("--field", action="append",
                        help="Field in 'entity.field' form. Repeat this flag for multiple fields.")
    parser.add_argument("--fields-dict", type=str,
                        help="JSON mapping of entity -> field, e.g. '{\"participant\": \"id\", \"questionnaire\": \"demog_gender_2_1\"}'")

    args = parser.parse_args()

    # Parse fields_dict JSON if supplied
    fields_dict_arg = None
    if args.fields_dict:
        try:
            fields_dict_arg = json.loads(args.fields_dict)
            if not isinstance(fields_dict_arg, dict):
                raise ValueError("fields-dict must parse to a JSON object")
        except Exception as e:
            logging.error(f"Invalid --fields-dict JSON: {e}")
            sys.exit(2)

    # If any of the new inputs are provided, we can ignore --input unless you want to enforce exclusivity.
    if (args.field or fields_dict_arg) and args.input != "PILOT_PHENOTYPES":
        logging.info("Ignoring --input because --field/--fields-dict provided.")

    fields(
        output_file=args.output,
        input_file=None if (args.field or fields_dict_arg) else args.input,
        cohort_key=args.cohort,
        dataset_id=args.dataset,
        sql_only=args.sql_only,
        fields=args.field,
        fields_dict=fields_dict_arg,
    )
