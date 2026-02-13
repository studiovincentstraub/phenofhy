import os
import json
import logging
import subprocess
from pathlib import Path
from typing import Dict

import pandas as pd

# --- Local imports ---
from .utils import find_latest_dx_file_id, get_dataset_id, load_file

# --- CONFIG ---
CONFIG_FILENAME = "config.json"
CONFIG_FILE_ID = find_latest_dx_file_id(name_pattern=CONFIG_FILENAME)
HELPERS_DIR = Path("/mnt/project/helpers")
METADATA_DIR = Path("./metadata")

logger = logging.getLogger(__name__)


# ----------------------------
# Public API
# ----------------------------
def metadata() -> Dict[str, pd.DataFrame]:
    """
    Ensure metadata dictionary files are downloaded, then load them into DataFrames.

    Returns
    -------
    dict[str, pd.DataFrame]
        Keys:
          - 'codings'
          - 'data_dictionary'
          - 'entity_dictionary'
    """
    files = _download_metadata_files()  # {'codings': Path, 'data_dictionary': Path, 'entity_dictionary': Path}
    dfs: Dict[str, pd.DataFrame] = {}

    for key, path in files.items():
        if path is None:
            raise RuntimeError(f"Expected {key} file is missing (None returned).")
        if not Path(path).exists():
            raise FileNotFoundError(f"File for {key} not found at {path}")
        df = load_file(path)
        if df is None:
            raise RuntimeError(f"Could not load {key} from {path}")
        dfs[key] = df

    return dfs


def _download_metadata_files() -> Dict[str, Path]:
    """
    Download the codings, data and entity dictionary assets into ./metadata/.

    Returns
    -------
    dict[str, Path]
        {
            "codings": Path,
            "data_dictionary": Path,
            "entity_dictionary": Path,
        }
    """
    # 1) Load project config
    try:
        cfg_target = HELPERS_DIR / CONFIG_FILENAME  # for clearer error messages
        config_path = _load_or_download_file(
            cfg_target,
            CONFIG_FILE_ID,
            description="Config file",
            config_mode=True,
        )
        with open(config_path, "r") as f:
            config = json.load(f)
    except Exception as e:
        raise RuntimeError(f"Failed to load project config: {e}")

    # 2) Resolve file infos from config
    try:
        codings_info = config["FILES"]["CODINGS"]
        data_dict_info = config["FILES"]["DATA_DICT"]
        entity_dict_info = config["FILES"]["ENTITY_DICT"]

        codings_src = resolve_path(config, codings_info)
        data_dict_src = resolve_path(config, data_dict_info)
        entity_dict_src = resolve_path(config, entity_dict_info)
    except Exception as e:
        raise RuntimeError(f"Config does not contain expected metadata entries: {e}")

    # 3) Fetch into ./metadata (force destination)
    fetched: Dict[str, Path] = {"codings": None, "data_dictionary": None, "entity_dictionary": None}  # type: ignore
    file_info_map = {
        "codings": codings_info,
        "data_dictionary": data_dict_info,
        "entity_dictionary": entity_dict_info,
    }

    try:
        fetched["codings"] = Path(
            _load_or_download_file(
                codings_src,
                codings_info["ID"],
                description=codings_info.get("FILENAME", "Codings"),
                dest_dir=METADATA_DIR,
            )
        )
    except Exception as e:
        logger.info(f"Fetching codings failed: {e}")

    try:
        fetched["data_dictionary"] = Path(
            _load_or_download_file(
                data_dict_src,
                data_dict_info["ID"],
                description=data_dict_info.get("FILENAME", "Data dictionary"),
                dest_dir=METADATA_DIR,
            )
        )
    except Exception as e:
        logger.info(f"Fetching data_dictionary failed: {e}")

    try:
        fetched["entity_dictionary"] = Path(
            _load_or_download_file(
                entity_dict_src,
                entity_dict_info["ID"],
                description=entity_dict_info.get("FILENAME", "Entity dictionary"),
                dest_dir=METADATA_DIR,
            )
        )
    except Exception as e:
        logger.info(f"Fetching entity_dictionary failed: {e}")

    # 4) Fallback to `dx extract_dataset -ddd` if any missing
    if any(v is None for v in fetched.values()):
        try:
            _run_dx_dump_dictionary(METADATA_DIR)
        except Exception as e:
            # If everything failed, surface the raw error; if partial success, raise a clearer message.
            if all(v is None for v in fetched.values()):
                raise
            raise RuntimeError(f"Partial failure: could not dump missing dictionary files via dx: {e}")

        dumped = _find_dumped_dictionary_files(METADATA_DIR)
        for key in ("codings", "data_dictionary", "entity_dictionary"):
            dumped_path = dumped.get(key)
            if fetched[key] is None:
                if dumped_path is None:
                    continue
                desired_name = file_info_map[key].get("FILENAME", dumped_path.name)
                desired_path = METADATA_DIR / desired_name
                if dumped_path != desired_path:
                    if desired_path.exists():
                        desired_path.unlink()
                    dumped_path.replace(desired_path)
                fetched[key] = desired_path
            elif dumped_path is not None and dumped_path.exists() and Path(fetched[key]) != dumped_path:
                dumped_path.unlink()

    # 5) Sanity: ensure presence (dx may omit empties by design; tighten or relax policy as needed)
    if any(fetched[k] is None for k in fetched):
        missing = [k for k in fetched if fetched[k] is None]
        raise RuntimeError(
            "The following metadata files could not be obtained (they may be empty in the dataset): "
            + ", ".join(missing)
        )

    return fetched  # type: ignore[return-value]


def field_list(
    fields=None,
    output_file: str | None = None,
    fields_list_name: str | None = None,
    input_file=None,                    # backward-compat alias
    input_file_name: str | None = None  # backward-compat alias
) -> pd.DataFrame | None:
    """
    Process a phenotype list (file path, file ID, list of fields, or dict of entity→field)
    into a merged metadata table.

    Parameters
    ----------
    fields : list[str] | dict[str, str] | str | None
        - list form: ["entity.field", ...]
        - dict form: {"entity": "field", ...}
        - str: path/ID/config key (legacy file behavior)
    output_file : str | None
        If provided, write CSV to this path and return None; otherwise return the DataFrame.
    fields_list_name : str | None
        Optional filename to use when downloading a direct file ID (new-style alias).
    input_file : any
        Backwards-compatible alias for `fields`. If `fields` is None, this is used.
    input_file_name : str | None
        Backwards-compatible alias for `fields_list_name`.

    Returns
    -------
    pd.DataFrame | None
        The processed/merged metadata table, or None if `output_file` is provided.
    """
    # --- Backward-compatibility shims
    if fields is None and input_file is not None:
        fields = input_file
    if fields_list_name is None and input_file_name is not None:
        fields_list_name = input_file_name

    # 1) Load config
    try:
        cfg_target = HELPERS_DIR / CONFIG_FILENAME
        config_path = _load_or_download_file(cfg_target, CONFIG_FILE_ID, "Config file", config_mode=True)
        with open(config_path, "r") as f:
            config = json.load(f)
    except Exception as e:
        raise RuntimeError(f"Failed to load config file from {cfg_target}: {e}")

    # 2) Load phenotype input
    try:
        if isinstance(fields, list):
            # Case: ["entity.field", ...]
            rows = []
            for item in fields:
                if "." not in item:
                    raise ValueError(f"Expected 'entity.field' format, got: {item}")
                entity, field = item.split(".", 1)
                rows.append({"entity": entity, "name": field, "coding_name": field})
            raw_pheno_df = pd.DataFrame(rows)

        elif isinstance(fields, dict):
            # Case: {"entity": "field", ...}
            rows = [{"entity": k, "name": v, "coding_name": v} for k, v in fields.items()]
            raw_pheno_df = pd.DataFrame(rows)

        else:
            # Legacy file-handling logic
            if fields == "-":
                pheno_info = config["FILES"]["PHENOTYPE_FILES"]["PILOT_PHENOTYPES"]
                pheno_file_path = resolve_path(config, pheno_info)
                pheno_file_path = _load_or_download_file(pheno_file_path, pheno_info["ID"], pheno_info["FILENAME"])
            elif isinstance(fields, str) and fields.endswith((".csv", ".tsv", ".xlsx")):
                pheno_file_path = Path(fields)
            elif isinstance(fields, str) and fields in config["FILES"]["PHENOTYPE_FILES"]:
                pheno_info = config["FILES"]["PHENOTYPE_FILES"][fields]
                pheno_file_path = resolve_path(config, pheno_info)
                pheno_file_path = _load_or_download_file(pheno_file_path, pheno_info["ID"], pheno_info["FILENAME"])
            elif isinstance(fields, str) and (fields.startswith("file-") or ":file-" in fields):
                fname = fields_list_name or f"{fields}.csv"
                pheno_file_path = _load_or_download_file(Path("inputs") / fname, fields, "Direct file ID")
            else:
                raise ValueError("Unsupported fields list format.")
            raw_pheno_df = load_file(pheno_file_path)

    except Exception as e:
        raise RuntimeError(f"Failed to resolve or load input: {e}")

    # 3) Get metadata dictionary files & load them
    coding_file = get_file(config["FILES"]["CODINGS"], config)
    data_dict_file = get_file(config["FILES"]["DATA_DICT"], config)

    coding_df = load_file(coding_file)
    data_dict_df = load_file(data_dict_file)

    # 4) Clean strings
    raw_pheno_df = strip_strings(raw_pheno_df)
    coding_df = strip_strings(coding_df)
    data_dict_df = strip_strings(data_dict_df)

    # 5) Determine merge keys
    if "phenotype" in raw_pheno_df.columns:
        merge_keys_left = ["coding_name", "phenotype"]
        merge_keys_right = ["coding_name", "meaning"]
    elif "trait" in raw_pheno_df.columns:
        merge_keys_left = ["coding_name", "trait"]
        merge_keys_right = ["coding_name", "meaning"]
    else:
        logger.warning("No 'phenotype' or 'trait' column found — merging only on 'coding_name'.")
        merge_keys_left = ["coding_name"]
        merge_keys_right = ["coding_name"]

    # 6) Merge with codings
    intermed_df = pd.merge(
        raw_pheno_df,
        coding_df,
        how="left",
        left_on=merge_keys_left,
        right_on=merge_keys_right,
        suffixes=("", "_from_coding"),
    )

    # 7) Fill codes, tidy types
    if "code_from_coding" in intermed_df.columns:
        intermed_df["code"] = intermed_df["code"].fillna(intermed_df["code_from_coding"]).infer_objects(copy=False)
    else:
        intermed_df["code"] = intermed_df["code"].infer_objects(copy=False)

    intermed_df["code"] = intermed_df["code"].apply(float_to_int_if_possible)

    # Drop redundant columns from codings
    intermed_df.drop(
        columns=[c for c in ["code_from_coding", "meaning", "concept", "display_order"] if c in intermed_df.columns],
        inplace=True,
    )

    # 8) Ensure we have a 'name' column for the dictionary merge
    if "name" not in intermed_df.columns and "coding_name" in intermed_df.columns:
        logger.warning("'name' column not found — deriving from 'coding_name'.")
        intermed_df["name"] = intermed_df["coding_name"].str.lower()

    # Strip all column names to avoid whitespace issues
    intermed_df.columns = intermed_df.columns.str.strip()
    data_dict_df.columns = data_dict_df.columns.str.strip()

    # Rename coding_name in data_dict_df if present to avoid collision
    if "coding_name" in data_dict_df.columns:
        logger.info("Renaming 'coding_name' in data_dict_df to avoid duplication.")
        data_dict_df = data_dict_df.rename(columns={"coding_name": "coding_name_from_dict"})

    # 9) Merge with the data dictionary
    processed_df = pd.merge(
        intermed_df,
        data_dict_df,
        how="left",
        on=["name", "entity"],
        suffixes=("", "_from_phenofhy_dict"),
    )

    # 10) Validate row counts when merge was constrained by phenotype/trait
    if set(merge_keys_left) >= {"coding_name", "phenotype"} or set(merge_keys_left) >= {"coding_name", "trait"}:
        assert len(processed_df) == len(raw_pheno_df), (
            f"Row count changed during merge! Original: {len(raw_pheno_df)}, After merge: {len(processed_df)}"
        )
    else:
        logger.warning("Row count may increase due to join on 'coding_name' only (no phenotype/trait column).")

    # 11) Optionally write output; return DataFrame or None
    if output_file:
        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        processed_df.to_csv(output_path, index=False)
        logger.info(f"Output saved to {output_path.resolve()}")
        return None
    else:
        return processed_df


# ----------------------------
# Internal utilities
# ----------------------------
def _clean_path_str(s: str) -> str:
    return str(s).strip().replace("\u00a0", " ")


# replace resolve_path with a sanitized version
def resolve_path(config: dict, file_info: dict) -> Path:
    base_key = _clean_path_str(file_info["BASE"])
    parts = [p.strip() for p in base_key.split(".")]

    node = config["BASE_PATHS"]
    for part in parts:
        node = node[part]  # your config already stores leaf values as full paths

    project_dir = Path(_clean_path_str(config["PROJECT_DIR_PATH"])).resolve()
    filename = _clean_path_str(file_info["FILENAME"])
    return project_dir / Path(_clean_path_str(node)) / filename


def get_file(file_info: dict, config: dict) -> Path:
    """Fetch a file defined in config FILES section and return the local Path."""
    file_path = resolve_path(config, file_info)
    return _load_or_download_file(file_path, file_info["ID"], file_info.get("FILENAME", file_path.name))


def float_to_int_if_possible(val):
    try:
        float_val = float(val)
        int_val = int(float_val)
        return int_val if float_val == int_val else float_val
    except (ValueError, TypeError):
        return val


def strip_strings(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for col in out.columns:
        if out[col].dtype == "object":
            out[col] = out[col].apply(lambda v: v.strip() if isinstance(v, str) else v)
    return out


def _run_dx_dump_dictionary(outdir: Path) -> None:
    """Run `dx extract_dataset -ddd` into outdir using utils.get_dataset_id()."""
    outdir.mkdir(parents=True, exist_ok=True)
    dataset_id = get_dataset_id()  # e.g. "project-xxxx:record-xxxx"
    try:
        subprocess.run(
            ["dx", "extract_dataset", dataset_id, "-ddd", "-o", str(outdir)],
            check=True,
            capture_output=True,
            text=True,
        )
    except FileNotFoundError as e:
        raise RuntimeError("`dx` CLI not found on PATH. Please install/activate DNAnexus dx-toolkit.") from e
    except subprocess.CalledProcessError as e:
        raise RuntimeError(
            f"dx extract_dataset failed (exit={e.returncode}).\nSTDOUT:\n{e.stdout}\nSTDERR:\n{e.stderr}"
        ) from e


def _find_dumped_dictionary_files(outdir: Path) -> Dict[str, Path | None]:
    """Find the three dictionary files produced by -ddd. Returns a dict of Paths (or None if missing)."""
    codings = next(outdir.glob("*.codings.csv"), None)
    data_dictionary = next(outdir.glob("*.data_dictionary.csv"), None)
    entity_dictionary = next(outdir.glob("*.entity_dictionary.csv"), None)
    return {
        "codings": Path(codings) if codings is not None else None,
        "data_dictionary": Path(data_dictionary) if data_dictionary is not None else None,
        "entity_dictionary": Path(entity_dictionary) if entity_dictionary is not None else None,
    }


def _load_or_download_file(
    file_path,
    file_id,
    description: str = "",
    validate_json: bool = False,
    config_mode: bool = False,
    dest_dir: Path | None = None,
) -> Path:
    """
    Load or download a file. If dest_dir is provided, the downloaded file
    will be placed in that directory with its original filename.
    """
    file_path = Path(file_path)
    validate_json = bool(validate_json or config_mode)

    def is_valid_json(path: Path) -> bool:
        try:
            with open(path, "r") as f:
                json.load(f)
            return True
        except Exception:
            return False

    project_root = Path("/mnt/project").resolve()

    # 1) Use existing file if valid
    if file_path.exists() and (not validate_json or is_valid_json(file_path)):
        return file_path

    # 2) Determine fallback target
    if dest_dir is not None:
        dest_dir = Path(dest_dir)
        dest_dir.mkdir(parents=True, exist_ok=True)
        fallback_path = dest_dir / file_path.name
    else:
        try:
            relative_path = file_path.resolve().relative_to(project_root)
        except Exception:
            relative_path = file_path.name
        fallback_path = Path(".") / relative_path
        fallback_path.parent.mkdir(parents=True, exist_ok=True)

    # 3) Use existing fallback if valid
    if fallback_path.exists() and (not validate_json or is_valid_json(fallback_path)):
        return fallback_path

    # 4) Download via dx
    rc = os.system(f"dx download {file_id} -o {fallback_path} --overwrite")
    if rc != 0 or not fallback_path.exists():
        raise FileNotFoundError(f"Failed to download {description} ({file_id})")

    return fallback_path


# ----------------------------
# CLI
# ----------------------------
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Process phenotype field list and save merged metadata.")
    # Back-compat positional args:
    parser.add_argument("fields", type=str, nargs="?", default=None,
                        help="Legacy: input file path/key/ID. Ignored if --field/--fields-dict provided.")
    parser.add_argument("output_file", type=str,
                        help="Output CSV file path (required if you want the result written to disk).")
    parser.add_argument("--fields_list_name", type=str, default=None,
                        help="Optional local name for downloaded input file (when using a DNAnexus file ID).")

    # New flexible inputs:
    parser.add_argument("--field", action="append",
                        help="Field in 'entity.field' form. Repeat this flag for multiple fields.")
    parser.add_argument("--fields-dict", type=str,
                        help='JSON mapping of entity -> field, e.g. \'{"participant":"id","questionnaire":"demog_gender_2_1"}\'')

    args = parser.parse_args()

    # Parse fields-dict JSON if supplied
    fields_dict_arg = None
    if args.fields_dict:
        try:
            fields_dict_arg = json.loads(args.fields_dict)
            if not isinstance(fields_dict_arg, dict):
                raise ValueError("fields-dict must parse to a JSON object")
        except Exception as e:
            logger.error(f"Invalid --fields-dict JSON: {e}")
            raise SystemExit(2)

    # Decide which input style to use
    if args.field or fields_dict_arg:
        # New style: ignore positional 'fields'
        field_list(
            fields=args.field if args.field else None,
            output_file=args.output_file,
            fields_list_name=args.fields_list_name,
            input_file=None,
            input_file_name=None,
        )
        if fields_dict_arg:
            # If both provided, merge both sources
            # Call again or merge before: here we call once with dict-only if list not given
            field_list(
                fields=fields_dict_arg,
                output_file=args.output_file,
                fields_list_name=args.fields_list_name,
            )
    else:
        # Legacy style
        field_list(
            fields=args.fields,
            output_file=args.output_file,
            fields_list_name=args.fields_list_name,
        )
