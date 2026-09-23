# icd.py
from __future__ import annotations

import logging
from typing import Any, Optional, Tuple, Dict, List
from collections.abc import Mapping

import pandas as pd

logger = logging.getLogger(__name__)

__all__ = [
    # pandas APIs
    "match_icd_traits",
    "get_matched_icd_traits",
    # spark APIs
    "match_icd_traits_spark",
    "get_matched_icd_traits_spark",
    # dispatchers
    "match_icd_traits_any",
    "get_matched_icd_traits_any",
]

# -------------------- Internal helpers (shared) --------------------
def _get_diag_cols(
    df: pd.DataFrame,
    explicit: Optional[List[str]] = None,
    *,
    prefix: Optional[str] = "nhse_eng_inpat.diag_4_",  # default prefix
    extra: Optional[List[str]] = None,
    all_if_none: bool = False                          # <-- new flag
) -> List[str]:
    """Resolve diagnosis columns from explicit names, prefixes, and extras.

    Args:
        df: Input pandas DataFrame.
        explicit: Explicit column names to use.
        prefix: Column prefix to match when explicit is None.
        extra: Additional column names to include when present in df.
        all_if_none: When True, return all columns except pid if none found.

    Returns:
        List of diagnosis column names.

    Raises:
        ValueError: If no diagnosis columns are found and all_if_none is False.
    """
    if explicit is not None:
        return list(explicit)

    cols = []
    if prefix:
        cols = [c for c in df.columns if c.startswith(prefix)]
    if extra:
        cols += [c for c in extra if c in df.columns]

    if not cols:
        if all_if_none:
            # return everything except obvious ID columns
            cols = [c for c in df.columns if not c.endswith(".pid")]
        else:
            raise ValueError(f"No diagnosis columns found (expected prefix '{prefix}').")

    return cols


def _normalize_codes(x: Any) -> List[str]:
    """Normalize codes to a clean list of strings.

    Args:
        x: Single code, list of codes, or stringified list.

    Returns:
        List of cleaned string codes.
    """
    if x is None or (isinstance(x, float) and pd.isna(x)):
        return []
    if isinstance(x, list):
        return [str(c).strip() for c in x if pd.notna(c)]
    s = str(x).strip()
    if s.startswith("[") and s.endswith("]"):
        return [c.strip() for c in s[1:-1].split(",") if c.strip()]
    return [s] if s else []


def _traits_to_codes(traits_and_codes: Any) -> Dict[str, List[str]]:
    """Convert trait/code inputs into a normalized mapping.

    Args:
        traits_and_codes: Dict of trait->codes, DataFrame with columns
            ['trait','ICD_code'], or (traits, codes) aligned sequences.

    Returns:
        Mapping of trait to a deduped list of codes (order-preserving).

    Raises:
        ValueError: If the input format is unsupported or missing columns.
    """
    if isinstance(traits_and_codes, Mapping):
        items = traits_and_codes.items()
    elif isinstance(traits_and_codes, pd.DataFrame):
        if not {"trait", "ICD_code"} <= set(traits_and_codes.columns):
            raise ValueError("DataFrame must contain 'trait' and 'ICD_code' columns.")
        items = zip(traits_and_codes["trait"], traits_and_codes["ICD_code"])
    elif isinstance(traits_and_codes, (list, tuple)) and len(traits_and_codes) == 2:
        traits, codes = traits_and_codes
        items = zip(traits, codes)
    else:
        raise ValueError("Provide a dict, a DataFrame with ['trait','ICD_code'], or (traits, codes).")

    out: Dict[str, List[str]] = {}
    for trait, codes in items:
        seen, cleaned = set(), []
        for c in _normalize_codes(codes):
            if c and c not in seen:
                cleaned.append(c)
                seen.add(c)
        out[str(trait)] = cleaned
    return out


def _to_str_series(s: pd.Series, *, use_pyarrow: bool = True) -> pd.Series:
    """Ensure a lean string dtype for fast vectorized ops.

    Args:
        s: Input series.
        use_pyarrow: Prefer pyarrow string dtype when available.

    Returns:
        Series converted to a string dtype.
    """
    if use_pyarrow:
        try:
            return s.astype("string[pyarrow]")
        except Exception:
            pass
    return s.astype("string")


def _is_spark_df(obj: Any) -> bool:
    """Return True if the object looks like a Spark DataFrame.

    Args:
        obj: Object to check.

    Returns:
        True when the object has Spark DataFrame attributes.
    """
    return hasattr(obj, "sparkSession") and hasattr(obj, "schema")

# -------------------- Pandas APIs --------------------

def match_icd_traits(
    raw_df: pd.DataFrame,
    traits_and_codes: Any,
    *,
    diag_cols: Optional[List[str]] = None,
    pid_col: str = "nhse_eng_inpat.pid",
    prefix_if_len_at_most: Optional[int] = 3,
    primary_only: bool = False,
    return_occurrence_counts: bool = True,
    use_pyarrow_strings: bool = True,
    chunksize: Optional[int] = None,
    diag_prefix: Optional[str] = "nhse_eng_inpat.diag_4_",
    extra_diag_cols: Optional[List[str]] = None,
    all_if_none: bool = False,
) -> Tuple[Dict[str, set], pd.DataFrame]:
    """Match ICD traits in a pandas DataFrame.

    Args:
        raw_df: Input DataFrame with diagnosis columns.
        traits_and_codes: Dict, DataFrame, or (traits, codes) mapping.
        diag_cols: Explicit diagnosis columns to scan.
        pid_col: Participant ID column name.
        prefix_if_len_at_most: Prefix length threshold for startswith matches.
        primary_only: Use only the primary diagnosis column.
        return_occurrence_counts: Include total occurrence counts per trait.
        use_pyarrow_strings: Prefer pyarrow string dtype for faster ops.
        chunksize: Optional chunk size for scanning large frames.
        diag_prefix: Prefix to select diagnosis columns when diag_cols is None.
        extra_diag_cols: Additional diagnosis columns to include when present.
        all_if_none: If no diagnosis columns found, use all non-pid columns.

    Returns:
        A tuple of (trait_to_pids, summary_df).
    """

    all_diag_cols = _get_diag_cols(
        raw_df,
        explicit=diag_cols,
        prefix=diag_prefix,
        extra=extra_diag_cols,
        all_if_none=all_if_none,
    )
    diag_cols = [min(all_diag_cols, key=lambda x: int(x.split("_")[-1]))] if primary_only else all_diag_cols

    trait_map = _traits_to_codes(traits_and_codes)

    # Pre-derive exact vs prefix per trait
    trait_specs: Dict[str, Tuple[set, Tuple[str, ...]]] = {}
    for trait, codes in trait_map.items():
        exact, prefixes = set(), []
        for c in codes:
            cc = c.strip()
            if not cc:
                continue
            if prefix_if_len_at_most is not None and len(cc) <= int(prefix_if_len_at_most):
                prefixes.append(cc)
            else:
                exact.add(cc)
        trait_specs[trait] = (exact, tuple(prefixes))

    n = len(raw_df)
    ranges = [(0, n)] if (chunksize is None or chunksize >= n) else \
             [(i, min(i + chunksize, n)) for i in range(0, n, chunksize)]

    trait_to_pids: Dict[str, set] = {trait: set() for trait in trait_map}
    occ_counts: Optional[Dict[str, int]] = {trait: 0 for trait in trait_map} if return_occurrence_counts else None

    for start, end in ranges:
        df_chunk = raw_df.iloc[start:end]
        row_any: Dict[str, pd.Series] = {trait: pd.Series(False, index=df_chunk.index) for trait in trait_map}

        for col in diag_cols:
            s = _to_str_series(df_chunk[col], use_pyarrow=use_pyarrow_strings)
            for trait, (exact, pref_tuple) in trait_specs.items():
                col_match = pd.Series(False, index=s.index)
                if exact:
                    col_match = s.isin(exact)
                if pref_tuple:
                    col_match = col_match | s.str.startswith(pref_tuple, na=False)
                if col_match.any():
                    row_any[trait] = row_any[trait] | col_match
                    if occ_counts is not None:
                        occ_counts[trait] += int(col_match.sum())

        pid_vals = df_chunk[pid_col]
        for trait, mask in row_any.items():
            if mask.any():
                trait_to_pids[trait].update(pid_vals[mask].dropna().unique())

    rows = []
    for trait, pids in trait_to_pids.items():
        row = {"trait": trait, "participant_count": len(pids)}
        if occ_counts is not None:
            row["occurrence_count"] = occ_counts[trait]
        rows.append(row)

    summary_df = (
        pd.DataFrame(rows)
        .sort_values(["participant_count", "trait"], ascending=[False, True])
        .reset_index(drop=True)
    )
    return trait_to_pids, summary_df


def get_matched_icd_traits(
    raw_df: pd.DataFrame,
    traits_and_codes: Any,
    *,
    diag_cols: Optional[List[str]] = None,
    prefix_if_len_at_most: int = 3,
    uppercase: bool = True,
    remove_chars: Tuple[str, ...] = (),
) -> pd.DataFrame:
    """Summarize ICD code matches per trait (pandas path).

    Args:
        raw_df: Input DataFrame with diagnosis columns.
        traits_and_codes: Dict, DataFrame, or (traits, codes) mapping.
        diag_cols: Explicit diagnosis columns to scan.
        prefix_if_len_at_most: Prefix length threshold for startswith matches.
        uppercase: Whether to uppercase observed codes before matching.
        remove_chars: Characters to remove from codes before matching.

    Returns:
        DataFrame with columns [trait, n_unique_codes, unique_codes].
    """
    diag_cols = _get_diag_cols(raw_df, diag_cols)

    def _norm_series(s: pd.Series) -> pd.Series:
        s = s.astype("string")
        if uppercase:
            s = s.str.upper()
        for ch in remove_chars:
            s = s.str.replace(ch, "", regex=False)
        return s.str.strip()

    observed: set[str] = set()
    for col in diag_cols:
        vals = _norm_series(raw_df[col]).dropna().unique()
        observed.update([v for v in map(str, vals) if v and v != "<NA>"])

    trait_map = _traits_to_codes(traits_and_codes)

    def _norm_code(c: str) -> str:
        cc = str(c).strip()
        if uppercase:
            cc = cc.upper()
        for ch in remove_chars:
            cc = cc.replace(ch, "")
        return cc

    specs: Dict[str, Tuple[set, Tuple[str, ...]]] = {}
    for trait, codes in trait_map.items():
        exact, prefixes = set(), []
        for c in codes:
            cc = _norm_code(c)
            if not cc:
                continue
            if len(cc) <= prefix_if_len_at_most:
                prefixes.append(cc)
            else:
                exact.add(cc)
        specs[trait] = (exact, tuple(prefixes))

    out_rows = []
    for trait, (exact, pref_tuple) in specs.items():
        matched = set()
        if exact:
            matched |= (observed & exact)
        if pref_tuple:
            matched |= {code for code in observed if code.startswith(pref_tuple)}
        codes_sorted = sorted(matched)
        out_rows.append(
            {"trait": trait, "n_unique_codes": len(codes_sorted), "unique_codes": codes_sorted}
        )

    return pd.DataFrame(out_rows).sort_values("trait").reset_index(drop=True)

# -------------------- Spark helpers + APIs --------------------

def _spark_diag_long(
    sdf,
    *,
    pid_col: str,
    diag_prefix: str,
    extra_diag_cols: Optional[List[str]] = None,  # <-- NEW
    primary_only: bool = False,                   # <-- NEW
    uppercase: bool = True,
    remove_chars: tuple = ()
):
    """Explode diagnosis columns into a long Spark DataFrame.

    Args:
        sdf: Spark DataFrame with diagnosis columns.
        pid_col: Participant ID column name.
        diag_prefix: Prefix to select diagnosis columns.
        extra_diag_cols: Optional extra diagnosis columns.
        primary_only: If True, keep only the primary diagnosis column.
        uppercase: Whether to uppercase codes.
        remove_chars: Characters to remove from codes.

    Returns:
        Spark DataFrame with columns [pid, code].
    """
    from pyspark.sql import functions as F
    import re

    def _q(name: str) -> str:
        return name if (name.startswith("`") and name.endswith("`")) else f"`{name}`"

    # gather diagnosis columns by prefix + optional extras
    diag_cols = [c for c in sdf.columns if c.startswith(diag_prefix)]
    if extra_diag_cols:
        diag_cols += [c for c in extra_diag_cols if c in sdf.columns]

    if not diag_cols:
        raise ValueError(f"No diagnosis columns found with prefix '{diag_prefix}'.")

    if primary_only:
        # pick the smallest numeric suffix, e.g. *_01 over *_02 …
        def suffix_key(c: str) -> int:
            m = re.search(r"(\d+)$", c)
            return int(m.group(1)) if m else 10**9
        diag_cols = [min(diag_cols, key=suffix_key)]

    arr = F.array(*[F.col(_q(c)) for c in diag_cols])
    long_df = (
        sdf.select(
            F.col(_q(pid_col)).alias("pid"),
            F.explode_outer(arr).alias("code"),
        )
        .filter(F.col("code").isNotNull())
    )

    if uppercase:
        long_df = long_df.withColumn("code", F.upper(F.col("code")))
    for ch in remove_chars:
        long_df = long_df.withColumn("code", F.regexp_replace(F.col("code"), ch, ""))

    return long_df


# --- Spark APIs ---

def get_matched_icd_traits_spark(
    sdf,
    traits_and_codes,
    *,
    pid_col: str = "nhse_eng_inpat.pid",
    diag_prefix: str = "nhse_eng_inpat.diag_4_",
    extra_diag_cols: Optional[List[str]] = None,     # <-- NEW
    primary_only: bool = False,                      # <-- NEW
    prefix_if_len_at_most: int = 3,
    uppercase: bool = True,
    remove_chars: tuple = (),
) -> pd.DataFrame:
    """Summarize ICD code matches per trait using Spark.

    Args:
        sdf: Spark DataFrame with diagnosis columns.
        traits_and_codes: Dict, DataFrame, or (traits, codes) mapping.
        pid_col: Participant ID column name.
        diag_prefix: Prefix to select diagnosis columns.
        extra_diag_cols: Optional extra diagnosis columns.
        primary_only: If True, keep only the primary diagnosis column.
        prefix_if_len_at_most: Prefix length threshold for startswith matches.
        uppercase: Whether to uppercase observed codes before matching.
        remove_chars: Characters to remove from codes before matching.

    Returns:
        DataFrame with columns [trait, n_unique_codes, unique_codes].
    """
    long_df = _spark_diag_long(
        sdf,
        pid_col=pid_col,
        diag_prefix=diag_prefix,
        extra_diag_cols=extra_diag_cols,             # <-- NEW
        primary_only=primary_only,                   # <-- NEW
        uppercase=uppercase,
        remove_chars=remove_chars,
    ).select("code").distinct()

    code_universe = [r["code"] for r in long_df.collect()]
    trait_map = _traits_to_codes(traits_and_codes)

    rows = []
    for trait, codes in trait_map.items():
        exact = {c for c in codes if len(c) > prefix_if_len_at_most}
        prefixes = tuple(c for c in codes if len(c) <= prefix_if_len_at_most)
        matched = set()
        if exact:
            matched |= (set(code_universe) & exact)
        if prefixes:
            matched |= {c for c in code_universe if c.startswith(prefixes)}
        lst = sorted(matched)
        rows.append({"trait": trait, "n_unique_codes": len(lst), "unique_codes": lst})

    return pd.DataFrame(rows).sort_values("trait").reset_index(drop=True)


def match_icd_traits_spark(
    sdf,
    traits_and_codes,
    *,
    pid_col: str = "nhse_eng_inpat.pid",
    diag_prefix: str = "nhse_eng_inpat.diag_4_",
    extra_diag_cols: Optional[List[str]] = None,
    primary_only: bool = False,
    prefix_if_len_at_most: Optional[int] = 3,
    uppercase: bool = True,
    remove_chars: tuple = (),
    return_occurrence_counts: bool = True,
    return_pids: bool = False,
    max_pids_collect: int = 200_000,
):
    """Match ICD traits in Spark with broadcasted trait codes.

    Args:
        sdf: Spark DataFrame with diagnosis columns.
        traits_and_codes: Dict, DataFrame, or (traits, codes) mapping.
        pid_col: Participant ID column name.
        diag_prefix: Prefix to select diagnosis columns.
        extra_diag_cols: Optional extra diagnosis columns.
        primary_only: If True, keep only the primary diagnosis column.
        prefix_if_len_at_most: Prefix length threshold for startswith matches.
        uppercase: Whether to uppercase observed codes before matching.
        remove_chars: Characters to remove from codes before matching.
        return_occurrence_counts: Include total occurrence counts per trait.
        return_pids: Whether to collect pid sets for small traits.
        max_pids_collect: Maximum pid set size to collect per trait.

    Returns:
        A tuple of (trait_to_pids, summary_df).
    """
    import pandas as pd
    from pyspark.sql import functions as F
    from pyspark.sql import types as T
    from pyspark.sql import SparkSession

    spark = SparkSession.builder.getOrCreate()
    spark.conf.set("spark.sql.adaptive.enabled", "true")
    spark.conf.set("spark.sql.adaptive.skewJoin.enabled", "true")

    # --- 1) Long-form diagnoses (pid, code) with your existing helper ---
    #     _spark_diag_long should already normalize (uppercase/remove_chars)
    long_df = (
        _spark_diag_long(
            sdf,
            pid_col=pid_col,
            diag_prefix=diag_prefix,
            extra_diag_cols=extra_diag_cols,
            primary_only=primary_only,
            uppercase=uppercase,
            remove_chars=remove_chars,
        )
        .select(F.col("pid"), F.col("code"))
        .where(F.col("code").isNotNull() & (F.length("code") > 0))
    )

    # --- 2) Build broadcastable Spark DF of (trait, icd_norm, is_prefix) ---
    # traits_and_codes can be a dict or a DF; reuse your existing converter
    trait_map = _traits_to_codes(traits_and_codes)

    # Flatten to rows
    rows = []
    for trait, codes in trait_map.items():
        for c in codes:
            cnorm = c.upper() if uppercase else c
            # remove characters the same way long_df did (keep logic identical)
            for ch in remove_chars:
                cnorm = cnorm.replace(ch, "")
            is_prefix = prefix_if_len_at_most is not None and len(cnorm) <= int(prefix_if_len_at_most)
            rows.append((trait, cnorm, bool(is_prefix)))

    pheno_pdf = pd.DataFrame(rows, columns=["trait", "icd_norm", "is_prefix"])
    pheno_sdf = spark.createDataFrame(
        pheno_pdf.astype({"trait": "string", "icd_norm": "string", "is_prefix": "bool"})
    )

    # --- 3) Prefix-aware join (single pass) ---
    # code startswith icd_norm when is_prefix else code == icd_norm
    # startswith with a column requires an expression:
    cond = F.when(
        F.col("is_prefix"),
        F.expr("code LIKE concat(icd_norm, '%')")
    ).otherwise(F.col("code") == F.col("icd_norm"))

    matched = (
        long_df.hint("shuffle_hash")                      # avoid broadcasting the big table
               .join(F.broadcast(pheno_sdf), cond, how="inner")
               .select("pid", "trait")
               .dropDuplicates(["pid", "trait"])          # (pid, trait) once
    )

    # --- 4) Summary aggregation in Spark ---
    agg_exprs = [F.countDistinct("pid").alias("participant_count")]
    if return_occurrence_counts:
        # For occurrences you need the non-distinct matches; recompute cheaply:
        # Join again without dropDuplicates to count occurrences per trait.
        # Alternatively, do it once up-front by keeping a "raw_matched".
        raw_matched = (
            long_df.hint("shuffle_hash")
                   .join(F.broadcast(pheno_sdf), cond, how="inner")
                   .select("pid", "trait")
        )
        occ_df = raw_matched.groupBy("trait").agg(F.count("*").alias("occurrence_count"))
        part_df = matched.groupBy("trait").agg(F.countDistinct("pid").alias("participant_count"))
        summary_s = (
            part_df.join(occ_df, on="trait", how="left")
                   .orderBy(F.desc("participant_count"), F.asc("trait"))
        )
    else:
        summary_s = (
            matched.groupBy("trait")
                   .agg(*agg_exprs)
                   .orderBy(F.desc("participant_count"), F.asc("trait"))
        )

    # --- 5) Optional, bounded pid collection (small traits only) ---
    trait_to_pids: Dict[str, set] = {}
    if return_pids:
        # Identify traits with participant_count <= cap
        small_traits = (
            summary_s.where(F.col("participant_count") <= F.lit(int(max_pids_collect)))
                     .select("trait")
        )
        pids_small = (
            matched.join(small_traits, on="trait", how="inner")
                   .groupBy("trait")
                   .agg(F.collect_set("pid").alias("pids"))
        )

        # Collect only the small traits’ pid sets
        for row in pids_small.toLocalIterator():
            trait_to_pids[row["trait"]] = set(row["pids"])

        # Ensure all traits exist in dict
        remaining = (
            summary_s.select("trait")
                     .where(~F.col("trait").isin(list(trait_to_pids.keys())))
                     .toLocalIterator()
        )
        for r in remaining:
            trait_to_pids[r["trait"]] = set()
    else:
        # Keep API: return empty sets when not requested
        for r in summary_s.select("trait").toLocalIterator():
            trait_to_pids[r["trait"]] = set()

    # --- 6) Return a small pandas summary (safe to collect) ---
    summary_df = summary_s.toPandas()
    return trait_to_pids, summary_df



# --- Dispatcher (keep stripping pandas-only kwargs, do NOT strip extra_diag_cols) ---

def match_icd_traits_any(df_or_sdf, *args, **kwargs):
    """Dispatch to pandas or Spark matcher based on input type.

    Args:
        df_or_sdf: pandas DataFrame or Spark DataFrame.
        *args: Positional arguments forwarded to the implementation.
        **kwargs: Keyword arguments forwarded to the implementation.

    Returns:
        The result of match_icd_traits or match_icd_traits_spark.
    """
    if _is_spark_df(df_or_sdf):
        # remove pandas-only args
        kwargs.pop("use_pyarrow_strings", None)
        kwargs.pop("chunksize", None)
        # Spark accepts: pid_col, diag_prefix, extra_diag_cols, primary_only, etc.
        return match_icd_traits_spark(df_or_sdf, *args, **kwargs)
    return match_icd_traits(df_or_sdf, *args, **kwargs)


def get_matched_icd_traits_any(df_or_sdf, *args, **kwargs):
    """Dispatch to pandas or Spark code-summary based on input type.

    Args:
        df_or_sdf: pandas DataFrame or Spark DataFrame.
        *args: Positional arguments forwarded to the implementation.
        **kwargs: Keyword arguments forwarded to the implementation.

    Returns:
        The result of get_matched_icd_traits or get_matched_icd_traits_spark.
    """
    if _is_spark_df(df_or_sdf):
        return get_matched_icd_traits_spark(df_or_sdf, *args, **kwargs)
    return get_matched_icd_traits(df_or_sdf, *args, **kwargs)
