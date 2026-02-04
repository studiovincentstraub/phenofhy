# phenofhy/calculate.py

import os
import re
import ast
import math
import logging
import difflib
import unicodedata
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Iterable, Literal, Optional, Dict, Tuple, Any, List, Union


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

try:
    from ._rules import DERIVED_CODEBOOK, DEFAULT_MEDICAT_GROUP_MAP
except Exception:
    # Ensure both symbols exist even if _rules isn't importable
    DERIVED_CODEBOOK = {}
    DEFAULT_MEDICAT_GROUP_MAP = {}


# ---------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------

def get_combined_count(df: pd.DataFrame, cols: List[str], value) -> int:
    """
    Count how many rows have the given value across multiple mutually-exclusive columns.
    Example: get_combined_count(df, ["demog_sex_1_1","demog_sex_2_1"], 1)
    """
    return ((df[cols] == value).any(axis=1)).sum()

def _normalize_category_label(v):
    if v is None or pd.isna(v):
        return "nan"

    s = str(v).strip()

    # Force correct canonical outputs
    if s.lower() == "current": return "Current"
    if s.lower() == "former":  return "Former"
    if s.lower() == "never":   return "Never"
    if s.lower() == "unknown": return "Unknown"
    if s.lower() == "missing": return "nan"

    # Normal NA handling
    if s.lower() in {"na", "n/a", "null", "<na>", ""}:
        return "nan"

    return s

def resolve_codebook_csv(
    codebook_csv: Optional[str] = None,
    metadata_dir: str = "./metadata",
    pattern: str = "*.codings.csv",
    prefer: Literal["mtime", "first", "error"] = "mtime",
) -> Optional[str]:
    """
    Resolve the codebook CSV path.

    Order of precedence:
      1) If `codebook_csv` is given, return it.
      2) Else, glob for `pattern` inside `metadata_dir`.
         - If exactly one match: return it.
         - If multiple:
             • prefer="mtime"  -> return most recently modified
             • prefer="first"  -> return the lexicographically first
             • prefer="error"  -> raise ValueError
         - If none: return None.
    """
    if codebook_csv:
        return codebook_csv

    matches = sorted(Path(metadata_dir).glob(pattern))
    if not matches:
        return None
    if len(matches) == 1:
        return str(matches[0])

    if prefer == "mtime":
        matches.sort(key=lambda p: p.stat().st_mtime, reverse=True)
        return str(matches[0])
    if prefer == "first":
        return str(matches[0])
    raise ValueError(f"Multiple codebooks matched: {[str(p) for p in matches]}")


# ---------------------------------------------------------------------
# Normalization + list-aware mapping helpers
# ---------------------------------------------------------------------

def _norm(x: Any) -> str:
    """Simple case/space normalizer."""
    return str(x).strip().lower()


def _norm_key(x: Any) -> str:
    """
    Robust, Unicode-tolerant normalizer for matching keys like (coding_name, meaning).
    - Unicode normalize (NFKC), fold diacritics, unify smart quotes/hyphens/slashes,
      collapse whitespace, lowercase.
    """
    s = str(x)
    s = unicodedata.normalize("NFKC", s)
    s = (
        s.replace("\u2019", "'").replace("\u2018", "'")
         .replace("\u201C", '"').replace("\u201D", '"')
         .replace("\u2212", "-").replace("\u2013", "-").replace("\u2014", "-")
         .replace("\u2044", "/").replace("\u2215", "/").replace("／", "/")
    )
    s = "".join(ch for ch in unicodedata.normalize("NFKD", s) if not unicodedata.combining(ch))
    s = " ".join(s.split()).lower().strip()
    return s


def _is_listlike_cell(v) -> bool:
    return isinstance(v, (list, tuple, set, np.ndarray))


def _maybe_number(x):
    """
    Coerce scalars/strings like '1', '1.0' → 1; '2.5' → 2.5; otherwise return as-is.
    """
    try:
        if isinstance(x, (int, np.integer)):
            return int(x)
        if isinstance(x, (float, np.floating)):
            return int(x) if float(x).is_integer() else float(x)
        s = str(x).strip()
        if s == "":
            return np.nan
        if re.fullmatch(r"[+-]?\d+", s):
            return int(s)
        if re.fullmatch(r"[+-]?\d+\.\d+", s):
            f = float(s)
            return int(f) if f.is_integer() else f
        return x
    except Exception:
        return x


def _coerce_cell_to_sequence(v):
    """
    Robustly turn a cell into a list of atomic values:
    - list/tuple/set/np.ndarray → list(...)
    - strings like "[1,2]" or "1,2" or "1|2" or "1;2" → parsed list
    - "None"/"NA"/"null"/"" → []
    - scalar "1"/1.0 → [1]
    """
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return []
    if isinstance(v, (list, tuple, set, np.ndarray)):
        return list(v)
    if isinstance(v, str):
        s = v.strip()
        if s == "" or s.lower() in {"na", "nan", "none", "null"}:
            return []
        if (s.startswith("[") and s.endswith("]")) or (s.startswith("(") and s.endswith(")")) or (s.startswith("{") and s.endswith("}")):
            try:
                parsed = ast.literal_eval(s)
                if isinstance(parsed, (list, tuple, set, np.ndarray)):
                    return [_maybe_number(x) for x in list(parsed)]
                return [_maybe_number(parsed)]
            except Exception:
                pass
        if any(d in s for d in (",", "|", ";")):
            parts = re.split(r"[,\|;]\s*", s)
            return [_maybe_number(p) for p in parts if p != ""]
        return [_maybe_number(s)]
    return [_maybe_number(v)]


def _explode_listlike(s: pd.Series) -> pd.Series:
    """Make one value per row, robust to stringified lists."""
    return s.astype(object).apply(_coerce_cell_to_sequence).explode().dropna()


def _map_series_listaware_to_codes(s: pd.Series, code2label: dict) -> pd.Series:
    def _map_cell(v):
        if v is None:
            return v
        if _is_listlike_cell(v):
            seq = list(v) if isinstance(v, np.ndarray) else list(v)
            out = []
            for x in seq:
                # try exact, string form, then normalized form
                if x in code2label:
                    out.append(code2label[x])
                    continue
                sx = str(x)
                if sx in code2label:
                    out.append(code2label[sx])
                    continue
                nk = _norm_key(x)
                if nk in code2label:
                    out.append(code2label[nk])
                    continue
                out.append(x)
            return out
        # scalar case
        if v in code2label:
            return code2label[v]
        sv = str(v)
        if sv in code2label:
            return code2label[sv]
        nk = _norm_key(v)
        if nk in code2label:
            return code2label[nk]
        return v
    return s.apply(_map_cell)


def _map_series_listaware_to_labels(s: pd.Series, code2label: dict) -> pd.Series:
    def _map_cell(v):
        if v is None:
            return v
        if _is_listlike_cell(v):
            seq = list(v) if isinstance(v, np.ndarray) else list(v)
            out = []
            for x in seq:
                if x in code2label:
                    out.append(code2label[x])
                    continue
                sx = str(x)
                if sx in code2label:
                    out.append(code2label[sx])
                    continue
                nk = _norm_key(x)
                if nk in code2label:
                    out.append(code2label[nk])
                    continue
                out.append(x)
            return out
        if v in code2label:
            return code2label[v]
        sv = str(v)
        if sv in code2label:
            return code2label[sv]
        nk = _norm_key(v)
        if nk in code2label:
            return code2label[nk]
        return v
    return s.apply(_map_cell)



def find_df_column_for_coding(df: pd.DataFrame, coding_name: str, entity: str) -> str:
    """
    Map (coding_name, entity) to the actual column in df.
    Strategy:
      1) look for columns that start with the expected prefix ('participant.' or 'questionnaire.')
         AND end with the coding_name (case-insensitive);
      2) else fallback to substring match.
    """
    pref = 'participant.' if _norm(entity) == 'participant' else 'questionnaire.'
    want = _norm_key(coding_name)

    for c in df.columns:
        cl = c.lower()
        if cl.startswith(pref) and cl.endswith(want):
            return c
    for c in df.columns:
        if want in c.lower():
            return c
    raise KeyError(f'Could not find a column for coding_name="{coding_name}" (entity="{entity}").')


def build_code_lookup(codings_df: pd.DataFrame) -> Tuple[Dict[Tuple[str, str], Any], Dict[str, Dict[str, Any]]]:
    """
    Build:
      1) lookup_exact: (norm(coding_name), norm(meaning)) -> code
      2) by_coding:    norm(coding_name) -> { norm(meaning) -> code }
    """
    lookup_exact: Dict[Tuple[str, str], Any] = {}
    by_coding: Dict[str, Dict[str, Any]] = {}
    for _, r in codings_df.iterrows():
        ck = _norm_key(r["coding_name"])
        mk = _norm_key(r["meaning"])
        code = r["code"]
        lookup_exact[(ck, mk)] = code
        by_coding.setdefault(ck, {})[mk] = code
    return lookup_exact, by_coding


# ---------------------------------------------------------------------
# Grouped statistics (list-aware)
# ---------------------------------------------------------------------

def summary(
    df: pd.DataFrame,
    traits: Optional[Iterable[str]] = None,
    *,
    stratify: Optional[Union[str, Dict[str, Iterable[Any]]]] = None,
    sex_col: str = "derived.sex",
    age_col: str = "derived.age_at_registration",
    age_bins: Optional[Dict[str, list]] = None,
    round_decimals: int = 2,
    categorical_traits: Optional[Iterable[str]] = None,
    label_mode: Literal["labels", "codes"] = "codes",
    codebook_csv: Optional[str] = None,
    metadata_dir: str = "./metadata",
    data_dictionary_csv: Optional[str] = None,
    local_codebook: Optional[Dict[str, Dict[Any, str]]] = None,
    autodetect_coded_categoricals: bool = True,
    autodetect_max_levels: int = 10,
    autodetect_exclude: Optional[Iterable[str]] = None,
    sex_keep: Optional[Dict[Any, str]] = None,
    granularity: Literal["variable", "category"] = "variable",
):
    """
    Compute grouped summaries.

    - If stratify is None: compute for whole sample (no sample column).
    - If stratify is a column name string: compute per-value of that column (all observed values).
    - If stratify is a single-key dict {col: [vals,...]}: compute per those values (subset).
      * allowed values are intersected with values actually present in the data.
    - granularity='variable' -> one row per variable (aggregate).
      granularity='category' -> one row per category/value (disaggregate).
    - Skips columns ending with '.pid'.
    - Uses codebook (codebook_csv) for mapping when provided and local derived codebook
      (DERIVED_CODEBOOK) if available.
    - Uses data_dictionary_csv to get human-readable descriptions for `trait`.
    """
    tmp = df.copy()

    # pick traits
    if traits is None:
        traits = [
            c for c in tmp.columns
            if not c.startswith("__")
            and not c.startswith("derived.age_group")
            and not c.startswith("derived.sex_tmp")
            and not str(c).lower().endswith(".pid")
        ]
    else:
        traits = [c for c in traits if c in tmp.columns and not str(c).lower().endswith(".pid")]

    if not traits:
        raise ValueError("No requested traits are present in the dataframe.")

    def _meta_key(col: str) -> str:
        return col.split(".")[-1].upper()

    # load codebook
    if codebook_csv is None:
        codebook_csv = resolve_codebook_csv(None, metadata_dir=metadata_dir, pattern="*.codings.csv", prefer="mtime")
    meta = None
    if codebook_csv:
        meta = pd.read_csv(codebook_csv)
        meta = meta.rename(columns={c: c.strip().lower() for c in meta.columns})
        required = {"coding_name", "code", "meaning"}
        missing = required - set(meta.columns)
        if missing:
            raise ValueError(f"Codebook CSV missing columns: {missing}")

    # load data dictionary for human descriptions
    data_dict = None
    if data_dictionary_csv:
        try:
            data_dict = pd.read_csv(data_dictionary_csv)
            data_dict = data_dict.rename(columns={c: c.strip().lower() for c in data_dict.columns})
        except Exception:
            data_dict = None

    # helper: meaning from code/coding
    def _meaning_for_code(col, code):
        if meta is not None:
            ck = col.split(".")[-1].upper()
            match = meta[(meta["coding_name"] == ck) & (meta["code"].astype(str) == str(code))]
            if not match.empty:
                return str(match["meaning"].iloc[0])
            match = meta[(meta["coding_name"] == ck) & (meta["code"] == code)]
            if not match.empty:
                return str(match["meaning"].iloc[0])
        derived_key = col if str(col).startswith("derived.") else f"derived.{str(col).split('.')[-1]}"
        if derived_key in (local_codebook or {}):
            m = (local_codebook or {})[derived_key]
            if code in m:
                return str(m[code])
            if str(code) in m:
                return str(m[str(code)])
        return str(code)

    # build code maps from codebook
    maps: Dict[str, Tuple[Dict[Any, str], Dict[str, Any]]] = {}
    if meta is not None:
        for col in traits:
            rows = meta.loc[meta["coding_name"] == _meta_key(col)]
            if rows.empty:
                continue
            s = tmp[col]

            def _coerce(v):
                if pd.api.types.is_integer_dtype(s.dtype) or str(s.dtype).startswith("Int"):
                    try:
                        return int(v)
                    except Exception:
                        pass
                if pd.api.types.is_float_dtype(s.dtype):
                    try:
                        return float(v)
                    except Exception:
                        pass
                return str(v)

            code2label = {_coerce(r.code): str(r.meaning) for _, r in rows.iterrows()}
            label2code: Dict[str, Any] = {}
            for k_code, v_meaning in code2label.items():
                label2code[v_meaning] = k_code
                try:
                    nk = _norm_key(v_meaning)
                    label2code[nk] = k_code
                except Exception:
                    pass
            maps[col] = (code2label, label2code)

    # local derived codebook
    if local_codebook is None:
        try:
            from ._rules import DERIVED_CODEBOOK as _DERIVED_CODEBOOK
        except Exception:
            try:
                from rules import DERIVED_CODEBOOK as _DERIVED_CODEBOOK
            except Exception:
                _DERIVED_CODEBOOK = {}
        local_codebook = _DERIVED_CODEBOOK

    for derived_col, mapping in (local_codebook or {}).items():
        if derived_col in traits and derived_col in tmp.columns:
            s = tmp[derived_col]
            def _coerce_for_series(v, s_series: pd.Series):
                if pd.api.types.is_integer_dtype(s_series.dtype) or str(s_series.dtype).startswith("Int"):
                    try:
                        return int(v)
                    except Exception:
                        pass
                if pd.api.types.is_float_dtype(s_series.dtype):
                    try:
                        return float(v)
                    except Exception:
                        pass
                return str(v)
            code2label = {_coerce_for_series(k, s): str(v) for k, v in mapping.items()}
            label2code = {}
            for k_code, v_meaning in code2label.items():
                label2code[v_meaning] = k_code
                try:
                    nk = _norm_key(v_meaning)
                    label2code[nk] = k_code
                except Exception:
                    pass
            maps[derived_col] = (code2label, label2code)

    # apply label_mode mapping where maps exist
    for col, (c2l, l2c) in maps.items():
        s = tmp[col]
        if label_mode == "labels":
            tmp[col] = _map_series_listaware_to_labels(s, c2l)
            has_listlikes = pd.Series(tmp[col]).astype(object).dropna().apply(_is_listlike_cell).any()
            if not has_listlikes:
                tmp[col] = pd.Series(tmp[col]).astype("category")
        else:
            tmp[col] = _map_series_listaware_to_codes(s, l2c)
            s2 = tmp[col]
            has_listlikes = pd.Series(s2).astype(object).dropna().apply(_is_listlike_cell).any()
            if not has_listlikes:
                if pd.api.types.is_integer_dtype(s2) or all(
                    (isinstance(v, (int, np.integer)) or pd.isna(v)) for v in s2
                ):
                    tmp[col] = pd.Series(s2).astype("Int64")

    # age bins if needed
    if age_bins is None:
        bins = [18, 30, 60, np.inf]
        labels = ["18-29", "30-59", "60+"]
    else:
        bins = age_bins.get("bins")
        labels = age_bins.get("labels")
        if bins is None or labels is None:
            raise ValueError("age_bins must provide both 'bins' and 'labels'.")

    if age_col in tmp.columns:
        tmp["derived.age_group"] = pd.cut(tmp[age_col], bins=bins, labels=labels, right=False)

    # -----------------------
    # prepare groups (STRATIFY)
    # -----------------------
    # --- prepare groups / stratify (supports codes or labels) ---
    if stratify is None:
        groups = [((), tmp)]
    else:
        # normalize input form
        if isinstance(stratify, dict):
            if len(stratify) != 1:
                raise ValueError("stratify dict must contain exactly one key, e.g. {'derived.sex':[1,2]} or {'derived.sex':['Male']}")
            strat_key, allowed = next(iter(stratify.items()))
            allowed = None if allowed is None else list(allowed)
        elif isinstance(stratify, str):
            strat_key = stratify
            allowed = None
        else:
            raise TypeError("stratify must be None, a column name string, or a single-key dict {col: [values]}")

        if strat_key not in tmp.columns:
            raise KeyError(f"Stratify column '{strat_key}' not found in df")

        # temp column name (avoid clobber)
        strat_col = "__summary_strat_col__"
        i = 0
        while strat_col in tmp.columns:
            i += 1
            strat_col = f"__summary_strat_col__{i}"

        base_series = tmp[strat_key]

        # helper: get mapping for this column if one exists (maps built earlier)
        col_map = maps.get(strat_key) if "maps" in locals() else None
        c2l = None
        l2c = None
        if col_map:
            c2l, l2c = col_map  # c2l: code->label, l2c: label->code (may include normalized keys)

        # If allowed is provided, try to convert allowed to match series values:
        if allowed is not None:
            # Build set of actual values present in series (excluding NaN)
            present_vals = set(pd.unique(base_series.dropna().tolist()))

            # If series contains strings (labels) prefer label-matching; else prefer code-matching.
            sample_nonnull = base_series.dropna().iloc[0] if not base_series.dropna().empty else None
            series_is_stringy = isinstance(sample_nonnull, str)

            allowed_matched = set()
            for a in allowed:
                # direct match to present values
                if a in present_vals:
                    allowed_matched.add(a)
                    continue
                # try stringified match
                if str(a) in present_vals:
                    allowed_matched.add(str(a))
                    continue

                # If series stores labels (strings) and we were given codes, convert code -> label via c2l
                if series_is_stringy and c2l is not None:
                    # try numeric / exact code match
                    try:
                        code_key = a
                        # handle string numeric -> int
                        if isinstance(a, str) and a.isdigit():
                            code_key = int(a)
                        if code_key in c2l:
                            allowed_matched.add(c2l[code_key])
                            continue
                        # also try string form of code
                        if str(code_key) in c2l:
                            allowed_matched.add(c2l[str(code_key)])
                            continue
                    except Exception:
                        pass
                    # try normalized label mapping (if a is label but normalized)
                    try:
                        nk = _norm_key(a)
                        # l2c sometimes contains normalized keys; get code then label
                        if l2c and nk in l2c:
                            code = l2c[nk]
                            if code in c2l:
                                allowed_matched.add(c2l[code])
                                continue
                    except Exception:
                        pass

                # If series stores codes and we were given labels, convert label -> code via l2c
                if (not series_is_stringy) and l2c is not None:
                    if a in l2c:
                        allowed_matched.add(l2c[a])
                        continue
                    try:
                        nk = _norm_key(a)
                        if nk in l2c:
                            allowed_matched.add(l2c[nk])
                            continue
                    except Exception:
                        pass

                # otherwise try to coerce numeric strings to numbers and match
                try:
                    if isinstance(a, str) and re.fullmatch(r"[+-]?\d+", a):
                        num = int(a)
                        if num in present_vals:
                            allowed_matched.add(num)
                            continue
                except Exception:
                    pass

            # final allowed set is intersection with present values (defensive)
            allowed_matched = {v for v in allowed_matched if v in present_vals}
            tmp[strat_col] = base_series.where(base_series.isin(allowed_matched))
        else:
            # no allowed filter: use series as-is
            tmp[strat_col] = base_series

        # Force plain python/object dtype to avoid pandas Categorical pitfalls with null categories
        try:
            tmp[strat_col] = tmp[strat_col].astype(object)
        except Exception:
            tmp[strat_col] = tmp[strat_col].apply(lambda v: v if (v is None or (not isinstance(v, float)) or not np.isnan(v)) else None)

        # Group — drop NaNs so missing/NaN strata are ignored
        grouped = tmp.groupby([strat_col], dropna=True, observed=False)
        groups = list(grouped)

    # autodetect categorical traits if not provided
    if categorical_traits is None:
        cat_traits: List[str] = []
        exclude = set(autodetect_exclude or [])
        for c in traits:
            if c in exclude:
                continue
            s = tmp[c]
            if pd.api.types.is_categorical_dtype(s) or pd.api.types.is_object_dtype(s):
                cat_traits.append(c)
                continue
            if autodetect_coded_categoricals and (pd.api.types.is_integer_dtype(s) or pd.api.types.is_float_dtype(s)):
                non_na = s.dropna()
                n_unique = non_na.nunique()
                try:
                    intlike = pd.api.types.is_integer_dtype(s) or (non_na.mod(1).eq(0).all())
                except Exception:
                    intlike = False
                if intlike and 1 < n_unique <= autodetect_max_levels:
                    cat_traits.append(c)
                elif pd.api.types.is_categorical_dtype(s):
                    cat_traits.append(c)
    else:
        cat_traits = list(set(categorical_traits) & set(traits))

    num_traits = [c for c in traits if c not in cat_traits]

    # helpers
    def _pretty_trait_name_from_col(colname: str) -> str:
        suffix = colname.split(".")[-1]
        if data_dict is not None:
            if "name" in data_dict.columns:
                matches = data_dict.loc[data_dict["name"] == suffix]
                if not matches.empty and "description" in matches.columns:
                    desc = matches["description"].dropna()
                    if not desc.empty:
                        return str(desc.iloc[0])
            if "coding_name" in data_dict.columns:
                matches = data_dict.loc[data_dict["coding_name"] == suffix.upper()]
                if not matches.empty and "description" in matches.columns:
                    desc = matches["description"].dropna()
                    if not desc.empty:
                        return str(desc.iloc[0])
        if "_" in suffix:
            return suffix.replace("_", " ").strip().title()
        # split on underscores also for derived like age_at_registration
        return suffix 

    # collect results
    numeric_rows = []
    categorical_rows = []

    for keys, sub in groups:
        keys_tuple = keys if isinstance(keys, tuple) else (keys,)
        size = len(sub)

        # numeric traits
        for c in num_traits:
            col_series = sub[c].astype(object)
            is_listlike = col_series.dropna().apply(_is_listlike_cell).any()
            if not is_listlike:
                coerced = pd.to_numeric(col_series.apply(lambda v: v if not _is_listlike_cell(v) else np.nan), errors="coerce")
                numeric_non_na = coerced.dropna()
                count = int(coerced.count())
                missing_count = size - count
                n_unique = int(coerced.nunique(dropna=True)) if not coerced.empty else 0
                if not numeric_non_na.empty:
                    mean_val = float(numeric_non_na.mean())
                    std_val = float(numeric_non_na.std(ddof=1))  # sample std
                    median_val = float(numeric_non_na.median())
                    min_val = float(numeric_non_na.min())
                    max_val = float(numeric_non_na.max())
                    range_val = max_val - min_val
                    q75 = float(numeric_non_na.quantile(0.75))
                    q25 = float(numeric_non_na.quantile(0.25))
                    iqr_val = q75 - q25
                else:
                    mean_val = std_val = median_val = min_val = max_val = range_val = iqr_val = np.nan

                numeric_rows.append({
                    **({} if not keys_tuple or keys_tuple[0] is None else {"sample": keys_tuple[0]}),
                    "trait": _pretty_trait_name_from_col(c),
                    "coding_name": c,
                    "derived": 1 if isinstance(c, str) and c.startswith("derived.") else 0,
                    "code": pd.NA,
                    "count": count,
                    "n_unique": n_unique,
                    "mean": round(mean_val, round_decimals) if not pd.isna(mean_val) else np.nan,
                    "median": round(median_val, round_decimals) if not pd.isna(median_val) else np.nan,
                    "std": round(std_val, round_decimals) if not pd.isna(std_val) else np.nan,
                    "min": round(min_val, round_decimals) if not pd.isna(min_val) else np.nan,
                    "max": round(max_val, round_decimals) if not pd.isna(max_val) else np.nan,
                    "range": round(range_val, round_decimals) if not pd.isna(range_val) else np.nan,
                    "IQR": round(iqr_val, round_decimals) if not pd.isna(iqr_val) else np.nan,
                    "missing_count": int(missing_count),
                    "missing_proportion": round((missing_count / size) if size else np.nan, round_decimals),
                    "sample_size": int(size),
                })
            else:
                exploded = _explode_listlike(col_series)
                n_categories = int(exploded.nunique()) if not exploded.empty else 0
                count = int(col_series.count())
                missing_count = size - count
                numeric_rows.append({
                    **({} if not keys_tuple or keys_tuple[0] is None else {"sample": keys_tuple[0]}),
                    "trait": _pretty_trait_name_from_col(c),
                    "coding_name": c,
                    "derived": 1 if isinstance(c, str) and c.startswith("derived.") else 0,
                    "code": pd.NA,
                    "count": int(count),
                    "n_unique": int(n_categories),
                    "mean": np.nan,
                    "median": np.nan,
                    "std": np.nan,
                    "min": np.nan,
                    "max": np.nan,
                    "range": np.nan,
                    "IQR": np.nan,
                    "missing_count": int(missing_count),
                    "missing_proportion": round((missing_count / size) if size else np.nan, round_decimals),
                    "sample_size": int(size),
                })

        # categorical traits
        for c in cat_traits:
            raw = sub[c].apply(_normalize_category_label)
            is_listlike = raw.dropna().apply(_is_listlike_cell).any()
            missing_count = int(raw.isna().sum())

            if granularity == "variable":
                if is_listlike:
                    exploded = _explode_listlike(raw)
                    n_categories = int(exploded.nunique()) if not exploded.empty else 0
                    count = int(raw.count())
                    mode = exploded.mode().iloc[0] if (not exploded.empty and not exploded.mode().empty) else pd.NA
                    raw_prop = (count / size) if size else np.nan
                    categorical_rows.append({
                        **({} if not keys_tuple or keys_tuple[0] is None else {"sample": keys_tuple[0]}),
                        "trait": _pretty_trait_name_from_col(c),
                        "coding_name": c,
                        "derived": 1 if isinstance(c, str) and c.startswith("derived.") else 0,
                        "code": pd.NA,
                        "categories": int(n_categories),
                        "count": int(count),
                        "proportion": round((count / size) if size else np.nan, round_decimals),
                        "percent": round(raw_prop * 100, round_decimals),
                        "mode": mode,
                        "missing_count": int(missing_count),
                        "missing_proportion": round((missing_count / size) if size else np.nan, round_decimals),
                        "sample_size": int(size),
                    })
                else:
                    vc = raw.value_counts(dropna=False)
                    n_categories = int(vc.size)
                    count = int(raw.count())
                    mode = vc.idxmax() if n_categories > 0 else pd.NA
                    categorical_rows.append({
                        **({} if not keys_tuple or keys_tuple[0] is None else {"sample": keys_tuple[0]}),
                        "trait": _pretty_trait_name_from_col(c),
                        "coding_name": c,
                        "derived": 1 if isinstance(c, str) and c.startswith("derived.") else 0,
                        "code": pd.NA,
                        "categories": int(n_categories),
                        "count": int(count),
                        "proportion": round((count / size) if size else np.nan, round_decimals),
                        "percent": round(count / size * 100, round_decimals) if size else pd.NA,
                        "mode": mode,
                        "missing_count": int(missing_count),
                        "missing_proportion": round((missing_count / size) if size else np.nan, round_decimals),
                        "sample_size": int(size),
                    })
            else:  # granularity == 'category'
                if is_listlike:
                    exploded = _explode_listlike(raw)
                    vc = exploded.value_counts(dropna=False)
                    for cat, cnt in vc.items():
                        meaning_val = _meaning_for_code(c, cat) if meta is not None else str(cat)
                        categorical_rows.append({
                            **({} if not keys_tuple or keys_tuple[0] is None else {"sample": keys_tuple[0]}),
                            "trait": _normalize_category_label(meaning_val),
                            "coding_name": c,
                            "derived": 1 if isinstance(c, str) and c.startswith("derived.") else 0,
                            "code": cat,
                            "count": int(cnt),
                            "proportion": round((cnt / size) if size else np.nan, round_decimals),
                            "percent": round((cnt / size) * 100, round_decimals) if size else pd.NA,
                            "mode": vc.idxmax() if len(vc) > 0 else pd.NA,
                            "missing_count": int(missing_count),
                            "missing_proportion": round((missing_count / size) if size else np.nan, round_decimals),
                            "sample_size": int(size),
                        })
                else:
                    vc = raw.value_counts(dropna=False)
                    for cat, cnt in vc.items():
                        meaning_val = _meaning_for_code(c, cat) if meta is not None else str(cat)
                        categorical_rows.append({
                            **({} if not keys_tuple or keys_tuple[0] is None else {"sample": keys_tuple[0]}),
                            "trait": meaning_val,
                            "coding_name": c,
                            "derived": 1 if isinstance(c, str) and c.startswith("derived.") else 0,
                            "code": cat,
                            "count": int(cnt),
                            "proportion": round((cnt / size) if size else np.nan, round_decimals),
                            "percent": round((cnt / size) * 100, round_decimals) if size else pd.NA,
                            "mode": vc.idxmax() if len(vc) > 0 else pd.NA,
                            "missing_count": int(missing_count),
                            "missing_proportion": round((missing_count / size) if size else np.nan, round_decimals),
                            "sample_size": int(size),
                        })
                        
    numeric_df = pd.DataFrame(numeric_rows)
    categorical_df = pd.DataFrame(categorical_rows)

    # reorder columns sensibly
    if not numeric_df.empty:
        desired_num_cols = ["trait", "coding_name", "derived", "code", "count", "n_unique",
                            "mean", "median", "std", "min", "max", "range", "IQR",
                            "missing_count", "missing_proportion", "sample", "sample_size"]
        cols_existing = [c for c in desired_num_cols if c in numeric_df.columns]
        numeric_df = numeric_df[cols_existing]

    if not categorical_df.empty:
        desired_cat_cols = ["trait", "coding_name", "derived", "code", "categories", "count",
                            "proportion", "percent", "mode", "missing_count", "missing_proportion",
                            "sample", "sample_size"]
        cols_existing = [c for c in desired_cat_cols if c in categorical_df.columns]
        categorical_df = categorical_df[cols_existing]

    return {"numeric": numeric_df.reset_index(drop=True), "categorical": categorical_df.reset_index(drop=True)}


# ---------------------------------------------------------------------
# Helpers to coerce flexible inputs (for trait_prevalence_using_grouped)
# ---------------------------------------------------------------------

def _coerce_codings_to_df(codings: Any) -> pd.DataFrame:
    """
    Accepts:
      - DataFrame with columns ['coding_name','code','meaning']
      - dict[str, dict[Any,str]]: { CODING_NAME: {code: meaning, ...}, ... }
      - list[dict]: each having keys 'coding_name', 'code', 'meaning'
    """
    if isinstance(codings, pd.DataFrame):
        df = codings.copy()
    elif isinstance(codings, dict):
        rows = []
        for ck, mapping in codings.items():
            for code, meaning in (mapping or {}).items():
                rows.append({"coding_name": str(ck), "code": code, "meaning": meaning})
        df = pd.DataFrame(rows, columns=["coding_name","code","meaning"])
    elif isinstance(codings, (list, tuple)):
        df = pd.DataFrame(list(codings))
    else:
        raise TypeError("codings must be a DataFrame, dict[coding_name -> {code: meaning}], or list of row dicts.")

    df = df.rename(columns={c: c.strip().lower() for c in df.columns})
    required = {"coding_name", "code", "meaning"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"codings is missing required columns: {missing}")
    df["coding_name"] = df["coding_name"].astype(str).str.upper().str.strip()
    return df[["coding_name","code","meaning"]]


def _coerce_traits_to_df(traits: Any, *, default_entity: str = "questionnaire") -> pd.DataFrame:
    """
    Accepts:
      - DataFrame with columns ['trait','coding_name','entity']
      - list[tuple]: [(entity, coding_name, trait), ...] or [(coding_name, trait), ...]
      - list[str]: ["entity.coding_name: trait", "entity.coding_name", ...]
      - dict[str,str]: {"entity.coding_name": "trait", ...} or {coding_name: "trait"} (entity defaults)
    """
    def _parse_item(s: str) -> Tuple[str,str,str]:
        txt = s.strip()
        trait = None
        if ":" in txt:
            left, right = txt.split(":", 1)
            trait = right.strip()
        else:
            left = txt

        left = left.strip()
        if "." in left:
            entity, coding = left.split(".", 1)
        else:
            entity, coding = default_entity, left

        if not trait:
            trait = coding
        return str(entity).strip(), str(coding).strip(), str(trait).strip()

    if isinstance(traits, pd.DataFrame):
        df = traits.copy()
        df = df.rename(columns={c: c.strip().lower() for c in df.columns})
        required = {"trait","coding_name","entity"}
        missing = required - set(df.columns)
        if missing:
            raise ValueError(f"traits is missing required columns: {missing}")
        return df[["trait","coding_name","entity"]]

    rows = []
    if isinstance(traits, dict):
        for k, v in traits.items():
            entity, coding, trait = _parse_item(f"{k}: {v}")
            rows.append({"trait": trait, "coding_name": coding, "entity": entity})
    elif isinstance(traits, (list, tuple)):
        if len(traits) > 0 and isinstance(traits[0], (list, tuple)) and len(traits[0]) in (2,3):
            for tup in traits:
                if len(tup) == 3:
                    entity, coding, trait = tup
                elif len(tup) == 2:
                    coding, trait = tup
                    entity = default_entity
                else:
                    raise ValueError("Trait tuple must be (entity,coding,trait) or (coding,trait).")
                rows.append({"trait": str(trait), "coding_name": str(coding), "entity": str(entity)})
        else:
            for s in traits:
                entity, coding, trait = _parse_item(str(s))
                rows.append({"trait": trait, "coding_name": coding, "entity": entity})
    else:
        raise TypeError("traits must be a DataFrame, list/tuple, list[str], or dict.")

    return pd.DataFrame(rows, columns=["trait","coding_name","entity"])


def _infer_traits_from_df(df: pd.DataFrame, codings_df: pd.DataFrame,
                          entities: Iterable[str] = ("participant", "questionnaire")) -> pd.DataFrame:
    """
    Auto-build a traits DataFrame from the columns in `df` and meanings in `codings_df`.

    For each column starting with 'participant.' or 'questionnaire.' whose suffix matches a
    coding_name in the codings file, emit one row per meaning:
      {'trait': <meaning>, 'coding_name': <CODING_NAME>, 'entity': <entity>}
    """
    codings_norm = codings_df.copy()
    codings_norm = codings_norm.rename(columns={c: c.strip().lower() for c in codings_norm.columns})
    codings_norm["coding_name"] = codings_norm["coding_name"].astype(str).str.upper().str.strip()

    meanings_by_ck: Dict[str, List[str]] = {}
    for _, r in codings_norm.iterrows():
        ck = str(r["coding_name"])
        meanings_by_ck.setdefault(ck, []).append(str(r["meaning"]))

    rows = []
    entities = set(e.strip().lower() for e in entities)
    for col in df.columns:
        cl = col.lower()
        if cl.startswith("participant."):
            entity = "participant"
        elif cl.startswith("questionnaire."):
            entity = "questionnaire"
        else:
            continue
        if entity not in entities:
            continue

        ck = col.split(".")[-1].upper()
        meanings = meanings_by_ck.get(ck)
        if not meanings:
            continue
        for m in meanings:
            rows.append({"trait": m, "coding_name": ck, "entity": entity})

    return pd.DataFrame(rows, columns=["trait","coding_name","entity"])


# ---------------------------------------------------------------------
# Prevalence using grouped stats (list-aware, robust meaning matching)
# ---------------------------------------------------------------------

def prevalence(
    df: pd.DataFrame,
    codings: Any | None = None,
    traits: Any | None = None,
    *,
    denominator: Literal["all", "nonmissing"] = "all",
    denominators: Optional[Iterable[str]] = None,
    eligibility: Optional[Dict[str, Union[str, List[str]]]] = None,
    wide_output: bool = True,
    participant_col: str = "participant.pid",
    metadata_dir: str = "./metadata",
    codebook_csv: Optional[str] = None,
    on_missing: Literal["warn", "ignore", "error"] = "warn",
    error_if_empty: bool = False,
) -> pd.DataFrame:
    """
    Compute prevalence including support for derived.* columns, a 'meaning' column,
    skip columns ending with '.pid', and add a binary 'derived' flag (1/0).

    NOTE: coding_name in results is always the original column name (entity.coding_name),
    not the uppercase codebook key.
    """
    import logging
    logger = logging.getLogger(__name__)
    import numpy as np
    from pathlib import Path

    # participant id fallback
    if participant_col in df.columns:
        _pid_col = participant_col
        _df = df
    else:
        _df = df.copy()
        temp_col = "__row_index_for_counts"
        i = 0
        while temp_col in _df.columns:
            i += 1
            temp_col = f"__row_index_for_counts_{i}"
        _df[temp_col] = range(len(_df))
        _pid_col = temp_col
        logger.warning("trait_prevalence: participant_col %r not found; using %r.", participant_col, _pid_col)

    # load codings/codebook if available
    codings_df = None
    if codings is not None:
        codings_df = _coerce_codings_to_df(codings)
    else:
        path_to_use = None
        if codebook_csv and Path(codebook_csv).exists():
            path_to_use = Path(codebook_csv)
        else:
            matches = sorted(Path(metadata_dir).glob("*.codings.csv"))
            if matches:
                path_to_use = max(matches, key=lambda p: p.stat().st_mtime)
        if path_to_use and Path(path_to_use).exists():
            codings_df = pd.read_csv(path_to_use)
            codings_df = _coerce_codings_to_df(codings_df)

    # pick columns
    if traits is None:
        cols_to_count = [c for c in _df.columns if not c.startswith("__")]
    else:
        cols_to_count = [c for c in traits if c in _df.columns]

    # skip any column ending with '.pid'
    cols_to_count = [c for c in cols_to_count if not str(c).lower().endswith(".pid")]

    # helper: continuous float guard
    def _is_continuous_float(s: pd.Series) -> bool:
        try:
            if pd.api.types.is_float_dtype(s.dtype):
                return s.nunique(dropna=True) > 15
        except Exception:
            pass
        return False

    cols_to_count = [c for c in cols_to_count if not _is_continuous_float(_df[c])]

    out_rows = []
    denom_all = int(_df[_pid_col].dropna().nunique())

    # codebook-driven columns (non-derived)
    if codings_df is not None:
        lookup_exact, by_coding = build_code_lookup(codings_df)
        present_codings = set(codings_df["coding_name"].astype(str).str.upper().str.strip())

        for col in cols_to_count:
            # skip derived here (handled below)
            if isinstance(col, str) and col.startswith("derived."):
                continue
            ck = col.split(".")[-1].upper()
            if ck not in present_codings:
                continue
            s = _df[[_pid_col, col]].copy()
            s[col] = s[col].astype(object).apply(_coerce_cell_to_sequence)
            s = s.explode(col).dropna(subset=[col])
            if s.empty:
                continue
            s["code_str"] = s[col].apply(lambda v: str(v).strip())
            grp = s.groupby("code_str", dropna=False)[_pid_col].nunique()
            trait = col.split(".")[-1]
            # <<-- keep coding_name as the original column name (entity.coding_name), not uppercase CK
            coding_name = col
            for code_str, cnt in grp.items():
                out_rows.append({
                    "trait": trait,
                    "coding_name": coding_name,
                    "code": code_str,
                    "count": int(cnt),
                    "__col__": col,
                    "derived": 0,
                })

    # derived.* and remaining raw categorical columns
    already_done = {r["__col__"] for r in out_rows}
    for col in cols_to_count:
        if col in already_done:
            continue

        # derived.* columns: coding_name should be full column name (e.g., 'derived.sex'), derived=1
        if isinstance(col, str) and col.startswith("derived."):
            trait = col.split(".")[-1]
            # <<-- use full column name for coding_name (entity.coding_name style) rather than uppercasing
            coding_name = col
            s = _df[col]
            # skip continuous by guard
            if _is_continuous_float(s):
                continue
            vc = s.value_counts(dropna=False)
            for cat, cnt in vc.items():
                out_rows.append({
                    "trait": trait,
                    "coding_name": coding_name,
                    "code": cat,
                    "count": int(cnt),
                    "__col__": col,
                    "derived": 1,
                })
            continue

        # raw categorical (no codebook): coding_name = full column name, derived=0
        s = _df[col]
        if _is_continuous_float(s):
            continue
        vc = s.value_counts(dropna=False)
        trait = col.split(".")[-1]
        coding_name = col
        for cat, cnt in vc.items():
            out_rows.append({
                "trait": trait,
                "coding_name": coding_name,
                "code": cat,
                "count": int(cnt),
                "__col__": col,
                "derived": 0,
            })

    base = pd.DataFrame(out_rows, columns=["trait", "coding_name", "code", "count", "__col__", "derived"])
    if base.empty:
        if error_if_empty:
            raise ValueError("trait_prevalence: empty result (no matched traits/columns or all counts were zero).")
        if denominators:
            if wide_output:
                return pd.DataFrame(columns=["trait", "coding_name", "code", "meaning", "count", "derived"])
            else:
                return pd.DataFrame(columns=["trait", "coding_name", "code", "meaning", "count", "derived", "denominator_type", "denominator", "prevalence"])
        else:
            return pd.DataFrame(columns=["trait", "coding_name", "code", "meaning", "count", "derived", "prevalence", "denominator"])

    # add meaning column
    def _lookup_meaning_for_row(r):
        coding = r["coding_name"]
        code = r["code"]
        # try codebook: codings_df stores CK as uppercase coding_name
        if codings_df is not None:
            # case 1: coding is CK (uppercase) from codebook
            ck_try = str(coding).upper()
            match = codings_df[codings_df["coding_name"] == ck_try]
            if not match.empty:
                possible = match.loc[match["code"] == code]
                if not possible.empty:
                    return str(possible["meaning"].values[0])
                possible = match.loc[match["code"].astype(str) == str(code)]
                if not possible.empty:
                    return str(possible["meaning"].values[0])
            # case 2: coding was stored as full column name; try suffix -> ck
            if "." in str(coding):
                ck_try2 = str(coding).split(".")[-1].upper()
                match2 = codings_df[codings_df["coding_name"] == ck_try2]
                if not match2.empty:
                    possible = match2.loc[match2["code"] == code]
                    if not possible.empty:
                        return str(possible["meaning"].values[0])
                    possible = match2.loc[match2["code"].astype(str) == str(code)]
                    if not possible.empty:
                        return str(possible["meaning"].values[0])

        # derived codebook: look up by derived column name (e.g., 'derived.sex')
        derived_key = None
        try:
            if isinstance(r["__col__"], str) and r["__col__"].startswith("derived."):
                derived_key = r["__col__"]
            else:
                derived_key = f"derived.{r['trait']}"
        except Exception:
            derived_key = None

        if derived_key and derived_key in globals().get("DERIVED_CODEBOOK", {}):
            mapping = globals()["DERIVED_CODEBOOK"].get(derived_key, {})
            # direct match
            if code in mapping:
                return str(mapping[code])
            if str(code) in mapping:
                return str(mapping[str(code)])
        return ""

    base["meaning"] = base.apply(_lookup_meaning_for_row, axis=1)

    # denominators
    denom_nonmissing = {col: int(_df.loc[_df[col].notna(), _pid_col].dropna().nunique()) for col in base["__col__"].unique()}

    eligibility = eligibility or {}
    elig_cols_map = {k: [v] if isinstance(v, str) else list(v) for k, v in eligibility.items()}
    elig_denoms = {}
    for name, cols in elig_cols_map.items():
        mask = pd.Series(False, index=_df.index)
        for c in cols:
            if c in _df.columns:
                mask = mask | _df[c].astype(object).map(lambda v: bool(_coerce_cell_to_sequence(v)))
        elig_denoms[name] = int(_df.loc[mask, _pid_col].dropna().nunique())

    def _denom_for(colname: str, denom_key: str) -> int:
        if denom_key == "all":
            return denom_all
        if denom_key == "nonmissing":
            return int(denom_nonmissing.get(colname, 0))
        if denom_key in elig_denoms:
            return int(elig_denoms[denom_key])
        return denom_all

    # format result
    if not denominators:
        base["denominator"] = base["__col__"].map(lambda c: _denom_for(c, denominator))
        base["prevalence"] = base.apply(lambda r: (r["count"] / r["denominator"]) if r["denominator"] > 0 else 0.0, axis=1)
        out = base.drop(columns=["__col__"])
        # ensure ordering
        return out[["trait", "coding_name", "derived", "code", "meaning", "count", "prevalence", "denominator"]]

    denoms = list(denominators)
    if wide_output:
        out = base.drop(columns=["__col__"]).copy()
        for key in denoms:
            key = str(key)
            denom_vals = base["__col__"].map(lambda c: _denom_for(c, key))
            out[f"denominator_{key}"] = denom_vals.values
            out[f"prevalence_{key}"] = (base["count"] / denom_vals.replace(0, np.nan)).fillna(0.0).values
        fixed = ["trait", "coding_name", "derived", "code", "meaning", "count"]
        dyn = [c for c in out.columns if c not in fixed]
        return out[fixed + sorted(dyn)]
    else:
        rows = []
        for key in denoms:
            key = str(key)
            denom_vals = base["__col__"].map(lambda c: _denom_for(c, key))
            prev_vals = (base["count"] / denom_vals.replace(0, np.nan)).fillna(0.0)
            part = base.drop(columns=["__col__"]).copy()
            part["denominator_type"] = key
            part["denominator"] = denom_vals.values
            part["prevalence"] = prev_vals.values
            rows.append(part)
        out = pd.concat(rows, ignore_index=True)
        return out[["trait", "coding_name", "derived", "code", "meaning", "count", "denominator_type", "denominator", "prevalence"]]
    
# ---------------------------------------------------------------------
# Medication-bundle across medicat_* columns
# ---------------------------------------------------------------------
 
def _coerce_medication_phenotypes_to_df(meds: Any) -> pd.DataFrame:
    """
    Coerce flexible medication phenotype specifications into a DataFrame with columns:
      ['trait', 'coding_name', 'medication']

    Supported inputs:
      - pd.DataFrame with those columns already present
      - list/tuple of tuples: (coding_name, medication, trait) or (coding_name, medication)
          * if (coding_name, medication) provided, trait defaults to medication string
          * medication may be a scalar or a list (will be exploded later)
      - list of dicts with keys 'trait','coding_name','medication'
      - dict mapping coding_name -> list_of_medication_labels (trait default = coding_name)
      - dict mapping trait -> {'coding_name': ck, 'medication': med_label_or_list}
    """
    rows = []

    if isinstance(meds, pd.DataFrame):
        df = meds.copy()
        df = df.rename(columns={c: c.strip() for c in df.columns})
        required = {"trait", "coding_name", "medication"}
        if not required.issubset(set(df.columns)):
            raise ValueError("medication_phenotypes DataFrame must contain columns: 'trait','coding_name','medication'")
        return df[["trait", "coding_name", "medication"]].copy()

    if isinstance(meds, dict):
        # two flavors:
        # 1) coding_name -> [med1, med2,...]
        # 2) trait -> {'coding_name': ck, 'medication': med_or_list}
        for k, v in meds.items():
            if isinstance(v, dict) and "coding_name" in v:
                trait = str(k)
                ck = str(v["coding_name"])
                meds_val = v.get("medication", [])
                rows.append({"trait": trait, "coding_name": ck, "medication": meds_val})
            else:
                # treat key as coding_name, value as list/str of medications; trait defaults to coding_name
                ck = str(k)
                trait = ck
                rows.append({"trait": trait, "coding_name": ck, "medication": v})
        return pd.DataFrame(rows, columns=["trait", "coding_name", "medication"])

    if isinstance(meds, (list, tuple)):
        for item in meds:
            if isinstance(item, dict):
                if not {"trait", "coding_name", "medication"}.issubset(set(item.keys())):
                    raise ValueError("Each dict in medication_phenotypes list must have keys 'trait','coding_name','medication'")
                rows.append({"trait": str(item["trait"]), "coding_name": str(item["coding_name"]), "medication": item["medication"]})
                continue
            if isinstance(item, (list, tuple)):
                if len(item) == 3:
                    ck, med, trait = item
                elif len(item) == 2:
                    ck, med = item
                    trait = med if not isinstance(med, (list, tuple)) else ck
                else:
                    raise ValueError("Tuple items must be (coding_name, medication, trait) or (coding_name, medication)")
                rows.append({"trait": str(trait), "coding_name": str(ck), "medication": med})
                continue
            # scalar string: interpret as coding_name and no meds (user likely made mistake)
            raise ValueError("Unsupported item in medication_phenotypes list/tuple; use tuple/dict forms.")
    else:
        raise TypeError("medication_phenotypes must be a DataFrame, dict, or list/tuple of tuples/dicts.")

    return pd.DataFrame(rows, columns=["trait", "coding_name", "medication"])


def medication_prevalence(
    df: pd.DataFrame,
    codings: Any | None,
    medication_phenotypes: Any,
    *,
    participant_col: str = "participant.pid",
    denominator: Literal["all", "nonmissing"] = "all",
    return_what: Literal["both", "per_medication", "group"] = "both",
    fuzzy: bool = True,
    fuzzy_cutoff: float = 0.82,
    metadata_dir: str = "./metadata",
    codebook_csv: Optional[str] = None,
):
    """
    Compute medication prevalence.

    Parameters
    ----------
    df : pd.DataFrame
        Questionnaire/participant dataframe (fully-qualified columns).
    codings : DataFrame/dict/list or None
        Codings mapping. If None, function will try to resolve a '*.codings.csv'
        from metadata_dir or use codebook_csv if provided.
    medication_phenotypes : flexible (DataFrame / list / dict)
        Specification of medication phenotypes. Coerced into rows of (trait, coding_name, medication).
    Other args: denominator, return_what, fuzzy matching, etc.

    Returns
    -------
    per_med_df, group_df   (or one of them depending on return_what)
    """
    logger = logging.getLogger(__name__)

    # --- resolve codings if not provided (same fallback logic as trait_prevalence_using_grouped) ---
    if codings is None:
        path_to_use = None
        if codebook_csv and Path(codebook_csv).exists():
            path_to_use = Path(codebook_csv)
        else:
            candidates = sorted(Path(metadata_dir).glob("*.codings.csv"), key=lambda p: p.stat().st_mtime, reverse=True)
            if candidates:
                path_to_use = candidates[0]
            else:
                try:
                    from . import load as _load_mod
                except Exception:
                    try:
                        import phenofhy.load as _load_mod
                    except Exception as e:
                        raise RuntimeError(
                            "Could not import phenofhy.load to fetch metadata. Ensure phenofhy.load is importable."
                        ) from e
                meta_files = _load_mod.metadata()
                path_to_use = Path(meta_files["codings"]) if "codings" in meta_files else None

        if not path_to_use or not Path(path_to_use).exists():
            raise FileNotFoundError(
                "No codings were provided and a codings CSV could not be found or fetched. "
                f"Tried metadata_dir={metadata_dir!r} and codebook_csv={codebook_csv!r}."
            )
        codings = pd.read_csv(path_to_use)

    # coerce codings
    codings_df = _coerce_codings_to_df(codings)

    # coerce medication_phenotypes
    meds_df = _coerce_medication_phenotypes_to_df(medication_phenotypes)

    # safety guards
    req = {"trait", "medication", "coding_name"}
    if not req.issubset(set(meds_df.columns)):
        raise ValueError("medication_phenotypes must result in columns: 'trait','coding_name','medication'")

    if participant_col not in df.columns:
        raise KeyError(f"participant column '{participant_col}' not found in df")

    # helpers
    def _ensure_list_of_strings(x) -> List[str]:
        if x is None or (isinstance(x, float) and pd.isna(x)):
            return []
        if isinstance(x, (list, tuple, set, np.ndarray)):
            return [str(v) for v in list(x)]
        s = str(x).strip()
        if s == "" or s.lower() in {"none", "na", "nan", "null"}:
            return []
        try:
            v = ast.literal_eval(s)
            if isinstance(v, (list, tuple, set, np.ndarray)):
                return [str(u) for u in list(v)]
            return [str(v)]
        except Exception:
            if any(d in s for d in (",", "|", ";")):
                return [t for t in re.split(r"[,\|;]\s*", s) if t != ""]
            return [s]

    def _as_code_str(x):
        if isinstance(x, (int, np.integer)):
            return str(int(x))
        if isinstance(x, (float, np.floating)) and pd.notna(x) and float(x).is_integer():
            return str(int(x))
        return str(x)

    # build lookup from codings
    _, by_coding = build_code_lookup(codings_df)

    def _resolve_codes_for_meaning(ck: str, meaning_text: str) -> List[Any]:
        cknorm = _norm_key(ck)
        want = _norm_key(meaning_text)
        choices = by_coding.get(cknorm, {})
        if not choices:
            return []
        if want in choices:
            return [choices[want]]
        picked = None
        if fuzzy:
            best = difflib.get_close_matches(want, list(choices.keys()), n=1, cutoff=fuzzy_cutoff)
            if best:
                picked = choices[best[0]]
        if picked is not None:
            return [picked]
        subs = [choices[k] for k in choices if want in k]
        return subs

    # denominators
    denom_all = int(df[participant_col].dropna().nunique())

    def _denom_for(colname: str) -> int:
        if denominator == "all":
            return denom_all
        mask = df[colname].notna()
        return int(df.loc[mask, participant_col].dropna().nunique())

    # explode meds rows so each row has single medication string
    work = meds_df.copy()
    work["medication"] = work["medication"].apply(_ensure_list_of_strings)
    work = work.explode("medication", ignore_index=True)

    per_med_rows = []
    trait_ck_to_pidset: Dict[Tuple[str, str], set] = {}

    for _, r in work.iterrows():
        trait = str(r["trait"])
        med_label = str(r["medication"])
        ck = str(r["coding_name"])

        # find questionnaire column for this coding_name
        try:
            col = find_df_column_for_coding(df, ck, entity="questionnaire")
        except KeyError:
            base = denom_all if denominator == "all" else 0
            per_med_rows.append({"trait": trait, "medication": med_label, "coding_name": ck,
                                 "count": 0, "prevalence": 0.0, "denominator": int(base)})
            continue
        if col not in df.columns:
            base = denom_all if denominator == "all" else 0
            per_med_rows.append({"trait": trait, "medication": med_label, "coding_name": ck,
                                 "count": 0, "prevalence": 0.0, "denominator": int(base)})
            continue

        # resolve code(s) corresponding to medication meaning
        codes = _resolve_codes_for_meaning(ck, med_label)
        code_set = {_as_code_str(c) for c in codes}
        denom = _denom_for(col)

        if not code_set or denom == 0:
            per_med_rows.append({"trait": trait, "medication": med_label, "coding_name": ck,
                                 "count": 0, "prevalence": 0.0, "denominator": int(denom)})
            continue

        s = df[col].apply(_coerce_cell_to_sequence)
        if not s.apply(bool).any():
            per_med_rows.append({"trait": trait, "medication": med_label, "coding_name": ck,
                                 "count": 0, "prevalence": 0.0, "denominator": int(denom)})
            continue

        tmp = (
            pd.DataFrame({participant_col: df[participant_col], "val": s})
            .explode("val")
            .dropna(subset=["val"])
        )
        if tmp.empty:
            per_med_rows.append({"trait": trait, "medication": med_label, "coding_name": ck,
                                 "count": 0, "prevalence": 0.0, "denominator": int(denom)})
            continue

        tmp["code_str"] = tmp["val"].apply(_as_code_str)
        hit_pids = tmp.loc[tmp["code_str"].isin(code_set), participant_col].dropna().unique()

        cnt = int(len(hit_pids))
        prev = (cnt / denom) if denom > 0 else 0.0
        per_med_rows.append({
            "trait": trait,
            "medication": med_label,
            "coding_name": ck,
            "count": cnt,
            "prevalence": prev,
            "denominator": int(denom),
        })

        if cnt > 0:
            trait_ck_to_pidset.setdefault((trait, ck), set()).update(hit_pids)

    per_med_df = (
        pd.DataFrame(per_med_rows, columns=["trait","medication","coding_name","count","prevalence","denominator"])
        .sort_values(["trait","coding_name","medication"])
        .reset_index(drop=True)
    )

    if return_what == "per_medication":
        return per_med_df

    # aggregate to trait-level groups (union across medications for same (trait, coding_name))
    group_rows = []
    for (trait, ck), pidset in trait_ck_to_pidset.items():
        try:
            col = find_df_column_for_coding(df, ck, entity="questionnaire")
        except KeyError:
            continue
        denom = _denom_for(col)
        cnt = int(len(pidset))
        prev = (cnt / denom) if denom > 0 else 0.0
        group_rows.append({
            "trait": trait,
            "coding_name": ck,
            "count": cnt,
            "prevalence": prev,
            "denominator": int(denom),
        })

    group_df = (
        pd.DataFrame(group_rows, columns=["trait","coding_name","count","prevalence","denominator"])
        .sort_values(["trait","coding_name"])
        .reset_index(drop=True)
    )

    if return_what == "group":
        return group_df

    return per_med_df, group_df


from typing import Optional, Dict, Iterable, Union, Tuple
import pandas as pd
import numpy as np
import logging

logger = logging.getLogger(__name__)

# assume DEFAULT_MEDICAT_GROUP_MAP and _is_listlike_cell are defined elsewhere in your module

def medication_summary(
    df,
    *,
    med_prefix: str = "derived.medicates_",
    group_map: Optional[Dict[str, Iterable[str]]] = DEFAULT_MEDICAT_GROUP_MAP,
    inplace: bool = True,
    return_summary: bool = False,
) -> Union[pd.DataFrame, Tuple[pd.DataFrame, pd.DataFrame]]:
    """
    Derive medication usage-pattern variables from binary domain flags and
    optionally return a compact summary DataFrame describing the new variables.

    Notes
    -----
    - Accepts either a pandas DataFrame or a (mapping_df, df) tuple (to be robust
      against callers that previously returned (mapping, df) tuples).
    - If `inplace=True` the supplied DataFrame is mutated and also returned. If
      False, a shallow copy is created and returned.
    - When `return_summary=True`, a second returned object is a DataFrame with
      one row per derived variable and the following columns:
        - var: column name
        - meaning: short human-friendly description
        - var_type: 'binary' or 'continuous'
        - mean: numeric mean (or proportion for binary)
        - std: sample std (NaN for many binaries)
        - mode: most frequent value (where sensible)
        - n_nonmissing: number of non-missing values
        - n_missing: number of missing rows
    """
    # Robustly accept (mapping_df, df) tuples
    if isinstance(df, tuple) and len(df) == 2 and isinstance(df[1], pd.DataFrame):
        # treat second element as df (common caller pattern)
        _, df = df

    if not inplace:
        df = df.copy()

    gm = group_map or {}

    # detect med domain columns
    med_cols = sorted([c for c in df.columns if isinstance(c, str) and c.startswith(med_prefix)])
    if med_cols:
        # coerce to numeric robustly: treat non-numeric as NaN -> fillna(0) -> int
        # If you want to preserve NA semantics instead, consider pandas "Int64" dtype (nullable integer)
        df[med_cols] = df[med_cols].apply(pd.to_numeric, errors="coerce").fillna(0).astype(int)
    else:
        logger.debug("medication_summary: no columns found with prefix %s", med_prefix)

    # core counts/flags
    if med_cols:
        df["derived.num_meds_domains"] = df[med_cols].sum(axis=1)
    else:
        df["derived.num_meds_domains"] = pd.Series(0, index=df.index, dtype=int)

    df["derived.any_meds_flag"] = (df["derived.num_meds_domains"] > 0).astype(int)
    df["derived.polypharmacy_flag"] = (df["derived.num_meds_domains"] >= 3).astype(int)

    # grouped theme flags (ensure missing component cols created as zeros)
    for gname, cols in gm.items():
        # prefer fully-qualified column names in mapping; accept partial names too
        cols_existing = [c for c in cols if c in df.columns]
        missing = [c for c in cols if c not in df.columns]
        if missing:
            # create zeros for missing components (safe, deterministic)
            for c in missing:
                df[c] = pd.Series(0, index=df.index, dtype=int)
            cols_existing = cols  # now effectively present
        # max across components -> binary indicator
        df[gname] = df[cols_existing].max(axis=1).astype(int)

    # --- multi-system: use the grouped columns created above ---
    # derive system column list from the group_map keys that were created as grouped flags
    # (prefer group_map keys that are present in df after the grouping step)
    system_cols = [gname for gname in (gm.keys() if gm is not None else []) if isinstance(gname, str) and (gname in df.columns)]
    # fallback: if none found, as a last resort look for any df cols that start with med_prefix_group style
    # (optional — keep this only if you expect other naming patterns)
    # if not system_cols:
    #     system_cols = [c for c in df.columns if isinstance(c, str) and c in gm.keys()]

    if system_cols:
        df["derived.n_systems_used"] = df[system_cols].sum(axis=1).astype(int)
    else:
        df["derived.n_systems_used"] = pd.Series(0, index=df.index, dtype=int)

    df["derived.multi_med_system_use"] = (df["derived.n_systems_used"] >= 2).astype(int)

    # proportion: proportion of *system groups* used  (0.0 - 1.0)
    # use the same system_cols computed above
    n_systems_total = len(system_cols)
    if n_systems_total > 0:
        df["derived.prop_systems_used"] = (df["derived.n_systems_used"] / float(n_systems_total)).astype(float)
    else:
        df["derived.prop_systems_used"] = pd.Series(0.0, index=df.index, dtype=float)

    # If no summary requested, return df (mutated or copy depending on inplace)
    if not return_summary:
        return df

    # ------------------------
    # Build compact summary DF
    # ------------------------
    # variables to include in summary (order matters)
    summary_vars = [
        "derived.num_meds_domains",
        "derived.any_meds_flag",
        "derived.polypharmacy_flag",
        "derived.medicates_cvd_diab",
        "derived.medicates_mental_pain",
        "derived.medicates_auto_endo",
        "derived.medicates_bone_supp",
        "derived.multi_med_system_use",
        "derived.prop_systems_used",
    ]

    # human-friendly meanings (updated multi-system meaning)
    meaning_map = {
        "derived.num_meds_domains": "Number of medication domains used",
        "derived.any_meds_flag": "Any medication use (≥1 domain)",
        "derived.polypharmacy_flag": "Polypharmacy (≥3 domains)",
        "derived.medicates_cvd_diab": "Use of cardiometabolic medications (cardio or diabetes)",
        "derived.medicates_mental_pain": "Use of mental-health or pain medications",
        "derived.medicates_auto_endo": "Use of immune / inflammatory system medications",
        "derived.medicates_bone_supp": "Use of supplements / bone / nutritional medications",
        "derived.multi_med_system_use": "Use across ≥2 medication system groups",
        "derived.prop_systems_used": "Proportion of *system groups* used (0–1)",
    }

    rows = []
    for var in summary_vars:
        if var not in df.columns:
            # skip absent vars but include a zero/NA-style row to make it explicit (optional)
            rows.append({
                "var": var,
                "meaning": meaning_map.get(var, ""),
                "var_type": "missing",
                "mean": np.nan,
                "std": np.nan,
                "mode": pd.NA,
                "sum": 0,
                "n_nonmissing": 0,
                "n_missing": len(df),
            })
            continue

        ser = df[var]
        n_nonmissing = int(ser.notna().sum())
        n_missing = int(len(ser) - n_nonmissing)

        # infer var type: binary if values subset of {0,1} (ignoring NA) or nunique<=2
        non_na_vals = ser.dropna()
        is_binary = False
        try:
            uniq = set(non_na_vals.astype(object).unique().tolist())
            if len(uniq) <= 2 and uniq <= {0, 1}:
                is_binary = True
        except Exception:
            # fallback: nunique check
            is_binary = (non_na_vals.nunique(dropna=True) <= 2)

        if is_binary:
            mean = float(non_na_vals.mean()) if n_nonmissing > 0 else np.nan
            std = float(non_na_vals.std(ddof=1)) if n_nonmissing > 1 else np.nan
            # mode: most frequent value (prefer 1 for proportion > 0.5)
            try:
                mode_val = non_na_vals.mode().iloc[0] if not non_na_vals.empty else pd.NA
            except Exception:
                mode_val = pd.NA
            sum_pos = int((non_na_vals == 1).sum()) if n_nonmissing > 0 else 0
            var_type = "binary"
        else:
            coerced = pd.to_numeric(ser.apply(lambda v: v if not _is_listlike_cell(v) else np.nan), errors="coerce")
            non_na_num = coerced.dropna()
            if not non_na_num.empty:
                mean = float(non_na_num.mean())
                std = float(non_na_num.std(ddof=1)) if len(non_na_num) > 1 else np.nan
                try:
                    mode_val = non_na_num.mode().iloc[0]
                except Exception:
                    mode_val = pd.NA
            else:
                mean = std = np.nan
                mode_val = pd.NA
            sum_pos = int((ser == 1).sum()) if ser.dtype.kind in ("i", "u", "b") else pd.NA
            var_type = "continuous" if coerced.notna().sum() > 0 else "other"

        rows.append({
            "var": var,
            "meaning": meaning_map.get(var, ""),
            "var_type": var_type,
            "mean": (round(mean, 6) if (mean is not None and not (pd.isna(mean))) else np.nan),
            "std": (round(std, 6) if (std is not None and not (pd.isna(std))) else np.nan),
            "mode": (mode_val if mode_val is not None else pd.NA),
            "sum": sum_pos,
            "n_nonmissing": n_nonmissing,
            "n_missing": n_missing,
        })

    summary_df = pd.DataFrame(rows, columns=["var", "meaning", "var_type", "mean", "std", "mode", "sum", "n_nonmissing", "n_missing"])
    return (df, summary_df) if return_summary else df



# ---------------------------------------------------------------------
# Specialised analysis funcs and dependencies
# ---------------------------------------------------------------------

def _binarize_series(s: pd.Series):
    """
    Map series to (mapped_series, is_binary_flag).
    - If the series has 0 non-NA uniques -> return all-zeros (binary).
    - If 1 unique -> map that value -> 0 (binary).
    - If 2 uniques -> deterministic mapping smaller->0, larger->1 (binary).
      Deterministic ordering tries numeric ordering, falls back to string ordering.
    - Otherwise returns original series and False.
    """
    # dropna uniques
    uniq_vals = pd.Series(s.dropna().unique())
    n = uniq_vals.size
    if n == 0:
        return pd.Series(np.zeros(len(s), dtype=int), index=s.index), True
    if n == 1:
        mapping = {uniq_vals.iloc[0]: 0}
        return s.map(mapping).fillna(0).astype(int), True
    if n == 2:
        # try numeric ordering first
        try:
            sorted_vals = sorted(uniq_vals.tolist())
        except Exception:
            # fall back to deterministic string ordering
            sorted_vals = sorted(uniq_vals.astype(str).tolist())
            # reconstruct original typed values where possible
            # but mapping by string representation is acceptable here
        mapping = {sorted_vals[0]: 0, sorted_vals[1]: 1}
        # attempt to map by equality to original objects; if that fails, map by string
        mapped = s.map(lambda v: mapping.get(v, mapping.get(str(v), None)))
        return mapped.fillna(0).astype(int), True
    return s, False


def matthews_corrcoef_series(a: pd.Series, b: pd.Series) -> float:
    """
    Compute Matthews correlation coefficient (phi) between two pandas Series.

    - Both inputs are binarized via _binarize_series (if they have <=2 unique non-NA values).
    - If either is not binary after checking, raises ValueError.
    - If denominator of phi formula is zero (degenerate confusion matrix),
      attempt to return Pearson on the binary-mapped arrays; if that fails, return 0.0.
    """
    a_bin, a_is_bin = _binarize_series(a)
    b_bin, b_is_bin = _binarize_series(b)

    if not (a_is_bin and b_is_bin):
        raise ValueError("Both series must be binary-like (<=2 unique values) to compute Matthews phi.")

    a_arr = a_bin.fillna(0).astype(int).to_numpy()
    b_arr = b_bin.fillna(0).astype(int).to_numpy()

    # confusion matrix entries
    TP = int(np.sum((a_arr == 1) & (b_arr == 1)))
    TN = int(np.sum((a_arr == 0) & (b_arr == 0)))
    FP = int(np.sum((a_arr == 0) & (b_arr == 1)))
    FN = int(np.sum((a_arr == 1) & (b_arr == 0)))

    num = TP * TN - FP * FN
    den = (TP + FP) * (TP + FN) * (TN + FP) * (TN + FN)
    if den == 0:
        # degenerate; attempt Pearson on mapped arrays (point-biserial equivalence)
        try:
            p = float(pd.Series(a_arr).corr(pd.Series(b_arr)))
            if np.isfinite(p):
                return float(p)
        except Exception:
            pass
        return 0.0
    return float(num / math.sqrt(den))


import os
from typing import Optional, Iterable
import numpy as np
import pandas as pd


def phi_corr(
    df: pd.DataFrame,
    vars_for_heatmap: Optional[Iterable[str]] = None,
    *,
    med_prefix: str = "derived.medicates_",
    outdir: Optional[str] = None,
    save_basename: str = "phi_corr",
) -> pd.DataFrame:
    """
    Compute a symmetric Pearson correlation matrix suitable for heatmaps.

    - Uses Pearson's r for every pair (this reproduces Pearson for continuous-continuous,
      point-biserial for continuous-binary, and Phi for binary-binary when binaries are 0/1).
    - Each pairwise correlation is computed on the intersection of non-missing samples.
    - If the pairwise overlap has < 2 samples or either variable is constant on the overlap,
      the correlation is set to 0.0 (stable, avoids NaN).
    - If vars_for_heatmap is None, default: all med_prefix cols + common usage-pattern vars (if present).
    - Returns DataFrame (index & columns = vars_for_heatmap) of floats in [-1, 1].
    """
    # choose variables if not provided
    if vars_for_heatmap is None:
        med_cols = sorted([c for c in df.columns if isinstance(c, str) and c.startswith(med_prefix)])
        usage_pattern_vars = [
            "derived.medicates_cvd_diab", "derived.medicates_mental_pain", "derived.medicates_auto_endo",
            "derived.medicates_bone_supp", "derived.polypharmacy_flag", "derived.num_meds_domains"
        ]
        vars_for_heatmap = med_cols + [v for v in usage_pattern_vars if v in df.columns]
    else:
        vars_for_heatmap = list(vars_for_heatmap)

    # if any requested variable is missing, create an all-NaN series (so pairwise overlap will be empty)
    missing = [v for v in vars_for_heatmap if v not in df.columns]
    if missing:
        # work on a copy of df but add NaN columns for missing vars (non-destructive)
        working_df = df.copy()
        for v in missing:
            working_df[v] = pd.Series(np.nan, index=df.index)
    else:
        working_df = df

    # work on a local copy of the selected columns (numeric coercion but keep NaN where not convertible)
    working = working_df[vars_for_heatmap].apply(pd.to_numeric, errors="coerce").copy()

    # initialize result
    corr_mat = pd.DataFrame(index=vars_for_heatmap, columns=vars_for_heatmap, dtype=float)

    # compute pairwise Pearson on pairwise non-missing samples
    for i_idx, i in enumerate(vars_for_heatmap):
        xi_full = working[i]
        for j_idx, j in enumerate(vars_for_heatmap):
            # mirror lower triangle
            if j_idx < i_idx:
                corr_mat.at[i, j] = corr_mat.at[j, i]
                continue

            xj_full = working[j]
            # align and keep only rows where both non-missing
            xi_pair, xj_pair = xi_full.align(xj_full, join="inner")
            mask = xi_pair.notna() & xj_pair.notna()
            xi = xi_pair[mask]
            xj = xj_pair[mask]

            # default fallback
            val = 0.0

            if len(xi) >= 2:
                # if either column constant on overlap, correlation undefined -> return 0
                try:
                    sdx = xi.std(ddof=1)
                    sdj = xj.std(ddof=1)
                except Exception:
                    sdx = float(np.nan)
                    sdj = float(np.nan)

                if (not np.isfinite(sdx)) or (not np.isfinite(sdj)) or (sdx == 0) or (sdj == 0):
                    val = 0.0
                else:
                    # safe Pearson
                    try:
                        val = float(xi.corr(xj, method="pearson"))
                        if not np.isfinite(val):
                            val = 0.0
                    except Exception:
                        val = 0.0

            # clip numerical noise and enforce float
            if val > 1.0:
                val = 1.0
            if val < -1.0:
                val = -1.0

            corr_mat.at[i, j] = float(val)
            corr_mat.at[j, i] = float(val)

    # final cleanup
    corr_mat = corr_mat.astype(float).fillna(0.0).clip(-1.0, 1.0)

    # optional save
    if outdir:
        os.makedirs(outdir, exist_ok=True)
        csv_path = os.path.join(outdir, f"{save_basename}.csv")
        try:
            corr_mat.to_csv(csv_path)
        except Exception as e:
            print(f"Warning: could not save CSV to {csv_path}: {e}")
        try:
            parquet_path = os.path.join(outdir, f"{save_basename}.parquet")
            corr_mat.to_parquet(parquet_path)
        except Exception:
            # ignore parquet failures quietly
            pass

    return corr_mat