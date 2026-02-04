import re
import ast
import glob
import logging
import numpy as np
import pandas as pd
from scipy.stats import entropy
from typing import Hashable, Callable, Optional, Iterable, List, Union, Dict, TypedDict, Any, Sequence, Tuple

# Import constants from _rules.py
from ._rules import MEDICAT_ABBREV, DEFAULT_MEDICAT_GROUP_MAP

logger = logging.getLogger(__name__)


# ------------------------------------------------------------------------------------
# Spec & Registry types
# ------------------------------------------------------------------------------------

class DeriveSpec(TypedDict, total=False):
    fn: Callable[[pd.DataFrame], pd.DataFrame]
    all_of: Optional[List[str]]   # (not used by the default "auto" selector, kept for compatibility)
    any_of: Optional[List[str]]   # columns hinting which entity this derive belongs to


# ------------------------------------------------------------------------------------
# Derive-specific utilities
# ------------------------------------------------------------------------------------

def make_colname(
    text: str,
    prefix: str = "col",
    max_len: int = 30,
    abbrev_map: Optional[Dict[str, str]] = MEDICAT_ABBREV,
) -> str:
    """
    Convert free-text into a compact snake_case column name.

    Falls back to the MEDICAT_ABBREV imported from _rules.py when abbrev_map is None.
    """
    # prefer explicit abbrev_map; otherwise use global MEDICAT_ABBREV; if that's missing, use empty dict
    sm = abbrev_map if abbrev_map is not None else (MEDICAT_ABBREV if 'MEDICAT_ABBREV' in globals() and MEDICAT_ABBREV is not None else {})
    # coerce to dict for safety (if someone passed a mapping-like)
    try:
        sm = dict(sm)
    except Exception:
        sm = {}

    s = str(text).lower()
    s = re.sub(r"\(.*?\)", "", s)
    s = re.sub(r"[^\w\s]", " ", s)
    tokens = [t for t in re.split(r"\s+", s) if t]

    # try to find best mapping by token prefix match (preserve order)
    for tok in tokens:
        for key, val in sm.items():
            if tok.startswith(key):
                cand = f"{prefix}{val}"
                cand = re.sub(r"_+", "_", cand).strip("_")
                return cand[:max_len]

    # fallback: use first two tokens joined
    fallback = "_".join(tokens[:2]) if tokens else "unknown"
    fallback = re.sub(r"[^\w]", "_", fallback)
    cand = f"{prefix}{fallback}"
    cand = re.sub(r"_+", "_", cand).strip("_")
    return cand[:max_len]


def _to_int_safe(x):
    """Robust conversion to int or None."""
    try:
        if pd.isna(x):
            return None
    except Exception:
        pass
    if isinstance(x, (int, np.integer)):
        return int(x)
    s = str(x).strip()
    if s == "":
        return None
    try:
        if "." in s:
            f = float(s)
            if f.is_integer():
                return int(f)
        return int(s)
    except Exception:
        return None


def _to_iterable_for_codes(x):
    if x is None:
        return None
    if isinstance(x, np.ndarray):
        return x.tolist()
    if isinstance(x, (list, tuple, set)):
        return list(x)
    if isinstance(x, (str, bytes)):
        try:
            return [int(x)]
        except Exception:
            return [x]
    try:
        if pd.isna(x):
            return None
    except Exception:
        pass
    return [x]


def _ensure_list_of_ints(cell) -> list:
    """
    Normalize a cell to a list of integer codes (ignores invalid tokens).
    Accepts lists/tuples/strings like "[1,2]", "1|2", "1,2", etc.
    """
    if cell is None or (isinstance(cell, float) and np.isnan(cell)):
        return []
    if isinstance(cell, (list, tuple, set, np.ndarray)):
        seq = list(cell)
    elif isinstance(cell, str):
        s = cell.strip()
        if s.startswith("[") or s.startswith("(") or s.startswith("{"):
            try:
                seq = list(ast.literal_eval(s))
            except Exception:
                seq = re.split(r"[^\d\-]+", s)
        else:
            if "," in s or ";" in s or "|" in s:
                seq = re.split(r"[,\;\|]", s)
            else:
                seq = [s]
    else:
        seq = [cell]

    out = []
    for token in seq:
        code_int = _to_int_safe(token)
        if code_int is not None:
            out.append(code_int)
    return out


def expand_multi_code_column(
    df: pd.DataFrame,
    codings_glob: str = "./metadata/*.codings.csv",
    coding_name: str = "MEDICAT_1_M",
    col: str = "questionnaire.medicat_1_m",
    prefix: str = "derived.medicates_",
    exclude_codes: Sequence[int] = (-7, -1, -3),
    abbrev_map: Optional[Dict[str, str]] = MEDICAT_ABBREV,
    inplace: bool = True,
) -> pd.DataFrame:
    """
    Create binary indicator columns from a coding file and a participant multi-code column.

    This is the generalized variant of 'get_dummies' for any multi-code field.

    Returns mapping DataFrame (raw_code, code_int, meaning, colname, prevalence).
    The transient helper column "<col>_codes" is removed from `df` before returning.
    """
    import glob

    matches = glob.glob(codings_glob)
    if not matches:
        raise FileNotFoundError(f"No codings files found for glob '{codings_glob}'")
    codings_path = matches[0]
    logger.debug("Using codings file: %s", codings_path)

    cod = pd.read_csv(codings_path, dtype=str, low_memory=False)
    cod_cols = {c.upper(): c for c in cod.columns}
    col_code = cod_cols.get("CODE", None) or cod_cols.get("VALUE", None) or cod_cols.get("CODES", None)
    col_name = cod_cols.get("CODING_NAME", None) or cod_cols.get("CODING", None) or cod_cols.get("CODINGNAME", None)
    col_mean = cod_cols.get("MEANING", None) or cod_cols.get("LABEL", None) or cod_cols.get("DESCRIPTION", None) or cod_cols.get("MEANINGS", None)

    if col_code is None or col_name is None or col_mean is None:
        raise ValueError(
            f"codings CSV missing expected columns. Found columns: {list(cod.columns)}. "
            "Expected some variant of ['code','coding_name','meaning']."
        )

    cod_sel = cod[cod[col_name].str.upper() == coding_name.upper()].copy()
    if cod_sel.empty:
        raise ValueError(f"No rows found in codings CSV for coding_name == '{coding_name}'")

    mapping_rows = []
    for _, row in cod_sel.iterrows():
        raw_code = row[col_code]
        meaning = row[col_mean] if pd.notna(row[col_mean]) else str(raw_code)
        code_int = _to_int_safe(raw_code)
        mapping_rows.append({"raw_code": raw_code, "code_int": code_int, "meaning": meaning})

    mapping_df = pd.DataFrame(mapping_rows)
    mapping_df_valid = mapping_df[mapping_df["code_int"].notna()].copy()
    mapping_df_valid["code_int"] = mapping_df_valid["code_int"].astype(int)
    mapping_df_valid = mapping_df_valid[~mapping_df_valid["code_int"].isin(exclude_codes)].copy()
    if mapping_df_valid.empty:
        raise ValueError("After excluding missing codes, no valid integer codes remain in the mapping.")

    mapping_df_valid["colname"] = mapping_df_valid["meaning"].apply(
        lambda m: make_colname(m, prefix=prefix, abbrev_map=abbrev_map)
    )

    dupes = mapping_df_valid["colname"].duplicated(keep=False)
    if dupes.any():
        for i in mapping_df_valid[dupes].index:
            row = mapping_df_valid.loc[i]
            mapping_df_valid.at[i, "colname"] = f"{row['colname']}_{int(row['code_int'])}"

    if not inplace:
        df = df.copy()

    cleaned_col = f"{col}_codes"
    df[cleaned_col] = df[col].apply(_ensure_list_of_ints)

    for _, r in mapping_df_valid.iterrows():
        code_int = int(r["code_int"])
        cname = r["colname"]
        df[cname] = df[cleaned_col].apply(lambda lst: 1 if code_int in lst else 0)

    prevalence = {r["colname"]: int(df[r["colname"]].sum()) for _, r in mapping_df_valid.iterrows()}
    mapping_df_valid = mapping_df_valid.reset_index(drop=True)
    mapping_df_valid["prevalence"] = mapping_df_valid["colname"].map(prevalence).fillna(0).astype(int)

    # remove the transient helper column before returning
    df.drop(columns=[cleaned_col], errors="ignore", inplace=True)

    logger.info("Created %d flag columns with prefix=%s", len(mapping_df_valid), prefix)
    return mapping_df_valid

    
# ------------------------------------------------------------------------------------
# Derive function participant implementations
#   NOTE: Each function *no-ops* (logs + returns df) if inputs are insufficient.
# ------------------------------------------------------------------------------------

def self_reported_sex(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create derived.sex (numeric codes, categorical dtype).
    """
    df = df.copy()
    v2 = df.get("participant.demog_sex_2_1", pd.NA)
    v1 = df.get("participant.demog_sex_1_1", pd.NA)
    use_v2 = v2.notna()
    sex_val = v2.where(use_v2, v1)

    def to_code(val, col):
        if pd.isna(val):
            return pd.NA
        v = int(val)
        if v == -3: return -3
        if v == 1: return 1
        if v == 2: return 2
        if v == 3:
            return 3 if col == "participant.demog_sex_2_1" else 4
        return pd.NA

    from_col = np.where(use_v2, "participant.demog_sex_2_1", "participant.demog_sex_1_1")
    codes = [to_code(val, col) for val, col in zip(sex_val, from_col)]

    # Store as Int64 and categorical with code categories
    code_categories = [1, 2, 3, 4, -3]
    df["derived.sex"] = pd.Series(codes, index=df.index, dtype="Int64")
    df["derived.sex"] = df["derived.sex"].astype(pd.CategoricalDtype(categories=code_categories))
    return df


def registration_date(df: pd.DataFrame) -> pd.DataFrame:
    """
    Derive 'derived.registration_date' as a proper datetime column.

    - If the column already exists, it will be coerced to datetime.
    - Otherwise, it is constructed from 'participant.registration_year' and
      'participant.registration_month' (day set to 1).
    - Returns the input DataFrame with a new/cleaned
      'derived.registration_date' column (datetime64[ns]).
    """
    cols = set(df.columns)
    df = df.copy()

    # Case 1: already present — just coerce
    if "derived.registration_date" in cols:
        df["derived.registration_date"] = pd.to_datetime(
            df["derived.registration_date"], errors="coerce"
        )
        return df

    # Case 2: build from year + month
    have_ym = {"participant.registration_year", "participant.registration_month"} <= cols
    if not have_ym:
        logger.info(
            "registration_date: skipped (no registration_date or registration_year/month)."
        )
        return df

    reg_dt = pd.to_datetime(
        dict(
            year=df["participant.registration_year"],
            month=df["participant.registration_month"],
            day=1,
        ),
        errors="coerce",
    )
    df["derived.registration_date"] = reg_dt
    return df


def age_at_registration(df: pd.DataFrame) -> pd.DataFrame:
    """
    Derive age at registration (float, continuous).
    """
    cols = set(df.columns)
    have_A = {"derived.registration_date", "participant.date_of_birth"} <= cols
    have_B = {"participant.registration_year", "participant.registration_month",
              "participant.birth_year", "participant.birth_month"} <= cols

    if not (have_A or have_B):
        logger.info("age_at_registration: skipped (no required input pattern present).")
        return df

    if have_A:
        reg_dt = pd.to_datetime(df["derived.registration_date"], errors="coerce")
        dob_dt = pd.to_datetime(df["participant.date_of_birth"], errors="coerce")
    else:
        reg_dt = pd.to_datetime(dict(
            year=df["participant.registration_year"],
            month=df["participant.registration_month"],
            day=1
        ), errors="coerce")
        dob_dt = pd.to_datetime(dict(
            year=df["participant.birth_year"],
            month=df["participant.birth_month"],
            day=1
        ), errors="coerce")

    out = (reg_dt - dob_dt).dt.days / 365.25
    df = df.copy()
    df["derived.age_at_registration"] = out
    return df


def age_group(df: pd.DataFrame, bins=None, labels=None) -> pd.DataFrame:
    """
    Derive 'derived.age_group' from existing derived.age_at_registration (preferred)
    or from registration / birth date fields (fallback).

    Default groups: all, 18-29, 30-59, 60+ (intervals are inclusive on the lower bound).

    Behavior:
      - If 'derived.age_group' already exists, it will be replaced.
      - If neither derived.age_at_registration nor the required raw date/year
        fields exist, the function is a no-op and returns the original df.
    """
    cols = set(df.columns)
    age = None

    if "derived.age_at_registration" in cols:
        # 1. Preferred path: use already-computed continuous age if present
        age = pd.to_numeric(df["derived.age_at_registration"], errors="coerce")

    else:
        # 2. Fallback path: calculate age from date/year fields
        
        # Check for full date fields (preferred fallback)
        have_A = {"derived.registration_date", "participant.date_of_birth"} <= cols
        # Check for year/month fields (less preferred fallback)
        have_B = {"participant.registration_year", "participant.registration_month",
                  "participant.birth_year", "participant.birth_month"} <= cols

        if not (have_A or have_B):
            logger.info("age_group: skipped (no derived.age_at_registration or required raw date fields).")
            return df

        if have_A:
            # Calculate age using full date objects
            reg_dt = pd.to_datetime(df["derived.registration_date"], errors="coerce")
            dob_dt = pd.to_datetime(df["participant.date_of_birth"], errors="coerce")
        else:
            # Use first-of-month approximation when only year/month available
            reg_dt = pd.to_datetime({
                'year': df["participant.registration_year"],
                'month': df["participant.registration_month"],
                'day': 1
            }, errors="coerce")
            dob_dt = pd.to_datetime({
                'year': df["participant.birth_year"],
                'month': df["participant.birth_month"],
                'day': 1
            }, errors="coerce")

        # Calculate continuous age in years: difference in days divided by 365.25
        age = (reg_dt - dob_dt).dt.days / 365.25

    # Define the bins and labels for categorical grouping
    if bins is None:
        bins = [18, 30, 60, float("inf")]
    if labels is None:
        labels = ["18-29", "30-59", "60+"]

    # Use pd.cut to group ages into the defined bins
    # right=False ensures the bin is inclusive on the left (e.g., age 30 goes into 30-59)
    out = pd.cut(age, bins=bins, labels=labels, right=False)

    # Convert the result to a proper categorical dtype to maintain ordering
    out = pd.Categorical(out, categories=labels, ordered=True)

    # Create a copy of the DataFrame and add the new derived column
    df2 = df.copy()
    df2["derived.age_group"] = out

    return df2
    
    
# ------------------------------------------------------------------------------------
# Derive function clinic measurement implementations
#   NOTE: Each function *no-ops* (logs + returns df) if inputs are insufficient.
# ------------------------------------------------------------------------------------

def bmi(df: pd.DataFrame) -> pd.DataFrame:
    """
    Derive BMI from clinic measurements (float, continuous).
    """
    cols = set(df.columns)
    need = {"clinic_measurements.height", "clinic_measurements.weight"}
    if not need <= cols:
        logger.info("bmi: skipped (missing %s).", sorted(need - cols))
        return df

    height_m = pd.to_numeric(df["clinic_measurements.height"], errors="coerce") / 100.0
    weight_kg = pd.to_numeric(df["clinic_measurements.weight"], errors="coerce")

    df = df.copy()
    with np.errstate(divide="ignore", invalid="ignore"):
        df["derived.bmi"] = weight_kg / (height_m ** 2)
    return df


def bmi_status(df: pd.DataFrame) -> pd.DataFrame:
    """
    Derive 'derived.bmi_status' (numeric code, categorical dtype).
    Categories: 0=underweight, 1=normal, 2=overweight, 3=obese, -9=missing
    """
    if "derived.bmi" not in df.columns:
        logger.info("bmi_status: skipped (requires 'derived.bmi').")
        return df

    df = df.copy()
    bmi_num = pd.to_numeric(df["derived.bmi"], errors="coerce")

    # Use integer codes for categories
    conditions = [
        bmi_num < 18.5,                  # 0 = underweight
        (bmi_num >= 18.5) & (bmi_num < 25),  # 1 = normal
        (bmi_num >= 25) & (bmi_num < 30),    # 2 = overweight
        bmi_num >= 30,                   # 3 = obese
    ]
    code_categories = [0, 1, 2, 3, -9]
    codes = np.select(conditions, [0, 1, 2, 3], default=-9)
    df["derived.bmi_status"] = pd.Series(codes, index=df.index, dtype="Int64")
    df["derived.bmi_status"] = df["derived.bmi_status"].astype(pd.CategoricalDtype(categories=code_categories))
    return df

    
# ------------------------------------------------------------------------------------
# Derive function questionnaire implementations
#   NOTE: Each function *no-ops* (logs + returns df) if inputs are insufficient.
# ------------------------------------------------------------------------------------


def vape_status(
    df: pd.DataFrame,
    numeric: bool = True,
    out_col: str = "derived.vape_status"
) -> pd.DataFrame:
    """
    Derive vaping status as numeric code and categorical dtype.
    1 = ever used, 0 = never used, -3 = prefer not to answer
    """
    v1 = df.get("questionnaire.smoke_vape_1_1", None)
    v2 = df.get("questionnaire.smoke_tobacco_type_1_m", None)

    if v1 is None and v2 is None:
        logger.info("vape_status: skipped (no input columns).")
        return df

    def to_list(x):
        if isinstance(x, (list, tuple, set, np.ndarray)): return list(x)
        if pd.isna(x): return []
        return [x]

    v2_lists = v2.map(to_list) if v2 is not None else pd.Series([], dtype="object", index=df.index)
    has1   = v2_lists.map(lambda L: 1 in L) if v2 is not None else pd.Series(False, index=df.index)
    haspnr = v2_lists.map(lambda L: -3 in L) if v2 is not None else pd.Series(False, index=df.index)
    hasany = v2_lists.map(lambda L: len(L) > 0) if v2 is not None else pd.Series(False, index=df.index)

    v2_status = pd.Series(pd.NA, index=df.index, dtype="object")
    if v2 is not None:
        v2_status.loc[has1] = "ever used"
        v2_status.loc[v2_status.isna() & haspnr] = "prefer not to answer"
        v2_status.loc[v2_status.isna() & hasany] = "never used"

    v1_map = {1: "ever used", 0: "never used", -3: "prefer not to answer"}
    v1_status = v1.map(v1_map) if v1 is not None else pd.Series(pd.NA, index=df.index, dtype="object")

    total = v1_status.fillna(v2_status)
    code_map = {"ever used": 1, "never used": 0, "prefer not to answer": -3}
    code_categories = [1, 0, -3]

    df = df.copy()
    codes = total.map(code_map).astype("Int64")
    df[out_col] = codes.astype(pd.CategoricalDtype(categories=code_categories))
    return df


def smoke_status_v1(df: pd.DataFrame, numeric: bool = True) -> pd.DataFrame:
    """
    Derive 'derived.smoke_status_v1' as numeric codes and categorical dtype.
    Codes: 2=current, 1=former, 0=never, -1=unknown/prefer not to answer, -9=missing
    """
    need = {
        "questionnaire.smoke_status_1_1",
        "questionnaire.smoke_tobacco_prev_1_1",
        "questionnaire.smoke_100_times_1_1",
    }
    cols = set(df.columns)
    if not need <= cols:
        logger.info("smoke_status_v1: skipped (missing %s).", sorted(need - cols))
        return df

    df = df.copy()
    smk = df["questionnaire.smoke_status_1_1"]
    prev = df["questionnaire.smoke_tobacco_prev_1_1"]
    times100 = df["questionnaire.smoke_100_times_1_1"]

    df["derived.smoke_status_v1"] = pd.NA

    # current
    mask_current = (
        (smk == 1)
        | ((smk == 2) & prev.isin([1, 2]))
        | ((smk == 2) & (prev == 3) & (times100 == 1))
    )
    df.loc[mask_current, "derived.smoke_status_v1"] = 2

    # former
    mask_former = (
        ((smk == 0) & prev.isin([1, 2]))
        | ((smk == 0) & (prev == 3) & (times100 == 1))
    )
    df.loc[mask_former, "derived.smoke_status_v1"] = 1

    # never
    mask_never = (
        (((smk == 0) | (smk == 2)) & (prev == 4))
        | ((smk == 0) & (prev == 3) & times100.isin([0, -1, -3]))
        | ((smk == 2) & (prev == 3))
    )
    df.loc[mask_never, "derived.smoke_status_v1"] = 0

    # unknown
    mask_unknown = (
        ((smk == -3) & prev.isin([-3, 1, 2, 3, 4]))
        | (((smk == 0) | (smk == 2)) & (prev == -3))
    )
    df.loc[mask_unknown, "derived.smoke_status_v1"] = -1

    # missing
    mask_missing = smk.isna() & prev.isna()
    df.loc[mask_missing, "derived.smoke_status_v1"] = -9

    # store as Int64 then as categorical dtype
    code_categories = [2, 1, 0, -1, -9]
    df["derived.smoke_status_v1"] = df["derived.smoke_status_v1"].astype("Int64")
    df["derived.smoke_status_v1"] = df["derived.smoke_status_v1"].astype(pd.CategoricalDtype(categories=code_categories))
    return df


def _recode_tobacco_status(
    df: pd.DataFrame,
    input_col: Hashable,
    output_col: Hashable,
) -> pd.DataFrame:
    """
    Recode array-like raw tobacco column -> numeric status:
      0=never (code 6 present), 1=smoked cigarettes (code 0), -3=pnr, 2=smoked other.
    """
    if input_col not in df.columns:
        logger.info("_recode_tobacco_status: skipped (missing %s).", input_col)
        return df

    vals = df[input_col].apply(_to_iterable_for_codes)

    def _has(v, code): return isinstance(v, (list, tuple)) and code in v

    out = pd.Series(pd.NA, index=df.index, dtype="Int64")
    has_6   = vals.apply(lambda v: _has(v, 6))
    has_0   = vals.apply(lambda v: _has(v, 0))
    has_m3  = vals.apply(lambda v: _has(v, -3))
    notnull = vals.notna()

    out.loc[has_6] = 0
    out.loc[has_0] = 1
    out.loc[has_m3] = -3
    out.loc[notnull & (~has_0) & (~has_6) & (~has_m3)] = 2

    df = df.copy()
    df[output_col] = out
    return df


def tobacco_ever(df: pd.DataFrame) -> pd.DataFrame:
    """Create 'questionnaire.tobacco_ever' from 'questionnaire.smoke_tobacco_type_1_m'."""
    return _recode_tobacco_status(
        df, "questionnaire.smoke_tobacco_type_1_m", "questionnaire.tobacco_ever"
    )


def tobacco_reg(df: pd.DataFrame) -> pd.DataFrame:
    """Create 'questionnaire.tobacco_reg' from 'questionnaire.smoke_reg_1_m'."""
    return _recode_tobacco_status(
        df, "questionnaire.smoke_reg_1_m", "questionnaire.tobacco_reg"
    )


def smoke_status_v2(df: pd.DataFrame, numeric: bool = True) -> pd.DataFrame:
    """
    Derive 'derived.smoke_status_v2' as numeric codes and categorical dtype.
    Codes: 2=current, 1=former, 0=never, -1=prefer not to answer/unknown, -9=missing
    """
    df = df.copy()

    # Try to (idempotently) create the final inputs from raw, if available
    if "questionnaire.tobacco_ever" not in df.columns and "questionnaire.smoke_tobacco_type_1_m" in df.columns:
        df = tobacco_ever(df)
    if "questionnaire.tobacco_reg" not in df.columns and "questionnaire.smoke_reg_1_m" in df.columns:
        df = tobacco_reg(df)

    need = {
        "questionnaire.tobacco_reg",
        "questionnaire.tobacco_ever",
        "questionnaire.smoke_status_2_1",
        "questionnaire.smoke_100_times_2_1",
    }
    cols = set(df.columns)
    if not need <= cols:
        logger.info("smoke_status_v2: skipped (missing %s).", sorted(need - cols))
        return df

    reg = df["questionnaire.tobacco_reg"]
    ever = df["questionnaire.tobacco_ever"]
    status = df["questionnaire.smoke_status_2_1"]
    times100 = df["questionnaire.smoke_100_times_2_1"]

    # Map string categories to numeric codes
    out = pd.Series(pd.NA, index=df.index, dtype="Int64")

    # current
    mask_current = ((reg == 1) & status.isin([1, 2, 3])) | ((reg == 1) & (ever == -3))
    out.loc[mask_current] = 2

    # former
    mask_former = (
        ((reg == 1) & (status == 0))
        | ((reg == 0) & ever.isin([1, 2]) & (times100 == 3))
        | ((reg == 2) & (ever == 1) & (times100 == 3))
    )
    out.loc[mask_former] = 1

    # never
    mask_never = (
        ((reg == 0) & ever.isin([0, -3, 2]))
        | (reg.isna() & (ever == 0))
        | (reg.isin([0, 2, -3]) & (ever == 1) & times100.isin([0, 1, 2]))
        | ((reg == 2) & ever.isin([2, -3]))
    )
    out.loc[mask_never] = 0

    # unknown
    mask_unknown = (
        ((reg == -3) & ever.isin([2, -3]))
        | (reg.isin([2, -3]) & (ever == 1) & times100.isin([3, -3]))
    )
    out.loc[mask_unknown] = -1

    # missing
    mask_missing = reg.isna() & ever.isna()
    out.loc[mask_missing] = -9

    code_categories = [2, 1, 0, -1, -9]
    df["derived.smoke_status_v2"] = out.astype(pd.CategoricalDtype(categories=code_categories))
    return df


def walk_16_10(
    df: pd.DataFrame,
    *,
    source_col: str = "questionnaire.activity_walk_days_2_1",
    out_col: str = "derived.walk_16_10",
    weeks_per_month: float = 4.0,
    unable_labels: tuple = ("Unable to walk",),
    treat_unable_as_zero: bool = True,
    dtype: str = "Int64",
) -> pd.DataFrame:
    """
    Binary: walked >=16 times per month for ≥10 minutes.
    If `source_col` is 'days per last week', threshold becomes ceil(16 / weeks_per_month) (default 4).
    """
    if source_col not in df.columns:
        logger.info("walk_16_10: skipped (missing %s).", source_col)
        return df

    s = df[source_col].copy()

    if unable_labels:
        s = s.replace(list(unable_labels), 0 if treat_unable_as_zero else pd.NA)

    s_num = pd.to_numeric(s, errors="coerce")
    weekly_threshold = int(np.ceil(16.0 / weeks_per_month))

    flag = (s_num >= weekly_threshold)
    out = flag.astype(dtype).where(~s_num.isna(), pd.NA)

    df = df.copy()
    df[out_col] = out
    return df


def medicat_expand(
    df: pd.DataFrame,
    *,
    codings_glob: str = "./metadata/*.codings.csv",
    coding_name: str = "MEDICAT_1_M",
    col: str = "questionnaire.medicat_1_m",
    prefix: str = "derived.medicates_",
    exclude_codes: Sequence[int] = (-7, -1, -3),
    abbrev_map: Optional[Dict[str, str]] = None,
) -> pd.DataFrame:
    """
    Wrapper derive that expands a multi-code raw column into binary flags.

    - Modifies `df` in-place by default (via expand_multi_code_column) and
      returns the DataFrame (so it fits DERIVE_REGISTRY expectations).
    - Swallows failures with a warning so auto-derive doesn't crash the whole pipeline.
    """
    if col not in df.columns:
        # Nothing to do
        return df

    try:
        expand_multi_code_column(
            df,
            codings_glob=codings_glob,
            coding_name=coding_name,
            col=col,
            prefix=prefix,
            exclude_codes=exclude_codes,
            abbrev_map=abbrev_map,
            inplace=True,
        )
    except Exception as e:
        logger.warning("medicat_expand: failed to expand codings (%s). Continuing. Error: %s", coding_name, e)
    return df


    
# ------------------------------------------------------------------------------------
# Derive function health records implementations
#   NOTE: Each function *no-ops* (logs + returns df) if inputs are insufficient.
# ------------------------------------------------------------------------------------

def any_hospital_contact(
    data: Union[Dict[str, pd.DataFrame], pd.DataFrame],
    merged: Optional[pd.DataFrame] = None,
    before_registration: bool = False
) -> pd.DataFrame:
    """Return merged DataFrame with derived.any_hospital_contact (0/1).
    Accepts a dict of entity DataFrames or a single merged DataFrame.
    """
    def pid_cols(df):
        return [c for c in df.columns if c == "pid" or c.lower().endswith(".pid") or c.lower().endswith("pid")]

    def unify_pid(df):
        pcols = pid_cols(df)
        if not pcols:
            raise ValueError("no pid-like columns found")
        if "pid" in pcols and len(pcols) == 1:
            return df  # already a single pid
        # create unified 'pid' without dropping originals
        df = df.copy()
        if "pid" not in df.columns:
            df["pid"] = df[pcols].bfill(axis=1).iloc[:, 0]
        else:
            # if 'pid' exists, prefer it but fill nulls from other pid cols
            df["pid"] = df[["pid"] + [c for c in pcols if c != "pid"]].bfill(axis=1).iloc[:, 0]
        return df

    def extract_events(df, key_names, date_names):
        if df is None: return None
        pcols = pid_cols(df)
        keys = [c for c in df.columns if any(c.lower().endswith(k) for k in key_names)]
        dates = [c for c in df.columns if any(c.lower().endswith(d) for d in date_names)]
        if not pcols or not keys or not dates:
            return None
        df2 = df[[*pcols, keys[0], dates[0]]].copy()
        df2 = unify_pid(df2)[["pid", keys[0], dates[0]]].rename(columns={keys[0]: "rec_key", dates[0]: "date"})
        df2["date"] = pd.to_datetime(df2["date"], errors="coerce")
        return df2.dropna(subset=["rec_key", "date", "pid"])

    # gather event DataFrames
    evs = []
    mapping = [
        (("aekey", "row_id"), ("arrivaldate",), "nhse_eng_ed"),
        (("epikey", "row_id"), ("admidate",), "nhse_eng_inpat"),
        (("attendkey", "row_id"), ("apptdate",), "nhse_eng_outpat"),
    ]

    if isinstance(data, dict):
        for keynames, datenames, ent in mapping:
            ev = extract_events(data.get(ent), keynames, datenames)
            if ev is not None:
                evs.append(ev)
    else:
        for keynames, datenames, _ in mapping:
            ev = extract_events(data, keynames, datenames)
            if ev is not None:
                evs.append(ev)

    # base merged frame to attach the derived column to
    target = merged if merged is not None else (data.get("participant") if isinstance(data, dict) else data)
    if target is None:
        raise ValueError("need a merged DF or participant DF available for registration check / output")

    target = unify_pid(target)

    if not evs:
        target["derived.any_hospital_contact"] = 0
        return target

    ev = pd.concat(evs, ignore_index=True).drop_duplicates(["pid", "rec_key"])

    if before_registration:
        reg_col = next((c for c in target.columns if "registration" in c.lower() and "date" in c.lower()), None)
        if reg_col is not None:
            regs = target[["pid", reg_col]].copy()
            regs[reg_col] = pd.to_datetime(regs[reg_col], errors="coerce")
            ev = ev.merge(regs.rename(columns={reg_col: "reg"}), on="pid", how="left")
            ev = ev[ev["reg"].notna() & (ev["date"] < ev["reg"])]

    has = ev.drop_duplicates(["pid"]).loc[:, ["pid"]].assign(**{"derived.any_hospital_contact": 1})
    out = target.merge(has, on="pid", how="left")
    out["derived.any_hospital_contact"] = out["derived.any_hospital_contact"].fillna(0).astype(int)
    return out


def ae_visits(
    data: Union[Dict[str, pd.DataFrame], pd.DataFrame],
    merged: Optional[pd.DataFrame] = None,
    before_registration: bool = False
) -> pd.DataFrame:
    """Return merged DataFrame with derived.ae_visits (int count of unique A&E visits per pid).
    Accepts a dict of entity DataFrames or a single merged DataFrame.
    If `before_registration=True` will only count visits before the participant's registration date
    (if a registration date column can be found on the target/merged frame).
    """
    def pid_cols(df):
        return [c for c in df.columns if c == "pid" or c.lower().endswith(".pid") or c.lower().endswith("pid")]

    def unify_pid(df):
        pcols = pid_cols(df)
        if not pcols:
            raise ValueError("no pid-like columns found")
        if "pid" in pcols and len(pcols) == 1:
            return df  # already a single pid
        # create unified 'pid' without dropping originals
        df = df.copy()
        if "pid" not in df.columns:
            df["pid"] = df[pcols].bfill(axis=1).iloc[:, 0]
        else:
            # if 'pid' exists, prefer it but fill nulls from other pid cols
            df["pid"] = df[["pid"] + [c for c in pcols if c != "pid"]].bfill(axis=1).iloc[:, 0]
        return df

    def extract_ae_events(df):
        """Expect an AE-like DF; return DataFrame with ['pid','rec_key','date'] or None if inputs insufficient."""
        if df is None:
            return None
        pcols = pid_cols(df)
        keys = [c for c in df.columns if c.lower().endswith("aekey") or c.lower().endswith(".aekey") or c.lower() == "aekey"]
        dates = [c for c in df.columns if "arrival" in c.lower() and "date" in c.lower()]
        if not pcols or not keys or not dates:
            return None
        df2 = df[[*pcols, keys[0], dates[0]]].copy()
        df2 = unify_pid(df2)[["pid", keys[0], dates[0]]].rename(columns={keys[0]: "rec_key", dates[0]: "date"})
        df2["date"] = pd.to_datetime(df2["date"], errors="coerce")
        return df2.dropna(subset=["rec_key", "date", "pid"])

    # --- gather AE events ---
    if isinstance(data, dict):
        ae_df = data.get("nhse_eng_ed")
        ev = extract_ae_events(ae_df)
    else:
        # data is a merged frame potentially containing aekey/arrivaldate columns
        ev = extract_ae_events(data)

    # base merged frame to attach the derived column to
    target = merged if merged is not None else (data.get("participant") if isinstance(data, dict) else data)
    if target is None:
        raise ValueError("need a merged DF or participant DF available for output / registration check")

    target = unify_pid(target)

    # if no events found, return zero counts
    if ev is None or ev.empty:
        target["derived.ae_visits"] = 0
        return target

    # optionally filter to pre-registration events
    if before_registration:
        reg_col = next((c for c in target.columns if "registration" in c.lower() and "date" in c.lower()), None)
        if reg_col is not None:
            regs = target[["pid", reg_col]].copy()
            regs[reg_col] = pd.to_datetime(regs[reg_col], errors="coerce")
            ev = ev.merge(regs.rename(columns={reg_col: "reg"}), on="pid", how="left")
            ev = ev[ev["reg"].notna() & (ev["date"] < ev["reg"])]
        else:
            # no registration date available -> treat as no filtering (counts all)
            pass

    # count unique AE keys per PID
    ev = ev.drop_duplicates(["pid", "rec_key"])
    counts = ev.groupby("pid", as_index=False)["rec_key"].agg(n_AE_visits="count").rename(columns={"n_AE_visits": "derived.ae_visits"})

    out = target.merge(counts, on="pid", how="left")
    out["derived.ae_visits"] = out["derived.ae_visits"].fillna(0).astype(int)
    return out


def apc_visits(
    data: Union[Dict[str, pd.DataFrame], pd.DataFrame],
    merged: Optional[pd.DataFrame] = None,
    before_registration: bool = False
) -> pd.DataFrame:
    """Return merged DataFrame with derived.apc_visits (int count of unique APC admissions per pid).
    Counts unique epikey values for rows with a non-null admission date (admidate).
    If before_registration=True, only counts admissions with ADMIDATE < registration_date (if available).
    """
    def pid_cols(df):
        return [c for c in df.columns if c == "pid" or c.lower().endswith(".pid") or c.lower().endswith("pid")]

    def unify_pid(df):
        pcols = pid_cols(df)
        if not pcols:
            raise ValueError("no pid-like columns found")
        if "pid" in pcols and len(pcols) == 1:
            return df
        df = df.copy()
        if "pid" not in df.columns:
            df["pid"] = df[pcols].bfill(axis=1).iloc[:, 0]
        else:
            df["pid"] = df[["pid"] + [c for c in pcols if c != "pid"]].bfill(axis=1).iloc[:, 0]
        return df

    def extract_apc_events(df):
        """Return DataFrame with ['pid','rec_key','date'] for APC-like DF, or None if insufficient."""
        if df is None:
            return None
        pcols = pid_cols(df)
        keys = [c for c in df.columns if c.lower().endswith("epikey") or c.lower().endswith(".epikey") or c.lower() == "epikey"]
        dates = [c for c in df.columns if "admidate" in c.lower() or ("admi" in c.lower() and "date" in c.lower())]
        if not pcols or not keys or not dates:
            return None
        df2 = df[[*pcols, keys[0], dates[0]]].copy()
        df2 = unify_pid(df2)[["pid", keys[0], dates[0]]].rename(columns={keys[0]: "rec_key", dates[0]: "date"})
        df2["date"] = pd.to_datetime(df2["date"], errors="coerce")
        # require an admission date to treat as admission event
        return df2.dropna(subset=["rec_key", "date", "pid"])

    # gather APC events
    if isinstance(data, dict):
        apc_df = data.get("nhse_eng_inpat")
        ev = extract_apc_events(apc_df)
    else:
        ev = extract_apc_events(data)

    # target frame to attach derived column
    target = merged if merged is not None else (data.get("participant") if isinstance(data, dict) else data)
    if target is None:
        raise ValueError("need a merged DF or participant DF available for output / registration check")

    target = unify_pid(target)

    # no events -> zeros
    if ev is None or ev.empty:
        target["derived.apc_visits"] = 0
        return target

    # optional pre-registration filter
    if before_registration:
        reg_col = next((c for c in target.columns if "registration" in c.lower() and "date" in c.lower()), None)
        if reg_col is not None:
            regs = target[["pid", reg_col]].copy()
            regs[reg_col] = pd.to_datetime(regs[reg_col], errors="coerce")
            ev = ev.merge(regs.rename(columns={reg_col: "reg"}), on="pid", how="left")
            ev = ev[ev["reg"].notna() & (ev["date"] < ev["reg"])]
        else:
            # no reg date -> include all admissions
            pass

    # count unique epikey per pid
    ev = ev.drop_duplicates(["pid", "rec_key"])
    counts = ev.groupby("pid", as_index=False)["rec_key"].agg(n_APC_visits="count").rename(columns={"n_APC_visits": "derived.apc_visits"})

    out = target.merge(counts, on="pid", how="left")
    out["derived.apc_visits"] = out["derived.apc_visits"].fillna(0).astype(int)
    return out


def op_visits(
    data: Union[Dict[str, pd.DataFrame], pd.DataFrame],
    merged: Optional[pd.DataFrame] = None,
    before_registration: bool = False
) -> pd.DataFrame:
    """Counts unique outpatient records (attendkey / attend_key) per pid and returns target with derived.op_visits (int)."""
    def extract_op(df: pd.DataFrame):
        if df is None:
            return None
        pcols = pid_cols(df)
        keys = [c for c in df.columns if c.lower().endswith("attendkey") or c.lower().endswith(".attendkey") or c.lower() == "attendkey" or c.lower().endswith("attend_key")]
        dates = [c for c in df.columns if "apptdate" in c.lower() or "appointment" in c.lower() and "date" in c.lower()]
        if not dates:
            dates = [c for c in df.columns if "appt" in c.lower() or "date" in c.lower()]
        if not pcols or not keys or not dates:
            return None
        df2 = df[[*pcols, keys[0], dates[0]]].copy()
        df2 = unify_pid(df2)[["pid", keys[0], dates[0]]].rename(columns={keys[0]: "rec_key", dates[0]: "date"})
        df2["date"] = pd.to_datetime(df2["date"], errors="coerce")
        return df2.dropna(subset=["rec_key", "date", "pid"])

    if isinstance(data, dict):
        op_df = data.get("nhse_eng_outpat")
        ev = extract_op(op_df)
    else:
        ev = extract_op(data)

    target = merged if merged is not None else (data.get("participant") if isinstance(data, dict) else data)
    if target is None:
        raise ValueError("need a merged DF or participant DF available for output / registration check")

    target = unify_pid(target)

    if ev is None or ev.empty:
        target["derived.op_visits"] = 0
        return target

    if before_registration:
        reg_col = find_reg_col(target)
        if reg_col is not None:
            regs = target[["pid", reg_col]].copy()
            regs[reg_col] = pd.to_datetime(regs[reg_col], errors="coerce")
            ev = ev.merge(regs.rename(columns={reg_col: "reg"}), on="pid", how="left")
            ev = ev[ev["reg"].notna() & (ev["date"] < ev["reg"])]

    ev = ev.drop_duplicates(["pid", "rec_key"])
    counts = ev.groupby("pid", as_index=False)["rec_key"].agg(derived_op_visits="count").rename(columns={"derived_op_visits": "derived.op_visits"})
    out = target.merge(counts, on="pid", how="left")
    out["derived.op_visits"] = out["derived.op_visits"].fillna(0).astype(int)
    return out


def total_hospital_contacts(
    data: Union[Dict[str, pd.DataFrame], pd.DataFrame],
    merged: Optional[pd.DataFrame] = None,
    before_registration: bool = False,
    winsorize_pct: float = 0.99
) -> pd.DataFrame:
    """
    Count distinct record IDs across AE (aekey), APC (epikey) and OP (attendkey) per PID.
    Optionally consider only events before registration (if a registration date column exists).
    Winsorise counts at the given percentile (default 0.99).
    Output column: derived.total_hospital_contacts (int)
    """
    # helper to extract events (generic)
    def extract_events_from_entity(df: pd.DataFrame, key_suffixes, date_keywords):
        if df is None:
            return None
        pcols = pid_cols(df)
        keys = [c for c in df.columns if any(c.lower().endswith(s) or c.lower() == s for s in key_suffixes)]
        dates = [c for c in df.columns if any(k in c.lower() for k in date_keywords)]
        if not dates:
            dates = [c for c in df.columns if "date" in c.lower()]
        if not pcols or not keys or not dates:
            return None
        df2 = df[[*pcols, keys[0], dates[0]]].copy()
        df2 = unify_pid(df2)[["pid", keys[0], dates[0]]].rename(columns={keys[0]: "rec_key", dates[0]: "date"})
        df2["date"] = pd.to_datetime(df2["date"], errors="coerce")
        return df2.dropna(subset=["rec_key", "date", "pid"])

    # gather event frames
    evs = []
    if isinstance(data, dict):
        ae_df = data.get("nhse_eng_ed")
        apc_df = data.get("nhse_eng_inpat")
        op_df = data.get("nhse_eng_outpat")
        ev_ae = extract_events_from_entity(ae_df, ["aekey"], ["arrivaldate", "arrival"])
        ev_apc = extract_events_from_entity(apc_df, ["epikey"], ["admidate", "admi", "episode", "epi"])
        ev_op = extract_events_from_entity(op_df, ["attendkey", "attend_key"], ["apptdate", "appt", "appointment"])
        for e in (ev_ae, ev_apc, ev_op):
            if e is not None and not e.empty:
                evs.append(e)
    else:
        # single merged DF: try to extract any of the keys/dates from the DF
        df = data
        ev_ae = extract_events_from_entity(df, ["aekey"], ["arrivaldate", "arrival"])
        ev_apc = extract_events_from_entity(df, ["epikey"], ["admidate", "admi", "episode", "epi"])
        ev_op = extract_events_from_entity(df, ["attendkey", "attend_key"], ["apptdate", "appt", "appointment"])
        for e in (ev_ae, ev_apc, ev_op):
            if e is not None and not e.empty:
                evs.append(e)

    target = merged if merged is not None else (data.get("participant") if isinstance(data, dict) else data)
    if target is None:
        raise ValueError("need a merged DF or participant DF available for output / registration check")
    target = unify_pid(target)

    if not evs:
        target["derived.total_hospital_contacts"] = 0
        return target

    ev = pd.concat(evs, ignore_index=True)
    ev = ev.drop_duplicates(["pid", "rec_key"])

    if before_registration:
        reg_col = find_reg_col(target)
        if reg_col is not None:
            regs = target[["pid", reg_col]].copy()
            regs[reg_col] = pd.to_datetime(regs[reg_col], errors="coerce")
            ev = ev.merge(regs.rename(columns={reg_col: "reg"}), on="pid", how="left")
            ev = ev[ev["reg"].notna() & (ev["date"] < ev["reg"])]
        # if no reg_col, fall back to including all events

    counts = ev.groupby("pid", as_index=False)["rec_key"].agg(n_contacts="count").rename(columns={"n_contacts": "derived.total_hospital_contacts"})
    out = target.merge(counts, on="pid", how="left")
    out["derived.total_hospital_contacts"] = out["derived.total_hospital_contacts"].fillna(0).astype(int)

    # winsorise at percentile if requested
    if winsorize_pct is not None and 0 < winsorize_pct < 1:
        q = out["derived.total_hospital_contacts"].quantile(winsorize_pct)
        # clamp > q to q (round to int)
        q_int = int(np.floor(q))
        out["derived.total_hospital_contacts"] = out["derived.total_hospital_contacts"].clip(upper=q_int).astype(int)

    return out

# ------------------------------------------------------------------------------------
# Registry tying names to implementations + light input hints (via any_of)
#   We prefer 'any_of' so derives can be attempted and no-op if inputs are absent.
#   This plays nicely with process_*(..., derive="auto") defaults.
# ------------------------------------------------------------------------------------

DERIVE_REGISTRY: Dict[str, DeriveSpec] = {
    # Participant entity
    "registration_date": {
        "fn": registration_date,
        "all_of": [
            "participant.registration_year",
            "participant.registration_month",
        ],
    },
    "age_at_registration": {
        "fn": age_at_registration,
        "all_of": [
            "participant.registration_year",
            "participant.registration_month",
            "participant.birth_year",
            "participant.birth_month",
        ],
    },

    # Age group
    "age_group": {
        "fn": age_group,
        "all_of": [
            "derived.age_at_registration",
        ],
    },
    # Self-reported sex
    "sex": {
        "fn": self_reported_sex,
        "all_of": ["participant.demog_sex_2_1", "participant.demog_sex_1_1"],
    },

    # Clinic measurements entity
    "bmi": {
        "fn": bmi,
        "all_of": ["clinic_measurements.height", "clinic_measurements.weight"],
    },
    "bmi_status": {
        "fn": bmi_status,
        "all_of": ["derived.bmi"],
    },

    # Questionnaire entity
    "vape_status": {
        "fn": vape_status,
        "all_of": ["questionnaire.smoke_vape_1_1", "questionnaire.smoke_tobacco_type_1_m"],
    },
    "smoke_status_v1": {
        "fn": smoke_status_v1,
        "all_of": [
            "questionnaire.smoke_status_1_1",
            "questionnaire.smoke_tobacco_prev_1_1",
            "questionnaire.smoke_100_times_1_1",
        ],
    },
    # small recoders (so smoke_status_v2 can work when only raw inputs exist)
    "tobacco_ever": {
        "fn": tobacco_ever,
        "all_of": ["questionnaire.smoke_tobacco_type_1_m"],
    },
    "tobacco_reg": {
        "fn": tobacco_reg,
        "all_of": ["questionnaire.smoke_reg_1_m"],
    },
    "smoke_status_v2": {
        "fn": smoke_status_v2,
        "all_of": [
            "questionnaire.smoke_reg_1_m",
            "questionnaire.smoke_tobacco_type_1_m",
            "questionnaire.smoke_status_2_1",
            "questionnaire.smoke_100_times_2_1",
        ],
    },
    "walk_16_10": {
        "fn": walk_16_10,
        "all_of": ["questionnaire.activity_walk_days_2_1"],
    },
    "medicat_expand": {
        "fn": medicat_expand,
        "all_of": ["questionnaire.medicat_1_m"],
    },
    "any_hospital_contact": {
        "fn": any_hospital_contact,
        "all_of": [
            # ED
            "nhse_eng_ed.aekey",
            "nhse_eng_ed.arrivaldate",
            # Inpatient
            "nhse_eng_inpat.epikey",
            "nhse_eng_inpat.admidate",
            # Outpatient
            "nhse_eng_outpat.attendkey",
            "nhse_eng_outpat.apptdate",
            "derived.registration_date",
        ],
    },
    "total_hospital_contacts": {
        "fn": total_hospital_contacts,
        "all_of": [
            "nhse_eng_ed.aekey", 
            "nhse_eng_ed.arrivaldate",
            "nhse_eng_inpat.epikey", 
            "nhse_eng_inpat.admidate",
            "nhse_eng_outpat.attendkey", 
            "nhse_eng_outpat.apptdate",
            "derived.registration_date"
        ],
    },
    "ae_visits": {
        "fn": ae_visits,
        "all_of": [
            "nhse_eng_ed.aekey",
            "nhse_eng_ed.arrivaldate",
            "derived.registration_date",
        ],
    },
    "apc_visits": {
        "fn": apc_visits,
        "all_of": [
            "nhse_eng_inpat.epikey",
            "nhse_eng_inpat.admidate",
            "derived.registration_date",
        ],
    },
    "op_visits": {
        "fn": op_visits,
        "all_of": ["nhse_eng_outpat.attendkey", "nhse_eng_outpat.apptdate", "derived.registration_date"],
    },
}

__all__ = [
    "DeriveSpec",
    "DERIVE_REGISTRY",
    # participant
    "self_reported_sex",
    "registration_date",
    "age_at_registration",
    "age_group",
    # clinic measurements
    "bmi",
    "bmi_status",
    # questionnaire
    "vape_status",
    "smoke_status_v1",
    "smoke_status_v2",
    "tobacco_ever",
    "tobacco_reg",
    "walk_16_10",
    "medicat_expand",
    # health records
    "any_hospital_contact",
    "total_hospital_contacts",
    "ae_visits",
    "apc_visits",
    "op_visits",
]
