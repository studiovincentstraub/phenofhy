# _filter_funcs.py
from typing import Dict, Tuple, List, Optional
from functools import reduce
import pandas as pd
import numpy as np
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def implausible_age_year_combinations(df: pd.DataFrame) -> pd.DataFrame:
    """
    Remove rows that contain known implausible sentinel values for birth
    year or birth month.
    - Keeps behavior identical to the original: removes sentinel -999 for
      'participant.birth_year' and 'participant.birth_month'.
    - Skips checks if columns are missing.
    - Returns a filtered DataFrame (does not modify in-place).
    """
    out = df
    if 'participant.birth_year' in out.columns:
        out = out[out['participant.birth_year'] != -999]
    if 'participant.birth_month' in out.columns:
        out = out[out['participant.birth_month'] != -999]
    return out


def _filter_clinic_measurements_plausibility(
    df: pd.DataFrame,
    ranges: Optional[Dict[str, Tuple[float, float]]] = None,
) -> pd.DataFrame:
    """
    Internal helper: drop rows where clinic measurement values are outside
    plausible bounds. NA values are preserved (not dropped).

    Default plausible ranges (can be overridden by passing `ranges`):
      - clinic_measurements.height   : (100, 250)  # cm
      - clinic_measurements.weight   : (30, 200)   # kg
      - clinic_measurements.waist    : (40, 150)   # cm

    The function will only check columns that exist in `df`.
    """
    default_ranges = {
        "clinic_measurements.height": (100.0, 250.0),
        "clinic_measurements.weight": (30.0, 200.0),
        "clinic_measurements.waist": (40.0, 150.0),
    }
    use_ranges = default_ranges if ranges is None else {**default_ranges, **ranges}

    mask = pd.Series(True, index=df.index)
    for col, (lo, hi) in use_ranges.items():
        if col not in df.columns:
            continue
        # keep rows where value is within [lo, hi] OR is NA
        cond = df[col].between(lo, hi, inclusive="both") | df[col].isna()
        mask &= cond

    return df.loc[mask]


def remove_known_errors(
    df: pd.DataFrame,
    *,
    clinic_ranges: Optional[Dict[str, Tuple[float, float]]] = None,
) -> pd.DataFrame:
    """
    Top-level cleaning wrapper that applies known error-removal helpers,
    but only when the required columns are present.

    - Age/year filtering runs only if birth year or birth month exists.
    - Clinic plausibility filtering runs only if at least one clinic
      measurement column exists (height, weight, waist).
    """
    out = df

    # 1) Implausible age/year combinations
    age_cols = {"participant.birth_year", "participant.birth_month"}
    if age_cols & set(out.columns):
        logger.info("Applying implausible age/year filters")
        out = implausible_age_year_combinations(out)
    else:
        logger.debug("Skipping age/year filters (no relevant columns found)")

    # 2) Clinic measurement plausibility
    clinic_cols = {
        "clinic_measurements.height",
        "clinic_measurements.weight",
        "clinic_measurements.waist",
    }
    if clinic_cols & set(out.columns):
        logger.info("Applying clinic measurement plausibility filters")
        out = _filter_clinic_measurements_plausibility(out, ranges=clinic_ranges)
    else:
        logger.debug("Skipping clinic filters (no clinic columns found)")

    return out


def apply_row_filters(
    df: pd.DataFrame,
    *,
    ranges: Optional[Dict[str, Tuple[float, float]]] = None,
    exprs: Optional[List[str]] = None,
    inclusive: str = "both",     # "both" | "neither" | "left" | "right"
    keep_na: bool = False,
    ignore_missing_range_cols: bool = False,
) -> pd.DataFrame:
    """
    Generic, reusable row-level filtering helper.

    - ranges: mapping col -> (lo, hi) to keep rows where col between(lo, hi).
    - exprs: list of pandas eval() expressions to be ANDed into the mask.
    - keep_na: if True, rows with NA in a range column are retained.
    - ignore_missing_range_cols: if True, absent range columns are skipped.
    """
    mask = pd.Series(True, index=df.index)

    if ranges:
        for col, (lo, hi) in ranges.items():
            if col not in df.columns:
                if ignore_missing_range_cols:
                    continue
                raise KeyError(f"Range filter column not found: {col}")
            cond = df[col].between(lo, hi, inclusive=inclusive)
            if keep_na:
                cond = cond | df[col].isna()
            mask &= cond

    if exprs:
        for e in exprs:
            mask &= df.eval(e)

    return df.loc[mask]


def floor_age_series(s: pd.Series) -> pd.Series:
    """
    Convert a series to floored integer ages, treating negative ages as missing.
    Returns a pandas nullable Int64 series.
    """
    s_num = pd.to_numeric(s, errors="coerce")
    s_num = s_num.where(s_num >= 0)                # negative -> NaN
    floored = pd.Series(np.floor(s_num), index=s.index)
    return floored.astype("Int64")


def filter_preferred_nonresponse(df: pd.DataFrame) -> pd.DataFrame:
    """
    Remove rows with non-substantive ("prefer not to say", "don't know", or missing)
    responses in key demographic variables.

    Columns are only checked if present.
    """
    logger.info("Applying preferred non-response exclusion filters")

    conditions = []

    if "participant.demog_sex_1_1" in df.columns:
        s = df["participant.demog_sex_1_1"]
        conditions.append(~s.isin([3, -3]) & s.notna())

    if "participant.demog_sex_2_1" in df.columns:
        s = df["participant.demog_sex_2_1"]
        conditions.append(~s.isin([3, -3]) & s.notna())

    if "participant.demog_ethnicity_1_1" in df.columns:
        s = df["participant.demog_ethnicity_1_1"]
        conditions.append(~s.isin([19, -3]))

    if "questionnaire.housing_income_1_1" in df.columns:
        s = df["questionnaire.housing_income_1_1"]
        conditions.append(~s.isin([-1, -3]) & s.notna())

    if not conditions:
        logger.warning("No preferred non-response filters applied (no columns found)")
        return df

    mask = reduce(lambda a, b: a & b, conditions)
    return df.loc[mask]
