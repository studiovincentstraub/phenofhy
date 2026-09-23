# _filter_funcs.py
from typing import Dict, Tuple, List, Optional
from functools import reduce
import pandas as pd
import numpy as np
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def implausible_age_year_combinations(df: pd.DataFrame) -> pd.DataFrame:
    """Remove rows with implausible birth year/month sentinel values.

    Args:
        df: Input dataframe.

    Returns:
        Filtered dataframe with sentinel -999 values removed when columns exist.
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
    """Filter out-of-range clinic measurements while preserving NAs.

    Args:
        df: Input dataframe.
        ranges: Optional mapping of column name to (low, high) bounds.

    Returns:
        Filtered dataframe with rows outside bounds removed.
    """
    default_ranges = {
        "clinic_measurements.height": (90, 299),
        "clinic_measurements.weight": (20.0, 400.0),
        "clinic_measurements.waist": (30.0, 200.0),
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
    """Apply known error-removal helpers when required columns exist.

    Args:
        df: Input dataframe.
        clinic_ranges: Optional mapping of clinic column ranges.

    Returns:
        Filtered dataframe with known error rows removed.
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
    """Apply range- and expression-based row filters.

    Args:
        df: Input dataframe.
        ranges: Mapping of column -> (low, high) bounds.
        exprs: Optional list of pandas eval() expressions to AND into the mask.
        inclusive: Bound inclusion for between().
        keep_na: If True, retain rows with NA in range columns.
        ignore_missing_range_cols: If True, skip missing range columns.

    Returns:
        Filtered dataframe.
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
    """Floor ages to integer years, treating negatives as missing.

    Args:
        s: Input series of ages.

    Returns:
        Nullable Int64 series of floored ages.
    """
    s_num = pd.to_numeric(s, errors="coerce")
    s_num = s_num.where(s_num >= 0)                # negative -> NaN
    floored = pd.Series(np.floor(s_num), index=s.index)
    return floored.astype("Int64")


def filter_preferred_nonresponse(df: pd.DataFrame) -> pd.DataFrame:
    """Remove rows with preferred non-response values in key demographics.

    Args:
        df: Input dataframe.

    Returns:
        Filtered dataframe with non-substantive responses removed.
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
