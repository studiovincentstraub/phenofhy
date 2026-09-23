"""Default coalescing rules for common traits.

Edit or extend DEFAULT_COALESCE_RULES as new questionnaire traits are added.
Import this dict in preprocess.py and pass it to process_fields by default.
"""

import re
import ast
import glob
import logging
import pandas as pd, numpy as np
from typing import Dict, Any, Iterable, Optional, Callable, Sequence, Tuple
from scipy.stats import entropy

logger = logging.getLogger(__name__)


CoalesceRule = Dict[str, Any]
CoalesceConfig = Dict[str, CoalesceRule]


def _default_normalize(x):
    """Normalize a value for coalescing.

    Args:
        x: Input value.

    Returns:
        Normalized value with empty strings converted to NA.
    """
    if isinstance(x, str):
        x = x.strip()        # REMOVE .lower()
        if x == "":
            return pd.NA
    return x

def rule_cat(
    sources: Iterable[str],
    *,
    informative: Optional[Iterable[Any]] = None,
    value_map: Optional[Dict[Any, Any]] = None,
    nonresponse: Optional[Iterable[Any]] = None,
    collapse: bool = True,
    priority: str = "last",
    unknown: str = "Unknown",
    missing: str = "nan",
    preserve_dtype: bool = True,
) -> CoalesceRule:
    """Build a categorical coalescing rule for coalesce_traits.

    Args:
        sources: Ordered list of source columns to coalesce.
        informative: Iterable of informative values. If omitted, non-null values not
            in nonresponse are treated as informative.
        value_map: Optional mapping to canonicalize labels or convert codes to labels.
        nonresponse: Values considered non-informative.
        collapse: If True, collapse non-informative values to unknown.
        priority: "first" or "last" to pick which source wins on ties.
        unknown: Label used for collapsed non-informative values.
        missing: Label used when collapse is False and all sources are NA.
        preserve_dtype: If True, preserve or extend categorical dtype categories.

    Returns:
        Rule dict compatible with coalesce_traits.
    """
    rule: CoalesceRule = {
        "sources": list(sources),
        "type": "categorical",
        "collapse_nonresponse": collapse,
        "priority": priority,
        "unknown_label": unknown,
        "missing_label": missing,
        "preserve_dtype": preserve_dtype,
    }
    if value_map is not None:
        rule["value_map"] = value_map
    if informative is not None:
        rule["informative"] = set(informative)
    if nonresponse is not None:
        rule["nonresponse_values"] = set(nonresponse)
    return rule


def rule_num(
    sources: Iterable[str],
    *,
    informative: Optional[Callable[[Any], bool]] = None,
    astype: str = "Int64",
    priority: str = "last",
) -> CoalesceRule:
    """Build a numeric coalescing rule for coalesce_traits.

    Args:
        sources: Ordered list of source columns to coalesce.
        informative: Optional predicate for valid values.
        astype: Final dtype cast for the derived column.
        priority: "first" or "last" to pick which source wins on ties.

    Returns:
        Rule dict compatible with coalesce_traits.
    """
    if informative is None:
        informative = lambda v: (pd.notna(v) and np.isfinite(v))
    return {
        "sources": list(sources),
        "type": "numeric",
        "informative": informative,
        "astype": astype,
        "priority": priority,
    }


def coalesce_traits(
    df: pd.DataFrame,
    rules: CoalesceConfig,
    *,
    global_unknown_label: str = "Unknown",
    global_missing_label: str = "nan",
) -> pd.DataFrame:
    """Coalesce versioned columns into unified traits.

    Args:
        df: Input dataframe.
        rules: Mapping of output column to coalescing rule.
        global_unknown_label: Default unknown label for categorical outputs.
        global_missing_label: Default missing label for categorical outputs.

    Returns:
        DataFrame with unified traits added.
    """
    out = df.copy()

    for out_col, spec in rules.items():
        sources: Iterable[str] = spec.get("sources", [])
        keep = [c for c in sources if c in out.columns]
        if not keep:
            continue

        trait_type = spec.get("type")
        normalize = spec.get("normalize", _default_normalize)
        # explicit value_map from rule (may be None)
        explicit_value_map = spec.get("value_map", None)
        informative = spec.get("informative")
        nonresp_vals = set(spec.get("nonresponse_values", []))
        collapse_nonresp = bool(spec.get("collapse_nonresponse", True))
        unknown_label = spec.get("unknown_label", global_unknown_label)
        missing_label = spec.get("missing_label", global_missing_label)
        preserve_dtype = bool(spec.get("preserve_dtype", True))
        code_map = spec.get("code_map")             # for categorical → integer recode
        astype = spec.get("astype")                 # final dtype cast

        # --- Normalize strings (trim+lower) first
        if normalize:
            norm = out[keep].apply(lambda col: col.map(normalize))
        else:
            norm = out[keep].copy()

        # --- Determine / apply a code->label mapping (value_map)
        # Priority: explicit rule value_map -> DERIVED_CODEBOOK lookup (out_col or a source)
        value_map = explicit_value_map
        if value_map is None and "DERIVED_CODEBOOK" in globals():
            dcb = globals().get("DERIVED_CODEBOOK", {})
            # prefer mapping keyed by the unified output
            if out_col in dcb:
                value_map = {str(k): str(v).strip() for k, v in dcb[out_col].items()}
            else:
                # otherwise look for any source-specific mapping and use first found
                for src in keep:
                    if src in dcb:
                        value_map = {str(k): str(v).strip() for k, v in dcb[src].items()}
                        break

        if value_map:
            # Build flexible lookup: accept int, string, padded, etc.
            lookup = {}
            for k, v in value_map.items():
                lookup[k] = v
                lookup[str(k)] = v
                try:
                    lookup[int(k)] = v
                except:
                    pass

            def _map_one(x):
                if pd.isna(x):
                    return x
                return lookup.get(x) or lookup.get(str(x)) or x

            norm = norm.apply(lambda col: col.map(_map_one))
                    
        # --- Infer trait type if not provided
        if trait_type is None:
            trait_type = "numeric" if all(pd.api.types.is_numeric_dtype(out[c]) for c in keep) else "categorical"

        # --- Build informative mask (per-column)
        if trait_type == "numeric":
            if callable(informative):
                ok = norm.apply(lambda col: col.map(informative))
            else:
                ok = norm.apply(lambda col: col.map(lambda v: pd.notna(v) and np.isfinite(v)))
        else:  # categorical
            if informative is None:
                ok = norm.apply(lambda col: col.map(lambda v: (pd.notna(v)) and (v not in nonresp_vals)))
            elif callable(informative):
                ok = norm.apply(lambda col: col.map(informative))
            else:
                inf_set = set(informative)
                ok = norm.apply(lambda col: col.map(lambda v: (v in inf_set)))

        # --- Mask non-informative, then pick preferred version according to priority
        masked = norm.where(ok)
        priority = spec.get("priority", "first")

        if priority == "last":
            unified = masked.ffill(axis=1).iloc[:, -1]
        else:
            unified = masked.bfill(axis=1).iloc[:, 0]

        # --- Handle nonresponse for categorical traits (operate on string labels)
        if trait_type == "categorical":

            # Ensure unified is a DataFrame for column-wise ops
            if isinstance(unified, pd.Series):
                unified = unified.to_frame()

            # Convert to object dtype BEFORE inserting strings to avoid Categorical errors
            unified = unified.astype(object)

            if collapse_nonresp:
                # Fill all NA with unknown label
                unified = unified.fillna(unknown_label)
            else:
                # Rows where all sources are NA → missing label
                both_na = norm[keep].isna().all(axis=1)
                unified = unified.where(~both_na, missing_label).fillna(unknown_label)

            # Replace pandas NA safely (no lowercase transformation!)
            unified = unified.replace({pd.NA: missing_label})


            # Preserve/extend category dtype: build categories from mapped values where possible
            if preserve_dtype:
                # Gather candidate categories:
                # Prefer known labels from DERIVED_CODEBOOK if present, else from the data
                candidate_labels = []
                dcb = globals().get("DERIVED_CODEBOOK", {})
                # prefer out_col mapping
                if out_col in dcb:
                    candidate_labels = [str(v).strip() for v in dcb[out_col].values()]
                else:
                    # otherwise, search source mappings
                    for src in keep:
                        if src in dcb:
                            candidate_labels = [str(v).strip() for v in dcb[src].values()]
                            break

                # include any observed labels from unified
                observed = pd.Series(unified.values.ravel()).dropna().unique().tolist()
                # combine (preserve ordering: candidate_labels first, then observed)
                wanted = pd.unique(pd.Series(list(candidate_labels) + observed + [unknown_label, missing_label], dtype="object"))
                unified = pd.Categorical(pd.Series(unified.values.ravel()), categories=wanted)
                unified = pd.Series(unified, index=df.index)
            else:
                # If not preserving dtype, just return Series of objects (strings)
                if isinstance(unified, pd.DataFrame) and unified.shape[1] == 1:
                    unified = unified.iloc[:, 0]
                unified = pd.Series(unified.values.ravel(), index=df.index)

            # Optional categorical -> integer coding
            if code_map:
                unified = pd.Series(unified).map(code_map)
                if astype:
                    unified = unified.astype(astype)
                else:
                    unified = unified.astype("Int64")

        else:
            # numeric: leave non-informative as NaN so downstream stats handle it
            if astype:
                unified = unified.astype(astype)

        # --- Ensure we have a Series aligned with the original DataFrame index
        if isinstance(unified, pd.DataFrame) and unified.shape[1] == 1:
            unified = unified.iloc[:, 0]

        if not isinstance(unified, pd.Series):
            if pd.api.types.is_scalar(unified):
                unified = pd.Series([unified] * len(df), index=df.index)
            else:
                unified = pd.Series(unified, index=df.index)

        unified = unified.reindex(df.index)

        out[out_col] = unified

    return out


    
def build_rules(overrides: CoalesceConfig | None = None,
               extend: CoalesceConfig | None = None) -> CoalesceConfig:
    """Combine default coalescing rules with user overrides.

    Args:
        overrides: Replace or modify existing default rules by key.
        extend: Add entirely new derived outputs.

    Returns:
        Combined rules dictionary.

    Raises:
        KeyError: If extend conflicts with an existing rule key.
    """
    rules = dict(DEFAULT_COALESCE_RULES)
    if overrides:
        rules.update(overrides)
    if extend:
        # guard against accidental overwrite
        for k in extend:
            if k in rules:
                raise KeyError(f"extend conflicts with existing rule: {k}")
        rules.update(extend)
    return rules



DEFAULT_COALESCE_RULES: CoalesceConfig = {
    # Smoking status (V1→V2)
    "derived.smoke_status": rule_cat(
        ["derived.smoke_status_v1", "derived.smoke_status_v2"],
        informative={"Current", "Former", "Never"},
        collapse=False,
        missing="nan",
        unknown="Unknown",
    ),

    # Age at first regular smoking (V1→V2), plausibility 5–100
    "derived.smoke_reg_first_age": rule_num(
        ["questionnaire.smoke_reg_first_age_1_1", "questionnaire.smoke_reg_first_age_2_1"],
        informative=lambda v: (pd.notna(v) and np.isfinite(v) and 5 <= float(v) <= 100),
        astype="Int64",
    ),
}


DERIVED_CODEBOOK = {
    "derived.sex": {
        1: "Male",
        2: "Female",
        3: "Intersex",
        4: "Other",
        -3: "Prefer not to answer",
    },
    "derived.bmi_status": {
        0: "Underweight",
        1: "Normal",
        2: "Overweight",
        3: "Obese",
        -9: "nan",
    },
    "derived.vape_status": {
        1: "Ever used",
        0: "Never used",
        -3: "Prefer not to answer",
    },
    "derived.smoke_status_v1": {
        2: "Current",
        1: "Former",
        0: "Never",
        -1: "Unknown",
        -9: "nan",
    },
    "derived.smoke_status_v2": {
        2: "Current",
        1: "Former",
        0: "Never",
        -1: "Unknown",
        -9: "nan",
    },
    "derived.smoke_status": {
        2: "Current",
        1: "Former",
        0: "Never",
        -1: "Unknown",
        -9: "nan",
    },
    "derived.walk_16_10": {
        1: "Meets threshold",
        0: "Below threshold",
    },
    # Add more as needed...
}


# --- Default presentation helpers for display.py (ordering & exclusions) ---

DEFAULT_CATEGORY_ORDER = {
    "questionnaire.alcohol_curr_1_1": [
        "Daily or almost daily",
        "3-4 times a week",
        "Once or twice a week",
        "1-3 times a month",
        "Special occasions only",
        "Never",
    ],
    "derived.smoke_status": [
        "Current",
        "Former",
        "Never",
        "Unknown",
        "nan",
    ],
    "derived.vape_status": [
        "ever used",
        "never used",
    ],
    "questionnaire.health_status_curr_1_1": [
        "Excellent",
        "Good",
        "Fair",
        "Poor",
    ],
    "questionnaire.health_status_chronic_1_1": [
        "Yes",
        "No",
    ],
    # add more traits here as you need...
}

DEFAULT_EXCLUDED_CATEGORIES = [
    "Do not know",
    "Unknown",
    "Prefer not to answer",
    "NA",
]


MEDICAT_ABBREV = {
    "autoimmune": "auto",
    "auto": "auto",
    "bone": "bone",
    "cancer": "cancer",
    "diabetic": "diab",
    "digest": "digest",
    "digestive": "digest",
    "endocrine": "endo",
    "circulatory": "cvd",
    "lung": "resp",
    "breathing": "resp",
    "mental": "mental",
    "insomnia": "mental",
    "neuro": "neuro",
    "neurological": "neuro",
    "pain": "pain",
    "repro": "repro",
    "supplement": "supp",
    "nutritional": "supp",
}


DEFAULT_MEDICAT_GROUP_MAP = {
    "derived.medicates_cvd_diab": ["derived.medicates_cvd", "derived.medicates_diab"],
    "derived.medicates_mental_pain": ["derived.medicates_mental", "derived.medicates_pain"],
    "derived.medicates_auto_endo": ["derived.medicates_auto", "derived.medicates_endo"],
    "derived.medicates_bone_supp": ["derived.medicates_bone", "derived.medicates_supp"],
}