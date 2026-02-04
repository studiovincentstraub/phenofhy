import logging
import numpy as np
import pandas as pd
from pathlib import Path
from functools import reduce
from collections.abc import Mapping
from pyspark.sql import functions as F
from functools import lru_cache, partial
from pyspark.sql import DataFrame as SparkDataFrame
from typing import Dict, Iterable, Tuple, List, Optional, Any, Union, Literal, Sequence

# project imports
from ._rules import DEFAULT_COALESCE_RULES, coalesce_traits
from ._filter_funcs import (
    remove_known_errors,
    apply_row_filters,
)
# registry (externalized)
try:
    from ._derive_funcs import DERIVE_REGISTRY as DEFAULT_DERIVE_REGISTRY, DeriveSpec, expand_multi_code_column 
except Exception:
    from _derive_funcs import DERIVE_REGISTRY as DEFAULT_DERIVE_REGISTRY, DeriveSpec, expand_multi_code_column   # type: ignore
    
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# local type aliases (renamed to avoid shadowing project symbols)
CoalesceRuleSpec = Dict[str, Any]         # per-trait rule
CoalesceRulesMap = Dict[str, CoalesceRuleSpec]  # {out_col: rule}


# ------------------------------------------------------------------------------------
# Core I/O + small helpers
# ------------------------------------------------------------------------------------

def load_input(input_data):
    if isinstance(input_data, (str, Path)):
        input_path = Path(input_data)
        logger.info("Reading input file: %s", input_path)
        return pd.read_csv(input_path)
    elif isinstance(input_data, pd.DataFrame):
        logger.info("Using input DataFrame with %s rows", len(input_data))
        return input_data.copy()
    else:
        raise ValueError("input_data must be a file path or a pandas DataFrame")
        

def _dfcol_to_meta_key(col: str) -> str:
    """'questionnaire.smoke_status_1_1' -> 'SMOKE_STATUS_1_1'."""
    return col.split(".")[-1].upper()


@lru_cache(maxsize=8)
def _load_codebook(csv_path: str) -> pd.DataFrame:
    """Load CSV with columns: coding_name, code, meaning."""
    df = pd.read_csv(csv_path, dtype={"coding_name": "string"})
    df = df.rename(columns={c: c.strip().lower() for c in df.columns})
    required = {"coding_name", "code", "meaning"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Codebook CSV missing columns: {missing}")
    return df


def _build_maps_for_traits(
    traits: Iterable[str],
    df: pd.DataFrame,
    csv_path: Optional[str],
) -> Dict[str, Tuple[Dict[Any, str], Dict[str, Any]]]:
    """
    Build code<->label maps from the external codebook for the given traits.
    Returns {trait: (code_to_label, label_to_code)} for traits present in df & CSV.
    """
    out: Dict[str, Tuple[Dict[Any, str], Dict[str, Any]]] = {}
    if not csv_path:
        return out
    meta = _load_codebook(csv_path)
    for col in traits:
        if col not in df.columns:
            continue
        key = _dfcol_to_meta_key(col)
        rows = meta.loc[meta["coding_name"] == key]
        if rows.empty:
            continue

        # Coerce code dtype to match df[col] when possible
        def _coerce(code):
            s = df[col]
            if pd.api.types.is_integer_dtype(s.dtype) or str(s.dtype).startswith("Int"):
                try:
                    return int(code)
                except Exception:
                    pass
            if pd.api.types.is_float_dtype(s.dtype):
                try:
                    return float(code)
                except Exception:
                    pass
            return str(code)

        c2l = {_coerce(r.code): str(r.meaning) for _, r in rows.iterrows()}
        l2c = {v: k for k, v in c2l.items()}
        out[col] = (c2l, l2c)
    return out


def resolve_categoricals_and_labels(
    df: pd.DataFrame,
    traits: Iterable[str],
    *,
    label_mode: str = "labels",                 # "labels" | "codes"
    codebook_csv: Optional[str] = None,
    autodetect_coded_categoricals: bool = True,
    autodetect_max_levels: int = 10,
) -> Tuple[pd.DataFrame, List[str]]:
    """
    Prepare a dataframe and the list of categorical traits for summary.
    """
    tmp = df.copy()
    traits = [t for t in traits if t in tmp.columns]

    # Codebook maps (if provided)
    maps = _build_maps_for_traits(traits, tmp, codebook_csv)

    # Apply chosen presentation
    if label_mode not in {"labels", "codes"}:
        raise ValueError("label_mode must be 'labels' or 'codes'.")

    for col in traits:
        if col in maps:
            code2label, label2code = maps[col]
            if label_mode == "labels":
                if not (pd.api.types.is_string_dtype(tmp[col].dtype) or pd.api.types.is_categorical_dtype(tmp[col].dtype)):
                    tmp[col] = tmp[col].map(code2label).astype("category")
            else:  # "codes"
                if pd.api.types.is_string_dtype(tmp[col].dtype) or pd.api.types.is_categorical_dtype(tmp[col].dtype):
                    tmp[col] = tmp[col].map(label2code)
                    if all(isinstance(v, (int, np.integer)) or pd.isna(v) for v in tmp[col]):
                        tmp[col] = tmp[col].astype("Int64")

    # Decide categorical traits without the user listing them
    cat_traits: List[str] = []
    for c in traits:
        s = tmp[c]
        if pd.api.types.is_categorical_dtype(s.dtype) or pd.api.types.is_object_dtype(s.dtype):
            cat_traits.append(c)
            continue
        if autodetect_coded_categoricals and (pd.api.types.is_integer_dtype(s.dtype) or pd.api.types.is_float_dtype(s.dtype)):
            non_na = s.dropna()
            try:
                intlike = pd.api.types.is_integer_dtype(s.dtype) or (non_na.mod(1).eq(0).all())
            except Exception:
                intlike = False
            if intlike and 1 < non_na.nunique() <= autodetect_max_levels:
                cat_traits.append(c)

    return tmp, cat_traits


def _require_inputs(df: pd.DataFrame, name: str, *, all_of: Optional[List[str]], any_of: Optional[List[str]]) -> None:
    """Strictly enforce input presence for a selected derivation."""
    cols = set(df.columns)
    if all_of:
        missing = set(all_of) - cols
        if missing:
            raise KeyError(f"{name}: missing required columns {missing}")
    if any_of:
        if not any(c in cols for c in any_of):
            raise KeyError(f"{name}: requires at least one of {any_of}, but none are present")


# ------------------------------------------------------------------------------------
# Entity-aware derivation selection and execution
# ------------------------------------------------------------------------------------

# If a derive references columns from multiple entities (e.g., BMI can be computed
# from clinic_measurements.*), use overrides to assign it to a "home" entity.
DERIVE_ENTITY_OVERRIDES: Dict[str, str] = {
    "bmi": "clinic_measurements",
    "bmi_status": "clinic_measurements",
}

def _columns_for_spec(spec: Dict[str, object]) -> List[str]:
    cols: List[str] = []
    for k in ("all_of", "any_of"):
        v = spec.get(k)  # type: ignore
        if isinstance(v, (list, tuple)):
            cols.extend([c for c in v if isinstance(c, str)])
    return cols

def _select_derives_for_entity(
    entity: str,
    registry: Dict[str, DeriveSpec],
    *,
    overrides: Optional[Dict[str, str]] = None,
) -> List[str]:
    """Select derives whose inputs fully belong to the given entity, plus explicit overrides."""
    
    pref = f"{entity}."
    selected = []
    
    for name, spec in registry.items():
        cols = _columns_for_spec(spec)
        
        # skip derives without declared inputs
        if not cols:
            continue
        
        # require *all* inputs to start with entity prefix OR be derived.*
        if all(c.startswith(pref) or c.startswith("derived.") for c in cols):
            selected.append(name)
    
    # apply explicit overrides as authoritative
    if overrides:
        for name, ent in overrides.items():
            if name in registry:
                if ent == entity and name not in selected:
                    selected.append(name)
                if ent != entity and name in selected:
                    selected.remove(name)
    
    selected.sort()
    return selected

def _spec_ready(df: pd.DataFrame, spec: Dict[str, Any]) -> bool:
    """
    Is this derivation runnable given what's in df *right now*?
    - all_of: must all be present
    - any_of: at least one present
    - ready (callable): optional custom readiness check
    """
    cols = set(df.columns)
    all_of = spec.get("all_of")
    any_of = spec.get("any_of")
    if all_of and not set(all_of).issubset(cols):
        return False
    if any_of and not any(c in cols for c in any_of):
        return False
    # optional custom readiness hook for complex cases (e.g., smoke_status_v2)
    ready_fn = spec.get("ready")
    if callable(ready_fn):
        try:
            return bool(ready_fn(df))
        except Exception:
            return False
    return True

def _run_derivations_auto(
    df: pd.DataFrame,
    selected: List[str],
    registry: Dict[str, DeriveSpec],
) -> pd.DataFrame:
    """
    Opportunistic execution:
      - repeatedly run any derives that are "ready" (inputs present),
        updating df each time; stop when no progress is possible.
      - never raises for missing inputs; simply skips unready derives.
    """
    remaining = list(selected)
    made_progress = True
    out = df
    while remaining and made_progress:
        made_progress = False
        next_remaining: List[str] = []
        for name in remaining:
            spec = registry[name]
            if _spec_ready(out, spec):  # type: ignore[arg-type]
                fn = spec["fn"]         # type: ignore[index]
                out = fn(out)           # type: ignore[misc]
                logger.info("Applied %s (auto)", name)
                made_progress = True
            else:
                next_remaining.append(name)
        remaining = next_remaining
    if remaining:
        logger.info("Auto-derive skipped (unready inputs): %s", remaining)
    return out

def _run_derivations_strict(
    df: pd.DataFrame,
    selected: List[str],
    registry: Dict[str, DeriveSpec],
) -> pd.DataFrame:
    """
    Strict execution:
      - raises KeyError when a selected derive is missing required inputs.
    """
    if not selected:
        return df
    unknown = [name for name in selected if name not in registry]
    if unknown:
        raise ValueError(f"Unknown derive names: {unknown}. Known: {list(registry.keys())}")
    out = df
    for name in selected:
        spec = registry[name]
        fn = spec["fn"]                       # type: ignore[index]
        all_of = spec.get("all_of")           # type: ignore[assignment]
        any_of = spec.get("any_of")           # type: ignore[assignment]
        _require_inputs(out, name, all_of=all_of, any_of=any_of)
        out = fn(out)                         # type: ignore[misc]
        logger.info("Applied %s (strict)", name)
    return out


def _resolve_selected_derives(
    *,
    derive: Union[bool, List[str], Literal["all", "auto"]],
    entity: str,
    registry: Dict[str, DeriveSpec],
    overrides: Optional[Dict[str, str]] = None,
) -> List[str]:
    """
    Selection semantics:
      - derive=False or None     -> []
      - derive=True or "all"     -> list(registry.keys())
      - derive="auto"            -> only derives relevant to `entity`
      - derive=[...]             -> explicit subset (validated later)
    """
    if derive is False or derive is None:
        return []
    if derive is True or (isinstance(derive, str) and derive.lower() == "all"):
        return list(registry.keys())
    if isinstance(derive, str) and derive.lower() == "auto":
        return _select_derives_for_entity(entity, registry, overrides=overrides)
    if isinstance(derive, (list, tuple, set)):
        return list(derive)
    raise ValueError("`derive` must be False/None, True/'all', 'auto', or a list of names.")


def _entity_fields_core(
    input_data,
    *,
    entity: str,
    derive: Union[bool, List[str], Literal["all", "auto"]] = "auto",
    derive_registry: Optional[Dict[str, DeriveSpec]] = None,
    coalesce_rules: Optional[Dict[str, dict]] = None,
    # age filtering (typically most useful for participant)
    auto_row_filters: bool = False,
    age_col: str = "derived.age_at_registration",
    min_age: int = 18,
    max_age: int = 110,
    age_group_bins: Optional[List[float]] = None,
    age_group_labels: Optional[List[str]] = None,
    floor_age: bool = False,
    extra_ranges: Optional[Dict[str, Tuple[float, float]]] = None,
    extra_exprs: Optional[List[str]] = None,
    keep_na_in_ranges: bool = False,
) -> pd.DataFrame:
    """
    Load → clean → (optionally) floor age → derive (scoped to `entity`) → (optionally) filter → (optionally) coalesce.
    """
    # 1) Load & clean
    df = load_input(input_data)
    df = remove_known_errors(df)
    logger.info("Using input DataFrame with %s rows", len(df))

    # 2) Choose registry & which derivations to run
    registry = DEFAULT_DERIVE_REGISTRY if derive_registry is None else derive_registry
    selected = _resolve_selected_derives(
        derive=derive, entity=entity, registry=registry, overrides=DERIVE_ENTITY_OVERRIDES
    )
    
    # Allow custom age groups
    if age_group_bins is not None or age_group_labels is not None:
        reg = dict(registry)                      # shallow copy; don't mutate global
        spec = reg.get("age_group", {}).copy()    # copy the spec dict
        orig_fn = spec.get("fn")
        if orig_fn is None:
            raise RuntimeError("age_group spec missing fn in registry")
        spec["fn"] = partial(orig_fn, bins=age_group_bins, labels=age_group_labels)
        reg["age_group"] = spec
        registry = reg

    # 3) floor (not round) age BEFORE running derives, if requested
    if floor_age and (age_col in df.columns):
        s = pd.to_numeric(df[age_col], errors="coerce")
        # optional: treat negative ages as missing (safer than allowing negative integers)
        s = s.where(s >= 0)
        # floor to get integer age (17.6 -> 17) and cast to pandas nullable Int
        df[age_col] = pd.Series(np.floor(s), index=df.index).astype("Int64")

    # 4) Run derivations
    if isinstance(derive, str) and derive.lower() == "auto":
        df = _run_derivations_auto(df, selected, registry)
    else:
        df = _run_derivations_strict(df, selected, registry)

    # 5) Optional row filters
    if auto_row_filters and age_col in df.columns:
        ranges = {age_col: (min_age, max_age)}
        if extra_ranges:
            ranges.update(extra_ranges)
        df = apply_row_filters(
            df,
            ranges=ranges,
            exprs=extra_exprs,
            inclusive="left",
            keep_na=keep_na_in_ranges,
        )

    # 6) Optional coalescing
    rules = DEFAULT_COALESCE_RULES if coalesce_rules is None else coalesce_rules
    if rules:
        df = coalesce_traits(df, rules)

    return df


# ------------------------------------------------------------------------------------
# Public entity-specific entry points
# ------------------------------------------------------------------------------------

def get_dummies(
    df: pd.DataFrame,
    codings_glob: str = "./metadata/*.codings.csv",
    coding_name: str = "MEDICAT_1_M",
    col: str = "questionnaire.medicat_1_m",
    prefix: str = "derived.medicates_",
    exclude_codes: Optional[Sequence[int]] = (-7, -1, -3),
    user_map: Optional[Dict[str, str]] = None,
    inplace: bool = True,
) -> Union[pd.DataFrame, Tuple[pd.DataFrame, pd.DataFrame]]:
    mapping_df = expand_multi_code_column(
        df=df,
        codings_glob=codings_glob,
        coding_name=coding_name,
        col=col,
        prefix=prefix,
        exclude_codes=exclude_codes,
        abbrev_map=user_map,
        inplace=inplace,
    )
    return df


def participant_fields(
    input_data,
    *,
    derive: Union[bool, List[str], Literal["all", "auto"]] = "auto",
    derive_registry: Optional[Dict[str, DeriveSpec]] = None,
    coalesce_rules: Optional[Dict[str, dict]] = None,
    auto_row_filters: bool = True,
    age_col: str = "derived.age_at_registration",
    min_age: int = 18,
    max_age: int = 110,
    floor_age: bool = True,
    age_group_bins: Optional[List[float]] = None,
    age_group_labels: Optional[List[str]] = None,
    extra_ranges: Optional[Dict[str, Tuple[float, float]]] = None,
    extra_exprs: Optional[List[str]] = None,
    keep_na_in_ranges: bool = False,
) -> pd.DataFrame:
    return _entity_fields_core(
        input_data,
        entity="participant",
        derive=derive,
        derive_registry=derive_registry,
        coalesce_rules=coalesce_rules,
        auto_row_filters=auto_row_filters,
        age_col=age_col,
        min_age=min_age,
        max_age=max_age,
        floor_age=floor_age,
        age_group_bins=age_group_bins,            
        age_group_labels=age_group_labels,        
        extra_ranges=extra_ranges,
        extra_exprs=extra_exprs,
        keep_na_in_ranges=keep_na_in_ranges,
    )


def questionnaire_fields(
    input_data,
    *,
    derive: Union[bool, List[str], Literal["all", "auto"]] = "auto",  # derive questionnaire traits if inputs exist
    derive_registry: Optional[Dict[str, DeriveSpec]] = None,
    coalesce_rules: Optional[Dict[str, dict]] = None,
    # questionnaire: no age filtering by default
    auto_row_filters: bool = False,
    age_col: str = "derived.age_at_registration",
    min_age: int = 18,
    max_age: int = 110,
    floor_age: bool = True,
    extra_ranges: Optional[Dict[str, Tuple[float, float]]] = None,
    extra_exprs: Optional[List[str]] = None,
    keep_na_in_ranges: bool = False,
) -> pd.DataFrame:
    """
    Questionnaire-focused processing:
      - By default (derive='auto'): derive vape_status, smoke_status_v1/v2, walk_16_10 when inputs exist.
      - Age filtering is OFF by default (can enable if desired).
    """
    return _entity_fields_core(
        input_data,
        entity="questionnaire",
        derive=derive,
        derive_registry=derive_registry,
        coalesce_rules=coalesce_rules,
        auto_row_filters=auto_row_filters,
        age_col=age_col,
        min_age=min_age,
        max_age=max_age,
        floor_age=floor_age,
        extra_ranges=extra_ranges,
        extra_exprs=extra_exprs,
        keep_na_in_ranges=keep_na_in_ranges,
    )


def clinic_measurements_fields(
    input_data,
    *,
    derive: Union[bool, List[str], Literal["all", "auto"]] = "auto",  # derive BMI/BMI_status if inputs exist
    derive_registry: Optional[Dict[str, DeriveSpec]] = None,
    coalesce_rules: Optional[Dict[str, dict]] = None,
    # clinic: age filter typically not relevant
    auto_row_filters: bool = False,
    age_col: str = "derived.age_at_registration",
    min_age: int = 18,
    max_age: int = 110,
    floor_age: bool = True,
    extra_ranges: Optional[Dict[str, Tuple[float, float]]] = None,
    extra_exprs: Optional[List[str]] = None,
    keep_na_in_ranges: bool = False,
) -> pd.DataFrame:
    """
    Clinic measurements–focused processing:
      - By default (derive='auto'): derive BMI/BMI_status when inputs exist.
    """
    return _entity_fields_core(
        input_data,
        entity="clinic_measurements",
        derive=derive,
        derive_registry=derive_registry,
        coalesce_rules=coalesce_rules,
        auto_row_filters=auto_row_filters,
        age_col=age_col,
        min_age=min_age,
        max_age=max_age,
        floor_age=floor_age,
        extra_ranges=extra_ranges,
        extra_exprs=extra_exprs,
        keep_na_in_ranges=keep_na_in_ranges,
    )


# ------------------------------------------------------------------------------------
# ICD utilities (unchanged)
# ------------------------------------------------------------------------------------

def _normalize_codes(x):
    if x is None or (isinstance(x, float) and pd.isna(x)):
        return []
    if isinstance(x, list):
        return [str(c).strip() for c in x if pd.notna(c)]
    s = str(x).strip()
    if s.startswith('[') and s.endswith(']'):
        return [c.strip() for c in s[1:-1].split(',') if c.strip()]
    return [s] if s else []


def _traits_to_codes(traits_and_codes):
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

    out = {}
    for trait, codes in items:
        code_list = _normalize_codes(codes)
        seen, cleaned = set(), []
        for c in code_list:
            if c and c not in seen:
                cleaned.append(c)
                seen.add(c)
        out[str(trait)] = cleaned
    return out