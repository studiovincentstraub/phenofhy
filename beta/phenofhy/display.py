# phenofhy/display.py
from __future__ import annotations
from typing import Any, Dict, Iterable, List, Optional
import re
import numpy as np
import pandas as pd

# NEW: soft import defaults from _rules
try:
    from ._rules import DEFAULT_CATEGORY_ORDER, DEFAULT_EXCLUDED_CATEGORIES
except Exception:
    # allow use outside package layout
    try:
        from _rules import DEFAULT_CATEGORY_ORDER, DEFAULT_EXCLUDED_CATEGORIES
    except Exception:
        DEFAULT_CATEGORY_ORDER = {}
        DEFAULT_EXCLUDED_CATEGORIES = []


def _normalize_label(s: Any) -> str:
    """Lowercase, trim, normalize dashes/spaces, basic range word→digit."""
    if s is None or (isinstance(s, float) and np.isnan(s)):
        return ""
    t = str(s).strip().lower()
    # normalize dashes to ASCII hyphen
    t = t.replace("–", "-").replace("—", "-").replace("−", "-")
    # basic range normalization (e.g., 'one or two' -> '1-2')
    words2num = {
        "one": "1", "two": "2", "three": "3", "four": "4", "five": "5",
        "six": "6", "seven": "7", "eight": "8", "nine": "9"
    }
    t = re.sub(
        r"\b(one|two|three|four|five|six|seven|eight|nine)\s+or\s+(one|two|three|four|five|six|seven|eight|nine)\b",
        lambda m: f"{words2num[m.group(1)]}-{words2num[m.group(2)]}", t
    )
    t = re.sub(
        r"\b(one|two|three|four|five|six|seven|eight|nine)\s+to\s+(one|two|three|four|five|six|seven|eight|nine)\b",
        lambda m: f"{words2num[m.group(1)]}-{words2num[m.group(2)]}", t
    )
    # collapse whitespace
    t = re.sub(r"\s+", " ", t)
    return t


def group_filter(
    categorical_df: pd.DataFrame,
    *,
    group_filter: Optional[Dict[str, Any]] = None,
    trait_labels: Optional[Dict[str, str]] = None,
    category_order: Optional[Dict[str, List[str]]] = None,   # will default to _rules
    exclude_categories: Optional[Iterable[str]] = None,       # will default to _rules
    round_decimals: int = 2,
    show_percent_sign: bool = False,
    na_label: str = "NA",
    spacer_between_traits: bool = False,
    keep_unlisted_at_end: bool = True,
    drop_label: bool = False,
) -> pd.DataFrame:
    """
    Build a 4-column Excel-ready table from the categorical output of grouped_trait_statistics().

    Returns a DataFrame with columns: ['label', 'N', '%', 'SD'] where:
      - A trait header row has 'label' = trait title, and N/%/SD empty.
      - Category rows follow in the specified order with counts/percent/SD.

    Parameters
    ----------
    categorical_df : DataFrame
        res["categorical"] from grouped_trait_statistics()
    group_filter : dict, optional
        Column->value filter to select one group (e.g., {"derived.age_group": "60+"}).
    trait_labels : dict, optional
        Mapping from trait codes to pretty names.
    category_order : dict, optional
        Desired category order for each trait (list of strings).
    exclude_categories : iterable, optional
        Category labels to hide (e.g., ["Do not know","Unknown","Prefer not to answer"]).
    round_decimals : int
        Rounding for % and SD.
    show_percent_sign : bool
        If True, % strings include a % symbol.
    na_label : str
        Display label for NaN categories.
    spacer_between_traits : bool
        If True, inserts a blank row after each trait block.
    keep_unlisted_at_end : bool
        If True, categories not specified in `category_order[trait]` are appended after ordered ones.
    drop_label : bool
        If True, returns only ['N','%','SD'] (no 'label' column).
    """
    # empty-fast path
    if categorical_df is None or categorical_df.empty:
        cols = ["N", "%", "SD"] if drop_label else ["label", "N", "%", "SD"]
        return pd.DataFrame(columns=cols)

    d = categorical_df.copy()
    
    # Defaults from _rules if not provided
    if category_order is None:
        category_order = DEFAULT_CATEGORY_ORDER
    if exclude_categories is None:
        exclude_categories = DEFAULT_EXCLUDED_CATEGORIES

    # optional: filter to a single group (e.g., one age band)
    if group_filter:
        for k, v in group_filter.items():
            if k not in d.columns:
                raise KeyError(f"Group column '{k}' not found in categorical_df.")
            d = d.loc[d[k] == v]
    if d.empty:
        cols = ["N", "%", "SD"] if drop_label else ["label", "N", "%", "SD"]
        return pd.DataFrame(columns=cols)

    # required columns
    req = {"trait", "category", "count", "percentage", "std"}
    missing = req - set(d.columns)
    if missing:
        raise ValueError(f"categorical_df missing columns: {missing}")

    # display fields
    d["__trait_disp__"] = d["trait"].map(lambda t: trait_labels.get(t, t) if trait_labels else t)
    d["__cat_disp__"]   = d["category"].map(lambda x: na_label if pd.isna(x) else str(x))
    d["__cat_norm__"]   = d["__cat_disp__"].map(_normalize_label)

    # exclude unwanted categories
    if exclude_categories:
        excl_norm = {_normalize_label(x) for x in exclude_categories}
        d = d.loc[~d["__cat_norm__"].isin(excl_norm)]

    # numbers
    d["__pct__"] = d["percentage"].round(round_decimals)
    d["__sd__"]  = d["std"].round(round_decimals)
    if show_percent_sign:
        d["__pct__"] = d["__pct__"].map(lambda x: f"{x:.{round_decimals}f}%" if pd.notnull(x) else "")

    rows: List[Dict[str, Any]] = []
    # keep trait appearance order from input
    for trait, sub in d.groupby("trait", sort=False):
        # trait header row
        rows.append({"label": sub["__trait_disp__"].iloc[0], "N": "", "%": "", "SD": ""})

        # order categories if specified
        if category_order and trait in category_order:
            desired = [str(c) for c in category_order[trait]]
            desired_norm = [_normalize_label(c) for c in desired]
            rank_map = {lab: i for i, lab in enumerate(desired_norm)}
            sub = sub.copy()
            sub["__rank__"] = sub["__cat_norm__"].map(lambda x: rank_map.get(x, np.inf))
            sub = sub.sort_values(["__rank__"])
            if not keep_unlisted_at_end:
                sub = sub.loc[sub["__cat_norm__"].isin(desired_norm)]

        # push rows
        for _, r in sub.iterrows():
            rows.append({
                "label": r["__cat_disp__"],
                "N": int(r["count"]) if pd.notnull(r["count"]) else "",
                "%": r["__pct__"],
                "SD": r["__sd__"],
            })

        if spacer_between_traits:
            rows.append({"label": "", "N": "", "%": "", "SD": ""})

    panel = pd.DataFrame(rows, columns=["label", "N", "%", "SD"])
    if drop_label:
        return panel.drop(columns=["label"])
    return panel
