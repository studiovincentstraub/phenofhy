# ICD Phenotypes

This tutorial shows examples for `icd.py`, which helps you define ICD-based phenotypes and match them to diagnosis columns. Use it to build case definitions, count affected participants, and summarize matched codes for reporting.

## Match ICD traits with `icd.match_icd_traits`

Build a trait map and match against diagnosis columns.

```python
from phenofhy.icd import match_icd_traits

traits_and_codes = {
    "ischemic_heart_disease": ["I20", "I21", "I22", "I23", "I24", "I25"],
    "stroke": ["I60", "I61", "I63", "I64"],
}

trait_to_pids, summary = match_icd_traits(
    raw_df,
    traits_and_codes,
    pid_col="nhse_eng_inpat.pid",
    diag_prefix="nhse_eng_inpat.diag_4_",
)
```

If you only need the list of matched codes per trait:

```python
from phenofhy.icd import get_matched_icd_traits

matched = get_matched_icd_traits(raw_df, traits_and_codes)
```

Notes:

- ICD codes shorter than or equal to length 3 are treated as prefixes by default.
- Use `primary_only=True` in `match_icd_traits` if you want only the primary diagnosis column.
