# Calculating Summary Stats

This tutorial shows examples for `calculate.py`, which provides derived summaries and prevalence metrics for phenotypes. Use it to compute prevalence, generate summary tables for numeric and categorical traits, and quantify medication usage patterns.

## Prevalence with `calculate.prevalence`

Compute counts and prevalence for categorical traits.

```python
from phenofhy.calculate import prevalence

prev = prevalence(
    df,
    traits=[
        "derived.sex",
        "questionnaire.smoke_status_2_1",
        "questionnaire.alcohol_curr_1_1",
    ],
    denominator="nonmissing",
)
```

Add multiple denominators in a wide output:

```python
prev = prevalence(
    df,
    traits=["derived.sex", "questionnaire.smoke_status_2_1"],
    denominators=["all", "nonmissing"],
    wide_output=True,
)
```

## Summary tables with `calculate.summary`

Use `summary()` to compute numeric and categorical summaries, optionally stratified.

```python
from phenofhy.calculate import summary

result = summary(
    df,
    stratify="derived.sex",
    categorical_traits=["questionnaire.smoke_status_2_1"],
    round_decimals=2,
)

numeric_df = result["numeric"]
categorical_df = result["categorical"]
```

## Medication prevalence

For medication phenotypes, use `calculate.medication_prevalence`.

```python
from phenofhy.calculate import medication_prevalence

per_med, grouped = medication_prevalence(
    df,
    codings=codings_df,
    medication_phenotypes=medication_phenotypes,
    denominator="all",
)
```
