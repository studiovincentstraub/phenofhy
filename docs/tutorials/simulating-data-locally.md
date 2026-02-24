# Simulating data locally

This tutorial shows how to generate an OFH-like sample dataframe locally using `phenofhy.simulate.simulate_phenotype_df()`.

Use this when you want to test preprocessing and analysis code outside the TRE without querying DNAnexus.

## Basic example

```python
from phenofhy.simulate import simulate_phenotype_df

# Uses default 1000 rows and default 15 OFH-like columns
df = simulate_phenotype_df(seed=42)

df.head()
```

## Choose sample size and fields

Pass one `entity.field` string or a list of fields.

```python
from phenofhy.simulate import simulate_phenotype_df

df = simulate_phenotype_df(
    sample=500,
    fields=[
        "participant.pid",
        "participant.birth_year",
        "participant.demog_sex_2_1",
        "participant.demog_ethnicity_1_1",
        "clinic_measurements.height",
        "clinic_measurements.weight",
    ],
    seed=7,
)
```

## Include non-response codes

By default, non-response/sentinel codes are excluded.
Set `include_nonresponse=True` if you want them included.

```python
from phenofhy.simulate import simulate_phenotype_df

df = simulate_phenotype_df(
    sample=1000,
    fields="participant.demog_ethnicity_1_1",
    include_nonresponse=True,
    seed=123,
)
```

## Control missingness

Use a global missing rate or a per-field mapping.

```python
from phenofhy.simulate import simulate_phenotype_df

df = simulate_phenotype_df(
    sample=1000,
    fields=[
        "participant.pid",
        "clinic_measurements.height",
        "clinic_measurements.weight",
    ],
    missing_rate={
        "participant.pid": 0.0,
        "clinic_measurements.height": 0.1,
        "clinic_measurements.weight": 0.15,
    },
    seed=2026,
)
```

## Notes

- Valid fields and base types come from `data_dictionary.csv`.
- Categorical code domains come from `codings.csv`.
- For numeric fields not present in codings, conservative fallback ranges are used.
- This simulator is intended for local testing and development, not for inference on real OFH participants.
