# Quickstart

Phenofhy helps you extract, process, and summarize OFH phenotype data inside the TRE, and also supports local testing on simulated data.

## Prerequisites

- You are running inside the OFH TRE with the `dx` toolkit available.
- `config.json` is available in `/mnt/project/helpers` (Phenofhy uses it to resolve files).
- If you have not uploaded Phenofhy to the TRE yet, see the [Installation](/getting-started/installation) page.

For local testing, you can skip TRE prerequisites and start with simulated data (see [Simulating data locally](/tutorials/simulating-data-locally)).

## Download Phenofhy into your JupyterLab session

Before running the workflow, download the Phenofhy package into your JupyterLab
Spark instance using `dx`. Replace `project_name` with your TRE project name:

```python
!dx download "project_name:/applets/phenofhy/" -r
```

This assumes Phenofhy was uploaded under the `applets` folder and contains the
beta source code (i.e., Python modules).

## Minimal workflow


```python
from phenofhy import extract, process, calculate, profile, utils

# 1) Extract a small set of fields
extract.fields(
    output_file="outputs/raw/phenos.csv",
    fields=[
        "participant.registration_year",
        "participant.registration_month",
        "participant.birth_year",
        "participant.birth_month",
        "participant.demog_sex_2_1",
        "questionnaire.smoke_status_2_1",
    ],
)

# 2) Process participant data (derives age, sex, age_group)
df = process.participant_fields("outputs/raw/phenos.csv")

# 3) Summaries
summary = calculate.summary(
    df,
    traits=["derived.age_at_registration", "derived.sex"],
    stratify="derived.sex",
)

# 4) Profile report
report = profile.phenotype_profile(
    df,
    phenotype="derived.age_at_registration",
    output="outputs/reports/age_profile.pdf",
)

# 5) Upload your results
report = utils.upload_files(
   files="outputs/reports/age_profile.pdf",
   dx_target="results"
)
```

## Next steps

- Use `calculate.prevalence()` for prevalence tables.
- Use `icd.match_icd_traits()` to match ICD codes to traits.
- Use `process.questionnaire_fields()` or `process.clinic_measurements_fields()` to work by entity.

## Local quick test (no TRE access required)

```python
from phenofhy.simulate import simulate_phenotype_df
from phenofhy import process, calculate

df = simulate_phenotype_df(sample=1000, seed=42)
df = process.participant_fields(df)

summary = calculate.summary(
    df,
    traits=["derived.age_at_registration", "derived.sex"],
    stratify="derived.sex",
)
```
