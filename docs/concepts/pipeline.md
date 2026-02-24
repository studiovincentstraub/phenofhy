# Processing pipeline

A typical Phenofhy workflow follows these steps.

For local development, the same downstream processing and summary steps can be tested on simulated data generated with `phenofhy.simulate.simulate_phenotype_df()`.

## 1) Choose phenotypes

Define a list of fields as `entity.field` strings or via a phenotype list file (which can be referenced in `config.json` using its file ID).

## 2) Extract raw data

Use `extract.fields()` to pull data (or generate SQL) from DNAnexus datasets.

For local tests, replace this extraction step with a simulated dataframe.

## 3) Process by entity

Use entity-specific helpers to clean and derive fields:

- `process.participant_fields()`
- `process.questionnaire_fields()`
- `process.clinic_measurements_fields()`

## 4) Summarize

Use `calculate.summary()` or `calculate.prevalence()` to create tables for QC or reporting.

## 5) Profile

Use `profile.phenotype_profile()` to produce a lightweight pre-GWAS phenotype report.


## 6) Upload

Use `utils.upload_files()` to upload any reports, figures, or tables to your TRE project.

For local testing, this upload step is optional.
