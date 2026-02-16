# Running a Pipeline

This **Tutorials** section provides guidance for using phenofhy. For full documentation of each module and function, see the [API reference](/api/).


## Running a Pipeline

There are typically two main ways you can use phenofhy in your phenotype workflow:

- A high level workflow that uses `pipeline.py` to go from fields to an analysis-ready table.
- A low level workflow that uses individual modules for more control.

Both paths produce a pandas DataFrame you can use for reporting, further preprocessing, and data analysis.

### High level workflow (pipeline)

Use the `pipeline.py` module when you want the full flow that phenofhy offers in one call.

#### Minimal workflow

```python
from phenofhy.pipeline import run_preprocessing_pipeline
from phenofhy.calculate import prevalence

fields = [
    "participant.pid",
    "participant.birth_year",
    "participant.demog_sex_1_1",
    "participant.demog_sex_2_1",
    "questionnaire.smoke_status_2_1",
    "clinic_measurements.height",
    "clinic_measurements.weight",
]

df = run_preprocessing_pipeline(
    fields=fields,
    cohort_key="FULL_SAMPLE_ID",
    derive_participant=True,
    derive_questionnaire=True,
    derive_questionnaire_mode="auto",
    derive_clinic=True,
)

prev = prevalence(df)
```

#### Next steps

- Use `phenofhy.profile.phenotype_profile()` to generate a phenotype report.
- Add your own custom fields to the `fields` list and rerun.

### Low level workflow (modules)

Use the modules directly when you want full control over each step, including customization and extra analyses with modules like `calculate.py` and `icd.py`.

```python
from phenofhy import load, extract, process
from phenofhy.calculate import prevalence

fields = [
    "participant.pid",
    "participant.birth_year",
    "participant.demog_sex_1_1",
    "participant.demog_sex_2_1",
    "questionnaire.smoke_status_2_1",
    "clinic_measurements.height",
    "clinic_measurements.weight",
]

fieldlist_path = "outputs/intermediate/demo_fields.csv"
load.field_list(fields=fields, output_file=fieldlist_path)

sql_path = "outputs/raw/demo_query.sql"
extract.fields(
    input_file=fieldlist_path,
    output_file=sql_path,
    cohort_key="FULL_SAMPLE_ID",
    sql_only=True,
)

raw_df = extract.sql_to_pandas(sql_path)

step_df = process.participant_fields(raw_df)
step_df = process.questionnaire_fields(step_df, derive="auto")

prev = prevalence(step_df)
```

## More tutorials

- [Profiling a Phenotype](/tutorials/profile)
- [Calculating Summary Stats](/tutorials/calculate)
- [ICD Phenotypes](/tutorials/icd)
- [TRE Utilities](/tutorials/utilities)
