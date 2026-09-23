# Processing Phenotypes

Phenofhy's processing functions clean extracted phenotype data, create derived
variables, coalesce related fields, and optionally apply row-level filters.
They accept either a pandas DataFrame or a path to a CSV file and return a
pandas DataFrame.

## Choose the processing level

Use the entity-specific functions when you want a clear, staged workflow:

- `process.participant_fields()` handles participant variables and applies
  participant-oriented derivations.
- `process.questionnaire_fields()` handles questionnaire variables and their
  derived measures.
- `process.clinic_measurements_fields()` handles clinic measurements and
  measurement-derived variables such as BMI.

A typical staged workflow is:

```python
from phenofhy import process

participant_df = process.participant_fields(raw_df)
questionnaire_df = process.questionnaire_fields(participant_df)
processed_df = process.clinic_measurements_fields(questionnaire_df)
```

Each function can also read a CSV directly:

```python
processed_df = process.participant_fields(
    "outputs/raw/phenotypes.csv",
    derive="auto",
)
```

## Derivation modes

The `derive` argument controls which registered derivations run:

- `"auto"` selects derivations whose input columns are available. This is the
  default and is usually the best choice for extracted data.
- `"all"` or `True` requests every derivation applicable to the entity.
- `False` or `None` disables derivation.
- A list of names runs only the selected derivations.

For example:

```python
age_df = process.participant_fields(
    raw_df,
    derive=[
        "registration_date",
        "age_at_registration",
        "age_group",
    ],
)
```

When a requested derivation does not have the required input columns, automatic
mode skips it. This allows the same processing call to work with different
field selections.

## Available derivations

The following derivations are registered in Phenofhy. With `derive="auto"`,
each one runs only when its required source columns are available and the
derivation is selected for the relevant processing function.

| Derivation | Main inputs or purpose |
| --- | --- |
| `registration_date` | Registration year and month; creates a datetime registration field. |
| `age_at_registration` | Registration date or registration/birth year and month; calculates age in years. |
| `age_group` | `derived.age_at_registration`; creates age bands. |
| `sex` | Versioned participant sex fields; creates a unified derived sex variable. |
| `bmi` | Clinic height and weight; calculates body mass index. |
| `bmi_status` | `derived.bmi`; creates BMI categories. |
| `vape_status` | Questionnaire vaping and tobacco-type fields; creates a vaping status. |
| `smoke_status_v1` | Version 1 smoking questionnaire fields; creates a normalized smoking status. |
| `tobacco_ever` | Questionnaire tobacco-type field; creates a tobacco-ever indicator. |
| `tobacco_reg` | Questionnaire regular-smoking field; creates a tobacco-regular indicator. |
| `smoke_status_v2` | Version 2 smoking questionnaire fields; creates a normalized smoking status. |
| `walk_16_10` | Questionnaire walking-days field; creates a walking threshold indicator. |
| `medicat_expand` | Multi-select medication field; expands medication codes into derived indicators. |
| `any_hospital_contact` | Emergency, inpatient, and outpatient event fields plus registration date; creates an any-contact indicator. |
| `total_hospital_contacts` | Hospital event fields plus registration date; counts contacts. |
| `ae_visits` | Emergency department event fields plus registration date; counts or derives emergency visits. |
| `apc_visits` | Inpatient event fields plus registration date; counts or derives inpatient visits. |
| `op_visits` | Outpatient event fields plus registration date; counts or derives outpatient visits. |

The exact output columns produced by a derivation are defined by its function
and may include several columns for expansions such as `medicat_expand`. The
hospital-contact derivations require the corresponding event entities, so they
will normally be skipped for extracts that contain only participant,
questionnaire, or clinic fields.

To see the registered names programmatically:

```python
from phenofhy._derive_funcs import DERIVE_REGISTRY

print(list(DERIVE_REGISTRY))
```

To run a selected subset explicitly, pass the registry names as a list:

```python
clinic_df = process.clinic_measurements_fields(
    raw_df,
    derive=["bmi", "bmi_status"],
)
```

## Participant processing

`participant_fields()` is the main entry point for participant-level data. It
can derive registration dates, age at registration, age groups, sex labels,
and other participant variables when their source fields are present.

Participant processing enables age filtering by default. The default range is
18 inclusive up to, but not including, 110. Disable it or change it when the
analysis requires a different population:

```python
participant_df = process.participant_fields(
    raw_df,
    auto_row_filters=False,
)

adult_df = process.participant_fields(
    raw_df,
    min_age=18,
    max_age=100,
    auto_row_filters=True,
)
```

Set `floor_age=True` to floor continuous ages before downstream derivations.
You can also provide custom age-group bins and labels:

```python
participant_df = process.participant_fields(
    raw_df,
    age_group_bins=[18, 30, 45, 60, 75, 110],
    age_group_labels=["18-29", "30-44", "45-59", "60-74", "75+"],
)
```

## Questionnaire and clinic processing

Questionnaire processing focuses on questionnaire-derived variables and does not
apply age filters by default:

```python
questionnaire_df = process.questionnaire_fields(
    participant_df,
    derive="auto",
)
```

Clinic processing focuses on measurements and derived measurements:

```python
clinic_df = process.clinic_measurements_fields(
    questionnaire_df,
    derive="auto",
    extra_ranges={
        "clinic_measurements.weight": (30, 250),
    },
    keep_na_in_ranges=True,
)
```

## Cleaning and filtering

All entity-processing functions apply known error filters before derivations.
These remove selected sentinel values and implausible clinic measurements when
the relevant columns are present. Missing columns are skipped, and missing
values are preserved by the plausibility filters.

You can add numeric ranges or pandas expressions with `extra_ranges` and
`extra_exprs`:

```python
processed_df = process.participant_fields(
    raw_df,
    extra_ranges={"participant.birth_year": (1900, 2020)},
    extra_exprs=["participant.demog_sex_2_1 >= 0"],
    keep_na_in_ranges=True,
)
```

For standalone filtering, see the [Rules and data cleaning](/api/rules) and
[Filter API](/api/filter) pages.

## Coalescing rules

Processing applies default coalescing rules after derivations. These combine
related variables, such as versioned smoking fields, into unified outputs.

Pass custom rules with `coalesce_rules`:

```python
from phenofhy import _rules

rules = _rules.build_rules(
    overrides={
        "derived.smoke_reg_first_age": _rules.rule_num(
            [
                "questionnaire.smoke_reg_first_age_1_1",
                "questionnaire.smoke_reg_first_age_2_1",
            ],
            informative=lambda value: value is not None and 5 <= float(value) <= 100,
            astype="Int64",
        ),
    }
)

processed_df = process.questionnaire_fields(
    raw_df,
    coalesce_rules=rules,
)
```

See [Rules and data cleaning](/api/rules) for categorical rules, numeric
rules, overrides, and extensions.

## Processing simulated data locally

Processing functions work with simulated data, so you can test downstream
logic without access to DNAnexus:

```python
from phenofhy import process, simulate

simulated_df = simulate.simulate_phenotype_df(sample=1000, seed=42)
simulated_df = process.participant_fields(simulated_df)
simulated_df = process.questionnaire_fields(simulated_df)
simulated_df = process.clinic_measurements_fields(simulated_df)
```

The result is an ordinary pandas DataFrame that can be passed to
`calculate`, `profile`, or your own analysis code.
