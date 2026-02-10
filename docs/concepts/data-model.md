# Data model

Phenofhy uses a simple naming convention for fields:

- Raw fields are `entity.field` (for example, `participant.birth_year`).
- Derived fields live under `derived.*` (for example, `derived.age_at_registration`).

## Metadata dictionaries

Phenofhy uses three dictionary files exported from DNAnexus:

- `*.codings.csv` for code-to-label mappings
- `*.data_dictionary.csv` for descriptions
- `*.entity_dictionary.csv` for entity metadata

These files are loaded into `./metadata` by `pipeline.metadata()`.

## Coding names

Coding names are matched using the field suffix, for example:

- `questionnaire.smoke_status_2_1` maps to coding name `SMOKE_STATUS_2_1`.
