# Data model

Phenofhy uses a simple naming convention for fields:

* Raw fields are `entity.field` (for example, `participant.birth_year`).
* Derived fields live under `derived.*` (for example, `derived.age_at_registration`).

## Metadata dictionaries

Phenofhy uses metadata dictionaries to describe fields, coding domains, and entities in the Our Future Health TRE.

For analyses running in the TRE, metadata exported from DNAnexus can include:

* `*.codings.csv` for code-to-label mappings
* `*.data_dictionary.csv` for field descriptions and metadata
* `*.entity_dictionary.csv` for entity metadata

These files are retrieved and processed by `pipeline.metadata()` for use with project data.

### Metadata used for simulation

Phenofhy also includes the metadata required by `phenofhy.simulate` within the installed package. Users do not need to download these files separately.

For example:

```
from phenofhy import simulate

df = simulate.simulate_phenotype_df(
    sample=500,
    seed=42,
)
```

The simulation utilities use the packaged data dictionary and coding information to generate synthetic OFH-like phenotype data.

## Coding names

Coding names link fields in the data dictionary to their permitted coded values.

For example:

* `participant.demog_sex_1_1` maps to coding name `DEMOG_SEX_1_1`.
* The corresponding coding domain defines the permitted values for that field.

Phenofhy handles this mapping internally when generating simulated phenotype data.
