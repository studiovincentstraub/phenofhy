# Pipeline

Utilities for metadata dictionaries and phenotype list processing.

## metadata()

```python
phenofhy.pipeline.metadata()
```

Load metadata dictionary files into DataFrames.

**Returns**

&nbsp;&nbsp;**out**: ***dict[str, pandas.DataFrame]***<br>
&nbsp;&nbsp;&nbsp;&nbsp;Dictionary with `codings`, `data_dictionary`, and `entity_dictionary`.

**Raises**

&nbsp;&nbsp;**RuntimeError**: ***Exception***<br>
&nbsp;&nbsp;&nbsp;&nbsp;If a required file is missing or fails to load.<br>
&nbsp;&nbsp;**FileNotFoundError**: ***Exception***<br>
&nbsp;&nbsp;&nbsp;&nbsp;If an expected file path does not exist.

## run_preprocessing_pipeline()

```python
phenofhy.pipeline.run_preprocessing_pipeline(fields, cohort_key="FULL_SAMPLE_ID",
	derive_participant=True, derive_questionnaire=True,
	derive_questionnaire_mode="auto", derive_clinic=True)
```

Extract and preprocess data with automatic field derivation.

This high-level function orchestrates the full phenotype preprocessing pipeline:
extracting fields from DNAnexus, then applying entity-specific derivations.

**Parameters**

&nbsp;&nbsp;**fields**: ***list[str] | dict[str, str]***<br>
&nbsp;&nbsp;&nbsp;&nbsp;List of "entity.field" strings or dict of entity->field mappings.<br>
&nbsp;&nbsp;**cohort_key**: ***str***<br>
&nbsp;&nbsp;&nbsp;&nbsp;Config key for the cohort dataset ID. Default: `"FULL_SAMPLE_ID"`.<br>
&nbsp;&nbsp;**derive_participant**: ***bool***<br>
&nbsp;&nbsp;&nbsp;&nbsp;Whether to derive participant fields (age groups, etc.). Default: `True`.<br>
&nbsp;&nbsp;**derive_questionnaire**: ***bool***<br>
&nbsp;&nbsp;&nbsp;&nbsp;Whether to derive questionnaire fields. Default: `True`.<br>
&nbsp;&nbsp;**derive_questionnaire_mode**: ***"all" | "auto"***<br>
&nbsp;&nbsp;&nbsp;&nbsp;Derivation mode for questionnaire. Default: `"auto"`.<br>
&nbsp;&nbsp;**derive_clinic**: ***bool***<br>
&nbsp;&nbsp;&nbsp;&nbsp;Whether to derive clinic measurement fields (BMI, etc.). Default: `True`.<br>

**Returns**

&nbsp;&nbsp;**out**: ***pandas.DataFrame***<br>
&nbsp;&nbsp;&nbsp;&nbsp;Processed DataFrame ready for analysis.

**Raises**

&nbsp;&nbsp;**RuntimeError**: ***Exception***<br>
&nbsp;&nbsp;&nbsp;&nbsp;If extraction or processing fails.<br>
&nbsp;&nbsp;**ValueError**: ***Exception***<br>
&nbsp;&nbsp;&nbsp;&nbsp;If fields format is unsupported.

**Example**
```python
from phenofhy import pipeline

# Extract and preprocess specific fields
df = pipeline.run_preprocessing_pipeline(
    fields=["participant.birth_year", "participant.birth_month"],
    derive_participant=True,
    derive_questionnaire=True,
)
```

## field_list()

```python
phenofhy.pipeline.field_list(fields=None, output_file=None, fields_list_name=None,
	input_file=None, input_file_name=None)
```

Build a merged metadata table from a phenotype list.

**Parameters**

&nbsp;&nbsp;**fields**: ***list[str] | dict[str, str] | str | None***<br>
&nbsp;&nbsp;&nbsp;&nbsp;List of "entity.field", dict of entity->field, or path/ID.<br>
&nbsp;&nbsp;**output_file**: ***str | None***<br>
&nbsp;&nbsp;&nbsp;&nbsp;If provided, write CSV to this path and return None.<br>
&nbsp;&nbsp;**fields_list_name**: ***str | None***<br>
&nbsp;&nbsp;&nbsp;&nbsp;Optional filename when downloading a direct file ID.<br>
&nbsp;&nbsp;**input_file**: ***any***<br>
&nbsp;&nbsp;&nbsp;&nbsp;Backward-compatible alias for `fields`.<br>
&nbsp;&nbsp;**input_file_name**: ***str | None***<br>
&nbsp;&nbsp;&nbsp;&nbsp;Backward-compatible alias for `fields_list_name`.<br>

**Returns**

&nbsp;&nbsp;**out**: ***pandas.DataFrame | None***<br>
&nbsp;&nbsp;&nbsp;&nbsp;Merged metadata table, or None if output_file is provided.

**Example**
```python
from phenofhy import pipeline

meta = pipeline.metadata()
fields = pipeline.field_list(fields=["participant.birth_year", "participant.birth_month"])
```
