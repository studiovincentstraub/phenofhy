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
