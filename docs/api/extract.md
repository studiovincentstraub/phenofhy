# Extract

Extraction helpers for DNAnexus datasets, including SQL export.

## fields()

```python
phenofhy.extract.fields(*fields, output_file=None, sep=",", sql_only=False,
	cohort_record_id=None, sql_file=None, max_rows=None, sanitize=True,
	replace_spaces=True, lower=True, return_sql=False)
```

Extract raw phenotype values to CSV, or generate SQL if sql_only is True.

**Parameters**

&nbsp;&nbsp;**\*fields**: ***str***<br>
&nbsp;&nbsp;&nbsp;&nbsp;One or more fully-qualified field names (e.g., "participant.birth_year").<br>
&nbsp;&nbsp;**output_file**: ***str | None***<br>
&nbsp;&nbsp;&nbsp;&nbsp;Output CSV path. If None and sql_only is False, returns a DataFrame.<br>
&nbsp;&nbsp;**sep**: ***str***<br>
&nbsp;&nbsp;&nbsp;&nbsp;CSV delimiter for output_file.<br>
&nbsp;&nbsp;**sql_only**: ***bool***<br>
&nbsp;&nbsp;&nbsp;&nbsp;If True, only generate SQL and do not execute.<br>
&nbsp;&nbsp;**cohort_record_id**: ***str | None***<br>
&nbsp;&nbsp;&nbsp;&nbsp;Optional cohort record ID to target a cohort dataset.<br>
&nbsp;&nbsp;**sql_file**: ***str | None***<br>
&nbsp;&nbsp;&nbsp;&nbsp;Optional file path to write SQL when sql_only is True or return_sql is True.<br>
&nbsp;&nbsp;**max_rows**: ***int | None***<br>
&nbsp;&nbsp;&nbsp;&nbsp;Optional row limit for extraction.<br>
&nbsp;&nbsp;**sanitize**: ***bool***<br>
&nbsp;&nbsp;&nbsp;&nbsp;Whether to sanitize column names and values.<br>
&nbsp;&nbsp;**replace_spaces**: ***bool***<br>
&nbsp;&nbsp;&nbsp;&nbsp;Replace spaces in column names when sanitize is True.<br>
&nbsp;&nbsp;**lower**: ***bool***<br>
&nbsp;&nbsp;&nbsp;&nbsp;Lowercase column names when sanitize is True.<br>
&nbsp;&nbsp;**return_sql**: ***bool***<br>
&nbsp;&nbsp;&nbsp;&nbsp;If True, return SQL string alongside extracted data.

**Returns**

&nbsp;&nbsp;**out**: ***pandas.DataFrame | str | tuple***<br>
&nbsp;&nbsp;&nbsp;&nbsp;DataFrame when executed, SQL string when sql_only, or a tuple when return_sql is True.

**Example**
```python
from phenofhy import extract

extract.fields(
    output_file="outputs/raw/phenos.csv",
    fields=["participant.birth_year", "participant.birth_month"],
)
```

## run_extraction()

```python
phenofhy.extract.run_extraction(fields, *, output_file=None, sep=",", sql_only=False,
	cohort_record_id=None, sql_file=None, max_rows=None, sanitize=True,
	replace_spaces=True, lower=True, return_sql=False)
```

Lower-level extraction entry point used by fields().

**Parameters**

&nbsp;&nbsp;**fields**: ***list[str]***<br>
&nbsp;&nbsp;&nbsp;&nbsp;List of fully-qualified field names.<br>
&nbsp;&nbsp;**output_file**: ***str | None***<br>
&nbsp;&nbsp;&nbsp;&nbsp;Output CSV path. If None and sql_only is False, returns a DataFrame.<br>
&nbsp;&nbsp;**sep**: ***str***<br>
&nbsp;&nbsp;&nbsp;&nbsp;CSV delimiter for output_file.<br>
&nbsp;&nbsp;**sql_only**: ***bool***<br>
&nbsp;&nbsp;&nbsp;&nbsp;If True, only generate SQL and do not execute.<br>
&nbsp;&nbsp;**cohort_record_id**: ***str | None***<br>
&nbsp;&nbsp;&nbsp;&nbsp;Optional cohort record ID to target a cohort dataset.<br>
&nbsp;&nbsp;**sql_file**: ***str | None***<br>
&nbsp;&nbsp;&nbsp;&nbsp;Optional file path to write SQL when sql_only is True or return_sql is True.<br>
&nbsp;&nbsp;**max_rows**: ***int | None***<br>
&nbsp;&nbsp;&nbsp;&nbsp;Optional row limit for extraction.<br>
&nbsp;&nbsp;**sanitize**: ***bool***<br>
&nbsp;&nbsp;&nbsp;&nbsp;Whether to sanitize column names and values.<br>
&nbsp;&nbsp;**replace_spaces**: ***bool***<br>
&nbsp;&nbsp;&nbsp;&nbsp;Replace spaces in column names when sanitize is True.<br>
&nbsp;&nbsp;**lower**: ***bool***<br>
&nbsp;&nbsp;&nbsp;&nbsp;Lowercase column names when sanitize is True.<br>
&nbsp;&nbsp;**return_sql**: ***bool***<br>
&nbsp;&nbsp;&nbsp;&nbsp;If True, return SQL string alongside extracted data.

**Returns**

&nbsp;&nbsp;**out**: ***pandas.DataFrame | str | tuple***<br>
&nbsp;&nbsp;&nbsp;&nbsp;DataFrame when executed, SQL string when sql_only, or a tuple when return_sql is True.

## sql_to_pandas()

```python
phenofhy.extract.sql_to_pandas(sql, *, cohort_record_id=None,
	sanitize=True, replace_spaces=True, lower=True)
```

Execute SQL and return a pandas DataFrame.

**Parameters**

&nbsp;&nbsp;**sql**: ***str***<br>
&nbsp;&nbsp;&nbsp;&nbsp;SQL query string.
&nbsp;&nbsp;**cohort_record_id**: ***str | None***<br>
&nbsp;&nbsp;&nbsp;&nbsp;Optional cohort record ID to target a cohort dataset.<br>
&nbsp;&nbsp;**sanitize**: ***bool***<br>
&nbsp;&nbsp;&nbsp;&nbsp;Whether to sanitize column names and values.<br>
&nbsp;&nbsp;**replace_spaces**: ***bool***<br>
&nbsp;&nbsp;&nbsp;&nbsp;Replace spaces in column names when sanitize is True.<br>
&nbsp;&nbsp;**lower**: ***bool***<br>
&nbsp;&nbsp;&nbsp;&nbsp;Lowercase column names when sanitize is True.<br>

**Returns**

&nbsp;&nbsp;**out**: ***pandas.DataFrame***<br>
&nbsp;&nbsp;&nbsp;&nbsp;Query results as a DataFrame.

## sql_to_spark()

```python
phenofhy.extract.sql_to_spark(sql, *, cohort_record_id=None,
	sanitize=True, replace_spaces=True, lower=True)
```

Execute SQL and return a Spark DataFrame.

**Parameters**

&nbsp;&nbsp;**sql**: ***str***<br>
&nbsp;&nbsp;&nbsp;&nbsp;SQL query string.
&nbsp;&nbsp;**cohort_record_id**: ***str | None***<br>
&nbsp;&nbsp;&nbsp;&nbsp;Optional cohort record ID to target a cohort dataset.<br>
&nbsp;&nbsp;**sanitize**: ***bool***<br>
&nbsp;&nbsp;&nbsp;&nbsp;Whether to sanitize column names and values.<br>
&nbsp;&nbsp;**replace_spaces**: ***bool***<br>
&nbsp;&nbsp;&nbsp;&nbsp;Replace spaces in column names when sanitize is True.<br>
&nbsp;&nbsp;**lower**: ***bool***<br>
&nbsp;&nbsp;&nbsp;&nbsp;Lowercase column names when sanitize is True.<br>

**Returns**

&nbsp;&nbsp;**out**: ***pyspark.sql.DataFrame***<br>
&nbsp;&nbsp;&nbsp;&nbsp;Query results as a Spark DataFrame.
