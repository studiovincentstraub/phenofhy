# Process

Entity-aware processing, derivations, and data cleaning.

## participant_fields()

```python
phenofhy.process.participant_fields(input_data, *, derive="auto", derive_registry=None,
	coalesce_rules=None, auto_row_filters=True, age_col="derived.age_at_registration",
	min_age=18, max_age=110, floor_age=True, age_group_bins=None,
	age_group_labels=None, extra_ranges=None, extra_exprs=None,
	keep_na_in_ranges=False)
```

Process participant entity fields with optional derives and filters.

**Parameters**

&nbsp;&nbsp;**input_data**: ***str | pandas.DataFrame***<br>
&nbsp;&nbsp;&nbsp;&nbsp;File path or DataFrame input.<br>
&nbsp;&nbsp;**derive**: ***bool | list[str] | Literal["all", "auto"]***<br>
&nbsp;&nbsp;&nbsp;&nbsp;Derivation selection policy.<br>
&nbsp;&nbsp;**derive_registry**: ***dict | None***<br>
&nbsp;&nbsp;&nbsp;&nbsp;Optional custom derive registry.<br>
&nbsp;&nbsp;**coalesce_rules**: ***dict | None***<br>
&nbsp;&nbsp;&nbsp;&nbsp;Optional coalesce rule overrides.<br>
&nbsp;&nbsp;**auto_row_filters**: ***bool***<br>
&nbsp;&nbsp;&nbsp;&nbsp;Whether to apply age-based filtering.<br>
&nbsp;&nbsp;**age_col**: ***str***<br>
&nbsp;&nbsp;&nbsp;&nbsp;Age column for filtering.<br>
&nbsp;&nbsp;**min_age**: ***int***<br>
&nbsp;&nbsp;&nbsp;&nbsp;Minimum age inclusive.<br>
&nbsp;&nbsp;**max_age**: ***int***<br>
&nbsp;&nbsp;&nbsp;&nbsp;Maximum age inclusive.<br>
&nbsp;&nbsp;**floor_age**: ***bool***<br>
&nbsp;&nbsp;&nbsp;&nbsp;Whether to floor age values before deriving.<br>
&nbsp;&nbsp;**age_group_bins**: ***list[float] | None***<br>
&nbsp;&nbsp;&nbsp;&nbsp;Optional custom age bins.<br>
&nbsp;&nbsp;**age_group_labels**: ***list[str] | None***<br>
&nbsp;&nbsp;&nbsp;&nbsp;Optional custom age labels.<br>
&nbsp;&nbsp;**extra_ranges**: ***dict | None***<br>
&nbsp;&nbsp;&nbsp;&nbsp;Optional extra numeric ranges for filtering.<br>
&nbsp;&nbsp;**extra_exprs**: ***list[str] | None***<br>
&nbsp;&nbsp;&nbsp;&nbsp;Optional query expressions for filtering.<br>
&nbsp;&nbsp;**keep_na_in_ranges**: ***bool***<br>
&nbsp;&nbsp;&nbsp;&nbsp;Whether to keep NA rows during range filters.<br>

**Returns**

&nbsp;&nbsp;**out**: ***pandas.DataFrame***<br>
&nbsp;&nbsp;&nbsp;&nbsp;Processed DataFrame.

## questionnaire_fields()

```python
phenofhy.process.questionnaire_fields(input_data, *, derive="auto",
	derive_registry=None, coalesce_rules=None, auto_row_filters=False,
	age_col="derived.age_at_registration", min_age=18, max_age=110,
	floor_age=True, extra_ranges=None, extra_exprs=None, keep_na_in_ranges=False)
```

Process questionnaire entity fields with optional derives.

**Parameters**

&nbsp;&nbsp;**input_data**: ***str | pandas.DataFrame***<br>
&nbsp;&nbsp;&nbsp;&nbsp;File path or DataFrame input.<br>
&nbsp;&nbsp;**derive**: ***bool | list[str] | Literal["all", "auto"]***<br>
&nbsp;&nbsp;&nbsp;&nbsp;Derivation selection policy.<br>
&nbsp;&nbsp;**derive_registry**: ***dict | None***<br>
&nbsp;&nbsp;&nbsp;&nbsp;Optional custom derive registry.<br>
&nbsp;&nbsp;**coalesce_rules**: ***dict | None***<br>
&nbsp;&nbsp;&nbsp;&nbsp;Optional coalesce rule overrides.<br>
&nbsp;&nbsp;**auto_row_filters**: ***bool***<br>
&nbsp;&nbsp;&nbsp;&nbsp;Whether to apply age-based filtering.<br>
&nbsp;&nbsp;**age_col**: ***str***<br>
&nbsp;&nbsp;&nbsp;&nbsp;Age column for filtering.<br>
&nbsp;&nbsp;**min_age**: ***int***<br>
&nbsp;&nbsp;&nbsp;&nbsp;Minimum age inclusive.<br>
&nbsp;&nbsp;**max_age**: ***int***<br>
&nbsp;&nbsp;&nbsp;&nbsp;Maximum age inclusive.<br>
&nbsp;&nbsp;**floor_age**: ***bool***<br>
&nbsp;&nbsp;&nbsp;&nbsp;Whether to floor age values before deriving.<br>
&nbsp;&nbsp;**extra_ranges**: ***dict | None***<br>
&nbsp;&nbsp;&nbsp;&nbsp;Optional extra numeric ranges for filtering.<br>
&nbsp;&nbsp;**extra_exprs**: ***list[str] | None***<br>
&nbsp;&nbsp;&nbsp;&nbsp;Optional query expressions for filtering.<br>
&nbsp;&nbsp;**keep_na_in_ranges**: ***bool***<br>
&nbsp;&nbsp;&nbsp;&nbsp;Whether to keep NA rows during range filters.<br>

**Returns**

&nbsp;&nbsp;**out**: ***pandas.DataFrame***<br>
&nbsp;&nbsp;&nbsp;&nbsp;Processed DataFrame.

## clinic_measurements_fields()

```python
phenofhy.process.clinic_measurements_fields(input_data, *, derive="auto",
	derive_registry=None, coalesce_rules=None, auto_row_filters=False,
	age_col="derived.age_at_registration", min_age=18, max_age=110,
	floor_age=True, extra_ranges=None, extra_exprs=None, keep_na_in_ranges=False)
```

Process clinic measurements fields with optional BMI derives.

**Parameters**

&nbsp;&nbsp;**input_data**: ***str | pandas.DataFrame***<br>
&nbsp;&nbsp;&nbsp;&nbsp;File path or DataFrame input.<br>
&nbsp;&nbsp;**derive**: ***bool | list[str] | Literal["all", "auto"]***<br>
&nbsp;&nbsp;&nbsp;&nbsp;Derivation selection policy.<br>
&nbsp;&nbsp;**derive_registry**: ***dict | None***<br>
&nbsp;&nbsp;&nbsp;&nbsp;Optional custom derive registry.<br>
&nbsp;&nbsp;**coalesce_rules**: ***dict | None***<br>
&nbsp;&nbsp;&nbsp;&nbsp;Optional coalesce rule overrides.<br>
&nbsp;&nbsp;**auto_row_filters**: ***bool***<br>
&nbsp;&nbsp;&nbsp;&nbsp;Whether to apply age-based filtering.<br>
&nbsp;&nbsp;**age_col**: ***str***<br>
&nbsp;&nbsp;&nbsp;&nbsp;Age column for filtering.<br>
&nbsp;&nbsp;**min_age**: ***int***<br>
&nbsp;&nbsp;&nbsp;&nbsp;Minimum age inclusive.<br>
&nbsp;&nbsp;**max_age**: ***int***<br>
&nbsp;&nbsp;&nbsp;&nbsp;Maximum age inclusive.<br>
&nbsp;&nbsp;**floor_age**: ***bool***<br>
&nbsp;&nbsp;&nbsp;&nbsp;Whether to floor age values before deriving.<br>
&nbsp;&nbsp;**extra_ranges**: ***dict | None***<br>
&nbsp;&nbsp;&nbsp;&nbsp;Optional extra numeric ranges for filtering.<br>
&nbsp;&nbsp;**extra_exprs**: ***list[str] | None***<br>
&nbsp;&nbsp;&nbsp;&nbsp;Optional query expressions for filtering.<br>
&nbsp;&nbsp;**keep_na_in_ranges**: ***bool***<br>
&nbsp;&nbsp;&nbsp;&nbsp;Whether to keep NA rows during range filters.<br>

**Returns**

&nbsp;&nbsp;**out**: ***pandas.DataFrame***<br>
&nbsp;&nbsp;&nbsp;&nbsp;Processed DataFrame.

## get_dummies()

```python
phenofhy.process.get_dummies(df, codings_glob="./metadata/*.codings.csv",
	coding_name="MEDICAT_1_M", col="questionnaire.medicat_1_m",
	prefix="derived.medicates_", exclude_codes=(-7, -1, -3), user_map=None,
	inplace=True)
```

Expand a coded multi-select column into dummy variables.

**Parameters**

&nbsp;&nbsp;**df**: ***pandas.DataFrame***<br>
&nbsp;&nbsp;&nbsp;&nbsp;Input dataframe to modify.<br>
&nbsp;&nbsp;**codings_glob**: ***str***<br>
&nbsp;&nbsp;&nbsp;&nbsp;Glob path to codings CSV files.<br>
&nbsp;&nbsp;**coding_name**: ***str***<br>
&nbsp;&nbsp;&nbsp;&nbsp;Coding name to match in codings CSVs.<br>
&nbsp;&nbsp;**col**: ***str***<br>
&nbsp;&nbsp;&nbsp;&nbsp;Column containing coded values.<br>
&nbsp;&nbsp;**prefix**: ***str***<br>
&nbsp;&nbsp;&nbsp;&nbsp;Prefix for generated dummy columns.<br>
&nbsp;&nbsp;**exclude_codes**: ***tuple[int, ...] | None***<br>
&nbsp;&nbsp;&nbsp;&nbsp;Codes to exclude from expansion.<br>
&nbsp;&nbsp;**user_map**: ***dict | None***<br>
&nbsp;&nbsp;&nbsp;&nbsp;Optional code-to-label mapping overrides.<br>
&nbsp;&nbsp;**inplace**: ***bool***<br>
&nbsp;&nbsp;&nbsp;&nbsp;If True, modify df in place.<br>

**Returns**

&nbsp;&nbsp;**out**: ***pandas.DataFrame | tuple[pandas.DataFrame, pandas.DataFrame]***<br>
&nbsp;&nbsp;&nbsp;&nbsp;Updated dataframe (and mapping dataframe if returned by helper).

## resolve_categoricals_and_labels()

```python
phenofhy.process.resolve_categoricals_and_labels(df, traits, *, label_mode="labels",
	codebook_csv=None, autodetect_coded_categoricals=True, autodetect_max_levels=10)
```

Prepare a DataFrame and categorical trait list for summary.

**Parameters**

&nbsp;&nbsp;**df**: ***pandas.DataFrame***<br>
&nbsp;&nbsp;&nbsp;&nbsp;Input dataframe.<br>
&nbsp;&nbsp;**traits**: ***Iterable[str]***<br>
&nbsp;&nbsp;&nbsp;&nbsp;Trait columns to evaluate.<br>
&nbsp;&nbsp;**label_mode**: ***Literal["labels", "codes"]***<br>
&nbsp;&nbsp;&nbsp;&nbsp;Map codes to labels or reverse.<br>
&nbsp;&nbsp;**codebook_csv**: ***str | None***<br>
&nbsp;&nbsp;&nbsp;&nbsp;Optional codings CSV path.<br>
&nbsp;&nbsp;**autodetect_coded_categoricals**: ***bool***<br>
&nbsp;&nbsp;&nbsp;&nbsp;Whether to infer coded categoricals from numeric columns.<br>
&nbsp;&nbsp;**autodetect_max_levels**: ***int***<br>
&nbsp;&nbsp;&nbsp;&nbsp;Max unique values to consider numeric as categorical.<br>

**Returns**

&nbsp;&nbsp;**out**: ***tuple[pandas.DataFrame, list[str]]***<br>
&nbsp;&nbsp;&nbsp;&nbsp;Processed dataframe and categorical trait list.

**Example**
```python
from phenofhy import process

df = process.participant_fields("outputs/raw/phenos.csv")
df2, cat_traits = process.resolve_categoricals_and_labels(df, traits=["derived.sex"])
```
