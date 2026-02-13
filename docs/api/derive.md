# _derive_funcs

Core derivation helpers used by `process`.

## expand_multi_code_column()

```python
phenofhy._derive_funcs.expand_multi_code_column(df, codings_glob="./metadata/*.codings.csv",
	coding_name="MEDICAT_1_M", col="questionnaire.medicat_1_m",
	prefix="derived.medicates_", exclude_codes=(-7, -1, -3),
	abbrev_map=MEDICAT_ABBREV, inplace=True)
```

Expand a multi-code column into binary indicator columns.

**Parameters**

&nbsp;&nbsp;**df**: ***pandas.DataFrame***<br>
&nbsp;&nbsp;&nbsp;&nbsp;Input dataframe containing the multi-code column.<br>
&nbsp;&nbsp;**codings_glob**: ***str***<br>
&nbsp;&nbsp;&nbsp;&nbsp;Glob pattern for codings CSV files.<br>
&nbsp;&nbsp;**coding_name**: ***str***<br>
&nbsp;&nbsp;&nbsp;&nbsp;Coding name to filter within the codings file.<br>
&nbsp;&nbsp;**col**: ***str***<br>
&nbsp;&nbsp;&nbsp;&nbsp;Multi-code column to expand.<br>
&nbsp;&nbsp;**prefix**: ***str***<br>
&nbsp;&nbsp;&nbsp;&nbsp;Prefix for derived indicator columns.<br>
&nbsp;&nbsp;**exclude_codes**: ***Sequence[int]***<br>
&nbsp;&nbsp;&nbsp;&nbsp;Codes to exclude from expansion.<br>
&nbsp;&nbsp;**abbrev_map**: ***dict | None***<br>
&nbsp;&nbsp;&nbsp;&nbsp;Optional mapping for abbreviating column names.<br>
&nbsp;&nbsp;**inplace**: ***bool***<br>
&nbsp;&nbsp;&nbsp;&nbsp;If True, mutate the input dataframe.

**Returns**

&nbsp;&nbsp;**mapping**: ***pandas.DataFrame***<br>
&nbsp;&nbsp;&nbsp;&nbsp;Mapping metadata for created columns.

## self_reported_sex()

```python
phenofhy._derive_funcs.self_reported_sex(df)
```

Create derived.sex as numeric codes with categorical dtype.

**Parameters**

&nbsp;&nbsp;**df**: ***pandas.DataFrame***<br>
&nbsp;&nbsp;&nbsp;&nbsp;Input dataframe with demographic sex fields.

**Returns**

&nbsp;&nbsp;**out**: ***pandas.DataFrame***<br>
&nbsp;&nbsp;&nbsp;&nbsp;Dataframe with derived.sex added.

## registration_date()

```python
phenofhy._derive_funcs.registration_date(df)
```

Derive derived.registration_date as a datetime column.

**Parameters**

&nbsp;&nbsp;**df**: ***pandas.DataFrame***<br>
&nbsp;&nbsp;&nbsp;&nbsp;Input dataframe.

**Returns**

&nbsp;&nbsp;**out**: ***pandas.DataFrame***<br>
&nbsp;&nbsp;&nbsp;&nbsp;Dataframe with derived.registration_date added or coerced.

## age_at_registration()

```python
phenofhy._derive_funcs.age_at_registration(df)
```

Derive derived.age_at_registration as continuous age in years.

**Parameters**

&nbsp;&nbsp;**df**: ***pandas.DataFrame***<br>
&nbsp;&nbsp;&nbsp;&nbsp;Input dataframe with registration and birth date fields.

**Returns**

&nbsp;&nbsp;**out**: ***pandas.DataFrame***<br>
&nbsp;&nbsp;&nbsp;&nbsp;Dataframe with derived.age_at_registration added.

## age_group()

```python
phenofhy._derive_funcs.age_group(df, bins=None, labels=None)
```

Derive derived.age_group from continuous age.

**Parameters**

&nbsp;&nbsp;**df**: ***pandas.DataFrame***<br>
&nbsp;&nbsp;&nbsp;&nbsp;Input dataframe.
&nbsp;&nbsp;**bins**: ***list | None***<br>
&nbsp;&nbsp;&nbsp;&nbsp;Optional list of bin edges.
&nbsp;&nbsp;**labels**: ***list | None***<br>
&nbsp;&nbsp;&nbsp;&nbsp;Optional list of labels for bins.

**Returns**

&nbsp;&nbsp;**out**: ***pandas.DataFrame***<br>
&nbsp;&nbsp;&nbsp;&nbsp;Dataframe with derived.age_group added.

## bmi()

```python
phenofhy._derive_funcs.bmi(df)
```

Derive derived.bmi from height and weight.

**Parameters**

&nbsp;&nbsp;**df**: ***pandas.DataFrame***<br>
&nbsp;&nbsp;&nbsp;&nbsp;Input dataframe with clinic_measurements.height/weight.

**Returns**

&nbsp;&nbsp;**out**: ***pandas.DataFrame***<br>
&nbsp;&nbsp;&nbsp;&nbsp;Dataframe with derived.bmi added.

## bmi_status()

```python
phenofhy._derive_funcs.bmi_status(df)
```

Derive derived.bmi_status as categorical BMI class codes.

**Parameters**

&nbsp;&nbsp;**df**: ***pandas.DataFrame***<br>
&nbsp;&nbsp;&nbsp;&nbsp;Input dataframe with derived.bmi.

**Returns**

&nbsp;&nbsp;**out**: ***pandas.DataFrame***<br>
&nbsp;&nbsp;&nbsp;&nbsp;Dataframe with derived.bmi_status added.

## vape_status()

```python
phenofhy._derive_funcs.vape_status(df, numeric=True, out_col="derived.vape_status")
```

Derive vaping status as numeric codes with categorical dtype.

**Parameters**

&nbsp;&nbsp;**df**: ***pandas.DataFrame***<br>
&nbsp;&nbsp;&nbsp;&nbsp;Input dataframe with vaping-related questionnaire fields.
&nbsp;&nbsp;**numeric**: ***bool***<br>
&nbsp;&nbsp;&nbsp;&nbsp;Unused; kept for compatibility.
&nbsp;&nbsp;**out_col**: ***str***<br>
&nbsp;&nbsp;&nbsp;&nbsp;Output column name.

**Returns**

&nbsp;&nbsp;**out**: ***pandas.DataFrame***<br>
&nbsp;&nbsp;&nbsp;&nbsp;Dataframe with derived vaping status column.

## smoke_status_v1()

```python
phenofhy._derive_funcs.smoke_status_v1(df, numeric=True)
```

Derive derived.smoke_status_v1 as categorical smoking status codes.

**Parameters**

&nbsp;&nbsp;**df**: ***pandas.DataFrame***<br>
&nbsp;&nbsp;&nbsp;&nbsp;Input dataframe with smoking questionnaire fields.
&nbsp;&nbsp;**numeric**: ***bool***<br>
&nbsp;&nbsp;&nbsp;&nbsp;Unused; kept for compatibility.

**Returns**

&nbsp;&nbsp;**out**: ***pandas.DataFrame***<br>
&nbsp;&nbsp;&nbsp;&nbsp;Dataframe with derived.smoke_status_v1 added.

## smoke_status_v2()

```python
phenofhy._derive_funcs.smoke_status_v2(df, numeric=True)
```

Derive derived.smoke_status_v2 as categorical smoking status codes.

**Parameters**

&nbsp;&nbsp;**df**: ***pandas.DataFrame***<br>
&nbsp;&nbsp;&nbsp;&nbsp;Input dataframe with smoking questionnaire fields.
&nbsp;&nbsp;**numeric**: ***bool***<br>
&nbsp;&nbsp;&nbsp;&nbsp;Unused; kept for compatibility.

**Returns**

&nbsp;&nbsp;**out**: ***pandas.DataFrame***<br>
&nbsp;&nbsp;&nbsp;&nbsp;Dataframe with derived.smoke_status_v2 added.

## medicat_expand()

```python
phenofhy._derive_funcs.medicat_expand(df, codings_glob="./metadata/*.codings.csv",
	coding_name="MEDICAT_1_M", col="questionnaire.medicat_1_m",
	prefix="derived.medicates_", exclude_codes=(-7, -1, -3), abbrev_map=None)
```

Expand medicat multi-code column into binary flags.

**Parameters**

&nbsp;&nbsp;**df**: ***pandas.DataFrame***<br>
&nbsp;&nbsp;&nbsp;&nbsp;Input dataframe.
&nbsp;&nbsp;**codings_glob**: ***str***<br>
&nbsp;&nbsp;&nbsp;&nbsp;Glob pattern for codings CSV files.
&nbsp;&nbsp;**coding_name**: ***str***<br>
&nbsp;&nbsp;&nbsp;&nbsp;Coding name to filter within the codings file.
&nbsp;&nbsp;**col**: ***str***<br>
&nbsp;&nbsp;&nbsp;&nbsp;Multi-code column to expand.
&nbsp;&nbsp;**prefix**: ***str***<br>
&nbsp;&nbsp;&nbsp;&nbsp;Prefix for derived indicator columns.
&nbsp;&nbsp;**exclude_codes**: ***Sequence[int]***<br>
&nbsp;&nbsp;&nbsp;&nbsp;Codes to exclude from expansion.
&nbsp;&nbsp;**abbrev_map**: ***dict | None***<br>
&nbsp;&nbsp;&nbsp;&nbsp;Optional mapping for abbreviating column names.

**Returns**

&nbsp;&nbsp;**out**: ***pandas.DataFrame***<br>
&nbsp;&nbsp;&nbsp;&nbsp;Dataframe with derived medication flags added.

## any_hospital_contact()

```python
phenofhy._derive_funcs.any_hospital_contact(data, merged=None, before_registration=False)
```

Derive derived.any_hospital_contact as a binary flag.

**Parameters**

&nbsp;&nbsp;**data**: ***dict | pandas.DataFrame***<br>
&nbsp;&nbsp;&nbsp;&nbsp;Dict of entity DataFrames or a merged DataFrame.
&nbsp;&nbsp;**merged**: ***pandas.DataFrame | None***<br>
&nbsp;&nbsp;&nbsp;&nbsp;Optional merged DataFrame to attach the derived column to.
&nbsp;&nbsp;**before_registration**: ***bool***<br>
&nbsp;&nbsp;&nbsp;&nbsp;If True, count only pre-registration contacts.

**Returns**

&nbsp;&nbsp;**out**: ***pandas.DataFrame***<br>
&nbsp;&nbsp;&nbsp;&nbsp;Dataframe with derived.any_hospital_contact added.

## ae_visits()

```python
phenofhy._derive_funcs.ae_visits(data, merged=None, before_registration=False)
```

Derive derived.ae_visits as count of A&E visits per participant.

**Parameters**

&nbsp;&nbsp;**data**: ***dict | pandas.DataFrame***<br>
&nbsp;&nbsp;&nbsp;&nbsp;Dict of entity DataFrames or a merged DataFrame.
&nbsp;&nbsp;**merged**: ***pandas.DataFrame | None***<br>
&nbsp;&nbsp;&nbsp;&nbsp;Optional merged DataFrame to attach the derived column to.
&nbsp;&nbsp;**before_registration**: ***bool***<br>
&nbsp;&nbsp;&nbsp;&nbsp;If True, count only pre-registration visits.

**Returns**

&nbsp;&nbsp;**out**: ***pandas.DataFrame***<br>
&nbsp;&nbsp;&nbsp;&nbsp;Dataframe with derived.ae_visits added.

## apc_visits()

```python
phenofhy._derive_funcs.apc_visits(data, merged=None, before_registration=False)
```

Derive derived.apc_visits as count of inpatient admissions per participant.

**Parameters**

&nbsp;&nbsp;**data**: ***dict | pandas.DataFrame***<br>
&nbsp;&nbsp;&nbsp;&nbsp;Dict of entity DataFrames or a merged DataFrame.
&nbsp;&nbsp;**merged**: ***pandas.DataFrame | None***<br>
&nbsp;&nbsp;&nbsp;&nbsp;Optional merged DataFrame to attach the derived column to.
&nbsp;&nbsp;**before_registration**: ***bool***<br>
&nbsp;&nbsp;&nbsp;&nbsp;If True, count only pre-registration admissions.

**Returns**

&nbsp;&nbsp;**out**: ***pandas.DataFrame***<br>
&nbsp;&nbsp;&nbsp;&nbsp;Dataframe with derived.apc_visits added.

## op_visits()

```python
phenofhy._derive_funcs.op_visits(data, merged=None, before_registration=False)
```

Derive derived.op_visits as count of outpatient visits per participant.

**Parameters**

&nbsp;&nbsp;**data**: ***dict | pandas.DataFrame***<br>
&nbsp;&nbsp;&nbsp;&nbsp;Dict of entity DataFrames or a merged DataFrame.
&nbsp;&nbsp;**merged**: ***pandas.DataFrame | None***<br>
&nbsp;&nbsp;&nbsp;&nbsp;Optional merged DataFrame to attach the derived column to.
&nbsp;&nbsp;**before_registration**: ***bool***<br>
&nbsp;&nbsp;&nbsp;&nbsp;If True, count only pre-registration visits.

**Returns**

&nbsp;&nbsp;**out**: ***pandas.DataFrame***<br>
&nbsp;&nbsp;&nbsp;&nbsp;Dataframe with derived.op_visits added.

## total_hospital_contacts()

```python
phenofhy._derive_funcs.total_hospital_contacts(data, merged=None, before_registration=False, winsorize_pct=0.99)
```

Derive derived.total_hospital_contacts as total visit counts.

**Parameters**

&nbsp;&nbsp;**data**: ***dict | pandas.DataFrame***<br>
&nbsp;&nbsp;&nbsp;&nbsp;Dict of entity DataFrames or a merged DataFrame.
&nbsp;&nbsp;**merged**: ***pandas.DataFrame | None***<br>
&nbsp;&nbsp;&nbsp;&nbsp;Optional merged DataFrame to attach the derived column to.
&nbsp;&nbsp;**before_registration**: ***bool***<br>
&nbsp;&nbsp;&nbsp;&nbsp;If True, count only pre-registration events.
&nbsp;&nbsp;**winsorize_pct**: ***float***<br>
&nbsp;&nbsp;&nbsp;&nbsp;Upper percentile to winsorize counts (0-1).

**Returns**

&nbsp;&nbsp;**out**: ***pandas.DataFrame***<br>
&nbsp;&nbsp;&nbsp;&nbsp;Dataframe with derived.total_hospital_contacts added.
