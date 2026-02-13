# ICD

ICD trait matching for pandas and Spark workflows.

## match_icd_traits()

```python
phenofhy.icd.match_icd_traits(raw_df, traits_and_codes, *, diag_cols=None,
	pid_col="nhse_eng_inpat.pid", prefix_if_len_at_most=3, primary_only=False,
	return_occurrence_counts=True, use_pyarrow_strings=True, chunksize=None,
	diag_prefix="nhse_eng_inpat.diag_4_", extra_diag_cols=None, all_if_none=False)
```

Match ICD traits in a pandas DataFrame.

**Parameters**

&nbsp;&nbsp;**raw_df**: ***pandas.DataFrame***<br>
&nbsp;&nbsp;&nbsp;&nbsp;Input dataframe with diagnosis columns.<br>
&nbsp;&nbsp;**traits_and_codes**: ***dict | pandas.DataFrame | tuple[list, list]***<br>
&nbsp;&nbsp;&nbsp;&nbsp;Trait/code mapping as dict, DataFrame with `trait`/`ICD_code`, or aligned sequences.<br>
&nbsp;&nbsp;**diag_cols**: ***list[str] | None***<br>
&nbsp;&nbsp;&nbsp;&nbsp;Explicit diagnosis columns to scan.<br>
&nbsp;&nbsp;**pid_col**: ***str***<br>
&nbsp;&nbsp;&nbsp;&nbsp;Participant ID column name.<br>
&nbsp;&nbsp;**prefix_if_len_at_most**: ***int | None***<br>
&nbsp;&nbsp;&nbsp;&nbsp;Codes at or below this length are treated as prefixes.<br>
&nbsp;&nbsp;**primary_only**: ***bool***<br>
&nbsp;&nbsp;&nbsp;&nbsp;If True, use only the primary diagnosis column.<br>
&nbsp;&nbsp;**return_occurrence_counts**: ***bool***<br>
&nbsp;&nbsp;&nbsp;&nbsp;If True, include occurrence counts in the summary.<br>
&nbsp;&nbsp;**use_pyarrow_strings**: ***bool***<br>
&nbsp;&nbsp;&nbsp;&nbsp;Prefer pyarrow string dtype for faster vectorized ops.<br>
&nbsp;&nbsp;**chunksize**: ***int | None***<br>
&nbsp;&nbsp;&nbsp;&nbsp;Optional chunk size for large dataframes.<br>
&nbsp;&nbsp;**diag_prefix**: ***str | None***<br>
&nbsp;&nbsp;&nbsp;&nbsp;Prefix to select diagnosis columns when `diag_cols` is None.<br>
&nbsp;&nbsp;**extra_diag_cols**: ***list[str] | None***<br>
&nbsp;&nbsp;&nbsp;&nbsp;Additional diagnosis columns to include when present.<br>
&nbsp;&nbsp;**all_if_none**: ***bool***<br>
&nbsp;&nbsp;&nbsp;&nbsp;If True, use all non-pid columns when no diagnosis columns are found.<br>

**Returns**

&nbsp;&nbsp;**out**: ***tuple[dict[str, set], pandas.DataFrame]***<br>
&nbsp;&nbsp;&nbsp;&nbsp;Trait-to-pids mapping and a summary DataFrame.

**Example**
```python
from phenofhy import icd

trait_map = {"asthma": ["J45", "J46"]}
trait_pids, summary = icd.match_icd_traits(df, trait_map)
```

## get_matched_icd_traits()

```python
phenofhy.icd.get_matched_icd_traits(raw_df, traits_and_codes, *, diag_cols=None,
	prefix_if_len_at_most=3, uppercase=True, remove_chars=())
```

Summarize matched ICD codes per trait (pandas path).

**Parameters**

&nbsp;&nbsp;**raw_df**: ***pandas.DataFrame***<br>
&nbsp;&nbsp;&nbsp;&nbsp;Input dataframe with diagnosis columns.<br>
&nbsp;&nbsp;**traits_and_codes**: ***dict | pandas.DataFrame | tuple[list, list]***<br>
&nbsp;&nbsp;&nbsp;&nbsp;Trait/code mapping as dict, DataFrame with `trait`/`ICD_code`, or aligned sequences.<br>
&nbsp;&nbsp;**diag_cols**: ***list[str] | None***<br>
&nbsp;&nbsp;&nbsp;&nbsp;Explicit diagnosis columns to scan.<br>
&nbsp;&nbsp;**prefix_if_len_at_most**: ***int***<br>
&nbsp;&nbsp;&nbsp;&nbsp;Codes at or below this length are treated as prefixes.<br>
&nbsp;&nbsp;**uppercase**: ***bool***<br>
&nbsp;&nbsp;&nbsp;&nbsp;Whether to uppercase codes before matching.<br>
&nbsp;&nbsp;**remove_chars**: ***tuple[str, ...]***<br>
&nbsp;&nbsp;&nbsp;&nbsp;Characters to remove before matching.<br>

**Returns**

&nbsp;&nbsp;**out**: ***pandas.DataFrame***<br>
&nbsp;&nbsp;&nbsp;&nbsp;DataFrame with columns `trait`, `n_unique_codes`, `unique_codes`.

## match_icd_traits_spark()

```python
phenofhy.icd.match_icd_traits_spark(sdf, traits_and_codes, *,
	pid_col="nhse_eng_inpat.pid", diag_prefix="nhse_eng_inpat.diag_4_",
	extra_diag_cols=None, primary_only=False, prefix_if_len_at_most=3,
	uppercase=True, remove_chars=(), return_occurrence_counts=True,
	return_pids=False, max_pids_collect=200_000)
```

Match ICD traits in Spark with broadcasted trait codes.

**Parameters**

&nbsp;&nbsp;**sdf**: ***pyspark.sql.DataFrame***<br>
&nbsp;&nbsp;&nbsp;&nbsp;Spark dataframe with diagnosis columns.<br>
&nbsp;&nbsp;**traits_and_codes**: ***dict | pandas.DataFrame | tuple[list, list]***<br>
&nbsp;&nbsp;&nbsp;&nbsp;Trait/code mapping as dict, DataFrame with `trait`/`ICD_code`, or aligned sequences.<br>
&nbsp;&nbsp;**pid_col**: ***str***<br>
&nbsp;&nbsp;&nbsp;&nbsp;Participant ID column name.<br>
&nbsp;&nbsp;**diag_prefix**: ***str***<br>
&nbsp;&nbsp;&nbsp;&nbsp;Prefix to select diagnosis columns.<br>
&nbsp;&nbsp;**extra_diag_cols**: ***list[str] | None***<br>
&nbsp;&nbsp;&nbsp;&nbsp;Additional diagnosis columns to include when present.<br>
&nbsp;&nbsp;**primary_only**: ***bool***<br>
&nbsp;&nbsp;&nbsp;&nbsp;If True, use only the primary diagnosis column.<br>
&nbsp;&nbsp;**prefix_if_len_at_most**: ***int | None***<br>
&nbsp;&nbsp;&nbsp;&nbsp;Codes at or below this length are treated as prefixes.<br>
&nbsp;&nbsp;**uppercase**: ***bool***<br>
&nbsp;&nbsp;&nbsp;&nbsp;Whether to uppercase codes before matching.<br>
&nbsp;&nbsp;**remove_chars**: ***tuple[str, ...]***<br>
&nbsp;&nbsp;&nbsp;&nbsp;Characters to remove before matching.<br>
&nbsp;&nbsp;**return_occurrence_counts**: ***bool***<br>
&nbsp;&nbsp;&nbsp;&nbsp;If True, include occurrence counts in the summary.<br>
&nbsp;&nbsp;**return_pids**: ***bool***<br>
&nbsp;&nbsp;&nbsp;&nbsp;If True, collect pid sets for small traits only.<br>
&nbsp;&nbsp;**max_pids_collect**: ***int***<br>
&nbsp;&nbsp;&nbsp;&nbsp;Max pid set size to collect per trait.<br>

**Returns**

&nbsp;&nbsp;**out**: ***tuple[dict[str, set], pandas.DataFrame]***<br>
&nbsp;&nbsp;&nbsp;&nbsp;Trait-to-pids mapping and a summary DataFrame.

## get_matched_icd_traits_spark()

```python
phenofhy.icd.get_matched_icd_traits_spark(sdf, traits_and_codes, *,
	pid_col="nhse_eng_inpat.pid", diag_prefix="nhse_eng_inpat.diag_4_",
	extra_diag_cols=None, primary_only=False, prefix_if_len_at_most=3,
	uppercase=True, remove_chars=())
```

Summarize matched ICD codes per trait (Spark path).

**Parameters**

&nbsp;&nbsp;**sdf**: ***pyspark.sql.DataFrame***<br>
&nbsp;&nbsp;&nbsp;&nbsp;Spark dataframe with diagnosis columns.<br>
&nbsp;&nbsp;**traits_and_codes**: ***dict | pandas.DataFrame | tuple[list, list]***<br>
&nbsp;&nbsp;&nbsp;&nbsp;Trait/code mapping as dict, DataFrame with `trait`/`ICD_code`, or aligned sequences.<br>
&nbsp;&nbsp;**pid_col**: ***str***<br>
&nbsp;&nbsp;&nbsp;&nbsp;Participant ID column name.<br>
&nbsp;&nbsp;**diag_prefix**: ***str***<br>
&nbsp;&nbsp;&nbsp;&nbsp;Prefix to select diagnosis columns.<br>
&nbsp;&nbsp;**extra_diag_cols**: ***list[str] | None***<br>
&nbsp;&nbsp;&nbsp;&nbsp;Additional diagnosis columns to include when present.<br>
&nbsp;&nbsp;**primary_only**: ***bool***<br>
&nbsp;&nbsp;&nbsp;&nbsp;If True, use only the primary diagnosis column.<br>
&nbsp;&nbsp;**prefix_if_len_at_most**: ***int***<br>
&nbsp;&nbsp;&nbsp;&nbsp;Codes at or below this length are treated as prefixes.<br>
&nbsp;&nbsp;**uppercase**: ***bool***<br>
&nbsp;&nbsp;&nbsp;&nbsp;Whether to uppercase codes before matching.<br>
&nbsp;&nbsp;**remove_chars**: ***tuple[str, ...]***<br>
&nbsp;&nbsp;&nbsp;&nbsp;Characters to remove before matching.<br>

**Returns**

&nbsp;&nbsp;**out**: ***pandas.DataFrame***<br>
&nbsp;&nbsp;&nbsp;&nbsp;DataFrame with columns `trait`, `n_unique_codes`, `unique_codes`.

## match_icd_traits_any()

```python
phenofhy.icd.match_icd_traits_any(df_or_sdf, *args, **kwargs)
```

Dispatch to pandas or Spark matching depending on input type.

**Parameters**

&nbsp;&nbsp;**df_or_sdf**: ***pandas.DataFrame | pyspark.sql.DataFrame***<br>
&nbsp;&nbsp;&nbsp;&nbsp;Input dataframe or Spark dataframe.<br>
&nbsp;&nbsp;**\*args**: ***tuple***<br>
&nbsp;&nbsp;&nbsp;&nbsp;Positional arguments forwarded to the implementation.<br>
&nbsp;&nbsp;**\*\*kwargs**: ***dict***<br>
&nbsp;&nbsp;&nbsp;&nbsp;Keyword arguments forwarded to the implementation.<br>

**Returns**

&nbsp;&nbsp;**out**: ***tuple[dict[str, set], pandas.DataFrame]***<br>
&nbsp;&nbsp;&nbsp;&nbsp;Trait-to-pids mapping and a summary DataFrame.

## get_matched_icd_traits_any()

```python
phenofhy.icd.get_matched_icd_traits_any(df_or_sdf, *args, **kwargs)
```

Dispatch to pandas or Spark code-summary depending on input type.

**Parameters**

&nbsp;&nbsp;**df_or_sdf**: ***pandas.DataFrame | pyspark.sql.DataFrame***<br>
&nbsp;&nbsp;&nbsp;&nbsp;Input dataframe or Spark dataframe.<br>
&nbsp;&nbsp;**\*args**: ***tuple***<br>
&nbsp;&nbsp;&nbsp;&nbsp;Positional arguments forwarded to the implementation.<br>
&nbsp;&nbsp;**\*\*kwargs**: ***dict***<br>
&nbsp;&nbsp;&nbsp;&nbsp;Keyword arguments forwarded to the implementation.<br>

**Returns**

&nbsp;&nbsp;**out**: ***pandas.DataFrame***<br>
&nbsp;&nbsp;&nbsp;&nbsp;DataFrame with columns `trait`, `n_unique_codes`, `unique_codes`.
