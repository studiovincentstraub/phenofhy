# _filter_funcs

Row-level filters and data cleaning helpers.

## remove_known_errors()

```python
phenofhy._filter_funcs.remove_known_errors(df, *, clinic_ranges=None)
```

Apply known error-removal helpers when required columns exist.

**Parameters**

&nbsp;&nbsp;**df**: ***pandas.DataFrame***<br>
&nbsp;&nbsp;&nbsp;&nbsp;Input dataframe.
&nbsp;&nbsp;**clinic_ranges**: ***dict | None***<br>
&nbsp;&nbsp;&nbsp;&nbsp;Optional mapping of clinic column ranges.

**Returns**

&nbsp;&nbsp;**out**: ***pandas.DataFrame***<br>
&nbsp;&nbsp;&nbsp;&nbsp;Filtered dataframe with known error rows removed.

**Example**
```python
from phenofhy import _filter_funcs

cleaned = _filter_funcs.remove_known_errors(df)
```

## apply_row_filters()

```python
phenofhy._filter_funcs.apply_row_filters(df, *, ranges=None, exprs=None, inclusive="both", keep_na=False, ignore_missing_range_cols=False)
```

Apply range- and expression-based row filters.

**Parameters**

&nbsp;&nbsp;**df**: ***pandas.DataFrame***<br>
&nbsp;&nbsp;&nbsp;&nbsp;Input dataframe.
&nbsp;&nbsp;**ranges**: ***dict | None***<br>
&nbsp;&nbsp;&nbsp;&nbsp;Mapping of column -> (low, high) bounds.
&nbsp;&nbsp;**exprs**: ***list[str] | None***<br>
&nbsp;&nbsp;&nbsp;&nbsp;Optional list of pandas eval() expressions to AND into the mask.
&nbsp;&nbsp;**inclusive**: ***str***<br>
&nbsp;&nbsp;&nbsp;&nbsp;Bound inclusion for between().
&nbsp;&nbsp;**keep_na**: ***bool***<br>
&nbsp;&nbsp;&nbsp;&nbsp;If True, retain rows with NA in range columns.
&nbsp;&nbsp;**ignore_missing_range_cols**: ***bool***<br>
&nbsp;&nbsp;&nbsp;&nbsp;If True, skip missing range columns.

**Returns**

&nbsp;&nbsp;**out**: ***pandas.DataFrame***<br>
&nbsp;&nbsp;&nbsp;&nbsp;Filtered dataframe.

**Example**
```python
from phenofhy import _filter_funcs

filtered = _filter_funcs.apply_row_filters(
	df,
	ranges={"clinic_measurements.weight": (30, 200)},
	keep_na=True,
)
```

## floor_age_series()

```python
phenofhy._filter_funcs.floor_age_series(s)
```

Floor ages to integer years, treating negatives as missing.

**Parameters**

&nbsp;&nbsp;**s**: ***pandas.Series***<br>
&nbsp;&nbsp;&nbsp;&nbsp;Input series of ages.

**Returns**

&nbsp;&nbsp;**out**: ***pandas.Series***<br>
&nbsp;&nbsp;&nbsp;&nbsp;Nullable Int64 series of floored ages.

**Example**
```python
from phenofhy import _filter_funcs

floored = _filter_funcs.floor_age_series(df["derived.age_at_registration"])
```

## filter_preferred_nonresponse()

```python
phenofhy._filter_funcs.filter_preferred_nonresponse(df)
```

Remove rows with preferred non-response values in key demographics.

**Parameters**

&nbsp;&nbsp;**df**: ***pandas.DataFrame***<br>
&nbsp;&nbsp;&nbsp;&nbsp;Input dataframe.

**Returns**

&nbsp;&nbsp;**out**: ***pandas.DataFrame***<br>
&nbsp;&nbsp;&nbsp;&nbsp;Filtered dataframe with non-substantive responses removed.

**Example**
```python
from phenofhy import _filter_funcs

cleaned = _filter_funcs.filter_preferred_nonresponse(df)
```
