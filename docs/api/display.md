# Display

Formatting helpers for categorical summary tables.

## group_filter()

```python
phenofhy.display.group_filter(df, group_col="group", level_col="level",
	variable_col="variable", value_col="value", *, group_order=None,
	variable_order=None, drop_missing_group=False, drop_missing_variable=False)
```

Filter and order grouped categorical summaries for presentation.

**Parameters**

&nbsp;&nbsp;**df**: ***pandas.DataFrame***<br>
&nbsp;&nbsp;&nbsp;&nbsp;Categorical summary dataframe.<br>
&nbsp;&nbsp;**group_col**: ***str***<br>
&nbsp;&nbsp;&nbsp;&nbsp;Column name identifying groups.<br>
&nbsp;&nbsp;**level_col**: ***str***<br>
&nbsp;&nbsp;&nbsp;&nbsp;Column name for category level labels.<br>
&nbsp;&nbsp;**variable_col**: ***str***<br>
&nbsp;&nbsp;&nbsp;&nbsp;Column name for variable names.<br>
&nbsp;&nbsp;**value_col**: ***str***<br>
&nbsp;&nbsp;&nbsp;&nbsp;Column name for formatted values.<br>
&nbsp;&nbsp;**group_order**: ***list[str] | None***<br>
&nbsp;&nbsp;&nbsp;&nbsp;Optional explicit ordering of groups.<br>
&nbsp;&nbsp;**variable_order**: ***list[str] | None***<br>
&nbsp;&nbsp;&nbsp;&nbsp;Optional explicit ordering of variables.<br>
&nbsp;&nbsp;**drop_missing_group**: ***bool***<br>
&nbsp;&nbsp;&nbsp;&nbsp;Whether to drop rows with missing group labels.<br>
&nbsp;&nbsp;**drop_missing_variable**: ***bool***<br>
&nbsp;&nbsp;&nbsp;&nbsp;Whether to drop rows with missing variable labels.<br>

**Returns**

&nbsp;&nbsp;**out**: ***pandas.DataFrame***<br>
&nbsp;&nbsp;&nbsp;&nbsp;Filtered and ordered categorical summary dataframe.

**Example**
```python
from phenofhy import display

panel = display.group_filter(categorical_df)
```
