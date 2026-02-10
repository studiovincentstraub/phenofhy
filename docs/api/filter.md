# _filter_funcs

Row-level filters and data cleaning helpers.

## Key functions

- `remove_known_errors()`
- `apply_row_filters()`
- `floor_age_series()`
- `filter_preferred_nonresponse()`

## Example

```python
from phenofhy import _filter_funcs

cleaned = _filter_funcs.remove_known_errors(df)
```
