# icd

ICD trait matching for pandas and Spark workflows.

## Key functions

- `match_icd_traits()`
- `get_matched_icd_traits()`
- `match_icd_traits_spark()`
- `get_matched_icd_traits_spark()`
- `match_icd_traits_any()`
- `get_matched_icd_traits_any()`

## Example

```python
from phenofhy import icd

trait_map = {"asthma": ["J45", "J46"]}
trait_pids, summary = icd.match_icd_traits(df, trait_map)
```
