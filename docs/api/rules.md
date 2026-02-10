# _rules

Coalescing rules and default codebooks used by processing.

## Key functions

- `rule_cat()`
- `rule_num()`
- `coalesce_traits()`
- `build_rules()`

## Key data

- `DEFAULT_COALESCE_RULES`
- `DERIVED_CODEBOOK`

## Example

```python
from phenofhy import _rules

rules = _rules.build_rules()
```
