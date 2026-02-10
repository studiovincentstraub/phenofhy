# _derive_funcs

Core derivation helpers used by `process`.

## Key functions

- `expand_multi_code_column()`
- `self_reported_sex()`
- `registration_date()`
- `age_at_registration()`
- `age_group()`
- `bmi()`
- `bmi_status()`
- `vape_status()`
- `smoke_status_v1()` and `smoke_status_v2()`
- `medicat_expand()`
- `any_hospital_contact()`
- `ae_visits()`, `apc_visits()`, `op_visits()`
- `total_hospital_contacts()`

## Example

```python
from phenofhy import _derive_funcs

mapping = _derive_funcs.expand_multi_code_column(df)
```
