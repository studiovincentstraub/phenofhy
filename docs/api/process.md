# process

Entity-aware processing, derivations, and data cleaning.

## Key functions

- `participant_fields()`
- `questionnaire_fields()`
- `clinic_measurements_fields()`
- `get_dummies()`
- `resolve_categoricals_and_labels()`

## Example

```python
from phenofhy import process

df = process.participant_fields("outputs/raw/phenos.csv")
```
