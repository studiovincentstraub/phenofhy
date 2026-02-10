# profile

Phenotype profile reports for pre-GWAS QA.

## Key functions

- `phenotype_profile()`

## Example

```python
from phenofhy import profile

report = profile.phenotype_profile(
    df,
    phenotype="derived.age_at_registration",
    output="outputs/reports/age_profile.pdf",
)
```
