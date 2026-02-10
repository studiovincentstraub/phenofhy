# calculate

Summaries, prevalence tables, and association metrics.

## Key functions

- `summary()`
- `prevalence()`
- `medication_prevalence()`
- `medication_summary()`
- `phi_corr()`
- `matthews_corrcoef_series()`

## Example

```python
from phenofhy import calculate

summary = calculate.summary(df, traits=["derived.age_at_registration"])
```
