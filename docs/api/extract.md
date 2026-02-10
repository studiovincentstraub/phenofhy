# extract

Extraction helpers for DNAnexus datasets, including SQL export.

## Key functions

- `fields()`
  - Extract raw phenotype values to CSV, or generate SQL if `sql_only=True`.
- `run_extraction()`
  - Lower-level entry point used by `fields()`.
- `sql_to_pandas()` and `sql_to_spark()`
  - Convert SQL to pandas or Spark dataframes.

## Example

```python
from phenofhy import extract

extract.fields(
    output_file="outputs/raw/phenos.csv",
    fields=["participant.birth_year", "participant.birth_month"],
)
```
