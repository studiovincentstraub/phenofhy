# pipeline

Utilities for metadata dictionaries and phenotype list processing.

## Key functions

- `metadata()`
  - Downloads `codings`, `data_dictionary`, and `entity_dictionary` into `./metadata`.
- `field_list()`
  - Accepts a list/dict of `entity.field` values or a phenotype list file and returns a merged metadata table.

## Example

```python
from phenofhy import pipeline

meta = pipeline.metadata()
fields = pipeline.field_list(fields=["participant.birth_year", "participant.birth_month"])
```
