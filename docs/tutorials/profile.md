# Profiling a Phenotype

This tutorial explains what `profile.py` produces and how to use it.

## What `phenotype_profile()` generates

`phenotype_profile()` creates a multi panel figure and summary metadata for a single phenotype. The figure is designed as a one page report with the following sections:

- Metadata: field name, entity, type, and units (if available in a data dictionary).
- Demographic summary table (counts and percent by sex, plus age summary stats).
- Phenotype completeness (missing vs non-missing for the target phenotype).
- Distribution by sex (histogram for continuous, bar chart for categorical).
- Mean or proportion trends by age and body measurements (binned by age, BMI, or height).

The function also returns a dictionary with the phenotype type, sample flow, and demographics table, and can optionally save the figure to a PDF.

## Example

```python
from phenofhy.profile import phenotype_profile

fig, meta = phenotype_profile(
    df,
    phenotype="clinic_measurements.heart_first_rate",
    age_bin_width=2,
    body_bin_width=1,
    min_n_per_bin=30,
    error_mode="se",
    output="outputs/profile_heart_rate.pdf",
)
```

## Notes

- For continuous phenotypes, trend plots show mean with SE by default (use `error_mode="sd"` for SD).
- For binary phenotypes, trend plots show proportion with SE by default.
- Categorical phenotypes do not show trend plots.
- The data dictionary lookup uses the first matching `*data_dictionary.csv` in `metadata_dir`.
