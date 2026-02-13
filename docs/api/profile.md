# Profile

Phenotype profile reports for pre-GWAS QA.

## phenotype_profile()

```python
phenofhy.profile.phenotype_profile(df, phenotype, *, sex_col="derived.sex",
	age_col="derived.age_at_registration", bmi_col="derived.bmi",
	height_col="clinic_measurements.height", metadata_dir="./metadata",
	age_bin_width=2, body_bin_width=1, min_n_per_bin=30, output=None)
```

Generate a phenotype profile figure and summary metadata.

**Parameters**

&nbsp;&nbsp;**df**: ***pandas.DataFrame***<br>
&nbsp;&nbsp;&nbsp;&nbsp;Input dataframe.<br>
&nbsp;&nbsp;**phenotype**: ***str***<br>
&nbsp;&nbsp;&nbsp;&nbsp;Phenotype column name to profile.<br>
&nbsp;&nbsp;**sex_col**: ***str***<br>
&nbsp;&nbsp;&nbsp;&nbsp;Sex column for stratified plots.<br>
&nbsp;&nbsp;**age_col**: ***str***<br>
&nbsp;&nbsp;&nbsp;&nbsp;Age column for age trend plot.<br>
&nbsp;&nbsp;**bmi_col**: ***str | None***<br>
&nbsp;&nbsp;&nbsp;&nbsp;BMI column for body trend plot.<br>
&nbsp;&nbsp;**height_col**: ***str | None***<br>
&nbsp;&nbsp;&nbsp;&nbsp;Height column for body trend plot.<br>
&nbsp;&nbsp;**metadata_dir**: ***str***<br>
&nbsp;&nbsp;&nbsp;&nbsp;Directory containing metadata dictionary files.<br>
&nbsp;&nbsp;**age_bin_width**: ***int***<br>
&nbsp;&nbsp;&nbsp;&nbsp;Age bin width in years.<br>
&nbsp;&nbsp;**body_bin_width**: ***int***<br>
&nbsp;&nbsp;&nbsp;&nbsp;Bin width for BMI/height trends.<br>
&nbsp;&nbsp;**min_n_per_bin**: ***int***<br>
&nbsp;&nbsp;&nbsp;&nbsp;Minimum count required per bin to plot.<br>
&nbsp;&nbsp;**output**: ***str | None***<br>
&nbsp;&nbsp;&nbsp;&nbsp;Optional output PDF path.<br>

**Returns**

&nbsp;&nbsp;**out**: ***tuple[matplotlib.figure.Figure, dict]***<br>
&nbsp;&nbsp;&nbsp;&nbsp;Figure and metadata summary.

**Raises**

&nbsp;&nbsp;**KeyError**: ***Exception***<br>
&nbsp;&nbsp;&nbsp;&nbsp;If the phenotype column is missing.

**Example**
```python
from phenofhy import profile

report = profile.phenotype_profile(
    df,
    phenotype="derived.age_at_registration",
    output="outputs/reports/age_profile.pdf",
)
```
