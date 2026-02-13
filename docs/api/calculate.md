# Calculate

Summaries, prevalence tables, and association metrics.

## summary()

```python
phenofhy.calculate.summary(df, traits=None, *, stratify=None, sex_col="derived.sex",
	age_col="derived.age_at_registration", age_bins=None, round_decimals=2,
	categorical_traits=None, label_mode="codes", codebook_csv=None,
	metadata_dir="./metadata", data_dictionary_csv=None, local_codebook=None,
	autodetect_coded_categoricals=True, autodetect_max_levels=10,
	autodetect_exclude=None, sex_keep=None, granularity="variable")
```

Compute grouped summaries for numeric and categorical traits.

**Parameters**

&nbsp;&nbsp;**df**: ***pandas.DataFrame***<br>
&nbsp;&nbsp;&nbsp;&nbsp;Input dataframe with participant, questionnaire, and derived columns.<br>
&nbsp;&nbsp;**traits**: ***Iterable[str] | None***<br>
&nbsp;&nbsp;&nbsp;&nbsp;Optional iterable of column names to summarize. If None, auto-detects usable columns and skips ones ending with `.pid`.<br>
&nbsp;&nbsp;**stratify**: ***str | dict | None***<br>
&nbsp;&nbsp;&nbsp;&nbsp;None for whole-sample summary, a column name to stratify by, or a single-key dict `{col: [values]}` to restrict strata.<br>
&nbsp;&nbsp;**sex_col**: ***str***<br>
&nbsp;&nbsp;&nbsp;&nbsp;Column name for sex (used for default age group derivation).<br>
&nbsp;&nbsp;**age_col**: ***str***<br>
&nbsp;&nbsp;&nbsp;&nbsp;Column name for age (used for default age group derivation).<br>
&nbsp;&nbsp;**age_bins**: ***dict | None***<br>
&nbsp;&nbsp;&nbsp;&nbsp;Optional dict with keys `bins` and `labels` for age groupings.<br>
&nbsp;&nbsp;**round_decimals**: ***int***<br>
&nbsp;&nbsp;&nbsp;&nbsp;Number of decimals to round numeric outputs.<br>
&nbsp;&nbsp;**categorical_traits**: ***Iterable[str] | None***<br>
&nbsp;&nbsp;&nbsp;&nbsp;Optional iterable of traits to treat as categorical.<br>
&nbsp;&nbsp;**label_mode**: ***Literal["labels", "codes"]***<br>
&nbsp;&nbsp;&nbsp;&nbsp;`labels` to map codes to labels, `codes` to keep codes.<br>
&nbsp;&nbsp;**codebook_csv**: ***str | None***<br>
&nbsp;&nbsp;&nbsp;&nbsp;Optional path to codings CSV; if None, resolved from metadata_dir.<br>
&nbsp;&nbsp;**metadata_dir**: ***str***<br>
&nbsp;&nbsp;&nbsp;&nbsp;Directory to search for codings and metadata files.<br>
&nbsp;&nbsp;**data_dictionary_csv**: ***str | None***<br>
&nbsp;&nbsp;&nbsp;&nbsp;Optional path to data dictionary CSV for trait descriptions.<br>
&nbsp;&nbsp;**local_codebook**: ***dict | None***<br>
&nbsp;&nbsp;&nbsp;&nbsp;Optional mapping of derived column name to `{code: label}`.<br>
&nbsp;&nbsp;**autodetect_coded_categoricals**: ***bool***<br>
&nbsp;&nbsp;&nbsp;&nbsp;Whether to auto-detect small-cardinality numeric categoricals.<br>
&nbsp;&nbsp;**autodetect_max_levels**: ***int***<br>
&nbsp;&nbsp;&nbsp;&nbsp;Max unique values to treat numeric as categorical.<br>
&nbsp;&nbsp;**autodetect_exclude**: ***Iterable[str] | None***<br>
&nbsp;&nbsp;&nbsp;&nbsp;Optional iterable of traits to exclude from auto-detect.<br>
&nbsp;&nbsp;**sex_keep**: ***dict | None***<br>
&nbsp;&nbsp;&nbsp;&nbsp;Optional mapping to recode or keep sex values (currently unused).<br>
&nbsp;&nbsp;**granularity**: ***Literal["variable", "category"]***<br>
&nbsp;&nbsp;&nbsp;&nbsp;`variable` for aggregated rows, `category` for per-category rows.<br>

**Returns**

&nbsp;&nbsp;**out**: ***dict[str, pandas.DataFrame]***<br>
&nbsp;&nbsp;&nbsp;&nbsp;Dictionary with two DataFrames: `numeric` and `categorical`.

**Example**
```python
from phenofhy import calculate

result = calculate.summary(df, traits=["derived.age_at_registration", "derived.bmi"], stratify="derived.sex", age_bins={"bins": [18, 30, 60, 120], "labels": ["18-29", "30-59", "60+"]})
```

## prevalence()

```python
phenofhy.calculate.prevalence(df, codings=None, traits=None, *, denominator="all",
	denominators=None, eligibility=None, wide_output=True,
	participant_col="participant.pid", metadata_dir="./metadata",
	codebook_csv=None, on_missing="warn", error_if_empty=False)
```

Compute prevalence counts and rates for coded or derived categorical traits.

**Parameters**

&nbsp;&nbsp;**df**: ***pandas.DataFrame***<br>
&nbsp;&nbsp;&nbsp;&nbsp;Input dataframe with participant, questionnaire, and derived columns.<br>
&nbsp;&nbsp;**codings**: ***pandas.DataFrame | dict | list | None***<br>
&nbsp;&nbsp;&nbsp;&nbsp;Codings mapping or None to resolve from metadata_dir or codebook_csv.<br>
&nbsp;&nbsp;**traits**: ***Iterable[str] | None***<br>
&nbsp;&nbsp;&nbsp;&nbsp;Optional traits to include; if None, all non-helper columns are used.<br>
&nbsp;&nbsp;**denominator**: ***Literal["all", "nonmissing"]***<br>
&nbsp;&nbsp;&nbsp;&nbsp;`all` for total participants or `nonmissing` per trait.<br>
&nbsp;&nbsp;**denominators**: ***Iterable[str] | None***<br>
&nbsp;&nbsp;&nbsp;&nbsp;Optional iterable of additional denominator keys; if provided, output includes prevalence per key.<br>
&nbsp;&nbsp;**eligibility**: ***dict | None***<br>
&nbsp;&nbsp;&nbsp;&nbsp;Optional mapping of eligibility name to column or list of columns used for custom denominators.<br>
&nbsp;&nbsp;**wide_output**: ***bool***<br>
&nbsp;&nbsp;&nbsp;&nbsp;If True, return wide columns for each denominator key.<br>
&nbsp;&nbsp;**participant_col**: ***str***<br>
&nbsp;&nbsp;&nbsp;&nbsp;Column name for participant id.<br>
&nbsp;&nbsp;**metadata_dir**: ***str***<br>
&nbsp;&nbsp;&nbsp;&nbsp;Directory to search for codings CSV if not provided.<br>
&nbsp;&nbsp;**codebook_csv**: ***str | None***<br>
&nbsp;&nbsp;&nbsp;&nbsp;Optional path to codings CSV.<br>
&nbsp;&nbsp;**on_missing**: ***Literal["warn", "ignore", "error"]***<br>
&nbsp;&nbsp;&nbsp;&nbsp;Behavior when metadata is missing.<br>
&nbsp;&nbsp;**error_if_empty**: ***bool***<br>
&nbsp;&nbsp;&nbsp;&nbsp;If True, raise when no results are produced.<br>

**Returns**

&nbsp;&nbsp;**out**: ***pandas.DataFrame***<br>
&nbsp;&nbsp;&nbsp;&nbsp;Prevalence counts and rates. Shape depends on denominator arguments and wide_output.

**Example**
```python
from phenofhy import calculate

prev = calculate.prevalence(df, traits=["derived.sex", "participant.demog_ethnicity_1_1"], denominator="nonmissing")
```

## medication_prevalence()

```python
phenofhy.calculate.medication_prevalence(df, codings, medication_phenotypes, *,
	participant_col="participant.pid", denominator="all", return_what="both",
	fuzzy=True, fuzzy_cutoff=0.82, metadata_dir="./metadata", codebook_csv=None)
```

Compute medication prevalence for coded medication traits.

**Parameters**

&nbsp;&nbsp;**df**: ***pandas.DataFrame***<br>
&nbsp;&nbsp;&nbsp;&nbsp;Questionnaire or participant dataframe (fully-qualified columns).<br>
&nbsp;&nbsp;**codings**: ***pandas.DataFrame | dict | list | None***<br>
&nbsp;&nbsp;&nbsp;&nbsp;Codings mapping or None to resolve from metadata_dir or codebook_csv.<br>
&nbsp;&nbsp;**medication_phenotypes**: ***DataFrame | list | dict***<br>
&nbsp;&nbsp;&nbsp;&nbsp;Flexible specification of medication phenotypes. Each row resolves to `(trait, coding_name, medication)`.<br>
&nbsp;&nbsp;**participant_col**: ***str***<br>
&nbsp;&nbsp;&nbsp;&nbsp;Column name for participant id.<br>
&nbsp;&nbsp;**denominator**: ***Literal["all", "nonmissing"]***<br>
&nbsp;&nbsp;&nbsp;&nbsp;Prevalence denominator.<br>
&nbsp;&nbsp;**return_what**: ***Literal["both", "per_medication", "group"]***<br>
&nbsp;&nbsp;&nbsp;&nbsp;Return per-medication, grouped, or both.<br>
&nbsp;&nbsp;**fuzzy**: ***bool***<br>
&nbsp;&nbsp;&nbsp;&nbsp;Whether to use fuzzy matching for medication meanings.<br>
&nbsp;&nbsp;**fuzzy_cutoff**: ***float***<br>
&nbsp;&nbsp;&nbsp;&nbsp;Similarity cutoff for fuzzy matching.<br>
&nbsp;&nbsp;**metadata_dir**: ***str***<br>
&nbsp;&nbsp;&nbsp;&nbsp;Directory to search for codings CSV if not provided.<br>
&nbsp;&nbsp;**codebook_csv**: ***str | None***<br>
&nbsp;&nbsp;&nbsp;&nbsp;Optional path to codings CSV.<br>

**Returns**

&nbsp;&nbsp;**out**: ***pandas.DataFrame | tuple[pandas.DataFrame, pandas.DataFrame]***<br>
&nbsp;&nbsp;&nbsp;&nbsp;Per-medication and grouped results depending on return_what.

**Example**
```python
from phenofhy import calculate

per_med, grouped = calculate.medication_prevalence(df, codings=None, medication_phenotypes={"lipids": {"coding_name": "MEDICATIONS", "medication": ["Atorvastatin", "Simvastatin"]}}, return_what="both")
```

## medication_summary()

```python
phenofhy.calculate.medication_summary(df, *, med_prefix="derived.medicates_",
	group_map=DEFAULT_MEDICAT_GROUP_MAP, inplace=True, return_summary=False)
```

Derive medication usage-pattern variables and optional summary.

**Parameters**

&nbsp;&nbsp;**df**: ***pandas.DataFrame | tuple***<br>
&nbsp;&nbsp;&nbsp;&nbsp;Input dataframe or a `(mapping_df, df)` tuple (compatibility).<br>
&nbsp;&nbsp;**med_prefix**: ***str***<br>
&nbsp;&nbsp;&nbsp;&nbsp;Prefix used to detect medication domain columns.<br>
&nbsp;&nbsp;**group_map**: ***dict | None***<br>
&nbsp;&nbsp;&nbsp;&nbsp;Mapping of grouped medication flags to constituent columns.<br>
&nbsp;&nbsp;**inplace**: ***bool***<br>
&nbsp;&nbsp;&nbsp;&nbsp;If True, mutate the input dataframe; otherwise return a copy.<br>
&nbsp;&nbsp;**return_summary**: ***bool***<br>
&nbsp;&nbsp;&nbsp;&nbsp;If True, also return a summary DataFrame of derived vars.<br>

**Returns**

&nbsp;&nbsp;**out**: ***pandas.DataFrame | tuple[pandas.DataFrame, pandas.DataFrame]***<br>
&nbsp;&nbsp;&nbsp;&nbsp;If return_summary is False, returns the mutated or copied DataFrame. If return_summary is True, returns `(df, summary_df)`.

**Example**
```python
from phenofhy import calculate

df2, summary_df = calculate.medication_summary(df, return_summary=True)
```

## phi_corr()

```python
phenofhy.calculate.phi_corr(df, vars_for_heatmap=None, *,
	med_prefix="derived.medicates_", outdir=None, save_basename="phi_corr")
```

Compute a Pearson correlation matrix suitable for heatmaps.

**Parameters**

&nbsp;&nbsp;**df**: ***pandas.DataFrame***<br>
&nbsp;&nbsp;&nbsp;&nbsp;Input dataframe.<br>
&nbsp;&nbsp;**vars_for_heatmap**: ***Iterable[str] | None***<br>
&nbsp;&nbsp;&nbsp;&nbsp;Optional iterable of columns to include. If None, uses medication prefix columns plus common usage-pattern vars.<br>
&nbsp;&nbsp;**med_prefix**: ***str***<br>
&nbsp;&nbsp;&nbsp;&nbsp;Prefix used to select medication columns when vars_for_heatmap is None.<br>
&nbsp;&nbsp;**outdir**: ***str | None***<br>
&nbsp;&nbsp;&nbsp;&nbsp;Optional directory to save CSV or parquet outputs.<br>
&nbsp;&nbsp;**save_basename**: ***str***<br>
&nbsp;&nbsp;&nbsp;&nbsp;Basename for saved outputs if outdir is provided.<br>

**Returns**

&nbsp;&nbsp;**out**: ***pandas.DataFrame***<br>
&nbsp;&nbsp;&nbsp;&nbsp;Square correlation matrix with values in [-1, 1].

**Example**
```python
from phenofhy import calculate

corr = calculate.phi_corr(df, outdir="outputs", save_basename="med_phi_corr")
```

## matthews_corrcoef_series()

```python
phenofhy.calculate.matthews_corrcoef_series(a, b)
```

Compute Matthews correlation coefficient (phi) for two series.

**Parameters**

&nbsp;&nbsp;**a**: ***pandas.Series***<br>
&nbsp;&nbsp;&nbsp;&nbsp;Series to compare (binarized if needed).<br>
&nbsp;&nbsp;**b**: ***pandas.Series***<br>
&nbsp;&nbsp;&nbsp;&nbsp;Series to compare (binarized if needed).<br>

**Returns**

&nbsp;&nbsp;**out**: ***float***<br>
&nbsp;&nbsp;&nbsp;&nbsp;Phi coefficient in [-1, 1]. If the confusion matrix is degenerate, returns Pearson on the binarized arrays or 0.0.

**Raises**

&nbsp;&nbsp;**ValueError**: ***Exception***<br>
&nbsp;&nbsp;&nbsp;&nbsp;If either series is not binary-like after binarization.

**Example**
```python
from phenofhy import calculate

phi = calculate.matthews_corrcoef_series(df["derived.any_meds_flag"], df["derived.polypharmacy_flag"])
```
