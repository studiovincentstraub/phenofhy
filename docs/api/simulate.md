# Simulate

Synthetic phenotype dataframe generation utilities for local testing and examples.

## simulate_phenotype_df()

```python
phenofhy.simulate.simulate_phenotype_df(sample=1000, fields=None, *,
	include_nonresponse=False, missing_rate=None, seed=None)
```

Simulate an OFH-like phenotype dataframe from metadata dictionaries and coding domains.

**Parameters**

&nbsp;&nbsp;**sample**: ***int***<br>
&nbsp;&nbsp;&nbsp;&nbsp;Number of rows to generate. Must be >= 1.<br>
&nbsp;&nbsp;**fields**: ***str | Sequence[str] | None***<br>
&nbsp;&nbsp;&nbsp;&nbsp;Single field name, sequence of field names, or None to use DEFAULT_FIELDS.<br>
&nbsp;&nbsp;**include_nonresponse**: ***bool***<br>
&nbsp;&nbsp;&nbsp;&nbsp;If True, keep known non-response/sentinel codes in coded domains.<br>
&nbsp;&nbsp;**missing_rate**: ***float | Mapping[str, float] | None***<br>
&nbsp;&nbsp;&nbsp;&nbsp;Global missingness probability or per-field mapping. If None, uses module defaults.<br>
&nbsp;&nbsp;**seed**: ***int | None***<br>
&nbsp;&nbsp;&nbsp;&nbsp;Optional random seed for reproducible output.<br>

**Returns**

&nbsp;&nbsp;**out**: ***pandas.DataFrame***<br>
&nbsp;&nbsp;&nbsp;&nbsp;Simulated dataframe with requested columns.

**Raises**

&nbsp;&nbsp;**ValueError**: ***Exception***<br>
&nbsp;&nbsp;&nbsp;&nbsp;If sample is < 1, requested fields are unknown, or missing_rate values are outside [0, 1].

**Example**
```python
from phenofhy import simulate

df = simulate.simulate_phenotype_df(
    sample=500,
    fields=["participant.pid", "participant.birth_year", "clinic_measurements.weight"],
    include_nonresponse=False,
    seed=42,
)
```

## _normalize_fields()

```python
phenofhy.simulate._normalize_fields(fields)
```

Normalize field selection input to a non-empty list of field names.

**Parameters**

&nbsp;&nbsp;**fields**: ***str | Sequence[str] | None***<br>
&nbsp;&nbsp;&nbsp;&nbsp;Single field name, sequence of names, or None.<br>

**Returns**

&nbsp;&nbsp;**out**: ***list[str]***<br>
&nbsp;&nbsp;&nbsp;&nbsp;Normalized field list. If fields is None, returns DEFAULT_FIELDS.

**Raises**

&nbsp;&nbsp;**ValueError**: ***Exception***<br>
&nbsp;&nbsp;&nbsp;&nbsp;If fields is an empty sequence.

## _helpers_dir()

```python
phenofhy.simulate._helpers_dir()
```

Resolve the helpers directory path used by metadata loaders.

**Returns**

&nbsp;&nbsp;**out**: ***pathlib.Path***<br>
&nbsp;&nbsp;&nbsp;&nbsp;Absolute path to beta/helpers.

## _data_dictionary()

```python
phenofhy.simulate._data_dictionary()
```

Load and cache metadata from data_dictionary.csv with a computed full_field column.

**Returns**

&nbsp;&nbsp;**out**: ***pandas.DataFrame***<br>
&nbsp;&nbsp;&nbsp;&nbsp;Data dictionary with entity, field, type, and full_field columns.

## _coded_domains()

```python
phenofhy.simulate._coded_domains(include_nonresponse)
```

Load and cache coded value domains from codings.csv.

**Parameters**

&nbsp;&nbsp;**include_nonresponse**: ***bool***<br>
&nbsp;&nbsp;&nbsp;&nbsp;Whether to keep known non-response/sentinel codes.

**Returns**

&nbsp;&nbsp;**out**: ***dict[str, list[float]]***<br>
&nbsp;&nbsp;&nbsp;&nbsp;Mapping of full field names to available numeric code domains.

**Raises**

&nbsp;&nbsp;**ValueError**: ***Exception***<br>
&nbsp;&nbsp;&nbsp;&nbsp;If codings.csv does not contain either coding_name or field.

## _filter_nonresponse_codes()

```python
phenofhy.simulate._filter_nonresponse_codes(field, codes, include_nonresponse)
```

Filter out known non-response codes for a coded field when requested.

**Parameters**

&nbsp;&nbsp;**field**: ***str***<br>
&nbsp;&nbsp;&nbsp;&nbsp;Full field name (entity.field) used for field-specific exclusions.<br>
&nbsp;&nbsp;**codes**: ***Sequence[float]***<br>
&nbsp;&nbsp;&nbsp;&nbsp;Candidate numeric codes.<br>
&nbsp;&nbsp;**include_nonresponse**: ***bool***<br>
&nbsp;&nbsp;&nbsp;&nbsp;If True, returns codes unchanged.

**Returns**

&nbsp;&nbsp;**out**: ***list[float]***<br>
&nbsp;&nbsp;&nbsp;&nbsp;Filtered code list with known sentinel values removed when include_nonresponse is False.

## _simulate_column()

```python
phenofhy.simulate._simulate_column(*, field, dtype, sample, rng, coded_domains)
```

Generate a synthetic column for one field using coded domains or dtype-specific fallbacks.

**Parameters**

&nbsp;&nbsp;**field**: ***str***<br>
&nbsp;&nbsp;&nbsp;&nbsp;Full field name (entity.field).<br>
&nbsp;&nbsp;**dtype**: ***str***<br>
&nbsp;&nbsp;&nbsp;&nbsp;Field dtype string (integer, float, string, date, datetime, or other).<br>
&nbsp;&nbsp;**sample**: ***int***<br>
&nbsp;&nbsp;&nbsp;&nbsp;Number of rows to generate.<br>
&nbsp;&nbsp;**rng**: ***numpy.random.Generator***<br>
&nbsp;&nbsp;&nbsp;&nbsp;Random number generator instance.<br>
&nbsp;&nbsp;**coded_domains**: ***Mapping[str, list[float]]***<br>
&nbsp;&nbsp;&nbsp;&nbsp;Per-field coded domains loaded from metadata.

**Returns**

&nbsp;&nbsp;**out**: ***pandas.Series***<br>
&nbsp;&nbsp;&nbsp;&nbsp;Simulated column values. Unknown dtypes return an all-NaN series.

## _apply_missingness()

```python
phenofhy.simulate._apply_missingness(col, *, field, missing_rate, rng)
```

Apply field-level missingness to a simulated column.

**Parameters**

&nbsp;&nbsp;**col**: ***pandas.Series***<br>
&nbsp;&nbsp;&nbsp;&nbsp;Input series to mask with missing values.<br>
&nbsp;&nbsp;**field**: ***str***<br>
&nbsp;&nbsp;&nbsp;&nbsp;Full field name used to resolve missingness probability.<br>
&nbsp;&nbsp;**missing_rate**: ***float | Mapping[str, float] | None***<br>
&nbsp;&nbsp;&nbsp;&nbsp;Global or per-field missingness specification.<br>
&nbsp;&nbsp;**rng**: ***numpy.random.Generator***<br>
&nbsp;&nbsp;&nbsp;&nbsp;Random number generator instance.<br>

**Returns**

&nbsp;&nbsp;**out**: ***pandas.Series***<br>
&nbsp;&nbsp;&nbsp;&nbsp;Series with missing entries injected (NaT for datetime-like, NaN otherwise).

**Raises**

&nbsp;&nbsp;**ValueError**: ***Exception***<br>
&nbsp;&nbsp;&nbsp;&nbsp;If resolved missing probability is greater than 1.

## _resolve_missing_rate()

```python
phenofhy.simulate._resolve_missing_rate(field, missing_rate)
```

Resolve missingness probability for one field from user input and defaults.

**Parameters**

&nbsp;&nbsp;**field**: ***str***<br>
&nbsp;&nbsp;&nbsp;&nbsp;Full field name (entity.field).<br>
&nbsp;&nbsp;**missing_rate**: ***float | Mapping[str, float] | None***<br>
&nbsp;&nbsp;&nbsp;&nbsp;Global value, per-field mapping, or None.

**Returns**

&nbsp;&nbsp;**out**: ***float***<br>
&nbsp;&nbsp;&nbsp;&nbsp;Resolved missingness probability in [0, 1].

**Raises**

&nbsp;&nbsp;**ValueError**: ***Exception***<br>
&nbsp;&nbsp;&nbsp;&nbsp;If resolved value is outside [0, 1].

## _random_token()

```python
phenofhy.simulate._random_token(rng, size)
```

Generate an uppercase alphanumeric random token.

**Parameters**

&nbsp;&nbsp;**rng**: ***numpy.random.Generator***<br>
&nbsp;&nbsp;&nbsp;&nbsp;Random number generator instance.<br>
&nbsp;&nbsp;**size**: ***int***<br>
&nbsp;&nbsp;&nbsp;&nbsp;Token length.

**Returns**

&nbsp;&nbsp;**out**: ***str***<br>
&nbsp;&nbsp;&nbsp;&nbsp;Random token using A-Z and 0-9.
