# Rules and data cleaning

Phenofhy uses rules in two related ways:

- **Coalescing rules** combine multiple versions of a trait into one derived
	column, such as combining version 1 and version 2 smoking variables.
- **Known-error filters** remove sentinel values and implausible measurements
	before derivations are applied.

The rule builders and default rule definitions live in the private
`phenofhy._rules` module. Most users should pass custom rules through the
public functions in `phenofhy.process` rather than editing package source.

## Coalescing rules

Coalescing combines source columns in a defined order. A rule specifies the
source columns, the trait type, which values are informative, and which source
takes priority.

### Categorical rules

Use `rule_cat()` for categorical values:

```python
from phenofhy import _rules

smoking_rule = _rules.rule_cat(
		[
				"questionnaire.smoke_status_1_1",
				"questionnaire.smoke_status_2_1",
		],
		informative={"Current", "Former", "Never"},
		collapse=False,
		priority="last",
)
```

Important options include:

- `informative`: values that count as substantive responses;
- `nonresponse`: values to treat as non-informative;
- `collapse`: whether non-informative values become `Unknown`;
- `priority`: `"first"` or `"last"` source preference;
- `value_map`: optional mapping to canonical labels;
- `preserve_dtype`: whether to preserve or extend categorical dtype metadata.

### Numeric rules

Use `rule_num()` for numeric values and optional validity predicates:

```python
age_rule = _rules.rule_num(
		[
				"questionnaire.smoke_reg_first_age_1_1",
				"questionnaire.smoke_reg_first_age_2_1",
		],
		informative=lambda value: value is not None and 5 <= float(value) <= 100,
		astype="Int64",
		priority="last",
)
```

The `informative` predicate should return `True` for values that may be used
in the unified output. Invalid values are ignored when the sources are
coalesced.

## Applying custom rules

Pass a mapping of output column names to rule definitions through
`coalesce_rules`:

```python
from phenofhy import process

rules = {
		"derived.smoking_status_custom": smoking_rule,
		"derived.smoking_age_custom": age_rule,
}

processed = process.questionnaire_fields(
		df,
		derive="auto",
		coalesce_rules=rules,
)
```

Passing `coalesce_rules` replaces the default coalescing rule mapping for that
processing call. To retain the defaults and add or override rules, use
`build_rules()`:

```python
rules = _rules.build_rules(
		overrides={"derived.smoking_status_custom": smoking_rule},
		extend={"derived.smoking_age_custom": age_rule},
)
```

`overrides` replaces an existing key. `extend` adds new keys and raises
`KeyError` if a key already exists, helping prevent accidental replacement.

Apply rules directly when needed:

```python
processed = _rules.coalesce_traits(df, rules)
```

## Default coalescing rules

`DEFAULT_COALESCE_RULES` currently includes:

- `derived.smoke_status`, combining `derived.smoke_status_v1` and
	`derived.smoke_status_v2` using the substantive labels `Current`, `Former`,
	and `Never`;
- `derived.smoke_reg_first_age`, combining questionnaire versions and keeping
	plausible ages from 5 through 100.

Inspect or extend the defaults without modifying the global dictionary:

```python
from phenofhy import _rules

rules = _rules.build_rules()
print(rules.keys())
```

## Derived codebooks and display defaults

`DERIVED_CODEBOOK` maps numeric derived values to presentation labels. For
example, `derived.sex` maps codes to `Male`, `Female`, `Intersex`, `Other`,
and `Prefer not to answer`.

The same module contains default category ordering and excluded categories
used by display/reporting helpers, along with medication group mappings used
when medication indicators are expanded.

## Known-error filters

Processing calls `remove_known_errors()` before running derivations. It:

- removes `-999` birth year and birth month sentinel values when those columns
	are present;
- filters clinic height, weight, and waist values to default plausible ranges;
- leaves missing values unchanged;
- skips a filter when its relevant columns are absent.

Use it directly when you want these checks outside the processing pipeline:

```python
from phenofhy import _filter_funcs

cleaned = _filter_funcs.remove_known_errors(
		df,
		clinic_ranges={"clinic_measurements.weight": (30, 250)},
)
```

For general range and expression filters, use `apply_row_filters()`:

```python
filtered = _filter_funcs.apply_row_filters(
		df,
		ranges={"clinic_measurements.weight": (30, 250)},
		exprs=["derived.age_at_registration >= 18"],
		keep_na=True,
)
```

Missing range columns raise `KeyError` by default. Set
`ignore_missing_range_cols=True` when a range should be applied only if its
column is available.
