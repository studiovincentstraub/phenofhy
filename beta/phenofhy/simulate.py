from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
import pandas as pd


DEFAULT_FIELDS: list[str] = [
    "participant.pid",
    "participant.birth_year",
    "participant.birth_month",
    "participant.registration_year",
    "participant.registration_month",
    "participant.demog_sex_1_1",
    "participant.demog_sex_2_1",
    "participant.demog_ethnicity_1_1",
    "questionnaire.demog_height_1_1",
    "questionnaire.smoke_status_2_1",
    "clinic_measurements.waist",
    "clinic_measurements.height",
    "clinic_measurements.weight",
    "questionnaire.alcohol_curr_1_1",
    "clinic_measurements.heart_first_rate",
]


KNOWN_NONRESPONSE_CODES: dict[str, set[float]] = {
    "participant.demog_sex_1_1": {3.0, -3.0},
    "participant.demog_sex_2_1": {3.0, -3.0},
    "participant.demog_ethnicity_1_1": {19.0, -3.0},
    "questionnaire.housing_income_1_1": {-1.0, -3.0},
}


GENERIC_INTEGER_RANGES: dict[str, tuple[int, int]] = {
    "participant.birth_year": (1935, 2007),
    "participant.birth_month": (1, 12),
    "participant.registration_year": (2022, 2026),
    "participant.registration_month": (1, 12),
    "clinic_measurements.height": (90, 299),
    "clinic_measurements.waist": (30, 200),
    "clinic_measurements.heart_first_rate": (35, 180),
}


GENERIC_FLOAT_RANGES: dict[str, tuple[float, float]] = {
    "questionnaire.demog_height_1_1": (120.0, 220.0),
    "clinic_measurements.weight": (20.0, 400.0),
}


DEFAULT_MISSINGNESS_BY_FIELD: dict[str, float] = {
    "participant.pid": 0.0,
    "participant.birth_year": 0.0,
    "participant.birth_month": 0.0,
    "participant.registration_year": 0.0,
    "participant.registration_month": 0.0,
    "participant.demog_sex_1_1": 0.65,
    "participant.demog_sex_2_1": 0.03,
    "participant.demog_ethnicity_1_1": 0.02,
    "questionnaire.demog_height_1_1": 0.06,
    "questionnaire.smoke_status_2_1": 0.55,
    "clinic_measurements.waist": 0.10,
    "clinic_measurements.height": 0.10,
    "clinic_measurements.weight": 0.10,
    "questionnaire.alcohol_curr_1_1": 0.05,
    "clinic_measurements.heart_first_rate": 0.12,
}


def simulate_phenotype_df(
    sample: int = 1000,
    fields: str | Sequence[str] | None = None,
    *,
    include_nonresponse: bool = False,
    missing_rate: float | Mapping[str, float] | None = None,
    seed: int | None = None,
) -> pd.DataFrame:
    """Simulate an OFH-like phenotype dataframe.

    Uses the local helper dictionaries as follows:
    - ``beta/helpers/data_dictionary.csv`` defines valid fields and base dtypes.
    - ``beta/helpers/codings.csv`` defines categorical code domains for coded fields.

    Conservative fallback ranges are used when a numeric field has no coding domain:
    - Integer default range: ``0..120``
    - Float default range: ``0.0..100.0``
    Field-specific conservative overrides are used when available, e.g.:
    - ``clinic_measurements.height``: ``90..299``
    - ``clinic_measurements.weight``: ``20.0..400.0``
    - ``clinic_measurements.waist``: ``30..200``

    Args:
        sample: Number of rows to generate.
        fields: A single ``entity.field`` string, a list/tuple of them, or ``None``.
            If ``None``, uses ``DEFAULT_FIELDS``.
        include_nonresponse: Whether to include non-response/sentinel codes from
            coded fields. Defaults to ``False``.
        missing_rate: Optional global missingness probability ``[0, 1]`` or per-field
            mapping. If ``None``, defaults are used for ``DEFAULT_FIELDS`` and ``0.05``
            for other fields.
        seed: Optional RNG seed for reproducibility.

    Returns:
        Simulated dataframe with requested columns.

    Raises:
        ValueError: If sample < 1, field names are invalid, or missing_rate is invalid.
    """
    if sample < 1:
        raise ValueError("sample must be >= 1")

    requested_fields = _normalize_fields(fields)
    metadata = _data_dictionary().copy()
    valid_fields = set(metadata["full_field"].tolist())

    unknown = [field for field in requested_fields if field not in valid_fields]
    if unknown:
        raise ValueError(f"Unknown fields: {unknown}")

    metadata = metadata.set_index("full_field")
    coded_domains = _coded_domains(include_nonresponse=include_nonresponse)
    rng = np.random.default_rng(seed)

    out: dict[str, pd.Series] = {}
    for field in requested_fields:
        dtype = str(metadata.at[field, "type"]).lower()
        col = _simulate_column(field=field, dtype=dtype, sample=sample, rng=rng, coded_domains=coded_domains)
        col = _apply_missingness(col, field=field, missing_rate=missing_rate, rng=rng)
        out[field] = col

    return pd.DataFrame(out)


def _normalize_fields(fields: str | Sequence[str] | None) -> list[str]:
    if fields is None:
        return list(DEFAULT_FIELDS)
    if isinstance(fields, str):
        return [fields]
    out = [str(field) for field in fields]
    if len(out) == 0:
        raise ValueError("fields cannot be empty")
    return out


@lru_cache(maxsize=1)
def _helpers_dir() -> Path:
    return Path(__file__).resolve().parents[1] / "helpers"


@lru_cache(maxsize=1)
def _data_dictionary() -> pd.DataFrame:
    path = _helpers_dir() / "data_dictionary.csv"
    df = pd.read_csv(path)
    df = df[["entity", "field", "type"]].copy()
    df["full_field"] = df["entity"].astype(str) + "." + df["field"].astype(str)
    return df


@lru_cache(maxsize=2)
def _coded_domains(include_nonresponse: bool) -> dict[str, list[float]]:
    path = _helpers_dir() / "codings.csv"
    df = pd.read_csv(path)
    if "coding_name" in df.columns:
        field_col = "coding_name"
    elif "field" in df.columns:
        field_col = "field"
    else:
        raise ValueError("codings.csv must contain either 'coding_name' or 'field' column")

    df = df[["entity", field_col, "code"]].copy()
    df["full_field"] = df["entity"].astype(str) + "." + df[field_col].astype(str)
    df["code"] = pd.to_numeric(df["code"], errors="coerce")

    domains: dict[str, list[float]] = {}
    for field, subdf in df.groupby("full_field", sort=False):
        codes = [float(v) for v in subdf["code"].dropna().unique().tolist()]
        filtered_codes = _filter_nonresponse_codes(field=field, codes=codes, include_nonresponse=include_nonresponse)
        domains[field] = filtered_codes if filtered_codes else codes
    return domains


def _filter_nonresponse_codes(field: str, codes: Sequence[float], include_nonresponse: bool) -> list[float]:
    if include_nonresponse:
        return list(codes)

    blocked = set(KNOWN_NONRESPONSE_CODES.get(field, set()))
    return [code for code in codes if code >= 0 and code not in blocked]


def _simulate_column(
    *,
    field: str,
    dtype: str,
    sample: int,
    rng: np.random.Generator,
    coded_domains: Mapping[str, list[float]],
) -> pd.Series:
    if field in coded_domains and len(coded_domains[field]) > 0:
        values = rng.choice(np.array(coded_domains[field], dtype=float), size=sample, replace=True)
        if dtype == "integer":
            values = np.rint(values)
        return pd.Series(values)

    if dtype == "integer":
        if field == "clinic_measurements.height":
            values = np.rint(np.clip(rng.normal(170.0, 10.0, size=sample), 90, 299))
            return pd.Series(values)
        if field == "clinic_measurements.waist":
            values = np.rint(np.clip(rng.normal(92.0, 14.0, size=sample), 30, 200))
            return pd.Series(values)
        if field == "clinic_measurements.heart_first_rate":
            values = np.rint(np.clip(rng.normal(72.0, 12.0, size=sample), 35, 180))
            return pd.Series(values)

        low, high = GENERIC_INTEGER_RANGES.get(field, (0, 120))
        return pd.Series(rng.integers(low, high + 1, size=sample))

    if dtype == "float":
        if field == "questionnaire.demog_height_1_1":
            values = np.clip(rng.normal(170.0, 11.0, size=sample), 120.0, 220.0)
            return pd.Series(np.round(values, 2))
        if field == "clinic_measurements.weight":
            values = np.clip(rng.normal(80.0, 18.0, size=sample), 20.0, 400.0)
            return pd.Series(np.round(values, 1))

        low, high = GENERIC_FLOAT_RANGES.get(field, (0.0, 100.0))
        values = rng.uniform(low, high, size=sample)
        return pd.Series(np.round(values, 2))

    if dtype == "string":
        if field.endswith(".pid"):
            return pd.Series([_random_token(rng, 7) for _ in range(sample)], dtype="object")
        if field.endswith(".id"):
            return pd.Series([_random_token(rng, 10) for _ in range(sample)], dtype="object")
        return pd.Series([_random_token(rng, 8) for _ in range(sample)], dtype="object")

    if dtype == "date":
        start = np.datetime64("2022-01-01")
        end = np.datetime64("2026-12-31")
        span_days = int((end - start) / np.timedelta64(1, "D"))
        offsets = rng.integers(0, span_days + 1, size=sample)
        return pd.Series((start + offsets.astype("timedelta64[D]")).astype("datetime64[ns]"))

    if dtype == "datetime":
        start = np.datetime64("2022-01-01T08:00:00")
        end = np.datetime64("2026-12-31T20:00:00")
        span_minutes = int((end - start) / np.timedelta64(1, "m"))
        offsets = rng.integers(0, span_minutes + 1, size=sample)
        return pd.Series((start + offsets.astype("timedelta64[m]")).astype("datetime64[ns]"))

    return pd.Series([np.nan] * sample)


def _apply_missingness(
    col: pd.Series,
    *,
    field: str,
    missing_rate: float | Mapping[str, float] | None,
    rng: np.random.Generator,
) -> pd.Series:
    miss_p = _resolve_missing_rate(field=field, missing_rate=missing_rate)
    if miss_p <= 0:
        return col
    if miss_p > 1:
        raise ValueError("missing_rate must be in [0, 1]")

    mask = rng.random(len(col)) < miss_p
    out = col.copy()

    if pd.api.types.is_datetime64_any_dtype(out):
        out.loc[mask] = pd.NaT
        return out

    out = out.astype("object")
    out.loc[mask] = np.nan
    return out


def _resolve_missing_rate(field: str, missing_rate: float | Mapping[str, float] | None) -> float:
    if missing_rate is None:
        return DEFAULT_MISSINGNESS_BY_FIELD.get(field, 0.05)

    if isinstance(missing_rate, Mapping):
        value = float(missing_rate.get(field, DEFAULT_MISSINGNESS_BY_FIELD.get(field, 0.05)))
    else:
        value = float(missing_rate)

    if value < 0 or value > 1:
        raise ValueError("missing_rate values must be in [0, 1]")
    return value


def _random_token(rng: np.random.Generator, size: int) -> str:
    alphabet = np.array(list("ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"))
    return "".join(rng.choice(alphabet, size=size, replace=True).tolist())
