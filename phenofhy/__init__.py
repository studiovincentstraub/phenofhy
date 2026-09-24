"""Phenofhy: A Python package for phenotyping and health data analysis."""

__version__ = "1.0.0"

from . import (
    _derive_funcs,
    _filter_funcs,
    _rules,
    utils,
    simulate,
    load,
    extract,
    icd,
    process,
    calculate,
    display,
    profile,
    pipeline,
)

from .config import init