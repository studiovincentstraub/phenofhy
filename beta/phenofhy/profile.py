"""
phenofhy.profile - MVP skeleton

This module implements an MVP API for generating a "pre-GWAS" phenotype profile.
It provides a single public function `phenotype_profile(...)` which is intentionally
small and returns both computed summary tables and matplotlib figures.

The implementation below is a focused skeleton that:
- validates inputs
- computes basic demographic summaries and a minimal sample flow
- computes age-binned means for a phenotype
- provides stub plotting functions (matplotlib)

TODO: incrementally implement plotting polish, PDF rendering, interactive Plotly mode,
and more complex diagnostics.

"""
from __future__ import annotations

import logging
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

logger = logging.getLogger(__name__)


# Public API ---------------------------------------------------------------

def phenotype_profile(
    df: pd.DataFrame,
    phenotype: str,
    *,
    sex_col: str = "derived.sex",
    age_col: str = "derived.age_at_registration",
    bmi_col: Optional[str] = "derived.bmi",
    height_col: Optional[str] = "clinic_measurements.height",
    ethnicity_col: Optional[str] = None,

    age_bin_width: int = 2,
    body_bin_method: str = "quantile",   # "quantile" | "fixed"
    body_bins: int = 10,

    output: Optional[str] = None,          # e.g. "phenotype_profile.pdf"
    mode: str = "pdf",                   # "pdf" | "interactive"
    title: Optional[str] = None,

    min_n_per_bin: int = 30,
    drop_missing: bool = True,
) -> Dict[str, Any]:
    """
    Generate a phenotype profile report for a single phenotype.

    Parameters
    ----------
    df
        Pre-processed pandas DataFrame (output of pipeline or process.* functions).
    phenotype
        Column name in `df` corresponding to the phenotype to profile.
    sex_col, age_col, bmi_col, height_col, ethnicity_col
        Column names for covariates. Columns can be None if not available.
    age_bin_width
        Width (in years) of age bins used for trend plots.
    body_bin_method
        How to bin BMI/height: 'quantile' or 'fixed'.
    body_bins
        Number of bins when using quantile binning (or fixed bins when method=='fixed').
    output
        If provided, write a PDF to this path (not yet implemented in MVP).
    mode
        'pdf' or 'interactive' (MVP implements 'pdf' via matplotlib canvases, 'interactive' is a stub).
    title
        Optional title for the report.
    min_n_per_bin
        Minimum sample size in a bin to include it in trend summaries.
    drop_missing
        Whether to drop rows with missing phenotype before calculations.

    Returns
    -------
    A dictionary containing summary dataframes and matplotlib figures.
    """

    # Validate inputs ----------------------------------------------------
    if phenotype not in df.columns:
        raise KeyError(f"Phenotype column '{phenotype}' not found in DataFrame columns: {list(df.columns)}")

    if mode not in ("pdf", "interactive"):
        raise ValueError("mode must be 'pdf' or 'interactive'")

    # Work on a copy
    working = df.copy()

    # Optionally drop missing phenotype
    if drop_missing:
        before = len(working)
        working = working[working[phenotype].notna()]
        logger.debug("Dropped %d rows with missing phenotype", before - len(working))

    # Infer phenotype type
    ptype = _infer_phenotype_type(working[phenotype])
    logger.info("Inferred phenotype '%s' as type '%s'", phenotype, ptype)

    # Compute basic summaries
    sample_flow = _compute_sample_flow(df, working, phenotype)
    demographics = _summarize_demographics(working, sex_col=sex_col, age_col=age_col, ethnicity_col=ethnicity_col)

    # Age binned means
    age_trends = None
    age_fig = None
    if age_col in working.columns:
        age_trends = _compute_age_bin_means(
            working, phenotype, age_col=age_col, sex_col=sex_col, bin_width=age_bin_width, min_n=min_n_per_bin
        )
        age_fig = _plot_age_trend(age_trends, phenotype, age_col)

    # BMI and height trends
    bmi_trends = None
    bmi_fig = None
    if bmi_col and bmi_col in working.columns:
        bmi_trends = _compute_body_bin_means(
            working, phenotype, body_col=bmi_col, sex_col=sex_col, method=body_bin_method, n_bins=body_bins, min_n=min_n_per_bin
        )
        bmi_fig = _plot_body_trend(bmi_trends, phenotype, bmi_col)

    height_trends = None
    height_fig = None
    if height_col and height_col in working.columns:
        height_trends = _compute_body_bin_means(
            working, phenotype, body_col=height_col, sex_col=sex_col, method=body_bin_method, n_bins=body_bins, min_n=min_n_per_bin
        )
        height_fig = _plot_body_trend(height_trends, phenotype, height_col)

    # Distribution plots
    dist_fig = _plot_distribution(working, phenotype, sex_col=sex_col, ptype=ptype)

    result = {
        "phenotype": phenotype,
        "type": ptype,
        "sample_flow": sample_flow,
        "demographics": demographics,
        "age_trends": age_trends,
        "bmi_trends": bmi_trends,
        "height_trends": height_trends,
        "figures": {
            "distribution": dist_fig,
            "age_trend": age_fig,
            "bmi_trend": bmi_fig,
            "height_trend": height_fig,
        },
    }

    # Render first (overview) panel if requested
    overview_fig = _plot_overview_panel(
        phenotype=phenotype,
        ptype=ptype,
        sample_flow=sample_flow,
        demographics=demographics,
        title=title,
    )

    result["figures"]["overview"] = overview_fig

    # --- PDF export (MVP) ---
    if output and mode == "pdf":
        from matplotlib.backends.backend_pdf import PdfPages
        with PdfPages(output) as pdf:
            # Enforce consistent ordering
            for key in ["overview", "distribution", "age_trend", "bmi_trend", "height_trend"]:
                fig = result.get("figures", {}).get(key)
                if fig is not None:
                    pdf.savefig(fig, bbox_inches="tight")
        logger.info("Phenotype profile PDF written to %s", output)

    return result


# ------------------------- helper implementations ------------------------

def _infer_phenotype_type(series: pd.Series) -> str:
    """Infer whether a phenotype is continuous, binary, or categorical.

    Simple rules for MVP:
    - continuous: dtype is float, many unique values (>50)
    - binary: two unique values (excluding NA)
    - categorical: otherwise
    """
    s = series.dropna()
    nunique = s.nunique()
    if pd.api.types.is_float_dtype(s.dtype) or (pd.api.types.is_integer_dtype(s.dtype) and nunique > 50):
        return "continuous"
    if nunique == 2:
        return "binary"
    return "categorical"


def _compute_sample_flow(original: pd.DataFrame, working: pd.DataFrame, phenotype: str) -> pd.DataFrame:
    """Return a small DataFrame with counts before and after phenotype filtering."""
    rows_orig = len(original)
    rows_after = len(working)
    missing = rows_orig - rows_after
    df = pd.DataFrame([
        {"step": "initial", "n": rows_orig},
        {"step": f"non-missing {phenotype}", "n": rows_after},
        {"step": f"missing {phenotype}", "n": missing},
    ])
    return df


def _summarize_demographics(df: pd.DataFrame, sex_col: str, age_col: str, ethnicity_col: Optional[str]) -> pd.DataFrame:
    """Create a small demographics summary table: counts and basic stats.

    Returns a tidy DataFrame with rows for sex, age (mean, sd, median, min, max), and ethnicity counts
    if available.
    """
    rows = []

    # Sex counts
    if sex_col in df.columns:
        sex_counts = df[sex_col].value_counts(dropna=False)
        for k, v in sex_counts.items():
            rows.append({"variable": "sex", "level": str(k), "n": int(v)})

    # Age stats
    if age_col in df.columns:
        a = df[age_col].dropna()
        rows.append({"variable": "age.mean", "level": "", "n": float(a.mean())})
        rows.append({"variable": "age.sd", "level": "", "n": float(a.std())})
        rows.append({"variable": "age.median", "level": "", "n": float(a.median())})
        rows.append({"variable": "age.min", "level": "", "n": float(a.min())})
        rows.append({"variable": "age.max", "level": "", "n": float(a.max())})

    # Ethnicity counts
    if ethnicity_col and ethnicity_col in df.columns:
        eth_counts = df[ethnicity_col].value_counts(dropna=False)
        for k, v in eth_counts.items():
            rows.append({"variable": "ethnicity", "level": str(k), "n": int(v)})

    return pd.DataFrame(rows)


def _compute_age_bin_means(
    df: pd.DataFrame, phenotype: str, age_col: str, sex_col: str, bin_width: int = 2, min_n: int = 30
) -> pd.DataFrame:
    """Bin age in fixed-width bins and compute mean phenotype per bin and sex.

    Returns a DataFrame with columns ['age_bin', 'age_bin_left', 'sex', 'n', 'mean', 'se']
    """
    # drop missing age or phenotype
    tmp = df[[age_col, phenotype, sex_col]].dropna()
    # create bins
    age_min = int(np.floor(tmp[age_col].min()))
    age_max = int(np.ceil(tmp[age_col].max()))
    bins = list(range(age_min, age_max + bin_width, bin_width))
    labels = [f"{b}-{b + bin_width - 1}" for b in bins[:-1]]
    tmp = tmp.assign(age_bin=pd.cut(tmp[age_col], bins=bins, labels=labels, right=False))

    out_rows = []
    grouped = tmp.groupby(["age_bin", sex_col])
    for (age_bin, sex), g in grouped:
        n = len(g)
        if n < min_n:
            continue
        mean = g[phenotype].mean()
        se = g[phenotype].std() / np.sqrt(n) if n > 1 else np.nan
        left = int(str(age_bin).split("-")[0]) if pd.notna(age_bin) else np.nan
        out_rows.append({"age_bin": age_bin, "age_bin_left": left, "sex": sex, "n": n, "mean": mean, "se": se})

    return pd.DataFrame(out_rows)


def _compute_body_bin_means(
    df: pd.DataFrame,
    phenotype: str,
    body_col: str,
    sex_col: str,
    method: str = "quantile",
    n_bins: int = 10,
    min_n: int = 30,
) -> pd.DataFrame:
    """Compute mean phenotype per body_measurement bin (BMI or height).

    For method=='quantile' we use pd.qcut; for 'fixed' we use pd.cut with equal-width bins.
    """
    tmp = df[[body_col, phenotype, sex_col]].dropna()
    if method == "quantile":
        try:
            tmp = tmp.assign(bin=pd.qcut(tmp[body_col], q=n_bins, duplicates="drop"))
        except Exception:
            tmp = tmp.assign(bin=pd.cut(tmp[body_col], bins=n_bins))
    else:
        tmp = tmp.assign(bin=pd.cut(tmp[body_col], bins=n_bins))

    out_rows = []
    grouped = tmp.groupby(["bin", sex_col])
    for (bin_, sex), g in grouped:
        n = len(g)
        if n < min_n:
            continue
        mean = g[phenotype].mean()
        se = g[phenotype].std() / np.sqrt(n) if n > 1 else np.nan
        left = g[body_col].min()
        right = g[body_col].max()
        out_rows.append({"bin": bin_, "bin_left": left, "bin_right": right, "sex": sex, "n": n, "mean": mean, "se": se})
    return pd.DataFrame(out_rows)


# ------------------------------ plotting --------------------------------


def _plot_age_trend_panel(
    df: pd.DataFrame,
    phenotype: str,
    *,
    age_col: str,
    sex_col: Optional[str] = None,
    bin_width: int = 2,
    min_n_per_bin: int = 30,
    ptype: str = "continuous",
    title: Optional[str] = None,
):
    """
    Panel 3: Phenotype vs age

    - Ages are binned into fixed-width bins (default: 2 years)
    - Plots mean phenotype per age bin
    - Stratifies by sex if available

    Returns a matplotlib.figure.Figure
    """
    fig, ax = plt.subplots(figsize=(11, 4))

    if age_col not in df.columns:
        ax.text(0.5, 0.5, "Age column not available", ha="center", va="center")
        ax.set_axis_off()
        return fig

    work = df[[phenotype, age_col] + ([sex_col] if sex_col and sex_col in df.columns else [])].dropna(subset=[phenotype, age_col])

    if work.empty:
        ax.text(0.5, 0.5, "No non-missing data for age trend", ha="center", va="center")
        ax.set_axis_off()
        return fig

    # Define age bins
    age_min = int(np.floor(work[age_col].min()))
    age_max = int(np.ceil(work[age_col].max()))
    bins = np.arange(age_min, age_max + bin_width, bin_width)

    work = work.copy()
    work["_age_bin"] = pd.cut(work[age_col], bins=bins, right=False)

    group_cols = ["_age_bin"]
    if sex_col and sex_col in work.columns:
        group_cols.append(sex_col)

    grouped = (
        work
        .groupby(group_cols)
        .agg(
            mean_val=(phenotype, "mean"),
            n=(phenotype, "count"),
        )
        .reset_index()
    )

    # Drop small bins
    grouped = grouped[grouped["n"] >= min_n_per_bin]

    if grouped.empty:
        ax.text(0.5, 0.5, "No age bins with sufficient sample size", ha="center", va="center")
        ax.set_axis_off()
        return fig

    # Compute bin midpoints for plotting
    grouped["age_mid"] = grouped["_age_bin"].apply(lambda x: (x.left + x.right) / 2)

    if sex_col and sex_col in grouped.columns:
        for sex, g in grouped.groupby(sex_col):
            ax.plot(g["age_mid"], g["mean_val"], marker="o", label=str(sex))
        ax.legend(title="Sex")
    else:
        ax.plot(grouped["age_mid"], grouped["mean_val"], marker="o")

    ax.set_xlabel("Age (years)")
    ax.set_ylabel(f"Mean {phenotype}")
    ax.set_title("Phenotype vs age")

    if title:
        ax.set_title(title)

    plt.tight_layout()
    return fig





def _plot_distribution_panel(
    df: pd.DataFrame,
    phenotype: str,
    *,
    sex_col: Optional[str] = None,
    ptype: str = "continuous",
    title: Optional[str] = None,
):
    """
    Panel 2: Phenotype distribution

    Layout:
      - Left: overall distribution (hist or bar)
      - Right: sex-stratified distribution (overlay)

    Returns a matplotlib.figure.Figure
    """
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))

    # --- Overall distribution ---
    ax0 = axes[0]
    s = df[phenotype].dropna()

    if ptype == "continuous":
        ax0.hist(s, bins=40)
        ax0.set_xlabel(phenotype)
        ax0.set_ylabel("Count")
    else:
        vc = s.value_counts()
        ax0.bar(vc.index.astype(str), vc.values)
        ax0.set_xticklabels(vc.index.astype(str), rotation=45, ha="right")
        ax0.set_ylabel("Count")

    ax0.set_title("Overall distribution")

    # --- Sex-stratified distribution ---
    ax1 = axes[1]
    if sex_col and sex_col in df.columns:
        for sex, g in df.groupby(sex_col):
            sg = g[phenotype].dropna()
            if sg.empty:
                continue
            if ptype == "continuous":
                ax1.hist(sg, bins=30, alpha=0.5, label=str(sex))
            else:
                vc = sg.value_counts()
                ax1.bar(vc.index.astype(str), vc.values, alpha=0.5, label=str(sex))
        ax1.legend(title="Sex")
        ax1.set_title("By sex")
    else:
        ax1.text(0.5, 0.5, "Sex column not available", ha="center")
        ax1.set_axis_off()

    if title:
        fig.suptitle(title, fontsize=13)

    plt.tight_layout()
    return fig





def _plot_overview_panel(
    *,
    phenotype: str,
    ptype: str,
    sample_flow: pd.DataFrame,
    demographics: pd.DataFrame,
    title: Optional[str] = None,
):
    """
    First-page overview panel combining:
      - phenotype metadata
      - sample size flow
      - basic demographic summaries

    Returns a matplotlib.figure.Figure
    """
    fig = plt.figure(figsize=(11, 8.5))
    gs = fig.add_gridspec(2, 2, height_ratios=[1, 1.3])

    # --- Panel A: phenotype metadata ---
    ax_meta = fig.add_subplot(gs[0, 0])
    ax_meta.axis("off")

    meta_lines = [
        f"Phenotype: {phenotype}",
        f"Type: {ptype}",
    ]

    for i, line in enumerate(meta_lines):
        ax_meta.text(0.0, 1.0 - i * 0.15, line, fontsize=11, va="top")

    ax_meta.set_title("Phenotype overview", loc="left", fontsize=12, fontweight="bold")

    # --- Panel B: sample flow ---
    ax_flow = fig.add_subplot(gs[0, 1])
    ax_flow.set_title("Sample size flow", loc="left", fontsize=12, fontweight="bold")

    y_pos = np.arange(len(sample_flow))
    ax_flow.barh(y_pos, sample_flow["n"])
    ax_flow.set_yticks(y_pos)
    ax_flow.set_yticklabels(sample_flow["step"])
    ax_flow.invert_yaxis()
    ax_flow.set_xlabel("N")

    for i, v in enumerate(sample_flow["n"]):
        ax_flow.text(v, i, f" {v}", va="center")

    # --- Panel C: demographics table ---
    ax_demo = fig.add_subplot(gs[1, :])
    ax_demo.axis("off")
    ax_demo.set_title("Demographic summary (phenotype non-missing)", loc="left", fontsize=12, fontweight="bold")

    if demographics.empty:
        ax_demo.text(0.5, 0.5, "No demographic information available", ha="center")
    else:
        table_df = demographics.copy()
        table_df = table_df.rename(columns={"variable": "Variable", "level": "Level", "n": "Value"})
        table = ax_demo.table(
            cellText=table_df.values,
            colLabels=table_df.columns,
            loc="upper left",
            cellLoc="left",
        )
        table.auto_set_font_size(False)
        table.set_fontsize(9)
        table.scale(1, 1.3)

    if title:
        fig.suptitle(title, fontsize=14, y=0.98)

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    return fig




def _plot_distribution(df: pd.DataFrame, phenotype: str, sex_col: str, ptype: str):
    """Plot main distribution and sex-stratified histograms for the phenotype.

    Returns a matplotlib.figure.Figure
    """
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    ax = axes[0]
    data = df[phenotype].dropna()
    if ptype == "continuous":
        ax.hist(data, bins=40)
        ax.set_title(f"Distribution of {phenotype}")
        ax.set_xlabel(phenotype)
    else:
        vc = data.value_counts()
        ax.bar(vc.index.astype(str), vc.values)
        ax.set_title(f"Counts of {phenotype}")
        ax.set_xticklabels(vc.index.astype(str), rotation=45, ha="right")

    # sex stratified
    ax2 = axes[1]
    if sex_col in df.columns:
        for sex, g in df.groupby(sex_col):
            ax2.hist(g[phenotype].dropna(), bins=30, alpha=0.5, label=str(sex))
        ax2.legend()
        ax2.set_title(f"{phenotype} by {sex_col}")
    else:
        ax2.text(0.5, 0.5, "sex column not available", ha="center")
        ax2.set_axis_off()

    plt.tight_layout()
    return fig


def _plot_age_trend(age_df: pd.DataFrame, phenotype: str, age_col: str):
    fig, ax = plt.subplots(figsize=(6, 4))
    if age_df is None or age_df.empty:
        ax.text(0.5, 0.5, "No age trend data available", ha="center")
    else:
        for sex, g in age_df.groupby("sex"):
            g = g.sort_values("age_bin_left")
            ax.errorbar(g["age_bin_left"], g["mean"], yerr=g["se"], label=str(sex), marker="o")
        ax.set_xlabel(age_col)
        ax.set_ylabel(f"mean({phenotype})")
        ax.legend()
        ax.set_title(f"{phenotype} mean by {age_col} (binned)")
    plt.tight_layout()
    return fig


def _plot_body_trend(body_df: pd.DataFrame, phenotype: str, body_col: str):
    fig, ax = plt.subplots(figsize=(6, 4))
    if body_df is None or body_df.empty:
        ax.text(0.5, 0.5, "No body trend data available", ha="center")
    else:
        # use bin_left as x-axis when available
        if "bin_left" in body_df.columns:
            for sex, g in body_df.groupby("sex"):
                g = g.sort_values("bin_left")
                ax.errorbar(g["bin_left"], g["mean"], yerr=g["se"], label=str(sex), marker="o")
            ax.set_xlabel(body_col)
        else:
            for sex, g in body_df.groupby("sex"):
                ax.errorbar(range(len(g)), g["mean"], yerr=g["se"], label=str(sex), marker="o")
            ax.set_xlabel("bin index")
        ax.set_ylabel(f"mean({phenotype})")
        ax.legend()
        ax.set_title(f"{phenotype} mean by {body_col} (binned)")
    plt.tight_layout()
    return fig


# ------------------------------ module test --------------------------------
if __name__ == "__main__":
    # Small smoke test
    pdf = pd.DataFrame({
        "derived.sex": np.random.choice(["M", "F"], size=1000),
        "derived.age_at_registration": np.random.randint(40, 85, size=1000),
        "derived.bmi": np.random.normal(27, 4, size=1000),
        "clinic_measurements.height": np.random.normal(170, 10, size=1000),
    })
    # make a synthetic phenotype with age trend
    pdf["derived.pheno"] = 0.1 * pdf["derived.age_at_registration"] + np.random.normal(0, 3, size=1000)

    res = phenotype_profile(pdf, "derived.pheno", age_bin_width=2)
    print(res.keys())