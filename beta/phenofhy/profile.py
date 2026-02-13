# phenofhy/profile.py

from __future__ import annotations
import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

logger = logging.getLogger(__name__)

TITLE_FONTSIZE = 12
TEXT_FONTSIZE = 10
TABLE_FONTSIZE = 10

COLOR_MALE = "#2563eb"
COLOR_FEMALE = "#f97316"
COLOR_ACCENT = "#0f172a"
COLOR_NEUTRAL = "#64748b"
COLOR_BAND_1 = "#f8fafc"
COLOR_BAND_2 = "#f1f5f9"
GRID_COLOR = "#e2e8f0"


# ============================================================
# PUBLIC API
# ============================================================

def phenotype_profile(
    df: pd.DataFrame,
    phenotype: str,
    *,
    sex_col: str = "derived.sex",
    age_col: str = "derived.age_at_registration",
    bmi_col: Optional[str] = "derived.bmi",
    height_col: Optional[str] = "clinic_measurements.height",
    metadata_dir: str = "./metadata",
    age_bin_width: int = 1,
    body_bin_width: int = 1,
    min_n_per_bin: int = 1,
    output: Optional[str] = None,
) -> Tuple[plt.Figure, Dict[str, Any]]:

    if phenotype not in df.columns:
        raise KeyError(f"{phenotype} not found in DataFrame")

    working = df.copy()
    # create standardized sex label column (may be None if sex_col absent)
    working["__sex_label"] = _standardize_sex(working[sex_col]) if sex_col in working.columns else pd.Series([pd.NA]*len(working), index=working.index)

    field_label = phenotype.split(".")[-1]
    ptype = _infer_phenotype_type(working[phenotype])

    sample_flow = _compute_sample_flow(df, working, phenotype)
    demographics = _summarize_demographics(working, sex_col=sex_col, age_col=age_col, ethnicity_col=None)

    # Multi-panel layout (4x2 grid; bottom row metadata spans both columns)
    fig = plt.figure(figsize=(13.5, 18))
    gs = fig.add_gridspec(
        5, 2,
        height_ratios=[1.1, 1.1, 1.1, 0.15, 0.3],
        hspace=0.55,
        wspace=0.3
    )

    ax_overview = fig.add_subplot(gs[4, :])  # metadata spans full width (now row 4)
    ax_demo = fig.add_subplot(gs[0, 0])
    ax_complete = fig.add_subplot(gs[0, 1])
    ax_dist = fig.add_subplot(gs[1, 0])
    ax_age = fig.add_subplot(gs[1, 1])
    ax_bmi = fig.add_subplot(gs[2, 0])
    ax_height = fig.add_subplot(gs[2, 1])
    # row 3 is the spacer (no axes)

    for ax in [ax_overview, ax_demo, ax_complete, ax_dist, ax_age, ax_bmi, ax_height]:
        ax.set_facecolor(COLOR_BAND_1)

    # draw panels
    _plot_overview(
        ax_overview,
        phenotype=phenotype,
        ptype=ptype,
        sample_flow=sample_flow,
        demographics=demographics,
        metadata_dir=metadata_dir,
    )
    _plot_demographics(ax_demo, demographics)
    _plot_completeness(ax_complete, sample_flow)
    _plot_distribution(ax_dist, working, phenotype, field_label, ptype, sex_label_col="__sex_label")
    _plot_age_trend(ax_age, working, phenotype, field_label, age_col, min_n_per_bin, age_bin_width)
    _plot_body_trend(ax_bmi, working, phenotype, field_label, bmi_col, min_n_per_bin, body_bin_width)
    _plot_body_trend(ax_height, working, phenotype, field_label, height_col, min_n_per_bin, body_bin_width)

    if ax_bmi.lines and ax_height.lines:
        y_max = max(ax_bmi.get_ylim()[1], ax_height.get_ylim()[1])
        ax_bmi.set_ylim(top=y_max)
        ax_height.set_ylim(top=y_max)

    fig.suptitle(f"Phenotype profile: {field_label}", fontsize=16, fontweight="bold", y=0.98)
    fig.subplots_adjust(left=0.06, right=0.94, top=0.94, bottom=0.07)

    missing = sample_flow.loc[sample_flow["step"].str.startswith("missing", na=False), "n"]
    missing_n = int(missing.iloc[0]) if not missing.empty else 0
    missing_pct = (missing_n / len(df) * 100.0) if len(df) > 0 else 0.0
    footer = f"Generated {datetime.now().strftime('%Y-%m-%d %H:%M')}"
    fig.text(0.06, 0.03, footer, fontsize=TEXT_FONTSIZE - 1, color=COLOR_NEUTRAL)

    if output:
        Path(output).parent.mkdir(parents=True, exist_ok=True)
        with PdfPages(output) as pdf:
            pdf.savefig(fig, bbox_inches="tight")
        logger.info("Saved profile to %s", output)

    return fig, {
        "phenotype": phenotype,
        "type": ptype,
        "sample_flow": sample_flow,
        "demographics": demographics,
    }


# ============================================================
# HELPERS
# ============================================================

def _standardize_sex(series: Optional[pd.Series]) -> pd.Series | None:
    """Map common sex encodings to 'Male'/'Female'. Return a Series of labels or series of NA if input is None."""
    if series is None:
        return None

    def mapper(v):
        if pd.isna(v):
            return pd.NA
        try:
            iv = int(v)
            if iv == 1:
                return "Male"
            if iv == 2:
                return "Female"
        except Exception:
            pass
        s = str(v).strip().lower()
        if s in {"m", "male", "man"}:
            return "Male"
        if s in {"f", "female", "woman"}:
            return "Female"
        return pd.NA

    # preserve index
    return series.map(mapper)


def _infer_phenotype_type(series: pd.Series) -> str:
    s = series.dropna()
    if s.empty:
        return "unknown"
    if pd.api.types.is_float_dtype(s.dtype) or s.nunique() > 50:
        return "continuous"
    if s.nunique() == 2:
        return "binary"
    return "categorical"


def _compute_sample_flow(original: pd.DataFrame, working: pd.DataFrame, phenotype: str) -> pd.DataFrame:
    return pd.DataFrame([
        {"step": "initial", "n": len(original)},
        {"step": f"non-missing {phenotype}", "n": len(working)},
        {"step": f"missing {phenotype}", "n": len(original) - len(working)},
    ])


def _summarize_demographics(
    df: pd.DataFrame,
    *,
    sex_col: str,
    age_col: str,
    ethnicity_col: Optional[str],
    max_decimals: int = 3,
) -> pd.DataFrame:
    """
    Compute structured demographic summary suitable for formatted rendering.

    Returns a dictionary with:
        - N
        - sex breakdown (list of dicts)
        - age summary (dict)
        - ethnicity breakdown (list of dicts) if available
    """
    rows = []
    N = len(df)

    rows.append({
        "Var": "Total",
        "Level/Stat": "",
        "N": int(N),
        "%": round(100.0, 1) if N > 0 else 0.0,
        "Value": "",
    })

    # Sex
    if sex_col in df.columns:
        if "__sex_label" in df.columns:
            ser = df["__sex_label"].copy()
        else:
            ser = df[sex_col].copy()
        ser = ser.fillna("Missing")
        counts = ser.value_counts(dropna=False)
        for level, count in counts.items():
            pct = round(100.0 * int(count) / N, 1) if N > 0 else 0.0
            rows.append({
                "Var": "Sex",
                "Level/Stat": str(level),
                "N": int(count),
                "%": pct,
                "Value": "",
            })

    # Age (continuous)
    if age_col in df.columns:
        a = df[age_col].dropna()
        if len(a) > 0:
            rows.extend([
                {"Var": "Age", "Level/Stat": "mean", "N": "", "%": "", "Value": round(float(a.mean()), 2)},
                {"Var": "Age", "Level/Stat": "sd", "N": "", "%": "", "Value": round(float(a.std()), 2)},
                {"Var": "Age", "Level/Stat": "median", "N": "", "%": "", "Value": round(float(a.median()), 2)},
                {"Var": "Age", "Level/Stat": "min", "N": "", "%": "", "Value": round(float(a.min()), 2)},
                {"Var": "Age", "Level/Stat": "max", "N": "", "%": "", "Value": round(float(a.max()), 2)},
            ])

    # Ethnicity
    if ethnicity_col and (ethnicity_col in df.columns):
        ec = df[ethnicity_col].fillna("Missing")
        counts = ec.value_counts(dropna=False)
        for level, count in counts.items():
            pct = round(100.0 * int(count) / N, 1) if N > 0 else 0.0
            rows.append({
                "Var": "Ethnicity",
                "Level/Stat": str(level),
                "N": int(count),
                "%": pct,
                "Value": "",
            })

    return pd.DataFrame(rows, columns=["Var", "Level/Stat", "N", "%", "Value"])


# ============================================================
# PLOTTING
# ============================================================

def _plot_overview(
    ax,
    *,
    phenotype: str,
    ptype: str,
    sample_flow: pd.DataFrame,
    demographics: Dict[str, Any],
    metadata_dir: str = "./metadata",
    title: Optional[str] = None,
):
    """
    Render phenotype metadata into the supplied axes.
    Shows phenotype info and data dictionary details if available.
    """

    # Clear axis
    ax.clear()
    ax.axis("off")

    if "." in phenotype:
        entity, field = phenotype.split(".", 1)
    else:
        entity, field = "", phenotype

    # Try to include metadata from data_dictionary.csv if available
    datadict_path = None
    metadata_root = Path(metadata_dir)
    for candidate in metadata_root.glob("*data_dictionary.csv"):
        datadict_path = candidate
        break

    units_text = "Not available"
    if datadict_path and datadict_path.exists():
        try:
            dd = pd.read_csv(datadict_path)
            # try to match by suffix
            key = phenotype.split(".")[-1]
            dd_cols = {c.lower(): c for c in dd.columns}
            match = pd.DataFrame()
            for col_key in ["name", "coding_name", "field", "phenotype", "variable", "code"]:
                if col_key in dd_cols:
                    col = dd_cols[col_key]
                    match = dd.loc[dd[col].astype(str).str.lower() == key.lower()]
                    if not match.empty:
                        break
                    match = dd.loc[dd[col].astype(str).str.lower() == phenotype.lower()]
                    if not match.empty:
                        break
            if match.empty:
                for col_key in ["name", "coding_name", "field", "phenotype", "variable", "code"]:
                    if col_key in dd_cols:
                        col = dd_cols[col_key]
                        match = dd.loc[dd[col].astype(str).str.lower().str.contains(key.lower(), na=False)]
                        if not match.empty:
                            break

            if not match.empty and "units" in dd.columns:
                units_text = str(match["units"].iloc[0])
        except Exception:
            # ignore metadata parsing failures
            pass

    table_rows = [
        [f"Field: {field}", f"Type: {ptype}"],
        [f"Entity: {entity}", f"Units: {units_text}"],
    ]

    table = ax.table(
        cellText=table_rows,
        cellLoc="left",
        colLoc="left",
        loc="center",
        bbox=[0.02, 0.25, 0.96, 0.5],
    )
    table.auto_set_font_size(False)
    table.set_fontsize(TEXT_FONTSIZE)
    for cell in table.get_celld().values():
        cell.set_edgecolor("none")
        cell.set_linewidth(0)
    if title:
        ax.set_title(title, pad=10, fontsize=TITLE_FONTSIZE, fontweight="bold", loc="center")
    else:
        ax.set_title("Metadata", pad=2, fontsize=TITLE_FONTSIZE, fontweight="bold", loc="left")

    return ax


def _plot_sample_flow(ax, sample_flow: pd.DataFrame):
    ax.clear()
    ax.set_title("Sample flow", fontsize=TITLE_FONTSIZE, fontweight="bold", loc="center")
    ax.barh(range(len(sample_flow)), sample_flow["n"], color="C2")
    ax.set_yticks(range(len(sample_flow)))
    ax.set_yticklabels(sample_flow["step"])
    ax.set_xlabel("N", fontsize=TEXT_FONTSIZE)
    ax.tick_params(axis="both", labelsize=TEXT_FONTSIZE)
    for i, v in enumerate(sample_flow["n"]):
        ax.text(v, i, f" {int(v)}", va="center", fontsize=TEXT_FONTSIZE)


def _plot_completeness(ax, sample_flow: pd.DataFrame):
    ax.clear()
    ax.set_title("Phenotype completeness", fontsize=TITLE_FONTSIZE, fontweight="bold", loc="center")

    if sample_flow.empty:
        ax.text(0.5, 0.5, "No sample flow data", ha="center", va="center", fontsize=TEXT_FONTSIZE)
        ax.set_axis_off()
        return

    non_missing = sample_flow.loc[sample_flow["step"].str.startswith("non-missing", na=False), "n"]
    missing = sample_flow.loc[sample_flow["step"].str.startswith("missing", na=False), "n"]
    nm_val = int(non_missing.iloc[0]) if not non_missing.empty else 0
    miss_val = int(missing.iloc[0]) if not missing.empty else 0
    total = nm_val + miss_val
    nm_pct = (nm_val / total * 100.0) if total > 0 else 0.0
    miss_pct = (miss_val / total * 100.0) if total > 0 else 0.0

    x = np.arange(2)
    bars = ax.bar(x, [nm_val, miss_val], color=[COLOR_ACCENT, COLOR_NEUTRAL])
    ax.set_xticks(x)
    ax.set_xticklabels(["Non-missing", "Missing"], rotation=15, ha="right", fontsize=TEXT_FONTSIZE)
    ax.set_ylabel("Count", fontsize=TEXT_FONTSIZE)
    ax.tick_params(axis="y", labelsize=TEXT_FONTSIZE)
    ax.grid(axis="y", color=GRID_COLOR, linewidth=0.8)

    for bar, pct in zip(bars, [nm_pct, miss_pct]):
        height = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            height * 0.98,
            f"{pct:.1f}%",
            ha="center",
            va="top",
            fontsize=TEXT_FONTSIZE,
            color="white" if height > 0 else "black",
        )


def _plot_demographics(ax, demographics: pd.DataFrame):
    ax.clear()
    ax.axis("off")
    ax.set_title("Demographic summary", fontsize=TITLE_FONTSIZE, fontweight="bold", loc="center")
    demo_df = demographics.copy()
    demo_df = demo_df.fillna("")

    table = ax.table(
        cellText=demo_df.values,
        colLabels=list(demo_df.columns),
        cellLoc="left",
        colLoc="left",
        loc="upper left",
        bbox=[0.02, 0.02, 0.96, 0.90],
    )
    table.auto_set_font_size(False)
    table.set_fontsize(TABLE_FONTSIZE)
    numeric_cols = [i for i, name in enumerate(demo_df.columns) if name in {"N", "%", "Value"}]
    for (row, col), cell in table.get_celld().items():
        if row == 0:
            cell.set_facecolor(COLOR_BAND_2)
            cell.set_text_props(fontweight="bold")
        else:
            if row % 2 == 0:
                cell.set_facecolor(COLOR_BAND_1)
            if col in numeric_cols:
                cell.set_text_props(ha="right")


def _plot_distribution(ax, df, phenotype, field_label, ptype, *, sex_label_col: str = "__sex_label"):
    ax.clear()
    ax.set_title("Distribution", fontsize=TITLE_FONTSIZE, fontweight="bold", loc="center")
    data = df[phenotype].dropna()

    if data.empty:
        ax.text(0.5, 0.5, "No data", ha="center")
        ax.set_axis_off()
        return

    # show Male/Female only
    if sex_label_col not in df.columns:
        ax.text(0.5, 0.5, "Sex column not available", ha="center")
        ax.set_axis_off()
        return

    sexes = df[sex_label_col].fillna(pd.NA)
    if sexes.notna().sum() == 0:
        ax.text(0.5, 0.5, "No sex labels available", ha="center")
        ax.set_axis_off()
        return

    if ptype == "continuous":
        annotations = []
        for label, color in [("Male", COLOR_MALE), ("Female", COLOR_FEMALE)]:
            sub = df.loc[sexes == label, phenotype].dropna()
            if sub.empty:
                continue
            ax.hist(sub, bins=30, alpha=0.35, label=label, color=color)
            mean = sub.mean()
            ax.axvline(mean, linestyle="--", color=color, linewidth=1.8)
            annotations.append((label, mean, color))
        for idx, (label, mean, color) in enumerate(annotations):
            y_pos = 0.96 - (idx * 0.07)
            ax.plot(
                [0.62, 0.69],
                [y_pos - 0.015, y_pos - 0.015],
                transform=ax.transAxes,
                linestyle="--",
                color=color,
                linewidth=1.8,
            )
            ax.text(
                0.98,
                y_pos,
                f"{label} mean: {mean:.2f}",
                transform=ax.transAxes,
                ha="right",
                va="top",
                fontsize=TEXT_FONTSIZE,
                color=color,
            )
        ax.set_xlabel(field_label, fontsize=TEXT_FONTSIZE)
        ax.set_ylabel("Count", fontsize=TEXT_FONTSIZE)
        ax.tick_params(axis="both", labelsize=TEXT_FONTSIZE)
        ax.grid(axis="y", color=GRID_COLOR, linewidth=0.8)
        ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.13), ncol=2, fontsize=TEXT_FONTSIZE, frameon=False)
    else:
        labels = None
        for label, color in [("Male", COLOR_MALE), ("Female", COLOR_FEMALE)]:
            sub = df.loc[sexes == label, phenotype].dropna()
            if sub.empty:
                continue
            vc = sub.value_counts().sort_index()
            labels = vc.index.astype(str) if labels is None else labels
            ax.bar(range(len(vc)), vc.values, alpha=0.4, label=label, color=color)
        if labels is not None:
            ax.set_xticks(range(len(labels)))
            ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=TEXT_FONTSIZE)
        ax.set_xlabel("Category", fontsize=TEXT_FONTSIZE)
        ax.set_ylabel("Count", fontsize=TEXT_FONTSIZE)
        ax.tick_params(axis="y", labelsize=TEXT_FONTSIZE)
        ax.grid(axis="y", color=GRID_COLOR, linewidth=0.8)
        ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.13), ncol=2, fontsize=TEXT_FONTSIZE, frameon=False)


def _plot_age_trend(ax, df, phenotype, field_label, age_col, min_n, age_bin_width):
    ax.clear()
    ax.set_title("Mean by age (binned)", fontsize=TITLE_FONTSIZE, fontweight="bold", loc="center")

    if age_col not in df.columns:
        ax.text(0.5, 0.5, "Age column not available", ha="center")
        ax.set_axis_off()
        return

    # keep only rows with phenotype and age
    tmp = df[[phenotype, age_col, "__sex_label"]].dropna(subset=[phenotype, age_col])
    if tmp.empty:
        ax.text(0.5, 0.5, "No non-missing data for age trend", ha="center")
        ax.set_axis_off()
        return

    # bins: 2-year bins from floor(min) to ceil(max)
    amin = int(np.floor(tmp[age_col].min()))
    amax = int(np.ceil(tmp[age_col].max()))
    bins = np.arange(amin, amax + age_bin_width + 1, age_bin_width) if (amax - amin) >= age_bin_width else np.arange(amin, amax + 1, 1)
    tmp["_bin"] = pd.cut(tmp[age_col], bins=bins, right=False, include_lowest=True)

    grouped = (
        tmp.groupby(["_bin", "__sex_label"], observed=False)
        .agg(mean=(phenotype, "mean"), sd=(phenotype, "std"), n=(phenotype, "count"))
        .reset_index()
    )

    # plot Male and Female only and in that order
    for label, color in [("Male", COLOR_MALE), ("Female", COLOR_FEMALE)]:
        g = grouped[grouped["__sex_label"] == label]
        g = g[g["n"] >= min_n]
        if g.empty:
            continue
        mids = g["_bin"].apply(lambda x: (x.left + x.right) / 2)
        # yerr is 1 SD
        ax.errorbar(mids, g["mean"], yerr=g["sd"], fmt="o", label=label, color=color)
    ax.set_xlabel("Age (years)", fontsize=TEXT_FONTSIZE)
    ax.set_ylabel(f"Mean {field_label}", fontsize=TEXT_FONTSIZE)
    ax.tick_params(axis="both", labelsize=TEXT_FONTSIZE)
    ax.grid(axis="y", color=GRID_COLOR, linewidth=0.8)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.13), ncol=2, fontsize=TEXT_FONTSIZE, frameon=False)


def _plot_body_trend(ax, df, phenotype, field_label, body_col, min_n, bin_width):
    # body_col might be None or missing
    ax.clear()
    title = f"Mean by {body_col.split('.')[-1] if body_col else ''} (binned)"
    ax.set_title(title, fontsize=TITLE_FONTSIZE, fontweight="bold", loc="center")

    if body_col is None or body_col not in df.columns:
        ax.text(0.5, 0.5, f"{body_col or 'Body measure'} not available", ha="center")
        ax.set_axis_off()
        return

    tmp = df[[phenotype, body_col, "__sex_label"]].dropna(subset=[phenotype, body_col])
    if tmp.empty:
        ax.text(0.5, 0.5, "No non-missing data", ha="center")
        ax.set_axis_off()
        return

    # restrict to Male/Female labels only
    tmp = tmp[tmp["__sex_label"].isin(["Male", "Female"])]
    if tmp.empty:
        ax.text(0.5, 0.5, "No Male/Female labelled rows", ha="center")
        ax.set_axis_off()
        return

    # use fixed-width bins (e.g., 1 unit) so every unit gets its own bin
    bmin = float(np.floor(tmp[body_col].min()))
    bmax = float(np.ceil(tmp[body_col].max()))
    if bin_width <= 0:
        raise ValueError("bin_width must be > 0")
    bins = np.arange(bmin, bmax + bin_width, bin_width)
    if len(bins) < 2:
        bins = np.array([bmin, bmax + 1.0])
    tmp["_bin"] = pd.cut(tmp[body_col], bins=bins, right=False, include_lowest=True)

    grouped = (
        tmp.groupby(["_bin", "__sex_label"], observed=False)
        .agg(mean=(phenotype, "mean"), sd=(phenotype, "std"), n=(phenotype, "count"))
        .reset_index()
    )

    for label, color in [("Male", COLOR_MALE), ("Female", COLOR_FEMALE)]:
        g = grouped[grouped["__sex_label"] == label]
        g = g[g["n"] >= min_n]
        if g.empty:
            continue
        mids = g["_bin"].apply(lambda x: (x.left + x.right) / 2)
        ax.errorbar(mids, g["mean"], yerr=g["sd"], fmt="o", label=label, color=color)
    ax.set_xlabel(body_col.split(".")[-1], fontsize=TEXT_FONTSIZE)
    ax.set_ylabel(f"Mean {field_label}", fontsize=TEXT_FONTSIZE)
    ax.tick_params(axis="both", labelsize=TEXT_FONTSIZE)
    ax.grid(axis="y", color=GRID_COLOR, linewidth=0.8)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.13), ncol=2, fontsize=TEXT_FONTSIZE, frameon=False)


# End of module
