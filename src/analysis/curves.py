"""
curves.py — Sensitivity curves and Exp1/Exp2 gap analysis.

Public API:
    load_results(results_jsonl, manifest_dir) -> pd.DataFrame
    plot_sensitivity_curve(df, dimension, vlm) -> matplotlib.Figure
    plot_exp_gap(df, dimension) -> matplotlib.Figure
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Magnitude ordering helpers
# ---------------------------------------------------------------------------

# For continuous dimensions: the x-axis is the numeric value extracted from
# degradation params.  For discrete dimensions it's a severity rank.

_CONTENT_SWAP_RANK = {"partial": 1, "similar": 2, "completely_wrong": 3}
_WORD_ERROR_RANK = {"wrong_word": 1, "extra_word": 1, "missing_word": 2}
_CASE_ERROR_RANK = {"first_char_flip": 1, "all_caps": 2, "all_lower": 2}
_FONT_WEIGHT_MAP = {"light": 300, "normal": 400, "medium": 500, "bold": 700}


def _params_to_numeric_magnitude(dimension: str, params: dict) -> float:
    """Extract a single numeric severity value from degradation params."""
    if dimension == "color_offset":
        return float(params.get("delta_e", 0))

    elif dimension == "position_offset":
        x = params.get("offset_x_px", 0)
        y = params.get("offset_y_px", 0)
        return float((x ** 2 + y ** 2) ** 0.5)

    elif dimension == "scale_error":
        return abs(float(params.get("scale_error_pct", 0)))

    elif dimension == "rotation":
        return abs(float(params.get("angle_deg", 0)))

    elif dimension == "font_weight":
        fw = params.get("font_weight", 400)
        if isinstance(fw, str):
            fw = _FONT_WEIGHT_MAP.get(fw.lower(), 400)
        return abs(int(fw) - 400)

    elif dimension == "font_style":
        return {"italic": 1.0, "oblique": 0.5}.get(
            str(params.get("font_style", "")), 0.0
        )

    elif dimension == "letter_spacing":
        return abs(float(params.get("letter_spacing_px", 0)))

    elif dimension == "opacity":
        return round(1.0 - float(params.get("opacity", 1.0)), 4)

    elif dimension == "char_substitution":
        return float(params.get("num_substitutions", 0))

    elif dimension == "word_error":
        return float(_WORD_ERROR_RANK.get(str(params.get("error_type", "")), 0))

    elif dimension == "case_error":
        return float(_CASE_ERROR_RANK.get(str(params.get("case_type", "")), 0))

    elif dimension == "content_swap":
        new_text = str(params.get("new_text", ""))
        # rank by the magnitude field carried in metadata when available, else
        # use text length as a rough proxy (shorter → smaller change).
        for label, rank in _CONTENT_SWAP_RANK.items():
            if new_text in ("Text", "XXXXXXXXXXX", "Sample Heading"):
                ranks = {"Text": 1, "Sample Heading": 2, "XXXXXXXXXXX": 3}
                return float(ranks.get(new_text, 0))
        return 0.0

    elif dimension == "gaussian_noise":
        return float(params.get("sigma", 0))

    elif dimension == "jpeg_compression":
        return 100.0 - float(params.get("quality", 100))

    elif dimension == "blur":
        return float(params.get("radius", 0))

    return 0.0


def _x_label(dimension: str) -> str:
    labels = {
        "color_offset": "ΔE (CIEDE2000)",
        "position_offset": "Offset magnitude (px)",
        "scale_error": "Scale error (%)",
        "rotation": "Rotation (°)",
        "font_weight": "Weight distance from 400",
        "font_style": "Severity",
        "letter_spacing": "|Letter spacing| (px)",
        "opacity": "Opacity reduction",
        "char_substitution": "Substituted characters",
        "word_error": "Severity",
        "case_error": "Severity",
        "content_swap": "Severity",
        "gaussian_noise": "Noise σ",
        "jpeg_compression": "Compression severity (100-quality)",
        "blur": "Blur radius",
    }
    return labels.get(dimension, dimension)


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_results(
    results_jsonl: str | Path,
    manifest_dir: str | Path,
) -> pd.DataFrame:
    """Load results.jsonl and join with per-stimulus metadata.

    Parameters
    ----------
    results_jsonl:
        Path to the JSONL file produced by runner.py.
    manifest_dir:
        Directory containing stimulus sub-directories each with metadata.json
        (e.g. ``data/full/``).

    Returns
    -------
    pd.DataFrame
        One row per result record, enriched with metadata columns:
        edit_type, edit_id, template_id, difficulty_tier,
        degradation_dimension, degradation_magnitude, degradation_layer,
        degradation_params, numeric_magnitude, model, experiment,
        parse_success, plus all score columns.
    """
    results_jsonl = Path(results_jsonl)
    manifest_dir = Path(manifest_dir)

    # -- Load results
    records: list[dict] = []
    with open(results_jsonl, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError:
                    pass

    # -- Load metadata index
    meta: dict[str, dict] = {}
    if manifest_dir.exists():
        for d in manifest_dir.iterdir():
            mpath = d / "metadata.json"
            if d.is_dir() and mpath.exists():
                m = json.loads(mpath.read_text(encoding="utf-8"))
                meta[m["stimulus_id"]] = m

    # -- Join
    rows = []
    for r in records:
        sid = r.get("stimulus_id", "")
        m = meta.get(sid, {})
        deg = m.get("degradation", {})
        tmpl = m.get("template", {})
        params = deg.get("params", {})
        dimension = deg.get("dimension", "unknown")

        row = {
            "stimulus_id": sid,
            "model": r.get("model", "unknown"),
            "experiment": r.get("experiment", ""),
            "parse_success": r.get("parse_success", True),
            # metadata
            "edit_type": m.get("edit_type", "unknown"),
            "edit_id": m.get("edit_id", "unknown"),
            "template_id": tmpl.get("template_id", "unknown"),
            "difficulty_tier": tmpl.get("difficulty_tier", "unknown"),
            "degradation_dimension": dimension,
            "degradation_magnitude": deg.get("magnitude", "unknown"),
            "degradation_layer": deg.get("layer", "unknown"),
            "degradation_params": params,
            "numeric_magnitude": _params_to_numeric_magnitude(dimension, params),
            # Exp1 scores
            "detected_difference": r.get("detected_difference"),
            "similarity_score": r.get("similarity_score"),
            "exp1_description": r.get("description"),
            # Exp2 scores
            "instruction_following": r.get("instruction_following"),
            "text_accuracy": r.get("text_accuracy"),
            "visual_consistency": r.get("visual_consistency"),
            "layout_preservation": r.get("layout_preservation"),
            "overall_quality": r.get("overall_quality"),
            "errors_noticed": r.get("errors_noticed"),
        }
        rows.append(row)

    df = pd.DataFrame(rows)
    return df


# ---------------------------------------------------------------------------
# Plot: sensitivity curve
# ---------------------------------------------------------------------------

def plot_sensitivity_curve(
    df: pd.DataFrame,
    dimension: str,
    vlm: str,
    *,
    exp2_score: str = "overall_quality",
    figsize: tuple[float, float] = (8, 5),
) -> plt.Figure:
    """Plot VLM score vs degradation magnitude, one line per edit type.

    Uses Experiment 2 (instruction-following evaluation) scores.

    Parameters
    ----------
    df:
        DataFrame from :func:`load_results`.
    dimension:
        Degradation dimension to plot (e.g. ``"color_offset"``).
    vlm:
        Model name to filter on (e.g. ``"gpt-4o"`` or ``"dummy"``).
    exp2_score:
        Which Exp2 score column to use on the y-axis.
    """
    sub = df[
        (df["degradation_dimension"] == dimension)
        & (df["model"] == vlm)
        & (df["experiment"] == "experiment_2")
        & df["parse_success"]
        & df[exp2_score].notna()
    ].copy()

    if sub.empty:
        fig, ax = plt.subplots(figsize=figsize)
        ax.text(0.5, 0.5, f"No data for dimension={dimension!r}, model={vlm!r}",
                ha="center", va="center", transform=ax.transAxes)
        ax.set_title(f"Sensitivity curve — {dimension} [{vlm}]")
        return fig

    edit_types = sorted(sub["edit_type"].unique())
    fig, ax = plt.subplots(figsize=figsize)

    for et in edit_types:
        et_sub = sub[sub["edit_type"] == et]
        grouped = (
            et_sub.groupby("numeric_magnitude")[exp2_score]
            .agg(["mean", "sem"])
            .reset_index()
            .sort_values("numeric_magnitude")
        )
        x = grouped["numeric_magnitude"].values
        y = grouped["mean"].values
        yerr = grouped["sem"].values
        line, = ax.plot(x, y, marker="o", label=et)
        ax.fill_between(x, y - yerr, y + yerr, alpha=0.15, color=line.get_color())

    # Also plot overall mean across edit types
    overall = (
        sub.groupby("numeric_magnitude")[exp2_score]
        .agg(["mean", "sem"])
        .reset_index()
        .sort_values("numeric_magnitude")
    )
    ax.plot(
        overall["numeric_magnitude"],
        overall["mean"],
        color="black",
        linewidth=2,
        linestyle="--",
        marker="s",
        label="overall mean",
    )

    ax.set_xlabel(_x_label(dimension))
    ax.set_ylabel(exp2_score.replace("_", " ").title() + " (1–5)")
    ax.set_title(f"Sensitivity curve — {dimension}\nmodel: {vlm}")
    ax.set_ylim(0.5, 5.5)
    ax.axhline(4.0, color="red", linestyle=":", linewidth=1, alpha=0.6,
               label="threshold (4.0)")
    ax.legend(loc="best", fontsize=8)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Plot: Exp1 vs Exp2 gap
# ---------------------------------------------------------------------------

def plot_exp_gap(
    df: pd.DataFrame,
    dimension: str,
    *,
    vlm: str | None = None,
    exp2_score: str = "overall_quality",
    figsize: tuple[float, float] = (8, 5),
) -> plt.Figure:
    """Plot Exp1 vs Exp2 scores for the same stimuli, grouped by magnitude.

    Exp1 uses ``similarity_score`` (higher = more similar).
    Exp2 uses ``overall_quality`` (or ``exp2_score``).

    The gap between the curves quantifies how much harder judging is vs
    simply detecting a difference.
    """
    filt = (df["degradation_dimension"] == dimension) & df["parse_success"]
    if vlm is not None:
        filt &= df["model"] == vlm

    exp1 = df[filt & (df["experiment"] == "experiment_1") & df["similarity_score"].notna()].copy()
    exp2 = df[filt & (df["experiment"] == "experiment_2") & df[exp2_score].notna()].copy()

    if exp1.empty and exp2.empty:
        fig, ax = plt.subplots(figsize=figsize)
        ax.text(0.5, 0.5, f"No data for dimension={dimension!r}",
                ha="center", va="center", transform=ax.transAxes)
        ax.set_title(f"Exp1 vs Exp2 gap — {dimension}")
        return fig

    fig, ax = plt.subplots(figsize=figsize)

    def _plot_series(data: pd.DataFrame, score_col: str, label: str, color: str, marker: str):
        if data.empty:
            return
        grouped = (
            data.groupby("numeric_magnitude")[score_col]
            .agg(["mean", "sem"])
            .reset_index()
            .sort_values("numeric_magnitude")
        )
        x = grouped["numeric_magnitude"].values
        y = grouped["mean"].values
        yerr = grouped["sem"].values
        ax.plot(x, y, color=color, marker=marker, label=label, linewidth=2)
        ax.fill_between(x, y - yerr, y + yerr, alpha=0.15, color=color)

    _plot_series(exp1, "similarity_score", "Exp1: similarity score", "#2196F3", "o")
    _plot_series(exp2, exp2_score, f"Exp2: {exp2_score.replace('_', ' ')}", "#FF5722", "s")

    # Shade the gap between curves where both exist
    mags_1 = set(exp1["numeric_magnitude"].unique())
    mags_2 = set(exp2["numeric_magnitude"].unique())
    shared_mags = sorted(mags_1 & mags_2)
    if shared_mags:
        m1_mean = (
            exp1[exp1["numeric_magnitude"].isin(shared_mags)]
            .groupby("numeric_magnitude")["similarity_score"]
            .mean()
            .reindex(shared_mags)
        )
        m2_mean = (
            exp2[exp2["numeric_magnitude"].isin(shared_mags)]
            .groupby("numeric_magnitude")[exp2_score]
            .mean()
            .reindex(shared_mags)
        )
        ax.fill_between(
            shared_mags,
            m2_mean.values,
            m1_mean.values,
            alpha=0.1,
            color="gray",
            label="gap",
        )

    ax.set_xlabel(_x_label(dimension))
    ax.set_ylabel("Score (1–5)")
    title_suffix = f" [{vlm}]" if vlm else ""
    ax.set_title(f"Exp1 vs Exp2 gap — {dimension}{title_suffix}")
    ax.set_ylim(0.5, 5.5)
    ax.axhline(4.0, color="red", linestyle=":", linewidth=1, alpha=0.6, label="threshold")
    ax.legend(loc="best", fontsize=9)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    return fig
