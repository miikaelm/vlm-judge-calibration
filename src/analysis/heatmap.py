"""
heatmap.py — Edit type × degradation dimension detection rate heatmap.

Public API:
    plot_detection_heatmap(df, vlm, experiment) -> matplotlib.Figure
    plot_score_heatmap(df, vlm, score_col) -> matplotlib.Figure
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
import pandas as pd


# Canonical ordering so axes are consistent across runs
_DIMENSION_ORDER = [
    "color_offset",
    "position_offset",
    "alignment_error",
    "scale_error",
    "rotation",
    "letter_spacing",
    "opacity",
    "char_substitution",
    "word_error",
    "case_error",
    "content_swap",
    "font_weight",
    "font_style",
    "font_family",
    "gaussian_noise",
    "jpeg_compression",
    "blur",
]

_EDIT_TYPE_ORDER = [
    "color",
    "scale",
    "rotation",
    "relocation",
    "font_weight",
    "italic",
    "letter_spacing",
    "font_family",
    # Legacy edit type names from older result files:
    "color_change",
    "position_change",
    "scale_change",
    "content_change",
]


def _present_dims(df: pd.DataFrame, order: list[str]) -> list[str]:
    """Return dimensions present in df, in canonical order."""
    present = set(df["degradation_dimension"].unique())
    ordered = [d for d in order if d in present]
    remaining = sorted(present - set(ordered))
    return ordered + remaining


def _present_edit_types(df: pd.DataFrame, order: list[str]) -> list[str]:
    present = set(df["edit_type"].unique())
    ordered = [e for e in order if e in present]
    remaining = sorted(present - set(ordered))
    return ordered + remaining


def plot_detection_heatmap(
    df: pd.DataFrame,
    vlm: str,
    *,
    experiment: str = "experiment_1",
    exclude_perfect: bool = True,
    figsize: tuple[float, float] | None = None,
) -> plt.Figure:
    """Heatmap of Exp1 detection rate: edit type (rows) × degradation dimension (cols).

    Cell value = fraction of stimuli where ``detected_difference`` is True,
    averaged across all magnitudes for that (edit_type, dimension) pair.

    Parameters
    ----------
    df:
        DataFrame from ``curves.load_results``.
    vlm:
        Model name to filter on.
    experiment:
        Which experiment to use. Experiment 1 has ``detected_difference``.
    exclude_perfect:
        When True (default), magnitude-0 stimuli are excluded — they are
        not detectable and must not inflate/distort detection-rate estimates.
    """
    sub = df[
        (df["model"] == vlm)
        & (df["experiment"] == experiment)
        & df["parse_success"]
        & df["detected_difference"].notna()
    ].copy()
    if exclude_perfect:
        sub = sub[sub["numeric_magnitude"] > 0]
    sub["detected"] = sub["detected_difference"].astype(bool).astype(float)

    dims = _present_dims(sub, _DIMENSION_ORDER)
    edit_types = _present_edit_types(sub, _EDIT_TYPE_ORDER)

    matrix = np.full((len(edit_types), len(dims)), np.nan)
    for i, et in enumerate(edit_types):
        for j, dim in enumerate(dims):
            cell = sub[(sub["edit_type"] == et) & (sub["degradation_dimension"] == dim)]
            if not cell.empty:
                matrix[i, j] = cell["detected"].mean()

    if figsize is None:
        figsize = (max(8, len(dims) * 0.9), max(4, len(edit_types) * 0.8))

    fig, ax = plt.subplots(figsize=figsize)
    im = ax.imshow(matrix, cmap="RdYlGn", vmin=0, vmax=1, aspect="auto")

    ax.set_xticks(range(len(dims)))
    ax.set_xticklabels([d.replace("_", "\n") for d in dims], fontsize=8)
    ax.set_yticks(range(len(edit_types)))
    ax.set_yticklabels([e.replace("_", " ") for e in edit_types], fontsize=9)

    plt.setp(ax.get_xticklabels(), rotation=30, ha="right")

    # Annotate cells
    for i in range(len(edit_types)):
        for j in range(len(dims)):
            val = matrix[i, j]
            if not np.isnan(val):
                text_color = "white" if val < 0.3 or val > 0.7 else "black"
                ax.text(j, i, f"{val:.0%}", ha="center", va="center",
                        fontsize=7, color=text_color, fontweight="bold")
            else:
                ax.text(j, i, "—", ha="center", va="center", fontsize=7, color="gray")

    cbar = fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02)
    cbar.set_label("Detection rate", fontsize=9)

    suffix = " (mag > 0 only)" if exclude_perfect else ""
    ax.set_title(
        f"Exp1 detection rate — model: {vlm}{suffix}\n"
        f"(fraction of stimuli where difference was detected)",
        fontsize=10,
    )
    ax.set_xlabel("Degradation dimension", fontsize=9)
    ax.set_ylabel("Edit type", fontsize=9)
    fig.tight_layout()
    return fig


def plot_detection_heatmap_by_dim(
    df: pd.DataFrame,
    vlm: str,
    *,
    experiment: str = "experiment_1",
    exclude_perfect: bool = True,
    figsize: tuple[float, float] | None = None,
) -> plt.Figure:
    """Bar chart of Exp1 detection rate by degradation dimension only.

    Edit type is intentionally excluded — Exp1 measures perceptual sensitivity
    to the degradation itself, independent of which edit type was applied.
    """
    sub = df[
        (df["model"] == vlm)
        & (df["experiment"] == experiment)
        & df["parse_success"]
        & df["detected_difference"].notna()
    ].copy()
    if exclude_perfect:
        sub = sub[sub["numeric_magnitude"] > 0]
    sub["detected"] = sub["detected_difference"].astype(bool).astype(float)

    dims = _present_dims(sub, _DIMENSION_ORDER)

    if sub.empty:
        fig, ax = plt.subplots(figsize=figsize or (8, 4))
        ax.text(0.5, 0.5, "No Exp1 data", ha="center", va="center", transform=ax.transAxes)
        ax.set_title(f"Exp1 detection rate — model: {vlm}")
        return fig

    det_rates = [sub[sub["degradation_dimension"] == d]["detected"].mean() for d in dims]
    ns = [int((sub["degradation_dimension"] == d).sum()) for d in dims]

    if figsize is None:
        figsize = (max(8, len(dims) * 1.1), 4)

    fig, ax = plt.subplots(figsize=figsize)
    colors = ["#1a9850" if v > 0.7 else "#fee08b" if v > 0.4 else "#d73027"
              for v in det_rates]
    bars = ax.bar(range(len(dims)), det_rates, color=colors, edgecolor="white", linewidth=0.8)

    for bar, val, n in zip(bars, det_rates, ns):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                f"{val:.0%}\n(n={n})", ha="center", va="bottom", fontsize=8)

    ax.set_xticks(range(len(dims)))
    ax.set_xticklabels([d.replace("_", "\n") for d in dims], fontsize=9)
    ax.set_ylim(0, 1.15)
    ax.set_ylabel("Detection rate", fontsize=9)
    ax.axhline(0.0, color="black", linewidth=0.5)
    suffix = " (mag > 0 only)" if exclude_perfect else ""
    ax.set_title(
        f"Exp1 detection rate by degradation dimension — model: {vlm}{suffix}",
        fontsize=10,
    )
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    return fig


def plot_score_heatmap(
    df: pd.DataFrame,
    vlm: str,
    *,
    score_col: str = "overall_quality",
    experiment: str = "experiment_2",
    exclude_perfect: bool = True,
    figsize: tuple[float, float] | None = None,
) -> plt.Figure:
    """Heatmap of mean VLM score: edit type (rows) × degradation dimension (cols).

    Useful for Exp2 where the score dimension is richer than a binary
    detected/not-detected flag.

    Parameters
    ----------
    df:
        DataFrame from ``curves.load_results``.
    vlm:
        Model name to filter on.
    score_col:
        Column to aggregate (default: ``"overall_quality"``).
    experiment:
        Which experiment to use.
    exclude_perfect:
        When True (default), magnitude-0 stimuli are excluded so the heatmap
        reflects performance on *degraded* edits only.
    """
    sub = df[
        (df["model"] == vlm)
        & (df["experiment"] == experiment)
        & df["parse_success"]
        & df[score_col].notna()
    ].copy()
    if exclude_perfect:
        sub = sub[sub["numeric_magnitude"] > 0]

    dims = _present_dims(sub, _DIMENSION_ORDER)
    edit_types = _present_edit_types(sub, _EDIT_TYPE_ORDER)

    matrix = np.full((len(edit_types), len(dims)), np.nan)
    for i, et in enumerate(edit_types):
        for j, dim in enumerate(dims):
            cell = sub[(sub["edit_type"] == et) & (sub["degradation_dimension"] == dim)]
            if not cell.empty:
                matrix[i, j] = cell[score_col].mean()

    if figsize is None:
        figsize = (max(8, len(dims) * 0.9), max(4, len(edit_types) * 0.8))

    fig, ax = plt.subplots(figsize=figsize)
    # reversed colormap: low score (bad) = red, high score (good) = green
    im = ax.imshow(matrix, cmap="RdYlGn", vmin=1, vmax=5, aspect="auto")

    ax.set_xticks(range(len(dims)))
    ax.set_xticklabels([d.replace("_", "\n") for d in dims], fontsize=8)
    ax.set_yticks(range(len(edit_types)))
    ax.set_yticklabels([e.replace("_", " ") for e in edit_types], fontsize=9)
    plt.setp(ax.get_xticklabels(), rotation=30, ha="right")

    for i in range(len(edit_types)):
        for j in range(len(dims)):
            val = matrix[i, j]
            if not np.isnan(val):
                text_color = "white" if val < 2.5 or val > 4.5 else "black"
                ax.text(j, i, f"{val:.2f}", ha="center", va="center",
                        fontsize=7, color=text_color, fontweight="bold")
            else:
                ax.text(j, i, "—", ha="center", va="center", fontsize=7, color="gray")

    cbar = fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02)
    cbar.set_label(score_col.replace("_", " ").title() + " (mean)", fontsize=9)

    suffix = " (mag > 0 only)" if exclude_perfect else ""
    ax.set_title(
        f"Exp2 mean {score_col.replace('_', ' ')} — model: {vlm}{suffix}",
        fontsize=10,
    )
    ax.set_xlabel("Degradation dimension", fontsize=9)
    ax.set_ylabel("Edit type", fontsize=9)
    fig.tight_layout()
    return fig


_EXP2_SCORE_COLS = [
    "instruction_following",
    "text_accuracy",
    "visual_consistency",
    "layout_preservation",
    "overall_quality",
]


def plot_perfect_detection_heatmap(
    df: pd.DataFrame,
    vlm: str,
    *,
    experiment: str = "experiment_1",
    figsize: tuple[float, float] | None = None,
) -> plt.Figure:
    """Bar chart of Exp1 false positive rate on perfect (magnitude=0) edits.

    Degradation dimension is not meaningful for perfect edits, so the x-axis
    is edit type only.  High bars (red) = the model hallucinates differences.
    """
    sub = df[
        (df["model"] == vlm)
        & (df["experiment"] == experiment)
        & df["parse_success"]
        & df["detected_difference"].notna()
        & (df["numeric_magnitude"] == 0)
    ].copy()
    sub["detected"] = sub["detected_difference"].astype(bool).astype(float)

    edit_types = _present_edit_types(sub, _EDIT_TYPE_ORDER)

    if sub.empty:
        fig, ax = plt.subplots(figsize=figsize or (6, 4))
        ax.text(0.5, 0.5, "No perfect-edit (mag=0) Exp1 data",
                ha="center", va="center", transform=ax.transAxes)
        ax.set_title(f"Exp1 false positive rate (mag=0) — model: {vlm}")
        return fig

    fp_rates = [sub[sub["edit_type"] == et]["detected"].mean() for et in edit_types]
    ns = [int((sub["edit_type"] == et).sum()) for et in edit_types]

    if figsize is None:
        figsize = (max(5, len(edit_types) * 1.4), 4)

    fig, ax = plt.subplots(figsize=figsize)
    colors = ["#d73027" if v > 0.2 else "#fee08b" if v > 0.05 else "#1a9850"
              for v in fp_rates]
    bars = ax.bar(range(len(edit_types)), fp_rates, color=colors, edgecolor="white", linewidth=0.8)

    for bar, val, n in zip(bars, fp_rates, ns):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                f"{val:.0%}\n(n={n})", ha="center", va="bottom", fontsize=8)

    ax.set_xticks(range(len(edit_types)))
    ax.set_xticklabels([e.replace("_", " ") for e in edit_types], fontsize=9)
    ax.set_ylim(0, 1.15)
    ax.set_ylabel("False positive rate", fontsize=9)
    ax.axhline(0.0, color="black", linewidth=0.5)
    ax.set_title(
        f"Exp1 false positive rate on perfect edits (mag=0) — model: {vlm}\n"
        f"(fraction of correct stimuli incorrectly flagged as different)",
        fontsize=10,
    )
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    return fig


def plot_perfect_score_heatmap(
    perfect_df: pd.DataFrame,
    vlm: str,
    *,
    experiment: str = "experiment_2",
    figsize: tuple[float, float] | None = None,
) -> plt.Figure:
    """Heatmap of Exp2 scores on perfect (magnitude=0) edits.

    Rows = edit types, columns = score dimensions.
    A well-calibrated model should score perfect edits at 5/5.  Cells below 5
    (yellow/red) reveal miscalibration on specific edit-type × score-dimension
    combinations.  Degradation dimension is omitted — it is meaningless for
    perfect edits.

    Parameters
    ----------
    perfect_df:
        DataFrame from ``curves.load_perfect_results`` (mag == 0 rows only).
    """
    score_cols = [c for c in _EXP2_SCORE_COLS if c in perfect_df.columns]
    sub = perfect_df[
        (perfect_df["model"] == vlm)
        & (perfect_df["experiment"] == experiment)
        & perfect_df["parse_success"]
    ].copy()
    # Keep rows where at least one score column is non-null
    sub = sub[sub[score_cols].notna().any(axis=1)]

    edit_types = _present_edit_types(sub, _EDIT_TYPE_ORDER)

    if sub.empty:
        fig, ax = plt.subplots(figsize=figsize or (8, 4))
        ax.text(0.5, 0.5, "No perfect-edit (mag=0) Exp2 data",
                ha="center", va="center", transform=ax.transAxes)
        ax.set_title(f"Exp2 perfect-edit scores (mag=0) — model: {vlm}")
        return fig

    n_cols = len(score_cols)
    n_rows = len(edit_types)
    matrix = np.full((n_rows, n_cols), np.nan)
    for i, et in enumerate(edit_types):
        et_sub = sub[sub["edit_type"] == et]
        for j, col in enumerate(score_cols):
            vals = et_sub[col].dropna()
            if not vals.empty:
                matrix[i, j] = vals.mean()

    if figsize is None:
        figsize = (max(6, n_cols * 1.5), max(3, n_rows * 0.9))

    fig, ax = plt.subplots(figsize=figsize)
    im = ax.imshow(matrix, cmap="RdYlGn", vmin=1, vmax=5, aspect="auto")

    ax.set_xticks(range(n_cols))
    ax.set_xticklabels([c.replace("_", "\n") for c in score_cols], fontsize=9)
    ax.set_yticks(range(n_rows))
    ax.set_yticklabels([e.replace("_", " ") for e in edit_types], fontsize=9)

    for i in range(n_rows):
        for j in range(n_cols):
            val = matrix[i, j]
            if not np.isnan(val):
                text_color = "white" if val < 2.5 else "black"
                ax.text(j, i, f"{val:.2f}", ha="center", va="center",
                        fontsize=8, color=text_color, fontweight="bold")
            else:
                ax.text(j, i, "—", ha="center", va="center", fontsize=8, color="gray")

    cbar = fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02)
    cbar.set_label("Mean score (ideal = 5)", fontsize=9)

    ax.set_title(
        f"Exp2 scores on perfect edits (mag=0) — model: {vlm}\n"
        f"(ideal = 5 everywhere; yellow/red = model penalises a correct edit)",
        fontsize=10,
    )
    ax.set_xlabel("Score dimension", fontsize=9)
    ax.set_ylabel("Edit type", fontsize=9)
    fig.tight_layout()
    return fig


def plot_noop_detection_heatmap(
    df: pd.DataFrame,
    vlm: str,
    *,
    experiment: str = "experiment_1",
    figsize: tuple[float, float] | None = None,
) -> plt.Figure:
    """Bar chart of Exp1 false positive rate on noop (unchanged image) stimuli.

    The model is asked whether there is a difference between source and source
    (identical images).  A high bar = the model hallucinates a difference even
    when nothing changed at all.  Results are grouped by edit type to show
    whether hallucination is instruction-driven.
    """
    sub = df[
        (df["model"] == vlm)
        & (df["experiment"] == experiment)
        & df["parse_success"]
        & df["detected_difference"].notna()
    ].copy()
    sub["detected"] = sub["detected_difference"].astype(bool).astype(float)

    edit_types = _present_edit_types(sub, _EDIT_TYPE_ORDER)

    if sub.empty:
        fig, ax = plt.subplots(figsize=figsize or (6, 4))
        ax.text(0.5, 0.5, "No noop Exp1 data",
                ha="center", va="center", transform=ax.transAxes)
        ax.set_title(f"Exp1 false positive rate (noop) — model: {vlm}")
        return fig

    fp_rates = [sub[sub["edit_type"] == et]["detected"].mean() for et in edit_types]
    ns = [int((sub["edit_type"] == et).sum()) for et in edit_types]

    if figsize is None:
        figsize = (max(5, len(edit_types) * 1.4), 4)

    fig, ax = plt.subplots(figsize=figsize)
    colors = ["#d73027" if v > 0.2 else "#fee08b" if v > 0.05 else "#1a9850"
              for v in fp_rates]
    bars = ax.bar(range(len(edit_types)), fp_rates, color=colors, edgecolor="white", linewidth=0.8)

    for bar, val, n in zip(bars, fp_rates, ns):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                f"{val:.0%}\n(n={n})", ha="center", va="bottom", fontsize=8)

    ax.set_xticks(range(len(edit_types)))
    ax.set_xticklabels([e.replace("_", " ") for e in edit_types], fontsize=9)
    ax.set_ylim(0, 1.15)
    ax.set_ylabel("False positive rate", fontsize=9)
    ax.axhline(0.0, color="black", linewidth=0.5)
    ax.set_title(
        f"Exp1 false positive rate on noop (image unchanged) — model: {vlm}\n"
        f"(fraction of identical-image stimuli incorrectly flagged as different)",
        fontsize=10,
    )
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    return fig


def plot_noop_score_heatmap(
    df: pd.DataFrame,
    vlm: str,
    *,
    experiment: str = "experiment_2",
    figsize: tuple[float, float] | None = None,
) -> plt.Figure:
    """Heatmap of Exp2 scores on noop (unchanged image) stimuli.

    Rows = edit types, columns = score dimensions.
    Since no edit was applied, a well-calibrated model should score low.
    High scores (now shown in red) reveal the model accepts an unapplied edit.
    """
    score_cols = [c for c in _EXP2_SCORE_COLS if c in df.columns]
    sub = df[
        (df["model"] == vlm)
        & (df["experiment"] == experiment)
        & df["parse_success"]
    ].copy()
    sub = sub[sub[score_cols].notna().any(axis=1)]

    edit_types = _present_edit_types(sub, _EDIT_TYPE_ORDER)

    if sub.empty:
        fig, ax = plt.subplots(figsize=figsize or (8, 4))
        ax.text(0.5, 0.5, "No noop Exp2 data",
                ha="center", va="center", transform=ax.transAxes)
        ax.set_title(f"Exp2 noop scores — model: {vlm}")
        return fig

    n_cols = len(score_cols)
    n_rows = len(edit_types)
    matrix = np.full((n_rows, n_cols), np.nan)
    for i, et in enumerate(edit_types):
        et_sub = sub[sub["edit_type"] == et]
        for j, col in enumerate(score_cols):
            vals = et_sub[col].dropna()
            if not vals.empty:
                matrix[i, j] = vals.mean()

    if figsize is None:
        figsize = (max(6, n_cols * 1.5), max(3, n_rows * 0.9))

    fig, ax = plt.subplots(figsize=figsize)
    # Reversed colormap: low score = green (correct for noop), high score = red (model was fooled)
    im = ax.imshow(matrix, cmap="RdYlGn_r", vmin=1, vmax=5, aspect="auto")

    ax.set_xticks(range(n_cols))
    ax.set_xticklabels([c.replace("_", "\n") for c in score_cols], fontsize=9)
    ax.set_yticks(range(n_rows))
    ax.set_yticklabels([e.replace("_", " ") for e in edit_types], fontsize=9)

    for i in range(n_rows):
        for j in range(n_cols):
            val = matrix[i, j]
            if not np.isnan(val):
                text_color = "white" if val > 3.5 or val < 1.5 else "black"
                ax.text(j, i, f"{val:.2f}", ha="center", va="center",
                        fontsize=8, color=text_color, fontweight="bold")
            else:
                ax.text(j, i, "—", ha="center", va="center", fontsize=8, color="gray")

    cbar = fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02)
    cbar.set_label("Mean score (ideal for noop = 1)", fontsize=9)

    ax.set_title(
        f"Exp2 scores on noop stimuli (image unchanged) — model: {vlm}\n"
        f"(ideal = 1 everywhere; yellow/red = model accepts unapplied edit)",
        fontsize=10,
    )
    ax.set_xlabel("Score dimension", fontsize=9)
    ax.set_ylabel("Edit type", fontsize=9)
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Detection rate heatmap: rows = dimension, cols = model
# ---------------------------------------------------------------------------

def plot_detection_rate_heatmap_dim_by_model(
    df: pd.DataFrame,
    models: list[str] | None = None,
    *,
    experiment: str = "experiment_1",
    exclude_perfect: bool = True,
    figsize: tuple[float, float] | None = None,
) -> plt.Figure:
    """Heatmap of Exp1 detection rate: degradation dimension (rows) × model (cols).

    Rows are ordered by ascending mean detection rate across all models, so
    the most-blind dimensions appear at the top.  Cell colour: RdYlGn 0→1.

    Parameters
    ----------
    df:
        Combined DataFrame from ``curves.load_results``.
    models:
        List of model names to include.  When None all models in df are used.
    experiment:
        Experiment with binary detected_difference (default: experiment_1).
    exclude_perfect:
        When True (default), magnitude-0 stimuli are excluded.
    """
    sub = df[
        (df["experiment"] == experiment)
        & df["parse_success"]
        & df["detected_difference"].notna()
    ].copy()
    if exclude_perfect:
        sub = sub[sub["numeric_magnitude"] > 0]

    if sub.empty:
        fig, ax = plt.subplots(figsize=figsize or (6, 4))
        ax.text(0.5, 0.5, "No Exp1 data", ha="center", va="center",
                transform=ax.transAxes)
        ax.set_title("Detection rate (dimension × model)")
        return fig

    sub["detected"] = sub["detected_difference"].astype(bool).astype(float)

    if models is None:
        models = sorted(sub["model"].unique().tolist())

    dims = _present_dims(sub, _DIMENSION_ORDER)

    # Compute detection rate matrix (dims × models)
    matrix = np.full((len(dims), len(models)), np.nan)
    for i, dim in enumerate(dims):
        for j, model in enumerate(models):
            cell = sub[(sub["degradation_dimension"] == dim) & (sub["model"] == model)]
            if not cell.empty:
                matrix[i, j] = cell["detected"].mean()

    # Sort rows by ascending mean detection rate (most blind first)
    row_means = np.nanmean(matrix, axis=1)
    sort_idx  = np.argsort(row_means)
    matrix    = matrix[sort_idx]
    dims_sorted = [dims[i] for i in sort_idx]

    if figsize is None:
        figsize = (max(4, len(models) * 1.8 + 2), max(4, len(dims_sorted) * 0.65))

    fig, ax = plt.subplots(figsize=figsize)
    im = ax.imshow(matrix, cmap="RdYlGn", vmin=0, vmax=1, aspect="auto")

    ax.set_yticks(range(len(dims_sorted)))
    ax.set_yticklabels([d.replace("_", " ") for d in dims_sorted], fontsize=9)
    ax.set_xticks(range(len(models)))
    ax.set_xticklabels(list(models), fontsize=9, rotation=20, ha="right")

    for i in range(len(dims_sorted)):
        for j in range(len(models)):
            val = matrix[i, j]
            if not np.isnan(val):
                text_color = "white" if val < 0.3 or val > 0.7 else "black"
                ax.text(j, i, f"{val:.0%}", ha="center", va="center",
                        fontsize=8, color=text_color, fontweight="bold")
            else:
                ax.text(j, i, "—", ha="center", va="center", fontsize=8, color="gray")

    cbar = fig.colorbar(im, ax=ax, fraction=0.04, pad=0.02)
    cbar.set_label("Detection rate", fontsize=9)

    suffix = " (mag > 0)" if exclude_perfect else ""
    ax.set_title(
        f"Exp1 detection rate — dimension × model{suffix}\n"
        f"(rows sorted by ascending mean detection rate)",
        fontsize=10,
    )
    ax.set_xlabel("Model", fontsize=9)
    ax.set_ylabel("Degradation dimension", fontsize=9)
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Score distribution histograms per (dimension, model, experiment)
# ---------------------------------------------------------------------------

def plot_score_distributions(
    df: pd.DataFrame,
    dimension: str,
    models: list[str] | None = None,
    *,
    exclude_perfect: bool = True,
    figsize_per_panel: tuple[float, float] = (3.5, 2.8),
) -> plt.Figure:
    """Multi-panel histogram of scores for a given degradation dimension.

    Layout: rows = models, cols = experiments.
    Exp1 panel shows similarity_score distribution (bar chart, 1–5).
    Exp2 panel shows overall_quality distribution (bar chart, 1–5).

    Parameters
    ----------
    df:
        Combined DataFrame from ``curves.load_results``.
    dimension:
        Degradation dimension to examine.
    models:
        Models to include (default: all models in df).
    exclude_perfect:
        When True (default), magnitude-0 stimuli are excluded.
    figsize_per_panel:
        (width, height) per (model, experiment) panel.
    """
    if models is None:
        models = sorted(df["model"].unique().tolist())

    sub_all = df[df["degradation_dimension"] == dimension].copy()
    if exclude_perfect:
        sub_all = sub_all[sub_all["numeric_magnitude"] > 0]

    experiments = [
        ("experiment_1", "similarity_score", "Exp1: similarity score"),
        ("experiment_2", "overall_quality",  "Exp2: overall quality"),
    ]

    n_rows = len(models)
    n_cols = len(experiments)
    fig_w  = figsize_per_panel[0] * n_cols
    fig_h  = figsize_per_panel[1] * n_rows
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(fig_w, fig_h),
                              squeeze=False, sharey=False)

    score_bins = [1, 2, 3, 4, 5]

    for row_i, model in enumerate(models):
        for col_j, (exp, score_col, exp_label) in enumerate(experiments):
            ax = axes[row_i][col_j]
            sub = sub_all[
                (sub_all["model"] == model)
                & (sub_all["experiment"] == exp)
                & sub_all["parse_success"]
                & sub_all[score_col].notna()
            ]

            if sub.empty:
                ax.text(0.5, 0.5, "no data", ha="center", va="center",
                        transform=ax.transAxes, fontsize=9, color="gray")
                ax.set_title(f"{model}\n{exp_label}", fontsize=8)
                ax.set_xlabel("Score (1–5)", fontsize=8)
                ax.set_ylabel("Count", fontsize=8)
                continue

            counts = sub[score_col].value_counts().reindex(score_bins, fill_value=0)
            bars = ax.bar(score_bins, counts.values,
                          color="steelblue", edgecolor="white", linewidth=0.6)
            for bar, cnt in zip(bars, counts.values):
                if cnt > 0:
                    ax.text(bar.get_x() + bar.get_width() / 2,
                            bar.get_height() + 0.3,
                            str(cnt), ha="center", va="bottom", fontsize=7)

            ax.set_xticks(score_bins)
            ax.set_xlabel("Score (1–5)", fontsize=8)
            ax.set_ylabel("Count", fontsize=8)
            ax.set_title(f"{model}\n{exp_label}", fontsize=8)
            ax.grid(axis="y", alpha=0.3)

    fig.suptitle(
        f"Score distributions — dimension: {dimension}" +
        (" (mag > 0)" if exclude_perfect else ""),
        fontsize=10,
    )
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Detection sensitivity forest plot (dimension × model)
# ---------------------------------------------------------------------------

_MODEL_COLORS = [
    "#2196F3",   # blue
    "#FF5722",   # deep orange
    "#4CAF50",   # green
    "#9C27B0",   # purple
    "#FF9800",   # amber
    "#00BCD4",   # cyan
]


def plot_blind_sensitive_heatmap(
    blind_sensitive_table: dict,
    *,
    figsize: tuple[float, float] | None = None,
) -> plt.Figure:
    """Forest plot of detection sensitivity margin Δ(d,m) per (dimension, model).

    Each row = degradation dimension; within a row, one point per model shows
    Δ(d,m) = DetectionRate(d,m) − FPR_m with 95% Newcombe CI error bars.
    A vertical reference line at Δ = 0 marks the blind-spot boundary.
    Dimensions are ordered by ascending mean Δ (most blind at bottom, most
    sensitive at top).  Points whose CI excludes zero (after BH correction)
    are coloured green (sensitive); points whose CI overlaps zero are coloured
    red (blind); unknown cells are grey.

    Parameters
    ----------
    blind_sensitive_table:
        The ``sensitivity_rank.blind_sensitive_table`` dict returned by
        ``compute_all_stats()``.  Each cell must contain:
            classification: str  ("sensitive" | "blind" | "unknown")
            delta:     float | None   (DetectionRate − FPR)
            ci_lower:  float | None   (Newcombe 95% CI lower)
            ci_upper:  float | None   (Newcombe 95% CI upper)
    """
    if not blind_sensitive_table:
        fig, ax = plt.subplots()
        ax.text(0.5, 0.5, "No blind/sensitive data", ha="center", va="center",
                transform=ax.transAxes)
        return fig

    models = sorted({m for d in blind_sensitive_table.values() for m in d.keys()})
    if not models:
        fig, ax = plt.subplots()
        ax.text(0.5, 0.5, "No models in table", ha="center", va="center",
                transform=ax.transAxes)
        return fig

    # Sort dimensions by ascending mean Δ (most blind at bottom → top is most sensitive)
    def _mean_delta(dim: str) -> float:
        deltas = [
            v.get("delta")
            for v in blind_sensitive_table.get(dim, {}).values()
            if isinstance(v, dict) and v.get("delta") is not None
        ]
        return float(np.mean(deltas)) if deltas else float("nan")

    dims = sorted(blind_sensitive_table.keys(), key=_mean_delta)

    if figsize is None:
        figsize = (max(6, len(models) * 1.6 + 4), max(4, len(dims) * 0.55 + 1.5))

    fig, ax = plt.subplots(figsize=figsize)

    n_models = len(models)
    # Vertical offsets so points from different models don't overlap within a row
    offsets = np.linspace(-0.3, 0.3, n_models) if n_models > 1 else [0.0]

    for m_idx, model in enumerate(models):
        color = _MODEL_COLORS[m_idx % len(_MODEL_COLORS)]
        offset = offsets[m_idx]

        x_vals, y_vals, x_lo_err, x_hi_err = [], [], [], []
        x_blind, y_blind = [], []
        x_unknown, y_unknown = [], []

        for row_i, dim in enumerate(dims):
            cell = blind_sensitive_table.get(dim, {}).get(model, {})
            if not isinstance(cell, dict):
                x_unknown.append(0.0)
                y_unknown.append(row_i + offset)
                continue

            cls    = cell.get("classification", "unknown")
            delta  = cell.get("delta")
            ci_lo  = cell.get("ci_lower")
            ci_hi  = cell.get("ci_upper")

            if delta is None:
                x_unknown.append(0.0)
                y_unknown.append(row_i + offset)
                continue

            if cls == "sensitive":
                x_vals.append(delta)
                y_vals.append(row_i + offset)
                x_lo_err.append(delta - (ci_lo if ci_lo is not None else delta))
                x_hi_err.append((ci_hi if ci_hi is not None else delta) - delta)
            elif cls == "blind":
                x_blind.append(delta)
                y_blind.append(row_i + offset)
                # Still draw CI for blind points (thinner, muted)
                ax.errorbar(
                    [delta], [row_i + offset],
                    xerr=[[delta - (ci_lo if ci_lo is not None else delta)],
                          [(ci_hi if ci_hi is not None else delta) - delta]],
                    fmt="none", ecolor=color, elinewidth=0.8, alpha=0.45,
                )
            else:
                x_unknown.append(0.0)
                y_unknown.append(row_i + offset)

        # Sensitive: filled marker with CI bars
        if x_vals:
            ax.errorbar(
                x_vals, y_vals,
                xerr=[x_lo_err, x_hi_err],
                fmt="o", color=color, markersize=6, linewidth=0,
                ecolor=color, elinewidth=1.5, capsize=3,
                label="_nolegend_",
            )
        # Blind: open marker
        if x_blind:
            ax.scatter(x_blind, y_blind, marker="o", facecolors="none",
                       edgecolors=color, s=36, linewidths=1.2)
        # Unknown: grey x
        if x_unknown:
            ax.scatter(x_unknown, y_unknown, marker="x", color="#aaaaaa",
                       s=24, linewidths=0.8)

    # One legend entry per model (use a proxy artist)
    legend_handles = []
    for m_idx, model in enumerate(models):
        color = _MODEL_COLORS[m_idx % len(_MODEL_COLORS)]
        legend_handles.append(
            plt.Line2D([0], [0], marker="o", color="w", markerfacecolor=color,
                       markeredgecolor=color, markersize=7, label=model)
        )
    # Classification legend entries
    legend_handles += [
        plt.Line2D([0], [0], marker="o", color="w", markerfacecolor="#555555",
                   markeredgecolor="#555555", markersize=7, label="sensitive (CI > 0)"),
        plt.Line2D([0], [0], marker="o", color="w", markerfacecolor="none",
                   markeredgecolor="#555555", markersize=7, label="blind (CI overlaps 0)"),
        plt.Line2D([0], [0], marker="x", color="#aaaaaa", markersize=7,
                   linewidth=0, label="unknown"),
    ]
    ax.legend(handles=legend_handles, loc="lower right", fontsize=8,
              framealpha=0.85, ncol=1)

    # Reference line at Δ = 0
    ax.axvline(0.0, color="black", linewidth=1.2, linestyle="--", alpha=0.7)

    ax.set_yticks(range(len(dims)))
    ax.set_yticklabels([d.replace("_", " ") for d in dims], fontsize=9)
    ax.set_xlabel("Sensitivity margin Δ = DetectionRate − FPR  (95% Newcombe CI)", fontsize=9)
    ax.set_xlim(-1.05, 1.05)
    ax.set_title(
        "Detection sensitivity margin per (dimension, model)\n"
        "filled = sensitive (BH-corrected p < 0.05), open = blind, × = unknown",
        fontsize=10,
    )
    ax.grid(axis="x", alpha=0.25)
    fig.tight_layout()
    return fig
