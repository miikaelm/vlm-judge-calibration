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
    "gaussian_noise",
    "jpeg_compression",
    "blur",
]

_EDIT_TYPE_ORDER = [
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
    df: pd.DataFrame,
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
    """
    score_cols = [c for c in _EXP2_SCORE_COLS if c in df.columns]
    sub = df[
        (df["model"] == vlm)
        & (df["experiment"] == experiment)
        & df["parse_success"]
        & (df["numeric_magnitude"] == 0)
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
