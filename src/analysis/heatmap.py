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
    """
    sub = df[
        (df["model"] == vlm)
        & (df["experiment"] == experiment)
        & df["parse_success"]
        & df["detected_difference"].notna()
    ].copy()
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

    ax.set_title(
        f"Exp1 detection rate — model: {vlm}\n"
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
    """
    sub = df[
        (df["model"] == vlm)
        & (df["experiment"] == experiment)
        & df["parse_success"]
        & df[score_col].notna()
    ].copy()

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

    ax.set_title(
        f"Exp2 mean {score_col.replace('_', ' ')} — model: {vlm}",
        fontsize=10,
    )
    ax.set_xlabel("Degradation dimension", fontsize=9)
    ax.set_ylabel("Edit type", fontsize=9)
    fig.tight_layout()
    return fig
