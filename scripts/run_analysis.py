#!/usr/bin/env python3
"""
run_analysis.py — Produce all analysis figures to outputs/figures/.

Usage:
    python scripts/run_analysis.py
    python scripts/run_analysis.py --results data/results.jsonl --manifest data/full/
    python scripts/run_analysis.py --vlm gpt-4o --output outputs/figures/
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # non-interactive backend — safe for scripts
import matplotlib.pyplot as plt

# Allow running from repo root without editable install
_repo_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_repo_root / "src"))

from analysis.curves import load_results, plot_sensitivity_curve, plot_exp_gap
from analysis.heatmap import plot_detection_heatmap, plot_score_heatmap


# ---------------------------------------------------------------------------
# Dimension groups for batch figure generation
# ---------------------------------------------------------------------------

_CONTINUOUS_DIMENSIONS = [
    "color_offset",
    "position_offset",
    "scale_error",
    "rotation",
    "letter_spacing",
    "opacity",
    "char_substitution",
    "gaussian_noise",
    "jpeg_compression",
    "blur",
]

_DISCRETE_DIMENSIONS = [
    "font_weight",
    "font_style",
    "word_error",
    "case_error",
    "content_swap",
]

_ALL_DIMENSIONS = _CONTINUOUS_DIMENSIONS + _DISCRETE_DIMENSIONS

_EXP2_SCORE_COLS = [
    "overall_quality",
    "instruction_following",
    "text_accuracy",
    "visual_consistency",
    "layout_preservation",
]


def save_fig(fig: plt.Figure, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved: {path.relative_to(path.parent.parent.parent)}")


def run_analysis(
    results_jsonl: Path,
    manifest_dir: Path,
    output_dir: Path,
    vlms: list[str],
) -> None:
    print(f"Loading results from {results_jsonl} ...")
    df = load_results(results_jsonl, manifest_dir)
    print(f"  {len(df)} records loaded ({df['stimulus_id'].nunique()} stimuli, "
          f"{df['model'].nunique()} model(s))")

    if df.empty:
        print("ERROR: no data — nothing to plot.")
        sys.exit(1)

    available_dims = set(df["degradation_dimension"].unique())
    available_models = set(df["model"].unique())

    if not vlms:
        vlms = sorted(available_models)
    else:
        missing = set(vlms) - available_models
        if missing:
            print(f"WARNING: model(s) not in results: {missing}")
        vlms = [v for v in vlms if v in available_models]

    if not vlms:
        print("ERROR: no matching models in results.")
        sys.exit(1)

    print(f"\nGenerating figures for model(s): {vlms}")
    print(f"Output directory: {output_dir}\n")

    for vlm in vlms:
        vlm_slug = vlm.replace("-", "_").replace(".", "_")
        vlm_dir = output_dir / vlm_slug

        # ------------------------------------------------------------------
        # 1. Sensitivity curves (Exp2 overall_quality, one per dimension)
        # ------------------------------------------------------------------
        print(f"[{vlm}] Sensitivity curves ...")
        for dim in _ALL_DIMENSIONS:
            if dim not in available_dims:
                continue
            fig = plot_sensitivity_curve(df, dim, vlm)
            save_fig(fig, vlm_dir / "sensitivity" / f"{dim}.png")

        # ------------------------------------------------------------------
        # 2. Sensitivity curves for additional Exp2 score columns
        # ------------------------------------------------------------------
        for score_col in _EXP2_SCORE_COLS[1:]:  # skip overall_quality (done above)
            for dim in _ALL_DIMENSIONS:
                if dim not in available_dims:
                    continue
                fig = plot_sensitivity_curve(df, dim, vlm, exp2_score=score_col)
                save_fig(fig, vlm_dir / "sensitivity" / score_col / f"{dim}.png")

        # ------------------------------------------------------------------
        # 3. Exp1 vs Exp2 gap curves (one per dimension)
        # ------------------------------------------------------------------
        print(f"[{vlm}] Exp1 vs Exp2 gap curves ...")
        for dim in _ALL_DIMENSIONS:
            if dim not in available_dims:
                continue
            fig = plot_exp_gap(df, dim, vlm=vlm)
            save_fig(fig, vlm_dir / "exp_gap" / f"{dim}.png")

        # ------------------------------------------------------------------
        # 4. Heatmaps
        # ------------------------------------------------------------------
        print(f"[{vlm}] Heatmaps ...")
        fig = plot_detection_heatmap(df, vlm)
        save_fig(fig, vlm_dir / "heatmaps" / "detection_rate.png")

        for score_col in _EXP2_SCORE_COLS:
            fig = plot_score_heatmap(df, vlm, score_col=score_col)
            save_fig(fig, vlm_dir / "heatmaps" / f"score_{score_col}.png")

    # ------------------------------------------------------------------
    # 5. Cross-model comparison heatmaps (if multiple models)
    # ------------------------------------------------------------------
    if len(vlms) > 1:
        print("\nCross-model comparison ...")
        _plot_cross_model_comparison(df, vlms, output_dir)

    print("\nDone.")


def _plot_cross_model_comparison(
    df: "pd.DataFrame",
    vlms: list[str],
    output_dir: Path,
) -> None:
    """Plot side-by-side overall_quality heatmaps for each model."""
    import matplotlib.pyplot as plt
    import numpy as np
    from analysis.heatmap import _present_dims, _present_edit_types, _DIMENSION_ORDER, _EDIT_TYPE_ORDER

    dims = _present_dims(df, _DIMENSION_ORDER)
    edit_types = _present_edit_types(df, _EDIT_TYPE_ORDER)

    n_models = len(vlms)
    fig, axes = plt.subplots(
        1, n_models,
        figsize=(max(6, len(dims) * 0.9) * n_models, max(4, len(edit_types) * 0.9)),
        squeeze=False,
    )

    for col, vlm in enumerate(vlms):
        ax = axes[0][col]
        sub = df[
            (df["model"] == vlm)
            & (df["experiment"] == "experiment_2")
            & df["parse_success"]
            & df["overall_quality"].notna()
        ]
        matrix = np.full((len(edit_types), len(dims)), np.nan)
        for i, et in enumerate(edit_types):
            for j, dim in enumerate(dims):
                cell = sub[(sub["edit_type"] == et) & (sub["degradation_dimension"] == dim)]
                if not cell.empty:
                    matrix[i, j] = cell["overall_quality"].mean()

        im = ax.imshow(matrix, cmap="RdYlGn", vmin=1, vmax=5, aspect="auto")
        ax.set_xticks(range(len(dims)))
        ax.set_xticklabels([d.replace("_", "\n") for d in dims], fontsize=7)
        ax.set_yticks(range(len(edit_types)))
        ax.set_yticklabels([e.replace("_", " ") for e in edit_types], fontsize=8)
        plt.setp(ax.get_xticklabels(), rotation=30, ha="right")
        ax.set_title(vlm, fontsize=10)
        for i in range(len(edit_types)):
            for j in range(len(dims)):
                val = matrix[i, j]
                if not np.isnan(val):
                    ax.text(j, i, f"{val:.1f}", ha="center", va="center", fontsize=6)

    fig.suptitle("Cross-model: Exp2 overall_quality (mean)", fontsize=12)
    fig.tight_layout()
    save_fig(fig, output_dir / "cross_model_comparison.png")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate all analysis figures from VLM evaluation results."
    )
    parser.add_argument(
        "--results",
        type=Path,
        default=Path("data/results.jsonl"),
        help="Path to results.jsonl (default: data/results.jsonl)",
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("data/full/"),
        help="Stimulus manifest directory (default: data/full/)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("outputs/figures/"),
        help="Output directory for figures (default: outputs/figures/)",
    )
    parser.add_argument(
        "--vlm",
        action="append",
        dest="vlms",
        default=None,
        metavar="MODEL",
        help="Model name(s) to plot. Can be repeated. Default: all in results.",
    )
    args = parser.parse_args()

    run_analysis(
        results_jsonl=args.results,
        manifest_dir=args.manifest,
        output_dir=args.output,
        vlms=args.vlms or [],
    )


if __name__ == "__main__":
    main()
