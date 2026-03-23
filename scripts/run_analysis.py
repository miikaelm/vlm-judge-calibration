#!/usr/bin/env python3
"""
run_analysis.py — Produce analysis figures to outputs/figures/.

Pass whichever result files you have; only the relevant artifacts are generated.

Usage:
    # Only perceptual sensitivity (Exp1) — detection heatmaps
    python scripts/run_analysis.py --results-exp1 data/results_exp1.jsonl --manifest data/full/

    # Only instruction-following (Exp2) — sensitivity curves + score heatmaps
    python scripts/run_analysis.py --results-exp2 data/results_exp2.jsonl --manifest data/full/

    # Both — all of the above plus Exp1-vs-Exp2 gap curves
    python scripts/run_analysis.py \\
        --results-exp1 data/results_exp1.jsonl \\
        --results-exp2 data/results_exp2.jsonl \\
        --manifest data/full/
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


def _load_combined(
    results_exp1: Path | None,
    results_exp2: Path | None,
    manifest_dir: Path,
):
    import pandas as pd
    frames = []
    if results_exp1 is not None:
        frames.append(load_results(results_exp1, manifest_dir))
    if results_exp2 is not None:
        frames.append(load_results(results_exp2, manifest_dir))
    return pd.concat(frames, ignore_index=True)


def run_analysis(
    results_exp1: Path | None,
    results_exp2: Path | None,
    manifest_dir: Path,
    output_dir: Path,
    model_name: str | None,
) -> None:
    if results_exp1 is None and results_exp2 is None:
        print("ERROR: supply at least one of --results-exp1 / --results-exp2.")
        sys.exit(1)

    sources = [p for p in (results_exp1, results_exp2) if p is not None]
    print(f"Loading results from: {', '.join(str(p) for p in sources)} ...")
    df = _load_combined(results_exp1, results_exp2, manifest_dir)
    print(f"  {len(df)} records loaded ({df['stimulus_id'].nunique()} stimuli, "
          f"{df['model'].nunique()} model(s))")

    if df.empty:
        print("ERROR: no data — nothing to plot.")
        sys.exit(1)

    has_exp1 = "experiment_1" in df["experiment"].values
    has_exp2 = "experiment_2" in df["experiment"].values

    available_dims = set(df["degradation_dimension"].unique())

    # If a display name is given, relabel the model column so all plot
    # functions see the name the user provided — no path inference.
    if model_name:
        df["model"] = model_name
        vlms = [model_name]
    else:
        vlms = sorted(df["model"].unique())

    print(f"\nGenerating figures for model(s): {vlms}")
    print(f"Experiments present: {'exp1 ' if has_exp1 else ''}{'exp2' if has_exp2 else ''}")
    print(f"Output directory: {output_dir}\n")

    # One model → figures go directly into output_dir.
    # Multiple models → one subdir per model name inside output_dir.
    def vlm_dir(name: str) -> Path:
        if len(vlms) == 1:
            return output_dir
        return output_dir / name.replace("-", "_").replace(".", "_")

    for vlm in vlms:
        vdir = vlm_dir(vlm)

        # ------------------------------------------------------------------
        # 1. Sensitivity curves (Exp2 overall_quality, one per dimension)
        # ------------------------------------------------------------------
        if has_exp2:
            print(f"[{vlm}] Sensitivity curves ...")
            for dim in _ALL_DIMENSIONS:
                if dim not in available_dims:
                    continue
                fig = plot_sensitivity_curve(df, dim, vlm)
                save_fig(fig, vdir / "sensitivity" / f"{dim}.png")

            # ----------------------------------------------------------------
            # 2. Sensitivity curves for additional Exp2 score columns
            # ----------------------------------------------------------------
            for score_col in _EXP2_SCORE_COLS[1:]:  # skip overall_quality (done above)
                for dim in _ALL_DIMENSIONS:
                    if dim not in available_dims:
                        continue
                    fig = plot_sensitivity_curve(df, dim, vlm, exp2_score=score_col)
                    save_fig(fig, vdir / "sensitivity" / score_col / f"{dim}.png")

        # ------------------------------------------------------------------
        # 3. Exp1 vs Exp2 gap curves — only when both experiments are present
        # ------------------------------------------------------------------
        if has_exp1 and has_exp2:
            print(f"[{vlm}] Exp1 vs Exp2 gap curves ...")
            for dim in _ALL_DIMENSIONS:
                if dim not in available_dims:
                    continue
                fig = plot_exp_gap(df, dim, vlm=vlm)
                save_fig(fig, vdir / "exp_gap" / f"{dim}.png")

        # ------------------------------------------------------------------
        # 4. Heatmaps
        # ------------------------------------------------------------------
        if has_exp1:
            print(f"[{vlm}] Detection heatmap (Exp1) ...")
            fig = plot_detection_heatmap(df, vlm)
            save_fig(fig, vdir / "heatmaps" / "detection_rate.png")

        if has_exp2:
            print(f"[{vlm}] Score heatmaps (Exp2) ...")
            for score_col in _EXP2_SCORE_COLS:
                fig = plot_score_heatmap(df, vlm, score_col=score_col)
                save_fig(fig, vdir / "heatmaps" / f"score_{score_col}.png")

    # ------------------------------------------------------------------
    # 5. Cross-model comparison heatmaps (if multiple models, Exp2 needed)
    # ------------------------------------------------------------------
    if len(vlms) > 1 and has_exp2:
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
        description="Generate analysis figures from VLM evaluation results.",
        epilog="Supply at least one of --results-exp1 / --results-exp2.",
    )
    parser.add_argument(
        "--results-exp1",
        type=Path,
        default=None,
        metavar="PATH",
        help="Exp1 (perceptual sensitivity) results JSONL. "
             "Enables: detection heatmaps, Exp1-vs-Exp2 gap curves (if exp2 also given).",
    )
    parser.add_argument(
        "--results-exp2",
        type=Path,
        default=None,
        metavar="PATH",
        help="Exp2 (instruction-following) results JSONL. "
             "Enables: sensitivity curves, score heatmaps, cross-model comparison.",
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
        "--model-name",
        default=None,
        metavar="NAME",
        help="Display name for the model (used in graph titles and output dirs). "
             "If omitted, the raw value from the results 'model' column is used.",
    )
    args = parser.parse_args()

    if args.results_exp1 is None and args.results_exp2 is None:
        parser.error("supply at least one of --results-exp1 / --results-exp2")

    run_analysis(
        results_exp1=args.results_exp1.resolve() if args.results_exp1 else None,
        results_exp2=args.results_exp2.resolve() if args.results_exp2 else None,
        manifest_dir=args.manifest.resolve(),
        output_dir=args.output.resolve(),
        model_name=args.model_name,
    )


if __name__ == "__main__":
    main()
