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

from analysis.curves import (
    load_results,
    load_noop_results,
    load_perfect_results,
    plot_sensitivity_curve,
    plot_sensitivity_curve_exp1,
    plot_exp_gap,
    plot_psychometric_curve,
)
from analysis.heatmap import (
    plot_detection_heatmap_by_dim,
    plot_detection_rate_heatmap_dim_by_model,
    plot_score_heatmap,
    plot_perfect_score_heatmap,
    plot_noop_score_heatmap,
    plot_score_distributions,
    plot_blind_sensitive_heatmap,
)
from analysis.stats import compute_all_stats


# ---------------------------------------------------------------------------
# Dimension groups for batch figure generation
# ---------------------------------------------------------------------------

_CONTINUOUS_DIMENSIONS = [
    "color_offset",
    "position_offset",
    "alignment_error",
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
    "font_weight_light",
    "font_weight_heavy",
    "font_style",
    "font_family",
    "word_error",
    "case_error",
    "content_swap",
]

_ALL_DIMENSIONS = _CONTINUOUS_DIMENSIONS + _DISCRETE_DIMENSIONS

_EXP_GAP_DIMENSIONS = list(_ALL_DIMENSIONS)

_EXP2_SCORE_COLS = [
    "narrow_mean",          # primary cross-model aggregate (computed column)
    "overall_quality",      # raw holistic score — within-model use only
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


def _load_noop_combined(
    results_exp1: Path | None,
    results_exp2: Path | None,
    manifest_dir: Path,
):
    import pandas as pd
    frames = []
    if results_exp1 is not None:
        frames.append(load_noop_results(results_exp1, manifest_dir))
    if results_exp2 is not None:
        frames.append(load_noop_results(results_exp2, manifest_dir))
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def _load_perfect_combined(
    results_exp1: Path | None,
    results_exp2: Path | None,
    manifest_dir: Path,
):
    import pandas as pd
    frames = []
    if results_exp1 is not None:
        frames.append(load_perfect_results(results_exp1, manifest_dir))
    if results_exp2 is not None:
        frames.append(load_perfect_results(results_exp2, manifest_dir))
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def run_analysis(
    results_exp1: Path | None,
    results_exp2: Path | None,
    manifest_dir: Path,
    output_dir: Path,
    model_name: str | None,
    gap_examples: bool = False,
) -> None:
    output_dir = output_dir.resolve()

    if results_exp1 is None and results_exp2 is None:
        print("ERROR: supply at least one of --results-exp1 / --results-exp2.")
        sys.exit(1)

    sources = [p for p in (results_exp1, results_exp2) if p is not None]
    print(f"Loading results from: {', '.join(str(p) for p in sources)} ...")
    df         = _load_combined(results_exp1, results_exp2, manifest_dir)
    noop_df    = _load_noop_combined(results_exp1, results_exp2, manifest_dir)
    perfect_df = _load_perfect_combined(results_exp1, results_exp2, manifest_dir)
    print(f"  {len(df)} records loaded ({df['stimulus_id'].nunique()} stimuli, "
          f"{df['model'].nunique()} model(s))")
    if not noop_df.empty:
        print(f"  {len(noop_df)} noop records loaded")
    if not perfect_df.empty:
        print(f"  {len(perfect_df)} perfect-edit records loaded")

    if df.empty:
        print("ERROR: no data — nothing to plot.")
        sys.exit(1)

    has_exp1 = "experiment_1" in df["experiment"].values
    has_exp2 = "experiment_2" in df["experiment"].values

    available_dims = set(df["degradation_dimension"].unique())

    if model_name:
        df["model"] = model_name
        if not noop_df.empty:
            noop_df["model"] = model_name
        if not perfect_df.empty:
            perfect_df["model"] = model_name
        vlms = [model_name]
    else:
        vlms = sorted(df["model"].unique())

    print(f"\nGenerating figures for model(s): {vlms}")
    print(f"Experiments present: {'exp1 ' if has_exp1 else ''}{'exp2' if has_exp2 else ''}")
    print(f"Output directory: {output_dir}\n")

    def vlm_dir(name: str) -> Path:
        if len(vlms) == 1:
            return output_dir
        safe = name.replace("/", "_").replace("\\", "_").replace("-", "_").replace(".", "_")
        return output_dir / safe

    # ------------------------------------------------------------------
    # Compute full stats bundle (needed for blind/sensitive heatmap)
    # ------------------------------------------------------------------
    stats = None
    if has_exp1 or has_exp2:
        print("Computing statistics bundle ...")
        try:
            stats = compute_all_stats(
                df,
                noop_df=noop_df if not noop_df.empty else None,
                perfect_df=perfect_df if not perfect_df.empty else None,
            )
            print("  done.\n")
        except Exception as exc:
            print(f"  WARNING: stats computation failed: {exc}\n")

    for vlm in vlms:
        vdir = vlm_dir(vlm)

        # ------------------------------------------------------------------
        # 1. Sensitivity curves — per VLM
        # ------------------------------------------------------------------
        if has_exp1:
            print(f"[{vlm}] Sensitivity curves (Exp1 — similarity score) ...")
            for dim in _ALL_DIMENSIONS:
                if dim not in available_dims:
                    continue
                fig = plot_sensitivity_curve_exp1(df, dim, vlm)
                save_fig(fig, vdir / "sensitivity" / "exp1" / f"{dim}.png")

        if has_exp2:
            # Primary sensitivity curves use overall_quality (within-model rank analysis).
            print(f"[{vlm}] Sensitivity curves (Exp2 — overall_quality, within-model) ...")
            for dim in _ALL_DIMENSIONS:
                if dim not in available_dims:
                    continue
                fig = plot_sensitivity_curve(df, dim, vlm)
                save_fig(fig, vdir / "sensitivity" / f"{dim}.png")

            # Per-score-col sensitivity curves (includes narrow_mean)
            for score_col in _EXP2_SCORE_COLS:
                if score_col == "overall_quality":
                    continue  # already generated above as the default
                for dim in _ALL_DIMENSIONS:
                    if dim not in available_dims:
                        continue
                    fig = plot_sensitivity_curve(df, dim, vlm, exp2_score=score_col)
                    save_fig(fig, vdir / "sensitivity" / score_col / f"{dim}.png")

        # ------------------------------------------------------------------
        # 2. Exp1 vs Exp2 gap curves
        # ------------------------------------------------------------------
        if has_exp1 and has_exp2:
            print(f"[{vlm}] Exp1 vs Exp2 gap curves ...")
            for dim in _EXP_GAP_DIMENSIONS:
                if dim not in available_dims:
                    continue
                fig = plot_exp_gap(
                    df, dim, vlm=vlm,
                    manifest_dir=manifest_dir if gap_examples else None,
                )
                save_fig(fig, vdir / "exp_gap" / f"{dim}.png")

        # ------------------------------------------------------------------
        # 3. Per-VLM detection heatmap (single model, edit_type × dimension)
        # ------------------------------------------------------------------
        if has_exp1:
            print(f"[{vlm}] Detection heatmap by dimension (Exp1) ...")
            fig = plot_detection_heatmap_by_dim(df, vlm)
            save_fig(fig, vdir / "heatmaps" / "detection_rate_by_dim.png")

        # ------------------------------------------------------------------
        # 4. Score heatmaps (Exp2)
        # ------------------------------------------------------------------
        if has_exp2:
            print(f"[{vlm}] Score heatmaps (Exp2) ...")
            for score_col in _EXP2_SCORE_COLS:
                fig = plot_score_heatmap(df, vlm, score_col=score_col)
                save_fig(fig, vdir / "heatmaps" / f"score_{score_col}.png")

        # ------------------------------------------------------------------
        # 5. Perfect-edit score heatmap (Exp2)
        # ------------------------------------------------------------------
        if has_exp2 and not perfect_df.empty:
            print(f"[{vlm}] Perfect-edit score heatmap (Exp2) ...")
            fig = plot_perfect_score_heatmap(perfect_df, vlm)
            save_fig(fig, vdir / "heatmaps" / "perfect_scores.png")

        # ------------------------------------------------------------------
        # 6. Noop score heatmap (Exp2)
        # ------------------------------------------------------------------
        if not noop_df.empty:
            noop_has_exp2 = "experiment_2" in noop_df["experiment"].values
            noop_vlm      = noop_df[noop_df["model"] == vlm]
            if noop_has_exp2 and not noop_vlm[noop_vlm["experiment"] == "experiment_2"].empty:
                print(f"[{vlm}] Noop score heatmap (Exp2) ...")
                fig = plot_noop_score_heatmap(noop_vlm, vlm)
                save_fig(fig, vdir / "heatmaps" / "noop_scores.png")

        # ------------------------------------------------------------------
        # 7. Score distribution histograms — per dimension
        # ------------------------------------------------------------------
        if has_exp1 or has_exp2:
            print(f"[{vlm}] Score distribution histograms ...")
            for dim in _ALL_DIMENSIONS:
                if dim not in available_dims:
                    continue
                fig = plot_score_distributions(df, dim, models=[vlm])
                save_fig(fig, vdir / "distributions" / f"{dim}.png")

    # ------------------------------------------------------------------
    # 8. Multi-model figures — only produced when multiple models present
    # ------------------------------------------------------------------

    # Detection rate heatmap: dimension × model
    if has_exp1:
        print("\nDetection rate heatmap (dimension × model) ...")
        fig = plot_detection_rate_heatmap_dim_by_model(df, models=vlms)
        save_fig(fig, output_dir / "heatmaps" / "detection_rate_dim_by_model.png")

    # Psychometric curves: detection rate + OQ, all models on one axes
    if has_exp1 or has_exp2:
        print("Psychometric curves (multi-model) ...")
        for dim in _ALL_DIMENSIONS:
            if dim not in available_dims:
                continue
            fig = plot_psychometric_curve(df, dim, models=vlms)
            save_fig(fig, output_dir / "psychometric" / f"{dim}.png")

    # Blind/sensitive binary heatmap
    if stats is not None:
        bst = stats.get("sensitivity_rank", {}).get("blind_sensitive_table")
        if bst:
            print("Blind/sensitive heatmap ...")
            fig = plot_blind_sensitive_heatmap(bst)
            save_fig(fig, output_dir / "heatmaps" / "blind_sensitive.png")

    # Cross-model narrow_mean comparison (primary aggregate — cross-model comparable)
    if len(vlms) > 1 and has_exp2:
        print("\nCross-model comparison (narrow_mean) ...")
        _plot_cross_model_comparison(df, vlms, output_dir)

    print("\nDone.")


def _plot_cross_model_comparison(
    df: "pd.DataFrame",
    vlms: list[str],
    output_dir: Path,
) -> None:
    """Plot side-by-side narrow_mean heatmaps for each model.

    Uses narrow_mean (mean of instruction_following, text_accuracy,
    visual_consistency, layout_preservation) as the cross-model comparable
    quality aggregate.  overall_quality is NOT used here because its holistic
    prompt scope is resolved differently by each model (noop divergence >2.5 pts).
    """
    import matplotlib.pyplot as plt
    import numpy as np
    from analysis.heatmap import _present_dims, _present_edit_types, _DIMENSION_ORDER, _EDIT_TYPE_ORDER

    dims       = _present_dims(df, _DIMENSION_ORDER)
    edit_types = _present_edit_types(df, _EDIT_TYPE_ORDER)

    n_models = len(vlms)
    fig, axes = plt.subplots(
        1, n_models,
        figsize=(max(6, len(dims) * 0.9) * n_models, max(4, len(edit_types) * 0.9)),
        squeeze=False,
    )

    for col, vlm in enumerate(vlms):
        ax  = axes[0][col]
        sub = df[
            (df["model"] == vlm)
            & (df["experiment"] == "experiment_2")
            & df["parse_success"]
            & df["narrow_mean"].notna()
        ]
        matrix = np.full((len(edit_types), len(dims)), np.nan)
        for i, et in enumerate(edit_types):
            for j, dim in enumerate(dims):
                cell = sub[(sub["edit_type"] == et) & (sub["degradation_dimension"] == dim)]
                if not cell.empty:
                    matrix[i, j] = cell["narrow_mean"].mean()

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

    fig.suptitle("Cross-model: Exp2 narrow_mean (mean of 4 narrow dims)", fontsize=12)
    fig.tight_layout()
    save_fig(fig, output_dir / "cross_model_comparison.png")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate analysis figures from VLM evaluation results.",
        epilog="Supply at least one of --results-exp1 / --results-exp2.",
    )
    parser.add_argument(
        "--results-exp1", type=Path, default=None, metavar="PATH",
        help="Exp1 (perceptual sensitivity) results JSONL.",
    )
    parser.add_argument(
        "--results-exp2", type=Path, default=None, metavar="PATH",
        help="Exp2 (instruction-following) results JSONL.",
    )
    parser.add_argument(
        "--manifest", type=Path, default=Path("data/full/"),
        help="Stimulus manifest directory (default: data/full/)",
    )
    parser.add_argument(
        "--output", type=Path, default=Path("outputs/figures/"),
        help="Output directory for figures (default: outputs/figures/)",
    )
    parser.add_argument(
        "--model-name", default=None, metavar="NAME",
        help="Display name for the model (used in graph titles and output dirs).",
    )
    parser.add_argument(
        "--gap-examples", action="store_true", default=False,
        help="Append example stimulus panels (images + scores) below each exp-gap curve. "
             "Slow to generate; off by default.",
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
        gap_examples=args.gap_examples,
    )


if __name__ == "__main__":
    main()
