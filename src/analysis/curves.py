"""
curves.py — Sensitivity curves and Exp1/Exp2 gap analysis.

Public API:
    load_results(results_jsonl, manifest_dir) -> pd.DataFrame
    plot_sensitivity_curve(df, dimension, vlm) -> matplotlib.Figure
    plot_exp_gap(df, dimension) -> matplotlib.Figure
"""

from __future__ import annotations

import json
import textwrap
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
        manifest_file = manifest_dir / "manifest.jsonl"
        if manifest_file.exists():
            with open(manifest_file, encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        try:
                            m = json.loads(line)
                            key = m.get("stimulus_id") or m.get("id", "")
                            if key:
                                meta[key] = m
                        except json.JSONDecodeError:
                            pass
        else:
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
        x = grouped["numeric_magnitude"].values.astype(float)
        y = grouped["mean"].values.astype(float)
        yerr = np.nan_to_num(grouped["sem"].values.astype(float))
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
# Helpers for the Exp-gap example panels
# ---------------------------------------------------------------------------

def _load_manifest_index(manifest_dir: Path) -> dict:
    """Load manifest.jsonl into a dict keyed by stimulus id."""
    manifest_file = manifest_dir / "manifest.jsonl"
    if not manifest_file.exists():
        return {}
    index: dict = {}
    with open(manifest_file, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                m = json.loads(line)
                key = m.get("stimulus_id") or m.get("id", "")
                if key:
                    index[key] = m
            except json.JSONDecodeError:
                pass
    return index


def _pick_example_stimuli(
    df: pd.DataFrame,
    dimension: str,
    vlm: str | None,
    manifest_index: dict,
    n: int = 10,
) -> list:
    """Pick n stimuli for the given dimension.

    Always includes at least one example from each key scale (low / medium /
    high magnitude) when enough stimuli are available.  The remaining slots are
    filled by a random sample from the leftover pool.  The final list is sorted
    by ascending magnitude for display.

    Returns a list of dicts with keys:
        stimulus_id, magnitude_label, numeric_magnitude,
        instruction, source_image, ground_truth_image, degraded_image,
        exp1 (dict), exp2 (dict)
    """
    import random

    filt = df["degradation_dimension"] == dimension
    if vlm is not None:
        filt &= df["model"] == vlm

    sub_e1 = df[filt & (df["experiment"] == "experiment_1")].copy()
    sub_e2 = df[filt & (df["experiment"] == "experiment_2")].copy()

    all_sids = set(sub_e1["stimulus_id"]).union(sub_e2["stimulus_id"])
    sids_with_meta = {sid for sid in all_sids if sid in manifest_index}
    if not sids_with_meta:
        return []

    # Build magnitude map — prefer exp2 row, fall back to exp1
    mag_map: dict = {}
    for sid in sids_with_meta:
        row = sub_e2[sub_e2["stimulus_id"] == sid]
        if row.empty:
            row = sub_e1[sub_e1["stimulus_id"] == sid]
        if not row.empty:
            mag_map[sid] = float(row["numeric_magnitude"].iloc[0])

    if not mag_map:
        return []

    sids_sorted = sorted(mag_map, key=lambda s: mag_map[s])
    total = len(sids_sorted)

    # Guaranteed anchor indices: lowest, median, highest magnitude
    if total == 1:
        anchor_indices = [0]
    elif total == 2:
        anchor_indices = [0, total - 1]
    else:
        anchor_indices = [0, total // 2, total - 1]

    anchor_sids = [sids_sorted[i] for i in anchor_indices]
    remaining = [s for s in sids_sorted if s not in set(anchor_sids)]

    # Fill up to n with random draws from the leftovers
    n_extra = max(0, n - len(anchor_sids))
    extra_sids = random.sample(remaining, min(n_extra, len(remaining)))

    chosen = sorted(set(anchor_sids) | set(extra_sids), key=lambda s: mag_map[s])

    # Assign magnitude labels by tertile
    mag_values = [mag_map[s] for s in chosen]
    if len(mag_values) > 1:
        lo_thresh = mag_values[len(mag_values) // 3]
        hi_thresh = mag_values[2 * len(mag_values) // 3]
    else:
        lo_thresh = hi_thresh = mag_values[0] if mag_values else 0.0

    def _mag_label(v: float) -> str:
        if v <= lo_thresh:
            return "low"
        if v >= hi_thresh:
            return "high"
        return "medium"

    results = []
    for sid in chosen:
        label = _mag_label(mag_map[sid])
        m = manifest_index[sid]

        e1_rows = sub_e1[sub_e1["stimulus_id"] == sid]
        e1: dict = {}
        if not e1_rows.empty:
            r = e1_rows.iloc[0]
            e1 = {
                "similarity_score": r.get("similarity_score"),
                "detected_difference": r.get("detected_difference"),
                "description": r.get("exp1_description") or "",
            }

        e2_rows = sub_e2[sub_e2["stimulus_id"] == sid]
        e2: dict = {}
        if not e2_rows.empty:
            r = e2_rows.iloc[0]
            e2 = {
                "instruction_following": r.get("instruction_following"),
                "text_accuracy": r.get("text_accuracy"),
                "visual_consistency": r.get("visual_consistency"),
                "layout_preservation": r.get("layout_preservation"),
                "overall_quality": r.get("overall_quality"),
                "errors_noticed": r.get("errors_noticed") or "",
            }

        results.append({
            "stimulus_id": sid,
            "magnitude_label": label,
            "numeric_magnitude": mag_map[sid],
            "instruction": m.get("edit_instruction", ""),
            "source_image": m.get("source_image", ""),
            "ground_truth_image": m.get("ground_truth_image", ""),
            "degraded_image": m.get("degraded_image", ""),
            "exp1": e1,
            "exp2": e2,
        })
    return results


def _format_example_text(ex: dict) -> str:
    """Format scores + instruction as a compact monospace text block."""
    lines = []

    instr = ex.get("instruction", "")
    lines.append("INSTRUCTION:")
    for chunk in textwrap.wrap(instr, width=42) or ["(none)"]:
        lines.append(f"  {chunk}")
    lines.append("")

    e1 = ex.get("exp1", {})
    lines.append("EXP 1  (perceptual sensitivity):")
    if e1:
        sim = e1.get("similarity_score")
        det = e1.get("detected_difference")
        lines.append(f"  Similarity     : {'—' if sim is None else f'{sim}/5'}")
        if det is None:
            det_str = "—"
        else:
            det_str = "Yes" if det else "No"
        lines.append(f"  Detected diff. : {det_str}")
        desc = str(e1.get("description") or "")
        if desc.strip():
            lines.append("  Comment:")
            for chunk in textwrap.wrap(desc, width=40):
                lines.append(f"    {chunk}")
    else:
        lines.append("  (no data)")
    lines.append("")

    e2 = ex.get("exp2", {})
    lines.append("EXP 2  (instruction-following):")
    if e2:
        for key, label in [
            ("instruction_following", "Instr. following"),
            ("text_accuracy",         "Text accuracy   "),
            ("visual_consistency",    "Visual consist. "),
            ("layout_preservation",   "Layout preserv. "),
            ("overall_quality",       "Overall quality "),
        ]:
            val = e2.get(key)
            score_str = f"{val}/5" if val is not None else "—"
            lines.append(f"  {label}: {score_str}")
        errors = str(e2.get("errors_noticed") or "")
        lines.append("")
        lines.append("  COMMENTS:")
        if errors.strip():
            for chunk in textwrap.wrap(errors, width=40):
                lines.append(f"    {chunk}")
        else:
            lines.append("    (none)")
    else:
        lines.append("  (no data)")

    return "\n".join(lines)


def _draw_gap_curve(
    ax: plt.Axes,
    exp1: pd.DataFrame,
    exp2: pd.DataFrame,
    exp2_score: str,
    dimension: str,
    vlm: str | None,
) -> None:
    """Draw the Exp1 vs Exp2 gap curve onto an existing Axes."""
    def _plot_series(data: pd.DataFrame, score_col: str, label: str, color: str, marker: str):
        if data.empty:
            return
        grouped = (
            data.groupby("numeric_magnitude")[score_col]
            .agg(["mean", "sem"])
            .reset_index()
            .sort_values("numeric_magnitude")
        )
        x = grouped["numeric_magnitude"].values.astype(float)
        y = grouped["mean"].values.astype(float)
        yerr = np.nan_to_num(grouped["sem"].values.astype(float))
        ax.plot(x, y, color=color, marker=marker, label=label, linewidth=2)
        ax.fill_between(x, y - yerr, y + yerr, alpha=0.15, color=color)

    _plot_series(exp1, "similarity_score", "Exp1: similarity score", "#2196F3", "o")
    _plot_series(exp2, exp2_score, f"Exp2: {exp2_score.replace('_', ' ')}", "#FF5722", "s")

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
            np.array(shared_mags, dtype=float),
            m2_mean.values.astype(float),
            m1_mean.values.astype(float),
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


def _add_example_panels(
    fig: plt.Figure,
    outer_gs,
    examples: list,
    manifest_dir: Path,
    start_row: int = 1,
) -> None:
    """Render example stimulus panels (images + scores) into outer_gs rows."""
    from matplotlib.image import imread as mpl_imread

    for i, ex in enumerate(examples):
        row_spec = outer_gs[start_row + i]
        inner_gs = row_spec.subgridspec(2, 1, height_ratios=[1, 9], hspace=0.1)

        # Header bar
        ax_hdr = fig.add_subplot(inner_gs[0])
        ax_hdr.axis("off")
        mag_val = ex["numeric_magnitude"]
        mag_str = f"{mag_val:.3g}" if mag_val != int(mag_val) else str(int(mag_val))
        ax_hdr.text(
            0.0, 0.5,
            f"  ▶  {ex['magnitude_label'].upper()} degradation  "
            f"│  stimulus: {ex['stimulus_id']}  │  magnitude = {mag_str}",
            transform=ax_hdr.transAxes, fontsize=8.5, fontweight="bold",
            va="center", color="#222222",
            bbox=dict(facecolor="#e8e8e8", edgecolor="none", boxstyle="square,pad=0.3"),
        )

        # Content: 3 images + score panel
        content_gs = inner_gs[1].subgridspec(1, 4, wspace=0.05, width_ratios=[1, 1, 1, 1.8])

        for j, (img_key, img_title) in enumerate([
            ("source_image",       "Source"),
            ("ground_truth_image", "Ground Truth"),
            ("degraded_image",     "Degraded"),
        ]):
            ax_img = fig.add_subplot(content_gs[j])
            img_rel = ex.get(img_key, "")
            loaded = False
            if img_rel:
                img_path = manifest_dir / img_rel
                if img_path.exists():
                    try:
                        img = mpl_imread(str(img_path))
                        ax_img.imshow(img)
                        loaded = True
                    except Exception:
                        pass
            if not loaded:
                ax_img.set_facecolor("#eeeeee")
                ax_img.text(0.5, 0.5, "Image\nnot found", ha="center", va="center",
                            transform=ax_img.transAxes, fontsize=7, color="#888888")
            ax_img.set_title(img_title, fontsize=8, pad=2)
            ax_img.axis("off")

        # Score / instruction text panel
        ax_txt = fig.add_subplot(content_gs[3])
        ax_txt.axis("off")
        ax_txt.text(
            0.04, 0.97,
            _format_example_text(ex),
            transform=ax_txt.transAxes,
            fontsize=6.5, va="top", ha="left",
            family="monospace",
            bbox=dict(
                facecolor="#f8f8f8", edgecolor="#cccccc",
                boxstyle="round,pad=0.5", linewidth=0.7,
            ),
        )


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
    manifest_dir: Path | None = None,
    n_examples: int = 10,
) -> plt.Figure:
    """Plot Exp1 vs Exp2 scores for the same stimuli, grouped by magnitude.

    Exp1 uses ``similarity_score`` (higher = more similar).
    Exp2 uses ``overall_quality`` (or ``exp2_score``).

    The gap between the curves quantifies how much harder judging is vs
    simply detecting a difference.

    When ``manifest_dir`` is given, example stimuli are appended below the
    curve (default 10), always including at least one from each key scale
    (low / medium / high magnitude), with the rest randomly sampled.
    """
    from matplotlib import gridspec as mgridspec

    filt = (df["degradation_dimension"] == dimension) & df["parse_success"]
    if vlm is not None:
        filt &= df["model"] == vlm

    exp1 = df[filt & (df["experiment"] == "experiment_1") & df["similarity_score"].notna()].copy()
    exp2 = df[filt & (df["experiment"] == "experiment_2") & df[exp2_score].notna()].copy()

    # Load examples when manifest directory is available
    examples: list = []
    if manifest_dir is not None:
        manifest_dir = Path(manifest_dir)
        manifest_index = _load_manifest_index(manifest_dir)
        examples = _pick_example_stimuli(df, dimension, vlm, manifest_index, n=n_examples)

    no_data = exp1.empty and exp2.empty

    if not examples:
        # Simple figure — original behaviour
        fig, ax = plt.subplots(figsize=figsize)
        if no_data:
            ax.text(0.5, 0.5, f"No data for dimension={dimension!r}",
                    ha="center", va="center", transform=ax.transAxes)
            ax.set_title(f"Exp1 vs Exp2 gap — {dimension}")
        else:
            _draw_gap_curve(ax, exp1, exp2, exp2_score, dimension, vlm)
        fig.tight_layout()
        return fig

    # Extended figure: curve on top + example panels below
    curve_h = float(figsize[1])
    example_h = 4.2
    fig_w = max(float(figsize[0]), 13.0)
    total_h = curve_h + len(examples) * example_h

    fig = plt.figure(figsize=(fig_w, total_h), layout="constrained")
    gs = mgridspec.GridSpec(
        1 + len(examples), 1,
        figure=fig,
        height_ratios=[curve_h] + [example_h] * len(examples),
        hspace=0.55,
    )

    ax = fig.add_subplot(gs[0])
    if no_data:
        ax.text(0.5, 0.5, f"No data for dimension={dimension!r}",
                ha="center", va="center", transform=ax.transAxes)
        ax.set_title(f"Exp1 vs Exp2 gap — {dimension}")
    else:
        _draw_gap_curve(ax, exp1, exp2, exp2_score, dimension, vlm)

    _add_example_panels(fig, gs, examples, manifest_dir, start_row=1)

    return fig
