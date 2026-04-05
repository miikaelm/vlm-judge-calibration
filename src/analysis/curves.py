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
        # "normal" is the most wrong degradation when the target was "italic"
        # (it is completely un-italic, not merely a subtle oblique variant).
        return {"italic": 1.0, "oblique": 0.5, "normal": 1.0}.get(
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

def _load_raw_results(
    results_jsonl: str | Path,
    manifest_dir: str | Path,
) -> pd.DataFrame:
    """Internal: load and join results + manifest without any row filtering."""
    results_jsonl = Path(results_jsonl)
    manifest_dir = Path(manifest_dir)

    records: list[dict] = []
    with open(results_jsonl, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError:
                    pass

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
            "edit_type": m.get("edit_type", "unknown"),
            "edit_id": m.get("edit_id", "unknown"),
            "template_id": tmpl.get("template_id", "unknown"),
            "difficulty_tier": tmpl.get("difficulty_tier", "unknown"),
            "degradation_dimension": dimension,
            "degradation_magnitude": deg.get("magnitude", "unknown"),
            "degradation_layer": deg.get("layer", "unknown"),
            "degradation_params": params,
            "numeric_magnitude": _params_to_numeric_magnitude(dimension, params),
            "detected_difference": r.get("detected_difference"),
            "similarity_score": r.get("similarity_score"),
            "exp1_description": r.get("description"),
            "instruction_following": r.get("instruction_following"),
            "text_accuracy": r.get("text_accuracy"),
            "visual_consistency": r.get("visual_consistency"),
            "layout_preservation": r.get("layout_preservation"),
            "overall_quality": r.get("overall_quality"),
            "errors_noticed": r.get("errors_noticed"),
        }
        rows.append(row)

    return pd.DataFrame(rows)


def load_results(
    results_jsonl: str | Path,
    manifest_dir: str | Path,
) -> pd.DataFrame:
    """Load results.jsonl and join with per-stimulus metadata.

    Noop records (``degradation_dimension == "noop"``) are excluded — use
    :func:`load_noop_results` to load those separately.

    Returns
    -------
    pd.DataFrame
        One row per result record, enriched with metadata columns:
        edit_type, edit_id, template_id, difficulty_tier,
        degradation_dimension, degradation_magnitude, degradation_layer,
        degradation_params, numeric_magnitude, model, experiment,
        parse_success, plus all score columns.
    """
    df = _load_raw_results(results_jsonl, manifest_dir)
    return df[df["degradation_dimension"] != "noop"].reset_index(drop=True)


def load_noop_results(
    results_jsonl: str | Path,
    manifest_dir: str | Path,
) -> pd.DataFrame:
    """Load only noop records from results.jsonl.

    Noop stimuli show the source image unchanged; the model should say there
    is no difference (Exp1) and score the edit low (Exp2) since it was never
    applied.  This data is kept separate for false-positive analysis.
    """
    df = _load_raw_results(results_jsonl, manifest_dir)
    return df[df["degradation_dimension"] == "noop"].reset_index(drop=True)


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

# Maps degradation dimension → the edit_type that is "in-dimension" for it.
# Dimensions absent from this map have no clear primary edit type; those
# samples are still included but all treated as secondary.
_DIM_TO_EDIT_TYPE: dict[str, str] = {
    "color_offset":    "color",
    "alignment_error": "relocation",
    "rotation":        "rotation",
    "scale_error":     "scale",
    "font_weight":     "font_weight",
    "font_style":      "italic",
    "letter_spacing":  "letter_spacing",
}


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
    n_secondary: int = 5,
) -> list:
    """Pick example stimuli for the given dimension.

    Strategy
    --------
    *In-dimension* samples (``sample_type="in_dimension"``):
        For dimensions that have a matching primary edit type (see
        ``_DIM_TO_EDIT_TYPE``), pick **one representative per unique
        numeric magnitude** where the edit type is that primary type.
        This ensures every severity level of the "correct" degradation
        is visible.

    *Secondary* samples (``sample_type="secondary"``):
        Up to ``n_secondary`` stimuli from the same dimension but a
        *different* edit type, stratified across magnitude (low / medium /
        high) where possible.

    For dimensions with no primary edit type all available stimuli are
    treated as secondary and the original anchor-plus-random selection
    is used.

    Returns a list of dicts with keys:
        stimulus_id, sample_type, edit_type, magnitude_label,
        numeric_magnitude, instruction, source_image, ground_truth_image,
        degraded_image, exp1 (dict), exp2 (dict)
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

    # Build magnitude map and edit_type map per stimulus
    mag_map: dict[str, float] = {}
    et_map: dict[str, str] = {}
    for sid in sids_with_meta:
        row = sub_e2[sub_e2["stimulus_id"] == sid]
        if row.empty:
            row = sub_e1[sub_e1["stimulus_id"] == sid]
        if not row.empty:
            mag_map[sid] = float(row["numeric_magnitude"].iloc[0])
            et_map[sid] = str(row["edit_type"].iloc[0])

    if not mag_map:
        return []

    primary_et = _DIM_TO_EDIT_TYPE.get(dimension)

    # ------------------------------------------------------------------ #
    # In-dimension: one sample per unique magnitude for the primary type  #
    # ------------------------------------------------------------------ #
    in_dim_sids: list[str] = []
    if primary_et is not None:
        # Group by magnitude; pick one sid per bucket (prefer exp2 data)
        mag_to_sids: dict[float, list[str]] = {}
        for sid, mag in mag_map.items():
            if et_map.get(sid) == primary_et:
                mag_to_sids.setdefault(mag, []).append(sid)

        for mag in sorted(mag_to_sids):
            bucket = mag_to_sids[mag]
            # Prefer a stimulus that has exp2 data
            has_e2 = [s for s in bucket
                      if not sub_e2[sub_e2["stimulus_id"] == s].empty]
            chosen_sid = has_e2[0] if has_e2 else bucket[0]
            in_dim_sids.append(chosen_sid)

    in_dim_set = set(in_dim_sids)

    # ------------------------------------------------------------------ #
    # Secondary: up to n_secondary from different edit types              #
    # ------------------------------------------------------------------ #
    secondary_pool = [
        sid for sid in mag_map
        if sid not in in_dim_set
        and mag_map[sid] > 0
        and (primary_et is None or et_map.get(sid) != primary_et)
    ]
    secondary_pool_sorted = sorted(secondary_pool, key=lambda s: mag_map[s])
    total_sec = len(secondary_pool_sorted)

    if primary_et is None:
        # No primary — fall back to anchor + random from the whole pool
        if total_sec == 1:
            anchor_indices = [0]
        elif total_sec == 2:
            anchor_indices = [0, total_sec - 1]
        else:
            anchor_indices = [0, total_sec // 2, total_sec - 1]
        anchor_sids = [secondary_pool_sorted[i] for i in anchor_indices]
        remaining = [s for s in secondary_pool_sorted if s not in set(anchor_sids)]
        extra = random.sample(remaining, min(max(0, n_secondary - len(anchor_sids)), len(remaining)))
        secondary_sids = sorted(set(anchor_sids) | set(extra), key=lambda s: mag_map[s])
    else:
        # Stratified by magnitude tertile
        if total_sec == 0:
            secondary_sids = []
        elif total_sec <= n_secondary:
            secondary_sids = secondary_pool_sorted
        else:
            lo = secondary_pool_sorted[: total_sec // 3]
            mid = secondary_pool_sorted[total_sec // 3 : 2 * total_sec // 3]
            hi = secondary_pool_sorted[2 * total_sec // 3 :]
            chosen_sec: list[str] = []
            for bucket in (lo, mid, hi):
                k = max(1, n_secondary // 3)
                chosen_sec.extend(random.sample(bucket, min(k, len(bucket))))
            # top up if slots remain
            picked = set(chosen_sec)
            remaining = [s for s in secondary_pool_sorted if s not in picked]
            while len(chosen_sec) < n_secondary and remaining:
                s = random.choice(remaining)
                chosen_sec.append(s)
                remaining.remove(s)
            secondary_sids = sorted(chosen_sec[:n_secondary], key=lambda s: mag_map[s])

    # ------------------------------------------------------------------ #
    # Assign magnitude labels                                             #
    # ------------------------------------------------------------------ #
    all_chosen = list(in_dim_sids) + [s for s in secondary_sids if s not in in_dim_set]
    all_mags = [mag_map[s] for s in all_chosen]
    if len(all_mags) > 1:
        lo_thresh = sorted(all_mags)[len(all_mags) // 3]
        hi_thresh = sorted(all_mags)[2 * len(all_mags) // 3]
    else:
        lo_thresh = hi_thresh = all_mags[0] if all_mags else 0.0

    def _mag_label(v: float) -> str:
        if v <= lo_thresh:
            return "low"
        if v >= hi_thresh:
            return "high"
        return "medium"

    def _build_entry(sid: str, sample_type: str) -> dict:
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
        return {
            "stimulus_id": sid,
            "sample_type": sample_type,
            "edit_type": et_map.get(sid, "unknown"),
            "magnitude_label": _mag_label(mag_map[sid]),
            "numeric_magnitude": mag_map[sid],
            "instruction": m.get("edit_instruction", ""),
            "source_image": m.get("source_image", ""),
            "ground_truth_image": m.get("ground_truth_image", ""),
            "degraded_image": m.get("degraded_image", ""),
            "exp1": e1,
            "exp2": e2,
        }

    results: list[dict] = []
    for sid in in_dim_sids:
        results.append(_build_entry(sid, "in_dimension"))
    for sid in secondary_sids:
        if sid not in in_dim_set:
            results.append(_build_entry(sid, "secondary"))
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

        # Header bar — colour-coded by sample type
        ax_hdr = fig.add_subplot(inner_gs[0])
        ax_hdr.axis("off")
        mag_val = ex["numeric_magnitude"]
        mag_str = f"{mag_val:.3g}" if mag_val != int(mag_val) else str(int(mag_val))
        sample_type = ex.get("sample_type", "")
        edit_type_str = ex.get("edit_type", "")
        if sample_type == "in_dimension":
            tag = "IN-DIM"
            hdr_bg = "#d4edda"   # light green
            hdr_fg = "#155724"
        elif sample_type == "secondary":
            tag = "SECONDARY"
            hdr_bg = "#fff3cd"   # light amber
            hdr_fg = "#856404"
        else:
            tag = ""
            hdr_bg = "#e8e8e8"
            hdr_fg = "#222222"
        tag_str = f"[{tag}]  " if tag else ""
        et_str = f"  │  edit: {edit_type_str}" if edit_type_str else ""
        ax_hdr.text(
            0.0, 0.5,
            f"  ▶  {tag_str}{ex['magnitude_label'].upper()} degradation"
            f"  │  stimulus: {ex['stimulus_id']}"
            f"  │  magnitude = {mag_str}{et_str}",
            transform=ax_hdr.transAxes, fontsize=8.5, fontweight="bold",
            va="center", color=hdr_fg,
            bbox=dict(facecolor=hdr_bg, edgecolor="none", boxstyle="square,pad=0.3"),
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
    n_secondary: int = 5,
) -> plt.Figure:
    """Plot Exp1 vs Exp2 scores for the same stimuli, grouped by magnitude.

    Exp1 uses ``similarity_score`` (higher = more similar).
    Exp2 uses ``overall_quality`` (or ``exp2_score``).

    The gap between the curves quantifies how much harder judging is vs
    simply detecting a difference.

    When ``manifest_dir`` is given, example stimuli are shown below the curve.
    All *in-dimension* samples (edit type matches the degradation dimension,
    one per unique magnitude level) are always included, followed by up to
    ``n_secondary`` randomly stratified *secondary* samples from other edit
    types for the same degradation dimension.
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
        examples = _pick_example_stimuli(df, dimension, vlm, manifest_index, n_secondary=n_secondary)

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
