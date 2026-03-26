"""
stats.py — Numerical analysis functions for VLM calibration experiments.

Public API:
    compute_all_stats(df) -> dict
    print_report(stats)
    save_report(stats, path)

The returned dict has sections:
    overview          — data quality / coverage
    exp1              — perceptual sensitivity (detection rate, thresholds, correlations)
    exp2              — instruction-following scores (descriptives, correlations, ceiling)
    cross_exp         — exp1 vs exp2 agreement and gap analysis
    sensitivity_rank  — per-dimension ranking by VLM sensitivity
"""

from __future__ import annotations

import json
import warnings
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats as scipy_stats


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

EXP2_SCORE_COLS = [
    "instruction_following",
    "text_accuracy",
    "visual_consistency",
    "layout_preservation",
    "overall_quality",
]


def _safe_spearman(x: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    """Return (rho, p_value) or (nan, nan) when insufficient variation."""
    if len(x) < 4 or np.std(x) == 0 or np.std(y) == 0:
        return float("nan"), float("nan")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        rho, pval = scipy_stats.spearmanr(x, y)
    return float(rho), float(pval)


def _safe_pointbiserial(cont: np.ndarray, binary: np.ndarray) -> tuple[float, float]:
    """Point-biserial correlation — appropriate when y is binary (0/1)."""
    cont = np.asarray(cont, dtype=float)
    binary = np.asarray(binary, dtype=float)
    if len(cont) < 4 or np.std(cont) == 0 or np.std(binary) == 0:
        return float("nan"), float("nan")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        r, pval = scipy_stats.pointbiserialr(cont, binary)
    return float(r), float(pval)


def _cronbach_alpha(score_matrix: np.ndarray) -> float:
    """Cronbach's alpha for an (n_items × k_scales) matrix.

    Each row is one stimulus; each column is one scale dimension.
    Returns nan when fewer than 2 valid scales exist.
    """
    if score_matrix.shape[1] < 2:
        return float("nan")
    item_vars = score_matrix.var(axis=0, ddof=1)
    total_var = score_matrix.sum(axis=1).var(ddof=1)
    if total_var == 0:
        return float("nan")
    k = score_matrix.shape[1]
    alpha = (k / (k - 1)) * (1 - item_vars.sum() / total_var)
    return float(alpha)


def _jnd_threshold(magnitudes: np.ndarray, detections: np.ndarray, target: float = 0.5) -> float:
    """Estimate the magnitude where detection rate crosses `target` by linear
    interpolation over the per-magnitude mean detection rate.

    Returns nan when the target is never reached or data is insufficient.
    """
    if len(magnitudes) < 2:
        return float("nan")
    grouped = (
        pd.DataFrame({"mag": magnitudes, "det": detections.astype(float)})
        .groupby("mag")["det"]
        .mean()
        .reset_index()
        .sort_values("mag")
    )
    mags = grouped["mag"].values
    rates = grouped["det"].values
    if rates.max() < target:
        return float("nan")
    if rates.min() >= target:
        return float(mags[0])
    for i in range(len(mags) - 1):
        if rates[i] <= target <= rates[i + 1]:
            # linear interpolation
            if rates[i + 1] == rates[i]:
                return float(mags[i])
            t = (target - rates[i]) / (rates[i + 1] - rates[i])
            return float(mags[i] + t * (mags[i + 1] - mags[i]))
    return float("nan")


# ---------------------------------------------------------------------------
# Overview
# ---------------------------------------------------------------------------

def _overview(df: pd.DataFrame) -> dict:
    out: dict[str, Any] = {}
    out["total_records"] = len(df)
    out["stimuli"] = int(df["stimulus_id"].nunique())
    out["models"] = sorted(df["model"].unique().tolist())

    for exp in ("experiment_1", "experiment_2"):
        sub = df[df["experiment"] == exp]
        out[exp] = {
            "records": len(sub),
            "parse_success_rate": float(sub["parse_success"].mean()) if len(sub) else float("nan"),
            "unique_stimuli": int(sub["stimulus_id"].nunique()),
        }

    out["dimensions"] = sorted(df["degradation_dimension"].unique().tolist())
    out["edit_types"] = sorted(df["edit_type"].unique().tolist())

    # coverage matrix: dimensions × edit_types — count of stimuli per cell
    e2 = df[df["experiment"] == "experiment_2"]
    coverage: dict[str, dict[str, int]] = {}
    for dim in out["dimensions"]:
        coverage[dim] = {}
        for et in out["edit_types"]:
            n = int(len(e2[(e2["degradation_dimension"] == dim) & (e2["edit_type"] == et)]))
            if n > 0:
                coverage[dim][et] = n
    out["coverage"] = coverage

    return out


# ---------------------------------------------------------------------------
# Exp1 — perceptual sensitivity
# ---------------------------------------------------------------------------

def _exp1_stats(df: pd.DataFrame) -> dict:
    e1 = df[
        (df["experiment"] == "experiment_1")
        & df["parse_success"]
        & df["detected_difference"].notna()
    ].copy()

    if e1.empty:
        return {"error": "no exp1 data"}

    e1["detected"] = e1["detected_difference"].astype(bool)

    out: dict[str, Any] = {}

    # --- global ---
    out["overall_detection_rate"] = float(e1["detected"].mean())
    out["overall_false_negative_rate"] = 1.0 - out["overall_detection_rate"]
    out["similarity_score_mean"] = float(e1["similarity_score"].mean())
    out["similarity_score_std"] = float(e1["similarity_score"].std())
    out["similarity_score_distribution"] = {
        str(v): int(n)
        for v, n in e1["similarity_score"].value_counts().sort_index().items()
    }

    # --- per dimension ---
    dims = sorted(e1["degradation_dimension"].unique())
    per_dim: dict[str, Any] = {}
    for dim in dims:
        sub = e1[e1["degradation_dimension"] == dim]
        det = sub["detected"].values
        sim = sub["similarity_score"].dropna().values
        mags = sub["numeric_magnitude"].values

        rho_det, p_det = _safe_pointbiserial(mags, det)
        rho_sim, p_sim = _safe_spearman(mags, sim) if len(sim) == len(mags) else (float("nan"), float("nan"))
        rho_sim_all, p_sim_all = _safe_spearman(mags, sub["similarity_score"].values)

        # per-magnitude detection rate table
        mag_table = (
            sub.groupby("numeric_magnitude")["detected"]
            .agg(["mean", "count"])
            .rename(columns={"mean": "detection_rate", "count": "n"})
            .reset_index()
            .sort_values("numeric_magnitude")
        )

        jnd = _jnd_threshold(mags, det, target=0.5)
        jnd75 = _jnd_threshold(mags, det, target=0.75)

        per_dim[dim] = {
            "n": int(len(sub)),
            "detection_rate": float(det.mean()),
            "false_negative_rate": float(1 - det.mean()),
            "similarity_score_mean": float(sub["similarity_score"].mean()),
            "similarity_score_std": float(sub["similarity_score"].std()),
            # correlation: magnitude → detected (point-biserial)
            "magnitude_detection_r": rho_det,
            "magnitude_detection_p": p_det,
            # correlation: magnitude → similarity score (Spearman)
            "magnitude_similarity_rho": rho_sim_all,
            "magnitude_similarity_p": p_sim_all,
            # thresholds
            "jnd_50pct": jnd,
            "jnd_75pct": jnd75,
            # per-magnitude breakdown
            "by_magnitude": {
                str(row["numeric_magnitude"]): {
                    "detection_rate": round(float(row["detection_rate"]), 4),
                    "n": int(row["n"]),
                }
                for _, row in mag_table.iterrows()
            },
        }

    out["by_dimension"] = per_dim

    # --- per edit type ---
    per_et: dict[str, Any] = {}
    for et in sorted(e1["edit_type"].unique()):
        sub = e1[e1["edit_type"] == et]
        per_et[et] = {
            "n": int(len(sub)),
            "detection_rate": float(sub["detected"].mean()),
            "similarity_score_mean": float(sub["similarity_score"].mean()),
        }
    out["by_edit_type"] = per_et

    return out


# ---------------------------------------------------------------------------
# Exp2 — instruction-following scores
# ---------------------------------------------------------------------------

def _exp2_stats(df: pd.DataFrame) -> dict:
    e2 = df[
        (df["experiment"] == "experiment_2")
        & df["parse_success"]
        & df["overall_quality"].notna()
    ].copy()

    if e2.empty:
        return {"error": "no exp2 data"}

    out: dict[str, Any] = {}

    # --- global descriptives per score column ---
    global_scores: dict[str, Any] = {}
    for col in EXP2_SCORE_COLS:
        vals = e2[col].dropna()
        global_scores[col] = {
            "mean": round(float(vals.mean()), 4),
            "std": round(float(vals.std()), 4),
            "median": round(float(vals.median()), 4),
            "min": int(vals.min()),
            "max": int(vals.max()),
            "ceiling_rate": round(float((vals == 5).mean()), 4),
            "floor_rate": round(float((vals == 1).mean()), 4),
            "distribution": {str(v): int(n) for v, n in vals.value_counts().sort_index().items()},
        }
    out["score_descriptives"] = global_scores

    # --- score intercorrelation matrix ---
    score_cols_avail = [c for c in EXP2_SCORE_COLS if c in e2.columns]
    corr_data = e2[score_cols_avail].dropna()
    corr_rho = corr_data.corr(method="spearman").round(4)
    out["score_intercorrelation_spearman"] = {
        col: corr_rho[col].to_dict() for col in score_cols_avail
    }

    # --- Cronbach's alpha across the 5 score dimensions ---
    score_matrix = corr_data.values
    out["cronbach_alpha"] = round(_cronbach_alpha(score_matrix), 4)

    # --- per dimension ---
    dims = sorted(e2["degradation_dimension"].unique())
    per_dim: dict[str, Any] = {}
    for dim in dims:
        sub = e2[e2["degradation_dimension"] == dim].copy()
        mags = sub["numeric_magnitude"].values
        dim_entry: dict[str, Any] = {"n": int(len(sub))}

        for col in EXP2_SCORE_COLS:
            vals = sub[col].dropna().values
            if len(vals) == 0:
                continue
            rho, p = _safe_spearman(mags[:len(vals)], vals) if len(vals) == len(mags) else _safe_spearman(mags, sub[col].values)
            dim_entry[col] = {
                "mean": round(float(vals.mean()), 4),
                "std": round(float(vals.std()), 4),
                "ceiling_rate": round(float((vals == 5).mean()), 4),
                "magnitude_rho": round(rho, 4) if not np.isnan(rho) else None,
                "magnitude_p": round(p, 4) if not np.isnan(p) else None,
            }

        # per-magnitude mean overall_quality table
        oq = sub[sub["overall_quality"].notna()].groupby("numeric_magnitude")["overall_quality"]
        mag_table = oq.agg(["mean", "std", "count"]).reset_index().sort_values("numeric_magnitude")
        dim_entry["overall_quality_by_magnitude"] = {
            str(row["numeric_magnitude"]): {
                "mean": round(float(row["mean"]), 4),
                "std": round(float(row["std"]) if not np.isnan(row["std"]) else 0.0, 4),
                "n": int(row["count"]),
            }
            for _, row in mag_table.iterrows()
        }

        per_dim[dim] = dim_entry

    out["by_dimension"] = per_dim

    # --- per edit type ---
    per_et: dict[str, Any] = {}
    for et in sorted(e2["edit_type"].unique()):
        sub = e2[e2["edit_type"] == et]
        per_et[et] = {
            "n": int(len(sub)),
            "overall_quality_mean": round(float(sub["overall_quality"].mean()), 4),
            "overall_quality_std": round(float(sub["overall_quality"].std()), 4),
        }
    out["by_edit_type"] = per_et

    return out


# ---------------------------------------------------------------------------
# Cross-experiment gap analysis
# ---------------------------------------------------------------------------

def _cross_exp_stats(df: pd.DataFrame) -> dict:
    e1 = df[
        (df["experiment"] == "experiment_1")
        & df["parse_success"]
        & df["similarity_score"].notna()
    ].copy()
    e2 = df[
        (df["experiment"] == "experiment_2")
        & df["parse_success"]
        & df["overall_quality"].notna()
    ].copy()

    if e1.empty or e2.empty:
        return {"error": "need both experiments"}

    out: dict[str, Any] = {}

    # --- stimulus-level alignment: stim_id present in both ---
    shared_sids = set(e1["stimulus_id"]) & set(e2["stimulus_id"])
    out["stimuli_in_both_experiments"] = len(shared_sids)

    merged = pd.merge(
        e1[["stimulus_id", "numeric_magnitude", "similarity_score", "detected_difference",
            "degradation_dimension", "edit_type"]],
        e2[["stimulus_id", "overall_quality", "instruction_following",
            "text_accuracy", "visual_consistency", "layout_preservation"]],
        on="stimulus_id",
    )
    out["merged_records"] = len(merged)

    if len(merged) < 4:
        out["error"] = "insufficient overlap for correlation"
        return out

    # global correlation: exp1 similarity_score vs exp2 overall_quality
    rho, p = _safe_spearman(merged["similarity_score"].values, merged["overall_quality"].values)
    out["global_spearman_sim_vs_oq"] = {"rho": round(rho, 4), "p": round(p, 6)}

    # correlation of exp1 detected_difference with exp2 overall_quality
    rho_d, p_d = _safe_pointbiserial(
        merged["overall_quality"].values.astype(float),
        pd.to_numeric(merged["detected_difference"], errors="coerce").fillna(0).values.astype(float),
    )
    out["global_pointbiserial_detected_vs_oq"] = {"r": round(rho_d, 4), "p": round(p_d, 6)}

    # --- per dimension ---
    per_dim: dict[str, Any] = {}
    for dim in sorted(merged["degradation_dimension"].unique()):
        sub = merged[merged["degradation_dimension"] == dim]
        if len(sub) < 3:
            continue

        rho, p = _safe_spearman(sub["similarity_score"].values, sub["overall_quality"].values)

        # mean gap: mean(exp2 oq) - mean(exp1 sim) — positive means exp2 scores higher
        # (inflated quality perception relative to perceptual similarity)
        gap = float(sub["overall_quality"].mean() - sub["similarity_score"].mean())

        # How often exp1 detects BUT exp2 scores ≥ 4 (false positive quality judgement)
        detected_mask = pd.to_numeric(sub["detected_difference"], errors="coerce").fillna(0).astype(bool)
        n_detected = detected_mask.sum()
        if n_detected > 0:
            exp2_high_given_detected = float((sub.loc[detected_mask, "overall_quality"] >= 4).mean())
        else:
            exp2_high_given_detected = float("nan")

        per_dim[dim] = {
            "n": int(len(sub)),
            "spearman_sim_vs_oq_rho": round(rho, 4) if not np.isnan(rho) else None,
            "spearman_sim_vs_oq_p": round(p, 4) if not np.isnan(p) else None,
            "mean_exp1_similarity": round(float(sub["similarity_score"].mean()), 4),
            "mean_exp2_oq": round(float(sub["overall_quality"].mean()), 4),
            "score_gap_exp2_minus_exp1": round(gap, 4),
            "exp2_high_score_rate_when_exp1_detected": (
                round(exp2_high_given_detected, 4)
                if not np.isnan(exp2_high_given_detected) else None
            ),
        }

    out["by_dimension"] = per_dim

    # --- overall blind-spot summary ---
    # Stimuli where exp1 detected difference (detected=True) but exp2 gave high score (≥4)
    det_bool = pd.to_numeric(merged["detected_difference"], errors="coerce").fillna(0).astype(bool)
    blind_spot = merged[det_bool & (merged["overall_quality"] >= 4)]
    out["blind_spot_count"] = len(blind_spot)
    out["blind_spot_rate"] = round(float(len(blind_spot) / max(1, det_bool.sum())), 4)

    return out


# ---------------------------------------------------------------------------
# Sensitivity ranking
# ---------------------------------------------------------------------------

def _sensitivity_rank(df: pd.DataFrame, exp1_stats: dict, exp2_stats: dict) -> dict:
    """Rank dimensions by how sensitive the VLM is to each degradation type."""
    dims = sorted(
        set(list(exp1_stats.get("by_dimension", {}).keys()))
        | set(list(exp2_stats.get("by_dimension", {}).keys()))
    )
    ranks: list[dict] = []
    for dim in dims:
        e1d = exp1_stats.get("by_dimension", {}).get(dim, {})
        e2d = exp2_stats.get("by_dimension", {}).get(dim, {})
        oq = e2d.get("overall_quality", {})

        entry: dict[str, Any] = {"dimension": dim}
        entry["exp1_detection_rate"] = e1d.get("detection_rate")
        entry["exp1_sim_magnitude_rho"] = e1d.get("magnitude_similarity_rho")
        entry["exp2_oq_mean"] = oq.get("mean")
        entry["exp2_oq_ceiling_rate"] = oq.get("ceiling_rate")
        entry["exp2_oq_magnitude_rho"] = oq.get("magnitude_rho")
        entry["jnd_50pct"] = e1d.get("jnd_50pct")
        ranks.append(entry)

    # Sort: low detection rate + high ceiling rate = least sensitive (blind)
    # Sort by exp1 detection rate ascending (most blind first)
    ranks_sorted_by_detection = sorted(
        [r for r in ranks if r["exp1_detection_rate"] is not None],
        key=lambda r: r["exp1_detection_rate"],
    )

    # Sort by exp2 oq mean descending (inflated = least sensitive)
    ranks_sorted_by_oq = sorted(
        [r for r in ranks if r["exp2_oq_mean"] is not None],
        key=lambda r: -r["exp2_oq_mean"],
    )

    return {
        "by_exp1_detection_rate_asc": [r["dimension"] for r in ranks_sorted_by_detection],
        "by_exp2_oq_mean_desc": [r["dimension"] for r in ranks_sorted_by_oq],
        "detail": {r["dimension"]: r for r in ranks},
    }


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def compute_all_stats(df: pd.DataFrame) -> dict:
    """Compute the full statistics bundle from a combined exp1+exp2 DataFrame.

    Parameters
    ----------
    df:
        Output of ``load_results(exp1_path, manifest) + load_results(exp2_path, manifest)``.

    Returns
    -------
    dict with keys: overview, exp1, exp2, cross_exp, sensitivity_rank
    """
    exp1_stats = _exp1_stats(df)
    exp2_stats = _exp2_stats(df)
    return {
        "overview": _overview(df),
        "exp1": exp1_stats,
        "exp2": exp2_stats,
        "cross_exp": _cross_exp_stats(df),
        "sensitivity_rank": _sensitivity_rank(df, exp1_stats, exp2_stats),
    }


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def save_report(stats: dict, path: str | Path) -> None:
    """Save statistics bundle as JSON."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2, ensure_ascii=False, default=str)


def print_report(stats: dict) -> None:
    """Print a human-readable summary to stdout."""
    ov = stats.get("overview", {})
    e1 = stats.get("exp1", {})
    e2 = stats.get("exp2", {})
    cx = stats.get("cross_exp", {})
    sr = stats.get("sensitivity_rank", {})

    sep = "-" * 68

    print(sep)
    print("VLM JUDGE CALIBRATION — NUMERICAL ANALYSIS REPORT")
    print(sep)

    # Overview
    print(f"\n{'OVERVIEW':}")
    print(f"  Total records       : {ov.get('total_records', '?')}")
    print(f"  Unique stimuli      : {ov.get('stimuli', '?')}")
    print(f"  Models              : {ov.get('models', [])}")
    for exp_key in ("experiment_1", "experiment_2"):
        ed = ov.get(exp_key, {})
        print(f"  {exp_key}: {ed.get('records','?')} records, "
              f"parse_success={ed.get('parse_success_rate','?'):.1%}, "
              f"stimuli={ed.get('unique_stimuli','?')}")
    print(f"  Degradation dims    : {ov.get('dimensions', [])}")
    print(f"  Edit types          : {ov.get('edit_types', [])}")

    # Exp1
    print(f"\n{'-'*68}")
    print("EXP 1 — PERCEPTUAL SENSITIVITY (binary detection + similarity)")
    print(f"{'-'*68}")
    if "error" not in e1:
        print(f"  Overall detection rate     : {e1.get('overall_detection_rate',0):.1%}")
        print(f"  Overall false-negative rate: {e1.get('overall_false_negative_rate',0):.1%}")
        print(f"  Mean similarity score      : {e1.get('similarity_score_mean',0):.3f} ± "
              f"{e1.get('similarity_score_std',0):.3f}")
        print(f"  Score distribution: {e1.get('similarity_score_distribution', {})}")
        print()
        print(f"  {'Dimension':<22} {'Det.Rate':>9} {'FNR':>7} {'Sim.Mean':>9} "
              f"{'Mag→Det r':>10} {'Mag→Sim ρ':>10} {'JND50%':>8}")
        print(f"  {'-'*22} {'-'*9} {'-'*7} {'-'*9} {'-'*10} {'-'*10} {'-'*8}")
        for dim, d in sorted(e1.get("by_dimension", {}).items()):
            jnd = d.get("jnd_50pct")
            jnd_str = f"{jnd:.3g}" if jnd is not None and not (isinstance(jnd, float) and np.isnan(jnd)) else "  —"
            rho_det = d.get("magnitude_detection_r")
            rho_det_str = f"{rho_det:+.3f}" if rho_det is not None and not np.isnan(rho_det) else "    —"
            rho_sim = d.get("magnitude_similarity_rho")
            rho_sim_str = f"{rho_sim:+.3f}" if rho_sim is not None and not np.isnan(rho_sim) else "    —"
            print(f"  {dim:<22} {d.get('detection_rate', 0):>9.1%} "
                  f"{d.get('false_negative_rate', 0):>7.1%} "
                  f"{d.get('similarity_score_mean', 0):>9.3f} "
                  f"{rho_det_str:>10} {rho_sim_str:>10} {jnd_str:>8}")

    # Exp2
    print(f"\n{'-'*68}")
    print("EXP 2 — INSTRUCTION-FOLLOWING (multi-dimensional scores 1–5)")
    print(f"{'-'*68}")
    if "error" not in e2:
        sd = e2.get("score_descriptives", {})
        print(f"\n  Global score descriptives:")
        print(f"  {'Score col':<24} {'Mean':>6} {'Std':>6} {'Median':>7} "
              f"{'Ceiling%':>9} {'Floor%':>7}")
        print(f"  {'-'*24} {'-'*6} {'-'*6} {'-'*7} {'-'*9} {'-'*7}")
        for col in EXP2_SCORE_COLS:
            s = sd.get(col, {})
            print(f"  {col:<24} {s.get('mean',0):>6.3f} {s.get('std',0):>6.3f} "
                  f"{s.get('median',0):>7.1f} "
                  f"{s.get('ceiling_rate',0):>9.1%} {s.get('floor_rate',0):>7.1%}")

        print(f"\n  Cronbach's alpha (score consistency): {e2.get('cronbach_alpha', float('nan')):.4f}")

        print(f"\n  Spearman intercorrelation matrix (overall_quality row):")
        ic = e2.get("score_intercorrelation_spearman", {})
        oq_row = ic.get("overall_quality", {})
        for col in EXP2_SCORE_COLS:
            print(f"    overall_quality ↔ {col:<24}: ρ = {oq_row.get(col, float('nan')):.4f}")

        print(f"\n  Per-dimension overall_quality stats:")
        print(f"  {'Dimension':<22} {'OQ.Mean':>8} {'OQ.Std':>7} "
              f"{'Ceil%':>7} {'Mag→OQ ρ':>10} {'p':>8}")
        print(f"  {'-'*22} {'-'*8} {'-'*7} {'-'*7} {'-'*10} {'-'*8}")
        for dim, d in sorted(e2.get("by_dimension", {}).items()):
            oq = d.get("overall_quality", {})
            rho = oq.get("magnitude_rho")
            p = oq.get("magnitude_p")
            rho_str = f"{rho:+.4f}" if rho is not None else "      —"
            p_str = f"{p:.4f}" if p is not None else "      —"
            print(f"  {dim:<22} {oq.get('mean', 0):>8.3f} {oq.get('std', 0):>7.3f} "
                  f"{oq.get('ceiling_rate', 0):>7.1%} {rho_str:>10} {p_str:>8}")

    # Cross-exp
    print(f"\n{'-'*68}")
    print("CROSS-EXPERIMENT GAP ANALYSIS")
    print(f"{'-'*68}")
    if "error" not in cx:
        g = cx.get("global_spearman_sim_vs_oq", {})
        gd = cx.get("global_pointbiserial_detected_vs_oq", {})
        print(f"  Stimuli in both experiments: {cx.get('stimuli_in_both_experiments', '?')}")
        print(f"  Merged records             : {cx.get('merged_records', '?')}")
        print(f"  Global Spearman ρ(sim_score, oq): {g.get('rho', float('nan')):.4f} "
              f"(p={g.get('p', float('nan')):.4f})")
        print(f"  Point-biserial r(detected, oq) : {gd.get('r', float('nan')):.4f} "
              f"(p={gd.get('p', float('nan')):.4f})")
        print(f"  Blind-spot count (detected=T, oq≥4): {cx.get('blind_spot_count','?')} "
              f"({cx.get('blind_spot_rate', 0):.1%} of detected stimuli)")

        print(f"\n  {'Dimension':<22} {'n':>4} {'E1.Sim':>7} {'E2.OQ':>7} "
              f"{'Gap':>7} {'ρ(sim,oq)':>10} {'BS%':>7}")
        print(f"  {'-'*22} {'-'*4} {'-'*7} {'-'*7} {'-'*7} {'-'*10} {'-'*7}")
        for dim, d in sorted(cx.get("by_dimension", {}).items()):
            rho = d.get("spearman_sim_vs_oq_rho")
            rho_str = f"{rho:+.4f}" if rho is not None else "      —"
            bs = d.get("exp2_high_score_rate_when_exp1_detected")
            bs_str = f"{bs:.1%}" if bs is not None else "    —"
            print(f"  {dim:<22} {d.get('n','?'):>4} "
                  f"{d.get('mean_exp1_similarity', 0):>7.3f} "
                  f"{d.get('mean_exp2_oq', 0):>7.3f} "
                  f"{d.get('score_gap_exp2_minus_exp1', 0):>+7.3f} "
                  f"{rho_str:>10} {bs_str:>7}")

    # Sensitivity ranking
    print(f"\n{'-'*68}")
    print("SENSITIVITY RANKING")
    print(f"{'-'*68}")
    print(f"  Most blind dims (lowest Exp1 detection rate):")
    for i, dim in enumerate(sr.get("by_exp1_detection_rate_asc", []), 1):
        det = sr["detail"][dim].get("exp1_detection_rate")
        det_str = f"{det:.1%}" if det is not None else "—"
        print(f"    {i}. {dim:<24} detection={det_str}")

    print(f"\n  Highest Exp2 score (most inflated dims):")
    for i, dim in enumerate(sr.get("by_exp2_oq_mean_desc", []), 1):
        oq = sr["detail"][dim].get("exp2_oq_mean")
        oq_str = f"{oq:.3f}" if oq is not None else "—"
        print(f"    {i}. {dim:<24} oq_mean={oq_str}")

    print(f"\n{sep}")
