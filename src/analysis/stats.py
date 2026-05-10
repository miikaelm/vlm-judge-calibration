"""
stats.py — Numerical analysis functions for VLM calibration experiments.

Public API:
    compute_all_stats(df) -> dict
    print_report(stats)
    save_report(stats, path)

The returned dict has sections:
    overview              — data quality / coverage (with per-model parse success)
    exp1.by_model         — perceptual sensitivity, indexed by model → dimension
    exp2.by_model         — instruction-following scores, indexed by model → dimension
    cross_exp.by_model    — Exp1 vs Exp2 agreement, indexed by model
    sensitivity_rank      — per (dimension, model) ranking with blind/sensitive classification
    failure_mode_table    — per-model summary of parse success, FPR, noop rates, score gap
    noop.by_model         — false positive rate analysis (Exp2 noop stimuli, by model)

Key changes from original:
    • All stats are computed per (dimension, model) cell — no cross-model pooling.
    • Spearman/point-biserial p-values carry both _raw and BH-corrected variants.
    • Threshold tiers use discrete tier-label lookup (no interpolation).
    • Blind spot classification uses a one-sided binomial test vs. the model's
      adjusted FPR (perfect_edits, excluding _FPR_EXCLUDED_DIMS).
    • Noop Exp2 metrics are above_1_rate (>1) and high_rating_rate (≥3).
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
# Constants
# ---------------------------------------------------------------------------

EXP2_SCORE_COLS = [
    "instruction_following",
    "text_accuracy",
    "visual_consistency",
    "layout_preservation",
    "overall_quality",
]

_BLIND_SPOT_ALPHA = 0.05   # FDR level for blind-spot binomial tests
_PRIMARY_ALPHA   = 0.05    # FDR level for primary Spearman tests

# Dimensions excluded from the model-level FPR used as the binomial-test baseline.
# alignment_error stimuli produce anomalously high false positive rates because
# models appear to trigger on the spatial unusualness of the relocated text
# (comparing against an internal position prior) rather than genuinely comparing
# the two identical images.  Excluding this dimension from the aggregate FPR
# gives a cleaner null baseline for classifying other dimensions as blind/sensitive.
# The per-dimension FPR breakdown (stored in perfect_edits.by_dimension) is
# unaffected and still includes alignment_error for full transparency.
_FPR_EXCLUDED_DIMS: frozenset[str] = frozenset({"alignment_error"})


# ---------------------------------------------------------------------------
# Helper: Benjamini-Hochberg FDR correction
# ---------------------------------------------------------------------------

def _bh_correction(pvalues: list[float], alpha: float = 0.05) -> list[float]:
    """Apply Benjamini-Hochberg FDR correction; return adjusted p-values.

    NaN entries are preserved as NaN.  All other adjusted values are ≤ 1.
    """
    n = len(pvalues)
    if n == 0:
        return []

    # Identify valid (non-NaN) entries
    valid_idx = [i for i, p in enumerate(pvalues) if not (isinstance(p, float) and np.isnan(p))]
    m = len(valid_idx)
    if m == 0:
        return list(pvalues)

    corrected = list(pvalues)  # copy; NaN slots stay NaN

    # Sort valid entries ascending by raw p-value; track original index
    sorted_valid = sorted(
        [(pvalues[i], i) for i in valid_idx],
        key=lambda x: x[0],
    )

    # BH step-up: traverse from largest to smallest to ensure monotonicity
    prev = 1.0
    for step, (p, orig_i) in enumerate(reversed(sorted_valid), start=1):
        rank = m - step + 1  # rank in ascending order (1 = smallest)
        adjusted = min(prev, p * m / rank)
        adjusted = min(adjusted, 1.0)
        corrected[orig_i] = adjusted
        prev = adjusted

    return corrected


# ---------------------------------------------------------------------------
# Helper: Spearman / point-biserial wrappers
# ---------------------------------------------------------------------------

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
    cont   = np.asarray(cont,   dtype=float)
    binary = np.asarray(binary, dtype=float)
    if len(cont) < 4 or np.std(cont) == 0 or np.std(binary) == 0:
        return float("nan"), float("nan")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        r, pval = scipy_stats.pointbiserialr(cont, binary)
    return float(r), float(pval)


# ---------------------------------------------------------------------------
# Helper: Newcombe CI for difference of two proportions
# ---------------------------------------------------------------------------

def _wilson_ci(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson score confidence interval for a proportion k/n."""
    if n == 0:
        return (0.0, 1.0)
    p = k / n
    z2 = z * z
    center = (p + z2 / (2 * n)) / (1 + z2 / n)
    margin = z * np.sqrt(p * (1 - p) / n + z2 / (4 * n * n)) / (1 + z2 / n)
    return max(0.0, center - margin), min(1.0, center + margin)


def _newcombe_ci(
    k1: int, n1: int, k2: int, n2: int, z: float = 1.96
) -> tuple[float, float, float]:
    """Newcombe hybrid score CI for the difference p1 - p2.

    Returns (delta, ci_lower, ci_upper).
    """
    p1 = k1 / n1 if n1 > 0 else 0.0
    p2 = k2 / n2 if n2 > 0 else 0.0
    l1, u1 = _wilson_ci(k1, n1, z)
    l2, u2 = _wilson_ci(k2, n2, z)
    delta = p1 - p2
    ci_lo = delta - np.sqrt((p1 - l1) ** 2 + (u2 - p2) ** 2)
    ci_hi = delta + np.sqrt((u1 - p1) ** 2 + (p2 - l2) ** 2)
    return delta, max(-1.0, ci_lo), min(1.0, ci_hi)


# ---------------------------------------------------------------------------
# Helper: Cronbach's alpha
# ---------------------------------------------------------------------------

def _cronbach_alpha(score_matrix: np.ndarray) -> float:
    """Cronbach's alpha for an (n_items × k_scales) matrix."""
    if score_matrix.shape[1] < 2:
        return float("nan")
    item_vars = score_matrix.var(axis=0, ddof=1)
    total_var = score_matrix.sum(axis=1).var(ddof=1)
    if total_var == 0:
        return float("nan")
    k = score_matrix.shape[1]
    return float((k / (k - 1)) * (1 - item_vars.sum() / total_var))


# ---------------------------------------------------------------------------
# Helper: discrete threshold tier  (replaces continuous _jnd_threshold)
# ---------------------------------------------------------------------------

def _threshold_tier(
    mag_table: pd.DataFrame, target: float = 0.5
) -> tuple[str, float | None]:
    """Return the smallest magnitude tier where detection rate >= target.

    Parameters
    ----------
    mag_table:
        DataFrame produced by groupby("degradation_magnitude").agg(…).
        Must have columns: degradation_magnitude, detection_rate,
        numeric_magnitude_mean.  Rows must be sorted ascending by
        numeric_magnitude_mean BEFORE calling this function.

    Returns
    -------
    (tier_label, numeric_magnitude_mean) where tier_label is the first tier
    to reach `target` (e.g. "small") and numeric_magnitude_mean is the
    corresponding physical magnitude in dimension-native units (ΔE, px, °, …).
    Returns ("not_crossed", None) when no tier reaches the threshold.
    No interpolation is performed — this is intentionally a discrete step function.
    """
    for _, row in mag_table.iterrows():
        if row["detection_rate"] >= target:
            return str(row["degradation_magnitude"]), float(row["numeric_magnitude_mean"])
    return "not_crossed", None


# ---------------------------------------------------------------------------
# Helper: Fleiss' kappa for inter-model agreement
# ---------------------------------------------------------------------------

def _fleiss_kappa(blind_sensitive: dict[str, dict[str, Any]]) -> float:
    """Compute Fleiss' kappa for inter-rater agreement on blind/sensitive.

    Parameters
    ----------
    blind_sensitive:
        Nested dict: dimension → model → {classification: "blind"|"sensitive"|"unknown"}

    Only rows (dimensions) where every model has a "blind" or "sensitive"
    classification are included; rows with any "unknown" entry are dropped.

    Returns
    -------
    Fleiss' kappa as a float, or nan when fewer than 2 complete rows exist.
    """
    dims   = sorted(blind_sensitive.keys())
    models = sorted({m for d in blind_sensitive.values() for m in d.keys()})

    if len(models) < 2:
        return float("nan")

    n = len(models)   # raters
    valid_rows: list[list[int]] = []

    for dim in dims:
        row_classes = [
            blind_sensitive[dim].get(m, {}).get("classification", "unknown")
            for m in models
        ]
        if all(c in ("blind", "sensitive") for c in row_classes):
            valid_rows.append([1 if c == "sensitive" else 0 for c in row_classes])

    N = len(valid_rows)
    if N < 2:
        return float("nan")

    matrix = np.array(valid_rows, dtype=float)   # N × n

    n_sensitive = matrix.sum(axis=1)              # per-subject sensitive count
    n_blind     = n - n_sensitive

    # Per-subject agreement proportion
    P_i     = (n_sensitive * (n_sensitive - 1) + n_blind * (n_blind - 1)) / (n * (n - 1))
    P_bar   = float(P_i.mean())

    # Marginal proportions (expected agreement by chance)
    p_sensitive = float(n_sensitive.sum()) / (N * n)
    p_blind     = 1.0 - p_sensitive
    P_e         = p_sensitive ** 2 + p_blind ** 2

    if P_e >= 1.0:
        return float("nan")

    return (P_bar - P_e) / (1.0 - P_e)


# ---------------------------------------------------------------------------
# Overview
# ---------------------------------------------------------------------------

def _overview(df: pd.DataFrame) -> dict:
    out: dict[str, Any] = {}
    out["total_records"] = len(df)
    out["stimuli"]       = int(df["stimulus_id"].nunique())
    out["models"]        = sorted(df["model"].unique().tolist())

    for exp in ("experiment_1", "experiment_2"):
        sub = df[df["experiment"] == exp]
        # Global parse success rate
        global_psr = float(sub["parse_success"].mean()) if len(sub) else float("nan")
        # Per-model parse success rate
        per_model: dict[str, Any] = {}
        for model in out["models"]:
            m_sub = sub[sub["model"] == model]
            per_model[model] = {
                "records": int(len(m_sub)),
                "parse_success_rate": (
                    float(m_sub["parse_success"].mean()) if len(m_sub) else float("nan")
                ),
            }
        out[exp] = {
            "records": int(len(sub)),
            "parse_success_rate": global_psr,
            "unique_stimuli": int(sub["stimulus_id"].nunique()),
            "by_model": per_model,
        }

    out["dimensions"] = sorted(df["degradation_dimension"].unique().tolist())
    out["edit_types"]  = sorted(df["edit_type"].unique().tolist())

    # Coverage matrix: dimensions × edit_types
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
# Exp1 — perceptual sensitivity (called once per model)
# ---------------------------------------------------------------------------

def _exp1_stats(df: pd.DataFrame, perfect_df: pd.DataFrame | None = None) -> dict:
    """Compute Exp1 statistics for a single-model DataFrame.

    ``df`` is expected to contain only degraded stimuli (numeric_magnitude > 0).
    Perfect-edit stimuli (mag == 0) are passed separately via ``perfect_df``.
    """
    e1 = df[
        (df["experiment"] == "experiment_1")
        & df["parse_success"]
        & df["detected_difference"].notna()
    ].copy()

    if e1.empty:
        return {"error": "no exp1 data"}

    e1["detected"] = e1["detected_difference"].astype(bool)

    out: dict[str, Any] = {}

    # --- global (degraded only) ---
    out["overall_detection_rate"]      = float(e1["detected"].mean())
    out["overall_false_negative_rate"] = 1.0 - out["overall_detection_rate"]
    out["similarity_score_mean"] = float(e1["similarity_score"].mean())
    out["similarity_score_std"]  = float(e1["similarity_score"].std())
    out["similarity_score_distribution"] = {
        str(v): int(n) for v, n in
        e1["similarity_score"].value_counts().sort_index().items()
    }

    # --- per dimension ---
    dims = sorted(e1["degradation_dimension"].unique())
    per_dim: dict[str, Any] = {}
    for dim in dims:
        sub = e1[e1["degradation_dimension"] == dim]
        det  = sub["detected"].values
        mags = sub["numeric_magnitude"].values

        rho_det, p_det         = _safe_pointbiserial(mags, det)
        rho_sim_all, p_sim_all = _safe_spearman(mags, sub["similarity_score"].values)

        # Per-magnitude detection rate table, sorted by mean numeric magnitude
        mag_table = (
            sub.groupby("degradation_magnitude")
            .agg(
                detection_rate=("detected", "mean"),
                n=("detected", "count"),
                numeric_magnitude_mean=("numeric_magnitude", "mean"),
            )
            .reset_index()
            .sort_values("numeric_magnitude_mean")
        )

        # Discrete threshold tiers (label + physical magnitude)
        threshold_50_label, threshold_50_val = _threshold_tier(mag_table, target=0.50)
        threshold_75_label, threshold_75_val = _threshold_tier(mag_table, target=0.75)

        per_dim[dim] = {
            "n": int(len(sub)),
            "detection_rate":      float(det.mean()),
            "false_negative_rate": float(1 - det.mean()),
            "similarity_score_mean": float(sub["similarity_score"].mean()),
            "similarity_score_std":  float(sub["similarity_score"].std()),
            "magnitude_detection_r": rho_det,
            "magnitude_detection_p": p_det,
            "magnitude_similarity_rho": rho_sim_all,
            "magnitude_similarity_p": p_sim_all,
            "threshold_tier_50pct": threshold_50_label,
            "threshold_numeric_50pct": round(threshold_50_val, 4) if threshold_50_val is not None else None,
            "threshold_tier_75pct": threshold_75_label,
            "threshold_numeric_75pct": round(threshold_75_val, 4) if threshold_75_val is not None else None,
            "by_magnitude": {
                str(row["degradation_magnitude"]): {
                    "detection_rate": round(float(row["detection_rate"]), 4),
                    "n": int(row["n"]),
                    "numeric_magnitude_mean": round(float(row["numeric_magnitude_mean"]), 4),
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
            "detection_rate":        float(sub["detected"].mean()),
            "similarity_score_mean": float(sub["similarity_score"].mean()),
        }
    out["by_edit_type"] = per_et

    # --- perfect edits (mag=0): false positive analysis ---
    if perfect_df is not None and not perfect_df.empty:
        e1_perf = perfect_df[
            (perfect_df["experiment"] == "experiment_1")
            & perfect_df["parse_success"]
            & perfect_df["detected_difference"].notna()
        ].copy()
        if not e1_perf.empty:
            e1_perf["detected"] = e1_perf["detected_difference"].astype(bool)
            # Compute FPR two ways:
            #   false_positive_rate         — excludes _FPR_EXCLUDED_DIMS (used as
            #                                 baseline for blind/sensitive tests)
            #   false_positive_rate_all_dims — includes every dimension (for display)
            e1_perf_for_fpr = e1_perf[
                ~e1_perf["degradation_dimension"].isin(_FPR_EXCLUDED_DIMS)
            ]
            fp_overall = (
                float(e1_perf_for_fpr["detected"].mean())
                if not e1_perf_for_fpr.empty
                else 0.0
            )
            fp_overall_all = float(e1_perf["detected"].mean())
            sim_vals   = e1_perf["similarity_score"].dropna()
            perf_dims: dict[str, Any] = {}
            for dim in sorted(e1_perf["degradation_dimension"].unique()):
                sub = e1_perf[e1_perf["degradation_dimension"] == dim]
                sv  = sub["similarity_score"].dropna()
                perf_dims[dim] = {
                    "n": int(len(sub)),
                    "false_positive_rate": round(float(sub["detected"].mean()), 4),
                    "similarity_score_mean": round(float(sv.mean()), 4) if len(sv) else None,
                    "similarity_score_distribution": {
                        str(v): int(n)
                        for v, n in sub["similarity_score"].value_counts().sort_index().items()
                    },
                }
            out["perfect_edits"] = {
                "n": int(len(e1_perf)),
                # Adjusted FPR (excl. alignment_error) — used as binomial-test baseline
                "false_positive_rate": round(fp_overall, 4),
                # Raw FPR including all dims — for reporting / transparency
                "false_positive_rate_all_dims": round(fp_overall_all, 4),
                "similarity_score_mean": (
                    round(float(sim_vals.mean()), 4) if len(sim_vals) else None
                ),
                "similarity_score_std": (
                    round(float(sim_vals.std()), 4) if len(sim_vals) else None
                ),
                "similarity_score_distribution": {
                    str(v): int(n)
                    for v, n in e1_perf["similarity_score"].value_counts().sort_index().items()
                },
                "by_dimension": perf_dims,
            }

    return out


# ---------------------------------------------------------------------------
# Exp2 — instruction-following scores (called once per model)
# ---------------------------------------------------------------------------

def _exp2_stats(df: pd.DataFrame, perfect_df: pd.DataFrame | None = None) -> dict:
    """Compute Exp2 statistics for a single-model DataFrame.

    ``df`` is expected to contain only degraded stimuli (numeric_magnitude > 0),
    so global score descriptives reflect degraded-only performance.
    Perfect-edit stimuli (mag == 0) are passed separately via ``perfect_df``.
    """
    e2 = df[
        (df["experiment"] == "experiment_2")
        & df["parse_success"]
        & df["overall_quality"].notna()
    ].copy()

    if e2.empty:
        return {"error": "no exp2 data"}

    out: dict[str, Any] = {}

    # --- global descriptives per score column (degraded stimuli only) ---
    global_scores: dict[str, Any] = {}
    for col in EXP2_SCORE_COLS:
        vals = e2[col].dropna()
        global_scores[col] = {
            "mean":         round(float(vals.mean()),   4),
            "std":          round(float(vals.std()),    4),
            "median":       round(float(vals.median()), 4),
            "min":          int(vals.min()),
            "max":          int(vals.max()),
            "ceiling_rate": round(float((vals == 5).mean()), 4),
            "floor_rate":   round(float((vals == 1).mean()), 4),
            "distribution": {str(v): int(n) for v, n in vals.value_counts().sort_index().items()},
        }
    out["score_descriptives"] = global_scores

    # --- score intercorrelation matrix (per model) ---
    score_cols_avail = [c for c in EXP2_SCORE_COLS if c in e2.columns]
    corr_data = e2[score_cols_avail].dropna()
    corr_rho  = corr_data.corr(method="spearman").round(4)
    out["score_intercorrelation_spearman"] = {
        col: corr_rho[col].to_dict() for col in score_cols_avail
    }

    # --- Cronbach's alpha across the 5 score dimensions (per model) ---
    score_matrix = corr_data.values
    out["cronbach_alpha"] = round(_cronbach_alpha(score_matrix), 4)

    # --- per dimension ---
    dims = sorted(e2["degradation_dimension"].unique())
    per_dim: dict[str, Any] = {}
    for dim in dims:
        sub  = e2[e2["degradation_dimension"] == dim].copy()
        mags = sub["numeric_magnitude"].values
        dim_entry: dict[str, Any] = {"n": int(len(sub))}

        for col in EXP2_SCORE_COLS:
            vals = sub[col].dropna().values
            if len(vals) == 0:
                continue
            col_mags = sub.loc[sub[col].notna(), "numeric_magnitude"].values
            rho, p = _safe_spearman(col_mags, vals)
            dim_entry[col] = {
                "mean":         round(float(vals.mean()),  4),
                "std":          round(float(vals.std()),   4),
                "ceiling_rate": round(float((vals == 5).mean()), 4),
                "magnitude_rho": round(rho, 4) if not np.isnan(rho) else None,
                "magnitude_p":   round(p,   4) if not np.isnan(p)   else None,
            }

        # Per-magnitude mean overall_quality table
        oq_sub    = sub[sub["overall_quality"].notna()]
        mag_table = (
            oq_sub.groupby("degradation_magnitude")
            .agg(
                mean=("overall_quality", "mean"),
                std=("overall_quality", "std"),
                count=("overall_quality", "count"),
                numeric_magnitude_mean=("numeric_magnitude", "mean"),
            )
            .reset_index()
            .sort_values("numeric_magnitude_mean")
        )
        dim_entry["overall_quality_by_magnitude"] = {
            str(row["degradation_magnitude"]): {
                "mean": round(float(row["mean"]), 4),
                "std":  round(float(row["std"]) if not np.isnan(row["std"]) else 0.0, 4),
                "n":    int(row["count"]),
                "numeric_magnitude_mean": round(float(row["numeric_magnitude_mean"]), 4),
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
            "overall_quality_std":  round(float(sub["overall_quality"].std()),  4),
        }
    out["by_edit_type"] = per_et

    # --- perfect edits (mag=0): false-rating analysis ---
    if perfect_df is not None and not perfect_df.empty:
        e2_perf = perfect_df[
            (perfect_df["experiment"] == "experiment_2")
            & perfect_df["parse_success"]
            & perfect_df["overall_quality"].notna()
        ].copy()
        if not e2_perf.empty:
            perf_scores: dict[str, Any] = {}
            for col in EXP2_SCORE_COLS:
                vals = e2_perf[col].dropna()
                if len(vals) == 0:
                    continue
                perf_scores[col] = {
                    "mean":          round(float(vals.mean()), 4),
                    "std":           round(float(vals.std()),  4) if len(vals) > 1 else 0.0,
                    "below_5_rate":  round(float((vals < 5).mean()),  4),
                    "bad_rating_rate": round(float((vals <= 3).mean()), 4),
                    "distribution":  {str(v): int(n) for v, n in vals.value_counts().sort_index().items()},
                }
            perf_by_dim: dict[str, Any] = {}
            for dim in sorted(e2_perf["degradation_dimension"].unique()):
                sub = e2_perf[e2_perf["degradation_dimension"] == dim]
                dim_entry = {"n": int(len(sub))}
                for col in EXP2_SCORE_COLS:
                    vals = sub[col].dropna().values
                    if len(vals) == 0:
                        continue
                    dim_entry[col] = {
                        "mean":          round(float(vals.mean()), 4),
                        "below_5_rate":  round(float((vals < 5).mean()), 4),
                        "bad_rating_rate": round(float((vals <= 3).mean()), 4),
                    }
                perf_by_dim[dim] = dim_entry

            out["perfect_edits"] = {
                "n": int(len(e2_perf)),
                "score_descriptives": perf_scores,
                "by_dimension": perf_by_dim,
            }

    return out


# ---------------------------------------------------------------------------
# Noop analysis  (Exp2 noop stimuli — called once per model)
# ---------------------------------------------------------------------------

def _noop_stats(noop_df: pd.DataFrame) -> dict:
    """Compute stats for noop stimuli (original unedited image shown as 'edited').

    For Exp2: model should score 1 everywhere — the edit was never applied.
    Metrics:
        above_1_rate    — proportion of stimuli receiving any score > 1
        high_rating_rate — proportion receiving any score ≥ 3

    NOTE ON Exp1 FPR:
        The Exp1 identity-control false positive rate (used in the blind/sensitive
        test and the failure-mode table) comes from *perfect edits*
        (numeric_magnitude == 0) in _exp1_stats — NOT from noop stimuli.
        Noop stimuli in Exp1 present two DIFFERENT images (source vs. a comparison
        that differs from source), so detecting a difference there is NOT a false
        positive. The field below is named detection_rate_on_noop_pairs to make
        this distinction explicit; do NOT use it as the Exp1 identity-control FPR.
    """
    out: dict[str, Any] = {}

    # --- Exp1 noop: detection on noop pairs (NOT the identity-control FPR) ---
    # Noop pairs in Exp1 show two *different* images, so detection here is a
    # true positive, not a false positive. The identity-control FPR comes from
    # exp1.by_model.<model>.perfect_edits.false_positive_rate (see _exp1_stats).
    e1 = noop_df[
        (noop_df["experiment"] == "experiment_1")
        & noop_df["parse_success"]
        & noop_df["detected_difference"].notna()
    ].copy()

    if not e1.empty:
        e1["detected"] = e1["detected_difference"].astype(bool)
        det_overall = float(e1["detected"].mean())
        sim_vals    = e1["similarity_score"].dropna()

        per_et: dict[str, Any] = {}
        for et in sorted(e1["edit_type"].unique()):
            sub = e1[e1["edit_type"] == et]
            sv  = sub["similarity_score"].dropna()
            per_et[et] = {
                "n": int(len(sub)),
                "detection_rate_on_noop_pairs": round(float(sub["detected"].mean()), 4),
                "similarity_score_mean": round(float(sv.mean()), 4) if len(sv) else None,
                "similarity_score_distribution": {
                    str(v): int(n)
                    for v, n in sub["similarity_score"].value_counts().sort_index().items()
                },
            }

        out["exp1"] = {
            "n": int(len(e1)),
            # IMPORTANT: this is detection_rate_on_noop_pairs, NOT the Exp1
            # identity-control FPR. Use exp1.by_model.<model>.perfect_edits
            # .false_positive_rate for the identity-control FPR.
            "detection_rate_on_noop_pairs": round(det_overall, 4),
            "similarity_score_mean": (
                round(float(sim_vals.mean()), 4) if len(sim_vals) else None
            ),
            "similarity_score_std": (
                round(float(sim_vals.std()), 4) if len(sim_vals) else None
            ),
            "similarity_score_distribution": {
                str(v): int(n)
                for v, n in e1["similarity_score"].value_counts().sort_index().items()
            },
            "by_edit_type": per_et,
        }

    # --- Exp2 noop ---
    e2 = noop_df[
        (noop_df["experiment"] == "experiment_2")
        & noop_df["parse_success"]
        & noop_df["overall_quality"].notna()
    ].copy()

    if not e2.empty:
        global_scores: dict[str, Any] = {}
        for col in EXP2_SCORE_COLS:
            vals = e2[col].dropna()
            if len(vals) == 0:
                continue
            global_scores[col] = {
                "mean": round(float(vals.mean()), 4),
                "std":  round(float(vals.std()),  4) if len(vals) > 1 else 0.0,
                # fraction with any score > 1 (model was fooled — scale floor is 1)
                "above_1_rate":    round(float((vals > 1).mean()), 4),
                # fraction with score ≥ 3 (high rating despite no edit)
                "high_rating_rate": round(float((vals >= 3).mean()), 4),
                "distribution": {
                    str(v): int(n) for v, n in vals.value_counts().sort_index().items()
                },
            }

        per_et_e2: dict[str, Any] = {}
        for et in sorted(e2["edit_type"].unique()):
            sub = e2[e2["edit_type"] == et]
            et_entry: dict[str, Any] = {"n": int(len(sub))}
            for col in EXP2_SCORE_COLS:
                vals = sub[col].dropna().values
                if len(vals) == 0:
                    continue
                et_entry[col] = {
                    "mean":          round(float(vals.mean()), 4),
                    "above_1_rate":   round(float((vals > 1).mean()),  4),
                    "high_rating_rate": round(float((vals >= 3).mean()), 4),
                }
            per_et_e2[et] = et_entry

        out["exp2"] = {
            "n": int(len(e2)),
            "score_descriptives": global_scores,
            "by_edit_type": per_et_e2,
        }

    return out


# ---------------------------------------------------------------------------
# Cross-experiment gap analysis (called once per model)
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

    rho, p = _safe_spearman(
        merged["similarity_score"].values, merged["overall_quality"].values
    )
    out["global_spearman_sim_vs_oq"] = {"rho": round(rho, 4), "p": round(p, 6)}

    rho_d, p_d = _safe_pointbiserial(
        merged["overall_quality"].values.astype(float),
        pd.to_numeric(merged["detected_difference"], errors="coerce").fillna(0).values.astype(float),
    )
    out["global_pointbiserial_detected_vs_oq"] = {"r": round(rho_d, 4), "p": round(p_d, 6)}

    per_dim: dict[str, Any] = {}
    for dim in sorted(merged["degradation_dimension"].unique()):
        sub = merged[merged["degradation_dimension"] == dim]
        if len(sub) < 3:
            continue

        rho, p = _safe_spearman(
            sub["similarity_score"].values, sub["overall_quality"].values
        )
        gap = float(sub["overall_quality"].mean() - sub["similarity_score"].mean())

        detected_mask = pd.to_numeric(
            sub["detected_difference"], errors="coerce"
        ).fillna(0).astype(bool)
        n_detected = detected_mask.sum()
        exp2_high_given_detected = (
            float((sub.loc[detected_mask, "overall_quality"] >= 4).mean())
            if n_detected > 0 else float("nan")
        )

        per_dim[dim] = {
            "n": int(len(sub)),
            "spearman_sim_vs_oq_rho": round(rho, 4) if not np.isnan(rho) else None,
            "spearman_sim_vs_oq_p":   round(p,   4) if not np.isnan(p)   else None,
            "mean_exp1_similarity": round(float(sub["similarity_score"].mean()), 4),
            "mean_exp2_oq":         round(float(sub["overall_quality"].mean()),  4),
            "score_gap_exp2_minus_exp1": round(gap, 4),
            "exp2_high_score_rate_when_exp1_detected": (
                round(exp2_high_given_detected, 4)
                if not np.isnan(exp2_high_given_detected) else None
            ),
        }

    out["by_dimension"] = per_dim

    det_bool  = pd.to_numeric(
        merged["detected_difference"], errors="coerce"
    ).fillna(0).astype(bool)
    blind_spot = merged[det_bool & (merged["overall_quality"] >= 4)]
    out["blind_spot_count"] = len(blind_spot)
    out["blind_spot_rate"]  = round(
        float(len(blind_spot) / max(1, det_bool.sum())), 4
    )

    return out


# ---------------------------------------------------------------------------
# FDR correction — applied to all primary test p-values after computation
# ---------------------------------------------------------------------------

def _apply_fdr_correction(out: dict, alpha: float = _PRIMARY_ALPHA) -> None:
    """Add BH-corrected p-values alongside raw p-values in the stats dict.

    Traverses exp1.by_model and exp2.by_model to collect all Spearman /
    point-biserial test p-values, applies BH correction jointly, and writes
    <original_key>_corrected entries into the same sub-dicts.

    The raw p-values at their original keys are left untouched.
    """
    entries: list[tuple[dict, str, float]] = []  # (mutable_dict, key, raw_p)

    # Collect exp1 p-values
    for _model, e1 in out.get("exp1", {}).get("by_model", {}).items():
        for _dim, d in e1.get("by_dimension", {}).items():
            for key in ("magnitude_detection_p", "magnitude_similarity_p"):
                p = d.get(key)
                if p is not None and not np.isnan(float(p)):
                    entries.append((d, key, float(p)))

    # Collect exp2 p-values
    for _model, e2 in out.get("exp2", {}).get("by_model", {}).items():
        for _dim, d in e2.get("by_dimension", {}).items():
            for col in EXP2_SCORE_COLS:
                col_dict = d.get(col, {})
                if not isinstance(col_dict, dict):
                    continue
                p = col_dict.get("magnitude_p")
                if p is not None and not np.isnan(float(p)):
                    entries.append((col_dict, "magnitude_p", float(p)))

    if not entries:
        return

    pvalues  = [p for _, _, p in entries]
    corrected = _bh_correction(pvalues, alpha)

    for (d, key, _raw_p), corr_p in zip(entries, corrected):
        d[key + "_corrected"] = round(corr_p, 6)


# ---------------------------------------------------------------------------
# Sensitivity ranking + blind spot classification (Issues 5, 8, 15)
# ---------------------------------------------------------------------------

def _sensitivity_rank(out: dict, exp1_by_model: dict, exp2_by_model: dict) -> dict:
    """Rank dimensions by sensitivity and classify (dimension, model) blind spots.

    Blind-spot criterion: detection_rate is NOT significantly above the model's
    adjusted Exp1 false positive rate (from perfect edits, excluding
    _FPR_EXCLUDED_DIMS).  A one-sided binomial test is used; p-values are
    BH-corrected across all (dimension, model) cells.

    Classification:
        "sensitive" — corrected p < 0.05
        "blind"     — corrected p ≥ 0.05  (cannot reject H0: rate ≤ FPR)
        "unknown"   — insufficient data (n < 4)
    """
    models = sorted(set(exp1_by_model.keys()) | set(exp2_by_model.keys()))

    # Collect all dimensions present across models
    all_dims: set[str] = set()
    for m in models:
        all_dims.update(exp1_by_model.get(m, {}).get("by_dimension", {}).keys())
        all_dims.update(exp2_by_model.get(m, {}).get("by_dimension", {}).keys())
    dims = sorted(all_dims)

    # ------------------------------------------------------------------ #
    # Detection rate matrix (dim × model)                                #
    # ------------------------------------------------------------------ #
    detection_rates: dict[str, dict[str, Any]] = {dim: {} for dim in dims}
    for dim in dims:
        for model in models:
            d = exp1_by_model.get(model, {}).get("by_dimension", {}).get(dim, {})
            detection_rates[dim][model] = d.get("detection_rate")

    # ------------------------------------------------------------------ #
    # Blind-spot binomial tests (vs. model adjusted FPR)                 #
    # ------------------------------------------------------------------ #
    # Collect (dim, model, k, n, fpr, n_ctrl) entries for tests
    test_entries: list[tuple[str, str, int, int, float, int]] = []

    for dim in dims:
        for model in models:
            e1m = exp1_by_model.get(model, {})
            d   = e1m.get("by_dimension", {}).get(dim, {})
            n   = d.get("n", 0)
            det = d.get("detection_rate")

            if det is None or n < 4:
                continue

            # Adjusted FPR from perfect edits (excludes _FPR_EXCLUDED_DIMS)
            perf   = e1m.get("perfect_edits", {})
            fpr    = perf.get("false_positive_rate", 0.0) or 0.0
            n_ctrl = perf.get("n", 0)
            k      = int(round(det * n))
            test_entries.append((dim, model, k, n, fpr, n_ctrl))

    # Run binomial tests against the model's adjusted FPR
    raw_pvals: list[float] = []
    for _dim, _model, k, n, fpr, _n_ctrl in test_entries:
        p_fpr = max(float(fpr), 1e-10)  # avoid p=0 in binomtest
        try:
            res = scipy_stats.binomtest(k, n, p_fpr, alternative="greater")
            raw_pvals.append(float(res.pvalue))
        except Exception:
            raw_pvals.append(float("nan"))

    # BH correction on blind-spot tests
    corrected_pvals = _bh_correction(raw_pvals, alpha=_BLIND_SPOT_ALPHA)

    # Assemble blind/sensitive classification table
    blind_sensitive: dict[str, dict[str, Any]] = {dim: {} for dim in dims}
    for (dim, model, k, n, fpr, n_ctrl), raw_p, corr_p in zip(
        test_entries, raw_pvals, corrected_pvals
    ):
        is_nan = np.isnan(raw_p) or np.isnan(corr_p)
        classification = (
            "unknown"
            if is_nan
            else ("sensitive" if corr_p < _BLIND_SPOT_ALPHA else "blind")
        )
        # Newcombe CI for Δ = DetectionRate − FPR
        k_ctrl = int(round(fpr * n_ctrl)) if n_ctrl > 0 else 0
        delta, ci_lo, ci_hi = _newcombe_ci(k, n, k_ctrl, n_ctrl)
        blind_sensitive[dim][model] = {
            "classification": classification,
            "detection_rate": detection_rates[dim].get(model),
            "p_raw":          round(raw_p,  6) if not np.isnan(raw_p)  else None,
            "p_corrected":    round(corr_p, 6) if not np.isnan(corr_p) else None,
            "delta":          round(delta,  4),
            "ci_lower":       round(ci_lo,  4),
            "ci_upper":       round(ci_hi,  4),
        }

    # Fill unknowns for (dim, model) cells that had no test
    for dim in dims:
        for model in models:
            if model not in blind_sensitive[dim]:
                blind_sensitive[dim][model] = {
                    "classification": "unknown",
                    "detection_rate": detection_rates[dim].get(model),
                    "p_raw":       None,
                    "p_corrected": None,
                    "delta":       None,
                    "ci_lower":    None,
                    "ci_upper":    None,
                }

    # ------------------------------------------------------------------ #
    # Summary statistics                                                  #
    # ------------------------------------------------------------------ #
    def _mean_det(dim: str) -> float:
        rates = [
            v for v in detection_rates[dim].values()
            if v is not None and not np.isnan(v)
        ]
        return float(np.mean(rates)) if rates else float("nan")

    dims_sorted_asc = sorted(dims, key=_mean_det)

    universally_blind = [
        d for d in dims
        if models
        and all(
            blind_sensitive[d].get(m, {}).get("classification") == "blind"
            for m in models
        )
    ]
    model_specific_blind: dict[str, list[str]] = {}
    for m in models:
        model_specific_blind[m] = [
            d for d in dims
            if blind_sensitive[d].get(m, {}).get("classification") == "blind"
            and d not in universally_blind
        ]

    # Detail table (also includes Exp2 OQ info for comprehensive ranking)
    detail: dict[str, Any] = {}
    for dim in dims:
        e2_samples: list[dict] = [
            exp2_by_model.get(m, {}).get("by_dimension", {}).get(dim, {})
            for m in models
        ]
        oq_means = [
            d.get("overall_quality", {}).get("mean")
            for d in e2_samples
            if isinstance(d.get("overall_quality"), dict)
        ]
        detail[dim] = {
            "mean_detection_rate":      _mean_det(dim),
            "detection_rate_by_model":  detection_rates[dim],
            "mean_exp2_oq": (
                round(float(np.mean([v for v in oq_means if v is not None])), 4)
                if oq_means else None
            ),
            "blind_sensitive_by_model": blind_sensitive[dim],
        }

    # ------------------------------------------------------------------ #
    # Inter-model agreement — Fleiss' kappa on blind/sensitive table     #
    # ------------------------------------------------------------------ #
    kappa = _fleiss_kappa(blind_sensitive)

    return {
        "dims_by_mean_detection_rate_asc": dims_sorted_asc,
        "blind_sensitive_table": blind_sensitive,
        "universally_blind": universally_blind,
        "model_specific_blind": model_specific_blind,
        "inter_model_fleiss_kappa": round(kappa, 4) if not np.isnan(kappa) else None,
        "detail": detail,
    }


# ---------------------------------------------------------------------------
# Failure-mode table
# ---------------------------------------------------------------------------

def _failure_mode_table(out: dict, models: list[str]) -> list[dict]:
    """Assemble per-model failure-mode summary table.

    Columns per model:
        parse_success_experiment_1 / _experiment_2
        exp1_false_positive_rate  (from perfect edits)
        noop_above_1_rate         (Exp2 noop, any score > 1)
        noop_high_rating_rate     (Exp2 noop, any score ≥ 3)
        score_gap_exp2_minus_exp1 (mean Exp2 OQ − mean Exp1 sim score)
    """
    rows: list[dict] = []
    ov = out.get("overview", {})

    for model in models:
        row: dict[str, Any] = {"model": model}

        # Parse success rates
        for exp_key in ("experiment_1", "experiment_2"):
            per_model = ov.get(exp_key, {}).get("by_model", {}).get(model, {})
            row[f"parse_success_{exp_key}"] = per_model.get("parse_success_rate")

        # Exp1 FPR from perfect edits
        e1m  = out.get("exp1", {}).get("by_model", {}).get(model, {})
        perf = e1m.get("perfect_edits", {})
        row["exp1_false_positive_rate"] = perf.get("false_positive_rate")

        # Noop Exp2 rates
        noop_m  = out.get("noop", {}).get("by_model", {}).get(model, {})
        noop_e2 = noop_m.get("exp2", {})
        oq_desc = noop_e2.get("score_descriptives", {}).get("overall_quality", {})
        row["noop_above_1_rate"]    = oq_desc.get("above_1_rate")
        row["noop_high_rating_rate"] = oq_desc.get("high_rating_rate")

        # Score gap (global)
        e2m        = out.get("exp2", {}).get("by_model", {}).get(model, {})
        oq_mean_e2 = e2m.get("score_descriptives", {}).get("overall_quality", {}).get("mean")
        sim_mean_e1 = e1m.get("similarity_score_mean")
        if oq_mean_e2 is not None and sim_mean_e1 is not None:
            row["score_gap_exp2_minus_exp1"] = round(
                float(oq_mean_e2) - float(sim_mean_e1), 4
            )
        else:
            row["score_gap_exp2_minus_exp1"] = None

        rows.append(row)

    return rows


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def compute_all_stats(
    df: pd.DataFrame,
    noop_df: pd.DataFrame | None = None,
    perfect_df: pd.DataFrame | None = None,
) -> dict:
    """Compute the full statistics bundle from a combined exp1+exp2 DataFrame.

    All stats are computed per (model, dimension) cell.  BH FDR correction
    (α=5 %) is applied jointly to all primary Spearman/point-biserial test
    p-values and, separately, to all blind-spot binomial test p-values.

    Parameters
    ----------
    df:
        Combined DataFrame from load_results() calls for Exp1 and Exp2.
        Contains only degraded stimuli (numeric_magnitude > 0).
    noop_df:
        Optional DataFrame from load_noop_results().  When provided, a
        'noop.by_model' section is included and the failure_mode_table is
        populated with noop rates.
    perfect_df:
        Optional DataFrame from load_perfect_results().  When provided,
        perfect-edit analysis is included in exp1 and exp2 sections.

    Returns
    -------
    dict with keys:
        overview, exp1, exp2, cross_exp, sensitivity_rank,
        failure_mode_table[, noop]
    """
    models = sorted(df["model"].unique().tolist())

    # ------------------------------------------------------------------ #
    # Per-model statistics                                                #
    # ------------------------------------------------------------------ #
    exp1_by_model:      dict[str, dict] = {}
    exp2_by_model:      dict[str, dict] = {}
    cross_exp_by_model: dict[str, dict] = {}

    for model in models:
        model_df   = df[df["model"] == model]
        model_perf = (
            perfect_df[perfect_df["model"] == model]
            if perfect_df is not None and not perfect_df.empty
            else None
        )
        exp1_by_model[model]      = _exp1_stats(model_df, perfect_df=model_perf)
        exp2_by_model[model]      = _exp2_stats(model_df, perfect_df=model_perf)
        cross_exp_by_model[model] = _cross_exp_stats(model_df)

    out = {
        "overview":   _overview(df),
        "exp1":       {"by_model": exp1_by_model},
        "exp2":       {"by_model": exp2_by_model},
        "cross_exp":  {"by_model": cross_exp_by_model},
    }

    # ------------------------------------------------------------------ #
    # BH FDR correction on primary Spearman tests                        #
    # ------------------------------------------------------------------ #
    _apply_fdr_correction(out)

    # ------------------------------------------------------------------ #
    # Noop analysis                                                       #
    # ------------------------------------------------------------------ #
    if noop_df is not None and not noop_df.empty:
        noop_by_model: dict[str, dict] = {}
        for model in models:
            model_noop = noop_df[noop_df["model"] == model]
            if not model_noop.empty:
                noop_by_model[model] = _noop_stats(model_noop)
        out["noop"] = {"by_model": noop_by_model}

    # ------------------------------------------------------------------ #
    # Sensitivity ranking + blind-spot classification                     #
    # (needs FDR-corrected primary p-values and noop section to exist)   #
    # ------------------------------------------------------------------ #
    out["sensitivity_rank"] = _sensitivity_rank(out, exp1_by_model, exp2_by_model)

    # ------------------------------------------------------------------ #
    # Failure-mode table (reads from all sections above)                  #
    # ------------------------------------------------------------------ #
    out["failure_mode_table"] = _failure_mode_table(out, models)

    return out


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def save_report(stats: dict, path: str | Path) -> None:
    """Save statistics bundle as JSON."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2, ensure_ascii=False, default=str)


def save_print_report(stats: dict, path: str | Path) -> None:  # noqa: C901
    """Write a thesis-focused human-readable statistics reference file.

    Covers all quantitative tables and key numbers cited in thesis chapters 4.1–4.7.
    Exp1 identity-control FPR is always sourced from perfect_edits (NOT noop).
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    lines: list[str] = []

    def _h(title: str) -> None:
        lines.append("")
        lines.append("=" * 72)
        lines.append(title)
        lines.append("=" * 72)

    def _sub(title: str) -> None:
        lines.append("")
        lines.append(title)
        lines.append("-" * len(title))

    def _row(*cells: str, widths: list[int] | None = None) -> None:
        if widths:
            lines.append("  " + "  ".join(str(c).ljust(w) for c, w in zip(cells, widths)))
        else:
            lines.append("  " + "  ".join(str(c) for c in cells))

    def _pct(v) -> str:
        if v is None or (isinstance(v, float) and np.isnan(v)):
            return "—"
        return f"{float(v):.1%}"

    def _f(v, decimals: int = 3) -> str:
        if v is None or (isinstance(v, float) and np.isnan(v)):
            return "—"
        return f"{float(v):.{decimals}f}"

    def _i(v) -> str:
        return str(v) if v is not None else "—"

    lines.append("VLM JUDGE CALIBRATION — THESIS STATISTICS REFERENCE")
    lines.append("(All Exp1 identity-control FPR values sourced from perfect_edits,")
    lines.append(" NOT from noop pairs. See stats.py note in _noop_stats.)")

    ov        = stats.get("overview", {})
    exp1_bm   = stats.get("exp1", {}).get("by_model", {})
    exp2_bm   = stats.get("exp2", {}).get("by_model", {})
    noop_bm   = stats.get("noop", {}).get("by_model", {})
    sr        = stats.get("sensitivity_rank", {})
    fmt       = stats.get("failure_mode_table", [])
    cross_bm  = stats.get("cross_exp", {}).get("by_model", {})
    models    = sorted(ov.get("models", []))

    # ------------------------------------------------------------------ #
    # 4.1 Data Quality                                                    #
    # ------------------------------------------------------------------ #
    _h("SECTION 4.1 — DATA QUALITY AND CORPUS OVERVIEW")

    _sub("Table 4.1: Stimulus counts by degradation dimension")
    all_dims = sorted(ov.get("dimensions", []))
    e2_all   = stats.get("exp2", {}).get("by_model", {})
    # Collect per-dimension n from exp1
    dim_counts: dict[str, int] = {}
    for model in models:
        for dim, d in exp1_bm.get(model, {}).get("by_dimension", {}).items():
            dim_counts[dim] = dim_counts.get(dim, 0) + d.get("n", 0)
    # Use the first model's counts (they should match across models)
    first_model = models[0] if models else None
    lines.append(f"  Total corpus  : {ov.get('total_records', '?')} records")
    lines.append(f"  Unique stimuli: {ov.get('stimuli', '?')}")
    lines.append(f"  Dimensions    : {len(ov.get('dimensions', []))}")
    lines.append(f"  Edit types    : {len(ov.get('edit_types', []))}")
    lines.append(f"  Models        : {models}")
    lines.append("")
    if first_model:
        lines.append(f"  {'Dimension':<28} {'Stimuli (Exp1 records, first model)':>10}")
        lines.append(f"  {'-'*28} {'-'*38}")
        for dim in sorted(
            exp1_bm.get(first_model, {}).get("by_dimension", {}).keys()
        ):
            n = exp1_bm[first_model]["by_dimension"][dim].get("n", "?")
            lines.append(f"  {dim:<28} {_i(n):>10}")

    _sub("Table 4.2: Response counts and parse success rates by model and experiment")
    w = [28, 14, 12, 14, 12]
    _row("Model", "Exp1 records", "Exp1 PSR", "Exp2 records", "Exp2 PSR", widths=w)
    _row("-"*28, "-"*14, "-"*12, "-"*14, "-"*12, widths=w)
    for model in models:
        e1d = ov.get("experiment_1", {}).get("by_model", {}).get(model, {})
        e2d = ov.get("experiment_2", {}).get("by_model", {}).get(model, {})
        _row(
            model,
            _i(e1d.get("records")),
            _pct(e1d.get("parse_success_rate")),
            _i(e2d.get("records")),
            _pct(e2d.get("parse_success_rate")),
            widths=w,
        )

    # ------------------------------------------------------------------ #
    # 4.2 Control Conditions                                              #
    # ------------------------------------------------------------------ #
    _h("SECTION 4.2 — CONTROL CONDITIONS")

    _sub("Table 4.3: Identity control — FPR and mean sim scores (Exp1 perfect_edits)")
    lines.append("  Source: exp1.by_model.<model>.perfect_edits")
    lines.append("  (Correctly-edited image shown as BOTH reference and comparison.)")
    lines.append(f"  NOTE: 'FPR (adjusted)' excludes {sorted(_FPR_EXCLUDED_DIMS)} — used as")
    lines.append("  the binomial-test baseline in Table 4.7 / 4.14. 'FPR (all dims)'")
    lines.append("  is the raw rate across all dimensions (shown for transparency).")
    w3 = [28, 6, 16, 16, 20]
    _row("Model", "n", "FPR (adjusted)", "FPR (all dims)", "Mean sim score", widths=w3)
    _row("-"*28, "-"*6, "-"*16, "-"*16, "-"*20, widths=w3)
    for model in models:
        perf = exp1_bm.get(model, {}).get("perfect_edits", {})
        _row(
            model,
            _i(perf.get("n")),
            _pct(perf.get("false_positive_rate")),
            _pct(perf.get("false_positive_rate_all_dims")),
            _f(perf.get("similarity_score_mean")),
            widths=w3,
        )

    lines.append("")
    lines.append("  Per-dimension FPR breakdown (from perfect_edits.by_dimension):")
    lines.append(f"  (Dimensions in {sorted(_FPR_EXCLUDED_DIMS)} are excluded from the adjusted FPR above.)")
    perf_dims: set[str] = set()
    for model in models:
        perf_dims.update(
            exp1_bm.get(model, {}).get("perfect_edits", {}).get("by_dimension", {}).keys()
        )
    if perf_dims:
        w_pd = [28] + [12] * len(models)
        _row("Dimension", *[f"{m[:10]} FPR" for m in models], widths=w_pd)
        _row("-"*28, *["-"*12]*len(models), widths=w_pd)
        for dim in sorted(perf_dims):
            excl_marker = " [EXCL]" if dim in _FPR_EXCLUDED_DIMS else ""
            vals = [
                _pct(
                    exp1_bm.get(m, {})
                    .get("perfect_edits", {})
                    .get("by_dimension", {})
                    .get(dim, {})
                    .get("false_positive_rate")
                )
                for m in models
            ]
            _row(dim + excl_marker, *vals, widths=w_pd)

    _sub("Table 4.4: Source-target control — detection on noop pairs (Exp1)")
    lines.append("  Source: noop.by_model.<model>.exp1")
    lines.append("  (Noop pairs present two DIFFERENT images; detection here is NOT FPR.)")
    w4 = [28, 6, 14, 20]
    _row("Model", "n", "Detection rate", "Mean sim score", widths=w4)
    _row("-"*28, "-"*6, "-"*14, "-"*20, widths=w4)
    for model in models:
        n1 = noop_bm.get(model, {}).get("exp1", {})
        _row(
            model,
            _i(n1.get("n")),
            _pct(n1.get("detection_rate_on_noop_pairs")),
            _f(n1.get("similarity_score_mean")),
            widths=w4,
        )

    _sub("Table 4.5: Instruction-not-followed control — Exp2 noop scores")
    lines.append("  Source: noop.by_model.<model>.exp2.score_descriptives")
    lines.append("  (Expected score = 1 everywhere; edit was never applied.)")
    w5 = [24] + [14] * len(models)
    _row("Score dimension", *[f"{m[:12]} mean" for m in models], widths=w5)
    _row("-"*24, *["-"*14]*len(models), widths=w5)
    for col in EXP2_SCORE_COLS:
        vals = [
            _f(
                noop_bm.get(m, {})
                .get("exp2", {})
                .get("score_descriptives", {})
                .get(col, {})
                .get("mean")
            )
            for m in models
        ]
        _row(col, *vals, widths=w5)

    # ------------------------------------------------------------------ #
    # 4.3 Exp1 Perceptual Sensitivity                                     #
    # ------------------------------------------------------------------ #
    _h("SECTION 4.3 — EXPERIMENT 1: PERCEPTUAL SENSITIVITY")

    _sub("Table 4.6: Overall detection rates and mean similarity scores (all degraded stimuli)")
    lines.append("  Source: exp1.by_model.<model>")
    w6 = [28, 16, 22, 20]
    _row("Model", "Detection rate", "Mean sim score", "Sim score std", widths=w6)
    _row("-"*28, "-"*16, "-"*22, "-"*20, widths=w6)
    for model in models:
        e1 = exp1_bm.get(model, {})
        _row(
            model,
            _pct(e1.get("overall_detection_rate")),
            _f(e1.get("similarity_score_mean")),
            _f(e1.get("similarity_score_std")),
            widths=w6,
        )

    _sub("Table 4.7: Per-dimension detection rates and Spearman ρ (Exp1)")
    lines.append("  Source: exp1.by_model.<model>.by_dimension + sensitivity_rank.blind_sensitive_table")
    lines.append(f"  Classification uses BH-corrected binomial test vs. adjusted FPR (excl. {sorted(_FPR_EXCLUDED_DIMS)}).")
    bs_table  = sr.get("blind_sensitive_table", {})
    all_dims2 = sorted(sr.get("detail", {}).keys())
    w7 = [24] + [14, 8, 6] * len(models)
    header_cells = ["Dimension"]
    for m in models:
        short = m[:8]
        header_cells += [f"{short} DR", f"{short} ρ", "cls"]
    _row(*header_cells, widths=w7)
    _row(*(["-"*24] + ["-"*14, "-"*8, "-"*6] * len(models)), widths=w7)
    for dim in all_dims2:
        row_cells = [dim]
        for model in models:
            d     = exp1_bm.get(model, {}).get("by_dimension", {}).get(dim, {})
            dr    = _pct(d.get("detection_rate"))
            rho   = d.get("magnitude_similarity_rho")
            rho_s = f"{rho:+.3f}" if rho is not None and not np.isnan(float(rho)) else "—"
            cls   = bs_table.get(dim, {}).get(model, {}).get("classification", "?")
            cls_s = "S" if cls == "sensitive" else ("B" if cls == "blind" else "?")
            row_cells += [dr, rho_s, cls_s]
        _row(*row_cells, widths=w7)

    lines.append("")
    lines.append("  Blind/Sensitive Summary:")
    ub = sr.get("universally_blind", [])
    lines.append(f"    Universally blind (all models) : {ub}")
    msb = sr.get("model_specific_blind", {})
    for m in models:
        dims_blind = msb.get(m, [])
        lines.append(f"    Model-specific blind [{m}]: {dims_blind}")
    kappa = sr.get("inter_model_fleiss_kappa")
    lines.append(f"    Inter-model Fleiss' kappa     : {_f(kappa, 3)}")

    # ------------------------------------------------------------------ #
    # 4.4 Detection Thresholds                                            #
    # ------------------------------------------------------------------ #
    _h("SECTION 4.4 — DETECTION THRESHOLDS")

    _sub("Table 4.8: 50% and 75% detection threshold tiers and numeric values (Exp1)")
    lines.append("  Source: exp1.by_model.<model>.by_dimension.threshold_tier_50pct etc.")
    w8 = [24] + [12, 9] * len(models)
    header_50 = ["Dimension"]
    for m in models:
        short = m[:8]
        header_50 += [f"{short} 50%tier", "val"]
    _row(*header_50, widths=w8)
    _row(*(["-"*24] + ["-"*12, "-"*9] * len(models)), widths=w8)
    for dim in all_dims2:
        row_cells = [dim]
        for model in models:
            d      = exp1_bm.get(model, {}).get("by_dimension", {}).get(dim, {})
            tier   = d.get("threshold_tier_50pct", "NC")
            val    = d.get("threshold_numeric_50pct")
            val_s  = _f(val, 2) if val is not None else "—"
            row_cells += [str(tier), val_s]
        _row(*row_cells, widths=w8)

    lines.append("")
    lines.append("  75% threshold tiers:")
    w8b = [24] + [12, 9] * len(models)
    header_75 = ["Dimension"]
    for m in models:
        short = m[:8]
        header_75 += [f"{short} 75%tier", "val"]
    _row(*header_75, widths=w8b)
    _row(*(["-"*24] + ["-"*12, "-"*9] * len(models)), widths=w8b)
    for dim in all_dims2:
        row_cells = [dim]
        for model in models:
            d      = exp1_bm.get(model, {}).get("by_dimension", {}).get(dim, {})
            tier   = d.get("threshold_tier_75pct", "NC")
            val    = d.get("threshold_numeric_75pct")
            val_s  = _f(val, 2) if val is not None else "—"
            row_cells += [str(tier), val_s]
        _row(*row_cells, widths=w8b)

    # ------------------------------------------------------------------ #
    # 4.5 Exp2 Edit Judgment                                              #
    # ------------------------------------------------------------------ #
    _h("SECTION 4.5 — EXPERIMENT 2: EDIT JUDGMENT SENSITIVITY")

    _sub("Table 4.9: Exp2 score descriptives per model (all degraded stimuli)")
    lines.append("  Source: exp2.by_model.<model>.score_descriptives")
    w9 = [24, 8] + [8, 10, 8] * len(models)
    header9 = ["Score dimension", "Stat"]
    for m in models:
        short = m[:6]
        header9 += [f"{short} mean", f"{short} ceil%", f"{short} flr%"]
    _row(*header9, widths=w9)
    _row(*(["-"*24, "-"*8] + ["-"*8, "-"*10, "-"*8] * len(models)), widths=w9)
    for col in EXP2_SCORE_COLS:
        cells9 = [col, ""]
        for m in models:
            s = exp2_bm.get(m, {}).get("score_descriptives", {}).get(col, {})
            cells9 += [
                _f(s.get("mean"), 2),
                _pct(s.get("ceiling_rate")),
                _pct(s.get("floor_rate")),
            ]
        _row(*cells9, widths=w9)

    _sub("Table 4.10: Spearman intercorrelations and Cronbach's alpha per model (Exp2)")
    lines.append("  Source: exp2.by_model.<model>.score_intercorrelation_spearman + cronbach_alpha")
    for model in models:
        e2 = exp2_bm.get(model, {})
        alpha = e2.get("cronbach_alpha")
        lines.append(f"  [{model}]  Cronbach's α = {_f(alpha, 3)}")
        corr = e2.get("score_intercorrelation_spearman", {})
        cols_avail = [c for c in EXP2_SCORE_COLS if c in corr]
        if cols_avail:
            short_labels = [c[:8] for c in cols_avail]
            lines.append("  " + " " * 26 + "  ".join(f"{s:>8}" for s in short_labels))
            for row_col in cols_avail:
                row_vals = corr.get(row_col, {})
                vals_str = "  ".join(
                    f"{row_vals.get(c, float('nan')):>8.3f}" for c in cols_avail
                )
                lines.append(f"    {row_col:<24}{vals_str}")
        lines.append("")

    _sub("Table 4.11: Exp2 overall_quality means and Spearman ρ per dimension")
    lines.append("  Source: exp2.by_model.<model>.by_dimension.overall_quality")
    w11 = [24] + [10, 8] * len(models)
    header11 = ["Dimension"]
    for m in models:
        short = m[:8]
        header11 += [f"{short} OQ μ", "ρ"]
    _row(*header11, widths=w11)
    _row(*(["-"*24] + ["-"*10, "-"*8] * len(models)), widths=w11)
    for dim in all_dims2:
        row_cells = [dim]
        for m in models:
            d  = exp2_bm.get(m, {}).get("by_dimension", {}).get(dim, {})
            oq = d.get("overall_quality", {})
            mn  = _f(oq.get("mean"), 2) if isinstance(oq, dict) else "—"
            rho = oq.get("magnitude_rho") if isinstance(oq, dict) else None
            rho_corr = (
                oq.get("magnitude_p_corrected") or oq.get("magnitude_p")
                if isinstance(oq, dict) else None
            )
            sig = ""
            if rho_corr is not None and not np.isnan(float(rho_corr)):
                if float(rho_corr) < 0.001:
                    sig = "***"
                elif float(rho_corr) < 0.01:
                    sig = "**"
                elif float(rho_corr) < 0.05:
                    sig = "*"
            rho_s = (f"{rho:+.3f}{sig}" if rho is not None and not np.isnan(float(rho)) else "n/s")
            row_cells += [mn, rho_s]
        _row(*row_cells, widths=w11)

    # ------------------------------------------------------------------ #
    # 4.6 Cross-Experiment Gap                                            #
    # ------------------------------------------------------------------ #
    _h("SECTION 4.6 — PERCEPTION-TO-JUDGMENT GAP (EXP1 vs EXP2)")

    _sub("Table 4.12: Global Spearman sim↔OQ and point-biserial detected↔OQ")
    lines.append("  Source: cross_exp.by_model.<model>")
    w12 = [28, 18, 18, 18]
    _row("Model", "Matched stimuli", "ρ (sim↔OQ)", "r_pb (det↔OQ)", widths=w12)
    _row("-"*28, "-"*18, "-"*18, "-"*18, widths=w12)
    for model in models:
        cx = cross_bm.get(model, {})
        rho_sim_oq = cx.get("global_spearman_sim_vs_oq", {}).get("rho")
        r_pb       = cx.get("global_pointbiserial_detected_vs_oq", {}).get("r")
        _row(
            model,
            _i(cx.get("stimuli_in_both_experiments")),
            _f(rho_sim_oq, 3),
            _f(r_pb, 3),
            widths=w12,
        )

    _sub("Table 4.13: Per-dimension score gap (Exp2 OQ − Exp1 sim) and cross-exp ρ")
    lines.append("  Source: cross_exp.by_model.<model>.by_dimension")
    w13 = [24] + [10, 8] * len(models)
    header13 = ["Dimension"]
    for m in models:
        short = m[:8]
        header13 += [f"{short} gap", "ρ"]
    _row(*header13, widths=w13)
    _row(*(["-"*24] + ["-"*10, "-"*8] * len(models)), widths=w13)
    for dim in all_dims2:
        row_cells = [dim]
        for m in models:
            d   = cross_bm.get(m, {}).get("by_dimension", {}).get(dim, {})
            gap = d.get("score_gap_exp2_minus_exp1")
            rho = d.get("spearman_sim_vs_oq_rho")
            gap_s = f"{gap:+.2f}" if gap is not None and not np.isnan(float(gap)) else "—"
            rho_p = d.get("spearman_sim_vs_oq_p")
            sig = ""
            if rho_p is not None and not np.isnan(float(rho_p)):
                if float(rho_p) < 0.001:
                    sig = "***"
                elif float(rho_p) < 0.01:
                    sig = "**"
                elif float(rho_p) < 0.05:
                    sig = "*"
                else:
                    sig = "n/s"
            rho_s = (
                f"{rho:+.3f}{sig}"
                if rho is not None and not np.isnan(float(rho)) else "n/s"
            )
            row_cells += [gap_s, rho_s]
        _row(*row_cells, widths=w13)

    # ------------------------------------------------------------------ #
    # 4.7 Cross-Model Blind Spots                                         #
    # ------------------------------------------------------------------ #
    _h("SECTION 4.7 — CROSS-MODEL AGREEMENT AND BLIND SPOTS")

    _sub("Table 4.14: Blind (B) / Sensitive (S) classification per (dimension, model)")
    lines.append("  Source: sensitivity_rank.blind_sensitive_table")
    lines.append(f"  FPR baseline = adjusted FPR from perfect_edits (excl. {sorted(_FPR_EXCLUDED_DIMS)}).")
    w14 = [24] + [18] * len(models)
    _row("Dimension", *[f"{m[:16]}" for m in models], widths=w14)
    _row("-"*24, *["-"*18]*len(models), widths=w14)
    for dim in all_dims2:
        row_cells = [dim]
        for m in models:
            cell  = bs_table.get(dim, {}).get(m, {})
            cls   = cell.get("classification", "?")
            dr    = cell.get("detection_rate")
            dr_s  = f"({_pct(dr)})" if dr is not None else ""
            cls_s = ("S" if cls == "sensitive" else ("B" if cls == "blind" else "?"))
            row_cells.append(f"{cls_s} {dr_s}")
        _row(*row_cells, widths=w14)

    lines.append("")
    lines.append("  Universally blind  : " + str(sr.get("universally_blind", [])))
    for m in models:
        dims_list = sr.get("model_specific_blind", {}).get(m, [])
        lines.append(f"  Model-specific blind [{m}]: {dims_list}")

    _sub("Table 4.15: Fleiss' kappa for inter-model agreement on blind/sensitive")
    lines.append("  Source: sensitivity_rank.inter_model_fleiss_kappa")
    kappa_val = sr.get("inter_model_fleiss_kappa")
    lines.append(f"  Across all {len(all_dims2)} dimensions ({len(models)} models): κ = {_f(kappa_val, 4)}")

    _sub("Sensitivity ranking (dimensions ordered by ascending mean detection rate)")
    lines.append("  Source: sensitivity_rank.dims_by_mean_detection_rate_asc")
    ranked = sr.get("dims_by_mean_detection_rate_asc", [])
    for i, dim in enumerate(ranked, 1):
        det = sr.get("detail", {}).get(dim, {}).get("mean_detection_rate")
        det_s = _pct(det)
        bs_strs = ", ".join(
            f"{m[:6]}: {bs_table.get(dim, {}).get(m, {}).get('classification', '?')}"
            for m in models
        )
        lines.append(f"  {i:>2}. {dim:<28} mean_det={det_s}  [{bs_strs}]")

    # ------------------------------------------------------------------ #
    # FAILURE-MODE SUMMARY                                                #
    # ------------------------------------------------------------------ #
    _h("FAILURE-MODE SUMMARY TABLE (per model)")
    lines.append(f"  exp1_FPR = adjusted FPR from perfect_edits (excl. {sorted(_FPR_EXCLUDED_DIMS)}).")
    lines.append("  Noop detection rates (Exp1 noop pairs) are a DIFFERENT measurement.")
    if fmt:
        w_fm = [28, 8, 8, 8, 10, 10, 8]
        _row("Model", "PSR-E1", "PSR-E2", "FPR", "Noop>1%", "Noop>=3%", "Gap", widths=w_fm)
        _row(*(["-"*w for w in w_fm]), widths=w_fm)
        for row in fmt:
            def _fmtp(v, pct: bool = True) -> str:
                if v is None or (isinstance(v, float) and np.isnan(v)):
                    return "—"
                return f"{v:.1%}" if pct else f"{v:+.3f}"
            _row(
                row["model"],
                _fmtp(row.get("parse_success_experiment_1")),
                _fmtp(row.get("parse_success_experiment_2")),
                _fmtp(row.get("exp1_false_positive_rate")),
                _fmtp(row.get("noop_above_1_rate")),
                _fmtp(row.get("noop_high_rating_rate")),
                _fmtp(row.get("score_gap_exp2_minus_exp1"), pct=False),
                widths=w_fm,
            )

    lines.append("")
    lines.append("=" * 72)
    lines.append("END OF THESIS STATISTICS REFERENCE")
    lines.append("=" * 72)

    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


def print_report(stats: dict) -> None:  # noqa: C901 (complexity OK for a report printer)
    """Print a human-readable summary to stdout."""
    ov = stats.get("overview", {})
    sr = stats.get("sensitivity_rank", {})

    sep = "-" * 72

    print(sep)
    print("VLM JUDGE CALIBRATION — NUMERICAL ANALYSIS REPORT")
    print(sep)

    # ------------------------------------------------------------------ #
    # Overview                                                            #
    # ------------------------------------------------------------------ #
    print(f"\nOVERVIEW")
    print(f"  Total records  : {ov.get('total_records', '?')}")
    print(f"  Unique stimuli : {ov.get('stimuli', '?')}")
    print(f"  Models         : {ov.get('models', [])}")

    for exp_key in ("experiment_1", "experiment_2"):
        ed = ov.get(exp_key, {})
        print(
            f"  {exp_key}: {ed.get('records','?')} records, "
            f"global parse_success={ed.get('parse_success_rate','?'):.1%}, "
            f"stimuli={ed.get('unique_stimuli','?')}"
        )
        for model, md in ed.get("by_model", {}).items():
            psr = md.get("parse_success_rate")
            psr_str = f"{psr:.1%}" if psr is not None and not np.isnan(psr) else "—"
            print(f"    [{model}] {md.get('records','?')} records, parse_success={psr_str}")

    print(f"  Degradation dims : {ov.get('dimensions', [])}")
    print(f"  Edit types       : {ov.get('edit_types', [])}")

    # ------------------------------------------------------------------ #
    # Failure-mode table                                                  #
    # ------------------------------------------------------------------ #
    fmt = stats.get("failure_mode_table", [])
    if fmt:
        print(f"\n{sep}")
        print("FAILURE-MODE TABLE (per model)")
        print(f"{sep}")
        hdr = (f"  {'Model':<28} {'PSR-E1':>7} {'PSR-E2':>7} "
               f"{'FPR':>7} {'Noop>1':>7} {'Noop>=3':>8} {'Gap':>8}")
        print(hdr)
        print("  " + "-" * (len(hdr) - 2))
        for row in fmt:
            def _fmt(v, pct: bool = True):
                if v is None or (isinstance(v, float) and np.isnan(v)):
                    return "    —"
                return f"{v:.1%}" if pct else f"{v:+.3f}"

            print(
                f"  {row['model']:<28} "
                f"{_fmt(row.get('parse_success_experiment_1')):>7} "
                f"{_fmt(row.get('parse_success_experiment_2')):>7} "
                f"{_fmt(row.get('exp1_false_positive_rate')):>7} "
                f"{_fmt(row.get('noop_above_1_rate')):>7} "
                f"{_fmt(row.get('noop_high_rating_rate')):>7} "
                f"{_fmt(row.get('score_gap_exp2_minus_exp1'), pct=False):>8}"
            )

    # ------------------------------------------------------------------ #
    # Exp1 — per model                                                    #
    # ------------------------------------------------------------------ #
    exp1_bm = stats.get("exp1", {}).get("by_model", {})
    if exp1_bm:
        print(f"\n{sep}")
        print("EXP 1 — PERCEPTUAL SENSITIVITY (binary detection + similarity)")
        print(f"{sep}")

        for model, e1 in sorted(exp1_bm.items()):
            if "error" in e1:
                print(f"\n  [{model}] ERROR: {e1['error']}")
                continue

            print(f"\n  ── Model: {model} ──")
            print(f"  Overall detection rate     : {e1.get('overall_detection_rate', 0):.1%}")
            print(f"  Mean similarity score      : {e1.get('similarity_score_mean', 0):.3f} ± "
                  f"{e1.get('similarity_score_std', 0):.3f}")

            perf1 = e1.get("perfect_edits")
            if perf1:
                fp = perf1.get("false_positive_rate", float("nan"))
                print(f"  Exp1 FPR (perfect edits)   : {fp:.1%}  (n={perf1['n']})")

            print(f"\n  {'Dimension':<22} {'Det%':>7} {'FNR':>6} {'Sim μ':>7} "
                  f"{'Det r':>8} {'Sim ρ':>8} {'Tier50':>10} {'Mag50':>9} {'Tier75':>10} {'Mag75':>9}")
            print(f"  {'-'*22} {'-'*7} {'-'*6} {'-'*7} {'-'*8} {'-'*8} {'-'*10} {'-'*9} {'-'*10} {'-'*9}")

            for dim, d in sorted(e1.get("by_dimension", {}).items()):
                r_det  = d.get("magnitude_detection_r")
                r_sim  = d.get("magnitude_similarity_rho")
                t50    = d.get("threshold_tier_50pct", "—")
                v50    = d.get("threshold_numeric_50pct")
                t75    = d.get("threshold_tier_75pct", "—")
                v75    = d.get("threshold_numeric_75pct")
                r_det_s = f"{r_det:+.3f}" if r_det is not None and not np.isnan(r_det) else "    —"
                r_sim_s = f"{r_sim:+.3f}" if r_sim is not None and not np.isnan(r_sim) else "    —"
                v50_s   = f"{v50:.3f}" if v50 is not None else "        —"
                v75_s   = f"{v75:.3f}" if v75 is not None else "        —"
                print(
                    f"  {dim:<22} {d.get('detection_rate', 0):>7.1%} "
                    f"{d.get('false_negative_rate', 0):>6.1%} "
                    f"{d.get('similarity_score_mean', 0):>7.3f} "
                    f"{r_det_s:>8} {r_sim_s:>8} "
                    f"{str(t50)[:10]:>10} {v50_s:>9} "
                    f"{str(t75)[:10]:>10} {v75_s:>9}"
                )

    # ------------------------------------------------------------------ #
    # Exp2 — per model                                                    #
    # ------------------------------------------------------------------ #
    exp2_bm = stats.get("exp2", {}).get("by_model", {})
    if exp2_bm:
        print(f"\n{sep}")
        print("EXP 2 — INSTRUCTION-FOLLOWING (multi-dimensional scores 1–5)")
        print(f"{sep}")

        for model, e2 in sorted(exp2_bm.items()):
            if "error" in e2:
                print(f"\n  [{model}] ERROR: {e2['error']}")
                continue

            print(f"\n  ── Model: {model} ──")
            print(f"  Cronbach's alpha: {e2.get('cronbach_alpha', float('nan')):.4f}")

            # Spearman correlation matrix among rubric dimensions
            corr_mat = e2.get("score_intercorrelation_spearman", {})
            if corr_mat:
                cols_avail = [c for c in EXP2_SCORE_COLS if c in corr_mat]
                short = {c: c[:8] for c in cols_avail}
                header = "  " + " " * 24 + "  ".join(f"{short[c]:>8}" for c in cols_avail)
                print("\n  Rubric Spearman correlation matrix:")
                print(header)
                for row_col in cols_avail:
                    row_vals = corr_mat.get(row_col, {})
                    vals_str = "  ".join(
                        f"{row_vals.get(c, float('nan')):>8.4f}" for c in cols_avail
                    )
                    print(f"  {row_col:<24}{vals_str}")

            sd = e2.get("score_descriptives", {})
            print(f"\n  {'Score col':<24} {'μ':>6} {'σ':>6} {'Med':>6} {'Ceil%':>8} {'Floor%':>7}")
            print(f"  {'-'*24} {'-'*6} {'-'*6} {'-'*6} {'-'*8} {'-'*7}")
            for col in EXP2_SCORE_COLS:
                s = sd.get(col, {})
                print(
                    f"  {col:<24} {s.get('mean', 0):>6.3f} {s.get('std', 0):>6.3f} "
                    f"{s.get('median', 0):>6.1f} "
                    f"{s.get('ceiling_rate', 0):>8.1%} "
                    f"{s.get('floor_rate', 0):>7.1%}"
                )

            print(f"\n  {'Dimension':<22} {'OQ μ':>7} {'σ':>6} {'Ceil%':>7} {'Mag ρ':>8} {'p_corr':>8}")
            print(f"  {'-'*22} {'-'*7} {'-'*6} {'-'*7} {'-'*8} {'-'*8}")
            for dim, d in sorted(e2.get("by_dimension", {}).items()):
                oq  = d.get("overall_quality", {})
                rho = oq.get("magnitude_rho")
                pc  = oq.get("magnitude_p_corrected") or oq.get("magnitude_p")
                rho_s = f"{rho:+.4f}" if rho is not None else "      —"
                p_s   = f"{pc:.4f}"   if pc  is not None else "      —"
                print(
                    f"  {dim:<22} {oq.get('mean', 0):>7.3f} {oq.get('std', 0):>6.3f} "
                    f"{oq.get('ceiling_rate', 0):>7.1%} {rho_s:>8} {p_s:>8}"
                )

    # ------------------------------------------------------------------ #
    # Sensitivity ranking + blind spots                                   #
    # ------------------------------------------------------------------ #
    print(f"\n{sep}")
    print("SENSITIVITY RANKING & BLIND SPOTS")
    print(f"{sep}")

    kappa = sr.get("inter_model_fleiss_kappa")
    kappa_str = f"{kappa:.4f}" if kappa is not None else "—"
    print(f"\n  Inter-model agreement (Fleiss' κ on blind/sensitive): {kappa_str}")

    ranked = sr.get("dims_by_mean_detection_rate_asc", [])
    print(f"\n  Dimensions ranked most-blind → most-sensitive (Exp1 detection rate):")
    for i, dim in enumerate(ranked, 1):
        det = sr.get("detail", {}).get(dim, {}).get("mean_detection_rate")
        det_s = f"{det:.1%}" if det is not None and not np.isnan(det) else "—"
        bs_by_m = sr.get("blind_sensitive_table", {}).get(dim, {})
        bs_strs = ", ".join(
            f"{m}: {v.get('classification', '?')}"
            for m, v in sorted(bs_by_m.items())
        )
        print(f"    {i:>2}. {dim:<24} mean_det={det_s}  [{bs_strs}]")

    ub = sr.get("universally_blind", [])
    if ub:
        print(f"\n  Universally blind (all models): {ub}")

    msb = sr.get("model_specific_blind", {})
    for m, dims_list in sorted(msb.items()):
        if dims_list:
            print(f"  Model-specific blind [{m}]: {dims_list}")

    # ------------------------------------------------------------------ #
    # Noop analysis                                                       #
    # ------------------------------------------------------------------ #
    noop_bm = stats.get("noop", {}).get("by_model", {})
    if noop_bm:
        print(f"\n{sep}")
        print("NOOP ANALYSIS — Exp2 (original image shown as 'edited')")
        print(f"{sep}")
        print(f"  Expected scores = 1 everywhere.  "
              f"above_1_rate: any score > 1; high_rating_rate: any score >= 3\n")

        for model, n_stats in sorted(noop_bm.items()):
            n2 = n_stats.get("exp2", {})
            if not n2:
                continue
            print(f"  ── Model: {model} (n={n2.get('n','?')}) ──")
            sd_n = n2.get("score_descriptives", {})
            print(f"  {'Score col':<24} {'mu':>6} {'sig':>5} {'>1 rate':>9} {'>=3 rate':>9}")
            print(f"  {'-'*24} {'-'*6} {'-'*5} {'-'*9} {'-'*9}")
            for col in EXP2_SCORE_COLS:
                s = sd_n.get(col, {})
                if not s:
                    continue
                print(
                    f"  {col:<24} {s.get('mean', 0):>6.3f} {s.get('std', 0):>6.3f} "
                    f"{s.get('above_1_rate', 0):>9.1%} "
                    f"{s.get('high_rating_rate', 0):>9.1%}"
                )
            print()

    print(f"\n{sep}")
