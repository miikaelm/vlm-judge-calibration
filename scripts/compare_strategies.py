"""
Compare combined vs separated prompt strategy results for exp2.

combined  = data/results/first/exp2.jsonl  (single query per stimulus)
separated = data/results/exp2.jsonl         (one query per dimension)

Pairs are matched by stimulus_id; only records present in both files
with parse_success=True are included in the analysis.
"""

import json
import sys
from pathlib import Path
from collections import defaultdict

import numpy as np
import pandas as pd
from scipy import stats

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parent.parent
COMBINED_PATH  = ROOT / "data/results/first/exp2.jsonl"
SEPARATED_PATH = ROOT / "data/results/exp2.jsonl"

SCORE_COLS = [
    "instruction_following",
    "text_accuracy",
    "visual_consistency",
    "layout_preservation",
    "overall_quality",
]

# ---------------------------------------------------------------------------
# Load helpers
# ---------------------------------------------------------------------------

def load_jsonl(path: Path) -> dict[str, dict]:
    """Return {stimulus_id: record} keeping only parse_success=True entries."""
    records = {}
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            if rec.get("parse_success", False):
                sid = rec["stimulus_id"]
                if sid in records:
                    print(f"  WARNING: duplicate stimulus_id '{sid}' in {path.name}, keeping first")
                else:
                    records[sid] = rec
    return records


def build_paired_df(combined: dict, separated: dict) -> pd.DataFrame:
    """Inner-join on stimulus_id; return long-and-wide paired DataFrame."""
    common = sorted(set(combined) & set(separated))
    print(f"\nMatched pairs: {len(common)}")
    print(f"  Combined-only (excluded):  {len(combined) - len(common)}")
    print(f"  Separated-only (excluded): {len(separated) - len(common)}")

    rows = []
    for sid in common:
        for col in SCORE_COLS:
            rows.append({
                "stimulus_id": sid,
                "dimension": col,
                "combined":  combined[sid][col],
                "separated": separated[sid][col],
            })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Statistics
# ---------------------------------------------------------------------------

def describe_scores(df: pd.DataFrame) -> pd.DataFrame:
    """Per-dimension descriptive stats for both strategies."""
    rows = []
    for dim in SCORE_COLS:
        sub = df[df["dimension"] == dim]
        for strategy in ("combined", "separated"):
            s = sub[strategy]
            rows.append({
                "dimension": dim,
                "strategy":  strategy,
                "n":         len(s),
                "mean":      s.mean(),
                "std":       s.std(),
                "median":    s.median(),
                "min":       s.min(),
                "max":       s.max(),
            })
    return pd.DataFrame(rows)


def run_tests(df: pd.DataFrame) -> pd.DataFrame:
    """
    For each dimension run:
      - Wilcoxon signed-rank test (non-parametric paired test)
      - Paired t-test (parametric)
      - Mean difference (separated − combined) with 95 % CI
      - Cohen's d
    """
    rows = []
    for dim in SCORE_COLS:
        sub  = df[df["dimension"] == dim]
        c    = sub["combined"].values
        s    = sub["separated"].values
        diff = s - c

        # Wilcoxon (handles tied non-normal data)
        try:
            w_stat, w_p = stats.wilcoxon(diff, zero_method="wilcox", alternative="two-sided")
        except ValueError:
            w_stat, w_p = np.nan, np.nan  # all zeros

        # Paired t-test
        t_stat, t_p = stats.ttest_rel(s, c)

        # Mean diff + 95 % CI via t-distribution
        n       = len(diff)
        mean_d  = diff.mean()
        se_d    = diff.std(ddof=1) / np.sqrt(n)
        ci_lo, ci_hi = stats.t.interval(0.95, df=n - 1, loc=mean_d, scale=se_d)

        # Cohen's d (paired)
        cohens_d = mean_d / diff.std(ddof=1) if diff.std(ddof=1) > 0 else 0.0

        rows.append({
            "dimension":      dim,
            "n_pairs":        n,
            "mean_diff (sep-com)": round(mean_d, 4),
            "95%_CI_lo":      round(ci_lo, 4),
            "95%_CI_hi":      round(ci_hi, 4),
            "cohen_d":        round(cohens_d, 4),
            "wilcoxon_stat":  round(w_stat, 2) if not np.isnan(w_stat) else "n/a",
            "wilcoxon_p":     round(w_p, 5)    if not np.isnan(w_p)    else "n/a",
            "ttest_stat":     round(t_stat, 4),
            "ttest_p":        round(t_p, 5),
            "significant_05": (w_p < 0.05) if not np.isnan(w_p) else False,
        })
    return pd.DataFrame(rows)


def agreement_stats(df: pd.DataFrame) -> pd.DataFrame:
    """Exact agreement rate and mean absolute difference per dimension."""
    rows = []
    for dim in SCORE_COLS:
        sub = df[df["dimension"] == dim]
        exact = (sub["combined"] == sub["separated"]).mean()
        mad   = (sub["combined"] - sub["separated"]).abs().mean()
        rows.append({
            "dimension":       dim,
            "exact_agreement": round(exact, 4),
            "mean_abs_diff":   round(mad, 4),
        })
    return pd.DataFrame(rows)


def score_distribution(df: pd.DataFrame) -> pd.DataFrame:
    """Value-counts for each strategy × dimension."""
    rows = []
    for dim in SCORE_COLS:
        sub = df[df["dimension"] == dim]
        for strategy in ("combined", "separated"):
            vc = sub[strategy].value_counts().sort_index()
            for score, cnt in vc.items():
                rows.append({
                    "dimension": dim,
                    "strategy":  strategy,
                    "score":     score,
                    "count":     cnt,
                    "pct":       round(cnt / len(sub) * 100, 1),
                })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("Loading files …")
    combined  = load_jsonl(COMBINED_PATH)
    separated = load_jsonl(SEPARATED_PATH)
    print(f"  combined  records loaded: {len(combined)}")
    print(f"  separated records loaded: {len(separated)}")

    df = build_paired_df(combined, separated)

    # --- Descriptive stats ---
    desc = describe_scores(df)
    print("\n" + "=" * 70)
    print("DESCRIPTIVE STATISTICS")
    print("=" * 70)
    print(desc.to_string(index=False))

    # --- Agreement ---
    agree = agreement_stats(df)
    print("\n" + "=" * 70)
    print("AGREEMENT (combined vs separated)")
    print("=" * 70)
    print(agree.to_string(index=False))

    # --- Hypothesis tests ---
    tests = run_tests(df)
    print("\n" + "=" * 70)
    print("HYPOTHESIS TESTS  (separated - combined)")
    print("Wilcoxon signed-rank + paired t-test, two-sided, alpha=0.05")
    print("=" * 70)
    print(tests.to_string(index=False))

    # --- Score distributions ---
    dist = score_distribution(df)
    print("\n" + "=" * 70)
    print("SCORE DISTRIBUTIONS")
    print("=" * 70)
    for dim in SCORE_COLS:
        sub = dist[dist["dimension"] == dim]
        print(f"\n  {dim}")
        pivot = sub.pivot_table(index="score", columns="strategy", values="pct", fill_value=0)
        print(pivot.to_string())

    # --- Summary of significant differences ---
    sig = tests[tests["significant_05"] == True]
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    if sig.empty:
        print("No dimension shows a statistically significant difference (α=0.05).")
    else:
        print(f"{len(sig)} dimension(s) show significant differences (Wilcoxon p < 0.05):")
        for _, row in sig.iterrows():
            direction = "HIGHER" if row["mean_diff (sep-com)"] > 0 else "LOWER"
            print(
                f"  {row['dimension']:25s}  mean diff = {row['mean_diff (sep-com)']:+.3f}"
                f"  (separated {direction})  p = {row['wilcoxon_p']}"
            )

    # --- Save results ---
    out_dir = ROOT / "data/results/comparison"
    out_dir.mkdir(parents=True, exist_ok=True)
    tests.to_csv(out_dir / "strategy_tests.csv", index=False)
    desc.to_csv(out_dir / "strategy_descriptive.csv", index=False)
    agree.to_csv(out_dir / "strategy_agreement.csv", index=False)
    dist.to_csv(out_dir / "strategy_distributions.csv", index=False)
    print(f"\nResults saved to {out_dir}/")


if __name__ == "__main__":
    main()
