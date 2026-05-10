#!/usr/bin/env python3
"""
run_stats.py — Compute numerical statistics from VLM evaluation results.

Produces:
  - A JSON file with all computed statistics (outputs/stats/stats.json)
  - A human-readable report printed to stdout

Usage:
    # Both experiments (recommended)
    python scripts/run_stats.py --results-exp1 exp1.jsonl --results-exp2 exp2.jsonl

    # Single experiment
    python scripts/run_stats.py --results-exp2 exp2.jsonl

    # Custom manifest and output paths
    python scripts/run_stats.py \\
        --results-exp1 exp1.jsonl \\
        --results-exp2 exp2.jsonl \\
        --manifest data/generated/ \\
        --output outputs/stats/stats.json
"""

from __future__ import annotations

import argparse
import io
import sys
from pathlib import Path

# Ensure UTF-8 output on Windows
if hasattr(sys.stdout, "buffer"):
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

_repo_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_repo_root / "src"))

import pandas as pd
from analysis.curves import load_results, load_noop_results, load_perfect_results
from analysis.stats import compute_all_stats, print_report, save_report, save_print_report


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compute numerical statistics from VLM calibration results.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--results-exp1", type=Path, default=None, metavar="PATH",
                        help="Exp1 results JSONL (perceptual sensitivity).")
    parser.add_argument("--results-exp2", type=Path, default=None, metavar="PATH",
                        help="Exp2 results JSONL (instruction-following).")
    parser.add_argument("--manifest", type=Path, default=Path("data/full/"),
                        help="Stimulus manifest directory (default: data/full/)")
    parser.add_argument("--output", type=Path, default=Path("outputs/stats/stats.json"),
                        help="Output JSON path (default: outputs/stats/stats.json)")
    parser.add_argument("--model-name", default=None, metavar="NAME",
                        help="Display name to relabel the model column.")
    parser.add_argument("--output-report", type=Path,
                        default=Path("outputs/stats/stats_report.txt"),
                        help="Output path for thesis-focused text report "
                             "(default: outputs/stats/stats_report.txt)")
    parser.add_argument("--no-save", action="store_true",
                        help="Print report only, do not write JSON or text report.")
    args = parser.parse_args()

    if args.results_exp1 is None and args.results_exp2 is None:
        parser.error("supply at least one of --results-exp1 / --results-exp2")

    frames: list[pd.DataFrame] = []
    noop_frames: list[pd.DataFrame] = []
    perfect_frames: list[pd.DataFrame] = []
    manifest_dir = args.manifest.resolve()

    if args.results_exp1 is not None:
        p = args.results_exp1.resolve()
        print(f"Loading Exp1: {p}")
        frames.append(load_results(p, manifest_dir))
        noop_frames.append(load_noop_results(p, manifest_dir))
        perfect_frames.append(load_perfect_results(p, manifest_dir))

    if args.results_exp2 is not None:
        p = args.results_exp2.resolve()
        print(f"Loading Exp2: {p}")
        frames.append(load_results(p, manifest_dir))
        noop_frames.append(load_noop_results(p, manifest_dir))
        perfect_frames.append(load_perfect_results(p, manifest_dir))

    df         = pd.concat(frames,         ignore_index=True)
    noop_df    = pd.concat(noop_frames,    ignore_index=True) if noop_frames    else pd.DataFrame()
    perfect_df = pd.concat(perfect_frames, ignore_index=True) if perfect_frames else pd.DataFrame()

    print(f"  {len(df)} records loaded "
          f"({df['stimulus_id'].nunique()} stimuli, "
          f"{df['model'].nunique()} model(s): {sorted(df['model'].unique().tolist())})")
    if not noop_df.empty:
        print(f"  {len(noop_df)} noop records loaded")
    if not perfect_df.empty:
        print(f"  {len(perfect_df)} perfect-edit records loaded")
    print()

    if args.model_name:
        df["model"] = args.model_name
        if not noop_df.empty:
            noop_df["model"] = args.model_name
        if not perfect_df.empty:
            perfect_df["model"] = args.model_name

    stats = compute_all_stats(
        df,
        noop_df=noop_df         if not noop_df.empty    else None,
        perfect_df=perfect_df   if not perfect_df.empty else None,
    )
    print_report(stats)

    if not args.no_save:
        out_path = args.output.resolve()
        save_report(stats, out_path)
        print(f"\nJSON report saved to: {out_path}")

        rpt_path = args.output_report.resolve()
        save_print_report(stats, rpt_path)
        print(f"Thesis text report  saved to: {rpt_path}")


if __name__ == "__main__":
    main()
