#!/usr/bin/env python3
"""
validate_results.py — Validate results.jsonl and print per-dimension statistics.

Checks:
  - Every stimulus directory in a manifest has both Exp1 and Exp2 entries
  - Parse failure rate
  - Score statistics (mean, std, min, max) per degradation dimension

Usage:
    python scripts/validate_results.py data/results.jsonl [--manifest data/full/]
"""

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path


def load_results(results_path: Path) -> list[dict]:
    records = []
    with open(results_path, encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as e:
                print(f"  WARNING: line {line_no} is malformed JSON: {e}", file=sys.stderr)
    return records


def load_stimulus_metadata(manifest_dir: Path) -> dict[str, dict]:
    """Load manifest.jsonl and return {record_id: record} for all entries."""
    for name in ("manifest.jsonl", "stimuli.jsonl"):
        p = manifest_dir / name
        if p.exists():
            meta = {}
            with open(p, encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        rec = json.loads(line)
                        meta[rec["id"]] = rec
            return meta
    return {}


def validate(results_path: Path, manifest_dir: Path | None) -> int:
    """Run validation. Returns exit code (0=ok, 1=errors)."""
    if not results_path.exists():
        print(f"ERROR: results file not found: {results_path}", file=sys.stderr)
        return 1

    records = load_results(results_path)
    print(f"Loaded {len(records)} result records from {results_path}")

    # -------------------------------------------------------------------------
    # 1. Parse failure rate
    # -------------------------------------------------------------------------
    n_total = len(records)
    n_parse_fail = sum(1 for r in records if not r.get("parse_success", True))
    pct_fail = 100 * n_parse_fail / n_total if n_total else 0
    print(f"\nParse failures: {n_parse_fail}/{n_total} ({pct_fail:.1f}%)")

    # -------------------------------------------------------------------------
    # 2. Coverage: every stimulus has both Exp1 and Exp2 entries
    # -------------------------------------------------------------------------
    by_stimulus: dict[str, set[str]] = defaultdict(set)
    for r in records:
        sid = r.get("stimulus_id", "")
        exp = r.get("experiment", "")
        by_stimulus[sid].add(exp)

    missing_exp1 = [sid for sid, exps in by_stimulus.items() if "experiment_1" not in exps]
    missing_exp2 = [sid for sid, exps in by_stimulus.items() if "experiment_2" not in exps]

    print(f"\nStimuli with both experiments: "
          f"{sum(1 for exps in by_stimulus.values() if len(exps) == 2)}/{len(by_stimulus)}")
    if missing_exp1:
        print(f"  Missing Exp1 ({len(missing_exp1)}): {missing_exp1[:5]}{'...' if len(missing_exp1) > 5 else ''}")
    if missing_exp2:
        print(f"  Missing Exp2 ({len(missing_exp2)}): {missing_exp2[:5]}{'...' if len(missing_exp2) > 5 else ''}")

    # -------------------------------------------------------------------------
    # 3. Dimension coverage check against manifest (if provided)
    # -------------------------------------------------------------------------
    if manifest_dir and manifest_dir.exists():
        metadata = load_stimulus_metadata(manifest_dir)
        n_manifest = len(metadata)
        n_in_results = sum(1 for sid in metadata if sid in by_stimulus)
        print(f"\nManifest coverage: {n_in_results}/{n_manifest} stimuli have results")
        missing_from_results = [sid for sid in metadata if sid not in by_stimulus]
        if missing_from_results:
            print(f"  Missing from results ({len(missing_from_results)}): "
                  f"{missing_from_results[:5]}{'...' if len(missing_from_results) > 5 else ''}")

    # -------------------------------------------------------------------------
    # 4. Per-dimension score statistics (Exp1 and Exp2 separately)
    # -------------------------------------------------------------------------
    # Join results with metadata for dimension info
    # First, build a dimension lookup from manifest or infer from stimulus_id
    dim_lookup: dict[str, str] = {}
    if manifest_dir and manifest_dir.exists():
        metadata = load_stimulus_metadata(manifest_dir)
        for sid, m in metadata.items():
            dim_lookup[sid] = m.get("degradation", {}).get("dimension", "unknown")

    print("\n" + "=" * 70)
    print("PER-DIMENSION STATISTICS")
    print("=" * 70)

    # Group results by experiment × dimension
    exp1_by_dim: dict[str, list[int]] = defaultdict(list)
    exp2_by_dim: dict[str, list[float]] = defaultdict(list)

    for r in records:
        if not r.get("parse_success", True):
            continue
        sid = r.get("stimulus_id", "")
        dim = dim_lookup.get(sid, _infer_dimension(sid))
        exp = r.get("experiment", "")

        if exp == "experiment_1":
            score = r.get("similarity_score")
            if score is not None:
                exp1_by_dim[dim].append(score)
        elif exp == "experiment_2":
            overall = r.get("overall_quality")
            if overall is not None:
                exp2_by_dim[dim].append(overall)

    all_dims = sorted(set(list(exp1_by_dim) + list(exp2_by_dim)))

    if not all_dims:
        print("  (no scored results to show)")
        return 0

    col_w = max(len(d) for d in all_dims) + 2
    header = f"{'Dimension':<{col_w}}  {'Exp1 similarity':>20}  {'Exp2 overall_quality':>22}"
    print(header)
    print("-" * len(header))

    for dim in all_dims:
        exp1_scores = exp1_by_dim.get(dim, [])
        exp2_scores = exp2_by_dim.get(dim, [])
        exp1_str = _stats_str(exp1_scores)
        exp2_str = _stats_str(exp2_scores)
        print(f"{dim:<{col_w}}  {exp1_str:>20}  {exp2_str:>22}")

    # -------------------------------------------------------------------------
    # 5. Summary
    # -------------------------------------------------------------------------
    errors = len(missing_exp1) + len(missing_exp2) + n_parse_fail
    print("\n" + "=" * 70)
    print(f"SUMMARY: {n_total} records | {n_parse_fail} parse failures | "
          f"{len(missing_exp1)} missing Exp1 | {len(missing_exp2)} missing Exp2")
    if errors:
        print("STATUS: ISSUES FOUND (see above)")
        return 1
    print("STATUS: OK")
    return 0


def _stats_str(scores: list) -> str:
    if not scores:
        return "—"
    n = len(scores)
    mean = sum(scores) / n
    if n > 1:
        variance = sum((x - mean) ** 2 for x in scores) / (n - 1)
        std = variance ** 0.5
    else:
        std = 0.0
    mn, mx = min(scores), max(scores)
    return f"n={n} m={mean:.2f} s={std:.2f} [{mn},{mx}]"


def _infer_dimension(stimulus_id: str) -> str:
    """Best-effort dimension inference from stimulus_id when no manifest is available."""
    known_dims = [
        "color_offset", "position_offset", "scale_error", "char_substitution",
        "font_weight", "font_style", "letter_spacing", "opacity", "rotation",
        "word_error", "case_error", "content_swap",
        "gaussian_noise", "jpeg_compression", "blur",
    ]
    for dim in known_dims:
        if dim.replace("_", "") in stimulus_id.replace("_", ""):
            return dim
    return "unknown"


def main():
    parser = argparse.ArgumentParser(description="Validate VLM evaluation results")
    parser.add_argument("results", type=Path, help="Path to results.jsonl")
    parser.add_argument(
        "--manifest", type=Path, default=None,
        help="Stimulus manifest directory (e.g. data/full/) for coverage check",
    )
    args = parser.parse_args()

    sys.exit(validate(args.results, args.manifest))


if __name__ == "__main__":
    main()
