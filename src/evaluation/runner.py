"""
runner.py — Evaluation runner: iterate a stimulus manifest, call the VLM, write results.jsonl.

Results are appended immediately after each call (crash-safe).

Standard (synchronous) usage:
    # Print planned calls, estimate cost for all models, no API calls:
    python src/evaluation/runner.py --manifest data/generated/ --model gpt-4o --dry-run

    # Run with real GPT-4o (requires OPENAI_API_KEY env var):
    python src/evaluation/runner.py --manifest data/generated/ --model gpt-4o

    # Run Experiment 1 (perceptual sensitivity) with fake API:
    python src/evaluation/runner.py --manifest data/generated/ --model dummy --experiment 1

    # Run Experiment 2 (instruction-following) with fake API:
    python src/evaluation/runner.py --manifest data/generated/ --model dummy --experiment 2

    # Run only first 5 stimuli:
    python src/evaluation/runner.py --manifest data/generated/ --model dummy --limit 5

Gemini Batch API (async, 50% cheaper, bypasses 250 req/day limit):
    Requires: pip install google-genai

    # Step 1 — build batch JSONL from manifest (test with --limit 3)
    python src/evaluation/runner.py batch prepare \\
        --manifest data/generated/ --model gemini-2.0-flash \\
        --experiment 1 --limit 3 --output data/batch/input.jsonl

    # Step 2 — upload and submit
    python src/evaluation/runner.py batch submit \\
        --input data/batch/input.jsonl --model gemini-2.0-flash

    # Step 3 — check status (repeat until SUCCEEDED)
    python src/evaluation/runner.py batch status --job batches/batch-abc123

    # Step 4 — list all jobs
    python src/evaluation/runner.py batch list

    # Step 5 — fetch and parse results
    python src/evaluation/runner.py batch fetch \\
        --job batches/batch-abc123 \\
        --input data/batch/input.jsonl \\
        --experiment 1 --model gemini-2.0-flash \\
        --output data/results_batch_exp1.jsonl
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
from pathlib import Path
from typing import Any

from api_tracker import DummyOpenAI, TrackedGemini, TrackedOpenAI, PRICING
from evaluation.judge import JudgeConfig, JudgeResult, run_judge
from evaluation.parser import log_parse_failure


DEFAULT_RESULTS_PATH = Path("data/results.jsonl")
DEFAULT_PARSE_FAILURES_PATH = Path("data/parse_failures.jsonl")

# Token estimates per stimulus call (for dry-run cost projection)
# GPT-4o high detail: 512×512 is scaled UP so shortest side = 768px → 768×768
# → 2×2 tile grid = 4 tiles × 170 + 85 base = 765 tokens/image × 2 images
# Gemini 3.1 Pro: images billed at ~258 tokens each (standard resolution) × 2
_EST_IMAGE_TOKENS_GPT = 765 * 2
_EST_IMAGE_TOKENS_GEMINI = 258 * 2
_EST_TEXT_TOKENS = 300   # system + user prompt text
_EST_OUTPUT_TOKENS = 120  # JSON response

# USD→EUR exchange rate (update as needed)
_USD_TO_EUR = 0.92

# Models to compare in dry-run cost summary
_DRY_RUN_COMPARE_MODELS = ["gpt-4o", "gemini-3.1-pro"]


def _load_manifest(manifest_dir: Path) -> list[dict]:
    """Read manifest.jsonl from manifest_dir and return list of entry dicts."""
    manifest_path = manifest_dir / "manifest.jsonl"
    entries = []
    with open(manifest_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                entries.append(json.loads(line))
    return entries


def _est_image_tokens(model: str) -> int:
    return _EST_IMAGE_TOKENS_GEMINI if model.startswith("gemini") else _EST_IMAGE_TOKENS_GPT


def _estimate_cost(model: str, n: int) -> float:
    prices = PRICING.get(model)
    if not prices:
        return 0.0
    input_tokens = (_est_image_tokens(model) + _EST_TEXT_TOKENS) * n
    output_tokens = _EST_OUTPUT_TOKENS * n
    return (input_tokens * prices["input"] + output_tokens * prices["output"]) / 1_000_000


def _result_to_dict(result: JudgeResult, prompt_variant: str) -> dict:
    d = {
        "stimulus_id": result.stimulus_id,
        "experiment": prompt_variant,
        "model": result.model,
        "parse_success": result.parse_success,
        "prompt_tokens": result.prompt_tokens,
        "completion_tokens": result.completion_tokens,
    }
    if prompt_variant == "experiment_2":
        d.update({
            "instruction_following": result.instruction_following,
            "text_accuracy": result.text_accuracy,
            "visual_consistency": result.visual_consistency,
            "layout_preservation": result.layout_preservation,
            "overall_quality": result.overall_quality,
            "errors_noticed": result.errors_noticed,
        })
    elif prompt_variant == "experiment_1":
        d.update({
            "detected_difference": result.detected_difference,
            "similarity_score": result.visual_consistency,
            "description": result.errors_noticed,
        })
    return d


def _make_client(model: str, prompt_variant: str) -> Any:
    if model == "dummy":
        return DummyOpenAI(prompt_variant=prompt_variant)
    if model.startswith("gemini"):
        api_key = os.environ.get("GEMINI_API_KEY")
        if not api_key:
            raise RuntimeError(
                "GEMINI_API_KEY environment variable not set. "
                "Use --model dummy for testing without an API key."
            )
        return TrackedGemini(api_key=api_key, note=f"eval_{prompt_variant}")
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError(
            "OPENAI_API_KEY environment variable not set. "
            "Use --model dummy for testing without an API key."
        )
    return TrackedOpenAI(api_key=api_key, note=f"eval_{prompt_variant}")


def _load_stimulus_ids(jsonl_path: Path) -> set[str]:
    """Return the set of stimulus_ids found in a results JSONL file."""
    ids: set[str] = set()
    with open(jsonl_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                ids.add(json.loads(line)["stimulus_id"])
    return ids


def _resolve_entries(
    manifest_dir: Path,
    limit: int | None,
    filter_ids: Path | None,
    presampled: list[dict] | None,
) -> list[dict]:
    """Load and filter/sample manifest entries. If *presampled* is given, use it directly."""
    if presampled is not None:
        return presampled
    entries = _load_manifest(manifest_dir)
    if filter_ids is not None:
        allowed = _load_stimulus_ids(filter_ids)
        before = len(entries)
        seen: set[str] = set()
        filtered: list[dict] = []
        for e in entries:
            if e["id"] in allowed and e["id"] not in seen:
                seen.add(e["id"])
                filtered.append(e)
        entries = filtered
        print(f"[runner] --filter-ids: kept {len(entries)}/{before} stimuli from {filter_ids}")
    elif limit is not None:
        entries = random.sample(entries, min(limit, len(entries)))
    return entries


def run_evaluation(
    manifest_dir: Path,
    model: str,
    prompt_variant: str = "experiment_2",
    dry_run: bool = False,
    output_path: Path = DEFAULT_RESULTS_PATH,
    parse_failures_path: Path = DEFAULT_PARSE_FAILURES_PATH,
    limit: int | None = None,
    filter_ids: Path | None = None,
    presampled_entries: list[dict] | None = None,
) -> None:
    manifest_dir = Path(manifest_dir)
    entries = _resolve_entries(manifest_dir, limit, filter_ids, presampled_entries)

    n = len(entries)
    print(f"[runner] Found {n} stimuli in {manifest_dir / 'manifest.jsonl'}")

    if dry_run:
        print(
            f"[runner] DRY RUN — {n} planned calls | "
            f"prompt={prompt_variant}"
        )
        print()
        # Cost summary for comparison models
        for m in _DRY_RUN_COMPARE_MODELS:
            usd = _estimate_cost(m, n)
            eur = usd * _USD_TO_EUR
            img_tok = _est_image_tokens(m)
            input_tok = (img_tok + _EST_TEXT_TOKENS) * n
            output_tok = _EST_OUTPUT_TOKENS * n
            total_tok = input_tok + output_tok
            prices = PRICING.get(m, {})
            price_note = (
                f"${prices.get('input', '?')}/M in + ${prices.get('output', '?')}/M out"
                if prices else "no pricing data"
            )
            print(
                f"  {m:<20} {total_tok:>9,} tokens  "
                f"~ ${usd:.2f} USD  /  ~EUR {eur:.2f}    ({price_note})"
            )
        print()
        for i, entry in enumerate(entries, 1):
            print(
                f"  [{i:03d}] {entry['id']} | "
                f"edit={entry['edit_type']} | "
                f"deg={entry['degradation']['dimension']}:{entry['degradation']['magnitude']}"
            )
        return

    client = _make_client(model, prompt_variant)
    config = JudgeConfig(model=model, prompt_variant=prompt_variant)

    output_path.parent.mkdir(parents=True, exist_ok=True)

    n_success = 0
    n_fail = 0

    for i, entry in enumerate(entries, 1):
        stimulus_id = entry["id"]
        print(f"[runner] [{i}/{n}] {stimulus_id} ...", end=" ", flush=True)

        try:
            result = run_judge(entry, manifest_dir, config, client)
        except Exception as e:
            print(f"ERROR: {e}")
            n_fail += 1
            continue

        if result.parse_success:
            if prompt_variant == "experiment_2":
                scores = (
                    f"{result.instruction_following}/"
                    f"{result.text_accuracy}/"
                    f"{result.visual_consistency}/"
                    f"{result.layout_preservation}/"
                    f"{result.overall_quality}"
                )
                print(f"ok (if/ta/vc/lp/oq: {scores})")
            else:
                print(
                    f"ok (detected={result.detected_difference}, "
                    f"similarity={result.visual_consistency})"
                )
            n_success += 1
        else:
            print("PARSE FAIL")
            n_fail += 1
            log_parse_failure(result.raw_response, stimulus_id, parse_failures_path)

        # Append result immediately — crash-safe
        with open(output_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(_result_to_dict(result, prompt_variant)) + "\n")

    print(f"\n[runner] Done. {n_success} succeeded, {n_fail} failed.")
    print(f"[runner] Results -> {output_path}")
    if n_fail:
        print(f"[runner] Parse failures -> {parse_failures_path}")


def _both_output_paths(base: Path) -> tuple[Path, Path]:
    """Derive exp1/exp2 output paths from a base path."""
    stem = base.stem
    suffix = base.suffix or ".jsonl"
    parent = base.parent
    return parent / f"{stem}_exp1{suffix}", parent / f"{stem}_exp2{suffix}"


def _batch_main(argv: list[str]) -> None:
    """Argument parser and dispatcher for 'batch' subcommands."""
    import os
    from evaluation.gemini_batch import (
        cmd_fetch,
        cmd_list,
        cmd_prepare,
        cmd_status,
        cmd_submit,
    )

    parser = argparse.ArgumentParser(
        prog="runner.py batch",
        description="Gemini Batch API commands (async, 50% cheaper than standard API)",
    )
    sub = parser.add_subparsers(dest="batch_cmd", required=True)

    # ---- prepare ----
    p_prep = sub.add_parser(
        "prepare",
        help="Build Gemini Batch API JSONL from manifest entries",
    )
    p_prep.add_argument("--manifest", required=True, type=Path,
                        help="Directory with manifest.jsonl and images/")
    p_prep.add_argument("--model", required=True,
                        help="Gemini model name (e.g. gemini-2.0-flash)")
    p_prep.add_argument("--experiment", required=True, choices=["1", "2"],
                        help="Experiment number")
    p_prep.add_argument("--output", required=True, type=Path,
                        help="Output path for batch JSONL (e.g. data/batch/input.jsonl)")
    p_prep.add_argument("--limit", type=int, default=None,
                        help="Randomly sample N stimuli (omit for all)")
    p_prep.add_argument("--filter-ids", type=Path, default=None,
                        help="JSONL file whose stimulus_ids define the subset to use")
    p_prep.add_argument("--temperature", type=float, default=0.0)
    p_prep.add_argument("--max-tokens", type=int, default=1024)

    # ---- submit ----
    p_sub = sub.add_parser(
        "submit",
        help="Upload batch JSONL to Gemini Files API and create a batch job",
    )
    p_sub.add_argument("--input", required=True, type=Path,
                       help="Batch JSONL built by 'batch prepare'")
    p_sub.add_argument("--model", required=True,
                       help="Gemini model name")
    p_sub.add_argument("--display-name", default=None,
                       help="Human-readable name for the batch job")

    # ---- status ----
    p_stat = sub.add_parser("status", help="Check batch job status")
    p_stat.add_argument("--job", required=True,
                        help="Batch job name (e.g. batches/batch-abc123)")

    # ---- list ----
    sub.add_parser("list", help="List all Gemini batch jobs")

    # ---- fetch ----
    p_fetch = sub.add_parser(
        "fetch",
        help="Download completed batch results and parse into results JSONL",
    )
    p_fetch.add_argument("--job", required=True,
                         help="Batch job name (e.g. batches/batch-abc123)")
    p_fetch.add_argument("--input", required=True, type=Path,
                         help="Original batch JSONL (used to find the sidecar manifest)")
    p_fetch.add_argument("--experiment", required=True, choices=["1", "2"],
                         help="Experiment number")
    p_fetch.add_argument("--model", required=True,
                         help="Gemini model name (for cost logging)")
    p_fetch.add_argument("--output", required=True, type=Path,
                         help="Results JSONL output path")
    p_fetch.add_argument("--parse-failures", type=Path,
                         default=DEFAULT_PARSE_FAILURES_PATH,
                         help="Parse failures log path")

    args = parser.parse_args(argv)

    api_key: str = os.environ.get("GEMINI_API_KEY", "")

    # Commands that need an API key
    if args.batch_cmd in ("submit", "status", "list", "fetch") and not api_key:
        print("ERROR: GEMINI_API_KEY environment variable not set.", file=sys.stderr)
        sys.exit(1)

    if args.batch_cmd == "prepare":
        prompt_variant = f"experiment_{args.experiment}"
        entries = _resolve_entries(args.manifest, args.limit, args.filter_ids, None)
        cmd_prepare(
            entries=entries,
            manifest_dir=args.manifest,
            prompt_variant=prompt_variant,
            output_path=args.output,
            temperature=args.temperature,
            max_tokens=args.max_tokens,
        )

    elif args.batch_cmd == "submit":
        cmd_submit(
            jsonl_path=args.input,
            model=args.model,
            api_key=api_key,
            display_name=args.display_name,
        )

    elif args.batch_cmd == "status":
        cmd_status(job_name=args.job, api_key=api_key)

    elif args.batch_cmd == "list":
        cmd_list(api_key=api_key)

    elif args.batch_cmd == "fetch":
        prompt_variant = f"experiment_{args.experiment}"
        cmd_fetch(
            job_name=args.job,
            jsonl_path=args.input,
            prompt_variant=prompt_variant,
            model=args.model,
            output_path=args.output,
            api_key=api_key,
            parse_failures_path=args.parse_failures,
        )


def main() -> None:
    # Dispatch to batch subcommand handler if first arg is 'batch'
    if len(sys.argv) > 1 and sys.argv[1] == "batch":
        _batch_main(sys.argv[2:])
        return

    parser = argparse.ArgumentParser(description="VLM evaluation runner")
    parser.add_argument(
        "--manifest", required=True, type=Path,
        help="Directory containing manifest.jsonl and images/ folder (e.g. data/generated/)",
    )
    parser.add_argument(
        "--model", default="gpt-4o",
        help="Model name (e.g. gpt-4o) or 'dummy' for fake API",
    )
    # --experiment 1/2/both is a convenience shorthand for --prompt-variant experiment_1/2
    exp_group = parser.add_mutually_exclusive_group()
    exp_group.add_argument(
        "--experiment", type=str, choices=["1", "2", "both"], default=None,
        help="Experiment to run: 1 (perceptual), 2 (instruction-following), or both",
    )
    exp_group.add_argument(
        "--prompt-variant", default=None,
        choices=["experiment_1", "experiment_2"],
        help="Which prompt/experiment to run (alternative to --experiment)",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Print planned calls with estimated cost; make no API calls",
    )
    parser.add_argument(
        "--output", default=DEFAULT_RESULTS_PATH, type=Path,
        help="Output JSONL path (default: data/results.jsonl). "
             "When --experiment both is used, _exp1/_exp2 suffixes are appended.",
    )
    parser.add_argument(
        "--limit", type=int, default=None,
        help="Process only N randomly-sampled stimuli. "
             "When --experiment both is used, the same sample is used for both.",
    )
    parser.add_argument(
        "--filter-ids", type=Path, default=None,
        help="Path to a results JSONL (e.g. exp1.jsonl); only stimuli whose "
             "stimulus_id appears in that file will be processed. "
             "Mutually exclusive with --limit.",
    )
    args = parser.parse_args()

    run_both = args.experiment == "both"

    if run_both:
        # Sample once so both experiments see identical stimuli
        manifest_dir = Path(args.manifest)
        presampled = _resolve_entries(manifest_dir, args.limit, args.filter_ids, None)
        print(f"[runner] Running both experiments on {len(presampled)} stimuli (shared sample).")

        out_exp1, out_exp2 = _both_output_paths(args.output)

        for variant, out_path in [("experiment_1", out_exp1), ("experiment_2", out_exp2)]:
            print(f"\n[runner] === {variant} -> {out_path} ===")
            run_evaluation(
                manifest_dir=manifest_dir,
                model=args.model,
                prompt_variant=variant,
                dry_run=args.dry_run,
                output_path=out_path,
                presampled_entries=presampled,
            )
        return

    # Single-experiment path
    if args.experiment is not None:
        prompt_variant = f"experiment_{args.experiment}"
    elif args.prompt_variant is not None:
        prompt_variant = args.prompt_variant
    else:
        prompt_variant = "experiment_2"

    run_evaluation(
        manifest_dir=args.manifest,
        model=args.model,
        prompt_variant=prompt_variant,
        dry_run=args.dry_run,
        output_path=args.output,
        limit=args.limit,
        filter_ids=args.filter_ids,
    )


if __name__ == "__main__":
    main()
