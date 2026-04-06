"""
runner.py — Evaluation runner: iterate a stimulus manifest, call the VLM, write results.jsonl.

Results are appended immediately after each call (crash-safe).

Usage:
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
"""

from __future__ import annotations

import argparse
import json
import os
import random
from pathlib import Path
from typing import Any

from src.api_tracker import DummyOpenAI, TrackedGemini, TrackedOpenAI, PRICING
from src.evaluation.judge import JudgeConfig, JudgeResult, run_judge
from src.evaluation.parser import log_parse_failure


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


def run_evaluation(
    manifest_dir: Path,
    model: str,
    prompt_variant: str = "experiment_2",
    dry_run: bool = False,
    output_path: Path = DEFAULT_RESULTS_PATH,
    parse_failures_path: Path = DEFAULT_PARSE_FAILURES_PATH,
    limit: int | None = None,
) -> None:
    manifest_dir = Path(manifest_dir)
    entries = _load_manifest(manifest_dir)
    if limit is not None:
        entries = random.sample(entries, min(limit, len(entries)))

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


def main() -> None:
    parser = argparse.ArgumentParser(description="VLM evaluation runner")
    parser.add_argument(
        "--manifest", required=True, type=Path,
        help="Directory containing manifest.jsonl and images/ folder (e.g. data/generated/)",
    )
    parser.add_argument(
        "--model", default="gpt-4o",
        help="Model name (e.g. gpt-4o) or 'dummy' for fake API",
    )
    # --experiment 1/2 is a convenience shorthand for --prompt-variant experiment_1/2
    exp_group = parser.add_mutually_exclusive_group()
    exp_group.add_argument(
        "--experiment", type=int, choices=[1, 2], default=None,
        help="Experiment number (1=perceptual sensitivity, 2=instruction-following)",
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
        help="Output JSONL path (default: data/results.jsonl)",
    )
    parser.add_argument(
        "--limit", type=int, default=None,
        help="Process only the first N stimuli",
    )
    args = parser.parse_args()

    # Resolve experiment variant from either flag (default: experiment_2)
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
    )


if __name__ == "__main__":
    main()
