"""
runner.py — Evaluation runner: iterate a stimulus manifest, call the VLM, write results.jsonl.

Results are appended immediately after each call (crash-safe).

Usage:
    # Print planned calls, estimate cost, no API calls:
    python src/evaluation/runner.py --manifest data/pilot/ --model gpt-4o --dry-run

    # Run with real GPT-4o (requires OPENAI_API_KEY env var):
    python src/evaluation/runner.py --manifest data/pilot/ --model gpt-4o

    # Run Experiment 1 (perceptual sensitivity) with fake API:
    python src/evaluation/runner.py --manifest data/pilot/ --model dummy --experiment 1

    # Run Experiment 2 (instruction-following) with fake API:
    python src/evaluation/runner.py --manifest data/pilot/ --model dummy --experiment 2

    # Run only first 5 stimuli:
    python src/evaluation/runner.py --manifest data/pilot/ --model dummy --limit 5
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

from src.api_tracker import DummyOpenAI, TrackedOpenAI, PRICING
from src.evaluation.judge import JudgeConfig, JudgeResult, run_judge
from src.evaluation.parser import log_parse_failure


DEFAULT_RESULTS_PATH = Path("data/results.jsonl")
DEFAULT_PARSE_FAILURES_PATH = Path("data/parse_failures.jsonl")

# Rough token estimates per stimulus call (for dry-run cost projection)
# 512×512 PNG at "high" detail ≈ 765 tokens/image × 2 images
_EST_IMAGE_TOKENS = 765 * 2
_EST_TEXT_TOKENS = 300   # system + user prompt text
_EST_OUTPUT_TOKENS = 120  # JSON response


def _find_stimulus_dirs(manifest_dir: Path) -> list[Path]:
    """Return all subdirectories containing a metadata.json, sorted by name."""
    return sorted(
        d for d in manifest_dir.iterdir()
        if d.is_dir() and (d / "metadata.json").exists()
    )


def _estimate_cost(model: str, n: int) -> float:
    prices = PRICING.get(model)
    if not prices:
        return 0.0
    input_tokens = (_EST_IMAGE_TOKENS + _EST_TEXT_TOKENS) * n
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
    stimulus_dirs = _find_stimulus_dirs(manifest_dir)
    if limit is not None:
        stimulus_dirs = stimulus_dirs[:limit]

    n = len(stimulus_dirs)
    print(f"[runner] Found {n} stimuli in {manifest_dir}")

    if dry_run:
        est_cost = _estimate_cost(model, n)
        print(
            f"[runner] DRY RUN — {n} planned calls | "
            f"model={model} | prompt={prompt_variant} | "
            f"est. cost=${est_cost:.4f}"
        )
        for i, d in enumerate(stimulus_dirs, 1):
            meta = json.loads((d / "metadata.json").read_text(encoding="utf-8"))
            print(
                f"  [{i:03d}] {meta['stimulus_id']} | "
                f"edit={meta['edit_type']} | "
                f"deg={meta['degradation']['dimension']}:{meta['degradation']['magnitude']}"
            )
        return

    client = _make_client(model, prompt_variant)
    config = JudgeConfig(model=model, prompt_variant=prompt_variant)

    output_path.parent.mkdir(parents=True, exist_ok=True)

    n_success = 0
    n_fail = 0

    for i, stimulus_dir in enumerate(stimulus_dirs, 1):
        meta = json.loads((stimulus_dir / "metadata.json").read_text(encoding="utf-8"))
        stimulus_id = meta["stimulus_id"]
        print(f"[runner] [{i}/{n}] {stimulus_id} ...", end=" ", flush=True)

        try:
            result = run_judge(stimulus_dir, config, client)
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
        help="Directory containing stimulus subdirectories (each with metadata.json)",
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
