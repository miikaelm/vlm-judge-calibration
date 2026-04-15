"""
run_qwen.py — Run Qwen2.5-VL inference on a stimulus manifest on LUMI (AMD ROCm).

Reads source/ground_truth/degraded images + metadata from a manifest directory,
runs Qwen inference for each stimulus, and writes results to a JSONL file in the
same format produced by src/evaluation/runner.py (compatible with validate_results.py
and run_analysis.py).

Usage:
    # Single experiment
    python scripts/run_qwen.py \\
        --manifest data/pilot/ \\
        --model /scratch/project_xxx/models/Qwen2.5-VL-7B-Instruct \\
        --experiment 2 \\
        --output data/results_qwen_exp2.jsonl

    python scripts/run_qwen.py \\
        --manifest data/full/ \\
        --model Qwen/Qwen2.5-VL-7B-Instruct \\
        --experiment 1 \\
        --output data/results_qwen_exp1.jsonl \\
        --limit 10

    # Both experiments in one run (model loaded once):
    python scripts/run_qwen.py \\
        --manifest data/full/ \\
        --model Qwen/Qwen2.5-VL-7B-Instruct \\
        --experiment both \\
        --output data/results/
    # → writes data/results/exp1.jsonl and data/results/exp2.jsonl
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import yaml
from PIL import Image

# Allow running from repo root without `pip install -e .`
sys.path.insert(0, str(Path(__file__).parent.parent))
from src.evaluation.parser import log_parse_failure, parse_response


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _find_manifest(manifest_dir: Path) -> Path:
    """Return path to manifest.jsonl inside manifest_dir (falls back to stimuli.jsonl)."""
    for name in ("manifest.jsonl", "stimuli.jsonl"):
        p = manifest_dir / name
        if p.exists():
            return p
    raise FileNotFoundError(f"No manifest.jsonl found in {manifest_dir}")


def _load_manifest(manifest_dir: Path) -> list[dict]:
    manifest_path = _find_manifest(manifest_dir)
    records = []
    with open(manifest_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def _load_prompt(prompts_path: Path, variant: str) -> dict:
    with open(prompts_path, encoding="utf-8") as f:
        data = yaml.safe_load(f)
    try:
        return data["prompts"][variant]
    except KeyError:
        raise ValueError(f"Prompt variant '{variant}' not found in {prompts_path}")


def _load_separated_prompts(prompts_path: Path) -> dict[str, dict]:
    """Load per-dimension prompts for the separated evaluation strategy."""
    with open(prompts_path, encoding="utf-8") as f:
        data = yaml.safe_load(f)
    try:
        return data["prompts"]["experiment_2_separated"]
    except KeyError:
        raise ValueError(f"'experiment_2_separated' prompts not found in {prompts_path}")


# Ordered list of (result_field_name, short_label) for separated mode.
_SEPARATED_DIMENSIONS = [
    ("instruction_following", "IF"),
    ("text_accuracy",         "TA"),
    ("visual_consistency",    "VC"),
    ("layout_preservation",   "LP"),
    ("overall_quality",       "OQ"),
]


def _load_model(model_name_or_path: str, device: str, torch_dtype_str: str):
    import torch
    from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

    dtype_map = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}
    torch_dtype = dtype_map.get(torch_dtype_str, torch.bfloat16)

    print(f"[run_qwen] Loading model: {model_name_or_path}")
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        model_name_or_path,
        torch_dtype=torch_dtype,
        device_map=device,
    )
    model.eval()
    processor = AutoProcessor.from_pretrained(model_name_or_path)
    print(f"[run_qwen] Model ready on device_map={device!r}, dtype={torch_dtype_str}")
    return model, processor


def _run_inference(model, processor, messages: list[dict], max_new_tokens: int) -> tuple[str, int, int]:
    """
    Run Qwen2.5-VL inference.

    messages: list of Qwen-format message dicts where images are PIL Image objects
    in content parts of type "image".

    Returns (generated_text, prompt_token_count, completion_token_count).
    """
    import torch

    text_prompt = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )

    images = [
        part["image"]
        for msg in messages
        for part in (msg["content"] if isinstance(msg["content"], list) else [])
        if isinstance(part, dict) and part.get("type") == "image"
    ]

    inputs = processor(
        text=[text_prompt],
        images=images or None,
        padding=True,
        return_tensors="pt",
    ).to(model.device)

    with torch.no_grad():
        output_ids = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)

    input_len = inputs["input_ids"].shape[1]
    new_tokens = output_ids[0][input_len:]
    generated = processor.decode(new_tokens, skip_special_tokens=True).strip()

    return generated, int(input_len), int(new_tokens.shape[0])


def _build_messages(
    prompt: dict,
    instruction: str,
    img1: Image.Image,
    img2: Image.Image,
) -> list[dict]:
    """Build Qwen-format messages for a stimulus."""
    system_text = prompt["system"].strip()
    user_text = prompt["user_template"].replace("{instruction}", instruction)

    return [
        {"role": "system", "content": system_text},
        {
            "role": "user",
            "content": [
                {"type": "text", "text": user_text},
                {"type": "image", "image": img1},
                {"type": "image", "image": img2},
            ],
        },
    ]


def _run_separated_inference(
    model,
    processor,
    separated_prompts: dict[str, dict],
    instruction: str,
    img1: Image.Image,
    img2: Image.Image,
    max_new_tokens: int,
) -> tuple[dict, int, int, bool]:
    """Run one inference call per dimension and aggregate into a combined-format dict.

    Returns (aggregated_dict, total_prompt_tokens, total_completion_tokens, all_ok).
    The aggregated_dict has the same keys as a combined exp2 response, with the
    rationale from each dimension concatenated into 'errors_noticed'.
    """
    scores: dict = {}
    rationales: list[str] = []
    total_prompt_tokens = 0
    total_completion_tokens = 0
    all_ok = True

    for field_name, abbr in _SEPARATED_DIMENSIONS:
        dim_prompt = separated_prompts[field_name]
        messages = _build_messages(dim_prompt, instruction, img1, img2)

        try:
            raw, pt, ct = _run_inference(model, processor, messages, max_new_tokens)
        except Exception as e:
            print(f"\n    [{abbr}] INFERENCE ERROR: {e}", end="")
            scores[field_name] = None
            rationales.append(f"{abbr}: [inference error]")
            all_ok = False
            continue

        total_prompt_tokens += pt
        total_completion_tokens += ct

        parsed = parse_response(raw)
        if parsed is not None and "score" in parsed:
            scores[field_name] = parsed["score"]
            rat = str(parsed.get("rationale", "")).strip()
            if rat:
                rationales.append(f"{abbr}: {rat}")
        else:
            scores[field_name] = None
            rationales.append(f"{abbr}: [parse error]")
            all_ok = False

    aggregated = {
        "instruction_following": scores.get("instruction_following"),
        "text_accuracy":         scores.get("text_accuracy"),
        "visual_consistency":    scores.get("visual_consistency"),
        "layout_preservation":   scores.get("layout_preservation"),
        "overall_quality":       scores.get("overall_quality"),
        "errors_noticed":        " | ".join(rationales),
    }
    return aggregated, total_prompt_tokens, total_completion_tokens, all_ok


def _clamp_score(val) -> int | None:
    if isinstance(val, int) and 1 <= val <= 5:
        return val
    if isinstance(val, float) and val == int(val) and 1 <= val <= 5:
        return int(val)
    return None


def _build_result_record(
    stimulus_id: str,
    model_name: str,
    variant: str,
    parsed: dict | None,
    raw_response: str,
    prompt_tokens: int,
    completion_tokens: int,
) -> dict:
    """Build a result record matching the schema written by runner.py."""
    record: dict = {
        "stimulus_id": stimulus_id,
        "experiment": variant,
        "model": model_name,
        "parse_success": parsed is not None,
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
    }

    if variant == "experiment_2":
        record.update({
            "instruction_following": _clamp_score(parsed.get("instruction_following")) if parsed else None,
            "text_accuracy": _clamp_score(parsed.get("text_accuracy")) if parsed else None,
            "visual_consistency": _clamp_score(parsed.get("visual_consistency")) if parsed else None,
            "layout_preservation": _clamp_score(parsed.get("layout_preservation")) if parsed else None,
            "overall_quality": _clamp_score(parsed.get("overall_quality")) if parsed else None,
            "errors_noticed": parsed.get("errors_noticed", "") if parsed else "",
        })
    else:  # experiment_1
        detected: bool | None = None
        if parsed is not None:
            raw_det = parsed.get("detected_difference")
            if isinstance(raw_det, bool):
                detected = raw_det
        record.update({
            "detected_difference": detected,
            "similarity_score": _clamp_score(parsed.get("similarity_score")) if parsed else None,
            "description": parsed.get("description", "") if parsed else "",
        })

    return record


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def _process_stimuli(
    model,
    processor,
    manifest_dir: Path,
    variant: str,
    output_path: Path,
    parse_failures_path: Path,
    records: list[dict],
    model_name_or_path: str,
    max_new_tokens: int,
    prompt_strategy: str = "combined",
) -> None:
    """Run inference for all manifest records under one variant and append to output_path."""
    prompts_path = Path(__file__).parent.parent / "configs" / "vlm_prompts.yaml"

    use_separated = (prompt_strategy == "separated" and variant == "experiment_2")
    if use_separated:
        separated_prompts = _load_separated_prompts(prompts_path)
        prompt = None
    else:
        prompt = _load_prompt(prompts_path, variant)
        separated_prompts = None

    n = len(records)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.is_dir():
        raise IsADirectoryError(
            f"Output path is a directory, not a file: {output_path}\n"
            "This can happen if a previous '--experiment both' run was given a file path as --output.\n"
            f"Fix: rm -rf {output_path}"
        )
    n_ok = n_fail = 0

    for i, record in enumerate(records, 1):
        stimulus_id = record["id"]
        instruction = record.get("edit_instruction", "")

        print(f"[run_qwen] [{i}/{n}] {stimulus_id} ({variant}, {prompt_strategy}) ...", end=" ", flush=True)

        if variant == "experiment_2":
            img1 = Image.open(manifest_dir / record["source_image"]).convert("RGB")
            img2 = Image.open(manifest_dir / record["degraded_image"]).convert("RGB")
        else:  # experiment_1
            img1 = Image.open(manifest_dir / record["ground_truth_image"]).convert("RGB")
            img2 = Image.open(manifest_dir / record["degraded_image"]).convert("RGB")

        if use_separated:
            assert separated_prompts is not None
            aggregated, prompt_tokens, completion_tokens, all_ok = _run_separated_inference(
                model, processor, separated_prompts, instruction, img1, img2, max_new_tokens,
            )
            if all_ok:
                print("ok")
                n_ok += 1
            else:
                print("PARTIAL FAIL")
                n_fail += 1
            result_record = _build_result_record(
                stimulus_id, model_name_or_path, variant,
                aggregated, "", prompt_tokens, completion_tokens,
            )
            result_record["parse_success"] = all_ok
        else:
            assert prompt is not None
            messages = _build_messages(prompt, instruction, img1, img2)
            try:
                raw_response, prompt_tokens, completion_tokens = _run_inference(
                    model, processor, messages, max_new_tokens
                )
            except Exception as e:
                print(f"INFERENCE ERROR: {e}")
                n_fail += 1
                continue

            parsed = parse_response(raw_response)

            if parsed is not None:
                print("ok")
                n_ok += 1
            else:
                print("PARSE FAIL")
                n_fail += 1
                log_parse_failure(raw_response, stimulus_id, parse_failures_path)

            result_record = _build_result_record(
                stimulus_id, model_name_or_path, variant,
                parsed, raw_response, prompt_tokens, completion_tokens,
            )

        with open(output_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(result_record) + "\n")

    print(f"\n[run_qwen] {variant} ({prompt_strategy}): {n_ok} succeeded, {n_fail} failed.")
    print(f"[run_qwen] Results -> {output_path}")



def run(
    manifest_dir: Path,
    model_name_or_path: str,
    variants: list[str],
    output_paths: list[Path],
    parse_failures_paths: list[Path],
    limit: int | None,
    device: str,
    torch_dtype: str,
    max_new_tokens: int,
    prompt_strategy: str = "combined",
) -> None:
    records = _load_manifest(manifest_dir)
    if limit is not None:
        records = records[:limit]
    n = len(records)
    print(f"[run_qwen] {n} records | variants={variants} | model={model_name_or_path} | strategy={prompt_strategy}")

    model, processor = _load_model(model_name_or_path, device, torch_dtype)

    for variant, output_path, parse_failures_path in zip(variants, output_paths, parse_failures_paths):
        print(f"\n[run_qwen] === Running {variant} ({prompt_strategy}) ===")
        _process_stimuli(
            model, processor, manifest_dir, variant,
            output_path, parse_failures_path, records,
            model_name_or_path, max_new_tokens,
            prompt_strategy=prompt_strategy,
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="Qwen2.5-VL inference on stimulus manifest")
    parser.add_argument("--manifest", required=True, type=Path,
                        help="Directory with stimulus subdirs (each containing metadata.json)")
    parser.add_argument("--model", required=True,
                        help="HuggingFace model ID or local path, e.g. Qwen/Qwen2.5-VL-7B-Instruct")
    parser.add_argument(
        "--experiment", choices=["1", "2", "both"], default="2",
        help="1=perceptual sensitivity, 2=instruction-following, both=run sequentially (model loaded once)",
    )
    parser.add_argument("--output", type=Path, default=Path("data/results"),
                        help="Output directory. Writes exp1.jsonl / exp2.jsonl inside it.")
    parser.add_argument("--parse-failures", type=Path, default=Path("data/parse_failures"),
                        help="Directory for parse failure logs. Writes exp1.jsonl / exp2.jsonl inside it.")
    parser.add_argument("--limit", type=int, default=None, help="Process only the first N stimuli")
    parser.add_argument("--device", default="auto",
                        help="Device map for transformers (auto, cuda, cpu)")
    parser.add_argument("--dtype", default="bfloat16",
                        choices=["bfloat16", "float16", "float32"],
                        help="Model dtype (bfloat16 recommended for MI250X)")
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument(
        "--prompt-strategy", choices=["combined", "separated"], default="combined",
        help=(
            "combined (default): single query per stimulus for all dimensions. "
            "separated: one query per dimension (exp2 only); rationales are aggregated "
            "into the errors_noticed field so the output format is identical."
        ),
    )
    args = parser.parse_args()

    if args.experiment == "both":
        variants = ["experiment_1", "experiment_2"]
    else:
        variants = [f"experiment_{args.experiment}"]

    output_paths = [args.output / f"exp{v[-1]}.jsonl" for v in variants]
    parse_failures_paths = [args.parse_failures / f"exp{v[-1]}.jsonl" for v in variants]

    run(
        manifest_dir=args.manifest,
        model_name_or_path=args.model,
        variants=variants,
        output_paths=output_paths,
        parse_failures_paths=parse_failures_paths,
        limit=args.limit,
        device=args.device,
        torch_dtype=args.dtype,
        max_new_tokens=args.max_new_tokens,
        prompt_strategy=args.prompt_strategy,
    )


if __name__ == "__main__":
    main()
