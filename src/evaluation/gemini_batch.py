"""
gemini_batch.py — Gemini Batch API for async VLM evaluation.

The standard Gemini API has a 250 requests/day free-tier quota.
The Batch API processes requests asynchronously (target: 24 h turnaround)
at 50% of the standard per-token price.

Requires: pip install google-genai

Workflow — run each step separately:

    # 1. Build the batch JSONL from manifest entries
    python src/evaluation/runner.py batch prepare \\
        --manifest data/generated/ --model gemini-2.0-flash \\
        --experiment 1 --limit 5 --output data/batch/input.jsonl

    # 2. Upload JSONL and submit to Gemini Batch API
    python src/evaluation/runner.py batch submit \\
        --input data/batch/input.jsonl --model gemini-2.0-flash

    # 3. Check status (repeat until SUCCEEDED)
    python src/evaluation/runner.py batch status --job batches/batch-abc123

    # 4. List all batch jobs
    python src/evaluation/runner.py batch list

    # 5. Fetch completed results and parse into results.jsonl
    python src/evaluation/runner.py batch fetch \\
        --job batches/batch-abc123 \\
        --input data/batch/input.jsonl \\
        --experiment 1 --model gemini-2.0-flash \\
        --output data/results_batch_exp1.jsonl

Notes:
  - 'prepare' writes a sidecar <output>.manifest.jsonl next to the batch JSONL.
    'fetch' reads this sidecar automatically when given --input.
  - Batch results are appended to --output (not overwritten), matching runner.py behaviour.
  - Token costs are logged to src/gemini_api_usage_log.csv at the 50% batch discount rate.
"""

from __future__ import annotations

import base64
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

from api_tracker import (
    DEFAULT_GEMINI_LOG_PATH,
    PRICING,
    _GEMINI_API_MODEL_NAMES,
    _append_log_row,
)
from evaluation.parser import log_parse_failure, parse_response

# Reverse map: API model name → friendly/pricing name
_GEMINI_API_MODEL_NAMES_REVERSE: dict[str, str] = {
    v: k for k, v in _GEMINI_API_MODEL_NAMES.items()
}


def _resolve_pricing_model(model: str) -> str:
    """Return the PRICING-table key for a model, handling API name aliases."""
    if model in PRICING:
        return model
    return _GEMINI_API_MODEL_NAMES_REVERSE.get(model, model)

_PROMPTS_PATH = Path(__file__).parent.parent.parent / "configs" / "vlm_prompts.yaml"

_BATCH_DONE_STATES = {
    "JOB_STATE_SUCCEEDED",
    "JOB_STATE_FAILED",
    "JOB_STATE_CANCELLED",
    "JOB_STATE_EXPIRED",
}

# Gemini Batch API is 50% cheaper than standard
_BATCH_DISCOUNT = 0.5

DEFAULT_BATCH_PARSE_FAILURES_PATH = Path("data/batch_parse_failures.jsonl")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _get_client(api_key: str) -> Any:
    try:
        from google import genai
    except ImportError:
        print(
            "ERROR: google-genai package required for Gemini Batch API.\n"
            "Install with: pip install google-genai",
            file=sys.stderr,
        )
        sys.exit(1)
    return genai.Client(api_key=api_key)


def _state_str(batch_job: Any) -> str:
    state = getattr(batch_job, "state", None)
    if state is None:
        return "UNKNOWN"
    return state.name if hasattr(state, "name") else str(state)


def _load_prompt(variant: str) -> dict:
    with open(_PROMPTS_PATH, encoding="utf-8") as f:
        data = yaml.safe_load(f)
    return data["prompts"][variant]


def _encode_image(image_path: Path) -> str:
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def _clamp_score(val: Any) -> int | None:
    if isinstance(val, int) and 1 <= val <= 5:
        return val
    if isinstance(val, float) and val == int(val) and 1 <= val <= 5:
        return int(val)
    return None


def _sidecar_path(jsonl_path: Path) -> Path:
    """Derive the sidecar manifest path from the batch JSONL path."""
    return jsonl_path.with_name(jsonl_path.stem + ".manifest.jsonl")


# ---------------------------------------------------------------------------
# Request / result building
# ---------------------------------------------------------------------------

def _build_gemini_request(
    entry: dict,
    manifest_dir: Path,
    prompt_variant: str,
    temperature: float,
    max_tokens: int,
) -> dict:
    """Return a Gemini-native GenerateContentRequest dict for one manifest entry."""
    prompt = _load_prompt(prompt_variant)
    system_text = prompt["system"].strip()

    if prompt_variant == "experiment_2":
        user_text = prompt["user_template"].replace("{instruction}", entry["edit_instruction"])
        img1_path = manifest_dir / entry["source_image"]
        img2_path = manifest_dir / entry["degraded_image"]
    elif prompt_variant == "experiment_1":
        user_text = prompt["user_template"]
        img1_path = manifest_dir / entry["ground_truth_image"]
        img2_path = manifest_dir / entry["degraded_image"]
    else:
        raise ValueError(f"Unknown prompt_variant: {prompt_variant!r}")

    img1_b64 = _encode_image(img1_path)
    img2_b64 = _encode_image(img2_path)

    return {
        "system_instruction": {
            "parts": [{"text": system_text}]
        },
        "contents": [
            {
                "role": "user",
                "parts": [
                    {"text": user_text},
                    {"inline_data": {"mime_type": "image/png", "data": img1_b64}},
                    {"inline_data": {"mime_type": "image/png", "data": img2_b64}},
                ],
            }
        ],
        "generation_config": {
            "temperature": temperature,
            "max_output_tokens": max_tokens,
        },
    }


def _build_result_dict(
    stimulus_id: str,
    prompt_variant: str,
    model: str,
    parsed: dict | None,
    parse_success: bool,
    prompt_tokens: int,
    completion_tokens: int,
) -> dict:
    """Build a result dict identical in schema to runner.py's _result_to_dict output."""
    d = {
        "stimulus_id": stimulus_id,
        "experiment": prompt_variant,
        "model": model,
        "parse_success": parse_success,
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
    }
    if prompt_variant == "experiment_2":
        d.update({
            "instruction_following": _clamp_score(parsed.get("instruction_following")) if parsed else None,
            "text_accuracy": _clamp_score(parsed.get("text_accuracy")) if parsed else None,
            "visual_consistency": _clamp_score(parsed.get("visual_consistency")) if parsed else None,
            "layout_preservation": _clamp_score(parsed.get("layout_preservation")) if parsed else None,
            "overall_quality": _clamp_score(parsed.get("overall_quality")) if parsed else None,
            "errors_noticed": parsed.get("errors_noticed", "") if parsed else "",
        })
    elif prompt_variant == "experiment_1":
        detected_difference = None
        if parsed is not None:
            raw_det = parsed.get("detected_difference")
            if isinstance(raw_det, bool):
                detected_difference = raw_det
        d.update({
            "detected_difference": detected_difference,
            "similarity_score": _clamp_score(parsed.get("similarity_score")) if parsed else None,
            "description": parsed.get("description", "") if parsed else "",
        })
    return d


# ---------------------------------------------------------------------------
# Commands
# ---------------------------------------------------------------------------

def cmd_prepare(
    entries: list[dict],
    manifest_dir: Path,
    prompt_variant: str,
    output_path: Path,
    temperature: float = 0.0,
    max_tokens: int = 1024,
) -> None:
    """
    Build the Gemini Batch API JSONL from manifest entries.

    Each line: {"key": "<stimulus_id>", "request": {...GenerateContentRequest...}}

    Also writes a sidecar <stem>.manifest.jsonl next to the output file so that
    'batch fetch' can reconstruct which entry each key belongs to.
    """
    manifest_dir = Path(manifest_dir)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    sidecar = _sidecar_path(output_path)

    print(f"[batch:prepare] {len(entries)} entries | experiment={prompt_variant}")
    print(f"[batch:prepare] Output -> {output_path}")
    print(f"[batch:prepare] Sidecar -> {sidecar}")
    print()

    n_ok = n_err = 0
    with open(output_path, "w", encoding="utf-8") as f_out, \
         open(sidecar, "w", encoding="utf-8") as f_side:
        for i, entry in enumerate(entries, 1):
            sid = entry["id"]
            print(f"  [{i:03d}/{len(entries)}] {sid} ...", end=" ", flush=True)
            try:
                req = _build_gemini_request(
                    entry, manifest_dir, prompt_variant, temperature, max_tokens
                )
                f_out.write(json.dumps({"key": sid, "request": req}) + "\n")
                f_side.write(json.dumps(entry) + "\n")
                n_ok += 1
                print("ok")
            except Exception as e:
                n_err += 1
                print(f"ERROR: {e}")

    size_mb = output_path.stat().st_size / 1_048_576
    print()
    print(f"[batch:prepare] {n_ok} ok, {n_err} errors | {size_mb:.1f} MB")
    print()
    print("Next step:")
    print(
        f"  python src/evaluation/runner.py batch submit"
        f" --input {output_path} --model <model>"
    )


def cmd_submit(
    jsonl_path: Path,
    model: str,
    api_key: str,
    display_name: str | None = None,
) -> str:
    """
    Upload the batch JSONL via Gemini Files API and submit a batch job.
    Saves the job name to a <stem>.job.txt sidecar for convenience.
    Returns the batch job name (e.g. 'batches/batch-abc123').
    """
    from google.genai import types as genai_types

    client = _get_client(api_key)
    api_model = _GEMINI_API_MODEL_NAMES.get(model, model)
    display_name = display_name or f"vlm-eval-{datetime.now(timezone.utc).strftime('%Y%m%d-%H%M%S')}"

    n_requests = sum(1 for ln in open(jsonl_path, encoding="utf-8") if ln.strip())
    size_mb = jsonl_path.stat().st_size / 1_048_576
    print(f"[batch:submit] Uploading {n_requests} requests ({size_mb:.1f} MB) -> Gemini Files API ...")

    uploaded = client.files.upload(
        file=str(jsonl_path),
        config=genai_types.UploadFileConfig(
            display_name=display_name,
            mime_type="jsonl",
        ),
    )
    print(f"[batch:submit] Uploaded: {uploaded.name}")

    print(f"[batch:submit] Creating batch job (model={api_model}) ...")
    batch_job = client.batches.create(
        model=api_model,
        src=uploaded.name,
        config={"display_name": display_name},
    )
    job_name = batch_job.name
    state = _state_str(batch_job)

    # Save job name next to the input file for easy copy-paste later
    job_file = jsonl_path.with_name(jsonl_path.stem + ".job.txt")
    job_file.write_text(job_name + "\n", encoding="utf-8")

    print(f"[batch:submit] Job created: {job_name}")
    print(f"[batch:submit] State:       {state}")
    print(f"[batch:submit] Saved job name -> {job_file}")
    print()
    print("Next steps:")
    print(f"  python src/evaluation/runner.py batch status --job {job_name}")
    print(
        f"  python src/evaluation/runner.py batch fetch"
        f" --job {job_name}"
        f" --input {jsonl_path}"
        f" --experiment <1|2> --model {model}"
        f" --output data/results_batch.jsonl"
    )
    return job_name


def cmd_status(job_name: str, api_key: str) -> str:
    """Print the current state of a batch job. Returns the state string."""
    client = _get_client(api_key)
    batch_job = client.batches.get(name=job_name)
    state = _state_str(batch_job)
    is_done = state in _BATCH_DONE_STATES

    print(f"[batch:status] {job_name}")
    print(f"  State: {state}")

    # Request counts (attribute names vary by SDK version)
    try:
        counts = batch_job.request_counts
        if counts is not None:
            parts = []
            for attr in ("total", "completed", "failed", "pending", "cancelled"):
                v = getattr(counts, attr, None)
                if v is not None:
                    parts.append(f"{attr}={v}")
            if parts:
                print(f"  Counts: {', '.join(parts)}")
    except AttributeError:
        pass

    if state == "JOB_STATE_SUCCEEDED":
        print("  Ready to fetch results.")
    elif is_done:
        print(f"  Job ended with state {state}. Cannot fetch results.")
    else:
        print("  Job still running — check again later (target: within 24 h).")

    return state


def cmd_list(api_key: str) -> None:
    """List all Gemini batch jobs for this API key."""
    client = _get_client(api_key)
    print("[batch:list]")
    found = 0
    for job in client.batches.list():
        found += 1
        state = _state_str(job)
        display = getattr(job, "display_name", "") or ""
        created = getattr(job, "create_time", "") or ""
        print(f"  {job.name:<45}  [{state:<25}]  {display}  {created}")
    if found == 0:
        print("  (no batch jobs found)")


def cmd_fetch(
    job_name: str,
    jsonl_path: Path,
    prompt_variant: str,
    model: str,
    output_path: Path,
    api_key: str,
    parse_failures_path: Path = DEFAULT_BATCH_PARSE_FAILURES_PATH,
) -> None:
    """
    Fetch a completed batch job's results, parse each response, and append
    to output_path in the same JSONL format used by runner.py.

    Token costs are logged to gemini_api_usage_log.csv at the 50% batch rate.
    """
    client = _get_client(api_key)
    batch_job = client.batches.get(name=job_name)
    state = _state_str(batch_job)

    if state != "JOB_STATE_SUCCEEDED":
        print(f"[batch:fetch] ERROR: job state is '{state}', expected JOB_STATE_SUCCEEDED.")
        if state in _BATCH_DONE_STATES:
            print("[batch:fetch] Job ended without success — no results to fetch.")
        else:
            print("[batch:fetch] Job not yet complete. Run 'batch status' and try again.")
        sys.exit(1)

    # Load sidecar manifest so we can validate keys
    sidecar = _sidecar_path(jsonl_path)
    entries_by_key: dict[str, dict] = {}
    with open(sidecar, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                entry = json.loads(line)
                entries_by_key[entry["id"]] = entry
    print(f"[batch:fetch] Sidecar: {len(entries_by_key)} entries from {sidecar}")

    # Retrieve result content
    dest = getattr(batch_job, "dest", None)
    if dest is None:
        _abort_fetch_no_dest(batch_job)

    file_name = getattr(dest, "file_name", None)
    inlined = getattr(dest, "inlined_responses", None)

    if file_name:
        print(f"[batch:fetch] Downloading results from {file_name} ...")
        raw_bytes = client.files.download(file=file_name)
        results_jsonl = raw_bytes.decode("utf-8")
        _parse_results_jsonl(
            results_jsonl, prompt_variant, model, output_path, parse_failures_path
        )

    elif inlined is not None:
        print(f"[batch:fetch] Reading {len(list(inlined))} inline responses ...")
        # Re-fetch because iterating above may exhaust the iterator
        batch_job = client.batches.get(name=job_name)
        inlined = getattr(batch_job.dest, "inlined_responses", [])
        _parse_sdk_responses(
            inlined, prompt_variant, model, output_path, parse_failures_path
        )

    else:
        _abort_fetch_no_dest(batch_job)


def _abort_fetch_no_dest(batch_job: Any) -> None:
    dest = getattr(batch_job, "dest", None)
    print("[batch:fetch] ERROR: Could not locate results in batch job object.")
    print("  This may be a google-genai SDK version issue.")
    print("  Try: pip install --upgrade google-genai")
    if dest is not None:
        attrs = [a for a in dir(dest) if not a.startswith("_")]
        print(f"  batch_job.dest attributes: {attrs}")
    sys.exit(1)


# ---------------------------------------------------------------------------
# Result parsing
# ---------------------------------------------------------------------------

def _parse_results_jsonl(
    results_jsonl: str,
    prompt_variant: str,
    model: str,
    output_path: Path,
    parse_failures_path: Path,
) -> None:
    """Parse a results JSONL string (file-based output) and write to output_path."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    raw_path = output_path.with_name(output_path.stem + ".raw.jsonl")
    raw_path.write_text(results_jsonl, encoding="utf-8")
    print(f"[batch:fetch] Raw API response saved -> {raw_path}")
    n_ok = n_parse_fail = n_err = 0
    total_prompt_toks = total_comp_toks = 0

    with open(output_path, "a", encoding="utf-8") as out_f:
        for line in results_jsonl.splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                print(f"  [WARN] Skipping unparseable line: {line[:80]}")
                continue

            key = obj.get("key", "?")

            if obj.get("error"):
                print(f"  [{key}] API ERROR: {obj['error']}")
                n_err += 1
                continue

            response = obj.get("response", {})
            candidates = response.get("candidates", [])
            if not candidates:
                print(f"  [{key}] ERROR: empty candidates")
                n_err += 1
                continue

            parts = candidates[0].get("content", {}).get("parts", [])
            raw_text = parts[0].get("text", "") if parts else ""

            # usageMetadata may use camelCase (REST) or snake_case (SDK-serialised)
            usage = response.get("usageMetadata", response.get("usage_metadata", {}))
            prompt_toks = usage.get("promptTokenCount", usage.get("prompt_token_count", 0))
            comp_toks = usage.get("candidatesTokenCount", usage.get("candidates_token_count", 0))

            total_prompt_toks += prompt_toks
            total_comp_toks += comp_toks

            n_ok, n_parse_fail = _write_result(
                out_f, key, raw_text, prompt_variant, model,
                prompt_toks, comp_toks, parse_failures_path, n_ok, n_parse_fail,
            )

    _print_fetch_summary(n_ok, n_parse_fail, n_err, model, total_prompt_toks, total_comp_toks, output_path)


def _parse_sdk_responses(
    responses: Any,
    prompt_variant: str,
    model: str,
    output_path: Path,
    parse_failures_path: Path,
) -> None:
    """Parse inline SDK response objects (inlined_responses path) and write to output_path."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    n_ok = n_parse_fail = n_err = 0
    total_prompt_toks = total_comp_toks = 0

    with open(output_path, "a", encoding="utf-8") as out_f:
        for resp in responses:
            key = getattr(resp, "key", "?")

            if getattr(resp, "error", None):
                print(f"  [{key}] API ERROR: {resp.error}")
                n_err += 1
                continue

            response = getattr(resp, "response", None)
            if response is None:
                print(f"  [{key}] ERROR: no response object")
                n_err += 1
                continue

            raw_text = getattr(response, "text", "")
            usage = getattr(response, "usage_metadata", None)
            prompt_toks = getattr(usage, "prompt_token_count", 0) if usage else 0
            comp_toks = getattr(usage, "candidates_token_count", 0) if usage else 0

            total_prompt_toks += prompt_toks
            total_comp_toks += comp_toks

            n_ok, n_parse_fail = _write_result(
                out_f, key, raw_text, prompt_variant, model,
                prompt_toks, comp_toks, parse_failures_path, n_ok, n_parse_fail,
            )

    _print_fetch_summary(n_ok, n_parse_fail, n_err, model, total_prompt_toks, total_comp_toks, output_path)


def _write_result(
    out_f,
    key: str,
    raw_text: str,
    prompt_variant: str,
    model: str,
    prompt_toks: int,
    comp_toks: int,
    parse_failures_path: Path,
    n_ok: int,
    n_parse_fail: int,
) -> tuple[int, int]:
    parsed = parse_response(raw_text)
    parse_success = parsed is not None

    if parse_success:
        print(f"  [{key}] ok")
        n_ok += 1
    else:
        print(f"  [{key}] PARSE FAIL")
        n_parse_fail += 1
        log_parse_failure(raw_text, key, parse_failures_path)

    out_f.write(
        json.dumps(
            _build_result_dict(key, prompt_variant, model, parsed, parse_success, prompt_toks, comp_toks)
        ) + "\n"
    )
    _append_log_row(
        DEFAULT_GEMINI_LOG_PATH,
        model=_resolve_pricing_model(model),
        prompt_tokens=prompt_toks,
        completion_tokens=comp_toks,
        note=f"batch_{prompt_variant}",
        discount=_BATCH_DISCOUNT,
    )
    return n_ok, n_parse_fail


def _print_fetch_summary(
    n_ok: int, n_parse_fail: int, n_err: int,
    model: str, prompt_toks: int, comp_toks: int, output_path: Path,
) -> None:
    print()
    print(f"[batch:fetch] {n_ok} ok, {n_parse_fail} parse failures, {n_err} errors.")
    print(f"[batch:fetch] Results -> {output_path}")

    prices = PRICING.get(model, {})
    if prices and (prompt_toks or comp_toks):
        full_usd = (prompt_toks * prices["input"] + comp_toks * prices["output"]) / 1_000_000
        batch_usd = full_usd * _BATCH_DISCOUNT
        print(
            f"[batch:fetch] Tokens: {prompt_toks}+{comp_toks} | "
            f"Cost: ${batch_usd:.4f} USD (50% batch discount applied)"
        )
