"""
judge.py — VLM judge: load a stimulus triple and call the VLM for Experiment 2 evaluation.

Usage:
    from src.evaluation.judge import JudgeConfig, JudgeResult, run_judge
    from src.api_tracker import DummyOpenAI

    config = JudgeConfig(model="dummy", prompt_variant="experiment_2")
    result = run_judge(Path("data/pilot/stimulus_001"), config, client=DummyOpenAI())
"""

from __future__ import annotations

import base64
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


_PROMPTS_PATH = Path(__file__).parent.parent.parent / "configs" / "vlm_prompts.yaml"


@dataclass
class JudgeConfig:
    model: str = "gpt-4o"
    temperature: float = 0.0
    max_tokens: int = 512
    prompt_variant: str = "experiment_2"
    prompts_path: Path = field(default_factory=lambda: _PROMPTS_PATH)
    log_note: str = ""


@dataclass
class JudgeResult:
    stimulus_id: str
    instruction_following: int | None
    text_accuracy: int | None
    visual_consistency: int | None
    layout_preservation: int | None
    overall_quality: int | None
    errors_noticed: str
    raw_response: str
    parse_success: bool
    model: str
    prompt_tokens: int
    completion_tokens: int
    # Experiment 1 only
    detected_difference: bool | None = None


def _load_prompt(prompts_path: Path, variant: str) -> dict:
    with open(prompts_path, encoding="utf-8") as f:
        data = yaml.safe_load(f)
    try:
        return data["prompts"][variant]
    except KeyError:
        raise ValueError(f"Prompt variant '{variant}' not found in {prompts_path}")


def _encode_image(image_path: Path) -> str:
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def _clamp_score(val: Any) -> int | None:
    """Return val if it is an integer in [1, 5], else None."""
    if isinstance(val, int) and 1 <= val <= 5:
        return val
    if isinstance(val, float) and val == int(val) and 1 <= val <= 5:
        return int(val)
    return None


def run_judge(stimulus_dir: Path, config: JudgeConfig, client: Any) -> JudgeResult:
    """
    Load a stimulus directory, call the VLM, and return a JudgeResult.

    client must expose .chat.completions.create() — use TrackedOpenAI for real
    API calls or DummyOpenAI for zero-cost smoke testing.

    For experiment_2: sends source.png + degraded.png + the edit instruction.
    For experiment_1: sends ground_truth.png + degraded.png (no instruction).
    """
    from src.evaluation.parser import parse_response

    stimulus_dir = Path(stimulus_dir)
    metadata = json.loads((stimulus_dir / "metadata.json").read_text(encoding="utf-8"))

    stimulus_id = metadata["stimulus_id"]
    instruction = metadata["edit_instruction"]
    files = metadata["files"]

    prompt = _load_prompt(config.prompts_path, config.prompt_variant)
    system_msg = prompt["system"].strip()

    if config.prompt_variant == "experiment_2":
        user_text = prompt["user_template"].replace("{instruction}", instruction)
        img1_path = stimulus_dir / files["source"]
        img2_path = stimulus_dir / files["degraded"]
    elif config.prompt_variant == "experiment_1":
        user_text = prompt["user_template"]
        img1_path = stimulus_dir / files["ground_truth"]
        img2_path = stimulus_dir / files["degraded"]
    else:
        raise ValueError(f"Unknown prompt_variant: {config.prompt_variant!r}")

    img1_b64 = _encode_image(img1_path)
    img2_b64 = _encode_image(img2_path)

    messages = [
        {"role": "system", "content": system_msg},
        {
            "role": "user",
            "content": [
                {"type": "text", "text": user_text},
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/png;base64,{img1_b64}",
                        "detail": "high",
                    },
                },
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/png;base64,{img2_b64}",
                        "detail": "high",
                    },
                },
            ],
        },
    ]

    response = client.chat.completions.create(
        model=config.model,
        messages=messages,
        max_tokens=config.max_tokens,
        temperature=config.temperature,
    )

    raw_response = response.choices[0].message.content
    usage = response.usage
    prompt_tokens = usage.prompt_tokens if usage else 0
    completion_tokens = usage.completion_tokens if usage else 0

    parsed = parse_response(raw_response)
    parse_success = parsed is not None

    if config.prompt_variant == "experiment_2":
        return JudgeResult(
            stimulus_id=stimulus_id,
            instruction_following=_clamp_score(parsed.get("instruction_following")) if parsed else None,
            text_accuracy=_clamp_score(parsed.get("text_accuracy")) if parsed else None,
            visual_consistency=_clamp_score(parsed.get("visual_consistency")) if parsed else None,
            layout_preservation=_clamp_score(parsed.get("layout_preservation")) if parsed else None,
            overall_quality=_clamp_score(parsed.get("overall_quality")) if parsed else None,
            errors_noticed=parsed.get("errors_noticed", "") if parsed else "",
            raw_response=raw_response,
            parse_success=parse_success,
            model=config.model,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
        )
    else:
        # experiment_1: perceptual sensitivity — ground_truth vs degraded comparison
        detected_difference: bool | None = None
        if parsed is not None:
            raw_det = parsed.get("detected_difference")
            if isinstance(raw_det, bool):
                detected_difference = raw_det
        return JudgeResult(
            stimulus_id=stimulus_id,
            instruction_following=None,
            text_accuracy=None,
            visual_consistency=_clamp_score(parsed.get("similarity_score")) if parsed else None,
            layout_preservation=None,
            overall_quality=None,
            errors_noticed=parsed.get("description", "") if parsed else "",
            raw_response=raw_response,
            parse_success=parse_success,
            model=config.model,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            detected_difference=detected_difference,
        )
