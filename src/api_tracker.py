"""
api_tracker.py — VLM API usage tracker and dummy client.

Copied from quantitative-text-editing/src/gen_pipeline/api_tracker.py and extended
with pricing entries for Gemini and Anthropic models.

Usage:
    from api_tracker import TrackedOpenAI, DummyOpenAI

    client = TrackedOpenAI(api_key=os.environ["OPENAI_API_KEY"], note="exp2_gpt4o")
    client = DummyOpenAI()   # zero-cost smoke testing

Log file: api_usage_log.csv — one row per call, appended immediately.
"""

from __future__ import annotations

import csv
import json
import random
import string
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

# ---------------------------------------------------------------------------
# Pricing table (USD per 1 000 000 tokens)
# ---------------------------------------------------------------------------

PRICING: dict[str, dict[str, float]] = {
    # OpenAI
    "gpt-4o":             {"input": 2.50,  "output": 10.00},
    "gpt-4o-mini":        {"input": 0.15,  "output": 0.60},
    "gpt-4.1":            {"input": 2.00,  "output": 8.00},
    "gpt-4.1-mini":       {"input": 0.40,  "output": 1.60},
    # Anthropic
    "claude-3-5-sonnet":  {"input": 3.00,  "output": 15.00},
    "claude-3-haiku":     {"input": 0.25,  "output": 1.25},
    # Gemini (approximate — check ai.google.dev for current rates)
    "gemini-1.5-pro":     {"input": 3.50,  "output": 10.50},
    "gemini-1.5-flash":   {"input": 0.075, "output": 0.30},
    "gemini-2.0-flash":   {"input": 0.10,  "output": 0.40},
    "gemini-2.5-pro":     {"input": 1.25,  "output": 10.00},  # verify at ai.google.dev
    # Gemini 3.1 Pro Preview — prompts <=200k tokens (our use case)
    "gemini-3.1-pro":     {"input": 2.00,  "output": 12.00},
}

DEFAULT_LOG_PATH = Path(__file__).parent / "api_usage_log.csv"

CSV_HEADER = [
    "timestamp_utc",
    "model",
    "prompt_tokens",
    "completion_tokens",
    "total_tokens",
    "prompt_cost_usd",
    "completion_cost_usd",
    "total_cost_usd",
    "note",
]


# ---------------------------------------------------------------------------
# Shared logging logic
# ---------------------------------------------------------------------------

def _compute_cost(model: str, prompt_tokens: int, completion_tokens: int) -> tuple[float, float]:
    prices = PRICING.get(model)
    if prices is None:
        print(f"  [api_tracker] WARNING: no pricing data for '{model}'. Logging $0.")
        return 0.0, 0.0
    prompt_cost = prompt_tokens * prices["input"] / 1_000_000
    completion_cost = completion_tokens * prices["output"] / 1_000_000
    return prompt_cost, completion_cost


def _append_log_row(
    log_path: Path,
    model: str,
    prompt_tokens: int,
    completion_tokens: int,
    note: str = "",
) -> None:
    prompt_cost, completion_cost = _compute_cost(model, prompt_tokens, completion_tokens)
    total_cost = prompt_cost + completion_cost
    total_tokens = prompt_tokens + completion_tokens
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")

    write_header = not log_path.exists()
    with open(log_path, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        if write_header:
            writer.writerow(CSV_HEADER)
        writer.writerow([
            ts, model, prompt_tokens, completion_tokens, total_tokens,
            f"{prompt_cost:.6f}", f"{completion_cost:.6f}", f"{total_cost:.6f}", note,
        ])

    print(
        f"  [api_tracker] {model} | "
        f"tokens: {prompt_tokens}+{completion_tokens}={total_tokens} | "
        f"cost: ${total_cost:.4f}"
    )


# ---------------------------------------------------------------------------
# Thin wrappers that mimic openai.OpenAI's chat.completions interface
# ---------------------------------------------------------------------------

@dataclass
class _FakeUsage:
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int


@dataclass
class _FakeMessage:
    content: str
    role: str = "assistant"


@dataclass
class _FakeChoice:
    message: _FakeMessage
    index: int = 0
    finish_reason: str = "stop"


@dataclass
class _FakeCompletion:
    choices: list[_FakeChoice]
    usage: _FakeUsage
    model: str = "dummy"
    id: str = "dummy-0"


class _TrackedCompletions:
    def __init__(self, real_completions: Any, log_path: Path, note: str) -> None:
        self._real = real_completions
        self._log_path = log_path
        self._note = note

    def create(self, **kwargs) -> Any:
        response = self._real.create(**kwargs)
        usage = response.usage
        if usage:
            _append_log_row(
                self._log_path,
                model=kwargs.get("model", "unknown"),
                prompt_tokens=usage.prompt_tokens,
                completion_tokens=usage.completion_tokens,
                note=self._note,
            )
        return response


class _TrackedChat:
    def __init__(self, real_chat: Any, log_path: Path, note: str) -> None:
        self.completions = _TrackedCompletions(real_chat.completions, log_path, note)


class TrackedOpenAI:
    """Drop-in replacement for openai.OpenAI that logs token usage and cost per call."""

    def __init__(
        self,
        api_key: str | None = None,
        log_path: Path | str = DEFAULT_LOG_PATH,
        note: str = "",
        **openai_kwargs: Any,
    ) -> None:
        try:
            from openai import OpenAI
        except ImportError:
            raise ImportError("openai package not installed. Run: pip install openai")

        self._client = OpenAI(api_key=api_key, **openai_kwargs)
        self._log_path = Path(log_path)
        self.chat = _TrackedChat(self._client.chat, self._log_path, note)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._client, name)


# ---------------------------------------------------------------------------
# Dummy client — zero cost, instant, returns plausible JSON
# ---------------------------------------------------------------------------

def _random_word(n: int = 6) -> str:
    return "".join(random.choices(string.ascii_lowercase, k=n))


class _DummyCompletions:
    def __init__(self, prompt_variant: str = "experiment_2") -> None:
        self._prompt_variant = prompt_variant

    def create(self, *, model: str = "dummy", messages: list, **_) -> _FakeCompletion:
        if self._prompt_variant == "experiment_1":
            # Experiment 1: perceptual sensitivity — ground_truth vs degraded comparison
            magnitude_hint = random.random()  # simulate degradation magnitude effect
            content = json.dumps({
                "detected_difference": magnitude_hint > 0.4,
                "similarity_score": max(1, min(5, round(5 - magnitude_hint * 4))),
                "description": "dummy response — no real inference performed",
            })
        else:
            # Experiment 2: instruction-following evaluation
            content = json.dumps({
                "instruction_following": random.randint(1, 5),
                "text_accuracy": random.randint(1, 5),
                "visual_consistency": random.randint(1, 5),
                "layout_preservation": random.randint(1, 5),
                "overall_quality": random.randint(1, 5),
                "errors_noticed": "dummy response — no real inference performed",
            })
        prompt_chars = sum(len(m.get("content", "") if isinstance(m.get("content"), str) else "") for m in messages)
        prompt_tokens = max(1, prompt_chars // 4)
        completion_tokens = max(1, len(content) // 4)
        return _FakeCompletion(
            choices=[_FakeChoice(message=_FakeMessage(content=content))],
            usage=_FakeUsage(prompt_tokens, completion_tokens, prompt_tokens + completion_tokens),
            model=model,
        )


class _DummyChat:
    def __init__(self, prompt_variant: str = "experiment_2") -> None:
        self.completions = _DummyCompletions(prompt_variant)


class DummyOpenAI:
    """Zero-cost mock for smoke-testing the evaluation pipeline without API calls."""

    def __init__(self, prompt_variant: str = "experiment_2", **_: Any) -> None:
        self.chat = _DummyChat(prompt_variant)
        print("  [api_tracker] DummyOpenAI active — no real API calls will be made.")


# ---------------------------------------------------------------------------
# Protocol
# ---------------------------------------------------------------------------

@runtime_checkable
class OpenAIClient(Protocol):
    """Structural type for any client with a .chat.completions.create interface."""
    chat: Any
