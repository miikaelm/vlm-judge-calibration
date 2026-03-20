"""
parser.py — Parse VLM response text into a structured dict.

Handles JSON in several formats:
  - Pure JSON string
  - JSON inside markdown code fences (```json ... ```)
  - JSON embedded within surrounding prose

Returns None on parse failure.
"""

from __future__ import annotations

import json
import re
from datetime import datetime, timezone
from pathlib import Path


def parse_response(raw_text: str) -> dict | None:
    """
    Extract and parse JSON from a VLM response string.

    Tries three strategies in order:
      1. Direct json.loads on stripped text
      2. Extract content inside ``` or ```json fences
      3. Regex-find the first {...} block in the text

    Returns the parsed dict on success, None if all strategies fail.
    """
    if not raw_text or not raw_text.strip():
        return None

    text = raw_text.strip()

    # Strategy 1: direct parse
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # Strategy 2: strip markdown code fences
    fence_match = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", text)
    if fence_match:
        try:
            return json.loads(fence_match.group(1))
        except json.JSONDecodeError:
            pass

    # Strategy 3: find the first {...} block
    brace_match = re.search(r"\{[\s\S]*\}", text)
    if brace_match:
        try:
            return json.loads(brace_match.group(0))
        except json.JSONDecodeError:
            pass

    return None


def log_parse_failure(raw_text: str, stimulus_id: str, log_path: Path) -> None:
    """Append a parse failure record to a JSONL file for debugging."""
    record = {
        "timestamp_utc": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S"),
        "stimulus_id": stimulus_id,
        "raw_response": raw_text,
    }
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with open(log_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record) + "\n")
