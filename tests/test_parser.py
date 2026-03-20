"""
tests/test_parser.py — Tests for src/evaluation/parser.py

10 test cases covering:
  - Pure JSON
  - JSON in ```json code fence
  - JSON in plain ``` code fence
  - JSON embedded in prose (after text)
  - JSON preceded by explanation text
  - JSON in code fence with surrounding prose
  - Completely malformed input
  - Truncated JSON
  - Empty string
  - JSON with extra whitespace / newlines
"""

import json
import pytest

from src.evaluation.parser import parse_response


# Canonical valid scores dict used across tests
VALID_SCORES = {
    "instruction_following": 4,
    "text_accuracy": 5,
    "visual_consistency": 3,
    "layout_preservation": 4,
    "overall_quality": 4,
    "errors_noticed": "slight color mismatch",
}


def test_pure_json():
    """Clean JSON string parses to the expected dict."""
    raw = json.dumps(VALID_SCORES)
    result = parse_response(raw)
    assert result == VALID_SCORES


def test_json_in_json_code_fence():
    """JSON wrapped in ```json ... ``` parses correctly."""
    raw = f"```json\n{json.dumps(VALID_SCORES)}\n```"
    result = parse_response(raw)
    assert result == VALID_SCORES


def test_json_in_plain_code_fence():
    """JSON wrapped in ``` ... ``` (no language tag) parses correctly."""
    raw = f"```\n{json.dumps(VALID_SCORES)}\n```"
    result = parse_response(raw)
    assert result == VALID_SCORES


def test_json_embedded_in_trailing_prose():
    """JSON followed by trailing prose still parses."""
    raw = f"{json.dumps(VALID_SCORES)}\n\nI hope that helps."
    result = parse_response(raw)
    assert result is not None
    assert result["instruction_following"] == VALID_SCORES["instruction_following"]


def test_json_preceded_by_leading_prose():
    """JSON preceded by explanation text parses correctly."""
    raw = f"Based on my analysis of the images, here are my scores:\n\n{json.dumps(VALID_SCORES)}"
    result = parse_response(raw)
    assert result is not None
    assert result["overall_quality"] == VALID_SCORES["overall_quality"]


def test_json_in_code_fence_with_surrounding_prose():
    """JSON inside a code fence surrounded by prose text parses correctly."""
    raw = (
        "I evaluated the image and here is the result:\n\n"
        f"```json\n{json.dumps(VALID_SCORES)}\n```\n\n"
        "Please let me know if you need more details."
    )
    result = parse_response(raw)
    assert result is not None
    assert result["text_accuracy"] == VALID_SCORES["text_accuracy"]


def test_completely_malformed_input():
    """Completely malformed input (no JSON at all) returns None."""
    raw = "This is just some text with no JSON at all."
    result = parse_response(raw)
    assert result is None


def test_truncated_json():
    """Truncated JSON (missing closing brace) returns None."""
    raw = '{"instruction_following": 4, "text_accuracy": 5'
    result = parse_response(raw)
    assert result is None


def test_empty_string():
    """Empty string returns None."""
    result = parse_response("")
    assert result is None


def test_json_with_extra_whitespace():
    """JSON with lots of indentation and newlines still parses correctly."""
    raw = """
    {
        "instruction_following":   4,
        "text_accuracy":           5,
        "visual_consistency":      3,
        "layout_preservation":     4,
        "overall_quality":         4,
        "errors_noticed":          ""
    }
    """
    result = parse_response(raw)
    assert result is not None
    assert result["instruction_following"] == 4
    assert result["errors_noticed"] == ""
