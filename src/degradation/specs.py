import random
from dataclasses import dataclass, field
from typing import Literal


def resolve_jitter(params: dict) -> dict:
    """Resolve min_X / max_X pairs by sampling uniformly, returning a new dict.

    Any key pair like ``min_foo`` / ``max_foo`` is replaced by a single ``foo``
    sampled from ``uniform(min_foo, max_foo)``.  All other keys pass through
    unchanged.  The original dict is not modified.
    """
    mins = {k[4:]: v for k, v in params.items() if k.startswith("min_")}
    maxs = {k[4:]: v for k, v in params.items() if k.startswith("max_")}
    resolved = dict(params)
    for key in mins:
        if key in maxs:
            resolved[key] = random.uniform(mins[key], maxs[key])
            del resolved[f"min_{key}"]
            del resolved[f"max_{key}"]
    return resolved


@dataclass
class DegradationSpec:
    dimension: str          # "color_offset", "position_offset", "scale_error", "char_substitution"
    magnitude: str          # human-readable label e.g. "delta_e_10"
    element_id: str         # which text element to corrupt
    target: Literal["correct_element", "wrong_element"] = "correct_element"
    layer: Literal["html", "image"] = "html"
    params: dict = field(default_factory=dict)
