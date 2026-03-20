from dataclasses import dataclass, field
from typing import Literal


@dataclass
class DegradationSpec:
    dimension: str          # "color_offset", "position_offset", "scale_error", "char_substitution"
    magnitude: str          # human-readable label e.g. "delta_e_10"
    element_id: str         # which text element to corrupt
    target: Literal["correct_element", "wrong_element"] = "correct_element"
    layer: Literal["html", "image"] = "html"
    params: dict = field(default_factory=dict)
