"""
layouts/core.py — LayoutDefinition, RoleConstraints, registry, and shared HTML helpers.

A LayoutDefinition declares:
  - roles: text elements present in the scene (e.g. ["title", "subtitle"])
  - supported_edits: which edit types work for this layout
  - role_constraints: per-role allowed edits and affordances
  - role_base_styles: default CSS properties per role (color, font_size_px, etc.)
  - default_content: default text per role
  - html_builder: function (contents, styles, bg) -> html_str

Usage:
    from layouts import get_layouts_for_edit, get_layout, all_layouts

    color_layouts = get_layouts_for_edit("color")
    layout = get_layout("title_subtitle")
    html = layout.html_builder(layout.default_content, layout.role_base_styles, layout.background)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable


# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------

@dataclass
class RoleConstraints:
    """Per-role constraints declaring which edit types are valid for this element."""
    color_editable: bool = True
    can_scale: bool = True          # font-size edits
    can_rotate: bool = False
    rotation_range: tuple[int, int] = (-30, 30)
    can_align: bool = False
    alignment_positions: list[str] = field(default_factory=list)


@dataclass
class LayoutDefinition:
    """Complete description of a scene layout."""
    name: str
    difficulty: str                  # "easy" | "medium" | "hard" | "multi"
    roles: list[str]
    # Edit types this layout supports.
    supported_edits: list[str]
    # Per-role edit constraints.
    role_constraints: dict[str, RoleConstraints]
    # Default CSS style properties per role.
    # Required keys per role: color, font_size_px, font_family, font_weight, font_style.
    # Optional keys: letter_spacing, line_height, rotation_deg, text_shadow, alignment.
    role_base_styles: dict[str, dict]
    # Default text content per role.
    default_content: dict[str, str]
    # CSS background value for the scene (e.g. "#FFFFFF", "linear-gradient(...)").
    background: str
    background_type: str             # "solid" | "gradient"
    # Renders the layout to an HTML string.
    # Signature: (contents, styles, bg) -> html_str
    # contents: {role: text_string}
    # styles:   {role: style_dict} — caller may override any property before passing in
    # bg:       CSS background value
    html_builder: Callable[[dict[str, str], dict[str, dict], str], str]
    notes: str = ""


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

_REGISTRY: dict[str, LayoutDefinition] = {}


def register(layout: LayoutDefinition) -> LayoutDefinition:
    """Register a layout and return it (usable as a statement)."""
    _REGISTRY[layout.name] = layout
    return layout


def get_layout(name: str) -> LayoutDefinition:
    if name not in _REGISTRY:
        raise KeyError(f"Unknown layout: {name!r}. Available: {sorted(_REGISTRY)}")
    return _REGISTRY[name]


def get_layouts_for_edit(edit_type: str) -> list[LayoutDefinition]:
    """Return all layouts that declare support for the given edit type."""
    return [layout for layout in _REGISTRY.values() if edit_type in layout.supported_edits]


def get_layouts_by_difficulty(difficulty: str) -> list[LayoutDefinition]:
    return [layout for layout in _REGISTRY.values() if layout.difficulty == difficulty]


def all_layouts() -> list[LayoutDefinition]:
    return list(_REGISTRY.values())


# ---------------------------------------------------------------------------
# Shared HTML rendering helpers
# ---------------------------------------------------------------------------

def _role_css(s: dict) -> str:
    """Convert a role style dict to an inline CSS string.

    Expected keys: color, font_size_px, font_family, font_weight, font_style.
    Optional keys: letter_spacing, line_height, rotation_deg, text_shadow.
    Keys that are layout/metadata (alignment, is_heading) are ignored.
    """
    parts = [
        f"color:{s['color']}",
        f"font-size:{s['font_size_px']}px",
        f"font-family:{s['font_family']}",
        f"font-weight:{s['font_weight']}",
        f"font-style:{s['font_style']}",
        f"letter-spacing:{s.get('letter_spacing', 'normal')}",
        "margin:0",
    ]
    lh = s.get("line_height")
    if lh is not None:
        parts.append(f"line-height:{lh}")
    rot = s.get("rotation_deg", 0.0)
    if rot:
        parts.append(f"transform:rotate({rot:.1f}deg)")
    text_shadow = s.get("text_shadow")
    if text_shadow:
        parts.append(f"text-shadow:{text_shadow}")
    return "; ".join(parts)
