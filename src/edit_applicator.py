"""
edit_applicator.py — Apply ground-truth (correct) edits to base HTML templates.

Each edit type modifies the HTML/CSS to produce the correct edited version,
which serves as the ground-truth image in a stimulus triple.
"""

import re

from src.degradation.engine import _inject_css_override


def apply_correct_edit(base_html: str, edit_type: str, edit_params: dict) -> str:
    """
    Apply the 'correct' (ground-truth) edit to base_html.
    Returns modified HTML string.

    Supported edit_types: color_change, scale_change, content_change, position_change
    """
    handlers = {
        "color_change": _apply_color_change,
        "scale_change": _apply_scale_change,
        "content_change": _apply_content_change,
        "position_change": _apply_position_change,
    }
    handler = handlers.get(edit_type)
    if handler is None:
        raise ValueError(f"Unknown edit_type: {edit_type!r}. Supported: {list(handlers)}")
    return handler(base_html, edit_params)


def _apply_color_change(base_html: str, edit_params: dict) -> str:
    """
    Change the color of element_id to new_color.

    edit_params:
        element_id (str): HTML id of the target element
        new_color (str): hex color string e.g. "#3B82F6"
    """
    element_id = edit_params["element_id"]
    new_color = edit_params["new_color"]
    css = f"color: {new_color} !important;"
    return _inject_css_override(base_html, element_id, css)


def _apply_scale_change(base_html: str, edit_params: dict) -> str:
    """
    Change the font size of element_id to new_font_size_px.

    edit_params:
        element_id (str): HTML id of the target element
        new_font_size_px (int): new font size in pixels
    """
    element_id = edit_params["element_id"]
    new_size = edit_params["new_font_size_px"]
    css = f"font-size: {new_size}px !important;"
    return _inject_css_override(base_html, element_id, css)


def _apply_content_change(base_html: str, edit_params: dict) -> str:
    """
    Replace the full text content of element_id with new_text.

    edit_params:
        element_id (str): HTML id of the target element
        new_text (str): replacement text
    """
    element_id = edit_params["element_id"]
    new_text = edit_params["new_text"]

    pattern = re.compile(
        r'(id=["\']' + re.escape(element_id) + r'["\'][^>]*>)([^<]*)',
        re.DOTALL,
    )
    match = pattern.search(base_html)
    if not match:
        return base_html

    return base_html[: match.start(2)] + new_text + base_html[match.end(2):]


def _apply_position_change(base_html: str, edit_params: dict) -> str:
    """
    Move element_id to an absolute position.

    edit_params:
        element_id (str): HTML id of the target element
        new_left_px (int): new left position in pixels
        new_top_px (int): new top position in pixels
    """
    element_id = edit_params["element_id"]
    left = edit_params["new_left_px"]
    top = edit_params["new_top_px"]
    css = f"position: absolute !important; left: {left}px !important; top: {top}px !important; right: auto !important; bottom: auto !important;"
    return _inject_css_override(base_html, element_id, css)
