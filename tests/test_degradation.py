"""Tests for the degradation engine — no rendering required."""
import re
import pytest
from src.degradation.specs import DegradationSpec
from src.degradation.engine import apply_degradation, _inject_css_override

SIMPLE_HTML = """<!DOCTYPE html>
<html><head><meta charset="UTF-8"><style>
#heading { color: #1A1A1A; font-size: 64px; }
</style></head><body>
<h1 id="heading">Sample Heading</h1>
</body></html>"""


def test_inject_css_override_before_head():
    result = _inject_css_override(SIMPLE_HTML, "heading", "color: red !important;")
    assert "</head>" in result
    # The style block should appear before </head>
    style_pos = result.rfind("<style>")
    head_pos = result.rfind("</head>")
    assert style_pos < head_pos
    assert "color: red !important;" in result


def test_degrade_color_injects_style():
    spec = DegradationSpec(
        dimension="color_offset",
        magnitude="delta_e_15",
        element_id="heading",
        params={"base_color": "#1A1A1A", "delta_e": 15.0},
    )
    result = apply_degradation(SIMPLE_HTML, spec)
    assert "#heading" in result
    assert "color:" in result
    # The injected color should be different from the base
    # Find all matches and take the last one (the injected override)
    matches = re.findall(r"#heading\s*\{[^}]*color:\s*(#[0-9A-Fa-f]{6})", result)
    assert len(matches) > 0
    injected_color = matches[-1]
    assert injected_color.upper() != "#1A1A1A"


def test_degrade_position_injects_transform():
    spec = DegradationSpec(
        dimension="position_offset",
        magnitude="offset_20px",
        element_id="heading",
        params={"offset_x_px": 20, "offset_y_px": 20},
    )
    result = apply_degradation(SIMPLE_HTML, spec)
    assert "transform: translate(20px, 20px)" in result


def test_degrade_scale_injects_font_size():
    spec = DegradationSpec(
        dimension="scale_error",
        magnitude="scale_25pct",
        element_id="heading",
        params={"base_font_size_px": 64, "scale_error_pct": 25.0},
    )
    result = apply_degradation(SIMPLE_HTML, spec)
    assert "font-size: 80px" in result  # 64 * 1.25 = 80


def test_degrade_char_substitution():
    spec = DegradationSpec(
        dimension="char_substitution",
        magnitude="1_similar",
        element_id="heading",
        params={"substitutions": [{"position": 0, "original": "S", "replacement": "5"}]},
    )
    result = apply_degradation(SIMPLE_HTML, spec)
    # The heading text "Sample Heading" should now start with "5"
    assert "5ample Heading" in result


def test_color_magnitude_produces_different_values():
    """Larger ΔE should produce more different colors."""
    from src.degradation.color_utils import hex_to_rgb, compute_delta_e

    base = "#3B82F6"

    small_spec = DegradationSpec(
        dimension="color_offset", magnitude="delta_e_5", element_id="heading",
        params={"base_color": base, "delta_e": 5.0},
    )
    large_spec = DegradationSpec(
        dimension="color_offset", magnitude="delta_e_30", element_id="heading",
        params={"base_color": base, "delta_e": 30.0},
    )

    small_result = apply_degradation(SIMPLE_HTML, small_spec)
    large_result = apply_degradation(SIMPLE_HTML, large_spec)

    # Extract injected colors — take the last match (the override, not the base style)
    def extract_color(html):
        matches = re.findall(r"#heading\s*\{[^}]*color:\s*(#[0-9A-Fa-f]{6})", html)
        return matches[-1] if matches else None

    small_color = extract_color(small_result)
    large_color = extract_color(large_result)

    assert small_color is not None
    assert large_color is not None

    small_de = compute_delta_e(hex_to_rgb(base), hex_to_rgb(small_color))
    large_de = compute_delta_e(hex_to_rgb(base), hex_to_rgb(large_color))

    assert large_de > small_de


def test_unknown_dimension_raises():
    spec = DegradationSpec(
        dimension="nonexistent", magnitude="x", element_id="heading", params={}
    )
    with pytest.raises(ValueError):
        apply_degradation(SIMPLE_HTML, spec)
