"""
test_layouts.py — Layout architecture verification.

Tests:
  1. All 11 layouts render to a valid 512×512 PNG with no Playwright console errors
  2. get_layout() and get_layouts_for_edit() behave correctly
  3. Each layout's roles, role_constraints, and role_base_styles are internally consistent
  4. Style override: modifying one property in styles produces different HTML
  5. Alignment positions: every declared alignment_positions value renders without crash

Run:
    pytest tests/test_layouts.py -v
"""

import copy
import sys
from pathlib import Path

import pytest
from PIL import Image

sys.path.insert(0, str(Path(__file__).parents[1] / "src"))

from render import render_html_sync, RenderConfig
from layouts import (
    all_layouts,
    get_layout,
    get_layouts_for_edit,
    get_layouts_by_difficulty,
)

RENDER_CFG = RenderConfig(width=1024, height=1024, downscale_to=512)
OUT_DIR = Path(__file__).parents[1] / "data" / "test_renders"

EXPECTED_LAYOUT_NAMES = {
    "solo_headline",
    "title_subtitle",
    "header_body",
    "title_byline",
    "name_card",
    "quote_attribution",
    "corner_badge",
    "split_panel",
    "banner_caption",
    "two_column_heading",
    "low_contrast_centered",
}


@pytest.fixture(scope="session", autouse=True)
def make_output_dir():
    OUT_DIR.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# 1. Rendering — one case per layout
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("layout", all_layouts(), ids=lambda l: l.name)
def test_layout_renders(layout):
    """Each layout renders to a 512×512 PNG with no console errors."""
    out_path = OUT_DIR / f"{layout.name}.png"
    html = layout.html_builder(layout.default_content, layout.role_base_styles, layout.background)

    assert isinstance(html, str) and len(html) > 100

    result = render_html_sync(html, out_path, config=RENDER_CFG)

    assert out_path.exists()
    assert out_path.stat().st_size > 0
    assert result.errors == [], f"Playwright console errors for '{layout.name}': {result.errors}"

    img = Image.open(out_path)
    assert img.size == (512, 512), f"Expected (512, 512), got {img.size}"
    assert img.format == "PNG"


# ---------------------------------------------------------------------------
# 2. Registry functions
# ---------------------------------------------------------------------------

def test_all_expected_layouts_registered():
    names = {l.name for l in all_layouts()}
    assert EXPECTED_LAYOUT_NAMES == names


def test_get_layout_known():
    layout = get_layout("solo_headline")
    assert layout.name == "solo_headline"
    assert layout.difficulty == "easy"


def test_get_layout_unknown():
    with pytest.raises(KeyError, match="Unknown layout"):
        get_layout("does_not_exist")


def test_get_layouts_for_edit_color():
    layouts = get_layouts_for_edit("color")
    # Every layout supports color edits
    assert len(layouts) == len(all_layouts())
    assert all("color" in l.supported_edits for l in layouts)


def test_get_layouts_for_edit_rotation():
    layouts = get_layouts_for_edit("rotation")
    names = {l.name for l in layouts}
    # Only layouts that explicitly declare rotation support
    assert "solo_headline" in names
    assert "title_byline" in names
    assert "two_column_heading" in names
    # Layouts without rotation should not appear
    assert "title_subtitle" not in names
    assert "header_body" not in names


def test_get_layouts_for_edit_alignment():
    layouts = get_layouts_for_edit("alignment")
    names = {l.name for l in layouts}
    assert "solo_headline" in names
    assert "title_byline" in names
    assert "corner_badge" in names
    assert "split_panel" in names
    assert "banner_caption" in names
    # Layouts without alignment should not appear
    assert "title_subtitle" not in names
    assert "name_card" not in names


def test_get_layouts_by_difficulty():
    easy = get_layouts_by_difficulty("easy")
    assert len(easy) >= 1
    assert all(l.difficulty == "easy" for l in easy)

    hard = get_layouts_by_difficulty("hard")
    assert len(hard) == 1
    assert hard[0].name == "low_contrast_centered"

    multi = get_layouts_by_difficulty("multi")
    assert len(multi) == 1
    assert multi[0].name == "two_column_heading"


# ---------------------------------------------------------------------------
# 3. Internal consistency of each layout definition
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("layout", all_layouts(), ids=lambda l: l.name)
def test_layout_roles_consistent(layout):
    """roles, role_constraints, role_base_styles, and default_content must all share the same keys."""
    assert set(layout.roles) == set(layout.role_constraints.keys()), (
        f"'{layout.name}': roles vs role_constraints mismatch"
    )
    assert set(layout.roles) == set(layout.role_base_styles.keys()), (
        f"'{layout.name}': roles vs role_base_styles mismatch"
    )
    assert set(layout.roles) == set(layout.default_content.keys()), (
        f"'{layout.name}': roles vs default_content mismatch"
    )


@pytest.mark.parametrize("layout", all_layouts(), ids=lambda l: l.name)
def test_role_base_styles_have_required_keys(layout):
    """Every role style dict must have the five required CSS keys."""
    required = {"color", "font_size_px", "font_family", "font_weight", "font_style"}
    for role, style in layout.role_base_styles.items():
        missing = required - set(style.keys())
        assert not missing, f"'{layout.name}'.{role} missing style keys: {missing}"


@pytest.mark.parametrize("layout", all_layouts(), ids=lambda l: l.name)
def test_supported_edits_backed_by_constraints(layout):
    """
    If a layout declares 'rotation', at least one role must have can_rotate=True.
    If it declares 'alignment', at least one role must have can_align=True.
    """
    if "rotation" in layout.supported_edits:
        assert any(rc.can_rotate for rc in layout.role_constraints.values()), (
            f"'{layout.name}' declares 'rotation' but no role has can_rotate=True"
        )
    if "alignment" in layout.supported_edits:
        assert any(rc.can_align for rc in layout.role_constraints.values()), (
            f"'{layout.name}' declares 'alignment' but no role has can_align=True"
        )


@pytest.mark.parametrize("layout", all_layouts(), ids=lambda l: l.name)
def test_default_content_non_empty(layout):
    for role, text in layout.default_content.items():
        assert text, f"'{layout.name}'.{role} default_content is empty"


# ---------------------------------------------------------------------------
# 4. Style override produces different HTML
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("layout", all_layouts(), ids=lambda l: l.name)
def test_style_override_changes_html(layout):
    """Changing one style property must produce different HTML output."""
    first_role = layout.roles[0]
    base_html = layout.html_builder(layout.default_content, layout.role_base_styles, layout.background)

    modified_styles = copy.deepcopy(layout.role_base_styles)
    original_color = modified_styles[first_role]["color"]
    modified_styles[first_role]["color"] = "#FF0000" if original_color != "#FF0000" else "#0000FF"

    modified_html = layout.html_builder(layout.default_content, modified_styles, layout.background)

    assert base_html != modified_html, (
        f"'{layout.name}': changing color of '{first_role}' produced identical HTML"
    )


# ---------------------------------------------------------------------------
# 5. Alignment positions — each declared position renders without crash
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("layout", all_layouts(), ids=lambda l: l.name)
def test_alignment_positions_render(layout):
    """For every role with can_align=True, each declared position must render without error."""
    for role, constraints in layout.role_constraints.items():
        if not constraints.can_align:
            continue
        for position in constraints.alignment_positions:
            styles = copy.deepcopy(layout.role_base_styles)
            styles[role]["alignment"] = position
            html = layout.html_builder(layout.default_content, styles, layout.background)
            assert isinstance(html, str) and len(html) > 100, (
                f"'{layout.name}'.{role} alignment='{position}' returned empty/short HTML"
            )
            # We only check the HTML string here — full render is covered by test_layout_renders
