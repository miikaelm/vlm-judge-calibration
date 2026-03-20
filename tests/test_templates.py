"""
test_templates.py — Milestone 1 verification.

Renders all 5 templates and asserts:
  - Output file exists and is a valid PNG
  - Image dimensions are 512×512 (downscaled from 1024×1024)
  - No Playwright console errors were emitted
  - get_template() and get_templates_by_difficulty() work correctly

Run:
    pytest tests/test_templates.py -v
"""

import sys
from pathlib import Path

import pytest
from PIL import Image

# Make src importable when running from repo root
sys.path.insert(0, str(Path(__file__).parents[1] / "src"))

from render import render_html_sync, RenderConfig
from templates.templates import (
    ALL_TEMPLATES,
    get_template,
    get_templates_by_difficulty,
)

RENDER_CFG = RenderConfig(width=1024, height=1024, downscale_to=512)
OUT_DIR = Path(__file__).parents[1] / "data" / "test_renders"


@pytest.fixture(scope="session", autouse=True)
def make_output_dir():
    OUT_DIR.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# Parametrized rendering test — one case per template
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("template", ALL_TEMPLATES, ids=lambda t: t.template_id)
def test_template_renders(template):
    """Each template should render to a 512×512 PNG with no console errors."""
    out_path = OUT_DIR / f"{template.template_id}.png"
    html = template.render_html(None)

    assert isinstance(html, str) and len(html) > 100, "render_html returned empty/short string"

    result = render_html_sync(html, out_path, config=RENDER_CFG)

    assert out_path.exists(), f"Output file not created: {out_path}"
    assert out_path.stat().st_size > 0, "Output file is empty"
    assert result.errors == [], f"Playwright console errors: {result.errors}"

    img = Image.open(out_path)
    assert img.size == (512, 512), f"Expected (512, 512), got {img.size}"
    assert img.format == "PNG"


# ---------------------------------------------------------------------------
# Text override test — verify text_content dict is applied
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("template", ALL_TEMPLATES, ids=lambda t: t.template_id)
def test_template_text_override(template):
    """Overriding text_content should change the rendered HTML string."""
    first_element = template.elements[0]
    override = {first_element.element_id: "OVERRIDE_TEXT_XYZ"}

    html_default = template.render_html(None)
    html_override = template.render_html(override)

    assert "OVERRIDE_TEXT_XYZ" in html_override, "Override text not found in HTML"
    assert "OVERRIDE_TEXT_XYZ" not in html_default, "Override text leaked into default HTML"


# ---------------------------------------------------------------------------
# Registry tests
# ---------------------------------------------------------------------------

def test_get_template_known():
    t = get_template("easy_single")
    assert t.template_id == "easy_single"
    assert t.difficulty == "easy"


def test_get_template_unknown():
    with pytest.raises(KeyError, match="Unknown template_id"):
        get_template("does_not_exist")


def test_get_templates_by_difficulty():
    easy = get_templates_by_difficulty("easy")
    assert len(easy) == 2
    assert all(t.difficulty == "easy" for t in easy)

    hard = get_templates_by_difficulty("hard")
    assert len(hard) == 1
    assert hard[0].template_id == "hard_low_contrast"

    multi = get_templates_by_difficulty("multi")
    assert len(multi) == 1
    assert multi[0].template_id == "medium_two_col"


def test_all_templates_have_elements():
    for t in ALL_TEMPLATES:
        assert len(t.elements) >= 1, f"{t.template_id} has no elements"
        assert t.num_elements == len(t.elements), (
            f"{t.template_id}: num_elements={t.num_elements} but {len(t.elements)} elements defined"
        )
        for el in t.elements:
            assert el.element_id, "element_id must be non-empty"
            assert el.default_text, "default_text must be non-empty"


def test_element_ids_appear_in_html():
    """Every element_id should appear as an HTML id attribute in the rendered HTML."""
    for t in ALL_TEMPLATES:
        html = t.render_html(None)
        for el in t.elements:
            assert f'id="{el.element_id}"' in html, (
                f"Template '{t.template_id}': element_id '{el.element_id}' "
                f"not found as id attribute in rendered HTML"
            )
