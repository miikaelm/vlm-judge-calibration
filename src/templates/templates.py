"""
templates.py — Parameterizable HTML/CSS templates for calibration stimuli.

Each Template renders a self-contained 1024×1024 HTML page with one or more
text elements. Elements are addressable by `id` attribute so the degradation
engine can inject targeted CSS overrides.

The render_html callable signature:
    render_html(text_content: dict[str, str] | None = None) -> str
where text_content optionally overrides the default text per element_id.

Difficulty tiers:
    easy   — large text (36–48px+), high contrast, solid white/light background
    medium — moderate text (20–28px), colored backgrounds, moderate contrast
    hard   — small text (12–16px), styled font, low contrast or complex background
    multi  — 3+ text elements of varying properties (for wrong-element targeting)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable


@dataclass
class TextElement:
    element_id: str      # matches the HTML `id` attribute — used by degradation engine
    default_text: str
    role: str            # semantic label: "heading", "subtext", "badge", "caption", etc.


@dataclass
class Template:
    template_id: str
    difficulty: str      # "easy" | "medium" | "hard" | "multi"
    elements: list[TextElement]
    render_html: Callable[[dict[str, str] | None], str]
    # Metadata recorded in stimulus metadata.json
    background_type: str = "solid"        # "solid" | "gradient" | "pattern"
    num_elements: int = 1
    font_size_range_px: tuple[int, int] = (36, 64)
    notes: str = ""


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def _text(overrides: dict[str, str] | None, element_id: str, default: str) -> str:
    if overrides and element_id in overrides:
        return overrides[element_id]
    return default


# ---------------------------------------------------------------------------
# Template 1 — easy_single
# Single large heading, white background, high contrast.
# ---------------------------------------------------------------------------

def _render_easy_single(text_content: dict[str, str] | None = None) -> str:
    heading = _text(text_content, "heading", "Sample Heading")
    return f"""<!DOCTYPE html>
<html><head><meta charset="UTF-8"><style>
* {{ margin: 0; padding: 0; box-sizing: border-box; }}
body {{
    width: 1024px; height: 1024px;
    background: #FFFFFF;
    display: flex; justify-content: center; align-items: center;
}}
#heading {{
    font-family: Arial, Helvetica, sans-serif;
    font-size: 64px;
    font-weight: bold;
    color: #1A1A1A;
    text-align: center;
    letter-spacing: 0.02em;
}}
</style></head><body>
<h1 id="heading">{heading}</h1>
</body></html>"""


EASY_SINGLE = Template(
    template_id="easy_single",
    difficulty="easy",
    elements=[TextElement("heading", "Sample Heading", "heading")],
    render_html=_render_easy_single,
    background_type="solid",
    num_elements=1,
    font_size_range_px=(64, 64),
    notes="Large bold heading centered on white background.",
)


# ---------------------------------------------------------------------------
# Template 2 — easy_card
# Heading + subtext on a light gray card, centered on white canvas.
# ---------------------------------------------------------------------------

def _render_easy_card(text_content: dict[str, str] | None = None) -> str:
    heading = _text(text_content, "heading", "Welcome Back")
    subtext = _text(text_content, "subtext", "Your dashboard is ready to view")
    return f"""<!DOCTYPE html>
<html><head><meta charset="UTF-8"><style>
* {{ margin: 0; padding: 0; box-sizing: border-box; }}
body {{
    width: 1024px; height: 1024px;
    background: #F0F0F0;
    display: flex; justify-content: center; align-items: center;
}}
.card {{
    background: #FFFFFF;
    border-radius: 16px;
    padding: 80px 100px;
    box-shadow: 0 4px 24px rgba(0,0,0,0.08);
    text-align: center;
    max-width: 760px;
}}
#heading {{
    font-family: Arial, Helvetica, sans-serif;
    font-size: 48px;
    font-weight: bold;
    color: #111111;
    margin-bottom: 24px;
}}
#subtext {{
    font-family: Arial, Helvetica, sans-serif;
    font-size: 24px;
    font-weight: normal;
    color: #555555;
}}
</style></head><body>
<div class="card">
    <h1 id="heading">{heading}</h1>
    <p id="subtext">{subtext}</p>
</div>
</body></html>"""


EASY_CARD = Template(
    template_id="easy_card",
    difficulty="easy",
    elements=[
        TextElement("heading", "Welcome Back", "heading"),
        TextElement("subtext", "Your dashboard is ready to view", "subtext"),
    ],
    render_html=_render_easy_card,
    background_type="solid",
    num_elements=2,
    font_size_range_px=(24, 48),
    notes="Card layout with heading and subtext on white card, light gray canvas.",
)


# ---------------------------------------------------------------------------
# Template 3 — medium_badge
# Centered heading on a blue background with a small corner badge.
# ---------------------------------------------------------------------------

def _render_medium_badge(text_content: dict[str, str] | None = None) -> str:
    heading = _text(text_content, "heading", "Annual Report")
    badge = _text(text_content, "badge", "2024")
    return f"""<!DOCTYPE html>
<html><head><meta charset="UTF-8"><style>
* {{ margin: 0; padding: 0; box-sizing: border-box; }}
body {{
    width: 1024px; height: 1024px;
    background: #1E3A5F;
    display: flex; justify-content: center; align-items: center;
    position: relative;
}}
#heading {{
    font-family: 'Trebuchet MS', Arial, sans-serif;
    font-size: 56px;
    font-weight: bold;
    color: #FFFFFF;
    text-align: center;
    letter-spacing: 0.04em;
    text-transform: uppercase;
}}
#badge {{
    position: absolute;
    top: 48px;
    right: 56px;
    font-family: Arial, Helvetica, sans-serif;
    font-size: 22px;
    font-weight: bold;
    color: #FFD700;
    background: rgba(255,255,255,0.12);
    padding: 8px 18px;
    border-radius: 8px;
    letter-spacing: 0.06em;
}}
</style></head><body>
<h1 id="heading">{heading}</h1>
<span id="badge">{badge}</span>
</body></html>"""


MEDIUM_BADGE = Template(
    template_id="medium_badge",
    difficulty="medium",
    elements=[
        TextElement("heading", "Annual Report", "heading"),
        TextElement("badge", "2024", "badge"),
    ],
    render_html=_render_medium_badge,
    background_type="solid",
    num_elements=2,
    font_size_range_px=(22, 56),
    notes="Dark blue background. Main heading centered, year badge in top-right corner.",
)


# ---------------------------------------------------------------------------
# Template 4 — medium_two_col
# Two-column layout: each column has a heading and body text. 4 elements total.
# Designed for wrong-element targeting tests.
# ---------------------------------------------------------------------------

def _render_medium_two_col(text_content: dict[str, str] | None = None) -> str:
    left_heading  = _text(text_content, "left_heading",  "Strategy")
    left_body     = _text(text_content, "left_body",     "Long-term growth through focused execution and customer value.")
    right_heading = _text(text_content, "right_heading", "Innovation")
    right_body    = _text(text_content, "right_body",    "Investing in research to build the next generation of products.")
    return f"""<!DOCTYPE html>
<html><head><meta charset="UTF-8"><style>
* {{ margin: 0; padding: 0; box-sizing: border-box; }}
body {{
    width: 1024px; height: 1024px;
    background: #FFF8F0;
    display: flex; justify-content: center; align-items: center;
}}
.container {{
    display: flex;
    gap: 0;
    width: 900px;
}}
.col {{
    flex: 1;
    padding: 60px 48px;
}}
.col:first-child {{
    border-right: 2px solid #D0C0B0;
}}
.col-heading {{
    font-family: Georgia, 'Times New Roman', serif;
    font-size: 36px;
    font-weight: bold;
    color: #2C1810;
    margin-bottom: 20px;
}}
.col-body {{
    font-family: Georgia, 'Times New Roman', serif;
    font-size: 20px;
    font-weight: normal;
    color: #5C4030;
    line-height: 1.6;
}}
</style></head><body>
<div class="container">
    <div class="col">
        <p id="left_heading" class="col-heading">{left_heading}</p>
        <p id="left_body" class="col-body">{left_body}</p>
    </div>
    <div class="col">
        <p id="right_heading" class="col-heading">{right_heading}</p>
        <p id="right_body" class="col-body">{right_body}</p>
    </div>
</div>
</body></html>"""


MEDIUM_TWO_COL = Template(
    template_id="medium_two_col",
    difficulty="multi",
    elements=[
        TextElement("left_heading",  "Strategy",   "heading"),
        TextElement("left_body",     "Long-term growth through focused execution and customer value.", "body"),
        TextElement("right_heading", "Innovation", "heading"),
        TextElement("right_body",    "Investing in research to build the next generation of products.", "body"),
    ],
    render_html=_render_medium_two_col,
    background_type="solid",
    num_elements=4,
    font_size_range_px=(20, 36),
    notes="Two-column layout on warm off-white. 4 elements for wrong-element targeting.",
)


# ---------------------------------------------------------------------------
# Template 5 — hard_low_contrast
# Small text on a dark gradient background, low contrast.
# Challenges VLMs that struggle with small / low-contrast text.
# ---------------------------------------------------------------------------

def _render_hard_low_contrast(text_content: dict[str, str] | None = None) -> str:
    heading = _text(text_content, "heading", "Limited Edition")
    caption = _text(text_content, "caption", "Exclusively crafted for discerning collectors")
    return f"""<!DOCTYPE html>
<html><head><meta charset="UTF-8"><style>
* {{ margin: 0; padding: 0; box-sizing: border-box; }}
body {{
    width: 1024px; height: 1024px;
    background: linear-gradient(145deg, #2A2A35 0%, #1A1A22 50%, #252530 100%);
    display: flex;
    flex-direction: column;
    justify-content: center;
    align-items: center;
    gap: 20px;
}}
#heading {{
    font-family: 'Palatino Linotype', Palatino, 'Book Antiqua', serif;
    font-size: 28px;
    font-weight: normal;
    color: #8A8098;
    letter-spacing: 0.18em;
    text-transform: uppercase;
    text-align: center;
}}
#caption {{
    font-family: 'Palatino Linotype', Palatino, 'Book Antiqua', serif;
    font-size: 14px;
    font-weight: normal;
    color: #6A6075;
    letter-spacing: 0.08em;
    text-align: center;
    font-style: italic;
}}
</style></head><body>
<h2 id="heading">{heading}</h2>
<p id="caption">{caption}</p>
</body></html>"""


HARD_LOW_CONTRAST = Template(
    template_id="hard_low_contrast",
    difficulty="hard",
    elements=[
        TextElement("heading", "Limited Edition", "heading"),
        TextElement("caption", "Exclusively crafted for discerning collectors", "caption"),
    ],
    render_html=_render_hard_low_contrast,
    background_type="gradient",
    num_elements=2,
    font_size_range_px=(14, 28),
    notes="Dark gradient bg, muted purple text, serif font, low contrast. Hard tier.",
)


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

ALL_TEMPLATES: list[Template] = [
    EASY_SINGLE,
    EASY_CARD,
    MEDIUM_BADGE,
    MEDIUM_TWO_COL,
    HARD_LOW_CONTRAST,
]

_REGISTRY: dict[str, Template] = {t.template_id: t for t in ALL_TEMPLATES}


def get_template(template_id: str) -> Template:
    if template_id not in _REGISTRY:
        raise KeyError(f"Unknown template_id: {template_id!r}. Available: {list(_REGISTRY)}")
    return _REGISTRY[template_id]


def get_templates_by_difficulty(difficulty: str) -> list[Template]:
    return [t for t in ALL_TEMPLATES if t.difficulty == difficulty]
