"""
layouts/definitions.py — All layout builder functions and register() calls.

Ported from quantitative-text-editing/src/gen_pipeline/layouts.py.
Adaptations from QTE:
  - html_builder signature is (contents, styles, bg) -> str  (single render, not a pair)
  - No jitter — padding/gap values are fixed; visual variety comes from style-dict overrides
  - role_base_styles include 'color' and 'font_family' (no separate style/color pools)
  - palette_slots replaced by color_editable: bool on RoleConstraints

Each layout builder reads styles[role_name] for CSS properties and contents[role_name] for text.
Callers (pipeline, degradation engine) modify the styles dict before passing it in.
"""

from __future__ import annotations

from .core import LayoutDefinition, RoleConstraints, _role_css, register


# ---------------------------------------------------------------------------
# Alignment position CSS lookup tables (shared across layouts)
# ---------------------------------------------------------------------------

_SOLO_POS_CSS: dict[str, str] = {
    "top-left":      "justify-content:flex-start; align-items:flex-start; padding:80px 0 0 60px",
    "top-center":    "justify-content:center;     align-items:flex-start; padding-top:80px",
    "top-right":     "justify-content:flex-end;   align-items:flex-start; padding:80px 60px 0 0",
    "center-left":   "justify-content:flex-start; align-items:center;     padding-left:60px",
    "center":        "justify-content:center;     align-items:center",
    "center-right":  "justify-content:flex-end;   align-items:center;     padding-right:60px",
    "bottom-left":   "justify-content:flex-start; align-items:flex-end;   padding:0 0 80px 60px",
    "bottom-center": "justify-content:center;     align-items:flex-end;   padding-bottom:80px",
    "bottom-right":  "justify-content:flex-end;   align-items:flex-end;   padding:0 60px 80px 0",
}

_BYLINE_POS_CSS: dict[str, str] = {
    "bottom-left":   "bottom:24px; left:24px",
    "bottom-center": "bottom:24px; left:0; right:0; text-align:center",
    "bottom-right":  "bottom:24px; right:24px",
}

_DESCRIPTOR_ALIGN_CSS: dict[str, str] = {
    "top":    "justify-content:flex-start; align-items:flex-start",
    "center": "justify-content:flex-start; align-items:center",
    "bottom": "justify-content:flex-start; align-items:flex-end",
}

_CAPTION_ALIGN_CSS: dict[str, str] = {
    "left":   "text-align:left",
    "center": "text-align:center",
    "right":  "text-align:right",
}

_BADGE_POS_CSS: dict[str, str] = {
    "top-left":     "top:24px; left:24px",
    "top-right":    "top:24px; right:24px",
    "bottom-left":  "bottom:24px; left:24px",
    "bottom-right": "bottom:24px; right:24px",
}


# ---------------------------------------------------------------------------
# Layout 1: solo_headline — single centered text element
#
# Supports: color, scale, rotation, alignment.
# The only single-role layout. Rotation is safe (no neighbouring elements).
# Alignment moves the headline to different screen regions.
# Ideal for isolating individual edit effects with zero confounds.
# Also used as the builder for low_contrast_centered (hard tier).
# ---------------------------------------------------------------------------

def _build_solo_headline(
    contents: dict[str, str],
    styles: dict[str, dict],
    bg: str,
) -> str:
    alignment = styles["headline"].get("alignment", "center")
    pos_css = _SOLO_POS_CSS[alignment]
    body = (
        f"margin:0; background:{bg}; display:flex; {pos_css};"
        "height:100vh; box-sizing:border-box;"
    )
    headline_css = _role_css(styles["headline"]) + "; text-align:center; max-width:80%"
    return (
        f'<!DOCTYPE html><html><body style="{body}">'
        f'<h1 style="{headline_css}">{contents["headline"]}</h1>'
        f'</body></html>'
    )


register(LayoutDefinition(
    name="solo_headline",
    id="sh",
    difficulty="easy",
    roles=["headline"],
    supported_edits=["color", "scale", "rotation", "alignment", "relocation", "font_weight", "italic", "letter_spacing"],
    primary_role="headline",
    role_constraints={
        "headline": RoleConstraints(
            color_editable=True,
            can_scale=True,
            can_rotate=True,
            rotation_range=(-25, 25),
            can_align=True,
            alignment_positions=[
                "top-left", "top-center", "top-right",
                "center-left", "center", "center-right",
                "bottom-left", "bottom-center", "bottom-right",
            ],
        ),
    },
    role_base_styles={
        "headline": {
            "color": "#1A1A1A",
            "font_size_px": 64,
            "font_family": "Arial, Helvetica, sans-serif",
            "font_weight": "bold",
            "font_style": "normal",
            "letter_spacing": "0.02em",
            "alignment": "center",
        },
    },
    default_content={"headline": "Sample Heading"},
    background="#FFFFFF",
    background_type="solid",
    html_builder=_build_solo_headline,
    notes="Single large heading on white. Easy tier. Supports all edit types.",
))


# ---------------------------------------------------------------------------
# Layout 2: title_subtitle — centered vertical stack
#
# Supports: color, scale, typography (font-weight/style).
# No rotation (elements too close). No alignment.
# ---------------------------------------------------------------------------

def _build_title_subtitle(
    contents: dict[str, str],
    styles: dict[str, dict],
    bg: str,
) -> str:
    body = (
        f"margin:0; background:{bg}; display:flex; flex-direction:column;"
        "justify-content:center; align-items:center; height:100vh; gap:20px;"
        "box-sizing:border-box; padding:40px;"
    )
    title_css = _role_css(styles["title"]) + "; text-align:center"
    subtitle_css = _role_css(styles["subtitle"]) + "; text-align:center"
    return (
        f'<!DOCTYPE html><html><body style="{body}">'
        f'<h1 style="{title_css}">{contents["title"]}</h1>'
        f'<h2 style="{subtitle_css}">{contents["subtitle"]}</h2>'
        f'</body></html>'
    )


register(LayoutDefinition(
    name="title_subtitle",
    id="ts",
    difficulty="easy",
    roles=["title", "subtitle"],
    supported_edits=["color", "scale", "font_weight", "italic", "letter_spacing"],
    primary_role="subtitle",  # title-type edits covered by solo_headline; this tests secondary text
    role_constraints={
        "title":    RoleConstraints(color_editable=True, can_scale=True),
        "subtitle": RoleConstraints(color_editable=True, can_scale=True),
    },
    role_base_styles={
        "title": {
            "color": "#111111",
            "font_size_px": 60,
            "font_family": "Arial, Helvetica, sans-serif",
            "font_weight": "bold",
            "font_style": "normal",
            "letter_spacing": "normal",
        },
        "subtitle": {
            "color": "#555555",
            "font_size_px": 30,
            "font_family": "Arial, Helvetica, sans-serif",
            "font_weight": "normal",
            "font_style": "normal",
            "letter_spacing": "normal",
        },
    },
    default_content={"title": "Welcome Back", "subtitle": "Your dashboard is ready to view"},
    background="#F0F0F0",
    background_type="solid",
    html_builder=_build_title_subtitle,
    notes="Centered vertical stack on light gray. Two elements. Easy tier.",
))


# ---------------------------------------------------------------------------
# Layout 3: header_body — left-aligned header + body paragraph
#
# Supports: color, scale.
# No rotation, no alignment.
# ---------------------------------------------------------------------------

def _build_header_body(
    contents: dict[str, str],
    styles: dict[str, dict],
    bg: str,
) -> str:
    body = (
        f"margin:0; background:{bg}; display:flex; flex-direction:column;"
        "justify-content:center; padding:56px; height:100vh; box-sizing:border-box;"
    )
    header_css = _role_css(styles["header"]) + "; margin-bottom:16px"
    body_css = _role_css(styles["body"]) + "; max-width:680px"
    return (
        f'<!DOCTYPE html><html><body style="{body}">'
        f'<h2 style="{header_css}">{contents["header"]}</h2>'
        f'<p style="{body_css}">{contents["body"]}</p>'
        f'</body></html>'
    )


register(LayoutDefinition(
    name="header_body",
    id="hb",
    difficulty="easy",
    roles=["header", "body"],
    supported_edits=["color", "scale", "font_weight", "italic", "letter_spacing"],
    primary_role="body",  # tests body/paragraph text color, smaller and lower contrast than a headline
    role_constraints={
        "header": RoleConstraints(color_editable=True, can_scale=True),
        "body":   RoleConstraints(color_editable=True, can_scale=True),
    },
    role_base_styles={
        "header": {
            "color": "#222222",
            "font_size_px": 48,
            "font_family": "Arial, Helvetica, sans-serif",
            "font_weight": "bold",
            "font_style": "normal",
            "letter_spacing": "normal",
        },
        "body": {
            "color": "#444444",
            "font_size_px": 28,
            "font_family": "Arial, Helvetica, sans-serif",
            "font_weight": "normal",
            "font_style": "normal",
            "letter_spacing": "normal",
            "line_height": 1.6,
        },
    },
    default_content={
        "header": "Getting Started",
        "body": "Follow these steps to configure your workspace and begin working with the platform.",
    },
    background="#FFFFFF",
    background_type="solid",
    html_builder=_build_header_body,
    notes="Left-aligned header + body paragraph on white. Easy tier.",
))


# ---------------------------------------------------------------------------
# Layout 4: title_byline — title centered, byline absolutely positioned
#
# Supports: color, scale, rotation, alignment.
# Rotation on title (±30°). Alignment on byline (bottom-left/center/right).
# ---------------------------------------------------------------------------

def _build_title_byline(
    contents: dict[str, str],
    styles: dict[str, dict],
    bg: str,
) -> str:
    body = (
        f"margin:0; background:{bg}; position:relative; height:100vh;"
        "display:flex; justify-content:center; align-items:center;"
    )
    title_css = _role_css(styles["title"]) + "; text-align:center"
    byline_alignment = styles["byline"].get("alignment", "bottom-left")
    pos_css = _BYLINE_POS_CSS[byline_alignment]
    byline_css = _role_css(styles["byline"]) + "; white-space:nowrap"
    return (
        f'<!DOCTYPE html><html><body style="{body}">'
        f'<h1 style="{title_css}">{contents["title"]}</h1>'
        f'<span style="position:absolute; {pos_css}; {byline_css}">{contents["byline"]}</span>'
        f'</body></html>'
    )


register(LayoutDefinition(
    name="title_byline",
    id="tb",
    difficulty="medium",
    roles=["title", "byline"],
    supported_edits=["color", "scale", "rotation", "alignment", "relocation"],
    primary_role="byline",  # tests absolutely-positioned small text; title-type covered by other layouts
    role_constraints={
        "title":  RoleConstraints(
            color_editable=True, can_scale=True,
            can_rotate=True, rotation_range=(-30, 30),
        ),
        "byline": RoleConstraints(
            color_editable=True, can_scale=True,
            can_align=True, alignment_positions=["bottom-left", "bottom-center", "bottom-right"],
        ),
    },
    role_base_styles={
        "title": {
            "color": "#1A1A2E",
            "font_size_px": 60,
            "font_family": "Arial, Helvetica, sans-serif",
            "font_weight": "bold",
            "font_style": "normal",
            "letter_spacing": "normal",
        },
        "byline": {
            "color": "#6C757D",
            "font_size_px": 28,
            "font_family": "Arial, Helvetica, sans-serif",
            "font_weight": "normal",
            "font_style": "normal",
            "letter_spacing": "0.05em",
            "alignment": "bottom-left",
        },
    },
    default_content={"title": "The Future of Design", "byline": "Studio Collective, 2024"},
    background="#F5F5F0",
    background_type="solid",
    html_builder=_build_title_byline,
    notes="Centered title + absolutely-positioned byline. Medium tier. Rotation on title, alignment on byline.",
))


# ---------------------------------------------------------------------------
# Layout 5: name_card — three-element centered stack
#
# Supports: color, scale.
# No rotation, no alignment.
# ---------------------------------------------------------------------------

def _build_name_card(
    contents: dict[str, str],
    styles: dict[str, dict],
    bg: str,
) -> str:
    body = (
        f"margin:0; background:{bg}; display:flex; flex-direction:column;"
        "justify-content:center; align-items:center; height:100vh; gap:10px;"
        "box-sizing:border-box; padding:40px;"
    )
    name_css = _role_css(styles["name"]) + "; text-align:center"
    job_css = _role_css(styles["job_title"]) + "; text-align:center"
    org_css = _role_css(styles["organization"]) + "; text-align:center; opacity:0.85"
    return (
        f'<!DOCTYPE html><html><body style="{body}">'
        f'<h1 style="{name_css}">{contents["name"]}</h1>'
        f'<p style="{job_css}">{contents["job_title"]}</p>'
        f'<p style="{org_css}">{contents["organization"]}</p>'
        f'</body></html>'
    )


register(LayoutDefinition(
    name="name_card",
    id="nc",
    difficulty="medium",
    roles=["name", "job_title", "organization"],
    supported_edits=["color", "scale", "font_weight"],
    primary_role="name",
    role_constraints={
        "name":         RoleConstraints(color_editable=True, can_scale=True),
        "job_title":    RoleConstraints(color_editable=True, can_scale=True),
        "organization": RoleConstraints(color_editable=True, can_scale=True),
    },
    role_base_styles={
        "name": {
            "color": "#1A1A1A",
            "font_size_px": 56,
            "font_family": "Georgia, 'Times New Roman', serif",
            "font_weight": "bold",
            "font_style": "normal",
            "letter_spacing": "normal",
        },
        "job_title": {
            "color": "#4A4A4A",
            "font_size_px": 28,
            "font_family": "Georgia, 'Times New Roman', serif",
            "font_weight": "normal",
            "font_style": "normal",
            "letter_spacing": "0.08em",
        },
        "organization": {
            "color": "#777777",
            "font_size_px": 26,
            "font_family": "Georgia, 'Times New Roman', serif",
            "font_weight": "normal",
            "font_style": "normal",
            "letter_spacing": "0.05em",
        },
    },
    default_content={
        "name": "Alexandra Chen",
        "job_title": "Senior Product Designer",
        "organization": "Meridian Systems",
    },
    background="#FAFAFA",
    background_type="solid",
    html_builder=_build_name_card,
    notes="Three-element centered stack. Medium tier. Three roles for wrong-element targeting.",
))


# ---------------------------------------------------------------------------
# Layout 6: quote_attribution — block quote with attribution line
#
# Supports: color, scale.
# Quote role defaults to italic — natural baseline for font-style edits.
# No rotation, no alignment.
# ---------------------------------------------------------------------------

def _build_quote_attribution(
    contents: dict[str, str],
    styles: dict[str, dict],
    bg: str,
) -> str:
    body = (
        f"margin:0; background:{bg}; display:flex; flex-direction:column;"
        "justify-content:center; padding:64px 80px; height:100vh;"
        "box-sizing:border-box;"
    )
    quote_css = _role_css(styles["quote"]) + "; max-width:720px"
    attr_css = _role_css(styles["attribution"]) + "; margin-top:24px"
    return (
        f'<!DOCTYPE html><html><body style="{body}">'
        f'<p style="{quote_css}">{contents["quote"]}</p>'
        f'<p style="{attr_css}">\u2014 {contents["attribution"]}</p>'
        f'</body></html>'
    )


register(LayoutDefinition(
    name="quote_attribution",
    id="qa",
    difficulty="medium",
    roles=["quote", "attribution"],
    supported_edits=["color", "scale", "italic"],
    primary_role="quote",
    role_constraints={
        "quote":       RoleConstraints(color_editable=True, can_scale=True),
        "attribution": RoleConstraints(color_editable=True, can_scale=True),
    },
    role_base_styles={
        "quote": {
            "color": "#2C2C2C",
            "font_size_px": 44,
            "font_family": "'Palatino Linotype', Palatino, 'Book Antiqua', serif",
            "font_weight": "normal",
            "font_style": "italic",
            "letter_spacing": "normal",
            "line_height": 1.5,
        },
        "attribution": {
            "color": "#888888",
            "font_size_px": 28,
            "font_family": "'Palatino Linotype', Palatino, 'Book Antiqua', serif",
            "font_weight": "bold",
            "font_style": "normal",
            "letter_spacing": "0.05em",
        },
    },
    default_content={
        "quote": "Design is not just what it looks like and feels like. Design is how it works.",
        "attribution": "Steve Jobs",
    },
    background="#F8F4EF",
    background_type="solid",
    html_builder=_build_quote_attribution,
    notes="Block quote (italic) + attribution. Medium tier. Italic default is a natural font-style edit target.",
))


# ---------------------------------------------------------------------------
# Layout 7: corner_badge — centered label with a corner-positioned badge
#
# Supports: color, scale, alignment.
# Badge alignment uses four corner positions.
# No rotation — badge text is short; rotation would be barely visible.
# ---------------------------------------------------------------------------

def _build_corner_badge(
    contents: dict[str, str],
    styles: dict[str, dict],
    bg: str,
) -> str:
    body = (
        f"margin:0; background:{bg}; position:relative; height:100vh;"
        "display:flex; justify-content:center; align-items:center;"
    )
    label_css = _role_css(styles["label"]) + "; text-align:center"
    badge_alignment = styles["badge"].get("alignment", "top-right")
    pos_css = _BADGE_POS_CSS[badge_alignment]
    badge_css = _role_css(styles["badge"]) + "; white-space:nowrap"
    return (
        f'<!DOCTYPE html><html><body style="{body}">'
        f'<h1 style="{label_css}">{contents["label"]}</h1>'
        f'<span style="position:absolute; {pos_css}; {badge_css}">{contents["badge"]}</span>'
        f'</body></html>'
    )


register(LayoutDefinition(
    name="corner_badge",
    id="cb",
    difficulty="medium",
    roles=["label", "badge"],
    supported_edits=["color", "scale", "alignment"],
    primary_role="badge",  # tests the small corner-positioned element against a dark background
    role_constraints={
        "label": RoleConstraints(color_editable=True, can_scale=True),
        "badge": RoleConstraints(
            color_editable=True, can_scale=True,
            can_align=True, alignment_positions=["top-left", "top-right", "bottom-left", "bottom-right"],
        ),
    },
    role_base_styles={
        "label": {
            "color": "#FFFFFF",
            "font_size_px": 56,
            "font_family": "'Trebuchet MS', Arial, sans-serif",
            "font_weight": "bold",
            "font_style": "normal",
            "letter_spacing": "0.04em",
        },
        "badge": {
            "color": "#FFD700",
            "font_size_px": 26,
            "font_family": "Arial, Helvetica, sans-serif",
            "font_weight": "bold",
            "font_style": "normal",
            "letter_spacing": "0.06em",
            "alignment": "top-right",
        },
    },
    default_content={"label": "Annual Report", "badge": "2024"},
    background="#1E3A5F",
    background_type="solid",
    html_builder=_build_corner_badge,
    notes="Dark blue bg. Centered label + corner badge. Medium tier. Badge alignment is the edit target.",
))


# ---------------------------------------------------------------------------
# Layout 8: split_panel — horizontal two-column layout
#
# Supports: color, scale, alignment (descriptor vertical position in right panel).
# No rotation.
# ---------------------------------------------------------------------------

def _build_split_panel(
    contents: dict[str, str],
    styles: dict[str, dict],
    bg: str,
) -> str:
    body = f"margin:0; background:{bg}; display:flex; height:100vh;"
    label_css = _role_css(styles["label"]) + "; text-align:center"
    descriptor_alignment = styles["descriptor"].get("alignment", "top")
    desc_align_css = _DESCRIPTOR_ALIGN_CSS[descriptor_alignment]
    descriptor_css = _role_css(styles["descriptor"])
    left_panel = (
        "display:flex; flex:1; justify-content:center; align-items:center;"
        "padding:40px; box-sizing:border-box;"
    )
    right_panel = (
        f"display:flex; flex:1; {desc_align_css};"
        "padding:40px; box-sizing:border-box;"
    )
    return (
        f'<!DOCTYPE html><html><body style="{body}">'
        f'<div style="{left_panel}">'
        f'<span style="{label_css}">{contents["label"]}</span>'
        f'</div>'
        f'<div style="{right_panel}">'
        f'<span style="{descriptor_css}">{contents["descriptor"]}</span>'
        f'</div>'
        f'</body></html>'
    )


register(LayoutDefinition(
    name="split_panel",
    id="sp",
    difficulty="medium",
    roles=["label", "descriptor"],
    supported_edits=["color", "scale", "alignment"],
    primary_role="label",
    role_constraints={
        "label":      RoleConstraints(color_editable=True, can_scale=True),
        "descriptor": RoleConstraints(
            color_editable=True, can_scale=True,
            can_align=True, alignment_positions=["top", "center", "bottom"],
        ),
    },
    role_base_styles={
        "label": {
            "color": "#2C2C2C",
            "font_size_px": 52,
            "font_family": "Arial, Helvetica, sans-serif",
            "font_weight": "bold",
            "font_style": "normal",
            "letter_spacing": "0.05em",
        },
        "descriptor": {
            "color": "#666666",
            "font_size_px": 28,
            "font_family": "Arial, Helvetica, sans-serif",
            "font_weight": "normal",
            "font_style": "normal",
            "letter_spacing": "normal",
            "alignment": "top",
        },
    },
    default_content={"label": "Strategy", "descriptor": "Long-term growth through focused execution and customer value."},
    background="#FFFFFF",
    background_type="solid",
    html_builder=_build_split_panel,
    notes="Two-column split. Medium tier. Descriptor vertical alignment is the edit target.",
))


# ---------------------------------------------------------------------------
# Layout 9: banner_caption — large top banner + bottom caption
#
# Supports: color, scale, alignment (caption text-align).
# No rotation.
# ---------------------------------------------------------------------------

def _build_banner_caption(
    contents: dict[str, str],
    styles: dict[str, dict],
    bg: str,
) -> str:
    body = (
        f"margin:0; background:{bg}; display:flex; flex-direction:column;"
        "justify-content:space-between; height:100vh;"
        "box-sizing:border-box; padding:60px 48px;"
    )
    banner_css = _role_css(styles["banner"]) + "; text-align:center"
    caption_alignment = styles["caption"].get("alignment", "center")
    align_css = _CAPTION_ALIGN_CSS[caption_alignment]
    caption_css = _role_css(styles["caption"]) + f"; {align_css}"
    return (
        f'<!DOCTYPE html><html><body style="{body}">'
        f'<h1 style="{banner_css}">{contents["banner"]}</h1>'
        f'<p style="{caption_css}">{contents["caption"]}</p>'
        f'</body></html>'
    )


register(LayoutDefinition(
    name="banner_caption",
    id="bc",
    difficulty="medium",
    roles=["banner", "caption"],
    supported_edits=["color", "scale", "alignment", "letter_spacing"],
    primary_role="banner",
    role_constraints={
        "banner": RoleConstraints(color_editable=True, can_scale=True),
        "caption": RoleConstraints(
            color_editable=True, can_scale=True,
            can_align=True, alignment_positions=["left", "center", "right"],
        ),
    },
    role_base_styles={
        "banner": {
            "color": "#1A1A1A",
            "font_size_px": 72,
            "font_family": "Arial, Helvetica, sans-serif",
            "font_weight": "bold",
            "font_style": "normal",
            "letter_spacing": "0.03em",
        },
        "caption": {
            "color": "#666666",
            "font_size_px": 28,
            "font_family": "Arial, Helvetica, sans-serif",
            "font_weight": "normal",
            "font_style": "normal",
            "letter_spacing": "normal",
            "alignment": "center",
        },
    },
    default_content={"banner": "Innovation Summit", "caption": "Bringing together the brightest minds in technology and design."},
    background="#FFFFFF",
    background_type="solid",
    html_builder=_build_banner_caption,
    notes="Large top banner + bottom caption. Medium tier. Caption text-align is the edit target.",
))


# ---------------------------------------------------------------------------
# Layout 10: two_column_heading — two side-by-side headings
#
# Supports: color, scale, rotation.
# Both roles can rotate independently. No alignment.
# Multi-element layout — either heading can be the edit target.
# ---------------------------------------------------------------------------

def _build_two_column_heading(
    contents: dict[str, str],
    styles: dict[str, dict],
    bg: str,
) -> str:
    body = f"margin:0; background:{bg}; display:flex; height:100vh;"
    cell = (
        "display:flex; flex:1; justify-content:center; align-items:center;"
        "padding:40px; box-sizing:border-box;"
    )
    left_css = _role_css(styles["left_heading"]) + "; text-align:center"
    right_css = _role_css(styles["right_heading"]) + "; text-align:center"
    return (
        f'<!DOCTYPE html><html><body style="{body}">'
        f'<div style="{cell}">'
        f'<span style="{left_css}">{contents["left_heading"]}</span>'
        f'</div>'
        f'<div style="{cell}">'
        f'<span style="{right_css}">{contents["right_heading"]}</span>'
        f'</div>'
        f'</body></html>'
    )


register(LayoutDefinition(
    name="two_column_heading",
    id="tch",
    difficulty="multi",
    roles=["left_heading", "right_heading"],
    supported_edits=["color", "scale", "rotation"],
    primary_role="left_heading",
    role_constraints={
        "left_heading":  RoleConstraints(
            color_editable=True, can_scale=True,
            can_rotate=True, rotation_range=(-20, 20),
        ),
        "right_heading": RoleConstraints(
            color_editable=True, can_scale=True,
            can_rotate=True, rotation_range=(-20, 20),
        ),
    },
    role_base_styles={
        "left_heading": {
            "color": "#2C1810",
            "font_size_px": 48,
            "font_family": "Georgia, 'Times New Roman', serif",
            "font_weight": "bold",
            "font_style": "normal",
            "letter_spacing": "normal",
        },
        "right_heading": {
            "color": "#2C1810",
            "font_size_px": 48,
            "font_family": "Georgia, 'Times New Roman', serif",
            "font_weight": "bold",
            "font_style": "normal",
            "letter_spacing": "normal",
        },
    },
    default_content={"left_heading": "Strategy", "right_heading": "Innovation"},
    background="#FFF8F0",
    background_type="solid",
    html_builder=_build_two_column_heading,
    notes="Two side-by-side headings. Multi tier. Both roles can be rotated — good for wrong-element targeting.",
))


# ---------------------------------------------------------------------------
# Layout 11: low_contrast_centered — single heading, dark bg, muted colors
#
# Supports: color, scale.
# Hard tier — low contrast between text and background challenges VLMs.
# Reuses _build_solo_headline (same single-role structure, different defaults).
# ---------------------------------------------------------------------------

register(LayoutDefinition(
    name="low_contrast_centered",
    id="lcc",
    difficulty="hard",
    roles=["headline"],
    supported_edits=["color", "scale"],
    primary_role="headline",
    role_constraints={
        "headline": RoleConstraints(color_editable=True, can_scale=True),
    },
    role_base_styles={
        "headline": {
            "color": "#8A8098",
            "font_size_px": 32,
            "font_family": "'Palatino Linotype', Palatino, 'Book Antiqua', serif",
            "font_weight": "normal",
            "font_style": "normal",
            "letter_spacing": "0.15em",
            "alignment": "center",
        },
    },
    default_content={"headline": "Limited Edition"},
    background="linear-gradient(145deg, #2A2A35 0%, #1A1A22 50%, #252530 100%)",
    background_type="gradient",
    html_builder=_build_solo_headline,
    notes="Dark gradient bg, muted purple serif, low contrast. Hard tier. Challenges VLMs on color perception.",
))


# ---------------------------------------------------------------------------
# Layout 12: gradient_headline — single heading on a vibrant gradient bg
#
# Supports: color, scale, rotation, alignment.
# Reuses _build_solo_headline. Easy tier with a gradient background.
# Tests VLM color perception against a non-white, non-solid backdrop.
# ---------------------------------------------------------------------------

register(LayoutDefinition(
    name="gradient_headline",
    id="gh",
    difficulty="easy",
    roles=["headline"],
    supported_edits=["color", "scale", "rotation", "alignment", "relocation"],
    primary_role="headline",
    role_constraints={
        "headline": RoleConstraints(
            color_editable=True,
            can_scale=True,
            can_rotate=True,
            rotation_range=(-25, 25),
            can_align=True,
            alignment_positions=[
                "top-left", "top-center", "top-right",
                "center-left", "center", "center-right",
                "bottom-left", "bottom-center", "bottom-right",
            ],
        ),
    },
    role_base_styles={
        "headline": {
            "color": "#FFFFFF",
            "font_size_px": 64,
            "font_family": "Arial, Helvetica, sans-serif",
            "font_weight": "bold",
            "font_style": "normal",
            "letter_spacing": "0.02em",
            "alignment": "center",
        },
    },
    default_content={"headline": "Sample Heading"},
    background="linear-gradient(135deg, #667eea 0%, #764ba2 100%)",
    background_type="gradient",
    html_builder=_build_solo_headline,
    notes="White headline on purple-blue gradient. Easy tier. Tests color against gradient bg.",
))


# ---------------------------------------------------------------------------
# Layout 13: pattern_headline — single heading on a grid-pattern bg
#
# Supports: color, scale, rotation, alignment.
# Reuses _build_solo_headline. Easy tier with a repeating CSS grid pattern.
# Tests VLM color perception against a textured (non-uniform) backdrop.
# ---------------------------------------------------------------------------

register(LayoutDefinition(
    name="pattern_headline",
    id="ph",
    difficulty="easy",
    roles=["headline"],
    supported_edits=["color", "scale", "rotation", "alignment", "relocation"],
    primary_role="headline",
    role_constraints={
        "headline": RoleConstraints(
            color_editable=True,
            can_scale=True,
            can_rotate=True,
            rotation_range=(-25, 25),
            can_align=True,
            alignment_positions=[
                "top-left", "top-center", "top-right",
                "center-left", "center", "center-right",
                "bottom-left", "bottom-center", "bottom-right",
            ],
        ),
    },
    role_base_styles={
        "headline": {
            "color": "#1A1A1A",
            "font_size_px": 64,
            "font_family": "Arial, Helvetica, sans-serif",
            "font_weight": "bold",
            "font_style": "normal",
            "letter_spacing": "0.02em",
            "alignment": "center",
        },
    },
    default_content={"headline": "Sample Heading"},
    background=(
        "radial-gradient(circle, rgba(80,40,120,0.22) 5px, transparent 5px) 0 0 / 32px 32px, "
        "radial-gradient(circle, rgba(80,40,120,0.10) 5px, transparent 5px) 16px 16px / 32px 32px, "
        "#f0ede8"
    ),
    background_type="pattern",
    html_builder=_build_solo_headline,
    notes="Dark headline on offset polka-dot pattern (warm off-white, purple dots). Easy tier. Tests color against patterned bg.",
))


# ---------------------------------------------------------------------------
# Layout 14: image_bg_headline — single heading on a photo-like complex bg
#
# Supports: color, scale, rotation, alignment.
# Reuses _build_solo_headline. Medium tier — complex multi-gradient bg simulates
# a natural scene (sky-to-ground), creating more visual noise around the text.
# ---------------------------------------------------------------------------

register(LayoutDefinition(
    name="image_bg_headline",
    id="ibh",
    difficulty="medium",
    roles=["headline"],
    supported_edits=["color", "scale", "rotation", "alignment"],
    primary_role="headline",
    role_constraints={
        "headline": RoleConstraints(
            color_editable=True,
            can_scale=True,
            can_rotate=True,
            rotation_range=(-25, 25),
            can_align=True,
            alignment_positions=[
                "top-left", "top-center", "top-right",
                "center-left", "center", "center-right",
                "bottom-left", "bottom-center", "bottom-right",
            ],
        ),
    },
    role_base_styles={
        "headline": {
            "color": "#FFFFFF",
            "font_size_px": 64,
            "font_family": "Arial, Helvetica, sans-serif",
            "font_weight": "bold",
            "font_style": "normal",
            "letter_spacing": "0.02em",
            "alignment": "center",
            "text_shadow": "0 2px 8px rgba(0,0,0,0.6)",
        },
    },
    default_content={"headline": "Sample Heading"},
    background=(
        "linear-gradient(rgba(17,201,111,0.29), rgba(17,201,111,0.29)), "
        "url('https://picsum.photos/seed/2250/800/600') center/cover"
    ),
    background_type="image",
    html_builder=_build_solo_headline,
    notes=(
        "White headline (with shadow) on real photo (picsum seed 2250) with green tint overlay. "
        "Medium tier. Tests color perception against a photographic background."
    ),
))


# ---------------------------------------------------------------------------
# Fixed layout sets: 1 or more layouts per (edit_type, difficulty) cell.
# ---------------------------------------------------------------------------

from .core import LAYOUT_SETS, validate_layout_sets  # noqa: E402

LAYOUT_SETS.update({
    # Color: 3 easy (solid / gradient / pattern bg) + 1 medium (image-like bg) = 4 total
    ("color", "easy"):   ["solo_headline", "gradient_headline", "pattern_headline"],
    ("color", "medium"): ["image_bg_headline"],

    # Scale: 3 easy (simple single/two-role layouts) + 1 medium (multi-role with context)
    ("scale", "easy"):   ["solo_headline", "title_subtitle", "header_body"],
    ("scale", "medium"): ["corner_badge",  "name_card",      "title_byline"],

    # Relocation (alignment position change): 3 easy (9-position grid) + 1 medium (3-position byline)
    ("relocation", "easy"):   ["solo_headline", "gradient_headline", "pattern_headline"],
    ("relocation", "medium"): ["title_byline"],

    # Font weight (100–900 scale): 3 easy (simple layouts) + 1 medium (serif multi-role)
    ("font_weight", "easy"):   ["solo_headline", "title_subtitle", "header_body"],
    ("font_weight", "medium"): ["name_card"],

    # Italic (font-style normal ↔ italic): 3 easy + 1 medium (quote has italic baseline)
    ("italic", "easy"):   ["solo_headline", "title_subtitle", "header_body"],
    ("italic", "medium"): ["quote_attribution"],

    # Letter spacing (tracking in px): 3 easy + 1 medium (large banner makes tracking visible)
    ("letter_spacing", "easy"):   ["solo_headline", "title_subtitle", "header_body"],
    ("letter_spacing", "medium"): ["banner_caption"],
})

validate_layout_sets()
