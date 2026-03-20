"""
engine.py — Degradation engine for injecting controlled errors into HTML.

Each degradation function injects a CSS override (or modifies HTML text)
to produce a specific type and magnitude of error for calibration stimuli.
"""

import re
from pathlib import Path

import yaml

from src.degradation.color_utils import offset_color_by_delta_e
from src.degradation.specs import DegradationSpec


def _inject_css_override(html: str, element_id: str, css_properties: str) -> str:
    """Inject a CSS override block before </head> in html."""
    override = f"\n<style>\n#{element_id} {{ {css_properties} }}\n</style>\n"
    if "</head>" in html:
        return html.replace("</head>", override + "</head>", 1)
    elif "</body>" in html:
        return html.replace("</body>", override + "</body>", 1)
    return html + override


def _extract_color(html: str, element_id: str) -> str | None:
    """
    Extract the color property for a given element_id from the HTML's <style> blocks.
    Returns hex string or None if not found.
    """
    style_blocks = re.findall(r"<style[^>]*>(.*?)</style>", html, re.DOTALL | re.IGNORECASE)
    pattern = re.compile(
        r"#" + re.escape(element_id) + r"\s*\{([^}]*)\}", re.DOTALL
    )
    for block in style_blocks:
        m = pattern.search(block)
        if m:
            color_m = re.search(r"(?<!\S)color:\s*(#[0-9A-Fa-f]{6})", m.group(1))
            if color_m:
                return color_m.group(1)
    return None


def degrade_color(base_html: str, spec: DegradationSpec) -> str:
    """
    Inject a color override that shifts the element's color by target ΔE.

    spec.params must contain:
        delta_e (float): target ΔE CIEDE2000 shift
    spec.params may contain:
        base_color (str): hex color of the correct color — if omitted, parsed from HTML
    """
    base_color = spec.params.get("base_color") or _extract_color(base_html, spec.element_id) or "#000000"
    delta_e = spec.params["delta_e"]
    new_color = offset_color_by_delta_e(base_color, delta_e)
    # Audit: store computed color back into params
    spec.params["computed_color"] = new_color
    css = f"color: {new_color} !important;"
    return _inject_css_override(base_html, spec.element_id, css)


def degrade_position(base_html: str, spec: DegradationSpec) -> str:
    """
    Shift the element's visual position by offset_x_px / offset_y_px using CSS transform.

    Using transform: translate() means the offset is always relative to the element's
    current rendered position — whether it's flexbox-centered, statically placed, or
    absolutely positioned. This avoids the element jumping to the origin.

    spec.params must contain:
        offset_x_px (int): horizontal offset in pixels (signed)
        offset_y_px (int): vertical offset in pixels (signed)
    """
    offset_x = spec.params["offset_x_px"]
    offset_y = spec.params["offset_y_px"]
    css = f"transform: translate({offset_x}px, {offset_y}px) !important;"
    return _inject_css_override(base_html, spec.element_id, css)


def _extract_font_size(html: str, element_id: str) -> int | None:
    """
    Extract the font-size (in px) for a given element_id from the HTML's <style> blocks.
    Returns None if not found.
    """
    # Match #element_id { ... font-size: Xpx ... } in any style block
    style_blocks = re.findall(r"<style[^>]*>(.*?)</style>", html, re.DOTALL | re.IGNORECASE)
    pattern = re.compile(
        r"#" + re.escape(element_id) + r"\s*\{([^}]*)\}", re.DOTALL
    )
    for block in style_blocks:
        m = pattern.search(block)
        if m:
            fs = re.search(r"font-size:\s*(\d+(?:\.\d+)?)px", m.group(1))
            if fs:
                return int(float(fs.group(1)))
    return None


def degrade_scale(base_html: str, spec: DegradationSpec) -> str:
    """
    Inject a font-size override scaled by scale_error_pct percent.

    spec.params must contain:
        scale_error_pct (float): percent change, e.g. 20 means 20% larger
    spec.params may contain:
        base_font_size_px (int): original font size in px — if omitted, parsed from HTML
    """
    base_size = spec.params.get("base_font_size_px") or _extract_font_size(base_html, spec.element_id) or 36
    error_pct = spec.params["scale_error_pct"]
    new_size = base_size * (1 + error_pct / 100)
    # Round to nearest integer for clean CSS
    new_size_px = round(new_size)
    css = f"font-size: {new_size_px}px !important;"
    return _inject_css_override(base_html, spec.element_id, css)


def _load_confusables() -> dict:
    """Load confusables table from configs/confusables.yaml."""
    configs_dir = Path(__file__).parents[2] / "configs"
    confusables_path = configs_dir / "confusables.yaml"
    if not confusables_path.exists():
        return {}
    with open(confusables_path, encoding="utf-8") as f:
        data = yaml.safe_load(f)
    return data.get("confusables", {})


def degrade_char_substitution(base_html: str, spec: DegradationSpec) -> str:
    """
    Replace characters in the text content of the target element.

    spec.params may contain:
        substitutions (list): list of {"position": int, "original": str, "replacement": str}
        num_substitutions (int): if substitutions not provided, pick this many from confusables
        tier (str): confusable tier label (unused for now, reserved)

    If substitutions not provided, loads confusables from configs/confusables.yaml.
    """
    element_id = spec.element_id
    substitutions = spec.params.get("substitutions")

    # Find the element text via regex: id="element_id" ... >TEXT<
    # Pattern handles both quote styles and attributes between id and >
    pattern = re.compile(
        r'(id=["\']' + re.escape(element_id) + r'["\'][^>]*>)([^<]*)',
        re.DOTALL,
    )
    match = pattern.search(base_html)
    if not match:
        # Element not found — return unchanged
        return base_html

    original_text = match.group(2)

    if substitutions:
        # Apply explicit substitutions
        chars = list(original_text)
        for sub in substitutions:
            pos = sub["position"]
            if pos < len(chars) and chars[pos] == sub["original"]:
                chars[pos] = sub["replacement"]
            elif pos < len(chars):
                # Position mismatch — replace anyway to apply the degradation
                chars[pos] = sub["replacement"]
        new_text = "".join(chars)
    else:
        # Auto-generate substitutions from confusables
        confusables = _load_confusables()
        num_subs = spec.params.get("num_substitutions", 1)
        chars = list(original_text)
        substituted = 0
        for i, ch in enumerate(chars):
            if substituted >= num_subs:
                break
            if ch in confusables and confusables[ch]:
                chars[i] = confusables[ch][0]
                substituted += 1
        new_text = "".join(chars)

    # Replace the matched text in the HTML
    new_html = base_html[: match.start(2)] + new_text + base_html[match.end(2):]
    return new_html


def degrade_font_weight(base_html: str, spec: DegradationSpec) -> str:
    """
    Inject a font-weight override.

    spec.params must contain:
        font_weight (str|int): e.g. "bold", "300", "500", "700"
    """
    weight = spec.params["font_weight"]
    css = f"font-weight: {weight} !important;"
    return _inject_css_override(base_html, spec.element_id, css)


def degrade_font_style(base_html: str, spec: DegradationSpec) -> str:
    """
    Inject a font-style override.

    spec.params must contain:
        font_style (str): "italic" or "oblique"
    """
    style = spec.params["font_style"]
    css = f"font-style: {style} !important;"
    return _inject_css_override(base_html, spec.element_id, css)


def degrade_letter_spacing(base_html: str, spec: DegradationSpec) -> str:
    """
    Inject a letter-spacing override.

    spec.params must contain:
        letter_spacing_px (float): offset in px (signed — positive = more space)
    """
    spacing = spec.params["letter_spacing_px"]
    css = f"letter-spacing: {spacing}px !important;"
    return _inject_css_override(base_html, spec.element_id, css)


def degrade_opacity(base_html: str, spec: DegradationSpec) -> str:
    """
    Inject an opacity override.

    spec.params must contain:
        opacity (float): value in [0, 1]
    """
    opacity = spec.params["opacity"]
    css = f"opacity: {opacity} !important;"
    return _inject_css_override(base_html, spec.element_id, css)


def degrade_rotation(base_html: str, spec: DegradationSpec) -> str:
    """
    Inject a rotation transform override.

    spec.params must contain:
        angle_deg (float): rotation angle in degrees (signed — positive = clockwise)
    """
    angle = spec.params["angle_deg"]
    css = f"transform: rotate({angle}deg) !important; display: inline-block;"
    return _inject_css_override(base_html, spec.element_id, css)


def _find_element_text(html: str, element_id: str) -> tuple[int, int, str] | None:
    """
    Locate element text in HTML. Returns (start, end, text) span or None.
    Same pattern as degrade_char_substitution — matches id="X" ... >TEXT<.
    """
    pattern = re.compile(
        r'(id=["\']' + re.escape(element_id) + r'["\'][^>]*>)([^<]*)',
        re.DOTALL,
    )
    match = pattern.search(html)
    if not match:
        return None
    return match.start(2), match.end(2), match.group(2)


def degrade_word_error(base_html: str, spec: DegradationSpec) -> str:
    """
    Apply word-level text errors.

    spec.params must contain:
        error_type (str): "wrong_word" | "missing_word" | "extra_word"
    spec.params may contain:
        word_index (int, default 0): which word to affect
        replacement (str): replacement word (for wrong_word, default "WRONG")
        extra_word (str): word to insert (for extra_word, default "EXTRA")
    """
    result = _find_element_text(base_html, spec.element_id)
    if result is None:
        return base_html
    start, end, text = result

    words = text.split()
    if not words:
        return base_html

    error_type = spec.params["error_type"]
    word_idx = spec.params.get("word_index", 0) % len(words)

    if error_type == "wrong_word":
        replacement = spec.params.get("replacement", "WRONG")
        words[word_idx] = replacement
    elif error_type == "missing_word":
        words.pop(word_idx)
    elif error_type == "extra_word":
        extra = spec.params.get("extra_word", "EXTRA")
        words.insert(word_idx + 1, extra)
    else:
        return base_html

    new_text = " ".join(words)
    return base_html[:start] + new_text + base_html[end:]


def degrade_case_error(base_html: str, spec: DegradationSpec) -> str:
    """
    Apply case errors to element text.

    spec.params must contain:
        case_type (str): "first_char_flip" | "all_caps" | "all_lower"
    """
    result = _find_element_text(base_html, spec.element_id)
    if result is None:
        return base_html
    start, end, text = result

    case_type = spec.params["case_type"]
    if case_type == "first_char_flip":
        if text:
            new_text = (text[0].lower() if text[0].isupper() else text[0].upper()) + text[1:]
        else:
            new_text = text
    elif case_type == "all_caps":
        new_text = text.upper()
    elif case_type == "all_lower":
        new_text = text.lower()
    else:
        new_text = text

    return base_html[:start] + new_text + base_html[end:]


def degrade_content_swap(base_html: str, spec: DegradationSpec) -> str:
    """
    Replace element text with wrong content.

    spec.params must contain:
        new_text (str): replacement text (always wrong relative to the edit instruction)
    """
    result = _find_element_text(base_html, spec.element_id)
    if result is None:
        return base_html
    start, end, _ = result
    new_text = spec.params["new_text"]
    return base_html[:start] + new_text + base_html[end:]


_HANDLERS = {
    "color_offset": degrade_color,
    "position_offset": degrade_position,
    "scale_error": degrade_scale,
    "char_substitution": degrade_char_substitution,
    "font_weight": degrade_font_weight,
    "font_style": degrade_font_style,
    "letter_spacing": degrade_letter_spacing,
    "opacity": degrade_opacity,
    "rotation": degrade_rotation,
    "word_error": degrade_word_error,
    "case_error": degrade_case_error,
    "content_swap": degrade_content_swap,
}


def apply_degradation(base_html: str, spec: DegradationSpec) -> str:
    handler = _HANDLERS.get(spec.dimension)
    if handler is None:
        raise ValueError(f"Unknown degradation dimension: {spec.dimension!r}")
    return handler(base_html, spec)
