#!/usr/bin/env python3
"""
generate.py — Layout-aware stimulus generation.

Generates calibration stimuli (source / ground_truth / degraded image triples)
from the layout registry + configs/degradations.yaml.

Each stimulus is one (layout, role, edit_type, target_value, degradation) combination.
Source shows the unedited layout; ground_truth shows the correct edit applied;
degraded shows the same edit with a controlled error.

Usage:
    # Dry run — print manifest without rendering
    python scripts/generate.py --edit-type color --count 10 --dry-run

    # Generate a small pilot set
    python scripts/generate.py --edit-type color --output-dir data/pilot --count 24 --seed 42

    # Single layout
    python scripts/generate.py --edit-type color --layout solo_headline --output-dir data/debug

    # Filter by difficulty
    python scripts/generate.py --edit-type scale --difficulty easy --output-dir data/scale_easy --seed 1

    # All layouts, all degradations, no count cap
    python scripts/generate.py --edit-type color --output-dir data/color_full --seed 42
"""

import argparse
import asyncio
import copy
import json
import sys
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import yaml
from skimage import color as skcolor

_PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(_PROJECT_ROOT / "src"))

from render import Renderer, RenderConfig
from layouts import (
    LayoutDefinition,
    all_layouts,
    get_layout,
    get_layouts_for_edit_difficulty,
    get_layouts_for_edit_all_difficulties,
)


# ---------------------------------------------------------------------------
# Edit target values (the "correct" values in the ground_truth image)
# ---------------------------------------------------------------------------

# Preset colors to use as edit targets for color edits.
_COLOR_TARGETS = [
    "#3B82F6",  # blue
    "#EF4444",  # red
    "#10B981",  # emerald
    "#F59E0B",  # amber
    "#8B5CF6",  # violet
    "#EC4899",  # pink
    "#06B6D4",  # cyan
]

# Scale multipliers applied to the role's base font_size_px.
_SCALE_MULTIPLIERS = [1.3, 1.5, 2.0, 0.75, 0.6]

# Rotation targets (degrees). Clamped to role's rotation_range at runtime.
_ROTATION_TARGETS_DEG = [10, -10, 20, -20, 30, -30]

# ---------------------------------------------------------------------------
# Relocation helpers — 3 × 3 alignment position grid (Chebyshev + Manhattan)
# ---------------------------------------------------------------------------

_ALIGNMENT_GRID: dict[str, tuple[int, int]] = {
    "top-left":      (0, 0), "top-center":    (0, 1), "top-right":    (0, 2),
    "center-left":   (1, 0), "center":        (1, 1), "center-right": (1, 2),
    "bottom-left":   (2, 0), "bottom-center": (2, 1), "bottom-right": (2, 2),
}


def _manhattan_distance(a: str, b: str) -> int:
    r1, c1 = _ALIGNMENT_GRID[a]
    r2, c2 = _ALIGNMENT_GRID[b]
    return abs(r1 - r2) + abs(c1 - c2)


# ---------------------------------------------------------------------------
# Font-weight helpers
# ---------------------------------------------------------------------------

_FONT_WEIGHT_NORM: dict[str, int] = {"normal": 400, "bold": 700, "lighter": 300, "bolder": 800}
_FONT_WEIGHT_TARGETS: list[int] = [100, 200, 300, 400, 500, 600, 700, 800, 900]


def _normalize_weight(w) -> int:
    if isinstance(w, int):
        return w
    return _FONT_WEIGHT_NORM.get(str(w).lower(), 400)


# ---------------------------------------------------------------------------
# Letter-spacing helpers
# ---------------------------------------------------------------------------

# Target letter-spacing values (px). Chosen to span a clear visible range.
_LETTER_SPACING_TARGETS_PX: list[float] = [2, 5, 10, -2, -5]


# ---------------------------------------------------------------------------
# Color utilities — LAB shift for color_offset degradation
# ---------------------------------------------------------------------------

def _hex_to_rgb01(hex_color: str) -> tuple[float, float, float]:
    h = hex_color.lstrip("#")
    return int(h[0:2], 16) / 255.0, int(h[2:4], 16) / 255.0, int(h[4:6], 16) / 255.0


def _rgb01_to_hex(r: float, g: float, b: float) -> str:
    return f"#{round(r * 255):02X}{round(g * 255):02X}{round(b * 255):02X}"


def _shift_color_lab(hex_color: str, delta_l: float) -> str:
    """Shift L* in CIELAB by delta_l, approximating a ΔE = delta_l offset.

    Direction is chosen to move away from the current lightness boundary
    (dark colors shift lighter, bright colors shift darker).
    """
    r, g, b = _hex_to_rgb01(hex_color)
    lab = skcolor.rgb2lab(np.array([[[r, g, b]]]))
    L, a, bv = float(lab[0, 0, 0]), float(lab[0, 0, 1]), float(lab[0, 0, 2])
    direction = -1.0 if L > 50 else 1.0
    L_new = float(np.clip(L + direction * abs(delta_l), 1.0, 99.0))
    shifted = np.array([[[L_new, a, bv]]])
    rgb_new = np.clip(skcolor.lab2rgb(shifted), 0.0, 1.0)
    return _rgb01_to_hex(float(rgb_new[0, 0, 0]), float(rgb_new[0, 0, 1]), float(rgb_new[0, 0, 2]))


# ---------------------------------------------------------------------------
# Stimulus spec dataclass
# ---------------------------------------------------------------------------

@dataclass
class StimulusSpec:
    stimulus_id: str
    layout_name: str
    edit_type: str
    target_role: str
    edit_property: str       # CSS property key in role style dict
    source_value: object     # value in source (base) layout
    target_value: object     # value in ground_truth
    deg_dimension: str       # degradation dimension (e.g. "color_offset")
    # All degradation levels for this stimulus.
    # Each entry: {id, magnitude, layer, params, degraded_value, secondary_degs}
    degradations: list = field(default_factory=list)


# ---------------------------------------------------------------------------
# Secondary degradation helpers
# ---------------------------------------------------------------------------

# Dimensions (other than the primary edit dim) that can be applied as style-dict
# overrides without needing element IDs or HTML surgery.
_SECONDARY_DIMS = ["scale_error", "rotation", "font_weight", "font_style", "letter_spacing", "opacity"]

# How many unrelated degradation dimensions to sample per layout.
_N_SECONDARY = 5

# Dims where the param can be positive or negative — pick one direction per layout.
_BIDIRECTIONAL_DIMS = {
    "scale_error": "scale_error_pct",
    "rotation": "angle_deg",
    "letter_spacing": "letter_spacing_px",
}


def _build_dim_configs(deg_configs: list[dict]) -> dict[str, dict]:
    """Build per-dimension config lists, split by direction for bidirectional dims.

    Returns {dim: {"positive": [...], "negative": [...]}} for bidirectional dims,
    {dim: {"all": [...]}} for unidirectional dims.
    Lists are sorted ascending by magnitude (smallest first).
    """
    result = {}
    for dim in _SECONDARY_DIMS:
        candidates = [
            d for d in deg_configs
            if d["dimension"] == dim and d.get("layer", "html") == "html"
        ]
        if not candidates:
            continue
        if dim in _BIDIRECTIONAL_DIMS:
            param_key = _BIDIRECTIONAL_DIMS[dim]
            pos_list = sorted(
                [c for c in candidates if c["params"].get(param_key, 0) > 0],
                key=lambda c: c["params"][param_key],
            )
            neg_list = sorted(
                [c for c in candidates if c["params"].get(param_key, 0) < 0],
                key=lambda c: -c["params"][param_key],  # ascending magnitude (−1 before −10)
            )
            result[dim] = {"positive": pos_list, "negative": neg_list}
        else:
            result[dim] = {"all": candidates}
    return result


def _pick_tiny_moderate_large(configs: list[dict]) -> list[dict]:
    """Return [tiny, moderate, large] from a magnitude-sorted list."""
    n = len(configs)
    if n == 0:
        return []
    return [configs[0], configs[n // 2], configs[-1]]


def _apply_secondary_degs_to_styles(
    styles: dict, role: str, secondary_degs: list[dict]
) -> dict:
    """Return a deep copy of styles with secondary degradation overrides applied."""
    import copy
    styles = copy.deepcopy(styles)
    base_size = styles[role].get("font_size_px", 36)
    for deg in secondary_degs:
        dim = deg["dimension"]
        params = deg["params"]
        if dim == "scale_error":
            error_pct = params["scale_error_pct"]
            new_size = max(12, min(200, round(base_size * (1 + error_pct / 100))))
            styles[role]["font_size_px"] = new_size
        elif dim == "rotation":
            styles[role]["rotation_deg"] = params["angle_deg"]
        elif dim == "font_weight":
            styles[role]["font_weight"] = params["font_weight"]
        elif dim == "font_style":
            styles[role]["font_style"] = params["font_style"]
        elif dim == "letter_spacing":
            styles[role]["letter_spacing"] = f"{params['letter_spacing_px']}px"
        elif dim == "opacity":
            styles[role]["opacity"] = params["opacity"]
    return styles


# ---------------------------------------------------------------------------
# Manifest builders — one per edit type
# ---------------------------------------------------------------------------

def _color_manifest(
    layouts: list[LayoutDefinition],
    deg_configs: list[dict],
    rng=None,
) -> list[StimulusSpec]:
    import random
    if rng is None:
        rng = random.Random(0)

    color_degs = [d for d in deg_configs if d["dimension"] == "color_offset"]
    dim_configs = _build_dim_configs(deg_configs)
    secondary_dims = list(dim_configs.keys())

    specs = []
    for layout in layouts:
        if "color" not in layout.supported_edits:
            continue
        role = layout.primary_role
        source_color = layout.role_base_styles[role]["color"]
        available_colors = [c for c in _COLOR_TARGETS if c.upper() != source_color.upper()]
        target_color = rng.choice(available_colors)

        # 1. Primary spec: color_offset only — no secondary degradations
        color_degradations = []
        for deg in color_degs:
            delta_l = deg["params"]["delta_e"]
            degraded_color = _shift_color_lab(target_color, delta_l)
            color_degradations.append({
                "id": deg["id"],
                "magnitude": deg["magnitude"],
                "layer": deg.get("layer", "html"),
                "params": deg["params"],
                "degraded_value": degraded_color,
                "secondary_degs": [],
            })

        specs.append(StimulusSpec(
            stimulus_id=f"color_{layout.id}_color_offset",
            layout_name=layout.name,
            edit_type="color",
            target_role=role,
            edit_property="color",
            source_value=source_color,
            target_value=target_color,
            deg_dimension="color_offset",
            degradations=color_degradations,
        ))

        # 2. Unrelated specs: color is correct, one unrelated property is degraded per spec.
        # Each layout independently samples dims and, for bidirectional dims, a direction.
        n_unrelated = min(_N_SECONDARY, len(secondary_dims))
        sampled_dims = rng.sample(secondary_dims, n_unrelated)
        for dim in sampled_dims:
            dim_data = dim_configs[dim]
            if "positive" in dim_data:
                # Bidirectional: pick one direction per layout so all 3 levels go the same way.
                direction = rng.choice(["positive", "negative"])
                candidates = dim_data[direction]
            else:
                candidates = dim_data["all"]

            three_levels = _pick_tiny_moderate_large(candidates)
            if not three_levels:
                continue

            unrelated_degradations = []
            for deg in three_levels:
                unrelated_degradations.append({
                    "id": deg["id"],
                    "magnitude": deg["magnitude"],
                    "layer": deg.get("layer", "html"),
                    "params": deg["params"],
                    "degraded_value": target_color,  # color is perfectly correct
                    "secondary_degs": [deg],          # only this unrelated dim changes
                })
            specs.append(StimulusSpec(
                stimulus_id=f"color_{layout.id}_{dim}",
                layout_name=layout.name,
                edit_type="color",
                target_role=role,
                edit_property="color",
                source_value=source_color,
                target_value=target_color,
                deg_dimension=dim,
                degradations=unrelated_degradations,
            ))

    return specs


def _scale_manifest(layouts: list[LayoutDefinition], deg_configs: list[dict], rng=None) -> list[StimulusSpec]:
    import random
    if rng is None:
        rng = random.Random(0)
    scale_degs = [d for d in deg_configs if d["dimension"] == "scale_error"]
    dim_configs = _build_dim_configs(deg_configs)
    secondary_dims = [d for d in dim_configs.keys() if d != "scale_error"]

    specs = []
    for layout in layouts:
        if "scale" not in layout.supported_edits:
            continue
        role = layout.primary_role
        base_size = layout.role_base_styles[role]["font_size_px"]
        multiplier = rng.choice(_SCALE_MULTIPLIERS)
        target_size = max(12, min(200, int(round(base_size * multiplier))))

        # 1. Primary spec: scale_error only
        degradations = []
        for deg in scale_degs:
            error_pct = deg["params"]["scale_error_pct"]
            degraded_size = max(12, min(200, int(round(target_size * (1 + error_pct / 100)))))
            degradations.append({
                "id": deg["id"],
                "magnitude": deg["magnitude"],
                "layer": deg.get("layer", "html"),
                "params": deg["params"],
                "degraded_value": degraded_size,
                "secondary_degs": [],
            })

        specs.append(StimulusSpec(
            stimulus_id=f"scale_{layout.id}_scale_error",
            layout_name=layout.name,
            edit_type="scale",
            target_role=role,
            edit_property="font_size_px",
            source_value=base_size,
            target_value=target_size,
            deg_dimension="scale_error",
            degradations=degradations,
        ))

        # 2. Unrelated specs: scale is correct, one unrelated property is degraded per spec.
        n_unrelated = min(_N_SECONDARY, len(secondary_dims))
        sampled_dims = rng.sample(secondary_dims, n_unrelated)
        for dim in sampled_dims:
            dim_data = dim_configs[dim]
            if "positive" in dim_data:
                direction = rng.choice(["positive", "negative"])
                candidates = dim_data[direction]
            else:
                candidates = dim_data["all"]

            three_levels = _pick_tiny_moderate_large(candidates)
            if not three_levels:
                continue

            unrelated_degradations = []
            for deg in three_levels:
                unrelated_degradations.append({
                    "id": deg["id"],
                    "magnitude": deg["magnitude"],
                    "layer": deg.get("layer", "html"),
                    "params": deg["params"],
                    "degraded_value": target_size,  # scale is perfectly correct
                    "secondary_degs": [deg],
                })
            specs.append(StimulusSpec(
                stimulus_id=f"scale_{layout.id}_{dim}",
                layout_name=layout.name,
                edit_type="scale",
                target_role=role,
                edit_property="font_size_px",
                source_value=base_size,
                target_value=target_size,
                deg_dimension=dim,
                degradations=unrelated_degradations,
            ))

    return specs


def _relocation_manifest(layouts: list[LayoutDefinition], deg_configs: list[dict], rng=None) -> list[StimulusSpec]:
    """Relocation: move text to a different alignment position.

    Primary degradation: wrong alignment positions ordered by Manhattan distance from target.
    Secondary degradations: unrelated property errors while position is correct.
    """
    import random
    if rng is None:
        rng = random.Random(0)
    dim_configs = _build_dim_configs(deg_configs)
    secondary_dims = list(dim_configs.keys())  # alignment_error is not in _SECONDARY_DIMS

    specs = []
    for layout in layouts:
        if "relocation" not in layout.supported_edits:
            continue
        role = layout.primary_role
        rc = layout.role_constraints[role]
        available_positions = rc.alignment_positions
        if not available_positions:
            continue

        base_alignment = layout.role_base_styles[role].get("alignment", "center")
        other_positions = [p for p in available_positions if p != base_alignment]
        if not other_positions:
            continue
        target_alignment = rng.choice(other_positions)

        # Wrong positions sorted by Manhattan distance from target (ascending = closest first).
        # Use only positions that are in _ALIGNMENT_GRID so distance is computable.
        wrong_candidates = sorted(
            [p for p in available_positions if p != target_alignment and p in _ALIGNMENT_GRID and target_alignment in _ALIGNMENT_GRID],
            key=lambda p: _manhattan_distance(target_alignment, p),
        )
        # Pick up to 3 spread levels (close / mid / far).
        if len(wrong_candidates) >= 3:
            n = len(wrong_candidates)
            level_positions = [wrong_candidates[0], wrong_candidates[n // 2], wrong_candidates[-1]]
            magnitudes = ["reloc_close", "reloc_medium", "reloc_far"]
        elif len(wrong_candidates) == 2:
            level_positions = [wrong_candidates[0], wrong_candidates[-1]]
            magnitudes = ["reloc_close", "reloc_far"]
        elif len(wrong_candidates) == 1:
            level_positions = [wrong_candidates[0]]
            magnitudes = ["reloc_far"]
        else:
            continue

        primary_degs = []
        for pos, mag in zip(level_positions, magnitudes):
            primary_degs.append({
                "id": f"alignment_error_{mag}",
                "magnitude": mag,
                "layer": "html",
                "params": {"wrong_alignment": pos},
                "degraded_value": pos,
                "secondary_degs": [],
            })

        specs.append(StimulusSpec(
            stimulus_id=f"relocation_{layout.id}_alignment_error",
            layout_name=layout.name,
            edit_type="relocation",
            target_role=role,
            edit_property="alignment",
            source_value=base_alignment,
            target_value=target_alignment,
            deg_dimension="alignment_error",
            degradations=primary_degs,
        ))

        # Unrelated specs: position is correct, one unrelated property is degraded.
        n_unrelated = min(_N_SECONDARY, len(secondary_dims))
        sampled_dims = rng.sample(secondary_dims, n_unrelated)
        for dim in sampled_dims:
            dim_data = dim_configs[dim]
            if "positive" in dim_data:
                direction = rng.choice(["positive", "negative"])
                candidates = dim_data[direction]
            else:
                candidates = dim_data["all"]

            three_levels = _pick_tiny_moderate_large(candidates)
            if not three_levels:
                continue

            unrelated_degradations = []
            for deg in three_levels:
                unrelated_degradations.append({
                    "id": deg["id"],
                    "magnitude": deg["magnitude"],
                    "layer": deg.get("layer", "html"),
                    "params": deg["params"],
                    "degraded_value": target_alignment,  # position is perfectly correct
                    "secondary_degs": [deg],
                })
            specs.append(StimulusSpec(
                stimulus_id=f"relocation_{layout.id}_{dim}",
                layout_name=layout.name,
                edit_type="relocation",
                target_role=role,
                edit_property="alignment",
                source_value=base_alignment,
                target_value=target_alignment,
                deg_dimension=dim,
                degradations=unrelated_degradations,
            ))

    return specs


def _font_weight_manifest(layouts: list[LayoutDefinition], deg_configs: list[dict], rng=None) -> list[StimulusSpec]:
    """Font weight: change the weight value (100–900).

    Primary degradation: wrong weight values sorted by distance from target.
    Secondary degradations: unrelated property errors while weight is correct.
    """
    import random
    if rng is None:
        rng = random.Random(0)
    fw_degs = [d for d in deg_configs if d["dimension"] == "font_weight"]
    dim_configs = _build_dim_configs(deg_configs)
    secondary_dims = [d for d in dim_configs.keys() if d != "font_weight"]

    specs = []
    for layout in layouts:
        if "font_weight" not in layout.supported_edits:
            continue
        role = layout.primary_role
        base_weight_raw = layout.role_base_styles[role]["font_weight"]
        base_weight = _normalize_weight(base_weight_raw)
        available_targets = [w for w in _FONT_WEIGHT_TARGETS if w != base_weight]
        target_weight = rng.choice(available_targets)

        # Wrong weights: all fw configs except target, sorted by distance from target.
        wrong_degs = sorted(
            [d for d in fw_degs if d["params"]["font_weight"] != target_weight],
            key=lambda d: abs(d["params"]["font_weight"] - target_weight),
        )
        three_levels = _pick_tiny_moderate_large(wrong_degs)
        if not three_levels:
            continue

        primary_degs = []
        for deg in three_levels:
            primary_degs.append({
                "id": deg["id"],
                "magnitude": deg["magnitude"],
                "layer": deg.get("layer", "html"),
                "params": deg["params"],
                "degraded_value": deg["params"]["font_weight"],
                "secondary_degs": [],
            })

        specs.append(StimulusSpec(
            stimulus_id=f"font_weight_{layout.id}_font_weight",
            layout_name=layout.name,
            edit_type="font_weight",
            target_role=role,
            edit_property="font_weight",
            source_value=base_weight_raw,
            target_value=target_weight,
            deg_dimension="font_weight",
            degradations=primary_degs,
        ))

        # Unrelated specs.
        n_unrelated = min(_N_SECONDARY, len(secondary_dims))
        sampled_dims = rng.sample(secondary_dims, n_unrelated)
        for dim in sampled_dims:
            dim_data = dim_configs[dim]
            if "positive" in dim_data:
                direction = rng.choice(["positive", "negative"])
                candidates = dim_data[direction]
            else:
                candidates = dim_data["all"]

            three_levels = _pick_tiny_moderate_large(candidates)
            if not three_levels:
                continue

            unrelated_degradations = []
            for deg in three_levels:
                unrelated_degradations.append({
                    "id": deg["id"],
                    "magnitude": deg["magnitude"],
                    "layer": deg.get("layer", "html"),
                    "params": deg["params"],
                    "degraded_value": target_weight,  # weight is perfectly correct
                    "secondary_degs": [deg],
                })
            specs.append(StimulusSpec(
                stimulus_id=f"font_weight_{layout.id}_{dim}",
                layout_name=layout.name,
                edit_type="font_weight",
                target_role=role,
                edit_property="font_weight",
                source_value=base_weight_raw,
                target_value=target_weight,
                deg_dimension=dim,
                degradations=unrelated_degradations,
            ))

    return specs


def _italic_manifest(layouts: list[LayoutDefinition], deg_configs: list[dict], rng=None) -> list[StimulusSpec]:
    """Italic: flip font-style between normal and italic.

    Primary degradation: wrong font-style values (oblique = visually close to italic, normal = clearly wrong).
    Secondary degradations: unrelated property errors while font-style is correct.
    """
    import random
    if rng is None:
        rng = random.Random(0)
    dim_configs = _build_dim_configs(deg_configs)
    secondary_dims = [d for d in dim_configs.keys() if d != "font_style"]

    specs = []
    for layout in layouts:
        if "italic" not in layout.supported_edits:
            continue
        role = layout.primary_role
        base_style = layout.role_base_styles[role].get("font_style", "normal")

        # Target: flip the current style.
        target_style = "italic" if base_style != "italic" else "normal"

        # Wrong values ordered closest-to-farthest from target.
        if target_style == "italic":
            # oblique ≈ italic visually (subtle difference), normal is clearly wrong
            wrong_styles = [("oblique", "style_oblique"), ("normal", "style_normal")]
        else:
            # flipping to normal: italic is close wrong, oblique is similar to italic
            wrong_styles = [("oblique", "style_oblique"), ("italic", "style_italic")]

        primary_degs = [
            {
                "id": f"italic_error_{style}",
                "magnitude": mag,
                "layer": "html",
                "params": {"font_style": style},
                "degraded_value": style,
                "secondary_degs": [],
            }
            for style, mag in wrong_styles
        ]

        specs.append(StimulusSpec(
            stimulus_id=f"italic_{layout.id}_font_style",
            layout_name=layout.name,
            edit_type="italic",
            target_role=role,
            edit_property="font_style",
            source_value=base_style,
            target_value=target_style,
            deg_dimension="font_style",
            degradations=primary_degs,
        ))

        # Unrelated specs.
        n_unrelated = min(_N_SECONDARY, len(secondary_dims))
        sampled_dims = rng.sample(secondary_dims, n_unrelated)
        for dim in sampled_dims:
            dim_data = dim_configs[dim]
            if "positive" in dim_data:
                direction = rng.choice(["positive", "negative"])
                candidates = dim_data[direction]
            else:
                candidates = dim_data["all"]

            three_levels = _pick_tiny_moderate_large(candidates)
            if not three_levels:
                continue

            unrelated_degradations = []
            for deg in three_levels:
                unrelated_degradations.append({
                    "id": deg["id"],
                    "magnitude": deg["magnitude"],
                    "layer": deg.get("layer", "html"),
                    "params": deg["params"],
                    "degraded_value": target_style,  # font-style is perfectly correct
                    "secondary_degs": [deg],
                })
            specs.append(StimulusSpec(
                stimulus_id=f"italic_{layout.id}_{dim}",
                layout_name=layout.name,
                edit_type="italic",
                target_role=role,
                edit_property="font_style",
                source_value=base_style,
                target_value=target_style,
                deg_dimension=dim,
                degradations=unrelated_degradations,
            ))

    return specs


def _letter_spacing_manifest(layouts: list[LayoutDefinition], deg_configs: list[dict], rng=None) -> list[StimulusSpec]:
    """Letter spacing: change tracking to a target px value.

    Primary degradation: wrong letter-spacing values sorted by distance from target.
    Secondary degradations: unrelated property errors while letter-spacing is correct.
    """
    import random
    if rng is None:
        rng = random.Random(0)
    ls_degs = [d for d in deg_configs if d["dimension"] == "letter_spacing"]
    dim_configs = _build_dim_configs(deg_configs)
    secondary_dims = [d for d in dim_configs.keys() if d != "letter_spacing"]

    specs = []
    for layout in layouts:
        if "letter_spacing" not in layout.supported_edits:
            continue
        role = layout.primary_role
        base_ls = layout.role_base_styles[role].get("letter_spacing", "normal")
        target_px = rng.choice(_LETTER_SPACING_TARGETS_PX)
        target_value = f"{target_px}px"

        # Wrong letter-spacings: all ls configs except target, sorted by distance from target.
        wrong_degs = sorted(
            [d for d in ls_degs if d["params"]["letter_spacing_px"] != target_px],
            key=lambda d: abs(d["params"]["letter_spacing_px"] - target_px),
        )
        three_levels = _pick_tiny_moderate_large(wrong_degs)
        if not three_levels:
            continue

        primary_degs = []
        for deg in three_levels:
            wrong_px = deg["params"]["letter_spacing_px"]
            primary_degs.append({
                "id": deg["id"],
                "magnitude": deg["magnitude"],
                "layer": deg.get("layer", "html"),
                "params": deg["params"],
                "degraded_value": f"{wrong_px}px",
                "secondary_degs": [],
            })

        specs.append(StimulusSpec(
            stimulus_id=f"letter_spacing_{layout.id}_letter_spacing",
            layout_name=layout.name,
            edit_type="letter_spacing",
            target_role=role,
            edit_property="letter_spacing",
            source_value=base_ls,
            target_value=target_value,
            deg_dimension="letter_spacing",
            degradations=primary_degs,
        ))

        # Unrelated specs.
        n_unrelated = min(_N_SECONDARY, len(secondary_dims))
        sampled_dims = rng.sample(secondary_dims, n_unrelated)
        for dim in sampled_dims:
            dim_data = dim_configs[dim]
            if "positive" in dim_data:
                direction = rng.choice(["positive", "negative"])
                candidates = dim_data[direction]
            else:
                candidates = dim_data["all"]

            three_levels = _pick_tiny_moderate_large(candidates)
            if not three_levels:
                continue

            unrelated_degradations = []
            for deg in three_levels:
                unrelated_degradations.append({
                    "id": deg["id"],
                    "magnitude": deg["magnitude"],
                    "layer": deg.get("layer", "html"),
                    "params": deg["params"],
                    "degraded_value": target_value,  # letter-spacing is perfectly correct
                    "secondary_degs": [deg],
                })
            specs.append(StimulusSpec(
                stimulus_id=f"letter_spacing_{layout.id}_{dim}",
                layout_name=layout.name,
                edit_type="letter_spacing",
                target_role=role,
                edit_property="letter_spacing",
                source_value=base_ls,
                target_value=target_value,
                deg_dimension=dim,
                degradations=unrelated_degradations,
            ))

    return specs


def _rotation_manifest(layouts: list[LayoutDefinition], deg_configs: list[dict], rng=None) -> list[StimulusSpec]:
    import random
    if rng is None:
        rng = random.Random(0)
    rot_degs = [d for d in deg_configs if d["dimension"] == "rotation"]
    specs = []
    for layout in layouts:
        if "rotation" not in layout.supported_edits:
            continue
        role = layout.primary_role
        rc = layout.role_constraints[role]
        lo, hi = rc.rotation_range
        raw = rng.choice(_ROTATION_TARGETS_DEG)
        target_deg = float(max(lo, min(hi, raw)))

        degradations = []
        for deg in rot_degs:
            error_deg = deg["params"]["angle_deg"]
            degradations.append({
                "id": deg["id"],
                "magnitude": deg["magnitude"],
                "layer": deg.get("layer", "html"),
                "params": deg["params"],
                "degraded_value": float(target_deg + error_deg),
                "secondary_degs": [],
            })

        specs.append(StimulusSpec(
            stimulus_id=f"rotation_{layout.id}_rotation",
            layout_name=layout.name,
            edit_type="rotation",
            target_role=role,
            edit_property="rotation_deg",
            source_value=0.0,
            target_value=target_deg,
            deg_dimension="rotation",
            degradations=degradations,
        ))
    return specs


_MANIFEST_BUILDERS = {
    "color":          _color_manifest,
    "scale":          _scale_manifest,
    "rotation":       _rotation_manifest,
    "relocation":     _relocation_manifest,
    "font_weight":    _font_weight_manifest,
    "italic":         _italic_manifest,
    "letter_spacing": _letter_spacing_manifest,
}


def build_manifest(
    layouts: list[LayoutDefinition],
    edit_type: str,
    deg_configs: list[dict],
    count: int | None = None,
    seed: int | None = None,
) -> list[StimulusSpec]:
    import random
    rng = random.Random(seed)  # deterministic regardless of seed=None (uses os.urandom)
    specs = _MANIFEST_BUILDERS[edit_type](layouts, deg_configs, rng)
    rng.shuffle(specs)
    if count is not None:
        specs = specs[:count]
    return specs


# ---------------------------------------------------------------------------
# Apply edit / degradation to style dict
# ---------------------------------------------------------------------------

def _apply_style_change(base_styles: dict, role: str, prop: str, value: object) -> dict:
    """Return a deep copy of base_styles with one property changed for one role."""
    styles = copy.deepcopy(base_styles)
    styles[role][prop] = value
    return styles


# ---------------------------------------------------------------------------
# Generation pipeline
# ---------------------------------------------------------------------------

async def generate_stimuli(
    specs: list[StimulusSpec],
    output_dir: Path,
    render_config: RenderConfig,
) -> Path:
    """Render all stimuli and write records to stimuli.jsonl.

    Appends to an existing JSONL file (crash-safe, resume-capable).
    Skips stimuli whose source image already exists.
    """
    image_dir = output_dir / "images"
    image_dir.mkdir(parents=True, exist_ok=True)
    jsonl_path = output_dir / "stimuli.jsonl"

    n = len(specs)
    n_ok = n_skip = n_fail = 0

    async with Renderer(render_config) as renderer:
        with jsonl_path.open("a", encoding="utf-8") as f:
            for i, spec in enumerate(specs, 1):
                src_png = image_dir / f"{spec.stimulus_id}_source.png"
                if src_png.exists():
                    print(f"  [{i}/{n}] SKIP (exists): {spec.stimulus_id}")
                    n_skip += 1
                    continue

                layout = get_layout(spec.layout_name)
                contents = layout.default_content
                bg = layout.background
                base_styles = layout.role_base_styles

                source_html = layout.html_builder(contents, base_styles, bg)
                correct_styles = _apply_style_change(
                    base_styles, spec.target_role, spec.edit_property, spec.target_value
                )
                ground_truth_html = layout.html_builder(contents, correct_styles, bg)

                try:
                    src_r = await renderer.render_html_string(
                        source_html, image_dir / f"{spec.stimulus_id}_source.png"
                    )
                    gt_r = await renderer.render_html_string(
                        ground_truth_html, image_dir / f"{spec.stimulus_id}_ground_truth.png"
                    )

                    all_errors = src_r.errors + gt_r.errors
                    degraded_images = {}
                    degraded_records = []

                    for deg in spec.degradations:
                        deg_styles = _apply_style_change(
                            correct_styles, spec.target_role, spec.edit_property, deg["degraded_value"]
                        )
                        if deg.get("secondary_degs"):
                            deg_styles = _apply_secondary_degs_to_styles(
                                deg_styles, spec.target_role, deg["secondary_degs"]
                            )
                        deg_html = layout.html_builder(contents, deg_styles, bg)
                        deg_r = await renderer.render_html_string(
                            deg_html, image_dir / f"{spec.stimulus_id}_{deg['magnitude']}.png"
                        )
                        all_errors += deg_r.errors
                        degraded_images[deg["magnitude"]] = str(deg_r.image_path.relative_to(output_dir))
                        degraded_records.append({
                            "id": deg["id"],
                            "magnitude": deg["magnitude"],
                            "layer": deg["layer"],
                            "params": deg["params"],
                            "degraded_value": deg["degraded_value"],
                        })

                    record = {
                        "stimulus_id": spec.stimulus_id,
                        "layout": spec.layout_name,
                        "layout_difficulty": layout.difficulty,
                        "edit_type": spec.edit_type,
                        "target_role": spec.target_role,
                        "edit": {
                            "property": spec.edit_property,
                            "source_value": spec.source_value,
                            "target_value": spec.target_value,
                        },
                        "degradation_dimension": spec.deg_dimension,
                        "degradations": degraded_records,
                        "source_image": str(src_r.image_path.relative_to(output_dir)),
                        "ground_truth_image": str(gt_r.image_path.relative_to(output_dir)),
                        "degraded_images": degraded_images,
                    }
                    if all_errors:
                        record["render_errors"] = all_errors

                    f.write(json.dumps(record) + "\n")
                    print(f"  [{i}/{n}] OK: {spec.stimulus_id}")
                    n_ok += 1

                except Exception as e:
                    print(f"  [{i}/{n}] FAIL: {spec.stimulus_id} — {e}")
                    n_fail += 1

    print(f"\nDone. {n_ok} generated, {n_skip} skipped, {n_fail} failed.")
    return jsonl_path


def generate_stimuli_sync(
    specs: list[StimulusSpec],
    output_dir: Path,
    render_config: RenderConfig | None = None,
) -> Path:
    return asyncio.run(generate_stimuli(specs, output_dir, render_config or RenderConfig()))


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    all_edit_types = sorted(_MANIFEST_BUILDERS)

    parser = argparse.ArgumentParser(
        description="Generate calibration stimuli from the layout registry.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Edit type selection:\n"
            "  --edit-type all              All registered edit types\n"
            "  --edit-type color scale      Specific types (space-separated)\n"
            "  --edit-type color            Single type (original behaviour)\n"
            "\n"
            "Layout selection (mutually exclusive):\n"
            "  --layout NAME      Single layout (e.g. solo_headline)\n"
            "  --difficulty TIER  All layouts at that difficulty (easy/medium/hard/multi)\n"
            "  (default)          All layouts that support the requested edit type\n"
        ),
    )

    parser.add_argument(
        "--edit-type", nargs="+", default=["color"],
        metavar="TYPE",
        help=(
            f"Edit type(s) to generate. Use 'all' for every type. "
            f"Available: {', '.join(all_edit_types)}. (default: color)"
        ),
    )

    layout_group = parser.add_mutually_exclusive_group()
    layout_group.add_argument(
        "--layout", default=None,
        help="Single layout name (e.g. solo_headline, corner_badge)",
    )
    layout_group.add_argument(
        "--difficulty", default=None,
        choices=["easy", "medium", "hard", "multi"],
        help="Filter layouts by difficulty tier",
    )

    parser.add_argument(
        "--output-dir", default="data/generated",
        help="Output directory for images/ and stimuli.jsonl (default: data/generated)",
    )
    parser.add_argument(
        "--seed", type=int, default=None,
        help="RNG seed for reproducible shuffling before --count truncation",
    )
    parser.add_argument(
        "--count", type=int, default=None,
        help="Maximum number of stimuli per edit type (default: all combinations)",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Print the manifest without rendering anything",
    )
    parser.add_argument(
        "--degradations", type=Path,
        default=_PROJECT_ROOT / "configs" / "degradations.yaml",
        help="Path to degradations YAML (default: configs/degradations.yaml)",
    )

    args = parser.parse_args()

    # Resolve edit type list — "all" expands to every registered type.
    if "all" in args.edit_type:
        edit_types = all_edit_types
    else:
        unknown = [t for t in args.edit_type if t not in _MANIFEST_BUILDERS]
        if unknown:
            parser.error(
                f"Unknown edit type(s): {', '.join(unknown)}. "
                f"Available: {', '.join(all_edit_types)}"
            )
        edit_types = args.edit_type

    # Load degradations once.
    with open(args.degradations) as f:
        deg_data = yaml.safe_load(f)
    deg_configs = deg_data["degradations"]

    # Build a manifest per edit type and collect all specs.
    all_specs: list[StimulusSpec] = []
    skipped_types: list[str] = []

    for edit_type in edit_types:
        if args.layout:
            layouts = [get_layout(args.layout)]
            layouts = [lay for lay in layouts if edit_type in lay.supported_edits]
            if not layouts:
                print(
                    f"[{edit_type}] Layout '{args.layout}' does not support this edit type — skipping."
                )
                skipped_types.append(edit_type)
                continue
        elif args.difficulty:
            try:
                layouts = get_layouts_for_edit_difficulty(edit_type, args.difficulty)
            except KeyError:
                print(
                    f"[{edit_type}] No layout set defined for difficulty '{args.difficulty}' — skipping."
                )
                skipped_types.append(edit_type)
                continue
        else:
            layouts = get_layouts_for_edit_all_difficulties(edit_type)
            if not layouts:
                print(
                    f"[{edit_type}] No LAYOUT_SETS entries — skipping. "
                    "Add cells to LAYOUT_SETS in definitions.py or use --layout NAME directly."
                )
                skipped_types.append(edit_type)
                continue

        specs = build_manifest(
            layouts=layouts,
            edit_type=edit_type,
            deg_configs=deg_configs,
            count=args.count,
            seed=args.seed,
        )
        if not specs:
            print(f"[{edit_type}] Manifest is empty — skipping.")
            skipped_types.append(edit_type)
            continue

        all_specs.extend(specs)

    if not all_specs:
        print("No stimuli to generate.")
        sys.exit(0)

    # Summary header — grouped by edit type.
    print(f"\nEdit types  : {', '.join(edit_types)}")
    if skipped_types:
        print(f"Skipped     : {', '.join(skipped_types)}")
    print(f"Total stimuli: {len(all_specs)}")
    for edit_type in edit_types:
        type_specs = [s for s in all_specs if s.edit_type == edit_type]
        if not type_specs:
            continue
        layout_names = sorted({s.layout_name for s in type_specs})
        dim_counts: dict[str, int] = {}
        for s in type_specs:
            dim_counts[s.deg_dimension] = dim_counts.get(s.deg_dimension, 0) + 1
        print(f"\n  [{edit_type}]  {len(type_specs)} stimuli  |  layouts: {', '.join(layout_names)}")
        for dim, cnt in sorted(dim_counts.items()):
            print(f"    {dim}: {cnt}")

    if args.dry_run:
        print(f"\nDRY RUN — manifest ({len(all_specs)} stimuli):")
        for spec in all_specs:
            magnitudes = ", ".join(d["magnitude"] for d in spec.degradations)
            print(
                f"  {spec.stimulus_id}\n"
                f"    {spec.layout_name}.{spec.target_role}  "
                f"{spec.source_value} -> {spec.target_value}  |  "
                f"degradations: [{magnitudes}]"
            )
        return

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"\nOutput      : {output_dir}\n")

    jsonl_path = generate_stimuli_sync(all_specs, output_dir)
    print(f"JSONL       : {jsonl_path}")


if __name__ == "__main__":
    main()
