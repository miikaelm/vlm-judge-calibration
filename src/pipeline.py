"""
pipeline.py — Orchestrator for stimulus triple generation.

A stimulus is a triple: (source.png, ground_truth.png, degraded.png) + metadata.json.

source.png       — original template, no edit applied
ground_truth.png — template with the correct edit applied
degraded.png     — template with correct edit + degradation applied
"""

import asyncio
import json
from dataclasses import dataclass, asdict
from pathlib import Path

from src.render import Renderer, RenderConfig
from src.templates.templates import Template, get_template
from src.edit_applicator import apply_correct_edit
from src.degradation.engine import apply_degradation
from src.degradation.specs import DegradationSpec


@dataclass
class EditSpec:
    edit_id: str
    edit_type: str
    template_id: str
    element_id: str
    edit_params: dict
    edit_instruction: str


@dataclass
class StimulusResult:
    stimulus_id: str
    output_dir: Path
    source_path: Path
    ground_truth_path: Path
    degraded_path: Path
    metadata_path: Path


async def generate_stimulus_async(
    renderer: Renderer,
    template: Template,
    edit_spec: EditSpec,
    degradation_spec: DegradationSpec,
    output_dir: Path,
    stimulus_id: str,
) -> StimulusResult:
    """Generate one stimulus triple. Caller owns the renderer context."""
    output_dir.mkdir(parents=True, exist_ok=True)

    # Generate the 3 HTML strings
    source_html = template.render_html(None)
    gt_html = apply_correct_edit(source_html, edit_spec.edit_type, edit_spec.edit_params)

    # Build DegradationSpec with element-specific params merged from edit
    deg_spec = DegradationSpec(
        dimension=degradation_spec.dimension,
        magnitude=degradation_spec.magnitude,
        element_id=degradation_spec.element_id or edit_spec.element_id,
        target=degradation_spec.target,
        layer=degradation_spec.layer,
        params={**degradation_spec.params},
    )

    # For color_offset degradation, we need to know the base color from the edit
    if deg_spec.dimension == "color_offset" and edit_spec.edit_type == "color_change":
        if "base_color" not in deg_spec.params:
            deg_spec.params["base_color"] = edit_spec.edit_params.get("new_color", "#000000")

    # For scale_error, we need the base font size from the edit
    if deg_spec.dimension == "scale_error" and edit_spec.edit_type == "scale_change":
        if "base_font_size_px" not in deg_spec.params:
            deg_spec.params["base_font_size_px"] = edit_spec.edit_params.get("new_font_size_px", 36)

    if deg_spec.layer == "image":
        # Image-layer degradations: render ground_truth HTML as base, apply pixel ops after
        degraded_html = gt_html
    else:
        degraded_html = apply_degradation(gt_html, deg_spec)

    # Render all three
    source_path = output_dir / "source.png"
    gt_path = output_dir / "ground_truth.png"
    degraded_path = output_dir / "degraded.png"

    await renderer.render_html_string(source_html, source_path)
    await renderer.render_html_string(gt_html, gt_path)
    await renderer.render_html_string(degraded_html, degraded_path)

    # Apply image-layer degradation (pixel-level) after rendering
    if deg_spec.layer == "image":
        from src.degradation.image_layer import apply_image_degradation
        apply_image_degradation(degraded_path, deg_spec, degraded_path)

    # Write metadata
    metadata = {
        "stimulus_id": stimulus_id,
        "edit_type": edit_spec.edit_type,
        "edit_instruction": edit_spec.edit_instruction,
        "edit_id": edit_spec.edit_id,
        "degradation": {
            "dimension": deg_spec.dimension,
            "magnitude": deg_spec.magnitude,
            "element_id": deg_spec.element_id,
            "target": deg_spec.target,
            "layer": deg_spec.layer,
            "params": deg_spec.params,
        },
        "template": {
            "template_id": template.template_id,
            "difficulty_tier": template.difficulty,
            "background_type": template.background_type,
            "num_text_elements": template.num_elements,
            "font_size_range_px": list(template.font_size_range_px),
        },
        "files": {
            "source": "source.png",
            "ground_truth": "ground_truth.png",
            "degraded": "degraded.png",
        },
    }
    metadata_path = output_dir / "metadata.json"
    metadata_path.write_text(json.dumps(metadata, indent=2))

    return StimulusResult(
        stimulus_id=stimulus_id,
        output_dir=output_dir,
        source_path=source_path,
        ground_truth_path=gt_path,
        degraded_path=degraded_path,
        metadata_path=metadata_path,
    )


def generate_stimulus(
    template: Template,
    edit_spec: EditSpec,
    degradation_spec: DegradationSpec,
    output_dir: Path,
    stimulus_id: str,
    render_config: RenderConfig | None = None,
) -> StimulusResult:
    """Synchronous wrapper for generate_stimulus_async."""
    async def _run():
        async with Renderer(render_config) as renderer:
            return await generate_stimulus_async(
                renderer, template, edit_spec, degradation_spec, output_dir, stimulus_id
            )
    return asyncio.run(_run())
