#!/usr/bin/env python3
"""
generate_pilot.py — Generates the pilot stimulus set.

Pilot: 2 templates × 4 edit types × 3 degradation magnitudes = up to 24 stimuli
(limited by available edit_configs in configs/edit_types.yaml).

Usage:
    python scripts/generate_pilot.py [--dry-run] [--output-dir data/pilot]
"""

import argparse
import asyncio
import json
import sys
from pathlib import Path

import yaml

# Ensure project root is on sys.path when running as a script
_PROJECT_ROOT = Path(__file__).parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))


def load_configs():
    root = Path(__file__).parent.parent
    with open(root / "configs" / "edit_types.yaml") as f:
        edit_data = yaml.safe_load(f)
    with open(root / "configs" / "degradations.yaml") as f:
        deg_data = yaml.safe_load(f)
    return edit_data["edit_configs"], deg_data["degradations"]


def build_manifest(edit_configs, degradation_configs):
    """Build list of (edit_config, degradation_config, stimulus_id) tuples."""
    manifest = []
    for edit in edit_configs:
        for deg in degradation_configs:
            stimulus_id = f"{edit['edit_id']}__{deg['id']}"
            manifest.append({
                "stimulus_id": stimulus_id,
                "edit": edit,
                "degradation": deg,
            })
    return manifest


async def run_generation(manifest, output_dir: Path, dry_run: bool):
    from src.render import Renderer, RenderConfig
    from src.templates.templates import get_template
    from src.edit_applicator import apply_correct_edit
    from src.degradation.engine import apply_degradation
    from src.degradation.specs import DegradationSpec
    from src.pipeline import generate_stimulus_async, EditSpec

    if dry_run:
        print(f"DRY RUN — {len(manifest)} stimuli planned:")
        for item in manifest:
            print(f"  {item['stimulus_id']}")
            print(f"    edit: {item['edit']['edit_type']} on {item['edit']['template_id']}.{item['edit']['element_id']}")
            print(f"    degradation: {item['degradation']['dimension']} @ {item['degradation']['magnitude']}")
        return

    output_dir.mkdir(parents=True, exist_ok=True)
    async with Renderer(RenderConfig()) as renderer:
        for item in manifest:
            stimulus_id = item["stimulus_id"]
            stim_dir = output_dir / stimulus_id

            template = get_template(item["edit"]["template_id"])

            edit_spec = EditSpec(
                edit_id=item["edit"]["edit_id"],
                edit_type=item["edit"]["edit_type"],
                template_id=item["edit"]["template_id"],
                element_id=item["edit"]["element_id"],
                edit_params=item["edit"]["edit_params"],
                edit_instruction=item["edit"]["edit_instruction"],
            )

            deg_cfg = item["degradation"]
            deg_spec = DegradationSpec(
                dimension=deg_cfg["dimension"],
                magnitude=deg_cfg["magnitude"],
                element_id=item["edit"]["element_id"],  # target same element as edit
                params=dict(deg_cfg.get("params", {})),
            )

            result = await generate_stimulus_async(
                renderer=renderer,
                template=template,
                edit_spec=edit_spec,
                degradation_spec=deg_spec,
                output_dir=stim_dir,
                stimulus_id=stimulus_id,
            )
            print(f"  OK: {stimulus_id}")

    # Write manifest summary
    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2))
    print(f"\nWrote {len(manifest)} stimuli to {output_dir}")
    print(f"Manifest: {manifest_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--output-dir", default="data/pilot")
    args = parser.parse_args()

    edit_configs, degradation_configs = load_configs()
    manifest = build_manifest(edit_configs, degradation_configs)

    output_dir = Path(args.output_dir)
    asyncio.run(run_generation(manifest, output_dir, args.dry_run))


if __name__ == "__main__":
    main()
