#!/usr/bin/env python3
"""
generate_full.py — Full sweep stimulus generation.

Uses 2 representative edit configs (one per primary template) × all degradation
configs → ~2 × 78 = ~156 stimuli into data/full/.

Usage:
    python scripts/generate_full.py [--dry-run] [--output-dir data/full]
    python scripts/generate_full.py --all-edits [--output-dir data/full]
"""

import argparse
import asyncio
import json
import sys
from pathlib import Path

import yaml

_PROJECT_ROOT = Path(__file__).parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

# One representative edit per primary template for the "2 templates × all configs" sweep.
_REPRESENTATIVE_EDIT_IDS = {
    "color_change_easy_single_blue",
    "color_change_medium_badge_red",
}


def load_configs(all_edits: bool):
    root = Path(__file__).parent.parent
    with open(root / "configs" / "edit_types.yaml") as f:
        edit_data = yaml.safe_load(f)
    with open(root / "configs" / "degradations.yaml") as f:
        deg_data = yaml.safe_load(f)

    edit_configs = edit_data["edit_configs"]
    if not all_edits:
        edit_configs = [e for e in edit_configs if e["edit_id"] in _REPRESENTATIVE_EDIT_IDS]

    return edit_configs, deg_data["degradations"]


def build_manifest(edit_configs, degradation_configs):
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
    from src.degradation.specs import DegradationSpec
    from src.pipeline import generate_stimulus_async, EditSpec

    if dry_run:
        print(f"DRY RUN — {len(manifest)} stimuli planned:")
        dims: dict[str, int] = {}
        for item in manifest:
            dim = item["degradation"]["dimension"]
            dims[dim] = dims.get(dim, 0) + 1
            print(
                f"  {item['stimulus_id']}"
                f"  [{item['degradation']['layer']}:{dim}]"
            )
        print(f"\nTotal: {len(manifest)} stimuli")
        print("Dimension breakdown:")
        for dim, count in sorted(dims.items()):
            print(f"  {dim}: {count}")
        return

    output_dir.mkdir(parents=True, exist_ok=True)
    n = len(manifest)
    n_ok = 0
    n_fail = 0

    async with Renderer(RenderConfig()) as renderer:
        for i, item in enumerate(manifest, 1):
            stimulus_id = item["stimulus_id"]
            stim_dir = output_dir / stimulus_id

            if (stim_dir / "metadata.json").exists():
                print(f"  [{i}/{n}] SKIP (exists): {stimulus_id}")
                n_ok += 1
                continue

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
                element_id=item["edit"]["element_id"],
                layer=deg_cfg.get("layer", "html"),
                params=dict(deg_cfg.get("params", {})),
            )

            try:
                await generate_stimulus_async(
                    renderer=renderer,
                    template=template,
                    edit_spec=edit_spec,
                    degradation_spec=deg_spec,
                    output_dir=stim_dir,
                    stimulus_id=stimulus_id,
                )
                print(f"  [{i}/{n}] OK: {stimulus_id}")
                n_ok += 1
            except Exception as e:
                print(f"  [{i}/{n}] FAIL: {stimulus_id} — {e}")
                n_fail += 1

    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2))
    print(f"\nDone. {n_ok} OK, {n_fail} failed.")
    print(f"Manifest: {manifest_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--output-dir", default="data/full")
    parser.add_argument(
        "--all-edits", action="store_true",
        help="Use all edit configs (default: 2 representative edits, one per primary template)",
    )
    args = parser.parse_args()

    edit_configs, degradation_configs = load_configs(args.all_edits)
    manifest = build_manifest(edit_configs, degradation_configs)
    print(f"Manifest: {len(edit_configs)} edits × {len(degradation_configs)} degradations = {len(manifest)} stimuli")

    output_dir = Path(args.output_dir)
    asyncio.run(run_generation(manifest, output_dir, args.dry_run))


if __name__ == "__main__":
    main()
