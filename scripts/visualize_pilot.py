#!/usr/bin/env python3
"""
visualize_pilot.py — Side-by-side grid images for the pilot stimulus set.

Each output image shows:
  Source | Ground Truth | Degraded
with column labels, a text panel below containing the edit instruction
and degradation info, and a metadata footer.

All grids are written into a single flat output directory.

Usage:
    python scripts/visualize_pilot.py [--pilot-dir data/pilot] [--output-dir data/pilot_grids] [--width 2400]
"""

from __future__ import annotations

import json
import sys
import textwrap
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


# ---------------------------------------------------------------------------
# Constants / tunables
# ---------------------------------------------------------------------------

_LABEL_FONT_SIZE = 20
_INFO_FONT_SIZE = 17
_LABEL_HEIGHT = 32
_TEXT_SIDE_PAD = 22
_TEXT_TOP_PAD = 14
_TEXT_BOTTOM_PAD = 14
_LINE_SPACING = 5
_DIVIDER_HEIGHT = 3
_SECTION_GAP = 10          # extra vertical gap between instruction and degradation sections

_BG_COLOR = (255, 255, 255)
_LABEL_BG_COLOR = (235, 235, 235)
_TEXT_BG_COLOR = (245, 245, 250)
_TEXT_COLOR = (30, 30, 30)
_MUTED_COLOR = (100, 100, 120)
_DIVIDER_COLOR = (200, 200, 200)
_LABEL_TEXT_COLOR = (70, 70, 70)
_COL_DIVIDER_COLOR = (180, 180, 180)

_COLUMNS = ["Source", "Ground Truth", "Degraded"]


# ---------------------------------------------------------------------------
# Font helpers
# ---------------------------------------------------------------------------

def _load_font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    candidates_bold = [
        "DejaVuSans-Bold.ttf",
        "Arial Bold.ttf",
        "arialbd.ttf",
        "LiberationSans-Bold.ttf",
    ]
    candidates_regular = [
        "DejaVuSans.ttf",
        "Arial.ttf",
        "arial.ttf",
        "Helvetica.ttf",
        "LiberationSans-Regular.ttf",
    ]
    for name in (candidates_bold if bold else candidates_regular):
        try:
            return ImageFont.truetype(name, size)
        except (IOError, OSError):
            pass
    return ImageFont.load_default()


# ---------------------------------------------------------------------------
# Core function
# ---------------------------------------------------------------------------

def make_stimulus_grid(
    source_path: Path,
    ground_truth_path: Path,
    degraded_path: Path,
    metadata: dict,
    output_path: Path,
    target_width: int | None = None,
) -> Path:
    """
    Compose a three-column grid image (source | ground truth | degraded)
    with an info panel below and save it to output_path.
    """
    imgs_raw = [
        Image.open(source_path).convert("RGB"),
        Image.open(ground_truth_path).convert("RGB"),
        Image.open(degraded_path).convert("RGB"),
    ]

    # Determine per-column width
    if target_width is not None:
        col_w = target_width // 3
    else:
        col_w = imgs_raw[0].width

    def _fit(img: Image.Image, width: int) -> Image.Image:
        ratio = width / img.width
        return img.resize((width, int(img.height * ratio)), Image.LANCZOS)

    imgs = [_fit(img, col_w) for img in imgs_raw]
    img_h = max(img.height for img in imgs)
    total_w = col_w * 3

    label_font = _load_font(_LABEL_FONT_SIZE, bold=True)
    info_font = _load_font(_INFO_FONT_SIZE)
    info_bold = _load_font(_INFO_FONT_SIZE, bold=True)

    # ---------------------------------------------------------------------------
    # Build text panel content
    # ---------------------------------------------------------------------------
    edit_instruction = metadata.get("edit_instruction", "")
    deg = metadata.get("degradation", {})
    deg_dimension = deg.get("dimension", "")
    deg_magnitude = deg.get("magnitude", "")
    deg_layer = deg.get("layer", "")
    deg_element = deg.get("element_id", "")
    deg_params = deg.get("params", {})

    edit_type = metadata.get("edit_type", "")
    stimulus_id = metadata.get("stimulus_id", "")

    # Format degradation info line
    deg_parts = []
    if deg_dimension:
        deg_parts.append(f"dimension: {deg_dimension}")
    if deg_magnitude:
        deg_parts.append(f"magnitude: {deg_magnitude}")
    if deg_layer:
        deg_parts.append(f"layer: {deg_layer}")
    if deg_element:
        deg_parts.append(f"element: {deg_element}")
    if deg_params:
        params_str = ", ".join(f"{k}={v}" for k, v in deg_params.items())
        deg_parts.append(f"params: {params_str}")
    degradation_line = "  |  ".join(deg_parts) if deg_parts else "(no degradation)"

    id_line = f"ID: {stimulus_id}  |  edit_type: {edit_type}"

    # Measure text heights using a dummy draw
    _dummy = Image.new("RGB", (1, 1))
    _draw = ImageDraw.Draw(_dummy)

    usable_w = total_w - 2 * _TEXT_SIDE_PAD
    avg_w_bbox = _draw.textbbox((0, 0), "M", font=info_font)
    avg_char_w = max(avg_w_bbox[2] - avg_w_bbox[0], 1)
    chars_per_line = max(usable_w // avg_char_w, 20)

    def _wrap(text: str) -> list[str]:
        lines = textwrap.wrap(text, width=chars_per_line)
        return lines if lines else [""]

    instruction_lines = _wrap(f"Instruction: {edit_instruction}")
    degradation_lines = _wrap(f"Degradation: {degradation_line}")
    id_lines = _wrap(id_line)

    lh_bbox = _draw.textbbox((0, 0), "Ag", font=info_font)
    line_h = (lh_bbox[3] - lh_bbox[1]) + _LINE_SPACING

    all_lines_count = (
        len(instruction_lines)
        + len(degradation_lines)
        + len(id_lines)
        + 2  # 2 section gaps
    )
    text_panel_h = _TEXT_TOP_PAD + all_lines_count * line_h + _SECTION_GAP + _TEXT_BOTTOM_PAD

    total_h = _LABEL_HEIGHT + img_h + _DIVIDER_HEIGHT + text_panel_h

    canvas = Image.new("RGB", (total_w, total_h), _BG_COLOR)
    draw = ImageDraw.Draw(canvas)

    # ---------------------------------------------------------------------------
    # Label bar
    # ---------------------------------------------------------------------------
    draw.rectangle([(0, 0), (total_w, _LABEL_HEIGHT)], fill=_LABEL_BG_COLOR)
    for i, label in enumerate(_COLUMNS):
        x_off = i * col_w
        bbox = draw.textbbox((0, 0), label, font=label_font)
        lw = bbox[2] - bbox[0]
        lh = bbox[3] - bbox[1]
        x = x_off + (col_w - lw) // 2
        y = (_LABEL_HEIGHT - lh) // 2
        draw.text((x, y), label, font=label_font, fill=_LABEL_TEXT_COLOR)

    # Column dividers in label bar
    for i in [1, 2]:
        draw.line([(i * col_w, 0), (i * col_w, _LABEL_HEIGHT)], fill=_COL_DIVIDER_COLOR, width=1)

    # ---------------------------------------------------------------------------
    # Images
    # ---------------------------------------------------------------------------
    img_y = _LABEL_HEIGHT
    for i, img in enumerate(imgs):
        canvas.paste(img, (i * col_w, img_y))

    # Column dividers between images
    for i in [1, 2]:
        draw.line(
            [(i * col_w, img_y), (i * col_w, img_y + img_h)],
            fill=_COL_DIVIDER_COLOR, width=1,
        )

    # ---------------------------------------------------------------------------
    # Divider between images and text panel
    # ---------------------------------------------------------------------------
    divider_y = img_y + img_h
    draw.rectangle(
        [(0, divider_y), (total_w, divider_y + _DIVIDER_HEIGHT)],
        fill=_DIVIDER_COLOR,
    )

    # ---------------------------------------------------------------------------
    # Text panel
    # ---------------------------------------------------------------------------
    text_y0 = divider_y + _DIVIDER_HEIGHT
    draw.rectangle([(0, text_y0), (total_w, total_h)], fill=_TEXT_BG_COLOR)

    y = text_y0 + _TEXT_TOP_PAD

    def _draw_lines(lines: list[str], font, color=_TEXT_COLOR):
        nonlocal y
        for line in lines:
            draw.text((_TEXT_SIDE_PAD, y), line, font=font, fill=color)
            y += line_h

    _draw_lines(instruction_lines, info_font)
    y += _SECTION_GAP // 2
    _draw_lines(degradation_lines, info_font, color=_MUTED_COLOR)
    y += _SECTION_GAP // 2
    _draw_lines(id_lines, info_font, color=_MUTED_COLOR)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output_path)
    return output_path


# ---------------------------------------------------------------------------
# Batch helper
# ---------------------------------------------------------------------------

def make_pilot_grids(
    pilot_dir: Path,
    output_dir: Path,
    target_width: int | None = None,
) -> list[Path]:
    """
    Scan pilot_dir for stimulus subdirectories, generate a grid for each,
    and write all grids into output_dir.

    Each stimulus directory must contain:
        source.png, ground_truth.png, degraded.png, metadata.json
    """
    pilot_dir = Path(pilot_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    written: list[Path] = []
    skipped = 0

    stimulus_dirs = sorted(
        d for d in pilot_dir.iterdir()
        if d.is_dir() and (d / "metadata.json").exists()
    )

    if not stimulus_dirs:
        print(f"[visualize] No stimulus directories found in {pilot_dir}", file=sys.stderr)
        return written

    for stim_dir in stimulus_dirs:
        stimulus_id = stim_dir.name
        source_path = stim_dir / "source.png"
        gt_path = stim_dir / "ground_truth.png"
        degraded_path = stim_dir / "degraded.png"
        metadata_path = stim_dir / "metadata.json"

        missing = [p.name for p in [source_path, gt_path, degraded_path] if not p.exists()]
        if missing:
            print(
                f"  [visualize] [{stimulus_id}] SKIP — missing: {', '.join(missing)}",
                file=sys.stderr,
            )
            skipped += 1
            continue

        with metadata_path.open("r", encoding="utf-8") as f:
            metadata = json.load(f)

        out_path = output_dir / f"{stimulus_id}_grid.png"
        make_stimulus_grid(
            source_path=source_path,
            ground_truth_path=gt_path,
            degraded_path=degraded_path,
            metadata=metadata,
            output_path=out_path,
            target_width=target_width,
        )
        print(f"  [visualize] {stimulus_id} -> {out_path.name}")
        written.append(out_path)

    return written


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Generate three-column grid images for all pilot stimuli.",
    )
    parser.add_argument(
        "--pilot-dir", type=Path, default=Path("data/pilot"),
        help="Directory containing stimulus subdirectories (default: data/pilot)",
    )
    parser.add_argument(
        "--output-dir", type=Path, default=Path("data/pilot_grids"),
        help="Directory to write grid PNGs into (default: data/pilot_grids)",
    )
    parser.add_argument(
        "--width", type=int, default=None,
        help="Total output image width in pixels (default: 3 × source image width)",
    )
    args = parser.parse_args()

    paths = make_pilot_grids(
        pilot_dir=args.pilot_dir,
        output_dir=args.output_dir,
        target_width=args.width,
    )
    print(f"\nDone. {len(paths)} grid image(s) written to {args.output_dir}")
