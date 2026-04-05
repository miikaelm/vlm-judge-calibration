"""
image_layer.py — Image-layer degradations applied after rendering.

These operate on the rendered PNG directly, unlike HTML-layer degradations
which modify the HTML before rendering. Used for noise, JPEG artifacts, and blur.
"""

from __future__ import annotations

import io
from pathlib import Path

import numpy as np
from PIL import Image, ImageFilter

from src.degradation.specs import DegradationSpec


def apply_gaussian_noise(image_path: Path, sigma: float, output_path: Path) -> None:
    """Add Gaussian noise with standard deviation sigma to all channels."""
    img = Image.open(image_path).convert("RGB")
    arr = np.array(img, dtype=np.float32)
    rng = np.random.default_rng(seed=42)  # deterministic seed
    noise = rng.normal(0, sigma, arr.shape).astype(np.float32)
    noisy = np.clip(arr + noise, 0, 255).astype(np.uint8)
    Image.fromarray(noisy).save(output_path, "PNG")


def apply_jpeg_compression(image_path: Path, quality: int, output_path: Path) -> None:
    """Re-encode as JPEG at given quality (1–95), then save as PNG."""
    img = Image.open(image_path).convert("RGB")
    buf = io.BytesIO()
    img.save(buf, "JPEG", quality=quality)
    buf.seek(0)
    Image.open(buf).convert("RGB").save(output_path, "PNG")


def apply_blur(image_path: Path, radius: float, output_path: Path) -> None:
    """Apply Gaussian blur with the given radius in pixels."""
    img = Image.open(image_path).convert("RGB")
    blurred = img.filter(ImageFilter.GaussianBlur(radius=radius))
    blurred.save(output_path, "PNG")


_IMAGE_HANDLERS: dict[str, object] = {
    "gaussian_noise": apply_gaussian_noise,
    "jpeg_compression": apply_jpeg_compression,
    "blur": apply_blur,
}


def apply_image_degradation(image_path: Path, spec: DegradationSpec, output_path: Path) -> None:
    """
    Dispatch to the appropriate image-layer degradation function.

    image_path and output_path may be the same file (in-place modification).
    """
    handler = _IMAGE_HANDLERS.get(spec.dimension)
    if handler is None:
        raise ValueError(
            f"Unknown image-layer degradation dimension: {spec.dimension!r}. "
            f"Known: {list(_IMAGE_HANDLERS)}"
        )
    if spec.dimension == "gaussian_noise":
        apply_gaussian_noise(image_path, spec.params["sigma"], output_path)
    elif spec.dimension == "jpeg_compression":
        apply_jpeg_compression(image_path, int(round(spec.params["quality"])), output_path)
    elif spec.dimension == "blur":
        apply_blur(image_path, spec.params["radius"], output_path)
