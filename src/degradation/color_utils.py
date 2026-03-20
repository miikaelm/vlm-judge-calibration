import numpy as np
from skimage.color import rgb2lab, deltaE_ciede2000, lab2rgb


def hex_to_rgb(hex_color: str) -> tuple[int, int, int]:
    h = hex_color.lstrip("#")
    if len(h) != 6:
        raise ValueError(f"Invalid hex color: {hex_color}")
    return (int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16))


def rgb_to_hex(rgb: tuple[int, int, int]) -> str:
    return f"#{rgb[0]:02X}{rgb[1]:02X}{rgb[2]:02X}"


def compute_delta_e(rgb1: tuple[int, int, int], rgb2: tuple[int, int, int]) -> float:
    lab1 = rgb2lab(np.array([[rgb1]], dtype=np.float64) / 255.0)
    lab2 = rgb2lab(np.array([[rgb2]], dtype=np.float64) / 255.0)
    return float(deltaE_ciede2000(lab1, lab2)[0, 0])


def offset_color_by_delta_e(base_hex: str, target_delta_e: float) -> str:
    """
    Shift a color in LAB space by approximately target_delta_e units.
    Shifts along the L* axis (lightness). Returns new hex color.
    """
    base_rgb = hex_to_rgb(base_hex)
    lab = rgb2lab(np.array([[base_rgb]], dtype=np.float64) / 255.0)[0, 0]

    # Try shifting L* up first, then down if that clips
    new_rgb = base_rgb
    for sign in [1, -1]:
        new_lab = lab.copy()
        new_lab[0] = np.clip(lab[0] + sign * target_delta_e, 0, 100)
        # Convert back to RGB
        new_rgb_f = lab2rgb(np.array([[new_lab]]))[0, 0]
        new_rgb = tuple(int(np.clip(c * 255, 0, 255)) for c in new_rgb_f)
        achieved = compute_delta_e(base_rgb, new_rgb)
        if achieved >= target_delta_e * 0.5:  # achieved at least half the target
            return rgb_to_hex(new_rgb)

    return rgb_to_hex(new_rgb)
