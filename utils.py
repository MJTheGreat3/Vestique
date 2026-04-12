from __future__ import annotations
import numpy as np
from PIL import Image


def pad_crop(image: Image.Image, box: tuple[int, int, int, int], pad: float = 0.05) -> Image.Image:
    """
    Crop `image` to `box` (x1, y1, x2, y2) with percentage padding on all sides.
    """
    w, h = image.size
    x1, y1, x2, y2 = box
    pad_x = int((x2 - x1) * pad)
    pad_y = int((y2 - y1) * pad)
    x1 = max(0, x1 - pad_x)
    y1 = max(0, y1 - pad_y)
    x2 = min(w, x2 + pad_x)
    y2 = min(h, y2 + pad_y)
    return image.crop((x1, y1, x2, y2))


def pil_to_rgb(image: Image.Image) -> Image.Image:
    """Ensure image is RGB (strip alpha, convert grayscale, etc.)."""
    return image.convert("RGB")


def resize_for_display(image: Image.Image, max_width: int = 600) -> Image.Image:
    """Downscale for display without upscaling small images."""
    w, h = image.size
    if w <= max_width:
        return image
    ratio = max_width / w
    return image.resize((max_width, int(h * ratio)), Image.LANCZOS)


def normalize_embedding(vec: np.ndarray) -> np.ndarray:
    """L2-normalize a 1D numpy vector."""
    norm = np.linalg.norm(vec)
    return vec / norm if norm > 0 else vec