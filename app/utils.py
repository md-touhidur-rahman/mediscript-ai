"""
utils.py
---------
Shared utility functions for MediScript AI.
"""

from __future__ import annotations

import numpy as np
from PIL import Image


def pil_to_numpy(image: Image.Image) -> np.ndarray:
    """Convert a PIL image to a NumPy array (RGB)."""
    if image.mode != "RGB":
        image = image.convert("RGB")
    return np.array(image)


def numpy_to_pil(array: np.ndarray) -> Image.Image:
    """Convert a NumPy array to a PIL image."""
    if array.dtype != np.uint8:
        array = array.astype(np.uint8)
    return Image.fromarray(array)


def bgr_to_pil(bgr_image: np.ndarray) -> Image.Image:
    """Convert a BGR numpy array (OpenCV format) to a PIL RGB Image."""
    rgb = bgr_image[:, :, ::-1]
    return Image.fromarray(rgb.astype(np.uint8))


def ensure_rgb_pil(image) -> Image.Image:
    """Ensure image is a PIL RGB Image, converting from BGR numpy array if needed."""
    if isinstance(image, np.ndarray):
        return bgr_to_pil(image)
    if isinstance(image, Image.Image):
        return image.convert("RGB")
    raise TypeError(f"Unsupported image type: {type(image)}")


def crop_region(image: Image.Image, box: list) -> Image.Image:
    """
    Crop a region from a PIL image given a PaddleOCR 4-corner bounding box.

    Parameters
    ----------
    image : PIL.Image
    box : list of 4 [x, y] corner points (clockwise from top-left).

    Returns
    -------
    PIL.Image
    """
    xs = [p[0] for p in box]
    ys = [p[1] for p in box]
    x_min = max(0, int(min(xs)))
    y_min = max(0, int(min(ys)))
    x_max = min(image.width, int(max(xs)))
    y_max = min(image.height, int(max(ys)))
    return image.crop((x_min, y_min, x_max, y_max))


def box_to_rect(box: list) -> tuple[int, int, int, int]:
    """
    Convert a PaddleOCR 4-corner box to (x_min, y_min, x_max, y_max).
    """
    xs = [p[0] for p in box]
    ys = [p[1] for p in box]
    return int(min(xs)), int(min(ys)), int(max(xs)), int(max(ys))


def ensure_min_size(image: Image.Image, min_h: int = 32, min_w: int = 32) -> Image.Image:
    """
    Pad a small crop to at least min_h x min_w so TrOCR doesn't fail.
    """
    w, h = image.size
    if h >= min_h and w >= min_w:
        return image

    new_w = max(w, min_w)
    new_h = max(h, min_h)

    padded = Image.new("RGB", (new_w, new_h), (255, 255, 255))
    padded.paste(image, (0, 0))
    return padded


def scale_image_for_display(
    image: Image.Image,
    max_width: int = 700,
    max_height: int = 700,
) -> Image.Image:
    """
    Resize image for UI display only.

    This does NOT affect OCR processing. It only keeps the image preview
    smaller and cleaner inside Streamlit.

    Parameters
    ----------
    image : PIL.Image
        Input image.
    max_width : int
        Maximum display width.
    max_height : int
        Maximum display height.

    Returns
    -------
    PIL.Image
        Resized copy for display, preserving aspect ratio.
    """
    w, h = image.size

    if w <= max_width and h <= max_height:
        return image

    scale_w = max_width / w
    scale_h = max_height / h
    scale = min(scale_w, scale_h)

    new_w = max(1, int(w * scale))
    new_h = max(1, int(h * scale))

    return image.resize((new_w, new_h), Image.LANCZOS)


def preprocess_for_ocr(image: Image.Image, max_dim: int = 1600) -> Image.Image:
    """
    Preprocess an image for better OCR performance:
    - Downscale if too large (speeds up PaddleOCR significantly)
    - Convert to RGB
    - Enhance contrast slightly

    Parameters
    ----------
    image : PIL.Image
    max_dim : int
        Maximum width or height in pixels. Default 1600.

    Returns
    -------
    PIL.Image
    """
    from PIL import ImageEnhance

    image = image.convert("RGB")

    # Downscale if image is very large
    w, h = image.size
    if max(w, h) > max_dim:
        scale = max_dim / max(w, h)
        image = image.resize((int(w * scale), int(h * scale)), Image.LANCZOS)

    # Mild contrast boost helps OCR on faint handwriting
    enhancer = ImageEnhance.Contrast(image)
    image = enhancer.enhance(1.3)

    return image