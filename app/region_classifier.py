from __future__ import annotations

import numpy as np

from config import OCRConfig
from utils import rect_size


def classify_region(
    crop_bgr: np.ndarray,
    rect,
    preliminary_text: str | None,
    cfg: OCRConfig,
) -> str:
    w, h = rect_size(rect)

    if w * h < cfg.ignore_small_area:
        return "ignore"

    if w < cfg.min_box_width or h < cfg.min_box_height:
        return "ignore"

    # Ignore only truly tiny regions
    if h < 12 or w < 18:
        return "ignore"

    # For now, keep classification permissive so handwriting is not suppressed.
    # We can add a better handwritten-vs-printed classifier later.
    return "handwritten"