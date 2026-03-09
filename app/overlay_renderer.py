"""
overlay_renderer.py
--------------------
Google Lens-style overlay renderer for MediScript AI.

Behavior:
  1. Uses filtered detector boxes directly.
  2. Expands each box slightly for clean erase.
  3. Samples local paper color.
  4. Draws typed text INSIDE the same region.
  5. Hard-clips long text so it never spills outside the box.
"""

from __future__ import annotations

from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from ocr_pipeline import PipelineResult, RegionResult


# ---------------------------------------------------------------------------
# Font loader
# ---------------------------------------------------------------------------

def _load_font(size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    candidates = [
        "/Library/Fonts/Arial.ttf",
        "/System/Library/Fonts/Supplemental/Arial.ttf",
        "/Library/Fonts/Helvetica.ttc",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
        "C:/Windows/Fonts/arial.ttf",
        str(Path(__file__).parent.parent / "fonts" / "DejaVuSans.ttf"),
    ]
    for path in candidates:
        if Path(path).is_file():
            try:
                return ImageFont.truetype(path, size)
            except Exception:
                continue
    return ImageFont.load_default()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _sample_bg(image: Image.Image, x0, y0, x1, y1) -> tuple:
    arr = np.array(image.convert("RGB"))
    h, w = arr.shape[:2]
    strip = 16
    samples = []

    if y0 - strip >= 0:
        samples.append(arr[max(0, y0 - strip):y0, max(0, x0):min(w, x1)])
    if y1 + strip <= h:
        samples.append(arr[y1:min(h, y1 + strip), max(0, x0):min(w, x1)])
    if x0 - strip >= 0:
        samples.append(arr[max(0, y0):min(h, y1), max(0, x0 - strip):x0])
    if x1 + strip <= w:
        samples.append(arr[max(0, y0):min(h, y1), x1:min(w, x1 + strip)])

    if not samples:
        return (255, 255, 255)

    usable = [s.reshape(-1, 3) for s in samples if s.size > 0]
    if not usable:
        return (255, 255, 255)

    all_px = np.concatenate(usable, axis=0)
    bg = np.percentile(all_px, 85, axis=0).astype(int)
    return int(bg[0]), int(bg[1]), int(bg[2])


def _text_bbox(draw: ImageDraw.ImageDraw, text: str, font) -> tuple[int, int]:
    try:
        bb = draw.textbbox((0, 0), text, font=font)
        return bb[2] - bb[0], bb[3] - bb[1]
    except Exception:
        size = getattr(font, "size", 12)
        return len(text) * max(5, size // 2), size


def _fit_font(draw: ImageDraw.ImageDraw, text: str, box_w: int, box_h: int):
    start = max(10, int(box_h * 0.72))
    font = _load_font(start)

    for size in range(start, 7, -1):
        font = _load_font(size)
        tw, th = _text_bbox(draw, text, font)
        if tw <= max(8, box_w) and th <= max(8, box_h):
            return font

    return _load_font(8)


def _clip_text_to_width(draw: ImageDraw.ImageDraw, text: str, font, max_w: int) -> str:
    text = (text or "").replace("\n", " ").strip()
    if not text:
        return ""

    while len(text) > 1:
        tw, _ = _text_bbox(draw, text, font)
        if tw <= max_w:
            return text
        text = text[:-1].rstrip()

    return text


def _expand_rect(rect, iw: int, ih: int, pad_x: int = 6, pad_y: int = 4):
    x0, y0, x1, y1 = rect
    return (
        max(0, x0 - pad_x),
        max(0, y0 - pad_y),
        min(iw, x1 + pad_x),
        min(ih, y1 + pad_y),
    )


def _valid_region(region: RegionResult, iw: int, ih: int) -> bool:
    x0, y0, x1, y1 = region.rect
    if x1 <= x0 or y1 <= y0:
        return False
    if x0 < 0 or y0 < 0 or x1 > iw or y1 > ih:
        return False
    if (x1 - x0) < 12 or (y1 - y0) < 8:
        return False
    if not str(region.final_text or "").strip():
        return False
    return True


# ---------------------------------------------------------------------------
# Renderer
# ---------------------------------------------------------------------------

class OverlayRenderer:
    """Google Lens-style typed overlay renderer."""

    def __init__(self, text_color=(10, 10, 40), padding=5):
        self.text_color = text_color
        self.padding = padding

    def _render_region(self, output: Image.Image, draw: ImageDraw.ImageDraw, region: RegionResult) -> None:
        iw, ih = output.size
        x0, y0, x1, y1 = _expand_rect(region.rect, iw, ih, pad_x=self.padding, pad_y=self.padding)

        box_w = max(1, x1 - x0)
        box_h = max(1, y1 - y0)

        text = str(region.final_text or "").replace("\n", " ").strip()
        if not text:
            return

        bg = _sample_bg(output, x0, y0, x1, y1)
        draw.rectangle([x0, y0, x1, y1], fill=bg)

        font = _fit_font(draw, text, box_w - 4, box_h - 2)
        text = _clip_text_to_width(draw, text, font, box_w - 4)
        if not text:
            return

        _, th = _text_bbox(draw, text, font)
        tx = x0 + 2
        ty = y0 + max(0, (box_h - th) // 2)

        draw.text((tx, ty), text, fill=self.text_color, font=font)

    def render(self, image: Image.Image, result: PipelineResult) -> Image.Image:
        output = image.convert("RGB").copy()

        if not result.regions:
            return output

        iw, ih = output.size
        draw = ImageDraw.Draw(output)

        for region in sorted(result.regions, key=lambda r: (r.rect[1], r.rect[0])):
            if _valid_region(region, iw, ih):
                self._render_region(output, draw, region)

        return output

    def render_with_boxes(
        self,
        image: Image.Image,
        result: PipelineResult,
        box_color=(50, 180, 255),
        line_width=2,
    ) -> Image.Image:
        rendered = self.render(image, result)
        draw = ImageDraw.Draw(rendered)
        iw, ih = rendered.size

        for r in sorted(result.regions, key=lambda r: (r.rect[1], r.rect[0])):
            if _valid_region(r, iw, ih):
                x0, y0, x1, y1 = _expand_rect(r.rect, iw, ih, pad_x=self.padding, pad_y=self.padding)
                draw.rectangle([x0, y0, x1, y1], outline=box_color, width=line_width)

        return rendered