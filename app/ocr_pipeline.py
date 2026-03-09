"""
ocr_pipeline.py
----------------
Main OCR pipeline for MediScript AI.

Recognition backends (selectable):
  - "paddle"  : PaddleOCR built-in recognition (fast, good for print)
  - "claude"  : Claude vision API (best for cursive handwriting, ~2-3s total)
  - "trocr"   : TrOCR transformer model (slow, avoid for long documents)

Detection is always done by PaddleOCR.
"""

from __future__ import annotations

import base64
import io
import os
from dataclasses import dataclass, field
from typing import Optional, Literal

import numpy as np
from PIL import Image

from utils import crop_region, box_to_rect, ensure_min_size, preprocess_for_ocr
from prescription_logic import PrescriptionCorrector


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class RegionResult:
    """Result for one detected text region."""
    box: list
    rect: tuple
    raw_text: str
    final_text: str
    corrections: list = field(default_factory=list)
    confidence: float = 0.0


@dataclass
class PipelineResult:
    """Aggregated result for the full image."""
    regions: list[RegionResult]
    full_text: str
    mode: str
    image_size: tuple


# ---------------------------------------------------------------------------
# Claude vision recognizer
# ---------------------------------------------------------------------------

def _pil_to_base64(image: Image.Image) -> str:
    """Convert PIL image to base64 JPEG string."""
    buf = io.BytesIO()
    image.convert("RGB").save(buf, format="JPEG", quality=90)
    return base64.b64encode(buf.getvalue()).decode("utf-8")


def _claude_recognize_batch(crops: list[Image.Image], api_key: str) -> list[str]:
    """
    Send all crops to Claude in a single API call for fast batch recognition.
    Returns list of recognized text strings, one per crop.
    """
    import anthropic

    client = anthropic.Anthropic(api_key=api_key)

    # Build content: all crops as images + single instruction
    content = []
    for i, crop in enumerate(crops):
        content.append({
            "type": "text",
            "text": f"Region {i + 1}:"
        })
        content.append({
            "type": "image",
            "source": {
                "type": "base64",
                "media_type": "image/jpeg",
                "data": _pil_to_base64(crop),
            }
        })

    content.append({
        "type": "text",
        "text": (
            "Each image above is a cropped region of handwritten text. "
            "For each region, transcribe ONLY the handwritten text exactly as written. "
            "Respond with ONLY a JSON array of strings in order, e.g.: "
            '[\"text 1\", \"text 2\", \"text 3\"]. '
            "No explanation, no markdown, just the JSON array."
        )
    })

    response = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=1024,
        messages=[{"role": "user", "content": content}]
    )

    raw = response.content[0].text.strip()

    # Parse JSON array
    import json
    try:
        # Strip markdown fences if present
        if raw.startswith("```"):
            raw = raw.split("```")[1]
            if raw.startswith("json"):
                raw = raw[4:]
        results = json.loads(raw.strip())
        if isinstance(results, list):
            return [str(r) for r in results]
    except Exception:
        pass

    # Fallback: split by newlines if JSON parse fails
    return [line.strip() for line in raw.split("\n") if line.strip()]


def _claude_recognize_full_image(image: Image.Image, api_key: str) -> str:
    """
    Send the full image to Claude for whole-document transcription.
    """
    import anthropic
    client = anthropic.Anthropic(api_key=api_key)
    response = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=2048,
        messages=[{
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": "image/jpeg",
                        "data": _pil_to_base64(image),
                    }
                },
                {
                    "type": "text",
                    "text": (
                        "Transcribe ALL handwritten text in this image exactly as written, "
                        "preserving line breaks. Output ONLY the transcribed text, "
                        "no explanation or commentary."
                    )
                }
            ]
        }]
    )
    return response.content[0].text.strip()


def _claude_detect_and_transcribe(image: Image.Image, api_key: str) -> list[dict]:
    """
    Ask Claude to detect text lines AND return their bounding boxes.
    Returns list of dicts: {text, x0, y0, x1, y1}
    Coordinates are fractions of image width/height (0.0-1.0).
    """
    import anthropic, json
    client = anthropic.Anthropic(api_key=api_key)
    w, h = image.size

    response = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=2048,
        messages=[{
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": "image/jpeg",
                        "data": _pil_to_base64(image),
                    }
                },
                {
                    "type": "text",
                    "text": (
                        f"This image is {w} pixels wide and {h} pixels tall. "
                        "Find every line of handwritten text. "
                        "For each line give the transcription and its bounding box in PIXELS (integer pixel coordinates). "
                        "x0,y0 = top-left corner. x1,y1 = bottom-right corner of that line. "
                        "Be precise — measure where the ink actually starts and ends on each line. "
                        "Respond ONLY with a JSON array, no markdown, no explanation: "
                        '[{"text": "Hello world", "x0": 120, "y0": 145, "x1": 890, "y1": 198}, ...]'
                    )
                }
            ]
        }]
    )

    raw = response.content[0].text.strip()
    try:
        if "```" in raw:
            raw = raw.split("```")[1]
            if raw.startswith("json"):
                raw = raw[4:]
        items = json.loads(raw.strip())
        result = []
        for item in items:
            x0 = int(item.get("x0", 0))
            y0 = int(item.get("y0", 0))
            x1 = int(item.get("x1", w))
            y1 = int(item.get("y1", h))
            # Clamp to image bounds
            x0 = max(0, min(x0, w))
            y0 = max(0, min(y0, h))
            x1 = max(x0+1, min(x1, w))
            y1 = max(y0+1, min(y1, h))
            result.append({
                "text": item.get("text", ""),
                "x0": x0, "y0": y0, "x1": x1, "y1": y1,
            })
        return result
    except Exception:
        # Fallback: return full transcription as single block
        text = raw.strip()
        return [{"text": text, "x0": 0, "y0": 0, "x1": w, "y1": h}]


# ---------------------------------------------------------------------------
# Pipeline class
# ---------------------------------------------------------------------------

class OCRPipeline:
    """
    End-to-end OCR pipeline.

    Parameters
    ----------
    recognizer : HandwritingRecognizer | None
        TrOCR recognizer (only used when backend='trocr').
    corrector : PrescriptionCorrector | None
        Only passed for prescription mode.
    paddle_lang : str
        Language for PaddleOCR detection (default 'en').
    backend : str
        Recognition backend: 'paddle' | 'claude' | 'trocr'
    anthropic_api_key : str | None
        Required when backend='claude'. Falls back to ANTHROPIC_API_KEY env var.
    """

    def __init__(
        self,
        recognizer=None,
        corrector: Optional[PrescriptionCorrector] = None,
        paddle_lang: str = "en",
        backend: Literal["paddle", "claude", "trocr"] = "claude",
        anthropic_api_key: Optional[str] = None,
    ) -> None:
        self.recognizer = recognizer
        self.corrector = corrector
        self._paddle = None
        self._paddle_lang = paddle_lang
        self.backend = backend
        self.api_key = anthropic_api_key or os.environ.get("ANTHROPIC_API_KEY", "")

    # ------------------------------------------------------------------
    # PaddleOCR lazy loader
    # ------------------------------------------------------------------

    def _get_paddle(self):
        if self._paddle is None:
            from paddleocr import PaddleOCR
            self._paddle = PaddleOCR(lang=self._paddle_lang)
        return self._paddle

    # ------------------------------------------------------------------
    # Detection via PaddleOCR
    # ------------------------------------------------------------------

    def _run_paddle_detect(self, image: Image.Image) -> list[dict]:
        """
        Run PaddleOCR detection only, return boxes + paddle's own text.
        """
        paddle = self._get_paddle()
        proc_image = preprocess_for_ocr(image)
        np_image = np.array(proc_image)
        result = paddle.ocr(np_image)

        regions = []
        if not result:
            return regions

        for item in result:
            try:
                boxes  = item.get("dt_polys")  or []
                texts  = item.get("rec_texts")  or []
                scores = item.get("rec_scores") or []

                for i, box in enumerate(boxes):
                    text  = texts[i]  if i < len(texts)  else ""
                    score = float(scores[i]) if i < len(scores) else 1.0
                    if hasattr(box, "tolist"):
                        box = box.tolist()
                    # Skip low-confidence detections (false positives)
                    if score < 0.75:
                        continue
                    # Skip boxes with no meaningful text
                    if not text or not text.strip():
                        continue
                    # Skip boxes shorter than 10px (noise)
                    if hasattr(box, "__len__") and len(box) == 4:
                        ys_box = [pt[1] for pt in box]
                        if max(ys_box) - min(ys_box) < 10:
                            continue
                    regions.append({
                        "box": box,
                        "paddle_text": text,
                        "confidence": score,
                    })
            except Exception:
                pass

        return regions

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def run(
        self,
        image: Image.Image,
        mode: str = "general",
        progress_callback=None,
    ) -> PipelineResult:
        """
        Run the full pipeline on an image.

        For 'claude' backend: sends full image to Claude in one call (fast).
        For 'paddle' backend: uses PaddleOCR text directly (fast).
        For 'trocr' backend: runs TrOCR on each crop (slow).
        """

        # ---- Claude mode: Claude detects lines AND transcribes (no PaddleOCR boxes) ----
        if self.backend == "claude" and self.api_key and self.api_key.strip():
            if progress_callback:
                progress_callback(0, 1)

            # Claude returns per-line text + bounding boxes
            claude_lines = _claude_detect_and_transcribe(image, self.api_key)

            region_results: list[RegionResult] = []
            all_texts = []

            for item in claude_lines:
                text = item["text"]
                x0, y0, x1, y1 = item["x0"], item["y0"], item["x1"], item["y1"]
                # Build a 4-corner box compatible with box_to_rect
                box = [[x0,y0],[x1,y0],[x1,y1],[x0,y1]]
                rect = (x0, y0, x1, y1)

                if mode == "prescription" and self.corrector is not None:
                    final_text, corrections = self.corrector.correct_text(text)
                else:
                    final_text = text
                    corrections = []

                all_texts.append(final_text)
                region_results.append(RegionResult(
                    box=box, rect=rect,
                    raw_text=text, final_text=final_text,
                    corrections=corrections, confidence=1.0,
                ))

            if progress_callback:
                progress_callback(1, 1)

            full_text = "\n".join(all_texts)

            return PipelineResult(
                regions=region_results,
                full_text=full_text,
                mode=mode,
                image_size=image.size,
            )

        # ---- Paddle / TrOCR mode ----
        raw_regions = self._run_paddle_detect(image)
        region_results = []
        total = len(raw_regions)

        for idx, r in enumerate(raw_regions):
            box         = r["box"]
            confidence  = r["confidence"]
            paddle_text = r["paddle_text"]
            rect        = box_to_rect(box)

            if self.backend == "trocr" and self.recognizer is not None:
                crop = crop_region(image, box)
                crop = ensure_min_size(crop)
                raw_text = self.recognizer.recognize(crop)
            else:
                raw_text = paddle_text

            if mode == "prescription" and self.corrector is not None:
                final_text, corrections = self.corrector.correct_text(raw_text)
            else:
                final_text = raw_text
                corrections = []

            region_results.append(RegionResult(
                box=box, rect=rect,
                raw_text=raw_text, final_text=final_text,
                corrections=corrections, confidence=confidence,
            ))

            if progress_callback:
                progress_callback(idx + 1, total)

        full_text = "\n".join(r.final_text for r in region_results)

        return PipelineResult(
            regions=region_results,
            full_text=full_text,
            mode=mode,
            image_size=image.size,
        )

    def run_single_word(
        self,
        image: Image.Image,
        top_k: int = 5,
    ) -> dict:
        """Single-word recognition mode."""
        image = ensure_min_size(image)

        if self.backend == "claude" and self.api_key:
            import anthropic, json
            client = anthropic.Anthropic(api_key=self.api_key)
            response = client.messages.create(
                model="claude-sonnet-4-6",
                max_tokens=256,
                messages=[{
                    "role": "user",
                    "content": [
                        {
                            "type": "image",
                            "source": {
                                "type": "base64",
                                "media_type": "image/jpeg",
                                "data": _pil_to_base64(image),
                            }
                        },
                        {
                            "type": "text",
                            "text": (
                                f"This is a handwritten word. Give me the top {top_k} most likely "
                                "transcriptions with confidence scores 0-1. "
                                "Respond ONLY with a JSON array like: "
                                '[{"text": "word", "score": 0.95}, ...]. No explanation.'
                            )
                        }
                    ]
                }]
            )
            raw = response.content[0].text.strip()
            try:
                if raw.startswith("```"):
                    raw = raw.split("```")[1]
                    if raw.startswith("json"):
                        raw = raw[4:]
                candidates = json.loads(raw.strip())
                best = candidates[0]["text"] if candidates else ""
                return {"best": best, "candidates": candidates}
            except Exception:
                return {"best": raw, "candidates": [{"text": raw, "score": 1.0}]}

        if self.recognizer is not None:
            candidates = self.recognizer.recognize_top_k(image, k=top_k)
            best = candidates[0]["text"] if candidates else ""
            return {"best": best, "candidates": candidates}

        return {"best": "", "candidates": []}