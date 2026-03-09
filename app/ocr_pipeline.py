"""
ocr_pipeline.py
----------------
Main OCR pipeline for MediScript AI.

Recognition backends:
  - "paddle" : PaddleOCR built-in recognition
  - "trocr"  : TrOCR on Paddle-detected crops

Detection is done by PaddleOCR.
This version is adjusted for PaddleOCR 3.x deployment stability on Streamlit Cloud:
- disables document preprocessing modules
- uses predict()
- parses 3.x result format safely
"""

from __future__ import annotations

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
        Recognition backend: 'paddle' | 'trocr'
    """

    def __init__(
        self,
        recognizer=None,
        corrector: Optional[PrescriptionCorrector] = None,
        paddle_lang: str = "en",
        backend: Literal["paddle", "trocr"] = "paddle",
    ) -> None:
        self.recognizer = recognizer
        self.corrector = corrector
        self._paddle = None
        self._paddle_lang = paddle_lang
        self.backend = backend

    # ------------------------------------------------------------------
    # PaddleOCR lazy loader
    # ------------------------------------------------------------------

    def _get_paddle(self):
        if self._paddle is None:
            from paddleocr import PaddleOCR

            self._paddle = PaddleOCR(
                lang=self._paddle_lang,
                use_doc_orientation_classify=False,
                use_doc_unwarping=False,
                use_textline_orientation=False,
            )
        return self._paddle

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _clean_text(text: str) -> str:
        return " ".join(str(text or "").strip().split())

    @staticmethod
    def _to_list_box(box) -> list:
        if hasattr(box, "tolist"):
            return box.tolist()
        return box

    # ------------------------------------------------------------------
    # Detection via PaddleOCR 3.x
    # ------------------------------------------------------------------

    def _run_paddle_detect(self, image: Image.Image) -> list[dict]:
        """
        Run PaddleOCR 3.x using predict() and normalize results.

        Returns
        -------
        list[dict]
            Each dict has:
              box         : list of 4 [x, y] points
              paddle_text : recognized text
              confidence  : float
        """
        paddle = self._get_paddle()
        proc_image = preprocess_for_ocr(image)
        np_image = np.array(proc_image.convert("RGB"))

        try:
            preds = paddle.predict(np_image)
        except Exception:
            return []

        regions = []
        if not preds:
            return regions

        for pred in preds:
            # PaddleOCR 3.x result objects usually expose .res
            res = pred.res if hasattr(pred, "res") else pred
            if not isinstance(res, dict):
                continue

            boxes = res.get("dt_polys") or []
            texts = res.get("rec_texts") or []
            scores = res.get("rec_scores") or []

            for i, box in enumerate(boxes):
                text = texts[i] if i < len(texts) else ""
                score = float(scores[i]) if i < len(scores) else 1.0

                if not text or not str(text).strip():
                    continue
                if score < 0.70:
                    continue

                box = self._to_list_box(box)
                rect = box_to_rect(box)
                x0, y0, x1, y1 = rect
                bw = x1 - x0
                bh = y1 - y0

                if bw < 20 or bh < 10:
                    continue

                regions.append(
                    {
                        "box": box,
                        "paddle_text": self._clean_text(text),
                        "confidence": score,
                    }
                )

        regions.sort(key=lambda r: (box_to_rect(r["box"])[1], box_to_rect(r["box"])[0]))
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

        For 'paddle' backend:
            Use PaddleOCR text directly.

        For 'trocr' backend:
            Use PaddleOCR detection, then TrOCR on each crop.
        """
        raw_regions = self._run_paddle_detect(image)
        region_results: list[RegionResult] = []
        total = len(raw_regions)

        if progress_callback:
            progress_callback(0, max(total, 1))

        for idx, r in enumerate(raw_regions):
            box = r["box"]
            confidence = float(r["confidence"])
            paddle_text = self._clean_text(r["paddle_text"])
            rect = box_to_rect(box)

            if self.backend == "trocr" and self.recognizer is not None:
                crop = crop_region(image, box)
                crop = ensure_min_size(crop)
                raw_text = self._clean_text(self.recognizer.recognize(crop))
            else:
                raw_text = paddle_text

            if mode == "prescription" and self.corrector is not None:
                final_text, corrections = self.corrector.correct_text(raw_text)
                final_text = self._clean_text(final_text)
            else:
                final_text = raw_text
                corrections = []

            region_results.append(
                RegionResult(
                    box=box,
                    rect=rect,
                    raw_text=raw_text,
                    final_text=final_text,
                    corrections=corrections,
                    confidence=confidence,
                )
            )

            if progress_callback:
                progress_callback(idx + 1, total)

        full_text = "\n".join(r.final_text for r in region_results if r.final_text)

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

        if self.recognizer is not None:
            candidates = self.recognizer.recognize_top_k(image, k=top_k)
            best = candidates[0]["text"] if candidates else ""
            return {"best": best, "candidates": candidates}

        return {"best": "", "candidates": []}
