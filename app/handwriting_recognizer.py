"""
handwriting_recognizer.py
--------------------------
Handwriting recognition using Microsoft TrOCR.
Supports single-word and multi-region recognition.
"""

from __future__ import annotations

from typing import Optional
from PIL import Image


class HandwritingRecognizer:
    """
    Wraps the TrOCR model for handwriting recognition.

    Uses 'microsoft/trocr-base-handwritten' by default, which works well
    for both printed-style and cursive handwriting.

    Parameters
    ----------
    model_name : str
        HuggingFace model identifier.
    device : str | None
        'cuda', 'cpu', or None (auto-detect).
    """

    DEFAULT_MODEL = "microsoft/trocr-base-handwritten"

    def __init__(
        self,
        model_name: str = DEFAULT_MODEL,
        device: Optional[str] = None,
    ) -> None:
        import torch
        from transformers import TrOCRProcessor, VisionEncoderDecoderModel

        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        self.processor = TrOCRProcessor.from_pretrained(model_name)
        self.model = VisionEncoderDecoderModel.from_pretrained(model_name)
        self.model.to(self.device)
        self.model.eval()

    def recognize(self, image: Image.Image) -> str:
        """
        Recognize handwritten text in a PIL image.

        Parameters
        ----------
        image : PIL.Image
            RGB image of a handwritten text region or word.

        Returns
        -------
        str
            Recognized text string.
        """
        import torch

        if image.mode != "RGB":
            image = image.convert("RGB")

        pixel_values = self.processor(
            images=image, return_tensors="pt"
        ).pixel_values.to(self.device)

        with torch.no_grad():
            generated_ids = self.model.generate(pixel_values)

        text = self.processor.batch_decode(
            generated_ids, skip_special_tokens=True
        )[0]
        return text.strip()

    def recognize_top_k(self, image: Image.Image, k: int = 5) -> list[dict]:
        """
        Return top-k candidate predictions with scores.
        Used in Word Recognition mode.

        Parameters
        ----------
        image : PIL.Image
            Cropped word image.
        k : int
            Number of candidates to return.

        Returns
        -------
        list[dict]
            Each dict has keys 'text' and 'score'.
        """
        import torch

        if image.mode != "RGB":
            image = image.convert("RGB")

        pixel_values = self.processor(
            images=image, return_tensors="pt"
        ).pixel_values.to(self.device)

        with torch.no_grad():
            outputs = self.model.generate(
                pixel_values,
                num_beams=max(k, 5),
                num_return_sequences=k,
                output_scores=True,
                return_dict_in_generate=True,
                early_stopping=True,
            )

        sequences = outputs.sequences
        scores = getattr(outputs, "sequences_scores", None)

        candidates = []
        for i, seq in enumerate(sequences):
            text = self.processor.decode(seq, skip_special_tokens=True).strip()
            score = float(scores[i]) if scores is not None else 0.0
            candidates.append({"text": text, "score": round(score, 4)})

        return candidates