"""
prescription_logic.py
-----------------------
Medicine name correction using RapidFuzz fuzzy matching.

Only used in Prescription OCR mode.
Does NOT affect general handwriting recognition.
"""

from __future__ import annotations

import os
import csv
from typing import Optional

from rapidfuzz import process, fuzz


# ---------------------------------------------------------------------------
# Built-in minimal medicine list (fallback when no CSV is provided)
# ---------------------------------------------------------------------------

_BUILTIN_MEDICINES: list[str] = [
    "Amoxicillin", "Azithromycin", "Metformin", "Atorvastatin", "Lisinopril",
    "Omeprazole", "Amlodipine", "Metoprolol", "Losartan", "Hydrochlorothiazide",
    "Simvastatin", "Levothyroxine", "Gabapentin", "Sertraline", "Ibuprofen",
    "Paracetamol", "Aspirin", "Cetirizine", "Loratadine", "Doxycycline",
    "Ciprofloxacin", "Clopidogrel", "Pantoprazole", "Escitalopram", "Fluoxetine",
    "Prednisolone", "Dexamethasone", "Furosemide", "Spironolactone", "Warfarin",
    "Insulin", "Glibenclamide", "Metronidazole", "Clindamycin", "Vancomycin",
    "Tramadol", "Codeine", "Morphine", "Diclofenac", "Naproxen",
    "Ranitidine", "Famotidine", "Loperamide", "Ondansetron", "Domperidone",
    "Salbutamol", "Montelukast", "Beclomethasone", "Ipratropium", "Theophylline",
    "Enalapril", "Ramipril", "Valsartan", "Carvedilol", "Bisoprolol",
    "Nifedipine", "Diltiazem", "Verapamil", "Digoxin", "Amiodarone",
]

# Common non-medicine words that should never be fuzzy-corrected
_STOPWORDS = {
    "my", "name", "is", "i", "am", "taking", "this", "course", "to", "improve",
    "so", "can", "enjoy", "writing", "in", "journals", "again", "as", "it",
    "stands", "now", "find", "handwriting", "be", "ununiformed", "and",
    "unattractive", "would", "like", "freehand", "flow", "much", "more",
    "smoothly", "effortlessly", "possible", "the", "a", "an", "of", "for",
    "with", "without", "after", "before", "food", "once", "twice", "daily",
    "morning", "night", "tablet", "tablets", "capsule", "capsules", "take",
    "one", "two", "three", "mg", "ml", "bid", "tid", "qid", "od", "bd", "sos",
}


class PrescriptionCorrector:
    """
    Corrects OCR-recognized text against a medicine dictionary using
    RapidFuzz fuzzy matching.

    Parameters
    ----------
    dictionary_path : str | None
        Path to a CSV file with medicine names (one per row, first column).
        Falls back to built-in list if None or file not found.
    score_threshold : int
        Minimum similarity score (0–100) to accept a correction.
    """

    def __init__(
        self,
        dictionary_path: Optional[str] = None,
        score_threshold: int = 88,
    ) -> None:
        self.score_threshold = score_threshold
        self.medicines = self._load_dictionary(dictionary_path)
        self._medicines_lower = {m.lower() for m in self.medicines}

    def _load_dictionary(self, path: Optional[str]) -> list[str]:
        if path and os.path.isfile(path):
            medicines = []
            with open(path, newline="", encoding="utf-8") as f:
                reader = csv.reader(f)
                for row in reader:
                    if row and row[0].strip():
                        medicines.append(row[0].strip())
            if medicines:
                return medicines
        return _BUILTIN_MEDICINES

    def _normalize_token(self, token: str) -> str:
        return token.strip(".,;:()[]{}\\/\"'").strip()

    def _should_attempt_correction(self, token: str) -> bool:
        """
        Be conservative. Only try to correct tokens that plausibly look like
        medicine names.
        """
        clean = self._normalize_token(token)
        if not clean:
            return False

        # Never correct very short tokens
        if len(clean) < 5:
            return False

        # Never correct pure numbers / dosage-like forms
        if any(ch.isdigit() for ch in clean):
            return False

        # Only alphabetical tokens
        if not clean.replace("-", "").isalpha():
            return False

        # Skip common instruction / sentence words
        if clean.lower() in _STOPWORDS:
            return False

        return True

    def correct_word(self, word: str) -> tuple[str, float]:
        """
        Try to correct a single word against the medicine dictionary.

        Returns
        -------
        (corrected_word, score)
            Original word returned unchanged if score < threshold.
        """
        clean = self._normalize_token(word)
        if not self._should_attempt_correction(clean):
            return word, 0.0

        # If already exactly a medicine, keep it
        if clean.lower() in self._medicines_lower:
            return clean, 100.0

        result = process.extractOne(
            clean,
            self.medicines,
            scorer=fuzz.WRatio,
        )

        if result is None:
            return word, 0.0

        match, score, _ = result

        # Extra safety: don't allow wildly different-length substitutions
        if abs(len(match) - len(clean)) > 4:
            return word, score

        if score >= self.score_threshold:
            return match, score

        return word, score

    def correct_text(self, text: str) -> tuple[str, list[dict]]:
        """
        Correct each token in a recognized text string.

        Parameters
        ----------
        text : str
            Full recognized text from one bounding-box region.

        Returns
        -------
        (corrected_text, corrections)
        """
        tokens = text.split()
        corrected_tokens = []
        corrections = []

        for token in tokens:
            stripped = self._normalize_token(token)
            corrected, score = self.correct_word(stripped)

            changed = corrected.lower() != stripped.lower()
            corrected_token = token.replace(stripped, corrected, 1) if changed else token

            corrected_tokens.append(corrected_token)
            corrections.append({
                "original": token,
                "corrected": corrected_token,
                "score": float(score),
                "changed": changed,
            })

        return " ".join(corrected_tokens), corrections