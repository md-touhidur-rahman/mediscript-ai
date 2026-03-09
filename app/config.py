from dataclasses import dataclass


@dataclass(frozen=True)
class OCRConfig:
    min_box_width: int = 18
    min_box_height: int = 10

    trocr_model_name: str = "microsoft/trocr-base-handwritten"
    recognition_num_beams: int = 8
    recognition_num_return_sequences: int = 5
    recognition_max_length: int = 64

    overlay_confidence_threshold: float = 0.0
    uncertain_confidence_threshold: float = 0.30

    medicine_correction_threshold: int = 88

    fill_region: bool = True
    overlay_margin: int = 4
    max_font_size: int = 42
    min_font_size: int = 12

    crop_padding: int = 8


DEFAULT_CONFIG = OCRConfig()