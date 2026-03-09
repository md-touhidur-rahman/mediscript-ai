#  MediScript AI

Live Demo: https://mediscript-ai-v1.streamlit.app/

A **general handwriting recognition system** that works for:
- Doctor prescriptions
- Student essays
- Handwritten notes
- Any handwritten document

Inspired by Google Lens: detects text regions, recognizes handwriting, and generates a layout-preserving overlay image with typed text replacing handwriting.

---

## Architecture

```
Image Input
    │
    ▼
[PaddleOCR]  ── Text Detection (bounding boxes only)
    │
    ▼
[Crop Regions]  ── Extract each detected text region from image
    │
    ▼
[TrOCR]  ── Handwriting Recognition per region
    │
    ▼
[RapidFuzz]  ── Optional medicine correction (Prescription mode only)
    │
    ▼
[OpenCV / Pillow]  ── Layout-preserving overlay renderer
    │
    ▼
Extracted Text  +  Overlay Image
```

---

## Project Structure

```
mediscript_ai/
├── app/
│   ├── streamlit_app.py        ← Streamlit UI (entry point)
│   ├── ocr_pipeline.py         ← Main pipeline orchestrator
│   ├── overlay_renderer.py     ← Layout-preserving overlay renderer
│   ├── handwriting_recognizer.py ← TrOCR handwriting recognition
│   ├── prescription_logic.py   ← RapidFuzz medicine correction
│   └── utils.py                ← Shared utilities
│
├── data/
│   └── training_labels.csv     ← Sample labels for fine-tuning
│
├── requirements.txt
└── README.md
```

---

## Modes

### Mode 1 — General OCR
- Input: any handwritten image
- Output: extracted text + overlay image
- No medicine correction

### Mode 2 — Prescription OCR
- Input: prescription image
- Output: extracted text + overlay image
- Optional: medicine name correction via RapidFuzz fuzzy matching
- Upload your own CSV medicine list or use the built-in list

### Mode 3 — Word Recognition
- Input: cropped single-word image
- Output: best prediction + top-k candidates with scores
- Useful for testing recognition accuracy

---

## Installation

### 1. Create a virtual environment (recommended)

```bash
python -m venv venv
source venv/bin/activate       # Linux / macOS
venv\Scripts\activate          # Windows
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

> **Note for CPU-only machines:** PaddlePaddle defaults to CPU. For GPU support, install the GPU version of PaddlePaddle separately from [https://www.paddlepaddle.org.cn/en](https://www.paddlepaddle.org.cn/en).

### 3. Run the app

```bash
streamlit run app/streamlit_app.py
```

The app will open in your browser at `http://localhost:8501`.

---

## Custom Medicine Dictionary

In Prescription OCR mode, you can upload a CSV file with your own medicine list.
Format: one medicine name per row, first column, no header required.

Example:
```
Amoxicillin
Metformin
Azithromycin
Omeprazole
```

---

## Key Design Decisions

| Decision | Reason |
|---|---|
| PaddleOCR for detection only | PaddleOCR's detection is fast and accurate; TrOCR handles handwriting better than PaddleOCR's built-in recognizer |
| TrOCR `trocr-base-handwritten` | Best open-source model for handwriting recognition |
| RapidFuzz WRatio scorer | Most robust for noisy OCR output (handles case, partial matches, transpositions) |
| Medicine correction is mode-gated | Ensures general OCR is never affected by prescription logic |
| Lazy model loading | PaddleOCR and TrOCR are only loaded when first needed |
| `@st.cache_resource` | Models are loaded once per Streamlit session, not on every rerun |

---

## Troubleshooting

**PaddleOCR import error:**
```bash
pip install paddlepaddle paddleocr --upgrade
```

**TrOCR slow on first run:**
The model (~1 GB) is downloaded from HuggingFace on first use. Subsequent runs use the cached model.

**Font not rendering on overlay:**
Install DejaVu fonts:
```bash
# Ubuntu/Debian
sudo apt-get install fonts-dejavu

# macOS
brew install --cask font-dejavu
```

**CUDA out of memory:**
Set `use_gpu=False` in `ocr_pipeline.py` or reduce image size before processing.
