"""
streamlit_app.py
----------------
MediScript AI — Streamlit frontend

Fresh UI version:
- Text output only
- No overlay rendering
- No Claude API
- Supports:
    1. General OCR
    2. Prescription OCR
    3. Word Recognition
"""

from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path

import pandas as pd
import streamlit as st
from PIL import Image

# Make local imports work when running:
# streamlit run app/streamlit_app.py
APP_DIR = Path(__file__).resolve().parent
if str(APP_DIR) not in sys.path:
    sys.path.insert(0, str(APP_DIR))

from ocr_pipeline import OCRPipeline
from prescription_logic import PrescriptionCorrector
from utils import scale_image_for_display


# ---------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------

st.set_page_config(
    page_title="MediScript AI",
    page_icon="🩺",
    layout="wide",
)


# ---------------------------------------------------------------------
# Cached loaders
# ---------------------------------------------------------------------

@st.cache_resource(show_spinner="Loading medicine corrector...")
def load_corrector(dictionary_path: str | None = None) -> PrescriptionCorrector:
    return PrescriptionCorrector(dictionary_path=dictionary_path)


@st.cache_resource(show_spinner="Loading OCR pipeline...")
def load_pipeline(backend: str) -> OCRPipeline:
    recognizer = None

    if backend == "trocr":
        from handwriting_recognizer import HandwritingRecognizer
        recognizer = HandwritingRecognizer()

    return OCRPipeline(
        recognizer=recognizer,
        backend=backend,
    )


# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------

def save_uploaded_csv(uploaded_file) -> str | None:
    if uploaded_file is None:
        return None

    with tempfile.NamedTemporaryFile(delete=False, suffix=".csv") as tmp:
        tmp.write(uploaded_file.read())
        return tmp.name


def render_header():
    st.markdown(
        """
        <div style="padding: 0.4rem 0 1rem 0;">
            <h1 style="margin-bottom: 0.2rem;">🩺 MediScript AI</h1>
            <p style="margin-top: 0; color: #666;">
                General handwriting OCR for documents, prescriptions, and single-word recognition.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_sidebar():
    st.sidebar.title("Settings")

    mode = st.sidebar.radio(
        "Mode",
        options=["General OCR", "Prescription OCR", "Word Recognition"],
        index=0,
    )

    backend = st.sidebar.radio(
        "Recognition Backend",
        options=["paddle", "trocr"],
        index=0,
        format_func=lambda x: {
            "paddle": "⚡ PaddleOCR (fast)",
            "trocr": "✍️ TrOCR (better for handwriting)",
        }[x],
    )

    apply_correction = False
    dict_file = None

    if mode == "Prescription OCR":
        st.sidebar.markdown("---")
        apply_correction = st.sidebar.checkbox(
            "Apply medicine correction",
            value=True,
        )
        dict_file = st.sidebar.file_uploader(
            "Medicine dictionary CSV",
            type=["csv"],
            help="Optional CSV file with one medicine name per row.",
        )

    st.sidebar.markdown("---")
    st.sidebar.caption("Fresh text-only UI")

    return mode, backend, apply_correction, dict_file


def render_document_mode(
    mode_key: str,
    backend: str,
    apply_correction: bool,
    dict_file,
):
    is_prescription = mode_key == "prescription"

    left, right = st.columns([1, 1.15], gap="large")

    with left:
        st.subheader("Upload Image")
        uploaded = st.file_uploader(
            "Choose an image",
            type=["png", "jpg", "jpeg", "bmp", "tiff", "webp"],
            key=f"upload_{mode_key}",
        )

        if uploaded is None:
            st.info("Upload an image to begin.")
            return

        image = Image.open(uploaded).convert("RGB")
        display_img = scale_image_for_display(image, max_width=650, max_height=650)

        st.image(
            display_img,
            caption=f"Preview ({display_img.width} × {display_img.height})",
        )

        if backend == "trocr":
            st.caption("TrOCR may be slower on long documents.")

    with right:
        st.subheader("Extracted Text")

        dict_path = save_uploaded_csv(dict_file) if (is_prescription and dict_file is not None) else None

        corrector = None
        if is_prescription and apply_correction:
            corrector = load_corrector(dict_path)

        pipeline = load_pipeline(backend)
        pipeline.corrector = corrector

        status = st.empty()
        progress = st.progress(0)

        def update_progress(current: int, total: int):
            percent = int((current / total) * 100) if total > 0 else 100
            progress.progress(percent)
            status.caption(f"Processing region {current} of {total}")

        with st.spinner("Running OCR..."):
            result = pipeline.run(
                image,
                mode=mode_key,
                progress_callback=update_progress,
            )

        progress.progress(100)
        status.caption(f"Done. Regions found: {len(result.regions)}")

        st.text_area(
            "Recognized text",
            value=result.full_text,
            height=320,
            key=f"text_{mode_key}",
        )

        c1, c2 = st.columns(2)
        with c1:
            st.metric("Regions", len(result.regions))
        with c2:
            st.metric("Backend", backend.upper())

        st.download_button(
            "Download text",
            data=result.full_text,
            file_name=f"{mode_key}_ocr_text.txt",
            mime="text/plain",
            use_container_width=True,
        )

        if is_prescription and apply_correction:
            rows = []
            for region in result.regions:
                for correction in region.corrections:
                    if correction.get("changed"):
                        rows.append(
                            {
                                "Original": correction["original"],
                                "Corrected": correction["corrected"],
                                "Score": f"{float(correction['score']):.1f}",
                            }
                        )

            st.markdown("---")
            st.subheader("Medicine Corrections")

            if rows:
                st.dataframe(pd.DataFrame(rows), use_container_width=True)
            else:
                st.success("No medicine corrections were applied.")

        with st.expander("Region Breakdown"):
            if not result.regions:
                st.write("No regions found.")
            else:
                for i, region in enumerate(result.regions, start=1):
                    st.markdown(f"**Region {i}**")
                    st.write(f"Raw text: {region.raw_text}")
                    st.write(f"Final text: {region.final_text}")
                    st.write(f"Confidence: {region.confidence:.2f}")
                    st.markdown("---")


def render_word_mode(backend: str):
    st.subheader("Single Word Recognition")

    if backend == "paddle":
        st.warning("Word Recognition works best with TrOCR. PaddleOCR may be weaker for single handwritten words.")

    uploaded = st.file_uploader(
        "Upload a cropped handwritten word",
        type=["png", "jpg", "jpeg", "bmp", "tiff", "webp"],
        key="word_upload",
    )

    top_k = st.slider("Number of candidates", min_value=1, max_value=10, value=5)

    if uploaded is None:
        st.info("Upload a cropped word image to begin.")
        return

    image = Image.open(uploaded).convert("RGB")

    left, right = st.columns([1, 1.15], gap="large")

    with left:
        display_img = scale_image_for_display(image, max_width=450, max_height=300)
        st.image(display_img, caption=f"Preview ({display_img.width} × {display_img.height})")

    with right:
        pipeline = load_pipeline(backend)

        with st.spinner("Recognizing word..."):
            result = pipeline.run_single_word(image, top_k=top_k)

        st.markdown("### Best Prediction")
        st.success(result.get("best", ""))

        candidates = result.get("candidates", [])
        if candidates:
            st.markdown("### Candidate List")
            df = pd.DataFrame(candidates)
            df.index = df.index + 1
            st.dataframe(df, use_container_width=True)

            st.download_button(
                "Download candidates",
                data=df.to_csv(index=False),
                file_name="word_candidates.csv",
                mime="text/csv",
                use_container_width=True,
            )


# ---------------------------------------------------------------------
# Main app
# ---------------------------------------------------------------------

render_header()
mode, backend, apply_correction, dict_file = render_sidebar()

if mode == "General OCR":
    render_document_mode(
        mode_key="general",
        backend=backend,
        apply_correction=False,
        dict_file=None,
    )
elif mode == "Prescription OCR":
    render_document_mode(
        mode_key="prescription",
        backend=backend,
        apply_correction=apply_correction,
        dict_file=dict_file,
    )
else:
    render_word_mode(backend)