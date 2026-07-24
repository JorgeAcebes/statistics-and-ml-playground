"""Minimal Streamlit interface for the alligator-versus-crocodile classifier."""

import pathlib
import sys

import fastcore.foundation
import streamlit as st
from fastai.vision.all import PILImage, load_learner


# The learner was exported on Windows, while Streamlit Community Cloud runs on Linux.
# This alias lets pickle resolve saved WindowsPath values during model loading.
if sys.platform != "win32":
    pathlib.WindowsPath = pathlib.PosixPath


def _starmap(self, function, *args, **kwargs):
    """Keep compatibility with learners exported using older fastcore versions."""
    return self.map(lambda item: function(*item, *args, **kwargs))


fastcore.foundation.L.starmap = _starmap

st.set_page_config(
    page_title="Alligator or Crocodile?",
    page_icon="🐊",
    layout="centered",
    initial_sidebar_state="collapsed",
)


@st.cache_resource(show_spinner="Loading the model…")
def get_learner():
    """Load the exported fastai learner once per application process on CPU."""
    model_path = pathlib.Path(__file__).with_name("model.pkl")
    return load_learner(model_path, cpu=True)


LABELS = {
    "alig": "Alligator",
    "croc": "Crocodile",
}


def percentage(value: float) -> str:
    """Format a probability for display."""
    return f"{value * 100:.1f}%"


st.markdown(
    """
    <style>
      .block-container { max-width: 720px; padding-top: 4rem; padding-bottom: 3rem; }
      h1 { font-size: 2.55rem !important; letter-spacing: -0.05em; margin-bottom: 0.3rem !important; }
      .subtitle { color: #6b7280; font-size: 1.05rem; margin-bottom: 2rem; }
      .result { border: 1px solid #e5e7eb; border-radius: 16px; padding: 1.25rem 1.4rem;
                background: #fafaf9; margin-top: 1rem; }
      .prediction { font-size: 1.7rem; font-weight: 700; letter-spacing: -0.03em; color: #18261d; }
      .confidence { color: #52715a; font-size: 1rem; margin-top: 0.25rem; }
      [data-testid="stFileUploader"] { border-radius: 14px; }
    </style>
    <h1>Alligator or crocodile?</h1>
    <p class="subtitle">Upload an image and let the model classify it.</p>
    """,
    unsafe_allow_html=True,
)

uploaded_file = st.file_uploader(
    "Choose an image",
    type=["jpg", "jpeg", "png", "webp"],
    help="Supported formats: JPG, PNG, and WEBP.",
)

if uploaded_file is None:
    st.caption("The model distinguishes alligators and crocodiles in photographs.")
else:
    try:
        image = PILImage.create(uploaded_file)
        st.image(image, caption="Selected image", use_container_width=True)

        learner = get_learner()
        with st.spinner("Analyzing the image…"):
            predicted_class, predicted_index, probabilities = learner.predict(image)

        predicted_class = str(predicted_class)
        confidence = float(probabilities[predicted_index])
        readable_label = LABELS.get(predicted_class, predicted_class.capitalize())

        st.markdown(
            f"""
            <div class="result">
              <div class="prediction">{readable_label}</div>
              <div class="confidence">Confidence: {percentage(confidence)}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        with st.expander("View probabilities"):
            for class_name, probability in zip(learner.dls.vocab, probabilities):
                label = LABELS.get(str(class_name), str(class_name).capitalize())
                st.progress(float(probability), text=f"{label}: {percentage(float(probability))}")
    except Exception:
        st.error("This image could not be processed. Please try another valid JPG, PNG, or WEBP image.")
