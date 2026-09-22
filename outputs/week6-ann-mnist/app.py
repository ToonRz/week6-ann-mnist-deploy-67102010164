from pathlib import Path

import numpy as np
import streamlit as st
from PIL import Image, UnidentifiedImageError


MODEL_FILENAME = "67102010164_mnist_model.keras"
MODEL_PATH = Path(__file__).resolve().parent / MODEL_FILENAME
WEIGHTS_PATH = Path(__file__).resolve().parent / "model_weights.npz"
CLASS_NAMES = [str(digit) for digit in range(10)]


st.set_page_config(
    page_title="MNIST Digit Recognizer",
    page_icon="✍️",
    layout="centered",
)

st.markdown(
    """
    <style>
        .block-container {max-width: 780px; padding-top: 2.5rem;}
        [data-testid="stMetricValue"] {color: #5b4bdb;}
        .subtitle {color: #667085; margin-top: -0.75rem; margin-bottom: 1.75rem;}
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource(show_spinner=False)
def load_weights(weights_path: str) -> dict[str, np.ndarray]:
    """Load the exported dense-layer weights without requiring TensorFlow."""
    with np.load(weights_path) as data:
        return {key: data[key] for key in ("w0", "b0", "w1", "b1")}


def predict(batch: np.ndarray, weights: dict[str, np.ndarray]) -> np.ndarray:
    hidden = np.maximum(0.0, batch.reshape(len(batch), -1) @ weights["w0"] + weights["b0"])
    logits = hidden @ weights["w1"] + weights["b1"]
    logits -= logits.max(axis=1, keepdims=True)
    probabilities = np.exp(logits)
    return probabilities / probabilities.sum(axis=1, keepdims=True)


def preprocess_image(image: Image.Image, invert: bool = False) -> tuple[np.ndarray, Image.Image]:
    """Apply the same grayscale, 28x28, and /255 pipeline used for training."""
    processed = image.convert("L").resize((28, 28))
    pixels = np.asarray(processed, dtype=np.float32) / 255.0
    if invert:
        pixels = 1.0 - pixels
        processed = Image.fromarray(np.uint8(np.rint(pixels * 255.0)), mode="L")
    return pixels[np.newaxis, ...], processed


st.title("Handwritten Digit Recognizer")
st.markdown(
    '<p class="subtitle">Upload an image and the ANN will predict a digit from 0 to 9.</p>',
    unsafe_allow_html=True,
)

uploaded_file = st.file_uploader(
    "Choose a digit image",
    type=["jpg", "jpeg", "png"],
    help="Supported formats: JPG, JPEG, and PNG.",
)
invert = st.toggle(
    "Invert black and white",
    value=False,
    help="Use this only if the digit polarity differs from the MNIST training images.",
)

if uploaded_file is None:
    st.info("Upload a JPG or PNG image to begin.")
    st.stop()

try:
    with Image.open(uploaded_file) as opened_image:
        opened_image.load()
        input_batch, processed_preview = preprocess_image(opened_image, invert=invert)
except (UnidentifiedImageError, OSError, ValueError) as exc:
    st.error("This file could not be read as a valid image. Please try another JPG or PNG file.")
    st.caption(f"Image error: {exc}")
    st.stop()

preview_column, result_column = st.columns([1, 1.25], gap="large")
with preview_column:
    st.subheader("Model input")
    st.image(
        processed_preview,
        caption="Grayscale • 28 × 28 pixels",
        width=224,
    )

with result_column:
    if not WEIGHTS_PATH.is_file():
        st.error("Exported model weights are missing: model_weights.npz")
        st.caption("Place the trained model in the same folder as app.py, then reload the app.")
        st.stop()

    try:
        with st.spinner("Recognizing the digit…"):
            probabilities = predict(input_batch, load_weights(str(WEIGHTS_PATH))).squeeze()
    except Exception as exc:
        st.error("The model could not be loaded or used for prediction.")
        st.caption(f"Model error: {exc}")
        st.stop()

    if probabilities.shape != (10,) or not np.all(np.isfinite(probabilities)):
        st.error("The model returned an invalid prediction. Expected 10 finite class probabilities.")
        st.stop()

    predicted_index = int(np.argmax(probabilities))
    confidence = float(probabilities[predicted_index])

    st.subheader("Prediction")
    st.metric("Predicted digit", CLASS_NAMES[predicted_index])
    st.metric("Confidence", f"{confidence:.2%}")
    st.progress(
        min(max(confidence, 0.0), 1.0),
        text=f"Digit {CLASS_NAMES[predicted_index]} confidence",
    )

with st.expander("View all class probabilities"):
    st.caption("Digits 0–9")
    st.bar_chart({"Probability": probabilities})
