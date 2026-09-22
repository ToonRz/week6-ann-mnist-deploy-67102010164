from pathlib import Path

import numpy as np
import streamlit as st
import tensorflow as tf
from PIL import Image, UnidentifiedImageError


MODEL_FILENAME = "67102010164_mnist_model.keras"
MODEL_PATH = Path(__file__).resolve().parent / MODEL_FILENAME
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
def load_model(model_path: str) -> tf.keras.Model:
    """Load the trained Keras model once per Streamlit process."""
    return tf.keras.models.load_model(model_path)


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
    if not MODEL_PATH.is_file():
        st.error(f"Model file not found: {MODEL_FILENAME}")
        st.caption("Place the trained model in the same folder as app.py, then reload the app.")
        st.stop()

    try:
        with st.spinner("Recognizing the digit…"):
            model = load_model(str(MODEL_PATH))
            probabilities = np.asarray(model.predict(input_batch, verbose=0)).squeeze()
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
