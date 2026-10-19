from __future__ import annotations

import time
from pathlib import Path

import cv2
import joblib
import numpy as np
import streamlit as st
from PIL import Image, ImageOps
from skimage.feature import graycomatrix, graycoprops, local_binary_pattern

ROOT = Path(__file__).resolve().parent
MODEL_PATH = ROOT / "classical_ml" / "final_rf_tuned.pkl"
ENCODER_PATH = ROOT / "classical_ml" / "rf_label_encoder.pkl"

st.set_page_config(
    page_title="CitrusCare | Citrus Leaf Screening",
    page_icon="🍊",
    layout="wide",
    initial_sidebar_state="expanded",
)

DISEASE_INFO = {
    "Anthracnose": (
        "A fungal disease that may cause dark lesions on citrus leaves, twigs, or fruit.",
        "Inspect affected growth and consult local agricultural extension services about suitable management.",
    ),
    "Black Spot": (
        "A citrus fungal disease associated with dark spots or lesions on affected plant tissue.",
        "Monitor symptoms and seek locally approved disease-management guidance.",
    ),
    "Canker": (
        "A bacterial disease that can cause raised, corky lesions, sometimes surrounded by a yellow halo.",
        "Avoid moving potentially infected plant material and ask a local plant-health expert for confirmation.",
    ),
    "Greening": (
        "Citrus greening (HLB) is serious, and visual symptoms can overlap with nutrient deficiencies.",
        "Arrange expert confirmation promptly and follow local plant-health authority guidance.",
    ),
    "Healthy": (
        "The image looks most similar to the model's Healthy training class.",
        "Continue routine monitoring. A model prediction cannot rule out every disease or plant stress.",
    ),
    "Melanose": (
        "A fungal disease that may cause small, rough, dark spots on citrus plant tissue.",
        "Monitor whether symptoms spread and consult locally approved guidance if concerned.",
    ),
}


@st.cache_resource(show_spinner="Loading the trained Random Forest model…")
def load_model():
    if not MODEL_PATH.is_file():
        raise FileNotFoundError(f"Missing model file: {MODEL_PATH}")
    if not ENCODER_PATH.is_file():
        raise FileNotFoundError(f"Missing label encoder: {ENCODER_PATH}")
    model = joblib.load(MODEL_PATH)
    encoder = joblib.load(ENCODER_PATH)
    if not hasattr(model, "predict_proba"):
        raise TypeError("The loaded model does not provide predict_proba().")
    if not hasattr(encoder, "inverse_transform"):
        raise TypeError("The label encoder is not compatible.")
    return model, encoder


def extract_features(pil_image: Image.Image):
    """Extract 96 HSV histogram + 4 GLCM + 10 LBP = 110 features."""
    pil_image = ImageOps.exif_transpose(pil_image).convert("RGB")
    rgb = np.asarray(pil_image)
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    bgr = cv2.resize(bgr, (256, 256), interpolation=cv2.INTER_AREA)

    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    histograms = [
        cv2.calcHist([hsv], [channel], None, [32], [0, 256]).flatten()
        for channel in (0, 1, 2)
    ]
    hsv_features = np.concatenate(histograms)

    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    glcm = graycomatrix(
        gray, distances=[1], angles=[0], levels=256,
        symmetric=True, normed=True
    )
    glcm_features = np.array([
        graycoprops(glcm, "contrast")[0, 0],
        graycoprops(glcm, "correlation")[0, 0],
        graycoprops(glcm, "energy")[0, 0],
        graycoprops(glcm, "homogeneity")[0, 0],
    ], dtype=np.float64)

    radius = 1
    points = 8 * radius
    lbp = local_binary_pattern(gray, points, radius, method="uniform")
    lbp_hist, _ = np.histogram(
        lbp.ravel(), bins=points + 2, range=(0, points + 2)
    )
    lbp_hist = lbp_hist.astype(np.float64)
    lbp_hist /= lbp_hist.sum() + 1e-6

    features = np.concatenate((hsv_features, glcm_features, lbp_hist))
    if features.size != 110:
        raise ValueError(f"Expected 110 features but extracted {features.size}.")
    if not np.isfinite(features).all():
        raise ValueError("Feature extraction produced invalid numeric values.")

    lbp_view = (lbp / (lbp.max() + 1e-6) * 255).astype(np.uint8)
    return features.reshape(1, -1), gray, lbp_view


st.markdown("""
<style>
.block-container {max-width: 1350px; padding-top: 1.7rem; padding-bottom: 3rem;}
.hero {padding: 1.5rem 1.7rem; border-radius: 18px; color: white;
background: linear-gradient(120deg,#123b2b,#17664a 60%,#2b8058); margin-bottom: 1rem;}
.hero h1 {color: white; font-size: 2.25rem; margin: 0 0 .35rem 0;}
.hero p {margin: 0; opacity: .94;}
.muted {color: #7b8490; font-size: .9rem;}
</style>
""", unsafe_allow_html=True)

st.markdown("""
<div class="hero">
  <h1>🍊 CitrusCare</h1>
  <p>Classical machine learning for citrus-leaf disease screening</p>
</div>
""", unsafe_allow_html=True)

with st.sidebar:
    st.markdown("## 🍃 CitrusCare")
    st.caption("Random Forest · Classical ML")
    st.divider()
    st.markdown("### How it works")
    st.write("1. Upload a clear leaf photo\n2. Extract colour and texture features\n3. Predict with the saved Random Forest\n4. Review the result and alternatives")
    st.divider()
    st.markdown("### Photo tips")
    st.write("• Use good, even lighting\n• Keep one leaf in focus\n• Avoid blur and heavy shadows\n• Include the full leaf where possible")
    st.divider()
    st.caption("For screening only—not a confirmed agricultural diagnosis.")

try:
    model, encoder = load_model()
except Exception as exc:
    st.error("The ML model could not be loaded.")
    st.code(str(exc))
    st.info(
        "Confirm that classical_ml/final_rf_tuned.pkl and "
        "classical_ml/rf_label_encoder.pkl exist beside this app.py folder, "
        "and install compatible scikit-learn and joblib versions."
    )
    st.stop()

class_names = [str(label) for label in encoder.classes_]
m1, m2, m3, m4 = st.columns(4)
m1.metric("Classifier", "Random Forest")
m2.metric("Input features", "110")
m3.metric("Feature groups", "3")
m4.metric("Classes in encoder", str(len(class_names)))

st.markdown("## 🔎 Analyse a citrus leaf")
st.write("Upload a JPG, JPEG, or PNG. Your image is resized for consistent feature extraction.")

uploaded = st.file_uploader(
    "Choose a citrus-leaf image",
    type=["jpg", "jpeg", "png"],
    help="A clear, well-lit image generally gives the model the best chance of useful results.",
)

if uploaded is None:
    st.info("👆 Upload an image to start the analysis.")
    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown("### 🎨 Colour")
        st.write("96 HSV histogram values")
    with c2:
        st.markdown("### 🧩 Texture")
        st.write("4 GLCM statistics")
    with c3:
        st.markdown("### 🔬 Local patterns")
        st.write("10 LBP histogram values")
    st.stop()

try:
    image = ImageOps.exif_transpose(Image.open(uploaded)).convert("RGB")
except Exception as exc:
    st.error(f"Could not open this image: {exc}")
    st.stop()

left, right = st.columns([1.05, 1])
with left:
    st.image(image, caption=f"{uploaded.name} · {image.width} × {image.height}", use_container_width=True)
with right:
    st.markdown("### Image details")
    st.write(f"**Dimensions:** {image.width} × {image.height}")
    st.write(f"**Format:** {uploaded.type or 'Image'}")
    st.write("**Processing size:** 256 × 256")
    st.caption("Predictions depend on image quality and similarity to the training data.")

if image.width < 100 or image.height < 100:
    st.warning("This image is small. Try a higher-resolution photo if possible.")

try:
    with st.spinner("Extracting image features and predicting…"):
        start = time.perf_counter()
        feature_row, gray, lbp_view = extract_features(image)
        probabilities = np.asarray(model.predict_proba(feature_row)[0], dtype=float)
        duration = time.perf_counter() - start

    model_classes = list(getattr(model, "classes_", range(len(probabilities))))
    # Use model's class order to map probabilities back to the label encoder.
    if len(model_classes) != len(probabilities):
        raise ValueError("Model class metadata and probability output do not match.")

    predicted_model_class = model_classes[int(np.argmax(probabilities))]
    predicted_label = str(encoder.inverse_transform(np.array([predicted_model_class]))[0])
    probability_by_label = {}
    for model_class, probability in zip(model_classes, probabilities):
        label = str(encoder.inverse_transform(np.array([model_class]))[0])
        probability_by_label[label] = float(probability)

    confidence = probability_by_label[predicted_label]
    details = DISEASE_INFO.get(
        predicted_label,
        ("The model selected this class based on its learned image patterns.",
         "Consider expert confirmation before taking action.")
    )

    st.divider()
    st.markdown("## 📋 Prediction result")
    result_col, confidence_col = st.columns([1.4, 1])
    with result_col:
        if predicted_label.strip().lower() == "healthy":
            st.success(f"### ✅ {predicted_label}")
        else:
            st.warning(f"### 🩺 {predicted_label}")
        st.write(details[0])
        st.markdown("**Suggested next step**")
        st.write(details[1])
    with confidence_col:
        st.metric("Model confidence", f"{confidence:.1%}")
        st.progress(float(np.clip(confidence, 0, 1)))
        st.caption(f"Inference time: {duration * 1000:.1f} ms")
        st.caption("Confidence is the model's probability estimate, not a guarantee of correctness.")

    st.markdown("### 📊 Class probabilities")
    ranked = sorted(probability_by_label.items(), key=lambda item: item[1], reverse=True)
    for label, probability in ranked:
        label_col, bar_col, pct_col = st.columns([1.3, 3, .7])
        label_col.write(label)
        bar_col.progress(float(np.clip(probability, 0, 1)))
        pct_col.write(f"{probability:.1%}")

    with st.expander("🔬 View feature-extraction images"):
        image_col1, image_col2 = st.columns(2)
        with image_col1:
            st.image(gray, caption="Grayscale image used for texture features", use_container_width=True)
        with image_col2:
            st.image(lbp_view, caption="Local Binary Pattern (LBP) visualisation", use_container_width=True)
        st.caption("The classifier receives 110 numeric features—not these preview images directly.")

    with st.expander("🧠 What the ML pipeline did"):
        st.markdown(
            "- **Preprocessing:** oriented and converted the photo to RGB, then resized it to 256 × 256.\n"
            "- **HSV colour histogram:** 96 values summarise colour distribution.\n"
            "- **GLCM texture:** 4 statistics describe neighbouring-pixel texture.\n"
            "- **LBP histogram:** 10 values summarise local texture patterns.\n"
            "- **Random Forest:** combines many decision trees to predict a class."
        )
        st.caption("This explanation reflects the feature extraction implemented in this app. For best reproducibility, it must match the preprocessing used when the saved model was trained.")

    st.warning(
        "Important: this is an AI screening aid. Leaf images can be ambiguous; "
        "confirm suspected disease with an agricultural professional before treatment."
    )

except Exception as exc:
    st.error("Prediction failed.")
    st.exception(exc)
    st.info(
        "If the error mentions feature count or feature names, the app's feature-extraction "
        "pipeline may not match the preprocessing used to train the saved model."
    )
