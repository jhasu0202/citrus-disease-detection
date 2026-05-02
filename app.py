import streamlit as st
import numpy as np
import joblib
from PIL import Image
import cv2
from skimage.feature import graycomatrix, graycoprops, local_binary_pattern

st.set_page_config(page_title="Citrus Disease Detection", page_icon="🍊")

# -------------------------
# FEATURE EXTRACTION (SINGLE IMAGE)
# -------------------------
def extract_features(image):

    image = cv2.resize(image, (256, 256))

    # HSV Histogram
    hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    hist_h = cv2.calcHist([hsv], [0], None, [32], [0, 256])
    hist_s = cv2.calcHist([hsv], [1], None, [32], [0, 256])
    hist_v = cv2.calcHist([hsv], [2], None, [32], [0, 256])
    hist_features = np.concatenate((hist_h, hist_s, hist_v)).flatten()

    # GLCM
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    glcm = graycomatrix(gray, distances=[1], angles=[0], levels=256, symmetric=True, normed=True)

    contrast = graycoprops(glcm, 'contrast')[0, 0]
    correlation = graycoprops(glcm, 'correlation')[0, 0]
    energy = graycoprops(glcm, 'energy')[0, 0]
    homogeneity = graycoprops(glcm, 'homogeneity')[0, 0]
    glcm_features = np.array([contrast, correlation, energy, homogeneity])

    # LBP
    radius = 1
    n_points = 8 * radius
    lbp_bins = n_points + 2

    lbp = local_binary_pattern(gray, n_points, radius, method="uniform")
    lbp_hist, _ = np.histogram(lbp.ravel(), bins=lbp_bins, range=(0, lbp_bins))
    lbp_hist = lbp_hist.astype("float")
    lbp_hist /= (lbp_hist.sum() + 1e-6)

    # FINAL FEATURE VECTOR
    features = np.concatenate((hist_features, glcm_features, lbp_hist))

    return features


# -------------------------
# LOAD MODEL + ENCODER
# -------------------------
@st.cache_resource
def load():
    model = joblib.load("classical_ml/final_rf_tuned.pkl")
    encoder = joblib.load("classical_ml/rf_label_encoder.pkl")
    return model, encoder

try:
    model, encoder = load()
except Exception as e:
    st.error(f"❌ Model load failed: {e}")
    st.stop()


# -------------------------
# UI
# -------------------------
st.title("🍊 Citrus Disease Detection")
st.write("Upload a citrus leaf image to detect disease using ML.")

uploaded_file = st.file_uploader("Upload Image", type=["jpg", "png", "jpeg"])

# -------------------------
# PREDICTION
# -------------------------
if uploaded_file:

    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True)

    image_np = np.array(image)

    try:
        features = extract_features(image_np)
        features = features.reshape(1, -1)

        pred = model.predict(features)
        label = encoder.inverse_transform(pred)[0]

        # Confidence
        prob = model.predict_proba(features)
        confidence = np.max(prob) * 100

        st.success("Prediction complete")

        st.subheader("🧠 Result")
        st.write(f"**{label}**")

        st.subheader("📊 Confidence")
        st.write(f"{confidence:.2f}%")

    except Exception as e:
        st.error(f"❌ Prediction failed: {e}")
