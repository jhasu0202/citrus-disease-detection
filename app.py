import streamlit as st
import numpy as np
import joblib
from PIL import Image
from skimage.color import rgb2gray
from skimage.feature import graycomatrix, graycoprops, local_binary_pattern

# -----------------------
# PAGE CONFIG
# -----------------------
st.set_page_config(page_title="Citrus AI System", layout="wide")

# -----------------------
# LOAD MODEL
# -----------------------
@st.cache_resource
def load_model():
    try:
        model = joblib.load("classical_ml/final_rf_tuned.pkl")
        le = joblib.load("classical_ml/rf_label_encoder.pkl")
        return model, le
    except Exception as e:
        st.error(f"Model loading failed: {e}")
        return None, None

model, label_encoder = load_model()

# -----------------------
# DOMAIN DATA
# -----------------------
disease_info = {
    "Anthracnose": "Fungal disease causing dark lesions.",
    "Black spot": "Black lesions in humid conditions.",
    "Canker": "Bacterial infection damaging leaves.",
    "Greening": "Severe citrus disease affecting growth.",
    "Healthy": "No disease detected.",
    "Melanose": "Small fungal spots on leaves."
}

treatment = {
    "Anthracnose": "Use fungicides.",
    "Black spot": "Apply copper spray.",
    "Canker": "Remove infected leaves.",
    "Greening": "Control insect vectors.",
    "Healthy": "No action needed.",
    "Melanose": "Improve airflow."
}

# -----------------------
# FEATURE EXTRACTION
# -----------------------
def extract_features(image):
    image = image.resize((256, 256))
    img = np.array(image)

    # HSV
    hsv = np.array(image.convert("HSV"))
    hist = []
    for i in range(3):
        h, _ = np.histogram(hsv[:, :, i], bins=32, range=(0, 255))
        hist.extend(h)

    # GLCM
    gray = rgb2gray(img)
    gray_u8 = (gray * 255).astype("uint8")
    glcm = graycomatrix(gray_u8, [1], [0], 256, symmetric=True, normed=True)

    glcm_features = [
        graycoprops(glcm, 'contrast')[0, 0],
        graycoprops(glcm, 'correlation')[0, 0],
        graycoprops(glcm, 'energy')[0, 0],
        graycoprops(glcm, 'homogeneity')[0, 0]
    ]

    # LBP
    radius = 1
    n_points = 8
    lbp = local_binary_pattern(gray_u8, n_points, radius, method="uniform")
    lbp_hist, _ = np.histogram(lbp.ravel(), bins=n_points + 2)

    lbp_hist = lbp_hist / (lbp_hist.sum() + 1e-6)

    features = np.concatenate([hist, glcm_features, lbp_hist]).reshape(1, -1)

    return features, gray, lbp

# -----------------------
# UI HEADER
# -----------------------
st.title("🍊 Citrus Disease Detection System")
st.write("Feature-engineered ML system using HSV + GLCM + LBP")

# -----------------------
# INPUT
# -----------------------
file = st.file_uploader("Upload leaf image", type=["jpg", "png", "jpeg"])

if file:
    if model is None:
        st.stop()

    try:
        image = Image.open(file).convert("RGB")

        col1, col2 = st.columns(2)

        with col1:
            st.image(image, caption="Uploaded Image")

        features, gray, lbp = extract_features(image)

        probs = model.predict_proba(features)[0]
        idx = np.argmax(probs)
        label = label_encoder.inverse_transform([idx])[0]
        confidence = probs[idx]

        with col2:
            st.success(f"Prediction: {label}")

            if confidence > 0.85:
                st.info("High confidence")
            elif confidence > 0.6:
                st.warning("Moderate confidence")
            else:
                st.error("Low confidence")

            st.progress(float(confidence))
            st.write(f"Confidence: {confidence:.2f}")

            st.write("Top Predictions:")
            top3 = np.argsort(probs)[::-1][:3]
            for i in top3:
                st.write(f"{label_encoder.classes_[i]} → {probs[i]:.2f}")

        # -----------------------
        # DETAILS
        # -----------------------
        st.subheader("Disease Info")
        st.write(disease_info.get(label, ""))

        st.subheader("Recommended Action")
        st.write(treatment.get(label, ""))

        # -----------------------
        # VISUALIZATION
        # -----------------------
        st.subheader("Model Insight")
        c1, c2 = st.columns(2)

        with c1:
            st.image(gray, caption="Grayscale")

        with c2:
            st.image(lbp, caption="LBP Texture")

    except Exception as e:
        st.error(f"Error: {e}")

# -----------------------
# SYSTEM INFO
# -----------------------
st.markdown("---")

st.subheader("Model Details")
st.write("""
- Random Forest
- Feature-based ML (~110 features)
- Accuracy: ~92%
""")

st.subheader("Limitations")
st.write("""
- Sensitive to lighting
- Needs single leaf
- May fail on unseen data
""")

st.subheader("Best Input Conditions")
st.info("Use clear image, natural light, single leaf")

st.caption("Built by Jhasveni • AI Engineer")
