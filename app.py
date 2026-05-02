import streamlit as st
import numpy as np
import joblib
from PIL import Image
from skimage.color import rgb2gray
from skimage.feature import graycomatrix, graycoprops, local_binary_pattern

# ---------------------------
# LOAD MODEL + LABEL ENCODER
# ---------------------------
model = joblib.load("classical_ml/final_rf_tuned.pkl")
label_encoder = joblib.load("classical_ml/rf_label_encoder.pkl")

# ---------------------------
# FEATURE EXTRACTION FUNCTION
# ---------------------------
def extract_features(image):

    # Resize
    image = image.resize((256, 256))
    img = np.array(image)

    # ---------- HSV HISTOGRAM ----------
    hsv = np.array(image.convert("HSV"))

    hist_h, _ = np.histogram(hsv[:, :, 0], bins=32, range=(0, 255))
    hist_s, _ = np.histogram(hsv[:, :, 1], bins=32, range=(0, 255))
    hist_v, _ = np.histogram(hsv[:, :, 2], bins=32, range=(0, 255))

    hist_features = np.concatenate([hist_h, hist_s, hist_v])

    # ---------- GLCM ----------
    gray = rgb2gray(img)
    gray = (gray * 255).astype("uint8")

    glcm = graycomatrix(gray, [1], [0], levels=256, symmetric=True, normed=True)

    contrast = graycoprops(glcm, 'contrast')[0, 0]
    correlation = graycoprops(glcm, 'correlation')[0, 0]
    energy = graycoprops(glcm, 'energy')[0, 0]
    homogeneity = graycoprops(glcm, 'homogeneity')[0, 0]

    glcm_features = np.array([contrast, correlation, energy, homogeneity])

    # ---------- LBP ----------
    radius = 1
    n_points = 8 * radius

    lbp = local_binary_pattern(gray, n_points, radius, method="uniform")
    lbp_hist, _ = np.histogram(lbp.ravel(), bins=n_points + 2, range=(0, n_points + 2))
    lbp_hist = lbp_hist.astype("float")
    lbp_hist /= (lbp_hist.sum() + 1e-6)

    # ---------- FINAL FEATURE VECTOR ----------
    features = np.concatenate([hist_features, glcm_features, lbp_hist])

    return features.reshape(1, -1)


# ---------------------------
# UI STARTS HERE
# ---------------------------
st.set_page_config(page_title="Citrus Disease Detection", layout="centered")

st.title("🍊 Citrus Disease Detection System")

st.write("""
Upload a citrus leaf image to detect disease using a machine learning model 
trained with **feature engineering (HSV + GLCM + LBP)** and Random Forest.
""")

# ---------------------------
# FILE UPLOAD
# ---------------------------
uploaded_file = st.file_uploader("Upload Image", type=["jpg", "png", "jpeg"])

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")

    st.image(image, caption="Uploaded Image", use_container_width=True)

    # Extract features
    features = extract_features(image)

    # Prediction
    prediction = model.predict(features)
    label = label_encoder.inverse_transform(prediction)[0]

    # Confidence
    probabilities = model.predict_proba(features)
    confidence = np.max(probabilities)

    # ---------------------------
    # OUTPUT
    # ---------------------------
    st.success(f"Prediction: {label}")
    st.info(f"Confidence: {confidence:.2f}")

    # ---------------------------
    # MODEL EXPLANATION
    # ---------------------------
    st.subheader("🧠 What the model analyzes")

    st.write("""
    • **Color Features (HSV Histogram)** – captures leaf color patterns  
    • **Texture Features (GLCM)** – identifies disease texture differences  
    • **Pattern Features (LBP)** – detects micro-pattern variations  
    """)

# ---------------------------
# PERFORMANCE SECTION
# ---------------------------
st.subheader("📊 Model Performance")

st.write("Model trained using Random Forest with ~92% accuracy on test data.")

# OPTIONAL (only if you upload images to repo)
# st.image("confusion_matrix.png")
# st.image("accuracy_plot.png")

# ---------------------------
# REAL WORLD IMPACT
# ---------------------------
st.subheader("🌱 Real-world Impact")

st.write("""
• Early detection of plant diseases  
• Helps farmers reduce crop loss  
• Supports precision agriculture  
""")

# ---------------------------
# FOOTER
# ---------------------------
st.markdown("---")
st.caption("Built by Jhasveni • Computer Vision & AI")
