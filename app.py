import streamlit as st
import numpy as np
import joblib
from PIL import Image
from skimage.color import rgb2gray
from skimage.feature import graycomatrix, graycoprops, local_binary_pattern

# -----------------------
# LOAD MODEL
# -----------------------
model = joblib.load("classical_ml/final_rf_tuned.pkl")
label_encoder = joblib.load("classical_ml/rf_label_encoder.pkl")

# -----------------------
# DISEASE INFO
# -----------------------
disease_info = {
    "Anthracnose": "Fungal disease causing dark lesions on leaves and fruits.",
    "Black spot": "Causes black circular spots on leaves; spreads in humid conditions.",
    "Canker": "Bacterial infection leading to lesions and leaf drop.",
    "Greening": "Severe disease affecting citrus growth and fruit quality.",
    "Healthy": "Leaf shows no visible disease symptoms.",
    "Melanose": "Fungal disease causing small dark spots on leaves."
}

treatment = {
    "Anthracnose": "Apply fungicides and remove infected areas.",
    "Black spot": "Use copper-based fungicide and avoid excess moisture.",
    "Canker": "Prune infected branches and apply antibacterial spray.",
    "Greening": "Control insect vectors and remove infected plants.",
    "Healthy": "No treatment required.",
    "Melanose": "Improve airflow and apply preventive fungicide."
}

# -----------------------
# FEATURE EXTRACTION
# -----------------------
def extract_features(image):

    image = image.resize((256, 256))
    img = np.array(image)

    # HSV histogram
    hsv = np.array(image.convert("HSV"))
    hist_h, _ = np.histogram(hsv[:, :, 0], bins=32, range=(0, 255))
    hist_s, _ = np.histogram(hsv[:, :, 1], bins=32, range=(0, 255))
    hist_v, _ = np.histogram(hsv[:, :, 2], bins=32, range=(0, 255))

    hist_features = np.concatenate([hist_h, hist_s, hist_v])

    # GLCM
    gray = rgb2gray(img)
    gray = (gray * 255).astype("uint8")

    glcm = graycomatrix(gray, [1], [0], levels=256, symmetric=True, normed=True)

    contrast = graycoprops(glcm, 'contrast')[0, 0]
    correlation = graycoprops(glcm, 'correlation')[0, 0]
    energy = graycoprops(glcm, 'energy')[0, 0]
    homogeneity = graycoprops(glcm, 'homogeneity')[0, 0]

    glcm_features = np.array([contrast, correlation, energy, homogeneity])

    # LBP
    radius = 1
    n_points = 8 * radius

    lbp = local_binary_pattern(gray, n_points, radius, method="uniform")
    lbp_hist, _ = np.histogram(lbp.ravel(), bins=n_points + 2, range=(0, n_points + 2))
    lbp_hist = lbp_hist / (lbp_hist.sum() + 1e-6)

    features = np.concatenate([hist_features, glcm_features, lbp_hist])

    return features.reshape(1, -1)

# -----------------------
# UI
# -----------------------
st.set_page_config(page_title="Citrus Disease Detection", layout="centered")

st.title("🍊 Citrus Disease Detection System")

st.write("""
Detect citrus leaf diseases using a **feature-engineered Machine Learning system**  
(Hue-Saturation-Value, Texture Analysis, Pattern Recognition + Random Forest).
""")

# -----------------------
# SUPPORTED CLASSES
# -----------------------
st.subheader("📂 Supported Diseases")
st.write("""
• Anthracnose  
• Black Spot  
• Canker  
• Greening  
• Melanose  
• Healthy  
""")

# -----------------------
# FILE UPLOAD
# -----------------------
uploaded_file = st.file_uploader("Upload Leaf Image", type=["jpg", "png", "jpeg"])

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")

    st.image(image, caption="Uploaded Image", use_container_width=True)

    # Quality warning
    if image.size[0] < 100 or image.size[1] < 100:
        st.warning("⚠️ Low-quality image may reduce accuracy.")

    # Feature extraction
    features = extract_features(image)

    # Prediction
    prediction = model.predict(features)
    label = label_encoder.inverse_transform(prediction)[0]

    # Confidence
    probs = model.predict_proba(features)
    confidence = np.max(probs)

    # -----------------------
    # RESULT
    # -----------------------
    st.success(f"🧠 Diagnosis: {label}")

    if confidence > 0.85:
        st.info("High confidence prediction")
    elif confidence > 0.65:
        st.warning("Moderate confidence — verify manually")
    else:
        st.error("Low confidence — image may be unclear")

    st.subheader("📊 Confidence Level")
    st.progress(float(confidence))
    st.write(f"{confidence:.2f}")

    # -----------------------
    # DETAILS
    # -----------------------
    st.subheader("🧾 Disease Details")
    st.write(disease_info.get(label, "No information available"))

    st.subheader("💡 Recommended Action")
    st.write(treatment.get(label, "No recommendation available"))

    # -----------------------
    # WHY PREDICTION
    # -----------------------
    st.subheader("🔍 Why this prediction?")

    st.write("""
The model analyzes:
• Color distribution (HSV histograms)  
• Texture properties (GLCM features)  
• Local patterns (LBP features)  

These combined features help distinguish visually similar diseases.
""")

# -----------------------
# SYSTEM INFO
# -----------------------
st.markdown("---")

st.subheader("⚙️ How it Works")
st.write("""
1. Image preprocessing  
2. Feature extraction (HSV + GLCM + LBP)  
3. Random Forest classification  
4. Disease prediction  
""")

st.subheader("📊 Model Validation")
st.write("""
• Trained on labeled citrus dataset  
• Evaluated on unseen test data  
• Achieved ~92% accuracy  
• Classical ML approach (feature-engineered)
""")

st.subheader("⚠️ Limitations")
st.write("""
• Requires clear leaf images  
• Performance drops with noise or blur  
• Cannot detect multiple diseases simultaneously  
""")

st.subheader("🌱 Real-World Impact")
st.write("""
• Early disease detection  
• Reduces crop loss  
• Supports precision agriculture  
""")

# -----------------------
# FOOTER
# -----------------------
st.markdown("---")
st.caption("Built by Jhasveni | Computer Vision & AI Systems")
