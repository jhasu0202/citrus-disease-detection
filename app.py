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
    "Greening": "Serious disease affecting citrus production and fruit quality.",
    "Healthy": "Leaf shows no disease symptoms.",
    "Melanose": "Fungal disease causing small dark spots on leaves."
}

treatment = {
    "Anthracnose": "Apply fungicides and remove infected parts.",
    "Black spot": "Use copper fungicide and maintain dryness.",
    "Canker": "Prune infected areas and apply antibacterial spray.",
    "Greening": "Control insect vectors and remove infected trees.",
    "Healthy": "No treatment required.",
    "Melanose": "Apply fungicides and improve airflow."
}

# -----------------------
# FEATURE EXTRACTION
# -----------------------
def extract_features(image):

    image = image.resize((256, 256))
    img = np.array(image)

    # HSV histogram
    hsv = np.array(image.convert("HSV"))
    hist_h, _ = np.histogram(hsv[:,:,0], bins=32, range=(0,255))
    hist_s, _ = np.histogram(hsv[:,:,1], bins=32, range=(0,255))
    hist_v, _ = np.histogram(hsv[:,:,2], bins=32, range=(0,255))

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
    lbp_hist, _ = np.histogram(lbp.ravel(), bins=n_points+2, range=(0, n_points+2))
    lbp_hist = lbp_hist / (lbp_hist.sum() + 1e-6)

    features = np.concatenate([hist_features, glcm_features, lbp_hist])

    return features.reshape(1, -1)


# -----------------------
# UI
# -----------------------
st.set_page_config(page_title="Citrus Disease Detection", layout="centered")

st.title("🍊 Citrus Disease Detection System")

st.write("""
Detect citrus leaf diseases using a **Machine Learning system** based on:
- Color Features (HSV)
- Texture Features (GLCM)
- Pattern Features (LBP)
- Random Forest Classifier
""")

# -----------------------
# UPLOAD
# -----------------------
uploaded_file = st.file_uploader("Upload Leaf Image", type=["jpg", "png", "jpeg"])

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")

    st.image(image, caption="Uploaded Image", use_container_width=True)

    # Quality check
    if image.size[0] < 100 or image.size[1] < 100:
        st.warning("⚠️ Low resolution image. Results may be inaccurate.")

    # Extract features
    features = extract_features(image)

    # Prediction
    prediction = model.predict(features)
    label = label_encoder.inverse_transform(prediction)[0]

    # Confidence
    probs = model.predict_proba(features)
    confidence = np.max(probs)

    # -----------------------
    # OUTPUT
    # -----------------------
    st.success(f"Prediction: {label}")

    st.subheader("📊 Confidence")
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
# EXTRA SECTIONS
# -----------------------
st.markdown("---")

st.subheader("⚙️ How it Works")
st.write("""
1. Image preprocessing  
2. Feature extraction (HSV, GLCM, LBP)  
3. Random Forest classification  
4. Disease prediction  
""")

st.subheader("📊 Model Performance")
st.write("Model achieved ~92% accuracy on test dataset.")

st.subheader("🌱 Real-World Impact")
st.write("""
• Early disease detection  
• Reduces crop loss  
• Supports farmers in decision-making  
""")

# -----------------------
# FOOTER
# -----------------------
st.markdown("---")
st.caption("Built by Jhasveni | Computer Vision & AI")
