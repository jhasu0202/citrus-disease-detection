import streamlit as st
import numpy as np
import joblib
from PIL import Image
from skimage.color import rgb2gray
from skimage.feature import graycomatrix, graycoprops, local_binary_pattern

# Load model
model = joblib.load("classical_ml/final_rf_tuned.pkl")
label_encoder = joblib.load("classical_ml/rf_label_encoder.pkl")

# Feature extraction (MATCH TRAINING)
def extract_features(image):

    image = image.resize((256, 256))
    img = np.array(image)

    # --- HSV approximation (without cv2)
    hsv = np.array(image.convert("HSV"))

    hist_h, _ = np.histogram(hsv[:,:,0], bins=32, range=(0,255))
    hist_s, _ = np.histogram(hsv[:,:,1], bins=32, range=(0,255))
    hist_v, _ = np.histogram(hsv[:,:,2], bins=32, range=(0,255))

    hist_features = np.concatenate([hist_h, hist_s, hist_v])

    # --- GLCM
    gray = rgb2gray(img)
    gray = (gray * 255).astype("uint8")

    glcm = graycomatrix(gray, [1], [0], levels=256, symmetric=True, normed=True)

    contrast = graycoprops(glcm, 'contrast')[0, 0]
    correlation = graycoprops(glcm, 'correlation')[0, 0]
    energy = graycoprops(glcm, 'energy')[0, 0]
    homogeneity = graycoprops(glcm, 'homogeneity')[0, 0]

    glcm_features = np.array([contrast, correlation, energy, homogeneity])

    # --- LBP
    radius = 1
    n_points = 8 * radius

    lbp = local_binary_pattern(gray, n_points, radius, method="uniform")
    lbp_hist, _ = np.histogram(lbp.ravel(), bins=n_points+2, range=(0, n_points+2))
    lbp_hist = lbp_hist / (lbp_hist.sum() + 1e-6)

    # Final feature vector
    features = np.concatenate([hist_features, glcm_features, lbp_hist])

    return features.reshape(1, -1)


# UI
st.title("🍊 Citrus Disease Detection")

uploaded_file = st.file_uploader("Upload leaf image", type=["jpg", "png", "jpeg"])

if uploaded_file:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_container_width=True)

    features = extract_features(image)

    pred = model.predict(features)
    label = label_encoder.inverse_transform(pred)

    st.success(f"Prediction: {label[0]}")
