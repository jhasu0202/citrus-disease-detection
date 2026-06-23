import streamlit as st
import numpy as np
import joblib
import time
import cv2
from PIL import Image
from skimage.feature import graycomatrix, graycoprops, local_binary_pattern

# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────
st.set_page_config(page_title="Citrus AI System", layout="wide")

# ─────────────────────────────────────────────
# LOAD MODEL
# ─────────────────────────────────────────────
@st.cache_resource
def load_model():
    try:
        model = joblib.load("classical_ml/final_rf_tuned.pkl")
        le    = joblib.load("classical_ml/rf_label_encoder.pkl")
        return model, le
    except Exception as e:
        st.error(f"Model failed to load: {e}")
        return None, None

model, label_encoder = load_model()

# ─────────────────────────────────────────────
# DOMAIN KNOWLEDGE
# ─────────────────────────────────────────────
disease_info = {
    "Anthracnose": "Fungal disease causing dark lesions on leaves and fruits.",
    "Black spot":  "Black lesions that develop in humid environments.",
    "Canker":      "Bacterial infection that damages leaves and stems.",
    "Greening":    "Severe citrus disease affecting growth and yield.",
    "Healthy":     "No disease detected. Leaf appears healthy.",
    "Melanose":    "Small fungal spots commonly seen on older leaves."
}

treatment = {
    "Anthracnose": "Apply appropriate fungicides and remove infected plant parts.",
    "Black spot":  "Use copper-based fungicide. Reduce leaf wetness and humidity.",
    "Canker":      "Prune infected branches. Disinfect tools between cuts.",
    "Greening":    "Control psyllid insect vectors. Remove and destroy infected trees.",
    "Healthy":     "No action required. Continue regular crop management.",
    "Melanose":    "Improve airflow around plants. Apply preventive fungicide sprays."
}

# ─────────────────────────────────────────────
# FEATURE EXTRACTION
# EXACTLY matches extract_features_new.py
# (the file used to create train_features.csv)
# ─────────────────────────────────────────────
def extract_features(pil_image):
    """
    Matches training code extract_features_new.py exactly:
      - Resize : 256 × 256  (NOT 224)
      - HSV    : all channels range [0, 256]  (NOT [0,180] for H)
      - GLCM   : distances=[1], angles=[0]
                 order: contrast, correlation, energy, homogeneity
      - LBP    : radius=1, n_points=8, uniform
                 normalised by sum  (NOT density=True)
      Total    : 96 + 4 + 10 = 110 features ✅
    """

    # ── Convert PIL → BGR numpy (same as cv2.imread) ─────────────
    img_rgb = np.array(pil_image.convert("RGB"))
    image   = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)

    # ── Resize to 256×256 (matches training) ─────────────────────
    image = cv2.resize(image, (256, 256))

    # ── HSV histogram — 96 features ──────────────────────────────
    # Training used [0,256] for ALL three channels including Hue
    hsv    = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    hist_h = cv2.calcHist([hsv], [0], None, [32], [0, 256])
    hist_s = cv2.calcHist([hsv], [1], None, [32], [0, 256])
    hist_v = cv2.calcHist([hsv], [2], None, [32], [0, 256])
    hist_features = np.concatenate((hist_h, hist_s, hist_v)).flatten()
    # 32 × 3 = 96 features

    # ── GLCM — 4 features ────────────────────────────────────────
    # Order must match training: contrast, correlation, energy, homogeneity
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    glcm        = graycomatrix(gray, distances=[1], angles=[0],
                               levels=256, symmetric=True, normed=True)
    contrast    = graycoprops(glcm, 'contrast')[0, 0]
    correlation = graycoprops(glcm, 'correlation')[0, 0]
    energy      = graycoprops(glcm, 'energy')[0, 0]
    homogeneity = graycoprops(glcm, 'homogeneity')[0, 0]
    glcm_features = np.array([contrast, correlation, energy, homogeneity])
    # 4 features

    # ── LBP histogram — 10 features ──────────────────────────────
    radius   = 1
    n_points = 8 * radius        # = 8
    lbp_bins = n_points + 2      # = 10
    lbp      = local_binary_pattern(gray, n_points, radius, method="uniform")
    lbp_hist, _ = np.histogram(lbp.ravel(), bins=lbp_bins, range=(0, lbp_bins))
    lbp_hist = lbp_hist.astype("float")
    lbp_hist /= (lbp_hist.sum() + 1e-6)   # normalise by sum (matches training)
    # 10 features

    # ── Concatenate ───────────────────────────────────────────────
    features = np.concatenate((hist_features, glcm_features, lbp_hist))
    # 96 + 4 + 10 = 110 ✅

    assert features.shape[0] == 110, \
        f"Feature mismatch: got {features.shape[0]}, expected 110"

    # LBP visualisation for display
    lbp_display = (lbp / (lbp.max() + 1e-6) * 255).astype(np.uint8)

    return features.reshape(1, -1), gray, lbp_display


# ─────────────────────────────────────────────
# UI
# ─────────────────────────────────────────────
st.title("🍊 AI-Powered Citrus Disease Detection System")
st.markdown("**92% accuracy on real-world dataset** — Computer vision for agricultural diagnosis.")

st.subheader("Dataset")
st.write("""
- Total Images: 2027  |  Train: 1808  |  Test: 219
- 6 classes: Anthracnose, Black Spot, Canker, Greening, Healthy, Melanose
""")

# ─────────────────────────────────────────────
# FILE UPLOADER
# ─────────────────────────────────────────────
uploaded_file = st.file_uploader("Upload a citrus leaf image", type=["jpg","jpeg","png"])

if uploaded_file:
    if model is None:
        st.stop()

    try:
        pil_image = Image.open(uploaded_file).convert("RGB")

        if pil_image.size[0] < 100 or pil_image.size[1] < 100:
            st.warning("⚠️ Low resolution image may reduce accuracy.")

        col1, col2 = st.columns([1.2, 1])

        with col1:
            st.image(pil_image, caption="Uploaded Leaf Image", use_container_width=True)
            st.info("💡 Best results: single leaf, natural daylight, plain background")

        # ── Predict ───────────────────────────────────────────────
        start = time.time()
        features, gray_u8, lbp_display = extract_features(pil_image)
        probs      = model.predict_proba(features)[0]
        idx        = np.argmax(probs)
        label      = label_encoder.inverse_transform([idx])[0]
        confidence = probs[idx]
        elapsed    = time.time() - start

        with col1:
            st.caption(f"⏱️ Inference time: {elapsed:.3f} seconds")

        # ── Results ───────────────────────────────────────────────
        with col2:
            st.markdown(f"## 🔬 Prediction: **{label}**")
            st.progress(min(float(confidence), 1.0))
            st.write(f"**Confidence: {confidence*100:.1f}%**")

            if confidence < 0.6:
                st.error("⚠️ Low confidence — try a clearer image.")
            elif confidence < 0.8:
                st.warning("Moderate confidence — verify manually.")
            else:
                st.success("✅ High confidence prediction")

            st.markdown("### 📊 Top-3 Predictions")
            for i in np.argsort(probs)[::-1][:3]:
                cls_name = label_encoder.classes_[i]
                st.progress(float(probs[i]))
                st.write(f"**{cls_name}** → {probs[i]*100:.1f}%")

            st.markdown("### 🧬 Why this prediction?")
            st.write(f"""
Detected patterns consistent with **{label}** based on:
- 🎨 **Colour variation** (HSV histogram)
- 🔲 **Texture patterns** (GLCM)
- 🔬 **Micro-structures** (LBP)
            """)

        st.subheader("📋 Disease Information")
        st.write(disease_info.get(label, "Information not available."))

        st.subheader("💊 Recommended Action")
        st.write(treatment.get(label, "No recommendation available."))

        st.subheader("🔍 Feature Visualisation")
        c1, c2 = st.columns(2)
        with c1:
            st.image(gray_u8, caption="Grayscale (GLCM + LBP input)", clamp=True)
        with c2:
            st.image(lbp_display, caption="LBP Pattern", clamp=True)

    except Exception as e:
        st.error(f"❌ Processing failed: {e}")

# ─────────────────────────────────────────────
# MODEL VALIDATION
# ─────────────────────────────────────────────
st.markdown("---")
st.subheader("📊 Model Validation")

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

try:
    test_df    = pd.read_csv("classical_ml/test_features.csv")
    X_test     = test_df.iloc[:, :-1].values
    y_test     = test_df.iloc[:, -1].values
    y_test_enc = label_encoder.transform(y_test)
    y_pred     = model.predict(X_test)
    cm         = confusion_matrix(y_test_enc, y_pred)
    fig, ax    = plt.subplots(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=label_encoder.classes_,
                yticklabels=label_encoder.classes_, ax=ax)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title("Confusion Matrix")
    st.pyplot(fig)
except Exception:
    st.warning("Confusion matrix unavailable (test_features.csv not found)")
    st.write("**Reported metrics:** Accuracy: 92% | Precision: 90% | Recall: 91% | F1: 90%")

st.subheader("⚖️ Model Comparison")
st.table({
    "Model":    ["Random Forest ✅", "XGBoost", "SVM"],
    "Accuracy": ["92%",              "89%",     "85%"]
})

st.markdown("---")
st.subheader("⚙️ System Design")
col_a, col_b = st.columns(2)

with col_a:
    st.markdown("**Why Random Forest?**")
    st.write("""
- Works well with handcrafted features (HSV + GLCM + LBP)
- Does not require large dataset like CNN
- Interpretable and computationally lightweight
    """)
    st.markdown("**Feature Engineering**")
    st.write("""
- **HSV (96 features):** colour variation in diseased vs healthy leaves
- **GLCM (4 features):** contrast, correlation, energy, homogeneity
- **LBP (10 features):** micro-structural boundary patterns
    """)

with col_b:
    st.markdown("**Known Limitations**")
    st.write("""
- Performance drops under extreme/uneven lighting
- Multiple overlapping leaves reduce accuracy
- Unseen disease types are not recognised
- Blurry or very low-resolution images give unreliable results
    """)
    st.markdown("**Confidence Interpretation**")
    st.write("""
- > 85% → Reliable prediction
- 60–85% → Moderate confidence, verify manually
- < 60% → Uncertain — retake photo in better lighting
    """)

st.markdown("---")
st.info("This system is designed for early screening and decision support — "
        "not a replacement for expert agronomic diagnosis.")
st.caption("Built by Jhasveni • Durgesh")
