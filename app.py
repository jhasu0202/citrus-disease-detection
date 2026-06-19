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
# FEATURE EXTRACTION  ← ALL BUGS FIXED HERE
# ─────────────────────────────────────────────
def extract_features(pil_image):
    """
    FIX SUMMARY vs old app.py:
    1. Resize to 224×224 (matches training, not 256×256)
    2. Use cv2.cvtColor for HSV  (not colorsys — different H range!)
    3. GLCM uses distances=[1,3,5], angles=[0,π/4,π/2,3π/4] (matches training)
    4. LBP uses radius=3, n_points=24 (matches training)
    5. NO final re-normalisation (scaler inside model pipeline handles this)
    """

    # ── Step 1: Resize to 224×224 (must match training) ──────────
    img_rgb = np.array(pil_image.resize((224, 224)))   # shape (224,224,3) RGB

    # ── Step 2: Convert to BGR for OpenCV ────────────────────────
    img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)

    # ── Step 3: HSV features via cv2  ← BUG FIX #1 ──────────────
    # OLD code used colorsys → H range [0,1] → scaled to [0,255]
    # cv2 HSV        → H range [0,180], S,V [0,255]  ← CORRECT (matches training)
    img_hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    hsv_features = []
    ranges = [(0, 180), (0, 256), (0, 256)]   # H, S, V ranges in cv2
    for i, (lo, hi) in enumerate(ranges):
        hist = cv2.calcHist([img_hsv], [i], None, [32], [lo, hi])
        hist = cv2.normalize(hist, hist).flatten()
        hsv_features.extend(hist)
    # = 96 features

    # ── Step 4: Grayscale for texture features ────────────────────
    gray_u8 = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)   # uint8 [0,255]

    # ── Step 5: GLCM features  ← BUG FIX #2 ──────────────────────
    # OLD code: distances=[1], angles=[0]  → only 4 features
    # FIXED:    distances=[1,3,5], angles=[0,π/4,π/2,3π/4] → 10 features
    glcm = graycomatrix(
        gray_u8,
        distances=[1, 3, 5],
        angles=[0, np.pi/4, np.pi/2, 3*np.pi/4],
        levels=256,
        symmetric=True,
        normed=True
    )
    props = ['contrast', 'dissimilarity', 'homogeneity', 'energy', 'correlation']
    glcm_features = []
    for prop in props:
        vals = graycoprops(glcm, prop)
        glcm_features.append(np.mean(vals))
        glcm_features.append(np.std(vals))
    # = 10 features

    # ── Step 6: LBP features  ← BUG FIX #3 ───────────────────────
    # OLD code: radius=1, n_points=8  → 10 histogram bins
    # FIXED:    radius=3, n_points=24 → 26 histogram bins
    radius   = 3
    n_points = 8 * radius   # = 24
    lbp = local_binary_pattern(gray_u8, n_points, radius, method='uniform')
    n_bins = n_points + 2   # = 26
    lbp_hist, _ = np.histogram(lbp.ravel(), bins=n_bins,
                               range=(0, n_bins), density=True)
    # = 26 features

    # ── Step 7: Concatenate all features ─────────────────────────
    features = np.concatenate([hsv_features, glcm_features, lbp_hist])
    # Total: 96 + 10 + 26 = 132 features
    # (adjust if your training notebook used different params)

    # ── BUG FIX #4: Do NOT re-normalise here ─────────────────────
    # OLD code: features = features / np.max(features)  ← WRONG
    # The StandardScaler INSIDE the sklearn Pipeline handles scaling.
    # Re-normalising here destroys what the model learned.

    # Also return gray and LBP visualisation for display
    lbp_display = (lbp / (lbp.max() + 1e-6) * 255).astype(np.uint8)

    return features.reshape(1, -1), gray_u8, lbp_display


# ─────────────────────────────────────────────
# UI HEADER
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
            st.warning("⚠️ Low resolution image may reduce accuracy. Use a clear close-up photo.")

        col1, col2 = st.columns([1.2, 1])

        with col1:
            st.image(pil_image, caption="Uploaded Leaf Image", use_container_width=True)
            st.info("💡 Best results: single leaf, natural daylight, plain background")

        # ── EXTRACT + PREDICT ─────────────────────────────────
        start = time.time()
        features, gray_u8, lbp_display = extract_features(pil_image)
        probs       = model.predict_proba(features)[0]
        idx         = np.argmax(probs)
        label       = label_encoder.inverse_transform([idx])[0]
        confidence  = probs[idx]
        elapsed     = time.time() - start

        with col1:
            st.caption(f"⏱️ Inference time: {elapsed:.3f} seconds")

        # ── RESULTS ──────────────────────────────────────────
        with col2:
            st.markdown(f"## 🔬 Prediction: **{label}**")
            st.progress(min(float(confidence), 1.0))
            st.write(f"**Confidence: {confidence*100:.1f}%**")

            if confidence < 0.6:
                st.error("⚠️ Low confidence — prediction may be unreliable. Try a clearer image.")
            elif confidence < 0.8:
                st.warning("Moderate confidence — verify manually if possible.")
            else:
                st.success("✅ High confidence prediction")

            st.markdown("### 📊 Top-3 Predictions")
            top3 = np.argsort(probs)[::-1][:3]
            for i in top3:
                cls_name = label_encoder.classes_[i]
                st.progress(float(probs[i]))
                st.write(f"**{cls_name}** → {probs[i]*100:.1f}%")

            st.markdown("### 🧬 Why this prediction?")
            st.write(f"""
Detected patterns consistent with **{label}** based on:
- 🎨 **Colour variation** (HSV histogram — leaf colour and saturation)
- 🔲 **Texture patterns** (GLCM — roughness, contrast, lesion texture)
- 🔬 **Micro-structures** (LBP — edge and boundary patterns)
            """)

        # ── DISEASE DETAILS ───────────────────────────────────
        st.subheader("📋 Disease Information")
        st.write(disease_info.get(label, "Information not available."))

        st.subheader("💊 Recommended Action")
        st.write(treatment.get(label, "No recommendation available."))

        # ── FEATURE VISUALISATIONS ────────────────────────────
        st.subheader("🔍 Feature Visualisation")
        c1, c2 = st.columns(2)
        with c1:
            st.image(gray_u8, caption="Grayscale (used for GLCM + LBP)", clamp=True)
        with c2:
            st.image(lbp_display, caption="LBP Pattern (texture micro-structures)", clamp=True)

    except Exception as e:
        st.error(f"❌ Processing failed: {e}")
        st.write("**Common causes:** corrupted image file, unsupported format, model not loaded.")

# ─────────────────────────────────────────────
# MODEL VALIDATION SECTION
# ─────────────────────────────────────────────
st.markdown("---")
st.subheader("📊 Model Validation")

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

try:
    test_df = pd.read_csv("classical_ml/test_features.csv")
    X_test  = test_df.iloc[:, :-1].values
    y_test  = test_df.iloc[:, -1].values
    y_test_enc = label_encoder.transform(y_test)
    y_pred  = model.predict(X_test)

    cm = confusion_matrix(y_test_enc, y_pred)
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=label_encoder.classes_,
                yticklabels=label_encoder.classes_, ax=ax)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title("Confusion Matrix")
    st.pyplot(fig)
except Exception:
    st.warning("Confusion matrix unavailable in deployment (test_features.csv not found)")
    st.write("**Reported metrics:** Accuracy: 92% | Precision: 90% | Recall: 91% | F1: 90%")

# ─────────────────────────────────────────────
# MODEL COMPARISON
# ─────────────────────────────────────────────
st.subheader("⚖️ Model Comparison")
st.table({
    "Model":    ["Random Forest ✅", "XGBoost", "SVM"],
    "Accuracy": ["92%",             "89%",     "85%"]
})

# ─────────────────────────────────────────────
# SYSTEM DESIGN
# ─────────────────────────────────────────────
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
- **HSV (96 features):** captures colour variation in diseased vs healthy leaves
- **GLCM (10 features):** captures texture roughness and lesion patterns
- **LBP (26 features):** captures micro-structural boundaries and edges
    """)

with col_b:
    st.markdown("**Known Limitations**")
    st.write("""
- Performance drops under extreme/uneven lighting
- Multiple overlapping leaves reduce accuracy
- Unseen disease types are not recognized
- Blurry or very low-resolution images give unreliable results
    """)
    st.markdown("**Confidence Interpretation**")
    st.write("""
- > 85% → Reliable prediction
- 60–85% → Moderate confidence, verify manually
- < 60% → Uncertain — retake photo in better lighting
    """)

st.markdown("---")
st.info("This system is designed for early screening and decision support — not a replacement for expert agronomic diagnosis.")
st.caption("Built by Jhasveni • Durgesh | IEEE Published Research")
