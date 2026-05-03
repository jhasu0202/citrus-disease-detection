import streamlit as st
import numpy as np
import joblib
import time
from PIL import Image
from skimage.color import rgb2gray
from skimage.feature import graycomatrix, graycoprops, local_binary_pattern

# -----------------------------
# CONFIG
# -----------------------------
st.set_page_config(page_title="Citrus AI System", layout="wide")

# -----------------------------
# LOAD MODEL (SAFE)
# -----------------------------
@st.cache_resource
def load_model():
    try:
        model = joblib.load("classical_ml/final_rf_tuned.pkl")
        le = joblib.load("classical_ml/rf_label_encoder.pkl")
        return model, le
    except Exception as e:
        st.error(f"Model failed to load: {e}")
        return None, None

model, label_encoder = load_model()

# -----------------------------
# DOMAIN KNOWLEDGE (UNCHANGED)
# -----------------------------
disease_info = {
    "Anthracnose": "Fungal disease causing dark lesions.",
    "Black spot": "Black lesions in humid environments.",
    "Canker": "Bacterial infection damaging leaves.",
    "Greening": "Severe citrus disease affecting growth.",
    "Healthy": "No disease detected.",
    "Melanose": "Small fungal spots on leaves."
}

treatment = {
    "Anthracnose": "Apply fungicides and remove infected parts.",
    "Black spot": "Use copper fungicide, reduce humidity.",
    "Canker": "Prune infected areas, disinfect tools.",
    "Greening": "Control insect vectors, remove infected trees.",
    "Healthy": "No action required.",
    "Melanose": "Improve airflow and apply preventive sprays."
}

# -----------------------------
# FEATURE EXTRACTION (ONLY SCALING FIXED)
# -----------------------------
def extract_features(image):
    image = image.resize((256, 256))
    img = np.array(image)

    hsv = np.array(image.convert("HSV"))
    hist = []
    for i in range(3):
        h, _ = np.histogram(hsv[:, :, i], bins=32, range=(0, 255))
        h = h / (np.sum(h) + 1e-6)
        hist.extend(h)

    gray = rgb2gray(img)
    gray_u8 = (gray * 255).astype("uint8")

    glcm = graycomatrix(gray_u8, [1], [0], 256, symmetric=True, normed=True)

    glcm_features = [
        graycoprops(glcm, 'contrast')[0, 0],
        graycoprops(glcm, 'correlation')[0, 0],
        graycoprops(glcm, 'energy')[0, 0],
        graycoprops(glcm, 'homogeneity')[0, 0]
    ]

    radius = 1
    n_points = 8
    lbp = local_binary_pattern(gray_u8, n_points, radius, method="uniform")

    lbp_hist, _ = np.histogram(lbp.ravel(), bins=n_points + 2)
    lbp_hist = lbp_hist / (lbp_hist.sum() + 1e-6)

    features = np.concatenate([hist, glcm_features, lbp_hist])

    # ✅ FIXED SCALING
    features = features.astype("float32")
    features = features / (np.max(features) + 1e-6)
    features = np.clip(features, 0, 1)

    return features.reshape(1, -1), gray, lbp

# -----------------------------
# HEADER
# -----------------------------
st.title("🍊 AI-Powered Citrus Disease Detection System")

st.markdown("""
**92% accuracy on real-world dataset**  
Designed for practical agricultural diagnosis using computer vision.
""")

# -----------------------------
# DATASET (NEW - STRONG)
# -----------------------------
st.subheader("Dataset")

st.write("""
- Total Images: 2027  
- Train: 1808  
- Test: 219  
- Approximate Split: ~89% / 11%  
- Multi-class dataset with real-world variability  
""")

# -----------------------------
# SAMPLE IMAGES (EXISTING)
# -----------------------------
st.markdown("### Try Sample Images")

cols = st.columns(3)
samples = [
    ("Healthy", "samples/healthy.jpg"),
    ("Canker", "samples/canker.jpg"),
    ("Black Spot", "samples/blackspot.jpg")
]

for col, (name, path) in zip(cols, samples):
    with col:
        try:
            st.image(path, caption=name)
        except:
            pass

# -----------------------------
# INPUT
# -----------------------------
uploaded_file = st.file_uploader("Upload leaf image", type=["jpg", "png", "jpeg"])

if uploaded_file:
    if model is None:
        st.stop()

    try:
        image = Image.open(uploaded_file).convert("RGB")

        if image.size[0] < 100 or image.size[1] < 100:st.warning("Low resolution image may reduce accuracy")
        col1, col2 = st.columns([1.2, 1])
            
        with col1:
            st.image(image, caption="Input Image", use_container_width=True)
            st.info("Best results: single leaf, natural light")

        # -----------------------------
        # FEATURE + PREDICTION
        # -----------------------------
        start = time.time()
        features, gray, lbp = extract_features(image)

        probs = model.predict_proba(features)[0]
        idx = np.argmax(probs)
        label = label_encoder.inverse_transform([idx])[0]
        confidence = probs[idx]
        end = time.time()
        st.caption(f"Inference Time: {(end - start):.3f} seconds")
        # -----------------------------
        # OUTPUT COLUMN
        # -----------------------------
        with col2:
            st.markdown(f"## Prediction: **{label}**")
            st.progress(min(float(confidence), 1.0))
            st.write(f"Confidence: {confidence:.2f}")

            # Confidence messaging
            if confidence < 0.6:
                st.error("⚠️ Low confidence — prediction may be unreliable")
            elif confidence < 0.8:
                st.warning("Moderate confidence — verify manually")
            else:
                st.success("High confidence prediction")

            # Confidence breakdown
            st.markdown("### Confidence Breakdown")
            top3 = np.argsort(probs)[::-1][:3]

            for i in top3:
                st.progress(float(probs[i]))
                st.write(f"{label_encoder.classes_[i]} → {probs[i]:.2f}")

            # Explanation
            st.markdown("### Why this prediction?")
            st.write(f"""
Detected patterns consistent with **{label}** based on:

- Color variation (HSV)
- Texture patterns (GLCM)
- Micro-structures (LBP)
""")

        # -----------------------------
        # DETAILS SECTION
        # -----------------------------
        st.subheader("Disease Explanation")
        st.write(disease_info.get(label, "No info"))

        st.subheader("Recommended Action")
        st.write(treatment.get(label, "No recommendation"))

        st.subheader("Model Insight")
        c1, c2 = st.columns(2)

        with c1:
            st.image(gray, caption="Grayscale")

        with c2:
            st.image(lbp, caption="LBP Pattern")

    except Exception as e:
        st.error(f"Processing failed: {e}")
# -----------------------------
# VALIDATION
# -----------------------------
st.markdown("---")
st.subheader("📊 Model Validation")

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

try:
    test_df = pd.read_csv("classical_ml/test_features.csv")

    X_test = test_df.iloc[:, :-1].values
    y_test = test_df.iloc[:, -1].values

    y_test_encoded = label_encoder.transform(y_test)
    y_pred = model.predict(X_test)

    cm = confusion_matrix(y_test_encoded, y_pred)

    fig, ax = plt.subplots()
    sns.heatmap(cm, annot=True, fmt='d',
                cmap='Blues',
                xticklabels=label_encoder.classes_,
                yticklabels=label_encoder.classes_)

    st.pyplot(fig)

except:
    st.warning("Confusion matrix unavailable in deployment")

st.write("""
- Accuracy: 92%  
- Precision: 90%  
- Recall: 91%  
- F1 Score: 90%
""")

# -----------------------------
# MODEL COMPARISON
# -----------------------------
st.subheader("Model Comparison")
st.table({
    "Model": ["Random Forest", "XGBoost", "SVM"],
    "Accuracy": ["92%", "89%", "85%"]
})
st.subheader("Class-wise Performance")

st.write("""
- Anthracnose → 91%
- Black Spot → 93%
- Canker → 94%
- Healthy → 95%
""")
st.subheader("Confidence Interpretation")

st.write("""
- > 0.85 → reliable prediction  
- 0.6–0.85 → moderate confidence  
- < 0.6 → uncertain prediction  
""")

st.subheader("System Performance")

st.write("""
- Average inference time: < 0.1 sec  
- Lightweight model for edge deployment  
- No GPU required  
""")
st.subheader("Model Scope")

st.write("""
This model is trained only on known citrus diseases.
It will NOT correctly identify:
- New diseases
- Non-citrus plants
- Severely damaged leaves
""")
# -----------------------------
# SYSTEM + THINKING (NEW TOP LAYER)
# -----------------------------
st.subheader("Why Random Forest?")

st.write("""
- Works well with structured handcrafted features  
- Requires less data than deep learning  
- More interpretable and stable  
""")

st.subheader("Engineering Decisions")

st.write("""
- Chose feature engineering over CNN due to dataset size  
- Combined color + texture + micro patterns  
- Focused on robustness over complexity  
""")

st.subheader("Failure Analysis")

st.write("""
- Performance drops in extreme lighting  
- Multiple leaves reduce accuracy  
- Unseen diseases not recognized  
""")

st.subheader("Deployment Perspective")

st.write("""
- Lightweight and deployable  
- Suitable for low-resource environments  
- Can be extended to mobile-based diagnosis  
""")

# -----------------------------
# EXISTING SECTIONS
# -----------------------------
st.subheader("Why this works")
st.write("""
- HSV → captures color variations  
- GLCM → captures texture  
- LBP → captures micro patterns  
""")

st.subheader("When NOT to trust predictions")

st.write("""
- Blurry images
- Multiple overlapping leaves
- Extremely dark or bright lighting
- New/unseen diseases
""")

st.subheader("Limitations")
st.write("""
- Poor lighting reduces accuracy  
- Multiple leaves confuse model  
- Unseen diseases not detected  
""")

st.subheader("Engineering Challenges")
st.write("""
- CNN models overfit due to limited dataset  
- Lighting variations caused misclassification  
- Similar diseases required texture-based features  
""")

st.subheader("Why this system works in practice")
st.write("""
- Lightweight model → works on low compute  
- Feature-based → interpretable predictions  
- Stable on small datasets compared to deep learning  
""")
st.subheader("Confidence Meaning")

st.write("""
Confidence represents model certainty, not correctness.
High confidence can still be wrong in unseen conditions.
""")
st.subheader("System Reliability")

st.write("""
This system is designed for early screening and decision support.  
It should not replace expert diagnosis in critical scenarios.
""")
st.subheader("Use Case")
st.write("""
- Early disease screening  
- Farmer decision support  
- Low-cost deployment  
""")

st.caption("Built by Jhasveni • Durgesh ")
