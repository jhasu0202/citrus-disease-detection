import streamlit as st
import numpy as np
import joblib
from PIL import Image

# -------------------------
# PAGE CONFIG
# -------------------------
st.set_page_config(
    page_title="Citrus Disease Detection",
    page_icon="🍊",
    layout="centered"
)

# -------------------------
# LOAD MODEL
# -------------------------
@st.cache_resource
def load_model():
    return joblib.load("clasical_ml/model_new.pkl")

try:
    model = load_model()
except Exception as e:
    st.error(f"❌ Model failed to load: {e}")
    st.stop()

# -------------------------
# TITLE
# -------------------------
st.title("🍊 Citrus Disease Detection")
st.markdown("Upload a citrus leaf image to detect disease using a trained ML model.")

# -------------------------
# FILE UPLOAD
# -------------------------
uploaded_file = st.file_uploader(
    "Upload Leaf Image",
    type=["jpg", "jpeg", "png"]
)

# -------------------------
# PREDICTION LOGIC
# -------------------------
if uploaded_file is not None:

    # Show image
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Preprocess
    image = image.resize((224, 224))  # adjust if your model uses different size
    image = np.array(image)

    # Handle grayscale or RGBA edge cases
    if image.shape[-1] == 4:
        image = image[:, :, :3]

    # Flatten for classical ML
    image = image.flatten().reshape(1, -1)

    # Predict
    try:
        prediction = model.predict(image)

        st.success("✅ Prediction Complete")

        # Output
        st.subheader("🧠 Result:")
        st.write(f"**{prediction[0]}**")

    except Exception as e:
        st.error(f"❌ Prediction failed: {e}")

# -------------------------
# FOOTER (optional but strong)
# -------------------------
st.markdown("---")
st.markdown(
    "Built with **Machine Learning · Computer Vision · Streamlit**"
)
