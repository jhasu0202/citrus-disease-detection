import streamlit as st
import numpy as np
import joblib
from PIL import Image

st.set_page_config(page_title="Citrus Disease Detection", page_icon="🍊")

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
    st.error(f"❌ Failed to load model: {e}")
    st.stop()

# -------------------------
# UI
# -------------------------
st.title("🍊 Citrus Disease Detection")
st.write("Upload a citrus leaf image to detect disease.")

uploaded_file = st.file_uploader("Upload Image", type=["jpg", "png", "jpeg"])

# -------------------------
# PROCESS
# -------------------------
if uploaded_file:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # preprocess
    image = image.resize((224, 224))
    image = np.array(image)

    if image.shape[-1] == 4:
        image = image[:, :, :3]

    image = image.flatten().reshape(1, -1)

    # predict
    pred = model.predict(image)
    label = encoder.inverse_transform(pred)[0]

    st.success("Prediction complete")

    st.subheader("🧠 Result:")
    st.write(f"**{label}**")
