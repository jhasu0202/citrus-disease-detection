import streamlit as st
import cv2
import numpy as np
from PIL import Image
import joblib

st.title("🍊 Citrus Disease Detection")

# Load your trained model
model = joblib.load("clasical_ml/model_new.pkl")

uploaded_file = st.file_uploader("Upload a leaf image", type=["jpg","png","jpeg"])

if uploaded_file:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # preprocess (you must match your training logic)
    img = np.array(image)
    img = cv2.resize(img, (224,224))
    img = img.flatten().reshape(1, -1)

    prediction = model.predict(img)

    st.success(f"Prediction: {prediction[0]}")
