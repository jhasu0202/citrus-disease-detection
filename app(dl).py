"""
Citrus Leaf Disease Detection — Streamlit Demo
------------------------------------------------
Loads the fine-tuned ResNet50+CBAM model and lets you upload a leaf
photo to get a live disease prediction with confidence scores.

Setup:
    1. Download your trained weights file from Google Drive:
       citrus_project/resnet50_cbam_finetuned_best.keras
       and place it in the SAME FOLDER as this app.py
    2. pip install streamlit tensorflow pillow numpy
    3. Run:  streamlit run app.py
"""

import streamlit as st
import numpy as np
from PIL import Image
import tensorflow as tf

from tensorflow.keras.applications import ResNet50
from tensorflow.keras.applications.resnet50 import preprocess_input as resnet_preprocess
from tensorflow.keras.layers import (GlobalAveragePooling2D, GlobalMaxPooling2D, Dense, Dropout,
                                       Reshape, Multiply, Concatenate, Conv2D, Add, Activation, Lambda)
from tensorflow.keras.models import Model

# ------------------- CONFIG -------------------
WEIGHTS_PATH = "resnet50_cbam_finetuned_best.keras"
IMG_SIZE = (224, 224)
CLASS_NAMES = ["Anthracnose", "Black_Spot", "Canker", "Healthy", "Melanose"]
# ------------------------------------------------


# ---- CBAM building blocks (must match training exactly) ----
def channel_attention(input_feature, ratio=8):
    channel = input_feature.shape[-1]
    avg_pool = GlobalAveragePooling2D()(input_feature)
    avg_pool = Reshape((1, 1, channel))(avg_pool)
    avg_pool = Dense(channel // ratio, activation="relu", use_bias=False)(avg_pool)
    avg_pool = Dense(channel, use_bias=False)(avg_pool)

    max_pool = GlobalMaxPooling2D()(input_feature)
    max_pool = Reshape((1, 1, channel))(max_pool)
    max_pool = Dense(channel // ratio, activation="relu", use_bias=False)(max_pool)
    max_pool = Dense(channel, use_bias=False)(max_pool)

    attention = Add()([avg_pool, max_pool])
    attention = Activation("sigmoid")(attention)
    return Multiply()([input_feature, attention])


def spatial_attention(input_feature):
    avg_pool = Lambda(lambda x: tf.reduce_mean(x, axis=-1, keepdims=True))(input_feature)
    max_pool = Lambda(lambda x: tf.reduce_max(x, axis=-1, keepdims=True))(input_feature)
    concat = Concatenate(axis=-1)([avg_pool, max_pool])
    attention = Conv2D(1, kernel_size=7, padding="same", activation="sigmoid", use_bias=False)(concat)
    return Multiply()([input_feature, attention])


def cbam_block(input_feature, ratio=8):
    x = channel_attention(input_feature, ratio)
    x = spatial_attention(x)
    return x


@st.cache_resource
def load_trained_model():
    base_model = ResNet50(input_shape=(224, 224, 3), include_top=False, weights=None)
    x = base_model.output
    x = cbam_block(x)
    x = GlobalAveragePooling2D()(x)
    x = Dense(128, activation="relu")(x)
    x = Dropout(0.3)(x)
    output = Dense(len(CLASS_NAMES), activation="softmax")(x)

    model = Model(inputs=base_model.input, outputs=output)
    model.load_weights(WEIGHTS_PATH)
    return model


def predict(model, pil_image):
    img = pil_image.convert("RGB").resize(IMG_SIZE)
    img_array = np.array(img).astype("float32")
    img_array = np.expand_dims(img_array, axis=0)
    img_array = resnet_preprocess(img_array)

    preds = model.predict(img_array, verbose=0)[0]
    predicted_class = CLASS_NAMES[np.argmax(preds)]
    confidence = float(np.max(preds)) * 100
    return predicted_class, confidence, preds


# ------------------- UI -------------------
st.set_page_config(page_title="Citrus Leaf Disease Detection", page_icon="🍊", layout="centered")

st.title("🍊 Citrus Leaf Disease Detection")
st.write("Upload a photo of a citrus leaf to detect **Anthracnose, Black Spot, Canker, Melanose, or Healthy**.")
st.caption("Model: ResNet50 + CBAM (attention), fine-tuned — 96.7% test accuracy")

with st.spinner("Loading model..."):
    model = load_trained_model()

uploaded_file = st.file_uploader("Upload a leaf image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    pil_image = Image.open(uploaded_file)

    col1, col2 = st.columns([1, 1])
    with col1:
        st.image(pil_image, caption="Uploaded image", use_container_width=True)

    with st.spinner("Analyzing..."):
        predicted_class, confidence, all_probs = predict(model, pil_image)

    with col2:
        if predicted_class == "Healthy":
            st.success(f"### {predicted_class}")
        else:
            st.warning(f"### {predicted_class}")
        st.metric("Confidence", f"{confidence:.1f}%")

    st.subheader("Full breakdown")
    for cname, prob in sorted(zip(CLASS_NAMES, all_probs), key=lambda x: -x[1]):
        st.write(f"**{cname}**")
        st.progress(float(prob))
        st.caption(f"{prob*100:.2f}%")

    if predicted_class != "Healthy":
        st.info(
            "This is a machine learning prediction, not a diagnosis. "
            "For confirmed disease identification, consult an agricultural expert."
        )
else:
    st.info("Upload an image above to get started.")

st.divider()
st.caption(
    "Trained on a merged, deduplicated dataset of 5 disease classes. "
    "Note: this model performs best on leaf-focused, dataset-style photos — "
    "results may be less reliable on images with different lighting/framing "
    "than the training data."
)
