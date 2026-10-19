# 🍊 CitrusCare — Citrus Leaf Disease Screening

A Streamlit web app that screens citrus-leaf images using **classical machine learning**. It extracts handcrafted colour and texture features from an uploaded image and passes those features to a saved Random Forest classifier.

> **Important:** CitrusCare is an educational screening aid, not a confirmed agricultural diagnosis. Model probabilities are estimates, not guarantees. Confirm suspected disease with a qualified agricultural professional before taking treatment action.

## ✨ Features

- Upload a citrus-leaf image in **JPG, JPEG, or PNG** format.
- Correct phone-photo orientation when EXIF orientation metadata is present.
- Resize the image to **256 × 256** for consistent feature extraction.
- Extract **110 numeric features**:
  - **96 HSV colour-histogram features** — 32 bins for each of Hue, Saturation, and Value.
  - **4 GLCM texture features** — contrast, correlation, energy, and homogeneity.
  - **10 LBP histogram features** — a summary of local texture patterns.
- Predict one of six learned categories with a saved **Random Forest** model.
- Display the top prediction, class-probability estimates, inference time, and feature-visualisation previews.

## 🦠 Prediction categories

The deployed model exposes these six labels:

1. Anthracnose
2. Black spot
3. Canker
4. Greening
5. Healthy
6. Melanose

These are the model's output labels. A prediction does not prove that a disease is present or absent.

## 🧠 How it works

```text
Upload citrus-leaf image
        ↓
Open image and correct EXIF orientation
        ↓
Convert colour representation and resize to 256 × 256
        ↓
Extract handcrafted features
  ├── HSV histograms: 96 values
  ├── GLCM texture:     4 values
  └── LBP histogram:   10 values
        ↓
Combine into one 110-value feature vector
        ↓
Load/use the saved Random Forest classifier
        ↓
Map model outputs to readable class labels
        ↓
Display prediction, probability estimates, and previews
```

The current `app.py` workflow performs **inference** using saved model files; launching the app does not train the model.

## 🛠️ Tech stack

- Python
- Streamlit
- OpenCV
- Pillow
- NumPy
- scikit-image
- scikit-learn
- joblib

## 📁 Project structure

```text
citrus-disease-detection/
├── app.py                         # Current classical-ML Streamlit app
├── app(dl).py                     # Separate deep-learning app
├── requirements.txt               # Python dependencies
├── runtime.txt                    # Runtime configuration, if used by deployment
├── Citrus_ResNet50_CBAM.ipynb     # Deep-learning notebook
└── classical_ml/
    ├── final_rf_tuned.pkl         # Saved Random Forest used by app.py
    ├── rf_label_encoder.pkl       # Encoded-label to class-name mapping
    ├── extract_features_new.py    # Feature-extraction reference script
    └── ...                        # Other training/data artefacts, if present
```

The dataset and model artefacts may not all be included in a fresh clone. Confirm the required model files are present before running the app.

## 🚀 Run locally (Windows PowerShell)

### 1. Clone the repository

```powershell
git clone https://github.com/jhasu0202/citrus-disease-detection.git
cd citrus-disease-detection
```

If you have already downloaded the repository, open PowerShell in the project folder that contains `app.py`.

### 2. Create and activate a virtual environment

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

If PowerShell blocks activation, use the Python executable in the environment directly or follow your system's approved PowerShell execution-policy guidance.

### 3. Install dependencies

```powershell
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Use the project's compatible dependency versions. Loading a saved scikit-learn model with a substantially different scikit-learn version can cause warnings or compatibility problems.

### 4. Confirm the model files exist

From the same folder as `app.py`, verify that these files are available:

```text
classical_ml/final_rf_tuned.pkl
classical_ml/rf_label_encoder.pkl
```

### 5. Start the app

```powershell
python -m streamlit run app.py
```

Open the **Local URL** printed by Streamlit, usually `http://localhost:8501`. Keep the terminal open while using the app. Press **Ctrl+C** in the terminal to stop it.

## 🖼️ Using the app

1. Choose a clear image containing a citrus leaf.
2. Prefer even natural light, a focused leaf, and minimal shadow or blur.
3. Review the uploaded image preview and image details.
4. Read the predicted category and compare the alternative class probabilities.
5. Open the feature-visualisation section to inspect the grayscale and LBP previews.
6. Treat the result as a screening suggestion, not as a final diagnosis.

## 📊 Dataset and evaluation status

The application exposes six classes, but **the exact dataset source, per-class image counts, and split used to train the currently deployed model should be documented from the training scripts and dataset manifest**.

Do not infer dataset composition or test performance from a single prediction. A sample result such as “90.8% model confidence” is not the same as 90.8% test accuracy.

Before publishing a definitive performance claim, verify and report:

- Dataset name/source and licence.
- Total number of images and number in each class.
- Exact train/validation/test split and random seed, if applicable.
- Whether splitting was stratified and whether duplicate/related images could cross subsets.
- Held-out test accuracy, per-class precision and recall, macro-F1, and a confusion matrix.
- The model and feature-extraction versions used for that evaluation.

**No test-accuracy figure is asserted in this README until it can be tied to a reproducible evaluation of the model being shipped.**

## ⚠️ Limitations

- Lighting, background, camera, and image quality can affect handcrafted colour and texture features.
- Visually similar diseases can produce similar symptoms.
- Model probability estimates may not be calibrated and are not guarantees of correctness.
- Results may not generalise to leaf images unlike those represented in the training data.
- The app classifies the uploaded image as a whole; it does not provide a verified diagnosis or identify exact lesion boundaries.

## 🔭 Possible future improvements

- Document dataset provenance, licence, class balance, and the verified split.
- Add a reproducible evaluation script and publish a confusion matrix and per-class metrics.
- Test on independent field images from different devices and lighting conditions.
- Evaluate probability calibration and add stronger input-quality checks.
- Compare the classical feature-based model with a CNN using the same leakage-safe evaluation protocol.

## 🔗 Repository and demo

- **GitHub repository:** https://github.com/jhasu0202/citrus-disease-detection
- **Streamlit demo:** https://citrus-disease-detection-cr6iiqj3enapgoqbucckpg.streamlit.app/

If the hosted demo is unavailable, run the app locally using the instructions above.

## 👤 Author

**GitHub:** [jhasu0202](https://github.com/jhasu0202)

---

*Built for learning and demonstration. CitrusCare is not a substitute for professional agricultural advice.*
