# 🍊 Citrus Disease Detection using Computer Vision

## 📌 Problem

Citrus crops are highly susceptible to diseases that significantly impact yield and quality.
Manual inspection by farmers is time-consuming, inconsistent, and often inaccurate, especially under varying lighting and environmental conditions.

---

## 💡 Solution

This project implements a computer vision-based system to automatically detect and classify diseases in citrus leaves from images.

The system processes leaf images, extracts relevant features, and predicts the disease category using a trained machine learning model.

---

## ⚙️ Approach

### 1. Data Preprocessing

* Image resizing and normalization
* Noise reduction and enhancement
* Handling variability in lighting conditions

### 2. Feature Extraction

* Color-based features
* Texture analysis
* Shape-based characteristics

### 3. Model Training

* Applied classical machine learning algorithms
* Trained on labeled citrus leaf dataset
* Optimized features to improve classification performance

---

## 📊 Results

* Achieved **92% accuracy** on the test dataset
* Dataset split: **80% training / 20% testing** *(modify if different)*

### Evaluation Metrics

* Accuracy: 92%
* Precision: XX% *(add if available)*
* Recall: XX% *(add if available)*
* Confusion Matrix used for error analysis

---

## 🧠 Key Observations

* Model accuracy improved after applying image preprocessing techniques
* Feature selection played a key role in distinguishing disease patterns
* Misclassifications mainly occurred between visually similar disease classes

---

## 🛠️ Tech Stack

* Python
* OpenCV
* scikit-learn
* NumPy, Pandas

---

## 📸 Sample Outputs

*(Add screenshots here — very important)*

* Input leaf image
* Predicted disease label
* Output visualization

---

## ⚠️ Limitations

* Sensitive to lighting variations
* Limited generalization on unseen datasets
* Requires better dataset diversity

---

## 🚀 Future Improvements

* Upgrade to deep learning models (CNN, ResNet)
* Add real-time detection via web/mobile app
* Improve robustness with larger and diverse datasets

---

## 📁 Project Structure

```
project/
│── data/ (not included in repo)
│── src/
│── models/
│── outputs/
│── main.py
│── requirements.txt
```

---

## ▶️ How to Run

```bash
pip install -r requirements.txt
python main.py
```

---

## 📎 Note

Dataset is not included in this repository due to size constraints.
Public citrus disease datasets can be used for testing and validation.

---
