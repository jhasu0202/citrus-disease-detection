import cv2
import numpy as np
import joblib
from skimage.feature import graycomatrix, graycoprops, local_binary_pattern

# -------------------------
# Load Model & Label Encoder
# -------------------------
model = joblib.load("final_rf_model.pkl")
le = joblib.load("rf_label_encoder.pkl")

# -------------------------
# Feature Extraction (MUST MATCH TRAINING)
# -------------------------
def extract_features(image_path):

    image = cv2.imread(image_path)

    if image is None:
        raise ValueError("Image not found or incorrect path.")

    image = cv2.resize(image, (256, 256))

    # HSV Histogram
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    hist_h = cv2.calcHist([hsv], [0], None, [32], [0, 256])
    hist_s = cv2.calcHist([hsv], [1], None, [32], [0, 256])
    hist_v = cv2.calcHist([hsv], [2], None, [32], [0, 256])
    hist_features = np.concatenate((hist_h, hist_s, hist_v)).flatten()

    # GLCM
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    glcm = graycomatrix(gray, distances=[1], angles=[0], levels=256, symmetric=True, normed=True)

    contrast = graycoprops(glcm, 'contrast')[0, 0]
    correlation = graycoprops(glcm, 'correlation')[0, 0]
    energy = graycoprops(glcm, 'energy')[0, 0]
    homogeneity = graycoprops(glcm, 'homogeneity')[0, 0]
    glcm_features = np.array([contrast, correlation, energy, homogeneity])

    # LBP
    radius = 1
    n_points = 8 * radius
    lbp_bins = n_points + 2
    lbp = local_binary_pattern(gray, n_points, radius, method="uniform")
    lbp_hist, _ = np.histogram(lbp.ravel(), bins=lbp_bins, range=(0, lbp_bins))
    lbp_hist = lbp_hist.astype("float")
    lbp_hist /= (lbp_hist.sum() + 1e-6)

    features = np.concatenate((hist_features, glcm_features, lbp_hist))

    return features.reshape(1, -1)


# -------------------------
# Predict Function
# -------------------------
def predict_disease(image_path):
    features = extract_features(image_path)
    prediction = model.predict(features)
    label = le.inverse_transform(prediction)
    return label[0]


# -------------------------
# Main
# -------------------------
if __name__ == "__main__":
    image_path = input("Enter path of leaf image: ")
    result = predict_disease(image_path)
    print("\nPredicted Disease:", result)
