import os
import cv2
import numpy as np
import pandas as pd
from skimage.feature import graycomatrix, graycoprops, local_binary_pattern

def extract_features_from_folder(folder_path, output_csv):

    data = []

    radius = 1
    n_points = 8 * radius
    lbp_bins = n_points + 2

    for label in os.listdir(folder_path):
        label_path = os.path.join(folder_path, label)

        if not os.path.isdir(label_path):
            continue

        print("Processing:", label)

        for file in os.listdir(label_path):
            if not file.lower().endswith(('.jpg', '.png', '.jpeg')):
                continue

            img_path = os.path.join(label_path, file)
            image = cv2.imread(img_path)

            if image is None:
                continue

            image = cv2.resize(image, (256, 256))

            # HSV histogram
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
            lbp = local_binary_pattern(gray, n_points, radius, method="uniform")
            lbp_hist, _ = np.histogram(lbp.ravel(), bins=lbp_bins, range=(0, lbp_bins))
            lbp_hist = lbp_hist.astype("float")
            lbp_hist /= (lbp_hist.sum() + 1e-6)

            features = np.concatenate((hist_features, glcm_features, lbp_hist))

            data.append(np.append(features, label))

    df = pd.DataFrame(data)
    df.to_csv(output_csv, index=False)
    print("Saved:", output_csv)


# Extract train
extract_features_from_folder("../train", "train_features.csv")

# Extract test
extract_features_from_folder("../test", "test_features.csv")
