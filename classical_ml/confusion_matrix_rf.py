import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

# Load data
test_df = pd.read_csv("test_features.csv")

X_test = test_df.iloc[:, :-1].values
y_test = test_df.iloc[:, -1].values

# Load model + encoder
model = joblib.load("final_rf_model.pkl")
le = joblib.load("rf_label_encoder.pkl")

y_test_encoded = le.transform(y_test)
y_pred = model.predict(X_test)

cm = confusion_matrix(y_test_encoded, y_pred)

plt.figure(figsize=(8,6))
sns.heatmap(cm, annot=True, fmt='d',
            xticklabels=le.classes_,
            yticklabels=le.classes_,
            cmap="Blues")

plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix - Random Forest")
plt.show()


