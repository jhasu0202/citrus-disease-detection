import pandas as pd
import joblib
from sklearn.metrics import classification_report, accuracy_score

# Load test
test_df = pd.read_csv("test_features.csv")

X_test = test_df.iloc[:, :-1].values
y_test = test_df.iloc[:, -1].values

# Load model + encoder
model = joblib.load("model_new.pkl")
le = joblib.load("label_encoder.pkl")

y_test_encoded = le.transform(y_test)

y_pred = model.predict(X_test)

print("Test Accuracy:", accuracy_score(y_test_encoded, y_pred))
print("\nClassification Report:\n")
print(classification_report(y_test_encoded, y_pred))

