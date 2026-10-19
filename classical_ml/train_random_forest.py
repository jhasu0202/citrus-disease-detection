import pandas as pd
import joblib
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

# -------------------------
# Load Train Data
# -------------------------
train_df = pd.read_csv("train_features.csv")
test_df = pd.read_csv("test_features.csv")

X_train = train_df.iloc[:, :-1].values
y_train = train_df.iloc[:, -1].values

X_test = test_df.iloc[:, :-1].values
y_test = test_df.iloc[:, -1].values

# -------------------------
# Encode Labels
# -------------------------
le = LabelEncoder()
y_train_encoded = le.fit_transform(y_train)
y_test_encoded = le.transform(y_test)

print("Classes:", le.classes_)

# -------------------------
# Train Random Forest
# -------------------------
rf_model = RandomForestClassifier(
    n_estimators=300,
    random_state=42
)

rf_model.fit(X_train, y_train_encoded)

# -------------------------
# Evaluate
# -------------------------
y_pred = rf_model.predict(X_test)

accuracy = accuracy_score(y_test_encoded, y_pred)

print("\nTest Accuracy:", accuracy)
print("\nClassification Report:\n")
print(classification_report(y_test_encoded, y_pred))

# -------------------------
# Save Model
# -------------------------
joblib.dump(rf_model, "final_rf_model.pkl")
joblib.dump(le, "rf_label_encoder.pkl")

print("\nRandom Forest model saved as final_rf_model.pkl")
print("Label encoder saved as rf_label_encoder.pkl")
