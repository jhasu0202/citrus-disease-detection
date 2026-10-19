import pandas as pd
import joblib
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier

# -------------------------
# Load Data
# -------------------------
train_df = pd.read_csv("train_features.csv")
test_df = pd.read_csv("test_features.csv")

X_train = train_df.iloc[:, :-1].values
y_train = train_df.iloc[:, -1].values

X_test = test_df.iloc[:, :-1].values
y_test = test_df.iloc[:, -1].values

# Encode labels
le = LabelEncoder()
y_train = le.fit_transform(y_train)
y_test = le.transform(y_test)

# Scale features (important for SVM, KNN, Logistic)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Store results
results = {}

# -------------------------
# Logistic Regression
# -------------------------
log_model = LogisticRegression(max_iter=5000)
log_model.fit(X_train_scaled, y_train)
log_pred = log_model.predict(X_test_scaled)
results["Logistic Regression"] = accuracy_score(y_test, log_pred)

# -------------------------
# KNN
# -------------------------
knn_model = KNeighborsClassifier(n_neighbors=5)
knn_model.fit(X_train_scaled, y_train)
knn_pred = knn_model.predict(X_test_scaled)
results["KNN"] = accuracy_score(y_test, knn_pred)

# -------------------------
# Random Forest
# -------------------------
rf_model = RandomForestClassifier(n_estimators=200, random_state=42)
rf_model.fit(X_train, y_train)
rf_pred = rf_model.predict(X_test)
results["Random Forest"] = accuracy_score(y_test, rf_pred)

# -------------------------
# SVM (RBF)
# -------------------------
svm_model = SVC(kernel='rbf')
svm_model.fit(X_train_scaled, y_train)
svm_pred = svm_model.predict(X_test_scaled)
results["SVM (RBF)"] = accuracy_score(y_test, svm_pred)

# -------------------------
# XGBoost
# -------------------------
xgb_model = XGBClassifier(
    n_estimators=200,
    max_depth=5,
    learning_rate=0.1,
    random_state=42,
    eval_metric='mlogloss'
)
xgb_model.fit(X_train, y_train)
xgb_pred = xgb_model.predict(X_test)
results["XGBoost"] = accuracy_score(y_test, xgb_pred)

# -------------------------
# Print Results
# -------------------------
print("\nModel Comparison (Test Accuracy):\n")
for model, acc in results.items():
    print(f"{model}: {acc:.4f}")

