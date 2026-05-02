import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score
import numpy as np

# Load data
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

# Train Random Forest
rf = RandomForestClassifier(n_estimators=200, random_state=42)
rf.fit(X_train, y_train)

# Train XGBoost
xgb = XGBClassifier(
    n_estimators=200,
    max_depth=5,
    learning_rate=0.1,
    random_state=42,
    eval_metric='mlogloss'
)
xgb.fit(X_train, y_train)

# Get probability predictions
rf_probs = rf.predict_proba(X_test)
xgb_probs = xgb.predict_proba(X_test)

# Average probabilities
avg_probs = (rf_probs + xgb_probs) / 2

# Final prediction
y_pred = np.argmax(avg_probs, axis=1)

print("Ensemble Accuracy:", accuracy_score(y_test, y_pred))
