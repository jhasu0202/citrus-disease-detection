import pandas as pd
import joblib
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, accuracy_score
from xgboost import XGBClassifier

# Load train
train_df = pd.read_csv("train_features.csv")

X_train = train_df.iloc[:, :-1].values
y_train = train_df.iloc[:, -1].values

le = LabelEncoder()
y_train = le.fit_transform(y_train)

# Train model
model = XGBClassifier(
    n_estimators=200,
    max_depth=5,
    learning_rate=0.1,
    random_state=42,
    eval_metric='mlogloss'
)

model.fit(X_train, y_train)

# Save label encoder + model
joblib.dump(model, "model_new.pkl")
joblib.dump(le, "label_encoder.pkl")

print("Training completed.")
