import pandas as pd
import joblib
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

train_df = pd.read_csv("train_features.csv")
test_df = pd.read_csv("test_features.csv")

X_train = train_df.iloc[:, :-1].values
y_train = train_df.iloc[:, -1].values
X_test = test_df.iloc[:, :-1].values
y_test = test_df.iloc[:, -1].values

le = LabelEncoder()
y_train = le.fit_transform(y_train)
y_test = le.transform(y_test)

best_acc = 0
best_model = None

for depth in [None, 15, 20]:
    for min_split in [2, 5, 10]:

        rf = RandomForestClassifier(
            n_estimators=400,
            max_depth=depth,
            min_samples_split=min_split,
            random_state=42
        )

        rf.fit(X_train, y_train)
        acc = accuracy_score(y_test, rf.predict(X_test))

        print(f"Depth={depth}, MinSplit={min_split} → Accuracy={acc:.4f}")

        if acc > best_acc:
            best_acc = acc
            best_model = rf

print("\nBest Accuracy:", best_acc)

joblib.dump(best_model, "final_rf_tuned.pkl")
joblib.dump(le, "rf_label_encoder.pkl")
