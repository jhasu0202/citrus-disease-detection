import joblib
import matplotlib.pyplot as plt
import numpy as np

model = joblib.load("final_rf_model.pkl")

importances = model.feature_importances_

# Get top 15 features
indices = np.argsort(importances)[-15:]

plt.figure(figsize=(8,6))
plt.barh(range(len(indices)), importances[indices])
plt.yticks(range(len(indices)), indices)
plt.title("Top 15 Feature Importances - Random Forest")
plt.xlabel("Importance")
plt.show()
