import joblib
import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, roc_curve, auc
import matplotlib.pyplot as plt
from data_processing import load_and_clean_data

# Load Model
model = joblib.load("rain_prediction_model.joblib")

# Load Test Data
data, _ = load_and_clean_data("dataset.csv")
X = data.drop(columns=['Rain_Today'])
y = data['Rain_Today']

# Predictions
y_pred = model.predict(X)
y_probs = model.predict_proba(X)[:, 1]

# 🔹 Accuracy Score
accuracy = accuracy_score(y, y_pred)
print(f"✅ Model Accuracy: {accuracy:.2f}")

# 🔹 Confusion Matrix
conf_matrix = confusion_matrix(y, y_pred)
print("✅ Confusion Matrix:\n", conf_matrix)

# 🔹 Classification Report
print("✅ Classification Report:\n", classification_report(y, y_pred))

# 🔹 ROC Curve & AUC Score
fpr, tpr, _ = roc_curve(y, y_probs)
roc_auc = auc(fpr, tpr)

plt.figure()
plt.plot(fpr, tpr, color='blue', label=f'ROC Curve (AUC = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='gray', linestyle='--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend(loc='lower right')
plt.show()

print(f"✅ AUC Score: {roc_auc:.2f}")
