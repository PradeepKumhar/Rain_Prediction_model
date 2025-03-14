import joblib
import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from data_processing import load_and_clean_data

# Load model & scaler
model = joblib.load("rain_prediction_model.joblib")

# Load test data
data, _ = load_and_clean_data("dataset.csv")
X = data.drop(columns=['Rain_Today'])
y = data['Rain_Today']

# Predictions
y_pred = model.predict(X)

# Evaluation
accuracy = accuracy_score(y, y_pred)
print(f"Model Accuracy: {accuracy:.2f}")

print("Confusion Matrix:\n", confusion_matrix(y, y_pred))
print("Classification Report:\n", classification_report(y, y_pred))
