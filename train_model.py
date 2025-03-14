from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import joblib
from data_processing import load_and_clean_data

# Load preprocessed data
data, scaler = load_and_clean_data("dataset.csv")

# Split data
X = data.drop(columns=['Rain_Today'])
y = data['Rain_Today']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train Model
model = LogisticRegression()
model.fit(X_train, y_train)

# Save Model & Scaler
joblib.dump(model, "rain_prediction_model.joblib")
joblib.dump(scaler, "scaler.joblib")

print("Model Training Complete ✅")
