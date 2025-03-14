from flask import Flask, request, jsonify
import joblib
import numpy as np

app = Flask(__name__)

# Load model & scaler
model = joblib.load("rain_prediction_model.joblib")
scaler = joblib.load("scaler.joblib")

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.json  # JSON input
        features = np.array(data['features']).reshape(1, -1)
        scaled_features = scaler.transform(features)  # Scale input

        prediction = model.predict(scaled_features)[0]
        return jsonify({'Rain_Today': 'Yes' if prediction == 1 else 'No'})
    
    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(debug=True)
