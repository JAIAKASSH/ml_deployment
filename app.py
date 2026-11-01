```python
import os
from pathlib import Path

import joblib
import numpy as np
from flask import Flask, request, jsonify

app = Flask(__name__)

BASE_DIR = Path(__file__).resolve().parent

# Load the trained model and scaler
model = joblib.load(BASE_DIR / "logistic_regression_model.pkl")
scaler = joblib.load(BASE_DIR / "scaler.pkl")

FEATURE_KEYS = [
    "Pregnancies",
    "Glucose",
    "BloodPressure",
    "SkinThickness",
    "Insulin",
    "BMI",
    "DiabetesPedigreeFunction",
    "Age",
]


@app.route("/", methods=["GET"])
def home():
    return jsonify({
        "status": "success",
        "message": "Diabetes Prediction API is running!"
    })


@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json(silent=True)

        if not isinstance(data, dict):
            return jsonify({
                "status": "error",
                "message": "Send a valid JSON object."
            }), 400

        missing = [key for key in FEATURE_KEYS if key not in data]

        if missing:
            return jsonify({
                "status": "error",
                "message": "Missing required features.",
                "missing_features": missing
            }), 400

        try:
            features = [float(data[key]) for key in FEATURE_KEYS]
        except (TypeError, ValueError):
            return jsonify({
                "status": "error",
                "message": "All features must be numeric."
            }), 400

        if not np.all(np.isfinite(features)):
            return jsonify({
                "status": "error",
                "message": "Features must be finite numbers."
            }), 400

        features_array = np.array([features], dtype=float)
        scaled_features = scaler.transform(features_array)

        prediction = int(model.predict(scaled_features)[0])

        probabilities = model.predict_proba(scaled_features)[0]
        classes = list(model.classes_)
        positive_index = classes.index(1)
        probability = float(probabilities[positive_index])

        return jsonify({
            "status": "success",
            "prediction": prediction,
            "prediction_label": (
                "Diabetes" if prediction == 1 else "No Diabetes"
            ),
            "probability": probability
        })

    except Exception:
        app.logger.exception("Prediction failed")

        return jsonify({
            "status": "error",
            "message": "Prediction failed. Check server logs."
        }), 500


if __name__ == "__main__":
    app.run(
        host="0.0.0.0",
        port=int(os.environ.get("PORT", 10000)),
        debug=False
    )
```
