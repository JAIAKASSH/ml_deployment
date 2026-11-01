flask_app_code = """
from flask import Flask, request, jsonify
import joblib
import numpy as np

app = Flask(__name__)

# Load the trained model and scaler
model = joblib.load('logistic_regression_model.pkl')
scaler = joblib.load('scaler.pkl')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json(force=True)
        
        # Features required by the model
        feature_keys = [
            'Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 
            'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age'
        ]
        
        # Extract values in the correct order
        features = [data[key] for key in feature_keys]
        features_array = np.array([features])
        
        # Scale features using the saved scaler
        scaled_features = scaler.transform(features_array)
        
        # Make prediction
        prediction = int(model.predict(scaled_features)[0])
        probability = float(model.predict_proba(scaled_features)[0][1])
        
        return jsonify({
            'status': 'success',
            'prediction': prediction,
            'prediction_label': 'Diabetes' if prediction == 1 else 'No Diabetes',
            'probability': probability
        })
        
    except KeyError as e:
        return jsonify({
            'status': 'error',
            'message': f'Missing feature in request payload: {str(e)}'
        }), 400
    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
"""

# Save the code to an app.py file
with open('app.py', 'w') as f:
    f.write(flask_app_code)

print("Flask application file 'app.py' has been successfully created!")
