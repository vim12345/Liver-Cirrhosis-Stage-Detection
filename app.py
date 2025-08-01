from flask import Flask, request, jsonify
import joblib
import numpy as np
import pandas as pd

app = Flask(__name__)

# Load model and tools
model = joblib.load("liver_stage_model.pkl")
scaler = joblib.load("scaler.pkl")
imputer = joblib.load("imputer.pkl")
encoders = joblib.load("encoders.pkl")

# Expected feature order
features = [
    'N_Days', 'Status', 'Drug', 'Age', 'Sex', 'Ascites', 'Hepatomegaly', 'Spiders',
    'Edema', 'Bilirubin', 'Cholesterol', 'Albumin', 'Copper', 'Alk_Phos',
    'SGOT', 'Tryglicerides', 'Platelets', 'Prothrombin'
]

@app.route('/predict', methods=['POST'])
def predict():
    try:
        input_data = request.get_json()
        df = pd.DataFrame([input_data])

        # Apply label encoders
        for col, le in encoders.items():
            df[col] = le.transform(df[col].astype(str))

        # Reorder and impute
        df = df[features]
        df_imputed = pd.DataFrame(imputer.transform(df), columns=features)

        # Scale
        df_scaled = scaler.transform(df_imputed)

        # Predict
        pred = model.predict(df_scaled)
        return jsonify({'predicted_stage': int(pred[0])})
    except Exception as e:
        return jsonify({'error': str(e)}), 400

if __name__ == '__main__':
    app.run(debug=True)