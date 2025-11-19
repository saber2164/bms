import os
import numpy as np
import pandas as pd
from flask import Flask, render_template, request, jsonify
import tensorflow as tf
from scripts.rul_predictor import RULPredictor
import joblib

app = Flask(__name__)

# Load Model and Scaler
SEQUENCE_LENGTH = 15
N_FEATURES = 6 # SoH, Temp, Re, Rct, IC_Peak, IC_Volt
EOL_SOH = 0.8

print("Loading model and scaler...")
try:
    # Initialize predictor
    predictor = RULPredictor(sequence_length=SEQUENCE_LENGTH, n_features=N_FEATURES)
    model = predictor.build_model()
    model.load_weights('outputs/rul_model.weights.h5')
    print("Model loaded.")
    
    scaler = joblib.load('outputs/feature_scaler.pkl')
    print("Scaler loaded.")
except Exception as e:
    print(f"Error loading model/scaler: {e}")
    model = None
    scaler = None

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/predict', methods=['POST'])
def predict():
    if model is None or scaler is None:
        return jsonify({'error': 'Model not loaded'}), 500
        
    try:
        data = request.json
        # Let's assume inputs are raw and we scale them roughly based on dataset stats.
        # SoH: 0-1 (already scaled effectively if capacity/nominal)
        # Temp: ~20-40?
        # Re: ~0.05
        # Rct: ~0.15
        
        # TODO: In a real app, save and load the scaler.
        # For now, we will pass values through as-is, assuming the user (or UI) handles normalization 
        # or the values are close enough to the training distribution (0-1).
        # Actually, let's just document that inputs should be normalized 0-1 for now.
        
        # Reshape for model: (1, SEQUENCE_LENGTH, N_FEATURES)
        input_reshaped = input_data.reshape(1, SEQUENCE_LENGTH, N_FEATURES)
        
        prediction = rul_predictor.model.predict(input_reshaped)
        rul_pred = float(prediction[0][0])
        
        return jsonify({'rul': rul_pred})

    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
