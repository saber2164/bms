#!/usr/bin/env python3
"""
Validates the trained OCV-LSTM model by plotting the OCV-SoC curve.

This script loads the trained OCV-LSTM model and generates OCV-SoC curves
for different temperatures to allow for visual inspection.
"""

import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt
import argparse

def validate_ocv_model(model_path, output_path):
    """
    Loads the OCV model and plots the OCV-SoC curves.
    """
    print(f"Loading OCV-LSTM model from {model_path}...")
    try:
        ocv_model = load_model(model_path)
    except Exception as e:
        print(f"Error loading model: {e}")
        return

    print("Generating OCV-SoC curves...")
    
    # Generate a range of SoC values
    soc_values = np.linspace(0, 1, 100)
    
    # Define a few representative temperatures
    temperatures = [10, 25, 40] # in degC
    
    plt.figure(figsize=(10, 6))
    
    for temp in temperatures:
        ocv_predictions = []
        for soc in soc_values:
            # Reshape for LSTM: (1, 1, 2) -> (batch_size, timesteps, features)
            model_input = np.array([[soc, temp]]).reshape((1, 1, 2))
            ocv = ocv_model.predict(model_input, verbose=0)[0, 0]
            ocv_predictions.append(ocv)
            
        plt.plot(soc_values, ocv_predictions, label=f'Temp = {temp}°C')

    plt.title('OCV-SoC Curve from LSTM Model')
    plt.xlabel('State of Charge (SoC)')
    plt.ylabel('Open-Circuit Voltage (OCV)')
    plt.grid(True)
    plt.legend()
    
    # Save the plot
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path)
    print(f"OCV-SoC curve plot saved to {output_path}")

def main():
    parser = argparse.ArgumentParser(description="Validate the OCV-LSTM model.")
    parser.add_argument('--model', type=str, default='outputs/final_ocv_lstm.keras', help='Path to the trained OCV-LSTM model.')
    parser.add_argument('--output', type=str, default='outputs/ocv_validation.png', help='Path to save the output plot.')
    args = parser.parse_args()
    
    validate_ocv_model(args.model, args.output)

if __name__ == "__main__":
    main()
