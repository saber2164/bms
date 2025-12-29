#!/usr/bin/env python3
"""
Refactored Inference Script for Dual UKF.

Improvements:
1.  **Scaler Loading**: Loads the `ocv_scaler.save` to ensure correct input scaling.
2.  **Model Loading**: Loads the new Dense OCV model.
3.  **Initialization**: Correctly initializes the DualUKF with the scaler.
"""

import os
import sys
import argparse
import joblib
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import load_model

# ensure repo root is on sys.path
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if REPO_ROOT not in sys.path:
    sys.path.append(REPO_ROOT)

from scripts.ukf_soc import DualUKF

def infer_single_file(input_csv, ocv_model_path, scaler_path, initial_params):
    print(f"Loading data from {input_csv}...")
    df = pd.read_csv(input_csv)

    print(f"Loading OCV model from {ocv_model_path}...")
    ocv_model = load_model(ocv_model_path)
    
    print(f"Loading Scaler from {scaler_path}...")
    scaler = joblib.load(scaler_path)

    # Initialize DUKF
    dukf = DualUKF(
        dt=initial_params.get('dt', 1.0),
        C_nom=initial_params.get('C_nom', 2.0),
        R0_nom=initial_params.get('R0_nom', 0.01),
        ocv_model=ocv_model,
        scaler=scaler
    )

    if 'initial_soc' in initial_params:
        dukf.x[0] = initial_params['initial_soc']

    results = []
    print("Running DUKF inference...")
    
    # Standardize columns if needed
    col_map = {
        'Current_measured': 'Current',
        'Voltage_measured': 'Voltage',
        'Temperature_measured': 'Temperature',
        'Time': 'Time'
    }
    df = df.rename(columns=col_map)
    
    required_cols = ['Voltage', 'Current', 'Temperature', 'Time']
    if not all(col in df.columns for col in required_cols):
        print(f"Error: Input file {input_csv} is missing required columns. Found: {df.columns.tolist()}")
        return

    for _, row in df.iterrows():
        v_meas = row['Voltage']
        i_k = row['Current']
        temp_k = row['Temperature']

        state, params = dukf.step(v_meas, i_k, temp_k)
        
        results.append({
            'Time': row['Time'],
            'Voltage': v_meas,
            'Current': i_k,
            'Temperature': temp_k,
            'SoC_estimated': state[0],
            'Q_max_estimated': params[0],
            'R_0_estimated': params[1]
        })

    results_df = pd.DataFrame(results)
    
    output_filename = os.path.join('outputs', os.path.basename(input_csv).replace('.csv', '_dukf_inference.csv'))
    results_df.to_csv(output_filename, index=False)
    print(f"Inference results saved to {output_filename}")

    print("\n--- Inference Summary ---")
    print(f"Final Estimated Capacity (Q_max): {results_df['Q_max_estimated'].iloc[-1]:.4f} Ah")
    print(f"Final Estimated Resistance (R_0): {results_df['R_0_estimated'].iloc[-1]:.4f} Ohms")
    print("-------------------------\n")

def main():
    parser = argparse.ArgumentParser(description="Run DUKF Inference.")
    parser.add_argument('--input', type=str, required=True)
    parser.add_argument('--ocv-model', type=str, default='outputs/final_ocv_model.keras')
    parser.add_argument('--scaler', type=str, default='outputs/ocv_scaler.save')
    parser.add_argument('--initial-soc', type=float, default=0.9)
    parser.add_argument('--C-nom', type=float, default=2.0)
    parser.add_argument('--R0-nom', type=float, default=0.01)
    args = parser.parse_args()

    initial_params = {
        'initial_soc': args.initial_soc,
        'C_nom': args.C_nom,
        'R0_nom': args.R0_nom
    }

    infer_single_file(args.input, args.ocv_model, args.scaler, initial_params)

if __name__ == "__main__":
    if not os.path.exists('outputs'):
        os.makedirs('outputs')
    main()
