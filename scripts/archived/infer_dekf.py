#!/usr/bin/env python3
"""
Inference script for the Dual EKF-LSTM model.

This script loads a trained OCV-LSTM model and the DualEKF estimator
to run inference on a given battery data CSV file.
"""

import os
import sys
import argparse
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import load_model

# ensure repo root is on sys.path
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if REPO_ROOT not in sys.path:
    sys.path.append(REPO_ROOT)

from scripts.dekf_soc import DualEKF

def infer_single_file(input_csv, ocv_model_path, initial_params):
    """
    Runs DEKF inference on a single file.

    Args:
        input_csv (str): Path to the input CSV file.
        ocv_model_path (str): Path to the trained OCV-LSTM Keras model.
        initial_params (dict): Dictionary of initial parameters for the DEKF.
    """
    print(f"Loading data from {input_csv}...")
    df = pd.read_csv(input_csv)

    # Load the trained OCV-LSTM model
    print(f"Loading OCV-LSTM model from {ocv_model_path}...")
    ocv_model = load_model(ocv_model_path)

    # Initialize the DEKF
    dekf = DualEKF(
        dt=initial_params.get('dt', 1.0),
        C_nom=initial_params.get('C_nom', 2.0),
        R0_nom=initial_params.get('R0_nom', 0.01),
        R_D_nom=initial_params.get('R_D_nom', 0.01),
        C_D_nom=initial_params.get('C_D_nom', 1000),
        eta_nom=initial_params.get('eta_nom', 0.99),
        ocv_lstm_model=ocv_model
    )

    # Set initial SoC if provided
    if 'initial_soc' in initial_params:
        dekf.x_k[0] = initial_params['initial_soc']

    results = []
    print("Running DEKF inference...")
    for _, row in df.iterrows():
        v_meas = row['Voltage_measured']
        i_k = row['Current_measured']
        temp_k = row['Temperature_measured']

        state, params = dekf.step(v_meas, i_k, temp_k)
        
        results.append({
            'Time': row['Time'],
            'Voltage_measured': v_meas,
            'Current_measured': i_k,
            'Temperature_measured': temp_k,
            'SoC_estimated': state[0],
            'Q_max_estimated': params[0],
            'R_0_estimated': params[1]
        })

    results_df = pd.DataFrame(results)
    
    # Save results
    output_filename = os.path.join('outputs', os.path.basename(input_csv).replace('.csv', '_dekf_inference.csv'))
    results_df.to_csv(output_filename, index=False)
    print(f"Inference results saved to {output_filename}")

    # Print final estimated parameters
    final_q_max = results_df['Q_max_estimated'].iloc[-1]
    final_r_0 = results_df['R_0_estimated'].iloc[-1]
    print("\n--- Inference Summary ---")
    print(f"Final Estimated Capacity (Q_max): {final_q_max:.4f} Ah")
    print(f"Final Estimated Resistance (R_0): {final_r_0:.4f} Ohms")
    print("-------------------------\\n")


def main():
    parser = argparse.ArgumentParser(description="Run DEKF-LSTM inference on a battery CSV file.")
    parser.add_argument('--input', type=str, required=True, help='Path to the input CSV file.')
    parser.add_argument('--ocv-model', type=str, default='outputs/final_ocv_lstm.keras', help='Path to the trained OCV-LSTM model.')
    parser.add_argument('--initial-soc', type=float, default=0.9, help='Initial State of Charge.')
    parser.add_argument('--C-nom', type=float, default=2.0, help='Initial nominal capacity in Ah.')
    parser.add_argument('--R0-nom', type=float, default=0.01, help='Initial internal resistance in Ohms.')
    args = parser.parse_args()

    initial_params = {
        'initial_soc': args.initial_soc,
        'C_nom': args.C_nom,
        'R0_nom': args.R0_nom
    }

    infer_single_file(args.input, args.ocv_model, initial_params)

if __name__ == "__main__":
    if not os.path.exists('outputs'):
        os.makedirs('outputs')
    main()
