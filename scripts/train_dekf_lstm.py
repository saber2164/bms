#!/usr/bin/env python3
"""
Refactored Training Script for OCV Model (Dense Neural Network).

Improvements:
1.  **Coulomb Counting**: Uses current integration for ground-truth SoC labels instead of linear time proxies.
2.  **Dense Architecture**: Replaces LSTM with a Feed-Forward Network (MLP) for static OCV mapping.
3.  **Scaler Saving**: Saves the MinMaxScaler to ensure consistent preprocessing during inference.
4.  **Data Filtering**: Improved low-current detection for OCV extraction.
"""

import os
import glob
import joblib
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import argparse

# --- 1. OCV Model Definition (Dense/MLP) ---
def create_ocv_model(input_shape):
    """
    Creates a Dense Neural Network for OCV prediction.
    Input: [SoC, Temperature]
    Output: [OCV]
    """
    model = Sequential([
        Input(shape=input_shape),
        Dense(64, activation='relu'),
        Dense(32, activation='relu'),
        Dense(16, activation='relu'),
        Dense(1)  # Output: OCV
    ])
    model.compile(optimizer=Adam(learning_rate=0.001), loss='mean_squared_error')
    return model

# --- 2. Data Preparation with Coulomb Counting ---
def prepare_data(data_dir, nominal_capacity=2.0, current_threshold=0.05, max_files=None):
    """
    Prepares training data by:
    1. Calculating SoC via Coulomb Counting.
    2. Extracting (SoC, Temp) -> Voltage pairs during low-current (rest) periods.
    """
    all_files = sorted(glob.glob(os.path.join(data_dir, "*.csv")))
    
    features = []  # [SoC, Temp]
    targets = []   # [Voltage]

    print(f"Scanning files in {data_dir}...")
    
    valid_files_count = 0
    processed_count = 0

    for file_path in all_files:
        if max_files and valid_files_count >= max_files:
            break
            
        processed_count += 1
        try:
            # Read only header first to check columns (optimization)
            df_header = pd.read_csv(file_path, nrows=0)
            
            # Standardize column names
            col_map = {
                'Current_measured': 'Current',
                'Voltage_measured': 'Voltage',
                'Temperature_measured': 'Temperature',
                'Time': 'Time'
            }
            df_header = df_header.rename(columns=col_map)
            
            required_cols = ['Current', 'Voltage', 'Temperature', 'Time']
            if not all(col in df_header.columns for col in required_cols):
                # Silently skip or print only if verbose
                continue

            # If columns exist, read full file
            df = pd.read_csv(file_path)
            df = df.rename(columns=col_map)

            # --- Coulomb Counting for SoC ---
            # Sort by time just in case
            df = df.sort_values('Time').reset_index(drop=True)
            
            # Calculate dt (time difference in seconds)
            dt = df['Time'].diff().fillna(0)
            
            initial_soc = 1.0 # Assume fully charged start for training files
            
            # Accumulate charge (Ah)
            # Current (A) * Time (s) / 3600 = Ah
            charge_change_ah = (df['Current'] * dt).cumsum() / 3600.0
            
            # Calculate SoC
            # If Current > 0 is discharge:
            df['SoC'] = initial_soc - (charge_change_ah / nominal_capacity)
            
            # Clip SoC to [0, 1]
            df['SoC'] = df['SoC'].clip(0, 1)

            # --- Extract OCV Points ---
            # Filter for low current (approx OCV conditions)
            rest_mask = df['Current'].abs() < current_threshold
            
            rest_data = df[rest_mask]
            
            if not rest_data.empty:
                # Append to dataset
                # Features: SoC, Temperature
                # Target: Voltage (which approximates OCV at rest)
                file_features = rest_data[['SoC', 'Temperature']].values
                file_targets = rest_data['Voltage'].values
                
                features.append(file_features)
                targets.append(file_targets)
                valid_files_count += 1
                print(f"Processed {file_path} ({valid_files_count}/{max_files if max_files else 'all'})")
                
        except Exception as e:
            print(f"Error processing {file_path}: {e}")

    if not features:
        return np.array([]), np.array([])

    return np.vstack(features), np.concatenate(targets)

# --- 3. Main Training Flow ---
def main():
    parser = argparse.ArgumentParser(description="Train OCV Model (Dense).")
    parser.add_argument('--data-dir', type=str, default='cleaned_dataset/data', help='Directory containing battery CSVs')
    parser.add_argument('--output-dir', type=str, default='outputs', help='Directory to save model and scaler')
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--max-files', type=int, default=None, help='Maximum number of valid files to use for training')
    args = parser.parse_args()

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    # 1. Prepare Data
    print("Preparing data...")
    X, y = prepare_data(args.data_dir, max_files=args.max_files)
    
    if len(X) == 0:
        print("No training data found. Check data directory and format.")
        return

    print(f"Training on {len(X)} samples.")

    # 2. Scale Features
    # It's crucial to scale inputs (SoC [0-1], Temp [~20-40]) to similar ranges [0-1]
    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Save Scaler
    scaler_path = os.path.join(args.output_dir, 'ocv_scaler.save')
    joblib.dump(scaler, scaler_path)
    print(f"Scaler saved to {scaler_path}")

    # 3. Split Data
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

    # 4. Create and Train Model
    # Input shape is (2,) -> [SoC, Temp]
    model = create_ocv_model(input_shape=(X_train.shape[1],))
    
    print("Starting training...")
    history = model.fit(
        X_train, y_train,
        validation_data=(X_test, y_test),
        epochs=args.epochs,
        batch_size=args.batch_size,
        verbose=1
    )

    # 5. Save Model
    model_path = os.path.join(args.output_dir, 'final_ocv_model.keras')
    model.save(model_path)
    print(f"Model saved to {model_path}")

if __name__ == "__main__":
    main()
