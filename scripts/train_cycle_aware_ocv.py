#!/usr/bin/env python3
"""
Cycle-Aware OCV Model Training Script.

This version trains an OCV model with 3 inputs:
1. SoC (State of Charge)
2. Temperature
3. CycleType (-1 for charge, 1 for discharge)

This should improve convergence by helping the model understand charge/discharge dynamics.
"""

import os
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

# --- 1. OCV Model Definition (Dense/MLP with 3 inputs) ---
def create_ocv_model(input_shape):
    """
    Creates a Dense Neural Network for OCV prediction.
    Input: [SoC,Temperature, CycleType]
    Output: [OCV]
    """
    model = Sequential([
        Input(shape=input_shape),
        Dense(64, activation='relu'),
        Dense(32, activation='relu'),
        Dense(16, activation='relu'),
        Dense(1)  # Output: OCV
    ])
    model.compile(optimizer=Adam(learning_rate=0.001), loss='mean_squared_error', metrics=['mae'])
    return model

# --- 2. Data Preparation from Processed Dataset ---
def prepare_data(train_path, nominal_capacity=2.0, current_threshold=0.05, sample_fraction=1.0):
    """
    Loads pre-processed data and prepares features.
    
    The processed dataset already has:
    - Voltage_measured
    - Current_measured
    - Temperature_measured  
    - cycle_type (-1 for charge, 1 for discharge)
    - Time
    
    We need to:
    1. Calculate SoC via Coulomb counting
    2. Extract OCV points (low current)
    3. Return [SoC, Temp, CycleType] -> Voltage pairs
    """
    print(f"Loading processed training data from {train_path}...")
    df = pd.read_csv(train_path)
    
    # Sample if needed (for faster testing)
    if sample_fraction < 1.0:
        df = df.sample(frac=sample_fraction, random_state=42)
    
    print(f"Loaded {len(df):,} total data points")
    
    # Calculate SoC via Coulomb Counting
    # Sort by time
    df = df.sort_values('Time').reset_index(drop=True)
    
    # Calculate dt
    dt = df['Time'].diff().fillna(0)
    
    # Accumulate charge (Ah)
    # Note: NASA dataset uses positive current for discharge
    charge_change_ah = (df['Current_measured'] * dt).cumsum() / 3600.0
    
    # Start assumption: First point is at SoC = 1.0 (fully charged)
    # This is a simplification - in reality we'd need cycle-level initialization
    df['SoC'] = 1.0 - (charge_change_ah / nominal_capacity)
    
    # Clip SoC to [0, 1]
    df['SoC'] = df['SoC'].clip(0, 1)
    
    # Extract OCV points (low current, approximates rest condition)
    print(f"Filteringfor low-current (OCV) conditions (|I| < {current_threshold} A)...")
    rest_mask = df['Current_measured'].abs() < current_threshold
    ocv_data = df[rest_mask].copy()
    
    print(f"Found {len(ocv_data):,} OCV points ({100*len(ocv_data)/len(df):.1f}% of total)")
    
    if len(ocv_data) == 0:
        print("WARNING: No OCV points found!")
        return np.array([]), np.array([])
    
    # Prepare features and targets
    # Features: [SoC, Temperature, CycleType]
    X = ocv_data[['SoC', 'Temperature_measured', 'cycle_type']].values
    # Target: Voltage (approximates OCV at rest)
    y = ocv_data['Voltage_measured'].values
    
    print(f"\nFeature Statistics:")
    print(f"SoC: min={X[:,0].min():.3f}, max={X[:,0].max():.3f}, mean={X[:,0].mean():.3f}")
    print(f"Temp: min={X[:,1].min():.1f}, max={X[:,1].max():.1f}, mean={X[:,1].mean():.1f}")
    print(f"CycleType: charge={np.sum(X[:,2]==-1)}, discharge={np.sum(X[:,2]==1)}")
    print(f"Voltage: min={y.min():.3f}, max={y.max():.3f}, mean={y.mean():.3f}")
    
    return X, y

# --- 3. Main Training Flow ---
def main():
    parser = argparse.ArgumentParser(description="Train Cycle-Aware OCV Model.")
    parser.add_argument('--train-data', type=str, default='cleaned_dataset/processed/train_cycles.csv',
                        help='Path to processed training data CSV')
    parser.add_argument('--output-dir', type=str, default='outputs', help='Directory to save model and scaler')
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--batch-size', type=int, default=128)
    parser.add_argument('--sample-fraction', type=float, default=1.0,
                        help='Fraction of data to use (for faster testing)')
    parser.add_argument('--capacity', type=float, default=1.5,
                        help='Nominal battery capacity in Ah for SoC calculation')
    args = parser.parse_args()

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    # 1. Prepare Data
    print("\n" + "="*60)
    print("CYCLE-AWARE OCV MODEL TRAINING")
    print("="*60 + "\n")
    
    X, y = prepare_data(args.train_data, nominal_capacity=args.capacity, sample_fraction=args.sample_fraction)
    
    if len(X) == 0:
        print("ERROR: No training data found.")
        return

    print(f"\n{len(X):,} training samples prepared.")

    # 2. Scale Features
    # Important: We scale all 3 features (SoC, Temp, CycleType) to [0, 1]
    print("\nScaling features...")
    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Save Scaler - CRITICAL for inference!
    scaler_path = os.path.join(args.output_dir, 'ocv_scaler_v2.save')
    joblib.dump(scaler, scaler_path)
    print(f"✓ Scaler saved to {scaler_path}")
    print(f"  Feature ranges: {scaler.data_min_} to {scaler.data_max_}")

    # 3. Split Data
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
    print(f"\nData split: {len(X_train):,} train, {len(X_test):,} test")

    # 4. Create and Train Model
    # Input shape is (3,) -> [SoC, Temp, CycleType]
    print(f"\nCreating model with input shape: ({X_train.shape[1]},)")
    model = create_ocv_model(input_shape=(X_train.shape[1],))
    model.summary()
    
    print("\n" + "="*60)
    print("Starting training...")
    print("="*60 + "\n")
    
    history = model.fit(
        X_train, y_train,
        validation_data=(X_test, y_test),
        epochs=args.epochs,
        batch_size=args.batch_size,
        verbose=1
    )

    # 5. Evaluate
    print("\n" + "="*60)
    print("Evaluation Results:")
    print("="*60)
    
    train_loss, train_mae = model.evaluate(X_train, y_train, verbose=0)
    test_loss, test_mae = model.evaluate(X_test, y_test, verbose=0)
    
    print(f"Train - MSE: {train_loss:.6f}, MAE: {train_mae:.6f}")
    print(f"Test  - MSE: {test_loss:.6f}, MAE: {test_mae:.6f}")

    # 6. Save Model
    model_path = os.path.join(args.output_dir, 'final_ocv_model_v2.keras')
    model.save(model_path)
    print(f"\n✓ Model saved to {model_path}")
    
    print("\n" + "="*60)
    print("TRAINING COMPLETE!")
    print("="*60)
    print(f"\nNext steps:")
    print(f"1. Update UKF to use: {model_path}")
    print(f"2. Update scaler to use: {scaler_path}")
    print(f"3. Ensure UKF passes 3 features: [SoC, Temp, CycleType]")

if __name__ == "__main__":
    main()
