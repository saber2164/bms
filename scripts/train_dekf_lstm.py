#!/usr/bin/env python3
"""
Training script for the DEKF-LSTM model.

This script is responsible for:
1. Defining the OCV-LSTM model architecture.
2. Preparing the data for training the OCV-LSTM.
3. Implementing the transfer learning workflow (pre-training and fine-tuning).
4. Saving the final trained OCV-LSTM model.
"""

import os
import glob
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Input
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

import argparse

# --- 1. OCV-LSTM Model Definition ---
def create_ocv_lstm_model(input_shape):
    """
    Creates the LSTM model for OCV prediction.
    
    Args:
        input_shape (tuple): The shape of the input data (e.g., (timesteps, n_features)).
    
    Returns:
        A Keras Sequential model.
    """
    model = Sequential([
        Input(shape=input_shape),
        LSTM(64, activation='relu', return_sequences=True),
        LSTM(32, activation='relu'),
        Dense(16, activation='relu'),
        Dense(1) # Output layer for OCV
    ])
    model.compile(optimizer=Adam(learning_rate=0.001), loss='mean_squared_error')
    return model

# --- 2. Data Preparation ---
def prepare_data_for_ocv_lstm(data_dir, current_threshold=0.01, max_files=None):
    """
    Prepares data for training the OCV-LSTM model by extracting OCV measurements
    from low-current conditions.
    """
    all_files = sorted(glob.glob(os.path.join(data_dir, "*.csv")))
    if max_files:
        all_files = all_files[:max_files]
    
    features = []
    targets = []

    # Lists of possible column names
    current_colnames = ['Current_measured', 'Sense_current', 'Battery_current']
    voltage_colnames = ['Voltage_measured', 'Voltage_load']
    temp_colnames = ['Temperature_measured']

    for file_path in all_files:
        df = pd.read_csv(file_path)
        
        # Find the correct column names
        current_col = next((col for col in current_colnames if col in df.columns), None)
        voltage_col = next((col for col in voltage_colnames if col in df.columns), None)
        temp_col = next((col for col in temp_colnames if col in df.columns), None)
        
        if not all([current_col, voltage_col, temp_col]):
            print(f"Warning: Skipping file {file_path} because it is missing one or more required columns (current, voltage, or temperature).")
            continue
            
        # Find low-current periods
        low_current_df = df[df[current_col].abs() < current_threshold]
        
        if not low_current_df.empty:
            initial_soc = 0.9 # Assumption
            
            for _, row in low_current_df.iterrows():
                if df['Time'].max() > 0:
                    time_ratio = row['Time'] / df['Time'].max()
                else:
                    time_ratio = 0
                soc_proxy = initial_soc * (1 - time_ratio)

                features.append([soc_proxy, row[temp_col]])
                targets.append(row[voltage_col])

    return np.array(features), np.array(targets)

# --- 3. Transfer Learning Workflow ---
def pretrain_lstm(model, X_train, y_train, epochs=50, batch_size=32):
    """
    Pre-trains the OCV-LSTM model on a large dataset.
    """
    print("--- Starting Pre-training ---")
    model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, validation_split=0.2, verbose=1)
    print("--- Pre-training Finished ---")
    return model

def finetune_lstm(model, X_train, y_train, epochs=20, batch_size=32):
    """
    Fine-tunes the OCV-LSTM model on a smaller, specific dataset.
    """
    print("--- Starting Fine-tuning ---")
    # Freeze the initial layers for fine-tuning
    for layer in model.layers[:-2]: # Freeze all but the last two Dense layers
        layer.trainable = False
    
    model.compile(optimizer=Adam(learning_rate=0.0001), loss='mean_squared_error') # Lower learning rate
    model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, validation_split=0.2, verbose=1)
    print("--- Fine-tuning Finished ---")
    return model

# --- 4. Main Training Orchestration ---
def main():
    parser = argparse.ArgumentParser(description="Train the OCV-LSTM model.")
    parser.add_argument('--num-files', type=int, default=None, help='Number of files to use for training.')
    args = parser.parse_args()

    # Assume a public dataset is available for pre-training
    # For this example, we will simulate this by splitting our own data
    print(f"Preparing data from {args.num_files or 'all'} files...")
    X, y = prepare_data_for_ocv_lstm('cleaned_dataset/data', max_files=args.num_files)

    if X.shape[0] == 0:
        print("Error: No data was prepared for training. This might be because all files were skipped.")
        exit(1)

    # Scale the features
    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Reshape data for LSTM: (samples, timesteps, features)
    X_scaled = X_scaled.reshape((X_scaled.shape[0], 1, X_scaled.shape[1]))
    
    # Split data into a "public" set for pre-training and a "private" set for fine-tuning
    X_pretrain, X_finetune, y_pretrain, y_finetune = train_test_split(X_scaled, y, test_size=0.3, random_state=42)

    # Create the model
    ocv_model = create_ocv_lstm_model(input_shape=(1, X_scaled.shape[2]))
    
    # Pre-training
    ocv_model = pretrain_lstm(ocv_model, X_pretrain, y_pretrain)
    
    # Save the pre-trained model
    ocv_model.save('outputs/pre_trained_ocv_lstm.keras')
    print("Pre-trained model saved to outputs/pre_trained_ocv_lstm.keras")

    # Fine-tuning
    ocv_model = finetune_lstm(ocv_model, X_finetune, y_finetune)

    # Save the final, fine-tuned model
    ocv_model.save('outputs/final_ocv_lstm.keras')
    print("Final fine-tuned model saved to outputs/final_ocv_lstm.keras")

if __name__ == "__main__":
    # Create output directory if it doesn't exist
    if not os.path.exists('outputs'):
        os.makedirs('outputs')
    main()
