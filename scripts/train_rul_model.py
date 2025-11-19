
import pandas as pd
import numpy as np
import tensorflow as tf
from rul_predictor import RULPredictor

def load_and_preprocess_data(soh_file: str, metadata_file: str) -> list[np.ndarray]:
    """
    Loads battery capacity and metadata, merges them, and returns a list of
    SoH degradation curves for each battery.

    Args:
        soh_file (str): Path to the soh_features.csv file.
        metadata_file (str): Path to the metadata.csv file.

    Returns:
        A list of numpy arrays, where each array represents the SoH
        history for a single battery.
    """
    try:
        soh_df = pd.read_csv(soh_file)
        meta_df = pd.read_csv(metadata_file)
    except FileNotFoundError as e:
        print(f"Error: {e}. Make sure both soh_features.csv and metadata.csv exist.")
        return []

    # Keep only relevant columns and filter for discharge cycles
    meta_df = meta_df[meta_df['type'] == 'discharge'][['filename', 'battery_id', 'test_id']]
    
    # Merge the two dataframes to associate each cycle with a battery_id
    merged_df = pd.merge(soh_df, meta_df, on='filename')

    # Filter out rows where capacity is zero or non-positive
    merged_df = merged_df[merged_df['calculated_capacity'] > 0]

    # Group data by each battery
    battery_groups = merged_df.groupby('battery_id')
    
    soh_curves = []
    curve_lengths = []
    for battery_id, group in battery_groups:
        # Sort by test_id to ensure correct sequence
        group = group.sort_values('test_id')
        
        # Use the first measurement as the nominal capacity for SoH calculation
        if not group.empty:
            nominal_capacity = group['calculated_capacity'].iloc[0]
            
            if nominal_capacity > 0:
                soh = group['calculated_capacity'] / nominal_capacity
                # Clip SoH to be max 1.0
                soh = np.clip(soh, 0, 1.0)
                soh_values = soh.values
                soh_curves.append(soh_values)
                curve_lengths.append(len(soh_values))
            
    print(f"Processed {len(soh_curves)} unique battery degradation curves.")
    if curve_lengths:
        print(f"Max curve length found: {max(curve_lengths)}")
    return soh_curves

def prepare_training_data(soh_curves: list[np.ndarray], sequence_length: int, end_of_life_soh: float) -> tuple[np.ndarray, np.ndarray]:
    """
    Creates training sequences (X) and RUL targets (y) from a list of SoH curves.

    Args:
        soh_curves (list): A list of SoH degradation curves.
        sequence_length (int): The length of the input sequences.
        end_of_life_soh (float): The SoH threshold for end-of-life.

    Returns:
        A tuple of (X_all, y_all) containing aggregated training data.
    """
    X_all, y_all = [], []
    
    for curve in soh_curves:
        try:
            # Find the EOL for this specific curve
            eol_cycle = np.where(curve < end_of_life_soh)[0][0]
        except IndexError:
            # If the battery never reaches EOL in the data, we can't use it for RUL prediction
            continue

        # The RUL at each point is the EOL cycle minus the current cycle
        rul_values = eol_cycle - np.arange(len(curve))

        if len(curve) > sequence_length:
            X, y = [], []
            for i in range(len(curve) - sequence_length):
                X.append(curve[i:(i + sequence_length)])
                # The target is the RUL at the end of the sequence
                y.append(rul_values[i + sequence_length - 1])
            X_all.extend(X)
            y_all.extend(y)
            
    return np.array(X_all), np.array(y_all)

def main():
    """
    Main function to orchestrate the model training process.
    """
    # --- 1. Configuration ---
    SOH_DATA_FILE = 'soh_features.csv'
    METADATA_FILE = 'cleaned_dataset/metadata.csv'
    SEQUENCE_LENGTH = 15  # Number of past cycles to use for prediction
    END_OF_LIFE_SOH = 0.8
    EPOCHS = 100
    BATCH_SIZE = 64
    MODEL_SAVE_PATH = 'rul_model.weights.h5'

    # --- 2. Load and Prepare Data ---
    print("Loading and preprocessing SoH data...")
    soh_curves = load_and_preprocess_data(SOH_DATA_FILE, METADATA_FILE)
    
    if not soh_curves:
        print("No data available for training. Exiting.")
        return

    print(f"Generating training sequences with sequence length {SEQUENCE_LENGTH}...")
    X_train, y_train = prepare_training_data(soh_curves, SEQUENCE_LENGTH, END_OF_LIFE_SOH)

    if X_train.shape[0] == 0:
        print("Not enough data to form training sequences. Try a smaller sequence length.")
        return
        
    print(f"Created {X_train.shape[0]} training samples.")

    # --- 3. Instantiate and Train the Model ---
    print("\n--- Initializing and Training RUL Predictor ---")
    rul_predictor = RULPredictor(sequence_length=SEQUENCE_LENGTH, end_of_life_soh=0.8)
    
    rul_predictor.model.fit(
        X_train, 
        y_train, 
        epochs=EPOCHS, 
        batch_size=BATCH_SIZE, 
        validation_split=0.2,
        verbose=1,
        callbacks=[
            tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
        ]
    )

    # --- 4. Save the Trained Model ---
    print(f"\n--- Training Complete. Saving model weights to {MODEL_SAVE_PATH} ---")
    rul_predictor.model.save_weights(MODEL_SAVE_PATH)
    print("Model saved successfully.")

if __name__ == '__main__':
    main()
