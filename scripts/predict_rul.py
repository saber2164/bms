
import pandas as pd
import numpy as np
import argparse
from rul_predictor import RULPredictor

def get_soh_history(battery_id: str, cycle_number: int, soh_file: str, metadata_file: str) -> np.ndarray:
    """
    Retrieves the SoH history for a specific battery up to a given cycle.

    Args:
        battery_id (str): The ID of the battery.
        cycle_number (int): The cycle number to predict from.
        soh_file (str): Path to the soh_features.csv file.
        metadata_file (str): Path to the metadata.csv file.

    Returns:
        A numpy array of the SoH sequence, or None if not found.
    """
    try:
        soh_df = pd.read_csv(soh_file)
        meta_df = pd.read_csv(metadata_file)
        # We also need IC features now
        ic_df = pd.read_csv('ic_features.csv')
    except FileNotFoundError as e:
        print(f"Error: {e}. Make sure data files exist.")
        return None

    # Fix filenames
    soh_df['filename'] = soh_df['filename'].astype(str).apply(lambda x: x if x.endswith('.csv') else x + '.csv')
    meta_df['filename'] = meta_df['filename'].astype(str).apply(lambda x: x if x.endswith('.csv') else x + '.csv')
    ic_df['filename'] = ic_df['filename'].astype(str).apply(lambda x: x if x.endswith('.csv') else x + '.csv')

    # Merge
    meta_df = meta_df[meta_df['type'] == 'discharge'][['filename', 'battery_id', 'test_id', 'Re', 'Rct', 'start_time']]
    merged_df = pd.merge(meta_df, soh_df, on='filename', how='inner')
    merged_df = pd.merge(merged_df, ic_df, on='filename', how='left')
    
    battery_df = merged_df[merged_df['battery_id'] == battery_id].copy()
    
    if battery_df.empty:
        print(f"Error: No data found for battery_id '{battery_id}'.")
        return None

    # Fill missing IC/Impedance
    battery_df['ic_peak_height'] = battery_df['ic_peak_height'].fillna(battery_df['ic_peak_height'].mean())
    battery_df['ic_peak_voltage'] = battery_df['ic_peak_voltage'].fillna(battery_df['ic_peak_voltage'].mean())
    battery_df['Re'] = pd.to_numeric(battery_df['Re'], errors='coerce').ffill().fillna(0)
    battery_df['Rct'] = pd.to_numeric(battery_df['Rct'], errors='coerce').ffill().fillna(0)

    # Calculate SoH (Robust Dynamic Nominal Capacity)
    # Use max capacity of first 50 cycles (or all if < 50)
    # This aligns with the updated training logic
    warmup_period = 50
    
    # Ensure we look at the *start* of the battery life, not just the window we loaded
    # We loaded everything up to cycle_number, sorted by test_id.
    # So we can just take the first 50 rows of battery_df.
    
    if len(battery_df) > warmup_period:
        early_data = battery_df.iloc[:warmup_period]
    else:
        early_data = battery_df
        
    nominal_capacity = early_data['calculated_capacity'].max()
    
    if nominal_capacity <= 0:
        print(f"Error: Invalid nominal capacity for battery {battery_id}.")
        return None

    battery_df['soh'] = np.clip(battery_df['calculated_capacity'] / nominal_capacity, 0, 1.0)
    
    # Sort by test_id
    battery_df = battery_df.sort_values('test_id')
    
    # Get history
    history_df = battery_df[battery_df['test_id'] <= cycle_number]
    
    if len(history_df) == 0:
        print(f"Error: No history found for battery '{battery_id}' up to cycle {cycle_number}.")
        return None
        
    # Select features
    feature_cols = ['soh', 'average_temp', 'Re', 'Rct', 'ic_peak_height', 'ic_peak_voltage']
    features = history_df[feature_cols].values
    
    return features

def main():
    """
    Main function to load the model and predict RUL for a given battery.
    """
    parser = argparse.ArgumentParser(description="Predict RUL for a given battery and cycle.")
    parser.add_argument("--battery_id", type=str, required=True, help="The ID of the battery (e.g., B0045).")
    parser.add_argument("--cycle_number", type=int, required=True, help="The current cycle number to predict from.")
    args = parser.parse_args()

    # --- 1. Configuration ---
    SOH_DATA_FILE = 'soh_features.csv'
    METADATA_FILE = 'cleaned_dataset/metadata.csv'
    SEQUENCE_LENGTH = 15
    MODEL_WEIGHTS_PATH = 'outputs/rul_model.weights.h5' # Correct path
    SCALER_PATH = 'outputs/feature_scaler.pkl'
    END_OF_LIFE_SOH = 0.8
    MAX_RUL = 2000

    # --- 2. Load SoH History ---
    print(f"Loading SoH history for battery '{args.battery_id}' up to cycle {args.cycle_number}...")
    features_history = get_soh_history(args.battery_id, args.cycle_number, SOH_DATA_FILE, METADATA_FILE)

    if features_history is None:
        return

    if len(features_history) < SEQUENCE_LENGTH:
        print(f"Error: Not enough historical data. Need at least {SEQUENCE_LENGTH} cycles, but got {len(features_history)}.")
        return

    # --- 3. Prepare Input ---
    import joblib
    try:
        scaler = joblib.load(SCALER_PATH)
    except FileNotFoundError:
        print(f"Error: Scaler not found at {SCALER_PATH}")
        return

    # Use last SEQUENCE_LENGTH cycles
    recent_features = features_history[-SEQUENCE_LENGTH:]
    
    # Scale
    scaled_features = scaler.transform(recent_features)
    
    # Reshape (1, SEQUENCE_LENGTH, n_features)
    input_sequence = scaled_features.reshape(1, SEQUENCE_LENGTH, 6)

    # --- 4. Load Model and Predict ---
    print("Initializing RUL predictor and loading weights...")
    
    rul_predictor = RULPredictor(sequence_length=SEQUENCE_LENGTH, n_features=6, end_of_life_soh=END_OF_LIFE_SOH)
    try:
        rul_predictor.model.load_weights(MODEL_WEIGHTS_PATH)
    except Exception as e:
        print(f"Error loading model weights from '{MODEL_WEIGHTS_PATH}': {e}")
        return

    print("Predicting RUL...")
    pred_scaled = rul_predictor.model.predict(input_sequence, verbose=0)[0][0]
    
    # Unscale RUL
    try:
        max_rul = joblib.load('outputs/max_rul.pkl')
    except FileNotFoundError:
        print("Error: max_rul.pkl not found. Using default 2000.")
        max_rul = 2000
        
    predicted_rul = pred_scaled * max_rul

    print(f"\n--- RUL Prediction ---")
    print(f"Battery ID:      {args.battery_id}")
    print(f"Current Cycle:   {args.cycle_number}")
    print(f"Predicted RUL:   {predicted_rul:.0f} cycles")
    print(f"End-of-Life Cycle: {args.cycle_number + predicted_rul:.0f}")
    print(f"----------------------")

if __name__ == '__main__':
    main()
