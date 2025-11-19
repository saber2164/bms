
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
    except FileNotFoundError as e:
        print(f"Error: {e}. Make sure both soh_features.csv and metadata.csv exist.")
        return None

    meta_df = meta_df[meta_df['type'] == 'discharge'][['filename', 'battery_id', 'test_id']]
    merged_df = pd.merge(soh_df, meta_df, on='filename')
    
    battery_df = merged_df[merged_df['battery_id'] == battery_id].sort_values('test_id')
    
    if battery_df.empty:
        print(f"Error: No data found for battery_id '{battery_id}'.")
        return None

    # Use the first measurement as the nominal capacity
    nominal_capacity = battery_df['calculated_capacity'].iloc[0]
    if nominal_capacity <= 0:
        print(f"Error: Invalid nominal capacity for battery {battery_id}.")
        return None

    battery_df['soh'] = np.clip(battery_df['calculated_capacity'] / nominal_capacity, 0, 1.0)
    
    # Get the history up to the specified cycle
    history = battery_df[battery_df['test_id'] <= cycle_number]['soh'].values
    
    if len(history) == 0:
        print(f"Error: No history found for battery '{battery_id}' up to cycle {cycle_number}.")
        return None
        
    return history

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
    SEQUENCE_LENGTH = 15  # Must match the trained model
    MODEL_WEIGHTS_PATH = 'rul_model.weights.h5'
    END_OF_LIFE_SOH = 0.8

    # --- 2. Load SoH History ---
    print(f"Loading SoH history for battery '{args.battery_id}' up to cycle {args.cycle_number}...")
    soh_history = get_soh_history(args.battery_id, args.cycle_number, SOH_DATA_FILE, METADATA_FILE)

    if soh_history is None:
        return

    if len(soh_history) < SEQUENCE_LENGTH:
        print(f"Error: Not enough historical data. Need at least {SEQUENCE_LENGTH} cycles, but got {len(soh_history)}.")
        return

    # --- 3. Load Model and Predict ---
    print("Initializing RUL predictor and loading weights...")
    rul_predictor = RULPredictor(sequence_length=SEQUENCE_LENGTH, end_of_life_soh=END_OF_LIFE_SOH)
    
    try:
        rul_predictor.model.load_weights(MODEL_WEIGHTS_PATH)
    except Exception as e:
        print(f"Error loading model weights from '{MODEL_WEIGHTS_PATH}': {e}")
        return

    print("Predicting RUL...")
    # Use the most recent `SEQUENCE_LENGTH` cycles for prediction
    recent_soh = soh_history[-SEQUENCE_LENGTH:]
    
    # Reshape for the model
    input_sequence = np.array(recent_soh).reshape(1, SEQUENCE_LENGTH, 1)
    
    predicted_rul = rul_predictor.predict_rul(input_sequence)

    print(f"\n--- RUL Prediction ---")
    print(f"Battery ID:      {args.battery_id}")
    print(f"Current Cycle:   {args.cycle_number}")
    print(f"Predicted RUL:   {predicted_rul:.0f} cycles")
    print(f"End-of-Life Cycle: {args.cycle_number + predicted_rul:.0f}")
    print(f"----------------------")

if __name__ == '__main__':
    main()
