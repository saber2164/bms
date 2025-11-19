
import os
import glob
import pandas as pd
import numpy as np
from tqdm import tqdm

# --- Constants ---
DATA_DIR = 'cleaned_dataset/data/'
OUTPUT_FILE = 'soh_features.csv'
VDTTI_START_TIME = 30  # seconds
VDTTI_END_TIME = 100   # seconds
VDTTI_WINDOW = 20       # seconds

def extract_health_indicators(file_path):
    """
    Extracts health indicators (VDTTI, capacity, temperature) from a single discharge file.

    Args:
        file_path (str): The path to the CSV file.

    Returns:
        A dictionary with the extracted features, or None if the file is skipped.
    """
    try:
        df = pd.read_csv(file_path)

        # --- Gracefully find column names ---
        current_col = next((col for col in ['Current_measured', 'Current (A)'] if col in df.columns), None)
        voltage_col = next((col for col in ['Voltage_measured', 'Voltage (V)'] if col in df.columns), None)
        temp_col = next((col for col in ['Temperature_measured', 'Temperature (C)'] if col in df.columns), None)
        time_col = next((col for col in ['Time', 'Time (s)'] if col in df.columns), 'Time')

        if not all([current_col, voltage_col, temp_col]):
            print(f"Warning: Skipping {os.path.basename(file_path)} due to missing essential columns.")
            return None

        # --- Filter for discharge cycle ---
        # Assuming discharge is where current is negative
        discharge_df = df[df[current_col] < -0.1].copy()
        if discharge_df.empty:
            return None

        # Reset time to start from 0 for the discharge cycle
        discharge_df[time_col] = discharge_df[time_col] - discharge_df[time_col].min()

        # --- 1. Calculate Current Charge Capacity (SoH) ---
        discharge_df['Time_delta'] = discharge_df[time_col].diff().fillna(0)
        # Integrate current over time to get capacity in Ampere-seconds, then convert to Ampere-hours
        calculated_capacity = (discharge_df[current_col].abs() * discharge_df['Time_delta']).sum() / 3600.0

        # --- 2. Calculate Voltage Disparity in Truncated Time Interval (VDTTI) ---
        # VDTTI calculation is removed for now as it is not working correctly.
        vdtti = np.nan

        # --- 3. Calculate Average Temperature ---
        average_temp = discharge_df[temp_col].mean()

        # --- 4. Get Cycle Number from Filename ---
        filename = os.path.basename(file_path)
        cycle_number = int(os.path.splitext(filename)[0])

        return {
            'filename': filename,
            'calculated_capacity': calculated_capacity,
            'average_temp': average_temp
        }

    except Exception as e:
        print(f"Error processing file {file_path}: {e}")
        return None

def main():
    """
    Main function to iterate through all data files, extract features,
    and save them to a summary CSV.
    """
    all_files = sorted(glob.glob(os.path.join(DATA_DIR, "*.csv")))
    if not all_files:
        print(f"Error: No CSV files found in {DATA_DIR}")
        return

    all_features = []
    for file_path in tqdm(all_files, desc="Extracting Health Indicators"):
        features = extract_health_indicators(file_path)
        if features:
            all_features.append(features)

    if not all_features:
        print("No features were extracted. Exiting.")
        return

    # --- Save to CSV ---
    features_df = pd.DataFrame(all_features)
    features_df.to_csv(OUTPUT_FILE, index=False)
    print(f"Successfully extracted features and saved to {OUTPUT_FILE}")
    print(features_df.head())

if __name__ == "__main__":
    main()
