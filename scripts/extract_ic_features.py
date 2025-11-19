import os
import numpy as np
import pandas as pd
from scipy.signal import savgol_filter
import matplotlib.pyplot as plt

# Configuration
DATA_DIR = 'cleaned_dataset/data'
METADATA_FILE = 'cleaned_dataset/metadata.csv'
OUTPUT_FILE = 'ic_features.csv'
VOLTAGE_WINDOW = 0.05  # V (for smoothing/differentiation window if needed, but we use savgol)

def calculate_dq_dv(voltage, capacity, window_length=21, polyorder=2):
    """
    Calculates dQ/dV using Savitzky-Golay smoothing.
    """
    # Sort by voltage (discharge: voltage decreases, capacity increases)
    # For dQ/dV, we usually look at it vs Voltage.
    # In discharge, V goes down.
    
    # Ensure data is sorted by voltage (ascending) for numerical derivative
    # But discharge data is usually time-sorted (V decreasing).
    # Let's sort by Voltage ascending to make dV positive.
    
    sort_idx = np.argsort(voltage)
    v_sorted = voltage[sort_idx]
    q_sorted = capacity[sort_idx]
    
    # Smooth V and Q
    try:
        if len(v_sorted) > window_length:
            v_smooth = savgol_filter(v_sorted, window_length, polyorder)
            q_smooth = savgol_filter(q_sorted, window_length, polyorder)
        else:
            v_smooth = v_sorted
            q_smooth = q_sorted
            
        # Calculate derivative
        dq = np.diff(q_smooth)
        dv = np.diff(v_smooth)
        
        # Avoid division by zero
        dv[dv == 0] = 1e-6
        
        dq_dv = dq / dv
        
        # Return corresponding voltage points (midpoints)
        v_mid = (v_smooth[1:] + v_smooth[:-1]) / 2
        
        return v_mid, dq_dv
        
    except Exception as e:
        # print(f"Error in smoothing: {e}")
        return np.array([]), np.array([])

def extract_features():
    print("Loading metadata...")
    try:
        meta_df = pd.read_csv(METADATA_FILE)
    except FileNotFoundError:
        print("Metadata file not found.")
        return

    # Filter for discharge cycles
    discharge_df = meta_df[meta_df['type'] == 'discharge']
    
    features = []
    
    print(f"Processing {len(discharge_df)} discharge cycles...")
    
    for i, (_, row) in enumerate(discharge_df.iterrows()):
        filename = row['filename']
        # metadata filename might be an integer or string. Ensure string.
        filename = str(filename)
        
        # Check if filename already has .csv
        if filename.endswith('.csv'):
            filepath = os.path.join(DATA_DIR, filename)
        else:
            filepath = os.path.join(DATA_DIR, filename + '.csv')
        
        if i < 5:
            print(f"Checking file: {filepath}")
        
        if not os.path.exists(filepath):
            if i < 5:
                print(f"File not found: {filepath}")
            continue
            
        try:
            df = pd.read_csv(filepath)
            
            # Check required columns
            if 'Voltage_measured' not in df.columns or 'Capacity' not in df.columns:
                # Some files might not have 'Capacity' column directly, usually it's calculated or 'Current_measured' integrated.
                # In this dataset, 'Capacity' might not be in the raw time-series file?
                # Let's check a raw file structure.
                # Wait, previous view of raw file showed: Time, Voltage_measured, Current_measured, Temperature_measured...
                # It did NOT show Capacity.
                # We need to calculate Capacity by integrating Current over Time.
                pass
            
            # Calculate Capacity if missing
            # Q = Integral(I dt)
            # Discharge current is usually negative? Or positive?
            # Let's assume standard: Discharge I < 0. Capacity = -Integral(I dt)
            # Or if I > 0 for discharge, then Q = Integral(I dt).
            # Let's check the raw file content again mentally... 
            # In '00001_dukf_inference.csv' (which mirrors raw), Current was ~ -1.0A.
            # So discharge is negative.
            
            time = df['Time'].values
            current = df['Current_measured'].values
            voltage = df['Voltage_measured'].values
            
            # Calculate Capacity (Ah)
            # dt in hours
            dt = np.diff(time, prepend=time[0]) / 3600.0
            # Accumulate charge (Ah)
            # Since I is negative during discharge, we take -I * dt
            # But we want "Capacity Discharged" so it starts at 0 and goes up.
            capacity_curve = np.cumsum(-current * dt)
            
            # Calculate dQ/dV
            v_mid, dq_dv = calculate_dq_dv(voltage, capacity_curve)
            
            if len(dq_dv) > 0:
                # Find Peak
                # We are looking for the main peak in the dQ/dV curve.
                # Usually around 3.6V - 3.8V for Li-ion.
                # We take the max value of dQ/dV.
                
                # Filter out noise at edges (very low/high voltage)
                mask = (v_mid > 3.0) & (v_mid < 4.1)
                if np.any(mask):
                    dq_dv_filtered = dq_dv[mask]
                    v_filtered = v_mid[mask]
                    
                    peak_idx = np.argmax(dq_dv_filtered)
                    ic_peak_height = dq_dv_filtered[peak_idx]
                    ic_peak_voltage = v_filtered[peak_idx]
                else:
                    ic_peak_height = 0
                    ic_peak_voltage = 0
            else:
                ic_peak_height = 0
                ic_peak_voltage = 0
                
            features.append({
                'filename': filename,
                'ic_peak_height': ic_peak_height,
                'ic_peak_voltage': ic_peak_voltage
            })
            
        except Exception as e:
            print(f"Error processing {filename}: {e}")
            
    # Save to CSV
    features_df = pd.DataFrame(features)
    features_df.to_csv(OUTPUT_FILE, index=False)
    print(f"Saved IC features to {OUTPUT_FILE}")
    print(features_df.head())

if __name__ == "__main__":
    extract_features()
