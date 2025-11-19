import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error
import joblib
import os
import random # Added for shuffling

# Configuration
METADATA_FILE = '/home/harshit/Documents/bms/cleaned_dataset/metadata.csv'
SOH_FEATURES_FILE = '/home/harshit/Documents/bms/soh_features.csv'
OUTPUT_DIR = 'outputs'
EOL_SOH = 0.8

# Ensure output directory exists
os.makedirs(OUTPUT_DIR, exist_ok=True)

def load_data():
    if not os.path.exists(METADATA_FILE) or not os.path.exists(SOH_FEATURES_FILE):
        raise FileNotFoundError("Ensure metadata.csv and soh_features.csv are in the correct paths.")

    meta_df = pd.read_csv(METADATA_FILE)
    soh_df = pd.read_csv(SOH_FEATURES_FILE)
    
    # Fix Filename matching
    # robustly handle .csv extension differences
    meta_df['filename'] = meta_df['filename'].astype(str).apply(lambda x: x if x.endswith('.csv') else x + '.csv')
    soh_df['filename'] = soh_df['filename'].astype(str).apply(lambda x: x if x.endswith('.csv') else x + '.csv')
    
    df = pd.merge(meta_df, soh_df, on='filename', how='inner')
    df = df[df['type'] == 'discharge']
    
    # Robust Time Parsing
    def parse_time_str(s):
        try:
            # Handle standard format "2008-10-23 12:00:00"
            return pd.to_datetime(s)
        except:
            try:
                # Handle the list format "[2008. 10. 23. 12. 00. 00]"
                s = str(s).strip('[]')
                parts = s.split()
                # Handle cases where parts are strings with dots or floats
                parts = [int(float(p.strip('.'))) for p in parts]
                return pd.Timestamp(year=parts[0], month=parts[1], day=parts[2], 
                                  hour=parts[3], minute=parts[4], second=parts[5])
            except Exception as e:
                return pd.NaT

    df['start_time'] = df['start_time'].apply(parse_time_str)
    df = df.dropna(subset=['start_time']) # Drop rows where time failed
    df = df.sort_values(['battery_id', 'start_time'])
    
    # Calculate SoH
    NOMINAL_CAPACITY = 2.0
    df['capacity'] = df['calculated_capacity']
    
    batteries = df['battery_id'].unique()
    processed_data = []
    
    for bat_id in batteries:
        bat_df = df[df['battery_id'] == bat_id].copy()
        bat_df['soh'] = bat_df['capacity'] / NOMINAL_CAPACITY
        bat_df['cycle'] = np.arange(1, len(bat_df) + 1)
        
        # Find True EOL
        below_eol = bat_df[bat_df['soh'] < EOL_SOH]
        
        if below_eol.empty:
            # Skip batteries that never failed (cannot validate RUL)
            continue
            
        eol_cycle = below_eol['cycle'].min()
        
        bat_df['rul'] = eol_cycle - bat_df['cycle']
        
        # STRICT EVALUATION: Only keep data UP TO EOL. 
        # Data after EOL is irrelevant for "Time to Failure" prediction
        bat_df = bat_df[bat_df['cycle'] <= eol_cycle]
        
        processed_data.append(bat_df)
        
    return pd.concat(processed_data)

def predict_rul_polynomial(history, current_cycle):
    """
    Robust polynomial fitting.
    """
    # Ensure we only look at past data
    # (Pass in pre-sliced history to speed this up, but filtering here is safer)
    data_slice = history[history['cycle'] <= current_cycle]
    
    # Need decent amount of data to fit a curve
    if len(data_slice) < 10: 
        return np.nan 
        
    x = data_slice['cycle'].values
    y = data_slice['soh'].values
    
    pred_rul = np.nan

    # Attempt 1: Quadratic Fit
    try:
        z = np.polyfit(x, y, 2) # ax^2 + bx + c
        p = np.poly1d(z)
        
        # Check curvature: if a > 0 (convex), the battery is "healing" which is wrong.
        # Force linear fit if curve is non-physical or finding roots is complex
        if z[0] > 0: 
            raise ValueError("Positive curvature")

        roots = (p - EOL_SOH).roots
        real_roots = [r.real for r in roots if np.isreal(r) and r.real > current_cycle]
        
        if real_roots:
            eol_cycle_pred = min(real_roots)
            pred_rul = eol_cycle_pred - current_cycle
    except:
        pass # Fall through to linear

    # Attempt 2: Linear Fit (Fallback)
    if np.isnan(pred_rul):
        try:
            z1 = np.polyfit(x, y, 1) # ax + b
            # Only if slope is negative (degrading)
            if z1[0] < 0:
                p1 = np.poly1d(z1)
                roots1 = (p1 - EOL_SOH).roots
                if roots1[0] > current_cycle:
                    pred_rul = roots1[0] - current_cycle
        except:
            pass
            
    return pred_rul

def evaluate_baseline():
    print("Loading and processing data...")
    df = load_data()
    
    batteries = df['battery_id'].unique()
    
    # Shuffle to ensure fair test set distribution
    # Fixed seed for reproducibility
    rng = np.random.default_rng(seed=42)
    rng.shuffle(batteries)
    
    split_idx = int(0.8 * len(batteries))
    test_batteries = batteries[split_idx:]
    
    print(f"Total Batteries: {len(batteries)}")
    print(f"Evaluating on {len(test_batteries)} test batteries...")
    
    y_true = []
    y_pred = []
    
    for bat_id in test_batteries:
        bat_df = df[df['battery_id'] == bat_id]
        
        # Simulate online prediction every 10 cycles
        # Starting from cycle 20 to give the polyfit enough data
        cycles_to_test = bat_df['cycle'].values[20::10]
        
        for cycle in cycles_to_test:
            # Get True RUL
            true_rul = bat_df[bat_df['cycle'] == cycle]['rul'].values[0]
            
            # Get Prediction
            # Pass only data up to 'cycle' to simulate real-time
            history_view = bat_df[bat_df['cycle'] <= cycle]
            pred_rul = predict_rul_polynomial(history_view, cycle)
            
            # Filter Logic:
            # 1. Must be not NaN
            # 2. Ignore predictions > 3000 cycles (usually mathematical artifacts)
            # 3. Ignore negatives (shouldn't happen with logic above but safe to check)
            if not np.isnan(pred_rul) and 0 <= pred_rul < 3000:
                y_true.append(true_rul)
                y_pred.append(pred_rul)
    
    if not y_true:
        print("No valid predictions generated. Check EOL threshold or data quality.")
        return

    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    
    print("-" * 30)
    print(f"Polynomial Baseline Results:")
    print(f"MAE:  {mae:.2f} cycles")
    print(f"RMSE: {rmse:.2f} cycles")
    print("-" * 30)
    
    with open(os.path.join(OUTPUT_DIR, 'polynomial_baseline.txt'), 'w') as f:
        f.write(f"MAE: {mae}\nMSE: {mse}\nRMSE: {rmse}")
        f.write(f"\nEvaluated on {len(y_true)} prediction points across {len(test_batteries)} batteries.")

if __name__ == "__main__":
    evaluate_baseline()