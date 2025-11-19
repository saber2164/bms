import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error
import matplotlib.pyplot as plt
import joblib
import os

# Configuration
METADATA_FILE = 'cleaned_dataset/metadata.csv'
SOH_FEATURES_FILE = 'soh_features.csv'
EOL_SOH = 0.8

def load_data():
    meta_df = pd.read_csv(METADATA_FILE)
    soh_df = pd.read_csv(SOH_FEATURES_FILE)
    
    # Merge
    # soh_features has filename without .csv? Or with?
    # Let's check. The extraction script output showed "00001.csv".
    # metadata has "00001.csv".
    # soh_features probably has "00001" or "00001.csv".
    # Let's assume we need to match them.
    
    # Check first item of soh_df
    first_filename = str(soh_df['filename'].iloc[0])
    if not first_filename.endswith('.csv'):
        soh_df['filename'] = soh_df['filename'].astype(str) + '.csv'
    
    df = pd.merge(meta_df, soh_df, on='filename', how='inner')
    
    # Filter discharge
    df = df[df['type'] == 'discharge']
    
    # Sort by time
    # Sort by time
    def parse_time_str(s):
        # Remove brackets and split by whitespace
        s = s.strip('[]')
        parts = s.split()
        # Convert to float then int
        parts = [int(float(p)) for p in parts]
        return f"{parts[0]}-{parts[1]}-{parts[2]} {parts[3]}:{parts[4]}:{parts[5]}"

    df['start_time'] = pd.to_datetime(df['start_time'].apply(parse_time_str))
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
        
        # Calculate True RUL
        # Find cycle where SoH crosses 0.8
        # If never crosses, assume last cycle is EOL? Or extrapolate?
        # For evaluation, we only care about data BEFORE EOL.
        
        eol_cycle = bat_df[bat_df['soh'] < EOL_SOH]['cycle'].min()
        
        if np.isnan(eol_cycle):
            # Battery never reached EOL in dataset
            # We can't evaluate RUL accuracy properly if we don't know true EOL.
            # But we can still predict.
            # For training/eval purposes, let's skip these or treat last cycle as EOL?
            # Let's skip for strict evaluation.
            continue
            
        bat_df['rul'] = eol_cycle - bat_df['cycle']
        # Filter out data after EOL
        bat_df = bat_df[bat_df['cycle'] <= eol_cycle]
        
        processed_data.append(bat_df)
        
    return pd.concat(processed_data)

def predict_rul_polynomial(cycle_data, current_cycle):
    """
    Fit polynomial to data up to current_cycle.
    Solve for SoH = 0.8.
    """
    history = cycle_data[cycle_data['cycle'] <= current_cycle]
    
    if len(history) < 5:
        return np.nan # Not enough data
        
    x = history['cycle'].values
    y = history['soh'].values
    
    # Fit 2nd degree polynomial: soh = ax^2 + bx + c
    try:
        z = np.polyfit(x, y, 2)
        p = np.poly1d(z)
        
        # Solve p(cycle) = 0.8
        # ax^2 + bx + c - 0.8 = 0
        roots = (p - EOL_SOH).roots
        
        # Filter real roots > current_cycle
        real_roots = [r.real for r in roots if np.isreal(r) and r.real > current_cycle]
        
        if not real_roots:
            # Maybe linear fit?
            z1 = np.polyfit(x, y, 1)
            p1 = np.poly1d(z1)
            roots1 = (p1 - EOL_SOH).roots
            real_roots = [r.real for r in roots1 if np.isreal(r) and r.real > current_cycle]
            
        if real_roots:
            eol_cycle_pred = min(real_roots)
            return eol_cycle_pred - current_cycle
        else:
            return np.nan
            
    except:
        return np.nan

def evaluate_baseline():
    print("Loading data...")
    df = load_data()
    
    batteries = df['battery_id'].unique()
    # Split train/test (use same logic as RUL model roughly)
    # We just need a test set to evaluate.
    # Let's take last 20% as test.
    split_idx = int(0.8 * len(batteries))
    test_batteries = batteries[split_idx:]
    
    print(f"Evaluating on {len(test_batteries)} batteries...")
    
    y_true = []
    y_pred = []
    
    for bat_id in test_batteries:
        bat_df = df[df['battery_id'] == bat_id]
        
        # Simulate online prediction
        # We don't need to predict for EVERY cycle, maybe every 10th to save time
        for cycle in bat_df['cycle'].values[::10]:
            true_rul = bat_df[bat_df['cycle'] == cycle]['rul'].values[0]
            pred_rul = predict_rul_polynomial(bat_df, cycle)
            
            if not np.isnan(pred_rul) and pred_rul < 5000: # Filter crazy predictions
                y_true.append(true_rul)
                y_pred.append(pred_rul)
                
    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    
    print(f"Polynomial Baseline Results:")
    print(f"MAE: {mae:.2f} cycles")
    print(f"MSE: {mse:.2f}")
    
    # Save dummy model file just to indicate existence
    with open('outputs/polynomial_baseline.txt', 'w') as f:
        f.write(f"MAE: {mae}\nMSE: {mse}")

if __name__ == "__main__":
    evaluate_baseline()
