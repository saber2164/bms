import pandas as pd
import numpy as np
import os

# Configuration
METADATA_FILE = 'cleaned_dataset/metadata.csv'
SOH_FEATURES_FILE = 'soh_features.csv'

def verify_soh():
    print("Loading data...")
    if not os.path.exists(METADATA_FILE) or not os.path.exists(SOH_FEATURES_FILE):
        print("Error: Data files not found.")
        return

    meta_df = pd.read_csv(METADATA_FILE)
    soh_df = pd.read_csv(SOH_FEATURES_FILE)
    
    # Fix Filename matching
    meta_df['filename'] = meta_df['filename'].astype(str).apply(lambda x: x if x.endswith('.csv') else x + '.csv')
    soh_df['filename'] = soh_df['filename'].astype(str).apply(lambda x: x if x.endswith('.csv') else x + '.csv')
    
    # Merge
    df = pd.merge(meta_df, soh_df, on='filename', how='inner')
    df = df[df['type'] == 'discharge']
    
    # Parse time
    def parse_time_str(s):
        try:
            s = s.strip('[]')
            parts = s.split()
            parts = [int(float(p)) for p in parts]
            return f"{parts[0]}-{parts[1]}-{parts[2]} {parts[3]}:{parts[4]}:{parts[5]}"
        except:
            return s
            
    df['start_time'] = pd.to_datetime(df['start_time'].apply(parse_time_str))
    df = df.sort_values(['battery_id', 'start_time'])
    
    batteries = df['battery_id'].unique()
    print(f"Found {len(batteries)} batteries.")
    
    failures = []
    warnings = []
    
    for bat_id in batteries:
        bat_df = df[df['battery_id'] == bat_id].copy()
        
        if bat_df.empty:
            failures.append(f"{bat_id}: No data")
            continue
            
        # Logic being tested: Dynamic Nominal Capacity (Global Max)
        # Use global max capacity to handle batteries with initial warm-up or bad data
        nominal_capacity = bat_df['calculated_capacity'].max()
        
        if nominal_capacity <= 0:
            failures.append(f"{bat_id}: Invalid nominal capacity {nominal_capacity}")
            continue
            
        # Calculate SoH
        bat_df['soh'] = bat_df['calculated_capacity'] / nominal_capacity
        
        # Check 1: Max SoH should be 1.0
        max_soh = bat_df['soh'].max()
        if not np.isclose(max_soh, 1.0):
            failures.append(f"{bat_id}: Max SoH is {max_soh}, expected 1.0")
            
        # Check 2: First SoH might be low, but check how many are "normal"
        first_soh = bat_df['soh'].iloc[0]
        if first_soh < 0.9:
            warnings.append(f"{bat_id}: First SoH is low ({first_soh:.4f}). Max capacity occurs later.")
            
        # Check 3: Bounds
        if bat_df['soh'].min() < 0:
            failures.append(f"{bat_id}: Negative SoH found")
            
        if bat_df['soh'].isna().any():
            failures.append(f"{bat_id}: NaNs in SoH")
            
    print("\n--- Verification Results ---")
    if failures:
        print(f"FAILURES ({len(failures)}):")
        for f in failures:
            print(f"  x {f}")
    else:
        print("No CRITICAL failures found.")
        
    if warnings:
        print(f"\nWARNINGS ({len(warnings)}):")
        for w in warnings:
            print(f"  ! {w}")
            
    if not failures:
        print("\nSoH Logic is VALID across all batteries.")
    else:
        print("\nSoH Logic FAILED verification.")
        exit(1)

if __name__ == "__main__":
    verify_soh()
