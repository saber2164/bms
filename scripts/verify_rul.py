import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt

# Configuration
METADATA_FILE = 'cleaned_dataset/metadata.csv'
SOH_FEATURES_FILE = 'soh_features.csv'
EOL_SOH = 0.8

def verify_rul():
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
    valid_batteries_count = 0
    
    ruls = []
    
    for bat_id in batteries:
        bat_df = df[df['battery_id'] == bat_id].copy()
        
        if bat_df.empty:
            continue
            
        # --- 1. Robust SoH Calculation ---
        if 'calculated_capacity' not in bat_df.columns:
             if 'capacity' in bat_df.columns:
                 bat_df['calculated_capacity'] = bat_df['capacity']
             else:
                 continue

        # Dynamic Nominal Capacity (Global Max)
        nominal_capacity = bat_df['calculated_capacity'].max()
        if nominal_capacity <= 0:
            failures.append(f"{bat_id}: Invalid nominal capacity")
            continue
            
        bat_df['soh'] = bat_df['calculated_capacity'] / nominal_capacity
        
        # Truncate Warm-up
        max_cap_idx = bat_df['calculated_capacity'].idxmax()
        bat_df = bat_df.loc[max_cap_idx:].copy()
        bat_df = bat_df.reset_index(drop=True)
        bat_df['cycle'] = np.arange(1, len(bat_df) + 1)
        
        # --- 2. RUL Calculation ---
        # Find first cycle where SoH < 0.8
        eol_indices = bat_df[bat_df['soh'] < EOL_SOH].index
        
        if len(eol_indices) == 0:
            warnings.append(f"{bat_id}: Never reached EOL (Min SoH: {bat_df['soh'].min():.4f})")
            continue
            
        eol_idx = eol_indices[0]
        eol_cycle = bat_df.loc[eol_idx, 'cycle']
        
        # Calculate RUL
        bat_df['true_rul'] = eol_cycle - bat_df['cycle']
        
        # Filter data after EOL
        bat_df = bat_df[bat_df['cycle'] <= eol_cycle]
        
        if bat_df.empty:
            failures.append(f"{bat_id}: Empty after EOL filtering")
            continue
            
        # --- 3. Assertions ---
        
        # Check 1: RUL should be positive (or 0 at EOL)
        if (bat_df['true_rul'] < 0).any():
            failures.append(f"{bat_id}: Negative RUL found")
            
        # Check 2: Max RUL should be realistic
        max_rul = bat_df['true_rul'].max()
        ruls.append(max_rul)
        
        if max_rul > 3000:
            warnings.append(f"{bat_id}: High Max RUL {max_rul} cycles")
        elif max_rul < 50:
            warnings.append(f"{bat_id}: Very Low Max RUL {max_rul} cycles (Bad battery?)")
            
        # Check 3: Monotonic Decrease
        # RUL should decrease by 1 every cycle.
        rul_diff = bat_df['true_rul'].diff().dropna()
        if not np.allclose(rul_diff, -1):
             failures.append(f"{bat_id}: RUL not monotonically decreasing by 1")
             
        valid_batteries_count += 1
        
    print("\n--- Verification Results ---")
    print(f"Valid Batteries for RUL: {valid_batteries_count}/{len(batteries)}")
    
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
            
    if ruls:
        print(f"\nRUL Statistics (Cycles):")
        print(f"  Mean: {np.mean(ruls):.2f}")
        print(f"  Median: {np.median(ruls):.2f}")
        print(f"  Min: {np.min(ruls)}")
        print(f"  Max: {np.max(ruls)}")
        
    if not failures:
        print("\nRUL Logic is VALID.")
    else:
        print("\nRUL Logic FAILED verification.")
        exit(1)

if __name__ == "__main__":
    verify_rul()
