import pandas as pd
import numpy as np
import os

SOH_FILE = 'soh_features.csv'
METADATA_FILE = 'cleaned_dataset/metadata.csv'

def analyze_truncation():
    if not os.path.exists(SOH_FILE) or not os.path.exists(METADATA_FILE):
        print("Data files not found.")
        return

    soh_df = pd.read_csv(SOH_FILE)
    meta_df = pd.read_csv(METADATA_FILE)
    
    # Fix filenames
    soh_df['filename'] = soh_df['filename'].astype(str).apply(lambda x: x if x.endswith('.csv') else x + '.csv')
    meta_df['filename'] = meta_df['filename'].astype(str).apply(lambda x: x if x.endswith('.csv') else x + '.csv')
    
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
    
    print(f"{'Battery':<10} {'Total Cycles':<15} {'Max Cap Cycle':<15} {'% Lost':<10} {'Max Cap':<10}")
    print("-" * 65)
    
    total_cycles_all = 0
    lost_cycles_all = 0
    
    for bat_id in df['battery_id'].unique():
        bat_df = df[df['battery_id'] == bat_id].copy()
        bat_df = bat_df.reset_index(drop=True)
        
        total_cycles = len(bat_df)
        max_cap_idx = bat_df['calculated_capacity'].idxmax()
        max_cap = bat_df['calculated_capacity'].max()
        
        # Cycles lost = max_cap_idx (since we start FROM there)
        lost = max_cap_idx
        pct_lost = (lost / total_cycles) * 100
        
        print(f"{bat_id:<10} {total_cycles:<15} {max_cap_idx:<15} {pct_lost:<10.1f} {max_cap:<10.4f}")
        
        total_cycles_all += total_cycles
        lost_cycles_all += lost
        
    print("-" * 65)
    print(f"Total Cycles: {total_cycles_all}")
    print(f"Total Lost: {lost_cycles_all}")
    print(f"Overall Data Loss: {(lost_cycles_all / total_cycles_all) * 100:.1f}%")

if __name__ == "__main__":
    analyze_truncation()
