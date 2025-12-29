import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

SOH_FILE = 'soh_features.csv'
METADATA_FILE = 'cleaned_dataset/metadata.csv'
IC_FILE = 'ic_features.csv'

def check_correlations():
    print("Loading data...")
    soh_df = pd.read_csv(SOH_FILE)
    meta_df = pd.read_csv(METADATA_FILE)
    ic_df = pd.read_csv(IC_FILE)
    
    # Fix filenames
    soh_df['filename'] = soh_df['filename'].astype(str).apply(lambda x: x if x.endswith('.csv') else x + '.csv')
    meta_df['filename'] = meta_df['filename'].astype(str).apply(lambda x: x if x.endswith('.csv') else x + '.csv')
    ic_df['filename'] = ic_df['filename'].astype(str).apply(lambda x: x if x.endswith('.csv') else x + '.csv')
    
    # Merge
    df = pd.merge(meta_df, soh_df, on='filename', how='inner')
    df = df.merge(ic_df, on='filename', how='left')
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
    
    # Calculate RUL for correlation check
    df_list = []
    for bat_id in df['battery_id'].unique():
        bat_df = df[df['battery_id'] == bat_id].copy()
        
        # SoH
        nominal_capacity = bat_df['calculated_capacity'].iloc[0]
        bat_df['soh'] = bat_df['calculated_capacity'] / nominal_capacity
        
        # RUL
        eol_indices = bat_df[bat_df['soh'] < 0.8].index
        if len(eol_indices) > 0:
            eol_idx = eol_indices[0]
            bat_df['rul'] = eol_idx - bat_df.index
            bat_df = bat_df[bat_df['rul'] >= 0]
            df_list.append(bat_df)
            
    full_df = pd.concat(df_list)
    
    # Features to check
    features = ['soh', 'average_temp', 'Re', 'Rct', 'ic_peak_height', 'ic_peak_voltage', 'rul']
    
    # Clean up features
    full_df['Re'] = pd.to_numeric(full_df['Re'], errors='coerce')
    full_df['Rct'] = pd.to_numeric(full_df['Rct'], errors='coerce')
    
    corr = full_df[features].corr()
    print("\nCorrelation with RUL:")
    print(corr['rul'].sort_values(ascending=False))
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(corr, annot=True, cmap='coolwarm')
    plt.title('Feature Correlation Matrix')
    plt.savefig('outputs/feature_correlation.png')
    print("\nCorrelation matrix saved to outputs/feature_correlation.png")

if __name__ == "__main__":
    check_correlations()
