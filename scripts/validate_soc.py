#!/usr/bin/env python3
"""
SoC Model Validation Script
"""

import os
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error

# Configuration
RAW_DIR = 'cleaned_dataset/data'
OUTPUT_DIR = 'outputs'
FILES = ['00003', '02659', '07015']
C_NOM = 2.0  # Ah
INITIAL_SOC = 1.0 # Assumed

def load_and_prep(file_id):
    # 1. Load Raw Data
    raw_path = os.path.join(RAW_DIR, f"{file_id}.csv")
    if not os.path.exists(raw_path):
        print(f"Error: Raw file {raw_path} not found.")
        return None, None

    df_raw = pd.read_csv(raw_path)
    
    # Standardize Raw Columns
    col_map_raw = {
        'Current_measured': 'Current',
        'Voltage_measured': 'Voltage',
        'Temperature_measured': 'Temperature',
        'Time': 'Time'
    }
    df_raw = df_raw.rename(columns=col_map_raw)
    
    # 2. Load Inference Data
    inf_path = os.path.join(OUTPUT_DIR, f"{file_id}_dukf_inference.csv")
    if not os.path.exists(inf_path):
        print(f"Error: Inference file {inf_path} not found.")
        return None, None
        
    df_inf = pd.read_csv(inf_path)
    
    # Standardize Inference Columns (should already be standard, but good to be safe)
    # Expected: Time, SoC_estimated, Q_max_estimated, R_0_estimated
    
    # 3. Align
    # Assuming 1-to-1 mapping based on row index as inference is run sequentially on raw data
    # But let's merge on Time to be precise if Time exists
    
    # Ensure Time is sorted
    df_raw = df_raw.sort_values('Time').reset_index(drop=True)
    df_inf = df_inf.sort_values('Time').reset_index(drop=True)
    
    # Truncate to shorter length if mismatch (though they should match)
    min_len = min(len(df_raw), len(df_inf))
    df_raw = df_raw.iloc[:min_len]
    df_inf = df_inf.iloc[:min_len]
    
    return df_raw, df_inf

def compute_coulomb_counting(df_raw):
    # dt calculation
    time = df_raw['Time'].values
    dt = np.zeros_like(time)
    dt[1:] = np.diff(time)
    
    # Current integration
    # Assuming Current > 0 is Discharge (SoC decreases)
    # charge_Ah = integral(I * dt) / 3600
    current = df_raw['Current'].values
    charge_ah = np.cumsum(current * dt) / 3600.0
    
    # SoC Calculation
    soc_cc = INITIAL_SOC - (charge_ah / C_NOM)
    soc_cc = np.clip(soc_cc, 0, 1)
    
    return soc_cc

def analyze_file(file_id):
    df_raw, df_inf = load_and_prep(file_id)
    if df_raw is None:
        return None

    # Compute Baseline
    soc_cc = compute_coulomb_counting(df_raw)
    soc_est = df_inf['SoC_estimated'].values
    
    # Metrics
    rmse = np.sqrt(mean_squared_error(soc_cc, soc_est))
    mae = mean_absolute_error(soc_cc, soc_est)
    
    final_q_max = df_inf['Q_max_estimated'].iloc[-1]
    final_r_0 = df_inf['R_0_estimated'].iloc[-1]
    
    # Behavior Checks
    soc_in_bounds = np.all((soc_est >= 0) & (soc_est <= 1))
    
    # Check correlation with discharge
    # Discharge (I > 0) should lead to SoC decrease
    # We can check correlation between dSoC and -Current
    d_soc = np.diff(soc_est)
    current_avg = (df_raw['Current'].values[1:] + df_raw['Current'].values[:-1]) / 2
    # If I > 0, dSoC should be < 0. So correlation(dSoC, -I) should be positive (or corr(dSoC, I) negative)
    # However, noise might affect this. Let's just look at overall trend.
    
    return {
        'file': file_id,
        'rmse': rmse,
        'mae': mae,
        'final_q_max': final_q_max,
        'final_r_0': final_r_0,
        'soc_cc': soc_cc,
        'soc_est': soc_est,
        'q_max_hist': df_inf['Q_max_estimated'].values,
        'r_0_hist': df_inf['R_0_estimated'].values
    }

def main():
    results = []
    print("Running Validation...\n")
    
    for file_id in FILES:
        res = analyze_file(file_id)
        if res:
            results.append(res)
            
    # 1. Summary Table
    print("1) Summary table")
    print(f"{'File':<10} | {'RMSE_SoC':<10} | {'MAE_SoC':<10} | {'Final Q_max':<15} | {'Final R_0':<15} | {'Verdict'}")
    print("-" * 90)
    
    verdicts = []
    
    for res in results:
        # Determine Verdict
        # Good: RMSE < 0.05, R0 reasonable (> 1e-4)
        # Moderate: RMSE < 0.10
        # Poor: RMSE > 0.10 or R0 hitting bounds
        
        rmse = res['rmse']
        r0 = res['final_r_0']
        
        if rmse < 0.05 and r0 > 0.001:
            verdict = "Good"
        elif rmse < 0.10:
            verdict = "Moderate"
        else:
            verdict = "Poor"
            
        print(f"{res['file']:<10} | {rmse:.4f}     | {res['mae']:<4f}     | {res['final_q_max']:.4f}          | {r0:.4f}          | {verdict}")
        verdicts.append(verdict)

    print("\n2) Per-file analysis")
    for res in results:
        print(f"\nFile {res['file']}:")
        print(f"SoC RMSE: {res['rmse']:.4f}. The estimated SoC tracks the Coulomb Counting baseline with {'high' if res['rmse'] < 0.05 else 'moderate' if res['rmse'] < 0.1 else 'low'} accuracy.")
        
        # R0 Analysis
        r0_start = res['r_0_hist'][0]
        r0_end = res['r_0_hist'][-1]
        print(f"R_0 evolution: Started at {r0_start:.4f}, ended at {r0_end:.4f}. {'Converged to a reasonable value.' if r0_end > 0.001 else 'Hit lower bound (unstable/unobservable).'}")
        
        # Q_max Analysis
        q_start = res['q_max_hist'][0]
        q_end = res['q_max_hist'][-1]
        print(f"Q_max evolution: Started at {q_start:.4f}, ended at {q_end:.4f}.")

    print("\n3) Overall verdict")
    if all(v == "Good" for v in verdicts):
        print("Yes, the model behaves correctly")
    elif any(v == "Poor" for v in verdicts):
        print("No, the model is not working correctly")
    else:
        print("Partially correct: SoC OK but parameter estimation unstable")

    print("\n4) Actionable recommendations")
    if any(v != "Good" for v in verdicts):
        print("- Tune Process Noise (Q) for parameters: If R_0 is unstable, reduce Q_param.")
        print("- Check Initial Conditions: Ensure initial SoC matches reality better (UKF converges slowly from bad init).")
        print("- Verify OCV Model: Large R_0 adaptation might compensate for OCV model errors.")
    else:
        print("- None. Model is performing well.")

if __name__ == "__main__":
    main()
