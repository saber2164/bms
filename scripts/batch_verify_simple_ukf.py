#!/usr/bin/env python3
"""
Batch verification script for SimplifiedDualUKF.
Runs the filter on 100 random files from the cleaned dataset.
Calculates metrics by comparing against Coulomb Counting (Reference SoC).
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import glob
import os
import sys
import random
from tqdm import tqdm

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from scripts.simple_ukf import SimplifiedDualUKF

def calculate_reference_soc(df, q_nom_ah):
    """
    Calculate Reference SoC using Coulomb Counting.
    Assumes starting at 100% SoC if voltage > 4.1V, else estimates from OCV.
    """
    # Time differences
    time = df['Time'].values
    dt = np.diff(time, prepend=time[0])
    dt[0] = 0 # First step has no dt
    
    # Current (negative for discharge in dataset? Let's check)
    # In 00001.csv, current is ~ -0.99 A during discharge.
    # Our UKF expects positive current for discharge.
    # So we flip the sign for the UKF, but for Coulomb counting:
    # SoC_next = SoC_prev + (I_charging * dt) / Q
    # If I is negative for discharge, then:
    # SoC_next = SoC_prev + (I_measured * dt) / (Q * 3600)
    
    current = df['Current_measured'].values
    
    # Estimate initial SoC based on first voltage
    v_start = df['Voltage_measured'].iloc[0]
    if v_start > 4.1:
        initial_soc = 1.0
    else:
        # Simple linear inversion: SoC = (V - 3.0) / 1.2
        initial_soc = np.clip((v_start - 3.0) / 1.2, 0, 1)
        
    soc_ref = np.zeros_like(time)
    soc_ref[0] = initial_soc
    
    for i in range(1, len(time)):
        # Coulomb counting
        # current[i] is negative for discharge
        delta_soc = (current[i] * dt[i]) / (q_nom_ah * 3600)
        soc_ref[i] = soc_ref[i-1] + delta_soc
        
    return soc_ref

def process_file(file_path, file_index):
    try:
        df = pd.read_csv(file_path)
        
        # Skip empty or too short files
        if len(df) < 100:
            return None
            
        # Parameters
        Q_nom = 2.0 # Assumed nominal capacity
        R0_nom = 0.05
        dt = 1.0 # Approximate, will use actual dt
        
        # Get Reference SoC
        ref_soc = calculate_reference_soc(df, Q_nom)
        
        # Initialize Filter
        ukf = SimplifiedDualUKF(dt=1.0, C_nom=Q_nom, R0_nom=R0_nom)
        ukf.x[0] = ref_soc[0] # Initialize with perfect knowledge for fairness in tracking test
        # Or initialize with slight error to test convergence
        # ukf.x[0] = ref_soc[0] * 0.9 
        
        est_soc = []
        est_q = []
        est_r0 = []
        
        # Run Filter
        times = df['Time'].values
        voltages = df['Voltage_measured'].values
        currents = df['Current_measured'].values
        temps = df['Temperature_measured'].values
        
        for i in range(len(df)):
            # Calculate dt
            if i > 0:
                dt = times[i] - times[i-1]
            else:
                dt = 1.0
            
            if dt <= 0: dt = 1.0 # Handle potential duplicate timestamps
            
            ukf.dt = dt
            
            # UKF expects positive current for discharge
            # Dataset has negative current for discharge
            i_meas_ukf = -currents[i] 
            
            soc, q, r0 = ukf.step(voltages[i], i_meas_ukf, temps[i])
            
            est_soc.append(soc)
            est_q.append(q)
            est_r0.append(r0)
            
        est_soc = np.array(est_soc)
        
        # Metrics
        mae = np.mean(np.abs(est_soc - ref_soc))
        rmse = np.sqrt(np.mean((est_soc - ref_soc)**2))
        
        return {
            'file': os.path.basename(file_path),
            'mae': mae,
            'rmse': rmse,
            'ref_soc': ref_soc,
            'est_soc': est_soc,
            'time': times,
            'voltage': voltages
        }
        
    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return None

def main():
    data_dir = 'cleaned_dataset/data'
    all_files = sorted(glob.glob(os.path.join(data_dir, '*.csv')))
    
    if not all_files:
        print("No files found!")
        return
        
    # Select 100 files (or fewer if not enough)
    num_files = min(100, len(all_files))
    # Randomly sample or take first 100? Let's take first 100 for reproducibility
    # selected_files = random.sample(all_files, num_files)
    selected_files = all_files[:num_files]
    
    print(f"Processing {num_files} files...")
    
    results = []
    
    for i, f in enumerate(tqdm(selected_files)):
        res = process_file(f, i)
        if res:
            results.append(res)
            
    # Aggregate Metrics
    maes = [r['mae'] for r in results]
    rmses = [r['rmse'] for r in results]
    
    avg_mae = np.mean(maes)
    avg_rmse = np.mean(rmses)
    
    print("\n" + "="*30)
    print(f"Batch Verification Results ({len(results)} files)")
    print("="*30)
    print(f"Average MAE:  {avg_mae:.4f}")
    print(f"Average RMSE: {avg_rmse:.4f}")
    print(f"Min MAE:      {min(maes):.4f}")
    print(f"Max MAE:      {max(maes):.4f}")
    print("="*30)
    
    # Plotting
    output_dir = 'outputs/batch_verification'
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. Error Distribution
    plt.figure(figsize=(10, 6))
    plt.hist(maes, bins=20, color='skyblue', edgecolor='black')
    plt.title('Distribution of Mean Absolute Error (MAE) across 100 files')
    plt.xlabel('MAE')
    plt.ylabel('Count')
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(output_dir, 'mae_distribution.png'))
    
    # 2. Best, Median, Worst Cases
    sorted_results = sorted(results, key=lambda x: x['mae'])
    best_case = sorted_results[0]
    median_case = sorted_results[len(sorted_results)//2]
    worst_case = sorted_results[-1]
    
    cases = [('Best', best_case), ('Median', median_case), ('Worst', worst_case)]
    
    plt.figure(figsize=(15, 12))
    for i, (label, res) in enumerate(cases):
        plt.subplot(3, 1, i+1)
        plt.plot(res['time']/60, res['ref_soc'], 'k--', label='Reference SoC (Coulomb Counting)', linewidth=2)
        plt.plot(res['time']/60, res['est_soc'], 'b-', label='UKF Estimate', linewidth=2)
        plt.title(f'{label} Case: {res["file"]} (MAE={res["mae"]:.4f})')
        plt.ylabel('SoC')
        if i == 2: plt.xlabel('Time (min)')
        plt.legend()
        plt.grid(True)
        
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'representative_cases.png'))
    
    print(f"\nPlots saved to {output_dir}")

if __name__ == "__main__":
    main()
