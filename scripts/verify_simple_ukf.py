#!/usr/bin/env python3
"""
Verification script for SimplifiedDualUKF.
Simulates a full discharge cycle and plots the estimated SoC vs True SoC.
"""

import numpy as np
import matplotlib.pyplot as plt
import sys
import os

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from scripts.simple_ukf import SimplifiedDualUKF

def simulate_discharge():
    print("Simulating discharge cycle...")
    
    # Simulation parameters
    dt = 1.0
    duration = 3600  # 1 hour
    time = np.arange(0, duration, dt)
    
    # Battery parameters (True)
    Q_true = 2.0  # Ah
    R0_true = 0.05 # Ohm
    
    # Initial State
    soc_true = 1.0
    
    # Filter Initialization (with some initial error)
    ukf = SimplifiedDualUKF(dt, C_nom=1.5, R0_nom=0.01) # Wrong initial params
    ukf.x[0] = 0.8 # Wrong initial SoC
    
    # Storage
    soc_true_hist = []
    soc_est_hist = []
    voltage_hist = []
    current_hist = []
    
    # Discharge current (1A constant)
    current = 1.0
    
    for t in time:
        # 1. Update True System
        # Coulomb counting for true SoC
        soc_true = soc_true - (current * dt / (Q_true * 3600))
        
        # True Voltage (Linear OCV + IR drop + Noise)
        # V = (1.2*SoC + 3.0) - I*R0
        ocv_true = 1.2 * soc_true + 3.0
        v_true = ocv_true - current * R0_true
        
        # Add measurement noise
        v_meas = v_true + np.random.normal(0, 0.005) # 5mV noise
        
        # 2. Update Filter
        soc_est, q_est, r_est = ukf.step(v_meas, current, temp_k=25.0)
        
        # Store
        soc_true_hist.append(soc_true)
        soc_est_hist.append(soc_est)
        voltage_hist.append(v_meas)
        current_hist.append(current)
        
    # Plotting
    plt.figure(figsize=(12, 8))
    
    plt.subplot(2, 1, 1)
    plt.plot(time/60, soc_true_hist, 'k--', label='True SoC', linewidth=2)
    plt.plot(time/60, soc_est_hist, 'b-', label='Estimated SoC', linewidth=2)
    plt.ylabel('SoC')
    plt.title('SoC Estimation Convergence (Linear OCV Model)')
    plt.legend()
    plt.grid(True)
    
    plt.subplot(2, 1, 2)
    plt.plot(time/60, voltage_hist, 'g-', label='Voltage', alpha=0.7)
    plt.ylabel('Voltage (V)')
    plt.xlabel('Time (min)')
    plt.grid(True)
    
    output_path = 'outputs/simple_ukf_verification.png'
    plt.savefig(output_path)
    print(f"Plot saved to {output_path}")
    
    # Calculate error metrics
    mae = np.mean(np.abs(np.array(soc_true_hist) - np.array(soc_est_hist)))
    final_error = abs(soc_true_hist[-1] - soc_est_hist[-1])
    
    print(f"Mean Absolute Error: {mae:.4f}")
    print(f"Final Estimation Error: {final_error:.4f}")
    
    if final_error < 0.05:
        print("✓ Convergence Successful!")
    else:
        print("✗ Convergence Failed")

if __name__ == "__main__":
    simulate_discharge()
