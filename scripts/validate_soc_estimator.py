#!/usr/bin/env python3
"""
Unit tests and validation for cycle-aware SoC estimator.
Tests OCV model, UKF, and end-to-end behavior.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
import joblib
from tensorflow.keras.models import load_model
from scripts.ukf_soc import DualUKF
import matplotlib.pyplot as plt

print("="*60)
print("SoC ESTIMATOR VALIDATION TESTS")
print("="*60)

# Load model and scaler
print("\n1. Loading model and scaler...")
try:
    ocv_model = load_model('outputs/final_ocv_model_v2.keras')
    scaler = joblib.load('outputs/ocv_scaler_v2.save')
    print("✓ Model and scaler loaded successfully")
    print(f"  Model input shape: {ocv_model.input_shape}")
    print(f"  Scaler expects {scaler.n_features_in_} features")
except Exception as e:
    print(f"✗ Error loading model/scaler: {e}")
    exit(1)

# Test 1: OCV Model Predictions
print("\n2. Testing OCV Model Predictions...")
print("-" * 60)

# Test different SoC, Temp, and CycleType combinations
test_cases = [
    # [SoC, Temp, CycleType], Expected behavior
    ([0.9, 25, 1], "High SoC, discharge"),
    ([0.5, 25, 1], "Mid SoC, discharge"),
    ([0.1, 25, 1], "Low SoC, discharge"),
    ([0.9, 25, -1], "High SoC, charge"),
    ([0.5, 25, -1], "Mid SoC, charge"),
    ([0.1, 25, -1], "Low SoC, charge"),
]

ocv_predictions = []
for inputs, desc in test_cases:
    # Scale input
    inputs_scaled = scaler.transform([inputs])
    # Predict
    ocv = ocv_model.predict(inputs_scaled, verbose=0)[0][0]
    ocv_predictions.append(ocv)
    print(f"  {desc:25s} | SoC={inputs[0]:.1f}, T={inputs[1]:.0f}°C, Type={inputs[2]:2.0f} → OCV={ocv:.3f}V")

# Check if predictions are reasonable (3.0V to 4.3V typical for Li-ion)
if all(3.0 < ocv < 4.3 for ocv in ocv_predictions):
    print("✓ OCV predictions in reasonable range (3.0-4.3V)")
else:
    print("✗ WARNING: Some OCV predictions out of typical range!")

# Check if discharge vs charge makes a difference
discharge_mid = ocv_predictions[1]  # Mid SoC discharge
charge_mid = ocv_predictions[4]  # Mid SoC charge
diff = abs(discharge_mid - charge_mid)
print(f"  OCV difference (discharge vs charge at 50% SoC): {diff*1000:.1f}mV")
if diff > 0.001:  # >1mV difference
    print("✓ Model differentiates between charge and discharge")
else:
    print("✗ WARNING: Model may not be using cycle_type effectively")

# Test 2: UKF Initialization
print("\n3. Testing UKF Initialization...")
print("-" * 60)

try:
    ukf = DualUKF(
        dt=1.0,
        C_nom=1.5,
        R0_nom=0.05,
        ocv_model=ocv_model,
        scaler=scaler
    )
    # Set initial SoC manually
    ukf.x[0] = 0.8
    print("✓ DualUKF initialized successfully")
    print(f"  Initial state: SoC={ukf.x[0]:.3f}, U_d={ukf.x[1]:.3f}")
    print(f"  Initial params: Q_max={ukf.theta[0]:.3f}Ah, R_0={ukf.theta[1]:.5f}Ω")
except Exception as e:
    print(f"✗ Error initializing UKF: {e}")
    import traceback
    traceback.print_exc()
    exit(1)

# Test 3: Gradual Discharge Simulation
print("\n4. Testing Gradual Discharge Simulation...")
print("-" * 60)

# Simulate constant discharge at 1A
current = 1.0  # A (positive = discharge)
temperature = 25.0  # °C
voltage_measured = 4.0  # V (will vary)

# Track SoC over time
soc_history = []
time_history = []
voltage_history = []

print("  Simulating 100 timesteps of constant 1A discharge...")
for i in range(100):
    # Prediction step
    ukf.predict(current)
    
    # For realistic voltage, use a simple model:
    # V ≈ OCV(SoC) - I*R0
    # We'll use a rough approximation
    # OCV ≈ 3.2 + 0.8*SoC (linear approximation)
    approx_ocv = 3.2 + 0.8 * ukf.x[0]
    voltage_measured = approx_ocv - current * ukf.theta[1]
    
    # Update step
    ukf.update(voltage_measured, current, temperature)
    
    # Record
    soc_history.append(ukf.x[0])
    time_history.append(i)
    voltage_history.append(voltage_measured)

print(f"  Initial SoC: {soc_history[0]:.3f}")
print(f"  Final SoC: {soc_history[-1]:.3f}")
print(f"  SoC change: {soc_history[0] - soc_history[-1]:.3f}")

# Check if SoC decreases
if soc_history[-1] < soc_history[0]:
    print("✓ SoC decreased during discharge")
else:
    print("✗ ERROR: SoC did not decrease!")

# Check for jumps (sudden changes >5%)
max_jump = 0
for i in range(1, len(soc_history)):
    jump = abs(soc_history[i] - soc_history[i-1])
    if jump > max_jump:
        max_jump = jump

print(f"  Max jump between timesteps: {max_jump*100:.2f}%")
if max_jump < 0.05:  # Less than 5% jump
    print("✓ No large jumps detected")
else:
    print(f"✗ WARNING: Large jumps detected! Max: {max_jump*100:.1f}%")

# Test 4: Visualize Results
print("\n5. Creating visualization...")
print("-" * 60)

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))

# Plot SoC
ax1.plot(time_history, [s*100 for s in soc_history], 'b-', linewidth=2)
ax1.set_xlabel('Time Step')
ax1.set_ylabel('SoC (%)')
ax1.set_title('SoC Estimation Over Time (1A Discharge)')
ax1.grid(True, alpha=0.3)

# Plot voltage
ax2.plot(time_history, voltage_history, 'r-', linewidth=2)
ax2.set_xlabel('Time Step')
ax2.set_ylabel('Voltage (V)')
ax2.set_title('Measured Voltage')
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plot_path = 'outputs/soc_validation_test.png'
plt.savefig(plot_path, dpi=150)
print(f"✓ Plot saved to {plot_path}")

# Test 5: Load Real Data and Test
print("\n6. Testing with Real Discharge Cycle...")
print("-" * 60)

try:
    # Load a real discharge cycle
    test_file = 'cleaned_dataset/data/00001.csv'
    df = pd.read_csv(test_file)
    print(f"  Loaded {test_file}")
    print(f"  {len(df)} data points")
    
    # Filter to first 50 points for quick test
    df = df.head(50)
    
    # Reset UKF
    ukf = DualUKF(
        dt=1.0,
        C_nom=1.5,
        R0_nom=0.05,
        ocv_model=ocv_model,
        scaler=scaler
    )
    ukf.x[0] = 0.9  # Set initial SoC
    
    real_soc = []
    for idx, row in df.iterrows():
        v = row['Voltage_measured']
        i = row['Current_measured']
        t = row['Temperature_measured']
        
        ukf.predict(i)
        ukf.update(v, i, t)
        real_soc.append(ukf.x[0])
    
    print(f"  Initial SoC: {real_soc[0]:.3f}")
    print(f"  Final SoC: {real_soc[-1]:.3f}")
    
    # Check for stability
    if len(real_soc) > 1:
        std_dev = np.std(np.diff(real_soc))
        print(f"  Std of SoC changes: {std_dev:.6f}")
        if std_dev < 0.01:
            print("✓ SoC changes are stable")
        else:
            print(f"✗ WARNING: High variance in SoC changes!")
    
except Exception as e:
    print(f"✗ Error testing with real data: {e}")

# Summary
print("\n" + "="*60)
print("TEST SUMMARY")
print("="*60)
print("\nIf you see warnings above, the model may need:")
print("1. Retraining with full dataset (currently trained on 10%)")
print("2. Better UKF parameter tuning (Q_x, Q_theta, R)")
print("3. Fixed SoC calculation (check coulomb counting)")
print("4. Better initial conditions")
print("\nCheck the plot at:", plot_path)
print("="*60)
