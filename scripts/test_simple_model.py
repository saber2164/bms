#!/usr/bin/env python3
"""Diagnostic for 2-input (SoC, Temp) OCV model"""
import numpy as np
import joblib
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt

print("Loading 2-input model (SoC, Temp only)...")
model = load_model('outputs/final_ocv_model.keras')
scaler = joblib.load('outputs/ocv_scaler.save')

socs = np.linspace(0.0, 1.0, 50)
temp = 25.0

ocvs = []
for soc in socs:
    inp = scaler.transform([[soc, temp]])  # Only 2 features
    ocv = model.predict(inp, verbose=0)[0][0]
    ocvs.append(ocv)

# Plot
plt.figure(figsize=(10, 6))
plt.plot(socs*100, ocvs, 'b-', label='OCV Curve', linewidth=2)
plt.xlabel('SoC (%)')
plt.ylabel('OCV (V)')
plt.title('OCV vs SoC at 25°C (2-Input Model)')
plt.grid(True, alpha=0.3)
plt.legend()
plt.savefig('outputs/simple_ocv_diagnostic.png', dpi=150)
print("Plot saved to outputs/simple_ocv_diagnostic.png")

print("\nOCV Curve:")
print(f"  At 100% SoC: {ocvs[-1]:.3f}V")
print(f"  At  50% SoC: {ocvs[25]:.3f}V")
print(f"  At   0% SoC: {ocvs[0]:.3f}V")

is_monotonic = all(ocvs[i] >= ocvs[i-1] for i in range(1, len(ocvs)))
print(f"  Monotonic: {is_monotonic}")

if is_monotonic:
    print("\n✓ MODEL IS MONOTONIC!")
else:
    print("\n✗ Model still non-monotonic")
