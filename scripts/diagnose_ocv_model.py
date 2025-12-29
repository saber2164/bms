#!/usr/bin/env python3
"""Quick diagnostic of OCV model issues"""
import numpy as np
import joblib
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt

print("Loading model...")
model = load_model('outputs/final_ocv_model_v2.keras')
scaler = joblib.load('outputs/ocv_scaler_v2.save')

# Test OCV predictions across SoC range
socs = np.linspace(0.0, 1.0, 50)
temp = 25.0

print("\n" + "="*60)
print("OCV vs SoC at 25°C")
print("="*60)

# Discharge
discharge_ocvs = []
for soc in socs:
    inp = scaler.transform([[soc, temp, 1]])  # discharge = 1
    ocv = model.predict(inp, verbose=0)[0][0]
    discharge_ocvs.append(ocv)

# Charge 
charge_ocvs = []
for soc in socs:
    inp = scaler.transform([[soc, temp, -1]])  # charge = -1
    ocv = model.predict(inp, verbose=0)[0][0]
    charge_ocvs.append(ocv)

# Plot
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

# Plot OCV curves
ax1.plot(socs*100, discharge_ocvs, 'b-', label='Discharge', linewidth=2)
ax1.plot(socs*100, charge_ocvs, 'r-', label='Charge', linewidth=2)
ax1.set_xlabel('SoC (%)')
ax1.set_ylabel('OCV (V)')
ax1.set_title('OCV vs SoC (Model Predictions)')
ax1.legend()
ax1.grid(True, alpha=0.3)

# Plot difference
diff = np.array(charge_ocvs) - np.array(discharge_ocvs)
ax2.plot(socs*100, diff*1000, 'g-', linewidth=2)
ax2.set_xlabel('SoC (%)')
ax2.set_ylabel('Difference (mV)')
ax2.set_title('Charge - Discharge OCV Difference')
ax2.grid(True, alpha=0.3)
ax2.axhline(y=0, color='k', linestyle='--', alpha=0.5)

plt.tight_layout()
plt.savefig('outputs/ocv_model_diagnostic.png', dpi=150)
print("Plot saved to outputs/ocv_model_diagnostic.png")

# Analysis
print("\nDISCHARGE OCV Curve:")
print(f"  At 100% SoC: {discharge_ocvs[-1]:.3f}V")
print(f"  At  50% SoC: {discharge_ocvs[25]:.3f}V")
print(f"  At   0% SoC: {discharge_ocvs[0]:.3f}V")
print(f"  Monotonic: {all(discharge_ocvs[i] >= discharge_ocvs[i-1] for i in range(1, len(discharge_ocvs)))}")

print("\nCHARGE OCV Curve:")
print(f"  At 100% SoC: {charge_ocvs[-1]:.3f}V")
print(f"  At  50% SoC: {charge_ocvs[25]:.3f}V")
print(f"  At   0% SoC: {charge_ocvs[0]:.3f}V")
print(f"  Monotonic: {all(charge_ocvs[i] >= charge_ocvs[i-1] for i in range(1, len(charge_ocvs)))}")

print("\nDIFFERENCE:")
print(f"  Avg: {np.mean(diff)*1000:.1f}mV")
print(f"  Max: {np.max(diff)*1000:.1f}mV")
print(f"  Min: {np.min(diff)*1000:.1f}mV")

print("\n" + "="*60)
print("DIAGNOSIS:")
print("="*60)

issues = []
if not all(discharge_ocvs[i] >= discharge_ocvs[i-1] for i in range(1, len(discharge_ocvs))):
    issues.append("✗ Discharge OCV is NOT monotonic (should increase with SoC)")
if not all(charge_ocvs[i] >= charge_ocvs[i-1] for i in range(1, len(charge_ocvs))):
    issues.append("✗ Charge OCV is NOT monotonic (should increase with SoC)")
if abs(np.mean(diff)) > 0.1:
    issues.append(f"✗ Average difference too large: {np.mean(diff)*1000:.0f}mV")
if discharge_ocvs[-1] < 3.8 or discharge_ocvs[-1] > 4.25:
    issues.append(f"✗ 100% SoC discharge OCV out of range: {discharge_ocvs[-1]:.3f}V (expect ~4.0-4.2V)")
if discharge_ocvs[0] < 2.8 or discharge_ocvs[0] > 3.5:
    issues.append(f"✗ 0% SoC discharge OCV out of range: {discharge_ocvs[0]:.3f}V (expect ~3.0-3.3V)")

if issues:
    print("\n** MODEL HAS ISSUES **\n")
    for issue in issues:
        print(issue)
    print("\nRECOMMENDATIONS:")
    print("1. Retrain model with full dataset (currently 10% sample)")
    print("2. Use larger network or more epochs")
    print("3. Check training data quality")
    print("4. Consider cycle_type may not be helpful")
else:
    print("✓ Model looks reasonable")
