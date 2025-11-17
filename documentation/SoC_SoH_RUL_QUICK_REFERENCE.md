# Quick Reference: SoC, SoH, RUL Formulas & Parameters

##  SoC (State of Charge) - Unscented Kalman Filter

### Core Formulas

| Name | Formula | Parameters |
|------|---------|------------|
| **Coulomb Counting** | $\text{SoC}_{k+1} = \text{SoC}_k - \frac{\Delta t \cdot I_k}{3600 \cdot C_{\text{nom}} \cdot \eta}$ | Δt=1s, C_nom=2.3Ah, η=0.99 |
| **Polarization RC** | $U_{p,k+1} = \alpha \cdot U_{p,k} + \beta \cdot I_k$ | α=exp(-Δt/R_D·C_D), β=R_D(1-α) |
| **Predicted Voltage** | $V_{\text{pred}} = \text{OCV}(\text{SoC}) - I_k \cdot R_0 - U_p$ | R0=0.05Ω |
| **Innovation** | $\nu = V_{\text{meas}} - V_{\text{pred}}$ | Drives SoC correction |
| **OCV Model** | $\text{OCV}(\text{SoC}, \text{Temp})$ | LSTM-based model |

### UKF Parameters

```python
ukf_params = {
    "dt": 1.0,              # Time step (seconds)
    "Q_max": 2.3,           # Nominal capacity (Ah)
    "R_0": 0.05,             # Series resistance (Ω)
    "R_D": 0.01,            # Polarization resistance (Ω)
    "C_D": 500.0,           # Polarization capacitance (F)
    "eta": 0.99,            # Coulombic efficiency
    "Q": [1e-6, 1e-6],      # Process noise covariance
    "R": 1e-2,              # Measurement noise variance
}
```

### Output
- **Column:** `UKF_SoC`
- **Range:** [0.0, 1.0]
- **Meaning:** Fraction of nominal capacity available

---

##  SoH (State of Health) - LSTM Prediction

### Training Label Calculation

```python
# Per discharge record:
C_nom = max(Capacity) for battery_id  # Battery's nominal capacity
SoH = Measured_Capacity / C_nom        # Normalized to [0, 1]

# Example:
# Battery new: 2.5 Ah
# After 100 cycles: 2.0 Ah measured
# SoH = 2.0 / 2.5 = 0.8 (80%)
```

### LSTM Architecture

```
Input Layer:     (batch, 50, 4)  [50 timesteps × 4 features]
    ↓
LSTM(64) + Dropout(0.2)
    ↓
LSTM(32) + Dropout(0.2)
    ↓
Dense(32, ReLU)
    ↓
Dense(2, Linear)  ← [pred_SoH, pred_RUL_scaled]
```

### Input Features
```python
features = [
    Voltage_measured,      # Raw voltage (V)
    Current_measured,      # Raw current (A)
    Temperature_measured,  # Raw temperature (°C)
    UKF_SoC               # Estimated state of charge
]  # Shape: (50, 4) per window
```

### Loss Function
$$L_{\text{SoH}} = \text{RMSE} = \sqrt{\frac{1}{N} \sum_i (\hat{y}_i - y_i)^2}$$

**Achieved:** RMSE = 0.0425 (≈4.25% error)

### Output
- **Column:** `pred_SoH`
- **Range:** [0.0, 1.0]
- **Meaning:** Remaining capacity fraction
- **Note:** Only populated at window-end rows (every 50 timesteps)

---

##  RUL (Remaining Useful Life) - LSTM Prediction

### Training Label Calculation

```python
# Per battery, chronologically sorted discharges:
for idx in range(num_discharges):
    future_indices = where(SoH[idx+1:] < 0.8)  # Threshold = 0.8
    if future_indices exist:
        RUL = index_of_first_future - idx    # Cycles until threshold
    else:
        RUL = num_discharges - idx - 1       # Remaining cycles

# Example:
# Discharge #1: SoH=1.00 → RUL=4 (threshold at #5)
# Discharge #2: SoH=0.96 → RUL=3 (threshold at #5)
# Discharge #3: SoH=0.92 → RUL=2 (threshold at #5)
# Discharge #4: SoH=0.88 → RUL=1 (threshold at #5)
# Discharge #5: SoH=0.80 → RUL=1 (at threshold)
```

### Scaling for Network

```python
# Scale raw RUL to [0, 1] for network stability
max_rul_for_battery = num_discharge_cycles  # Total cycles per battery
RUL_scaled = RUL_cycles / max_rul_for_battery

# Example:
# RUL = 20 cycles, max = 100 cycles
# RUL_scaled = 0.20
```

### Loss Function
$$L_{\text{RUL}} = \text{MAE} = \frac{1}{N} \sum_i |\hat{y}_i - y_i|$$

**Achieved:** MAE = 0.0838 (≈8.38% scaled error)

### Output
- **Column:** `pred_RUL_scaled`
- **Range:** [0.0, 1.0]
- **Meaning:** Scaled fraction of cycles remaining
- **Note:** Only populated at window-end rows (every 50 timesteps)

### Converting to Absolute Cycles

```python
# If you know max cycles in training:
RUL_absolute = pred_RUL_scaled × rul_max

# Example:
# pred_RUL_scaled = 0.0684
# rul_max = 2000 (max cycles seen during training)
# RUL_absolute = 0.0684 × 2000 ≈ 137 cycles
```

---

##  Key Relationships & Thresholds

### SoH Interpretation

| SoH Value | Status | Degradation | RUL Expectation |
|-----------|--------|-------------|-----------------|
| 0.95–1.00 | Excellent | Fresh battery | High (0.5–1.0) |
| 0.85–0.95 | Good | Early degradation | Good (0.2–0.5) |
| 0.80–0.85 | Fair | Noticeable aging | Fair (0.05–0.2) |
| <0.80 | Critical | End-of-life threshold | Low (0.0–0.05) |

### SoH ↔ RUL Correlation

```python
# Inverse relationship by definition:
# Higher SoH → Battery fresher → More cycles remaining (higher RUL)
# Lower SoH → Battery aged → Fewer cycles remaining (lower RUL)

# Mathematical:
RUL_cycles = cycles_until_SoH_drops_below_0.8(current_SoH)
```

---

##  Tuning Parameters

### When to Adjust

| Problem | Parameter | Action |
|---------|-----------|--------|
| SoC stuck at 0 or 1 | `C_nom`, `R0`, `eta` | Verify battery specs, refit OCV |
| SoC oscillating | `Q`, `R` (noise) | ↑ Q = trust measurements more |
| Slow SoC response | `max_soc_step` | ↑ Increase to allow faster tracking |
| Unrealistic SoH/RUL | seq_len | ↓ Reduce if overfitting |
| Slow inference | seq_len | ↓ Try 25 instead of 50 |

### Recommended CLI Overrides

```bash
# Use different capacity
python3 scripts/infer_ukf.py --input data.csv \
  --ukf-params '{"Q_max": 2.5, "R_0": 0.06}'

# Tighter sequence window
python3 scripts/infer_ukf.py --input data.csv \
  --seq-len 25  # Default: 50

# Diagnostic mode to inspect UKF
python3 scripts/infer_ukf.py --input data.csv \
  --diagnostic  # Outputs .ukf_diag.csv

# Convert RUL to absolute cycles
python3 scripts/infer_ukf.py --input data.csv \
  --rul-max 2000  # Max cycles in dataset
```

---

##  Training Metrics Summary

| Metric | Value | Interpretation |
|--------|-------|---|
| **Dataset** | 1500 files (50% of 2,769) | Balanced coverage, CPU-friendly |
| **Train/Val Split** | 85% / 15% | Standard practice |
| **Windows** | ~75K train, ~13K val | Overlapping 50-step sequences |
| **SoH RMSE** | 0.0425 | ±4.25% typical error |
| **RUL MAE** | 0.0838 | ±8.38% scaled error |
| **Improvement** | +52% SoH, +46% RUL | vs. baseline model |

---

##  Diagnostic Mode Output

Run with `--diagnostic` to get per-step UKF details:

```csv
# outputs/eda/data.ukf_diag.csv
prior_soc,v_pred,innovation,post_soc
0.9500,3.650,0.040,0.9504
0.9504,3.645,0.045,0.9509
0.9509,3.641,0.050,0.9514
...
```

**Use for:**
- Identifying where large innovations occur
- Debugging EKF saturation
- Validating OCV polynomial fit
- Tuning process/measurement noise (Q, R)

---

##  Quick Start: Running Inference

```bash
# Simplest (uses all defaults)
python3 scripts/infer_ukf.py --input cleaned_dataset/data/00001.csv

# With diagnostics and custom output
python3 scripts/infer_ukf.py \
  --input cleaned_dataset/data/00001.csv \
  --out my_predictions.csv \
  --diagnostic

# With custom parameters and absolute RUL
python3 scripts/infer_ukf.py \
  --input cleaned_dataset/data/00001.csv \
  --ukf-params '{"Q_max": 2.4}' \
  --rul-max 2000 \
  --seq-len 25

# Refit OCV polynomial
python3 scripts/infer_ukf.py \
  --input cleaned_dataset/data/00001.csv \
  --auto-init-ocv
```

---

##  Files for Each Component

| Component | File | Key Functions |
|-----------|------|---|
| **SoC (UKF)** | `scripts/ukf_soc.py` | `DualUKF`, `UnscentedTransform` |
| **Training** | `scripts/train_dekf_lstm.py` | `prepare_dataset`, `build_lstm` |
| **Inference** | `scripts/infer_ukf.py` | `infer_single_file`, `run_ukf_for_file` |
| **Artifacts** | `outputs/eda/` | `.keras` model, `.npy` OCV, `.json` metrics |

---

##  Summary

**SoC** = Real-time charge level via **UKF** (physics-based)  
**SoH** = Capacity degradation predicted by **LSTM** (data-driven)  
**RUL** = Remaining cycles predicted by **LSTM** (data-driven)

All three work together:
1. UKF produces reliable SoC estimate from voltage/current
2. SoC becomes a feature for the LSTM
3. LSTM learns capacity degradation patterns (SoH, RUL) from historical discharge data
4. Predictions help schedule maintenance before failure

