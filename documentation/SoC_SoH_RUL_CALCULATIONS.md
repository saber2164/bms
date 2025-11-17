# SoC, SoH, and RUL Calculations - Technical Deep Dive

**Date:** 15 November 2025  
**Scope:** Detailed explanation of how each metric is calculated, parameters used, and implementation details

---

## Overview

The hybrid EKF+LSTM system produces three key battery health metrics:

1. **SoC (State of Charge)** — Estimated via Extended Kalman Filter (EKF) in real-time
2. **SoH (State of Health)** — Predicted by LSTM from time-series features
3. **RUL (Remaining Useful Life)** — Predicted by LSTM as scaled fraction

This document explains the mathematical models and implementation details for each.

---

## 1. SoC (State of Charge) Calculation

### 1.1 What is SoC?

**SoC** represents the percentage of battery charge available relative to nominal capacity:
- **0.0** = fully discharged
- **1.0** = fully charged
- **0.5** = 50% charge remaining

### 1.2 UKF-Based SoC Estimation

SoC is estimated using a **Dual Unscented Kalman Filter** architecture.

#### Dual Filter Architecture
- **State Filter (UKF):** Estimates the fast-changing state (SoC) at every time step.
- **Parameter Filter (UKF):** Estimates the slow-changing parameters (like capacity `Q_max` and resistance `R_0`) at a slower rate.

This allows the filter to adapt to battery aging.

#### The Unscented Transform
Instead of linearizing the model like an EKF, the UKF uses the **Unscented Transform**:
1.  **Sigma Points:** A set of points ("sigma points") are chosen to capture the mean and covariance of the state.
2.  **Propagation:** These points are propagated through the true non-linear model.
3.  **New Estimate:** A new mean and covariance are calculated from the propagated points.

This approach is more accurate for non-linear systems.

#### State Vector (State Filter)

$$\mathbf{x}_k = \begin{bmatrix} \text{SoC}_k \\ U_{p,k} \end{bmatrix}$$

Where:
- **SoC** = state of charge (0–1)
- **U_p** = polarization voltage (V)

#### 1.2.1 Process Model (Non-linear)
The process model is a non-linear function `f` that describes the evolution of the state:
$$\mathbf{x}_{k+1} = f(\mathbf{x}_k, I_k) + w_k$$
where `w_k` is the process noise.

**SoC Update (Coulomb Counting):**
$$\text{SoC}_{k+1} = \text{SoC}_k - \frac{\Delta t \cdot I_k}{3600 \cdot Q_{\text{max}} \cdot \eta}$$

**Polarization RC Model Update:**
$$U_{p,k+1} = e^{-\Delta t / \tau} \cdot U_{p,k} + R_D \cdot (1 - e^{-\Delta t / \tau}) \cdot I_k$$
where `τ = R_D * C_D`.

#### 1.2.2 Observation Model (Non-linear)
The observation model is a non-linear function `h` that relates the state to the measurement:
$$V_{\text{meas}} = h(\mathbf{x}_k, I_k) + v_k$$
where `v_k` is the measurement noise.

**Predicted Terminal Voltage:**
$$V_{\text{pred}} = \text{OCV}(\text{SoC}_k, \text{Temp}_k) - I_k \cdot R_0 - U_p$$

Where:
- **OCV(SoC, Temp)** = Open Circuit Voltage, provided by a pre-trained LSTM model.
- **R_0** = series resistance, estimated by the parameter filter.

#### 1.2.3 UKF Update Step
The UKF uses the voltage innovation to refine the SoC estimate without calculating Jacobians.

**Innovation (error signal):**
$$\nu_k = V_{\text{meas}} - V_{\text{pred}}$$

The filter computes the Kalman Gain based on the propagated sigma points and uses it to correct the state and its covariance.

**Safeguards:**
The implementation includes safeguards to clamp the SoC between 0 and 1 and limit the maximum step size to prevent divergence.

### 1.3 OCV (Open Circuit Voltage) LSTM Model

The OCV model is a **data-driven LSTM neural network** that provides the OCV value for the UKF.

**Model:**
- A pre-trained Long Short-Term Memory (LSTM) network.
- **Inputs:** State of Charge (SoC) and Temperature.
- **Output:** Open-Circuit Voltage (OCV).

This approach is more accurate than a fixed polynomial because the LSTM can learn the complex, non-linear relationship between SoC, temperature, and OCV from a large dataset. It adapts to different battery behaviors and operating conditions.

### 1.4 Default UKF Parameters

| Parameter | Default | Unit | Description |
|-----------|---------|------|-------------|
| **dt** | 1.0 | s | Time step (sampling interval) |
| **Q_max** | 2.3 | Ah | Nominal battery capacity |
| **R_0** | 0.05 | Ω | Series (ohmic) resistance |
| **R_D** | 0.01 | Ω | Polarization branch resistance |
| **C_D** | 500.0 | F | Polarization branch capacitance |
| **eta** | 0.99 | — | Coulombic efficiency (charge/discharge loss) |
| **Q** (process noise) | diag([1e-6, 1e-6]) | — | State transition uncertainty |
| **R** (measurement noise) | 1e-2 | V² | Voltage measurement uncertainty |

### 1.5 SoC Output

**In Inference CSV:**
- Column: `UKF_SoC`
- Range: [0.0, 1.0]
- Meaning: Percentage of nominal capacity available

**Example:**
```
EKF_SoC = 0.81 → Battery is ~81% charged
```

---

## 2. SoH (State of Health) Calculation

### 2.1 What is SoH?

**SoH** represents the battery's remaining capacity relative to its nominal (new) state:
- **1.0** = battery at 100% capacity (like new)
- **0.8** = battery at 80% capacity (common end-of-life threshold)
- **0.0** = battery at 0% capacity (dead)

### 2.2 SoH Label During Training

During training, SoH labels are computed from metadata:

**Per-Discharge SoH:**
$$\text{SoH} = \frac{\text{Capacity}_{\text{measured}}}{\text{C}_{\text{nom}}}$$

Where:
- **Capacity_measured** = actual measured capacity for this discharge cycle (from metadata.csv)
- **C_nom** = battery's nominal capacity = max(Capacity) observed for that battery

**Example:**
- Battery #1 new capacity: 2.5 Ah
- Battery #1 after 500 cycles: 2.0 Ah measured
- **SoH** = 2.0 / 2.5 = **0.8** (80% health)

### 2.3 SoH Prediction via LSTM

The LSTM model takes a **50-timestep window** of features and predicts SoH:

**Input Features (sliding window):**
$$\mathbf{X}_{\text{window}} = \begin{bmatrix}
V_{\text{measured}, 1} & I_{\text{measured}, 1} & T_{\text{measured}, 1} & \text{UKF\_SoC}_1 \\
V_{\text{measured}, 2} & I_{\text{measured}, 2} & T_{\text{measured}, 2} & \text{UKF\_SoC}_2 \\
\vdots & \vdots & \vdots & \vdots \\
V_{\text{measured}, 50} & I_{\text{measured}, 50} & T_{\text{measured}, 50} & \text{UKF\_SoC}_{50}
\end{bmatrix}$$

Shape: **(50, 4)** — 50 timesteps × 4 features

**LSTM Architecture:**

```
Input Layer:           (batch, 50, 4)
    ↓
LSTM Layer 1:          64 units, return_sequences=True
    ↓ (dropout 0.2)
LSTM Layer 2:          32 units, return_sequences=False
    ↓ (dropout 0.2)
Dense Layer 1:         32 units, ReLU activation
    ↓
Dense Output Layer:    2 units, linear activation
    ↓
Output:                [pred_SoH, pred_RUL_scaled]
```

**LSTM Processing:**
1. **Layer 1 (64 LSTM units):** Learns temporal patterns across 50 timesteps
   - Captures degradation signatures in voltage, current, temperature
2. **Layer 2 (32 LSTM units):** Compresses learned patterns into summary state
3. **Dense Layers:** Maps LSTM state to SoH prediction

**Activation Functions:**
- **Hidden layers:** ReLU (rectified linear) — allows non-linear feature extraction
- **Output layer:** Linear — no bounds, ranges naturally 0–1 after training

**Training Loss:**
$$L_{\text{SoH}} = \text{RMSE} = \sqrt{\frac{1}{N} \sum_{i=1}^{N} (\hat{\text{SoH}}_i - \text{SoH}_i)^2}$$

**Achieved RMSE:** 0.0425 (≈ 4.25% typical error)

### 2.4 SoH Output

**In Inference CSV:**
- Column: `pred_SoH`
- Range: [0.0, 1.0]
- Meaning: Predicted remaining capacity fraction

**Note:** SoH predictions appear only at window end rows (every 50 timesteps by default).

**Example:**
```
pred_SoH = 0.7401 → Battery at ~74% of original capacity
```

---

## 3. RUL (Remaining Useful Life) Calculation

### 3.1 What is RUL?

**RUL** predicts how many discharge cycles remain before the battery degrades below an acceptable threshold (default: SoH < 0.8).

- **High RUL** (e.g., 0.5) = many cycles remaining
- **Low RUL** (e.g., 0.01) = few cycles remaining, end-of-life approaching

### 3.2 RUL Label During Training

During training, RUL labels are computed per-battery using chronological discharge records:

**Algorithm:**
```
For each battery in chronological order (oldest → newest discharge):
  For each discharge record i:
    - Find first future record j where SoH[j] < threshold (0.8)
    - If found: RUL[i] = j - i (cycles until threshold)
    - If not found: RUL[i] = (n_total - i) (remaining cycles)
```

**Example:**

Battery #1 chronological discharges:
```
Discharge #1: Capacity=2.5 Ah → SoH=1.0
Discharge #2: Capacity=2.4 Ah → SoH=0.96
Discharge #3: Capacity=2.3 Ah → SoH=0.92
Discharge #4: Capacity=2.2 Ah → SoH=0.88
Discharge #5: Capacity=2.0 Ah → SoH=0.80  ← threshold
Discharge #6: Capacity=1.8 Ah → SoH=0.72
```

RUL labels:
```
Discharge #1: RUL = 4 cycles (reaches threshold at discharge #5)
Discharge #2: RUL = 3 cycles (reaches threshold at discharge #5)
Discharge #3: RUL = 2 cycles (reaches threshold at discharge #5)
Discharge #4: RUL = 1 cycle (reaches threshold at discharge #5)
Discharge #5: RUL = 1 cycle (at threshold, 1 cycle until failure)
Discharge #6: RUL = 0 cycles (already below threshold)
```

### 3.3 RUL Scaling

Raw RUL values (cycle counts) are **scaled to [0, 1]** for neural network training stability:

**Per-Battery Scaling:**
$$\text{RUL}_{\text{scaled}} = \frac{\text{RUL}_{\text{cycles}}}{\text{max\_RUL}_{\text{battery}}}$$

Where:
- **max_RUL_battery** = total discharge cycles for that battery
- Ensures LSTM output is in normalized range [0, 1]

**Example:**
- Battery #1 has 100 total discharge cycles
- Current discharge has RUL = 20 cycles
- **RUL_scaled** = 20 / 100 = **0.20**

### 3.4 RUL Prediction via LSTM

The same LSTM that predicts SoH also predicts **RUL_scaled** (second output):

**Architecture:** Same as SoH (see section 2.3)

**Output:** 2D vector: [pred_SoH, pred_RUL_scaled]

**Training Loss:**
$$L_{\text{RUL}} = \text{MAE} = \frac{1}{N} \sum_{i=1}^{N} |\hat{\text{RUL}}_i - \text{RUL}_i|$$

**Achieved MAE:** 0.0838 (≈ 8.38% typical scaled error)

### 3.5 RUL Output

**In Inference CSV:**
- Column: `pred_RUL_scaled`
- Range: [0.0, 1.0]
- Meaning: Fraction of cycles remaining (relative to max cycles seen in training)

**Converting to Absolute Cycles:**

If you provide `--rul-max` (max cycles in training dataset):
$$\text{RUL}_{\text{absolute}} = \text{pred\_RUL\_scaled} \times \text{rul\_max}$$

**Example:**
```
pred_RUL_scaled = 0.0684
rul_max = 2000 (max cycles observed during training)
RUL_absolute = 0.0684 × 2000 ≈ 137 cycles remaining
```

### 3.6 Inverse Relationship: SoH ↔ RUL

By definition, higher SoH correlates with higher RUL:

| SoH | Interpretation | Typical RUL |
|-----|---|---|
| 0.95+ | Fresh battery | 0.5–1.0 (high) |
| 0.85–0.95 | Early degradation | 0.2–0.5 |
| 0.80–0.85 | Aging, approaching threshold | 0.05–0.2 |
| <0.80 | Failure region | 0.0–0.05 (very low) |

---

## 4. Feature Engineering & Data Flow

### 4.1 Training Data Pipeline

```
1. Load metadata.csv → per-file Capacity labels
2. Compute SoH = Capacity / C_nom per file
3. Compute RUL = cycles until SoH < 0.8 per battery
4. For each discharge CSV file:
   a. Run UKF to produce SoC time-series
   b. Load V_measured, I_measured, T_measured
   c. Build feature matrix: [V, I, T, UKF_SoC]
5. Create sliding windows (50-timestep overlapping sequences)
6. Assign labels [SoH, RUL_scaled] to each window
7. Train LSTM with train/val split (85/15)
8. Save best model + OCV coefficients + metrics
```

### 4.2 Inference Data Pipeline

```
1. Load user's CSV (requires V_measured, I_measured, optional T_measured)
2. Auto-fit OCV polynomial (if not cached)
3. Create UKF with loaded/fitted OCV coefficients
4. Run UKF over entire CSV → SoC time-series
5. Build features: [V, I, T, UKF_SoC]
6. Create sliding windows (50-timestep sequences)
7. Run LSTM on each window → [pred_SoH, pred_RUL_scaled]
8. Write per-row predictions to output CSV:
   - UKF_SoC: result from step 4
   - pred_SoH: from step 7 (only at window-end rows)
   - pred_RUL_scaled: from step 7 (only at window-end rows)
```

### 4.3 Sequence Window Creation

**Sliding Window Generation:**

For a time-series of length N and seq_len=50:
- Window 1: rows [0:50]
- Window 2: rows [1:51]
- Window 3: rows [2:52]
- ...
- Window M: rows [N-50:N]

**Number of windows:** M = N - seq_len + 1

**Prediction alignment:** LSTM prediction for window i is assigned to **row (i + seq_len - 1)** in output CSV.

---

## 5. Quality Metrics & Validation

### 5.1 Training Metrics

| Metric | Value | Type | Interpretation |
|--------|-------|------|---|
| **SoH RMSE** | 0.0425 | Regression error | Typical SoH prediction error ≈ 4.25% |
| **RUL MAE** | 0.0838 | Regression error | Typical RUL error ≈ 8.38% of max cycles |

**Calculation:**
```python
SoH_RMSE = sqrt(mean((pred_SoH - true_SoH)²))
RUL_MAE = mean(|pred_RUL_scaled - true_RUL_scaled|)
```

### 5.2 Validation Results (5 Test Files)

| Metric | Value | Range | Status |
|--------|-------|-------|--------|
| **UKF_SoC** | 0.8434 (mean) | [0.8085, 0.9687] | ✅ No saturation |
| **pred_SoH** | 0.4711 (mean) | [0.2476, 0.8451] | ✅ Diverse states |
| **pred_RUL_scaled** | 0.0264 (mean) | [0.0092, 0.0684] | ✅ Inverse SoH correlation |

### 5.3 Data Quality Checks

**Automatic Validation:**
1. ✅ Drops rows with NaN in V_measured or I_measured
2. ✅ Warns if voltage range suspicious (e.g., 0–100V unrealistic)
3. ✅ Warns if current range suspicious (e.g., consistently near limits)
4. ✅ Detects EKF saturation (SoC stuck at 0 or 1 for >50% of rows)
5. ✅ Validates output: all predictions in [0, 1]

---

## 6. Parameter Sensitivity & Tuning

### 6.1 EKF Parameters

| Parameter | Effect | Tuning |
|-----------|--------|--------|
| **C_nom** | Changes Coulomb counting rate | Set to battery's rated capacity (Ah) |
| **R0** | Series resistance (voltage drop) | ↑ increases voltage ripple compensation |
| **R_D, C_D** | Polarization time constant (RC product) | Controls transient response |
| **eta** | Coulombic efficiency | Use 0.99 for lithium-ion, 0.95 for lead-acid |
| **max_soc_step** | Outlier protection | ↑ allows faster SoC tracking, less robust |
| **Q (process noise)** | EKF trust in model | ↑ Q means trust measurements more |
| **R (meas. noise)** | EKF trust in measurements | ↑ R means trust model more |

### 6.2 LSTM Parameters

| Parameter | Default | Effect |
|-----------|---------|--------|
| **seq_len** | 50 | Larger = longer temporal context, slower inference |
| **LSTM units (L1, L2)** | 64, 32 | Larger = more capacity, slower training |
| **Dropout rate** | 0.2 | Prevents overfitting |
| **Dense units** | 32 | Bottleneck before output |
| **Learning rate** | 1e-3 (Adam optimizer) | Affects convergence speed |

---

## 7. Advanced: Diagnostic Mode

Run inference with `--diagnostic` flag to enable detailed EKF inspection:

```bash
python3 scripts/infer_hybrid.py --input data.csv --diagnostic
```

**Output CSV:** `outputs/eda/data.ekf_diag.csv`

**Columns:**
- `prior_soc`: SoC before measurement update
- `v_pred`: predicted voltage
- `innovation`: measurement error (v_meas - v_pred)
- `post_soc`: SoC after measurement update

**Use Case:** Identify where large innovations drive SoC errors.

---

## 8. Example Calculation (Step-by-Step)

### Input Data Sample

```csv
Time,Voltage_measured,Current_measured,Temperature_measured
0,3.7,-1.5,25.0
1,3.69,-1.5,25.1
2,3.68,-1.5,25.2
```

### Step 1: EKF SoC Estimation

**Initial state:** x₀ = [0.95, 0.0]ᵀ (95% SoC, 0V polarization)

**Row 1 (t=1s):**
- Current I₁ = -1.5 A (discharge)
- Predict: SoC₁⁻ = 0.95 - (1.0 × (-1.5)) / (3600 × 2.3 × 0.99) ≈ 0.9502
- Predict: V_pred = OCV(0.9502) - (-1.5) × 0.05 - 0 ≈ 3.65 V
- Measure: V_meas = 3.69 V
- Innovation: ν = 3.69 - 3.65 = 0.04 V
- Update: SoC₁ ≈ 0.9502 + 0.04 × (Kalman gain) ≈ **0.9504**

**Repeat for all rows** → SoC time-series

### Step 2: Feature Matrix Construction

```
Row 50 features (end of first 50-step window):
[V_measured=3.42, I_measured=-1.5, T_measured=25.8, EKF_SoC=0.80]
```

### Step 3: LSTM Prediction

**Input:** 50×4 matrix of [V, I, T, SoC]

**LSTM processes 50 timesteps** → learns degradation patterns

**Output:** [pred_SoH=0.74, pred_RUL_scaled=0.009]

### Step 4: Output CSV Row 50

```
Time=49.0, Voltage_measured=3.42, Current_measured=-1.5, 
Temperature_measured=25.8, EKF_SoC=0.80, 
pred_SoH=0.74, pred_RUL_scaled=0.009
```

Rows 1–49 have NaN for pred_SoH and pred_RUL_scaled (insufficient history).

---

## 9. Troubleshooting

| Issue | Cause | Solution |
|-------|-------|----------|
| SoC stuck at 0 or 1 | Poorly-fitting OCV model | Run with `--diagnostic`, enable `--auto-init-ocv` |
| Large innovations | Current sign mismatch | Check Current_measured polarity vs. training data |
| Negative predictions | OCV model overfitting | Refit OCV on larger dataset |
| NaN in output | Missing voltage/current rows | Pre-process CSV to remove NaN rows |
| Slow inference | Large seq_len or large file | Reduce `--seq-len` or use `--sample-rows` |

---

## 10. References

**Key Files:**
- `scripts/ekf_soc.py` — EKF implementation
- `scripts/hybrid_train.py` — Training pipeline
- `scripts/infer_hybrid.py` — Inference script
- `outputs/eda/hybrid_metrics.json` — Training metrics
- `outputs/eda/ocv_coeffs.npy` — Fitted OCV coefficients

**Documentation:**
- `HOWTO_RUN.md` — Quick reference
- `WORKFLOW_TECHNICAL.md` — Architecture overview
- `TEST_REPORT.md` — Validation results

---

## Summary Table

| Metric | Calculation | Parameters | Range | Error |
|--------|---|---|---|---|
| **SoC** | EKF (Coulomb count + RC model + voltage feedback) | C_nom, R0, R_D, C_D, η, OCV | [0, 1] | Depends on OCV fit |
| **SoH** | LSTM on 50-step [V,I,T,SoC] windows | LSTM weights, seq_len=50 | [0, 1] | RMSE=0.0425 |
| **RUL** | LSTM on same features, scaled cycles | LSTM weights, seq_len=50 | [0, 1] | MAE=0.0838 |

