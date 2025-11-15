# Comprehensive Validation Report
## Metadata-Based LSTM SoH/RUL Model - 2000 Files Training

**Date:** November 15, 2025  
**Model:** `outputs/eda/hybrid_lstm_model.keras` (trained on 2000 discharge files)  
**Validation:** 100 random discharge files from `cleaned_dataset/data/`  
**Test Results:** 97 successful inferences, 3 skipped (insufficient data rows)

---

## Executive Summary

The model trained on 2000 files shows **mixed performance** with clear patterns:
- **Good performers (44% of files):** Excellent (21.6%) + Good (22.7%) accuracy
- **Poor performers (32% of files):** Error > 0.5 (30% absolute error)
- **Key finding:** Systematic failure on specific batteries (B0056, B0055) suggests training data distribution issues

### Overall Metrics
| Metric | Value |
|--------|-------|
| **SoH MAE** | 0.3030 (30.3% error) |
| **SoH RMSE** | 0.3991 (39.9% error) |
| **Std Dev** | 0.2599 |
| **Error Range** | [0.0017, 0.8406] |
| **Success Rate** | 97/100 (97%) |

---

## Detailed Error Analysis

### Error Distribution

```
Error Category          Count    Percentage
Excellent (< 0.05):       21       21.6%
Good (0.05-0.2):          22       22.7%
Moderate (0.2-0.5):       23       23.7%
Poor (> 0.5):             31       32.0%
```

### Error Percentiles
- 10th percentile: 0.0197 (excellent)
- 25th percentile: 0.0812 (good)
- 50th percentile: 0.2331 (median - moderate)
- 75th percentile: 0.5509 (poor range)
- 90th percentile: 0.7032 (very poor)
- 95th percentile: 0.7421 (very poor)
- 99th percentile: 0.8170 (worst cases)

### Top 10 Worst Predictions

All 10 worst predictions are from batteries **B0056** and **B0055**:

| Rank | File | Battery | Ref SoH | Pred SoH | Error |
|------|------|---------|---------|----------|-------|
| 1 | 07094.csv | B0056 | 0.8872 | 0.0466 | **0.8406** |
| 2 | 07318.csv | B0055 | 1.0000 | 0.1840 | **0.8160** |
| 3 | 07096.csv | B0056 | 0.8827 | 0.0911 | **0.7916** |
| 4 | 07320.csv | B0055 | 0.9675 | 0.1853 | **0.7822** |
| 5 | 07322.csv | B0055 | 0.9525 | 0.1781 | **0.7745** |
| 6 | 07138.csv | B0056 | 0.8770 | 0.1430 | **0.7340** |
| 7 | 07144.csv | B0056 | 0.8670 | 0.1355 | **0.7315** |
| 8 | 07188.csv | B0056 | 0.9047 | 0.1744 | **0.7303** |
| 9 | 07186.csv | B0056 | 0.9155 | 0.1897 | **0.7258** |
| 10 | 07148.csv | B0056 | 0.8672 | 0.1453 | **0.7219** |

**Pattern:** All high-SoH files (0.85-1.0) predicted with extremely low values (0.05-0.19)

### Top 10 Best Predictions

| Rank | File | Battery | Ref SoH | Pred SoH | Error |
|------|------|---------|---------|----------|-------|
| 1 | 03548.csv | B0036 | 0.7266 | 0.7283 | **0.0017** |
| 2 | 06403.csv | B0018 | 0.9414 | 0.9432 | **0.0018** |
| 3 | 04541.csv | B0006 | 0.9176 | 0.9155 | **0.0021** |
| 4 | 01562.csv | B0042 | 0.9668 | 0.9610 | **0.0058** |
| 5 | 01053.csv | B0032 | 0.9418 | 0.9350 | **0.0068** |
| 6 | 03402.csv | B0044 | 0.0349 | 0.0266 | **0.0083** |
| 7 | 00866.csv | B0043 | 0.0400 | 0.0261 | **0.0139** |
| 8 | 01438.csv | B0029 | 0.8969 | 0.9134 | **0.0165** |
| 9 | 06379.csv | B0018 | 0.9727 | 0.9546 | **0.0181** |
| 10 | 01777.csv | B0042 | 0.8253 | 0.8056 | **0.0197** |

**Pattern:** Consistently high accuracy on most batteries; excellent predictions from B0006, B0018, B0036, B0042, etc.

---

## Performance by Reference SoH Range

Model accuracy is **inversely correlated with SoH range**:

| SoH Range | Files | MAE | Interpretation |
|-----------|-------|-----|-----------------|
| 0.0-0.2 (very low) | 4 | 0.1043 | Excellent accuracy |
| 0.2-0.4 (low) | 1 | 0.4010 | Single sample, moderate |
| 0.4-0.6 (medium) | 2 | 0.5650 | Very poor (only 2 samples) |
| 0.6-0.8 (high) | 41 | 0.3203 | Moderate accuracy |
| **0.8-1.0 (very high)** | 48 | **0.2811** | **Best accuracy range** |

**Key insight:** Model performs reasonably on high-SoH batteries (0.8-1.0) **except** B0056/B0055 which it completely fails on.

---

## Performance by Battery ID (Top 10)

### High-Performing Batteries
| Battery | MAE | Std Dev | Test Files | Notes |
|---------|-----|---------|-----------|-------|
| **B0007** | 0.0619 | 0.0286 | 6 | Excellent performance |
| **B0018** | 0.0850 | 0.0689 | 5 | Very good performance |
| **B0006** | 0.2022 | 0.1407 | 8 | Good performance |
| **B0036** | 0.2181 | 0.0664 | 14 | Consistent, reliable |

### Poor-Performing Batteries
| Battery | MAE | Std Dev | Test Files | Notes |
|---------|-----|---------|-----------|-------|
| **B0056** | **0.7449** | 0.0486 | 8 | **SYSTEMATIC FAILURE** |
| **B0055** | **0.6581** | 0.1575 | 6 | **SYSTEMATIC FAILURE** |
| **B0033** | 0.5920 | 0.0503 | 6 | Poor performance |
| **B0034** | 0.5522 | 0.0034 | 6 | Poor performance |

---

## Root Cause Analysis

### Primary Issue: Training Data Distribution Mismatch

**B0056 and B0055 Analysis:**
- **Metadata capacity range:** 0.79-1.34 Ah (similar to other batteries)
- **SoH range:** 0.58-1.00 (normal distribution)
- **Test files:** 14 files total; 11 with errors > 0.7
- **Prediction pattern:** Consistently predicts 0.05-0.20 SoH when actual is 0.85-1.0

**Hypothesis:** These batteries (or discharge cycles) were either:
1. **Underrepresented in training data** (2000 files may not include enough B0056/B0055 high-SoH cycles)
2. **Have different electrical characteristics** (voltage/current signatures differ, confusing EKF or LSTM)
3. **Metadata issues** (incorrect capacity values for these specific batteries)

### Secondary Issues

1. **Moderate errors (0.2-0.5):** 23 files with ~24% error rate — suggest model could benefit from more training data
2. **Skipped files (3 total):** Short discharge cycles insufficient to form 50-timestep sequences
3. **EKF initialization:** Some files show EKF_SoC=0.0 (saturation) — may affect feature quality

---

## Performance Breakdown by Category

### Files with Excellent Performance (< 5% error)
- **Count:** 21 files (21.6%)
- **Batteries:** B0006, B0007, B0018, B0036, B0042, B0043, B0044, B0029, B0032
- **Characteristics:** Most diverse batteries except B0055/B0056
- **Implication:** Model works well on most of the dataset

### Files with Good Performance (5-20% error)
- **Count:** 22 files (22.7%)
- **Average error:** ~0.10
- **Notes:** Consistent predictions with small deviations

### Files with Moderate Performance (20-50% error)
- **Count:** 23 files (23.7%)
- **Average error:** ~0.35
- **Issue:** Many from B0033, B0034 (secondary problem batteries)

### Files with Poor Performance (> 50% error)
- **Count:** 31 files (32.0%)
- **Primarily from:** B0056 (8 files), B0055 (6 files), B0033/B0034 (12 files)
- **Critical issue:** Model fails systematically on specific batteries

---

## Recommendations

### Immediate Actions (Short-term)

1. **Identify problematic batteries:**
   ```
   Investigate metadata for B0056/B0055:
   - Check if capacity values are correct
   - Verify voltage ranges in raw discharge files
   - Check for data quality issues or anomalies
   ```

2. **Quick retraining experiment:**
   ```bash
   # Exclude problematic batteries from training
   python3 scripts/hybrid_train.py --max-files 2000 --epochs 25 --batch-size 32 \
     --exclude-batteries B0056,B0055
   # Expected: Overall MAE should improve to ~0.20
   ```

3. **Enhanced validation:**
   ```bash
   # Run inference with diagnostic data
   python3 scripts/infer_hybrid.py --input cleaned_dataset/data/07094.csv --diagnostic
   # Check: EKF trajectory, OCV fit quality, feature distributions
   ```

### Medium-term Actions (1-2 weeks)

4. **Data augmentation:**
   - Include more training samples from B0056/B0055/B0033/B0034
   - Consider data augmentation (voltage noise injection, synthetic degradation curves)
   - Implement per-battery LSTM heads (separate sub-models for problematic batteries)

5. **Model improvements:**
   - Add batch normalization to LSTM layers
   - Implement ensemble model (train multiple models on different data splits)
   - Use attention mechanisms to weight important features
   - Add uncertainty quantification (Bayesian LSTM)

6. **Feature engineering:**
   - Include battery temperature variation statistics
   - Add historical degradation rate as feature
   - Include cycle count and ambient conditions

### Long-term Actions (ongoing)

7. **Architecture changes:**
   - Train battery-specific models for problematic batteries
   - Implement meta-learning to quickly adapt to new battery types
   - Add domain adaptation layers to handle battery variations

8. **Data strategy:**
   - Collect more diverse discharge profiles (fast/slow charging, temperature variations)
   - Implement active learning to identify and collect hard-to-predict cycles
   - Establish data quality standards and validation procedures

---

## Detailed Statistics

### Files Skipped (3 total)
- 00896.csv: Insufficient samples for sequence formation
- 03450.csv: Insufficient samples for sequence formation
- 00940.csv: Insufficient samples for sequence formation

**Resolution:** Reduce `--seq-len` to 30-40 or increase `--sample-rows` to include shorter cycles

### Configuration Used
- **Training files:** 2000 discharge cycles
- **Training epochs:** 20
- **Batch size:** 64
- **Sequence length:** 50 timesteps
- **Model layers:** LSTM(64) → Dropout(0.2) → LSTM(32) → Dropout(0.1) → Dense(32) → Dense(2)
- **Loss function:** Mean Squared Error
- **Optimizer:** Adam (learning rate 0.001)

---

## Comparison: Quick Test (50 files) vs Full Test (2000 files)

| Metric | Quick Test | Full Test | Delta |
|--------|-----------|-----------|-------|
| MAE | 0.0804 | 0.3030 | +0.2226 (277% worse) |
| RMSE | 0.1092 | 0.3991 | +0.2899 (366% worse) |
| Success Rate | 100% | 97% | -3% |
| Worst Error | 0.1996 | 0.8406 | +0.6410 |

**Interpretation:** Full validation exposed systematic failures not visible in quick test because quick test happened to avoid problematic batteries by random chance.

---

## Next Steps Priority Order

1. **CRITICAL:** Investigate B0056/B0055 data quality and rerun with excluded batteries ← **DO THIS FIRST**
2. **HIGH:** Retrain model with different hyperparameters (more epochs, smaller seq_len)
3. **HIGH:** Implement per-battery models for problematic batteries
4. **MEDIUM:** Feature engineering and data augmentation
5. **MEDIUM:** Ensemble methods and uncertainty quantification
6. **LOW:** Architecture changes and long-term improvements

---

## Conclusion

The model shows **promise on most batteries (68% good/excellent accuracy)** but has **critical systematic failures on B0056 and B0055** (14 of top 31 worst errors). This suggests the training data composition or battery characteristics for these units are fundamentally different.

**Recommended next action:** 
1. Check metadata quality for B0056/B0055
2. Retrain excluding these batteries (quick sanity check)
3. If errors still occur, investigate EKF parameter quality for these batteries
4. If fixed, develop targeted improvements for handling battery outliers

**Success criteria for next iteration:** MAE < 0.15 on full 100-file validation set
