# Rules & How to run

This file contains the minimal, opinionated rules and runnable commands to run EDA, inference and training in this repository.

## Purpose
- Provide a single reference for reproducible runs (inference, training, diagnostics).
- Describe the expected input CSV schema and the main CLI options.

## Prerequisites
- Python 3.10+ with the packages listed in `requirements.txt` installed in your active environment.
- Work from the repository root (project path). Example:

```bash
cd /home/harshit/Documents/bms
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Input CSV schema
- Required columns (case-sensitive): `Voltage_measured`, `Current_measured`.
- Optional but recommended: `Temperature_measured`, `Time`.
- Files are expected in `cleaned_dataset/data/*.csv`. Per-file metadata can be found in `cleaned_dataset/metadata.csv`.

## Inference (single CSV)
Runs the hybrid EKF+LSTM pipeline and writes per-row predictions to `outputs/eda/<base>.inference.csv`.

Basic usage (defaults):

```bash
python3 scripts/infer_hybrid.py --input cleaned_dataset/data/00007.csv
```

Useful flags:
- `--model`: Keras `.keras` model file (default `outputs/eda/hybrid_lstm_model.keras`).
- `--init-soc`: optional initial SoC (0..1) to seed EKF.
- `--no-auto-init`: disable automatic OCV fitting (by default the script will attempt to fit and reuse empirical OCV coefficients when missing).
- `--diagnostic`: write per-step EKF diagnostics to `outputs/eda/<base>.ekf_diag.csv`.
- `--out`: override output CSV path.

Example (auto OCV + diagnostics):

```bash
python3 scripts/infer_hybrid.py --input cleaned_dataset/data/00007.csv --diagnostic --out outputs/eda/00007.inference.csv
```

If the EKF SoC collapses to 0 or 1, run the script with `--diagnostic` to inspect `outputs/eda/<base>.ekf_diag.csv` and consider enabling auto-init (default) or supplying a better `--init-soc`.

## Training (hybrid LSTM)
There is a training pipeline in `scripts/hybrid_train.py` that:
- runs the EKF across training files to produce SoC features,
- constructs sliding windows, and
- trains a Keras LSTM model to predict SoH and scaled RUL.

Run a quick demo training (small subset):

```bash
python3 scripts/hybrid_train.py --demo
```

For a full training, see the script help. Models are saved to `outputs/eda/`.

## Diagnostics & Tips
- If EKF updates produce very large innovations (v_meas - v_pred) the empirical OCV fit may be missing; run with `--diagnostic` and enable auto-init.
- The inference script now persists fitted OCV coefficients to `outputs/eda/ocv_coeffs.npy` for reuse.
- The EKF has a per-update guard limiting SoC step size; if you need different behavior, edit `scripts/ekf_soc.py` parameter `max_soc_step` in the constructor.

## Where outputs are written
- Predictions and inference CSVs: `outputs/eda/*.inference.csv`
- EKF diagnostic CSVs: `outputs/eda/*.ekf_diag.csv`
- Fitted OCV coefficients: `outputs/eda/ocv_coeffs.npy`

## Contact
For deeper changes (new models, different sampling dt, or thermal integration), consult `WORKFLOW_TECHNICAL.md`.
