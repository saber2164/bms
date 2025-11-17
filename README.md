# Battery SOH & RUL Estimator (BMS)

This repository contains code, data utilities, and documentation for a hybrid battery State-of-Health (SoH) and Remaining Useful Life (RUL) estimator that combines an Extended Kalman Filter (EKF) SoC feature extractor with an LSTM-based predictor.

Contents
- `cleaned_dataset/` — per-discharge CSV files and `metadata.csv` describing EKF parameters and labels.
- `scripts/` — training, inference, EKF, and validation scripts (see `scripts/README.md` or inline `scripts` docstrings).
- `outputs/` — generated models, validation reports, and plots.
- `docs/` and `documentation/` — detailed reports, validation summaries, and quick references.

Key files
- `scripts/hybrid_train.py` — main training pipeline that uses EKF-derived SoC as an input feature and trains the LSTM SoH/RUL model.
- `scripts/infer_hybrid.py` — single-file inference using EKF + trained LSTM model.
- `scripts/ekf_soc.py` — EKF implementation and OCV fitting utilities.
- `scripts/metadata_loader.py` — utilities to load per-file EKF parameters from `cleaned_dataset/metadata.csv`.
- `outputs/eda/hybrid_lstm_model.keras` — canonical trained model (if present).

Quick start

1. Create a Python environment and install dependencies (see `requirements.txt`):

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

2. Run a short training run (example):

```bash
# train on a subset (set --max-files to limit); tune epochs/batch/seq-len as needed
python3 scripts/hybrid_train.py --max-files 200 --epochs 10 --batch-size 64 --seq-len 50
```

3. Run inference on a single discharge file:

```bash
python3 scripts/infer_hybrid.py --input-file cleaned_dataset/data/00001.csv --metadata cleaned_dataset/metadata.csv --model outputs/eda/hybrid_lstm_model.keras
```

4. Run the validation workflow (example uses 100 random files):

```bash
python3 scripts/test_metadata_based_model.py --n-samples 100 --out-dir outputs/eda
```

Repository structure

```
bms/
├─ cleaned_dataset/
│  ├─ metadata.csv
│  └─ data/ (per-discharge CSVs: 00001.csv ...)
├─ scripts/
│  ├─ hybrid_train.py
│  ├─ infer_hybrid.py
│  ├─ ekf_soc.py
│  └─ metadata_loader.py
├─ outputs/
│  └─ eda/ (models, predictions, reports)
├─ documentation/
└─ README.md
```

Model & outputs
- Trained models are saved under `outputs/eda/` by default (file `hybrid_lstm_model.keras`).
- Validation reports and per-file predictions are saved into `outputs/eda/` during validation runs.

Contributing and development notes
- Use `scripts/` utilities to reproduce training and validation. The code is organized for research workflows; small edits and experiments are expected.
- Tests and validation harnesses are in `scripts/` and `documentation/`.


Or use the GitHub CLI (if installed and authenticated):

```bash
# interactive create + push
gh repo create bms-soh-estimator --public --source=. --remote=origin --push
```

License
- This project is released under the MIT License. See `LICENSE` for details.

Where to go next
- Re-run training with full dataset: increase `--max-files` and `--epochs` in `scripts/hybrid_train.py`.
- Investigate the validation report in `outputs/eda/test_metadata_based_model_report.json` and `documentation/COMPREHENSIVE_VALIDATION_REPORT.md`.

