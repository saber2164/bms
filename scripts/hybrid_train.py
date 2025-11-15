#!/usr/bin/env python3
"""
Hybrid EKF+LSTM training pipeline

This script builds a hybrid model that combines an EKF-based SoC estimator
as an input feature and an LSTM that predicts SoH and RUL from short
time-series sequences.

Usage (recommended):
  # dry/demo run on first 50 discharge files (fast)
  python scripts/hybrid_train.py --max-files 50 --seq-len 50 --epochs 3 --demo

Production-style training (may be slow on CPU):
  python scripts/hybrid_train.py --seq-len 50 --epochs 20

Important design notes:
- SoH label: per-discharge Capacity divided by the battery's nominal capacity
  (we take nominal = max Capacity observed for that battery in metadata).
- RUL label: remaining number of discharge cycles until SoH falls below 0.8
  (computed per-battery from chronological discharge records). If never
  reaches threshold, RUL is set to the number of remaining cycles.
- EKF: we run an EKF (from `scripts/ekf_soc.py`) over each timeseries to
  produce an estimated SoC time-series used as an input feature to the LSTM.

This file is CPU-friendly: defaults limit data and training for quick runs.
"""

import argparse
import os
import json
from typing import List, Tuple

import numpy as np
import pandas as pd
import shutil

from sklearn.model_selection import train_test_split

# Ensure repository root is on sys.path so `from scripts...` works when running the
# script directly. This fixes ModuleNotFoundError for local imports.
import sys
REPO_ROOT = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

# Import EKF class and OCV fitter from local script
from scripts.ekf_soc import EKF_SoC_Estimator, fit_ocv_poly
from scripts.metadata_loader import MetadataLoader, get_ekf_params_for_file

try:
    import tensorflow as tf
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import LSTM, Dense, Dropout
    from tensorflow.keras.optimizers import Adam
    from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
except Exception:
    tf = None

ROOT = os.path.dirname(os.path.dirname(__file__))
DATA_DIR = os.path.join(ROOT, "cleaned_dataset", "data")
METADATA = os.path.join(ROOT, "cleaned_dataset", "metadata.csv")
OUT_DIR = os.path.join(ROOT, "outputs", "eda")
os.makedirs(OUT_DIR, exist_ok=True)


def load_metadata() -> pd.DataFrame:
    return pd.read_csv(METADATA)


def compute_nominal_capacity(meta: pd.DataFrame) -> dict:
    """Compute C_nom per battery: max Capacity observed for that battery."""
    d = {}
    for bid, g in meta.groupby("battery_id"):
        caps = pd.to_numeric(g["Capacity"], errors="coerce").dropna()
        if len(caps) > 0:
            d[bid] = float(caps.max())
    return d


def discharge_records(meta: pd.DataFrame) -> pd.DataFrame:
    # return only discharge rows that have capacity values
    df = meta[meta["type"] == "discharge"].copy()
    df["Capacity"] = pd.to_numeric(df["Capacity"], errors="coerce")
    df = df[df["Capacity"].notna()]
    # ensure test ordering for RUL; use test_id if present else index
    if "test_id" in df.columns:
        df = df.sort_values(["battery_id", "test_id"])  # chronological per battery
    return df


def compute_rul_per_battery(dis_df: pd.DataFrame, threshold=0.8) -> dict:
    """For each discharge row, compute RUL in cycles until SoH < threshold.

    Returns mapping filename -> rul (int)
    """
    rul_map = {}
    for bid, g in dis_df.groupby("battery_id"):
        g = g.sort_values("test_id").reset_index(drop=True)
        caps = g["Capacity"].astype(float).to_numpy()
        if np.nanmax(caps) <= 0:
            # skip invalid
            for fn in g["filename"]:
                rul_map[fn] = np.nan
            continue
        Cnom = np.nanmax(caps)
        soh = caps / Cnom
        n = len(soh)
        for idx in range(n):
            # find first future index where soh < threshold
            future = np.where(soh[idx + 1 :] < threshold)[0]
            if len(future) == 0:
                rul = n - 1 - idx  # remaining cycles
            else:
                rul = int(future[0] + 1)
            rul_map[g.loc[idx, "filename"]] = rul
    return rul_map


def run_ekf_on_file(path: str, ekf_params: dict, sample_rows: int = None) -> np.ndarray:
    """Run EKF on time-series file and return SoC estimate array aligned to rows read."""
    df = pd.read_csv(path, nrows=sample_rows)
    # ensure columns exist
    required = ["Voltage_measured", "Current_measured", "Temperature_measured"]
    for c in required:
        if c not in df.columns:
            raise ValueError(f"Missing column {c} in {path}")

    ekf = EKF_SoC_Estimator(**ekf_params)
    socs = []
    for i, row in df.iterrows():
        i_k = float(row.get("Current_measured", 0.0))
        ekf.predict(i_k)
        # create synthetic measurement from data
        v_meas = float(row.get("Voltage_measured", np.nan))
        ekf.update(v_meas, i_k)
        socs.append(float(ekf.x[0]))
    return np.array(socs)


def build_lstm(input_shape: Tuple[int, int]):
    if tf is None:
        raise ImportError("TensorFlow not installed in environment")
    model = Sequential()
    model.add(LSTM(64, return_sequences=True, input_shape=input_shape))
    model.add(Dropout(0.2))
    model.add(LSTM(32, return_sequences=False))
    model.add(Dropout(0.1))
    model.add(Dense(32, activation="relu"))
    # output two values: SoH (0..1) and RUL (cycles). We'll scale RUL during training.
    model.add(Dense(2, activation="linear"))
    model.compile(optimizer=Adam(1e-3), loss="mse")
    return model


def create_sequences_from_series(features: np.ndarray, seq_len: int) -> np.ndarray:
    N, F = features.shape
    M = N - seq_len + 1
    if M <= 0:
        return np.zeros((0, seq_len, F))
    X = np.zeros((M, seq_len, F), dtype=float)
    for i in range(M):
        X[i] = features[i : i + seq_len]
    return X


def prepare_dataset(dis_df: pd.DataFrame, max_files: int = None, seq_len: int = 50, sample_rows: int = 2000,
                    ekf_defaults: dict = None, metadata_loader: MetadataLoader = None) -> Tuple[np.ndarray, np.ndarray]:
    """Prepare X, y_soh, y_rul arrays from discharge records using metadata for per-file parameters.

    X shape: (samples, seq_len, n_features)
    y shape: (samples, 2) columns [SoH, RUL_scaled]
    """
    if ekf_defaults is None:
        ekf_defaults = dict(dt=1.0, C_nom=2.3, R0=0.05, R_D=0.01, C_D=500.0, eta=0.99)

    # compute nominal C per battery
    Cnom_map = compute_nominal_capacity(pd.read_csv(METADATA))
    rul_map = compute_rul_per_battery(dis_df)

    # map battery -> list of filenames for OCV fitting
    battery_files_map = dis_df.groupby("battery_id")["filename"].apply(list).to_dict()
    battery_ocv_cache: dict = {}

    X_list = []
    y_list = []
    files = dis_df["filename"].tolist()
    if max_files:
        files = files[:max_files]

    for fn in files:
        path = os.path.join(DATA_DIR, fn)
        if not os.path.exists(path):
            continue
        # select ekf params per battery and file
        row = dis_df[dis_df["filename"] == fn].iloc[0]
        bid = row["battery_id"]
        Cnom = Cnom_map.get(bid, ekf_defaults.get("C_nom", 2.3))
        
        # Try to get metadata-based parameters if loader is available
        ekf_params = ekf_defaults.copy()
        ekf_params["C_nom"] = float(Cnom)
        
        if metadata_loader is not None:
            try:
                meta_params = metadata_loader.get_ekf_params(fn)
                ekf_params.update(meta_params)
            except Exception:
                pass  # fall back to defaults

        # Fit or reuse empirical OCV for this battery (fast cached fit across its files)
        if bid not in battery_ocv_cache:
            fns = battery_files_map.get(bid, [])
            full_paths = [os.path.join(DATA_DIR, x) for x in fns if os.path.exists(os.path.join(DATA_DIR, x))]
            if len(full_paths) > 0:
                try:
                    coeffs = fit_ocv_poly(full_paths, Cnom, degree=3)
                    battery_ocv_cache[bid] = coeffs
                except Exception:
                    battery_ocv_cache[bid] = None
            else:
                battery_ocv_cache[bid] = None

        if battery_ocv_cache.get(bid) is not None:
            ekf_params["ocv_coeffs"] = battery_ocv_cache[bid]

        try:
            soc_est = run_ekf_on_file(path, ekf_params, sample_rows=sample_rows)
        except Exception as e:
            print(f"EKF failed for {fn}: {e}")
            continue

        df = pd.read_csv(path, nrows=len(soc_est))
        # build features: [V, I, T, ekf_soc]
        feat_cols = []
        for c in ["Voltage_measured", "Current_measured", "Temperature_measured"]:
            if c in df.columns:
                feat_cols.append(df[c].to_numpy(dtype=float))
            else:
                feat_cols.append(np.zeros(len(df)))
        feat_cols.append(soc_est)
        features = np.vstack(feat_cols).T  # (N,4)

        # sliding windows
        Xf = create_sequences_from_series(features, seq_len)
        if Xf.size == 0:
            continue

        # labels: SoH and RUL for this file
        cap = float(row["Capacity"])
        soh = cap / Cnom if Cnom > 0 else 0.0
        rul = float(rul_map.get(fn, 0))

        # scale RUL by an upper bound for network stability (max cycles per battery)
        max_rul = dis_df[dis_df["battery_id"] == bid].shape[0]
        rul_scaled = rul / max(1.0, max_rul)

        n_samples = Xf.shape[0]
        X_list.append(Xf)
        y_list.append(np.tile([soh, rul_scaled], (n_samples, 1)))

    if len(X_list) == 0:
        return np.zeros((0, seq_len, 4)), np.zeros((0, 2))

    X = np.vstack(X_list)
    y = np.vstack(y_list)
    return X, y


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-files", type=int, default=None, help="Limit number of discharge files to use (fast debug)")
    parser.add_argument("--seq-len", type=int, default=50)
    parser.add_argument("--sample-rows", type=int, default=2000)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--demo", action="store_true", help="Run a quick demo (overrides epochs to 3)")
    args = parser.parse_args()

    if args.demo:
        args.epochs = min(3, args.epochs)

    meta = load_metadata()
    dis_df = discharge_records(meta)
    print(f"Found {len(dis_df)} discharge records with capacity labels")

    # Load metadata loader for per-file parameter extraction
    try:
        metadata_loader = MetadataLoader(METADATA)
        print(f"MetadataLoader initialized with {len(metadata_loader.filename_index)} files")
    except Exception as e:
        print(f"Warning: Could not initialize MetadataLoader: {e}")
        metadata_loader = None

    if args.max_files is None:
        # default: limit to 500 files to keep CPU time reasonable unless user requested full run
        args.max_files = min(500, len(dis_df))

    print(f"Preparing dataset from up to {args.max_files} files (seq_len={args.seq_len})")

    X, y = prepare_dataset(dis_df, max_files=args.max_files, seq_len=args.seq_len, sample_rows=args.sample_rows,
                          metadata_loader=metadata_loader)
    if X.shape[0] == 0:
        print("No training samples generated. Exiting.")
        return

    # train/test split by sample (files were limited already)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15, random_state=42)
    print("Training samples:", X_train.shape, "Test samples:", X_test.shape)

    if tf is None:
        print("TensorFlow not available in environment. Install tensorflow to train the model.")
        return

    model = build_lstm((args.seq_len, X.shape[2]))
    model.summary()

    # train with sensible callbacks: EarlyStopping + ModelCheckpoint (best on val_loss)
    checkpoint_path = os.path.join(OUT_DIR, "hybrid_lstm_best.keras")
    callbacks = [
        EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True),
        ModelCheckpoint(checkpoint_path, monitor="val_loss", save_best_only=True, save_weights_only=False)
    ]

    model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=args.epochs,
              batch_size=args.batch_size, callbacks=callbacks)

    # evaluate
    preds = model.predict(X_test)
    # unscale RUL (we scaled by per-file max_rul; here we kept as fraction so report relative errors)
    soh_rmse = np.sqrt(np.mean((preds[:, 0] - y_test[:, 0]) ** 2))
    rul_mae = np.mean(np.abs(preds[:, 1] - y_test[:, 1]))
    metrics = {"soh_rmse": float(soh_rmse), "rul_mae_scaled": float(rul_mae)}
    print("Metrics:", metrics)

    # save model and metrics
    model_path = os.path.join(OUT_DIR, "hybrid_lstm_model.keras")
    # if checkpointed best model exists, use it as the canonical saved model
    if os.path.exists(checkpoint_path):
        try:
            shutil.copy(checkpoint_path, model_path)
        except Exception:
            # fallback to saving current model
            model.save(model_path)
    else:
        model.save(model_path)
    with open(os.path.join(OUT_DIR, "hybrid_metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)

    print("Saved model to", model_path)


if __name__ == "__main__":
    main()
