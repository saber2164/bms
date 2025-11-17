#!/usr/bin/env python3
"""
Inference utility for the hybrid EKF+LSTM model.

Usage examples:
  # predict for a single CSV using the default saved model
  python scripts/infer_hybrid.py --input cleaned_dataset/data/00001.csv

  # specify model and EKF params
  python scripts/infer_hybrid.py --input cleaned_dataset/data/00001.csv \
      --model outputs/eda/hybrid_lstm_model.keras --ekf-params '{"C_nom":2.3, "R0":0.05}'

What the script does:
 - Loads a trained hybrid LSTM model (default: outputs/eda/hybrid_lstm_model.keras)
 - Runs the EKF (`scripts/ekf_soc.EKF_SoC_Estimator`) over the input CSV to produce a SoC time-series
 - Builds sliding windows of features [Voltage, Current, Temperature, EKF_SoC] with `--seq-len`
 - Runs the LSTM model to predict SoH and scaled RUL for each window
 - Writes an output CSV alongside the input with columns: time, Voltage, Current, Temperature, EKF_SoC, pred_SoH, pred_RUL_scaled
 - Prints a short summary: final SoC, median/last SoH and RUL predictions.

Required columns in the input CSV (case-sensitive):
 - Voltage_measured (V)
 - Current_measured (A)
 - Temperature_measured (degC) -- optional but recommended (we include it as a feature)
 - Time (seconds) -- optional but preserved if present

Notes / assumptions:
 - EKF requires a nominal capacity `C_nom` (Ah). You can provide per-file EKF params via `--ekf-params` JSON.
 - The model returns SoH in range ~0..1 and RUL as a scaled fraction (same scaling used during training). If you want absolute cycles, provide `--rul-max` used at training time.

"""

import os
import sys
import json
import argparse
from typing import Optional

# ensure repo root is on sys.path so local imports resolve when running the script directly
REPO_ROOT = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

import numpy as np
import pandas as pd

from scripts.ekf_soc import EKF_SoC_Estimator, fit_ocv_poly
from scripts.metadata_loader import MetadataLoader, get_ekf_params_for_file

try:
    import tensorflow as tf
    from tensorflow.keras.models import load_model
except Exception:
    tf = None
    load_model = None


def check_required_columns(df: pd.DataFrame):
    req = ["Voltage_measured", "Current_measured"]
    for c in req:
        if c not in df.columns:
            raise ValueError(f"Input CSV missing required column: {c}. Available columns: {list(df.columns)[:10]}")


def run_ekf_for_file(path: str, ekf_params: dict, sample_rows: Optional[int] = None,
                     init_soc: Optional[float] = None, diagnostic: bool = False,
                     diag_out: Optional[str] = None) -> np.ndarray:
    df = pd.read_csv(path, nrows=sample_rows)
    check_required_columns(df)
    
    # Drop rows with NaN in required columns to avoid EKF state corruption
    req_cols = ["Voltage_measured", "Current_measured"]
    df = df.dropna(subset=req_cols)
    
    if len(df) == 0:
        raise ValueError("No valid measurement rows after removing NaN values")

    # allow ekf_params to include ocv_coeffs (or be None)
    ekf = EKF_SoC_Estimator(**ekf_params)
    # allow caller to override initial SoC if provided
    if init_soc is not None:
        try:
            ekf.x[0] = float(np.clip(float(init_soc), 0.0, 1.0))
        except Exception:
            pass
    socs = []
    diag_rows = []
    for idx, row in df.iterrows():
        i_k = float(row.get("Current_measured", 0.0))

        prior_x = ekf.x.copy()
        ekf.predict(i_k)

        # predicted terminal voltage before measurement update
        try:
            v_pred = float(ekf._observation_function(ekf.x, i_k))
        except Exception:
            v_pred = np.nan

        v_meas = float(row.get("Voltage_measured", np.nan))
        innovation = float(v_meas - v_pred) if np.isfinite(v_meas) and np.isfinite(v_pred) else np.nan

        ekf.update(v_meas, i_k)
        post_x = ekf.x.copy()

        socs.append(float(post_x[0]))

        if diagnostic:
            diag_rows.append({
                "index": int(idx),
                "Voltage_measured": v_meas,
                "Current_measured": i_k,
                "prior_soc": float(prior_x[0]),
                "prior_up": float(prior_x[1]),
                "v_pred": v_pred,
                "innovation": innovation,
                "post_soc": float(post_x[0]),
                "post_up": float(post_x[1]),
            })

    # write diagnostic CSV if requested
    if diagnostic and diag_out is not None and len(diag_rows) > 0:
        try:
            diag_df = pd.DataFrame(diag_rows)
            os.makedirs(os.path.dirname(diag_out), exist_ok=True)
            diag_df.to_csv(diag_out, index=False)
        except Exception:
            pass

    return np.array(socs)


def create_sequences(features: np.ndarray, seq_len: int) -> np.ndarray:
    N, F = features.shape
    M = N - seq_len + 1
    if M <= 0:
        return np.zeros((0, seq_len, F))
    X = np.zeros((M, seq_len, F), dtype=float)
    for i in range(M):
        X[i] = features[i : i + seq_len]
    return X


def get_reference_soh_rul(input_filename: str, metadata_path: str) -> dict:
    """Load reference SoH and RUL from metadata for a discharge file.
    
    Returns dict with keys:
        - battery_id: battery identifier
        - capacity_actual: actual capacity from metadata
        - capacity_nominal: max capacity for this battery
        - soh_reference: capacity_actual / capacity_nominal
        - rul_reference: remaining cycles until SoH < 0.8
    
    Returns empty dict if file not found in metadata.
    """
    try:
        df = pd.read_csv(metadata_path)
        
        # Find this file in metadata
        file_row = df[df["filename"] == input_filename]
        if file_row.empty:
            return {}
        
        file_row = file_row.iloc[0]
        battery_id = file_row.get("battery_id")
        
        if pd.isna(battery_id):
            return {}
        
        # Get actual capacity
        cap_actual = pd.to_numeric(file_row.get("Capacity"), errors="coerce")
        if pd.isna(cap_actual):
            return {}
        
        # Get nominal (max) capacity for this battery
        battery_rows = df[df["battery_id"] == battery_id]
        discharge_rows = battery_rows[battery_rows["type"] == "discharge"].copy()
        discharge_rows["Capacity"] = pd.to_numeric(discharge_rows["Capacity"], errors="coerce")
        discharge_rows = discharge_rows.dropna(subset=["Capacity"])
        
        if discharge_rows.empty:
            return {}
        
        cap_nominal = float(discharge_rows["Capacity"].max())
        soh_ref = float(cap_actual / cap_nominal) if cap_nominal > 0 else 1.0
        
        # Compute RUL: cycles remaining until SoH < 0.8
        discharge_rows = discharge_rows.sort_values("test_id")
        caps = discharge_rows["Capacity"].values
        soh_values = caps / cap_nominal
        current_idx = discharge_rows[discharge_rows["filename"] == input_filename].index[0]
        current_pos = list(discharge_rows.index).index(current_idx)
        
        future_low_soh = np.where(soh_values[current_pos + 1:] < 0.8)[0]
        if len(future_low_soh) > 0:
            rul_ref = int(future_low_soh[0] + 1)
        else:
            rul_ref = int(len(discharge_rows) - current_pos - 1)
        
        return {
            "battery_id": str(battery_id),
            "capacity_actual": float(cap_actual),
            "capacity_nominal": float(cap_nominal),
            "soh_reference": soh_ref,
            "rul_reference": rul_ref,
        }
    
    except Exception as e:
        # Silently fail and return empty dict
        return {}


def infer_single_file(input_csv: str, model_path: str, ekf_params: dict, seq_len: int,
                      sample_rows: Optional[int], rul_max: Optional[float], out_csv: Optional[str],
                      init_soc: Optional[float] = None, diagnostic: bool = False, diag_out: Optional[str] = None):
    df = pd.read_csv(input_csv, nrows=sample_rows)
    check_required_columns(df)
    
    # Drop rows with NaN in required columns early to ensure consistency with EKF
    req_cols = ["Voltage_measured", "Current_measured"]
    df = df.dropna(subset=req_cols)
    if len(df) == 0:
        raise ValueError("No valid measurement rows after removing NaN values")
    
    # Data validation: check voltage range is reasonable for Li-ion cells (2.5–4.3 V typical)
    v_measured = df["Voltage_measured"]
    v_min, v_max = v_measured.min(), v_measured.max()
    data_quality_warning = None
    if v_min < 1.5 or v_max > 5.0:
        data_quality_warning = f"Voltage range [{v_min:.2f}, {v_max:.2f}] V may indicate corrupted or non-BMS data (expected ~2.5–4.3 V)."
    
    current_measured = df["Current_measured"]
    i_rms = np.sqrt((current_measured ** 2).mean())
    if i_rms < 0.001:
        data_quality_warning = (data_quality_warning or "") + f" RMS current {i_rms:.6f} A is very low (near-zero). Data may be corrupted."

    # run EKF (optionally with provided initial SoC). If diagnostic is set, also write EKF diag CSV.
    soc_est = run_ekf_for_file(input_csv, ekf_params, sample_rows=sample_rows, init_soc=init_soc,
                              diagnostic=diagnostic, diag_out=diag_out)

    # build feature matrix
    cols = ["Voltage_measured", "Current_measured", "Temperature_measured"]
    feat_cols = []
    for c in cols:
        if c in df.columns:
            feat_cols.append(pd.to_numeric(df[c], errors="coerce").fillna(0.0).to_numpy(dtype=float))
        else:
            feat_cols.append(np.zeros(len(df)))
    feat_cols.append(soc_est)
    features = np.vstack(feat_cols).T  # (N, 4)

    # sequences
    X = create_sequences(features, seq_len)
    if X.shape[0] == 0:
        raise RuntimeError("Not enough samples to form a single sequence. Increase seq_len or provide longer data.")

    # load model
    if load_model is None:
        raise RuntimeError("TensorFlow/Keras not available in environment")
    model = load_model(model_path)

    preds = model.predict(X, verbose=0)
    # preds: (n_windows, 2) -> [SoH, RUL_scaled]

    # map predictions back to time alignment: we associate each window's prediction with the window's end index
    n_windows = preds.shape[0]
    indices = np.arange(seq_len - 1, seq_len - 1 + n_windows)

    # create output DF with per-row data; fill pred columns only for rows that have a window ending there
    out_df = df.copy()
    out_df = out_df.reset_index(drop=True)
    out_df["EKF_SoC"] = soc_est
    out_df["pred_SoH"] = np.nan
    out_df["pred_RUL_scaled"] = np.nan

    for i, idx in enumerate(indices):
        out_df.at[idx, "pred_SoH"] = float(preds[i, 0])
        out_df.at[idx, "pred_RUL_scaled"] = float(preds[i, 1])

    # aggregate summary: last available window prediction (most recent)
    last_pred = preds[-1]
    last_soh = float(last_pred[0])
    last_rul_scaled = float(last_pred[1])
    last_rul_abs = None
    if rul_max is not None:
        last_rul_abs = last_rul_scaled * float(rul_max)

    # Detect EKF saturation issues (SoC stuck at 0 or 1)
    valid_socs = soc_est[~np.isnan(soc_est)]
    saturation_fraction = 0.0
    saturation_warning = None
    if len(valid_socs) > 0:
        saturation_count = np.sum((valid_socs == 0.0) | (valid_socs == 1.0))
        saturation_fraction = saturation_count / len(valid_socs)
        if saturation_fraction > 0.5:
            saturation_warning = f"EKF_SoC appears saturated (stuck at 0 or 1): {saturation_fraction*100:.1f}% of valid samples. Consider using --auto-init-ocv or adjusting EKF params."

    # write out CSV
    if out_csv is None:
        base = os.path.splitext(os.path.basename(input_csv))[0]
        out_csv = os.path.join("outputs", "eda", f"{base}.inference.csv")
    os.makedirs(os.path.dirname(out_csv), exist_ok=True)
    out_df.to_csv(out_csv, index=False)

    summary = {
        "input": input_csv,
        "out_csv": out_csv,
        "last_ekf_soc": float(soc_est[-1]) if len(soc_est) > 0 else None,
        "last_pred_soh": last_soh,
        "last_pred_rul_scaled": last_rul_scaled,
        "last_pred_rul_abs": last_rul_abs,
        "n_windows": int(n_windows),
        "data_quality_warning": data_quality_warning,
        "ekf_saturation_warning": saturation_warning,
    }
    return summary


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--input", required=True, help="Input CSV file path")
    p.add_argument("--model", default=os.path.join("outputs", "eda", "hybrid_lstm_model.keras"),
                   help="Trained hybrid model path (Keras .keras).")
    p.add_argument("--seq-len", type=int, default=50)
    p.add_argument("--sample-rows", type=int, default=None)
    p.add_argument("--ekf-params", type=str, default=None,
                   help='JSON string of EKF params (dt, C_nom, R0, R_D, C_D, eta). Example: "{\"C_nom\":2.3}"')
    p.add_argument("--init-soc", type=float, default=None,
                   help="Optional initial SoC to seed the EKF (0..1). If omitted and auto-init is enabled, the script will try to estimate SoC from OCV.")
    p.add_argument("--auto-init-ocv", action="store_true",
                   help="(Deprecated) explicit flag to run OCV-based init. Prefer leaving auto-init enabled by default.")
    p.add_argument("--no-auto-init", action="store_true",
                   help="Disable automatic OCV fitting/init when ocv_coeffs aren't provided in ekf params.")
    p.add_argument("--diagnostic", action="store_true",
                   help="Write per-step EKF diagnostics (prior/post SoC, predicted V, innovation) to a companion CSV")
    p.add_argument("--ocv-degree", type=int, default=3, help="Polynomial degree to use when fitting OCV")
    p.add_argument("--ocv-current-threshold", type=float, default=0.01,
                   help="Absolute current threshold (A) to consider a point as low-current for OCV fitting/estimation")
    p.add_argument("--rul-max", type=float, default=None,
                   help="If provided, this will be used to invert scaled RUL to absolute cycles")
    p.add_argument("--out", type=str, default=None, help="Optional output CSV path")
    args = p.parse_args()

    if args.ekf_params:
        ekf_params = json.loads(args.ekf_params)
    else:
        # Try to load from metadata first, fall back to defaults
        try:
            input_filename = os.path.basename(args.input)
            metadata_path = os.path.join(REPO_ROOT, "cleaned_dataset", "metadata.csv")
            ekf_params = get_ekf_params_for_file(input_filename, metadata_path=metadata_path)
        except Exception:
            # If metadata loading fails, use reasonable defaults
            ekf_params = dict(dt=1.0, C_nom=2.3, R0=0.05, R_D=0.01, C_D=500.0, eta=0.99)

    # If user requested auto OCV initialization (or ocv_coeffs missing and auto is enabled),
    # fit OCV on this file and then estimate an initial SoC if --init-soc was not provided.
    init_soc = args.init_soc
    do_auto_init = args.auto_init_ocv or (not args.no_auto_init and "ocv_coeffs" not in ekf_params)
    if do_auto_init:
        try:
            # fit polynomial using a pool of dataset files (more robust than single-file fit)
            data_dir = os.path.join(REPO_ROOT, "cleaned_dataset", "data")
            files = []
            if os.path.isdir(data_dir):
                import glob
                all_files = sorted(glob.glob(os.path.join(data_dir, "*.csv")))
                # limit number of files to keep fit time reasonable
                files = all_files[:100]
            if len(files) == 0:
                files = [args.input]

            coeffs = fit_ocv_poly(files, C_nom=float(ekf_params.get("C_nom", 2.3)),
                                  degree=int(args.ocv_degree), current_threshold=float(args.ocv_current_threshold))
            # attach OCV curve to EKF params so EKF uses it during updates
            ekf_params["ocv_coeffs"] = coeffs.tolist() if hasattr(coeffs, "tolist") else list(coeffs)

            # persist fitted coefficients for later reuse (helpful for repeated inference)
            try:
                os.makedirs(os.path.join("outputs", "eda"), exist_ok=True)
                np.save(os.path.join("outputs", "eda", "ocv_coeffs.npy"), np.array(coeffs))
            except Exception:
                pass

            # estimate SoC by inverting OCV at the first low-current measured voltage
            if init_soc is None:
                df_tmp = pd.read_csv(args.input, nrows=2000)
                # find low-current row
                if "Current_measured" in df_tmp.columns:
                    cur = pd.to_numeric(df_tmp["Current_measured"], errors="coerce").fillna(0.0)
                    mask = cur.abs() <= float(args.ocv_current_threshold)
                    if mask.any():
                        idx = int(mask.idxmax())
                        v0 = float(pd.to_numeric(df_tmp.loc[idx, "Voltage_measured"]))
                    else:
                        v0 = float(pd.to_numeric(df_tmp.loc[0, "Voltage_measured"]))
                else:
                    v0 = float(pd.to_numeric(df_tmp.loc[0, "Voltage_measured"]))

                # invert OCV via bisection between 0..1
                def estimate_soc_from_ocv(coeffs_arr, v_target, iters=50):
                    # coeffs_arr: highest-degree first
                    def ocv_at(s):
                        return float(np.polyval(coeffs_arr, float(s)))

                    lo, hi = 0.0, 1.0
                    vlo, vhi = ocv_at(lo), ocv_at(hi)
                    # if not bracketed, return midpoint as fallback
                    if not (min(vlo, vhi) <= v_target <= max(vlo, vhi)):
                        return 0.9
                    for _ in range(iters):
                        mid = 0.5 * (lo + hi)
                        vm = ocv_at(mid)
                        if vm == v_target:
                            return mid
                        # choose side that contains the target
                        if (vm < v_target and vlo < vhi) or (vm > v_target and vlo > vhi):
                            lo = mid
                        else:
                            hi = mid
                    return 0.5 * (lo + hi)

                try:
                    init_soc = float(estimate_soc_from_ocv(coeffs, v0))
                except Exception:
                    init_soc = None
        except Exception:
            # if ocv fit fails, continue with defaults / user-provided init_soc
            pass

    # prepare diagnostic output path if requested
    diag_out = None
    if args.diagnostic:
        base = os.path.splitext(os.path.basename(args.input))[0]
        diag_out = os.path.join("outputs", "eda", f"{base}.ekf_diag.csv")

    # Get reference SoH/RUL from metadata if available
    metadata_path = os.path.join(REPO_ROOT, "cleaned_dataset", "metadata.csv")
    reference_data = {}
    input_filename = os.path.basename(args.input)
    try:
        reference_data = get_reference_soh_rul(input_filename, metadata_path)
    except Exception:
        pass

    summary = infer_single_file(args.input, args.model, ekf_params, args.seq_len, args.sample_rows, args.rul_max, args.out, init_soc=init_soc, diagnostic=args.diagnostic, diag_out=diag_out)
    
    # Add reference data to summary
    if reference_data:
        summary["reference"] = reference_data
    
    print("Inference summary:")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
