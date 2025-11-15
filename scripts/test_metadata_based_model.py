#!/usr/bin/env python3
"""
Test script for metadata-based LSTM SoH/RUL model with random discharge files.

This script:
1. Randomly selects discharge files from cleaned_dataset/data
2. Trains a small hybrid model on a subset of files (optional)
3. Runs inference on random test files
4. Compares predicted vs reference SoH/RUL from metadata
5. Generates a comprehensive test report

Usage:
  python scripts/test_metadata_based_model.py --max-train 100 --num-test 10 --quick
  python scripts/test_metadata_based_model.py --max-train 500 --num-test 50
"""

import os
import sys
import json
import argparse
import numpy as np
import pandas as pd
from typing import List, Tuple, Dict
import random

# Ensure repo root on path
REPO_ROOT = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from scripts.metadata_loader import MetadataLoader, get_ekf_params_for_file
from scripts.hybrid_train import (
    load_metadata, discharge_records, compute_nominal_capacity, 
    compute_rul_per_battery, prepare_dataset, build_lstm, run_ekf_on_file
)
from scripts.infer_hybrid import get_reference_soh_rul, infer_single_file

# Try to import TensorFlow
try:
    import tensorflow as tf
    from tensorflow.keras.models import load_model
except Exception:
    tf = None
    load_model = None


def get_random_discharge_files(max_count: int = None) -> List[str]:
    """Get list of discharge files from metadata."""
    metadata_path = os.path.join(REPO_ROOT, "cleaned_dataset", "metadata.csv")
    meta = pd.read_csv(metadata_path)
    
    dis_df = meta[meta["type"] == "discharge"].copy()
    dis_df["Capacity"] = pd.to_numeric(dis_df["Capacity"], errors="coerce")
    dis_df = dis_df[dis_df["Capacity"].notna()]
    
    files = dis_df["filename"].unique().tolist()
    
    if max_count and len(files) > max_count:
        files = random.sample(files, max_count)
    
    return files


def train_test_model(train_files: int = 100, epochs: int = 3, seq_len: int = 50,
                    batch_size: int = 32, sample_rows: int = 2000) -> Tuple[str, Dict]:
    """Train a small hybrid model on a subset of files."""
    print("\n" + "="*100)
    print("TRAINING HYBRID MODEL")
    print("="*100)
    
    if tf is None:
        print("TensorFlow not available. Skipping model training.")
        return None, {}
    
    # Load metadata and discharge records
    meta = load_metadata()
    dis_df = discharge_records(meta)
    
    print(f"Total discharge records: {len(dis_df)}")
    print(f"Using {train_files} files for training")
    
    # Initialize metadata loader
    metadata_path = os.path.join(REPO_ROOT, "cleaned_dataset", "metadata.csv")
    try:
        metadata_loader = MetadataLoader(metadata_path)
    except Exception as e:
        print(f"Warning: Could not initialize MetadataLoader: {e}")
        metadata_loader = None
    
    # Prepare dataset
    X, y = prepare_dataset(dis_df, max_files=train_files, seq_len=seq_len, 
                          sample_rows=sample_rows, metadata_loader=metadata_loader)
    
    if X.shape[0] == 0:
        print("ERROR: No training samples generated")
        return None, {}
    
    print(f"Training data: X shape {X.shape}, y shape {y.shape}")
    
    # Build and train model
    model = build_lstm((seq_len, X.shape[2]))
    
    print(f"Training for {epochs} epochs...")
    history = model.fit(X, y, validation_split=0.2, epochs=epochs, batch_size=batch_size, verbose=1)
    
    # Save model
    out_dir = os.path.join(REPO_ROOT, "outputs", "eda")
    os.makedirs(out_dir, exist_ok=True)
    model_path = os.path.join(out_dir, "test_hybrid_model.keras")
    model.save(model_path)
    
    print(f"Model saved to: {model_path}")
    
    metrics = {
        "train_files": train_files,
        "epochs": epochs,
        "seq_length": seq_len,
        "batch_size": batch_size,
        "final_loss": float(history.history["loss"][-1]) if "loss" in history.history else None,
        "final_val_loss": float(history.history["val_loss"][-1]) if "val_loss" in history.history else None,
    }
    
    return model_path, metrics


def test_inference_on_random_files(model_path: str, num_test_files: int = 10,
                                   sample_rows: int = 2000) -> Dict:
    """Run inference on random files and compare with reference."""
    print("\n" + "="*100)
    print("TESTING INFERENCE ON RANDOM FILES")
    print("="*100)
    
    if load_model is None:
        print("TensorFlow not available. Skipping inference test.")
        return {}
    
    if not os.path.exists(model_path):
        print(f"ERROR: Model not found: {model_path}")
        return {}
    
    # Get random test files
    all_files = get_random_discharge_files()
    test_files = random.sample(all_files, min(num_test_files, len(all_files)))
    
    print(f"Testing on {len(test_files)} random files:")
    for f in test_files[:5]:
        print(f"  - {f}")
    if len(test_files) > 5:
        print(f"  ... and {len(test_files) - 5} more")
    
    # Prepare output directory
    test_out_dir = os.path.join(REPO_ROOT, "outputs", "eda", "test_results")
    os.makedirs(test_out_dir, exist_ok=True)
    
    data_dir = os.path.join(REPO_ROOT, "cleaned_dataset", "data")
    metadata_path = os.path.join(REPO_ROOT, "cleaned_dataset", "metadata.csv")
    
    results = {
        "num_files": len(test_files),
        "predictions": [],
        "summary_metrics": {}
    }
    
    soh_errors = []
    rul_errors = []
    successful_inferences = 0
    
    for test_file in test_files:
        input_path = os.path.join(data_dir, test_file)
        
        if not os.path.exists(input_path):
            print(f"  WARNING: File not found: {input_path}")
            continue
        
        try:
            # Get reference values from metadata
            reference = get_reference_soh_rul(test_file, metadata_path)
            if not reference:
                print(f"  SKIP {test_file}: No reference data in metadata")
                continue
            
            # Get EKF params from metadata
            try:
                ekf_params = get_ekf_params_for_file(test_file, metadata_path=metadata_path)
            except Exception:
                ekf_params = dict(dt=1.0, C_nom=2.3, R0=0.05, R_D=0.01, C_D=500.0, eta=0.99)
            
            # Run inference
            out_csv = os.path.join(test_out_dir, f"{os.path.splitext(test_file)[0]}.inference.csv")
            summary = infer_single_file(input_path, model_path, ekf_params, seq_len=50,
                                       sample_rows=sample_rows, rul_max=None, out_csv=out_csv,
                                       init_soc=None, diagnostic=False, diag_out=None)
            
            pred_soh = summary["last_pred_soh"]
            ref_soh = reference["soh_reference"]
            soh_error = abs(pred_soh - ref_soh)
            soh_errors.append(soh_error)
            
            # RUL error (scaled)
            pred_rul = summary["last_pred_rul_scaled"]
            rul_errors.append(float(pred_rul) if pred_rul is not None else np.nan)
            
            result_entry = {
                "filename": test_file,
                "battery_id": reference.get("battery_id"),
                "pred_soh": float(pred_soh),
                "ref_soh": float(ref_soh),
                "soh_error": float(soh_error),
                "pred_rul_scaled": float(pred_rul) if pred_rul is not None else None,
                "ref_rul": reference.get("rul_reference"),
                "ekf_soc_final": summary.get("last_ekf_soc"),
                "warnings": {
                    "data_quality": summary.get("data_quality_warning"),
                    "ekf_saturation": summary.get("ekf_saturation_warning"),
                }
            }
            
            results["predictions"].append(result_entry)
            successful_inferences += 1
            
            status = "OK"
            if soh_error > 0.2:
                status = "HIGH ERROR"
            
            print(f"  [{status}] {test_file}: ref_SoH={ref_soh:.4f}, pred_SoH={pred_soh:.4f}, error={soh_error:.4f}")
        
        except Exception as e:
            print(f"  ERROR {test_file}: {str(e)[:100]}")
            results["predictions"].append({
                "filename": test_file,
                "error": str(e)
            })
    
    # Compute summary metrics
    if len(soh_errors) > 0:
        soh_errors = np.array(soh_errors)
        results["summary_metrics"] = {
            "successful_inferences": successful_inferences,
            "total_attempted": len(test_files),
            "soh_mae": float(np.mean(soh_errors)),
            "soh_rmse": float(np.sqrt(np.mean(soh_errors**2))),
            "soh_std": float(np.std(soh_errors)),
            "soh_min_error": float(np.min(soh_errors)),
            "soh_max_error": float(np.max(soh_errors)),
        }
    
    return results


def main():
    parser = argparse.ArgumentParser(description="Test metadata-based hybrid model")
    parser.add_argument("--max-train", type=int, default=100,
                       help="Max files to use for training")
    parser.add_argument("--num-test", type=int, default=10,
                       help="Number of random files to test on")
    parser.add_argument("--seq-len", type=int, default=50,
                       help="Sequence length for LSTM")
    parser.add_argument("--epochs", type=int, default=3,
                       help="Training epochs")
    parser.add_argument("--batch-size", type=int, default=32,
                       help="Batch size")
    parser.add_argument("--sample-rows", type=int, default=2000,
                       help="Rows per file to load")
    parser.add_argument("--skip-train", action="store_true",
                       help="Skip training and use existing model")
    parser.add_argument("--model", type=str, default=None,
                       help="Path to pre-trained model (overrides --skip-train)")
    parser.add_argument("--quick", action="store_true",
                       help="Quick test with defaults (50 train, 5 test, 2 epochs)")
    args = parser.parse_args()
    
    # Apply quick mode
    if args.quick:
        args.max_train = 50
        args.num_test = 5
        args.epochs = 2
        print("Quick mode: 50 train files, 5 test files, 2 epochs")
    
    # Main test report
    test_report = {
        "timestamp": pd.Timestamp.now().isoformat(),
        "configuration": {
            "max_train_files": args.max_train,
            "num_test_files": args.num_test,
            "seq_length": args.seq_len,
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "sample_rows": args.sample_rows,
        },
        "training": {},
        "inference": {},
    }
    
    # Train or use existing model
    if args.model:
        model_path = args.model
        print(f"Using pre-trained model: {model_path}")
    elif args.skip_train:
        model_path = os.path.join(REPO_ROOT, "outputs", "eda", "hybrid_lstm_model.keras")
        print(f"Using existing model: {model_path}")
    else:
        model_path, train_metrics = train_test_model(
            train_files=args.max_train,
            epochs=args.epochs,
            seq_len=args.seq_len,
            batch_size=args.batch_size,
            sample_rows=args.sample_rows
        )
        test_report["training"] = train_metrics
    
    if model_path and os.path.exists(model_path):
        # Run inference tests
        inference_results = test_inference_on_random_files(
            model_path,
            num_test_files=args.num_test,
            sample_rows=args.sample_rows
        )
        test_report["inference"] = inference_results
    
    # Print summary
    print("\n" + "="*100)
    print("TEST SUMMARY")
    print("="*100)
    
    if "summary_metrics" in test_report["inference"]:
        metrics = test_report["inference"]["summary_metrics"]
        print(f"\nInference Results:")
        print(f"  Successful inferences: {metrics['successful_inferences']}/{metrics['total_attempted']}")
        print(f"  SoH MAE: {metrics['soh_mae']:.6f}")
        print(f"  SoH RMSE: {metrics['soh_rmse']:.6f}")
        print(f"  SoH Std Dev: {metrics['soh_std']:.6f}")
        print(f"  SoH Error Range: [{metrics['soh_min_error']:.6f}, {metrics['soh_max_error']:.6f}]")
    
    # Save report
    out_dir = os.path.join(REPO_ROOT, "outputs", "eda")
    os.makedirs(out_dir, exist_ok=True)
    report_path = os.path.join(out_dir, "test_metadata_based_model_report.json")
    
    with open(report_path, "w") as f:
        json.dump(test_report, f, indent=2)
    
    print(f"\nFull report saved to: {report_path}")
    
    # Also save detailed predictions CSV
    if "predictions" in test_report["inference"] and test_report["inference"]["predictions"]:
        pred_df = pd.DataFrame([
            p for p in test_report["inference"]["predictions"]
            if "filename" in p and "error" not in p
        ])
        if len(pred_df) > 0:
            pred_csv_path = os.path.join(out_dir, "test_predictions.csv")
            pred_df.to_csv(pred_csv_path, index=False)
            print(f"Detailed predictions saved to: {pred_csv_path}")
    
    print("\n" + "="*100)


if __name__ == "__main__":
    main()
