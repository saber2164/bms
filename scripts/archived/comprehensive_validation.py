#!/usr/bin/env python3
"""
Comprehensive Validation Report Generator
Tests the hybrid model on a large random sample with detailed analysis
"""

import os
import sys
import json
import numpy as np
import pandas as pd
from typing import Dict, List
import random

REPO_ROOT = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from scripts.infer_hybrid import get_reference_soh_rul, infer_single_file
from scripts.metadata_loader import get_ekf_params_for_file

try:
    from tensorflow.keras.models import load_model
except Exception:
    load_model = None


def validate_on_random_files(model_path: str, num_files: int = 100, 
                            sample_rows: int = 2000) -> Dict:
    """Run comprehensive validation on random files."""
    
    print("\n" + "="*120)
    print("COMPREHENSIVE MODEL VALIDATION")
    print("="*120)
    
    if load_model is None:
        print("ERROR: TensorFlow not available")
        return {}
    
    if not os.path.exists(model_path):
        print(f"ERROR: Model not found: {model_path}")
        return {}
    
    print(f"\nModel: {model_path}")
    print(f"Validation files: {num_files}")
    print(f"Sample rows per file: {sample_rows}\n")
    
    # Load metadata
    metadata_path = os.path.join(REPO_ROOT, "cleaned_dataset", "metadata.csv")
    meta = pd.read_csv(metadata_path)
    
    # Get all discharge files
    dis_df = meta[meta["type"] == "discharge"].copy()
    dis_df["Capacity"] = pd.to_numeric(dis_df["Capacity"], errors="coerce")
    dis_df = dis_df[dis_df["Capacity"].notna()]
    
    all_files = dis_df["filename"].unique().tolist()
    test_files = random.sample(all_files, min(num_files, len(all_files)))
    
    print(f"Testing on {len(test_files)} random discharge files...\n")
    
    data_dir = os.path.join(REPO_ROOT, "cleaned_dataset", "data")
    test_out_dir = os.path.join(REPO_ROOT, "outputs", "eda", "validation_results")
    os.makedirs(test_out_dir, exist_ok=True)
    
    results = {
        "timestamp": pd.Timestamp.now().isoformat(),
        "model_path": model_path,
        "num_files": len(test_files),
        "predictions": [],
        "by_battery": {},
        "error_analysis": {},
        "performance_metrics": {},
    }
    
    soh_errors = []
    rul_errors = []
    successful = 0
    failed = 0
    
    battery_data = {}  # Track per-battery performance
    
    for idx, test_file in enumerate(test_files, 1):
        input_path = os.path.join(data_dir, test_file)
        
        if not os.path.exists(input_path):
            failed += 1
            continue
        
        try:
            # Get reference values
            reference = get_reference_soh_rul(test_file, metadata_path)
            if not reference:
                failed += 1
                continue
            
            # Get EKF params from metadata
            try:
                ekf_params = get_ekf_params_for_file(test_file, metadata_path=metadata_path)
            except Exception:
                ekf_params = dict(dt=1.0, C_nom=2.3, R0=0.05, R_D=0.01, C_D=500.0, eta=0.99)
            
            # Run inference
            out_csv = os.path.join(test_out_dir, f"{os.path.splitext(test_file)[0]}.csv")
            summary = infer_single_file(input_path, model_path, ekf_params, seq_len=50,
                                       sample_rows=sample_rows, rul_max=None, out_csv=out_csv,
                                       init_soc=None, diagnostic=False, diag_out=None)
            
            pred_soh = summary["last_pred_soh"]
            ref_soh = reference["soh_reference"]
            soh_error = abs(pred_soh - ref_soh)
            soh_errors.append(soh_error)
            
            pred_rul = summary["last_pred_rul_scaled"]
            rul_errors.append(float(pred_rul) if pred_rul is not None else np.nan)
            
            bid = reference.get("battery_id")
            
            # Track per-battery
            if bid not in battery_data:
                battery_data[bid] = {"errors": [], "count": 0}
            battery_data[bid]["errors"].append(soh_error)
            battery_data[bid]["count"] += 1
            
            result_entry = {
                "filename": test_file,
                "battery_id": bid,
                "pred_soh": float(pred_soh),
                "ref_soh": float(ref_soh),
                "soh_error": float(soh_error),
                "soh_error_pct": float(soh_error * 100),
                "pred_rul_scaled": float(pred_rul) if pred_rul is not None else None,
                "ref_rul": reference.get("rul_reference"),
                "ekf_soc_final": summary.get("last_ekf_soc"),
                "capacity_actual": reference.get("capacity_actual"),
                "capacity_nominal": reference.get("capacity_nominal"),
            }
            
            results["predictions"].append(result_entry)
            successful += 1
            
            # Progress indicator
            if idx % 10 == 0:
                avg_err = np.mean(soh_errors[-10:])
                print(f"  [{idx:3d}/{len(test_files)}] Processed {test_file} | Recent avg error: {avg_err:.4f}")
        
        except Exception as e:
            failed += 1
            pass
    
    # Compute summary metrics
    print(f"\n  Successful: {successful}/{len(test_files)} ({100*successful/len(test_files):.1f}%)")
    
    if len(soh_errors) > 0:
        soh_errors = np.array(soh_errors)
        
        # Overall metrics
        results["performance_metrics"] = {
            "successful_inferences": successful,
            "failed_inferences": failed,
            "success_rate": float(successful / len(test_files)),
            "soh_mae": float(np.mean(soh_errors)),
            "soh_rmse": float(np.sqrt(np.mean(soh_errors**2))),
            "soh_std": float(np.std(soh_errors)),
            "soh_min_error": float(np.min(soh_errors)),
            "soh_max_error": float(np.max(soh_errors)),
            "soh_median_error": float(np.median(soh_errors)),
            "soh_q25_error": float(np.percentile(soh_errors, 25)),
            "soh_q75_error": float(np.percentile(soh_errors, 75)),
        }
        
        # Error bins
        error_bins = [0.0, 0.02, 0.05, 0.1, 0.15, 0.2, 1.0]
        bin_counts = [0] * (len(error_bins) - 1)
        for err in soh_errors:
            for i in range(len(error_bins) - 1):
                if error_bins[i] <= err < error_bins[i+1]:
                    bin_counts[i] += 1
                    break
        
        results["error_analysis"]["error_distribution"] = {
            f"{error_bins[i]:.3f}-{error_bins[i+1]:.3f}": count 
            for i, count in enumerate(bin_counts)
        }
        
        # Per-battery metrics
        for bid, data in battery_data.items():
            errors = np.array(data["errors"])
            results["by_battery"][bid] = {
                "count": int(data["count"]),
                "mae": float(np.mean(errors)),
                "rmse": float(np.sqrt(np.mean(errors**2))),
                "std": float(np.std(errors)),
                "min": float(np.min(errors)),
                "max": float(np.max(errors)),
            }
    
    return results


def main():
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, 
                       default=os.path.join("outputs/eda/hybrid_lstm_model.keras"),
                       help="Model path to validate")
    parser.add_argument("--num-files", type=int, default=100,
                       help="Number of random files to test")
    parser.add_argument("--sample-rows", type=int, default=2000,
                       help="Rows per file")
    args = parser.parse_args()
    
    # Run validation
    results = validate_on_random_files(args.model, num_files=args.num_files, 
                                      sample_rows=args.sample_rows)
    
    if not results:
        print("ERROR: Validation failed")
        return
    
    # Print detailed report
    print("\n" + "="*120)
    print("VALIDATION RESULTS SUMMARY")
    print("="*120)
    
    metrics = results["performance_metrics"]
    
    print(f"\nOverall Performance:")
    print(f"  Success Rate: {metrics['success_rate']*100:.1f}% ({metrics['successful_inferences']}/{metrics['successful_inferences']+metrics['failed_inferences']})")
    print(f"\nSoH Prediction Accuracy:")
    print(f"  Mean Absolute Error (MAE):  {metrics['soh_mae']:.6f} ({metrics['soh_mae']*100:.2f}%)")
    print(f"  Root Mean Squared Error:    {metrics['soh_rmse']:.6f} ({metrics['soh_rmse']*100:.2f}%)")
    print(f"  Standard Deviation:         {metrics['soh_std']:.6f}")
    print(f"  Median Error:               {metrics['soh_median_error']:.6f} ({metrics['soh_median_error']*100:.2f}%)")
    print(f"  Error Range:                [{metrics['soh_min_error']:.6f}, {metrics['soh_max_error']:.6f}]")
    print(f"  25th-75th Percentile:       [{metrics['soh_q25_error']:.6f}, {metrics['soh_q75_error']:.6f}]")
    
    print(f"\nError Distribution:")
    for range_str, count in results["error_analysis"]["error_distribution"].items():
        pct = 100 * count / metrics['successful_inferences']
        print(f"  {range_str}: {count:3d} files ({pct:5.1f}%)")
    
    print(f"\nTop Performing Batteries (lowest error):")
    battery_sorted = sorted(results["by_battery"].items(), 
                           key=lambda x: x[1]["mae"])[:5]
    for bid, metrics_b in battery_sorted:
        print(f"  {bid}: MAE={metrics_b['mae']:.6f}, count={metrics_b['count']}")
    
    print(f"\nMost Challenging Batteries (highest error):")
    battery_sorted = sorted(results["by_battery"].items(), 
                           key=lambda x: x[1]["mae"], reverse=True)[:5]
    for bid, metrics_b in battery_sorted:
        print(f"  {bid}: MAE={metrics_b['mae']:.6f}, count={metrics_b['count']}")
    
    # Save results
    out_dir = os.path.join(REPO_ROOT, "outputs", "eda")
    os.makedirs(out_dir, exist_ok=True)
    
    report_path = os.path.join(out_dir, f"validation_report_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.json")
    with open(report_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nFull report saved to: {report_path}")
    
    # Save predictions CSV
    if results["predictions"]:
        pred_df = pd.DataFrame(results["predictions"])
        pred_csv = os.path.join(out_dir, "validation_predictions.csv")
        pred_df.to_csv(pred_csv, index=False)
        print(f"Predictions CSV saved to: {pred_csv}")
    
    print("="*120 + "\n")


if __name__ == "__main__":
    main()
