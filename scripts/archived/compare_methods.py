#!/usr/bin/env python3
"""
Compares the different battery state estimation methods implemented in this project.

This script runs the inference for the EKF, DEKF, and DUKF models on a given
input file and generates a comparative report with graphs and tables.
"""

import os
import time
import subprocess
import pandas as pd
import matplotlib.pyplot as plt
import argparse

def run_comparison(input_file):
    """
    Runs the comparison and generates the report.
    """
    if not os.path.exists(input_file):
        print(f"Error: Input file not found at {input_file}")
        return

    base_name = os.path.splitext(os.path.basename(input_file))[0]
    
    models = {
        "Hybrid EKF": {
            "script": "scripts/infer_hybrid.py",
            "output_file": f"outputs/eda/{base_name}.inference.csv",
            "soc_col": "EKF_SoC"
        },
        "Dual EKF": {
            "script": "scripts/infer_dekf.py",
            "output_file": f"outputs/{base_name}_dekf_inference.csv",
            "soc_col": "SoC_estimated",
            "r0_col": "R_0_estimated"
        },
        "Dual UKF": {
            "script": "scripts/infer_ukf.py",
            "output_file": f"outputs/{base_name}_dukf_inference.csv",
            "soc_col": "SoC_estimated",
            "r0_col": "R_0_estimated"
        }
    }
    
    results = {}

    # --- Run Inference for Each Model ---
    for model_name, config in models.items():
        print(f"--- Running {model_name} Inference ---")
        start_time = time.time()
        
        process = subprocess.run(
            ['python3', config['script'], '--input', input_file],
            capture_output=True, text=True
        )
        
        end_time = time.time()
        execution_time = end_time - start_time
        
        if process.returncode != 0:
            print(f"Error running {model_name} inference.")
            print(process.stderr)
            continue
            
        results[model_name] = {
            "execution_time": execution_time,
            "output_df": pd.read_csv(config['output_file'])
        }
        print(f"Finished in {execution_time:.2f} seconds.")

    # --- Generate Comparison Plots ---
    print("\n--- Generating Comparison Plots ---")
    plt.style.use('seaborn-v0_8-whitegrid')
    
    # SoC Comparison Plot
    plt.figure(figsize=(12, 7))
    for model_name, result in results.items():
        df = result['output_df']
        soc_col = models[model_name]['soc_col']
        if soc_col in df.columns:
            plt.plot(df['Time'], df[soc_col], label=model_name)
            
    plt.title('SoC Estimation Comparison', fontsize=16)
    plt.xlabel('Time (s)')
    plt.ylabel('State of Charge (SoC)')
    plt.legend()
    plt.grid(True)
    soc_plot_path = f"outputs/{base_name}_soc_comparison.png"
    plt.savefig(soc_plot_path)
    print(f"SoC comparison plot saved to {soc_plot_path}")

    # R0 Comparison Plot
    plt.figure(figsize=(12, 7))
    for model_name in ["Dual EKF", "Dual UKF"]:
        if model_name in results:
            df = results[model_name]['output_df']
            r0_col = models[model_name]['r0_col']
            if r0_col in df.columns:
                plt.plot(df['Time'], df[r0_col], label=model_name)

    plt.title('Internal Resistance (R_0) Estimation Comparison', fontsize=16)
    plt.xlabel('Time (s)')
    plt.ylabel('Resistance (Ohms)')
    plt.legend()
    plt.grid(True)
    r0_plot_path = f"outputs/{base_name}_r0_comparison.png"
    plt.savefig(r0_plot_path)
    print(f"R0 comparison plot saved to {r0_plot_path}")

    # --- Generate Summary Table ---
    print("\n--- Comparison Summary ---")
    summary_data = []
    for model_name, result in results.items():
        df = result['output_df']
        soc_col = models[model_name]['soc_col']
        soc_variance = df[soc_col].var() if soc_col in df.columns else "N/A"
        
        summary_data.append({
            "Model": model_name,
            "Execution Time (s)": f"{result['execution_time']:.2f}",
            "SoC Variance": f"{soc_variance:.6f}" if isinstance(soc_variance, float) else "N/A"
        })
        
    summary_df = pd.DataFrame(summary_data)
    print(summary_df.to_string(index=False))
    
    # Save summary to a text file
    report_path = f"outputs/{base_name}_comparison_report.txt"
    with open(report_path, 'w') as f:
        f.write("Comparison Summary\n")
        f.write("====================\n\n")
        f.write(summary_df.to_string(index=False))
        f.write("\n\nNotes:\n")
        f.write("- SoC Variance: A lower variance can indicate a more stable and less noisy estimate.\n")
        f.write("- Execution Time: Shows the computational cost of each model.\n")
        
    print(f"\nFull comparison report saved to {report_path}")


def main():
    parser = argparse.ArgumentParser(description="Compare battery state estimation methods.")
    parser.add_argument('--input', type=str, default='cleaned_dataset/data/00001.csv', help='Input CSV file for comparison.')
    args = parser.parse_args()
    
    run_comparison(args.input)

if __name__ == "__main__":
    main()
