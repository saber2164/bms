#!/usr/bin/env python3
"""
Runs batch inference on a random sample of "good" files and checks for anomalies.
"""

import os
import random
import subprocess
import json
import argparse

def run_batch_inference(files_list, num_to_run):
    """
    Runs inference on a random sample of files and reports anomalies.
    """
    with open(files_list, 'r') as f:
        good_files = [line.strip() for line in f.readlines()]
        
    if len(good_files) < num_to_run:
        print(f"Warning: Requested to run on {num_to_run} files, but only {len(good_files)} good files were found.")
        num_to_run = len(good_files)
        
    random_sample = random.sample(good_files, num_to_run)
    
    anomalies = []
    
    print(f"--- Starting batch inference on {num_to_run} random files ---")
    
    for i, file_path in enumerate(random_sample):
        print(f"Running inference on file {i+1}/{num_to_run}: {file_path}")
        
        try:
            process = subprocess.run(
                ['python3', 'scripts/infer_ukf.py', '--input', file_path],
                capture_output=True,
                text=True,
                timeout=300 # 5 minute timeout per file
            )
            
            if process.returncode != 0:
                anomalies.append({
                    "file": file_path,
                    "anomaly": "Inference script failed.",
                    "details": process.stderr
                })
                continue

            # The summary is printed to stdout, but it's not in a clean JSON format.
            # The last few lines of stdout contain the summary.
            summary_str = ""
            in_summary = False
            for line in process.stdout.strip().split('\n'):
                if "--- Inference Summary ---" in line:
                    in_summary = True
                    continue
                if in_summary:
                    summary_str += line
            
            # A bit of a hack to parse the summary, as it's not clean JSON
            try:
                # This is very brittle. A better solution would be to make the inference script output clean JSON.
                r0_line = [line for line in summary_str.split('\n') if "Final Estimated Resistance" in line][0]
                r0_val = float(r0_line.split(':')[1].strip().split(' ')[0])
                
                if r0_val == 1e-4:
                    anomalies.append({
                        "file": file_path,
                        "anomaly": "Resistance estimate hit the lower bound constraint.",
                        "details": f"Final R_0: {r0_val}"
                    })
            except Exception as e:
                anomalies.append({
                    "file": file_path,
                    "anomaly": "Could not parse inference output.",
                    "details": str(e)
                })

        except subprocess.TimeoutExpired:
            anomalies.append({
                "file": file_path,
                "anomaly": "Inference timed out.",
                "details": None
            })
        except Exception as e:
            anomalies.append({
                "file": file_path,
                "anomaly": "An unexpected error occurred.",
                "details": str(e)
            })
            
    print("\n--- Batch Inference Finished ---")
    
    if not anomalies:
        print("No anomalies found.")
    else:
        print(f"Found {len(anomalies)} anomalies:")
        for anomaly in anomalies:
            print(json.dumps(anomaly, indent=2))

def main():
    parser = argparse.ArgumentParser(description="Run batch inference and check for anomalies.")
    parser.add_argument('--files-list', type=str, default='good_files.txt', help='Path to the list of good files.')
    parser.add_argument('--num-files', type=int, default=100, help='Number of random files to run inference on.')
    args = parser.parse_args()
    
    run_batch_inference(args.files_list, args.num_files)

if __name__ == "__main__":
    main()
