#!/usr/bin/env python3
"""
Scans the data directory to find files that are suitable for training and inference.

A "good" file is one that contains at least one valid column name for each of the
required measurements (current, voltage, and temperature).
"""

import os
import glob
import pandas as pd
import argparse

def find_good_files(data_dir):
    """
    Scans the data directory and returns a list of "good" file paths.
    """
    all_files = sorted(glob.glob(os.path.join(data_dir, "*.csv")))
    good_files = []

    # Lists of possible column names
    current_colnames = ['Current_measured', 'Sense_current', 'Battery_current']
    voltage_colnames = ['Voltage_measured', 'Voltage_load']
    temp_colnames = ['Temperature_measured']

    for file_path in all_files:
        try:
            df = pd.read_csv(file_path, nrows=1) # Read only the header
            
            current_col = next((col for col in current_colnames if col in df.columns), None)
            voltage_col = next((col for col in voltage_colnames if col in df.columns), None)
            temp_col = next((col for col in temp_colnames if col in df.columns), None)
            
            if all([current_col, voltage_col, temp_col]):
                good_files.append(file_path)
        except Exception as e:
            print(f"Error processing file {file_path}: {e}")
            continue
            
    return good_files

def main():
    parser = argparse.ArgumentParser(description="Find good files for training/inference.")
    parser.add_argument('--data-dir', type=str, default='cleaned_dataset/data', help='Path to the data directory.')
    parser.add_argument('--output-file', type=str, default='good_files.txt', help='Path to the output file.')
    args = parser.parse_args()

    print(f"Scanning {args.data_dir} for good files...")
    good_files = find_good_files(args.data_dir)
    
    print(f"Found {len(good_files)} good files.")
    
    with open(args.output_file, 'w') as f:
        for file_path in good_files:
            f.write(f"{file_path}\n")
            
    print(f"List of good files saved to {args.output_file}")

if __name__ == "__main__":
    main()
