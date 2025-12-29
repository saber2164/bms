#!/usr/bin/env python3
"""
Prepare cycle-aware dataset for SoC estimation training.

This script:
1. Loads metadata.csv to identify charge and discharge cycles
2. Processes each cycle and adds cycle_type feature
3. Validates data quality
4. Creates train/test splits
5. Saves processed data for model training
"""

import pandas as pd
import numpy as np
from pathlib import Path
import argparse
from tqdm import tqdm

def load_metadata(metadata_path):
    """Load and parse metadata CSV."""
    df = pd.read_csv(metadata_path)
    print(f"Loaded {len(df)} records from metadata")
    print(f"Cycle types: {df['type'].value_counts().to_dict()}")
    return df

def process_cycle_file(file_path, cycle_type):
    """
    Process a single cycle file and add cycle_type feature.
    
    Args:
        file_path: Path to CSV file
        cycle_type: 'charge' or 'discharge'
    
    Returns:
        DataFrame with added cycle_type column (-1 for charge, 1 for discharge)
    """
    try:
        df = pd.read_csv(file_path)
        
        # Validate required columns
        required_cols = ['Voltage_measured', 'Current_measured', 'Temperature_measured']
        
        # Check for required columns
        if not all(col in df.columns for col in required_cols):
            return None
        
        # Select only the relevant columns
        df = df[required_cols + ['Time']].copy()
        
        # Add cycle type: -1 for charge, 1 for discharge
        df['cycle_type'] = -1 if cycle_type == 'charge' else 1
        
        # Basic data validation - check for nulls
        if df['Voltage_measured'].isnull().any() or \
           df['Current_measured'].isnull().any() or \
           df['Temperature_measured'].isnull().any():
            return None
        
        # Filter out invalid voltage/current/temperature
        valid_voltage = (df['Voltage_measured'] > 2.0) & (df['Voltage_measured'] < 5.0)
        valid_current = df['Current_measured'].abs() < 10.0
        valid_temp = (df['Temperature_measured'] > -10) & (df['Temperature_measured'] < 60)
        
        df = df[valid_voltage & valid_current & valid_temp].copy()
        
        if len(df) < 10:  # Skip very short cycles
            return None
        
        return df[['Voltage_measured', 'Current_measured', 'Temperature_measured', 'cycle_type', 'Time']]
        
    except Exception as e:
        # Silently skip errors to avoid cluttering output
        return None

def main(args):
    # Paths
    base_dir = Path(args.data_dir)
    metadata_path = base_dir / 'metadata.csv'
    data_dir = base_dir / 'data'
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load metadata
    metadata = load_metadata(metadata_path)
    
    # Filter to charge and discharge cycles only
    charge_cycles = metadata[metadata['type'] == 'charge']
    discharge_cycles = metadata[metadata['type'] == 'discharge']
    
    print(f"\nProcessing {len(charge_cycles)} charge cycles and {len(discharge_cycles)} discharge cycles")
    
    # Process all cycles
    all_processed_data = []
    cycle_stats = {'charge': 0, 'discharge': 0, 'failed': 0}
    
    # Process charge cycles
    print("\nProcessing charge cycles...")
    for idx, row in tqdm(charge_cycles.iterrows(), total=len(charge_cycles), desc="Charge"):
        file_path = data_dir / row['filename']
        if file_path.exists():
            df = process_cycle_file(file_path, 'charge')
            if df is not None:
                all_processed_data.append(df)
                cycle_stats['charge'] += 1
            else:
                cycle_stats['failed'] += 1
    
    # Process discharge cycles
    print("\nProcessing discharge cycles...")
    for idx, row in tqdm(discharge_cycles.iterrows(), total=len(discharge_cycles), desc="Discharge"):
        file_path = data_dir / row['filename']
        if file_path.exists():
            df = process_cycle_file(file_path, 'discharge')
            if df is not None:
                all_processed_data.append(df)
                cycle_stats['discharge'] += 1
            else:
                cycle_stats['failed'] += 1
    
    # Combine all data
    print(f"\nCombining all processed cycles...")
    combined_df = pd.concat(all_processed_data, ignore_index=True)
    
    # Statistics
    print(f"\n{'='*60}")
    print(f"Processing Summary:")
    print(f"{'='*60}")
    print(f"Charge cycles processed: {cycle_stats['charge']}")
    print(f"Discharge cycles processed: {cycle_stats['discharge']}")
    print(f"Failed/skipped: {cycle_stats['failed']}")
    print(f"Total data points: {len(combined_df):,}")
    print(f"\nData Statistics:")
    print(combined_df.describe())
    
    # Validate cycle_type distribution
    print(f"\nCycle type distribution:")
    print(combined_df['cycle_type'].value_counts())
    
    # Save processed dataset
    output_path = output_dir / 'processed_cycles.csv'
    combined_df.to_csv(output_path, index=False)
    print(f"\nSaved processed dataset to: {output_path}")
    
    # Create train/test split (80/20)
    from sklearn.model_selection import train_test_split
    train_df, test_df = train_test_split(combined_df, test_size=0.2, random_state=42, shuffle=True)
    
    train_path = output_dir / 'train_cycles.csv'
    test_path = output_dir / 'test_cycles.csv'
    
    train_df.to_csv(train_path, index=False)
    test_df.to_csv(test_path, index=False)
    
    print(f"\nTrain set: {len(train_df):,} points -> {train_path}")
    print(f"Test set: {len(test_df):,} points -> {test_path}")
    print(f"\n{'='*60}")
    print("Dataset preparation complete!")
    print(f"{'='*60}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Prepare cycle-aware battery dataset')
    parser.add_argument('--data-dir', type=str, default='cleaned_dataset',
                        help='Directory containing metadata.csv and data/ folder')
    parser.add_argument('--output-dir', type=str, default='cleaned_dataset/processed',
                        help='Output directory for processed data')
    
    args = parser.parse_args()
    main(args)
