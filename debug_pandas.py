import pandas as pd
import sys

file_path = 'cleaned_dataset/data/07015.csv'
try:
    df = pd.read_csv(file_path)
    print("Original Columns:", df.columns.tolist())
    
    col_map = {
        'Current_measured': 'Current',
        'Voltage_measured': 'Voltage',
        'Temperature_measured': 'Temperature',
        'Time': 'Time'
    }
    df = df.rename(columns=col_map)
    print("Renamed Columns:", df.columns.tolist())
    
    if 'Voltage' in df.columns:
        print("Voltage column found.")
    else:
        print("Voltage column NOT found.")
        
except Exception as e:
    print(f"Error: {e}")
