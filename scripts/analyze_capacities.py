import pandas as pd
import matplotlib.pyplot as plt

def analyze_capacities():
    meta_df = pd.read_csv('cleaned_dataset/metadata.csv')
    
    # Filter discharge
    discharge_df = meta_df[meta_df['type'] == 'discharge'].copy()
    discharge_df['Capacity'] = pd.to_numeric(discharge_df['Capacity'], errors='coerce')
    
    # Group by battery
    batteries = discharge_df['battery_id'].unique()
    
    print(f"Found {len(batteries)} batteries.")
    print("Initial Capacities:")
    
    initial_caps = []
    
    for bat in batteries:
        bat_df = discharge_df[discharge_df['battery_id'] == bat].sort_values('test_id')
        if not bat_df.empty:
            init_cap = bat_df['Capacity'].iloc[0]
            initial_caps.append(init_cap)
            print(f"{bat}: {init_cap:.4f} Ah")
            
    avg_init = sum(initial_caps) / len(initial_caps)
    print(f"\nAverage Initial Capacity: {avg_init:.4f} Ah")
    
    # Plot histogram
    plt.figure(figsize=(10, 6))
    plt.hist(initial_caps, bins=20)
    plt.title('Distribution of Initial Capacities')
    plt.xlabel('Capacity (Ah)')
    plt.ylabel('Count')
    plt.savefig('outputs/initial_capacities.png')
    print("Saved histogram to outputs/initial_capacities.png")

if __name__ == "__main__":
    analyze_capacities()
