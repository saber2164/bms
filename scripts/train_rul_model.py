
import pandas as pd
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
import joblib
from rul_predictor import RULPredictor
MAX_RUL = 2000 
def load_and_preprocess_data(soh_file: str, metadata_file: str, ic_file: str) -> tuple[list[np.ndarray], list[np.ndarray], int]:
    """
    Loads battery capacity and metadata, merges them, and returns a list of
    feature sequences for each battery.

    Args:
        soh_file (str): Path to the soh_features.csv file.
        metadata_file (str): Path to the metadata.csv file.
        ic_file (str): Path to the ic_features.csv file.

    Returns:
        A tuple containing:
        - A list of numpy arrays (features).
        - A list of numpy arrays (RUL targets).
        - The number of features.
    """
    try:
        soh_df = pd.read_csv(soh_file)
        meta_df = pd.read_csv(metadata_file)
        ic_df = pd.read_csv(ic_file)
    except FileNotFoundError as e:
        print(f"Error: {e}. Make sure all data files exist.")
        return [], [], 0

    # Sort metadata by battery_id and test_id to ensure correct order for ffill
    meta_df = meta_df.sort_values(['battery_id', 'test_id'])

    # Forward fill Re and Rct from impedance tests to discharge tests
    meta_df['Re'] = pd.to_numeric(meta_df['Re'], errors='coerce')
    meta_df['Rct'] = pd.to_numeric(meta_df['Rct'], errors='coerce')
    meta_df['Re'] = meta_df.groupby('battery_id')['Re'].ffill()
    meta_df['Rct'] = meta_df.groupby('battery_id')['Rct'].ffill()

    # Keep only discharge cycles and relevant columns
    meta_df_discharge = meta_df[meta_df['type'] == 'discharge'][['filename', 'battery_id', 'test_id', 'Re', 'Rct', 'start_time']]
    
    # Merge the two dataframes
    # Ensure filenames match (some might have .csv extension issues)
    meta_df_discharge['filename'] = meta_df_discharge['filename'].astype(str)
    if len(meta_df_discharge) > 0 and not meta_df_discharge['filename'].iloc[0].endswith('.csv'):
         meta_df_discharge['filename'] = meta_df_discharge['filename'] + '.csv'
         
    soh_df['filename'] = soh_df['filename'].astype(str)
    if len(soh_df) > 0 and not soh_df['filename'].iloc[0].endswith('.csv'):
        soh_df['filename'] = soh_df['filename'] + '.csv'

    ic_df['filename'] = ic_df['filename'].astype(str)
    # ic_features.csv was generated with .csv extension in filename column
    
    df = pd.merge(meta_df_discharge, soh_df, on='filename', how='inner')
    df = pd.merge(df, ic_df, on='filename', how='left') # Left join to keep all discharge cycles
    
    # Fill missing IC features (if any)
    df['ic_peak_height'] = df['ic_peak_height'].fillna(df['ic_peak_height'].mean())
    df['ic_peak_voltage'] = df['ic_peak_voltage'].fillna(df['ic_peak_voltage'].mean())

    # Filter out rows where capacity is zero or non-positive
    df = df[df['calculated_capacity'] > 0]
    
    # Fill any remaining NaNs
    df['Re'] = df['Re'].fillna(df['Re'].mean())
    df['Rct'] = df['Rct'].fillna(df['Rct'].mean())

    # Sort by time
    # Fix time parsing if needed (as seen in poly script)
    def parse_time_str(s):
        try:
            s = s.strip('[]')
            parts = s.split()
            parts = [int(float(p)) for p in parts]
            return f"{parts[0]}-{parts[1]}-{parts[2]} {parts[3]}:{parts[4]}:{parts[5]}"
        except:
            return s 
            
    # Check if start_time needs parsing (it might be already parsed or not)
    # In original script it was using eval.
    # Let's use the robust parser.
    df['start_time'] = pd.to_datetime(df['start_time'].apply(parse_time_str))
    
    df = df.sort_values(['battery_id', 'start_time'])
    
    # Features to use
    feature_cols = ['soh', 'average_temp', 'Re', 'Rct', 'ic_peak_height', 'ic_peak_voltage']
    
    processed_feature_curves = []
    processed_rul_curves = []
    
    for bat_id in df['battery_id'].unique():
        bat_df = df[df['battery_id'] == bat_id].copy()

        # Calculate SoH relative to initial capacity (iloc[0]) as requested
        # This is the standard definition, assuming fresh battery at start
        nominal_capacity = bat_df['calculated_capacity'].iloc[0]
        
        # No truncation needed if we assume start is fresh, or we can keep the max_cap logic if we want robust start
        # But user requested iloc[0]. Let's stick to that.
        # However, we should still probably remove the "warm-up" if it exists, or just let the model handle it?
        # User instruction: "Remove the hardcoded NOMINAL_CAPACITY = 2.0. Replace it with dynamic calculation: nominal_capacity = bat_df['calculated_capacity'].iloc[0]."
        # User didn't explicitly say "remove truncation", but truncation was based on max_cap_idx.
        # If we use iloc[0], max_cap_idx might be later.
        # Let's remove the truncation logic to strictly follow "Fix Training Logic" instructions which didn't mention truncation.
        # Or better, keep truncation but based on the new nominal capacity? No, truncation was "start from max".
        # If we use iloc[0], we are saying t=0 is the reference.
        
        bat_df['soh'] = bat_df['calculated_capacity'] / nominal_capacity
        bat_df['soh'] = np.clip(bat_df['soh'], 0, 1.0)
        
        # Truncate "warm-up" phase: Start from the point of maximum capacity
        # This ensures we are modeling degradation, not the initial capacity increase
        # We calculate max capacity index on the *original* bat_df (before any other slicing if any)
        # But here bat_df is the full history for the battery.
        max_cap_idx = bat_df['calculated_capacity'].idxmax()
        bat_df = bat_df.loc[max_cap_idx:].copy()
        bat_df = bat_df.reset_index(drop=True)

        
        # RUL calculation
        bat_df = bat_df.sort_values('test_id').reset_index(drop=True)
        eol_indices = bat_df[bat_df['soh'] < 0.8].index
        
        if len(eol_indices) > 0:
             eol_idx = eol_indices[0]
             bat_df['rul'] = eol_idx - bat_df.index
             bat_df = bat_df[bat_df['rul'] >= 0]
             
             if not bat_df.empty:
                processed_feature_curves.append(bat_df[feature_cols].values)
                processed_rul_curves.append(bat_df['rul'].values)
        # else:
             # If battery never reaches EOL, it's not useful for RUL prediction
             # We skip it.
             
    print(f"Processed {len(processed_feature_curves)} unique battery degradation curves.")
    print(f"Features used: {feature_cols}")
    
    return processed_feature_curves, processed_rul_curves, len(feature_cols)

def get_train_test_data(soh_file: str, metadata_file: str, ic_file: str, sequence_length: int, end_of_life_soh: float, test_split: float = 0.2, MAX_RUL = None) -> tuple:
    """
    Loads data, splits it, scales it, and creates sequences for training and testing.
    
    Returns:
        X_train, y_train, X_test, y_test, n_features
    """
    print("Loading and preprocessing SoH data...")
    feature_curves, rul_curves, n_features = load_and_preprocess_data(soh_file, metadata_file, ic_file)
    
    if not feature_curves:
        print("No data available.")
        return None, None, None, None, 0

    # Split batteries into training and testing sets
    n_batteries = len(feature_curves)
    n_test = int(n_batteries * test_split)
    
    # Random shuffle (optional, but good practice)
    indices = np.arange(n_batteries)
    np.random.shuffle(indices)
    
    train_indices = indices[:-n_test]
    test_indices = indices[-n_test:]
    
    train_features = [feature_curves[i] for i in train_indices]
    train_ruls = [rul_curves[i] for i in train_indices]
    test_features = [feature_curves[i] for i in test_indices]
    test_ruls = [rul_curves[i] for i in test_indices]
    
    # Fit scaler on training data ONLY
    # Flatten training features to fit scaler
    all_train_features = np.vstack(train_features)
    scaler = MinMaxScaler()
    scaler.fit(all_train_features)
    
    # Save scaler for inference
    import joblib
    joblib.dump(scaler, 'outputs/feature_scaler.pkl')
    print("Scaler saved to outputs/feature_scaler.pkl")
    MAX_RUL = 2000  # Assuming max RUL for scaling
    # Helper to create sequences
    def create_sequences(features_list, ruls_list):
        X, y = [], []
        for features, ruls in zip(features_list, ruls_list):
            # Scale features
            scaled_features = scaler.transform(features)
            
            for i in range(len(scaled_features) - sequence_length):
                X.append(scaled_features[i:i+sequence_length])
                # y.append(ruls[i+sequence_length]) # Predict RUL at the end of sequence
                raw_rul = ruls[i+sequence_length]
                y.append(raw_rul)
        return np.array(X), np.array(y)

    print("Creating sequences...")
    X_train, y_train = create_sequences(train_features, train_ruls)
    X_test, y_test = create_sequences(test_features, test_ruls)
    
    # Scale targets
    max_rul = np.max(y_train)
    y_train = y_train / max_rul
    y_test = y_test / max_rul
    
    # Save max_rul
    joblib.dump(max_rul, 'outputs/max_rul.pkl')
    print(f"Max RUL: {max_rul} (Saved to outputs/max_rul.pkl)")
    
    print(f"Training samples: {len(X_train)}")
    print(f"Testing samples: {len(X_test)}")
    
    return X_train, y_train, X_test, y_test, n_features, max_rul


def prepare_training_data(soh_curves: list[np.ndarray], sequence_length: int, end_of_life_soh: float) -> tuple[np.ndarray, np.ndarray]:
    """
    Creates training sequences (X) and RUL targets (y) from a list of SoH curves.

    Args:
        soh_curves (list): A list of SoH degradation curves.
        sequence_length (int): The length of the input sequences.
        end_of_life_soh (float): The SoH threshold for end-of-life.

    Returns:
        A tuple of (X_all, y_all) containing aggregated training data.
    """
    X_all, y_all = [], []
    
    for curve in soh_curves:
        try:
            # Find the EOL for this specific curve. 
            # We assume the first column is SoH.
            eol_cycle = np.where(curve[:, 0] < end_of_life_soh)[0][0]
        except IndexError:
            # If the battery never reaches EOL in the data, we can't use it for RUL prediction
            continue

        # The RUL at each point is the EOL cycle minus the current cycle
        rul_values = eol_cycle - np.arange(len(curve))

        if len(curve) > sequence_length:
            X, y = [], []
            for i in range(len(curve) - sequence_length):
                X.append(curve[i:(i + sequence_length)])
                # The target is the RUL at the end of the sequence
                y.append(rul_values[i + sequence_length - 1])
            X_all.extend(X)
            y_all.extend(y)
            
    return np.array(X_all), np.array(y_all)

def main():
    """
    Main function to orchestrate the model training and validation process.
    """
    # Configuration
    SOH_FILE = 'soh_features.csv'
    METADATA_FILE = 'cleaned_dataset/metadata.csv'
    IC_FILE = 'ic_features.csv'
    SEQUENCE_LENGTH = 15
    END_OF_LIFE_SOH = 0.8
    EPOCHS = 50
    BATCH_SIZE = 32
    TRAIN_TEST_SPLIT = 0.8
    MAX_RUL = 2000  # Assuming max RUL for scaling
    # Get data
    X_train, y_train, X_test, y_test, n_features,MAX_RUL = get_train_test_data(
        SOH_FILE, METADATA_FILE, IC_FILE, SEQUENCE_LENGTH, END_OF_LIFE_SOH, 1.0 - TRAIN_TEST_SPLIT, MAX_RUL
    )
    
    if X_train is None:
        return

    # Build model
    predictor = RULPredictor(sequence_length=SEQUENCE_LENGTH, n_features=n_features)
    model = predictor.build_model()
    
    # Train
    print("Training model...")
    history = model.fit(
        X_train, y_train,
        validation_data=(X_test, y_test),
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        callbacks=[
            tf.keras.callbacks.EarlyStopping(patience=10, restore_best_weights=True),
            tf.keras.callbacks.ReduceLROnPlateau(factor=0.5, patience=5)
        ]
    )
    # Save model
    model.save_weights('outputs/rul_model.weights.h5')
    print("Model weights saved to outputs/rul_model.weights.h5")

    # --- 5. Validate the Model on the Test Set ---
    print("\n--- Validating Model on Test Set ---")
    
    if X_test.shape[0] == 0:
        print("Not enough data to form test sequences.")
        return

    print(f"Created {X_test.shape[0]} test samples.")
    
    # Make predictions
    y_pred = predictor.model.predict(X_test).flatten()
    y_test_cycles = y_test * MAX_RUL
    y_pred_cycles = y_pred * MAX_RUL

    # Calculate metrics
    mse = np.mean(np.square(y_test_cycles - y_pred_cycles))
    mae = np.mean(np.abs(y_test_cycles - y_pred_cycles))
    print(f"\nTest Set Validation Metrics:")
    print(f"  Mean Squared Error (MSE): {mse:.2f}")
    print(f"  Mean Absolute Error (MAE): {mae:.2f} cycles")
    
    print("\nSample Predictions (Actual vs Predicted):")
    for i in range(min(10, len(y_test_cycles))):
        print(f"  Sample {i}: Actual={y_test_cycles[i]:.2f}, Predicted={y_pred_cycles[i]:.2f}")

    # --- 6. Visualize Results ---
    print("\n--- Generating Validation Plot ---")
    plt.figure(figsize=(12, 6))
    
    # Plot actual vs. predicted RUL for a sample from the test set
    sample_indices = np.random.choice(len(y_test), size=min(500, len(y_test)), replace=False)
    plt.scatter(y_test_cycles[sample_indices], y_pred_cycles[sample_indices], alpha=0.5, label='Predicted vs. Actual')
    
    # Plot the ideal line
    min_val = min(np.min(y_test_cycles), np.min(y_pred_cycles))
    max_val = max(np.max(y_test_cycles), np.max(y_pred_cycles))
    plt.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label='Ideal Fit')
    
    plt.title('RUL Prediction: Actual vs. Predicted on Test Set')
    plt.xlabel('Actual RUL (Cycles)')
    plt.ylabel('Predicted RUL (Cycles)')
    plt.legend()
    plt.grid(True)
    
    plot_filename = 'rul_validation_plot.png'
    plt.savefig(plot_filename)
    print(f"Validation plot saved to {plot_filename}")


if __name__ == '__main__':
    main()
