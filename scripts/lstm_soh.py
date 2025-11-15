#!/usr/bin/env python3
"""
LSTM-based SoH estimator with metadata integration

This script provides:
 - create_sequences(data, labels, seq_length): convert time-series to overlapping sequences
 - build_lstm_model(input_shape): constructs a Keras Sequential LSTM model
 - main(): loads real discharge data from metadata and cleaned_dataset, trains on actual SoH labels

Notes:
 - The script uses tensorflow.keras. If TensorFlow is not available in the environment,
   the build/train section will raise ImportError; the code guards for that and explains what to do.
 - Normalization is implemented with simple mean/std computed on training data (no sklearn dependency).
 - SoH labels are computed from actual Capacity values in metadata per discharge cycle.
"""
import numpy as np
import os
from typing import Tuple
import sys
import pandas as pd


def create_sequences(data: np.ndarray, labels: np.ndarray, seq_length: int) -> Tuple[np.ndarray, np.ndarray]:
    """Create overlapping sequences from time-series data.

    Parameters
    ----------
    data : np.ndarray
        Array of shape (N, F) containing features [V, I, T, Cyc].
    labels : np.ndarray
        Array of shape (N,) containing SoH label at each timestep.
    seq_length : int
        Length of sequences (timesteps) to create.

    Returns
    -------
    X : np.ndarray
        Array of shape (M, seq_length, F)
    y : np.ndarray
        Array of shape (M,) where each y is the label at the end of the sequence.
    """
    N, F = data.shape
    if len(labels) != N:
        raise ValueError("data and labels must have same first dim")

    M = N - seq_length + 1
    if M <= 0:
        return np.zeros((0, seq_length, F)), np.zeros((0,))

    X = np.zeros((M, seq_length, F), dtype=float)
    y = np.zeros((M,), dtype=float)
    for i in range(M):
        X[i] = data[i : i + seq_length]
        y[i] = labels[i + seq_length - 1]
    return X, y


def build_lstm_model(input_shape: Tuple[int, int]):
    """Build a Keras Sequential LSTM model for regression of SoH.

    input_shape: (seq_length, n_features)
    """
    try:
        from tensorflow.keras.models import Sequential
        from tensorflow.keras.layers import LSTM, Dense, Dropout
        from tensorflow.keras.optimizers import Adam
    except Exception as e:
        raise ImportError("TensorFlow is required to build the LSTM model: " + str(e))

    seq_len, n_features = input_shape
    model = Sequential()
    model.add(LSTM(64, return_sequences=True, input_shape=(seq_len, n_features)))
    model.add(Dropout(0.2))
    model.add(LSTM(32, return_sequences=False))
    model.add(Dropout(0.1))
    model.add(Dense(16, activation="relu"))
    model.add(Dense(1, activation="linear"))

    model.compile(optimizer=Adam(learning_rate=1e-3), loss="mean_squared_error")
    return model


def _standardize(train: np.ndarray, val: np.ndarray = None):
    """Simple StandardScaler replacement using train stats."""
    mu = train.mean(axis=0)
    sigma = train.std(axis=0) + 1e-8
    train_s = (train - mu) / sigma
    if val is None:
        return train_s, mu, sigma
    else:
        return train_s, (val - mu) / sigma, mu, sigma


def load_real_discharge_data(max_files: int = None, sample_rows: int = 2000) -> Tuple[np.ndarray, np.ndarray]:
    """Load real discharge data from metadata and CSV files.
    
    Returns:
        data: (N, 4) array with features [V, I, T, cycle_fraction]
        soh: (N,) array with SoH labels
    """
    # Get repo root
    repo_root = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))
    metadata_path = os.path.join(repo_root, "cleaned_dataset", "metadata.csv")
    data_dir = os.path.join(repo_root, "cleaned_dataset", "data")
    
    if not os.path.exists(metadata_path):
        raise FileNotFoundError(f"Metadata file not found: {metadata_path}")
    
    # Load metadata
    meta = pd.read_csv(metadata_path)
    
    # Get discharge records with capacity
    dis_df = meta[meta["type"] == "discharge"].copy()
    dis_df["Capacity"] = pd.to_numeric(dis_df["Capacity"], errors="coerce")
    dis_df = dis_df[dis_df["Capacity"].notna()].sort_values(["battery_id", "test_id"])
    
    # Compute nominal capacity per battery
    cnom_map = {}
    for bid, g in dis_df.groupby("battery_id"):
        caps = g["Capacity"].values
        cnom_map[bid] = float(np.max(caps)) if len(caps) > 0 else 2.3
    
    print(f"Found {len(dis_df)} discharge records across {len(cnom_map)} batteries")
    print(f"Nominal capacities: {cnom_map}")
    
    # Collect data from files
    all_data = []
    all_soh = []
    file_count = 0
    
    files = dis_df["filename"].unique()
    if max_files:
        files = files[:max_files]
    
    for fn in files:
        path = os.path.join(data_dir, fn)
        if not os.path.exists(path):
            print(f"Warning: File not found: {path}")
            continue
        
        # Get SoH label for this file
        row = dis_df[dis_df["filename"] == fn].iloc[0]
        bid = row["battery_id"]
        cap = float(row["Capacity"])
        cnom = cnom_map.get(bid, 2.3)
        soh_label = cap / cnom if cnom > 0 else 1.0
        
        # Load CSV
        try:
            df = pd.read_csv(path, nrows=sample_rows)
            if len(df) == 0:
                continue
            
            # Extract features
            V = pd.to_numeric(df.get("Voltage_measured", []), errors="coerce").fillna(0.0).values
            I = pd.to_numeric(df.get("Current_measured", []), errors="coerce").fillna(0.0).values
            T = pd.to_numeric(df.get("Temperature_measured", []), errors="coerce").fillna(25.0).values
            
            # Cycle fraction (time-normalized)
            cyc_frac = np.linspace(0.0, 1.0, len(df))
            
            # Stack features
            data = np.vstack([V, I, T, cyc_frac]).T  # (N, 4)
            
            # All rows in this file get the same SoH label
            soh_arr = np.full(len(df), soh_label, dtype=float)
            
            all_data.append(data)
            all_soh.append(soh_arr)
            file_count += 1
            
            if file_count % 50 == 0:
                print(f"Loaded {file_count} files ({len(np.concatenate(all_data))} total timesteps)")
        
        except Exception as e:
            print(f"Error loading {fn}: {e}")
            continue
    
    if len(all_data) == 0:
        raise ValueError("No discharge data files loaded")
    
    data = np.vstack(all_data)
    soh = np.concatenate(all_soh)
    
    print(f"Total data shape: {data.shape}, labels shape: {soh.shape}")
    print(f"SoH range: [{soh.min():.4f}, {soh.max():.4f}]")
    
    return data, soh


def main():
    """Train LSTM model on real discharge data with metadata SoH labels."""
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-files", type=int, default=100, help="Max discharge files to load")
    parser.add_argument("--sample-rows", type=int, default=2000, help="Rows per file")
    parser.add_argument("--seq-len", type=int, default=50, help="Sequence length")
    parser.add_argument("--epochs", type=int, default=5, help="Training epochs")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size")
    args = parser.parse_args()
    
    # Load real data
    try:
        data, soh_labels = load_real_discharge_data(max_files=args.max_files, sample_rows=args.sample_rows)
    except Exception as e:
        print(f"Error loading data: {e}")
        return
    
    # train/val split
    split = int(len(data) * 0.8)
    train_data = data[:split]
    val_data = data[split:]
    train_labels = soh_labels[:split]
    val_labels = soh_labels[split:]

    # standardize features using training stats
    train_data_s, val_data_s, mu, sigma = _standardize(train_data, val_data)

    seq_length = args.seq_len
    X_train, y_train = create_sequences(train_data_s, train_labels, seq_length)
    X_val, y_val = create_sequences(val_data_s, val_labels, seq_length)

    print(f"X_train shape: {X_train.shape}, y_train shape: {y_train.shape}")
    print(f"X_val shape: {X_val.shape}, y_val shape: {y_val.shape}")

    # build model
    try:
        model = build_lstm_model((seq_length, X_train.shape[2]))
    except ImportError as e:
        print("TensorFlow not available; skipping model build/train. Error:", e)
        return

    # training
    print(f"Training for {args.epochs} epochs with batch size {args.batch_size}...")
    history = model.fit(X_train, y_train, validation_data=(X_val, y_val), 
                       epochs=args.epochs, batch_size=args.batch_size, verbose=1)

    # Evaluate
    val_loss = model.evaluate(X_val, y_val, verbose=0)
    print(f"Final validation loss: {val_loss:.6f}")

    # Save model artifact to outputs
    out_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "outputs", "eda")
    os.makedirs(out_dir, exist_ok=True)
    model_path = os.path.join(out_dir, "lstm_soh_model.keras")
    model.save(model_path)
    print(f"Model trained and saved to: {model_path}")
    
    # Also save normalization stats for inference
    stats_path = os.path.join(out_dir, "lstm_soh_stats.npz")
    np.savez(stats_path, mu=mu, sigma=sigma)
    print(f"Normalization stats saved to: {stats_path}")


if __name__ == "__main__":
    main()
