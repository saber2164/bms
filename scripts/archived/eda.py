#!/usr/bin/env python3
"""
Lightweight EDA for cleaned_dataset/data CSVs.

Behavior:
- Reads `cleaned_dataset/metadata.csv` for file-level metadata
- Samples up to `sample_rows` rows from each CSV (default first N rows) to compute statistics
- Produces `outputs/eda/summary.csv` with per-file aggregated stats joined to metadata
- Saves a few plots in `outputs/eda/` (histograms for Voltage/Current/Temperature and Voltage vs Current scatter)

Usage: python scripts/eda.py [--max-files N] [--sample-rows N]

Notes:
- Default sampling avoids reading every row for very large datasets. Adjust flags to suit full processing.
"""
import argparse
import os
import glob
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

ROOT = os.path.dirname(os.path.dirname(__file__))
DATA_DIR = os.path.join(ROOT, "cleaned_dataset", "data")
METADATA = os.path.join(ROOT, "cleaned_dataset", "metadata.csv")
OUT_DIR = os.path.join(ROOT, "outputs", "eda")


def ensure_out():
    os.makedirs(OUT_DIR, exist_ok=True)


def read_metadata():
    if os.path.exists(METADATA):
        try:
            return pd.read_csv(METADATA)
        except Exception:
            # fallback: read raw and parse basic columns
            return pd.read_csv(METADATA, header=0)
    return pd.DataFrame()


def per_file_stats(path, sample_rows=5000):
    # Read a sample (first sample_rows) to keep memory bounded
    try:
        df = pd.read_csv(path, nrows=sample_rows)
    except Exception as e:
        print(f"Failed to read {path}: {e}")
        return None

    numeric = df.select_dtypes(include=[np.number])
    stats = {}
    stats["filename"] = os.path.basename(path)
    stats["rows_sampled"] = len(df)
    if len(numeric.columns) == 0:
        return stats

    for c in numeric.columns:
        col = numeric[c].dropna()
        stats[f"{c}_mean"] = float(col.mean()) if len(col) else np.nan
        stats[f"{c}_std"] = float(col.std()) if len(col) else np.nan
        stats[f"{c}_min"] = float(col.min()) if len(col) else np.nan
        stats[f"{c}_max"] = float(col.max()) if len(col) else np.nan

    # time duration if 'Time' present
    if "Time" in df.columns:
        try:
            stats["time_min"] = float(df["Time"].min())
            stats["time_max"] = float(df["Time"].max())
            stats["time_duration"] = float(stats["time_max"] - stats["time_min"]) if pd.notna(stats.get("time_max")) else np.nan
        except Exception:
            pass

    return stats


def aggregate_and_plot(sampled_values, meta_df):
    # sampled_values: list of DataFrames (each is a sampled df)
    # concat them (may be large; we sampled from each) for global plots
    if len(sampled_values) == 0:
        print("No sampled data collected for plotting.")
        return

    big = pd.concat(sampled_values, ignore_index=True)
    numeric = big.select_dtypes(include=[np.number])

    # Histograms
    cols_to_plot = []
    for c in ["Voltage_measured", "Current_measured", "Temperature_measured"]:
        if c in big.columns:
            cols_to_plot.append(c)

    sns.set(style="darkgrid")
    for c in cols_to_plot:
        plt.figure(figsize=(6,4))
        sns.histplot(big[c].dropna(), bins=80, kde=False)
        plt.title(f"Histogram: {c}")
        plt.tight_layout()
        outp = os.path.join(OUT_DIR, f"hist_{c}.png")
        plt.savefig(outp)
        plt.close()

    # Voltage vs Current scatter (sample if too big)
    if "Voltage_measured" in big.columns and "Current_measured" in big.columns:
        samp = big.sample(min(len(big), 20000))
        plt.figure(figsize=(6,5))
        plt.scatter(samp["Voltage_measured"], samp["Current_measured"], s=1, alpha=0.4)
        plt.xlabel("Voltage_measured")
        plt.ylabel("Current_measured")
        plt.title("Voltage vs Current (sample)")
        plt.tight_layout()
        plt.savefig(os.path.join(OUT_DIR, "voltage_vs_current.png"))
        plt.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-files", type=int, default=None, help="Max files to process (default=all)")
    parser.add_argument("--sample-rows", type=int, default=5000, help="Rows to read per file (default=5000)")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    ensure_out()

    meta = read_metadata()
    files = sorted(glob.glob(os.path.join(DATA_DIR, "*.csv")))
    if args.max_files:
        files = files[: args.max_files]

    summaries = []
    sampled_dfs = []
    print(f"Processing {len(files)} files (sample_rows={args.sample_rows})")
    for i, f in enumerate(files, 1):
        if args.verbose and i % 50 == 0:
            print(f"  file {i}/{len(files)}: {os.path.basename(f)}")

        stats = per_file_stats(f, sample_rows=args.sample_rows)
        if stats is None:
            continue
        summaries.append(stats)

        # also keep sampled frame for global plots
        try:
            sdf = pd.read_csv(f, nrows=args.sample_rows)
            sampled_dfs.append(sdf)
        except Exception:
            pass

    summary_df = pd.DataFrame(summaries)
    # join metadata on filename
    if not meta.empty and "filename" in meta.columns:
        merged = summary_df.merge(meta, left_on="filename", right_on="filename", how="left")
    else:
        merged = summary_df

    out_summary = os.path.join(OUT_DIR, "summary.csv")
    merged.to_csv(out_summary, index=False)
    print(f"Wrote summary to {out_summary}")

    # produce global plots from sampled_dfs
    aggregate_and_plot(sampled_dfs, meta)
    print(f"Plots written to {OUT_DIR}")


if __name__ == "__main__":
    main()
