"""
Step 1: Load and combine CICIDS2017 CSV file(s) from data/raw/.
CICIDS2017 is often distributed as multiple daily CSVs (Monday.csv, Tuesday.csv, etc.)
This script combines whatever CSVs are found into a single DataFrame.
"""
import pandas as pd
import glob
import os

RAW_DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "raw")
PROCESSED_DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "processed")


def load_all_csvs(raw_dir=RAW_DATA_DIR):
    csv_files = glob.glob(os.path.join(raw_dir, "*.csv"))
    if not csv_files:
        raise FileNotFoundError(
            f"No CSV files found in {raw_dir}. "
            "Download CICIDS2017 from Kaggle and place CSVs here, "
            "or run generate_synthetic_sample.py first to test the pipeline."
        )

    print(f"Found {len(csv_files)} CSV file(s):")
    for f in csv_files:
        print(f"  - {os.path.basename(f)}")

    dfs = []
    for f in csv_files:
        df = pd.read_csv(f, low_memory=False)
        # CICIDS2017 CSVs sometimes have leading whitespace in column names
        df.columns = df.columns.str.strip()
        dfs.append(df)

    combined = pd.concat(dfs, ignore_index=True)
    print(f"\nCombined shape: {combined.shape}")
    return combined


if __name__ == "__main__":
    os.makedirs(PROCESSED_DATA_DIR, exist_ok=True)
    df = load_all_csvs()
    out_path = os.path.join(PROCESSED_DATA_DIR, "combined_raw.parquet")
    df.to_parquet(out_path, index=False)
    print(f"\nSaved combined raw data to: {out_path}")
