"""
Generates a small synthetic sample matching the CICIDS2017 schema.
Used ONLY to validate the pipeline logic locally before running on the real dataset.
Do NOT use this synthetic data in the actual project report/results.
"""
import numpy as np
import pandas as pd
import os

RAW_DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "raw")

# Subset of real CICIDS2017 feature columns (the real dataset has ~78-80)
FEATURE_COLUMNS = [
    "Destination Port", "Flow Duration", "Total Fwd Packets", "Total Backward Packets",
    "Total Length of Fwd Packets", "Total Length of Bwd Packets",
    "Fwd Packet Length Max", "Fwd Packet Length Min", "Fwd Packet Length Mean",
    "Bwd Packet Length Max", "Bwd Packet Length Min", "Bwd Packet Length Mean",
    "Flow Bytes/s", "Flow Packets/s", "Flow IAT Mean", "Flow IAT Std",
    "Fwd IAT Mean", "Bwd IAT Mean", "Fwd PSH Flags", "SYN Flag Count",
    "ACK Flag Count", "Average Packet Size",
]

LABELS = {
    "BENIGN": 0.80,
    "DoS Hulk": 0.06,
    "PortScan": 0.05,
    "DDoS": 0.04,
    "FTP-Patator": 0.02,
    "SSH-Patator": 0.015,
    "Bot": 0.01,
    "Web Attack – Brute Force": 0.005,
}

def generate(n_rows=20000, seed=42):
    rng = np.random.default_rng(seed)
    labels = rng.choice(list(LABELS.keys()), size=n_rows, p=list(LABELS.values()))

    data = {}
    for col in FEATURE_COLUMNS:
        base = rng.exponential(scale=500, size=n_rows)
        # Give attack rows a shifted distribution so the model has signal to learn
        attack_mask = labels != "BENIGN"
        base[attack_mask] *= rng.uniform(1.5, 4.0, size=attack_mask.sum())
        data[col] = base

    df = pd.DataFrame(data)
    df["Label"] = labels

    # Inject some realistic messiness: missing values, duplicates, infinities
    n_missing = int(n_rows * 0.01)
    missing_idx = rng.choice(n_rows, n_missing, replace=False)
    df.loc[missing_idx, "Flow Bytes/s"] = np.nan

    inf_idx = rng.choice(n_rows, 20, replace=False)
    df.loc[inf_idx, "Flow Packets/s"] = np.inf

    df = pd.concat([df, df.sample(50, random_state=seed)], ignore_index=True)  # duplicates

    return df

if __name__ == "__main__":
    os.makedirs(RAW_DATA_DIR, exist_ok=True)
    df = generate()
    out_path = os.path.join(RAW_DATA_DIR, "synthetic_sample.csv")
    df.to_csv(out_path, index=False)
    print(f"Synthetic sample saved: {out_path}")
    print(f"Shape: {df.shape}")
    print(df["Label"].value_counts())
