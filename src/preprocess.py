"""
Step 3: Preprocessing for CICIDS2017 (cleaned single-file version).

- Cleans infinities/NaNs and duplicates
- Encodes labels (multiclass; also derives a binary Normal/Attack label)
- Drops constant (zero-variance) features
- Splits into train/test (stratified)
- Resamples the TRAINING SET ONLY: undersamples the majority class and
  applies SMOTE to minority classes. Full SMOTE-to-majority is infeasible
  here (1,075:1 imbalance would generate ~14M synthetic rows).
"""
import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
import joblib

PROCESSED_DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "processed")
LABEL_COL = "Attack Type"
BENIGN_VALUE = "Normal Traffic"
RANDOM_STATE = 42

# Resampling targets (tune these if you hit memory limits)
MAJORITY_CAP = 200_000    # undersample Normal Traffic down to this
MINORITY_TARGET = 50_000  # SMOTE minority classes up to this


def clean(df: pd.DataFrame) -> pd.DataFrame:
    before = len(df)
    df = df.drop_duplicates()
    print(f"Dropped {before - len(df):,} duplicate rows")

    numeric_cols = df.select_dtypes(include="number").columns
    df[numeric_cols] = df[numeric_cols].replace([np.inf, -np.inf], np.nan)

    before = len(df)
    df = df.dropna()
    print(f"Dropped {before - len(df):,} rows with NaN/inf")
    return df


def drop_constant_features(df: pd.DataFrame) -> pd.DataFrame:
    """Drop only truly constant columns. Variance thresholds are unreliable
    on unscaled data where features span microseconds to binary flags."""
    numeric_cols = df.select_dtypes(include="number").columns
    constant = [c for c in numeric_cols if df[c].nunique() <= 1]
    if constant:
        print(f"Dropping {len(constant)} constant columns: {constant}")
        df = df.drop(columns=constant)
    return df


def build_resampling_strategies(y_train, label_encoder):
    """Work out per-class targets for undersampling then oversampling."""
    counts = pd.Series(y_train).value_counts().to_dict()
    names = {i: n for i, n in enumerate(label_encoder.classes_)}

    under = {c: min(n, MAJORITY_CAP) for c, n in counts.items()}
    after_under = under.copy()
    over = {c: max(n, MINORITY_TARGET) for c, n in after_under.items()}

    print("\nResampling plan:")
    for c in sorted(counts):
        print(f"  {names[c]:<18} {counts[c]:>9,} -> {under[c]:>8,} -> {over[c]:>8,}")
    return under, over


def preprocess(df: pd.DataFrame, use_resampling=True):
    df = clean(df)
    df = drop_constant_features(df)

    df["Binary_Label"] = (df[LABEL_COL] != BENIGN_VALUE).astype(int)

    feature_cols = [c for c in df.columns if c not in [LABEL_COL, "Binary_Label"]]
    X = df[feature_cols].astype(np.float32)   # halves memory vs float64
    y_multiclass = df[LABEL_COL]
    y_binary = df["Binary_Label"]

    le = LabelEncoder()
    y_encoded = le.fit_transform(y_multiclass)
    print(f"\nClasses: {list(le.classes_)}")

    X_train, X_test, y_train, y_test, yb_train, yb_test = train_test_split(
        X, y_encoded, y_binary,
        test_size=0.2, random_state=RANDOM_STATE, stratify=y_encoded
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train).astype(np.float32)
    X_test_scaled = scaler.transform(X_test).astype(np.float32)

    if use_resampling:
        under_strategy, over_strategy = build_resampling_strategies(y_train, le)

        rus = RandomUnderSampler(sampling_strategy=under_strategy,
                                 random_state=RANDOM_STATE)
        X_train_scaled, y_train = rus.fit_resample(X_train_scaled, y_train)

        smote = SMOTE(sampling_strategy=over_strategy,
                      random_state=RANDOM_STATE, k_neighbors=5)
        X_train_scaled, y_train = smote.fit_resample(X_train_scaled, y_train)

        X_train_scaled = X_train_scaled.astype(np.float32)
        print(f"\nFinal training distribution: {np.bincount(y_train)}")

    return {
        "X_train": X_train_scaled, "X_test": X_test_scaled,
        "y_train": y_train, "y_test": y_test,
        "yb_train": yb_train, "yb_test": yb_test,
        "label_encoder": le, "scaler": scaler,
        "feature_names": feature_cols,
    }


if __name__ == "__main__":
    path = os.path.join(PROCESSED_DATA_DIR, "combined_raw.parquet")
    df = pd.read_parquet(path)
    print(f"Loaded {df.shape[0]:,} rows x {df.shape[1]} columns")

    result = preprocess(df)

    out_path = os.path.join(PROCESSED_DATA_DIR, "preprocessed.pkl")
    joblib.dump(result, out_path)
    print(f"\nSaved preprocessed data to: {out_path}")
    print(f"Train: {result['X_train'].shape} | Test: {result['X_test'].shape}")
