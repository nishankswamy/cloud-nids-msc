"""
Step 3: Preprocessing.
- Cleans infinities/NaNs and duplicates
- Encodes labels (multiclass; also derives a binary Benign/Attack label)
- Drops low-variance / redundant features
- Splits into train/test
- Applies SMOTE to the training set only (never touch the test set) to address class imbalance
"""
import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from imblearn.over_sampling import SMOTE
import joblib

PROCESSED_DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "processed")
LABEL_COL = "Label"
RANDOM_STATE = 42


def clean(df: pd.DataFrame) -> pd.DataFrame:
    df = df.drop_duplicates()
    numeric_cols = df.select_dtypes(include="number").columns
    df[numeric_cols] = df[numeric_cols].replace([np.inf, -np.inf], np.nan)
    df = df.dropna()
    return df


def drop_low_variance_features(df: pd.DataFrame, threshold=0.01) -> pd.DataFrame:
    numeric_cols = df.select_dtypes(include="number").columns
    variances = df[numeric_cols].var()
    low_var_cols = variances[variances < threshold].index.tolist()
    if low_var_cols:
        print(f"Dropping {len(low_var_cols)} low-variance columns: {low_var_cols}")
        df = df.drop(columns=low_var_cols)
    return df


def preprocess(df: pd.DataFrame, use_smote=True):
    df = clean(df)
    df = drop_low_variance_features(df)

    # Binary label: BENIGN vs ATTACK (useful as a first, simpler classification task)
    df["Binary_Label"] = (df[LABEL_COL] != "BENIGN").astype(int)

    feature_cols = [c for c in df.columns if c not in [LABEL_COL, "Binary_Label"]]
    X = df[feature_cols]
    y_multiclass = df[LABEL_COL]
    y_binary = df["Binary_Label"]

    le = LabelEncoder()
    y_multiclass_encoded = le.fit_transform(y_multiclass)

    X_train, X_test, y_train, y_test, yb_train, yb_test = train_test_split(
        X, y_multiclass_encoded, y_binary,
        test_size=0.2, random_state=RANDOM_STATE, stratify=y_multiclass_encoded
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    if use_smote:
        print(f"Pre-SMOTE training class distribution: {np.bincount(y_train)}")
        smote = SMOTE(random_state=RANDOM_STATE, k_neighbors=3)
        X_train_scaled, y_train = smote.fit_resample(X_train_scaled, y_train)
        print(f"Post-SMOTE training class distribution: {np.bincount(y_train)}")

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
    result = preprocess(df)

    out_path = os.path.join(PROCESSED_DATA_DIR, "preprocessed.pkl")
    joblib.dump(result, out_path)
    print(f"\nSaved preprocessed data to: {out_path}")
    print(f"Train shape: {result['X_train'].shape} | Test shape: {result['X_test'].shape}")
