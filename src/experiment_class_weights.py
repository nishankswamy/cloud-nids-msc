"""
E2: Cost-sensitive learning as an alternative to synthetic oversampling.

Instead of SMOTE-ing minority classes up, train on the original (imbalanced)
distribution and weight the loss so minority-class errors cost more.
Hypothesis: this improves precision on rare classes (esp. Bots) without
the over-generous decision boundary SMOTE produces.

Outputs a comparison against the E1 baseline.
"""
import os
import time
import json
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.utils.class_weight import compute_sample_weight
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report, confusion_matrix
)
from xgboost import XGBClassifier

BASE = os.path.join(os.path.dirname(__file__), "..")
PROCESSED = os.path.join(BASE, "data", "processed")
MODELS = os.path.join(BASE, "models")
RESULTS = os.path.join(BASE, "docs", "results")
LABEL_COL = "Attack Type"
BENIGN = "Normal Traffic"
SEED = 42


def prepare_unbalanced():
    """Same split/scaling as E1, but NO resampling."""
    df = pd.read_parquet(os.path.join(PROCESSED, "combined_raw.parquet"))
    df = df.drop_duplicates()
    num = df.select_dtypes(include="number").columns
    df[num] = df[num].replace([np.inf, -np.inf], np.nan)
    df = df.dropna()

    feats = [c for c in df.columns if c != LABEL_COL]
    X = df[feats].astype(np.float32)
    le = LabelEncoder()
    y = le.fit_transform(df[LABEL_COL])

    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.2, random_state=SEED, stratify=y
    )
    sc = StandardScaler()
    X_tr = sc.fit_transform(X_tr).astype(np.float32)
    X_te = sc.transform(X_te).astype(np.float32)
    print(f"Train: {X_tr.shape} (imbalanced) | Test: {X_te.shape}")
    print(f"Train distribution: {np.bincount(y_tr)}")
    return X_tr, X_te, y_tr, y_te, le, feats


def run():
    os.makedirs(RESULTS, exist_ok=True)
    X_tr, X_te, y_tr, y_te, le, feats = prepare_unbalanced()

    sample_w = compute_sample_weight("balanced", y_tr)

    models = {
        "rf_class_weighted": (
            RandomForestClassifier(n_estimators=200, max_depth=20,
                                   class_weight="balanced",
                                   n_jobs=-1, random_state=SEED),
            {},
        ),
        "xgb_class_weighted": (
            XGBClassifier(n_estimators=200, max_depth=8, learning_rate=0.1,
                          eval_metric="mlogloss", n_jobs=-1, random_state=SEED),
            {"sample_weight": sample_w},
        ),
    }

    rows = []
    for name, (model, fit_kw) in models.items():
        print(f"\nTraining {name}...")
        t0 = time.time()
        model.fit(X_tr, y_tr, **fit_kw)
        elapsed = time.time() - t0
        print(f"  Done in {elapsed:.1f}s")

        joblib.dump(model, os.path.join(MODELS, f"{name}.pkl"))
        y_pred = model.predict(X_te)

        rows.append({
            "model": name,
            "accuracy": accuracy_score(y_te, y_pred),
            "precision_macro": precision_score(y_te, y_pred, average="macro", zero_division=0),
            "recall_macro": recall_score(y_te, y_pred, average="macro", zero_division=0),
            "f1_macro": f1_score(y_te, y_pred, average="macro", zero_division=0),
            "train_secs": round(elapsed, 1),
        })

        report = classification_report(y_te, y_pred, target_names=le.classes_,
                                       zero_division=0, output_dict=True)
        with open(os.path.join(RESULTS, f"{name}_report.json"), "w") as f:
            json.dump(report, f, indent=2)

        print(f"\n--- {name} ---")
        print(classification_report(y_te, y_pred, target_names=le.classes_,
                                    zero_division=0))
        print("Confusion matrix:")
        print(confusion_matrix(y_te, y_pred))

    df = pd.DataFrame(rows).sort_values("f1_macro", ascending=False)
    df.to_csv(os.path.join(RESULTS, "e2_class_weights.csv"), index=False)
    print("\n" + "=" * 70)
    print("E2 RESULTS (cost-sensitive, no oversampling)")
    print("=" * 70)
    print(df.to_string(index=False))
    print(f"\nSaved to {RESULTS}/e2_class_weights.csv")


if __name__ == "__main__":
    run()
