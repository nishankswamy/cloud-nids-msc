"""
E3: Seed stability analysis.

E2 suggested class weighting beats SMOTE for XGBoost (macro F1 0.969 vs
0.955), and that Random Forest degrades under class weighting. Both claims
rest on a single run at seed 42. This script repeats both strategies across
multiple seeds and reports mean +/- std, so differences can be separated
from run-to-run variance.

Usage:
    python src/experiment_seed_stability.py            # 3 seeds (default)
    python src/experiment_seed_stability.py --seeds 5  # 5 seeds
    python src/experiment_seed_stability.py --models xgboost   # xgb only
"""
import os
import time
import json
import argparse
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.utils.class_weight import compute_sample_weight
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
)
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from xgboost import XGBClassifier

BASE = os.path.join(os.path.dirname(__file__), "..")
PROCESSED = os.path.join(BASE, "data", "processed")
RESULTS = os.path.join(BASE, "docs", "results")
LABEL_COL = "Attack Type"
BENIGN = "Normal Traffic"

MAJORITY_CAP = 200_000
MINORITY_TARGET = 50_000

_CACHE = {}


def load_clean():
    """Load and clean once; reused across all seeds."""
    if "df" in _CACHE:
        return _CACHE["df"]
    df = pd.read_parquet(os.path.join(PROCESSED, "combined_raw.parquet"))
    df = df.drop_duplicates()
    num = df.select_dtypes(include="number").columns
    df[num] = df[num].replace([np.inf, -np.inf], np.nan)
    df = df.dropna()
    _CACHE["df"] = df
    return df


def split_and_scale(df, seed):
    feats = [c for c in df.columns if c != LABEL_COL]
    X = df[feats].astype(np.float32)
    le = LabelEncoder()
    y = le.fit_transform(df[LABEL_COL])

    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.2, random_state=seed, stratify=y
    )
    sc = StandardScaler()
    X_tr = sc.fit_transform(X_tr).astype(np.float32)
    X_te = sc.transform(X_te).astype(np.float32)
    return X_tr, X_te, y_tr, y_te, le


def apply_resampling(X_tr, y_tr, seed):
    """E1 strategy: undersample majority, SMOTE minorities."""
    counts = pd.Series(y_tr).value_counts().to_dict()
    under = {c: min(n, MAJORITY_CAP) for c, n in counts.items()}
    over = {c: max(v, MINORITY_TARGET) for c, v in under.items()}

    rus = RandomUnderSampler(sampling_strategy=under, random_state=seed)
    X_r, y_r = rus.fit_resample(X_tr, y_tr)
    sm = SMOTE(sampling_strategy=over, random_state=seed, k_neighbors=5)
    X_r, y_r = sm.fit_resample(X_r, y_r)
    return X_r.astype(np.float32), y_r


def build_model(family, strategy, seed):
    if family == "xgboost":
        return XGBClassifier(n_estimators=200, max_depth=8, learning_rate=0.1,
                             eval_metric="mlogloss", n_jobs=-1, random_state=seed)
    cw = "balanced" if strategy == "class_weights" else None
    return RandomForestClassifier(n_estimators=200, max_depth=20,
                                  class_weight=cw, n_jobs=-1, random_state=seed)


def run_one(family, strategy, seed, le_classes, df):
    X_tr, X_te, y_tr, y_te, le = split_and_scale(df, seed)

    fit_kw = {}
    if strategy == "smote":
        X_tr, y_tr = apply_resampling(X_tr, y_tr, seed)
    elif strategy == "class_weights" and family == "xgboost":
        fit_kw["sample_weight"] = compute_sample_weight("balanced", y_tr)

    model = build_model(family, strategy, seed)
    t0 = time.time()
    model.fit(X_tr, y_tr, **fit_kw)
    elapsed = time.time() - t0

    y_pred = model.predict(X_te)
    bots_idx = list(le.classes_).index("Bots")
    benign_idx = list(le.classes_).index(BENIGN)
    cm = confusion_matrix(y_te, y_pred)

    return {
        "family": family,
        "strategy": strategy,
        "seed": seed,
        "accuracy": accuracy_score(y_te, y_pred),
        "precision_macro": precision_score(y_te, y_pred, average="macro", zero_division=0),
        "recall_macro": recall_score(y_te, y_pred, average="macro", zero_division=0),
        "f1_macro": f1_score(y_te, y_pred, average="macro", zero_division=0),
        "bots_precision": precision_score(y_te, y_pred, labels=[bots_idx],
                                          average="macro", zero_division=0),
        "bots_recall": recall_score(y_te, y_pred, labels=[bots_idx],
                                    average="macro", zero_division=0),
        "benign_as_bots": int(cm[benign_idx, bots_idx]),
        "train_secs": round(elapsed, 1),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--models", nargs="+", default=["xgboost", "random_forest"])
    args = ap.parse_args()

    os.makedirs(RESULTS, exist_ok=True)
    seeds = [42, 7, 123, 2024, 31337][:args.seeds]
    strategies = ["smote", "class_weights"]

    df = load_clean()
    print(f"Data: {df.shape[0]:,} rows")
    print(f"Seeds: {seeds}")
    print(f"Runs: {len(seeds) * len(strategies) * len(args.models)}\n")

    rows = []
    total = len(seeds) * len(strategies) * len(args.models)
    n = 0
    for seed in seeds:
        for family in args.models:
            for strategy in strategies:
                n += 1
                print(f"[{n}/{total}] {family} / {strategy} / seed={seed} ...",
                      end=" ", flush=True)
                r = run_one(family, strategy, seed, None, df)
                rows.append(r)
                print(f"F1={r['f1_macro']:.4f} BotsP={r['bots_precision']:.3f} "
                      f"({r['train_secs']}s)")

                pd.DataFrame(rows).to_csv(
                    os.path.join(RESULTS, "e3_seed_stability_raw.csv"), index=False)

    raw = pd.DataFrame(rows)

    summary = raw.groupby(["family", "strategy"]).agg(
        f1_mean=("f1_macro", "mean"), f1_std=("f1_macro", "std"),
        prec_mean=("precision_macro", "mean"),
        bots_p_mean=("bots_precision", "mean"), bots_p_std=("bots_precision", "std"),
        bots_r_mean=("bots_recall", "mean"),
        benign_as_bots_mean=("benign_as_bots", "mean"),
        train_mean=("train_secs", "mean"),
    ).round(4).reset_index()

    summary.to_csv(os.path.join(RESULTS, "e3_seed_stability_summary.csv"), index=False)

    print("\n" + "=" * 78)
    print(f"E3 SUMMARY — mean +/- std across {len(seeds)} seeds")
    print("=" * 78)
    print(summary.to_string(index=False))

    # Paired comparison: does class weighting beat SMOTE, per seed?
    print("\n" + "=" * 78)
    print("PAIRED COMPARISON (class_weights - smote), per seed")
    print("=" * 78)
    pivot = raw.pivot_table(index=["family", "seed"], columns="strategy",
                            values="f1_macro")
    if {"smote", "class_weights"}.issubset(pivot.columns):
        pivot["delta_f1"] = pivot["class_weights"] - pivot["smote"]
        print(pivot.round(4).to_string())
        print("\nMean delta by model family:")
        for fam, grp in pivot.groupby(level=0):
            d = grp["delta_f1"]
            direction = "class_weights better" if d.mean() > 0 else "SMOTE better"
            consistent = "consistent" if (d > 0).all() or (d < 0).all() else "INCONSISTENT"
            print(f"  {fam:<15} {d.mean():+.4f} (std {d.std():.4f}) "
                  f"-> {direction}, {consistent} across seeds")

    print(f"\nSaved: {RESULTS}/e3_seed_stability_summary.csv")


if __name__ == "__main__":
    main()
