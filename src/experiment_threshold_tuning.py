"""
E4: Decision-threshold tuning for the Bots class.

E3 selected XGBoost + class weights (macro F1 0.969), but Bots precision
remains 0.675 - roughly one in three Bots alerts is a false positive.
Default multiclass prediction takes argmax over class probabilities, which
gives no control over that trade-off.

This script introduces an explicit threshold rule for Bots:

    predict Bots  if  P(Bots) >= t
    otherwise     argmax over the remaining classes

Sweeping t traces a precision-recall curve for Bots. The threshold is
selected on a held-out VALIDATION set and only then applied to the test
set, so reported test metrics remain unbiased.

Split: 60% train / 20% validation / 20% test (stratified).
"""
import os
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.utils.class_weight import compute_sample_weight
from sklearn.metrics import (
    precision_score, recall_score, f1_score, classification_report,
    confusion_matrix
)
from xgboost import XGBClassifier

BASE = os.path.join(os.path.dirname(__file__), "..")
PROCESSED = os.path.join(BASE, "data", "processed")
RESULTS = os.path.join(BASE, "docs", "results")
MODELS = os.path.join(BASE, "models")
LABEL_COL = "Attack Type"
TARGET_CLASS = "Bots"
SEED = 42


def load_split():
    df = pd.read_parquet(os.path.join(PROCESSED, "combined_raw.parquet"))
    df = df.drop_duplicates()
    num = df.select_dtypes(include="number").columns
    df[num] = df[num].replace([np.inf, -np.inf], np.nan)
    df = df.dropna()

    feats = [c for c in df.columns if c != LABEL_COL]
    X = df[feats].astype(np.float32)
    le = LabelEncoder()
    y = le.fit_transform(df[LABEL_COL])

    # 60 / 20 / 20
    X_tr, X_tmp, y_tr, y_tmp = train_test_split(
        X, y, test_size=0.4, random_state=SEED, stratify=y)
    X_val, X_te, y_val, y_te = train_test_split(
        X_tmp, y_tmp, test_size=0.5, random_state=SEED, stratify=y_tmp)

    sc = StandardScaler()
    X_tr = sc.fit_transform(X_tr).astype(np.float32)
    X_val = sc.transform(X_val).astype(np.float32)
    X_te = sc.transform(X_te).astype(np.float32)

    print(f"Train {X_tr.shape} | Val {X_val.shape} | Test {X_te.shape}")
    return X_tr, X_val, X_te, y_tr, y_val, y_te, le, feats


def threshold_predict(proba, target_idx, t):
    """Predict target class if P(target) >= t, else argmax over others."""
    masked = proba.copy()
    masked[:, target_idx] = -np.inf
    fallback = masked.argmax(axis=1)
    return np.where(proba[:, target_idx] >= t, target_idx, fallback)


def sweep(proba, y_true, target_idx, thresholds):
    rows = []
    for t in thresholds:
        y_pred = threshold_predict(proba, target_idx, t)
        rows.append({
            "threshold": round(float(t), 4),
            "bots_precision": precision_score(y_true, y_pred, labels=[target_idx],
                                              average="macro", zero_division=0),
            "bots_recall": recall_score(y_true, y_pred, labels=[target_idx],
                                        average="macro", zero_division=0),
            "bots_f1": f1_score(y_true, y_pred, labels=[target_idx],
                                average="macro", zero_division=0),
            "macro_f1": f1_score(y_true, y_pred, average="macro", zero_division=0),
            "false_positives": int(((y_pred == target_idx) & (y_true != target_idx)).sum()),
            "missed": int(((y_pred != target_idx) & (y_true == target_idx)).sum()),
        })
    return pd.DataFrame(rows)


def plot_curves(df, chosen, out_path):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

    ax = axes[0]
    ax.plot(df["bots_recall"], df["bots_precision"], marker=".", lw=1.2)
    sel = df[df["threshold"] == chosen].iloc[0]
    ax.scatter([sel["bots_recall"]], [sel["bots_precision"]], s=90,
               color="crimson", zorder=5,
               label=f"selected t={chosen:.2f}")
    ax.set_xlabel("Recall (Bots)")
    ax.set_ylabel("Precision (Bots)")
    ax.set_title("Precision-Recall trade-off, Bots class (validation)")
    ax.grid(alpha=0.3)
    ax.legend()

    ax = axes[1]
    ax.plot(df["threshold"], df["bots_precision"], label="Bots precision")
    ax.plot(df["threshold"], df["bots_recall"], label="Bots recall")
    ax.plot(df["threshold"], df["macro_f1"], label="Macro F1", ls="--")
    ax.axvline(chosen, color="crimson", ls=":", label=f"selected t={chosen:.2f}")
    ax.set_xlabel("Decision threshold for Bots")
    ax.set_ylabel("Score")
    ax.set_title("Metrics vs threshold (validation)")
    ax.grid(alpha=0.3)
    ax.legend()

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    print(f"Saved plot: {out_path}")


def main():
    os.makedirs(RESULTS, exist_ok=True)
    X_tr, X_val, X_te, y_tr, y_val, y_te, le, feats = load_split()
    target_idx = list(le.classes_).index(TARGET_CLASS)

    print("\nTraining XGBoost + class weights on 60% train split...")
    sw = compute_sample_weight("balanced", y_tr)
    model = XGBClassifier(n_estimators=200, max_depth=8, learning_rate=0.1,
                          eval_metric="mlogloss", n_jobs=-1, random_state=SEED)
    model.fit(X_tr, y_tr, sample_weight=sw)

    # ---- Baseline (argmax, no threshold) on test, for reference ----
    base_pred = model.predict(X_te)
    base = {
        "bots_precision": precision_score(y_te, base_pred, labels=[target_idx],
                                          average="macro", zero_division=0),
        "bots_recall": recall_score(y_te, base_pred, labels=[target_idx],
                                    average="macro", zero_division=0),
        "bots_f1": f1_score(y_te, base_pred, labels=[target_idx],
                            average="macro", zero_division=0),
        "macro_f1": f1_score(y_te, base_pred, average="macro", zero_division=0),
        "false_positives": int(((base_pred == target_idx) & (y_te != target_idx)).sum()),
    }

    # ---- Sweep on VALIDATION ----
    print("Sweeping thresholds on validation set...")
    proba_val = model.predict_proba(X_val)
    thresholds = np.concatenate([
        np.arange(0.05, 0.95, 0.05),
        np.arange(0.95, 0.999, 0.005),
        np.array([0.9990, 0.9995, 0.9999, 0.99995]),
    ])
    val_df = sweep(proba_val, y_val, target_idx, thresholds)
    val_df.to_csv(os.path.join(RESULTS, "e4_threshold_sweep_val.csv"), index=False)

    # ---- Selection rules ----
    best_f1_t = float(val_df.loc[val_df["bots_f1"].idxmax(), "threshold"])
    hp = val_df[val_df["bots_precision"] >= 0.90]
    hp_t = float(hp.loc[hp["bots_recall"].idxmax(), "threshold"]) if len(hp) else None

    print("\n" + "=" * 78)
    print("VALIDATION SWEEP (every 4th row)")
    print("=" * 78)
    print(val_df.iloc[::4].to_string(index=False))

    print(f"\nMax Bots F1 on validation at t = {best_f1_t}")
    if hp_t is not None:
        print(f"Highest recall with precision >= 0.90 at t = {hp_t}")
    else:
        print("No threshold reaches precision >= 0.90 on validation.")

    chosen = best_f1_t

    # ---- Apply chosen threshold to TEST (held out throughout) ----
    proba_te = model.predict_proba(X_te)
    tuned_pred = threshold_predict(proba_te, target_idx, chosen)
    tuned = {
        "bots_precision": precision_score(y_te, tuned_pred, labels=[target_idx],
                                          average="macro", zero_division=0),
        "bots_recall": recall_score(y_te, tuned_pred, labels=[target_idx],
                                    average="macro", zero_division=0),
        "bots_f1": f1_score(y_te, tuned_pred, labels=[target_idx],
                            average="macro", zero_division=0),
        "macro_f1": f1_score(y_te, tuned_pred, average="macro", zero_division=0),
        "false_positives": int(((tuned_pred == target_idx) & (y_te != target_idx)).sum()),
    }

    print("\n" + "=" * 78)
    print(f"TEST SET — baseline (argmax) vs tuned (t = {chosen})")
    print("=" * 78)
    comp = pd.DataFrame([
        {"config": "argmax (baseline)", **base},
        {"config": f"threshold t={chosen}", **tuned},
    ])
    print(comp.round(4).to_string(index=False))

    print("\nFull classification report — tuned:")
    print(classification_report(y_te, tuned_pred, target_names=le.classes_,
                                zero_division=0))
    print("Confusion matrix — tuned:")
    print(confusion_matrix(y_te, tuned_pred))

    comp.to_csv(os.path.join(RESULTS, "e4_test_comparison.csv"), index=False)
    with open(os.path.join(RESULTS, "e4_chosen_threshold.json"), "w") as f:
        json.dump({"target_class": TARGET_CLASS, "chosen_threshold": chosen,
                   "high_precision_threshold": hp_t,
                   "selected_on": "validation", "seed": SEED}, f, indent=2)

    plot_curves(val_df, chosen, os.path.join(RESULTS, "e4_threshold_curves.png"))
    print(f"\nSaved: {RESULTS}/e4_*")


if __name__ == "__main__":
    main()
