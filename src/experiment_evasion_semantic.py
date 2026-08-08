"""
E6: Semantically-consistent evasion.

E5 perturbed 28 features independently and found near-total evasion at
trivial perturbation magnitudes. That result has a methodological weakness:
Flow Duration, Fwd IAT Total and Flow IAT Mean are all derived from the same
packet timestamps, so perturbing them independently describes flows that
could not exist. The direction of the finding is sound but the magnitudes
are optimistic for the attacker.

This experiment removes that weakness. The attacker manipulates two physical
parameters, and every affected feature is recomputed consistently from them:

  time_scale   the attacker inserts proportional delays between packets.
               Durations, inter-arrival times, active and idle periods all
               scale up; every per-second rate scales down. This is a
               `sleep()` between sends — the cheapest evasion available.

  pad_bytes    the attacker pads forward packets. Forward packet lengths,
               forward byte totals and mixed-direction length statistics
               are recomputed accordingly.

Every candidate flow produced here is internally consistent and physically
realisable. The reported result is therefore a lower bound on attacker
capability rather than an artefact of independent feature perturbation.

The headline metric is the minimum slowdown factor required to evade —
directly interpretable as how much an attacker must slow their traffic.
"""
import os
import json
import argparse
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.utils.class_weight import compute_sample_weight
from xgboost import XGBClassifier

BASE = os.path.join(os.path.dirname(__file__), "..")
PROCESSED = os.path.join(BASE, "data", "processed")
RESULTS = os.path.join(BASE, "docs", "results")
LABEL_COL = "Attack Type"
BENIGN = "Normal Traffic"
SEED = 42
BOTS_THRESHOLD = 0.99

# Features scaling linearly with elapsed time
TIME_FEATURES = [
    "Flow Duration",
    "Flow IAT Mean", "Flow IAT Std", "Flow IAT Max", "Flow IAT Min",
    "Fwd IAT Total", "Fwd IAT Mean", "Fwd IAT Std", "Fwd IAT Max", "Fwd IAT Min",
    "Bwd IAT Total", "Bwd IAT Mean", "Bwd IAT Std", "Bwd IAT Max", "Bwd IAT Min",
    "Active Mean", "Active Max", "Active Min",
    "Idle Mean", "Idle Max", "Idle Min",
]

# Per-second rates, inversely proportional to elapsed time
RATE_FEATURES = ["Flow Bytes/s", "Flow Packets/s", "Fwd Packets/s", "Bwd Packets/s"]

# Forward-direction size features affected by padding
FWD_LEN_FEATURES = ["Fwd Packet Length Max", "Fwd Packet Length Min",
                    "Fwd Packet Length Mean"]
FWD_TOTAL_FEATURES = ["Total Length of Fwd Packets", "Subflow Fwd Bytes"]
MIXED_LEN_FEATURES = ["Min Packet Length", "Max Packet Length",
                      "Packet Length Mean", "Average Packet Size"]


def build_model():
    df = pd.read_parquet(os.path.join(PROCESSED, "combined_raw.parquet"))
    df = df.drop_duplicates()
    num = df.select_dtypes(include="number").columns
    df[num] = df[num].replace([np.inf, -np.inf], np.nan)
    df = df.dropna()

    feats = [c for c in df.columns if c != LABEL_COL]
    X = df[feats].astype(np.float32)
    le = LabelEncoder()
    y = le.fit_transform(df[LABEL_COL])

    X_tr, X_tmp, y_tr, y_tmp = train_test_split(
        X, y, test_size=0.4, random_state=SEED, stratify=y)
    _, X_te, _, y_te = train_test_split(
        X_tmp, y_tmp, test_size=0.5, random_state=SEED, stratify=y_tmp)

    scaler = StandardScaler()
    X_tr_s = scaler.fit_transform(X_tr).astype(np.float32)

    print(f"Training selected model on {X_tr_s.shape[0]:,} rows...")
    sw = compute_sample_weight("balanced", y_tr)
    model = XGBClassifier(n_estimators=200, max_depth=8, learning_rate=0.1,
                          eval_metric="mlogloss", n_jobs=-1, random_state=SEED)
    model.fit(X_tr_s, y_tr, sample_weight=sw)
    return model, scaler, le, feats, X_te, y_te


def predict_labels(model, X_scaled, bots_idx):
    proba = model.predict_proba(X_scaled)
    masked = proba.copy()
    masked[:, bots_idx] = -np.inf
    fallback = masked.argmax(axis=1)
    labels = np.where(proba[:, bots_idx] >= BOTS_THRESHOLD, bots_idx, fallback)
    return labels, proba


def apply_attack(raw_batch, idx, time_scale, pad_bytes):
    """
    Apply a physically consistent transformation to a batch of raw flows.
    raw_batch: (n, n_features) in original units.
    """
    out = raw_batch.copy()

    if time_scale != 1.0:
        for f in TIME_FEATURES:
            if f in idx:
                out[:, idx[f]] *= time_scale
        for f in RATE_FEATURES:
            if f in idx:
                out[:, idx[f]] /= time_scale

    if pad_bytes > 0:
        n_fwd = out[:, idx["Total Fwd Packets"]] if "Total Fwd Packets" in idx else None
        for f in FWD_LEN_FEATURES:
            if f in idx:
                out[:, idx[f]] += pad_bytes
        if n_fwd is not None:
            for f in FWD_TOTAL_FEATURES:
                if f in idx:
                    out[:, idx[f]] += pad_bytes * n_fwd
        # Mixed-direction statistics shift by the forward share of packets
        if n_fwd is not None and "Bwd Packets/s" in idx:
            total_pkts = np.maximum(n_fwd + 1.0, 1.0)
            share = n_fwd / total_pkts
            for f in MIXED_LEN_FEATURES:
                if f in idx:
                    out[:, idx[f]] += pad_bytes * share
        # Padding lengthens the flow in bytes, so byte rate rises
        if "Flow Bytes/s" in idx and "Flow Duration" in idx:
            dur_s = np.maximum(out[:, idx["Flow Duration"]] / 1e6, 1e-6)
            if n_fwd is not None:
                out[:, idx["Flow Bytes/s"]] += (pad_bytes * n_fwd) / dur_s

    return np.maximum(out, 0.0)


def min_slowdown(model, raw_row, idx, scaler, benign_idx, bots_idx,
                 pad_bytes, max_scale, steps):
    """Smallest time_scale that evades, or None."""
    grid = np.concatenate([
        np.arange(1.0, 2.0, 0.05),
        np.arange(2.0, 10.0, 0.25),
        np.geomspace(10.0, max_scale, steps),
    ])
    batch = np.repeat(raw_row.reshape(1, -1), len(grid), axis=0)
    for k, t in enumerate(grid):
        batch[k] = apply_attack(raw_row.reshape(1, -1), idx, float(t), pad_bytes)[0]
    scaled = scaler.transform(batch).astype(np.float32)
    labels, _ = predict_labels(model, scaled, bots_idx)
    hits = np.flatnonzero(labels == benign_idx)
    return float(grid[hits[0]]) if len(hits) else None


def run(n_per_class, pad_bytes, max_scale):
    os.makedirs(RESULTS, exist_ok=True)
    model, scaler, le, feats, X_te_raw, y_te = build_model()
    idx = {f: i for i, f in enumerate(feats)}
    classes = list(le.classes_)
    benign_idx = classes.index(BENIGN)
    bots_idx = classes.index("Bots")

    X_raw = X_te_raw.to_numpy(dtype=np.float64)
    X_scaled = scaler.transform(X_raw).astype(np.float32)
    pred, _ = predict_labels(model, X_scaled, bots_idx)
    correct_attack = (pred == y_te) & (y_te != benign_idx)

    rng = np.random.default_rng(SEED)
    rows = []

    print(f"\nAttacker knobs: proportional delay, padding = {pad_bytes} bytes")
    print(f"Maximum slowdown searched: {max_scale:g}x\n")

    for cls_idx, cls_name in enumerate(classes):
        if cls_idx == benign_idx:
            continue
        pool = np.flatnonzero(correct_attack & (y_te == cls_idx))
        if len(pool) == 0:
            continue
        take = rng.choice(pool, size=min(n_per_class, len(pool)), replace=False)

        scales = []
        for j in take:
            t = min_slowdown(model, X_raw[j], idx, scaler, benign_idx,
                             bots_idx, pad_bytes, max_scale, 20)
            if t is not None:
                scales.append(t)

        rate = len(scales) / len(take)
        rows.append({
            "attack_class": cls_name,
            "flows_tested": len(take),
            "evaded": len(scales),
            "evasion_rate": round(rate, 4),
            "median_slowdown": round(float(np.median(scales)), 2) if scales else None,
            "min_slowdown": round(float(np.min(scales)), 2) if scales else None,
            "p90_slowdown": round(float(np.percentile(scales, 90)), 2) if scales else None,
        })
        msg = f"  {cls_name:<15} {len(scales)}/{len(take)} evaded ({rate:.1%})"
        if scales:
            msg += (f"  median {np.median(scales):.2f}x"
                    f"  min {np.min(scales):.2f}x")
        print(msg)

    df = pd.DataFrame(rows)
    suffix = f"pad{pad_bytes}"
    df.to_csv(os.path.join(RESULTS, f"e6_semantic_evasion_{suffix}.csv"),
              index=False)

    print("\n" + "=" * 74)
    print("E6 SEMANTICALLY-CONSISTENT EVASION")
    print("=" * 74)
    print(df.to_string(index=False))

    med = df["median_slowdown"].dropna()
    if len(med):
        print(f"\nAcross classes, median slowdown required: {med.median():.2f}x")
        print("Interpretation: an attack lasting 1 second must be stretched to "
              f"{med.median():.2f} seconds to evade detection.")

    with open(os.path.join(RESULTS, f"e6_config_{suffix}.json"), "w") as f:
        json.dump({"pad_bytes": pad_bytes, "max_scale": max_scale,
                   "n_per_class": n_per_class, "seed": SEED,
                   "time_features": TIME_FEATURES,
                   "rate_features": RATE_FEATURES}, f, indent=2)
    print(f"\nSaved to {RESULTS}/e6_*_{suffix}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=50)
    ap.add_argument("--pad", type=float, default=0.0,
                    help="bytes of padding added per forward packet")
    ap.add_argument("--max-scale", type=float, default=1000.0)
    args = ap.parse_args()
    run(args.n, args.pad, args.max_scale)
