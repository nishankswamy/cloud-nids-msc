"""
E5: Adversarial evasion testing against the deployed detection model.

The STRIDE threat model recorded model evasion as High severity, unmitigated
and untested. This experiment tests it.

Method
------
Two threat conditions are compared:

  UNCONSTRAINED  every feature may be perturbed. This is the standard
                 adversarial-ML setting and is what most published evasion
                 results measure.

  CONSTRAINED    only features an attacker can actually manipulate may be
                 perturbed. Backward-direction statistics are the victim's
                 responses; TCP flag counts are fixed by protocol semantics;
                 initial window size and minimum segment size come from the
                 sending host's TCP stack; destination port is fixed by the
                 targeted service. None of these are freely chosen by an
                 attacker who still needs the attack to work.

The gap between the two is the finding. Unconstrained results overstate the
operational threat.

Attack algorithm
----------------
Gradient-free, because the deployed artefact is a tree ensemble. For each
flow: sample random perturbation directions within the permitted feature
set at increasing magnitude until the prediction flips to benign, then
binary-search the magnitude to find the minimum perturbation that still
evades. Reports the L2 norm of that minimum perturbation in units of
standard deviations, so magnitudes are comparable across features.

Non-negativity is enforced, since negative durations, counts and lengths
are physically impossible.
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

# Feature indices an attacker can manipulate while keeping the attack functional.
# Forward-direction timing and sizing only.
CONTROLLABLE = {
    1,                              # Flow Duration (extend by idling)
    2, 3,                           # Total Fwd Packets, Total Length of Fwd Packets
    4, 5, 6, 7,                     # Fwd Packet Length max/min/mean/std (padding)
    12, 13,                         # Flow Bytes/s, Flow Packets/s (rate control)
    14, 15, 16, 17,                 # Flow IAT mean/std/max/min
    18, 19, 20, 21, 22,             # Fwd IAT total/mean/std/max/min
    28,                             # Fwd Header Length (option padding)
    30,                             # Fwd Packets/s
    41, 44,                         # Subflow Fwd Bytes, act_data_pkt_fwd
    46, 47, 48,                     # Active mean/max/min
    49, 50, 51,                     # Idle mean/max/min
}

# Fixed: victim responses, protocol semantics, OS stack, target service.
FIXED_REASONS = {
    0: "destination port fixed by targeted service",
    8: "backward packet length set by victim",
    9: "backward packet length set by victim",
    10: "backward packet length set by victim",
    11: "backward packet length set by victim",
    23: "backward timing set by victim",
    24: "backward timing set by victim",
    25: "backward timing set by victim",
    26: "backward timing set by victim",
    27: "backward timing set by victim",
    29: "backward header length set by victim",
    31: "backward packet rate set by victim",
    37: "FIN flag count fixed by protocol semantics",
    38: "PSH flag count fixed by protocol semantics",
    39: "ACK flag count fixed by protocol semantics",
    42: "initial window size set by sending TCP stack",
    43: "initial window size set by receiving TCP stack",
    45: "minimum segment size set by MSS negotiation",
}


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
    """Apply the deployed decision rule, including the E4 Bots threshold."""
    proba = model.predict_proba(X_scaled)
    masked = proba.copy()
    masked[:, bots_idx] = -np.inf
    fallback = masked.argmax(axis=1)
    labels = np.where(proba[:, bots_idx] >= BOTS_THRESHOLD, bots_idx, fallback)
    return labels, proba


def clip_valid(x_scaled, scaler):
    """Project back to physically realisable traffic: no negative durations,
    counts, lengths or rates. Flows violating this are not achievable by any
    attacker, so an 'evasion' relying on them is not a real evasion."""
    raw = x_scaled * scaler.scale_ + scaler.mean_
    raw = np.maximum(raw, 0.0)
    return ((raw - scaler.mean_) / scaler.scale_).astype(np.float32)


def evade_one(model, x_scaled, mask, benign_idx, bots_idx, rng, scaler,
              max_mag=6.0, tries=40):
    """
    Find the smallest perturbation (in scaled units) that makes this flow
    classify as benign. Returns (success, magnitude, perturbed_vector).
    """
    def is_benign(v):
        lab, _ = predict_labels(model, v.reshape(1, -1), bots_idx)
        return lab[0] == benign_idx

    # Escalating random search for any successful direction
    found_mag, found_vec = None, None
    for mag in np.linspace(0.25, max_mag, 24):
        for _ in range(tries):
            delta = rng.normal(0, 1, size=x_scaled.shape) * mask
            norm = np.linalg.norm(delta)
            if norm == 0:
                continue
            cand = clip_valid(x_scaled + (delta / norm) * mag, scaler)
            if is_benign(cand):
                found_mag, found_vec = mag, cand
                break
        if found_mag is not None:
            break

    if found_mag is None:
        return False, None, None

    # Binary search along the successful direction for the minimum magnitude
    direction = (found_vec - x_scaled)
    direction = direction / np.linalg.norm(direction)
    lo, hi = 0.0, found_mag
    best = found_vec
    for _ in range(18):
        mid = (lo + hi) / 2
        cand = clip_valid(x_scaled + direction * mid, scaler)
        if is_benign(cand):
            hi, best = mid, cand
        else:
            lo = mid
    return True, hi, best


def run(n_per_class, conditions):
    os.makedirs(RESULTS, exist_ok=True)
    model, scaler, le, feats, X_te_raw, y_te = build_model()

    classes = list(le.classes_)
    benign_idx = classes.index(BENIGN)
    bots_idx = classes.index("Bots")
    n_feat = len(feats)

    masks = {
        "unconstrained": np.ones(n_feat, dtype=np.float32),
        "constrained": np.array([1.0 if i in CONTROLLABLE else 0.0
                                 for i in range(n_feat)], dtype=np.float32),
    }
    print(f"\nControllable features: {int(masks['constrained'].sum())} of {n_feat}")

    # Sample correctly-classified attack flows only — evading a flow the
    # model already gets wrong proves nothing.
    X_te_s = scaler.transform(X_te_raw).astype(np.float32)
    pred, proba = predict_labels(model, X_te_s, bots_idx)
    correct_attack = (pred == y_te) & (y_te != benign_idx)

    rng = np.random.default_rng(SEED)
    rows, per_feature_use = [], np.zeros(n_feat)

    for cls_idx, cls_name in enumerate(classes):
        if cls_idx == benign_idx:
            continue
        pool = np.flatnonzero(correct_attack & (y_te == cls_idx))
        if len(pool) == 0:
            continue
        take = rng.choice(pool, size=min(n_per_class, len(pool)), replace=False)

        for cond in conditions:
            mask = masks[cond]
            successes, mags = 0, []
            for j in take:
                ok, mag, vec = evade_one(model, X_te_s[j], mask,
                                         benign_idx, bots_idx, rng, scaler)
                if ok:
                    successes += 1
                    mags.append(mag)
                    if cond == "constrained":
                        d = np.abs(vec - X_te_s[j])
                        if d.sum() > 0:
                            per_feature_use += d / d.sum()
            rate = successes / len(take)
            rows.append({
                "attack_class": cls_name,
                "condition": cond,
                "flows_tested": len(take),
                "evaded": successes,
                "evasion_rate": round(rate, 4),
                "median_perturbation": round(float(np.median(mags)), 3) if mags else None,
                "min_perturbation": round(float(np.min(mags)), 3) if mags else None,
            })
            print(f"  {cls_name:<15} {cond:<14} "
                  f"{successes}/{len(take)} evaded ({rate:.1%})"
                  + (f", median L2 {np.median(mags):.2f} sd" if mags else ""))

    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(RESULTS, "e5_evasion.csv"), index=False)

    print("\n" + "=" * 74)
    print("E5 EVASION RESULTS")
    print("=" * 74)
    print(df.to_string(index=False))

    if "constrained" in conditions and "unconstrained" in conditions:
        piv = df.pivot_table(index="attack_class", columns="condition",
                             values="evasion_rate")
        piv["gap"] = piv["unconstrained"] - piv["constrained"]
        print("\nEvasion rate by condition:")
        print(piv.round(3).to_string())

    # Express a representative perturbation in original feature units, so
    # magnitudes can be judged against physically meaningful quantities.
    if per_feature_use.sum() > 0:
        share = per_feature_use / per_feature_use.sum()
        print("\nTypical absolute change at median perturbation (top features):")
        med = df[df.condition == "constrained"]["median_perturbation"].median()
        for i in np.argsort(share)[::-1][:5]:
            if share[i] > 0.001:
                units = med * share[i] * scaler.scale_[i]
                print(f"  {feats[i]:<28} {units:,.4f} (units of that feature)")

    # Express a representative perturbation in original feature units, so
    # magnitudes can be judged against physically meaningful quantities.
    if per_feature_use.sum() > 0:
        share = per_feature_use / per_feature_use.sum()
        print("\nTypical absolute change at median perturbation (top features):")
        med = df[df.condition == "constrained"]["median_perturbation"].median()
        for i in np.argsort(share)[::-1][:5]:
            if share[i] > 0.001:
                units = med * share[i] * scaler.scale_[i]
                print(f"  {feats[i]:<28} {units:,.4f} (units of that feature)")

    if per_feature_use.sum() > 0:
        share = per_feature_use / per_feature_use.sum()
        top = np.argsort(share)[::-1][:10]
        print("\nMost-exploited features under realistic constraints:")
        for i in top:
            if share[i] > 0.001:
                print(f"  {share[i]:6.1%}  {feats[i]}")
        pd.DataFrame({"feature": [feats[i] for i in top],
                      "share": [round(float(share[i]), 4) for i in top]}
                     ).to_csv(os.path.join(RESULTS, "e5_exploited_features.csv"),
                              index=False)

    with open(os.path.join(RESULTS, "e5_config.json"), "w") as f:
        json.dump({
            "controllable_features": [feats[i] for i in sorted(CONTROLLABLE)],
            "fixed_features": {feats[i]: r for i, r in FIXED_REASONS.items()},
            "n_per_class": n_per_class,
            "seed": SEED,
        }, f, indent=2)

    print(f"\nSaved to {RESULTS}/e5_*")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=30,
                    help="flows tested per attack class")
    ap.add_argument("--conditions", nargs="+",
                    default=["constrained", "unconstrained"])
    args = ap.parse_args()
    run(args.n, args.conditions)
