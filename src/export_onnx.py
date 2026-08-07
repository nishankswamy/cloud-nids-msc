"""
Step: Export the selected model (XGBoost + class weights, per E3/E4) to ONNX.

Rationale for ONNX over a pickled model in production:
  - No deserialisation of executable code at inference (pickle loading
    executes arbitrary bytecode; ONNX is a data format). Mitigates a
    tampering vector in the deployed system.
  - Inference needs only onnxruntime (~15MB), not xgboost + sklearn (~300MB).
  - Portable across runtimes and hardware architectures.

The script converts, then VERIFIES that ONNX predictions match the original
model exactly before saving. An unverified conversion is worse than none.
"""
import os
import json
import numpy as np
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.utils.class_weight import compute_sample_weight
from sklearn.metrics import f1_score
from xgboost import XGBClassifier

import onnxmltools
from onnxmltools.convert.common.data_types import FloatTensorType
import onnxruntime as ort

BASE = os.path.join(os.path.dirname(__file__), "..")
PROCESSED = os.path.join(BASE, "data", "processed")
ARTEFACTS = os.path.join(BASE, "artefacts")
LABEL_COL = "Attack Type"
SEED = 42
BOTS_THRESHOLD = 0.99   # selected on validation in E4


def build():
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
    X_val, X_te, y_val, y_te = train_test_split(
        X_tmp, y_tmp, test_size=0.5, random_state=SEED, stratify=y_tmp)

    scaler = StandardScaler()
    X_tr_s = scaler.fit_transform(X_tr).astype(np.float32)
    X_te_s = scaler.transform(X_te).astype(np.float32)

    print(f"Training on {X_tr_s.shape[0]:,} rows, {X_tr_s.shape[1]} features")
    sw = compute_sample_weight("balanced", y_tr)
    model = XGBClassifier(n_estimators=200, max_depth=8, learning_rate=0.1,
                          eval_metric="mlogloss", n_jobs=-1, random_state=SEED)
    model.fit(X_tr_s, y_tr, sample_weight=sw)
    return model, scaler, le, feats, X_te_s, y_te


def verify(model, onnx_path, X_sample):
    """ONNX output must match the original model's probabilities."""
    sess = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
    input_name = sess.get_inputs()[0].name
    outputs = sess.run(None, {input_name: X_sample})

    onnx_labels = outputs[0]
    raw_proba = outputs[1]
    if isinstance(raw_proba, list):
        onnx_proba = np.array([[row[k] for k in sorted(row)] for row in raw_proba],
                              dtype=np.float64)
    else:
        onnx_proba = np.asarray(raw_proba, dtype=np.float64)

    orig_proba = model.predict_proba(X_sample)
    orig_labels = model.predict(X_sample)

    max_diff = float(np.abs(onnx_proba - orig_proba).max())
    label_match = float((onnx_labels.ravel() == orig_labels).mean())

    print(f"\nVerification on {len(X_sample):,} samples")
    print(f"  Max probability difference: {max_diff:.3e}")
    print(f"  Label agreement:            {label_match:.6f}")

    ok = max_diff < 1e-4 and label_match == 1.0
    print("  RESULT:", "PASS" if ok else "FAIL")
    return ok, max_diff, label_match


def main():
    os.makedirs(ARTEFACTS, exist_ok=True)
    model, scaler, le, feats, X_te, y_te = build()

    n_features = X_te.shape[1]
    initial_type = [("input", FloatTensorType([None, n_features]))]
    print("\nConverting to ONNX...")
    onnx_model = onnxmltools.convert_xgboost(model, initial_types=initial_type)

    onnx_path = os.path.join(ARTEFACTS, "nids_model.onnx")
    with open(onnx_path, "wb") as f:
        f.write(onnx_model.SerializeToString())
    size_mb = os.path.getsize(onnx_path) / 1e6
    print(f"Wrote {onnx_path} ({size_mb:.2f} MB)")

    sample = X_te[:5000]
    ok, max_diff, label_match = verify(model, onnx_path, sample)
    if not ok:
        raise SystemExit("ONNX verification FAILED — do not deploy this artefact.")

    # Scaler params travel as plain JSON so inference needs no sklearn
    scaler_path = os.path.join(ARTEFACTS, "scaler.json")
    with open(scaler_path, "w") as f:
        json.dump({"mean": scaler.mean_.tolist(),
                   "scale": scaler.scale_.tolist()}, f)

    meta = {
        "model": "xgboost_class_weighted",
        "selected_in": "E3 (seed stability), threshold from E4",
        "n_features": int(n_features),
        "feature_names": feats,
        "classes": le.classes_.tolist(),
        "bots_class_index": int(list(le.classes_).index("Bots")),
        "bots_threshold": BOTS_THRESHOLD,
        "seed": SEED,
        "onnx_size_mb": round(size_mb, 3),
        "verification": {"max_prob_diff": max_diff, "label_agreement": label_match},
    }
    with open(os.path.join(ARTEFACTS, "model_meta.json"), "w") as f:
        json.dump(meta, f, indent=2)

    joblib.dump(model, os.path.join(BASE, "models", "xgb_selected.pkl"))

    print(f"\nArtefacts written to {ARTEFACTS}/")
    print(f"  nids_model.onnx   {size_mb:.2f} MB")
    print(f"  scaler.json")
    print(f"  model_meta.json")


if __name__ == "__main__":
    main()
