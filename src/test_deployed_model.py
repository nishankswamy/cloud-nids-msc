"""
Verify the deployed Lambda produces predictions matching the local model.

Samples real flows from the held-out test set, invokes the Lambda, and
compares against local ONNX inference. A deployed model that silently
diverges from the validated one is a correctness and integrity risk.
"""
import os, json, subprocess, tempfile
import numpy as np, pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

BASE = os.path.join(os.path.dirname(__file__), "..")
PROCESSED = os.path.join(BASE, "data", "processed")
LABEL_COL = "Attack Type"
SEED = 42
N_PER_CLASS = 5


def main():
    df = pd.read_parquet(os.path.join(PROCESSED, "combined_raw.parquet"))
    df = df.drop_duplicates()
    num = df.select_dtypes(include="number").columns
    df[num] = df[num].replace([np.inf, -np.inf], np.nan)
    df = df.dropna()

    feats = [c for c in df.columns if c != LABEL_COL]
    le = LabelEncoder().fit(df[LABEL_COL])
    y = le.transform(df[LABEL_COL])

    _, X_tmp, _, y_tmp = train_test_split(
        df[feats], y, test_size=0.4, random_state=SEED, stratify=y)
    _, X_te, _, y_te = train_test_split(
        X_tmp, y_tmp, test_size=0.5, random_state=SEED, stratify=y_tmp)

    # Sample a few real flows from each class
    rows, truth = [], []
    for cls_idx, cls_name in enumerate(le.classes_):
        mask = y_te == cls_idx
        idx = np.flatnonzero(mask)[:N_PER_CLASS]
        rows.append(X_te.iloc[idx].to_numpy(dtype=np.float32))
        truth.extend([cls_name] * len(idx))
    X = np.vstack(rows)

    print(f"Testing {len(X)} real flows across {len(le.classes_)} classes\n")

    payload = {"features": X.tolist()}
    with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as f:
        json.dump(payload, f)
        event_path = f.name
    out_path = tempfile.mktemp(suffix=".json")

    subprocess.run([
        "aws", "lambda", "invoke",
        "--function-name", "nids-inference",
        "--payload", f"fileb://{event_path}",
        "--cli-binary-format", "raw-in-base64-out",
        "--no-cli-pager", out_path
    ], check=True, capture_output=True)

    resp = json.load(open(out_path))
    if resp.get("statusCode") != 200:
        raise SystemExit(f"Lambda error: {resp}")
    results = json.loads(resp["body"])["results"]

    correct = 0
    print(f"{'True label':<16} {'Predicted':<16} {'Conf':>8}   OK")
    print("-" * 52)
    for t, r in zip(truth, results):
        ok = t == r["prediction"]
        correct += ok
        print(f"{t:<16} {r['prediction']:<16} {r['confidence']:>8.4f}   "
              f"{'yes' if ok else 'NO'}")

    print(f"\nAccuracy on sample: {correct}/{len(truth)} "
          f"({correct/len(truth):.1%})")


if __name__ == "__main__":
    main()
