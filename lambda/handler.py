"""
Lambda inference handler for the cloud NIDS.

Loads the ONNX model and scaler parameters from S3 on cold start, caches
them in the execution environment, and scores incoming network flow
feature vectors.

Applies the Bots decision threshold selected on validation in E4 (t=0.99):
predict Bots only if P(Bots) >= t, else argmax over the remaining classes.
"""
import os
import json
import logging

import boto3
import numpy as np
import onnxruntime as ort

logger = logging.getLogger()
logger.setLevel(logging.INFO)

BUCKET = os.environ["MODEL_BUCKET"]
MODEL_KEY = os.environ.get("MODEL_KEY", "model/nids_model.onnx")
SCALER_KEY = os.environ.get("SCALER_KEY", "model/scaler.json")
BOTS_THRESHOLD = float(os.environ.get("BOTS_THRESHOLD", "0.99"))

CLASSES = ["Bots", "Brute Force", "DDoS", "DoS",
           "Normal Traffic", "Port Scanning", "Web Attacks"]
BOTS_IDX = CLASSES.index("Bots")

_session = None
_mean = None
_scale = None


def _load():
    """Cold-start load. Cached across warm invocations."""
    global _session, _mean, _scale
    if _session is not None:
        return

    s3 = boto3.client("s3")
    tmp_model = "/tmp/model.onnx"

    logger.info("Cold start: loading model from s3://%s/%s", BUCKET, MODEL_KEY)
    s3.download_file(BUCKET, MODEL_KEY, tmp_model)
    _session = ort.InferenceSession(tmp_model, providers=["CPUExecutionProvider"])

    obj = s3.get_object(Bucket=BUCKET, Key=SCALER_KEY)
    params = json.loads(obj["Body"].read())
    _mean = np.array(params["mean"], dtype=np.float32)
    _scale = np.array(params["scale"], dtype=np.float32)
    logger.info("Model loaded: %d features", len(_mean))


def _probabilities(raw):
    """onnxruntime returns ZipMap output as a list of dicts by default."""
    if isinstance(raw, list):
        return np.array([[row[k] for k in sorted(row)] for row in raw],
                        dtype=np.float32)
    return np.asarray(raw, dtype=np.float32)


def predict(features):
    _load()
    X = np.asarray(features, dtype=np.float32)
    if X.ndim == 1:
        X = X.reshape(1, -1)
    if X.shape[1] != len(_mean):
        raise ValueError(f"Expected {len(_mean)} features, got {X.shape[1]}")

    X = ((X - _mean) / _scale).astype(np.float32)

    input_name = _session.get_inputs()[0].name
    outputs = _session.run(None, {input_name: X})
    proba = _probabilities(outputs[1])

    masked = proba.copy()
    masked[:, BOTS_IDX] = -np.inf
    fallback = masked.argmax(axis=1)
    labels = np.where(proba[:, BOTS_IDX] >= BOTS_THRESHOLD, BOTS_IDX, fallback)

    return [
        {
            "prediction": CLASSES[int(i)],
            "confidence": round(float(proba[n, int(i)]), 6),
            "is_attack": CLASSES[int(i)] != "Normal Traffic",
        }
        for n, i in enumerate(labels)
    ]


def lambda_handler(event, context):
    try:
        body = event.get("body", event)
        if isinstance(body, str):
            body = json.loads(body)

        features = body.get("features")
        if features is None:
            return {"statusCode": 400,
                    "body": json.dumps({"error": "Missing 'features'"})}

        results = predict(features)
        n_attacks = sum(r["is_attack"] for r in results)
        logger.info("Scored %d flows, %d flagged as attacks",
                    len(results), n_attacks)

        return {
            "statusCode": 200,
            "headers": {"Content-Type": "application/json"},
            "body": json.dumps({"results": results,
                                "flows_scored": len(results),
                                "attacks_detected": n_attacks}),
        }
    except ValueError as e:
        logger.warning("Bad request: %s", e)
        return {"statusCode": 400, "body": json.dumps({"error": str(e)})}
    except Exception:
        logger.exception("Inference failed")
        return {"statusCode": 500,
                "body": json.dumps({"error": "Internal error"})}
