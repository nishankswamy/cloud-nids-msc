"""
NIDS analyst dashboard.

Local Streamlit interface to the deployed AWS Lambda inference endpoint.
Provides detection review (sample from held-out test data, or upload flows)
and operational telemetry pulled from CloudWatch metrics.

Run:  streamlit run dashboard/app.py
"""
import os
import json
import time
from datetime import datetime, timedelta, timezone

import boto3
import numpy as np
import pandas as pd
import streamlit as st

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROCESSED = os.path.join(BASE, "data", "processed")
FUNCTION = "nids-inference"
REGION = "eu-central-1"
LABEL_COL = "Attack Type"
CLASSES = ["Bots", "Brute Force", "DDoS", "DoS",
           "Normal Traffic", "Port Scanning", "Web Attacks"]

st.set_page_config(page_title="NIDS Analyst Console",
                   page_icon=None, layout="wide")


@st.cache_resource
def clients():
    return (boto3.client("lambda", region_name=REGION),
            boto3.client("cloudwatch", region_name=REGION),
            boto3.client("logs", region_name=REGION))


@st.cache_data(show_spinner="Loading test data...")
def load_test_pool(n=4000):
    """Sample of held-out flows for on-demand scoring."""
    path = os.path.join(PROCESSED, "combined_raw.parquet")
    if not os.path.exists(path):
        return None, None
    df = pd.read_parquet(path).drop_duplicates()
    num = df.select_dtypes(include="number").columns
    df[num] = df[num].replace([np.inf, -np.inf], np.nan)
    df = df.dropna()
    per_class = max(1, n // df[LABEL_COL].nunique())
    parts = []
    for cls, grp in df.groupby(LABEL_COL, sort=False):
        parts.append(grp.sample(min(len(grp), per_class), random_state=42))
    pool = pd.concat(parts, ignore_index=True)
    feats = [c for c in df.columns if c != LABEL_COL]
    return pool, feats


def invoke(features):
    lam, _, _ = clients()
    t0 = time.perf_counter()
    resp = lam.invoke(
        FunctionName=FUNCTION,
        Payload=json.dumps({"features": features}).encode())
    elapsed = (time.perf_counter() - t0) * 1000
    payload = json.loads(resp["Payload"].read())
    if payload.get("statusCode") != 200:
        raise RuntimeError(payload.get("body", "unknown error"))
    return json.loads(payload["body"])["results"], elapsed


def metrics(hours=24):
    _, cw, _ = clients()
    end = datetime.now(timezone.utc)
    start = end - timedelta(hours=hours)
    out = {}
    for name, stat in [("Invocations", "Sum"), ("Errors", "Sum"),
                       ("Duration", "Average"), ("Throttles", "Sum")]:
        r = cw.get_metric_statistics(
            Namespace="AWS/Lambda", MetricName=name,
            Dimensions=[{"Name": "FunctionName", "Value": FUNCTION}],
            StartTime=start, EndTime=end, Period=3600, Statistics=[stat])
        pts = sorted(r["Datapoints"], key=lambda d: d["Timestamp"])
        out[name] = pd.DataFrame(
            [{"time": p["Timestamp"], "value": p[stat]} for p in pts])
    return out


st.title("Network intrusion detection console")
st.caption(f"Inference endpoint: {FUNCTION} ({REGION}) — "
           "XGBoost, class-weighted, Bots threshold 0.99")

tab_detect, tab_ops, tab_about = st.tabs(
    ["Detection", "Operations", "Model"])

with tab_detect:
    left, right = st.columns([1, 2])

    with left:
        st.subheader("Input")
        mode = st.radio("Source", ["Sample held-out flows", "Upload CSV"])
        results, truth, latency = None, None, None

        if mode == "Sample held-out flows":
            pool, feats = load_test_pool()
            if pool is None:
                st.error("combined_raw.parquet not found. "
                         "Run src/load_data.py first.")
            else:
                n = st.slider("Flows to score", 10, 500, 100, step=10)
                only_attacks = st.checkbox("Attack traffic only")
                if st.button("Score flows", type="primary"):
                    sel = pool[pool[LABEL_COL] != "Normal Traffic"] \
                        if only_attacks else pool
                    sample = sel.sample(min(n, len(sel)), random_state=None)
                    X = sample[feats].to_numpy(dtype=np.float32).tolist()
                    with st.spinner("Invoking Lambda..."):
                        results, latency = invoke(X)
                    truth = sample[LABEL_COL].tolist()
                    st.session_state.update(results=results, truth=truth,
                                            latency=latency)
        else:
            up = st.file_uploader("CSV of flow features", type="csv")
            st.caption("52 numeric feature columns, no label column.")
            if up is not None and st.button("Score uploaded flows",
                                            type="primary"):
                df = pd.read_csv(up)
                df = df.select_dtypes(include="number")
                if df.shape[1] != 52:
                    st.error(f"Expected 52 features, got {df.shape[1]}")
                else:
                    X = df.to_numpy(dtype=np.float32).tolist()
                    with st.spinner("Invoking Lambda..."):
                        results, latency = invoke(X)
                    st.session_state.update(results=results, truth=None,
                                            latency=latency)

    with right:
        st.subheader("Detections")
        results = st.session_state.get("results")
        truth = st.session_state.get("truth")
        latency = st.session_state.get("latency")

        if not results:
            st.info("Score some flows to see detections.")
        else:
            df = pd.DataFrame(results)
            if truth:
                df.insert(0, "actual", truth)
                df["correct"] = df["actual"] == df["prediction"]

            n = len(df)
            attacks = int(df["is_attack"].sum())
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Flows scored", n)
            c2.metric("Attacks detected", attacks)
            c3.metric("Attack rate", f"{attacks / n:.1%}")
            if truth:
                c4.metric("Accuracy", f"{df['correct'].mean():.1%}")
            else:
                c4.metric("Round-trip", f"{latency:.0f} ms")

            st.bar_chart(df["prediction"].value_counts())

            show_only = st.checkbox("Show attacks only", value=True)
            view = df[df["is_attack"]] if show_only else df
            st.dataframe(view.sort_values("confidence", ascending=False),
                         use_container_width=True, height=340)

            if truth:
                wrong = df[~df["correct"]]
                if len(wrong):
                    st.warning(f"{len(wrong)} misclassified")
                    st.dataframe(wrong, use_container_width=True)
                else:
                    st.success("All flows classified correctly")

            st.download_button("Export detections (CSV)",
                               df.to_csv(index=False),
                               "detections.csv", "text/csv")

with tab_ops:
    st.subheader("Operational telemetry")
    hours = st.selectbox("Window", [3, 6, 24, 72], index=2,
                         format_func=lambda h: f"Last {h}h")
    try:
        m = metrics(hours)
        inv = m["Invocations"]["value"].sum() if len(m["Invocations"]) else 0
        err = m["Errors"]["value"].sum() if len(m["Errors"]) else 0
        thr = m["Throttles"]["value"].sum() if len(m["Throttles"]) else 0
        dur = m["Duration"]["value"].mean() if len(m["Duration"]) else 0

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Invocations", int(inv))
        c2.metric("Errors", int(err))
        c3.metric("Error rate", f"{(err / inv if inv else 0):.2%}")
        c4.metric("Mean duration", f"{dur:.0f} ms")

        for label, key in [("Invocations per hour", "Invocations"),
                           ("Mean duration (ms)", "Duration")]:
            if len(m[key]):
                st.caption(label)
                st.line_chart(m[key].set_index("time")["value"])

        if thr:
            st.error(f"{int(thr)} throttled invocations in window")
    except Exception as e:
        st.error(f"Could not read CloudWatch metrics: {e}")

with tab_about:
    st.subheader("Deployed model")
    meta_path = os.path.join(BASE, "artefacts", "model_meta.json")
    if os.path.exists(meta_path):
        st.json(json.load(open(meta_path)))
    else:
        st.info("model_meta.json not found.")
    st.markdown(
        "Architecture, design decisions and measured results are documented "
        "in `docs/DEPLOYMENT.md`. Experiment record in `docs/EXPERIMENTS.md`.")
