"""
Threat register for the NIDS analyst console.

Renders the STRIDE threat model as an interactive risk register: likelihood
and impact scoring, a risk matrix, and a priority-ordered remediation queue.

Scores are qualitative judgements recorded during threat modelling, not
measurements. Where empirical evidence exists (notably the evasion
experiments E5 and E6), the likelihood score reflects it and the evidence
column names the source.
"""
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# likelihood and impact on 1-5 scales; risk = likelihood x impact
THREATS = [
    dict(id="T01", component="Detection model", category="Spoofing",
         threat="Rate-independent attacks evade detection via proportional delay",
         likelihood=5, impact=5, status="Open",
         evidence="E6: Web Attacks 100% evaded at 1.65x slowdown",
         action="Add payload or behavioural features independent of flow timing"),
    dict(id="T02", component="Detection model", category="Spoofing",
         threat="Slow port scanning evades detection",
         likelihood=5, impact=4, status="Open",
         evidence="E6: Port Scanning 100% evaded at 2.88x slowdown",
         action="Cross-flow correlation over longer observation windows"),
    dict(id="T03", component="Detection model", category="Spoofing",
         threat="Bot beaconing evades via jittered long intervals",
         likelihood=4, impact=4, status="Open",
         evidence="E6: Bots 100% evaded at 9.38x slowdown",
         action="Periodicity detection across sessions rather than per-flow"),
    dict(id="T04", component="Endpoint", category="Denial of service",
         threat="Authenticated caller floods the endpoint without rate limiting",
         likelihood=2, impact=3, status="Partially mitigated",
         evidence="Account concurrency quota of 10 bounds blast radius",
         action="Per-caller rate limiting requires API Gateway or equivalent"),
    dict(id="T05", component="Detection model", category="Info disclosure",
         threat="Model extraction through repeated queries enables offline evasion",
         likelihood=2, impact=4, status="Partially mitigated",
         evidence="Single authorised principal; no query rate limiting",
         action="Query budgets per principal; monitor for probing patterns"),
    dict(id="T06", component="Artefact store", category="Repudiation",
         threat="Object-level access to model artefacts not attributable",
         likelihood=2, impact=2, status="Accepted",
         evidence="CloudTrail data events disabled on cost grounds",
         action="Enable data events if artefact integrity becomes contested"),
    dict(id="T07", component="Detection model", category="Tampering",
         threat="Training data poisoning shifts the decision boundary",
         likelihood=1, impact=5, status="Mitigated",
         evidence="No automated retraining; role holds no write permission",
         action="Re-assess if a retraining pipeline is introduced"),
    dict(id="T08", component="Endpoint", category="Spoofing",
         threat="Unauthenticated invocation of the inference endpoint",
         likelihood=1, impact=5, status="Mitigated",
         evidence="Unsigned request returns HTTP 403; verified by test",
         action="None; retest after any endpoint configuration change"),
    dict(id="T09", component="Lambda function", category="Elevation",
         threat="Compromised function reaches other account resources",
         likelihood=1, impact=5, status="Mitigated",
         evidence="Zero managed policies; three resource-scoped inline policies",
         action="Three EC2 Describe actions remain unscopable by platform"),
    dict(id="T10", component="Lambda function", category="Info disclosure",
         threat="Data exfiltrated from the inference environment",
         likelihood=1, impact=4, status="Mitigated",
         evidence="No internet route; egress limited to the S3 prefix list",
         action="None; verify after any security group change"),
    dict(id="T11", component="Artefact store", category="Tampering",
         threat="Model artefact overwritten or corrupted",
         likelihood=1, impact=5, status="Mitigated",
         evidence="Versioning enabled; inference role holds read only",
         action="Consider MFA delete for stronger guarantees"),
    dict(id="T12", component="Network", category="Repudiation",
         threat="Network activity within the VPC not attributable",
         likelihood=2, impact=2, status="Mitigated",
         evidence="VPC Flow Logs enabled, 14-day retention",
         action="None"),
    dict(id="T13", component="Lambda function", category="Tampering",
         threat="Deserialisation of a stored artefact executes attacker code",
         likelihood=1, impact=5, status="Mitigated",
         evidence="ONNX data format replaces pickle; no code execution path",
         action="None"),
    dict(id="T14", component="Endpoint", category="Info disclosure",
         threat="Error responses disclose internal implementation detail",
         likelihood=1, impact=2, status="Mitigated",
         evidence="Generic message returned; detail logged internally",
         action="None"),
    dict(id="T15", component="Detection model", category="Denial of service",
         threat="False positive volume exhausts analyst capacity",
         likelihood=3, impact=2, status="Partially mitigated",
         evidence="127 false Bots alerts per 504,118 flows after E4 tuning",
         action="Confidence floor routing uncertain predictions to review"),
]

BANDS = [(15, "High", "#E24B4A"), (8, "Medium", "#EF9F27"), (0, "Low", "#639922")]


def _band(score):
    for threshold, name, colour in BANDS:
        if score >= threshold:
            return name, colour
    return "Low", "#639922"


def build_frame():
    df = pd.DataFrame(THREATS)
    df["risk"] = df["likelihood"] * df["impact"]
    df[["band", "colour"]] = df["risk"].apply(
        lambda s: pd.Series(_band(s)))
    return df.sort_values("risk", ascending=False).reset_index(drop=True)


def _matrix_figure(df):
    fig, ax = plt.subplots(figsize=(6.2, 5.0))

    for i in range(1, 6):
        for j in range(1, 6):
            _, colour = _band(i * j)
            ax.add_patch(plt.Rectangle((i - 0.5, j - 0.5), 1, 1,
                                       facecolor=colour, alpha=0.16,
                                       edgecolor="white", linewidth=1.5))

    rng = np.random.default_rng(7)
    for _, r in df.iterrows():
        jx = r["likelihood"] + rng.uniform(-0.22, 0.22)
        jy = r["impact"] + rng.uniform(-0.22, 0.22)
        ax.scatter(jx, jy, s=260, color=r["colour"],
                   edgecolor="white", linewidth=1.5, zorder=3)
        ax.text(jx, jy, r["id"][1:], ha="center", va="center",
                fontsize=7.5, color="white", fontweight="bold", zorder=4)

    ax.set_xlim(0.5, 5.5)
    ax.set_ylim(0.5, 5.5)
    ax.set_xticks(range(1, 6))
    ax.set_yticks(range(1, 6))
    ax.set_xlabel("Likelihood", fontsize=10)
    ax.set_ylabel("Impact", fontsize=10)
    ax.set_title("Risk matrix — STRIDE threats", fontsize=11, pad=12)
    ax.grid(False)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.tick_params(length=0)
    fig.tight_layout()
    return fig


def render():
    df = build_frame()

    st.subheader("Threat register")
    st.caption("STRIDE threat model of the deployed system. Likelihood and "
               "impact are qualitative judgements; risk = likelihood x impact.")

    high = df[df.band == "High"]
    open_items = df[df.status == "Open"]
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Threats identified", len(df))
    c2.metric("High risk", len(high))
    c3.metric("Open", len(open_items))
    c4.metric("Mitigated", int((df.status == "Mitigated").sum()))

    if len(open_items):
        top = open_items.iloc[0]
        st.error(f"**Highest open risk — {top['id']}:** {top['threat']}  \n"
                 f"{top['evidence']}")

    left, right = st.columns([1, 1])

    with left:
        st.pyplot(_matrix_figure(df), use_container_width=True)

    with right:
        st.markdown("**Remediation priority**")
        st.caption("Open and partially mitigated threats, highest risk first.")
        queue = df[df.status != "Mitigated"].head(6)
        for _, r in queue.iterrows():
            st.markdown(
                f"<div style='border-left:4px solid {r['colour']};"
                f"padding:6px 10px;margin-bottom:8px;'>"
                f"<b>{r['id']} · {r['band']} ({r['risk']})</b><br>"
                f"<span style='font-size:0.86em'>{r['threat']}</span><br>"
                f"<span style='font-size:0.8em;opacity:0.75'>→ {r['action']}</span>"
                f"</div>", unsafe_allow_html=True)

    st.markdown("---")

    f1, f2, f3 = st.columns(3)
    bands = f1.multiselect("Risk band", ["High", "Medium", "Low"],
                           default=["High", "Medium", "Low"])
    statuses = f2.multiselect("Status", sorted(df.status.unique()),
                              default=sorted(df.status.unique()))
    components = f3.multiselect("Component", sorted(df.component.unique()),
                                default=sorted(df.component.unique()))

    view = df[df.band.isin(bands) & df.status.isin(statuses)
              & df.component.isin(components)]

    st.dataframe(
        view[["id", "component", "category", "threat", "likelihood",
              "impact", "risk", "band", "status", "evidence"]],
        use_container_width=True, height=380, hide_index=True)

    st.download_button("Export threat register (CSV)",
                       df.drop(columns=["colour"]).to_csv(index=False),
                       "threat_register.csv", "text/csv")

    st.caption("Scores for T01 to T03 reflect measured evasion results from "
               "experiments E5 and E6 rather than estimated likelihood.")
