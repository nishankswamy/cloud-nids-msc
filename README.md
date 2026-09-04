# Cloud-Native Network Intrusion Detection

A machine learning intrusion detection system deployed on AWS with a threat-modelled security architecture. Built as an MSc Cyber Security dissertation project (Arden University, COM7014).

The project does two things: it uses machine learning **for** security, and it secures the machine learning system **itself**. The second half is the part most detection projects skip.

**Macro F1 0.972** across seven traffic classes · **649 ms** cold start on Graviton · deployed in a private subnet with **no internet route** and **no NAT cost**.

---

## Architecture

```
Analyst workstation
        │  SigV4-signed request
        ▼
┌───────────────────────────────────────┐
│  Trust boundary — IAM authentication  │
│         Lambda Function URL           │
└───────────────────┬───────────────────┘
                    ▼
┌───────────────────────────────────────┐
│  Trust boundary — VPC, no internet    │
│                                       │
│   Lambda inference ──▶ S3 gateway     │
│   ONNX, scoped role      endpoint     │
└──────────┬─────────────────┬──────────┘
           ▼                 ▼
    CloudWatch Logs      S3 bucket
    audit trail          encrypted model
```

| Layer | Service | Configuration |
|---|---|---|
| Compute | Lambda | Python 3.12, arm64 Graviton, 1024 MB, 10 s timeout |
| Storage | S3 | Private, SSE-S3, versioning, TLS enforced by bucket policy |
| Network | VPC | Two private subnets, no internet gateway, no NAT gateway |
| AWS access | Gateway endpoint | S3 only, no charge |
| Identity | IAM role | One scoped inline policy (s3:GetObject, single prefix); two AWS managed policies also attached |
| Interface | Function URL | AWS_IAM auth, single named principal |

---

## Results

Model selection was driven by four controlled experiments, not by picking a favourite algorithm.

| Experiment | Question | Outcome |
|---|---|---|
| E1 | Which model family, with hybrid resampling? | XGBoost 0.955 macro F1, ahead of Random Forest and an MLP |
| E2 | Class weighting or synthetic oversampling? | XGBoost improved to 0.969; Random Forest degraded |
| E3 | Is that difference real or noise? | Confirmed across 3 seeds, effect ~10× the variance |
| E4 | Can the weakest class be improved? | Bots precision 0.672 → 0.742 via validation-tuned threshold |

**Final: XGBoost with class weighting, macro F1 0.9721.**

| Class | Precision | Recall | F1 | Test support |
|---|---|---|---|---|
| Bots | 0.74 | 0.94 | 0.83 | 390 |
| Brute Force | 1.00 | 1.00 | 1.00 | 1,830 |
| DDoS | 1.00 | 1.00 | 1.00 | 25,603 |
| DoS | 1.00 | 1.00 | 1.00 | 38,749 |
| Normal Traffic | 1.00 | 1.00 | 1.00 | 418,979 |
| Port Scanning | 0.99 | 1.00 | 0.99 | 18,139 |
| Web Attacks | 0.98 | 0.99 | 0.98 | 428 |

Accuracy is not reported as a headline metric. The test set is 83% benign traffic, so a trivial majority classifier scores 0.83 — macro F1 is the meaningful figure.

### Principal finding: imbalance handling is not model-agnostic

Across three seeds, class weighting improved XGBoost by +0.0128 macro F1 (SD 0.0012) and degraded Random Forest by −0.0149 (SD 0.0020). The direction held on every seed for both families.

The same technique helps one architecture and harms another, consistently. A likely mechanism: Random Forest applies weights at node impurity, so at a 1,075:1 imbalance each minority sample carries ~1,075× weight, letting individual trees claim large regions of feature space from 1,559 real samples. XGBoost applies weights to gradient contributions inside a regularised, depth-limited ensemble, which constrains the effect.

Studies reporting one imbalance technique on one model risk over-generalising.

---

## Security

The system was threat-modelled with STRIDE, extended with MITRE ATLAS categories for machine-learning-specific threats. 18 security requirements trace from an identified threat through to verification evidence.

**Controls verified by test**

| Control | Evidence |
|---|---|
| Endpoint rejects unauthenticated requests | Unsigned request returns HTTP 403; signed request returns 200 |
| No internet route from the inference function | Operates correctly in a subnet with no IGW and no NAT |
| Egress restricted to required destinations | Default allow-all revoked; HTTPS to the S3 prefix list only |
| Least privilege on the execution role | S3 access is one inline, read-only, single-prefix statement — no PutObject, ListBucket, or wildcards |
| Inference cannot modify the model | Role holds object read only — no write, delete, or list |
| Format conversion preserves behaviour | 5,000 samples: max probability delta 2.4e-07, label agreement 1.000000 |

**Design decisions with security rationale**

*ONNX rather than a pickled model.* Loading a pickle deserialises and executes arbitrary bytecode, so write access to the model object would mean remote code execution in the inference environment. ONNX is parsed as data. It also cuts the dependency footprint from ~300 MB to ~15 MB, removing the need for a container image.

*No write permission anywhere in the inference path.* This is the structural defence against model poisoning — not a policy that could be misconfigured, but a capability the role does not have.

*Gateway endpoint rather than NAT.* The textbook private-subnet pattern costs ~$32/month for a NAT gateway. A gateway endpoint gives the same isolation for nothing.

**Gaps recorded rather than hidden**

Model evasion was tested (E5, E6 — see `docs/EXPERIMENTS.md`) and confirmed as the highest-severity open risk, not merely assumed. Under attacker-realistic constraints, five of six attack classes reach total or near-total evasion — most via cheap timing manipulation alone (proportional packet delay, no exploit required), at a median slowdown of roughly 1–15x depending on class. Brute Force is the one consistently robust class. No mitigation for this is currently deployed. Per-caller rate limiting is absent, a consequence of choosing a Function URL over API Gateway on cost grounds. Three EC2 `Describe` actions cannot be resource-scoped — AWS does not support it — and remain wildcarded. The execution role also carries two AWS managed policies (`AWSLambdaBasicExecutionRole`, `AWSLambdaVPCAccessExecutionRole`) whose actions are granted on `Resource: *` rather than scoped to what the function actually needs — a known over-permission on the logging/ENI-management side of the role, distinct from the tightly-scoped S3 grant, and not yet remediated.

---

## Measured behaviour

| Metric | Public | Private subnet |
|---|---|---|
| Cold start | 681 ms | 649 ms |
| Invocation | 618 ms | 626 ms |
| Memory used / allocated | 187 MB / 1024 MB | 187 MB / 1024 MB |

The VPC cold-start penalty widely described in the literature — 8 to 10 seconds while an ENI is provisioned per invocation — did not appear. AWS re-architected Lambda VPC networking in 2019 to use pre-provisioned function-level interfaces; much published guidance still reflects the older behaviour. Network isolation cost nothing measurable in latency and nothing at all in money.

---

## Repository

```
src/          data pipeline, training, four experiments, ONNX export, deployment tests
lambda/       inference handler and arm64 build process
infra/        IAM policy documents, VPC creation script
dashboard/    Streamlit analyst console
docs/         experiment log, deployment record, threat model, requirements, hardening
artefacts/    deployed ONNX model, scaler parameters, metadata
```

Milestones are tagged, giving a timestamped record of development from scaffolding through to the hardened endpoint.

---

## Running it

Requires Python 3.12+, Docker (for the arm64 Lambda build), and AWS CLI credentials.

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

Download CICIDS2017 (cleaned distribution) into `data/raw/`, then:

```bash
python src/load_data.py        # combine and cache as parquet
python src/explore_data.py     # class balance, data quality
python src/preprocess.py       # clean, split, scale, resample
python src/train_models.py     # train three model families
python src/evaluate.py         # compare, per-class report
```

Reproduce the experiments:

```bash
python src/experiment_class_weights.py       # E2
python src/experiment_seed_stability.py      # E3
python src/experiment_threshold_tuning.py    # E4
```

Export and deploy:

```bash
python src/export_onnx.py                    # convert and verify
./infra/create_vpc.sh                        # network layer
python src/test_deployed_model.py            # verify against real flows
```

Analyst console:

```bash
streamlit run dashboard/app.py
```

The dataset is not committed — it is ~684 MB and belongs to the Canadian Institute for Cybersecurity, University of New Brunswick.

---

## Notes on the data

CICIDS2017, cleaned distribution. 2,520,751 flows, 52 features, 7 classes, 1,075:1 imbalance between the largest and smallest.

The cleaned distribution already had duplicates, nulls and low-variance features removed by its publisher; that work is not claimed here. Known limitations of the dataset — label noise reported in the literature, and its age relative to current attack patterns — are discussed in the project's evaluation.

Original dataset: Canadian Institute for Cybersecurity, University of New Brunswick.

---

## Status

Academic project, under active development. The artefact is complete and verified; the accompanying dissertation is in progress.
