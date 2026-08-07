# Cloud Deployment — Architecture and Build Record

Deployment of the selected model (XGBoost + class weights, threshold
t=0.99, per E3/E4) to AWS. Region: eu-central-1. Post-July-2025 Free Plan.

## 1. Architecture

| Layer | Service | Configuration |
|---|---|---|
| Storage | S3 | Private, SSE-S3, versioning on, TLS enforced |
| Compute | Lambda | Python 3.12, arm64 (Graviton), 1024MB, 30s |
| Network | VPC | Two private subnets, no IGW, no NAT |
| AWS access | VPC gateway endpoint | S3 only, free |
| Identity | IAM role | Inline s3:GetObject on one prefix |
| Observability | CloudWatch Logs | Structured invocation logging |

## 2. Design decisions

### 2.1 ONNX instead of a pickled model

Loading a pickle deserialises and executes arbitrary Python bytecode. An
attacker able to write to the model object would achieve remote code
execution in the inference environment. ONNX is a data format parsed by a
runtime, not executed, removing the vector entirely. Recorded as a
Tampering mitigation in the threat model.

Size: onnxruntime (~15MB) replaces xgboost + scikit-learn (~300MB),
bringing the package under Lambda's 250MB limit with no container needed.

Verification against the source model on 5,000 held-out samples:
max probability difference 2.384e-07, label agreement 1.000000. The
export script refuses to write artefacts if verification fails.

### 2.2 arm64 (Graviton) instead of x86_64

The development machine is an Apple M1. Building x86_64 would require
QEMU emulation, slowing ML dependency builds substantially. Graviton
gives native builds locally and ~20% lower cost per GB-second.

Environment note: uname -m reports x86_64 despite the M1 chip, because
the shell runs under Rosetta 2 (arch reports i386, Homebrew at
/usr/local). Docker Desktop runs natively regardless, so container builds
are true arm64. This mismatch is a known source of deployment failures.

### 2.3 Function URL instead of API Gateway

API Gateway's free allowance is time-limited under the current Free Plan.
Function URLs are free and support AWS_IAM auth (SigV4-signed requests).

Accepted limitation: forgoes throttling, WAF integration, and usage
plans. Rate limiting must instead use Lambda reserved concurrency, a
coarser control. A cost-driven trade-off, not an equivalent substitute.

### 2.4 S3 gateway endpoint instead of NAT Gateway

The conventional private-subnet Lambda pattern needs a NAT Gateway at
~$32/month, which would exhaust the $200 credit within six months of idle
operation. A VPC gateway endpoint for S3 gives the same private
connectivity at no charge, retaining a subnet with no internet route.

### 2.5 SSE-S3 instead of SSE-KMS

SSE-KMS with a customer-managed key would give rotation control and
per-request audit granularity but costs $1/month per key. SSE-S3 chosen
on cost grounds. Accepted limitation: reduced key management control and
no separate key-usage audit trail.

## 3. Network design

- VPC CIDR 10.0.0.0/16
- Two private subnets (10.0.1.0/24 AZ-a, 10.0.2.0/24 AZ-b); Lambda
  requires multiple AZs for availability
- Route table with no internet gateway and no NAT gateway
- S3 gateway endpoint associated with the route table
- Security group: default allow-all egress revoked, replaced with HTTPS
  (443) to the S3 managed prefix list only

AWS creates security groups permitting unrestricted outbound traffic by
default. Explicitly revoking this constrains data exfiltration paths and
is recorded as an Information Disclosure mitigation.

## 4. Identity and access

Role nids-lambda-execution-role. Trust policy: lambda.amazonaws.com only.

The entire S3 grant is one inline statement allowing s3:GetObject on
arn:aws:s3:::BUCKET/model/* and nothing else.

Deliberately absent: s3:PutObject (a compromised function cannot
overwrite the model, the primary data-poisoning mitigation),
s3:ListBucket (no enumeration), and any wildcard action or resource.

Managed policies attached: AWSLambdaBasicExecutionRole (logs),
AWSLambdaVPCAccessExecutionRole (ENI management).

Known over-permission: both managed policies grant their actions on
Resource *. Replacing them with scoped inline equivalents is identified
as further hardening work.

## 5. Build process

Dependencies are installed inside an arm64 Linux container matching the
Lambda runtime, because onnxruntime ships compiled binaries that cannot
be installed correctly from macOS. Test suites, headers, and bytecode
caches are then stripped, reducing 119MB to a 29MB archive. boto3 is not
bundled, being present in the Lambda runtime already.

## 6. Measured results

### 6.1 Cold start (pre-VPC)

| Metric | Value |
|---|---|
| Init duration | 681 ms |
| Invocation duration | 618 ms |
| Memory used / allocated | 187 MB / 1024 MB |
| Package size | 29 MB zipped |

Memory utilisation at 18% suggests the allocation can be reduced. Lambda
scales CPU proportionally to memory, so this is a latency-versus-cost
trade-off rather than pure waste; measuring the curve is further work.

### 6.2 Functional verification

35 real flows (5 per class) from the held-out test split were sent to the
deployed endpoint: 35/35 correct.

This is a deployment sanity check, not a performance measurement. The
sample was small and non-random (first N per class). Reported model
performance remains the E4 test-set figures (macro F1 0.972). The purpose
was to confirm the deployed artefact matches the validated model.

### 6.3 Observed runtime warnings

onnxruntime logs cpuinfo_initialize failed and Unknown CPU vendor in
Lambda, as the sandbox does not expose /sys/devices/system/cpu/. The
runtime falls back to generic arm64 code paths. Functionally harmless but
may cost inference performance: an observed limitation of running ML in
restricted execution environments.

### 6.4 CloudWatch logging from a VPC Lambda

Predicted that log delivery does not traverse the customer VPC interface
and therefore needs no CloudWatch Logs interface endpoint (~$7/month).
Confirmed in the pre-VPC deployment. TO BE RE-CONFIRMED after VPC
attachment.

## 7. Troubleshooting log

| Issue | Cause | Resolution |
|---|---|---|
| entrypoint requires handler name | Lambda image ENTRYPOINT expects a handler arg | --entrypoint /bin/sh override |
| Role cannot be assumed by Lambda | Role did not exist; console creation silently failed | Recreated via CLI, policies saved to infra/ |
| libomp.dylib not found | macOS ships no OpenMP runtime | brew install libomp, brew link --force |
| SMOTE infeasible at scale | 1,075:1 imbalance implies ~14M synthetic rows | Hybrid undersample + oversample (E1) |

Console-created resources proved unreliable and unverifiable. All
subsequent infrastructure was created via CLI with policy documents
committed to infra/, giving a reproducible and auditable record.

## 8. Status

- [x] Model converted to ONNX and verified
- [x] S3 bucket with four controls
- [x] Least-privilege IAM role
- [x] Lambda deployed on arm64 and functionally verified
- [x] VPC, private subnets, S3 gateway endpoint, restricted SG
- [ ] Lambda attached to VPC and re-verified
- [ ] Function URL with IAM authentication
- [ ] STRIDE threat model against the built architecture
