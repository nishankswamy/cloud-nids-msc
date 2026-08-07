# Project Progress Log

## Week 1
- **2026-08-07** — Environment set up (Python 3.13, venv, requirements frozen).
  Repo initialised and pushed to GitHub. Resolved macOS OpenMP dependency
  issue blocking XGBoost (`brew install libomp` + force link).

## Week 1 (cont.)
- **2026-08-07** — Model deployed to AWS Lambda (ARM64/Graviton, eu-central-1).
  Converted XGBoost to ONNX (2.76MB, verified: max prob diff 2.4e-07, 100%
  label agreement). Packaged with onnxruntime via Docker ARM64 build, 29MB ZIP.
  Least-privilege IAM role: s3:GetObject on one prefix + CloudWatch Logs only.
  First invocation: 681ms init, 618ms duration, 187MB/1024MB memory used.
  Troubleshooting: console role creation silently failed; recreated via CLI
  and saved policy documents to infra/ for reproducibility.
