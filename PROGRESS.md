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

## Week 2
- **2026-08-08** — Adversarial evasion testing (E5, E6). Confirmed the
  STRIDE-flagged "model evasion" risk empirically rather than leaving it
  assessed-but-untested: black-box random-search attack (E5) found total
  evasion (100%) for 5/6 attack classes under attacker-realistic feature
  constraints; a semantically-consistent follow-up (E6, two physical
  attacker knobs — proportional delay and packet padding) confirmed the
  finding at more conservative magnitudes (median slowdown ~1-15x for the
  vulnerable classes). Brute Force was the one consistently robust class.
- **2026-09-04** — Wrote up E5/E6 in `docs/EXPERIMENTS.md` and updated the
  README's Security > Gaps section, which previously still said evasion
  was "untested" despite the results already being committed.
