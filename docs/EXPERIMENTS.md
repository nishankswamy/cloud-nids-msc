# Experiment Log

Chronological record of experiments, decisions, and results for the
COM7014 NIDS project. Each entry: what was tried, why, what happened,
what it means.

---

## E1 — Baseline: hybrid resampling (undersample + SMOTE)

**Date:** 2026-08-07
**Commit / tag:** `v0.2-baseline-models`

### Setup
- Dataset: CICIDS2017 (cleaned), 2,520,751 rows x 52 features, 7 classes
- Class imbalance: 1,075:1 (Normal Traffic 2,095,057 vs Bots 1,948)
- Split: 80/20 stratified. Test set left at original imbalanced distribution.
- Scaling: StandardScaler, fit on train only
- Resampling (train only): RandomUnderSampler majority to 200,000,
  then SMOTE minorities up to 50,000
- Training set after resampling: 679,962 rows

**Why hybrid rather than plain SMOTE:** oversampling all classes to the
majority count would generate ~14M synthetic rows (~7GB), infeasible on
available hardware. Combining undersampling with oversampling follows
Chawla et al. (2002), who propose it in the original SMOTE paper.

### Results

| Model | Accuracy | Precision (macro) | Recall (macro) | F1 (macro) | Train (s) | Size |
|---|---|---|---|---|---|---|
| XGBoost | 0.9987 | 0.9291 | 0.9981 | **0.9553** | 40.1 | 3.7 MB |
| Random Forest | 0.9973 | 0.8819 | 0.9964 | 0.9130 | 56.3 | 44 MB |
| MLP (64,32) | 0.9864 | 0.7343 | 0.9925 | 0.7753 | 147.5 | 76 KB |

Per-class, XGBoost:

| Class | Precision | Recall | F1 | Support |
|---|---|---|---|---|
| Bots | 0.57 | 0.99 | 0.72 | 389 |
| Brute Force | 0.99 | 1.00 | 1.00 | 1,830 |
| DDoS | 1.00 | 1.00 | 1.00 | 25,603 |
| DoS | 1.00 | 1.00 | 1.00 | 38,749 |
| Normal Traffic | 1.00 | 1.00 | 1.00 | 418,979 |
| Port Scanning | 0.99 | 1.00 | 0.99 | 18,139 |
| Web Attacks | 0.96 | 1.00 | 0.98 | 429 |

### Observations
1. **Bots precision (0.57) is the weakest result.** 296 Normal Traffic
   flows misclassified as Bots vs 385 true positives — a 43% false
   discovery rate for that class.
2. **Likely cause:** ~97% of Bots training data was synthetic (1,559 real
   samples interpolated to 50,000), producing an over-generous decision
   boundary.
3. **Systematic precision/recall asymmetry** (macro P 0.93 vs macro R 0.998)
   across all oversampled classes — consistent with oversampling bias.
4. **MLP underperforms on every axis**, consistent with the literature on
   tree ensembles outperforming neural networks on tabular data
   (Grinsztajn et al., 2022).
5. **Accuracy is misleading here** — 83% of the test set is Normal Traffic,
   so a trivial majority-class classifier scores 0.83. Macro F1 is the
   appropriate headline metric.

### Next
Compare against cost-sensitive learning (class weights) instead of
synthetic oversampling — see E2.

---
## E2 — Cost-sensitive learning (class weights, no oversampling)

**Date:** 2026-08-07
**Script:** `src/experiment_class_weights.py`

### Setup
Identical split, scaling and seed to E1. No resampling — models trained on
the full imbalanced training set (2,016,472 rows) with class weighting:
- Random Forest: `class_weight="balanced"`
- XGBoost: `sample_weight` from `compute_sample_weight("balanced", y)`

**Hypothesis:** cost-sensitive learning improves minority-class precision
relative to SMOTE, by penalising errors rather than synthesising samples.

### Results vs E1 baseline

| Model | Strategy | Macro P | Macro R | Macro F1 | Bots P | Bots FP | Train (s) |
|---|---|---|---|---|---|---|---|
| XGBoost | SMOTE (E1) | 0.929 | 0.998 | 0.955 | 0.57 | 296 | 40.1 |
| XGBoost | Class weights (E2) | 0.950 | 0.997 | **0.969** | **0.67** | **190** | 115.6 |
| Random Forest | SMOTE (E1) | 0.882 | 0.996 | 0.913 | 0.30 | 914 | 56.3 |
| Random Forest | Class weights (E2) | 0.879 | 0.997 | 0.900 | **0.20** | **1,551** | 127.4 |

*Bots FP = Normal Traffic flows misclassified as Bots.*

### Observations
1. **Hypothesis partially supported — result is model-dependent.**
   XGBoost improved (F1 0.955 -> 0.969; Bots precision 0.57 -> 0.67;
   false positives on Bots down 36%). Random Forest degraded sharply
   (Bots precision 0.57 -> 0.20; false positives up 5.2x).
2. **Proposed mechanism.** RF applies weights to node impurity, so at a
   1,075:1 ratio each Bots sample carries ~1,075x weight, letting individual
   trees claim large regions of feature space from only 1,559 real samples.
   XGBoost applies weights to gradient contributions within a regularised,
   depth-limited boosted ensemble, constraining the same effect.
3. **Implication:** imbalance-handling strategy cannot be selected
   independently of model architecture. Reporting a single technique on a
   single model risks over-generalisation.
4. **Cost:** class weighting trained on ~3x more rows, so training time
   roughly tripled for XGBoost (40s -> 116s). Still trivial at this scale,
   but relevant if retraining frequently in production.
5. **Best configuration so far:** XGBoost + class weights, macro F1 0.969.

### Limitations
- Single random seed (42); differences of ~0.01 F1 are not established as
  significant without repeated runs.
- Bots remains the weakest class in both configurations (389 test samples,
  1,559 training samples) — a data scarcity problem that neither resampling
  nor reweighting fully solves.

### Next
E3: repeated runs across seeds to establish whether the XGBoost gap is
stable; consider precision-recall threshold tuning for Bots specifically.
## E3 — Seed stability analysis

**Date:** 2026-08-07
**Script:** `src/experiment_seed_stability.py --seeds 3`
**Seeds:** 42, 7, 123 (12 runs: 2 families x 2 strategies x 3 seeds)

### Motivation
E2's conclusion rested on a single run per configuration. This repeats both
strategies across seeds to separate genuine effects from run-to-run variance.

### Summary (mean across 3 seeds)

| Family | Strategy | Macro F1 (SD) | Macro P | Bots P (SD) | Bots R | Benign->Bots FP | Train (s) |
|---|---|---|---|---|---|---|---|
| XGBoost | Class weights | **0.9692** (0.0019) | 0.949 | **0.675** (0.013) | 0.992 | 186 | 136 |
| XGBoost | SMOTE | 0.9564 (0.0027) | 0.931 | 0.579 (0.013) | 0.993 | 281 | 46 |
| Random Forest | SMOTE | 0.9129 (0.0009) | 0.883 | 0.296 (0.006) | 0.987 | 914 | 79 |
| Random Forest | Class weights | 0.8980 (0.0025) | 0.878 | 0.189 (0.009) | 0.987 | 173 | 173 |

### Paired comparison (class_weights - smote), per seed

| Family | Seed 42 | Seed 7 | Seed 123 | Mean delta (SD) | Direction |
|---|---|---|---|---|---|
| XGBoost | +0.0140 | +0.0117 | +0.0127 | **+0.0128** (0.0012) | Class weights better, 3/3 |
| Random Forest | -0.0128 | -0.0153 | -0.0166 | **-0.0149** (0.0020) | SMOTE better, 3/3 |

### Findings
1. **A model x strategy interaction effect is confirmed.** Class weighting
   improves XGBoost and degrades Random Forest, consistently, on every seed.
   The technique is not model-agnostic.
2. **Effects exceed noise by roughly an order of magnitude.** XGBoost delta
   +0.0128 against SD 0.0012; RF delta -0.0149 against SD 0.0020. Direction
   held 3/3 for both families.
3. **Seed variance is small overall** (macro F1 SD 0.0009-0.0027). Single-run
   figures in E1/E2 were therefore representative, but this could not be
   known without repetition.
4. **Bots precision is also stable across seeds** (SD 0.006-0.013),
   contrary to the expectation that a class with only 1,559 training samples
   would show high sampling variance. Instability lies in strategy choice,
   not in sampling.
5. **Best configuration: XGBoost + class weights**, macro F1 0.9692,
   Bots precision 0.675, 186 benign flows misclassified as Bots per
   ~504K test flows. Selected for deployment.

### Correction to E2
E1 reported per-class metrics for the best model (XGBoost) only. The Bots
precision of 0.57 quoted for Random Forest + SMOTE in the E2 table was
therefore incorrect; E3 measures it at 0.296. The direction of the RF
degradation under class weighting stands (0.296 -> 0.189), but the
magnitude was overstated.

### Limitations
- Three seeds permits descriptive comparison only; no formal significance
  testing. Five or more seeds would support a paired t-test or Wilcoxon.
- Only two imbalance strategies compared. Others (ADASYN, Tomek links,
  focal loss, threshold tuning) untested.
- Bots precision remains the binding constraint at 0.675 — roughly one in
  three Bots alerts is a false positive.

### Next
E4: precision-recall threshold tuning for the Bots class on the selected
XGBoost + class weights configuration.
