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
## E4 — Decision-threshold tuning for the Bots class

**Date:** 2026-08-07
**Script:** `src/experiment_threshold_tuning.py`
**Config:** XGBoost + class weights (selected in E3), seed 42

### Motivation
E3 left Bots precision at 0.675 — roughly one in three Bots alerts false.
Default argmax prediction offers no control over that trade-off. This
experiment introduces an explicit threshold rule:
`predict Bots if P(Bots) >= t, else argmax over remaining classes`.

### Method
Split changed to 60/20/20 (train/validation/test), stratified. The threshold
was swept and selected on **validation only**; the test set was untouched
until the chosen threshold was applied once. This avoids selecting a
hyperparameter on the data used to report results.

### Test-set results

| Config | Bots P | Bots R | Bots F1 | Macro F1 | Bots FP |
|---|---|---|---|---|---|
| argmax (baseline) | 0.672 | 0.992 | 0.801 | 0.9682 | 189 |
| threshold t=0.99 | **0.742** | 0.939 | **0.829** | **0.9721** | **127** |

### Findings
1. **Thresholding improved precision by 10.4% relative** (0.672 -> 0.742)
   and cut false positives by 33% (189 -> 127 per ~504K flows).
2. **Cost: recall fell 0.992 -> 0.939**, i.e. missed bots rose from 3 to 24
   out of 390.
3. **Macro F1 improved rather than degraded** (0.9682 -> 0.9721), because
   the baseline was over-predicting Bots enough that trimming helped the
   aggregate.
4. **The selection generalised.** Validation-chosen threshold produced
   comparable test behaviour, supporting the split methodology.
5. **No threshold achieved precision >= 0.90** anywhere on the sweep. With
   1,559 training samples, thresholding alone cannot yield a high-precision
   bot detector. This is a data-scarcity ceiling, not a tuning failure.
6. **Optimum pinned at the grid boundary (t=0.99).** Bots F1 was still
   increasing at the highest threshold tested, so the true optimum may lie
   above the sweep range.

### Operational interpretation
At t=0.99 the system raises 127 false Bots alerts per ~504K flows while
detecting 94% of genuine bot traffic. Whether this is acceptable depends on
analyst capacity and the cost asymmetry between a missed bot and a wasted
investigation — a deployment-context decision rather than a modelling one.
The threshold is a tunable operational control, not a fixed property.

### Limitations
- Single seed (42); E3's variance estimates suggest ~+/-0.002 macro F1, but
  threshold stability across seeds was not tested.
- Only the Bots class was tuned. Per-class thresholds for all classes were
  not explored.
- Probability calibration (Platt scaling, isotonic regression) untested;
  the boundary-pinned optimum suggests the model's probabilities may be
  poorly calibrated for this class.

### Artefacts
- `docs/results/e4_threshold_sweep_val.csv`
- `docs/results/e4_test_comparison.csv`
- `docs/results/e4_threshold_curves.png`
- `docs/results/e4_chosen_threshold.json`

---

## E5 — Adversarial evasion testing

**Date:** 2026-08-08
**Script:** `src/experiment_evasion.py`

### Motivation
The STRIDE threat model (extended with MITRE ATLAS categories for ML-specific
threats) recorded model evasion as a High-severity threat, unmitigated and
untested against the deployed artefact. This experiment tests it directly
rather than leaving it as an assessed-but-unverified risk.

### Method
Gradient-free black-box attack against the deployed configuration (XGBoost +
class weighting + the E4 Bots threshold, macro F1 0.9721), since the
artefact is a tree ensemble with no exposed gradient. For each
correctly-classified attack flow in the test set: escalating random search
across perturbation magnitude and direction until the prediction flips to
Normal Traffic, then a binary search along the successful direction for the
minimum L2 magnitude (in standard-deviation units) that still evades.
Perturbed vectors are clipped to non-negative raw values, since negative
durations, counts and lengths cannot occur.

Two conditions, 30 flows per attack class, seed 42:

- **Unconstrained** — all 52 features may be perturbed. The standard
  adversarial-ML setting, and what most published evasion results measure.
- **Constrained** — only the 28 features an attacker can actually
  manipulate (forward-direction timing and sizing) may be perturbed.
  Backward-direction statistics are the victim's responses, TCP flag counts
  are fixed by protocol semantics, initial window size and MSS come from
  the TCP stack, and destination port is fixed by the targeted service —
  none of these are attacker-controlled degrees of freedom.

Only flows the deployed model already classifies correctly were attacked;
evading a flow it already gets wrong would prove nothing.

### Results

| Attack class | Unconstrained evasion | Constrained evasion | Constrained median perturbation (SD units) |
|---|---|---|---|
| Bots | 100% | 100% | 0.007 |
| Brute Force | 100% | 100% | 0.142 |
| DDoS | 100% | 100% | 0.111 |
| DoS | 100% | 70% | 0.896 |
| Port Scanning | 100% | 100% | 0.008 |
| Web Attacks | 100% | 100% | 0.055 |

Full data: `docs/results/e5_evasion.csv`. Most-exploited features under the
realistic (constrained) condition — Flow Duration, Fwd IAT Max, Flow IAT
Mean, Flow IAT Max, Fwd IAT Total — together account for ~22% of total
perturbation weight (`docs/results/e5_exploited_features.csv`).

### Findings
1. **Unconstrained evasion is total.** Every class reaches 100% evasion,
   most at near-zero perturbation (median 0.000-0.004 SD): the model has
   essentially no margin against an attacker with unrestricted feature
   access.
2. **Constrained evasion is still severe.** Five of six classes still reach
   100% evasion using only attacker-realistic features; only DoS drops, to
   70%, at a much larger required perturbation (median 0.896 SD vs <=0.14 SD
   for the rest).
3. **The gap between conditions is the intended finding.** Realistic
   constraints reduce evasion for DoS specifically — large-volume floods
   are harder to disguise by nudging forward-timing features alone — but
   barely touch the other five classes, which evade almost as easily
   whether or not the attacker is realistically constrained.
4. **The exploited features are exactly the timing statistics the model
   relies on for detection** (Flow/Fwd IAT, Flow Duration): the same
   features driving 0.97 macro F1 are the ones an attacker can cheaply
   manipulate, since none of them are set by the victim or the protocol.

### Limitations
- Single seed (42); no variance estimate across seeds as in E3.
- Random-search attack, not a gradient-based or genetic attack — a
  stronger optimiser could find smaller or more reliable perturbations, so
  these numbers are upper bounds on required attacker effort, not lower
  bounds.
- Evaluated offline against cached test-set flows, not against the live
  Lambda endpoint or a real network stack.
- Does not test whether any control other than the ML classifier itself
  (rate limiting, behavioural correlation) would catch the perturbed
  traffic.

### Follow-up
E6 addresses one further weakness directly: E5 perturbs correlated timing
features (Flow Duration, Fwd IAT Total, Flow IAT Mean, etc.) independently,
describing flows that could not physically occur from a single attacker
action.

### Artefacts
- `docs/results/e5_evasion.csv`
- `docs/results/e5_exploited_features.csv`
- `docs/results/e5_config.json`

## E6 — Semantically-consistent evasion

**Date:** 2026-08-08
**Script:** `src/experiment_evasion_semantic.py`

### Motivation
E5's independent-feature perturbation has a methodological weakness: Flow
Duration, Fwd IAT Total and Flow IAT Mean are all derived from the same
packet timestamps, so perturbing them independently describes flows that
could not exist on a real network. The direction of the E5 finding is
sound but its magnitudes are optimistic for the attacker. E6 removes that
weakness by reducing the attack to two physical parameters an attacker
actually controls, with every affected feature recomputed consistently
from them.

### Method
Two attacker knobs, both physically realisable:

- **time_scale** — proportional delay inserted between packets (a
  `sleep()` between sends). All duration/inter-arrival/active/idle
  features scale up by this factor; every per-second rate feature scales
  down by the same factor.
- **pad_bytes** — padding added to forward packets. Forward packet-length
  and forward-byte-total features, and the forward share of
  mixed-direction length statistics, are recomputed accordingly; byte rate
  is adjusted to stay internally consistent with the padded byte count and
  the (possibly stretched) duration.

For each correctly-classified attack flow (50 per class, seed 42), the
minimum `time_scale` that flips the prediction to Normal Traffic was found
by grid search (0.05 steps to 2x, 0.25 steps to 10x, then a geometric sweep
to `max_scale`), at two fixed padding levels: 0 bytes and 20 bytes/packet.
Every candidate flow is internally consistent and physically realisable, so
the result is a lower bound on attacker capability rather than an artefact
of independent perturbation.

### Results

**No padding (pad_bytes = 0), max slowdown searched 50x:**

| Attack class | Evasion rate | Median slowdown | Min slowdown |
|---|---|---|---|
| Bots | 100% | 10.88x | 1.15x |
| Web Attacks | 100% | 11.85x | 1.5x |
| Port Scanning | 96% | 14.65x | 5.5x |
| DoS | 12% | 38.92x | 1.05x |
| Brute Force | 0% | — | — |
| DDoS | 0% | — | — |

**With padding (pad_bytes = 20 bytes/packet), max slowdown searched 1000x:**

| Attack class | Evasion rate | Median slowdown | Min slowdown |
|---|---|---|---|
| Bots | 100% | 9.38x | 1.0x |
| Port Scanning | 100% | 2.88x | 1.0x |
| Web Attacks | 100% | 1.65x | 1.45x |
| DDoS | 88% | 7.0x | 1.0x |
| DoS | 20% | 7.25x | 1.05x |
| Brute Force | 4% | 358.43x | 233.57x |

Full data: `docs/results/e6_semantic_evasion_pad0.0.csv`,
`docs/results/e6_semantic_evasion_pad20.0.csv`.

### Findings
1. **Timing manipulation alone (no padding) evades Bots, Web Attacks and
   Port Scanning almost completely** at a moderate (~11-15x) slowdown, but
   Brute Force and DDoS show 0% evasion within the 50x search range —
   those classes' detection does not hinge on timing alone.
2. **Adding 20 bytes/packet of padding changes which classes are
   vulnerable, not whether evasion is possible overall.** DDoS jumps from
   0% to 88% evasion once padding is available, and the slowdown required
   for Port Scanning and Web Attacks drops sharply (14.65x -> 2.88x,
   11.85x -> 1.65x) — padding and timing are compensating levers, not
   independent risks.
3. **Brute Force is the most robust class in both conditions** (0%, then
   4% evasion, and the one flow that did evade needed a 233-358x
   slowdown) — consistent with Brute Force being the only class with
   perfect precision and recall in the base model (README results table),
   and suggesting its detection relies on features outside this attack's
   two knobs.
4. **Even under the more conservative, physically-realisable model, four
   of six classes remain majority-evadable** (Bots, Web Attacks, Port
   Scanning always; DDoS once padding is available) at slowdowns of
   roughly 1-15x — stretching a one-second flood to somewhere between one
   and fifteen seconds, well within reach of an unhurried attacker.
5. **This is a materially lower-magnitude version of the E5 finding, not a
   contradiction of it.** Requiring realistic, internally-consistent
   perturbations narrows which classes evade easily (Brute Force, and
   DDoS without padding, resist) but does not remove the underlying
   vulnerability for the rest.

### Limitations
- Only two attacker knobs tested (time_scale, pad_bytes); other physically
  realisable manipulations (packet fragmentation, decoy traffic,
  protocol-specific tricks) are untested.
- Grid search over time_scale at one fixed padding level at a time, not a
  joint optimisation over both knobs — the reported slowdowns are not
  necessarily the jointly optimal (potentially smaller) combination.
- Evaluated offline against cached flows, not the live endpoint, as in E5.
- Single seed (42).

### Operational interpretation
Taken with E5, these results confirm the STRIDE assessment of model
evasion as High severity, and close its "untested" status: for most attack
classes, an attacker who can insert delay and/or pad packets — both cheap,
no-exploit-required actions — can evade the deployed classifier at
moderate cost. No mitigation for this is currently deployed (see README,
Security > Gaps).

### Artefacts
- `docs/results/e6_semantic_evasion_pad0.0.csv`, `e6_config_pad0.0.json`
- `docs/results/e6_semantic_evasion_pad20.0.csv`, `e6_config_pad20.0.json`
