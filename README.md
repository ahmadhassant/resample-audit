# resample-audit

De-biased validity audit and information-gain benchmark for oversamplers and
synthetic minority generators (SMOTE and its ~100 variants, GAN/VAE/diffusion
tabular generators, or anything exposing `fit_resample(X, y)`).

## Why

The standard way to check synthetic minority data compares each synthetic
point against the very sample that generated it — the parents sit in the
reference, so the check cannot fail. `resample-audit` measures validity
against data the generator has never seen (`ER_split`, a consistent estimate
of the population invalidity `1 − V(G)`), and asks the only question that
matters in deployment: does the oversampler beat what class weights or a
threshold move give for free?

A generator **passes the standard** only if

1. **valid** — the upper Wilson bound of `ER_split` is below `er_max`
   (default 0.10), and
2. **informative** — `I(G)` = F1 over the best trivial baseline
   (`no_resample`, `class_weight`, `threshold_move`) is positive,

measured with an honest protocol: split first, resample the training half
only, evaluate on untouched test data.

## Install

```
pip install resample-audit
```

Requires only `numpy` and `scikit-learn`. `imbalanced-learn` is optional
(for the resamplers themselves).

## Use

```python
from imblearn.over_sampling import SMOTE
from resample_audit import audit

report = audit(X, y, SMOTE)        # class, instance, or zero-arg factory
print(report)

report.er_split          # de-biased invalidity estimate (est. 1 - V(G))
report.er_split_ci       # Wilson 95% CI
report.er_naive          # the classical parent-retained check, for contrast
report.info_gain         # F1 over the best trivial baseline
report.delta_brier       # calibration cost (>0 = worse than no_resample)
report.passes_standard   # valid AND informative
```

Command line:

```
resample-audit data.csv --resampler imblearn.over_sampling.SMOTE
resample-audit data.csv --no-benchmark --json
```

CSV = numeric features + a label column (default: last; minority = rarer
value unless `--minority-label` is given).

## What the numbers mean

- `ER_naive` — share of synthetic points whose nearest real neighbour is
  majority, judged against the generator's *own* training sample. Parents
  are retained in the reference, so this is biased low (often ~0).
- `ER_split` — same vote, but the generator sees only half of each class and
  is judged against the *withheld* half. Consistent for `1 − V(G)`; validated
  against held-out ground truth in the paper (corr 0.79–0.93 across
  datasets vs 0.13–0.43 for the naive check).
- `bias_gain` — `ER_split − ER_naive`: the invalidity the classical check
  hides.
- `I(G)` and the deltas — the honest benchmark. PR-AUC/ROC-AUC deltas near
  zero with a positive F1 delta are the signature of a threshold shift in
  disguise; Brier/ECE deltas above zero are the calibration cost.

Distance metric defaults to Hassanat (scale-free, the paper's instrument);
`metric="euclidean"` is faster on large data.

## Cite

Paper in preparation ("Stop oversampling: a validity theory and a de-biased
test", 2026). Earlier instruments: Tarawneh, Hassanat & Altarawneh, "Stop
Oversampling for Class Imbalance Learning: A Review", IEEE Access 2022;
Hassanat et al., "The Jeopardy of Learning from Over-Sampled
Class-Imbalanced Medical Datasets", 2023.

MIT license.
