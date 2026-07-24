# resample-audit

[![PyPI](https://img.shields.io/pypi/v/resample-audit.svg)](https://pypi.org/project/resample-audit/)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.21444930.svg)](https://doi.org/10.5281/zenodo.21444930)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

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

A full walkthrough (one-line audit, reading the report, comparing resamplers,
the overlap-vs-separable horns, auditing your own generator) is in
[`examples/tutorial.ipynb`](examples/tutorial.ipynb) — regenerate it with
`python examples/build_tutorial.py`.

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

If you use `resample-audit`, please cite the paper it implements:

Hassanat, A. B., Tarawneh, A. S., & Altarawneh, G. A. (2026). *Synthetic
minority data is redundant or invalid: a data-dependent validity theory and a
de-biased test*. arXiv:2607.20787. https://doi.org/10.48550/arXiv.2607.20787

```bibtex
@article{hassanat2026synthetic,
  author  = {Hassanat, Ahmad B. and Tarawneh, Ahmad S. and Altarawneh, Ghada A.},
  title   = {Synthetic minority data is redundant or invalid: a data-dependent
             validity theory and a de-biased test},
  journal = {arXiv preprint arXiv:2607.20787},
  year    = {2026},
  doi     = {10.48550/arXiv.2607.20787}
}
```

For the software itself: Hassanat, A. B. (2026). *resample-audit: de-biased
validity and information-gain audit for oversamplers*. Zenodo.
https://doi.org/10.5281/zenodo.21444930 (concept DOI — resolves to the latest version)

Earlier instruments: Tarawneh, Hassanat & Altarawneh, "Stop Oversampling for
Class Imbalance Learning: A Review", IEEE Access 2022; Hassanat et al., "The
Jeopardy of Learning from Over-Sampled Class-Imbalanced Medical Datasets", 2023.

MIT license.
