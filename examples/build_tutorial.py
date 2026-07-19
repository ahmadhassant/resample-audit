"""Build examples/tutorial.ipynb and execute it end-to-end so every output
is real. Run:  python build_tutorial.py  (from the examples/ folder)."""
import os
import nbformat as nbf
from nbformat.v4 import new_notebook, new_markdown_cell, new_code_cell

HERE = os.path.dirname(os.path.abspath(__file__))
nb = new_notebook()
C = []


def md(t): C.append(new_markdown_cell(t))
def code(t): C.append(new_code_cell(t))


md("""# resample-audit — tutorial

`resample-audit` asks two questions about any oversampler / synthetic-minority
generator and applies **one standard**:

1. **Is the synthetic minority actually minority?** — measured by `ER_split`, a
   *de-biased* estimate of the population invalidity `1 − V(G)`. The classical
   check (`ER_naive`) leaves the generator's own parent points in the
   reference, so it almost never fails; the split estimator judges synthetic
   points against **withheld real data** the generator never saw.
2. **Does it add information a trivial decision-rule change couldn't?** —
   measured by `I(G)`, the F1 improvement over the best of
   `{no_resample, class_weight, threshold_move}`, on an honest
   split-then-resample protocol, with PR-AUC / ROC-AUC / Brier / ECE deltas.

A generator **passes** only if it is *valid AND informative*. This notebook
walks through both, on data that needs no download.""")

md("""## 1. Install

```bash
pip install resample-audit          # core: numpy + scikit-learn only
pip install imbalanced-learn        # for the SMOTE variants used below
```""")

md("## 2. A dataset with class imbalance\n"
   "We synthesise an imbalanced, *overlapping* problem — the regime where "
   "oversampling is usually deployed.")
code("""import numpy as np
from sklearn.datasets import make_classification

X, y = make_classification(
    n_samples=4000, n_features=20, n_informative=6, n_redundant=2,
    weights=[0.94, 0.06], class_sep=0.8, flip_y=0.03, random_state=0)

n1, n0 = int((y == 1).sum()), int((y == 0).sum())
print(f"n={len(y)}  majority={n0}  minority={n1}  IR={n0/n1:.1f}")""")

md("## 3. One-line audit\n"
   "Pass the data and a resampler (a class, an instance, or a zero-arg "
   "factory). Here, imbalanced-learn's SMOTE.")
code("""from imblearn.over_sampling import SMOTE
from resample_audit import audit

report = audit(X, y, SMOTE, seed=0)
print(report)""")

md("""## 4. Reading the report

The two clauses are independent, and this is the whole point:

- **`ER_naive` ≈ 0** — under the classical parent-retained test, SMOTE's points
  look almost perfectly valid.
- **`ER_split` ≈ 0.8** — judged against withheld real data, ~80% of those
  "minority" points have a *majority* nearest neighbour. That gap
  (`bias_gain`) is the invalidity the classical test hides.
- **`I(G)` < 0** — even so, SMOTE does not beat the best trivial baseline on
  F1; and `ΔPR-AUC ≈ 0` with `ΔBrier > 0` means it only shifted the operating
  point while *worsening* calibration.""")
code("""print(f"ER_naive       {report.er_naive:.3f}  CI {report.er_naive_ci}")
print(f"ER_split       {report.er_split:.3f}  CI {report.er_split_ci}")
print(f"hidden bias    {report.bias_gain:+.3f}   (split - naive)")
print(f"valid?         {report.valid}")
print(f"info gain I(G) {report.info_gain:+.3f}   over '{report.best_baseline}'")
print(f"ΔPR-AUC {report.delta_pr_auc:+.4f}  ΔROC-AUC {report.delta_roc_auc:+.4f}"
      f"  ΔBrier {report.delta_brier:+.4f}  ΔECE {report.delta_ece:+.4f}")
print(f"passes standard? {report.passes_standard}")""")

md("## 5. Comparing several resamplers\n"
   "The verdict is a property of the *problem* (overlap), not of any one "
   "method — so the whole family fails together.")
code("""import pandas as pd
from imblearn.over_sampling import (BorderlineSMOTE, ADASYN, RandomOverSampler)

rows = []
for R in (SMOTE, BorderlineSMOTE, ADASYN, RandomOverSampler):
    r = audit(X, y, R, seed=0)
    rows.append(dict(method=R.__name__, ER_naive=round(r.er_naive, 3),
                     ER_split=round(r.er_split, 3),
                     info_gain=round(r.info_gain, 3),
                     dPR_AUC=round(r.delta_pr_auc, 4),
                     dBrier=round(r.delta_brier, 4),
                     passes=r.passes_standard))
pd.DataFrame(rows).set_index("method")""")

md("""## 6. Why validity depends on overlap

The invalidity `1 − V(G)` is a floor set by class overlap, not sample size.
Where the classes separate, interpolation lands inside genuine minority
territory and the same SMOTE is **valid**; where they overlap, it can't be.""")
code("""for sep, label in [(2.5, "well separated"), (0.6, "heavy overlap")]:
    Xs, ys = make_classification(
        n_samples=4000, n_features=20, n_informative=6, n_redundant=2,
        weights=[0.94, 0.06], class_sep=sep, flip_y=0.0, random_state=1)
    r = audit(Xs, ys, SMOTE, benchmark=False, seed=0)
    print(f"class_sep={sep:<4} ({label:14}) ER_split={r.er_split:.3f}  "
          f"valid={r.valid}")""")

md("## 7. Auditing your own generator\n"
   "Anything with `fit_resample(X, y)` (imbalanced-learn convention), a "
   "`sample(X, y)` method (smote_variants), or a plain callable works. Here a "
   "deliberately bad generator that jitters points around the *majority* "
   "mean — it should be caught.")
code("""def majority_faker(X, y):
    rng = np.random.default_rng(0)
    mu = X[y == 0].mean(0)
    syn = mu + 0.1 * rng.standard_normal((300, X.shape[1]))
    return np.vstack([X, syn]), np.r_[y, np.ones(300)]

r = audit(X, y, lambda: majority_faker, benchmark=False, seed=0)
print(f"majority_faker  ER_split={r.er_split:.3f}  valid={r.valid}  "
      f"(should be ~1.0 / False)")""")

md("""## 8. Command line

```bash
# audit a CSV (numeric features + label column; minority = the rarer value)
resample-audit data.csv --resampler imblearn.over_sampling.SMOTE

# validity instrument only, machine-readable
resample-audit data.csv --no-benchmark --json

# pick the label column and metric
resample-audit data.csv --label-col Class --metric euclidean
```""")

md("""## 9. Interpreting results

- **`ER_split` is the headline.** It estimates the fraction of synthetic
  "minority" that is majority in truth (`1 − V(G)`). Low is good; the Wilson
  CI upper bound must clear `er_max` (default 0.10) to be called *valid*.
- **`bias_gain = ER_split − ER_naive`** is exactly what the classical test
  hides — expect it to grow with the imbalance ratio.
- **A positive `ΔF1` with `ΔPR-AUC ≈ 0` and `ΔBrier > 0`** is the signature of
  a *threshold shift in disguise*: the model's ranking didn't improve, the
  operating point just moved — reproducible for free by `threshold_move`, and
  at a calibration cost.
- **The standard is a bar, not a ban.** A generator earns its place by
  clearing *both* clauses on *your* data. `resample-audit` is how you check.

Distance defaults to Hassanat (scale-free); `metric="euclidean"` is faster on
large data. See the README for the theory and citations.""")

nb["cells"] = C
nb["metadata"] = {"kernelspec": {"name": "python3",
                                 "display_name": "Python 3",
                                 "language": "python"},
                  "language_info": {"name": "python"}}

path = os.path.join(HERE, "tutorial.ipynb")
nbf.write(nb, path)
print("wrote", path, "-- executing...")

from nbconvert.preprocessors import ExecutePreprocessor
ep = ExecutePreprocessor(timeout=300, kernel_name="python3")
ep.preprocess(nb, {"metadata": {"path": HERE}})
nbf.write(nb, path)
print("executed OK ->", path)
