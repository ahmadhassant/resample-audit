"""resample-audit — de-biased validity + information-gain audit for
oversamplers and synthetic minority generators.

Quickstart
----------
>>> from imblearn.over_sampling import SMOTE
>>> from resample_audit import audit
>>> report = audit(X, y, lambda: SMOTE())
>>> print(report)
>>> report.er_split, report.info_gain, report.passes_standard

`audit` measures two things and applies one standard:
  * validity  — ER_split, a consistent, de-biased estimate of the share of
    synthetic "minority" points that are majority in truth (1 - V(G)); the
    classical parent-retained check (ER_naive) is reported alongside to
    show what it hides.
  * information gain — honest split-then-resample benchmark against the
    trivial alternatives (class weights, threshold moving) with F1,
    PR-AUC/ROC-AUC and Brier/ECE deltas.
A generator passes only if it is valid AND adds information.
"""
from __future__ import annotations

import numpy as np

from ._distance import nn_is_majority, wilson_ci
from ._validity import synthetic_minority, validity_audit
from ._benchmark import benchmark_audit
from ._report import AuditReport

__version__ = "0.1.1"
__all__ = ["audit", "AuditReport", "validity_audit", "benchmark_audit",
           "synthetic_minority", "nn_is_majority", "wilson_ci",
           "__version__"]


def _as_factory(resampler):
    """Accept an instance, a class, or a zero-arg factory; return a factory
    producing fresh instances (fresh state per validity split)."""
    if isinstance(resampler, type):
        return resampler, resampler.__name__
    if callable(resampler) and not hasattr(resampler, "fit_resample") \
            and not hasattr(resampler, "sample"):
        try:
            inst = resampler()
        except TypeError as e:
            raise TypeError("resampler factory must be callable with no "
                            "arguments") from e
        return resampler, type(inst).__name__
    # instance: reuse via a fresh clone when sklearn-compatible
    name = type(resampler).__name__
    if hasattr(resampler, "get_params"):
        from sklearn.base import clone
        return (lambda: clone(resampler)), name
    return (lambda: resampler), name


def audit(X, y, resampler, *, minority_label=None, n_splits=5, n_query=1000,
          ref_cap=20000, metric="hassanat", er_max=0.10, benchmark=True,
          clf=None, test_size=0.25, seed=0) -> AuditReport:
    """Audit one resampler on one dataset.

    Parameters
    ----------
    X, y : dataset (numeric features; binary labels).
    resampler : imblearn-style instance/class, or zero-arg factory returning
        one; must expose fit_resample(X, y) or sample(X, y).
    minority_label : which label is the minority; default = the rarer one.
    n_splits, n_query, ref_cap : validity-instrument knobs (paper defaults).
    metric : "hassanat" (paper instrument) or "euclidean" (faster).
    er_max : validity bar — PASS requires the upper Wilson bound of
        ER_split to be below this (default 0.10).
    benchmark : also run the information-gain benchmark (default True).
    clf : sklearn classifier for the benchmark (default LogisticRegression).
    """
    X = np.asarray(X, dtype=float); y = np.asarray(y)
    labs, counts = np.unique(y, return_counts=True)
    if len(labs) != 2:
        raise ValueError(f"need binary labels, got {len(labs)} classes")
    if minority_label is None:
        minority_label = labs[np.argmin(counts)]
    y01 = (y == minority_label).astype(int)
    Xmin, Xmaj = X[y01 == 1], X[y01 == 0]
    if len(Xmin) < 12:
        raise ValueError(f"minority has {len(Xmin)} rows; the split "
                         "instrument needs at least 12")

    factory, name = _as_factory(resampler)
    v = validity_audit(factory, Xmin, Xmaj, n_splits=n_splits,
                       n_query=n_query, ref_cap=ref_cap, metric=metric,
                       seed=seed)
    rep = AuditReport(resampler=name, er_max=er_max, metric=metric, **v)
    if benchmark:
        b = benchmark_audit(factory, X, y01, clf=clf, test_size=test_size,
                            seed=seed)
        rep.arms = b["arms"]; rep.best_baseline = b["best_baseline"]
        rep.info_gain = b["info_gain"]
        rep.delta_pr_auc = b["delta_pr_auc"]
        rep.delta_roc_auc = b["delta_roc_auc"]
        rep.delta_brier = b["delta_brier"]; rep.delta_ece = b["delta_ece"]
    return rep
