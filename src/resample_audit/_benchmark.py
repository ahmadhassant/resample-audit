"""Information-gain benchmark: does the oversampler beat what a trivial
decision-rule adjustment gives for free?

Honest protocol (identical to the research harness): split FIRST, resample
the training half only, evaluate on untouched test data. Baselines:
  * no_resample     — the classifier as-is, threshold 0.5
  * class_weight    — cost-sensitive reweighting (no fabricated data)
  * threshold_move  — no_resample probabilities, threshold picked by 3-fold
                      out-of-fold F1 on the training half
Metrics: F1 (operating point), ROC-AUC / PR-AUC (ranking), Brier / ECE
(calibration). I(G) = F1_oversampled - best baseline F1.
"""
from __future__ import annotations

import numpy as np

from ._validity import _call_resampler


def _ece(y, p, bins=10):
    edges = np.linspace(0, 1, bins + 1); e = 0.0
    for i in range(bins):
        hi = p <= edges[i + 1] if i == bins - 1 else p < edges[i + 1]
        m = (p >= edges[i]) & hi
        if m.sum():
            e += m.mean() * abs(y[m].mean() - p[m].mean())
    return float(e)


def _metrics(y, p, thr):
    from sklearn.metrics import (precision_score, recall_score, f1_score,
                                 roc_auc_score, average_precision_score,
                                 brier_score_loss)
    pred = (p >= thr).astype(int)
    return dict(threshold=float(thr),
                precision=float(precision_score(y, pred, zero_division=0)),
                recall=float(recall_score(y, pred, zero_division=0)),
                f1=float(f1_score(y, pred, zero_division=0)),
                roc_auc=float(roc_auc_score(y, p))
                if len(np.unique(y)) > 1 else float("nan"),
                pr_auc=float(average_precision_score(y, p)),
                brier=float(brier_score_loss(y, p)), ece=_ece(y, p))


def _best_thr(y, p):
    from sklearn.metrics import f1_score
    bt, bf = 0.5, -1.0
    for t in np.quantile(p, np.linspace(0.02, 0.98, 120)):
        f = f1_score(y, (p >= t).astype(int), zero_division=0)
        if f > bf:
            bf, bt = f, t
    return float(bt)


def _default_clf():
    from sklearn.linear_model import LogisticRegression
    return LogisticRegression(max_iter=2000)


def _fit_proba(clf_factory, Xtr, ytr, Xte, balanced=False):
    from sklearn.base import clone
    est = clone(clf_factory) if hasattr(clf_factory, "get_params") \
        else clf_factory()
    if balanced:
        params = est.get_params()
        if "class_weight" in params:
            est.set_params(class_weight="balanced")
        elif "scale_pos_weight" in params:      # xgboost
            est.set_params(scale_pos_weight=float(np.sum(ytr == 0)
                                                  / max(1, np.sum(ytr == 1))))
        else:
            return None                          # no balancing knob
    Xtr = np.clip(np.asarray(Xtr, dtype=np.float64), -1e10, 1e10)
    Xte = np.clip(np.asarray(Xte, dtype=np.float64), -1e10, 1e10)
    est.fit(Xtr, ytr)
    return est.predict_proba(Xte)[:, 1]


def benchmark_audit(resampler_factory, X, y, *, clf=None, test_size=0.25,
                    seed=0):
    """Run the honest benchmark. Returns dict with per-arm metrics and the
    information gain I(G) = F1_oversampled - best trivial baseline F1."""
    from sklearn.model_selection import train_test_split, cross_val_predict
    from sklearn.preprocessing import StandardScaler
    from sklearn.base import clone
    X = np.asarray(X, dtype=float); y = np.asarray(y).astype(int)
    clf = clf if clf is not None else _default_clf()
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=test_size,
                                          stratify=y, random_state=seed)
    sc = StandardScaler().fit(Xtr)
    Xtr_s, Xte_s = sc.transform(Xtr), sc.transform(Xte)

    arms = {}
    p = _fit_proba(clf, Xtr_s, ytr, Xte_s)
    arms["no_resample"] = _metrics(yte, p, 0.5)
    pb = _fit_proba(clf, Xtr_s, ytr, Xte_s, balanced=True)
    if pb is not None:
        arms["class_weight"] = _metrics(yte, pb, 0.5)
    oof = cross_val_predict(clone(clf), Xtr_s, ytr, cv=3,
                            method="predict_proba")[:, 1]
    arms["threshold_move"] = _metrics(yte, p, _best_thr(ytr, oof))

    Xb, yb = _call_resampler(resampler_factory(), Xtr_s, ytr)
    arms["oversample"] = _metrics(yte, _fit_proba(clf, Xb, yb.astype(int),
                                                  Xte_s), 0.5)

    base = {k: v for k, v in arms.items() if k != "oversample"}
    best_name = max(base, key=lambda k: base[k]["f1"])
    ov, nb = arms["oversample"], arms[best_name]
    return dict(arms=arms, best_baseline=best_name,
                info_gain=ov["f1"] - nb["f1"],
                delta_pr_auc=ov["pr_auc"] - arms["no_resample"]["pr_auc"],
                delta_roc_auc=ov["roc_auc"] - arms["no_resample"]["roc_auc"],
                delta_brier=ov["brier"] - arms["no_resample"]["brier"],
                delta_ece=ov["ece"] - arms["no_resample"]["ece"])
