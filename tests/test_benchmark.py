import numpy as np

from resample_audit._benchmark import benchmark_audit, _ece, _best_thr
from conftest import InterpolatingOversampler


def test_ece_perfect_and_worst():
    y = np.r_[np.zeros(500), np.ones(500)]
    p = y.astype(float) * 0.999 + 0.0005          # near-perfect confidence
    assert _ece(y, p) < 0.01
    assert _ece(y, 1.0 - p) > 0.9                 # anti-calibrated


def test_best_thr_recovers_separator():
    from sklearn.metrics import f1_score
    rng = np.random.default_rng(0)
    p = np.r_[rng.uniform(0.0, 0.4, 900), rng.uniform(0.6, 1.0, 100)]
    y = (p > 0.5).astype(int)
    t = _best_thr(y, p)
    # 120-point quantile grid: near-perfect recovery (may miss the exact gap)
    assert f1_score(y, (p >= t).astype(int)) >= 0.99


def test_benchmark_arms_and_deltas(overlap_world):
    X, y, _, _ = overlap_world
    b = benchmark_audit(InterpolatingOversampler, X, y, seed=0)
    assert set(b["arms"]) == {"no_resample", "class_weight",
                              "threshold_move", "oversample"}
    ov = b["arms"]["oversample"]
    assert b["delta_brier"] == ov["brier"] - b["arms"]["no_resample"]["brier"]
    assert b["info_gain"] == ov["f1"] - b["arms"][b["best_baseline"]]["f1"]
    for m in b["arms"].values():
        assert 0.0 <= m["f1"] <= 1.0 and 0.0 <= m["pr_auc"] <= 1.0
    # deterministic under the same seed
    b2 = benchmark_audit(InterpolatingOversampler, X, y, seed=0)
    assert b2["info_gain"] == b["info_gain"]


def test_classifier_without_class_weight_skips_that_arm(overlap_world):
    from sklearn.naive_bayes import GaussianNB
    X, y, _, _ = overlap_world
    b = benchmark_audit(InterpolatingOversampler, X, y, clf=GaussianNB(),
                        seed=0)
    assert "class_weight" not in b["arms"]
    assert "threshold_move" in b["arms"]
