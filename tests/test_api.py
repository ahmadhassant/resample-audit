import numpy as np
import pytest

from resample_audit import audit, AuditReport
from conftest import InterpolatingOversampler, MajorityCloner


def test_end_to_end_report(overlap_world):
    X, y, _, _ = overlap_world
    rep = audit(X, y, InterpolatingOversampler, seed=0)
    assert isinstance(rep, AuditReport)
    assert rep.resampler == "InterpolatingOversampler"
    assert 0.0 <= rep.er_split <= 1.0
    assert rep.er_split_ci[0] <= rep.er_split <= rep.er_split_ci[1]
    assert rep.info_gain is not None
    assert rep.best_baseline in ("no_resample", "class_weight",
                                 "threshold_move")
    for k in ("no_resample", "class_weight", "threshold_move", "oversample"):
        assert 0.0 <= rep.arms[k]["f1"] <= 1.0
    # overlap world: interpolation is invalid -> fails the standard
    assert rep.valid is False
    assert rep.passes_standard is False
    txt = str(rep)
    assert "STANDARD" in txt and "ER_split" in txt
    d = rep.to_dict()
    assert d["passes_standard"] is False


def test_validity_only(separable_world):
    X, y, _, _ = separable_world
    rep = audit(X, y, InterpolatingOversampler, benchmark=False, seed=0)
    assert rep.valid is True             # separable: interpolation is valid
    assert rep.info_gain is None and rep.passes_standard is None
    assert "I(G)" not in str(rep)


def test_cloner_fails_standard(separable_world):
    X, y, _, _ = separable_world
    rep = audit(X, y, MajorityCloner, seed=0)
    assert rep.valid is False
    assert rep.passes_standard is False


def test_accepts_instance_and_factory(overlap_world):
    X, y, _, _ = overlap_world
    r1 = audit(X, y, InterpolatingOversampler(seed=3), benchmark=False)
    r2 = audit(X, y, lambda: InterpolatingOversampler(seed=3),
               benchmark=False)
    assert abs(r1.er_split - r2.er_split) < 1e-9   # same seeds -> identical


def test_label_inference_and_errors(overlap_world):
    X, y, _, _ = overlap_world
    # string labels, minority inferred as the rarer
    ys = np.where(y == 1, "case", "control")
    rep = audit(X, ys, InterpolatingOversampler, benchmark=False)
    assert rep.n_synth > 0
    with pytest.raises(ValueError):
        audit(X, np.zeros(len(X)), InterpolatingOversampler)   # one class
    with pytest.raises(TypeError):
        audit(X, y, object())                                  # no interface
