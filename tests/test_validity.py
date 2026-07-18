import numpy as np
import pytest

from resample_audit import validity_audit, synthetic_minority, wilson_ci
from conftest import (InterpolatingOversampler, MajorityCloner,
                      ShufflingOversampler, make_world)


def test_separable_generator_is_valid(separable_world):
    _, _, Xmin, Xmaj = separable_world
    v = validity_audit(InterpolatingOversampler, Xmin, Xmaj, seed=0)
    assert v["er_split"] < 0.05
    assert v["er_naive"] < 0.05
    assert v["splits_used"] == 5


def test_majority_cloner_caught_by_split(separable_world):
    """Points fabricated at the majority mean must be flagged ~100%."""
    _, _, Xmin, Xmaj = separable_world
    v = validity_audit(MajorityCloner, Xmin, Xmaj, seed=0)
    assert v["er_split"] > 0.95


def test_naive_biased_low_under_overlap(overlap_world):
    """The parent-retained reference must under-report invalidity
    (Lemma 0L); the split estimate must sit near the population value
    (~0.55 for delta=1.4, IR=5 — cf. closed form 0.613 at n->inf)."""
    _, _, Xmin, Xmaj = overlap_world
    v = validity_audit(InterpolatingOversampler, Xmin, Xmaj, seed=0)
    assert v["er_naive"] < v["er_split"]
    assert v["bias_gain"] > 0.10
    assert 0.40 < v["er_split"] < 0.80


def test_shuffled_output_fallback(separable_world):
    """Generators that reorder rows go through the set-difference path and
    must yield the same synthetic count."""
    _, _, Xmin, Xmaj = separable_world
    a = synthetic_minority(InterpolatingOversampler(), Xmin, Xmaj)
    b = synthetic_minority(ShufflingOversampler(), Xmin, Xmaj)
    assert a.shape == b.shape
    # same multiset of rows
    sa = sorted(r.tobytes() for r in a.astype(np.float64))
    sb = sorted(r.tobytes() for r in b.astype(np.float64))
    assert sa == sb


def test_euclidean_metric_close_to_hassanat(overlap_world):
    _, _, Xmin, Xmaj = overlap_world
    vh = validity_audit(InterpolatingOversampler, Xmin, Xmaj, seed=0)
    ve = validity_audit(InterpolatingOversampler, Xmin, Xmaj, seed=0,
                        metric="euclidean")
    assert abs(vh["er_split"] - ve["er_split"]) < 0.10


def test_tiny_minority_raises():
    _, _, Xmin, Xmaj = make_world(delta=8.0, n1=8)
    with pytest.raises(ValueError):
        validity_audit(InterpolatingOversampler, Xmin[:5], Xmaj)


def test_wilson_ci_properties():
    lo, hi = wilson_ci(0, 100)
    assert lo == 0.0 and 0.0 < hi < 0.06
    lo, hi = wilson_ci(100, 100)
    assert hi > 0.999 and lo > 0.94
    lo, hi = wilson_ci(50, 100)
    assert lo < 0.5 < hi
    assert np.isnan(wilson_ci(0, 0)[0])
