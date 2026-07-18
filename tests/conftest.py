"""Shared fixtures: synthetic 2-Gaussian worlds + minimal resamplers.

No imbalanced-learn dependency — a 30-line SMOTE stand-in (interpolation
between minority neighbours) and pathological generators exercise every
code path.
"""
import numpy as np
import pytest


class InterpolatingOversampler:
    """Minimal SMOTE: interpolate between a minority point and one of its
    K nearest minority neighbours. Appends synthetic rows (imblearn
    convention)."""

    def __init__(self, k=5, n_syn=400, seed=0):
        self.k, self.n_syn, self.seed = k, n_syn, seed

    def fit_resample(self, X, y):
        rng = np.random.default_rng(self.seed)
        X = np.asarray(X, dtype=float); y = np.asarray(y)
        Xmin = X[y == 1]
        d2 = ((Xmin[:, None, :] - Xmin[None, :, :]) ** 2).sum(-1)
        np.fill_diagonal(d2, np.inf)
        nn = np.argsort(d2, axis=1)[:, :self.k]
        i = rng.integers(0, len(Xmin), self.n_syn)
        j = nn[i, rng.integers(0, self.k, self.n_syn)]
        lam = rng.random((self.n_syn, 1))
        syn = Xmin[i] + lam * (Xmin[j] - Xmin[i])
        return np.vstack([X, syn]), np.r_[y, np.ones(self.n_syn)]


class MajorityCloner:
    """Pathological: labels points drawn at the MAJORITY mean as minority."""

    def fit_resample(self, X, y):
        rng = np.random.default_rng(0)
        X = np.asarray(X, dtype=float); y = np.asarray(y)
        mu = X[y == 0].mean(0)
        syn = mu + 0.1 * rng.standard_normal((300, X.shape[1]))
        return np.vstack([X, syn]), np.r_[y, np.ones(300)]


class ShufflingOversampler(InterpolatingOversampler):
    """Same synthesis, but returns a SHUFFLED full set — exercises the
    set-difference fallback in synthetic_minority()."""

    def fit_resample(self, X, y):
        Xr, yr = super().fit_resample(X, y)
        p = np.random.default_rng(1).permutation(len(Xr))
        return Xr[p], yr[p]


def make_world(delta, n1=150, ir=5.0, d=2, seed=0):
    """2-Gaussian world: minority N((delta,0..),I) vs majority N(0,I)."""
    rng = np.random.default_rng(seed)
    mu1 = np.zeros(d); mu1[0] = delta
    Xmin = rng.multivariate_normal(mu1, np.eye(d), n1)
    Xmaj = rng.multivariate_normal(np.zeros(d), np.eye(d), int(n1 * ir))
    X = np.vstack([Xmaj, Xmin])
    y = np.r_[np.zeros(len(Xmaj)), np.ones(n1)].astype(int)
    return X, y, Xmin, Xmaj


@pytest.fixture
def separable_world():
    return make_world(delta=8.0)


@pytest.fixture
def overlap_world():
    return make_world(delta=1.4)
