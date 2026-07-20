"""Faithful implementation of SMOTEFUNA (Tarawneh, Hassanat, Almohammadi,
Chetverikov & Bellinger, "SMOTEFUNA: Synthetic Minority Over-Sampling
Technique Based on Furthest Neighbour Algorithm", IEEE Access 8, 2020,
DOI 10.1109/ACCESS.2020.2983003), for auditing with resample-audit.

Algorithm 1 of the paper, verbatim:
  * Rate = |majority| - |minority| synthetic points are generated.
  * Repeat: pick a random minority seed SE1; find SE2 = the minority example
    with MAXIMUM (Manhattan) distance from SE1 (the "furthest neighbour");
    draw SEnew feature-by-feature uniformly in [min(S1j,S2j), max(S1j,S2j)]
    (a random point in the hyper-cuboid spanned by SE1 and SE2).
  * BUILT-IN CHECK: accept SEnew iff theta <= beta, where
       theta = min Manhattan distance from SEnew to the minority class,
       beta  = min Manhattan distance from SEnew to the majority class.
    i.e. accept iff SEnew's nearest neighbour in the training set is minority;
    otherwise discard and regenerate. Distance = Manhattan (Eq. 1).

That acceptance rule is exactly the classical (2022) nearest-neighbour
validity check, embedded in the generator -- so ER_naive is ~0 by design and
the interesting question is what ER_split says on withheld data.

imbalanced-learn compatible: `.fit_resample(X, y)` returns the original data
with the synthetic minority rows appended.
"""
from __future__ import annotations

import numpy as np


class SMOTEFUNA:
    def __init__(self, random_state=None, batch=4096, max_attempts=500):
        self.random_state = random_state
        self.batch = batch
        self.max_attempts = max_attempts   # safety cap (paper's "go to 5" loop)

    def fit_resample(self, X, y):
        rng = np.random.default_rng(self.random_state)
        X = np.asarray(X, dtype=float); y = np.asarray(y)
        labs, cnt = np.unique(y, return_counts=True)
        min_lab = labs[np.argmin(cnt)]
        Xmin = X[y == min_lab]; Xmaj = X[y != min_lab]
        n_min = len(Xmin)
        rate = len(Xmaj) - n_min
        if rate <= 0 or n_min < 2:
            return X.copy(), y.copy()

        # furthest minority neighbour of every minority point (Manhattan)
        # pairwise L1 within minority; argmax per row
        d = np.abs(Xmin[:, None, :] - Xmin[None, :, :]).sum(-1)
        furthest = d.argmax(1)                      # index of SE2 for each SE1

        synth = np.empty((rate, X.shape[1]))
        got = 0; attempts = 0
        while got < rate and attempts < self.max_attempts:
            attempts += 1
            k = min(self.batch, (rate - got) * 4)   # over-draw; some rejected
            seeds = rng.integers(0, n_min, k)
            S1 = Xmin[seeds]; S2 = Xmin[furthest[seeds]]
            lo = np.minimum(S1, S2); hi = np.maximum(S1, S2)
            cand = lo + rng.random((k, X.shape[1])) * (hi - lo)
            # theta = nearest minority (L1), beta = nearest majority (L1)
            theta = np.abs(cand[:, None, :] - Xmin[None, :, :]).sum(-1).min(1)
            beta = np.abs(cand[:, None, :] - Xmaj[None, :, :]).sum(-1).min(1)
            ok = cand[theta <= beta]
            take = min(len(ok), rate - got)
            synth[got:got + take] = ok[:take]; got += take

        synth = synth[:got]
        Xr = np.vstack([X, synth])
        yr = np.r_[y, np.full(got, min_lab)]
        return Xr, yr

    # smote_variants-style alias, so the harness/tool accept it either way
    def sample(self, X, y):
        return self.fit_resample(X, y)


if __name__ == "__main__":
    # quick self-check: ER_naive should be ~0 (built-in check), ER_split higher
    from sklearn.datasets import make_classification
    from resample_audit import audit
    X, y = make_classification(n_samples=3000, weights=[0.93, 0.07],
                               class_sep=0.8, random_state=0)
    rep = audit(X, y, lambda: SMOTEFUNA(random_state=0), seed=0)
    print(rep)
