"""De-biased validity instrument: ER_naive (parent-retained, 2022 protocol)
vs ER_split (independent-reference reform).

Protocol identical to the research harness (`validity_benchmark.validity_one`):
  * ER_naive : 1-NN vote of synthetic points against the generator's own
    training sample (the parents remain in the reference — Lemma 0L bias).
  * ER_split : n_splits times, randomly halve each class; generate on half A,
    vote against half B only (never seen by the generator; capped at ref_cap).
    Consistent for 1 - V(G) (Prop 1).
"""
from __future__ import annotations

import numpy as np

from ._distance import nn_is_majority, wilson_ci


def _call_resampler(resampler, X, y):
    """Support imblearn-style .fit_resample, smote_variants-style .sample,
    and plain callables (X, y) -> (Xr, yr)."""
    if hasattr(resampler, "fit_resample"):
        Xr, yr = resampler.fit_resample(X, y)
    elif hasattr(resampler, "sample"):
        Xr, yr = resampler.sample(X, y)
    elif callable(resampler):
        Xr, yr = resampler(X, y)
    else:
        raise TypeError("resampler must expose fit_resample(X, y), "
                        "sample(X, y), or be callable(X, y)")
    return np.asarray(Xr, dtype=float), np.asarray(yr)


def synthetic_minority(resampler, Xmin, Xmaj):
    """Run the generator on (majority + minority) and return the synthetic
    minority rows it created.

    Primary convention (imblearn, smote_variants): new rows are appended
    after the original data. Fallback for generators that reorder: minority
    rows of the output whose bytes do not match any input minority row.
    """
    X = np.vstack([Xmaj, Xmin]).astype(float)
    y = np.r_[np.zeros(len(Xmaj)), np.ones(len(Xmin))]
    Xr, yr = _call_resampler(resampler, X, y)
    if len(Xr) >= len(X) and np.array_equal(yr[len(X):],
                                            np.ones(len(Xr) - len(X))):
        appended = Xr[len(X):]
        if len(appended):
            return appended
    # fallback: set-difference on rows (handles shuffled output)
    seen = {a.tobytes() for a in np.ascontiguousarray(
        Xmin.astype(np.float64))}
    out = [r for r, lab in zip(Xr, yr) if lab == 1
           and np.ascontiguousarray(r.astype(np.float64)).tobytes()
           not in seen]
    return np.asarray(out, dtype=float).reshape(-1, X.shape[1])


def validity_audit(resampler_factory, Xmin, Xmaj, *, n_splits=5, n_query=1000,
                   ref_cap=20000, metric="hassanat", seed=0):
    """Compute ER_naive and ER_split for one generator.

    Parameters
    ----------
    resampler_factory : zero-arg callable returning a FRESH resampler
        (a new instance per split avoids state leakage between fits).
    Xmin, Xmaj : arrays of real minority / majority rows.
    Returns a dict (see keys below); ER values are majority-vote rates of
    synthetic points, i.e. estimates of 1 - V(G).
    """
    rng = np.random.default_rng(seed)
    Xmin = np.asarray(Xmin, dtype=float); Xmaj = np.asarray(Xmaj, dtype=float)

    # --- naive (2022 protocol: parents retained in the reference) ---------
    syn = synthetic_minority(resampler_factory(), Xmin, Xmaj)
    if len(syn) == 0:
        raise ValueError("generator produced no synthetic minority points")
    refX = np.vstack([Xmaj, Xmin]).astype(np.float32)
    refy = np.r_[np.zeros(len(Xmaj)), np.ones(len(Xmin))]
    if len(refX) > ref_cap:                       # keep ALL parents (the bias)
        mi = np.where(refy == 1)[0]; mj = np.where(refy == 0)[0]
        keep = np.r_[mi, rng.choice(mj, ref_cap - len(mi), replace=False)]
        refX, refy = refX[keep], refy[keep]
    q = syn if len(syn) <= n_query else syn[rng.choice(len(syn), n_query,
                                                       replace=False)]
    hit = nn_is_majority(q, refX, refy, metric=metric)
    er_naive = float(hit.mean())
    naive_ci = wilson_ci(int(hit.sum()), len(hit))

    # --- split (reform: reference independent of the generator) -----------
    vals = []
    for _ in range(n_splits):
        pm = rng.permutation(len(Xmin)); pj = rng.permutation(len(Xmaj))
        a, b = len(Xmin) // 2, len(Xmaj) // 2
        Amin, Bmin = Xmin[pm[:a]], Xmin[pm[a:]]
        Amaj, Bmaj = Xmaj[pj[:b]], Xmaj[pj[b:]]
        if len(Amin) < 6:
            continue
        ss = synthetic_minority(resampler_factory(), Amin, Amaj)
        if len(ss) == 0:
            continue
        qs = ss if len(ss) <= n_query else ss[rng.choice(len(ss), n_query,
                                                         replace=False)]
        Bx = np.vstack([Bmaj, Bmin]).astype(np.float32)
        By = np.r_[np.zeros(len(Bmaj)), np.ones(len(Bmin))]
        if len(Bx) > ref_cap:
            idx = rng.choice(len(Bx), ref_cap, replace=False)
            Bx, By = Bx[idx], By[idx]
        vals.append(float(nn_is_majority(qs, Bx, By, metric=metric).mean()))
    if not vals:
        raise ValueError("ER_split unavailable: minority too small to halve "
                         "(need >= 12 minority rows) or generator failed on "
                         "every split")
    er_split = float(np.mean(vals))
    split_ci = wilson_ci(int(round(er_split * n_query)), n_query)
    return dict(n_synth=int(len(syn)),
                er_naive=er_naive, er_naive_ci=naive_ci,
                er_split=er_split, er_split_ci=split_ci,
                er_split_std=float(np.std(vals)), splits_used=len(vals),
                bias_gain=er_split - er_naive)
