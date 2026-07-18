"""Nearest-neighbour majority vote under the Hassanat or Euclidean metric.

The Hassanat implementation is a verbatim port of the research harness
(`validity_benchmark.nn_is_majority`), including the chunked broadcasting
(caps each temporary at ~40 MB) and the fast non-negative branch, so audit
numbers are bit-identical to the paper's instrument.
"""
from __future__ import annotations

import numpy as np


def nn_is_majority(query, refX, refy, metric="hassanat", chunk=None):
    """For each query point: is its nearest reference neighbour majority (0)?

    Parameters
    ----------
    query : (m, d) float array of synthetic points.
    refX  : (n, d) float array of real reference points.
    refy  : (n,) labels, 1 = minority, 0 = majority.
    metric : "hassanat" (default, the paper's instrument) or "euclidean".
    chunk : rows of `query` per broadcast block; None = auto (~40 MB temps).

    Returns
    -------
    (m,) bool array — True where the nearest real neighbour is majority.
    """
    query = np.ascontiguousarray(query, dtype=np.float32)
    refX = np.ascontiguousarray(refX, dtype=np.float32)
    refy = np.asarray(refy)
    if metric == "euclidean":
        # scipy-free, exact: argmin over squared distances, chunked
        if chunk is None:
            chunk = int(np.clip(1e7 // max(1, refX.shape[0]), 2, 2048))
        out = np.empty(len(query), dtype=bool)
        r2 = (refX * refX).sum(1)
        for s in range(0, len(query), chunk):
            Q = query[s:s + chunk]
            d2 = r2[None, :] - 2.0 * (Q @ refX.T)
            out[s:s + chunk] = (refy[d2.argmin(1)] == 0)
        return out
    if metric != "hassanat":
        raise ValueError(f"unknown metric {metric!r} (hassanat|euclidean)")
    nonneg = bool(query.min() >= 0 and refX.min() >= 0)
    if chunk is None:
        chunk = int(np.clip(1e7 // max(1, refX.shape[0] * refX.shape[1]),
                            2, 128))
    out = np.empty(len(query), dtype=bool)
    B = refX[None, :, :]
    for s in range(0, len(query), chunk):
        Q = query[s:s + chunk][:, None, :]
        mx = np.maximum(Q, B); mn = np.minimum(Q, B)
        if nonneg:
            d = (mx - mn) / (1.0 + mx)
        else:
            am = np.abs(mn)
            d = np.where(mn >= 0, (mx - mn) / (1.0 + mx),
                         (mx - mn) / (1.0 + mx + am))
        out[s:s + chunk] = (refy[d.sum(2).argmin(1)] == 0)
    return out


def wilson_ci(k, n, z=1.96):
    """Wilson score interval for a binomial proportion k/n."""
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n; d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (float(max(0.0, c - h)), float(min(1.0, round(c + h, 12))))
