"""Command line: resample-audit data.csv [--resampler pkg.mod:Class] ...

CSV format: numeric feature columns; label column = --label-col (default:
last column). Minority label inferred as the rarer value unless given.
"""
from __future__ import annotations

import argparse
import importlib
import sys

import numpy as np


def _resolve(spec):
    """'imblearn.over_sampling:SMOTE' or dotted path -> class."""
    mod, _, obj = spec.replace(":", ".").rpartition(".")
    if not mod:
        raise SystemExit(f"--resampler {spec!r}: give a full import path "
                         "like imblearn.over_sampling.SMOTE")
    try:
        return getattr(importlib.import_module(mod), obj)
    except (ImportError, AttributeError) as e:
        raise SystemExit(f"cannot import {spec!r}: {e}")


def main(argv=None):
    ap = argparse.ArgumentParser(
        prog="resample-audit",
        description="De-biased validity (ER_split) + information-gain audit "
                    "of an oversampler on a CSV dataset.")
    ap.add_argument("csv", help="dataset; numeric features + label column")
    ap.add_argument("--resampler", default="imblearn.over_sampling.SMOTE",
                    help="import path of the resampler class "
                         "(default: imblearn SMOTE)")
    ap.add_argument("--label-col", default=None,
                    help="label column name or index (default: last)")
    ap.add_argument("--minority-label", default=None,
                    help="minority value in the label column "
                         "(default: the rarer one)")
    ap.add_argument("--metric", default="hassanat",
                    choices=["hassanat", "euclidean"])
    ap.add_argument("--n-splits", type=int, default=5)
    ap.add_argument("--n-query", type=int, default=1000)
    ap.add_argument("--ref-cap", type=int, default=20000)
    ap.add_argument("--er-max", type=float, default=0.10)
    ap.add_argument("--no-benchmark", action="store_true",
                    help="validity instrument only")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--json", action="store_true", help="emit JSON")
    a = ap.parse_args(argv)

    import csv as _csv
    with open(a.csv, newline="") as f:
        rows = list(_csv.reader(f))
    hdr = rows[0]
    try:                                    # header row present?
        [float(v) for v in hdr]
        data, names = rows, [str(i) for i in range(len(rows[0]))]
    except ValueError:
        data, names = rows[1:], hdr
    li = (len(names) - 1 if a.label_col is None
          else int(a.label_col) if a.label_col.lstrip("-").isdigit()
          else names.index(a.label_col))
    lab_raw = np.array([r[li] for r in data if r])
    X = np.array([[float(v) for j, v in enumerate(r) if j != li]
                  for r in data if r])
    labs = np.unique(lab_raw)
    if a.minority_label is not None:
        minority = a.minority_label
        if minority not in labs:
            raise SystemExit(f"--minority-label {minority!r} not in {labs}")
    else:
        minority = labs[np.argmin([(lab_raw == v).sum() for v in labs])]
    y = (lab_raw == minority).astype(int)

    from resample_audit import audit
    rep = audit(X, y, _resolve(a.resampler), n_splits=a.n_splits,
                n_query=a.n_query, ref_cap=a.ref_cap, metric=a.metric,
                er_max=a.er_max, benchmark=not a.no_benchmark, seed=a.seed)
    if a.json:
        import json
        print(json.dumps(rep.to_dict(), default=float, indent=2))
    else:
        print(rep)
    return 0


if __name__ == "__main__":
    sys.exit(main())
