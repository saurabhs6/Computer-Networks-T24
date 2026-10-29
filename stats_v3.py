"""Shared helpers to aggregate the JSON-lines results of the revised study."""
import collections, json, math
import numpy as np
import runner as R
from scipy import stats

NAME = {"M1": "M1", "M2": "M2", "M3": "M3", "M2-EIG": "M2+EIG", "M1-BF": "M1+BF", "OPT": "OPT", "RCI": "RCI",
        "C22": "C12", "Z10": "Z10", "Y14": "Y14", "Y4": "Y4", "IDEAL": "Ideal"}


def group(path, by=("method", "snr"), **flt):
    """rows grouped by the job fields in `by` (method 'IDEAL' for noiseless runs), then by seed."""
    g = collections.defaultdict(dict)
    for r in R.load(path, **flt):
        j = {**R.DEFAULT, **r["job"]}
        j["method"] = "IDEAL" if j["ideal"] else j["method"]
        g[tuple(j.get(k) for k in by)][j["seed"]] = r
    return g


def mean_sd(vals):
    v = np.asarray(vals, float)
    return float(v.mean()), float(v.std(ddof=1)) if len(v) > 1 else 0.0


def paired_ci(a, b, level=0.95):
    """mean difference a - b over common seeds, t-interval, and whether 0 lies inside."""
    seeds = sorted(set(a) & set(b))
    d = np.array([a[s] - b[s] for s in seeds], float)
    n = len(d); m = float(d.mean()); s = float(d.std(ddof=1))
    h = float(stats.t.ppf(0.5 + level / 2, n - 1) * s / math.sqrt(n))
    return dict(n=n, diff=m, lo=m - h, hi=m + h, within_noise=bool(m - h <= 0 <= m + h))


def per_seed(rows, field="acc10", scale=100.0):
    return {s: scale * r[field] for s, r in rows.items()}
