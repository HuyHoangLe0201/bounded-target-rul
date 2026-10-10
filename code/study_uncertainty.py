"""How much of the head ranking survives the sample size?

The unit of independence in these benchmarks is the BEARING, not the prediction
point: every point inside a record shares that record's history. Seventeen and
fifteen records give an effective sample size of order ten, and a ranking of six
heads read off point estimates at that size is not obviously meaningful. This
script measures it directly with a cluster bootstrap over bearings, so the paper
can state which parts of the ordering are stable and which are not.

Reported per benchmark:
  * bootstrap distribution of each head's MAE and CRPS,
  * P(head ranks first), the honest form of "head X wins",
  * paired differences against the leading head, with a percentile interval;
    paired because every head sees the identical resampled bearings.

Usage:  python study_uncertainty.py
"""
from __future__ import annotations

import json
import os
from collections import defaultdict

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "..", "results", "regression")
HEADS = ["BS", "BS-t", "Weibull", "LogNormal", "IGauss", "Gaussian", "Point(MAE)"]
NAME = {"BS": "BS", "BS-t": "BS-$t$", "Weibull": "Weibull",
        "LogNormal": "lognormal", "IGauss": "inverse Gaussian",
        "Gaussian": "Gaussian", "Point(MAE)": "point head (MAE loss)"}
B = 10_000
QLEV = np.linspace(0.005, 0.995, 101)


def crps_from_quantiles(q, y):
    """CRPS by the quantile-loss (pinball) identity, integrated over levels.

    CRPS = 2 * int_0^1 QL_tau(q_tau, y) dtau, with QL the pinball loss. The
    stored grid is the same one the tables were built from.
    """
    q = np.asarray(q)
    ind = (y < q).astype(float)
    return 2.0 * np.trapezoid((QLEV - ind) * (y - q), QLEV)


def load():
    with open(os.path.join(RES, "results_both.json")) as f:
        rows = json.load(f)["rows"]
    per = defaultdict(lambda: defaultdict(list))   # dataset -> head -> per-bearing
    bearings = defaultdict(set)
    tmp = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    for r in rows:
        d, h, b = r["dataset"], r["head"], r["bearing"]
        bearings[d].add(b)
        ae = abs(r["median"] - r["true"])
        cr = crps_from_quantiles(r["q"], r["true"]) if r.get("q") else np.nan
        tmp[d][h][b].append((ae, cr))
    for d in tmp:
        for h in tmp[d]:
            per[d][h] = {b: np.array(v) for b, v in tmp[d][h].items()}
    return per, {d: sorted(v) for d, v in bearings.items()}


def bootstrap(per, bearings, dataset, metric_idx, rng):
    """Cluster bootstrap over bearings; returns (B x n_heads) metric matrix."""
    bs = bearings[dataset]
    heads = [h for h in HEADS if h in per[dataset]]
    out = np.full((B, len(heads)), np.nan)
    idx = rng.integers(0, len(bs), size=(B, len(bs)))
    for b in range(B):
        pick = [bs[i] for i in idx[b]]
        for j, h in enumerate(heads):
            vals = [per[dataset][h][p][:, metric_idx] for p in pick
                    if p in per[dataset][h]]
            if vals:
                v = np.concatenate(vals)
                out[b, j] = np.nanmean(v)
    return heads, out


def report(dataset, metric_name, metric_idx, per, bearings, rng, out):
    heads, M = bootstrap(per, bearings, dataset, metric_idx, rng)
    ok = ~np.all(np.isnan(M), axis=0)
    heads = [h for h, k in zip(heads, ok) if k]
    M = M[:, ok]
    point = np.nanmean(M, axis=0)
    best = int(np.argmin(np.nanmean(M, axis=0)))
    win = (np.nanargmin(M, axis=1)[:, None] == np.arange(M.shape[1])).mean(0)

    print(f"\n### {dataset} -- {metric_name}   ({len(bearings[dataset])} bearings, "
          f"{B} bootstrap resamples over bearings)")
    print(f"{'head':22s} {'mean':>7s} {'2.5%':>7s} {'97.5%':>7s} "
          f"{'P(first)':>9s}  diff vs {NAME[heads[best]]}")
    rec = []
    for j, h in enumerate(heads):
        lo, hi = np.nanpercentile(M[:, j], [2.5, 97.5])
        d = M[:, j] - M[:, best]
        dlo, dhi = np.nanpercentile(d, [2.5, 97.5])
        excl = "  *" if (dlo > 0 or dhi < 0) else "   "
        s = (f"{NAME[h]:22s} {point[j]:7.3f} {lo:7.3f} {hi:7.3f} {win[j]:9.3f}"
             f"  [{dlo:+.3f}, {dhi:+.3f}]{excl}")
        print(s)
        rec.append({"head": h, "mean": float(point[j]), "lo": float(lo),
                    "hi": float(hi), "p_first": float(win[j]),
                    "diff_lo": float(dlo), "diff_hi": float(dhi)})
    print("   * = the paired 95 % interval on the difference excludes zero")
    out[f"{dataset}/{metric_name}"] = {"best": heads[best], "rows": rec}


if __name__ == "__main__":
    per, bearings = load()
    rng = np.random.default_rng(0)
    out = {}
    for dset in ("pronostia", "xjtu"):
        report(dset, "MAE", 0, per, bearings, rng, out)
        report(dset, "CRPS", 1, per, bearings, rng, out)
    with open(os.path.join(RES, "uncertainty.json"), "w") as f:
        json.dump({"n_boot": B, "results": out}, f, indent=1)
    print("\n-> results/regression/uncertainty.json")
