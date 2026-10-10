# -*- coding: utf-8 -*-
r"""Can the two benchmarks together resolve what neither resolves alone?

Paper 2's open question is its own headline.  The inverse Gaussian head leads on
both benchmarks, but PRONOSTIA separates it from BS and not from Weibull, and
XJTU-SY does the reverse, so on eleven and fifteen bearings the paper can only
say the ordering is directional.  That is honest and it is also weaker than the
data may support: the two benchmarks are separate evidence about the same
question, and pooling them gives twenty-six clusters instead of eleven.

Pooling raw scores would be wrong -- the two archives differ in life scale, so a
pooled mean absolute error is mostly a statement about which archive has the
larger seconds.  Two combinations that are legitimate:

  1. STRATIFIED, on within-benchmark standardised differences.  Resample
     bearings within each benchmark, compute each head's paired difference from
     the leading head inside that benchmark, standardise by that benchmark's own
     spread, and average.  Units cancel; what survives is "better on both".
  2. RANK, the weakest assumption available.  Resample as above and average each
     head's rank within each benchmark.  A head that is first on both benchmarks
     in a resample gets average rank 1.

Both are paired: every head sees the identical resampled bearings, so the
comparison is not confounded by which bearings the resample happened to draw.

    python code/study_joint.py
"""
from __future__ import annotations

import io
import json
import os
from collections import defaultdict

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "..", "results", "regression")
HEADS = ["BS", "BS-t", "Weibull", "LogNormal", "IGauss", "Gaussian"]
NAME = {"BS": "BS", "BS-t": "BS-t", "Weibull": "Weibull", "LogNormal": "lognormal",
        "IGauss": "inverse Gaussian", "Gaussian": "Gaussian"}
B = 10_000
QLEV = np.linspace(0.005, 0.995, 101)
rng = np.random.default_rng(20260806)


def crps(qs, y):
    """CRPS by the pinball identity on the stored quantile grid."""
    q = np.asarray(qs, float)
    lo = QLEV * (y - q)
    hi = (QLEV - 1.0) * (y - q)
    return 2.0 * np.trapezoid(np.where(y >= q, lo, hi), QLEV)


rows = json.load(io.open(os.path.join(RES, "results_both.json"),
                         encoding="utf-8"))["rows"]

#  per (dataset, bearing, head): mean CRPS and mean absolute error
acc = defaultdict(lambda: defaultdict(list))
for r in rows:
    if r["head"] not in HEADS:
        continue
    key = (r["dataset"], r["bearing"])
    acc[key][r["head"]].append(
        (crps(r["q"], r["true"]), abs(r["median"] - r["true"])))

per = {k: {h: (float(np.mean([x[0] for x in v])),
              float(np.mean([x[1] for x in v]))) for h, v in d.items()}
       for k, d in acc.items()}
sets = sorted({k[0] for k in per})
bearings = {d: sorted(b for (dd, b) in per if dd == d) for d in sets}
for d in sets:
    print(f"  {d}: {len(bearings[d])} bearings")
print(f"  pooled: {sum(len(v) for v in bearings.values())} clusters\n")


def scores(sample, metric):
    """Mean score per head over a resampled set of bearings, per dataset."""
    out = {}
    for d in sets:
        vals = {h: [] for h in HEADS}
        for b in sample[d]:
            for h in HEADS:
                vals[h].append(per[(d, b)][h][metric])
        out[d] = {h: float(np.mean(v)) for h, v in vals.items()}
    return out


def draw():
    return {d: list(rng.choice(bearings[d], len(bearings[d]), replace=True))
            for d in sets}


payload = {}
for metric, mname in ((0, "CRPS"), (1, "absolute error")):
    print(f"  ── {mname} " + "─" * 46)
    first_rank, first_std, diff_vs_weib = defaultdict(int), defaultdict(int), []
    for _ in range(B):
        s = scores(draw(), metric)
        #  rank combination
        ranks = defaultdict(float)
        for d in sets:
            order = sorted(HEADS, key=lambda h: s[d][h])
            for i, h in enumerate(order):
                ranks[h] += (i + 1) / len(sets)
        first_rank[min(ranks, key=ranks.get)] += 1
        #  standardised combination
        z = defaultdict(float)
        for d in sets:
            v = np.array([s[d][h] for h in HEADS])
            sd = v.std() or 1.0
            for h, x in zip(HEADS, v):
                z[h] += (x - v.mean()) / sd / len(sets)
        first_std[min(z, key=z.get)] += 1
        diff_vs_weib.append(z["IGauss"] - z["Weibull"])

    print(f"     P(leads on average rank across both benchmarks)")
    for h, n in sorted(first_rank.items(), key=lambda kv: -kv[1]):
        if n:
            print(f"       {NAME[h]:20s} {n / B:6.3f}")
    print(f"     P(leads on the standardised combination)")
    for h, n in sorted(first_std.items(), key=lambda kv: -kv[1]):
        if n:
            print(f"       {NAME[h]:20s} {n / B:6.3f}")
    d = np.asarray(diff_vs_weib)
    lo, hi = np.percentile(d, [2.5, 97.5])
    print(f"     inverse Gaussian minus Weibull, standardised: "
          f"{d.mean():+.3f}  95% [{lo:+.3f},{hi:+.3f}]"
          f"   {'separated' if hi < 0 or lo > 0 else 'NOT separated'}\n")

    #  Both metrics are stored. Keeping only CRPS left the absolute-error
    #  figures the manuscript quotes -- the lead probability and the margin over
    #  Weibull -- alive nowhere but a console log, so nothing could check them.
    payload.setdefault("n_clusters", sum(len(v) for v in bearings.values()))
    payload[mname] = {
        "p_rank_first": {h: n / B for h, n in first_rank.items()},
        "p_std_first": {h: n / B for h, n in first_std.items()},
        "ig_minus_weibull": {"mean": float(d.mean()),
                             "ci": [float(lo), float(hi)]},
    }
    if metric == 0:      # kept flat as well, so existing readers still work
        payload["p_rank_first"] = payload[mname]["p_rank_first"]
        payload["p_std_first"] = payload[mname]["p_std_first"]
        payload["ig_minus_weibull"] = payload[mname]["ig_minus_weibull"]
json.dump(payload, io.open(os.path.join(RES, "joint.json"), "w",
                          encoding="utf-8"), indent=1)
print("  wrote results/regression/joint.json")
