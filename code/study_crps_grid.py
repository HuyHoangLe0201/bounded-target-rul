"""Numerical check of the quantile-grid CRPS used for the restricted laws.

The restricted (post-hoc truncated) laws are scored by integrating the pinball
loss over the 101 stored levels 0.005..0.995, which leaves out 1 % of mass in
the two tails.  Three variants of the integral are compared:

  grid101    the stored levels (what the paper reports)
  dense      the quantile function interpolated to 1000 midpoint levels
             on [0.005, 0.995]
  tails      dense levels over the whole of (0,1), the outer 0.5 % on each side
             extrapolated linearly from the last two stored levels and clipped
             to the target's support [0, 1]

and, for each, the pooled IG - rival margins of the restricted laws.
Writes results/regression/crps_grid.json.
"""
import json
import os

import numpy as np

import study_review_p2 as S

INNER = (np.arange(1000) + 0.5) / 1000 * (S.TAU[-1] - S.TAU[0]) + S.TAU[0]
FULL = (np.arange(2000) + 0.5) / 2000


def crps_levels(qf, levels, y):
    u = y - qf
    return 2.0 * np.mean(np.maximum(levels * u, (levels - 1.0) * u))


def variants(q, y):
    q = np.asarray(q, float)
    out = {"grid101": S.crps_q(q, y)}
    out["dense"] = crps_levels(np.interp(INNER, S.TAU, q), INNER, y) * (S.TAU[-1] - S.TAU[0])
    lo_s = (q[1] - q[0]) / (S.TAU[1] - S.TAU[0])
    hi_s = (q[-1] - q[-2]) / (S.TAU[-1] - S.TAU[-2])
    qf = np.interp(FULL, S.TAU, q)
    qf = np.where(FULL < S.TAU[0], q[0] - lo_s * (S.TAU[0] - FULL), qf)
    qf = np.where(FULL > S.TAU[-1], q[-1] + hi_s * (FULL - S.TAU[-1]), qf)
    out["tails"] = crps_levels(np.clip(qf, 0.0, 1.0), FULL, y)
    return out


def main():
    _, rows = S.load()
    c = {}
    for r in rows:
        v = variants(S.truncate(np.asarray(r["q"], float))[0], r["true"])
        for k, x in v.items():
            c.setdefault(k, {}).setdefault((r["dataset"], r["bearing"]), {}).setdefault(r["head"], []).append(x)
    out = {}
    for k, cc in c.items():
        res = {}
        for ds in ("pronostia", "xjtu"):
            for h in S.HEADS:
                res[f"{ds}/{h}"] = float(np.mean([x for (dd, _), hh in cc.items() if dd == ds for x in hh[h]]))
        for rival in ("BS", "Weibull", "LogNormal", "Gaussian"):
            res[f"pooled IG-{rival}"] = float(np.mean([np.mean(hh["IGauss"]) - np.mean(hh[rival])
                                                      for hh in cc.values()]))
        res["pooled BS-Weibull"] = float(np.mean([np.mean(hh["BS"]) - np.mean(hh["Weibull"]) for hh in cc.values()]))
        res["pooled BS-LogNormal"] = float(np.mean([np.mean(hh["BS"]) - np.mean(hh["LogNormal"]) for hh in cc.values()]))
        out[k] = res
    keys = list(out["grid101"])
    out["max_abs_dev_vs_grid101"] = {k: max(abs(out[v][k] - out["grid101"][k]) for v in ("dense", "tails"))
                                     for k in keys}
    json.dump(out, open(os.path.join(S.RES, "crps_grid.json"), "w"), indent=1)
    for k in keys:
        print(f"{k:22s} " + "  ".join(f"{v}: {out[v][k]:+.4f}" for v in ("grid101", "dense", "tails")))


if __name__ == "__main__":
    main()
