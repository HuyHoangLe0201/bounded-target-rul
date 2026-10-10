"""The predictive laws in hours of remaining life.

At inspection k the elapsed time t_k is known, and the total life is
T = t_k + R_k, so the normalised target y = R_k / T = R_k / (t_k + R_k) is an
exact reparametrisation of the RUL:

    R_k = t_k * y / (1 - y),        increasing in y on (0, 1).

A predictive law for y on (0,1) therefore induces one for R in hours, with the
quantiles mapped through the same function. Coverage of a central interval and
the PIT are unchanged by the map, and the median maps to the median, so the
absolute error of the median is recomputed here in hours.

The CRPS is not: it needs E[R] < infinity, and R has a pole at y = 1. Any law
on y with positive density at one -- every truncated or restricted head here --
induces an RUL law with infinite mean, hence infinite CRPS in hours. The CRPS
values in hours below are therefore NOT reported in the paper: they are finite
only because the stored quantile grid stops at 0.995, and they grow without
bound as that level approaches one. Mass at y >= 1 maps to no valid RUL (it would need
t_k < 0), so the six conventional heads enter after restriction to (0,1)
(study_review_p2.truncate); the four bounded heads enter as fitted.

Reads results_both_causal.json; writes results/regression/hours.json.
"""
import json
import os

import numpy as np

import study_review_p2 as S

DT = {"pronostia": 10.0, "xjtu": 60.0}       # seconds between acquisitions
NB, SEED = 10000, 4
CONV = ["BS", "BS-t", "Weibull", "LogNormal", "IGauss", "Gaussian"]
BOUND = ["LogitNormal", "Beta", "BS-trunc", "IGauss-trunc"]
EPS = 1e-9


def to_hours(qy, t_h):
    qy = np.clip(qy, EPS, 1 - 1e-6)
    return t_h * qy / (1 - qy)


def main():
    d = json.load(open(os.path.join(S.RES, "results_both_causal.json")))
    rows = [r for r in d["rows"] if r["q"] is not None and r["head"] in CONV + BOUND]
    per = {}          # (dataset, bearing) -> head -> list of (crps, ae, cov)
    for r in rows:
        t_h = r["k"] * DT[r["dataset"]] / 3600.0
        q = np.asarray(r["q"], float)
        if r["head"] in CONV:
            q = S.truncate(q)[0]
        qr = to_hours(q, t_h)
        y = r["true"]
        R = t_h * y / (1 - y)
        lo, hi = S.interval(qr, 0.025)
        med = float(np.interp(0.5, S.TAU, qr))
        per.setdefault((r["dataset"], r["bearing"]), {}).setdefault(r["head"], []).append(
            (S.crps_q(qr, R), abs(med - R), float(lo <= R <= hi)))
    arch = {ds: sorted(b for (dd, b) in per if dd == ds) for ds in DT}

    out = {"heads": {}, "pairs": {}}
    for ds in DT:
        for h in CONV + BOUND:
            v = np.array([x for b in arch[ds] for x in per[(ds, b)][h]])
            out["heads"][f"{ds}/{h}"] = {"CRPS_h": float(v[:, 0].mean()), "MAE_h": float(v[:, 1].mean()),
                                         "cov95": float(v[:, 2].mean()), "n": int(len(v))}

    rng = np.random.default_rng(SEED)

    def margins(sample, a, b):
        bm = {ds: [np.mean([x[1] for x in per[(ds, z)][a]]) - np.mean([x[1] for x in per[(ds, z)][b]])
                   for z in sample[ds]] for ds in sample}
        return {"pronostia": float(np.mean(bm["pronostia"])), "xjtu": float(np.mean(bm["xjtu"])),
                "pooled": float(np.mean(bm["pronostia"] + bm["xjtu"]))}

    #  paired comparisons on the absolute error of the median, in hours
    fams = {"vs_conventional_restricted": [(a, b) for a in ("BS-trunc", "IGauss-trunc") for b in CONV],
            "bounded": [("IGauss-trunc", "BS-trunc"), ("IGauss-trunc", "Beta"), ("IGauss-trunc", "LogitNormal"),
                        ("BS-trunc", "Beta"), ("BS-trunc", "LogitNormal")]}
    for fam, pairs in fams.items():
        for a, b in pairs:
            pt = margins(arch, a, b)
            bs = {k: [] for k in pt}
            for _ in range(NB):
                smp = {ds: [arch[ds][i] for i in rng.choice(len(arch[ds]), len(arch[ds]))] for ds in arch}
                for k, v in margins(smp, a, b).items():
                    bs[k].append(v)
            out["pairs"][f"{a} - {b}"] = {"family": fam, **{
                k: {"margin": pt[k], "ci95": [float(np.quantile(bs[k], 0.025)), float(np.quantile(bs[k], 0.975))],
                    "p_boot": float(min(1.0, 2 * min(np.mean(np.array(bs[k]) >= 0), np.mean(np.array(bs[k]) <= 0))))}
                for k in pt}}
        keys = [f"{a} - {b}" for a, b in pairs]
        run = 0.0
        for i, k in enumerate(sorted(keys, key=lambda k: out["pairs"][k]["pooled"]["p_boot"])):
            run = max(run, min(1.0, (len(keys) - i) * out["pairs"][k]["pooled"]["p_boot"]))
            out["pairs"][k]["pooled"]["p_holm"] = run
    json.dump(out, open(os.path.join(S.RES, "hours.json"), "w"), indent=1)
    for k, v in out["heads"].items():
        print(f"{k:24s} CRPS {v['CRPS_h']:.3f} h  MAE {v['MAE_h']:.3f} h  cov95 {v['cov95']:.3f}")
    for k, v in out["pairs"].items():
        p = v["pooled"]
        print(f"{k:28s} " + "  ".join(f"{ds}: {v[ds]['margin']:+.3f} [{v[ds]['ci95'][0]:+.3f},{v[ds]['ci95'][1]:+.3f}]"
                                      for ds in ("pronostia", "xjtu", "pooled")) + f"  holm={p['p_holm']:.3f}")


if __name__ == "__main__":
    main()
