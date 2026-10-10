"""Heads that respect the (0,1] target in training, against each other and
against the post-hoc restriction of the unbounded first-passage heads.

Reads results/regression/results_both_causal.json (written with GBS_BOUNDED=1, so it
holds the four bounded heads beside the six of the main comparison). Bearings are
resampled within each archive, every bearing weighted equally in the pooled
margin, and the pooled p-values of each family of comparisons are Holm-adjusted.

Writes results/regression/bounded.json.
"""
import json
import os

import numpy as np

import study_review_p2 as S
from study_crps_grid import variants

NB, SEED = 10000, 2
BOUNDED = ["LogitNormal", "Beta", "BS-trunc", "IGauss-trunc"]
#  family 1: the four support-respecting heads against each other
PAIRS_BOUNDED = [("IGauss-trunc", "BS-trunc"), ("IGauss-trunc", "Beta"),
                 ("IGauss-trunc", "LogitNormal"), ("BS-trunc", "Beta"),
                 ("BS-trunc", "LogitNormal"), ("Beta", "LogitNormal")]
#  family 2: trained with the truncated likelihood vs restricted after fitting
PAIRS_TRAINED_VS_POSTHOC = [("IGauss-trunc", "IGauss@post"), ("BS-trunc", "BS@post")]
#  family 3: the truncated first-passage heads against all six conventional
#  heads as fitted, so that "beats every conventional head" is tested directly
PAIRS_VS_UNBOUNDED = [(a, b) for a in ("BS-trunc", "IGauss-trunc")
                      for b in ("BS", "BS-t", "Weibull", "LogNormal", "IGauss", "Gaussian")]


def main():
    d = json.load(open(os.path.join(S.RES, "results_both_causal.json")))
    rows = [r for r in d["rows"] if r["q"] is not None]
    rng = np.random.default_rng(SEED)
    c, cd = {}, {}      # CRPS on the 101 stored levels, and on 1000 interpolated levels
    for r in rows:
        q = np.asarray(r["q"], float)
        key = (r["dataset"], r["bearing"])
        c.setdefault(key, {}).setdefault(r["head"], []).append(S.crps_q(q, r["true"]))
        cd.setdefault(key, {}).setdefault(r["head"], []).append(variants(q, r["true"])["dense"])
        if r["head"] in ("BS", "IGauss"):     # post-hoc restriction of the unbounded heads
            c[key].setdefault(r["head"] + "@post", []).append(S.crps_q(S.truncate(q)[0], r["true"]))
            cd[key].setdefault(r["head"] + "@post", []).append(variants(S.truncate(q)[0], r["true"])["dense"])
    arch = {ds: sorted(b for (dd, b) in c if dd == ds) for ds in ("pronostia", "xjtu")}

    def margins(sample, a, b):
        bm = {ds: [np.mean(c[(ds, x)][a]) - np.mean(c[(ds, x)][b]) for x in sample[ds]] for ds in sample}
        return {"pronostia": float(np.mean(bm["pronostia"])), "xjtu": float(np.mean(bm["xjtu"])),
                "pooled": float(np.mean(bm["pronostia"] + bm["xjtu"]))}

    out = {"heads": {}, "pairs": {}}
    for ds in ("pronostia", "xjtu"):
        for h in BOUNDED + ["BS@post", "IGauss@post"]:
            out["heads"][f"{ds}/{h}"] = float(np.mean([v for (dd, _), hh in c.items() if dd == ds for v in hh[h]]))
    for fam, pairs in (("bounded", PAIRS_BOUNDED), ("trained_vs_posthoc", PAIRS_TRAINED_VS_POSTHOC),
                       ("vs_unbounded", PAIRS_VS_UNBOUNDED)):
        for a, b in pairs:
            pt = margins(arch, a, b)
            dense = float(np.mean([np.mean(cd[(ds, x)][a]) - np.mean(cd[(ds, x)][b])
                                   for ds in arch for x in arch[ds]]))
            bs = {k: [] for k in pt}
            for _ in range(NB):
                smp = {ds: [arch[ds][i] for i in rng.choice(len(arch[ds]), len(arch[ds]))] for ds in arch}
                for k, v in margins(smp, a, b).items():
                    bs[k].append(v)
            out["pairs"][f"{a} - {b}"] = {"family": fam, "pooled_dense_margin": dense, **{
                k: {"margin": pt[k],
                    "ci95": [float(np.quantile(bs[k], 0.025)), float(np.quantile(bs[k], 0.975))],
                    "p_boot": float(min(1.0, 2 * min(np.mean(np.array(bs[k]) >= 0),
                                                     np.mean(np.array(bs[k]) <= 0))))}
                for k in pt}}
        keys = [f"{a} - {b}" for a, b in pairs]
        run = 0.0
        for i, k in enumerate(sorted(keys, key=lambda k: out["pairs"][k]["pooled"]["p_boot"])):
            run = max(run, min(1.0, (len(keys) - i) * out["pairs"][k]["pooled"]["p_boot"]))
            out["pairs"][k]["pooled"]["p_holm"] = run
    json.dump(out, open(os.path.join(S.RES, "bounded.json"), "w"), indent=1)
    for k, v in out["heads"].items():
        print(f"{k:28s} CRPS {v:.4f}")
    for k, v in out["pairs"].items():
        p = v["pooled"]
        print(f"{k:30s} " + "  ".join(f"{ds}: {v[ds]['margin']:+.4f} [{v[ds]['ci95'][0]:+.4f},{v[ds]['ci95'][1]:+.4f}]"
                                      for ds in ("pronostia", "xjtu", "pooled")) + f"  holm={p['p_holm']:.3f}")


if __name__ == "__main__":
    main()
