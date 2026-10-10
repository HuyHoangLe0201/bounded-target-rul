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


FULL = (np.arange(2000) + 0.5) / 2000


def crps_variant(q, y, kind, lo, hi):
    """CRPS on the y scale under different treatments of the two tails.

    inner01 / inner025: the pinball integral over [0.01,0.99] / [0.025,0.975]
    only; full: over all of (0,1), the quantile function extrapolated linearly
    beyond the outermost stored levels and clipped to the head's support [lo,hi].
    """
    if kind in ("inner01", "inner025"):
        a = 0.01 if kind == "inner01" else 0.025
        m = (S.TAU >= a - 1e-12) & (S.TAU <= 1 - a + 1e-12)
        u = y - q[m]
        return 2.0 * np.mean(np.maximum(S.TAU[m] * u, (S.TAU[m] - 1.0) * u)) * (1 - 2 * a)
    ls = (q[1] - q[0]) / (S.TAU[1] - S.TAU[0])
    hs = (q[-1] - q[-2]) / (S.TAU[-1] - S.TAU[-2])
    qf = np.interp(FULL, S.TAU, q)
    qf = np.where(FULL < S.TAU[0], q[0] - ls * (S.TAU[0] - FULL), qf)
    qf = np.where(FULL > S.TAU[-1], q[-1] + hs * (FULL - S.TAU[-1]), qf)
    qf = np.clip(qf, lo, hi)
    u = y - qf
    return 2.0 * np.mean(np.maximum(FULL * u, (FULL - 1.0) * u))


def support(head):
    if head in BOUNDED or head.endswith("@post"):
        return 0.0, 1.0
    return (-np.inf, np.inf) if head == "Gaussian" else (0.0, np.inf)


def main():
    d = json.load(open(os.path.join(S.RES, "results_both_causal.json")))
    rows = [r for r in d["rows"] if r["q"] is not None]
    rng = np.random.default_rng(SEED)
    c, cd = {}, {}      # CRPS on the 101 stored levels, and on 1000 interpolated levels
    TAILS = ("inner01", "inner025", "full")
    ct = {k: {} for k in TAILS}
    for r in rows:
        q = np.asarray(r["q"], float)
        key = (r["dataset"], r["bearing"])
        c.setdefault(key, {}).setdefault(r["head"], []).append(S.crps_q(q, r["true"]))
        cd.setdefault(key, {}).setdefault(r["head"], []).append(variants(q, r["true"])["dense"])
        if r["head"] in ("BS", "IGauss"):     # post-hoc restriction of the unbounded heads
            c[key].setdefault(r["head"] + "@post", []).append(S.crps_q(S.truncate(q)[0], r["true"]))
            cd[key].setdefault(r["head"] + "@post", []).append(variants(S.truncate(q)[0], r["true"])["dense"])
        for kind in TAILS:
            for h, qq in ((r["head"], q),) + (((r["head"] + "@post", S.truncate(q)[0]),)
                                              if r["head"] in ("BS", "IGauss") else ()):
                ct[kind].setdefault(key, {}).setdefault(h, []).append(
                    crps_variant(qq, r["true"], kind, *support(h)))
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
    #  tail treatment: pooled margins of every pair under each variant, and for
    #  the full-domain variant the bootstrap interval and Holm p as well
    allpairs = [(fam, a, b) for fam, pairs in (("bounded", PAIRS_BOUNDED),
                ("trained_vs_posthoc", PAIRS_TRAINED_VS_POSTHOC), ("vs_unbounded", PAIRS_VS_UNBOUNDED))
                for a, b in pairs]
    out["tails"] = {}
    for kind in TAILS:
        cc = ct[kind]
        res = {}
        for fam, a, b in allpairs:
            pm = lambda smp: float(np.mean([np.mean(cc[(ds, x)][a]) - np.mean(cc[(ds, x)][b])
                                            for ds in smp for x in smp[ds]]))
            res[f"{a} - {b}"] = {"pooled": pm(arch)}
            if kind == "full":
                bs = [pm({ds: [arch[ds][i] for i in rng.choice(len(arch[ds]), len(arch[ds]))] for ds in arch})
                      for _ in range(NB)]
                res[f"{a} - {b}"].update(ci95=[float(np.quantile(bs, 0.025)), float(np.quantile(bs, 0.975))],
                                         p_boot=float(min(1.0, 2 * min(np.mean(np.array(bs) >= 0),
                                                                       np.mean(np.array(bs) <= 0)))))
        out["tails"][kind] = res
        out["tails"][kind + "_max_abs_dev"] = max(abs(res[k]["pooled"] - out["pairs"][k]["pooled"]["margin"])
                                                  for k in res)
    for fam in ("bounded", "trained_vs_posthoc", "vs_unbounded"):
        keys = [f"{a} - {b}" for f, a, b in allpairs if f == fam]
        run = 0.0
        for i, k in enumerate(sorted(keys, key=lambda k: out["tails"]["full"][k]["p_boot"])):
            run = max(run, min(1.0, (len(keys) - i) * out["tails"]["full"][k]["p_boot"]))
            out["tails"]["full"][k]["p_holm"] = run
    json.dump(out, open(os.path.join(S.RES, "bounded.json"), "w"), indent=1)
    for kind in TAILS:
        print(f"tails {kind:9s} max |pooled margin change| = {out['tails'][kind + '_max_abs_dev']:.4f}")
    for k, v in out["tails"]["full"].items():
        print(f"  full {k:28s} {v['pooled']:+.4f} [{v['ci95'][0]:+.4f},{v['ci95'][1]:+.4f}] holm={v['p_holm']:.3f}"
              f"   (grid {out['pairs'][k]['pooled']['margin']:+.4f}, holm {out['pairs'][k]['pooled']['p_holm']:.3f})")
    for k, v in out["heads"].items():
        print(f"{k:28s} CRPS {v:.4f}")
    for k, v in out["pairs"].items():
        p = v["pooled"]
        print(f"{k:30s} " + "  ".join(f"{ds}: {v[ds]['margin']:+.4f} [{v[ds]['ci95'][0]:+.4f},{v[ds]['ci95'][1]:+.4f}]"
                                      for ds in ("pronostia", "xjtu", "pooled")) + f"  holm={p['p_holm']:.3f}")


if __name__ == "__main__":
    main()
