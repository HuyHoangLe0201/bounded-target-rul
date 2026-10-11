"""Heads trained on the remaining-to-elapsed ratio w = R/t.

On w the target's support is (0, inf). The five positive conventional families
match it; the Gaussian does not, so it enters twice: untruncated (its mass at
w <= 0 read as failure now, y = 0, as in eq. Rext) and truncated at zero,
fitted by log f(w) - log(1 - F(0)). Every law is also stored on the fraction
scale through y = w/(1+w), so the two designs are compared at the same points.

Three scores per prediction:
  CRPS_y   on the fraction, against the truncated BS and IG heads trained on y
  CRPS_w   on w, by the kernel form, so for the heads trained on w only
  CRPS_h   in hours, t_k * CRPS_w (CRPS scales with a positive constant), so
           each inspection is weighted by its own elapsed time
The truncated heads trained on y have infinite mean on w and in hours, so the
kernel form does not apply to them; study_hours_crps.py scores every head in
hours by the integral definition instead, and the paper reports that.

Two protocols: the reported prediction points, and inspections fixed in
operating time (GBS_EVAL=grid). Bearing-level paired margins, pooled over the
26 bearings, bootstrap within each benchmark, Holm within each family.
Writes results/regression/ratio.json.
"""
import json
import os

import numpy as np

import study_review_p2 as S

NB, SEED = 10000, 6
DT = {"pronostia": 10.0, "xjtu": 60.0}
WHEADS = ["BS", "BS-t", "Weibull", "LogNormal", "IGauss", "Gaussian", "Gaussian-trunc0"]
TRUNC = ["BS-trunc", "IGauss-trunc"]
PROTOCOLS = {
    "main": (["results_both_ratio.json", "results_both_ratio_g0.json"], "results_both_causal.json"),
    "grid": (["results_both_ratio_grid.json", "results_both_ratio_grid_g0.json"], "results_both_grid.json"),
}


def crps(q, y):
    return S.crps_q(np.asarray(q, float), y)


def analyse(ratio_files, frac_file, rng):
    c = {}       # (ds, bearing) -> score key -> list
    tables = {}
    for f in ratio_files:
        d = json.load(open(os.path.join(S.RES, f)))
        for k, v in d["tables"].items():
            tables.setdefault(k, {}).update({h: {m: v[h][m] for m in ("MAE", "CRPS", "cov95", "PIT_KS")}
                                             for h in v if h in WHEADS})
        for r in d["rows"]:
            if r["q"] is None or r["head"] not in WHEADS:
                continue
            key = (r["dataset"], r["bearing"])
            w_true = r["true"] / (1 - r["true"])
            t_h = r["k"] * DT[r["dataset"]] / 3600.0
            cw = crps(r["q_w"], w_true)
            dd = c.setdefault(key, {})
            dd.setdefault(r["head"] + "@y", []).append(crps(r["q"], r["true"]))
            dd.setdefault(r["head"] + "@w", []).append(cw)
            dd.setdefault(r["head"] + "@h", []).append(t_h * cw)
    for r in json.load(open(os.path.join(S.RES, frac_file)))["rows"]:
        if r["head"] in TRUNC and (r["dataset"], r["bearing"]) in c:
            c[(r["dataset"], r["bearing"])].setdefault(r["head"] + "@y", []).append(crps(r["q"], r["true"]))
    arch = {ds: sorted(b for (dd, b) in c if dd == ds) for ds in DT}

    out = {"heads": {}, "tables": tables, "pairs": {},
           "n_points": {ds: sum(len(c[(ds, b)]["IGauss@y"]) for b in arch[ds]) for ds in arch}}
    for ds in arch:
        for key in [h + s for h in WHEADS for s in ("@y", "@w", "@h")] + [h + "@y" for h in TRUNC]:
            out["heads"][f"{ds}/{key}"] = float(np.mean([v for b in arch[ds] for v in c[(ds, b)][key]]))

    def margins(sample, a, b):
        bm = {ds: [np.mean(c[(ds, x)][a]) - np.mean(c[(ds, x)][b]) for x in sample[ds]] for ds in sample}
        return {"pronostia": float(np.mean(bm["pronostia"])), "xjtu": float(np.mean(bm["xjtu"])),
                "pooled": float(np.mean(bm["pronostia"] + bm["xjtu"]))}

    def pooled_mean(key):
        return float(np.mean([np.mean(c[k][key]) for k in c]))

    lead_w = min(WHEADS, key=lambda h: pooled_mean(h + "@w"))
    lead_h = min(WHEADS, key=lambda h: pooled_mean(h + "@h"))
    out["leaders"] = {"w": lead_w, "h": lead_h}
    fams = {
        "fraction": [(a + "@y", b + "@y") for a in TRUNC for b in WHEADS],
        "ratio": [(lead_w + "@w", b + "@w") for b in WHEADS if b != lead_w],
        "hours": [(lead_h + "@h", b + "@h") for b in WHEADS if b != lead_h],
    }
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
    return out


def main():
    rng = np.random.default_rng(SEED)
    res = {}
    for name, (rf, ff) in PROTOCOLS.items():
        if not all(os.path.exists(os.path.join(S.RES, f)) for f in rf + [ff]):
            print(f"skip {name}")
            continue
        res[name] = analyse(rf, ff, rng)
        o = res[name]
        print(f"=== {name}: points {o['n_points']}, leaders {o['leaders']}")
        for k, v in o["heads"].items():
            print(f"   {k:34s} {v:.4f}")
        for k, v in o["pairs"].items():
            p = v["pooled"]
            print(f"   {k:36s} pooled {p['margin']:+.4f} [{p['ci95'][0]:+.4f},{p['ci95'][1]:+.4f}]"
                  f" holm={p['p_holm']:.3f}   P {v['pronostia']['margin']:+.4f}  X {v['xjtu']['margin']:+.4f}")
    json.dump(res, open(os.path.join(S.RES, "ratio.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
