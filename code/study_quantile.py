"""A distribution-free baseline: an ensemble of quantile-regression heads.

The quantile head shares the backbone, features, schedule and seeds of the
likelihood heads and is trained by the pinball loss over 101 monotone quantiles
of the fraction. It is compared with the truncated BS and IG heads at the
reported and at the fixed-time prediction points, by the same bearing-level
paired bootstrap (Holm over the two comparisons of each protocol).

Writes results/regression/quantile.json.
"""
import json
import os

import numpy as np

import study_review_p2 as S

NB, SEED = 10000, 7
PROTOCOLS = {"main": ("results_both_quant.json", "results_both_causal.json"),
             "grid": ("results_both_quant_grid.json", "results_both_grid.json")}
TRUNC = ["BS-trunc", "IGauss-trunc"]


def analyse(qf, ff, rng):
    dq = json.load(open(os.path.join(S.RES, qf)))
    df = json.load(open(os.path.join(S.RES, ff)))
    c, outside = {}, []
    for r in dq["rows"]:
        if r["head"] != "Quantile":
            continue
        q = np.asarray(r["q"], float)
        outside.append(float(np.mean((q <= 0) | (q >= 1))))
        c.setdefault((r["dataset"], r["bearing"]), {}).setdefault("Quantile", []).append(S.crps_q(q, r["true"]))
    for r in df["rows"]:
        if r["head"] in TRUNC and (r["dataset"], r["bearing"]) in c:
            c[(r["dataset"], r["bearing"])].setdefault(r["head"], []).append(
                S.crps_q(np.asarray(r["q"], float), r["true"]))
    arch = {ds: sorted(b for (dd, b) in c if dd == ds) for ds in ("pronostia", "xjtu")}

    def margins(sample, a, b):
        bm = {ds: [np.mean(c[(ds, x)][a]) - np.mean(c[(ds, x)][b]) for x in sample[ds]] for ds in sample}
        return {"pronostia": float(np.mean(bm["pronostia"])), "xjtu": float(np.mean(bm["xjtu"])),
                "pooled": float(np.mean(bm["pronostia"] + bm["xjtu"]))}

    out = {"tables": {k: v["Quantile"] for k, v in dq["tables"].items() if v and "Quantile" in v},
           "share_of_quantiles_outside_01": float(np.mean(outside)), "pairs": {}}
    for a in TRUNC:
        pt = margins(arch, a, "Quantile")
        bs = {k: [] for k in pt}
        for _ in range(NB):
            smp = {ds: [arch[ds][i] for i in rng.choice(len(arch[ds]), len(arch[ds]))] for ds in arch}
            for k, v in margins(smp, a, "Quantile").items():
                bs[k].append(v)
        out["pairs"][f"{a} - Quantile"] = {
            k: {"margin": pt[k], "ci95": [float(np.quantile(bs[k], 0.025)), float(np.quantile(bs[k], 0.975))],
                "p_boot": float(min(1.0, 2 * min(np.mean(np.array(bs[k]) >= 0), np.mean(np.array(bs[k]) <= 0))))}
            for k in pt}
    keys = list(out["pairs"])
    run = 0.0
    for i, k in enumerate(sorted(keys, key=lambda k: out["pairs"][k]["pooled"]["p_boot"])):
        run = max(run, min(1.0, (len(keys) - i) * out["pairs"][k]["pooled"]["p_boot"]))
        out["pairs"][k]["pooled"]["p_holm"] = run
    return out


def main():
    rng = np.random.default_rng(SEED)
    res = {}
    for name, (qf, ff) in PROTOCOLS.items():
        if not all(os.path.exists(os.path.join(S.RES, f)) for f in (qf, ff)):
            print("skip", name)
            continue
        res[name] = o = analyse(qf, ff, rng)
        print(f"=== {name}: share of quantiles outside (0,1) {o['share_of_quantiles_outside_01']:.3f}")
        for k, v in o["tables"].items():
            print(f"   {k:16s} MAE {v['MAE']:.4f} CRPS {v['CRPS']:.4f} cov95 {v['cov95']:.3f} PIT-KS {v['PIT_KS']:.3f}")
        for k, v in o["pairs"].items():
            p = v["pooled"]
            print(f"   {k:24s} pooled {p['margin']:+.4f} [{p['ci95'][0]:+.4f},{p['ci95'][1]:+.4f}] holm={p['p_holm']:.3f}"
                  f"   P {v['pronostia']['margin']:+.4f} [{v['pronostia']['ci95'][0]:+.4f},{v['pronostia']['ci95'][1]:+.4f}]"
                  f"  X {v['xjtu']['margin']:+.4f} [{v['xjtu']['ci95'][0]:+.4f},{v['xjtu']['ci95'][1]:+.4f}]")
    json.dump(res, open(os.path.join(S.RES, "quantile.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
