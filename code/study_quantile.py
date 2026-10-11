"""Distribution-free baselines: ensembles of quantile-regression heads.

Two quantile heads share the backbone, features, schedule and seeds of the
likelihood heads and are trained by the pinball loss over 101 monotone quantiles
of the fraction:

  Quantile        unconstrained quantiles, which can leave (0,1)
  Quantile-logit  the same head with its quantiles on the logit scale, mapped to
                  (0,1) by the logistic function, so that the support is respected

They are compared with the truncated BS and IG heads at the reported and at the
fixed-time prediction points by the bearing-level paired bootstrap. The four
truncated-minus-quantile margins of each protocol form one Holm family; the
constrained-minus-unconstrained margin (the effect of the support within the
quantile heads) is reported on its own.

Writes results/regression/quantile.json.
"""
import json
import os

import numpy as np

import study_review_p2 as S

NB, SEED = 10000, 7
PROTOCOLS = {"main": (["results_both_quant.json", "results_both_quantl.json"], "results_both_causal.json"),
             "grid": (["results_both_quant_grid.json", "results_both_quantl_grid.json"], "results_both_grid.json")}
QHEADS = ["Quantile", "Quantile-logit"]
TRUNC = ["BS-trunc", "IGauss-trunc"]
PAIRS = [(a, b) for b in QHEADS for a in TRUNC]
SUPPORT_PAIR = ("Quantile-logit", "Quantile")


def analyse(qfiles, ff, rng):
    c, outside, tables = {}, {h: [] for h in QHEADS}, {}
    for qf in qfiles:
        dq = json.load(open(os.path.join(S.RES, qf)))
        for k, v in dq["tables"].items():
            for h in QHEADS:
                if v and h in v:
                    tables.setdefault(k, {})[h] = v[h]
        for r in dq["rows"]:
            if r["head"] not in QHEADS:
                continue
            q = np.asarray(r["q"], float)
            outside[r["head"]].append(float(np.mean((q <= 0) | (q >= 1))))
            c.setdefault((r["dataset"], r["bearing"]), {}).setdefault(r["head"], []).append(S.crps_q(q, r["true"]))
    for r in json.load(open(os.path.join(S.RES, ff)))["rows"]:
        if r["head"] in TRUNC and (r["dataset"], r["bearing"]) in c:
            c[(r["dataset"], r["bearing"])].setdefault(r["head"], []).append(
                S.crps_q(np.asarray(r["q"], float), r["true"]))
    arch = {ds: sorted(b for (dd, b) in c if dd == ds) for ds in ("pronostia", "xjtu")}

    def margins(sample, a, b):
        bm = {ds: [np.mean(c[(ds, x)][a]) - np.mean(c[(ds, x)][b]) for x in sample[ds]] for ds in sample}
        return {"pronostia": float(np.mean(bm["pronostia"])), "xjtu": float(np.mean(bm["xjtu"])),
                "pooled": float(np.mean(bm["pronostia"] + bm["xjtu"]))}

    def boot(a, b):
        pt = margins(arch, a, b)
        bs = {k: [] for k in pt}
        for _ in range(NB):
            smp = {ds: [arch[ds][i] for i in rng.choice(len(arch[ds]), len(arch[ds]))] for ds in arch}
            for k, v in margins(smp, a, b).items():
                bs[k].append(v)
        return {k: {"margin": pt[k], "ci95": [float(np.quantile(bs[k], 0.025)), float(np.quantile(bs[k], 0.975))],
                    "p_boot": float(min(1.0, 2 * min(np.mean(np.array(bs[k]) >= 0), np.mean(np.array(bs[k]) <= 0))))}
                for k in pt}

    out = {"tables": tables,
           "share_of_quantiles_outside_01": {h: float(np.mean(v)) for h, v in outside.items()},
           "pairs": {}}
    for a, b in PAIRS:
        out["pairs"][f"{a} - {b}"] = {"family": "truncated_vs_quantile", **boot(a, b)}
    keys = [f"{a} - {b}" for a, b in PAIRS]
    run = 0.0
    for i, k in enumerate(sorted(keys, key=lambda k: out["pairs"][k]["pooled"]["p_boot"])):
        run = max(run, min(1.0, (len(keys) - i) * out["pairs"][k]["pooled"]["p_boot"]))
        out["pairs"][k]["pooled"]["p_holm"] = run
    a, b = SUPPORT_PAIR
    v = boot(a, b)
    v["pooled"]["p_holm"] = v["pooled"]["p_boot"]          # a single comparison
    out["pairs"][f"{a} - {b}"] = {"family": "support_within_quantile", **v}
    return out


def main():
    rng = np.random.default_rng(SEED)
    res = {}
    for name, (qfs, ff) in PROTOCOLS.items():
        if not all(os.path.exists(os.path.join(S.RES, f)) for f in qfs + [ff]):
            print("skip", name)
            continue
        res[name] = o = analyse(qfs, ff, rng)
        print(f"=== {name}: share of quantiles outside (0,1) {o['share_of_quantiles_outside_01']}")
        for k, v in o["tables"].items():
            for h, t in v.items():
                print(f"   {k:16s} {h:15s} MAE {t['MAE']:.4f} CRPS {t['CRPS']:.4f} cov95 {t['cov95']:.3f} PIT-KS {t['PIT_KS']:.3f}")
        for k, v in o["pairs"].items():
            p = v["pooled"]
            print(f"   {k:32s} pooled {p['margin']:+.4f} [{p['ci95'][0]:+.4f},{p['ci95'][1]:+.4f}] holm={p['p_holm']:.3f}"
                  f"   P {v['pronostia']['margin']:+.4f}  X {v['xjtu']['margin']:+.4f}")
    json.dump(res, open(os.path.join(S.RES, "quantile.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
