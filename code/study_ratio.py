"""Heads trained on the remaining-to-elapsed ratio w = R/t.

On w the target's support is (0, inf), the support of the conventional heads,
so no truncation is needed, and a law with a finite mean has a finite CRPS on
w -- and in hours, since R = t*w at the known elapsed time. The run
results_both_ratio.json stores every law on the fraction scale too, through
y = w/(1+w), so the two designs are compared at the same prediction points:

  fraction  CRPS on y for the six heads trained on w, against the truncated
            BS and IG heads trained on y (results_both_causal.json)
  ratio     CRPS on w for the six heads trained on w; the truncated heads have
            an infinite CRPS there (their law of y has positive density at one)

Bearing-level paired margins, pooled over the 26 bearings, bootstrap within each
benchmark, Holm within each family. Writes results/regression/ratio.json.
"""
import json
import os

import numpy as np

import study_review_p2 as S

NB, SEED = 10000, 6
CONV = ["BS", "BS-t", "Weibull", "LogNormal", "IGauss", "Gaussian"]
TRUNC = ["BS-trunc", "IGauss-trunc"]


def crps(q, y):
    return S.crps_q(np.asarray(q, float), y)


def main():
    rr = json.load(open(os.path.join(S.RES, "results_both_ratio.json")))
    rc = json.load(open(os.path.join(S.RES, "results_both_causal.json")))
    c = {}   # (ds, bearing) -> key -> list
    for r in rr["rows"]:
        if r["q"] is None:
            continue
        key = (r["dataset"], r["bearing"])
        w_true = r["true"] / (1 - r["true"])
        c.setdefault(key, {}).setdefault(r["head"] + "@ratio", []).append(crps(r["q"], r["true"]))
        c[key].setdefault(r["head"] + "@ratio_w", []).append(crps(r["q_w"], w_true))
    for r in rc["rows"]:
        if r["head"] in TRUNC:
            c[(r["dataset"], r["bearing"])].setdefault(r["head"], []).append(crps(r["q"], r["true"]))
    arch = {ds: sorted(b for (dd, b) in c if dd == ds) for ds in ("pronostia", "xjtu")}

    out = {"heads": {}, "tables": {k: {h: {m: v[h][m] for m in ("MAE", "CRPS", "cov95", "PIT_KS")}
                                       for h in v} for k, v in rr["tables"].items() if v},
           "pairs": {}}
    for ds in arch:
        for h in [x + "@ratio" for x in CONV] + [x + "@ratio_w" for x in CONV] + TRUNC:
            out["heads"][f"{ds}/{h}"] = float(np.mean([v for b in arch[ds] for v in c[(ds, b)][h]]))

    rng = np.random.default_rng(SEED)

    def margins(sample, a, b):
        bm = {ds: [np.mean(c[(ds, x)][a]) - np.mean(c[(ds, x)][b]) for x in sample[ds]] for ds in sample}
        return {"pronostia": float(np.mean(bm["pronostia"])), "xjtu": float(np.mean(bm["xjtu"])),
                "pooled": float(np.mean(bm["pronostia"] + bm["xjtu"]))}

    #  leader on w among the ratio-trained heads (pooled bearing-equal CRPS on w)
    pooled_w = {h: np.mean([np.mean(c[k][h + "@ratio_w"]) for k in c]) for h in CONV}
    lead = min(pooled_w, key=pooled_w.get)
    fams = {
        "fraction: truncated vs ratio-trained": [(a, b + "@ratio") for a in TRUNC for b in CONV],
        "ratio scale: leader vs other ratio-trained": [(lead + "@ratio_w", b + "@ratio_w")
                                                       for b in CONV if b != lead],
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
    out["ratio_leader"] = lead
    json.dump(out, open(os.path.join(S.RES, "ratio.json"), "w"), indent=1)
    for k, v in out["heads"].items():
        print(f"{k:28s} {v:.4f}")
    print("leader on w:", lead)
    for k, v in out["pairs"].items():
        p = v["pooled"]
        print(f"{k:36s} " + "  ".join(f"{ds}: {v[ds]['margin']:+.4f} [{v[ds]['ci95'][0]:+.4f},{v[ds]['ci95'][1]:+.4f}]"
                                      for ds in ("pronostia", "xjtu", "pooled")) + f"  holm={p['p_holm']:.3f}")


if __name__ == "__main__":
    main()
