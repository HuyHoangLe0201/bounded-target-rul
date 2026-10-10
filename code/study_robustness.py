"""Do the pooled comparisons survive changes of protocol?

The comparisons of study_bounded.py (truncated first-passage heads against the
six conventional heads and against the two bounded families) are repeated on
every run that changes one element of the protocol:

  main     the reported run (results_both_causal.json)
  grid     prediction points at every 50th (PRONOSTIA) or 10th (XJTU-SY)
           acquisition after the healthy window, chosen without the total life
  seeds5   ensemble seeds 5..9 instead of 0..4: retraining variability
  lr1e-3   learning rate 1e-3 for every head (reported: 3e-3)
  lr1e-2   learning rate 1e-2 for every head

CRPS from the stored quantile grid for every head alike; bearing-level paired
differences; pooled margin = mean over the 26 bearings; bearings resampled within
each benchmark; Holm within each family. Missing runs are skipped.

Writes results/regression/robustness.json.
"""
import json
import os

import numpy as np

import study_review_p2 as S

NB, SEED = 10000, 5
RUNS = {"main": "results_both_causal.json", "grid": "results_both_grid.json",
        "seeds5": "results_both_seeds5.json", "lr1e-3": "results_both_lr1e-3.json",
        "lr1e-2": "results_both_lr1e-2.json",
        #  backbone sweep: the six conventional heads of each arm, paired with the
        #  four bounded heads run afterwards on the same backbone and settings
        "w32": ["results_both_w32.json", "results_both_w32b.json"],
        "w48": ["results_both_w48.json", "results_both_w48b.json"],
        "w96": ["results_both_w96.json", "results_both_w96b.json"],
        "big": ["results_both_big.json", "results_both_bigb.json"],
        "w256": ["results_both_w256.json", "results_both_w256b.json"]}
CONV = ["BS", "BS-t", "Weibull", "LogNormal", "IGauss", "Gaussian"]
FAMILIES = {
    "vs_conventional": [(a, b) for a in ("BS-trunc", "IGauss-trunc") for b in CONV],
    "bounded": [("IGauss-trunc", "BS-trunc"), ("IGauss-trunc", "Beta"), ("IGauss-trunc", "LogitNormal"),
                ("BS-trunc", "Beta"), ("BS-trunc", "LogitNormal")],
}


def load_merged(paths):
    """Rows and tables of one or several runs; several runs are concatenated."""
    d = {"rows": [], "tables": {}}
    for p in paths:
        x = json.load(open(p))
        d["rows"] += x["rows"]
        for k, v in x["tables"].items():
            d["tables"].setdefault(k, {}).update(v)
    return d


def analyse(paths, rng):
    d = load_merged(paths)
    c = {}
    for r in d["rows"]:
        if r["q"] is None:
            continue
        c.setdefault((r["dataset"], r["bearing"]), {}).setdefault(r["head"], []).append(
            S.crps_q(np.asarray(r["q"], float), r["true"]))
    arch = {ds: sorted(b for (dd, b) in c if dd == ds) for ds in ("pronostia", "xjtu")}

    def margins(sample, a, b):
        bm = {ds: [np.mean(c[(ds, x)][a]) - np.mean(c[(ds, x)][b]) for x in sample[ds]] for ds in sample}
        return {"pronostia": float(np.mean(bm["pronostia"])), "xjtu": float(np.mean(bm["xjtu"])),
                "pooled": float(np.mean(bm["pronostia"] + bm["xjtu"]))}

    out = {"tables": {k: {h: {m: v[h][m] for m in ("MAE", "CRPS", "cov95", "PIT_KS", "n")}
                          for h in v} for k, v in d["tables"].items() if v},
           "pairs": {}}
    for fam, pairs in FAMILIES.items():
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
    n_pts = {ds: sum(len(c[(ds, x)]["IGauss"]) for x in arch[ds]) for ds in arch}
    out["n_points"] = n_pts
    return out


def main():
    rng = np.random.default_rng(SEED)
    res = {}
    for name, f in RUNS.items():
        ps = [os.path.join(S.RES, x) for x in ([f] if isinstance(f, str) else f)]
        if not all(os.path.exists(x) for x in ps):
            print(f"skip {name} ({f} missing)")
            continue
        res[name] = analyse(ps, rng)
        n_sig = sum(v["pooled"]["p_holm"] < 0.05 and v["pooled"]["margin"] < 0
                    for k, v in res[name]["pairs"].items() if v["family"] == "vs_conventional")
        print(f"{name:7s} points {res[name]['n_points']}  truncated heads beat conventional "
              f"(Holm p<0.05) in {n_sig}/12 pooled comparisons")
        for k, v in res[name]["pairs"].items():
            p = v["pooled"]
            print(f"   {k:28s} pooled {p['margin']:+.4f} [{p['ci95'][0]:+.4f},{p['ci95'][1]:+.4f}] holm={p['p_holm']:.3f}"
                  f"   P {v['pronostia']['margin']:+.4f}  X {v['xjtu']['margin']:+.4f}")
    json.dump(res, open(os.path.join(S.RES, "robustness.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
