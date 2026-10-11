"""Variability over the choice of training bearings: PRONOSTIA re-split.

The bootstrap over test bearings holds the fitted models fixed, and the seed and
learning-rate runs keep the official split. Here the PRONOSTIA records are
re-partitioned: in each of five splits, two bearings per operating condition
are drawn for training, as in the official split, the second one of condition 2
serving for early stopping as Bearing2_2 does there, and the other eleven are
held out. Every head is refitted with the settings of the reported run, and the
pooled comparisons are repeated on each split's eleven test bearings.

  python study_resplit.py fit      # writes results_resplit.json (rows per split)
  python study_resplit.py score    # writes resplit.json
"""
import json
import os
import sys
import time

os.environ.setdefault("GBS_BOUNDED", "1")

import numpy as np

RES = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results", "regression")
ROWS = os.path.join(RES, "results_resplit.json")
N_SPLITS, SPLIT_SEED, NB, SEED = 5, 2026, 10000, 12
CONV = ["BS", "BS-t", "Weibull", "LogNormal", "IGauss", "Gaussian"]
FAMILIES = {
    "vs_conventional": [(a, b) for a in ("BS-trunc", "IGauss-trunc") for b in CONV],
    "bounded": [("IGauss-trunc", "Beta"), ("IGauss-trunc", "LogitNormal"),
                ("BS-trunc", "Beta"), ("BS-trunc", "LogitNormal")],
    "pair": [("IGauss-trunc", "BS-trunc")],
}


def splits():
    import rul_dataset as D
    names = sorted(D.PRONOSTIA_LEARN + D.PRONOSTIA_TEST)
    cond = {c: [b for b in names if b.startswith(f"Bearing{c}_")] for c in "123"}
    official = sorted(D.PRONOSTIA_LEARN)
    rng = np.random.default_rng(SPLIT_SEED)
    out, seen = [], {tuple(official)}
    while len(out) < N_SPLITS:
        pick = {c: [str(b) for b in rng.choice(cond[c], 2, replace=False)] for c in "123"}
        tr = sorted(pick["1"] + pick["2"] + pick["3"])
        if tuple(tr) in seen:
            continue
        seen.add(tuple(tr))
        out.append({"train": tr, "validation": pick["2"][1], "test": [b for b in names if b not in tr]})
    return out


def fit():
    import rul_dataset as D
    import study_regression as SR
    rows, t0 = [], time.time()
    raw = D.load("pronostia", t_scale=1.0)
    for i, sp in enumerate(splits()):
        ml = float(np.median([raw[b]["n"] for b in sp["train"]])) * D.DT["pronostia"]
        data = D.load("pronostia", t_scale=ml)
        rr = SR._fold(data, [b for b in sp["train"] if b != sp["validation"]], sp["validation"],
                      sp["test"], "pronostia")
        for r in rr:
            r.pop("samples", None)
            r["split"] = i
        rows += rr
        print(f"split {i} {sp['train']} (validation {sp['validation']}) done ({time.time()-t0:.0f} s)", flush=True)
    json.dump({"splits": splits(), "rows": rows}, open(ROWS, "w"))


def compare(rows, rng):
    import study_review_p2 as S
    c = {}
    for r in rows:
        if r["q"] is not None:
            c.setdefault(r["bearing"], {}).setdefault(r["head"], []).append(S.crps_q(np.asarray(r["q"], float), r["true"]))
    bs_ = sorted(c)
    out = {}
    for fam, pairs in FAMILIES.items():
        for a, b in pairs:
            d = np.array([np.mean(c[x][a]) - np.mean(c[x][b]) for x in bs_])
            boot = np.array([d[rng.integers(0, len(d), len(d))].mean() for _ in range(NB)])
            out[f"{a} - {b}"] = {"family": fam, "margin": float(d.mean()),
                                 "ci95": [float(np.quantile(boot, 0.025)), float(np.quantile(boot, 0.975))],
                                 "p_boot": float(min(1.0, 2 * min(np.mean(boot >= 0), np.mean(boot <= 0))))}
        keys = [f"{a} - {b}" for a, b in pairs]
        run = 0.0
        for i, k in enumerate(sorted(keys, key=lambda k: out[k]["p_boot"])):
            run = max(run, min(1.0, (len(keys) - i) * out[k]["p_boot"]))
            out[k]["p_holm"] = run
    crps = {h: float(np.mean([np.mean(c[x][h]) for x in bs_])) for h in c[bs_[0]]}
    return out, crps


def score():
    d = json.load(open(ROWS))
    rng = np.random.default_rng(SEED)
    official = [r for r in json.load(open(os.path.join(RES, "results_both_causal.json")))["rows"]
                if r["dataset"] == "pronostia"]
    res = {"splits": d["splits"], "per_split": {}}
    runs = [("official", official)] + [(str(i), [r for r in d["rows"] if r["split"] == i]) for i in range(N_SPLITS)]
    for name, rr in runs:
        pairs, crps = compare(rr, rng)
        res["per_split"][name] = {"pairs": pairs, "crps": crps}
    keys = list(res["per_split"]["official"]["pairs"])
    summ = {}
    for k in keys:
        ms = [res["per_split"][str(i)]["pairs"][k] for i in range(N_SPLITS)]
        summ[k] = {"n_favourable": int(sum(m["margin"] < 0 for m in ms)),
                   "n_significant": int(sum(m["margin"] < 0 and m["p_holm"] < 0.05 for m in ms)),
                   "n_significant_against": int(sum(m["margin"] > 0 and m["p_holm"] < 0.05 for m in ms)),
                   "margins": [m["margin"] for m in ms],
                   "official": res["per_split"]["official"]["pairs"][k]["margin"]}
    res["summary"] = summ
    json.dump(res, open(os.path.join(RES, "resplit.json"), "w"), indent=1)
    for i, sp in enumerate(d["splits"]):
        print(f"split {i}: train {sp['train']}, validation {sp['validation']}")
    for k, v in summ.items():
        print(f"   {k:28s} favourable in {v['n_favourable']}/{N_SPLITS}, significant in {v['n_significant']}/{N_SPLITS}, "
              f"against {v['n_significant_against']}   official {v['official']:+.4f}   "
              f"splits {' '.join(f'{m:+.3f}' for m in v['margins'])}")
    for name in res["per_split"]:
        cr = res["per_split"][name]["crps"]
        print(f"   {name:9s} best: {sorted(cr, key=cr.get)[:4]}")


if __name__ == "__main__":
    {"fit": fit, "score": score}[sys.argv[1] if len(sys.argv) > 1 else "score"]()
