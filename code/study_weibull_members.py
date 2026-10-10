"""Shape parameter of every Weibull ensemble member at every prediction point.

The 1/R-weighted median needs E[1/R] < infinity.  The Weibull predictive law is
an equal mixture of M = 5 Weibull members, and the mixture has a finite E[1/R]
only if every member does, i.e. only if every member shape k_m > 1.  The stored
results keep 400 draws per prediction, not member parameters, so this script
refits the Weibull head alone with the identical folds, seeds and settings of
study_regression.py, records (k_m, lambda_m) per member and point, and checks
that the refit reproduces the stored predictive medians.

Writes results/regression/weibull_members.json.
"""
import json
import os
import time

import numpy as np
import torch

import rul_dataset as D
import rulreg as R
import study_regression as SR


def fold(data, tr_b, va_b, te_b, dataset):
    Xtr, ytr, wtr, _ = D.stack(data, tr_b)
    Xva, yva, _, _ = D.stack(data, [va_b])
    Xtr_s, Xva_s, (mu, sd) = D.standardise(Xtr, Xva)
    f32 = lambda a: a.astype(np.float32)
    head = SR.make_head("Weibull")
    models = SR._fit_ensemble(head, f32(Xtr_s), f32(ytr), f32(wtr), f32(Xva_s), f32(yva))
    out = []
    for b in te_b:
        rec = data[b]
        pts, _ = SR._eval_points(rec, dataset, b)
        if not pts:
            continue
        Xe = f32((rec["X"][pts] - mu) / sd)
        T = max(rec["t"][-1], 1e-9)
        pr = R.predict(head, models, Xe)
        with torch.no_grad():
            P = [head.params(m(torch.tensor(Xe))) for m in models]
        for j, k in enumerate(pts):
            if rec["rul"][k] / T <= 0:
                continue
            out.append({"dataset": dataset, "bearing": b, "k": int(k),
                        "shape": [float(p[0][j]) for p in P],
                        "scale": [float(p[1][j]) for p in P],
                        "median": float(np.quantile(pr["samples"][j], 0.5))})
    return out


def main():
    rows = []
    t0 = time.time()
    raw = D.load("pronostia", t_scale=1.0)
    ml = float(np.median([raw[b]["n"] for b in D.PRONOSTIA_LEARN])) * D.DT["pronostia"]
    data = D.load("pronostia", t_scale=ml)
    rows += fold(data, [b for b in D.PRONOSTIA_LEARN if b != "Bearing2_2"],
                 "Bearing2_2", D.PRONOSTIA_TEST, "pronostia")
    print(f"pronostia done ({time.time()-t0:.0f} s)", flush=True)
    raw = D.load("xjtu", t_scale=1.0)
    names = sorted(raw)
    for cond in ("1", "2", "3"):
        te = [b for b in names if b.startswith(f"Bearing{cond}_")]
        tr_all = [b for b in names if b not in te]
        ml = float(np.median([raw[b]["n"] for b in tr_all])) * D.DT["xjtu"]
        data = D.load("xjtu", t_scale=ml)
        va = tr_all[len(tr_all) // 2]
        rows += fold(data, [b for b in tr_all if b != va], va, te, "xjtu")
        print(f"xjtu condition {cond} done ({time.time()-t0:.0f} s)", flush=True)

    stored = {(r["dataset"], r["bearing"], r["k"]): r["median"]
              for r in json.load(open(os.path.join(SR.RES, "results_both.json")))["rows"]
              if r["head"] == "Weibull"}
    summ = {}
    for d in ("pronostia", "xjtu"):
        R_ = [r for r in rows if r["dataset"] == d]
        kmin = np.array([min(r["shape"]) for r in R_])
        dev = [abs(r["median"] - stored[(d, r["bearing"], r["k"])]) for r in R_]
        summ[d] = {"n": len(R_), "n_min_shape_le_1": int(np.sum(kmin <= 1.0)),
                   "min_shape": float(kmin.min()), "median_min_shape": float(np.median(kmin)),
                   "all_members_shape_min": float(min(min(r["shape"]) for r in R_)),
                   "max_abs_median_dev_from_stored": float(max(dev))}
    json.dump({"summary": summ, "rows": rows},
              open(os.path.join(SR.RES, "weibull_members.json"), "w"), indent=1)
    print(json.dumps(summ, indent=1))


if __name__ == "__main__":
    main()
