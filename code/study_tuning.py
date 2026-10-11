"""Equal-budget tuning of the learning rate for every head, on validation data only.

Every head shares one learning rate in the reported run (3e-3, fixed before the
bounded heads were added), and the robustness runs change it for all heads at
once. Here each head chooses its own: for every fold, each head is trained at the
same three learning rates {1e-3, 3e-3, 1e-2}, and the one with the lowest
ensemble CRPS on that fold's validation bearing is kept. The validation bearing is
the one already used for early stopping; no test bearing enters the choice.

Training is deterministic, so the test predictions at a chosen rate are those
already stored by the runs at that rate (results_both_causal.json for 3e-3,
results_both_lr1e-3.json, results_both_lr1e-2.json); the refit checks this by
reproducing the stored predictive medians. The tuned predictions are assembled
from the stored runs and compared exactly as in study_robustness.py.

  python study_tuning.py fit      # refit, writes tuning_validation.json
  python study_tuning.py score    # writes results_both_tuned.json and tuning.json
"""
import json
import os
import sys
import time

os.environ.setdefault("GBS_BOUNDED", "1")

import numpy as np

RES = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results", "regression")
LRS = [1e-3, 3e-3, 1e-2]
STORED = {1e-3: "results_both_lr1e-3.json", 3e-3: "results_both_causal.json", 1e-2: "results_both_lr1e-2.json"}
VAL = os.path.join(RES, "tuning_validation.json")


def fit_fold(data, tr_b, va_b, te_b, dataset, stored):
    import rul_dataset as D
    import rulreg as R
    import study_regression as SR
    f32 = lambda a: a.astype(np.float32)
    Xtr, ytr, wtr, _ = D.stack(data, tr_b)
    Xva, yva, _, _ = D.stack(data, [va_b])
    Xtr_s, Xva_s, (mu, sd) = D.standardise(Xtr, Xva)
    out = []
    for lr in LRS:
        for h in SR.HEAD_ORDER:
            head = SR.make_head(h)
            fits = [R.train_head(head, f32(Xtr_s), f32(ytr), f32(wtr), f32(Xva_s), f32(yva),
                                 R.TrainCfg(epochs=SR.EPOCHS, seed=SR.SEED0 + s, hidden=SR.HIDDEN, lr=lr))
                    for s in range(SR.N_ENSEMBLE)]
            models = [m for m, _ in fits]
            pc = R.predict(head, models, f32(Xva_s))
            crps_val = float(np.mean(R.crps_from_samples(pc["samples"], yva)))
            dev = 0.0
            for b in te_b:
                rec = data[b]
                pts, _ = SR._eval_points(rec, dataset, b)
                if not pts:
                    continue
                T = max(rec["t"][-1], 1e-9)
                S = R.predict(head, models, f32((rec["X"][pts] - mu) / sd))["samples"]
                for j, k in enumerate(pts):
                    if rec["rul"][k] / T > 0:
                        dev = max(dev, abs(float(np.quantile(S[j], 0.5)) - stored[lr][(dataset, b, int(k), h)]))
            out.append({"dataset": dataset, "validation": va_b, "test": list(te_b), "head": h, "lr": lr,
                        "val_crps": crps_val, "val_nll": float(np.mean([v for _, v in fits])),
                        "max_median_dev": dev})
            print(f"  {dataset} {va_b} lr={lr:g} {h:13s} val CRPS {crps_val:.4f}  dev {dev:.1e}", flush=True)
    return out


def fit():
    import rul_dataset as D
    lock = VAL + ".lock"
    if os.path.exists(lock) or os.path.exists(VAL):
        print(f"{VAL} exists or is being written; delete it (and {lock}) to refit")
        return
    open(lock, "w").close()
    stored = {}
    for lr, f in STORED.items():
        stored[lr] = {(r["dataset"], r["bearing"], r["k"], r["head"]): r["median"]
                      for r in json.load(open(os.path.join(RES, f)))["rows"] if r["q"] is not None}
    rows, t0 = [], time.time()
    raw = D.load("pronostia", t_scale=1.0)
    ml = float(np.median([raw[b]["n"] for b in D.PRONOSTIA_LEARN])) * D.DT["pronostia"]
    data = D.load("pronostia", t_scale=ml)
    rows += fit_fold(data, [b for b in D.PRONOSTIA_LEARN if b != "Bearing2_2"],
                     "Bearing2_2", D.PRONOSTIA_TEST, "pronostia", stored)
    print(f"pronostia done ({time.time()-t0:.0f} s)", flush=True)
    raw = D.load("xjtu", t_scale=1.0)
    names = sorted(raw)
    for cond in ("1", "2", "3"):
        te = [b for b in names if b.startswith(f"Bearing{cond}_")]
        tr_all = [b for b in names if b not in te]
        ml = float(np.median([raw[b]["n"] for b in tr_all])) * D.DT["xjtu"]
        data = D.load("xjtu", t_scale=ml)
        va = tr_all[len(tr_all) // 2]
        rows += fit_fold(data, [b for b in tr_all if b != va], va, te, "xjtu", stored)
        print(f"xjtu condition {cond} done ({time.time()-t0:.0f} s)", flush=True)
    json.dump(rows, open(VAL, "w"), indent=1)
    os.remove(lock)


def score():
    import study_robustness as RB
    val = json.load(open(VAL))
    print("max |median - stored| over every refit:", max(v["max_median_dev"] for v in val))
    choice, by_nll = {}, {}
    for v in val:
        #  one choice per fold and head; two XJTU-SY folds share a validation bearing
        key = (v["dataset"], v["test"][0], v["head"])
        if key not in choice or v["val_crps"] < choice[key]["val_crps"]:
            choice[key] = v
        if key not in by_nll or v["val_nll"] < by_nll[key]["val_nll"]:
            by_nll[key] = v
    stored = {lr: json.load(open(os.path.join(RES, f))) for lr, f in STORED.items()}
    rows = []
    for (ds, va, h), v in choice.items():
        test = set(v["test"])
        rows += [r for r in stored[v["lr"]]["rows"]
                 if r["dataset"] == ds and r["head"] == h and r["bearing"] in test and r["q"] is not None]
    for r in rows:
        r.pop("samples", None)
    json.dump({"rows": rows, "tables": {}}, open(os.path.join(RES, "results_both_tuned.json"), "w"))
    res = RB.analyse([os.path.join(RES, "results_both_tuned.json")], np.random.default_rng(11))
    res["choice"] = {f"{ds}/{va}/{h}": v["lr"] for (ds, va, h), v in choice.items()}
    res["agree_with_nll"] = int(sum(choice[k]["lr"] == by_nll[k]["lr"] for k in choice))
    res["n_choices"] = len(choice)
    res["n_at_reported_lr"] = int(sum(v["lr"] == 3e-3 for v in choice.values()))
    json.dump(res, open(os.path.join(RES, "tuning.json"), "w"), indent=1)
    from collections import Counter
    print("chosen rates per head:", {h: dict(Counter(v["lr"] for (d, a, hh), v in choice.items() if hh == h))
                                     for h in sorted({k[2] for k in choice})})
    print(f"choices at the reported 3e-3: {res['n_at_reported_lr']}/{res['n_choices']}; "
          f"same choice by validation NLL: {res['agree_with_nll']}/{res['n_choices']}")
    for k, x in res["pairs"].items():
        p = x["pooled"]
        print(f"   {k:28s} pooled {p['margin']:+.4f} [{p['ci95'][0]:+.4f},{p['ci95'][1]:+.4f}] holm={p['p_holm']:.3f}"
              f"   P {x['pronostia']['margin']:+.4f}  X {x['xjtu']['margin']:+.4f}")


if __name__ == "__main__":
    {"fit": fit, "score": score}[sys.argv[1] if len(sys.argv) > 1 else "score"]()
