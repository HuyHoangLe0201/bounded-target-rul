"""
Direct RUL regression: which predictive distribution should the head use?

Target = fraction of life remaining, RUL_k / T_total in (0,1] (the protocol under
which the PRONOSTIA literature reports MAE/RMSE; see rul_dataset.stack for why
absolute-seconds RUL is not identifiable on this rig). Er% and the IEEE PHM 2012
score are invariant to that rescaling and remain comparable with the challenge.

Every head sits on an identical backbone, sees identical features and is trained
with an identical schedule and seeds, so differences arise under one shared
protocol. A point head trained by MAE, and a split-conformal
recalibration of each distributional head, are included to ask
whether the family matters once coverage is fixed post hoc.

Usage:  python study_regression.py [pronostia|xjtu|both]
"""
from __future__ import annotations

import json
import os
import sys
import time

import numpy as np
import torch

import rul_dataset as D
import rulreg as R

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "..", "results", "regression")
os.makedirs(RES, exist_ok=True)

HEAD_ORDER = ["BS", "BS-t", "Weibull", "LogNormal", "IGauss", "Gaussian"]
#  GBS_BOUNDED=1 adds the four heads that respect the (0,1] target in training
if os.environ.get("GBS_BOUNDED", "0") == "1":
    HEAD_ORDER = HEAD_ORDER + ["LogitNormal", "Beta", "BS-trunc", "IGauss-trunc"]
N_ENSEMBLE = 5
EVAL_FRACS = (0.4, 0.5, 0.6, 0.7, 0.8, 0.9)
EPOCHS = int(os.environ.get("GBS_EPOCHS", 250))

#  Backbone width.  The reported results use the default; setting GBS_HIDDEN
#  (e.g. "256,128,64") re-runs the identical protocol on a larger network, which
#  is how the ordering is tested for representation-independence.  RES_TAG keeps
#  the two sets of result files apart.
HIDDEN = tuple(int(h) for h in os.environ.get("GBS_HIDDEN", "48,24").split(","))
RES_TAG = os.environ.get("GBS_TAG", "")

#  Robustness options; the defaults reproduce the reported run exactly.
#    GBS_SEED0  first ensemble seed (members use SEED0 .. SEED0+4)
#    GBS_LR     learning rate of every head, the point head included
#    GBS_EVAL   "fractions" (40..90 % of life, plus the official PHM 2012 point)
#               or "grid": every STRIDE-th acquisition after the healthy window,
#               a set of inspections known in operation, chosen without the
#               bearing's total life
SEED0 = int(os.environ.get("GBS_SEED0", 0))
LR = float(os.environ.get("GBS_LR", 3e-3))
EVAL = os.environ.get("GBS_EVAL", "fractions")
STRIDE = {"pronostia": 50, "xjtu": 10}      # 500 s and 10 min


def make_head(name):
    if name == "BS-t":
        return R.BSTHead(nu=6.0)
    return R.HEADS[name]() if name in R.HEADS else R.BOUNDED_HEADS[name]()


def _fit_ensemble(head, Xtr, ytr, wtr, Xva, yva, epochs=EPOCHS, n=N_ENSEMBLE):
    return [R.train_head(head, Xtr, ytr, wtr, Xva, yva,
                         R.TrainCfg(epochs=epochs, seed=SEED0 + s, hidden=HIDDEN, lr=LR))[0]
            for s in range(n)]


def _fit_point(Xtr, ytr, wtr, Xva, yva, epochs=EPOCHS, n=N_ENSEMBLE):
    """Point head: identical backbone, MAE loss, no distribution.

    Included so that 'the distributional head costs nothing in point accuracy'
    is a measured statement rather than an assumption.
    """
    models = []
    for s in range(n):
        torch.manual_seed(SEED0 + s)
        m = R.MLP(Xtr.shape[1], hidden=HIDDEN)
        opt = torch.optim.AdamW(m.parameters(), lr=LR, weight_decay=1e-3)
        sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
        Xt, yt = torch.tensor(Xtr), torch.tensor(ytr)
        wt = torch.tensor(wtr)
        Xv, yv = torch.tensor(Xva), torch.tensor(yva)
        best, state, bad = np.inf, None, 0
        g = torch.Generator().manual_seed(SEED0 + s)
        for ep in range(epochs):
            m.train()
            perm = torch.randperm(len(yt), generator=g)
            for i in range(0, len(yt), 256):
                idx = perm[i:i + 256]
                pred = m(Xt[idx])[:, 0]
                loss = ((pred - yt[idx]).abs() * wt[idx]).sum() / wt[idx].sum()
                opt.zero_grad(); loss.backward()
                torch.nn.utils.clip_grad_norm_(m.parameters(), 5.0); opt.step()
            sch.step()
            m.eval()
            with torch.no_grad():
                v = (m(Xv)[:, 0] - yv).abs().mean().item()
            if v < best - 1e-5:
                best, bad = v, 0
                state = {k: t.clone() for k, t in m.state_dict().items()}
            else:
                bad += 1
                if bad >= 40:
                    break
        if state is not None:
            m.load_state_dict(state)
        m.eval()
        models.append(m)
    return models


def _point_predict(models, X):
    Xt = torch.tensor(X)
    with torch.no_grad():
        P = torch.stack([m(Xt)[:, 0] for m in models])
    return P.mean(0).numpy(), P.var(0, unbiased=False).numpy()


def conformal_widen(samples, cal_samples, cal_y, alpha=0.05):
    """Split-conformal recalibration of a predictive distribution.

    Nonconformity is the absolute deviation from the predictive median scaled by
    the predictive interquartile range, so the correction adapts to how uncertain
    each point claims to be. Returns recalibrated (lo, hi).
    """
    def med_iqr(S):
        q = np.quantile(S, [0.25, 0.5, 0.75], axis=1)
        return q[1], np.maximum(q[2] - q[0], 1e-9)

    m_c, s_c = med_iqr(cal_samples)
    score = np.abs(cal_y - m_c) / s_c
    n = len(score)
    k = min(n - 1, int(np.ceil((n + 1) * (1 - alpha))) - 1)
    q = np.sort(score)[max(k, 0)]
    m_t, s_t = med_iqr(samples)
    return m_t - q * s_t, m_t + q * s_t


def _metrics(point, S, y, v_al=None, v_ep=None, extra_lo=None, extra_hi=None):
    """`point` is the PREDICTIVE MEDIAN.

    Mean absolute error and the IEEE PHM relative error Er are both minimised by
    the median of the predictive distribution, not its mean. Scoring a skewed
    head by its mean therefore penalises the skew rather than the fit: for a BS
    head with a learned shape alpha = 4 the mean is nine times the median. The
    median is used as the point summary for every head, which for the Gaussian
    head is the same number and for the point head trained by MAE is what that
    loss already targets.
    """
    mean = point
    lo, hi = np.quantile(S, [0.025, 0.975], axis=1)
    lo90, hi90 = np.quantile(S, [0.05, 0.95], axis=1)
    err = mean - y
    er = 100.0 * (y - mean) / y
    out = {
        "n": len(y),
        "MAE": float(np.mean(np.abs(err))),
        "RMSE": float(np.sqrt(np.mean(err ** 2))),
        "MedAPE_pct": float(np.median(np.abs(err / y)) * 100),
        "mean_absEr_pct": float(np.mean(np.abs(er))),
        "PHM_score": float(np.mean(D.phm_score(er))),
        "CRPS": float(np.mean(R.crps_from_samples(S, y))),
        "cov95": float(np.mean((lo <= y) & (y <= hi))),
        "cov90": float(np.mean((lo90 <= y) & (y <= hi90))),
        "MPIW": float(np.mean(hi - lo)),
        "PIT_KS": float(R.ks_uniform(R.pit_values(S, y))),
    }
    if v_al is not None:
        out["rho"] = float(np.mean(v_ep / np.maximum(v_ep + v_al, 1e-12)))
    if extra_lo is not None:
        out["cov95_conf"] = float(np.mean((extra_lo <= y) & (y <= extra_hi)))
        out["MPIW_conf"] = float(np.mean(extra_hi - extra_lo))
    return out


def _eval_points(rec, dataset, bearing):
    n = rec["n"]
    if EVAL == "grid":
        k0 = max(20, rec.get("nh", 0))
        return list(range(k0, n - 1, STRIDE[dataset])), None
    pts = {int(f * (n - 1)) for f in EVAL_FRACS}
    official = None
    if dataset == "pronostia" and bearing in D.PHM12_TRUNC_N:
        official = D.PHM12_TRUNC_N[bearing] - 1
        pts.add(official)
    return sorted(k for k in pts if max(20, rec.get("nh", 0)) <= k < n - 1), official


def _fold(data, tr_b, va_b, te_b, dataset):
    """Fit every head on one fold and evaluate on the held-out bearings."""
    Xtr, ytr, wtr, _ = D.stack(data, tr_b)
    Xva, yva, _, _ = D.stack(data, [va_b])
    Xtr_s, Xva_s, (mu, sd) = D.standardise(Xtr, Xva)
    f32 = lambda a: a.astype(np.float32)

    rows = []
    fits = {}
    for h in HEAD_ORDER:
        head = make_head(h)
        fits[h] = (head, _fit_ensemble(head, f32(Xtr_s), f32(ytr), f32(wtr),
                                       f32(Xva_s), f32(yva)))
    point_models = _fit_point(f32(Xtr_s), f32(ytr), f32(wtr), f32(Xva_s), f32(yva))

    # calibration set for split conformal: the validation bearing
    cal = {}
    for h, (head, models) in fits.items():
        pc = R.predict(head, models, f32(Xva_s))
        cal[h] = (pc["samples"], yva)

    for b in te_b:
        rec = data[b]
        pts, official = _eval_points(rec, dataset, b)
        if not pts:
            continue
        Xe = f32((rec["X"][pts] - mu) / sd)
        T = max(rec["t"][-1], 1e-9)
        for h, (head, models) in fits.items():
            pr = R.predict(head, models, Xe)
            clo, chi = conformal_widen(pr["samples"], cal[h][0], cal[h][1])
            for j, k in enumerate(pts):
                y = float(rec["rul"][k] / T)
                if y <= 0:
                    continue
                rows.append({
                    "dataset": dataset, "head": h, "bearing": b, "k": k,
                    "official": bool(k == official),
                    "life_frac": float(rec["t"][k] / T),
                    "true": y, "mean": float(pr["mean"][j]),
                    "median": float(np.quantile(pr["samples"][j], 0.5)),
                    "q": np.quantile(pr["samples"][j],
                                     np.linspace(0.005, 0.995, 101)).tolist(),
                    "v_al": float(pr["v_aleatory"][j]),
                    "v_ep": float(pr["v_epistemic"][j]),
                    "conf_lo": float(clo[j]), "conf_hi": float(chi[j]),
                    "samples": pr["samples"][j].tolist(),
                })
        pm, pv = _point_predict(point_models, Xe)
        for j, k in enumerate(pts):
            y = float(rec["rul"][k] / T)
            if y <= 0:
                continue
            rows.append({"dataset": dataset, "head": "Point(MAE)", "bearing": b,
                         "k": k, "official": bool(k == official),
                         "life_frac": float(rec["t"][k] / T), "true": y,
                         "mean": float(pm[j]), "median": float(pm[j]), "q": None,
                         "v_al": 0.0, "v_ep": float(pv[j]),
                         "conf_lo": np.nan, "conf_hi": np.nan, "samples": None})
    return rows


def summarise(rows, subset=None):
    if subset is not None:
        rows = [r for r in rows if subset(r)]
    out = {}
    for h in HEAD_ORDER + ["Point(MAE)"]:
        s = [r for r in rows if r["head"] == h]
        if not s:
            continue
        y = np.array([r["true"] for r in s])
        mean = np.array([r["median"] for r in s])        # median: see _metrics
        if s[0]["samples"] is None:                      # point head
            err = mean - y
            er = 100.0 * (y - mean) / y
            out[h] = {"n": len(y), "MAE": float(np.mean(np.abs(err))),
                      "RMSE": float(np.sqrt(np.mean(err ** 2))),
                      "MedAPE_pct": float(np.median(np.abs(err / y)) * 100),
                      "mean_absEr_pct": float(np.mean(np.abs(er))),
                      "PHM_score": float(np.mean(D.phm_score(er))),
                      "CRPS": np.nan, "cov95": np.nan, "cov90": np.nan,
                      "MPIW": np.nan, "PIT_KS": np.nan, "rho": np.nan,
                      "cov95_conf": np.nan, "MPIW_conf": np.nan}
            continue
        S = np.sort(np.array([r["samples"] for r in s]), axis=1)
        out[h] = _metrics(mean, S, y,
                          np.array([r["v_al"] for r in s]),
                          np.array([r["v_ep"] for r in s]),
                          np.array([r["conf_lo"] for r in s]),
                          np.array([r["conf_hi"] for r in s]))
    return out


HDR = ["n", "MAE", "RMSE", "MedAPE_pct", "mean_absEr_pct", "PHM_score", "CRPS",
       "cov95", "MPIW", "cov95_conf", "MPIW_conf", "PIT_KS", "rho"]


def show(title, tbl):
    print(f"\n{title}")
    print(f"  {'head':11s} " + " ".join(f"{h:>9s}" for h in HDR))
    for k, v in tbl.items():
        print(f"  {k:11s} " + " ".join(
            f"{v[h]:9.0f}" if h == "n" else f"{v.get(h, float('nan')):9.4f}"
            for h in HDR))


def save(tables, rows, which):
    with open(os.path.join(RES, f"results_{which}{RES_TAG}.json"), "w") as f:
        json.dump({"tables": tables,
                   "rows": [{k: v for k, v in r.items() if k != "samples"}
                            for r in rows]}, f, indent=1, default=float)


if __name__ == "__main__":
    which = sys.argv[1] if len(sys.argv) > 1 else "both"
    allrows, tables = [], {}

    if which in ("pronostia", "both"):
        print("=" * 96 + "\nPRONOSTIA (official split: 6 learning / 11 test)\n" + "=" * 96,
              flush=True)
        raw = D.load("pronostia", t_scale=1.0)
        ml = float(np.median([raw[b]["n"] for b in D.PRONOSTIA_LEARN])) * D.DT["pronostia"]
        data = D.load("pronostia", t_scale=ml)
        t0 = time.time()
        rows = _fold(data, [b for b in D.PRONOSTIA_LEARN if b != "Bearing2_2"],
                     "Bearing2_2", D.PRONOSTIA_TEST, "pronostia")
        allrows += rows
        tables["pronostia/all"] = summarise(rows)
        show("[pronostia] all prediction points", tables["pronostia/all"])
        tables["pronostia/official"] = summarise(rows, lambda r: r["official"])
        show("[pronostia] official IEEE PHM 2012 truncation points",
             tables["pronostia/official"])
        print(f"\n  ({time.time()-t0:.0f} s)", flush=True)
        save(tables, allrows, which)
        print("  checkpoint written", flush=True)

    if which in ("xjtu", "both"):
        print("\n" + "=" * 96 +
              "\nXJTU-SY (leave-one-operating-condition-out)\n" + "=" * 96, flush=True)
        raw = D.load("xjtu", t_scale=1.0)
        names = sorted(raw)
        t0 = time.time()
        rows = []
        for cond in ("1", "2", "3"):
            te = [b for b in names if b.startswith(f"Bearing{cond}_")]
            tr_all = [b for b in names if b not in te]
            #  The elapsed-time scale is a training quantity, so it is rebuilt
            #  inside the fold from the two regimes actually in training. Taking
            #  the median over all fifteen records instead lets the held-out
            #  regime set the scale of a feature used to predict it: harmless on
            #  conditions 2 and 3, which move it by 1 per cent, but condition 1
            #  holds out the five shortest records and moves it by 168.
            #  PRONOSTIA never had this problem -- it uses the fixed official
            #  split, so its scale comes from the learning bearings by
            #  construction.
            ml = float(np.median([raw[b]["n"] for b in tr_all])) * D.DT["xjtu"]
            data = D.load("xjtu", t_scale=ml)
            va = tr_all[len(tr_all) // 2]
            rows += _fold(data, [b for b in tr_all if b != va], va, te, "xjtu")
            print(f"  condition {cond} done ({time.time()-t0:.0f} s)", flush=True)
        allrows += rows
        tables["xjtu/all"] = summarise(rows)
        show("[xjtu] leave-one-condition-out, all prediction points",
             tables["xjtu/all"])

    save(tables, allrows, which)
    #  RES_TAG has to appear here. Without it a sweep arm reports that it wrote
    #  results_both.json -- the file the papers are built from -- when it wrote
    #  results_both_w96.json, which reads like the main results have just been
    #  overwritten by a backbone that is not the reported one.
    print(f"\nwrote results/regression/results_{which}{RES_TAG}.json")
