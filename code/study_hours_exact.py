"""Integral CRPS in hours from the exact parametric laws.

study_hours_crps.py integrates the CRPS in hours from the 101 stored quantiles of
each predictive law, with an assumed tail beyond q_0.995. The weight (1-y)^-2
makes the region near y = 1 decisive, and the stored draws are not exact there:
a truncated head samples by ppf(q F(1)) with q clamped to [1e-6, 1-1e-6], so a
member with F(1) < 1e-6 returns draws of exactly one, an atom at infinite RUL
that its law does not have.

This script refits every head with the identical folds, seeds and settings of
study_regression.py (as study_weibull_members.py does), keeps each member's
parameters at both sets of prediction points, and integrates

    CRPS_h = t * int_0^inf {F_Y(y(s)) - 1(s >= s*)}^2 e^s ds,   y = 1 - e^-s,

from the exact mixture CDF in double precision, with log-CDFs and log-survival
functions so that 1 - F stays accurate near y = 1. Restricted heads condition
the ensemble mixture on (0,1), as study_review_p2.truncate does; truncated heads
condition each member, as rulreg.TruncatedHead does. A Beta member with b <= 1/2
has 1 - F ~ (1-y)^b, so its integral diverges and its CRPS in hours is infinite.

  python study_hours_exact.py fit     # refit, writes hours_exact_params.json
  python study_hours_exact.py score   # integrate and bootstrap, writes hours_exact.json
"""
import json
import os
import sys
import time

import numpy as np
from scipy import special as sp

RES = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results", "regression")
PARAMS = os.path.join(RES, "hours_exact_params.json")
DT = {"pronostia": 10.0, "xjtu": 60.0}
FRACTION_HEADS = ["BS", "BS-t", "Weibull", "LogNormal", "IGauss", "Gaussian",
                  "LogitNormal", "Beta", "BS-trunc", "IGauss-trunc"]
RATIO_HEADS = ["BS", "BS-t", "Weibull", "LogNormal", "IGauss", "Gaussian", "Gaussian-trunc0"]
CONV = ["BS", "BS-t", "Weibull", "LogNormal", "IGauss", "Gaussian"]
NB, SEED = 10000, 9


# ── stage 1: refit and keep the member parameters ─────────────────────────
def eval_points(rec, dataset, bearing, protocol):
    import rul_dataset as D
    import study_regression as SR
    n = rec["n"]
    k0 = max(20, rec.get("nh", 0))
    if protocol == "grid":
        return list(range(k0, n - 1, SR.STRIDE[dataset]))
    pts = {int(f * (n - 1)) for f in SR.EVAL_FRACS}
    if dataset == "pronostia" and bearing in D.PHM12_TRUNC_N:
        pts.add(D.PHM12_TRUNC_N[bearing] - 1)
    return sorted(k for k in pts if k0 <= k < n - 1)


def fit_fold(data, tr_b, va_b, te_b, dataset):
    import torch
    import rul_dataset as D
    import rulreg as R
    import study_regression as SR
    f32 = lambda a: a.astype(np.float32)
    rows = []
    for target, heads in (("fraction", FRACTION_HEADS), ("ratio", RATIO_HEADS)):
        Xtr, ytr, wtr, _ = D.stack(data, tr_b, target=target)
        Xva, yva, _, _ = D.stack(data, [va_b], target=target)
        Xtr_s, Xva_s, (mu, sd) = D.standardise(Xtr, Xva)
        for h in heads:
            head = SR.make_head(h)
            models = SR._fit_ensemble(head, f32(Xtr_s), f32(ytr), f32(wtr), f32(Xva_s), f32(yva))
            for b in te_b:
                rec = data[b]
                T = max(rec["t"][-1], 1e-9)
                for protocol in ("main", "grid"):
                    pts = eval_points(rec, dataset, b, protocol)
                    if not pts:
                        continue
                    Xe = f32((rec["X"][pts] - mu) / sd)
                    with torch.no_grad():
                        P = [[x.double().numpy() for x in head.params(m(torch.tensor(Xe)))] for m in models]
                    med = None
                    if protocol == "main":    # reproduction check against the stored run
                        S = R.predict(head, models, Xe)["samples"]
                        med = np.quantile(SR._to_fraction(S) if target == "ratio" else S, 0.5, axis=1)
                    for j, k in enumerate(pts):
                        y = float(rec["rul"][k] / T)
                        if y <= 0:
                            continue
                        rows.append({"protocol": protocol, "dataset": dataset, "bearing": b, "k": int(k),
                                     "head": h + ("@w" if target == "ratio" else ""), "true": y,
                                     "p1": [float(p[0][j]) for p in P], "p2": [float(p[1][j]) for p in P],
                                     **({"median": float(med[j])} if med is not None else {})})
            print(f"  {dataset} {target} {h} fitted", flush=True)
    return rows


def fit():
    import rul_dataset as D
    rows, t0 = [], time.time()
    raw = D.load("pronostia", t_scale=1.0)
    ml = float(np.median([raw[b]["n"] for b in D.PRONOSTIA_LEARN])) * D.DT["pronostia"]
    data = D.load("pronostia", t_scale=ml)
    rows += fit_fold(data, [b for b in D.PRONOSTIA_LEARN if b != "Bearing2_2"],
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
        rows += fit_fold(data, [b for b in tr_all if b != va], va, te, "xjtu")
        print(f"xjtu condition {cond} done ({time.time()-t0:.0f} s)", flush=True)
    json.dump(rows, open(PARAMS, "w"))


# ── stage 2: exact integral in double precision ───────────────────────────
def _xi(u):
    return np.sqrt(u) - 1.0 / np.sqrt(u)


def _logdiff(la, lb):
    """log(exp(la) - exp(lb)) for la >= lb."""
    with np.errstate(divide="ignore", invalid="ignore"):
        d = np.where(np.isneginf(lb), -np.inf, np.minimum(lb - la, 0.0))
        return la + np.log(-np.expm1(d))


def logcdf_logsf(fam, a, b, x, omx=None):
    """log F and log(1 - F) of one member at x (the head's own variable)."""
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        if fam in ("BS", "BS-t"):
            z = _xi(np.maximum(x, 1e-300) / b) / a
            if fam == "BS":
                lF, lS = sp.log_ndtr(z), sp.log_ndtr(-z)
            else:
                lF, lS = np.log(sp.stdtr(6.0, z)), np.log(sp.stdtr(6.0, -z))
        elif fam == "Weibull":
            r = (np.maximum(x, 0.0) / b) ** a
            lF, lS = np.log(-np.expm1(-r)), -r
        elif fam == "LogNormal":
            z = (np.log(np.maximum(x, 1e-300)) - a) / b
            lF, lS = sp.log_ndtr(z), sp.log_ndtr(-z)
        elif fam == "IGauss":
            xx = np.maximum(x, 1e-300)
            r = np.sqrt(b / xx)
            u, v, c = r * (xx / a - 1.0), -r * (xx / a + 1.0), 2.0 * b / a
            l1, l2 = sp.log_ndtr(u), c + sp.log_ndtr(v)
            lF = np.logaddexp(l1, l2)
            lS = _logdiff(sp.log_ndtr(-u), l2)
        elif fam == "Gaussian":
            z = (x - a) / b
            lF, lS = sp.log_ndtr(z), sp.log_ndtr(-z)
        elif fam == "LogitNormal":
            z = (np.log(x) - np.log(omx) - a) / b
            lF, lS = sp.log_ndtr(z), sp.log_ndtr(-z)
        elif fam == "Beta":
            lF, lS = np.log(sp.betainc(a, b, x)), np.log(sp.betainc(b, a, omx))
        else:
            raise ValueError(fam)
        if fam in ("BS", "BS-t", "Weibull", "LogNormal", "IGauss"):
            lF = np.where(x <= 0, -np.inf, lF)
            lS = np.where(x <= 0, 0.0, lS)
    return lF, lS


def _logdiff_cdf(lFa, lSa, lFb, lSb):
    """log(F(a) - F(b)) for a >= b, from whichever side keeps precision."""
    use_f = lFa <= np.log(0.5)
    return np.where(use_f, _logdiff(lFa, lFb), _logdiff(lSb, lSa))


def law_on_grid(head, p1, p2, y, omy):
    """(F, 1-F) of the predictive mixture on the grid y (1-y given exactly)."""
    base = head.split("@")[0]
    fam = base.replace("-trunc0", "").replace("-trunc", "")
    M = len(p1)
    lF, lS = [], []
    if head.endswith("@w"):                      # law on w = y/(1-y) = expm1(s)
        w = y / omy
        for m in range(M):
            f, s_ = logcdf_logsf(fam, p1[m], p2[m], w)
            if base == "Gaussian-trunc0":        # conditioned on w > 0
                f0, s0 = logcdf_logsf(fam, p1[m], p2[m], np.zeros(1))
                f, s_ = _logdiff_cdf(f, s_, f0, s0) - s0, s_ - s0
            lF.append(f)
            lS.append(s_)
        F = np.mean(np.exp(lF), 0)
        S = np.mean(np.exp(lS), 0)
        return F, S
    if base in ("LogitNormal", "Beta"):
        for m in range(M):
            f, s_ = logcdf_logsf(fam, p1[m], p2[m], y, omy)
            lF.append(f)
            lS.append(s_)
        return np.mean(np.exp(lF), 0), np.mean(np.exp(lS), 0)
    one, zero = np.ones(1), np.zeros(1)
    lg, ld, lD = [], [], []
    for m in range(M):
        f, s_ = logcdf_logsf(fam, p1[m], p2[m], y)
        f1, s1 = logcdf_logsf(fam, p1[m], p2[m], one)
        f0, s0 = logcdf_logsf(fam, p1[m], p2[m], zero)
        lg.append(_logdiff_cdf(f, s_, f0, s0))   # F(y) - F(0)
        ld.append(_logdiff_cdf(f1, s1, f, s_))   # F(1) - F(y)
        lD.append(_logdiff_cdf(f1, s1, f0, s0))  # F(1) - F(0)
    lg, ld, lD = np.array(lg), np.array(ld), np.array(lD)
    if base.endswith("-trunc"):                  # each member conditioned on (0,1)
        return np.mean(np.exp(lg - lD), 0), np.mean(np.exp(ld - lD), 0)
    norm = sp.logsumexp(lD)                      # the mixture conditioned on (0,1)
    return np.exp(sp.logsumexp(lg, axis=0) - norm), np.exp(sp.logsumexp(ld, axis=0) - norm)


def crps_hours_exact(head, p1, p2, y_true, t_h):
    s_true = -np.log1p(-y_true)
    yb = np.linspace(0.0, 0.95, 4001)
    s = np.unique(np.concatenate([
        -np.log1p(-yb), np.arange(3.0, 36.0, 0.004),
        -np.log1p(-np.clip(np.linspace(y_true - 0.01, y_true + 0.01, 1001), 0.0, 0.999)),
        [s_true]]))
    y, omy = -np.expm1(-s), np.exp(-s)
    F, S = law_on_grid(head, p1, p2, y, omy)
    lo, hi = s <= s_true, s >= s_true
    with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
        g1 = F[lo] ** 2 * np.exp(s[lo])
        g2 = np.exp(2.0 * np.log(np.maximum(S[hi], 0.0)) + s[hi])
    val = np.trapezoid(g1, s[lo]) + np.trapezoid(g2, s[hi])
    if head.split("@")[0] == "Beta" and min(p2) <= 0.5:
        val = np.inf                             # 1 - F ~ (1-y)^b: the integral diverges
    return t_h * val


def _check_uniform():
    """Beta(1,1) at y* = 1/2 and t = 1 has CRPS_h = 2 - 2 log 2 = 0.613706."""
    v = crps_hours_exact("Beta", [1.0], [1.0], 0.5, 1.0)
    assert abs(v - 0.6137056) < 1e-4, v
    return v


def score():
    _check_uniform()
    rows = json.load(open(PARAMS))
    out = {"reproduction": {}}
    # the refit must reproduce the stored predictions
    stored = {}
    for f, suf in (("results_both_causal.json", ""), ("results_both_ratio.json", "@w"),
                   ("results_both_ratio_g0.json", "@w")):
        for r in json.load(open(os.path.join(RES, f)))["rows"]:
            if r["q"] is not None:
                stored[(r["dataset"], r["bearing"], r["k"], r["head"] + suf)] = r["median"]
    dev = {}
    for r in rows:
        if r["protocol"] == "main" and (r["dataset"], r["bearing"], r["k"], r["head"]) in stored:
            dev.setdefault(r["head"], []).append(abs(r["median"] - stored[(r["dataset"], r["bearing"], r["k"], r["head"])]))
    out["reproduction"] = {h: float(max(v)) for h, v in dev.items()}
    print("max |median - stored| per head:", out["reproduction"], flush=True)

    rng = np.random.default_rng(SEED)
    for protocol in ("main", "grid"):
        cc, n_inf = {}, {}
        t0 = time.time()
        for r in rows:
            if r["protocol"] != protocol:
                continue
            key = r["head"] if r["head"].endswith("@w") or r["head"] not in CONV else r["head"] + "@restricted"
            v = crps_hours_exact(r["head"], np.array(r["p1"]), np.array(r["p2"]), r["true"],
                                 r["k"] * DT[r["dataset"]] / 3600.0)
            n_inf[key] = n_inf.get(key, 0) + int(not np.isfinite(v))
            cc.setdefault((r["dataset"], r["bearing"]), {}).setdefault(key, []).append(float(v))
        print(f"{protocol}: integrated in {time.time()-t0:.0f} s", flush=True)
        arch = {ds: sorted(b for (d, b) in cc if d == ds) for ds in DT}
        keys = sorted({k for d in cc.values() for k in d})
        res = {"heads": {f"{ds}/{k}": float(np.mean([x for b in arch[ds] for x in cc[(ds, b)][k]]))
                         for ds in DT for k in keys},
               "pooled": {k: float(np.mean([np.mean(cc[kk][k]) for kk in cc])) for k in keys},
               "n_infinite": n_inf, "pairs": {}}

        def margins(sample, a, b):
            bm = {ds: [np.mean(cc[(ds, x)][a]) - np.mean(cc[(ds, x)][b]) for x in sample[ds]] for ds in sample}
            return {"pronostia": float(np.mean(bm["pronostia"])), "xjtu": float(np.mean(bm["xjtu"])),
                    "pooled": float(np.mean(bm["pronostia"] + bm["xjtu"]))}

        fams = {"vs_w": [(a, b + "@w") for a in ("BS-trunc", "IGauss-trunc") for b in RATIO_HEADS],
                "vs_restricted": [(a, b + "@restricted") for a in ("BS-trunc", "IGauss-trunc") for b in CONV]}
        for fam, pairs in fams.items():
            for a, b in pairs:
                pt = margins(arch, a, b)
                bs = {k: [] for k in pt}
                for _ in range(NB):
                    smp = {ds: [arch[ds][i] for i in rng.choice(len(arch[ds]), len(arch[ds]))] for ds in arch}
                    for k, x in margins(smp, a, b).items():
                        bs[k].append(x)
                entry = {"family": fam}
                for k in pt:
                    arr = np.array(bs[k])
                    entry[k] = {"margin": pt[k], "ci95": [float(np.quantile(arr, 0.025)), float(np.quantile(arr, 0.975))],
                                "p_boot": float(min(1.0, 2 * min(np.mean(arr >= 0), np.mean(arr <= 0))))}
                res["pairs"][f"{a} - {b}"] = entry
            ks = [f"{a} - {b}" for a, b in pairs]
            run = 0.0
            for i, k in enumerate(sorted(ks, key=lambda k: res["pairs"][k]["pooled"]["p_boot"])):
                run = max(run, min(1.0, (len(ks) - i) * res["pairs"][k]["pooled"]["p_boot"]))
                res["pairs"][k]["pooled"]["p_holm"] = run
        out[protocol] = res
        print(f"##### {protocol}")
        for k in sorted(res["pooled"], key=res["pooled"].get):
            print(f"   {k:24s} pooled {res['pooled'][k]:9.3f} h   P {res['heads']['pronostia/' + k]:9.3f}"
                  f"  X {res['heads']['xjtu/' + k]:9.3f}   infinite at {n_inf.get(k, 0)}")
        for k, x in res["pairs"].items():
            p = x["pooled"]
            print(f"   {k:34s} {p['margin']:+.3f} [{p['ci95'][0]:+.3f},{p['ci95'][1]:+.3f}] holm={p['p_holm']:.3f}")
    json.dump(out, open(os.path.join(RES, "hours_exact.json"), "w"), indent=1)


if __name__ == "__main__":
    {"fit": fit, "score": score}[sys.argv[1] if len(sys.argv) > 1 else "score"]()
