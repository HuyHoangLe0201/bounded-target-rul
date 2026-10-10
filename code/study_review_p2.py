"""Re-analysis of the stored paper-2 predictions for the robustness analyses of the paper.

Uses results/regression/results_both.json only (101 stored predictive quantiles
per prediction); nothing is retrained.  Reports, per benchmark and head:

  1. points     the prediction points of every held-out bearing (life fractions)
  2. calib_ci   bearing-level bootstrap intervals for cov95, MPIW and PIT-KS
  3. trunc      the predictive law truncated to (0,1] after the fact: does the
                ordering survive once the mass above one is removed?
  4. pooling    the IG - Weibull CRPS margin under four ways of combining the
                two benchmarks, bearings resampled within each archive
  5. b14        the official subset rescored with the published Bearing1_4 RUL
  6. er_opt     the point summary that minimises the expected relative error
                (a 1/R-weighted median), against the median used in the paper

CRPS and coverage are recomputed from the quantile grid; the script first checks
them against the stored tables.  Writes results/regression/review_p2.json.
"""
import json
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "..", "results", "regression")
TAU = np.linspace(0.005, 0.995, 101)
HEADS = ["BS", "BS-t", "Weibull", "LogNormal", "IGauss", "Gaussian"]
NB, SEED = 10000, 0
NGRID, NGRID_CHECK = 10000, 2000  # PHM-optimal search grid and its resolution check
TAU_DENSE = np.linspace(0.0005, 0.9995, 1000)
#  Bearing1_4: archive end implies 2890 s at the official truncation, the
#  challenge published 339 s (see the Ground truth paragraph of the paper)
B14_ARCHIVE, B14_PUBLISHED, DT = 2890.0, 339.0, 10.0


def phm(er):
    er = np.asarray(er, float)
    return np.where(er <= 0, np.exp(-np.log(0.5) * er / 5.0), np.exp(np.log(0.5) * er / 20.0))


def phm_argmax(q, ng):
    """Grid maximiser of the expected PHM score; returns (index, grid)."""
    qq = np.interp(TAU_DENSE, TAU, q).astype(np.float32)
    grid = np.linspace(0.2 * q[0], q[-1], ng).astype(np.float32)
    c1, c2 = np.float32(np.log(2) / 5), np.float32(-np.log(2) / 20)
    best, ib = -1.0, 0
    for c in np.array_split(np.arange(ng), max(1, ng // 1000)):
        er = 100 * (qq[None, :] - grid[c][:, None]) / qq[None, :]
        e = np.exp(np.where(er <= 0, c1 * er, c2 * er)).mean(axis=1)
        i = int(np.argmax(e))
        if e[i] > best:
            best, ib = float(e[i]), int(c[i])
    return ib, grid


def crps_q(q, y):
    """CRPS = 2 * integral of the pinball loss over tau, on the stored grid."""
    u = y - q
    return 2.0 * np.mean(np.maximum(TAU * u, (TAU - 1.0) * u))


def pit_q(q, y):
    return float(np.interp(y, q, TAU, left=TAU[0] / 2, right=1 - TAU[0] / 2))


def ks_u(p):
    p = np.sort(np.asarray(p))
    n = len(p)
    i = np.arange(1, n + 1)
    return float(max(np.max(i / n - p), np.max(p - (i - 1) / n)))


def interval(q, a):
    return np.interp(a, TAU, q), np.interp(1 - a, TAU, q)


def truncate(q):
    """Quantiles of the law conditioned on the target's support 0 < Y <= 1.

    Returns (quantiles, retained mass).  For the five positive families only the
    upper cut matters; the Gaussian also loses its mass below zero."""
    F0 = float(np.interp(0.0, q, TAU, left=0.0, right=1.0))
    F1 = float(np.interp(1.0, q, TAU, left=0.0, right=1.0))
    if F0 <= TAU[0] and F1 >= TAU[-1]:
        return q.copy(), 1.0
    F1 = max(F1, F0 + 1e-3)
    return np.interp(F0 + TAU * (F1 - F0), TAU, q, left=q[0], right=q[-1]), F1 - F0


def wmedian(q, w):
    o = np.argsort(q)
    c = np.cumsum(w[o]) / np.sum(w)
    return float(q[o][np.searchsorted(c, 0.5)])


def load():
    d = json.load(open(os.path.join(RES, "results_both.json")))
    return d["tables"], [r for r in d["rows"] if r["head"] in HEADS]


def per_row(r, trunc=False):
    q = np.asarray(r["q"], float)
    if trunc:
        q, _ = truncate(q)
    y = r["true"]
    lo, hi = interval(q, 0.025)
    med = float(np.interp(0.5, TAU, q))
    return {"crps": crps_q(q, y), "cov": float(lo <= y <= hi), "w": hi - lo,
            "pit": pit_q(q, y), "ae": abs(med - y), "med": med}


def boot_ci(vals_by_bearing, stat, rng):
    bs = list(vals_by_bearing)
    out = []
    for _ in range(NB):
        pick = rng.choice(len(bs), len(bs))
        out.append(stat([v for i in pick for v in vals_by_bearing[bs[i]]]))
    return [float(np.quantile(out, 0.025)), float(np.quantile(out, 0.975))]


def main():
    tables, rows = load()
    rng = np.random.default_rng(SEED)
    out = {}
    dsets = {"pronostia": "pronostia/all", "xjtu": "xjtu/all"}

    # 1. prediction points per bearing
    pts = {}
    for r in rows:
        if r["head"] == "IGauss":
            pts.setdefault(r["dataset"], {}).setdefault(r["bearing"], []).append(
                (round(r["life_frac"], 3), bool(r["official"])))
    out["points"] = {d: {b: sorted(v) for b, v in sorted(bb.items())} for d, bb in pts.items()}

    # sanity: recomputed metrics against the stored tables
    chk = {}
    for d, key in dsets.items():
        for h in HEADS:
            R = [per_row(r) for r in rows if r["dataset"] == d and r["head"] == h]
            chk[f"{d}/{h}"] = {
                "CRPS": [float(np.mean([x["crps"] for x in R])), tables[key][h]["CRPS"]],
                "cov95": [float(np.mean([x["cov"] for x in R])), tables[key][h]["cov95"]],
                "PIT_KS": [ks_u([x["pit"] for x in R]), tables[key][h]["PIT_KS"]]}
    out["check"] = chk

    # 2. bearing-level intervals for the calibration summaries
    ci = {}
    for d in dsets:
        for h in HEADS:
            byb = {}
            for r in rows:
                if r["dataset"] == d and r["head"] == h:
                    byb.setdefault(r["bearing"], []).append(per_row(r))
            ci[f"{d}/{h}"] = {
                "cov95": boot_ci(byb, lambda L: np.mean([x["cov"] for x in L]), rng),
                "MPIW": boot_ci(byb, lambda L: np.mean([x["w"] for x in L]), rng),
                "PIT_KS": boot_ci(byb, lambda L: ks_u([x["pit"] for x in L]), rng)}
    out["calib_ci"] = ci

    # 3. truncation to (0,1]
    tr = {}
    for d in dsets:
        for h in HEADS:
            R0 = [per_row(r) for r in rows if r["dataset"] == d and r["head"] == h]
            R1 = [per_row(r, True) for r in rows if r["dataset"] == d and r["head"] == h]
            F1 = [truncate(np.asarray(r["q"], float))[1] for r in rows
                  if r["dataset"] == d and r["head"] == h]
            tr[f"{d}/{h}"] = {
                "mass_above_1": float(1 - np.mean(F1)),
                "CRPS": [float(np.mean([x["crps"] for x in R0])), float(np.mean([x["crps"] for x in R1]))],
                "MAE": [float(np.mean([x["ae"] for x in R0])), float(np.mean([x["ae"] for x in R1]))],
                "cov95": [float(np.mean([x["cov"] for x in R0])), float(np.mean([x["cov"] for x in R1]))],
                "PIT_KS": [ks_u([x["pit"] for x in R0]), ks_u([x["pit"] for x in R1])]}
    out["trunc"] = tr

    # 4. IG - Weibull CRPS margin under four pooling rules
    crps = {}
    for r in rows:
        crps.setdefault((r["dataset"], r["bearing"]), {}).setdefault(r["head"], []).append(
            crps_q(np.asarray(r["q"], float), r["true"]))
    arch = {d: sorted(b for (dd, b) in crps if dd == d) for d in dsets}

    def rule_stats(sample):
        """sample: {dataset: [bearing,...]} -> margins (IG - Weibull) by rule"""
        res = {}
        bmean = {d: [{h: np.mean(crps[(d, b)][h]) for h in HEADS} for b in sample[d]] for d in dsets}
        pts_ = {d: [{h: crps[(d, b)][h] for h in HEADS} for b in sample[d]] for d in dsets}
        # (a) every bearing weighted equally across the 26
        allb = bmean["pronostia"] + bmean["xjtu"]
        res["bearing_equal"] = np.mean([x["IGauss"] - x["Weibull"] for x in allb])
        # (b) each benchmark weighted equally
        res["benchmark_equal"] = 0.5 * sum(np.mean([x["IGauss"] - x["Weibull"] for x in bmean[d]])
                                           for d in dsets)
        # (c) every prediction point weighted equally
        diffs = [a - b for d in dsets for x in pts_[d] for a, b in zip(x["IGauss"], x["Weibull"])]
        res["point_equal"] = np.mean(diffs)
        # (d) standardised by each benchmark's spread of head means (paper's rule)
        s = 0.0
        for d in dsets:
            hm = {h: np.mean([x[h] for x in bmean[d]]) for h in HEADS}
            sd = np.std(list(hm.values()), ddof=1)
            s += (hm["IGauss"] - hm["Weibull"]) / max(sd, 1e-12)
        res["standardised"] = s / 2
        return res

    point = rule_stats(arch)
    boots = {k: [] for k in point}
    for _ in range(NB):
        smp = {d: [arch[d][i] for i in rng.choice(len(arch[d]), len(arch[d]))] for d in dsets}
        for k, v in rule_stats(smp).items():
            boots[k].append(v)
    out["pooling"] = {k: {"margin": float(point[k]),
                          "ci95": [float(np.quantile(boots[k], 0.025)), float(np.quantile(boots[k], 0.975))],
                          "p_ig_better": float(np.mean(np.array(boots[k]) < 0))}
                      for k in point}

    # 5. official subset with the published Bearing1_4 RUL
    b14 = {}
    off = [r for r in d_rows(rows) if r["official"]] if False else [r for r in rows if r["official"]]
    for which in ("archive", "published"):
        tab = {}
        for h in HEADS:
            ae, er, cr, cv = [], [], [], []
            for r in off:
                if r["head"] != h:
                    continue
                y = r["true"]
                if r["bearing"] == "Bearing1_4" and which == "published":
                    t_tr = B14_ARCHIVE / y - B14_ARCHIVE      # truncation time
                    y = B14_PUBLISHED / (t_tr + B14_PUBLISHED)
                med = r["median"]
                q = np.asarray(r["q"], float)
                lo, hi = interval(q, 0.025)
                ae.append(abs(med - y))
                er.append(100 * (y - med) / y)
                cr.append(crps_q(q, y))
                cv.append(float(lo <= y <= hi))
            tab[h] = {"MAE": float(np.mean(ae)), "mean_absEr": float(np.mean(np.abs(er))),
                      "PHM": float(np.mean(phm(er))), "CRPS": float(np.mean(cr)),
                      "cov95": float(np.mean(cv))}
        for m in ("MAE", "mean_absEr", "PHM", "CRPS"):
            order = sorted(HEADS, key=lambda h: tab[h][m] * (-1 if m == "PHM" else 1))
            tab[f"order_{m}"] = order
        b14[which] = tab
    out["b14"] = b14

    # 6. loss-matched point summaries: the 1/R-weighted median for |Er| and the
    #    maximiser of the expected PHM score.  Both need Er(Y, a) = 100(Y-a)/Y
    #    to be defined over the predictive law, i.e. Y > 0; for the Gaussian,
    #    whose law gives mass to Y <= 0, neither summary is defined (None).
    eo = {}
    for d in dsets:
        for h in HEADS:
            er_med, er_opt, s_med, s_opt = [], [], [], []
            s_fine, edge, a_opt, a_dev = [], [], [], []
            for r in rows:
                if r["dataset"] != d or r["head"] != h:
                    continue
                q = np.asarray(r["q"], float)
                y = r["true"]
                med = float(np.interp(0.5, TAU, q))
                er_med.append(abs(100 * (y - med) / y))
                s_med.append(float(phm(100 * (y - med) / y)))
                if h == "Gaussian":
                    continue
                #  E|R-a|/R is finite only if E[1/R] is: always for BS, BS-t,
                #  lognormal and IG, but for a Weibull only when the shape k > 1,
                #  which fails for part of the predictions (study_weibull_shape.py)
                if h != "Weibull":
                    opt = wmedian(q, 1.0 / np.maximum(q, 1e-6))
                    er_opt.append(abs(100 * (y - opt) / y))
                #  expected score under the predictive law, its quantile function
                #  interpolated to 1000 levels (on the 101 stored atoms alone the
                #  expected score has spurious near-equal peaks), maximised on a
                #  grid over [0.2 q_0.005, q_0.995]; the coarser grid checks
                #  that the result has converged in the grid resolution
                for ng, store in ((NGRID, s_opt), (NGRID_CHECK, s_fine)):
                    i, grid = phm_argmax(q, ng)
                    store.append(float(phm(100 * (y - grid[i]) / y)))
                    if ng == NGRID:
                        edge.append(i in (0, ng - 1))
                        a_opt.append(float(grid[i]))
                    else:
                        a_dev.append(abs(float(grid[i]) - a_opt[-1]))
            nan = None
            eo[f"{d}/{h}"] = {"absEr_median": float(np.mean(er_med)),
                              "absEr_weighted_median": float(np.mean(er_opt)) if er_opt else nan,
                              "PHM_median": float(np.mean(s_med)),
                              "PHM_optimal": float(np.mean(s_opt)) if s_opt else nan,
                              "PHM_optimal_fine": float(np.mean(s_fine)) if s_fine else nan,
                              "argmax_max_shift": float(max(a_dev)) if a_dev else nan,
                              "argmax_on_edge": int(sum(edge))}
    out["er_opt"] = eo

    json.dump(out, open(os.path.join(RES, "review_p2.json"), "w"), indent=1)
    print(json.dumps({k: out[k] for k in ("check",)}, indent=0)[:1500])
    for k in ("pooling",):
        print(k, json.dumps(out[k], indent=1))


def d_rows(rows):
    return rows


if __name__ == "__main__":
    main()
