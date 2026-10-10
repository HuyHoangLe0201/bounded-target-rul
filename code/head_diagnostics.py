"""
Shape diagnostics for the fitted predictive heads.

The head comparison says which family predicts better; this says why. Each row of
the results file carries 101 quantiles of the predictive distribution, from which
the realised spread and right-tail heaviness of each head can be read directly
without refitting anything.
"""
from __future__ import annotations

import json
import os

import numpy as np


# BS and IG quantiles, so that this module does not need the wider library
def _bs_ppf(q, a, b):
    from scipy.stats import norm
    z = norm.ppf(q)
    return b / 4.0 * (a * z + np.sqrt(4.0 + (a * z) ** 2)) ** 2


def _ig_ppf(q, mu, lam):
    from scipy.stats import invgauss
    return invgauss.ppf(q, mu / lam, scale=lam)

HERE = os.path.dirname(os.path.abspath(__file__))
RG = os.path.join(HERE, "..", "results", "regression")
QGRID = np.linspace(0.005, 0.995, 101)


def qidx(p):
    return int(np.argmin(np.abs(QGRID - p)))


def main():
    with open(os.path.join(RG, "results_both.json")) as f:
        rows = json.load(f)["rows"]
    rows = [r for r in rows if r.get("q")]
    shape = {}

    i05, i25, i50, i75, i95 = (qidx(p) for p in (0.05, 0.25, 0.5, 0.75, 0.95))
    for dset in ("pronostia", "xjtu"):
        rs = [r for r in rows if r["dataset"] == dset]
        if not rs:
            continue
        print(f"\n{dset.upper()}")
        print(f"  {'head':11s} {'med':>7s} {'IQR/med':>8s} {'q95/med':>8s} "
              f"{'med/q05':>8s} {'tail ratio':>11s} {'>1 mass':>8s}")
        print("  " + "-" * 66)
        for h in ["BS", "BS-t", "Weibull", "LogNormal", "IGauss", "Gaussian"]:
            s = [r for r in rs if r["head"] == h]
            if not s:
                continue
            Q = np.array([r["q"] for r in s])
            med = Q[:, i50]
            ok = med > 1e-9
            Q, med = Q[ok], med[ok]
            iqr = (Q[:, i75] - Q[:, i25]) / med
            right = Q[:, i95] / med                      # right-tail extent
            left = med / np.maximum(Q[:, i05], 1e-9)     # left-tail extent
            # asymmetry: how much longer the right tail is than the left
            tail = right / left
            # a target in (0,1]: mass the head puts on impossible values
            above1 = (Q > 1.0).mean(axis=1)
            shape.setdefault(dset, {})[h] = {
                "med": float(np.median(med)), "iqr_med": float(np.median(iqr)),
                "q95_med": float(np.median(right)), "med_q05": float(np.median(left)),
                "tail_ratio": float(np.median(tail)), "mass_above_1": float(np.mean(above1))}
            print(f"  {h:11s} {np.median(med):7.3f} {np.median(iqr):8.2f} "
                  f"{np.median(right):8.2f} {np.median(left):8.2f} "
                  f"{np.median(tail):11.2f} {np.mean(above1):8.3f}")
    with open(os.path.join(RG, "head_shape.json"), "w") as f:
        json.dump(shape, f, indent=1)
    print("\n  -> results/regression/head_shape.json")
    print("\n  IQR/med, q95/med and med/q05 are relative spreads; tail ratio > 1 means"
          "\n  the right tail is the longer one. '>1 mass' is the average predictive"
          "\n  probability placed above 1, i.e. on a remaining life longer than the"
          "\n  whole test -- values the target cannot take.")


def matched_tails():
    """BS versus IG at matched median and interquartile range.

    Backs Remark 'The comparison that matters is at matched quantiles'. The
    mixture representation BS = 1/2 IG + 1/2 RIG adds right-tail mass at fixed
    (mu, lambda), but that is not the comparison a fitted model makes: once both
    families are matched on median and IQR, BS is the LIGHTER-tailed of the two
    as dispersion grows.
    """
    from scipy import optimize

    def ig_q(p, mu, lam):
        return float(_ig_ppf(p, mu, lam))

    print("\nBS vs IG at matched median (=1) and IQR")
    print(f"  {'IQR/median':>10} {'BS alpha':>9} {'BS q95':>8} {'IG q95':>8} {'BS/IG':>7}")
    print("  " + "-" * 46)
    rows = []
    for tgt in (0.4, 0.8, 1.2, 1.6, 2.0):
        a = optimize.brentq(
            lambda a: (_bs_ppf(0.75, a, 1.0) - _bs_ppf(0.25, a, 1.0)) - tgt,
            1e-3, 20.0)
        bs95 = float(_bs_ppf(0.95, a, 1.0))
        sol = optimize.fsolve(
            lambda v: [ig_q(0.5, *np.exp(v)) - 1.0,
                       (ig_q(0.75, *np.exp(v)) - ig_q(0.25, *np.exp(v))) - tgt],
            [0.1, 0.5])
        ig95 = ig_q(0.95, *np.exp(sol))
        print(f"  {tgt:10.1f} {a:9.3f} {bs95:8.2f} {ig95:8.2f} {bs95/ig95:7.2f}")
        rows.append({"iqr_over_median": tgt, "bs_alpha": a, "bs_q95": bs95,
                     "ig_q95": ig95, "ratio": bs95 / ig95})
    with open(os.path.join(RG, "matched_tails.json"), "w") as f:
        json.dump(rows, f, indent=1)


if __name__ == "__main__":
    main()
    matched_tails()
