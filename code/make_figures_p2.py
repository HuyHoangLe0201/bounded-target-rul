"""Calibration and PIT figures of paper 2."""
import json
import os

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import binom

import figstyle

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "..", "results")
FIG = os.path.join(RES, "figures")
os.makedirs(FIG, exist_ok=True)

figstyle.apply()
C, HEADC, HEADL = figstyle.C, figstyle.HEADC, figstyle.HEADL
QLEV = np.linspace(0.005, 0.995, 101)


def save(fig, name):
    figstyle.save(fig, name, FIG)
    print(f"  -> results/figures/{name}  "
          f"({figstyle.width_of(name):.2f} in wide, as printed)")


def jload(*p):
    with open(os.path.join(RES, *p)) as f:
        return json.load(f)


def _rows_by_head(rows, dataset):
    out = {}
    for r in rows:
        if r["dataset"] != dataset or r.get("q") is None:
            continue
        out.setdefault(r["head"], []).append(r)
    return out


def _pit(rows):
    """Probability integral transform from the stored 101-quantile grid."""
    return np.array([float(np.interp(r["true"], r["q"], QLEV)) for r in rows])


def fig_calibration():
    rows = jload("regression", "results_both.json")["rows"]
    fig, ax = plt.subplots(1, 2, figsize=figstyle.size("fig_calibration", 0.44))
    for a, (dset, title) in zip(ax, [("pronostia", "(a) PRONOSTIA"),
                                     ("xjtu", "(b) XJTU-SY")]):
        by = _rows_by_head(rows, dset)
        a.plot([0, 1], [0, 1], "--", color="k", lw=0.8, label="perfect")
        for h in HEADC:
            if h not in by:
                continue
            q = np.array([r["q"] for r in by[h]])
            y = np.array([r["true"] for r in by[h]])[:, None]
            a.plot(QLEV, (y <= q).mean(0), color=HEADC[h], label=HEADL[h])
        a.set_xlabel("nominal level")
        a.set_title(title)
        a.set_xlim(0, 1); a.set_ylim(0, 1)
    ax[0].set_ylabel("empirical coverage")
    ax[1].legend(loc="upper left", ncol=1, framealpha=0.9)
    save(fig, "fig_calibration")


def fig_pit():
    d = jload("regression", "results_both.json")
    by = _rows_by_head(d["rows"], "pronostia")
    # KS is read from the stored table rather than recomputed, so the number
    # printed on a panel is character-for-character the one in Table 1
    ks_tab = {h: v["PIT_KS"] for h, v in d["tables"]["pronostia/all"].items()}
    fig, ax = plt.subplots(2, 3, figsize=figstyle.size("fig_pit", 0.50),
                           sharex=True, sharey=True,
                           gridspec_kw={"hspace": 0.30, "wspace": 0.10})
    for a, h in zip(ax.ravel(), HEADC):
        p = _pit(by[h])
        a.hist(p, bins=10, range=(0, 1), color=HEADC[h], alpha=0.85,
               edgecolor="white", linewidth=0.6)
        #  Under a calibrated head each bin holds Binomial(n, 1/10) counts, so
        #  a 95 per cent band says which bars are a real departure and which are
        #  the sampling noise of ten bins over seventy-seven predictions. A flat
        #  reference line alone cannot distinguish the two.
        n = len(p)
        lo, hi = binom.ppf([0.025, 0.975], n, 0.1)
        a.axhspan(lo, hi, color="#000000", alpha=0.07, lw=0, zorder=0)
        a.axhline(n / 10, color="k", ls="--", lw=0.7, zorder=1)
        a.set_title(f"{HEADL[h]}   KS $={ks_tab[h]:.3f}$")
        a.set_xlim(0, 1)
    for a in ax[1]:
        a.set_xlabel("PIT")
    for a in ax[:, 0]:
        a.set_ylabel("count")
    save(fig, "fig_pit")


if __name__ == "__main__":
    fig_calibration()
    fig_pit()
