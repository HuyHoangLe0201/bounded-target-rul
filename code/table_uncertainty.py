"""LaTeX table for the bearing-level bootstrap of study_uncertainty.py.

Kept in its own module because make_tables.py builds its tables with nested
f-strings, and this one carries enough LaTeX backslashes that mixing the two
styles is how escaping errors get introduced.
"""
from __future__ import annotations

import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(HERE, "..", "results", "regression", "uncertainty.json")

ORDER = ["BS", "BS-t", "Weibull", "LogNormal", "IGauss", "Gaussian", "Point(MAE)"]
NAME = {"BS": "BS", "BS-t": "BS-$t$", "Weibull": "Weibull",
        "LogNormal": "lognormal", "IGauss": "inverse Gaussian",
        "Gaussian": "Gaussian", "Point(MAE)": "point head (MAE loss)"}

CAPTION = (
    "Paired cluster bootstrap over bearings: P(first), probability of the "
    "lowest MAE; brackets, 95\\,\\% interval on the gap to the leader; "
    "$\\dagger$, not separated from it."
)


def build():
    with open(SRC) as f:
        d = json.load(f)
    P = {r["head"]: r for r in d["results"]["pronostia/MAE"]["rows"]}
    X = {r["head"]: r for r in d["results"]["xjtu/MAE"]["rows"]}

    out = [
        r"\begin{table}[htbp]",
        r"\centering\small",
        r"\caption{" + CAPTION.format(nb=d["n_boot"]) + "}",
        r"\label{tab:uncertainty}",
        # no adjustbox here: make_tables.write() wraps every tabular itself
        r"\begin{tabular}{@{}l rcr rcr@{}}",
        r"\toprule",
        r"& \multicolumn{3}{c}{PRONOSTIA (11 bearings)}"
        r" & \multicolumn{3}{c}{XJTU-SY (15 bearings)}\\",
        r"\cmidrule(lr){2-4}\cmidrule(lr){5-7}",
        r"head & MAE & diff.\ from leader & P(first)"
        r" & MAE & diff.\ from leader & P(first)\\",
        r"\midrule",
    ]
    for h in ORDER:
        cells = []
        for tab in (P, X):
            r = tab.get(h)
            if r is None:
                cells += ["---", "---", "---"]
                continue
            tie = r["diff_lo"] <= 0.0 <= r["diff_hi"]
            lead = abs(r["diff_lo"]) < 1e-12 and abs(r["diff_hi"]) < 1e-12
            span = "---" if lead else \
                f"$[{r['diff_lo']:+.3f},\\,{r['diff_hi']:+.3f}]$"
            pf = f"{r['p_first']:.3f}" + (r"$^\dagger$" if tie and not lead else "")
            cells += [f"{r['mean']:.3f}", span, pf]
        out.append(NAME[h] + " & " + " & ".join(cells) + r"\\")
    out += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    return "\n".join(out) + "\n"


if __name__ == "__main__":
    print(build())
