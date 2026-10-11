"""LaTeX table for the fitted-shape diagnostics of head_diagnostics.py.

This table used to be typed by hand in the manuscript, which contradicted the
paper's own claim that every table is emitted from the stored result files. The
hand-typed values were correct when checked, but nothing would have caught them
drifting after a re-run.
"""
from __future__ import annotations

import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(HERE, "..", "results", "regression", "head_shape.json")

ORDER = ["BS", "BS-t", "Weibull", "LogNormal", "IGauss", "Gaussian"]
NAME = {"BS": "BS", "BS-t": "BS-$t$", "Weibull": "Weibull",
        "LogNormal": "lognormal", "IGauss": "inverse Gaussian",
        "Gaussian": "Gaussian"}

CAPTION = (
    "Shape of the fitted laws, medians over prediction points: interquartile "
    "range (IQR) over the median, and mass above $y=1$."
)


def build():
    with open(SRC) as f:
        d = json.load(f)
    P, X = d.get("pronostia", {}), d.get("xjtu", {})
    out = [
        r"\begin{table}[htbp]",
        r"\centering\small",
        r"\caption{" + CAPTION + "}",
        r"\label{tab:headshape}",
        r"\begin{tabular}{@{}l rr rr@{}}",
        r"\toprule",
        r"& \multicolumn{2}{c}{PRONOSTIA} & \multicolumn{2}{c}{XJTU-SY}\\",
        r"\cmidrule(lr){2-3}\cmidrule(lr){4-5}",
        r"head & IQR/med & $>1$ mass & IQR/med & $>1$ mass\\",
        r"\midrule",
    ]
    for h in ORDER:
        c = []
        for tab in (P, X):
            r = tab.get(h)
            c += ["---", "---"] if r is None else \
                 [f"{r['iqr_med']:.2f}", f"{r['mass_above_1']:.3f}"]
        out.append(NAME[h] + " & " + " & ".join(c) + r"\\")
    out += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    return "\n".join(out) + "\n"


if __name__ == "__main__":
    print(build())
