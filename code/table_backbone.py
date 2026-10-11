"""LaTeX table for the backbone sweep."""
import io
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(HERE, "..", "results", "regression", "backbone.json")

LABEL = {"MAE": "mean absolute error", "CRPS": "CRPS",
         "cov95": "coverage (from nominal)", "PIT_KS": "PIT--KS"}
SPLIT = {"pronostia/all": "PRONOSTIA (all)",
         "pronostia/official": "PRONOSTIA (official)",
         "xjtu/all": "XJTU-SY"}


def build():
    d = json.load(io.open(SRC, encoding="utf-8"))
    rows = ""
    for split in ("pronostia/all", "xjtu/all", "pronostia/official"):
        if split not in d:
            continue
        first = True
        for key in ("MAE", "CRPS", "cov95", "PIT_KS"):
            if key not in d[split]:
                continue
            r = d[split][key]
            name = SPLIT[split] if first else ""
            first = False
            if r["n_distinct_winners"] == 1:
                verdict = f"\\textbf{{{r['winners'][0]}}} in all six"
            else:
                verdict = f"{r['n_distinct_winners']} different: " \
                          + ", ".join(r["winners"])
            rows += (f"{name} & {LABEL[key]} & ${r['tau_min']:+.2f}$ & "
                     f"${r['tau_mean']:+.2f}$ & {verdict}" r"\\" "\n")
        rows += r"\addlinespace" "\n"
    return f"""\\begin{{table}}[htbp]
\\centering\\small
\\caption{{Orderings of the heads across six backbones, compared by Kendall's
$\\tau$; coverage is ranked by distance from nominal.}}
\\label{{tab:backbone}}
\\begin{{adjustbox}}{{max width=\\linewidth}}
\\begin{{tabular}}{{@{{}}l l r r l@{{}}}}
\\toprule
split & criterion & $\\tau_{{\\min}}$ & $\\bar\\tau$ & best head across the six\\\\
\\midrule
{rows}\\bottomrule
\\end{{tabular}}
\\end{{adjustbox}}
\\end{{table}}
"""


if __name__ == "__main__":
    out = os.path.join(HERE, "..", "results", "tables", "tab_backbone.tex")
    io.open(out, "w", encoding="utf-8").write(build())
    print("  -> results/tables/tab_backbone.tex")
