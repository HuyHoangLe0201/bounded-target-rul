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
\\caption{{Is the ordering a property of the families or of the network? The whole
protocol --- features, splits, seeds, schedule, heads --- is re-run on six
backbones ($32$--$16$, $48$--$24$, $96$--$48$, $128$--$64$--$32$, $256$--$128$,
all at a common epoch budget, plus the reported configuration), and the six
orderings of the distributional heads are compared pairwise by Kendall's
$\\tau$. Absolute accuracy improves with capacity and is not the question. The
last column reports how many different heads come first across the six.
Coverage is ranked by distance from nominal rather than magnitude, since a head
can reach high coverage by being wide.}}
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
