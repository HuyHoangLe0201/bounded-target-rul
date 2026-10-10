"""Report the epistemic share of the predictive variance.

sec_head defines the ensemble decomposition and the ratio
rho = V_ep / (V_ep + V_al), and Appendix app:bstvar derives the BS-t variance it
needs -- but the ratio was never reported anywhere in the paper.  The per-row
components are stored by study_regression.py, so the number exists; this script
computes it and writes results/regression/rho.json so the prose can quote a
stored result rather than an ad-hoc calculation.

    python code/study_rho.py
"""
import io
import json
import os
import statistics as st
from collections import defaultdict

os.chdir(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

#  results_both.json carries both benchmarks; results.json is PRONOSTIA only.
SRC = "results/regression/results_both.json"
rows = json.load(io.open(SRC, encoding="utf-8"))["rows"]

by = defaultdict(list)
for r in rows:
    va, ve = r.get("v_al"), r.get("v_ep")
    if va is None or ve is None or (va + ve) <= 0:
        continue
    by[(r["dataset"], r["head"])].append(ve / (ve + va))

out = {}
for (ds, head), v in sorted(by.items()):
    out.setdefault(ds, {})[head] = {
        "n": len(v),
        "median": st.median(v),
        "q1": st.quantiles(v, n=4)[0] if len(v) > 3 else None,
        "q3": st.quantiles(v, n=4)[2] if len(v) > 3 else None,
        "frac_above_half": sum(x > 0.5 for x in v) / len(v),
    }

#  the point head has no aleatory term at all, so rho == 1 identically; it is a
#  check on the decomposition rather than a result about uncertainty
for ds in out:
    dist = {h: s["median"] for h, s in out[ds].items() if not h.startswith("Point")}
    out[ds]["_distributional_median_range"] = [min(dist.values()), max(dist.values())]

with io.open("results/regression/rho.json", "w", encoding="utf-8") as f:
    json.dump(out, f, indent=1)

print(f"{'dataset':<10}{'head':<12}{'n':>5}{'median rho':>12}{'>0.5':>8}")
for ds in sorted(out):
    for head, s in sorted(out[ds].items()):
        if head.startswith("_"):
            continue
        print(f"{ds:<10}{head:<12}{s['n']:>5}{s['median']:>12.3f}"
              f"{s['frac_above_half']:>8.3f}")
    lo, hi = out[ds]["_distributional_median_range"]
    print(f"  -> distributional heads span {lo:.3f} to {hi:.3f}")
print("\nwrote results/regression/rho.json")
