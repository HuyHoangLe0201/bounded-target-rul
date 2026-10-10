"""Is the ordering of predictive heads a property of the families or the network?

Paper 2's central claim is a RANKING of distribution families under one fixed
backbone, and its Limitations section conceded that the ranking might be a
property of that backbone.  This tests it across a SWEEP of backbones rather
than against a single alternative, which is what a claim about representations
needs.

The protocol is identical in every arm except network width: same features, same
splits, same seeds, same schedule, same heads, same epoch budget.  Compare the
ORDERINGS, not the absolute numbers -- absolute accuracy improves with capacity
and that is not the question.

    python code/run_backbone_sweep.py     # produces the arms
    python code/study_backbone.py
"""
from __future__ import annotations

import io
import itertools
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "..", "results", "regression")

METRICS = [("MAE", "MAE"), ("CRPS", "CRPS"), ("cov95", "cov95"),
           ("PIT_KS", "PIT-KS")]
NOMINAL = 0.95

#  (tag, human label).  "" is the paper's reported configuration.
ARMS = [("", "48-24 (reported, 250 ep)"), ("_w32", "32-16"), ("_w48", "48-24"),
        ("_w96", "96-48"), ("_big", "128-64-32"), ("_w256", "256-128")]


def load(tag):
    p = os.path.join(RES, f"results_both{tag}.json")
    if not os.path.exists(p):
        return None
    return json.load(io.open(p, encoding="utf-8"))["tables"]


def order(table, key):
    """Heads ranked best-first, distributional heads only.

    Coverage is scored by DISTANCE FROM NOMINAL, not magnitude: this paper's own
    finding is that a head can reach high coverage by being wide, so ranking by
    raw coverage would call the widest head the best calibrated.
    """
    rows = [(h, v.get(key)) for h, v in table.items()
            if v.get(key) is not None and not h.startswith("Point")]
    if key == "cov95":
        rows = [(h, abs(v - NOMINAL)) for h, v in rows]
    rows.sort(key=lambda t: t[1])          # every metric: smaller is better
    return [h for h, _ in rows]


def tau(a, b):
    idx = {h: i for i, h in enumerate(b)}
    n, conc, disc = len(a), 0, 0
    for i in range(n):
        for j in range(i + 1, n):
            if idx[a[i]] - idx[a[j]] < 0:
                conc += 1
            else:
                disc += 1
    tot = n * (n - 1) / 2
    return (conc - disc) / tot if tot else float("nan")


arms = [(t, lab, load(t)) for t, lab in ARMS]
arms = [(t, lab, d) for t, lab, d in arms if d]
print(f"{len(arms)} arms present: " + ", ".join(lab for _, lab, _ in arms) + "\n")

splits = [s for s in arms[0][2] if all(s in d for _, _, d in arms)]
out = {}

for split in splits:
    print(f"=== {split}")
    out[split] = {}
    for key, label in METRICS:
        orders = {lab: order(d[split], key) for _, lab, d in arms}
        tops = {lab: o[0] for lab, o in orders.items()}
        #  pairwise tau over every pair of arms: the spread is the answer, a
        #  single comparison against one alternative is not
        pairs = list(itertools.combinations(orders, 2))
        taus = [tau(orders[a], orders[b]) for a, b in pairs]
        uniq = sorted(set(tops.values()))
        out[split][key] = {
            "orders": orders, "tops": tops,
            "tau_min": min(taus), "tau_mean": sum(taus) / len(taus),
            "n_distinct_winners": len(uniq), "winners": uniq,
        }
        print(f"  {label:<8} tau min {min(taus):+.2f} mean {sum(taus)/len(taus):+.2f}"
              f"   winners across arms: {', '.join(uniq)}")
    print()

with io.open(os.path.join(RES, "backbone.json"), "w", encoding="utf-8") as f:
    json.dump(out, f, indent=1)
print("wrote results/regression/backbone.json")
