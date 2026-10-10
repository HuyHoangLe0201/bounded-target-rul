"""Paired IG comparisons with and without truncating each predictive law to (0,1].

Companion to study_review_p2.py: bearing-level bootstrap (within each archive)
of the CRPS margins of IG against every rival and of BS against Weibull and
lognormal, per benchmark and pooled with every
bearing weighted equally.  Writes results/regression/review_p2_trunc.json.
"""
import json
import os

import numpy as np

import study_review_p2 as S

NB, SEED = 10000, 1
#  IG against every rival, and BS against the two it is claimed to beat once
#  restricted: a tie IG-BS plus IG < Weibull does not imply BS < Weibull
PAIRS = [("IGauss", r) for r in ("BS", "Weibull", "LogNormal", "Gaussian")] +         [("BS", r) for r in ("Weibull", "LogNormal")]
SHORT = {"IGauss": "IG", "BS": "BS"}


def main():
    _, rows = S.load()
    rng = np.random.default_rng(SEED)
    c = {}
    for r in rows:
        q = np.asarray(r["q"], float)
        key = (r["dataset"], r["bearing"])
        c.setdefault(key, {}).setdefault(r["head"], {"raw": [], "trunc": []})
        c[key][r["head"]]["raw"].append(S.crps_q(q, r["true"]))
        c[key][r["head"]]["trunc"].append(S.crps_q(S.truncate(q)[0], r["true"]))
    arch = {d: sorted(b for (dd, b) in c if dd == d) for d in ("pronostia", "xjtu")}

    def margins(sample, kind, rival, lead="IGauss"):
        bm = {d: [np.mean(c[(d, b)][lead][kind]) - np.mean(c[(d, b)][rival][kind])
                  for b in sample[d]] for d in sample}
        return {"pronostia": float(np.mean(bm["pronostia"])), "xjtu": float(np.mean(bm["xjtu"])),
                "pooled": float(np.mean(bm["pronostia"] + bm["xjtu"]))}

    out = {}
    for kind in ("raw", "trunc"):
        for lead, rival in PAIRS:
            pt = margins(arch, kind, rival, lead)
            bs = {k: [] for k in pt}
            for _ in range(NB):
                smp = {d: [arch[d][i] for i in rng.choice(len(arch[d]), len(arch[d]))] for d in arch}
                for k, v in margins(smp, kind, rival, lead).items():
                    bs[k].append(v)
            a = 0.05 / (2 * len(PAIRS))      # Bonferroni over the pairs of one kind
            out[f"{kind}/{SHORT[lead]}-{rival}"] = {
                k: {"margin": pt[k], "ci95": [float(np.quantile(bs[k], 0.025)), float(np.quantile(bs[k], 0.975))],
                    "ci_bonf": [float(np.quantile(bs[k], a)), float(np.quantile(bs[k], 1 - a))],
                    "p_boot": float(min(1.0, 2 * min(np.mean(np.array(bs[k]) >= 0),
                                                     np.mean(np.array(bs[k]) <= 0))))}
                for k in pt}
    #  Holm step-down over the pooled comparisons of each kind
    for kind in ("raw", "trunc"):
        keys = [f"{kind}/{SHORT[l]}-{r}" for l, r in PAIRS]
        order = sorted(keys, key=lambda k: out[k]["pooled"]["p_boot"])
        run = 0.0
        for i, k in enumerate(order):
            run = max(run, min(1.0, (len(keys) - i) * out[k]["pooled"]["p_boot"]))
            out[k]["pooled"]["p_holm"] = run
    json.dump(out, open(os.path.join(S.RES, "review_p2_trunc.json"), "w"), indent=1)
    for k, v in out.items():
        print(f"{k:22s} " + "  ".join(f"{d}: {x['margin']:+.4f} [{x['ci95'][0]:+.4f},{x['ci95'][1]:+.4f}]"
                                      for d, x in v.items()))


if __name__ == "__main__":
    main()
