"""CRPS in hours by the integral definition, for every head on the fraction.

CRPS(F, x) = int (F(r) - 1{r >= x})^2 dr is finite whenever the tail of F decays
faster than r^(-1/2); a finite mean is needed only for the kernel form
E|X-x| - E|X-X'|/2. With R = t y/(1-y) and dr = t dy/(1-y)^2,

    CRPS_h = t * int_0^1 (F_Y(y) - 1{y >= y*})^2 / (1-y)^2 dy,

which is finite for any law of Y with bounded density near one: the truncated
heads have infinite mean RUL but finite CRPS in hours. A law with an atom at
y >= 1 (an unrestricted conventional head under eq. Rext) has an atom at R = inf
and infinite CRPS; such heads enter restricted to (0,1) here.

F_Y is piecewise linear through the stored quantiles, with the outer 0.5 % on
each side spread uniformly over (0, q_0.005) and (q_0.995, 1). Two variants:
  trimmed   the upper tail beyond q_0.995 omitted (a lower bound)
  uniform   the upper tail included as uniform mass up to one (heavy: a Pareto
            tail of index one in hours, which overstates lighter tails)
The heads trained on w = R/t are scored the same way from their fraction-scale
quantiles. Two protocols: the reported prediction points and inspections fixed in
operating time (GBS_EVAL=grid). Bearing-level paired margins pooled over the 26
bearings, bootstrap within each benchmark, Holm over each family.
Writes results/regression/hours_crps.json.
"""
import json
import os

import numpy as np

import study_review_p2 as S

DT = {"pronostia": 10.0, "xjtu": 60.0}
NB, SEED = 10000, 8
CONV = ["BS", "BS-t", "Weibull", "LogNormal", "IGauss", "Gaussian"]
BOUND = ["LogitNormal", "Beta", "BS-trunc", "IGauss-trunc"]
WHEADS = ["BS", "BS-t", "Weibull", "LogNormal", "IGauss", "Gaussian", "Gaussian-trunc0"]
PROTOCOLS = {
    "main": ("results_both_causal.json", ["results_both_ratio.json", "results_both_ratio_g0.json"]),
    "grid": ("results_both_grid.json", ["results_both_ratio_grid.json", "results_both_ratio_grid_g0.json"]),
}


def crps_hours(q, y_true, t_h, variant):
    """Integral CRPS in hours of the law of Y given by quantiles q at TAU."""
    q = np.clip(np.maximum.accumulate(np.asarray(q, float)), 1e-9, 1 - 1e-7)
    ys = np.concatenate([[0.0], q, [1.0]])
    Fs = np.concatenate([[0.0], S.TAU, [1.0]])
    top = q[-1]
    # integrate on a grid dense near the top quantile, up to q_0.995
    g = np.unique(np.concatenate([np.linspace(0.0, top, 4000),
                                  top - (top) * np.geomspace(1e-6, 1.0, 2000)[::-1]]))
    g = g[(g >= 0) & (g <= top)]
    F = np.interp(g, ys, Fs)
    ind = (g >= y_true).astype(float)
    integrand = (F - ind) ** 2 / (1.0 - g) ** 2
    val = np.trapezoid(integrand, g)
    if variant == "uniform":
        # beyond q_0.995: 1 - F = 0.005 (1-y)/(1-top); integrand constant = (0.005/(1-top))^2
        # when y* <= top; otherwise split at y*
        c = 0.005 / (1.0 - top)
        if y_true <= top:
            val += c * c * (1.0 - top)
        else:
            gg = np.linspace(top, 1.0 - 1e-9, 4000)
            FF = 1.0 - c * (1.0 - gg)
            ii = (gg >= y_true).astype(float)
            val += np.trapezoid((FF - ii) ** 2 / (1.0 - gg) ** 2, gg)
    return t_h * val


def load_rows(frac_file, ratio_files):
    rows = []
    for r in json.load(open(os.path.join(S.RES, frac_file)))["rows"]:
        if r["q"] is None or r["head"] not in CONV + BOUND:
            continue
        q = np.asarray(r["q"], float)
        if r["head"] in CONV:
            q = S.truncate(q)[0]
        rows.append((r, q, r["head"] + ("@restricted" if r["head"] in CONV else "")))
    for f in ratio_files:
        for r in json.load(open(os.path.join(S.RES, f)))["rows"]:
            if r["q"] is None or r["head"] not in WHEADS:
                continue
            rows.append((r, np.asarray(r["q"], float), r["head"] + "@w"))
    return rows


def pinned(rows):
    """Inspections at which a compared head puts its 0.995 quantile within 1e-6 of
    one. Its CRPS in hours is then set by the clip at 1 - 1e-7, a lower bound."""
    return {(r["dataset"], r["bearing"], r["k"]) for r, q, key in rows
            if q[-1] >= 1 - 1e-6 and key not in ("Beta", "LogitNormal")}


def analyse(rows, rng, variants=("trimmed", "uniform"), exclude=()):
    c = {v: {} for v in variants}
    for r, q, key in rows:
        if (r["dataset"], r["bearing"], r["k"]) in exclude:
            continue
        t_h = r["k"] * DT[r["dataset"]] / 3600.0
        for v in c:
            c[v].setdefault((r["dataset"], r["bearing"]), {}).setdefault(key, []).append(
                crps_hours(q, r["true"], t_h, v.split("_")[0]))
    out = {}
    for v, cc in c.items():
        arch = {ds: sorted(b for (dd, b) in cc if dd == ds) for ds in DT}
        keys = sorted({k for d in cc.values() for k in d})
        res = {"heads": {f"{ds}/{k}": float(np.mean([x for b in arch[ds] for x in cc[(ds, b)][k]]))
                         for ds in DT for k in keys},
               "pooled": {k: float(np.mean([np.mean(cc[kk][k]) for kk in cc])) for k in keys},
               "pairs": {}}

        def margins(sample, a, b):
            bm = {ds: [np.mean(cc[(ds, x)][a]) - np.mean(cc[(ds, x)][b]) for x in sample[ds]] for ds in sample}
            return {"pronostia": float(np.mean(bm["pronostia"])), "xjtu": float(np.mean(bm["xjtu"])),
                    "pooled": float(np.mean(bm["pronostia"] + bm["xjtu"]))}

        fams = {"vs_w": [(a, b + "@w") for a in ("BS-trunc", "IGauss-trunc") for b in WHEADS],
                "vs_restricted": [(a, b + "@restricted") for a in ("BS-trunc", "IGauss-trunc") for b in CONV]}
        for fam, pairs in fams.items():
            for a, b in pairs:
                pt = margins(arch, a, b)
                entry = {"family": fam, **{k: {"margin": pt[k]} for k in pt}}
                if v.startswith("uniform"):            # the bootstrap is run on the main variant only
                    bs = {k: [] for k in pt}
                    for _ in range(NB):
                        smp = {ds: [arch[ds][i] for i in rng.choice(len(arch[ds]), len(arch[ds]))] for ds in arch}
                        for k, x in margins(smp, a, b).items():
                            bs[k].append(x)
                    for k in pt:
                        entry[k].update(ci95=[float(np.quantile(bs[k], 0.025)), float(np.quantile(bs[k], 0.975))],
                                        p_boot=float(min(1.0, 2 * min(np.mean(np.array(bs[k]) >= 0),
                                                                      np.mean(np.array(bs[k]) <= 0)))))
                res["pairs"][f"{a} - {b}"] = entry
            if v.startswith("uniform"):
                ks = [f"{a} - {b}" for a, b in pairs]
                run = 0.0
                for i, k in enumerate(sorted(ks, key=lambda k: res["pairs"][k]["pooled"]["p_boot"])):
                    run = max(run, min(1.0, (len(ks) - i) * res["pairs"][k]["pooled"]["p_boot"]))
                    res["pairs"][k]["pooled"]["p_holm"] = run
        out[v] = res
        print(f"=== {v}")
        for k in sorted(res["pooled"], key=res["pooled"].get):
            print(f"   {k:24s} pooled {res['pooled'][k]:.3f} h   P {res['heads']['pronostia/' + k]:.3f}  X {res['heads']['xjtu/' + k]:.3f}")
        for k, x in res["pairs"].items():
            p = x["pooled"]
            extra = (f" [{p['ci95'][0]:+.3f},{p['ci95'][1]:+.3f}] holm={p['p_holm']:.3f}" if "ci95" in p else "")
            print(f"   {k:32s} {p['margin']:+.3f}{extra}   P {x['pronostia']['margin']:+.3f}  X {x['xjtu']['margin']:+.3f}")
    return out


def main():
    rng = np.random.default_rng(SEED)
    out = {}
    for name, (frac_file, ratio_files) in PROTOCOLS.items():
        print(f"##### protocol {name}")
        rows = load_rows(frac_file, ratio_files)
        out[name] = analyse(rows, rng)
        # sensitivity: drop the inspections pinned at the bound, for every head, so
        # the pairs stay matched; this can only favour the head that is pinned
        pin = pinned(rows)
        out[name]["pinned"] = sorted(pin)
        if pin:
            print(f"##### protocol {name}, without {len(pin)} pinned inspections {sorted(pin)}")
            out[name].update(analyse(rows, rng, ("uniform_excl",), pin))
    json.dump(out, open(os.path.join(S.RES, "hours_crps.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
