"""LaTeX tables for the paper-2 review analyses (results/regression/review_p2*.json).

  tab_support      CRPS, MAE and coverage before and after restricting each
                   predictive law to the target's support (0,1)
  tab_calib_ci     bearing-level bootstrap intervals for coverage and PIT-KS
  tab_points       prediction points per held-out bearing
"""
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "..", "results", "regression")
TAB = os.path.join(HERE, "..", "results", "tables")
HEADS = ["BS", "BS-t", "Weibull", "LogNormal", "IGauss", "Gaussian"]
NAME = {"BS": "BS", "BS-t": "BS-$t$", "Weibull": "Weibull", "LogNormal": "lognormal",
        "IGauss": "inverse Gaussian", "Gaussian": "Gaussian"}
DS = {"pronostia": "PRONOSTIA", "xjtu": "XJTU-SY"}


def pfmt(p):
    """A bootstrap p-value: below 0.001 it is reported as a bound, never as 0.000."""
    return "$<0.001$" if p < 0.001 else f"{p:.3f}"


def w(name, lines):
    open(os.path.join(TAB, name + ".tex"), "w", encoding="utf-8", newline="\n").write("\n".join(lines) + "\n")


def main():
    d = json.load(open(os.path.join(RES, "review_p2.json")))
    t = json.load(open(os.path.join(RES, "review_p2_trunc.json")))

    L = [r"\begin{table}[htbp]", r"\centering\small",
         r"\caption{Each predictive law before and after restriction to the target's support"
         r" $(0,1)$, applied after fitting. Outside: mean predictive mass outside $(0,1)$.}",
         r"\label{tab:support}",
         r"\begin{adjustbox}{max width=\linewidth}",
         r"\begin{tabular}{@{}l r rr rr rr@{}}", r"\toprule",
         r"& & \multicolumn{2}{c}{CRPS} & \multicolumn{2}{c}{MAE} & \multicolumn{2}{c}{cov$_{95}$}\\",
         r"\cmidrule(lr){3-4}\cmidrule(lr){5-6}\cmidrule(lr){7-8}",
         r"head & outside & raw & restricted & raw & restricted & raw & restricted\\", r"\midrule"]
    for ds, lab in DS.items():
        L.append(r"\multicolumn{8}{@{}l}{\emph{" + lab + r"}}\\")
        for h in HEADS:
            v = d["trunc"][f"{ds}/{h}"]
            L.append(f"\\quad {NAME[h]} & {v['mass_above_1']:.3f} & {v['CRPS'][0]:.4f} & {v['CRPS'][1]:.4f}"
                     f" & {v['MAE'][0]:.4f} & {v['MAE'][1]:.4f} & {v['cov95'][0]:.3f} & {v['cov95'][1]:.3f}\\\\")
    L += [r"\midrule",
          r"\multicolumn{8}{@{}l}{\emph{Pooled CRPS margin, 26 bearings"
          r" (raw; restricted): $95\,\%$ bootstrap interval and Holm-adjusted $p$ over six}}\\"]
    SH = {"IG": "IG", "BS": "BS"}
    for lead, rival in [("IG", "BS"), ("IG", "Weibull"), ("IG", "LogNormal"), ("IG", "Gaussian"),
                        ("BS", "Weibull"), ("BS", "LogNormal")]:
        a, b = t[f"raw/{lead}-{rival}"]["pooled"], t[f"trunc/{lead}-{rival}"]["pooled"]
        f = lambda x: (f"${x['margin']:+.3f}$ [${x['ci95'][0]:+.3f}$, ${x['ci95'][1]:+.3f}$]"
                       f" ({pfmt(x['p_holm'])})")
        L.append(f"\\quad {SH[lead]} $-$ {NAME[rival]} & & \\multicolumn{{3}}{{l}}{{{f(a)}}}"
                 f" & \\multicolumn{{3}}{{l}}{{{f(b)}}}\\\\")
    L += [r"\bottomrule", r"\end{tabular}", r"\end{adjustbox}", r"\end{table}"]
    w("tab_support", L)

    L = [r"\begin{table}[htbp]", r"\centering\small",
         r"\caption{Calibration summaries with $95\,\%$ intervals from resampling held-out bearings"
         r" ($10\,000$ resamples). Nominal coverage is $0.95$.}",
         r"\label{tab:calib_ci}",
         r"\begin{tabular}{@{}l rr rr@{}}", r"\toprule",
         r"& \multicolumn{2}{c}{PRONOSTIA} & \multicolumn{2}{c}{XJTU-SY}\\",
         r"\cmidrule(lr){2-3}\cmidrule(lr){4-5}",
         r"head & cov$_{95}$ & PIT-KS & cov$_{95}$ & PIT-KS\\", r"\midrule"]
    for h in HEADS:
        cells = []
        for ds in DS:
            v = d["calib_ci"][f"{ds}/{h}"]
            cells.append(f"[{v['cov95'][0]:.2f}, {v['cov95'][1]:.2f}] & [{v['PIT_KS'][0]:.2f}, {v['PIT_KS'][1]:.2f}]")
        L.append(f"{NAME[h]} & " + " & ".join(cells) + r"\\")
    L += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    w("tab_calib_ci", L)

    L = [r"\begin{table}[htbp]", r"\centering\small",
         r"\caption{Prediction points per held-out bearing, as fractions of the bearing's life;"
         r" $^{\ast}$ marks the official IEEE PHM 2012 truncation point. On Bearing1\_4 it falls"
         r" three acquisitions before the $80\,\%$ point; both are kept.}",
         r"\label{tab:points}",
         r"\begin{tabular}{@{}l l@{}}", r"\toprule", r"bearing & life fractions\\", r"\midrule"]
    for ds, lab in DS.items():
        L.append(r"\multicolumn{2}{@{}l}{\emph{" + lab + r"}}\\")
        for b, pts in d["points"][ds].items():
            s = ", ".join(f"{f:.3f}" + ("$^{\\ast}$" if o else "") for f, o in pts)
            L.append(f"\\quad {b.replace('_', chr(92) + '_')} & {s}\\\\")
    L += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    w("tab_points", L)
    L = [r"\begin{table}[htbp]", r"\centering\small",
         r"\caption{Mean $|\mathrm{Er}|$ [\%] and PHM score at the predictive median and at the"
         r" summary that optimises each criterion in expectation. ---: undefined (see text).}",
         r"\label{tab:eropt}",
         r"\begin{adjustbox}{max width=\linewidth}",
         r"\begin{tabular}{@{}l rr rr rr rr@{}}", r"\toprule",
         r"& \multicolumn{4}{c}{PRONOSTIA} & \multicolumn{4}{c}{XJTU-SY}\\",
         r"\cmidrule(lr){2-5}\cmidrule(lr){6-9}",
         r"& \multicolumn{2}{c}{$|\mathrm{Er}|$} & \multicolumn{2}{c}{PHM}"
         r" & \multicolumn{2}{c}{$|\mathrm{Er}|$} & \multicolumn{2}{c}{PHM}\\",
         r"\cmidrule(lr){2-3}\cmidrule(lr){4-5}\cmidrule(lr){6-7}\cmidrule(lr){8-9}",
         r"head & median & optimal & median & optimal & median & optimal & median & optimal\\",
         r"\midrule"]
    f1 = lambda x: "---" if x is None else f"{x:.1f}"
    f3 = lambda x: "---" if x is None else f"{x:.3f}"
    for h in HEADS:
        cells = []
        for ds in DS:
            v = d["er_opt"][f"{ds}/{h}"]
            cells.append(f"{f1(v['absEr_median'])} & {f1(v['absEr_weighted_median'])}"
                         f" & {f3(v['PHM_median'])} & {f3(v['PHM_optimal'])}")
        L.append(f"{NAME[h]} & " + " & ".join(cells) + r"\\")
    L += [r"\bottomrule", r"\end{tabular}", r"\end{adjustbox}", r"\end{table}"]
    w("tab_eropt", L)
    L = [r"\begin{table}[htbp]", r"\centering\small",
         r"\caption{The eleven official truncation points scored with the archive RUL of"
         r" Bearing1\_4 and with the published one; every other bearing is unchanged.}",
         r"\label{tab:b14}",
         r"\begin{adjustbox}{max width=\linewidth}",
         r"\begin{tabular}{@{}l rr rr rr rr rr@{}}", r"\toprule",
         r"& \multicolumn{2}{c}{MAE} & \multicolumn{2}{c}{$|\mathrm{Er}|$ [\%]}"
         r" & \multicolumn{2}{c}{PHM} & \multicolumn{2}{c}{CRPS} & \multicolumn{2}{c}{cov$_{95}$}\\",
         r"\cmidrule(lr){2-3}\cmidrule(lr){4-5}\cmidrule(lr){6-7}\cmidrule(lr){8-9}\cmidrule(lr){10-11}",
         r"head" + r" & archive & published" * 5 + r"\\", r"\midrule"]
    for h in HEADS:
        a, p = d["b14"]["archive"][h], d["b14"]["published"][h]
        L.append(f"{NAME[h]} & {a['MAE']:.3f} & {p['MAE']:.3f} & {a['mean_absEr']:.1f} & {p['mean_absEr']:.1f}"
                 f" & {a['PHM']:.3f} & {p['PHM']:.3f} & {a['CRPS']:.3f} & {p['CRPS']:.3f}"
                 f" & {a['cov95']:.2f} & {p['cov95']:.2f}\\\\")
    L += [r"\bottomrule", r"\end{tabular}", r"\end{adjustbox}", r"\end{table}"]
    w("tab_b14", L)
    bounded_table()
    print("wrote tab_support, tab_calib_ci, tab_points, tab_eropt, tab_b14, tab_bounded,"
          " tab_margins, tab_robust, tab_hours")


def bounded_table():
    """Heads trained on the bounded target, beside the strongest heads as fitted."""
    import numpy as np
    import study_review_p2 as S
    c = json.load(open(os.path.join(RES, "results_both_causal.json")))
    b = json.load(open(os.path.join(RES, "bounded.json")))
    rows = [r for r in c["rows"] if r["q"] is not None]
    SHOW = ["IGauss-trunc", "BS-trunc", "Beta", "LogitNormal", "IGauss", "Weibull", "Gaussian"]
    NM = {"IGauss-trunc": "IG, truncated at one", "BS-trunc": "BS, truncated at one",
          "Beta": "Beta", "LogitNormal": "logit-normal", "IGauss": "IG (as fitted)",
          "Weibull": "Weibull (as fitted)", "Gaussian": "Gaussian (as fitted)"}
    rng = np.random.default_rng(3)

    def cov_ci(ds, h):
        byb = {}
        for r in rows:
            if r["dataset"] == ds and r["head"] == h:
                lo, hi = S.interval(np.asarray(r["q"], float), 0.025)
                byb.setdefault(r["bearing"], []).append(float(lo <= r["true"] <= hi))
        ks = list(byb)
        bs = [np.mean([v for i in rng.choice(len(ks), len(ks)) for v in byb[ks[i]]]) for _ in range(S.NB)]
        return np.quantile(bs, 0.025), np.quantile(bs, 0.975)

    L = [r"\begin{table}[htbp]", r"\centering\small",
         r"\caption{Heads trained on a likelihood that respects the target's support $(0,1)$,"
         r" beside the strongest heads as fitted. cov$_{95}$ with its bearing-level $95\,\%$ interval.}",
         r"\label{tab:bounded}",
         r"\begin{adjustbox}{max width=\linewidth}",
         r"\begin{tabular}{@{}l rrrr rrrr@{}}", r"\toprule",
         r"& \multicolumn{4}{c}{PRONOSTIA} & \multicolumn{4}{c}{XJTU-SY}\\",
         r"\cmidrule(lr){2-5}\cmidrule(lr){6-9}",
         r"head & MAE & CRPS & cov$_{95}$ & PIT-KS & MAE & CRPS & cov$_{95}$ & PIT-KS\\", r"\midrule"]
    for h in SHOW:
        cells = []
        for ds, key in (("pronostia", "pronostia/all"), ("xjtu", "xjtu/all")):
            v = c["tables"][key][h]
            lo, hi = cov_ci(ds, h)
            cells.append(f"{v['MAE']:.3f} & {v['CRPS']:.3f} & {v['cov95']:.2f} [{lo:.2f}, {hi:.2f}] & {v['PIT_KS']:.3f}")
        L.append(f"{NM[h]} & " + " & ".join(cells) + r"\\")
        if h == "LogitNormal":
            L.append(r"\midrule")
    #  the point head, so that the comparison on its own loss can be read off directly
    pm = [c["tables"][k]["Point(MAE)"]["MAE"] for k in ("pronostia/all", "xjtu/all")]
    L.append(r"\midrule")
    #  the distribution-free baseline, bootstrapped after the heads above so that
    #  their intervals are unchanged
    qp = os.path.join(RES, "results_both_quant.json")
    if os.path.exists(qp):
        dq = json.load(open(qp))
        rows += [r for r in dq["rows"] if r["head"] == "Quantile"]
        cells = []
        for ds, key in (("pronostia", "pronostia/all"), ("xjtu", "xjtu/all")):
            v = dq["tables"][key]["Quantile"]
            lo, hi = cov_ci(ds, "Quantile")
            cells.append(f"{v['MAE']:.3f} & {v['CRPS']:.3f} & {v['cov95']:.2f} [{lo:.2f}, {hi:.2f}] & {v['PIT_KS']:.3f}")
        L.append("quantile regression & " + " & ".join(cells) + r"\\")
    qlp = os.path.join(RES, "results_both_quantl.json")
    if os.path.exists(qlp):
        dq = json.load(open(qlp))
        rows += [r for r in dq["rows"] if r["head"] == "Quantile-logit"]
        cells = []
        for ds, key in (("pronostia", "pronostia/all"), ("xjtu", "xjtu/all")):
            v = dq["tables"][key]["Quantile-logit"]
            lo, hi = cov_ci(ds, "Quantile-logit")
            cells.append(f"{v['MAE']:.3f} & {v['CRPS']:.3f} & {v['cov95']:.2f} [{lo:.2f}, {hi:.2f}] & {v['PIT_KS']:.3f}")
        L.append(r"quantile regression in $(0,1)$ & " + " & ".join(cells) + r"\\")
    L.append(f"point head (MAE loss) & {pm[0]:.3f} & --- & --- & --- & {pm[1]:.3f} & --- & --- & ---\\\\")
    L += [r"\bottomrule", r"\end{tabular}", r"\end{adjustbox}", r"\end{table}"]
    w("tab_bounded", L)
    margins_table(b)
    robustness_table()
    hours_table()
    ratio_table()
    quantile_table()


SH = {"IGauss-trunc": r"IG$_{\rm tr}$", "BS-trunc": r"BS$_{\rm tr}$", "Beta": "Beta",
      "LogitNormal": "logit-normal", "IGauss": "IG", "Weibull": "Weibull", "Gaussian": "Gaussian",
      "IGauss@post": "IG restricted", "BS@post": "BS restricted",
      "BS": "BS", "BS-t": "BS-$t$", "LogNormal": "lognormal"}
PAIR_ORDER = ([("IGauss-trunc", r) for r in ("BS", "BS-t", "Weibull", "LogNormal", "IGauss", "Gaussian")]
              + [("BS-trunc", r) for r in ("BS", "BS-t", "Weibull", "LogNormal", "IGauss", "Gaussian")]
              + [("IGauss-trunc", "BS-trunc"), ("IGauss-trunc", "Beta"), ("IGauss-trunc", "LogitNormal"),
                 ("BS-trunc", "Beta"), ("BS-trunc", "LogitNormal")])
BLOCK = {0: r"\emph{Truncated heads against the conventional heads as fitted (Holm over 12)}",
         12: r"\emph{Among the support-respecting heads (Holm over 6)}"}


def margins_table(b):
    """Paired CRPS margins per benchmark and pooled, for every comparison claimed,
    with the pooled margin at inspections fixed in operating time beside them."""
    ci = lambda x: f"${x['margin']:+.3f}$ [${x['ci95'][0]:+.3f}$, ${x['ci95'][1]:+.3f}$]"
    rp = os.path.join(RES, "robustness.json")
    grid = json.load(open(rp))["grid"]["pairs"] if os.path.exists(rp) else None
    ncol = 5 if grid else 4
    L = [r"\begin{table}[htbp]", r"\centering\footnotesize",
         r"\caption{Paired CRPS margins, row head minus rival, with $95\,\%$ bootstrap intervals over"
         r" bearings. Pooled: the $26$ bearings weighted equally, with the Holm-adjusted $p$."
         r" Fixed-time: the pooled margin at inspections fixed in operating time"
         r" (\Cref{sec:robust}). Negative favours the row head.}",
         r"\label{tab:margins}",
         r"\begin{adjustbox}{max width=\linewidth}",
         r"\begin{tabular}{@{}l " + "l" * (ncol - 1) + r"@{}}", r"\toprule",
         r"comparison & PRONOSTIA ($11$) & XJTU-SY ($15$) & pooled ($26$), Holm $p$"
         + (r" & fixed-time, Holm $p$" if grid else "") + r"\\", r"\midrule"]
    for i, (a, r_) in enumerate(PAIR_ORDER):
        if i in BLOCK:
            if i:
                L.append(r"\addlinespace")
            L.append(r"\multicolumn{" + str(ncol) + r"}{@{}l}{" + BLOCK[i] + r"}\\")
        v = b["pairs"][f"{a} - {r_}"]
        row = (f"\\quad {SH[a]} $-$ {SH[r_]} & {ci(v['pronostia'])} & {ci(v['xjtu'])}"
               f" & {ci(v['pooled'])} ({pfmt(v['pooled']['p_holm'])})")
        if grid:
            g = grid[f"{a} - {r_}"]["pooled"]
            row += f" & {ci(g)} ({pfmt(g['p_holm'])})"
        L.append(row + r"\\")
    L += [r"\bottomrule", r"\end{tabular}", r"\end{adjustbox}", r"\end{table}"]
    w("tab_margins", L)


def robustness_table():
    """Pooled margins under each change of protocol; * marks Holm-adjusted p < 0.05."""
    path = os.path.join(RES, "robustness.json")
    if not os.path.exists(path):
        return
    r = json.load(open(path))
    COLS = [("main", "reported"), ("grid", "fixed-time points"), ("seeds5", "seeds 5--9"),
            ("lr1e-3", r"lr $10^{-3}$"), ("lr1e-2", r"lr $10^{-2}$")]
    COLS = [c for c in COLS if c[0] in r]
    cell = lambda v: (f"${v['margin']:+.3f}" + (r"^{\ast}" if v["p_holm"] < 0.05 else r"\phantom{^{\ast}}") + "$")
    L = [r"\begin{table}[htbp]", r"\centering\footnotesize",
         r"\caption{Pooled CRPS margins under changes of protocol; $^{\ast}$: Holm-adjusted $p<0.05$"
         r" within the block. Fixed-time points: every $50$th (PRONOSTIA) or $10$th (XJTU-SY)"
         r" acquisition after the healthy window.}",
         r"\label{tab:robust}",
         r"\begin{adjustbox}{max width=\linewidth}",
         r"\begin{tabular}{@{}l " + "r" * len(COLS) + r"@{}}", r"\toprule",
         "comparison & " + " & ".join(c[1] for c in COLS) + r"\\", r"\midrule"]
    for i, (a, r_) in enumerate(PAIR_ORDER):
        if i in BLOCK:
            if i:
                L.append(r"\addlinespace")
            L.append(r"\multicolumn{" + str(len(COLS) + 1) + r"}{@{}l}{" + BLOCK[i] + r"}\\")
        L.append(f"\\quad {SH[a]} $-$ {SH[r_]} & "
                 + " & ".join(cell(r[c]["pairs"][f"{a} - {r_}"]["pooled"]) for c, _ in COLS) + r"\\")
    L += [r"\bottomrule", r"\end{tabular}", r"\end{adjustbox}", r"\end{table}"]
    w("tab_robust", L)

    #  the same comparisons with their bootstrap intervals, for the appendix
    ci = lambda v: f"[${v['ci95'][0]:+.3f}$, ${v['ci95'][1]:+.3f}$]"
    L = [r"\begin{table}[htbp]", r"\centering\scriptsize",
         r"\caption{The pooled margins of \Cref{tab:robust} with their $95\,\%$ bootstrap intervals"
         r" over bearings, by protocol.}",
         r"\label{tab:robust_ci}",
         r"\begin{adjustbox}{max width=\linewidth}",
         r"\begin{tabular}{@{}l " + "c" * len(COLS) + r"@{}}", r"\toprule",
         "comparison & " + " & ".join(c[1] for c in COLS) + r"\\", r"\midrule"]
    for i, (a, r_) in enumerate(PAIR_ORDER):
        if i in BLOCK:
            if i:
                L.append(r"\addlinespace")
            L.append(r"\multicolumn{" + str(len(COLS) + 1) + r"}{@{}l}{" + BLOCK[i] + r"}\\")
        L.append(f"\\quad {SH[a]} $-$ {SH[r_]} & "
                 + " & ".join(ci(r[c]["pairs"][f"{a} - {r_}"]["pooled"]) for c, _ in COLS) + r"\\")
    L += [r"\bottomrule", r"\end{tabular}", r"\end{adjustbox}", r"\end{table}"]
    w("tab_robust_ci", L)

    #  the same comparisons across the backbone sweep
    BB = [("main", "48--24, reported"), ("w32", "32--16"), ("w48", "48--24"), ("w96", "96--48"),
          ("big", "128--64--32"), ("w256", "256--128")]
    if not all(k in r for k, _ in BB):
        return
    L = [r"\begin{table}[htbp]", r"\centering\footnotesize",
         r"\caption{Pooled CRPS margins across backbones; $^{\ast}$: Holm-adjusted $p<0.05$ within the"
         r" block. All but the reported backbone are trained for $300$ epochs.}",
         r"\label{tab:robust_bb}",
         r"\begin{adjustbox}{max width=\linewidth}",
         r"\begin{tabular}{@{}l " + "r" * len(BB) + r"@{}}", r"\toprule",
         "comparison & " + " & ".join(c[1] for c in BB) + r"\\", r"\midrule"]
    for i, (a, r_) in enumerate(PAIR_ORDER):
        if i in BLOCK:
            if i:
                L.append(r"\addlinespace")
            L.append(r"\multicolumn{" + str(len(BB) + 1) + r"}{@{}l}{" + BLOCK[i] + r"}\\")
        L.append(f"\\quad {SH[a]} $-$ {SH[r_]} & "
                 + " & ".join(cell(r[c]["pairs"][f"{a} - {r_}"]["pooled"]) for c, _ in BB) + r"\\")
    L += [r"\bottomrule", r"\end{tabular}", r"\end{adjustbox}", r"\end{table}"]
    w("tab_robust_bb", L)


def ratio_table():
    """The time-scale comparison: CRPS on the fraction and the integral CRPS in
    hours for the heads trained on w, the truncated heads and the conventional
    heads restricted after fitting; and the paired margins."""
    rp, hp = os.path.join(RES, "ratio.json"), os.path.join(RES, "hours_crps.json")
    if not (os.path.exists(rp) and os.path.exists(hp)):
        return
    rr = json.load(open(rp))
    r = rr["main"]
    hj = json.load(open(hp))
    hc, hg = hj["main"]["uniform"], hj["grid"]["uniform_excl"]
    c = json.load(open(os.path.join(RES, "results_both_causal.json")))["tables"]
    tr = json.load(open(os.path.join(RES, "review_p2.json")))["trunc"]
    NMR = {"BS": "BS", "BS-t": "BS-$t$", "Weibull": "Weibull", "LogNormal": "lognormal",
           "IGauss": "IG", "Gaussian": "Gaussian", "Gaussian-trunc0": "Gaussian, truncated at zero",
           "BS-trunc": "BS", "IGauss-trunc": "IG"}
    L = [r"\begin{table}[htbp]", r"\centering\small",
         r"\caption{CRPS on the fraction ($y$) and integral CRPS in hours ($h$) for the heads trained on"
         r" $w=R/t$, the heads truncated at one and the conventional heads restricted to $(0,1)$ after"
         r" fitting. Coverage and PIT are the same on both scales. Reported prediction points.}",
         r"\label{tab:ratio}",
         r"\begin{adjustbox}{max width=\linewidth}",
         r"\begin{tabular}{@{}l rrrr rrrr@{}}", r"\toprule",
         r"& \multicolumn{4}{c}{PRONOSTIA} & \multicolumn{4}{c}{XJTU-SY}\\",
         r"\cmidrule(lr){2-5}\cmidrule(lr){6-9}",
         r"head & CRPS$_y$ & CRPS$_h$ & cov$_{95}$ & PIT-KS & CRPS$_y$ & CRPS$_h$ & cov$_{95}$ & PIT-KS\\",
         r"\midrule"]

    def block(title, items):
        L.append(r"\multicolumn{9}{@{}l}{\emph{" + title + r"}}\\")
        for h, cy, ch, cv, pk in items:
            cells = []
            for ds in ("pronostia", "xjtu"):
                cells.append(f"{cy(ds):.3f} & {ch(ds):.2f} & {cv(ds):.2f} & {pk(ds):.3f}")
            L.append(f"\\quad {NMR[h]} & " + " & ".join(cells) + r"\\")

    key = {"pronostia": "pronostia/all", "xjtu": "xjtu/all"}
    block("trained on $w$", [(h, (lambda ds, h=h: r["heads"][f"{ds}/{h}@y"]),
                              (lambda ds, h=h: hc["heads"][f"{ds}/{h}@w"]),
                              (lambda ds, h=h: r["tables"][key[ds]][h]["cov95"]),
                              (lambda ds, h=h: r["tables"][key[ds]][h]["PIT_KS"]))
                             for h in ("BS", "BS-t", "Weibull", "LogNormal", "IGauss", "Gaussian", "Gaussian-trunc0")])
    block("trained on $y$, truncated at one", [(h, (lambda ds, h=h: r["heads"][f"{ds}/{h}@y"]),
                                                (lambda ds, h=h: hc["heads"][f"{ds}/{h}"]),
                                                (lambda ds, h=h: c[key[ds]][h]["cov95"]),
                                                (lambda ds, h=h: c[key[ds]][h]["PIT_KS"]))
                                               for h in ("BS-trunc", "IGauss-trunc")])
    block("trained on $y$, restricted to $(0,1)$ after fitting",
          [(h, (lambda ds, h=h: tr[f"{ds}/{h}"]["CRPS"][1]),
            (lambda ds, h=h: hc["heads"][f"{ds}/{h}@restricted"]),
            (lambda ds, h=h: tr[f"{ds}/{h}"]["cov95"][1]),
            (lambda ds, h=h: tr[f"{ds}/{h}"]["PIT_KS"][1]))
           for h in ("BS", "BS-t", "Weibull", "LogNormal", "IGauss", "Gaussian")])
    L += [r"\bottomrule", r"\end{tabular}", r"\end{adjustbox}", r"\end{table}"]
    w("tab_ratio", L)

    cell = lambda v: (f"${v['margin']:+.3f}$ [${v['ci95'][0]:+.3f}$, ${v['ci95'][1]:+.3f}$]"
                      f" ({pfmt(v['p_holm'])})")
    nm = lambda k: NMR[k.split("@")[0]] + (r"$_{\rm tr}$" if k.split("@")[0].endswith("trunc") else "")
    L = [r"\begin{table}[htbp]", r"\centering\scriptsize",
         r"\caption{Pooled margins for the time-scale comparison, with $95\,\%$ bootstrap intervals over"
         r" bearings and Holm-adjusted $p$ within each block. On the fraction: CRPS, at the reported and"
         r" the fixed-time points. In hours: integral CRPS, at the fixed-time points without the three"
         r" inspections at which a truncated head collapses onto the bound (\Cref{sec:hours}).}",
         r"\label{tab:ratio_pairs}",
         r"\begin{adjustbox}{max width=\linewidth}",
         r"\begin{tabular}{@{}l l l@{}}", r"\toprule",
         r"comparison & reported points & fixed-time points\\", r"\midrule",
         r"\multicolumn{3}{@{}l}{\emph{On the fraction: truncated head minus head trained on $w$ (Holm over 14)}}\\"]
    for k, v in rr["main"]["pairs"].items():
        if v["family"] != "fraction":
            continue
        a, b = k.split(" - ")
        g = rr["grid"]["pairs"].get(k)
        L.append(f"\\quad {nm(a)} $-$ {nm(b)} & {cell(v['pooled'])} & {cell(g['pooled']) if g else '---'}\\\\")
    for fam, title in (("vs_w", r"In hours: truncated head minus head trained on $w$ (Holm over 14)"),
                       ("vs_restricted", r"In hours: truncated head minus restricted conventional head (Holm over 12)")):
        L.append(r"\addlinespace")
        L.append(r"\multicolumn{3}{@{}l}{\emph{" + title + r"}}\\")
        for k, v in hc["pairs"].items():
            if v["family"] != fam:
                continue
            a, b = k.split(" - ")
            L.append(f"\\quad {nm(a)} $-$ {nm(b)} & {cell(v['pooled'])} & {cell(hg['pairs'][k]['pooled'])}\\\\")
    L += [r"\bottomrule", r"\end{tabular}", r"\end{adjustbox}", r"\end{table}"]
    w("tab_ratio_pairs", L)


def quantile_table():
    """Truncated heads against the two quantile heads, per benchmark and pooled,
    at the reported and at the fixed-time prediction points."""
    path = os.path.join(RES, "quantile.json")
    if not os.path.exists(path):
        return
    q = json.load(open(path))
    ci = lambda x: f"${x['margin']:+.3f}$ [${x['ci95'][0]:+.3f}$, ${x['ci95'][1]:+.3f}$]"
    NMQ = {"BS-trunc": r"BS$_{\rm tr}$", "IGauss-trunc": r"IG$_{\rm tr}$",
           "Quantile": "QR", "Quantile-logit": r"QR$_{(0,1)}$"}
    L = [r"\begin{table}[htbp]", r"\centering\footnotesize",
         r"\caption{CRPS margins against the quantile-regression (QR) heads: QR unconstrained, QR$_{(0,1)}$ kept in"
         r" $(0,1)$ by a logit link. Holm over the four truncated-minus-quantile margins of each"
         r" protocol; the last row is a single comparison.}",
         r"\label{tab:quantile}",
         r"\begin{adjustbox}{max width=\linewidth}",
         r"\begin{tabular}{@{}l llll@{}}", r"\toprule",
         r"comparison & PRONOSTIA & XJTU-SY & pooled, Holm $p$ & fixed-time pooled, Holm $p$\\", r"\midrule"]
    for k, v in q["main"]["pairs"].items():
        a, b = k.split(" - ")
        if v["family"] != "truncated_vs_quantile":
            L.append(r"\addlinespace")
        g = q.get("grid", {}).get("pairs", {}).get(k)
        L.append(f"{NMQ[a]} $-$ {NMQ[b]} & {ci(v['pronostia'])} & {ci(v['xjtu'])}"
                 f" & {ci(v['pooled'])} ({pfmt(v['pooled']['p_holm'])})"
                 f" & {(ci(g['pooled']) + ' (' + pfmt(g['pooled']['p_holm']) + ')') if g else '---'}\\\\")
    L += [r"\bottomrule", r"\end{tabular}", r"\end{adjustbox}", r"\end{table}"]
    w("tab_quantile", L)


def hours_table():
    """Absolute error of the predictive median in hours of remaining life."""
    path = os.path.join(RES, "hours.json")
    if not os.path.exists(path):
        return
    h = json.load(open(path))
    NMH = {"BS": "BS, restricted", "BS-t": "BS-$t$, restricted", "Weibull": "Weibull, restricted",
           "LogNormal": "lognormal, restricted", "IGauss": "IG, restricted", "Gaussian": "Gaussian, restricted",
           "LogitNormal": "logit-normal", "Beta": "Beta", "BS-trunc": "BS, truncated at one",
           "IGauss-trunc": "IG, truncated at one"}
    L = [r"\begin{table}[htbp]", r"\centering\small",
         r"\caption{Mean absolute error of the predictive median in hours, after mapping each law"
         r" to remaining life by $R=t_k\,y/(1-y)$. Coverage and PIT are unchanged by the map.}",
         r"\label{tab:hours}",
         r"\begin{tabular}{@{}l rr@{}}", r"\toprule",
         r"head & PRONOSTIA & XJTU-SY\\", r"\midrule"]
    for k in ("BS", "BS-t", "Weibull", "LogNormal", "IGauss", "Gaussian", "LogitNormal", "Beta",
              "BS-trunc", "IGauss-trunc"):
        L.append(f"{NMH[k]} & {h['heads']['pronostia/' + k]['MAE_h']:.2f}"
                 f" & {h['heads']['xjtu/' + k]['MAE_h']:.2f}\\\\")
        if k == "Gaussian":
            L.append(r"\midrule")
    L += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    w("tab_hours", L)


if __name__ == "__main__":
    main()
