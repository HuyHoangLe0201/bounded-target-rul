"""LaTeX tables for the paper-2 review analyses (results/regression/review_p2*.json).

  tab_support      CRPS, MAE and coverage before and after restricting each
                   predictive law to the target's support (0,1]
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
         r" $(0,1]$, applied after fitting. Outside: mean predictive mass outside $(0,1]$.}",
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
         r"\caption{Heads trained on a likelihood that respects the target's support $(0,1]$,"
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
    L += [r"\bottomrule", r"\end{tabular}", r"\end{adjustbox}", r"\end{table}"]
    w("tab_bounded", L)
    margins_table(b)
    robustness_table()
    hours_table()


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
    """Paired CRPS margins per benchmark and pooled, for every comparison claimed."""
    ci = lambda x: f"${x['margin']:+.3f}$ [${x['ci95'][0]:+.3f}$, ${x['ci95'][1]:+.3f}$]"
    L = [r"\begin{table}[htbp]", r"\centering\footnotesize",
         r"\caption{Paired CRPS margins, row head minus rival, with $95\,\%$ bootstrap intervals over"
         r" bearings. Pooled: the $26$ bearings weighted equally, with the Holm-adjusted $p$."
         r" Negative favours the row head.}",
         r"\label{tab:margins}",
         r"\begin{adjustbox}{max width=\linewidth}",
         r"\begin{tabular}{@{}l l l l@{}}", r"\toprule",
         r"comparison & PRONOSTIA ($11$) & XJTU-SY ($15$) & pooled ($26$), Holm $p$\\", r"\midrule"]
    for i, (a, r_) in enumerate(PAIR_ORDER):
        if i in BLOCK:
            if i:
                L.append(r"\addlinespace")
            L.append(r"\multicolumn{4}{@{}l}{" + BLOCK[i] + r"}\\")
        v = b["pairs"][f"{a} - {r_}"]
        L.append(f"\\quad {SH[a]} $-$ {SH[r_]} & {ci(v['pronostia'])} & {ci(v['xjtu'])}"
                 f" & {ci(v['pooled'])} ({pfmt(v['pooled']['p_holm'])})\\\\")
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
