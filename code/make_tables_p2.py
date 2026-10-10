"""Main regression tables of paper 2 (tab_regression.tex)."""
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
RG = os.path.join(HERE, "..", "results", "regression")
TAB = os.path.join(HERE, "..", "results", "tables")


def jload(p):
    with open(p) as f:
        return json.load(f)


RHEAD = ["BS", "BS-t", "Weibull", "LogNormal", "IGauss", "Gaussian", "Point(MAE)"]


RNAME = {"BS": "BS", "BS-t": "BS-$t$", "Weibull": "Weibull",
         "LogNormal": "lognormal", "IGauss": "inverse Gaussian",
         "Gaussian": "Gaussian", "Point(MAE)": "point head (MAE loss)"}


RSUB = {"pronostia/all": "PRONOSTIA, all prediction points",
        "pronostia/official": "PRONOSTIA, official IEEE PHM 2012 truncation points",
        "xjtu/all": "XJTU-SY, leave-one-operating-condition-out"}


def _fmt(v, k):
    x = v.get(k, float("nan"))
    if x != x:
        return "---"
    if k == "n":
        return f"{x:.0f}"
    if k in ("MAE", "RMSE", "CRPS", "MPIW", "MPIW_conf"):
        return f"{x:.4f}"
    if k in ("MedAPE_pct", "mean_absEr_pct"):
        return f"{x:.1f}"
    return f"{x:.3f}"


def t_regression():
    d = jload(os.path.join(RG, "results_both.json"))["tables"]
    # n and the epistemic ratio are dropped from the table (n goes in the
    # caption, rho is not referenced in the text) so that eleven columns fit the
    # text width instead of overflowing it by 97 pt.
    cols = ["MAE", "RMSE", "mean_absEr_pct", "PHM_score", "CRPS",
            "cov95", "MPIW", "PIT_KS"]
    hdr = (r"head & MAE & RMSE & $|\mathrm{Er}|$ [\%] & PHM & CRPS "
           r"& cov$_{95}$ & MPIW & PIT-KS\\")
    out = ""
    for key, label in RSUB.items():
        if key not in d or not d[key]:
            continue
        body = ""
        for h in RHEAD:
            if h not in d[key]:
                continue
            v = d[key][h]
            body += RNAME[h] + " & " + " & ".join(_fmt(v, c) for c in cols) + "\\\\\n"
        tag = key.replace("/", "_")
        n = int(next(iter(d[key].values()))["n"])
        out += f"""\\begin{{table}}[htbp]
\\centering\\footnotesize
\\caption{{{label}; $n={n}$ prediction points. All heads share one backbone,
feature set, schedule and set of seeds. Mean absolute error (MAE), root mean squared
error (RMSE), continuous ranked probability score (CRPS) and mean prediction
interval width (MPIW) are on the normalised
scale (fraction of life); $|\\mathrm{{Er}}|$ and the prognostics and health
management (PHM) challenge score are scale-invariant
and therefore comparable with the IEEE PHM 2012 challenge. cov$_{{95}}$ is the
empirical coverage of the nominal 95\\,\\% interval; PIT-KS is the Kolmogorov--Smirnov
distance of the probability integral transform from uniformity (0 is perfect).}}
\\label{{tab:reg_{tag}}}
\\begin{{tabular}}{{@{{}}l rr rr rr rr r@{{}}}}
\\toprule
{hdr}
\\midrule
{body}\\bottomrule
\\end{{tabular}}
\\end{{table}}

"""
    return out


if __name__ == "__main__":
    open(os.path.join(TAB, "tab_regression.tex"), "w", encoding="utf-8").write(t_regression())
    import table_uncertainty
    open(os.path.join(TAB, "tab_uncertainty.tex"), "w", encoding="utf-8").write(table_uncertainty.build())
    print("wrote tab_regression.tex, tab_uncertainty.tex")
