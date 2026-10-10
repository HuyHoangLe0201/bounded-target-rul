# bounded-target-rul

Code and stored results for

> H. H. Le and K.-A. Nguyen, *Bounded Target, Truncated Likelihood: A Controlled
> Comparison of Predictive Distributions for Bearing Remaining Useful Life*
> (submitted).

Ten likelihood heads predict the remaining fraction of a bearing's life,
`R_k / T_total ∈ (0,1]`, on one small MLP backbone with one causal feature set,
one training schedule and one set of seeds:

| group | heads |
|---|---|
| conventional | Birnbaum–Saunders, BS-Student-*t* (ν = 6), Weibull, lognormal, inverse Gaussian, Gaussian |
| respect the bound | Beta, logit-normal, BS truncated at one, IG truncated at one |
| baseline | point head trained by mean absolute error |

Each head is a deep ensemble of five members. The heads are scored on PRONOSTIA
(IEEE PHM 2012) and XJTU-SY with CRPS, interval coverage and width, PIT
calibration, and a cluster bootstrap over held-out bearings.

## Data

The bearing archives are third-party data and are **not** redistributed here,
neither raw nor as derived features. Download them from their publishers:

* PRONOSTIA / IEEE PHM 2012 Prognostic Challenge: the folder with `Learning_set/`
  and `Full_Test_Set/`.
* XJTU-SY: the folder with `35Hz12kN/`, `37.5Hz11kN/` and `40Hz10kN/`.

Then extract the features (written to `results/features*/`):

```bash
python code/extract_features.py /path/to/phm-ieee-2012-data-challenge-dataset-master
python code/extract_xjtu.py /path/to/XJTU-SY_Bearing_Datasets
```

(`PRONOSTIA_DIR` and `XJTU_DIR` environment variables work as well.)

## What is stored

`results/regression/` holds every prediction the paper is built from, so the
tables can be checked without retraining:

* `results_both.json`: per-inspection predictions of the six conventional heads
  and the point head (101 predictive quantiles each), with the summary tables.
* `results_both_causal.json`: the same run including the four bounded heads.
* `results_both_{w32,w48,w96,w256,big}.json`: the backbone sweep.
* `results_both_{grid,seeds5,lr1e-3,lr1e-2}.json`: the robustness runs (prediction
  points fixed in operating time; seeds 5–9; learning rates 1e-3 and 1e-2).
* the analysis outputs (`uncertainty`, `joint`, `bounded`, `review_p2`,
  `review_p2_trunc`, `crps_grid`, `head_shape`, `rho`, `weibull_members`,
  `backbone`, `hours`, `robustness`).

## Reproducing

Tables and figures from the stored results (minutes, no data needed):

```bash
cd code
python make_tables_p2.py        # tab_regression, tab_uncertainty
python table_review_p2.py       # tab_support, tab_bounded, tab_margins, tab_robust, tab_hours,
                                # tab_calib_ci, tab_points, tab_eropt, tab_b14
python table_backbone.py
python table_headshape.py
python make_figures_p2.py       # fig_calibration, fig_pit
python head_diagnostics.py      # matched-quantile tail comparison
```

The full pipeline (needs the extracted features; about 25 minutes on a laptop
CPU for the main run, about 90 minutes for the sweep):

```bash
cd code
python check_causal_features.py                                  # features at k use acquisitions 1..k only
GBS_BOUNDED=1 GBS_TAG=_causal python study_regression.py both    # all ten heads + point head
#  results_both.json is the six conventional heads and the point head of that run
python study_uncertainty.py; python study_joint.py; python study_rho.py
python study_review_p2.py; python study_review_p2_trunc.py; python study_crps_grid.py
python study_bounded.py; python study_weibull_members.py; python study_hours.py
python run_backbone_sweep.py; python study_backbone.py
#  robustness: one element of the protocol changed at a time
GBS_BOUNDED=1 GBS_EVAL=grid GBS_TAG=_grid python study_regression.py both
GBS_BOUNDED=1 GBS_SEED0=5 GBS_TAG=_seeds5 python study_regression.py both
GBS_BOUNDED=1 GBS_LR=1e-3 GBS_TAG=_lr1e-3 python study_regression.py both
GBS_BOUNDED=1 GBS_LR=1e-2 GBS_TAG=_lr1e-2 python study_regression.py both
python study_robustness.py
```

The target `y = R/T_total` is an exact reparametrisation of the RUL at a known
elapsed time `t`: `R = t*y/(1-y)`. `study_hours.py` maps every predictive law to
hours; coverage and PIT are unchanged by the map, the CRPS is not (it is infinite
in hours for any law with positive density at `y = 1`).

Seeds are fixed (ensemble members use seeds 0–4); on CPU the refit in
`study_weibull_members.py` reproduces the stored predictive medians exactly.

## Protocol in brief

* Splits: PRONOSTIA official (6 learning / 11 test bearings, Bearing2_2 held out
  for early stopping); XJTU-SY leave-one-operating-condition-out.
* Prediction points: 40, 50, …, 90 % of each held-out bearing's life, plus the
  official truncation point on PRONOSTIA.
* Features: robust z-scores against a fixed healthy window (40 / 10
  acquisitions), with trailing smoothing, differences, slopes and an alarm
  fraction; training points start after the healthy window.
* Truncated heads maximise `log f(y) − log F(1)`.

## Licence

MIT, see `LICENSE`. The licence covers the code and the stored results, not the
PRONOSTIA or XJTU-SY data.
