# scanpath-perturbation-profiles

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.24132704.svg)](https://doi.org/10.5281/zenodo.24132704)

Code, package and aggregate results for "Characterizing Scanpath Decoding Pipelines with Perturbation Profiles"
(Mozaffari & Nadi). A perturbation profile reports how much of a scanpath descriptor's participant-held-out
discrimination is lost when one property of every scanpath (order, duration coupling, duration distribution,
orientation, fine spatial structure) is destroyed and the classifier is refit.

All data are public (GazeBase, SB-SAT, EMIP) and are fetched from their original sources; no raw recording is
redistributed here.

## Layout

| Path | What it is |
|---|---|
| `pkg/` | the `scanpathdecomp` package (transformations, refit bootstrap, k-fold evaluation, within-fold AUC, descriptors) and its tests |
| `analysis/open_data.py` | loaders, the thirteen binary problems and their permutation schemes |
| `analysis/run_profiles.py`, `analysis/run_all.py` | one profile (problem x descriptor); every profile, in parallel and resumable |
| `analysis/observed_auc_ci.py` | observed AUC with a refit-bootstrap interval and a permutation distribution |
| `analysis/select_cells.py`, `analysis/evidence_ledger.py` | the selection rule and the ledger of checks for the interpreted cells |
| `analysis/controls.py` | fold partitions, same-program AUC, synthetic-gaze control, per-fold AUCs |
| `analysis/run_convergence.py` | longer runs and the second learner |
| `analysis/support_retention.py` | support retention, support-only descriptor, support-matched transformations |
| `analysis/contrast_null.py`, `analysis/fold_donor_check.py` | three permutation distributions for negative sensitivities; training-fold donor pools |
| `analysis/gazebase_extract.py` | I-VT fixation extraction from the public GazeBase samples |
| `analysis/sim_design.py`, `analysis/sim_calibration.py`, `analysis/sim_extended.py` | generative model and the simulation campaigns |
| `analysis/summarize*.py` | summaries of profiles and simulations |
| `data/raw/` | SB-SAT and EMIP files as downloaded (not committed) |
| `data/derived/gazebase_r1_all_fixations.csv` | fixations of all 322 participants of GazeBase Round 1, derived with `gazebase_extract.py` |
| `results/profiles_v5/`, `results/profiles_v5_pooled/` | the 116 profiles (within-fold AUC with class-balanced folds; pooled AUC) |
| `results/observed_auc_v5/`, `results/observed_auc_v5_pooled/` | observed AUCs with permutation distributions |
| `results/convergence_v5/`, `results/convergence_v5_t7/` | longer runs and boosted trees for the interpreted cells |
| `results/support_v5/` | support retention and the support controls |
| `results/simulation_fold/`, `results/simulation/` | simulation campaigns (within-fold AUC; the earlier pooled-AUC campaign) |
| `results/twobytwo/` | permutation means by metric and fold assignment |
| `results/summaries/` | evidence ledger, controls and simulation summaries |
| `manuscript/` | `main.tex`, `supplement.tex`, figures, and the scripts that write the tables and figures |

## Citation

The version used for the article is archived on Zenodo:

> Mozaffari, S., & Nadi, A. (2026). *Perturbation profiles for scanpath decoding pipelines: code, package and results*
> (Version 0.6.3) [Computer software]. Zenodo. https://doi.org/10.5281/zenodo.24132704

`CITATION.cff` holds the same reference for GitHub's "Cite this repository" button.

## Licences

The code is released under the MIT licence (`LICENSE`). The derived GazeBase table is a derivative of GazeBase v2.0
(Griffith, Lohr, Abdulin & Komogortsev, 2021, *Scientific Data*, 8, 184), which is distributed under CC BY 4.0;
the derived table is shared under the same licence. EMIP (OSF, osf.io/53kts) is also
distributed under CC BY 4.0. SB-SAT (github.com/ahnchive/SB-SAT) and the corrected EMIP release
(github.com/nalmadi/EMIP-Toolkit) state no licence; neither is redistributed here, and their authors ask that
Ahn et al. (2020) and Al Madi et al. (2021) be cited when they are used.

## Reproduce

Quick check from a fresh clone (GazeBase only; the derived fixation table is in `data/derived/`; about one minute):

```
pip install -r requirements.txt && pip install -e pkg[fag]
cd analysis
python run_profiles.py gb_vd1_blg classic 6 60 42 /tmp/check
# /tmp/check/prof_gb_vd1_blg_classic_R6b60_s42.csv, row "order":
# auc_observed 0.886, contribution -0.114, interval [-0.137, -0.091]  (Table 7 and Table S2 of the article)
```

Full analysis (within-fold AUC and class-balanced folds are the defaults of the scripts):

```
# data: SB-SAT from github.com/ahnchive/SB-SAT (fixation/18sat_fixfinal.csv, 18sat_labels.csv),
#       EMIP corrected release from github.com/nalmadi/EMIP-Toolkit ("Corrected EMIP Dataset"),
#       EMIP metadata (emip_metadata.csv) from osf.io/53kts  -> data/raw/
#       GazeBase v2.0 (figshare 12912257) only if the derived table is to be rebuilt:
#       python gazebase_extract.py GAZEBASE_DIR ../data/derived/gazebase_r1_all_fixations.csv 0 1 4
cd analysis
export SPO_CACHE_DIR=../results/cache_v5
SPO_OUT=../results/profiles_v5 python run_all.py 4 6 60 42 gb em sb
SPO_METRIC=pooled SPO_OUT=../results/profiles_v5_pooled python run_all.py 4 6 60 42 gb em sb
SPO_OAUC_OUT=../results/observed_auc_v5 python observed_auc_ci.py 4 200 1000 gb em sb
SPO_METRIC=pooled SPO_OAUC_OUT=../results/observed_auc_v5_pooled python observed_auc_ci.py 4 0 200 gb em sb
python select_cells.py ../results      # selection rule -> results/summaries/interpreted_cells.csv; then the checks
python controls.py partitions 20 6
python controls.py stratum 200 1000; python controls.py synthetic 200; python controls.py folds
SPO_CONV_OUT=../results/convergence_v5 python run_convergence.py 4 svm 20 300 42 43 44
SPO_CONV_OUT=../results/convergence_v5 python run_convergence.py 4 xgb 6 60 42
SPO_SUPPORT_OUT=../results/support_v5 python support_retention.py tables
SPO_SUPPORT_OUT=../results/support_v5 python support_retention.py matched em_halves
SPO_SUPPORT_OUT=../results/support_v5 python support_retention.py support em_halves
python contrast_null.py 1000 6 42; python fold_donor_check.py 6 42
python evidence_ledger.py ../results
# simulations
SPO_SIM_OUT=../results/simulation_fold python sim_calibration.py 4 ref
SPO_SIM_OUT=../results/simulation_fold python sim_extended.py 4 val direct ext extref big bigref
SPO_METRIC=pooled SPO_STRATIFY=0 SPO_SIM_OUT=../results/simulation python sim_calibration.py 4 rec
# tables, figure and manuscript
cd ../manuscript
python tools/make_tables.py ../results/profiles_v5 . 6 60 42
SPO_TAB_SUFFIX=_pooled python tools/make_tables.py ../results/profiles_v5_pooled . 6 60 42
python tools/make_ledger_tables.py ../results . && python tools/make_fig_profiles.py ../results figs/fig_profiles.pdf
latexmk -pdf main.tex && latexmk -pdf supplement.tex && python tools/make_brm.py .
```
