"""Summaries of the extended simulation campaigns (sim_extended.py).

Usage:
    python summarize_extended.py RESULTS_DIR

Writes to RESULTS_DIR/summaries:
    sim_val_se.csv   bootstrap SE over empirical SD and coverage at nominal 80% and 90%, per contrast
    sim_val.csv      calibration at R = 6, B = 60 per design and size (zero-target rejections, coverage, width)
    sim_direct.csv   corrected versus direct refit bootstrap on the same datasets, per cell and contrast
    sim_big.csv      calibration at 322 participants x 4 trials (no signal, signal elsewhere)
    sim_ext.csv      new designs: target (with its Monte Carlo SE), coverage, rejection rate, width, observed AUC
and prints the headline numbers.
"""
from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd

from summarize_simulation import load, upper_bound, zero_by_argument

KEYS = ["design", "size", "descriptor", "component"]
PLANTED_ORDER = ("orderw", "negorder")
NOMINAL = {80: 1.2816, 90: 1.6449}     # normal quantiles of the lower nominal levels
N_VAL = 40                             # datasets per cell of the calibration summary


def targets(ref: pd.DataFrame) -> pd.DataFrame:
    """Reference value of every contrast with its Monte Carlo standard error.

    Args:
        ref: Reference-run rows (point estimates).

    Returns:
        One row per design x size x descriptor x contrast.
    """
    return ref.groupby(KEYS).agg(theta=("contribution", "mean"),
                                 theta_se=("contribution", lambda v: v.std(ddof=1) / np.sqrt(len(v))),
                                 n_ref=("contribution", "size")).reset_index()


def assess(C: pd.DataFrame, th: pd.DataFrame) -> pd.DataFrame:
    """Attach targets and per-interval coverage.

    Args:
        C: Interval rows.
        th: Targets.

    Returns:
        The rows of genuine intervals, with target, coverage and width.
    """
    C = C[~C.exact_invariant.astype(bool)].merge(th, on=KEYS, how="left")
    # order carries the planted signal in the two order designs, so its target is zero only elsewhere
    C["zero_arg"] = [zero_by_argument(d, c) and d not in PLANTED_ORDER for d, c in zip(C.design, C.component)]
    C["target"] = np.where(C.zero_arg, 0.0, C.theta)
    C["covers"] = (C.ci_lo <= C.target) & (C.target <= C.ci_hi)
    C["width"] = C.ci_hi - C.ci_lo
    for level, z in NOMINAL.items():
        C[f"covers{level}"] = (C.contribution - z * C.se <= C.target) & (C.target <= C.contribution + z * C.se)
    return C


def se_calibration(C: pd.DataFrame, by: list) -> pd.DataFrame:
    """Bootstrap standard error against the spread of the estimate, and coverage at lower nominal levels.

    Args:
        C: Assessed interval rows.
        by: Grouping keys (a group holds the datasets of one contrast).

    Returns:
        Per group: mean bootstrap SE, empirical SD of the estimate across datasets, their ratio, and the coverage
        of the normal interval at nominal 80%, 90% and 95%.
    """
    G = C.groupby(by).agg(n=("contribution", "size"), se_boot=("se", "mean"), sd_emp=("contribution", "std"),
                          cov80=("covers80", "mean"), cov90=("covers90", "mean"),
                          cov95=("covers", "mean")).reset_index()
    G["se_ratio"] = G.se_boot / G.sd_emp
    return G


def per_cell(C: pd.DataFrame, by: list) -> pd.DataFrame:
    """Aggregate coverage and rejection by the given keys.

    Args:
        C: Assessed interval rows.
        by: Grouping keys.

    Returns:
        Summary with counts of intervals and of datasets.
    """
    z = C[C.zero_arg]
    a = C.groupby(by).agg(n_intervals=("covers", "size"), n_datasets=("dataset", "nunique"),
                          covered=("covers", "sum"), width=("width", "mean"), auc_obs=("auc_obs", "mean"))
    zz = z.groupby(by).agg(n_zero=("excludes0", "size"), rej_zero=("excludes0", "sum"))
    out = a.join(zz, how="left").fillna({"n_zero": 0, "rej_zero": 0}).reset_index()
    out["coverage"] = out.covered / out.n_intervals
    return out


def main() -> None:
    """Build and print every summary."""
    res = sys.argv[1]
    sim = os.path.join(res, "simulation")
    out = os.path.join(res, "summaries")
    os.makedirs(out, exist_ok=True)

    # validation at the settings of the reported profiles; targets from the R = 20 reference runs. The classic
    # descriptor's duration-marginal target was estimated with the dataset-wide donor pool, so it is not compared.
    th = targets(load(os.path.join(sim, "ref")))
    Vall = assess(load(os.path.join(sim, "val")), th)
    Vall = Vall[~((Vall.descriptor == "classic") & (Vall.component == "duration_marginal"))]
    V = Vall[Vall.dataset < N_VAL]           # the direct-versus-corrected cells have more datasets
    Vs = per_cell(V, ["design", "size"])
    Vs.to_csv(os.path.join(out, "sim_val.csv"), index=False)
    zr = V[V.zero_arg]
    print(f"VAL: {len(V)} intervals from {V.groupby(['design', 'size']).dataset.nunique().sum()} datasets; "
          f"zero-target {int(zr.excludes0.sum())}/{len(zr)} excluded zero; coverage {int(V.covers.sum())}/{len(V)} "
          f"({V.covers.mean():.3f}); max target SE {th.theta_se.max():.4f}")
    print(Vs.round(3).to_string(index=False))
    worst = zr.groupby(KEYS).excludes0.sum().max()
    print(f"worst zero-target contrast: {int(worst)} of 40 (upper bound {upper_bound(int(worst), 40):.3f})")
    j = th.merge(V[["design", "size"]].drop_duplicates(), on=["design", "size"])
    print("target SE (cells in val): median", round(j.theta_se.median(), 4), "max", round(j.theta_se.max(), 4))

    Sv = se_calibration(V, KEYS)
    Sv.to_csv(os.path.join(out, "sim_val_se.csv"), index=False)
    print(f"SE ratio (bootstrap SE / empirical SD) over {len(Sv)} contrasts: median {Sv.se_ratio.median():.2f}, "
          f"range {Sv.se_ratio.min():.2f}-{Sv.se_ratio.max():.2f}; coverage at nominal 80% {V.covers80.mean():.3f}, "
          f"90% {V.covers90.mean():.3f}, 95% {V.covers.mean():.3f}")

    # direct versus corrected on the same datasets
    D = assess(load(os.path.join(sim, "direct")), th)
    D = D[~((D.descriptor == "classic") & (D.component == "duration_marginal"))]
    M = D.merge(Vall, on=KEYS + ["dataset"], suffixes=("_direct", "_corr"))
    Ds = M.groupby(KEYS).agg(n=("covers_direct", "size"), cov_direct=("covers_direct", "mean"),
                             cov_corr=("covers_corr", "mean"), w_direct=("width_direct", "mean"),
                             w_corr=("width_corr", "mean"), rej_direct=("excludes0_direct", "mean"),
                             rej_corr=("excludes0_corr", "mean"), zero=("zero_arg_corr", "first")).reset_index()
    Ds["width_ratio"] = Ds.w_corr / Ds.w_direct
    for tag in ("direct", "corr"):
        T = M.rename(columns={f"contribution_{tag}": "contribution", f"se_{tag}": "se", f"covers_{tag}": "covers",
                              f"covers80_{tag}": "covers80", f"covers90_{tag}": "covers90"})
        S = se_calibration(T, KEYS)[KEYS + ["se_ratio", "cov80", "cov90"]]
        Ds = Ds.merge(S.rename(columns={c: f"{c}_{tag}" for c in ("se_ratio", "cov80", "cov90")}), on=KEYS)
    Ds.to_csv(os.path.join(out, "sim_direct.csv"), index=False)
    print(f"DIRECT: {len(M)} paired intervals; coverage direct {M.covers_direct.mean():.3f} vs corrected "
          f"{M.covers_corr.mean():.3f}; mean width {M.width_direct.mean():.3f} vs {M.width_corr.mean():.3f}; "
          f"zero-target rejections direct {int(M[M.zero_arg_corr].excludes0_direct.sum())} vs corrected "
          f"{int(M[M.zero_arg_corr].excludes0_corr.sum())} of {int(M.zero_arg_corr.sum())}")
    print(Ds.round(3).to_string(index=False))

    # new designs against their own references
    the = targets(load(os.path.join(sim, "extref")))
    E = assess(load(os.path.join(sim, "ext")), the)
    Es = E.groupby(KEYS).agg(theta=("theta", "first"), theta_se=("theta_se", "first"), zero=("zero_arg", "first"),
                             n=("covers", "size"), coverage=("covers", "mean"), excl=("excludes0", "mean"),
                             width=("width", "mean"), estimate=("contribution", "mean"),
                             auc_obs=("auc_obs", "mean")).reset_index()
    Es = Es.merge(se_calibration(E, KEYS)[KEYS + ["se_ratio", "cov80", "cov90"]], on=KEYS)
    Es.to_csv(os.path.join(out, "sim_ext.csv"), index=False)

    # calibration at the largest analysed sample size (322 participants x 4 trials)
    if os.path.isdir(os.path.join(sim, "big")):
        G = assess(load(os.path.join(sim, "big")), targets(load(os.path.join(sim, "bigref"))))
        G = G[~((G.descriptor == "classic") & (G.component == "duration_marginal"))]
        Gs = per_cell(G, KEYS).merge(se_calibration(G, KEYS)[KEYS + ["se_ratio", "cov80", "cov90"]], on=KEYS)
        Gs.to_csv(os.path.join(out, "sim_big.csv"), index=False)
        print(f"BIG (322 x 4): {G.dataset.nunique()} datasets per cell, {len(G)} intervals; zero-target excluded "
              f"{int(Gs.rej_zero.sum())}/{int(Gs.n_zero.sum())}; coverage {int(G.covers.sum())}/{len(G)} = "
              f"{G.covers.mean():.3f}; "
              f"lowest per contrast {Gs.coverage.min():.3f}; SE ratio median {Gs.se_ratio.median():.2f} "
              f"({Gs.se_ratio.min():.2f}-{Gs.se_ratio.max():.2f}); coverage at nominal 80% {G.covers80.mean():.3f}, "
              f"90% {G.covers90.mean():.3f}; mean width by design "
              f"{G.groupby('design').width.mean().round(3).to_dict()}; mean observed AUC "
              f"{G.groupby('design').auc_observed.mean().round(3).to_dict()}")
    ze = E[E.zero_arg]
    print(f"EXT: {len(E)} intervals from {E.groupby(['design', 'size', 'descriptor']).dataset.nunique().sum()} "
          f"datasets; coverage {int(E.covers.sum())}/{len(E)} ({E.covers.mean():.3f}); zero-target "
          f"{int(ze.excludes0.sum())}/{len(ze)}; lowest per-contrast coverage {Es.coverage.min():.3f}")
    pd.set_option("display.width", 220)
    print(Es.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
