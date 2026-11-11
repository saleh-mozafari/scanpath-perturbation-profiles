"""Summaries of the simulation campaign, the convergence runs and the learner comparison.

Usage:
    python summarize_simulation.py RESULTS_DIR OUT_DIR

RESULTS_DIR holds simulation/{rec,cal,ref}, convergence/{svm,xgb}, profiles_v3, observed_auc_v3 and
observed_auc_xgb. Writes CSV summaries to OUT_DIR and prints the headline numbers.
"""
from __future__ import annotations

import glob
import os
import sys

import numpy as np
import pandas as pd
from scipy.stats import beta

KEYS = ["design", "size", "descriptor", "component"]


def upper_bound(k: int, n: int, level: float = 0.95) -> float:
    """Exact one-sided upper confidence bound on a binomial rate (Clopper-Pearson).

    Args:
        k: Events.
        n: Trials.
        level: Confidence level.

    Returns:
        The upper bound.
    """
    return 1.0 if k >= n else float(beta.ppf(level, k + 1, n - k))


def zero_by_argument(design: str, component: str) -> bool:
    """Whether a contrast's target is zero by an exchangeability argument.

    Args:
        design: Generative design.
        component: Transformation.

    Returns:
        True for order in every design and rotation in the design without the participant offset.
    """
    return component == "order" or (component == "rotation" and design.endswith("n") and design != "none")


def load(d: str) -> pd.DataFrame:
    """Concatenate every CSV in a directory.

    Args:
        d: Directory.

    Returns:
        The concatenated table.
    """
    files = sorted(glob.glob(os.path.join(d, "*.csv")))
    return pd.concat([pd.read_csv(f) for f in files], ignore_index=True)


def calibration(res: str) -> pd.DataFrame:
    """Per-contrast rejection rates and coverage of the estimated target.

    Args:
        res: Results directory.

    Returns:
        One row per design x size x descriptor x contrast.
    """
    C = load(os.path.join(res, "simulation", "cal"))
    R = load(os.path.join(res, "simulation", "ref"))
    C = C[~C.exact_invariant.astype(bool)]
    th = R.groupby(KEYS).agg(theta=("contribution", "mean"), theta_se=("contribution", lambda v: v.std(ddof=1) / np.sqrt(len(v))),
                             n_ref=("contribution", "size")).reset_index()
    C = C.merge(th, on=KEYS, how="left")
    C["zero_arg"] = [zero_by_argument(d, c) for d, c in zip(C.design, C.component)]
    C["target"] = np.where(C.zero_arg, 0.0, C.theta)
    C["covers"] = (C.ci_lo <= C.target) & (C.target <= C.ci_hi)
    C["width"] = C.ci_hi - C.ci_lo
    g = C.groupby(KEYS + ["zero_arg"])
    S = g.agg(n=("excludes0", "size"), rejections=("excludes0", "sum"), coverage=("covers", "mean"),
              theta=("theta", "first"), theta_se=("theta_se", "first"), width=("width", "mean"),
              auc=("auc_obs", "mean"), fallback=("variance_fallback", "sum"), degenerate=("degenerate", "sum")).reset_index()
    S["rate"] = S.rejections / S.n
    S["upper95"] = [upper_bound(int(k), int(n)) for k, n in zip(S.rejections, S.n)]
    return S


def pooled_rates(res: str) -> pd.DataFrame:
    """Pooled rejection rate over contrasts with target zero by argument, per design and size.

    The dataset is the unit: the per-dataset fraction of rejected contrasts is averaged over datasets, with its
    standard error.

    Args:
        res: Results directory.

    Returns:
        One row per design x size.
    """
    C = load(os.path.join(res, "simulation", "cal"))
    C = C[~C.exact_invariant.astype(bool)]
    C = C[[zero_by_argument(d, c) for d, c in zip(C.design, C.component)]]
    q = C.groupby(["design", "size", "dataset"]).excludes0.mean().reset_index()
    return q.groupby(["design", "size"]).excludes0.agg(rate="mean", se=lambda v: v.std(ddof=1) / np.sqrt(len(v)),
                                                     datasets="size").reset_index()


def recovery(res: str) -> pd.DataFrame:
    """Mean sensitivity and the fraction of datasets whose interval excludes zero, per scenario.

    Args:
        res: Results directory.

    Returns:
        One row per scenario x descriptor x component.
    """
    R = load(os.path.join(res, "simulation", "rec"))
    return R.groupby(["design", "descriptor", "component"]).agg(
        auc=("auc_obs", "mean"), contrib=("contribution", "mean"), sd=("contribution", "std"),
        frac_excl=("excludes0", "mean"), exact=("exact_invariant", "all"), n=("dataset", "size")).reset_index()


def convergence(res: str) -> pd.DataFrame:
    """Interpreted cells: the main run (R = 6, B = 60) against R = 20, B = 300 under three seeds.

    Args:
        res: Results directory.

    Returns:
        One row per cell x run.
    """
    M = load(os.path.join(res, "profiles_v3"))
    V = load(os.path.join(res, "convergence", "svm"))
    keep = V[["problem", "descriptor", "component"]].drop_duplicates()
    M = M.merge(keep, on=["problem", "descriptor", "component"]).assign(run="R6b60_s42")
    V = V.assign(run=lambda d: "R20b300_s" + d.seed.astype(str))
    A = pd.concat([M, V], ignore_index=True)
    A["excl"] = (A.ci_lo > 0) | (A.ci_hi < 0)
    return A[["problem", "descriptor", "component", "run", "auc_observed", "contribution", "ci_lo", "ci_hi", "excl"]]


def learners(res: str) -> tuple:
    """Observed AUCs under both learners, and interpreted-cell sensitivities under both.

    Args:
        res: Results directory.

    Returns:
        (observed-AUC table, sensitivity table).
    """
    S = load(os.path.join(res, "observed_auc_v3"))[["problem", "descriptor", "auc", "p_perm"]]
    X = load(os.path.join(res, "observed_auc_xgb"))[["problem", "descriptor", "auc", "p_perm"]]
    O = S.merge(X, on=["problem", "descriptor"], suffixes=("_svm", "_xgb"))
    V = load(os.path.join(res, "convergence", "svm"))
    V = V[V.seed == 42]
    G = load(os.path.join(res, "convergence", "xgb"))
    K = ["problem", "descriptor", "component"]
    T = V[K + ["auc_observed", "contribution", "ci_lo", "ci_hi"]].merge(
        G[K + ["auc_observed", "contribution", "ci_lo", "ci_hi"]], on=K, suffixes=("_svm", "_xgb"))
    return O, T


def main() -> None:
    """Write and print every summary."""
    res, out = sys.argv[1], sys.argv[2]
    os.makedirs(out, exist_ok=True)
    pd.set_option("display.width", 250)
    S = calibration(res)
    S.to_csv(os.path.join(out, "calibration_contrasts.csv"), index=False)
    P = pooled_rates(res)
    P.to_csv(os.path.join(out, "calibration_pooled.csv"), index=False)
    print("== pooled rejection rate, contrasts with target zero by argument ==")
    print(P.round(4).to_string(index=False))
    print("== per contrast ==")
    print(S.round(3).to_string(index=False))
    Rr = recovery(res)
    Rr.to_csv(os.path.join(out, "recovery.csv"), index=False)
    print("== recovery ==")
    print(Rr.round(3).to_string(index=False))
    Cv = convergence(res)
    Cv.to_csv(os.path.join(out, "convergence.csv"), index=False)
    print("== convergence ==")
    print(Cv.round(3).sort_values(["problem", "descriptor", "component", "run"]).to_string(index=False))
    O, T = learners(res)
    O.to_csv(os.path.join(out, "learners_observed_auc.csv"), index=False)
    T.to_csv(os.path.join(out, "learners_sensitivity.csv"), index=False)
    print("== learners: observed AUC ==")
    print(O.round(3).to_string(index=False))
    print("== learners: interpreted sensitivities ==")
    print(T.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
