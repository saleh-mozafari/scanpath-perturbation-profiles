"""Evidence ledger of the interpreted cells.

Usage:
    python evidence_ledger.py RESULTS_DIR

For every interpreted cell (results/summaries/interpreted_cells.csv) the ledger collects the checks the cell has to
pass: the primary estimate (within-fold AUC, class-balanced folds), the pooled-AUC estimate, the fold-partition
range, the three long runs (R = 20, B = 300) and the boosted-trees run, and the sensitivity normalized by the
decodable margin, C / (AUC - permutation mean), under both metrics. A cell is called robust when its sign and the
exclusion of zero hold in the primary run, the pooled run, every long run and every fold partition, and, for the
AOI graph, on the padded graph that loses no fixation under any transformation; the boosted-trees result is reported next to it as a learner-dependence check.
Writes RESULTS_DIR/summaries/evidence_ledger.csv and prints the counts.
"""
from __future__ import annotations

import glob
import os
import sys

import numpy as np
import pandas as pd

KEYS = ["problem", "descriptor", "component"]


def _read(pattern: str) -> pd.DataFrame:
    """Concatenate the CSV files matching a pattern.

    Args:
        pattern: Glob pattern.

    Returns:
        The concatenated table.
    """
    return pd.concat([pd.read_csv(f) for f in sorted(glob.glob(pattern))], ignore_index=True)


def _excl(lo: pd.Series, hi: pd.Series) -> pd.Series:
    """Whether an interval excludes zero.

    Args:
        lo: Lower bounds.
        hi: Upper bounds.

    Returns:
        Boolean series.
    """
    return (lo > 0) | (hi < 0)


def ledger(res: str) -> pd.DataFrame:
    """Build the ledger.

    Args:
        res: Results directory.

    Returns:
        One row per interpreted cell.
    """
    summ = os.path.join(res, "summaries")
    cells = pd.read_csv(os.path.join(summ, "interpreted_cells.csv"))[KEYS]
    prim = _read(os.path.join(res, "profiles_v5", "prof_*.csv"))[KEYS + ["auc_observed", "contribution", "ci_lo",
                                                                        "ci_hi"]]
    pool = _read(os.path.join(res, "profiles_v5_pooled", "prof_*.csv"))[KEYS + ["auc_observed", "contribution",
                                                                               "ci_lo", "ci_hi"]]
    L = cells.merge(prim, on=KEYS).merge(pool, on=KEYS, suffixes=("", "_pooled"))
    oa = _read(os.path.join(res, "observed_auc_v5", "oauc_*.csv"))[["problem", "descriptor", "perm_mean"]]
    op = _read(os.path.join(res, "observed_auc_v5_pooled", "oauc_*.csv"))[["problem", "descriptor", "perm_mean"]]
    L = L.merge(oa, on=["problem", "descriptor"]).merge(op, on=["problem", "descriptor"], suffixes=("", "_pooled"))
    L["normalized"] = L.contribution / (L.auc_observed - L.perm_mean)
    L["normalized_pooled"] = L.contribution_pooled / (L.auc_observed_pooled - L.perm_mean_pooled)
    L["pooled_ok"] = _excl(L.ci_lo_pooled, L.ci_hi_pooled) & (np.sign(L.contribution_pooled) == np.sign(L.contribution))

    part = pd.read_csv(os.path.join(summ, "control_partitions.csv"))[KEYS + ["c_min", "c_max"]]
    L = L.merge(part, on=KEYS, how="left")
    L["partitions_ok"] = (L.c_min * L.c_max > 0) & (np.sign(L.c_min) == np.sign(L.contribution))

    long_ = _read(os.path.join(res, "convergence_v5", "svm", "conv_*.csv"))
    long_["ok"] = _excl(long_.ci_lo, long_.ci_hi)
    g = long_.groupby(KEYS).agg(long_n=("ok", "size"), long_excl=("ok", "sum"), long_min=("contribution", "min"),
                                long_max=("contribution", "max"), long_lo=("ci_lo", "min"),
                                long_hi=("ci_hi", "max")).reset_index()
    L = L.merge(g, on=KEYS, how="left")
    L["long_ok"] = (L.long_excl == L.long_n) & (L.long_min * L.long_max > 0) & (
        np.sign(L.long_min) == np.sign(L.contribution))

    xgb = _read(os.path.join(res, "convergence_v5", "xgb", "conv_*.csv"))[KEYS + ["auc_observed", "contribution",
                                                                                 "ci_lo", "ci_hi"]]
    L = L.merge(xgb, on=KEYS, how="left", suffixes=("", "_xgb"))
    L["xgb_ok"] = _excl(L.ci_lo_xgb, L.ci_hi_xgb) & (np.sign(L.contribution_xgb) == np.sign(L.contribution))
    # bounded support: an AOI-graph cell must also hold on the padded graph, which loses no fixation
    full = _read(os.path.join(res, "profiles_v5", "prof_*.csv"))
    pad = full[full.descriptor == "fag_aoipad"][["problem", "component", "contribution", "ci_lo", "ci_hi"]]
    L = L.merge(pad, on=["problem", "component"], how="left", suffixes=("", "_pad"))
    held = _excl(L.ci_lo_pad, L.ci_hi_pad) & (np.sign(L.contribution_pad) == np.sign(L.contribution))
    L["support_ok"] = (L.descriptor != "fag_aoi") | held
    L["robust"] = L.pooled_ok & L.partitions_ok & L.long_ok & L.support_ok
    return L


def main() -> None:
    """Write the ledger and print its counts."""
    res = sys.argv[1]
    L = ledger(res)
    L.to_csv(os.path.join(res, "summaries", "evidence_ledger.csv"), index=False)
    L["dataset"] = L.problem.str[:2]
    print(L.groupby("dataset")[["pooled_ok", "partitions_ok", "long_ok", "support_ok", "robust", "xgb_ok"]].sum().assign(
        n=L.groupby("dataset").size()).to_string())
    both = L[L.robust & L.xgb_ok]
    print(f"{len(L)} interpreted cells; {int(L.robust.sum())} robust; {len(both)} robust and confirmed with trees")
    cols = KEYS + ["contribution", "ci_lo", "ci_hi", "normalized", "normalized_pooled", "pooled_ok", "long_ok",
                   "xgb_ok"]
    pd.set_option("display.width", 250)
    print(L[~(L.robust & L.xgb_ok)][cols].round(3).to_string(index=False))


if __name__ == "__main__":
    main()
