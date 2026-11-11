"""Apply the selection rule and write the list of interpreted cells.

Usage:
    python select_cells.py RESULTS_DIR

A cell (problem, descriptor, transformation) is interpreted if the observed AUC exceeds its permutation
distribution (p <= 0.05), the refit-bootstrap interval of the observed AUC lies above 0.5, and the sensitivity
interval excludes zero. Reads RESULTS_DIR/profiles_v5 and RESULTS_DIR/observed_auc_v5 and writes
RESULTS_DIR/summaries/interpreted_cells.csv.
"""
from __future__ import annotations

import glob
import os
import sys

import pandas as pd

ALPHA = 0.05


def select(res: str) -> pd.DataFrame:
    """The interpreted cells.

    Args:
        res: Results directory.

    Returns:
        One row per interpreted cell with its observed AUC, its interval and p-value, and the sensitivity.
    """
    prof = pd.concat([pd.read_csv(f) for f in sorted(glob.glob(os.path.join(res, "profiles_v5", "prof_*.csv")))])
    prof = prof[(prof.ci_lo > 0) | (prof.ci_hi < 0)]
    auc = pd.concat([pd.read_csv(f) for f in sorted(glob.glob(os.path.join(res, "observed_auc_v5", "oauc_*.csv")))])
    auc = auc[["problem", "descriptor", "ci_lo", "ci_hi", "p_perm"]].rename(columns={"ci_lo": "a_lo", "ci_hi": "a_hi"})
    M = prof.merge(auc, on=["problem", "descriptor"])
    M = M[(M.p_perm <= ALPHA) & (M.a_lo > 0.5)]
    return M[["problem", "descriptor", "component", "auc_observed", "a_lo", "a_hi", "p_perm", "contribution",
              "ci_lo", "ci_hi"]]


if __name__ == "__main__":
    out = select(sys.argv[1])
    out.to_csv(os.path.join(sys.argv[1], "summaries", "interpreted_cells.csv"), index=False)
    print(f"{len(out)} interpreted cells")
