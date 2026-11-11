"""Collect the profile outputs into one long table and one wide summary.

Usage:
    python summarize.py [R] [B] [SEED]

Writes ../results/summary_long.csv (one row per problem, descriptor and transformation) and
../results/summary_wide.csv (observed AUC and each sensitivity with its interval, one row per
problem and descriptor). A sensitivity is marked "*" when its 95% interval excludes zero and "=" when
it is an exact invariance.
"""
from __future__ import annotations

import glob
import os
import sys

import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "..", "results")


def cell(r: pd.Series) -> str:
    """Format one sensitivity as 'C [lo, hi]' with a significance mark.

    Args:
        r: A row of the long table.

    Returns:
        The formatted cell.
    """
    if bool(r.get("exact_invariant", False)):
        return "0 ="
    mark = "*" if (r.ci_lo > 0 or r.ci_hi < 0) else ""
    return f"{r.contribution:+.3f} [{r.ci_lo:+.3f}, {r.ci_hi:+.3f}]{mark}"


def main() -> None:
    """Build and save the long and wide summaries."""
    r, b, seed = (sys.argv[1:4] + ["6", "60", "42"][len(sys.argv[1:4]):])
    files = sorted(glob.glob(os.path.join(RES, "profiles", f"prof_*_R{r}b{b}_s{seed}.csv")))
    L = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    L.to_csv(os.path.join(RES, "summary_long.csv"), index=False)
    L["cell"] = L.apply(cell, axis=1)
    W = L.pivot_table(index=["problem", "descriptor"], columns="component", values="cell", aggfunc="first")
    auc = L.groupby(["problem", "descriptor"])["auc_observed"].first().round(3)
    n = L.groupby(["problem", "descriptor"])[["n_trials_all", "n_participants"]].first()
    W = pd.concat([n, auc, W], axis=1).reset_index()
    W.to_csv(os.path.join(RES, "summary_wide.csv"), index=False)
    with pd.option_context("display.width", 250, "display.max_columns", 30, "display.max_colwidth", 30):
        print(W.drop(columns=["n_trials_all"]).to_string(index=False))


if __name__ == "__main__":
    main()
