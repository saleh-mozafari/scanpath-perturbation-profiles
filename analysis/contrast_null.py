"""Permutation nulls for the transformed pipeline and for the contrast, in selected profile cells.

Usage:
    python contrast_null.py [N_PERM] [R] [SEED]

A negative sensitivity means the transformed pipeline separates the classes better than the observed one. Whether
that is information made accessible by the transformation, or a change in evaluation bias, is judged against three
permutation distributions, all built with the problem's own permutation scheme (label units shuffled within
exchangeability blocks) and the same transformed trials: the observed AUC, the transformed AUC (mean over R
replicates) and the contrast (observed minus transformed). Labels do not enter the transformations used here, so
the transformed features are computed once. Writes results/summaries/contrast_null.csv.
"""
from __future__ import annotations

import os
import sys
import warnings
from typing import List, Tuple

import numpy as np
import pandas as pd

from scanpathdecomp.decompose import _featurize, _loso, _prep, default_nulls

from observed_auc_ci import METRIC, STRATIFY, permute_labels, unit_block
from open_data import PROBLEMS
from run_profiles import CV_FOLDS, HERE, JITTER_LINES, descriptor_set, learner

warnings.filterwarnings("ignore")
CELLS: List[Tuple[str, str, str]] = [("em_program", "classic", "order"), ("gb_vd1_halves", "classic", "order"),
                                     ("gb_tex_vd1", "classic", "order"),
                                     ("gb_vd1_blg", "classic", "order"), ("em_halves", "rqa_half", "resolution(s=37)")]


def cell(problem: str, dname: str, comp: str, n_perm: int, n_reps: int, seed: int) -> dict:
    """Permutation nulls of one cell.

    Args:
        problem: Problem key.
        dname: Descriptor key.
        comp: Transformation name.
        n_perm: Permutations.
        n_reps: Transformed replicates.
        seed: Random seed.

    Returns:
        Observed values, null means and one-sided p-values.
    """
    p = PROBLEMS[problem]
    trials, positive = p.loader()
    for i, t in enumerate(trials):
        t.meta["trial_id"] = i
    desc, centre = descriptor_set(p)[dname]
    fn = default_nulls(trials, tuple(s * p.line for s in JITTER_LINES), rotation_center=centre)[comp]
    rng = np.random.default_rng(seed)
    M0 = _featurize(trials, desc)
    P0 = _prep(M0, positive)
    reps = []
    for _ in range(n_reps):
        rr = np.random.default_rng(rng.integers(0, 2**32 - 1))
        Mn = _featurize([fn(t, rr) for t in trials], desc).set_index("_idx").loc[M0["_idx"].values].reset_index()
        reps.append(_prep(Mn, positive))

    def stats(y: np.ndarray) -> Tuple[float, float]:
        a0 = _loso((P0[0], y, P0[2], P0[3], P0[4]), learner, metric=METRIC, cv_folds=CV_FOLDS, stratify=STRATIFY)[1]
        at = float(np.mean([_loso((Pn[0], y, Pn[2], Pn[3], Pn[4]), learner, metric=METRIC, cv_folds=CV_FOLDS,
                                  stratify=STRATIFY)[1] for Pn in reps]))
        return a0, at

    a0, at = stats(P0[1])
    units, blocks = unit_block(p, trials, M0["_idx"].values)
    null = np.array([stats(permute_labels(P0[1], units, blocks, rng)) for _ in range(n_perm)])
    c0, cn = a0 - at, null[:, 0] - null[:, 1]
    pv = lambda obs, dist, sign=1: (1 + np.sum(sign * dist >= sign * obs)) / (1 + len(dist))  # noqa: E731
    return dict(problem=problem, descriptor=dname, component=comp, metric=METRIC, auc_obs=a0, auc_trans=at,
                contrast=c0, null_obs_mean=null[:, 0].mean(), null_trans_mean=null[:, 1].mean(),
                null_contrast_mean=cn.mean(), null_contrast_sd=cn.std(ddof=1), p_obs=pv(a0, null[:, 0]),
                p_trans=pv(at, null[:, 1]), p_contrast_neg=pv(c0, cn, -1), p_contrast_pos=pv(c0, cn),
                n_perm=n_perm, n_reps=n_reps)


def main() -> None:
    """Run every cell and save the table."""
    n_perm, n_reps, seed = (int(v) for v in (sys.argv[1:4] + ["200", "6", "42"][len(sys.argv[1:4]):]))
    rows = []
    for c in CELLS:
        rows.append(cell(*c, n_perm, n_reps, seed))
        print({k: (round(v, 3) if isinstance(v, float) else v) for k, v in rows[-1].items()}, flush=True)
    out = os.path.join(HERE, "..", "results", "summaries")
    os.makedirs(out, exist_ok=True)
    pd.DataFrame(rows).to_csv(os.path.join(out, f"contrast_null_{METRIC}.csv"), index=False)


if __name__ == "__main__":
    main()
