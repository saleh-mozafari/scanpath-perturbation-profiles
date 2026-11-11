"""Longer runs (R = 20, B = 300) of the cells the paper interprets, under three seeds, with either learner.

The cells are read from results/summaries/interpreted_cells.csv (or the file named by SPO_CELLS).
Only the transformations a cell interprets are re-run, so each job profiles one descriptor on one problem for a
handful of contrasts. Features of every descriptor are cached (as for the graph descriptors in run_profiles.py),
so a job interrupted by a time limit resumes where it stopped.

Usage:
    python run_convergence.py WORKERS LEARNER R B SEED [SEED ...]

LEARNER is "svm" (the paper's learner) or "xgb" (gradient-boosted trees, a robustness check).
Writes ../results/convergence/<learner>/conv_<problem>_<descriptor>_R<R>b<B>_s<seed>.csv.
"""
from __future__ import annotations

import os
import sys
import traceback
import warnings
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Callable, Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin

from scanpathdecomp import decompose, default_nulls
from open_data import PROBLEMS
from run_profiles import (CV_FOLDS, CV_SEED, HERE, JITTER_LINES, METRIC, STRATIFY, FeatureCache, descriptor_set,
                          learner)

warnings.filterwarnings("ignore")
OUT = os.environ.get("SPO_CONV_OUT", os.path.join(HERE, "..", "results", "convergence"))

CELL_FILE = os.environ.get("SPO_CELLS", os.path.join(HERE, "..", "results", "summaries", "interpreted_cells.csv"))


def load_cells(path: str = CELL_FILE) -> Dict[Tuple[str, str], List[str]]:
    """The interpreted cells: (problem, descriptor) -> component names as in the main run.

    Args:
        path: CSV with the columns problem, descriptor and component (written by the selection rule).

    Returns:
        Mapping from (problem, descriptor) to the list of interpreted components.
    """
    cells: Dict[Tuple[str, str], List[str]] = {}
    for row in pd.read_csv(path).itertuples():
        cells.setdefault((row.problem, row.descriptor), []).append(row.component)
    return cells


CELLS = load_cells()


class XGBMargin(ClassifierMixin, BaseEstimator):
    """Gradient-boosted trees with a decision_function (the log-odds margin), for the profile's AUCs.

    Hyperparameters are fixed in advance (shallow trees, moderate shrinkage) rather than tuned, so the learner
    adds no selection step that the participant-held-out evaluation would have to account for.
    """

    def __init__(self, n_estimators: int = 200, max_depth: int = 3, learning_rate: float = 0.05,
                 subsample: float = 0.8, random_state: int = 0) -> None:
        """Store the hyperparameters.

        Args:
            n_estimators: Number of trees.
            max_depth: Maximum tree depth.
            learning_rate: Shrinkage.
            subsample: Row subsampling per tree.
            random_state: Seed of the subsampling.
        """
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.learning_rate = learning_rate
        self.subsample = subsample
        self.random_state = random_state

    def fit(self, X: np.ndarray, y: np.ndarray) -> "XGBMargin":
        """Fit the booster.

        Args:
            X: Features.
            y: Binary labels.

        Returns:
            self
        """
        from xgboost import XGBClassifier
        self.model_ = XGBClassifier(n_estimators=self.n_estimators, max_depth=self.max_depth,
                                    learning_rate=self.learning_rate, subsample=self.subsample,
                                    random_state=self.random_state, n_jobs=1, tree_method="hist",
                                    eval_metric="logloss")
        self.model_.fit(X, y)
        self.classes_ = self.model_.classes_
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predicted class labels.

        Args:
            X: Features.

        Returns:
            Labels.
        """
        return np.asarray(self.model_.predict(X))

    def decision_function(self, X: np.ndarray) -> np.ndarray:
        """Log-odds margin of the positive class.

        Args:
            X: Features.

        Returns:
            Margins.
        """
        return np.asarray(self.model_.predict(X, output_margin=True), dtype=float)


LEARNERS: Dict[str, Callable] = {"svm": learner, "xgb": XGBMargin}


def nulls_of(problem: str, dname: str, trials: list, centre: Callable | None) -> Dict[str, Callable]:
    """The interpreted transformations of one cell, built exactly as in the main run.

    Args:
        problem: Problem key.
        dname: Descriptor key.
        trials: The problem's trials.
        centre: Rotation centre function, or None for the frame centre.

    Returns:
        Mapping from component name (as in the main run) to transformation.
    """
    p = PROBLEMS[problem]
    scales = tuple(s * p.line for s in JITTER_LINES)
    alln = default_nulls(trials, scales, centre)
    names = CELLS[(problem, dname)]
    return {n: alln[n] for n in names}


def one(problem: str, dname: str, lname: str, n_reps: int, n_boot: int, seed: int) -> str:
    """Run one cell and save it.

    Args:
        problem: Problem key.
        dname: Descriptor key.
        lname: Learner key.
        n_reps: Replicates.
        n_boot: Bootstrap draws.
        seed: Seed.

    Returns:
        A status line.
    """
    try:
        p = PROBLEMS[problem]
        trials, positive = p.loader()
        desc, centre = descriptor_set(p)[dname]
        cdir = os.environ.get("SPO_CACHE_DIR", os.path.join(HERE, "..", "results", "cache"))
        os.makedirs(cdir, exist_ok=True)
        desc = FeatureCache(desc, os.path.join(cdir, f"conv_{problem}_{dname}_s{seed}.sqlite"))   # one writer per file
        r = decompose(trials, desc, positive=positive, nulls=nulls_of(problem, dname, trials, centre),
                      n_reps=n_reps, n_boot=n_boot, seed=seed, ci_method="refit",
                      learner_factory=LEARNERS[lname], cv_folds=CV_FOLDS, metric=METRIC, stratify=STRATIFY,
                      cv_seed=CV_SEED)
        if r.contributions.empty or not np.isfinite(r.table.auc_mean.iloc[0]):
            raise RuntimeError("every evaluation failed; check the learner")
        C = r.contributions.assign(problem=problem, descriptor=dname, learner=lname, n_reps=n_reps,
                                   n_boot_req=n_boot, seed=seed, line_px=p.line)
        d = os.path.join(OUT, lname)
        os.makedirs(d, exist_ok=True)
        C.to_csv(os.path.join(d, f"conv_{problem}_{dname}_R{n_reps}b{n_boot}_s{seed}.csv"), index=False)
        return f"done {problem} {dname} {lname} s{seed}"
    except Exception:
        return f"FAILED {problem} {dname} {lname} s{seed}\n{traceback.format_exc()}"


def main() -> None:
    """Run every pending cell for the requested learner and seeds."""
    workers, lname, n_reps, n_boot = int(sys.argv[1]), sys.argv[2], int(sys.argv[3]), int(sys.argv[4])
    seeds = [int(s) for s in sys.argv[5:]]
    pending = [(k, d, s) for s in seeds for (k, d) in CELLS
               if not os.path.exists(os.path.join(OUT, lname, f"conv_{k}_{d}_R{n_reps}b{n_boot}_s{s}.csv"))]
    print(f"{len(pending)} pending", flush=True)
    with ProcessPoolExecutor(max_workers=workers) as ex:
        futs = [ex.submit(one, k, d, lname, n_reps, n_boot, s) for (k, d, s) in pending]
        for f in as_completed(futs):
            print(f.result(), flush=True)


if __name__ == "__main__":
    main()
