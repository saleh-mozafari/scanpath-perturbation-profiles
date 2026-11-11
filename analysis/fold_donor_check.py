"""Training-fold donor pool for the duration-marginal transformation, compared with the leave-one-participant-out pool.

Usage:
    python fold_donor_check.py [R] [SEED]

For every problem and the classic descriptor (the only descriptor with a duration-marginal sensitivity that the
paper interprets), each cross-validation fold redraws the durations of its training trials from the other training
participants and those of its test trials from the training participants only, so no held-out duration enters any
transformed trial the model is trained or tested on. The within-fold AUC (class-balanced folds) is averaged over R
replicates and compared with the AUC under the leave-one-participant-out pool of the main analysis
(results/profiles_v5).
Writes results/summaries/fold_donor_check.csv.
"""
from __future__ import annotations

import glob
import os
import sys
import warnings
from typing import Dict, List

import numpy as np
import pandas as pd

from scanpathdecomp import Trial, descriptors
from scanpathdecomp.decompose import _auc_within_folds, _group_kfold_splits

from open_data import PROBLEMS
from run_profiles import CV_FOLDS, HERE, learner

warnings.filterwarnings("ignore")
RES = os.path.join(HERE, "..", "results")


def features(trials: List[Trial]) -> np.ndarray:
    """Classic features (mean duration, mean saccade amplitude) of every trial.

    Args:
        trials: The trials (each with at least two fixations).

    Returns:
        Array of shape (n, 2).
    """
    f = descriptors.classic()
    return np.array([[v for v in f(t).values()] for t in trials])


def pools(trials: List[Trial], rows: np.ndarray) -> Dict[str, np.ndarray]:
    """Leave-one-participant-out duration pools restricted to a set of trials.

    Args:
        trials: All trials.
        rows: Indices of the trials that may donate.

    Returns:
        Mapping from participant to the pooled durations of the other donating participants; the key "*" holds
        every donated duration.
    """
    by: Dict[str, List[np.ndarray]] = {}
    for i in rows:
        by.setdefault(trials[i].group, []).append(np.asarray(trials[i].dur, dtype=float))
    alld = {g: np.concatenate(v) for g, v in by.items()}
    out = {g: np.concatenate([alld[h] for h in alld if h != g]) for g in alld}
    out["*"] = np.concatenate(list(alld.values()))
    return out


def run(n_reps: int = 6, seed: int = 42) -> pd.DataFrame:
    """Compute the training-fold check for every problem.

    Args:
        n_reps: Replicates.
        seed: Random seed.

    Returns:
        One row per problem.
    """
    out = []
    for k, p in PROBLEMS.items():
        trials, positive = p.loader()
        trials = [t for t in trials if len(t) >= 2 and t.dur is not None]
        y = np.array([t.label == positive for t in trials], dtype=int)
        g = np.array([t.group for t in trials])
        splits = _group_kfold_splits(g, CV_FOLDS, labels=y)     # class-balanced folds, as in the main analysis
        rng = np.random.default_rng(seed)
        aucs = []
        for _ in range(n_reps):
            s = np.full(len(trials), np.nan)
            for tr, te in splits:
                pl = pools(trials, tr)
                moved = list(trials)
                for i in tr:
                    moved[i] = trials[i].replace(dur=rng.choice(pl[trials[i].group], len(trials[i])))
                for i in te:
                    moved[i] = trials[i].replace(dur=rng.choice(pl["*"], len(trials[i])))
                X = features(moved)
                m = learner().fit(X[tr], y[tr])
                s[te] = m.decision_function(X[te])
            aucs.append(_auc_within_folds(y, s, [te for _, te in splits]))
        tab = glob.glob(os.path.join(RES, "profiles_v5", f"tab_{k}_classic_R6b60_s42.csv"))
        loo = obs = np.nan
        if tab:
            T = pd.read_csv(tab[0])
            loo = float(T.loc[T.condition == "duration_marginal", "auc_mean"].item())
            obs = float(T.loc[T.condition == "observed", "auc_mean"].item())
        out.append(dict(problem=k, auc_observed=obs, auc_dm_loo=loo, auc_dm_trainfold=float(np.mean(aucs)),
                        auc_dm_trainfold_sd=float(np.std(aucs, ddof=1)), n_reps=n_reps))
        print(out[-1], flush=True)
    D = pd.DataFrame(out)
    D["sens_loo"] = D.auc_observed - D.auc_dm_loo
    D["sens_trainfold"] = D.auc_observed - D.auc_dm_trainfold
    os.makedirs(os.path.join(RES, "summaries"), exist_ok=True)
    D.to_csv(os.path.join(RES, "summaries", "fold_donor_check.csv"), index=False)
    return D


if __name__ == "__main__":
    a = sys.argv
    print(run(*(int(v) for v in a[1:3])).round(3).to_string(index=False))
