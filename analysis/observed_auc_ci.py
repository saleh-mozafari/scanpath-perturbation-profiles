"""Observed AUC with a refit participant-bootstrap interval and a permutation p-value, per problem and descriptor.

Usage:
    python observed_auc_ci.py WORKERS B N_PERM PROBLEM_PREFIX [PROBLEM_PREFIX ...]

The interval resamples participants with replacement and refits the cross-validated classifier on every draw.
The permutation test shuffles labels in the way the design allows: across participants when the label is constant
within every participant, otherwise within participants (each participant keeps its own label counts). The p-value
is (1 + #{perm AUC >= observed}) / (1 + N_PERM); the lower-tail probability, the permutation mean and SD
are stored too. Writes ../results/observed_auc/oauc_<problem>_<descriptor>.csv.
Graph descriptors read their features from the feature cache filled by run_profiles.py. Set SPO_LEARNER=xgb
for the gradient-boosted-trees learner (default: the linear SVM).
"""
from __future__ import annotations

import os
import sys
import warnings
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import List, Tuple

import numpy as np
import pandas as pd

from scanpathdecomp.decompose import _featurize, _loso, _prep
from open_data import PROBLEMS, Problem
from run_all import jobs_for
from run_convergence import LEARNERS
from run_profiles import CACHED, CV_FOLDS, CV_SEED, HERE, FeatureCache, descriptor_set

warnings.filterwarnings("ignore")
METRIC = os.environ.get("SPO_METRIC", "fold")           # "fold" (within-fold AUC; primary) or "pooled"
STRATIFY = os.environ.get("SPO_STRATIFY", "1") == "1"   # class-balanced participant folds (primary)
OUT = os.environ.get("SPO_OAUC_OUT", os.path.join(HERE, "..", "results", "observed_auc"))


def permute_labels(y: np.ndarray, units: np.ndarray, blocks: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Shuffle binary labels between label units within exchangeability blocks.

    Each unit (the level at which a label was assigned, such as a participant or a passage) keeps one label; within
    each block the units' labels are permuted, and every trial of a unit receives its unit's new label.

    Args:
        y: Labels (0/1) per trial.
        units: Label unit per trial (labels are constant within a unit).
        blocks: Exchangeability block per trial (a unit lies in one block).
        rng: Random generator.

    Returns:
        Permuted labels.
    """
    out = y.copy()
    for b in np.unique(blocks):
        mb = blocks == b
        us = list(dict.fromkeys(units[mb].tolist()))
        lab = [y[mb & (units == u)][0] for u in us]
        new = rng.permutation(lab)
        for u, lv in zip(us, new):
            out[mb & (units == u)] = lv
    return out


def unit_block(p: Problem, trials: list, rows: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Permutation unit and block of every evaluated row.

    Args:
        p: The problem.
        trials: Its trials, with meta["trial_id"] set.
        rows: Trial index of each evaluated row (the _idx column of the feature table).

    Returns:
        (units, blocks) as integer codes aligned with the rows (tuple-valued units are coded, not compared).
    """
    ucode: dict = {}
    bcode: dict = {}
    units = np.array([ucode.setdefault(p.perm_unit(trials[i]), len(ucode)) for i in rows])
    blocks = np.array([bcode.setdefault(p.perm_block(trials[i]), len(bcode)) for i in rows])
    return units, blocks


def one(job: Tuple[str, str], n_boot: int, n_perm: int, seed: int = 42) -> str:
    """Compute the observed AUC, its interval and its permutation p-value for one job.

    Args:
        job: (problem, descriptor).
        n_boot: Bootstrap draws.
        n_perm: Permutations.
        seed: Random seed.

    Returns:
        A one-line status message.
    """
    k, d = job
    p = PROBLEMS[k]
    trials, positive = p.loader()
    for i, t in enumerate(trials):
        t.meta["trial_id"] = i
    desc, _ = descriptor_set(p)[d]
    if d in CACHED:
        cdir = os.environ.get("SPO_CACHE_DIR", os.path.join(HERE, "..", "results", "cache"))
        desc = FeatureCache(desc, os.path.join(cdir, f"{k}_{d}.sqlite"))
    M = _featurize(trials, desc)
    P = _prep(M, positive)
    lf = LEARNERS[os.environ.get("SPO_LEARNER", "svm")]
    _, auc = _loso(P, lf, metric=METRIC, cv_folds=CV_FOLDS, stratify=STRATIFY, cv_seed=CV_SEED)
    if not np.isfinite(auc):
        raise RuntimeError(f"{k} {d}: the observed evaluation failed; check the learner")
    rng = np.random.default_rng(seed)
    groups = np.unique(P[2])
    boots: List[float] = []
    for _ in range(n_boot):
        pick = rng.choice(groups, len(groups), replace=True)
        idx = np.concatenate([P[3][s] for s in pick])
        _, a = _loso(P, lf, idx, METRIC, CV_FOLDS, STRATIFY, CV_SEED)
        if np.isfinite(a):
            boots.append(float(a))
    X, y, g, gidx, ridx = P
    units, blocks = unit_block(p, trials, M["_idx"].values)
    for u in np.unique(units):
        if len(np.unique(y[units == u])) != 1:
            raise RuntimeError(f"{k}: label varies within permutation unit {u}")
    perms: List[float] = []
    for _ in range(n_perm):
        yp = permute_labels(y, units, blocks, rng)
        _, a = _loso((X, yp, g, gidx, ridx), lf, metric=METRIC, cv_folds=CV_FOLDS, stratify=STRATIFY, cv_seed=CV_SEED)
        if np.isfinite(a):
            perms.append(float(a))
    lo, hi = np.percentile(boots, [2.5, 97.5]) if boots else (np.nan, np.nan)
    pv = (1 + sum(a >= auc for a in perms)) / (1 + len(perms))
    pl = (1 + sum(a <= auc for a in perms)) / (1 + len(perms))
    os.makedirs(OUT, exist_ok=True)
    pd.DataFrame([dict(problem=k, descriptor=d, auc=auc, ci_lo=lo, ci_hi=hi, p_perm=pv, p_lower=pl, perm_mean=float(np.mean(perms)),
                       perm_sd=float(np.std(perms, ddof=1)),
                       n_boot=len(boots), n_perm=len(perms), n_trials=len(M), cv_folds=CV_FOLDS,
                       metric=METRIC, stratify=STRATIFY, perm_scheme=p.perm_note)]) \
        .to_csv(os.path.join(OUT, f"oauc_{k}_{d}.csv"), index=False)
    return f"{k} {d}: AUC {auc:.3f} [{lo:.3f}, {hi:.3f}], p = {pv:.3f}, null mean {np.mean(perms):.3f}"


def main() -> None:
    """Run every pending job."""
    workers, n_boot, n_perm = (int(v) for v in sys.argv[1:4])
    pending = [j for j in jobs_for(sys.argv[4:]) if not os.path.exists(os.path.join(OUT, f"oauc_{j[0]}_{j[1]}.csv"))]
    print(f"{len(pending)} pending", flush=True)
    with ProcessPoolExecutor(max_workers=workers) as ex:
        futs = [ex.submit(one, j, n_boot, n_perm) for j in pending]
        for f in as_completed(futs):
            print(f.result(), flush=True)


if __name__ == "__main__":
    main()
