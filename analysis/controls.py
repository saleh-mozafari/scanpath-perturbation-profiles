"""Controls requested for specific results.

Usage:
    python controls.py synthetic [N_PERM]      layout-only control for the EMIP program contrast
    python controls.py stratum [B] [N_PERM]    same-program AUC for pooled correctness
    python controls.py partitions [N] [R]      fold-partition variability of the interpreted cells
    python controls.py folds                   per-fold AUCs and SVM weights on EMIP expertise

``synthetic`` replaces every trial of the program contrast by fixations drawn uniformly over the bounding box of its
program's code tokens (same count and durations), so nothing of the participant's gaze remains; a descriptor that
still separates the programs does so from the stimulus layout alone.
``stratum`` evaluates pooled correctness with an AUC over correct-incorrect pairs from the same test fold and the
same program, with a refit participant bootstrap and labels permuted within program.
``partitions`` recomputes the observed AUC and the sensitivities of the interpreted cells under N seeded fold
partitions (point estimates) and reports their mean and SD.
``folds`` reports, for EMIP expertise, the AUC of every test fold and the sign pattern of the SVM weights.
Outputs go to results/summaries/control_*.csv.
"""
from __future__ import annotations

import os
import sys
import warnings
from typing import Callable, Dict

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from scanpathdecomp import decompose
from scanpathdecomp.decompose import _featurize, _group_kfold_splits, _loso, _prep, default_nulls

from observed_auc_ci import permute_labels, unit_block
from open_data import PROBLEMS
from run_profiles import CACHED, CV_FOLDS, HERE, JITTER_LINES, FeatureCache, descriptor_set, learner

warnings.filterwarnings("ignore")
OUT = os.path.join(HERE, "..", "results", "summaries")
CACHE = os.environ.get("SPO_CACHE_DIR", os.path.join(HERE, "..", "results", "cache"))


def _desc(problem: str, dname: str):
    """Descriptor and rotation centre of a problem, cached when the descriptor is expensive.

    Args:
        problem: Problem key.
        dname: Descriptor key.

    Returns:
        (descriptor, rotation centre).
    """
    desc, centre = descriptor_set(PROBLEMS[problem])[dname]
    if dname in CACHED:
        os.makedirs(CACHE, exist_ok=True)
        desc = FeatureCache(desc, os.path.join(CACHE, f"{problem}_{dname}.sqlite"))
    return desc, centre


def synthetic(n_perm: int = 200, seed: int = 42) -> pd.DataFrame:
    """Layout-only control for the program contrast.

    Args:
        n_perm: Permutations for the p-value.
        seed: Random seed.

    Returns:
        One row per descriptor with the AUC on real and on synthetic gaze.
    """
    p = PROBLEMS["em_program"]
    trials, positive = p.loader()
    for i, t in enumerate(trials):
        t.meta["trial_id"] = i
    assert p.aoi_points is not None
    pts = p.aoi_points()
    rng = np.random.default_rng(seed)
    synth = []
    for t in trials:
        a = pts[t.meta["stimulus"]]
        lo, hi = a.min(0), a.max(0)
        xy = rng.uniform(lo, hi, size=(len(t), 2))
        synth.append(t.replace(x=xy[:, 0], y=xy[:, 1]))
    rows = []
    for dname, (desc, _) in descriptor_set(p).items():
        if dname in ("count", "classic", "duration_trend"):
            continue
        out = dict(descriptor=dname)
        for tag, data in (("real", trials), ("synthetic", synth)):
            M = _featurize(data, desc)
            P = _prep(M, positive)
            a = _loso(P, learner, metric="fold", cv_folds=CV_FOLDS, stratify=True)[1]
            units, blocks = unit_block(p, data, M["_idx"].values)
            null = [_loso((P[0], permute_labels(P[1], units, blocks, rng), P[2], P[3], P[4]), learner, metric="fold",
                          cv_folds=CV_FOLDS, stratify=True)[1] for _ in range(n_perm)]
            out[f"auc_{tag}"] = a
            out[f"p_{tag}"] = (1 + sum(v >= a for v in null)) / (1 + len(null))
        rows.append(out)
        print(out, flush=True)
    D = pd.DataFrame(rows)
    D.to_csv(os.path.join(OUT, "control_synthetic_gaze.csv"), index=False)
    return D


def _stratum_auc(y: np.ndarray, s: np.ndarray, strata: np.ndarray) -> float:
    """AUC over positive-negative pairs that share a stratum, pair-weighted.

    Args:
        y: Binary labels.
        s: Scores.
        strata: Stratum code per row.

    Returns:
        The AUC, or nan if no stratum holds both classes.
    """
    num = den = 0.0
    for c in np.unique(strata):
        m = strata == c
        n1, n0 = int(y[m].sum()), int((1 - y[m]).sum())
        if n1 and n0:
            num += roc_auc_score(y[m], s[m]) * n1 * n0
            den += n1 * n0
    return num / den if den else float("nan")


def stratum(n_boot: int = 200, n_perm: int = 1000, seed: int = 42) -> pd.DataFrame:
    """Same-program, same-fold AUC of pooled correctness.

    Args:
        n_boot: Refit participant-bootstrap draws.
        n_perm: Permutations (labels shuffled within program).
        seed: Random seed.

    Returns:
        One row per descriptor.
    """
    p = PROBLEMS["em_correct"]
    trials, positive = p.loader()
    for i, t in enumerate(trials):
        t.meta["trial_id"] = i
    rows = []
    for dname in descriptor_set(p):
        desc, _ = _desc("em_correct", dname)
        M = _featurize(trials, desc)
        X, y, g, gidx, ridx = _prep(M, positive)
        prog = np.array([str(trials[i].meta["stimulus"]).split("_")[0] for i in M["_idx"].values])

        def evaluate(rows_: np.ndarray, labels: np.ndarray) -> float:
            Xb, yb, gb, pb = X[rows_], labels[rows_], g[rows_], prog[rows_]
            sc = np.full(len(rows_), np.nan)
            code = np.empty(len(rows_), dtype=object)
            for f, (tr, te) in enumerate(_group_kfold_splits(gb, CV_FOLDS, labels=yb)):
                m = learner().fit(Xb[tr], yb[tr])
                sc[te] = m.decision_function(Xb[te])
                for i in te:
                    code[i] = f"{f}|{pb[i]}"
            return _stratum_auc(yb, sc, code)

        allrows = np.arange(len(y))
        a = evaluate(allrows, y)
        rng = np.random.default_rng(seed)
        groups = np.unique(g)
        boots = []
        for _ in range(n_boot):
            pick = rng.choice(groups, len(groups), replace=True)
            v = evaluate(np.concatenate([gidx[s] for s in pick]), y)
            if np.isfinite(v):
                boots.append(v)
        units, blocks = unit_block(p, trials, M["_idx"].values)
        null = np.array([evaluate(allrows, permute_labels(y, units, blocks, rng)) for _ in range(n_perm)])
        lo, hi = np.percentile(boots, [2.5, 97.5])
        rows.append(dict(descriptor=dname, auc_same_program=a, ci_lo=lo, ci_hi=hi, null_mean=float(np.nanmean(null)),
                         null_sd=float(np.nanstd(null, ddof=1)),
                         p_upper=(1 + int(np.sum(null >= a))) / (1 + len(null)), n_boot=len(boots), n_perm=n_perm))
        print(rows[-1], flush=True)
    D = pd.DataFrame(rows)
    D.to_csv(os.path.join(OUT, "control_same_program_auc.csv"), index=False)
    return D


def partitions(n_part: int = 20, n_reps: int = 6, seed: int = 42) -> pd.DataFrame:
    """Variability of the interpreted cells over fold partitions.

    Args:
        n_part: Number of partitions (fold seeds 0 .. n_part - 1).
        n_reps: Transformed replicates.
        seed: Seed of the transformations (the same for every partition).

    Returns:
        One row per problem, descriptor and transformation with the mean and SD over partitions.
    """
    cells = pd.read_csv(os.path.join(OUT, "interpreted_cells.csv"))
    rows = []
    for (problem, dname), grp in cells.groupby(["problem", "descriptor"]):
        p = PROBLEMS[problem]
        trials, positive = p.loader()
        desc, centre = _desc(problem, dname)
        nl = default_nulls(trials, tuple(s * p.line for s in JITTER_LINES), rotation_center=centre)
        nl = {k: v for k, v in nl.items() if k in set(grp.component)}
        for cs in range(n_part):
            r = decompose(trials, desc, positive=positive, nulls=nl, n_reps=n_reps, n_boot=0, seed=seed,
                          learner_factory=learner, cv_folds=CV_FOLDS, metric="fold", stratify=True, cv_seed=cs)
            for c in r.contributions.itertuples():
                rows.append(dict(problem=problem, descriptor=dname, component=c.component, cv_seed=cs,
                                 auc_observed=c.auc_observed, contribution=c.contribution))
        print(problem, dname, flush=True)
    D = pd.DataFrame(rows)
    D.to_csv(os.path.join(OUT, "control_partitions_long.csv"), index=False)
    S = D.groupby(["problem", "descriptor", "component"]).agg(
        auc_mean=("auc_observed", "mean"), auc_sd=("auc_observed", "std"), c_mean=("contribution", "mean"),
        c_sd=("contribution", "std"), c_min=("contribution", "min"), c_max=("contribution", "max"),
        n=("contribution", "size")).reset_index()
    S.to_csv(os.path.join(OUT, "control_partitions.csv"), index=False)
    return S


def folds() -> pd.DataFrame:
    """Per-fold AUCs and weight signs on EMIP expertise.

    Returns:
        One row per descriptor and fold.
    """
    p = PROBLEMS["em_expertise"]
    trials, positive = p.loader()
    rows = []
    for dname in ("directed", "fag_aoi", "classic"):
        desc, _ = _desc("em_expertise", dname)
        M = _featurize(trials, desc)
        X, y, g, _, _ = _prep(M, positive)
        feats = [c for c in M.columns if not c.startswith("_")]
        full = learner().fit(X, y)
        w_full = np.sign(full[-1].coef_.ravel())
        for f, (tr, te) in enumerate(_group_kfold_splits(g, CV_FOLDS, labels=y)):
            m = learner().fit(X[tr], y[tr])
            sc = m.decision_function(X[te])
            ok = len(np.unique(y[te])) == 2
            rows.append(dict(descriptor=dname, fold=f, n_test=len(te), n_pos=int(y[te].sum()),
                             auc=roc_auc_score(y[te], sc) if ok else np.nan,
                             auc_train=roc_auc_score(y[tr], m.decision_function(X[tr])),
                             same_signs_as_full=int((np.sign(m[-1].coef_.ravel()) == w_full).sum()),
                             n_features=len(feats)))
    D = pd.DataFrame(rows)
    D.to_csv(os.path.join(OUT, "control_expertise_folds.csv"), index=False)
    return D


if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    a = sys.argv
    ints = [int(v) for v in a[2:]]
    table: Dict[str, Callable[..., pd.DataFrame]] = {"synthetic": synthetic, "stratum": stratum,
                                                     "partitions": partitions, "folds": folds}
    fn = table[a[1]]
    pd.set_option("display.width", 200)
    print(fn(*ints).round(3).to_string(index=False))
