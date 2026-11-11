"""Perturbation profile of a scanpath descriptor.

For a descriptor f, a binary participant-labelled task, and a set of perturbations, evaluate the
participant-held-out AUC on the observed trials and on R perturbed replicates per perturbation.
The *contribution* (perturbation sensitivity) of a perturbation is
    C = AUC_obs - mean_r AUC(perturbed_r).
It is a property of (descriptor, learner, evaluation, task, perturbation) — not an additive share
of information — and contributions of different perturbations need not sum to the total signal.

Intervals: a participant-clustered bootstrap that REFITS the classifier on every resample
(ci_method="refit", default; see _refit_ci for the variance correction and its assumptions), a
direct variant that averages k perturbed replicates inside each draw (ci_method="refit_direct"),
or the fast but anti-conservative resampling of fixed out-of-fold scores (ci_method="scores").
"""
from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import Callable, Dict, Optional, Sequence
import numpy as np
import pandas as pd
from scipy.stats import rankdata
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import cross_val_predict
from .trial import Trial
from . import nulls as N


def _auc_pooled(y, s):
    """Mann-Whitney AUC on pooled scores (identical to sklearn's roc_auc_score)."""
    y = np.asarray(y); r = rankdata(s); n1 = y.sum(); n0 = len(y) - n1
    if n1 == 0 or n0 == 0:
        return np.nan
    return (r[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)


def _auc_within(y, s, g):
    """Mean over participants of the within-participant AUC (participants with one class are skipped).
    Estimates within-person discrimination rather than pooled discrimination."""
    y = np.asarray(y); s = np.asarray(s); g = np.asarray(g); vals = []
    for p in np.unique(g):
        m = g == p; a = _auc_pooled(y[m], s[m])
        if np.isfinite(a):
            vals.append(a)
    return float(np.mean(vals)) if vals else np.nan


def roc_auc_score(y, s, g=None, metric="pooled"):
    return _auc_within(y, s, g) if metric == "within" else _auc_pooled(y, s)


def default_learner():
    return make_pipeline(StandardScaler(), SVC(kernel="linear", C=1.0))


def _featurize(trials: Sequence[Trial], descriptor: Callable) -> pd.DataFrame:
    rows = []
    for k, t in enumerate(trials):
        f = descriptor(t)
        if f is None:
            continue
        f = dict(f); f["_idx"] = k; f["_label"] = t.label; f["_group"] = t.group
        rows.append(f)
    return pd.DataFrame(rows)


def _prep(M: pd.DataFrame, positive):
    feats = [c for c in M.columns if not c.startswith("_")]
    X = M[feats].values; yv = (M["_label"] == positive).astype(int).values; gv = M["_group"].values
    gidx = {s_: np.where(gv == s_)[0] for s_ in np.unique(gv)}
    return X, yv, gv, gidx, M["_idx"].values


def _group_kfold_splits(groups, k, seed=0, labels=None):
    """k-fold splits over participants with a platform-independent assignment.

    Participants (sorted identifiers) are put in a seeded random order and dealt to the folds in turn, so every
    fold holds about the same number of participants. Unlike scikit-learn's GroupKFold, the assignment does not
    depend on how a platform's sort orders participants of equal size, so results reproduce across machines.
    With ``labels``, participants are first ordered by their share of positive trials (ties in the seeded order)
    and then dealt in turn, so every fold receives a similar mix of classes (class-balanced folds).

    Args:
        groups: Participant identifier per row.
        k: Number of folds.
        seed: Seed of the participant order.
        labels: Optional binary label per row, for class-balanced folds.

    Returns:
        List of (train indices, test indices).
    """
    groups = np.asarray(groups)
    ug = np.unique(groups)
    order = ug[np.random.default_rng(seed).permutation(len(ug))]
    if labels is not None:
        labels = np.asarray(labels)
        share = {g: float(labels[groups == g].mean()) for g in ug}
        rank = {g: i for i, g in enumerate(order)}
        order = np.array(sorted(order, key=lambda g: (share[g], rank[g])), dtype=ug.dtype)
    fold_of = {g: i % k for i, g in enumerate(order)}
    f = np.array([fold_of[g] for g in groups])
    rows = np.arange(len(groups))
    return [(rows[f != j], rows[f == j]) for j in range(k)]


def _auc_within_folds(y, s, folds):
    """AUC over the positive-negative pairs that lie in the same test fold.

    Each fold's AUC compares only scores produced by one fitted model, so differences in score offset between
    fold models cannot enter. Folds are weighted by their number of positive-negative pairs, which makes the
    result the proportion of correctly ordered within-fold pairs.

    Args:
        y: Binary labels.
        s: Out-of-fold decision values.
        folds: List of index arrays, one per test fold.

    Returns:
        The within-fold AUC, or nan if no fold contains both classes.
    """
    y = np.asarray(y); s = np.asarray(s); num = 0.0; den = 0.0
    for te in folds:
        yy = y[te]; n1 = yy.sum(); n0 = len(yy) - n1
        if n1 == 0 or n0 == 0:
            continue
        num += _auc_pooled(yy, s[te]) * n1 * n0; den += n1 * n0
    return num / den if den > 0 else np.nan


def _loso(P, learner_factory, idx=None, metric="pooled", cv_folds=None, stratify=False, cv_seed=0):
    """Participant-held-out decision values and AUC on the trials in idx.

    A multiset of participants is allowed: duplicated participants keep one identity, so folds stay
    participant-pure.

    Args:
        P: Prepared feature tuple from _prep.
        learner_factory: Returns a fresh estimator with a decision_function.
        idx: Row indices to evaluate (default: all rows).
        metric: "pooled" (all out-of-fold scores ranked together), "within" (mean within-participant AUC) or
            "fold" (AUC over positive-negative pairs from the same test fold).
        cv_folds: None for leave-one-participant-out; an integer k for k-fold cross-validation over
            participants, which is much faster when there are many participants.
        stratify: With k folds, deal participants so that every fold has a similar mix of classes.
        cv_seed: Seed of the participant order that defines the k-fold partition.

    Returns:
        (decision values, AUC), or (None, nan) if the AUC is undefined.
    """
    X, yv, gv, gidx, _ = P
    if idx is None:
        idx = np.arange(len(yv))
    Xb, yb, gb = X[idx], yv[idx], gv[idx]
    if yb.min() == yb.max() or len(np.unique(gb)) < 3:
        return None, np.nan
    n_g = len(np.unique(gb))
    if cv_folds is None or cv_folds >= n_g:
        rows = np.arange(len(gb))
        cv = [(rows[gb != g], rows[gb == g]) for g in np.unique(gb)]
    else:
        cv = _group_kfold_splits(gb, int(cv_folds), seed=cv_seed, labels=yb if stratify else None)
    try:
        pb = cross_val_predict(learner_factory(), Xb, yb, groups=gb, cv=cv, method="decision_function")
    except Exception:
        return None, np.nan
    if metric == "fold":
        return pb, _auc_within_folds(yb, pb, [te for _, te in cv])
    return pb, roc_auc_score(yb, pb, gb, metric)


@dataclass
class DecompositionResult:
    table: pd.DataFrame          # one row per condition (observed + each perturbation)
    contributions: pd.DataFrame  # one row per perturbation: contribution, CI, p, se, n_boot, method
    descriptor: str
    n_trials: int
    n_groups: int
    metric: str = "pooled"
    boot: Optional[pd.DataFrame] = None   # per-draw bootstrap AUCs (component, draw, a_obs, a_null) for refit methods

    def __repr__(self):
        return (f"DecompositionResult({self.descriptor}; n_trials={self.n_trials}, n_groups={self.n_groups}, metric={self.metric})\n"
                + self.table.to_string(index=False) + "\n\n" + self.contributions.to_string(index=False))


def default_nulls(trials: Sequence[Trial], resolution_scales=(10, 30, 60, 120), rotation_center=None, donor="loo"):
    """The default set of transformations.

    Args:
        trials: The trials (for the duration donor pool).
        resolution_scales: Jitter SDs.
        rotation_center: Centre of rotation, a point or a function of the trial; None for the frame centre.
        donor: Donor pool of the duration-marginal transformation: "loo" (default; each trial draws from the
            durations of the other participants only, so no trial ever receives its own participant's durations)
            or "all" (every duration of the problem, the trial's own included).

    Returns:
        Mapping from transformation name to function(trial, rng).
    """
    has_dur = any(t.dur is not None for t in trials)
    nulls: Dict[str, Callable] = {"order": N.order, "duration_coupling": N.duration_coupling}
    if has_dur and donor == "loo":
        nulls["duration_marginal"] = N.duration_marginal_loo(trials)
    elif has_dur:
        nulls["duration_marginal"] = N.duration_marginal(np.concatenate([t.dur for t in trials if t.dur is not None]))
    nulls["rotation"] = (lambda t, rng: N.rotation(t, rng, center=rotation_center(t) if callable(rotation_center) else rotation_center))
    for s in resolution_scales:
        nulls[f"resolution(s={s:g})"] = N.resolution(s)
    return nulls


def decompose(trials: Sequence[Trial], descriptor: Callable, *, positive=None,
              learner_factory: Callable = default_learner,
              nulls: Optional[Dict[str, Callable]] = None,
              resolution_scales: Sequence[float] = (10, 30, 60, 120),
              rotation_center=None,
              n_reps: int = 20, n_boot: int = 200, seed: int = 0,
              ci_method: str = "refit", k_direct: Optional[int] = None, direct_interval: str = "normal",
              metric: str = "pooled", boot_replicates: bool = True,
              cv_folds: Optional[int] = None, stratify: bool = False, donor: str = "loo",
              cv_seed: int = 0) -> DecompositionResult:
    """Run the perturbation profile.

    trials            : list of Trial (label, group set)
    descriptor        : Trial -> dict of features, or None
    positive          : label of the positive class (default: lexicographically larger)
    learner_factory   : returns a fresh sklearn estimator with decision_function
    nulls             : {name: fn(trial, rng)}; default = default_nulls(trials, resolution_scales, rotation_center)
    rotation_center   : (cx, cy) or callable(trial)->(cx, cy) for the rotation null (default: frame centre)
    n_reps            : perturbed replicates per null
    n_boot            : bootstrap draws; 0 returns point estimates only (no intervals), which is ~R*B/(R+1)
                        times cheaper and is how a reference value for the estimand is estimated
    ci_method         : 'refit' (refit bootstrap with Monte-Carlo variance correction; default),
                        'refit_direct' (refit bootstrap averaging k_direct replicates inside each draw — k_direct=None
                        means all R replicates, i.e. the same estimator as the point estimate; no correction; R times
                        slower), or 'scores' (resample fixed out-of-fold scores; anti-conservative)
    direct_interval   : for 'refit_direct': 'normal' (point +/- 1.96 * bootstrap SD of a_obs - mean_k a_null; same
                        construction as 'refit') or 'percentile'
    metric            : 'pooled' AUC over all out-of-fold trials, or 'within' (mean within-participant AUC)
    cv_folds          : None (default) for leave-one-participant-out; k for k-fold cross-validation over
                        participants. Every evaluation (observed, transformed, bootstrap) uses the same scheme.
    stratify          : with k folds, deal participants so that folds have similar class mixes
    donor             : duration-marginal donor pool, "loo" (other participants only; default) or "all"
    metric            : also 'fold' (AUC over positive-negative pairs within the same test fold)
    cv_seed           : seed of the participant order that defines the k-fold partition (default 0); vary it to
                        check that a result does not depend on one partition

    Exact invariances (the transformed features equal the observed features in every replicate) are reported with
    ci_lo = ci_hi = 0 and exact_invariant=True; they are not estimates. A cell whose bootstrap draws all give
    a_obs == a_null without feature invariance (e.g. at ceiling) is flagged degenerate=True. If the corrected variance
    is not positive for a non-invariant component, the uncorrected paired bootstrap variance is used instead
    (conservative) and variance_fallback=True is set.
    """
    by_group: dict = {}
    for t in trials:
        by_group.setdefault(t.group, set()).add(t.label)
    if metric == "pooled" and all(len(v) == 1 for v in by_group.values()):
        warnings.warn("The label is constant within every participant: the pooled cross-validated AUC is biased "
                      "below 0.5 under no signal, and perturbations can change that bias. Consider metric='fold' "
                      "with cv_folds=k and stratify=True, and read AUCs against a permutation distribution.",
                      UserWarning, stacklevel=2)
    rng = np.random.default_rng(seed)
    labels = sorted(set(t.label for t in trials), key=str)
    assert len(labels) == 2, "binary tasks only"
    positive = labels[1] if positive is None else positive
    def _loso_cv(P, lf, idx=None, metric="pooled"):
        return _loso(P, lf, idx, metric, cv_folds, stratify, cv_seed)

    if nulls is None:
        nulls = default_nulls(trials, resolution_scales, rotation_center, donor)

    M0 = _featurize(trials, descriptor); P0 = _prep(M0, positive)
    p0, auc0 = _loso_cv(P0, learner_factory, metric=metric)
    n_all = len(trials)
    S0 = np.full(n_all, np.nan); Y = np.full(n_all, -1); G = np.array([t.group for t in trials], dtype=object)
    S0[P0[4]] = p0; Y[P0[4]] = P0[1]

    rows = [dict(condition="observed", auc_mean=auc0, auc_sd=0.0, n_reps=1, n_trials=len(M0))]
    null_scores: Dict[str, np.ndarray] = {}; null_preps: Dict[str, list] = {}; null_aucs: Dict[str, list] = {}; null_featinv: Dict[str, bool] = {}
    feats0 = [c for c in M0.columns if not c.startswith("_")]
    for name, fn in nulls.items():
        aucs, reps, preps = [], [], []; featinv = True
        for r in range(n_reps):
            rr = np.random.default_rng(rng.integers(0, 2**32 - 1))
            Mn = _featurize([fn(t, rr) for t in trials], descriptor)
            if len(Mn) < 10 or Mn["_label"].nunique() < 2:
                continue
            # exact invariance = the transformed FEATURES equal the observed features (to 1e-9) for every replicate
            featinv = featinv and len(Mn) == len(M0) and np.array_equal(Mn["_idx"].values, M0["_idx"].values) \
                and np.allclose(Mn[feats0].values.astype(float), M0[feats0].values.astype(float), rtol=1e-9, atol=1e-9)
            Pn = _prep(Mn, positive); p, a = _loso_cv(Pn, learner_factory, metric=metric)
            if p is None:
                continue
            S = np.full(n_all, np.nan); S[Pn[4]] = p
            aucs.append(a); reps.append(S); preps.append(Pn)
        null_scores[name] = np.array(reps) if reps else np.zeros((0, n_all)); null_preps[name] = preps; null_aucs[name] = aucs; null_featinv[name] = bool(featinv and reps)
        rows.append(dict(condition=name, auc_mean=float(np.mean(aucs)) if aucs else np.nan,
                         auc_sd=float(np.std(aucs, ddof=1)) if len(aucs) > 1 else 0.0, n_reps=len(aucs),
                         n_trials=int(np.isfinite(null_scores[name]).any(0).sum()) if reps else 0))
    table = pd.DataFrame(rows)

    groups = np.unique(G[P0[4]]); brng = np.random.default_rng(seed + 1)
    contrib = []; boot_rows = []
    for name, R in null_scores.items():
        if len(R) == 0:
            continue
        point = auc0 - table.loc[table.condition == name, "auc_mean"].item()
        Fr = null_preps[name]; R_ = len(Fr)
        if n_boot == 0:
            # point estimate only: no interval. Used to estimate a reference value for the estimand on
            # datasets that are independent of those whose intervals are being assessed.
            contrib.append(dict(component=name, auc_observed=auc0, auc_null=table.loc[table.condition == name, "auc_mean"].item(),
                                contribution=point, ci_lo=np.nan, ci_hi=np.nan, se=np.nan, p_boot=np.nan, n_boot=0,
                                ci_method="none", share=point / max(auc0 - 0.5, 1e-9),
                                exact_invariant=null_featinv[name], variance_fallback=False, degenerate=False))
            continue
        if ci_method in ("refit", "refit_direct"):
            obs_list, null_list = [], []
            for _ in range(n_boot):
                pick = brng.choice(groups, len(groups), replace=True)
                idx0 = np.concatenate([P0[3][s_] for s_ in pick if s_ in P0[3]])
                _, a_obs = _loso_cv(P0, learner_factory, idx0, metric)
                if ci_method == "refit":
                    Pn = Fr[brng.integers(0, R_)]
                    idxn = np.concatenate([Pn[3][s_] for s_ in pick if s_ in Pn[3]])
                    _, a_null = _loso_cv(Pn, learner_factory, idxn, metric)
                else:   # direct: average k_direct distinct replicates (default: all R) inside the draw
                    kk_ = R_ if k_direct is None else min(k_direct, R_)
                    ks = np.arange(R_) if kk_ == R_ else brng.choice(R_, kk_, replace=False); vals = []
                    for kk in ks:
                        Pn = Fr[kk]; idxn = np.concatenate([Pn[3][s_] for s_ in pick if s_ in Pn[3]])
                        _, a = _loso_cv(Pn, learner_factory, idxn, metric); vals.append(a)
                    a_null = float(np.nanmean(vals))
                if np.isfinite(a_obs) and np.isfinite(a_null):
                    obs_list.append(a_obs); null_list.append(a_null)
            A_obs, A_null = np.array(obs_list), np.array(null_list)
            diff = A_obs - A_null
            zero_paths = bool(len(diff) > 0 and np.all(np.abs(diff) < 1e-12) and np.all(np.abs(np.array(null_aucs[name]) - auc0) < 1e-12))
            exact_inv = null_featinv[name]            # invariant descriptor: identically zero, no interval
            degenerate = bool(zero_paths and not exact_inv)   # e.g. ceiling: every resample gives the same AUC, but the features differ
            fallback = False
            boot_rows += [dict(component=name, draw=b_, a_obs=float(A_obs[b_]), a_null=float(A_null[b_])) for b_ in range(len(A_obs))]
            if exact_inv or degenerate:
                se, lo, hi, pv = 0.0, 0.0, 0.0, 1.0
            elif ci_method == "refit":
                # Estimator: C = a_obs - mean over R replicates of a_null_r. Each bootstrap draw uses ONE
                # replicate, so var_b(a_null_b) = v_f + v_e, where v_f is the participant-sampling variance
                # of the null AUC and v_e the Monte-Carlo variance of a single replicate; the estimator's
                # null term has variance v_f + v_e/R. v_e is estimated from the spread of the R replicate
                # AUCs on the full data (assumption: replicate-to-replicate variance does not depend on
                # the participant resample). A non-positive corrected variance falls back to the
                # uncorrected paired variance var_b(a_obs - a_null), which overstates the estimator's
                # variance by v_e (1 - 1/R) and is therefore conservative.
                v_e = float(table.loc[table.condition == name, "auc_sd"].item() ** 2) if R_ > 1 else 0.0
                v_e = min(v_e, A_null.var(ddof=1))
                cov = np.cov(A_obs, A_null)[0, 1]
                var_C = A_obs.var(ddof=1) + (A_null.var(ddof=1) - v_e) + v_e / R_ - 2 * cov
                if not var_C > 0:
                    fallback = True; var_C = diff.var(ddof=1)
                se = float(np.sqrt(max(var_C, 0.0))); lo, hi = point - 1.96 * se, point + 1.96 * se
                from scipy.stats import norm
                pv = float(2 * norm.sf(abs(point) / se)) if se > 0 else 1.0
            else:   # refit_direct: the draw's a_null already averages k replicates; no correction
                se = float(diff.std(ddof=1))
                if direct_interval == "normal":
                    lo, hi = point - 1.96 * se, point + 1.96 * se
                    from scipy.stats import norm
                    pv = float(2 * norm.sf(abs(point) / se)) if se > 0 else 1.0
                else:
                    lo, hi = np.percentile(diff, [2.5, 97.5]); pv = float(min(1.0, 2 * min((diff <= 0).mean(), (diff >= 0).mean())))
            nb = len(A_obs)
        else:   # 'scores'
            valid = ~np.isnan(S0) & np.isfinite(R).all(0); ds = []
            for _ in range(n_boot):
                pick = brng.choice(groups, len(groups), replace=True)
                idx = np.concatenate([np.where((G == s_) & valid)[0] for s_ in pick]); yy = Y[idx]
                if yy.min() == yy.max():
                    continue
                gg = G[idx]
                a_obs = roc_auc_score(yy, S0[idx], gg, metric)
                reps_pick = brng.integers(0, len(R), len(R)) if boot_replicates else np.arange(len(R))
                a_null = np.mean([roc_auc_score(yy, R[r, idx], gg, metric) for r in reps_pick])
                ds.append(a_obs - a_null)
            dd_ = np.array(ds); se = float(dd_.std(ddof=1)); lo, hi = np.percentile(dd_, [2.5, 97.5])
            pv = float(min(1.0, 2 * min((dd_ <= 0).mean(), (dd_ >= 0).mean()))); nb = len(dd_); exact_inv = null_featinv[name]; fallback = False; degenerate = False
        contrib.append(dict(component=name, auc_observed=auc0, auc_null=table.loc[table.condition == name, "auc_mean"].item(),
                            contribution=point, ci_lo=lo, ci_hi=hi, se=se, p_boot=pv, n_boot=nb, ci_method=ci_method,
                            share=point / max(auc0 - 0.5, 1e-9), exact_invariant=exact_inv, variance_fallback=fallback, degenerate=degenerate))
    return DecompositionResult(table=table, contributions=pd.DataFrame(contrib),
                               descriptor=getattr(descriptor, "__name__", "descriptor"),
                               n_trials=len(M0), n_groups=len(groups), metric=metric,
                               boot=pd.DataFrame(boot_rows) if boot_rows else None)
