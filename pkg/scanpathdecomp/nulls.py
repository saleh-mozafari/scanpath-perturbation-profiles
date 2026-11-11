"""Perturbations ("nulls"). Each takes (trial, rng) and returns a new Trial in which one
property of the scanpath has been destroyed. What each one PRESERVES is stated exactly, because
that is what the profile measures against.

order              permute the fixation sequence        keeps: multiset of locations, multiset of durations,
                                                         every order-free property (recurrence, density, ...)
duration_coupling  permute durations within the trial   keeps: locations, order, the trial's duration multiset
                                                         (mean, variance, total dwell, count); destroys only
                                                         which fixation carries which duration
duration_marginal  replace durations by a draw from     keeps: locations, order, count; destroys the trial's
                   the pooled durations of all trials    own duration distribution (mean, spread) as well as
                                                         the assignment
rotation           rigid rotation and/or reflection     keeps: all pairwise distances and angles, order,
                   about a centre                        durations, and each fixation's distance from the
                                                         centre (eccentricity); destroys orientation and
                                                         side relative to the stimulus. NOT a complete
                                                         removal of stimulus location.
resolution(s)      isotropic Gaussian jitter, SD s      keeps: order, durations, orientation, structure
                                                         coarser than ~s; destroys structure finer than s.
                                                         Two coincident points end up on average s*sqrt(pi)
                                                         apart (RMS 2s).
"""
from __future__ import annotations
import numpy as np
from .trial import Trial


def order(trial: Trial, rng: np.random.Generator) -> Trial:
    i = rng.permutation(len(trial))
    return trial.replace(x=trial.x[i], y=trial.y[i], dur=None if trial.dur is None else trial.dur[i])


def duration_coupling(trial: Trial, rng: np.random.Generator) -> Trial:
    """Permute durations across the fixations of the trial (marginal duration distribution preserved)."""
    if trial.dur is None:
        return trial
    return trial.replace(dur=rng.permutation(trial.dur))


def duration_marginal(pool: np.ndarray):
    """Factory: replace the trial's durations by a random draw (without replacement) from `pool`,
    the concatenated durations of all trials. Destroys the trial's own duration distribution."""
    pool = np.asarray(pool, dtype=float)
    def _null(trial: Trial, rng: np.random.Generator) -> Trial:
        if trial.dur is None:
            return trial
        n = len(trial)
        return trial.replace(dur=rng.choice(pool, n, replace=n > len(pool)))
    _null.__name__ = "duration_marginal"
    return _null


def duration_marginal_loo(trials):
    """Factory: like duration_marginal, but each trial's durations are drawn from the pooled durations of
    the OTHER participants only (leave-one-participant-out donor pool). Use as a robustness check on the
    fixed dataset-wide pool: no held-out participant's durations enter the transformed data of its own trials."""
    groups = {}
    for t in trials:
        if t.dur is not None:
            groups.setdefault(t.group, []).append(np.asarray(t.dur, dtype=float))
    alld = {g: np.concatenate(v) for g, v in groups.items()}
    pools = {g: np.concatenate([alld[h] for h in alld if h != g]) for g in alld}
    def _null(trial: Trial, rng: np.random.Generator) -> Trial:
        if trial.dur is None:
            return trial
        pool = pools[trial.group]; n = len(trial)
        return trial.replace(dur=rng.choice(pool, n, replace=n > len(pool)))
    _null.__name__ = "duration_marginal_loo"
    return _null


def rotation(trial: Trial, rng: np.random.Generator, mode: str = "rotation", center=None) -> Trial:
    """Rigid rotation/reflection about a centre. Preserves intrinsic geometry AND eccentricity.

    mode='rotation'  : uniform random angle plus a random reflection. Points may leave a rectangular
                       support; use with support-free descriptors or report the retained fraction.
                       Destroys orientation AND handedness (chirality: clockwise vs counter-clockwise
                       circulation, left-to-right vs right-to-left sweeps).
    mode='proper'    : uniform random angle, no reflection. Destroys orientation but preserves handedness;
                       comparing it with mode='rotation' separates the two.
    mode='dihedral'  : random non-identity element of the rectangle's symmetry group (horizontal flip,
                       vertical flip, 180-degree rotation). Preserves a centred rectangular support.
    center           : (cx, cy). Default: centre of trial.frame if set, else the trial's own centroid.
                       For a descriptor with bounded support, rotate about the SUPPORT's centre.
    """
    if center is not None:
        cx, cy = center
    elif trial.frame is not None:
        cx, cy = trial.frame[0] / 2.0, trial.frame[1] / 2.0
    else:
        cx, cy = trial.x.mean(), trial.y.mean()
    x, y = trial.x - cx, trial.y - cy
    if mode == "dihedral":
        k = rng.integers(0, 3)
        if k == 0:   x = -x
        elif k == 1: y = -y
        else:        x, y = -x, -y
    elif mode in ("rotation", "proper"):
        th = rng.uniform(0, 2 * np.pi)
        if mode == "rotation" and rng.random() < 0.5:
            y = -y
        x, y = np.cos(th) * x - np.sin(th) * y, np.sin(th) * x + np.cos(th) * y
    else:
        raise ValueError(mode)
    return trial.replace(x=x + cx, y=y + cy)


def resolution(s: float):
    """Factory: jitter every fixation by isotropic Gaussian noise with SD s (same unit as x, y)."""
    def _null(trial: Trial, rng: np.random.Generator) -> Trial:
        n = len(trial)
        return trial.replace(x=trial.x + rng.normal(0, s, n), y=trial.y + rng.normal(0, s, n))
    _null.__name__ = f"resolution(s={s:g})"
    return _null


def identity(trial: Trial, rng: np.random.Generator) -> Trial:
    return trial


# backwards-compatible aliases (names used in the first manuscript draft)
duration = duration_coupling
anchoring = rotation

NULL_REGISTRY = {"order": order, "duration_coupling": duration_coupling, "duration_marginal": duration_marginal, "duration_marginal_loo": duration_marginal_loo,
                 "rotation": rotation, "resolution": resolution}
