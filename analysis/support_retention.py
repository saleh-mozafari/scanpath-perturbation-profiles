"""Support retention of the graph descriptors under each transformation, and support-matched controls.

Usage:
    python support_retention.py tables [R] [SEED]
    python support_retention.py matched PROBLEM [R] [B] [SEED]
    python support_retention.py support PROBLEM [R] [B] [SEED]

A fixation that a transformation moves outside the triangulation of the graph descriptor falls in no triangle and
is dropped from the graph. ``tables`` reports, for every problem, triangulation (AOI or lattice), transformation and
class, the mean share of fixations and of dwell time that stay on the support (results/support/retention.csv).
``support`` profiles a descriptor made only of the two retention shares on the AOI triangulation, which shows
whether support loss alone separates the classes. ``matched`` profiles the AOI graph under rotation and jitter
redrawn until the trial keeps at least the dwell share it had before the transformation (support-matched
transformations), so that a sensitivity that survives cannot come from differential support loss.
"""
from __future__ import annotations

import os
import sys
import warnings
from typing import Callable, Dict, List, Optional

import numpy as np
import pandas as pd

from scanpathdecomp import Trial, decompose
from scanpathdecomp.decompose import default_nulls
from scanpathdecomp.fag import StimulusTriangulation

from open_data import PROBLEMS, Problem
from run_profiles import (CV_FOLDS, HERE, JITTER_LINES, METRIC, PAD_LINES, STRATIFY, FeatureCache, descriptor_set,
                          learner, padded)

warnings.filterwarnings("ignore")
OUT = os.environ.get("SPO_SUPPORT_OUT", os.path.join(HERE, "..", "results", "support"))
MAX_TRIES = 200


def triangulations(p: Problem) -> Dict[str, Callable[[Trial], StimulusTriangulation]]:
    """The triangulations used by the graph descriptors of a problem.

    Args:
        p: The problem.

    Returns:
        Mapping from descriptor key to a function returning the trial's triangulation.
    """
    lat = StimulusTriangulation.lattice(p.frame, 2.0 * p.line)
    out: Dict[str, Callable[[Trial], StimulusTriangulation]] = {"fag_lattice": lambda t: lat}
    if p.aoi_points is not None:
        tri = {k: StimulusTriangulation.from_points(v) for k, v in p.aoi_points().items()}
        out["fag_aoi"] = lambda t: tri[t.meta["stimulus"]]
        ptri = {k: StimulusTriangulation.from_points(padded(v, p.frame, PAD_LINES * p.line))
                for k, v in p.aoi_points().items()}
        out["fag_aoipad"] = lambda t: ptri[t.meta["stimulus"]]
    return out


def retained(T: StimulusTriangulation, t: Trial) -> Dict[str, float]:
    """Share of a trial's fixations and dwell time that fall inside a triangulation.

    Args:
        T: The triangulation.
        t: The trial.

    Returns:
        Dict with the fixation share (fix) and the dwell share (dwell).
    """
    q = np.stack([t.x, t.y], 1) + T._nudge
    inside = T.tri.find_simplex(q) >= 0
    d = t.dur if t.dur is not None else np.ones(len(t))
    return dict(fix=float(inside.mean()), dwell=float(d[inside].sum() / max(d.sum(), 1e-12)))


def tables(n_reps: int = 6, seed: int = 42) -> pd.DataFrame:
    """Mean retention by problem, triangulation, transformation and class.

    Args:
        n_reps: Transformed replicates per trial.
        seed: Random seed.

    Returns:
        The retention table (also written to results/support/retention.csv).
    """
    rows: List[dict] = []
    for k, p in PROBLEMS.items():
        trials, positive = p.loader()
        centre = descriptor_set(p).get("fag_aoi", (None, None))[1]
        for dname, get_tri in triangulations(p).items():
            nl = default_nulls(trials, tuple(s * p.line for s in JITTER_LINES),
                               rotation_center=centre if dname.startswith("fag_aoi") else None)
            rng = np.random.default_rng(seed)
            for i, t in enumerate(trials):
                T = get_tri(t)
                cls = "positive" if t.label == positive else "negative"
                base = dict(problem=k, descriptor=dname, cls=cls, trial=i)
                rows.append({**base, "transformation": "observed", **retained(T, t)})
                for name, f in nl.items():
                    for r in range(n_reps):
                        rows.append({**base, "transformation": name, "rep": r, **retained(T, f(t, rng))})
    D = pd.DataFrame(rows)
    per_trial = D.groupby(["problem", "descriptor", "transformation", "cls", "trial"])[["fix", "dwell"]].mean()
    S = per_trial.groupby(["problem", "descriptor", "transformation", "cls"]).agg(
        fix_mean=("fix", "mean"), dwell_mean=("dwell", "mean"), dwell_min=("dwell", "min"),
        n_trials=("dwell", "size")).reset_index()
    W = S.pivot_table(index=["problem", "descriptor", "transformation"], columns="cls",
                      values=["fix_mean", "dwell_mean"]).reset_index()
    W.columns = ["_".join(c).strip("_") for c in W.columns]
    W["dwell_gap"] = W["dwell_mean_positive"] - W["dwell_mean_negative"]
    os.makedirs(OUT, exist_ok=True)
    S.to_csv(os.path.join(OUT, "retention_long.csv"), index=False)
    W.to_csv(os.path.join(OUT, "retention.csv"), index=False)
    return W


def matched(f: Callable[[Trial, np.random.Generator], Trial], T_of: Callable[[Trial], StimulusTriangulation],
            tol: float = 0.02, stats: Optional[Dict[str, int]] = None) -> Callable[[Trial, np.random.Generator], Trial]:
    """Wrap a transformation so that the transformed trial keeps at least its observed dwell share on the support.

    Args:
        f: The transformation.
        T_of: Trial -> triangulation.
        tol: Allowed loss of dwell share.
        stats: Optional counter dict; receives the number of calls and of calls that hit MAX_TRIES.

    Returns:
        A transformation that redraws f until the condition holds (after MAX_TRIES draws the best draw is kept).
    """
    def _g(t: Trial, rng: np.random.Generator) -> Trial:
        T = T_of(t)
        target = retained(T, t)["dwell"] - tol
        best, best_share = None, -1.0
        if stats is not None:
            stats["calls"] = stats.get("calls", 0) + 1
        for _ in range(MAX_TRIES):
            u = f(t, rng)
            s = retained(T, u)["dwell"]
            if s >= target:
                return u
            if s > best_share:
                best, best_share = u, s
        if stats is not None:
            stats["unmatched"] = stats.get("unmatched", 0) + 1
        return best if best is not None else t
    return _g


def _profile(problem: str, dname: str, desc: Callable, nulls: Optional[Dict[str, Callable]], tag: str,
             n_reps: int, n_boot: int, seed: int) -> pd.DataFrame:
    """Run and save one profile.

    Args:
        problem: Problem key.
        dname: Descriptor label for the output.
        desc: The descriptor.
        nulls: Transformations (None for the default set).
        tag: File tag.
        n_reps: Replicates per transformation.
        n_boot: Bootstrap draws.
        seed: Random seed.

    Returns:
        The sensitivity table.
    """
    p = PROBLEMS[problem]
    trials, positive = p.loader()
    centre = descriptor_set(p)["fag_aoi"][1]
    r = decompose(trials, desc, positive=positive, nulls=nulls, n_reps=n_reps, n_boot=n_boot, seed=seed,
                  resolution_scales=tuple(s * p.line for s in JITTER_LINES), rotation_center=centre,
                  ci_method="refit", learner_factory=learner, cv_folds=CV_FOLDS, metric=METRIC,
                  stratify=STRATIFY)
    C = r.contributions.assign(problem=problem, descriptor=dname, control=tag, n_reps=n_reps, n_boot=n_boot,
                               seed=seed)
    os.makedirs(OUT, exist_ok=True)
    C.to_csv(os.path.join(OUT, f"{tag}_{problem}_R{n_reps}b{n_boot}_s{seed}.csv"), index=False)
    r.table.assign(problem=problem, descriptor=dname, control=tag).to_csv(
        os.path.join(OUT, f"tab_{tag}_{problem}_R{n_reps}b{n_boot}_s{seed}.csv"), index=False)
    return C


def run_matched(problem: str, n_reps: int = 6, n_boot: int = 60, seed: int = 42) -> pd.DataFrame:
    """Profile the AOI graph under support-matched rotation and jitter.

    Args:
        problem: Problem key (with AOI anchors).
        n_reps: Replicates per transformation.
        n_boot: Bootstrap draws.
        seed: Random seed.

    Returns:
        The sensitivity table.
    """
    p = PROBLEMS[problem]
    trials, _ = p.loader()
    desc, centre = descriptor_set(p)["fag_aoi"]
    cdir = os.environ.get("SPO_CACHE_DIR", os.path.join(HERE, "..", "results", "cache"))
    os.makedirs(cdir, exist_ok=True)
    desc = FeatureCache(desc, os.path.join(cdir, f"{problem}_fag_aoi.sqlite"))
    T_of = triangulations(p)["fag_aoi"]
    base = default_nulls(trials, tuple(s * p.line for s in JITTER_LINES), rotation_center=centre)
    stats: Dict[str, Dict[str, int]] = {}
    nl = {}
    for k, v in base.items():
        if k == "rotation" or k.startswith("resolution"):
            stats[k] = {}
            nl[f"{k}[matched]"] = matched(v, T_of, stats=stats[k])
    C = _profile(problem, "fag_aoi", desc, nl, "matched", n_reps, n_boot, seed)
    pd.DataFrame([dict(problem=problem, transformation=k, **v) for k, v in stats.items()]).to_csv(
        os.path.join(OUT, f"matched_stats_{problem}.csv"), index=False)
    return C


def support_descriptor(T_of: Callable[[Trial], StimulusTriangulation]) -> Callable[[Trial], dict]:
    """Descriptor made only of the support-retention shares.

    Args:
        T_of: Trial -> triangulation.

    Returns:
        Trial -> {fix, dwell}.
    """
    def _f(t: Trial) -> dict:
        return retained(T_of(t), t)
    _f.__name__ = "aoi_support"
    return _f


def run_support(problem: str, n_reps: int = 6, n_boot: int = 60, seed: int = 42) -> pd.DataFrame:
    """Profile the retention-only descriptor under the default transformations.

    Args:
        problem: Problem key (with AOI anchors).
        n_reps: Replicates per transformation.
        n_boot: Bootstrap draws.
        seed: Random seed.

    Returns:
        The sensitivity table.
    """
    p = PROBLEMS[problem]
    return _profile(problem, "aoi_support", support_descriptor(triangulations(p)["fag_aoi"]), None, "support",
                    n_reps, n_boot, seed)


if __name__ == "__main__":
    a = sys.argv
    if a[1] == "tables":
        print(tables(*(int(v) for v in a[2:4])).to_string())
    else:
        ints = [int(v) for v in a[3:6]]
        fn = run_matched if a[1] == "matched" else run_support
        print(fn(a[2], *ints).to_string())
