"""Perturbation profiles of every descriptor on the public-data problems.

Usage:
    python run_profiles.py PROBLEM DESCRIPTOR [R] [B] [SEED] [OUTDIR]

PROBLEM is a key of ``open_data.PROBLEMS``; DESCRIPTOR a key of ``descriptor_set``. Writes
``prof_<problem>_<descriptor>_R<R>b<B>_s<seed>.csv`` (sensitivities with refit-bootstrap intervals)
and ``tab_...csv`` (AUC under each transformation) to OUTDIR.
"""
from __future__ import annotations

import atexit
import hashlib
import json
import os
import sqlite3
import sys
import time
import warnings
from collections import defaultdict
from typing import Callable, Dict, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC

from scanpathdecomp import Trial, decompose, descriptors
from scanpathdecomp.fag import StimulusTriangulation, fixation_adjacency_graph

from open_data import PROBLEMS, Problem

warnings.filterwarnings("ignore")
HERE = os.path.dirname(os.path.abspath(__file__))
JITTER_LINES = (0.25, 0.5, 1.0, 2.0)
CV_FOLDS = 10   # participant-held-out 10-fold cross-validation (GroupKFold) on every problem
METRIC = os.environ.get("SPO_METRIC", "fold")           # "fold" (within-fold AUC; primary) or "pooled"
STRATIFY = os.environ.get("SPO_STRATIFY", "1") == "1"   # class-balanced participant folds (primary)
CV_SEED = int(os.environ.get("SPO_CV_SEED", "0"))       # seed of the fold partition
PAD_LINES = 2.0                                         # spacing of the padding anchors, in line units


def learner() -> Pipeline:
    """Linear SVM (liblinear, C = 1) on standardized features, the learner for every problem.

    Returns:
        A fresh scikit-learn pipeline with a decision_function.
    """
    return make_pipeline(StandardScaler(), LinearSVC(C=1.0, max_iter=20000, random_state=0))


def dispersion() -> Callable[[Trial], dict]:
    """Baseline that reads only spatial spread.

    Returns:
        Descriptor giving the SD of x, the SD of y and the mean pairwise distance.
    """
    def _f(t: Trial) -> dict:
        P = np.stack([t.x, t.y], 1)
        D = np.sqrt(((P[:, None] - P[None]) ** 2).sum(-1))
        mpd = float(D[np.triu_indices(len(t), 1)].mean()) if len(t) > 1 else 0.0
        return dict(sdx=float(t.x.std()), sdy=float(t.y.std()), mpd=mpd)
    _f.__name__ = "dispersion"
    return _f


def count() -> Callable[[Trial], dict]:
    """Baseline that reads only the number of fixations.

    Returns:
        Descriptor giving the fixation count.
    """
    def _f(t: Trial) -> dict:
        return dict(n=float(len(t)))
    _f.__name__ = "count"
    return _f


def duration_trend() -> Callable[[Trial], Optional[dict]]:
    """Descriptor that reads how durations depend on sequence position.

    Returns:
        Descriptor giving the correlation of duration with fixation index and the mean duration.
    """
    def _f(t: Trial) -> Optional[dict]:
        if len(t) < 5 or t.dur is None:
            return None
        r = np.corrcoef(np.arange(len(t)), t.dur)[0, 1]
        return dict(dur_trend=float(0.0 if np.isnan(r) else r), dur_mean=float(t.dur.mean()))
    _f.__name__ = "duration_trend"
    return _f


def padded(points: np.ndarray, frame: tuple, spacing: float) -> np.ndarray:
    """AOI anchors padded with lattice anchors, so that no transformation moves a fixation off the support.

    Lattice nodes fill the disc around the anchors' centre that contains the frame under any rotation about that
    centre, plus a margin for jitter; nodes closer than one spacing to an AOI anchor are left out, so the
    triangulation inside the code is essentially unchanged.

    Args:
        points: AOI anchor coordinates, shape (n, 2).
        frame: (width, height) of the trial frame.
        spacing: Distance between padding anchors.

    Returns:
        The AOI anchors followed by the padding anchors.
    """
    c = points.mean(0)
    corners = np.array([[0, 0], [frame[0], 0], [0, frame[1]], [frame[0], frame[1]]], dtype=float)
    radius = float(np.sqrt(((corners - c) ** 2).sum(1)).max()) + 4.0 * spacing
    xs = np.arange(c[0] - radius, c[0] + radius + spacing, spacing)
    ys = np.arange(c[1] - radius, c[1] + radius + spacing, spacing)
    gx, gy = np.meshgrid(xs, ys)
    L = np.stack([gx.ravel() + 1e-6 * spacing * gy.ravel() / max(frame[1], 1.0), gy.ravel()], 1)
    L = L[np.sqrt(((L - c) ** 2).sum(1)) <= radius + spacing]
    d = np.sqrt(((L[:, None, :] - points[None, :, :]) ** 2).sum(-1)).min(1)
    return np.vstack([points, L[d >= spacing]])


def descriptor_set(p: Problem) -> Dict[str, Tuple[Callable, Optional[Callable]]]:
    """Descriptors for one problem, with their rotation centre.

    Args:
        p: The problem; its line spacing sets every spatial constant.

    Returns:
        Mapping from descriptor key to (descriptor, rotation centre or None for the frame centre).
    """
    L = p.line
    lattice = StimulusTriangulation.lattice(p.frame, 2.0 * L)   # anchors two reference scales apart
    out: Dict[str, Tuple[Callable, Optional[Callable]]] = {
        "rqa_half": (descriptors.rqa(0.5 * L), None),
        "rqa_line": (descriptors.rqa(1.0 * L), None),
        "directed": (descriptors.directed_graph((L, L)), None),
        "classic": (descriptors.classic(), None),
        "count": (count(), None),
        "dispersion": (dispersion(), None),
        "duration_trend": (duration_trend(), None),
    }
    fl = fixation_adjacency_graph(defaultdict(lambda: lattice), stimulus_key="stimulus")
    fl.__name__ = "fag_lattice"
    out["fag_lattice"] = (fl, None)
    if p.aoi_points is not None:
        pts = p.aoi_points()
        tri = {k: StimulusTriangulation.from_points(v) for k, v in pts.items()}
        centre = {k: tuple(v.mean(0)) for k, v in pts.items()}
        fa = fixation_adjacency_graph(tri, stimulus_key="stimulus")
        fa.__name__ = "fag_aoi"
        out["fag_aoi"] = (fa, lambda t: centre[t.meta["stimulus"]])
        ptri = {k: StimulusTriangulation.from_points(padded(v, p.frame, PAD_LINES * L)) for k, v in pts.items()}
        fp = fixation_adjacency_graph(ptri, stimulus_key="stimulus")
        fp.__name__ = "fag_aoipad"
        out["fag_aoipad"] = (fp, lambda t: centre[t.meta["stimulus"]])
    return out


class FeatureCache:
    """Disk cache of a descriptor's features, keyed by the trial's exact coordinates and durations.

    Expensive descriptors (the graph descriptors) are cached so that a profile interrupted by a time limit
    resumes where it stopped: the transformations are seeded, so a rerun produces the same transformed
    trials and finds their features in the cache. The cache changes no result.
    """

    def __init__(self, descriptor: Callable[[Trial], Optional[dict]], path: str, every: int = 50) -> None:
        """Wrap a descriptor.

        Args:
            descriptor: Trial -> feature dict or None.
            path: SQLite file holding the cache.
            every: Commit after this many new entries.
        """
        self.descriptor = descriptor
        self.path = path
        self.every = every
        self.__name__ = getattr(descriptor, "__name__", "descriptor")
        self._con: Optional[sqlite3.Connection] = None
        self._pid = -1
        self._pending = 0

    def _db(self) -> sqlite3.Connection:
        """Open the connection lazily, once per process."""
        if self._con is None or self._pid != os.getpid():
            self._con = sqlite3.connect(self.path, timeout=60)
            self._con.execute("CREATE TABLE IF NOT EXISTS f (k TEXT PRIMARY KEY, v TEXT)")
            self._pid = os.getpid()
            atexit.register(self.flush)
        return self._con

    def flush(self) -> None:
        """Commit pending entries."""
        if self._con is not None and self._pid == os.getpid():
            self._con.commit()
            self._pending = 0

    def __call__(self, t: Trial) -> Optional[dict]:
        """Return the cached features of a trial, computing and storing them if absent.

        Args:
            t: The trial.

        Returns:
            Feature dict, or None if the descriptor is undefined for the trial.
        """
        h = hashlib.sha1()
        for a in (t.x, t.y, t.dur if t.dur is not None else np.zeros(0)):
            h.update(np.ascontiguousarray(a, dtype=np.float64).tobytes())
        h.update(str(t.meta.get("stimulus", "")).encode())
        key = h.hexdigest()
        db = self._db()
        row = db.execute("SELECT v FROM f WHERE k = ?", (key,)).fetchone()
        if row is not None:
            return json.loads(row[0])
        f = self.descriptor(t)
        val = None if f is None else {k: float(v) for k, v in f.items()}
        db.execute("INSERT OR REPLACE INTO f VALUES (?, ?)", (key, json.dumps(val)))
        self._pending += 1
        if self._pending >= self.every:
            self.flush()
        return val


CACHED = ("fag_lattice", "fag_aoi", "fag_aoipad", "rqa_half", "rqa_line", "directed")


def run(problem: str, dname: str, n_reps: int = 6, n_boot: int = 60, seed: int = 42,
        outdir: Optional[str] = None) -> pd.DataFrame:
    """Profile one descriptor on one problem and save the result.

    Args:
        problem: Key of PROBLEMS.
        dname: Key of descriptor_set(problem).
        n_reps: Transformed replicates per transformation (R).
        n_boot: Refit-bootstrap draws (B).
        seed: Random seed.
        outdir: Output directory (default ../results/profiles).

    Returns:
        The sensitivity table.
    """
    p = PROBLEMS[problem]
    trials, positive = p.loader()
    desc, centre = descriptor_set(p)[dname]
    if dname in CACHED:
        cdir = os.environ.get("SPO_CACHE_DIR", os.path.join(HERE, "..", "results", "cache"))
        os.makedirs(cdir, exist_ok=True)
        desc = FeatureCache(desc, os.path.join(cdir, f"{problem}_{dname}.sqlite"))
    scales = tuple(s * p.line for s in JITTER_LINES)
    t0 = time.time()
    r = decompose(trials, desc, positive=positive, n_reps=n_reps, n_boot=n_boot, seed=seed,
                  resolution_scales=scales, rotation_center=centre, ci_method="refit",
                  learner_factory=learner, cv_folds=CV_FOLDS, metric=METRIC, stratify=STRATIFY, cv_seed=CV_SEED)
    n_part = len({t.group for t in trials})
    print(f"{problem} {dname}: {len(trials)} trials, {n_part} participants, {time.time() - t0:.0f} s", flush=True)
    outdir = outdir or os.path.join(HERE, "..", "results", "profiles")
    os.makedirs(outdir, exist_ok=True)
    tag = f"{problem}_{dname}_R{n_reps}b{n_boot}_s{seed}"
    meta = dict(problem=problem, descriptor=dname, n_reps=n_reps, n_boot=n_boot, seed=seed,
                n_trials_all=len(trials), n_participants=n_part, line_px=p.line, cv_folds=CV_FOLDS,
                metric=METRIC, stratify=STRATIFY, donor="loo", cv_seed=CV_SEED)
    C = r.contributions.assign(**meta)
    C["scale_lines"] = C["component"].str.extract(r"s=([0-9.]+)")[0].astype(float) / p.line
    C.to_csv(os.path.join(outdir, f"prof_{tag}.csv"), index=False)
    r.table.assign(**meta).to_csv(os.path.join(outdir, f"tab_{tag}.csv"), index=False)
    return C


if __name__ == "__main__":
    a = sys.argv
    run(a[1], a[2], int(a[3]) if len(a) > 3 else 6, int(a[4]) if len(a) > 4 else 60,
        int(a[5]) if len(a) > 5 else 42, a[6] if len(a) > 6 else None)
