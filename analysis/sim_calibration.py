"""Simulation checks of the perturbation profile under the open-data pipeline.

The pipeline is the one the paper uses on real data: a linear SVM (liblinear, C = 1) on standardized features
and ten-fold cross-validation over participants. Three kinds of run, each one job per
(mode, design, size, descriptor, dataset) and one CSV per job, so the campaign can be resumed:

* ``rec``  recovery: a property is planted (order, duration trend, duration marginal, side, eccentricity, scale)
           and each scenario is profiled with a descriptor that can read it and one that cannot
           (R = 5, B = 60, 20 datasets at 20 x 8).
* ``cal``  calibration: designs in which the target of some contrasts is zero by argument, at the sample sizes of
           the analysed problems (R = 10, B = 150, 40 datasets per cell).
* ``ref``  reference: the estimand of every calibration contrast, estimated on datasets with disjoint seeds
           (point estimates only, R = 20, 60 datasets per cell).

Sizes (participants x trials per participant; W = label within participants, B = label between participants):
W60x4 (GazeBase), W112x2 (EMIP within), B95x20 (SB-SAT), B65x2 (EMIP expertise).

Usage:
    python sim_calibration.py WORKERS MODE [MODE ...]
"""
from __future__ import annotations

import os
import sys
import traceback
import warnings
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Any, Callable, Dict, List, Tuple

import numpy as np
import pandas as pd

from scanpathdecomp import Trial, decompose
from scanpathdecomp import nulls as N
from run_profiles import CV_FOLDS, HERE, METRIC, STRATIFY, learner
from sim_design import DESC, PLAN, make_dataset

warnings.filterwarnings("ignore")
OUT = os.environ.get("SPO_SIM_OUT", os.path.join(HERE, "..", "results", "simulation"))

SIZES: Dict[str, Tuple[bool, int, int]] = {          # name -> (between, participants, trials per participant)
    "W20x8": (False, 20, 8),
    "W60x4": (False, 60, 4),
    "W112x2": (False, 112, 2),
    "B95x20": (True, 95, 20),
    "B65x2": (True, 65, 2),
    "W322x4": (False, 322, 4),         # GazeBase Round 1
}
CAL_SIZES = ["W60x4", "W112x2", "B95x20", "B65x2"]
CAL_DESIGNS: Dict[str, List[str]] = {                # design -> descriptors profiled
    "none": ["rqa", "directed", "classic"],
    "scale20ws": ["rqa", "directed"],
    "scale20wsn": ["rqa", "directed"],
}
REC_SCENARIOS = ["none", "order", "duration", "durmarg", "anchoring", "eccentric", "scale20", "scale60"]
REC_SCALES = (5, 10, 20, 40, 80, 160, 320)
SETTINGS = {"rec": dict(n_reps=5, n_boot=60, n_datasets=20, seed_base=1000),
            "cal": dict(n_reps=10, n_boot=150, n_datasets=40, seed_base=1000),
            "ref": dict(n_reps=20, n_boot=0, n_datasets=60, seed_base=2000)}


def make(design: str, seed: int, size: str) -> List[Trial]:
    """Draw one synthetic dataset.

    For a between-participant size, each participant is assigned one class and keeps only trials of that class,
    drawn from the same generative model (twice as many are drawn and half kept), so the class-conditional law
    of a trial is unchanged and only the label's design level differs.

    Args:
        design: Scenario name understood by sim_design.make_dataset.
        seed: Random seed of the dataset.
        size: Key of SIZES.

    Returns:
        The trials.
    """
    between, ng, pg = SIZES[size]
    if not between:
        return make_dataset(design, seed, n_groups=ng, per_group=pg)
    trials = make_dataset(design, seed, n_groups=ng, per_group=2 * pg)
    keep = {f"P{g:02d}": ("A" if g % 2 == 0 else "B") for g in range(ng)}
    return [t for t in trials if t.label == keep[t.group]]


def nulls_for(dname: str, trials: List[Trial], design: str) -> Dict[str, Callable]:
    """The contrasts checked for one descriptor in one design.

    Order (every design) and rotation in the no-offset design have target zero by argument; the others have
    targets estimated by the reference runs. Jitter at 40 px under the strong-signal design is a contrast with a
    clearly nonzero target, for coverage away from zero.

    Args:
        dname: Descriptor key.
        trials: The dataset (for the duration donor pool).
        design: Scenario name.

    Returns:
        Mapping from contrast name to transformation.
    """
    if dname == "classic":
        pool = np.concatenate([t.dur for t in trials])
        return {"order": N.order, "rotation": N.rotation, "duration_coupling": N.duration_coupling,
                "duration_marginal": N.duration_marginal(pool)}
    nl: Dict[str, Callable] = {"order": N.order, "rotation": N.rotation}
    if design == "none":
        nl.update({"resolution(s=40)": N.resolution(40), "resolution(s=160)": N.resolution(160)})
    if design == "scale20ws":
        nl["resolution(s=40)"] = N.resolution(40)
    return nl


def jobs(modes: List[str]) -> List[Tuple[str, str, str, str, int]]:
    """Every (mode, design, size, descriptor, dataset) job of the requested modes.

    Args:
        modes: Subset of "rec", "cal", "ref".

    Returns:
        Job tuples, ordered so that every cell gets datasets early.
    """
    out = []
    for m in modes:
        st = SETTINGS[m]
        for ds in range(st["n_datasets"]):
            if m == "rec":
                for sc in REC_SCENARIOS:
                    for dn in PLAN[sc]:
                        out.append((m, sc, "W20x8", dn, ds))
            else:
                for design, descs in CAL_DESIGNS.items():
                    for size in CAL_SIZES:
                        for dn in descs:
                            out.append((m, design, size, dn, ds))
    return out


def path_of(job: Tuple[str, str, str, str, int]) -> str:
    """Output file of a job.

    Args:
        job: (mode, design, size, descriptor, dataset).

    Returns:
        The CSV path.
    """
    m, design, size, dn, ds = job
    return os.path.join(OUT, m, f"{design}_{size}_{dn}_{ds:03d}.csv")


def one(job: Tuple[str, str, str, str, int]) -> str:
    """Run one job and save its rows.

    Args:
        job: (mode, design, size, descriptor, dataset).

    Returns:
        A status line.
    """
    m, design, size, dn, ds = job
    st = SETTINGS[m]
    try:
        trials = make(design, st["seed_base"] + ds, size)
        kw: Dict[str, Any] = dict(n_reps=st["n_reps"], n_boot=st["n_boot"], seed=ds, positive="B",
                                  learner_factory=learner, cv_folds=CV_FOLDS, ci_method="refit", metric=METRIC,
                                  stratify=STRATIFY)
        if m == "rec":
            r = decompose(trials, DESC[dn], resolution_scales=REC_SCALES, **kw)
        else:
            r = decompose(trials, DESC[dn], nulls=nulls_for(dn, trials, design), **kw)
        obs = r.table.loc[r.table.condition == "observed", "auc_mean"].item()
        C = r.contributions.assign(mode=m, design=design, size=size, descriptor=dn, dataset=ds, auc_obs=obs,
                                   n_trials=len(trials), n_reps=st["n_reps"], n_boot=st["n_boot"])
        C["excludes0"] = (C.ci_lo > 0) | (C.ci_hi < 0)
        os.makedirs(os.path.dirname(path_of(job)), exist_ok=True)
        C.to_csv(path_of(job), index=False)
        return f"done {job}"
    except Exception:
        return f"FAILED {job}\n{traceback.format_exc()}"


def collect(mode: str) -> pd.DataFrame:
    """Concatenate the finished jobs of one mode.

    Args:
        mode: "rec", "cal" or "ref".

    Returns:
        One row per dataset x descriptor x contrast.
    """
    d = os.path.join(OUT, mode)
    files = [os.path.join(d, f) for f in sorted(os.listdir(d))] if os.path.isdir(d) else []
    return pd.concat([pd.read_csv(f) for f in files], ignore_index=True) if files else pd.DataFrame()


def main() -> None:
    """Run every pending job of the requested modes."""
    workers = int(sys.argv[1])
    pending = [j for j in jobs(sys.argv[2:]) if not os.path.exists(path_of(j))]
    print(f"{len(pending)} pending", flush=True)
    with ProcessPoolExecutor(max_workers=workers) as ex:
        futs = [ex.submit(one, j) for j in pending]
        for f in as_completed(futs):
            res = f.result()
            if res.startswith("FAILED"):
                print(res, flush=True)


if __name__ == "__main__":
    main()
