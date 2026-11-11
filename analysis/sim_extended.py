"""Extended simulation checks of the interval procedure, at the Monte Carlo settings of the main analyses.

Three campaigns, one job per (campaign, design, size, descriptor, dataset) and one CSV per job (resumable):

* ``val``     the calibration designs and sizes of sim_calibration.py rerun at R = 6, B = 60 (the settings of every
              reported profile) with the leave-one-participant-out duration donor pool; 40 datasets per cell.
* ``direct``  the same datasets for a subset of cells, with the direct refit bootstrap that averages all R
              transformed replicates inside every bootstrap draw (no variance correction; R times slower).
              200 datasets per cell (SPO_DIRECT_N); the corrected interval is run on the same datasets.
* ``ext``     new designs: a weak spread contrast (scale20w), a weak order contrast (orderw), a design in which
              shuffling the order exposes information the observed descriptor misses (negorder, negative
              sensitivity), and the zero-target designs under unequal trial counts, class imbalance and strong
              participant heterogeneity; 40 datasets per cell.
* ``extref``  reference values of every ``ext`` contrast (point estimates, R = 20, 60 datasets, disjoint seeds).

Sizes beyond those of sim_calibration.py (participants x trials; W = label within, B = label between participants):
W60u / B65u  unequal trial counts (each participant keeps 1-4 trials per class, or 1-4 trials);
W60i / B65i  class imbalance (one positive in four trials within participants; 16 of 65 participants positive);
W60h / B65h  strong heterogeneity (participant offset SD 100 instead of 25, and participant scaling exp(N(0, 0.3))).

Usage:
    python sim_extended.py WORKERS CAMPAIGN [CAMPAIGN ...]
"""
from __future__ import annotations

import os
import sys
import traceback
import warnings
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Any, Callable, Dict, List, Tuple

import numpy as np

from scanpathdecomp import Trial, decompose
from scanpathdecomp import nulls as N
from run_profiles import CV_FOLDS, METRIC, STRATIFY, learner
from sim_calibration import CAL_DESIGNS, CAL_SIZES, OUT
from sim_calibration import make as make_base
from sim_design import DESC, make_dataset

warnings.filterwarnings("ignore")

N_DIRECT = int(os.environ.get("SPO_DIRECT_N", "200"))   # datasets of the direct-versus-corrected cells
SETTINGS: Dict[str, Dict[str, Any]] = {"val": dict(n_reps=6, n_boot=60, n_datasets=40, seed_base=1000, ci="refit"),
            "direct": dict(n_reps=6, n_boot=60, n_datasets=N_DIRECT, seed_base=1000, ci="refit_direct"),
            "ext": dict(n_reps=6, n_boot=60, n_datasets=40, seed_base=3000, ci="refit"),
            "extref": dict(n_reps=20, n_boot=0, n_datasets=60, seed_base=5000, ci="refit"),
            "big": dict(n_reps=6, n_boot=60, n_datasets=40, seed_base=7000, ci="refit"),
            "bigref": dict(n_reps=20, n_boot=0, n_datasets=60, seed_base=9000, ci="refit")}
BIG_DESIGNS = (("none", ("rqa", "directed", "classic")), ("scale20ws", ("rqa", "directed")))
BIG_CELLS = [(d, "W322x4", dn) for d, dns in BIG_DESIGNS for dn in dns]
DIRECT_CELLS = [("none", "W60x4", "rqa"), ("none", "B65x2", "rqa"), ("none", "W60x4", "classic"),
                ("scale20ws", "W60x4", "rqa"), ("scale20ws", "B65x2", "rqa"), ("scale20wsn", "W60x4", "rqa")]
EXT_CELLS: List[Tuple[str, str, str]] = (
    [(d, s, dn) for d, dn in (("scale20w", "rqa"), ("orderw", "directed"), ("negorder", "classic"))
     for s in ("W60x4", "B65x2")]
    + [(d, s, "rqa") for d in ("none", "scale20wsn", "scale20w")
       for s in ("W60u", "B65u", "W60i", "B65i", "W60h", "B65h")])
HET = dict(offset_sd=100.0, scale_sd=0.3)


def make(design: str, seed: int, size: str) -> List[Trial]:
    """Draw one synthetic dataset, including the unequal, imbalanced and heterogeneous sizes.

    Args:
        design: Scenario name understood by sim_design.make_dataset.
        seed: Random seed of the dataset.
        size: A size of sim_calibration.SIZES or one of W60u, B65u, W60i, B65i, W60h, B65h.

    Returns:
        The trials.
    """
    if size in ("W60h", "B65h"):
        if size == "W60h":
            return make_dataset(design, seed, n_groups=60, per_group=4, **HET)
        trials = make_dataset(design, seed, n_groups=65, per_group=4, **HET)
        keep = {f"P{g:02d}": ("A" if g % 2 == 0 else "B") for g in range(65)}
        return [t for t in trials if t.label == keep[t.group]]
    if size not in ("W60u", "B65u", "W60i", "B65i"):
        return make_base(design, seed, size)
    sub = np.random.default_rng(seed + 777)
    if size[0] == "W":
        trials = make_dataset(design, seed, n_groups=60, per_group=8)
        out = []
        for g in range(60):
            mine = [t for t in trials if t.group == f"P{g:02d}"]
            a = [t for t in mine if t.label == "A"]
            b = [t for t in mine if t.label == "B"]
            if size == "W60u":
                m = int(sub.integers(1, 5))
                out += a[:m] + b[:m]
            else:
                out += a[:3] + b[:1]
        return out
    trials = make_dataset(design, seed, n_groups=65, per_group=8)
    pos = set(range(16)) if size == "B65i" else set(range(0, 65, 2))
    out = []
    for g in range(65):
        lab = "B" if g in pos else "A"
        mine = [t for t in trials if t.group == f"P{g:02d}" and t.label == lab]
        m = int(sub.integers(1, 5)) if size == "B65u" else 2
        out += mine[:m]
    return out


def nulls_for(dname: str, trials: List[Trial], design: str) -> Dict[str, Callable]:
    """The contrasts of one descriptor in one design.

    Args:
        dname: Descriptor key.
        trials: The dataset (for the duration donor pool).
        design: Scenario name.

    Returns:
        Mapping from contrast name to transformation.
    """
    if dname == "classic":
        return {"order": N.order, "rotation": N.rotation, "duration_coupling": N.duration_coupling,
                "duration_marginal": N.duration_marginal_loo(trials)}
    nl: Dict[str, Callable] = {"order": N.order, "rotation": N.rotation}
    if design in ("none", "scale20ws", "scale20w", "orderw"):
        nl["resolution(s=40)"] = N.resolution(40)
    if design == "none":
        nl["resolution(s=160)"] = N.resolution(160)
    return nl


def jobs(campaigns: List[str]) -> List[Tuple[str, str, str, str, int]]:
    """Every job of the requested campaigns.

    Args:
        campaigns: Subset of "val", "direct", "ext", "extref", "big", "bigref".

    Returns:
        Job tuples (campaign, design, size, descriptor, dataset), datasets outermost.
    """
    out = []
    for c in campaigns:
        if c == "val":
            cells = [(d, s, dn) for d, dns in CAL_DESIGNS.items() for s in CAL_SIZES for dn in dns]
        elif c == "direct":
            cells = DIRECT_CELLS
        elif c in ("big", "bigref"):
            cells = BIG_CELLS
        else:
            cells = EXT_CELLS
        for ds in range(SETTINGS[c]["n_datasets"]):
            out += [(c, d, s, dn, ds) for d, s, dn in cells]
            if c == "direct":    # the corrected interval on the same dataset, beyond the datasets of ``val``
                out += [("val", d, s, dn, ds) for d, s, dn in cells]
    return out


def path_of(job: Tuple[str, str, str, str, int]) -> str:
    """Output file of a job.

    Args:
        job: (campaign, design, size, descriptor, dataset).

    Returns:
        The CSV path.
    """
    c, design, size, dn, ds = job
    return os.path.join(OUT, c, f"{design}_{size}_{dn}_{ds:03d}.csv")


def one(job: Tuple[str, str, str, str, int]) -> str:
    """Run one job and save its rows.

    Args:
        job: (campaign, design, size, descriptor, dataset).

    Returns:
        A status line.
    """
    c, design, size, dn, ds = job
    st = SETTINGS[c]
    try:
        trials = make(design, int(st["seed_base"]) + ds, size)
        r = decompose(trials, DESC[dn], nulls=nulls_for(dn, trials, design), n_reps=st["n_reps"],
                      n_boot=st["n_boot"], seed=ds, positive="B", learner_factory=learner, cv_folds=CV_FOLDS,
                      ci_method=st["ci"], metric=METRIC, stratify=STRATIFY)
        obs = r.table.loc[r.table.condition == "observed", "auc_mean"].item()
        C = r.contributions.assign(mode=c, design=design, size=size, descriptor=dn, dataset=ds, auc_obs=obs,
                                   n_trials=len(trials), n_pos=sum(t.label == "B" for t in trials),
                                   n_reps=st["n_reps"], n_boot=st["n_boot"])
        C["excludes0"] = (C.ci_lo > 0) | (C.ci_hi < 0)
        os.makedirs(os.path.dirname(path_of(job)), exist_ok=True)
        C.to_csv(path_of(job), index=False)
        return f"done {job}"
    except Exception:
        return f"FAILED {job}\n{traceback.format_exc()}"


def main() -> None:
    """Run every pending job of the requested campaigns."""
    workers = int(sys.argv[1])
    pending = [j for j in jobs(sys.argv[2:]) if not os.path.exists(path_of(j))]
    if os.environ.get("SPO_REVERSE") == "1":    # a second machine can work from the other end of the queue
        pending = pending[::-1]
    print(f"{len(pending)} pending", flush=True)
    with ProcessPoolExecutor(max_workers=workers) as ex:
        futs = [ex.submit(one, j) for j in pending]
        for f in as_completed(futs):
            res = f.result()
            if res.startswith("FAILED"):
                print(res, flush=True)


if __name__ == "__main__":
    main()
