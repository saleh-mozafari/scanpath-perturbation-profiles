"""Run the perturbation profile for many (problem, descriptor) jobs in parallel, skipping finished ones.

Usage:
    python run_all.py WORKERS R B SEED PROBLEM_PREFIX [PROBLEM_PREFIX ...]

Set SPO_DESCRIPTORS to a comma-separated list to restrict the descriptors, and SPO_OUT to change the output directory.

Every problem whose key starts with one of the prefixes is profiled under every descriptor of its
descriptor set. A job is skipped when its output file already exists, so the run can be resumed.
"""
from __future__ import annotations

import os
import sys
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import List, Tuple

from open_data import PROBLEMS
from run_profiles import HERE, descriptor_set, run

OUT = os.environ.get("SPO_OUT", os.path.join(HERE, "..", "results", "profiles"))
CHEAP_FIRST = ["count", "classic", "duration_trend", "dispersion", "directed", "rqa_half", "rqa_line", "fag_aoi",
               "fag_aoipad", "fag_lattice"]


def jobs_for(prefixes: List[str]) -> List[Tuple[str, str]]:
    """List (problem, descriptor) pairs, cheap descriptors first.

    Args:
        prefixes: Problem-key prefixes to include.

    Returns:
        Ordered list of jobs.
    """
    probs = [k for k in PROBLEMS if any(k.startswith(p) for p in prefixes)]
    only = [d for d in os.environ.get("SPO_DESCRIPTORS", "").split(",") if d]
    out = []
    for d in CHEAP_FIRST:
        if only and d not in only:
            continue
        for k in probs:
            if d in descriptor_set(PROBLEMS[k]):
                out.append((k, d))
    return out


def _one(job: Tuple[str, str], r: int, b: int, seed: int) -> str:
    """Run one job, returning a status line.

    Args:
        job: (problem, descriptor).
        r: Replicates per transformation.
        b: Bootstrap draws.
        seed: Random seed.

    Returns:
        A one-line status message.
    """
    k, d = job
    try:
        run(k, d, r, b, seed, OUT)
        return f"done {k} {d}"
    except Exception:
        return f"FAILED {k} {d}\n{traceback.format_exc()}"


def main() -> None:
    """Parse arguments and run the pending jobs."""
    workers, r, b, seed = (int(v) for v in sys.argv[1:5])
    pending = [j for j in jobs_for(sys.argv[5:])
               if not os.path.exists(os.path.join(OUT, f"prof_{j[0]}_{j[1]}_R{r}b{b}_s{seed}.csv"))]
    print(f"{len(pending)} jobs pending", flush=True)
    with ProcessPoolExecutor(max_workers=workers) as ex:
        futs = [ex.submit(_one, j, r, b, seed) for j in pending]
        for f in as_completed(futs):
            print(f.result(), flush=True)


if __name__ == "__main__":
    main()
