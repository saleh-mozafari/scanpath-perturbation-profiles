"""Extract fixation events from GazeBase v2.0 (Griffith et al., 2021; CC BY 4.0).

Fixations are detected with I-VT (30 deg/s, minimum 80 ms) on the raw 1 kHz samples for every task, because the
dataset's own per-sample labels are absent for the video and game tasks; one detector throughout avoids confounding
task with detector. Centroids are in degrees of visual angle.

Usage:
    python gazebase_extract.py GAZEBASE_DIR OUT_CSV [N_SUBJECTS] [ROUND] [WORKERS]

Each participant is written to its own part file next to OUT_CSV (``<OUT_CSV>.parts/``), so an interrupted run
resumes where it stopped; OUT_CSV is written once every participant is done. N_SUBJECTS = 0 takes every participant
of the round.
"""
from __future__ import annotations

import io
import os
import sys
import zipfile
from concurrent.futures import ProcessPoolExecutor
from typing import List, Tuple

import numpy as np
import pandas as pd

MIN_DUR = 80          # ms (1 sample = 1 ms)
VEL = 30.0            # deg/s: I-VT saccade velocity threshold
SMOOTH = 5            # samples: moving-average window before differencing
COLS = ["subject", "session", "task", "k", "x", "y", "t0", "dur"]


def ivt(x: np.ndarray, y: np.ndarray, valid: np.ndarray, min_dur: int = MIN_DUR, vel: float = VEL,
        smooth: int = SMOOTH) -> List[Tuple[int, int]]:
    """Velocity-threshold fixation detection (Salvucci & Goldberg, 2000) on 1 kHz samples.

    Positions are smoothed with a short moving average, velocity is |dP/dt| in deg/s, valid samples below the
    threshold form fixation candidates, and runs of at least min_dur ms are fixations.

    Args:
        x: Horizontal gaze position in degrees.
        y: Vertical gaze position in degrees.
        valid: Sample validity.
        min_dur: Minimum fixation duration in samples.
        vel: Velocity threshold in deg/s.
        smooth: Moving-average window in samples.

    Returns:
        List of (start, end) sample indices, end exclusive.
    """
    k = np.ones(smooth) / smooth
    xs = np.convolve(np.nan_to_num(x), k, mode="same")
    ys = np.convolve(np.nan_to_num(y), k, mode="same")
    v = np.sqrt(np.diff(xs) ** 2 + np.diff(ys) ** 2) * 1000.0
    v = np.concatenate([[v[0]], v])
    fix = valid & (v < vel)
    d = np.diff(np.concatenate([[0], fix.astype(int), [0]]))
    st = np.where(d == 1)[0]
    en = np.where(d == -1)[0]
    return [(int(s), int(e)) for s, e in zip(st, en) if e - s >= min_dur]


def one(zpath: str, part: str) -> str:
    """Extract the fixations of one participant archive into a part file.

    Args:
        zpath: Path of Subject_XXXX.zip.
        part: Output CSV for this participant.

    Returns:
        The participant identifier.
    """
    rows = []
    with zipfile.ZipFile(zpath) as z:
        for member in z.namelist():
            if not member.endswith(".csv"):
                continue
            base = os.path.basename(member)[:-4]          # S_1001_S1_TEX
            _, sid, sess, task = base.split("_")
            d = pd.read_csv(io.BytesIO(z.read(member)), usecols=["n", "x", "y", "val"])
            x, y = d.x.values.astype(float), d.y.values.astype(float)
            valid = (d.val.values == 0) & np.isfinite(x) & np.isfinite(y)
            for k, (s, e) in enumerate(ivt(x, y, valid)):
                rows.append((sid, sess, task, k, float(x[s:e].mean()), float(y[s:e].mean()), float(d.n.values[s]),
                             float(e - s)))
    tmp = part + ".tmp"
    pd.DataFrame(rows, columns=COLS).to_csv(tmp, index=False)
    os.replace(tmp, part)
    return os.path.basename(zpath)


def main() -> None:
    """Extract every requested participant, then write the combined table."""
    root, out = sys.argv[1], sys.argv[2]
    n_subj = int(sys.argv[3]) if len(sys.argv) > 3 else 60
    rnd = int(sys.argv[4]) if len(sys.argv) > 4 else 1
    workers = int(sys.argv[5]) if len(sys.argv) > 5 else 1
    rdir = os.path.join(root, f"Round_{rnd}")
    zips = sorted(f for f in os.listdir(rdir) if f.endswith(".zip"))
    if n_subj > 0:
        zips = zips[:n_subj]
    pdir = out + ".parts"
    os.makedirs(pdir, exist_ok=True)
    parts = {z: os.path.join(pdir, z.replace(".zip", ".csv")) for z in zips}
    todo = [z for z in zips if not os.path.exists(parts[z])]
    print(f"{len(todo)} of {len(zips)} participants to extract", flush=True)
    with ProcessPoolExecutor(max_workers=workers) as ex:
        for name in ex.map(one, [os.path.join(rdir, z) for z in todo], [parts[z] for z in todo]):
            print(name, flush=True)
    F = pd.concat([pd.read_csv(parts[z], dtype={"subject": str}) for z in zips], ignore_index=True)
    F.to_csv(out, index=False)
    med = F.groupby("task").apply(lambda g: g.groupby(["subject", "session"]).size().median(), include_groups=False)
    print(med.rename("median fix/trial"))


if __name__ == "__main__":
    main()
