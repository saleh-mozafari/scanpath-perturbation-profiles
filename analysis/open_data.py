"""Loaders for the public datasets and the binary problems profiled in the open-data paper.

Every problem is built only from files that are publicly released by their authors:

* SB-SAT (Ahn et al., 2020): Stony Brook SAT reading fixations, EyeLink 1000, 95 participants,
  https://github.com/ahnchive/SB-SAT (files ``18sat_fixfinal.csv`` and ``18sat_labels.csv``).
* EMIP, corrected release (Al Madi et al., 2021) of the Eye Movements in Programming dataset
  (Bednarik et al., 2020), https://github.com/nalmadi/EMIP-Toolkit (``finalset_line_part.csv``).

Spatial constants are expressed in units of the stimulus's line spacing, measured on the stimuli
(SB-SAT: 63 px between text lines; EMIP: 37 px between code lines), so that the descriptor
parameters and jitter scales mean the same thing on both datasets.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Hashable, List, Optional, Tuple

import numpy as np
import pandas as pd

from scanpathdecomp import Trial

RAW = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data", "raw")
MIN_FIX = 5


@dataclass
class Problem:
    """A binary classification problem on one public dataset.

    Attributes:
        key: Short identifier used in file names.
        dataset: Dataset name.
        description: One-line description of the contrast.
        label_level: "within" if the label varies within participants, "between" if it is constant
            within every participant, "mixed" otherwise.
        frame: (width, height) of the display in pixels.
        line: Line spacing of the stimulus text in pixels (the unit of every spatial constant).
        loader: Function returning (trials, positive label).
        aoi_points: Optional mapping from stimulus id to an array of AOI centres, for the AOI graph.
        perm_unit: The unit to which a label is assigned (labels are constant within a unit), as a function of a
            trial; permutation tests shuffle labels between units.
        perm_block: The exchangeability block, as a function of a trial; labels are shuffled only between units
            of the same block.
        perm_note: One-line description of the permutation scheme, for the methods table.
    """

    key: str
    dataset: str
    description: str
    label_level: str
    frame: Tuple[float, float]
    line: float
    loader: Callable[[], Tuple[List[Trial], str]]
    aoi_points: Optional[Callable[[], Dict[str, np.ndarray]]] = field(default=None)
    perm_unit: Callable[[Trial], Hashable] = field(default=lambda t: t.meta["trial_id"])
    perm_block: Callable[[Trial], Hashable] = field(default=lambda t: t.group)
    perm_note: str = "trial labels shuffled within participant"


# --------------------------------------------------------------------------------------------- SB-SAT
SB_FRAME = (1024.0, 768.0)
SB_LINE = 63.0


def _sbsat_fixations(page_type: str = "reading") -> pd.DataFrame:
    """Fixations of the first visit to every page of the given type, inside the display.

    Args:
        page_type: "reading" or "question".

    Returns:
        DataFrame with columns subj, book, page, trial, k, x, y, dur.
    """
    cols = ["RECORDING_SESSION_LABEL", "TRIAL_INDEX", "CURRENT_FIX_X", "CURRENT_FIX_Y", "CURRENT_FIX_DURATION",
            "type", "book_name", "page"]
    F = pd.read_csv(os.path.join(RAW, "sbsat_fix.csv"), usecols=cols)
    F = F[F["type"] == page_type].rename(columns={
        "RECORDING_SESSION_LABEL": "subj", "TRIAL_INDEX": "trial", "CURRENT_FIX_X": "x", "CURRENT_FIX_Y": "y",
        "CURRENT_FIX_DURATION": "dur", "book_name": "book"})
    F["k"] = F.groupby(["subj", "trial"]).cumcount()
    # a page can be revisited (participants paged back); keep the first visit to each page
    first = F.groupby(["subj", "book", "page"])["trial"].min().rename("first").reset_index()
    F = F.merge(first, on=["subj", "book", "page"])
    F = F[F["trial"] == F["first"]].drop(columns="first")
    inside = (F.x >= 0) & (F.x <= SB_FRAME[0]) & (F.y >= 0) & (F.y <= SB_FRAME[1])
    return F[inside]


def _to_trials(F: pd.DataFrame, keys: List[str], label_of: Callable[[tuple], Optional[str]],
               group_col: str, frame: Tuple[float, float], stim_of: Callable[[tuple], str]) -> List[Trial]:
    """Group a fixation table into Trial objects.

    Args:
        F: Fixation table with columns x, y, dur, k and the grouping keys.
        keys: Columns that identify one trial.
        label_of: Maps a key tuple to a class label, or None to drop the trial.
        group_col: Column holding the participant identifier (must be in ``keys``).
        frame: Display size in pixels.
        stim_of: Maps a key tuple to a stimulus identifier.

    Returns:
        Trials with at least MIN_FIX fixations.
    """
    gi = keys.index(group_col)
    out = []
    for key, g in F.groupby(keys, sort=True):
        lab = label_of(key)
        if lab is None or len(g) < MIN_FIX:
            continue
        g = g.sort_values("k")
        out.append(Trial(g.x.values, g.y.values, g.dur.values.astype(float), label=lab, group=str(key[gi]),
                         meta=dict(stimulus=stim_of(key), key=key), frame=frame))
    return out


def sbsat_native() -> Tuple[List[Trial], str]:
    """Native vs. non-native English readers on the reading pages (between-participant label).

    Returns:
        Trials and the positive label.
    """
    F = _sbsat_fixations()
    L = pd.read_csv(os.path.join(RAW, "sbsat_labels.csv"))
    native = L.groupby("subj")["native"].first().to_dict()
    trials = _to_trials(F, ["subj", "book", "page"], lambda k: "native" if native[k[0]] == 1 else "nonnative",
                        "subj", SB_FRAME, lambda k: f"{k[1]}-{k[2]}")
    return trials, "nonnative"


def sbsat_comprehension() -> Tuple[List[Trial], str]:
    """Reading pages of passages answered well (accuracy >= 0.8) vs. poorly (<= 0.4).

    Returns:
        Trials and the positive label.
    """
    F = _sbsat_fixations()
    L = pd.read_csv(os.path.join(RAW, "sbsat_labels.csv"))
    acc = {(r.subj, r.book): r.acc for r in L.itertuples()}

    def lab(k: tuple) -> Optional[str]:
        a = acc.get((k[0], k[1]))
        if a is None:
            return None
        return "high" if a >= 0.8 else ("low" if a <= 0.4 else None)

    trials = _to_trials(F, ["subj", "book", "page"], lab, "subj", SB_FRAME, lambda k: f"{k[1]}-{k[2]}")
    return trials, "low"


# ----------------------------------------------------------------------------------------------- EMIP
EM_FRAME = (1920.0, 1080.0)
EM_LINE = 37.0


def _emip_fixations(require_both: bool = True) -> pd.DataFrame:
    """EMIP corrected fixations.

    Args:
        require_both: If True, keep only participants with both programs in the same stimulus version.

    Returns:
        DataFrame with columns participant, program, version, code_file, k, x, y, dur and AOI geometry.
    """
    E = pd.read_csv(os.path.join(RAW, "emip_finalset_line_part.csv"))
    E = E.rename(columns={"x_cord": "x", "y_cord": "y", "duration": "dur"})
    stem = E["code_file"].str.replace(".jpg", "", regex=False)
    E["program"] = stem.str.split("_").str[0]
    E["version"] = stem.str.split("_").str[1]
    E = E.sort_values(["participant", "code_file", "timestamp"])
    E["k"] = E.groupby(["participant", "code_file"]).cumcount()
    if not require_both:
        return E
    both = E.groupby(["participant", "version"])["program"].nunique()
    keep = both[both == 2].reset_index()[["participant", "version"]]
    return E.merge(keep, on=["participant", "version"])


def emip_program() -> Tuple[List[Trial], str]:
    """Reading the Rectangle vs. the Vehicle program (within-participant; differs in stimulus layout).

    Returns:
        Trials and the positive label.
    """
    E = _emip_fixations()
    trials = _to_trials(E, ["participant", "code_file", "program"], lambda k: str(k[2]), "participant",
                        EM_FRAME, lambda k: str(k[1]))
    return trials, "vehicle"


def emip_halves() -> Tuple[List[Trial], str]:
    """First vs. second half of each program reading, by fixation index (within-participant).

    Returns:
        Trials and the positive label.
    """
    E = _emip_fixations()
    n = E.groupby(["participant", "code_file"])["k"].transform("size")
    r = E.groupby(["participant", "code_file"])["k"].rank(method="first") - 1
    # equal halves: with an odd number of fixations the middle one is dropped, so count cannot separate the halves
    E = E[(r < n // 2) | (r >= n - n // 2)]
    r = r.loc[E.index]
    n = n.loc[E.index]
    E = E.assign(half=np.where(r < n // 2, "first", "second"))
    trials = _to_trials(E, ["participant", "code_file", "half"], lambda k: str(k[2]), "participant",
                        EM_FRAME, lambda k: str(k[1]))
    return trials, "second"


def _emip_meta() -> pd.DataFrame:
    """EMIP participant metadata (Bednarik et al., 2020), indexed by participant id.

    Returns:
        The metadata table from ``emip_metadata.csv``.
    """
    return pd.read_csv(os.path.join(RAW, "emip_metadata.csv")).set_index("id")


def emip_expertise() -> Tuple[List[Trial], str]:
    """Self-rated high vs. low or no programming expertise (between-participant; "medium" excluded).

    Every corrected program reading of a participant is a trial, whether or not both programs are present.

    Returns:
        Trials and the positive label.
    """
    E = _emip_fixations(require_both=False)
    exp = _emip_meta()["expertise_programming"].to_dict()

    def lab(k: tuple) -> Optional[str]:
        e = exp.get(k[0])
        return "high" if e == "high" else ("low" if e in ("low", "none") else None)

    trials = _to_trials(E, ["participant", "code_file"], lab, "participant", EM_FRAME, lambda k: str(k[1]))
    return trials, "low"


def emip_correct(program: Optional[str] = None) -> Tuple[List[Trial], str]:
    """Program readings followed by a correct vs. an incorrect comprehension answer (mixed label).

    Pooled over both programs the label is confounded with the program (most Vehicle answers are wrong,
    most Rectangle answers right), so ``program`` restricts the problem to one program.

    Args:
        program: "rectangle" or "vehicle" to use one program only; None pools both.

    Returns:
        Trials and the positive label.
    """
    E = _emip_fixations(require_both=False)
    if program is not None:
        E = E[E["program"] == program]
    M = _emip_meta()

    def lab(k: tuple) -> Optional[str]:
        if k[0] not in M.index:
            return None
        c = M.loc[k[0], f"correct_{k[2]}"]
        return "correct" if c == 1 else "incorrect"

    trials = _to_trials(E, ["participant", "code_file", "program"], lab, "participant", EM_FRAME,
                        lambda k: str(k[1]))
    return trials, "incorrect"


def emip_aoi_points() -> Dict[str, np.ndarray]:
    """Centres of the token AOIs of every EMIP stimulus, from the AOI boxes in the corrected release.

    Returns:
        Mapping from code_file to an (n, 2) array of AOI centres in screen pixels.
    """
    E = pd.read_csv(os.path.join(RAW, "emip_finalset_line_part.csv"))
    A = E.drop_duplicates(["code_file", "aoi_x", "aoi_y", "aoi_width", "aoi_height"])
    out = {}
    for cf, g in A.groupby("code_file"):
        out[str(cf)] = np.stack([g.aoi_x + g.aoi_width / 2.0, g.aoi_y + g.aoi_height / 2.0], 1).astype(float)
    return out


# ------------------------------------------------------------------------------------------- GazeBase
GB_HW, GB_HH = 23.3, 15.1     # half-width and half-height of the display in degrees
GB_FRAME = (2 * GB_HW, 2 * GB_HH)
GB_UNIT = 2.0                 # reference scale in degrees (GazeBase has tasks without text lines)
GB_CSV = os.environ.get("SPO_GB_CSV", os.path.join(os.path.dirname(RAW), "derived", "gazebase_r1_all_fixations.csv"))


def _gb_select(F: pd.DataFrame, spec: str) -> List[Tuple[pd.DataFrame, str, str]]:
    """Fixations of one side of a GazeBase pair.

    Args:
        F: GazeBase fixation table (subject, session, task, k, x, y, dur), centred degrees.
        spec: A task ("TEX"), a task in one session ("TEX@S1"), or one half of a task ("VD1#1").

    Returns:
        List of (fixations, subject, session) per recording.
    """
    task, mode, arg = spec, "", ""
    if "@" in spec:
        task, arg = spec.split("@")
        mode = "session"
    if "#" in spec:
        task, arg = spec.split("#")
        mode = "half"
    out = []
    for (s, sess), g in F[F.task == task].groupby(["subject", "session"]):
        g = g.sort_values("k")
        if mode == "session" and sess != arg:
            continue
        if mode == "half":
            n = len(g)
            g = g.iloc[: n // 2] if arg == "1" else g.iloc[n - n // 2:]    # odd n: the middle fixation is dropped
        if len(g) >= MIN_FIX:
            out.append((g, str(s), str(sess)))
    return out


def gazebase(pair: str) -> Callable[[], Tuple[List[Trial], str]]:
    """Loader factory for one GazeBase Round-1 binary problem.

    Args:
        pair: Two specs separated by a colon, e.g. "TEX:VD1" or "VD1#1:VD1#2".

    Returns:
        A loader returning trials (coordinates in a (2*HW, 2*HH) degree frame) and the positive label.
    """
    def _load() -> Tuple[List[Trial], str]:
        F = pd.read_csv(GB_CSV)
        a, b = pair.split(":")
        trials = []
        for spec in (a, b):
            for g, subj, sess in _gb_select(F, spec):
                trials.append(Trial(g.x.values + GB_HW, g.y.values + GB_HH, g.dur.values.astype(float), label=spec,
                                    group=subj, meta=dict(stimulus=f"{spec}-{sess}", session=sess), frame=GB_FRAME))
        return trials, b
    return _load


def _participant(t: Trial) -> Hashable:
    """Permutation unit or block: the participant."""
    return t.group


def _everyone(t: Trial) -> Hashable:
    """Permutation block: all trials together."""
    return "all"


def _recording(t: Trial) -> Hashable:
    """Permutation block for halves of one recording: participant and session or program."""
    return (t.group, t.meta.get("session", t.meta.get("stimulus")))


def _program(t: Trial) -> Hashable:
    """Permutation block for pooled correctness: the program (Rectangle or Vehicle) of the trial."""
    return str(t.meta["stimulus"]).split("_")[0]


def _passage(t: Trial) -> Hashable:
    """Permutation unit for SB-SAT comprehension: the participant's passage (its pages share one label)."""
    return (t.group, t.meta["key"][1])


BETWEEN: Dict[str, Any] = dict(perm_unit=_participant, perm_block=_everyone, perm_note="participant labels shuffled across participants")
WITHIN: Dict[str, Any] = dict(perm_note="trial labels shuffled within participant")
HALVES: Dict[str, Any] = dict(perm_block=_recording, perm_note="half labels swapped within each recording")


PROBLEMS: Dict[str, Problem] = {
    "gb_tex_vd1": Problem("gb_tex_vd1", "GazeBase", "reading vs. video viewing", "within",
                          GB_FRAME, GB_UNIT, gazebase("TEX:VD1"), **WITHIN),
    "gb_vd1_blg": Problem("gb_vd1_blg", "GazeBase", "video viewing vs. ball-catching game", "within",
                          GB_FRAME, GB_UNIT, gazebase("VD1:BLG"), **WITHIN),
    "gb_vd1_vd2": Problem("gb_vd1_vd2", "GazeBase", "video segment 1 vs. 2", "within",
                          GB_FRAME, GB_UNIT, gazebase("VD1:VD2"), **WITHIN),
    "gb_tex_s1s2": Problem("gb_tex_s1s2", "GazeBase", "reading, session 1 vs. 2", "within",
                           GB_FRAME, GB_UNIT, gazebase("TEX@S1:TEX@S2"), **WITHIN),
    "gb_vd1_halves": Problem("gb_vd1_halves", "GazeBase", "video 1, first vs. second half", "within",
                             GB_FRAME, GB_UNIT, gazebase("VD1#1:VD1#2"), **HALVES),
    "sb_native": Problem("sb_native", "SB-SAT", "native vs. non-native English readers, reading pages", "between",
                         SB_FRAME, SB_LINE, sbsat_native, **BETWEEN),
    "sb_compr": Problem("sb_compr", "SB-SAT", "high vs. low passage comprehension, reading pages", "mixed",
                        SB_FRAME, SB_LINE, sbsat_comprehension, perm_unit=_passage,
                        perm_note="passage labels shuffled within participant (a passage's pages share its label)"),
    "em_program": Problem("em_program", "EMIP", "Rectangle vs. Vehicle program", "within",
                          EM_FRAME, EM_LINE, emip_program, emip_aoi_points, **WITHIN),
    "em_expertise": Problem("em_expertise", "EMIP", "high vs. low/no self-rated programming expertise", "between",
                            EM_FRAME, EM_LINE, emip_expertise, emip_aoi_points, **BETWEEN),
    "em_correct": Problem("em_correct", "EMIP", "correct vs. incorrect answer, both programs pooled (confounded)",
                          "mixed", EM_FRAME, EM_LINE, emip_correct, emip_aoi_points, perm_block=_program,
                          perm_note="trial labels shuffled within program"),
    "em_correctrect": Problem("em_correctrect", "EMIP", "correct vs. incorrect answer, Rectangle program only",
                              "between", EM_FRAME, EM_LINE, lambda: emip_correct("rectangle"), emip_aoi_points,
                              **BETWEEN),
    "em_correctveh": Problem("em_correctveh", "EMIP", "correct vs. incorrect answer, Vehicle program only",
                             "between", EM_FRAME, EM_LINE, lambda: emip_correct("vehicle"), emip_aoi_points,
                             **BETWEEN),
    "em_halves": Problem("em_halves", "EMIP", "first vs. second half of a program reading", "within",
                         EM_FRAME, EM_LINE, emip_halves, emip_aoi_points, **HALVES),
}
