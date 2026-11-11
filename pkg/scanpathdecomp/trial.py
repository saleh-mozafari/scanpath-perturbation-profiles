from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any, Optional
import numpy as np


@dataclass
class Trial:
    """One scanpath: time-ordered fixation coordinates and durations.

    x, y      : fixation centroids (any consistent unit: px or degrees)
    dur       : fixation durations (any consistent unit); may be None if unused
    label     : class label for the classification task (e.g. 'IS'/'DS', 'reading'/'video')
    group     : participant identifier; used for participant-wise cross-validation and
                participant-clustered bootstrap. Required.
    meta      : anything else (stimulus id, session, ...)
    frame     : (width, height) of the stimulus frame, used by the anchoring null and by
                grid-based descriptors. If None, the anchoring null rotates about the
                centroid of the trial's own fixations.
    """
    x: np.ndarray
    y: np.ndarray
    dur: Optional[np.ndarray]
    label: Any
    group: Any
    meta: dict = field(default_factory=dict)
    frame: Optional[tuple] = None

    def __post_init__(self):
        self.x = np.asarray(self.x, dtype=float)
        self.y = np.asarray(self.y, dtype=float)
        if self.dur is not None:
            self.dur = np.asarray(self.dur, dtype=float)
            assert len(self.dur) == len(self.x)
        assert len(self.x) == len(self.y)

    def __len__(self):
        return len(self.x)

    def replace(self, **kw) -> "Trial":
        d = dict(x=self.x, y=self.y, dur=self.dur, label=self.label, group=self.group,
                 meta=self.meta, frame=self.frame)
        d.update(kw)
        return Trial(**d)
