"""scanpathdecomp — what does a scanpath descriptor use?

A perturbation profile for trial-level scanpath descriptors. Each perturbation destroys one
property of the scanpath (fixation order, duration-position coupling, the duration marginal,
orientation/side relative to the stimulus, or spatial structure finer than a scale s) and
preserves stated others. Re-evaluating a descriptor + participant-held-out classifier under each
perturbation measures how sensitive its discrimination is to that property. Contributions are
not additive shares of information.

Typical use
-----------
>>> from scanpathdecomp import Trial, decompose, descriptors
>>> trials = [Trial(x, y, dur, label='IS', group='P01'), ...]
>>> res = decompose(trials, descriptors.rqa(radius=60), n_reps=20)
>>> print(res.table)      # observed AUC, AUC under each null, contribution with 95% CI
"""
from .trial import Trial
from .nulls import order, duration_coupling, duration_marginal, duration_marginal_loo, rotation, resolution, NULL_REGISTRY, duration, anchoring
from .decompose import decompose, DecompositionResult, default_nulls
from . import descriptors

__all__ = ["Trial", "order", "duration_coupling", "duration_marginal", "duration_marginal_loo", "rotation",
           "resolution", "NULL_REGISTRY", "duration", "anchoring",
           "decompose", "DecompositionResult", "default_nulls", "descriptors"]
__version__ = "0.6.2"
