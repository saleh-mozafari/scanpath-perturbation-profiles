# scanpathdecomp

**What do scanpath descriptors use?**

A perturbation profile for trial-level scanpath descriptors. Given any descriptor
(recurrence quantification, transition-graph features, entropies, classic statistics, your own)
and any participant-labelled binary task, the package re-evaluates the descriptor, with the
classifier refit under leave-one-participant-out cross-validation, after each of five
transformations. Each transformation destroys a stated property of every scanpath and
preserves stated others; what it does **not** remove is listed, because the names are not
enough:

| transformation | destroys | preserves | not removed |
|---|---|---|---|
| `order` | serial order of fixations | multisets of locations and durations | — |
| `duration_coupling` | which fixation carries which duration | locations, order, the trial's duration distribution (mean, spread, total dwell, count) | marginal duration information |
| `duration_marginal(pool)` | the trial's duration distribution (and the coupling) | locations, order, count | — |
| `rotation` | orientation and side (rigid rotation/reflection about a centre) | all pairwise distances/angles, order, durations, each fixation's eccentricity | eccentricity; any centre-symmetric location contrast |
| `resolution(s)` | spatial structure finer than *s* (Gaussian jitter; coincident points end ≈ s√π apart) | order, durations, coarse density | — |

The drop in participant-held-out AUC under a transformation is the descriptor's
**sensitivity** to that property, with a participant bootstrap that refits the classifier
(`ci_method="refit"`, Monte-Carlo-corrected and conservative; or `"refit_direct"`, which averages all R
replicates inside every draw at R times the cost and is closer to nominal coverage). Exact invariances are reported
as 0 with no interval; a non-positive corrected variance falls back to the conservative paired variance and is flagged. Sensitivities are descriptor- and learner-specific, need not add
up to the total signal, and a zero is ambiguous between invariance, redundancy, precision and
information the transformation leaves intact. Sweeping *s* gives a jitter-sensitivity curve,
not a spatial scale of behaviour. `metric="within"` evaluates the mean within-participant AUC
instead of the pooled AUC. (`decompose()` keeps its name for backward compatibility; the
result tables call the quantity `contribution` for the same reason.)

```python
from scanpathdecomp import Trial, decompose, descriptors

trials = [Trial(x, y, dur, label="native", group="P01", frame=(1024, 768)), ...]
res = decompose(trials, descriptors.rqa(radius=60), n_reps=20)
print(res)
```

The approach unifies and extends the perturbation-based validation of Dewhurst et al.
(2012) and Fahimi & Bruce (2021, *Behavior Research Methods*): those studies perturbed
scanpaths to characterise the *robustness of similarity metrics*; here the outcome is
*participant-held-out decoding of a behavioural label after refitting*, the transformations are
stated with their invariants, and the result is a per-descriptor, per-task profile. No priority
is claimed for any individual transformation.

## Descriptors

Built in: `descriptors.rqa(radius)`, `descriptors.directed_graph(tile)`, `descriptors.classic()`, and
`fag.fixation_adjacency_graph(triangulations)`, the undirected Fixation Adjacency Graph. Its triangulation can be
built from a stimulus mask (`StimulusTriangulation(mask)`, needs `pip install -e .[fag]`), from any set of anchor
points such as the centres of a stimulus's areas of interest (`StimulusTriangulation.from_points(points)`), or on a
stimulus-independent lattice that covers the frame under rotation (`StimulusTriangulation.lattice(frame, spacing)`).
For a descriptor with bounded spatial support, pass `rotation_center=` to `decompose()` so it rotates about the
support's centre.

## Evaluation

`decompose()` evaluates leave-one-participant-out by default. With many participants, `cv_folds=k` switches every
evaluation (observed, transformed and bootstrap) to k-fold cross-validation over participants, which is much faster.

## Install

```
pip install -e .[fag]
pytest
```

## Citation

Mozaffari, S., & Nadi, A. (in preparation). Perturbation profiles of scanpath classifiers: what reading,
code-comprehension and video decoders are sensitive to.

MIT licence.
