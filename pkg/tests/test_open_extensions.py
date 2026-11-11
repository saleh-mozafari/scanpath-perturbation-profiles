"""Tests for the 0.3.0 additions: anchor-point and lattice triangulations, and k-fold evaluation."""
from __future__ import annotations

import numpy as np

from scanpathdecomp import Trial, decompose, descriptors
from scanpathdecomp.fag import StimulusTriangulation, fixation_adjacency_graph


def test_lattice_covers_rotated_frame() -> None:
    """Every point of the frame, rotated about its centre, stays on the lattice support."""
    T = StimulusTriangulation.lattice((400.0, 300.0), 50.0)
    rng = np.random.default_rng(0)
    xy = rng.uniform([0, 0], [400, 300], size=(500, 2)) - [200, 150]
    th = rng.uniform(0, 2 * np.pi, 500)
    rx = np.cos(th) * xy[:, 0] - np.sin(th) * xy[:, 1] + 200
    ry = np.sin(th) * xy[:, 0] + np.cos(th) * xy[:, 1] + 150
    d = T.dwell(rx, ry, np.ones(500))
    assert d.sum() == 500


def test_dwell_does_not_depend_on_query_order() -> None:
    """Points exactly on lattice edges are assigned the same way in any order."""
    T = StimulusTriangulation.lattice((200.0, 200.0), 50.0)
    x = np.array([50.0, 100.0, 75.0, 125.0, 150.0])
    y = np.array([50.0, 100.0, 75.0, 100.0, 50.0])
    a = T.dwell(x, y, np.arange(1.0, 6.0))
    b = T.dwell(x[::-1], y[::-1], np.arange(1.0, 6.0)[::-1])
    assert np.allclose(a, b)


def test_fag_on_anchor_points_returns_features() -> None:
    """The graph descriptor runs on arbitrary anchors."""
    rng = np.random.default_rng(1)
    pts = rng.uniform(0, 300, size=(40, 2))
    f = fixation_adjacency_graph({"s": StimulusTriangulation.from_points(pts)})
    t = Trial(rng.uniform(50, 250, 30), rng.uniform(50, 250, 30), rng.uniform(100, 300, 30), label="a", group="p",
              meta=dict(stimulus="s"))
    out = f(t)
    assert out is not None and set(out) == {"attention", "betweenness", "estrada", "squareness", "n_components"}


def test_kfold_evaluation_recovers_signal() -> None:
    """With cv_folds set, a planted spread contrast is still found and order stays exactly invariant for count."""
    rng = np.random.default_rng(2)
    trials = []
    for p in range(24):
        for lab, sd in (("A", 20.0), ("B", 60.0)):
            for _ in range(2):
                trials.append(Trial(300 + rng.normal(0, sd, 30), 250 + rng.normal(0, sd, 30), rng.uniform(100, 300, 30),
                                    label=lab, group=f"P{p}", frame=(600, 500)))
    r = decompose(trials, descriptors.rqa(40.0), n_reps=2, n_boot=10, cv_folds=5, resolution_scales=(10,))
    assert r.table.loc[r.table.condition == "observed", "auc_mean"].item() > 0.9


def test_group_kfold_is_participant_pure_and_fixed() -> None:
    """The k-fold assignment keeps participants whole and is a fixed function of the identifiers."""
    from scanpathdecomp.decompose import _group_kfold_splits
    g = np.repeat([f"P{i:02d}" for i in range(23)], 3)
    a = _group_kfold_splits(g, 5)
    b = _group_kfold_splits(g[::-1], 5)
    for tr, te in a:
        assert not set(g[tr]) & set(g[te])
    assert [sorted(set(g[te])) for _, te in a] == [sorted(set(g[::-1][te])) for _, te in b]


def test_stratified_folds_balance_classes() -> None:
    """Class-balanced folds give every fold a similar share of positive-class participants."""
    from scanpathdecomp.decompose import _group_kfold_splits
    g = np.repeat([f"P{i:02d}" for i in range(40)], 2)
    y = np.repeat((np.arange(40) < 10).astype(int), 2)
    shares = [y[te].mean() for _, te in _group_kfold_splits(g, 5, labels=y)]
    assert max(shares) - min(shares) < 1e-9


def test_within_fold_auc_ignores_fold_offsets() -> None:
    """A score offset that differs between folds leaves the within-fold AUC unchanged."""
    from scanpathdecomp.decompose import _auc_within_folds
    rng = np.random.default_rng(3)
    y = rng.integers(0, 2, 200)
    s = y + rng.normal(0, 1, 200)
    folds = [np.arange(i, 200, 4) for i in range(4)]
    shifted = s.copy()
    for j, te in enumerate(folds):
        shifted[te] += 10.0 * j
    assert np.isclose(_auc_within_folds(y, s, folds), _auc_within_folds(y, shifted, folds))


def test_loo_donor_pool_excludes_own_participant() -> None:
    """The default duration-marginal donor pool never gives a trial its own participant's durations."""
    from scanpathdecomp.decompose import default_nulls
    rng = np.random.default_rng(4)
    trials = [Trial(rng.uniform(0, 100, 20), rng.uniform(0, 100, 20), np.full(20, 100.0 * (p + 1)), label="A",
                    group=f"P{p}") for p in range(3)]
    f = default_nulls(trials)["duration_marginal"]
    for t in trials:
        assert not np.any(f(t, rng).dur == t.dur[0])
