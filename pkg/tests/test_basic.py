import numpy as np
from scanpathdecomp import Trial, decompose, descriptors, nulls


def _synthetic(n_groups=12, per_group=6, seed=0):
    """Two classes: 'A' scanpaths revisit a few spots (high recurrence); 'B' wander.
    Order and duration carry no information by construction."""
    rng = np.random.default_rng(seed); trials = []
    for g in range(n_groups):
        for k in range(per_group):
            lab = "A" if k % 2 == 0 else "B"
            n = 40
            if lab == "A":
                centers = rng.uniform(100, 500, (4, 2))
                pts = centers[rng.integers(0, 4, n)] + rng.normal(0, 8, (n, 2))
            else:
                pts = rng.uniform(0, 600, (n, 2)) * [1, 500 / 600]
            trials.append(Trial(pts[:, 0], pts[:, 1], rng.uniform(100, 400, n), lab, f"P{g:02d}", frame=(600, 500)))
    return trials


def test_nulls_preserve_invariants():
    rng = np.random.default_rng(1)
    t = _synthetic(2, 2)[0]
    o = nulls.order(t, rng)
    assert sorted(o.x) == sorted(t.x) and sorted(o.dur) == sorted(t.dur)
    d = nulls.duration_coupling(t, rng)
    assert np.allclose(d.x, t.x) and sorted(d.dur) == sorted(t.dur)
    a = nulls.rotation(t, rng)
    D0 = np.sqrt((t.x[:, None] - t.x[None]) ** 2 + (t.y[:, None] - t.y[None]) ** 2)
    D1 = np.sqrt((a.x[:, None] - a.x[None]) ** 2 + (a.y[:, None] - a.y[None]) ** 2)
    assert np.allclose(D0, D1)            # rigid: all pairwise distances preserved
    r = nulls.resolution(5.0)(t, rng)
    assert np.abs(r.x - t.x).mean() < 15


def test_decomposition_recovers_known_structure():
    trials = _synthetic()
    res = decompose(trials, descriptors.rqa(radius=40), n_reps=5, n_boot=40, seed=0,
                    resolution_scales=(10, 120, 400))
    c = res.contributions.set_index("component")
    assert res.table.loc[res.table.condition == "observed", "auc_mean"].item() > 0.9
    assert abs(c.loc["order", "contribution"]) < 0.05        # order carries nothing
    assert abs(c.loc["duration_coupling", "contribution"]) < 0.05   # RQA ignores durations
    assert abs(c.loc["duration_marginal", "contribution"]) < 0.05
    assert abs(c.loc["rotation", "contribution"]) < 0.05            # RQA is rotation-invariant
    assert c.loc["resolution(s=120)", "contribution"] > 0.1  # fine returns carry part of the signal
    assert c.loc["resolution(s=400)", "contribution"] > c.loc["resolution(s=120)", "contribution"]  # monotone in s


def test_rqa_rotation_exact_with_pairs_at_the_radius():
    """Integer points with pairs exactly at the radius (a 36-48-60 triangle) must give identical recurrence
    features after any rigid rotation; before 0.2.0 floating-point error flipped such pairs."""
    import numpy as np
    from scanpathdecomp import Trial, descriptors, nulls
    x = np.array([0, 36, 36, 0, 100, 136, 20, 56], float)
    y = np.array([0, 48, 0, 60, 100, 148, 10, 58], float)
    t = Trial(x, y, np.full(8, 200.0), label="A", group="p", frame=(600, 500))
    f = descriptors.rqa(60.0, min_len=5)
    ref = f(t)
    rng = np.random.default_rng(0)
    for mode in ("rotation", "proper", "dihedral"):
        for _ in range(200):
            assert f(nulls.rotation(t, rng, mode=mode)) == ref
