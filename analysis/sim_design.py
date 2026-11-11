"""Simulation study: does the decomposition recover a PLANTED signal and stay flat when there is none?

Generative model. n_groups participants x per_group trials, half labelled A, half B. Both default to the
values used throughout the published campaign (20 x 8) and are arguments so that the calibration can be
repeated at the sample sizes of the analysed datasets (39 participants x 10 trials; 60 x 4) -- the
estimand depends on the participant count, so a calibration at one count does not transfer to another. Each trial has
n fixations in a 600x500 frame. Participants carry a random offset (individual differences). Five
scenarios plant a signal in exactly one component:

  none       : A and B are identically distributed                        -> every contribution ~ 0
  order      : same location multiset; B visits its locations in a left-to-right sweep, A in random order
  duration   : same locations & order; in B, durations increase along the trial, in A they are random
  anchoring  : same intrinsic geometry; B's cloud is centred top-left, A's bottom-right
  scale(sig) : both classes revisit 4 clusters; B returns within SD sig, A within 3*sig (suffix "w": 1.35*sig, moderate AUC;
               suffix "s": cluster centres uniform in a centred disc, so the design is exactly rotation-symmetric)

Each scenario is evaluated with a descriptor that can read the planted component (and, as a
control, one that cannot), over `n_datasets` independent synthetic datasets. Reported: mean and SD
of every contribution, the fraction of datasets whose 95% CI excludes 0 (power for planted
components, false-positive rate for the others), and for the scale scenario the recovered knee.
Output: sim_results.csv, sim_summary.csv

Everything below the docstring is importable: make_dataset(), DESC and PLAN are used by
simulation_fpr.py, simulation_direct.py and notebooks/03_simulation_recovery.ipynb, and run() is what
both the command line and the notebooks call, so there is one code path.
Usage: python simulation_study.py outdir n_datasets scenarios offset n_reps
Output: sim_results_<scenarios>_<offset>.csv and the printed summary."""
import sys, os, numpy as np, pandas as pd, warnings
warnings.filterwarnings("ignore")
from scanpathdecomp import Trial, decompose, descriptors

W, H, NG, PG, NF = 600, 500, 20, 8, 40        # frame, participants, trials per participant, fixations per trial
DEFAULT_SCENARIOS = ["none", "order", "duration", "anchoring", "scale20", "scale60"]


def make_dataset(scenario, seed, n_groups=NG, per_group=PG, offset_sd=25.0, scale_sd=0.0):
    """One synthetic dataset. A trailing "n" on the scenario name removes the participant random effect
    (the draw is still made, so the two designs share a random-number stream and differ only by the offset).
    Without the offset the trials of a participant are i.i.d., so a design whose per-trial law is invariant
    under rotation about the frame centre has a JOINT law that is invariant too, and the true sensitivity to
    an independently drawn per-trial rotation is exactly zero for a participant-held-out learner. With the
    offset, only the per-trial law is invariant; see notebooks/04 and the reference simulation.
    offset_sd sets the SD of the participant offset; scale_sd > 0 adds a participant-specific scaling of the
    scanpath about the frame centre by exp(N(0, scale_sd)) (strong participant heterogeneity). Scaling about the
    frame centre commutes with rotation about it, so it keeps the rotation invariance of a symmetric design."""
    no_offset = scenario.endswith("n") and scenario not in ("none", "duration")
    if no_offset:
        scenario = scenario[:-1]
    rng = np.random.default_rng(seed); trials = []
    for g in range(n_groups):
        off = rng.normal(0, 25, 2) * (offset_sd / 25.0)              # participant random effect
        if no_offset:
            off = np.zeros(2)
        gscale = float(np.exp(rng.normal(0, scale_sd))) if scale_sd > 0 else 1.0
        for k in range(per_group):
            lab = "A" if k % 2 == 0 else "B"
            if scenario == "none":
                P = rng.uniform([0, 0], [W, H], (NF, 2)); d = rng.uniform(100, 400, NF)
            elif scenario == "order":
                P = rng.uniform([0, 0], [W, H], (NF, 2)); d = rng.uniform(100, 400, NF)
                if lab == "B": P = P[np.argsort(P[:, 0] + rng.normal(0, 40, NF))]   # noisy left-to-right sweep
            elif scenario == "orderw":           # weak sweep: the left-to-right sort is much noisier
                P = rng.uniform([0, 0], [W, H], (NF, 2)); d = rng.uniform(100, 400, NF)
                if lab == "B": P = P[np.argsort(P[:, 0] + rng.normal(0, 150, NF))]
            elif scenario == "negorder":         # amplitudes match in the observed order but not after shuffling:
                d = rng.uniform(100, 400, NF)    # A sweeps a wide field in order, B visits a narrow field at random
                if lab == "A":
                    P = rng.uniform([0, 0], [W, H], (NF, 2)); P = P[np.argsort(P[:, 0] + rng.normal(0, 40, NF))]
                else:
                    P = np.array([W / 2, H / 2]) + rng.uniform(-172.5, 172.5, (NF, 2))
            elif scenario == "duration":
                P = rng.uniform([0, 0], [W, H], (NF, 2)); d = rng.uniform(100, 400, NF)
                if lab == "B": d = np.sort(d)                                          # durations grow along the trial
            elif scenario == "anchoring":
                c = np.array([150, 125]) if lab == "B" else np.array([450, 375])
                P = c + rng.normal(0, 60, (NF, 2)); d = rng.uniform(100, 400, NF)
            elif scenario == "durmarg":          # class-dependent duration MARGINAL, same locations/order
                P = rng.uniform([0, 0], [W, H], (NF, 2)); d = rng.uniform(100, 400, NF) if lab == "A" else rng.uniform(220, 520, NF)
            elif scenario == "eccentric":        # B central, A peripheral; rotation preserves eccentricity
                if lab == "B":
                    P = np.array([W / 2, H / 2]) + rng.normal(0, 50, (NF, 2))
                else:
                    th = rng.uniform(0, 2 * np.pi, NF); rad = rng.normal(190, 20, NF)
                    P = np.stack([W / 2 + rad * np.cos(th), H / 2 + rad * np.sin(th)], 1)
                d = rng.uniform(100, 400, NF)
            elif scenario.startswith("scale"):
                sym = scenario.endswith("s"); s_ = scenario.rstrip("s")           # "...s": rotation-symmetric design
                weak = s_.endswith("w"); sig = float(s_[5:].rstrip("w"))
                if sym:   # cluster centres uniform in a disc of radius 150 about the frame centre: the class-conditional
                          # distribution is then exactly invariant to rotation/reflection about that centre (and order is
                          # exchangeable), so the TRUE sensitivity to rotation and order is exactly zero for any descriptor
                    rad = 150 * np.sqrt(rng.uniform(0, 1, 4)); th = rng.uniform(0, 2 * np.pi, 4)
                    cen = np.stack([W / 2 + rad * np.cos(th), H / 2 + rad * np.sin(th)], 1)
                else:
                    cen = rng.uniform([100, 100], [W - 100, H - 100], (4, 2))
                sd = sig if lab == "B" else (1.35 if weak else 3.0) * sig     # B returns tightly, A loosely (weak: moderate AUC)
                P = cen[rng.integers(0, 4, NF)] + rng.normal(0, sd, (NF, 2))
                d = rng.uniform(100, 400, NF)
            else:
                raise ValueError(scenario)
            if gscale != 1.0:
                P = np.array([W / 2, H / 2]) + gscale * (P - np.array([W / 2, H / 2]))
            P = P + off
            trials.append(Trial(P[:, 0], P[:, 1], d, lab, f"P{g:02d}", frame=(W, H)))
    return trials


def duration_trend():
    """Reads duration STRUCTURE (not just the mean): correlation of duration with fixation index."""
    def _f(t):
        if len(t) < 5: return None
        r = np.corrcoef(np.arange(len(t)), t.dur)[0, 1]
        return dict(dur_trend=float(0.0 if np.isnan(r) else r), dur_mean=float(t.dur.mean()))
    _f.__name__ = "duration_trend"; return _f


def position():
    """Reads absolute position (stimulus-anchored)."""
    def _f(t): return dict(mx=float(t.x.mean()), my=float(t.y.mean()))
    _f.__name__ = "position"; return _f


def mean_duration():
    """Reads only the duration MARGINAL (mean and SD)."""
    def _f(t): return dict(dmean=float(t.dur.mean()), dsd=float(t.dur.std()))
    _f.__name__ = "mean_duration"; return _f


def eccentricity():
    """Reads distance from the frame centre (preserved by rotation about that centre)."""
    def _f(t):
        cx, cy = (t.frame[0] / 2, t.frame[1] / 2) if t.frame else (t.x.mean(), t.y.mean())
        r = np.sqrt((t.x - cx) ** 2 + (t.y - cy) ** 2); return dict(ecc_mean=float(r.mean()), ecc_sd=float(r.std()))
    _f.__name__ = "eccentricity"; return _f


DESC = {"rqa": descriptors.rqa(40), "directed": descriptors.directed_graph((60, 50)), "classic": descriptors.classic(),
        "duration_trend": duration_trend(), "position": position(), "mean_duration": mean_duration(), "eccentricity": eccentricity()}
# which descriptor is the "reader" for each scenario, and which is the invariant control
PLAN = {"none": ["rqa", "directed"],
        "order": ["directed", "rqa"], "duration": ["duration_trend", "rqa"], "durmarg": ["mean_duration", "duration_trend"],
        "anchoring": ["position", "rqa"], "eccentric": ["eccentricity", "position"], "scale20": ["rqa"], "scale60": ["rqa"], "scale20w": ["rqa", "directed"],
        "scale20ws": ["rqa", "directed"], "scale20s": ["rqa", "directed"],
        "scale20wsn": ["rqa", "directed"], "scale20sn": ["rqa", "directed"], "nonen": ["rqa", "directed"]}
SCALES = (5, 10, 20, 40, 80, 160, 320)


def summarise(R):
    """Mean and SD of every sensitivity and the fraction of datasets whose 95% interval excludes zero."""
    return R.groupby(["scenario", "descriptor", "component"]).agg(
        auc_obs=("auc_obs", "mean"), contrib_mean=("contribution", "mean"), contrib_sd=("contribution", "std"),
        frac_ci_excl0=("excludes0", "mean"), n=("dataset", "count")).reset_index()


def run(out=".", n_datasets=20, scenarios=None, offset=0, n_reps=5, save=True, verbose=True):
    """Recovery simulation. Returns (per-dataset DataFrame, summary DataFrame)."""
    scenarios = list(scenarios) if scenarios else list(DEFAULT_SCENARIOS)
    rows = []
    for sc in scenarios:
        for ds in range(offset, offset + n_datasets):
            trials = make_dataset(sc, 1000 + ds)
            for dn in PLAN[sc]:
                r = decompose(trials, DESC[dn], positive="B", n_reps=n_reps, n_boot=60, seed=ds, resolution_scales=SCALES)
                obs = r.table.loc[r.table.condition == "observed", "auc_mean"].item()
                for _, c in r.contributions.iterrows():
                    rows.append(dict(scenario=sc, dataset=ds, descriptor=dn, auc_obs=obs, component=c.component,
                                     contribution=c.contribution, ci_lo=c.ci_lo, ci_hi=c.ci_hi,
                                     excludes0=bool(c.ci_lo > 0 or c.ci_hi < 0)))
            if verbose:
                print(sc, ds, flush=True)
    R = pd.DataFrame(rows); S = summarise(R)
    if save:
        os.makedirs(out, exist_ok=True)
        R.to_csv(os.path.join(out, f"sim_results_{'_'.join(scenarios)}_{offset}.csv"), index=False)
    if verbose:
        print(S.round(3).to_string(index=False))
    return R, S


if __name__ == "__main__":
    run(out=sys.argv[1] if len(sys.argv) > 1 else ".",
        n_datasets=int(sys.argv[2]) if len(sys.argv) > 2 else 20,
        scenarios=sys.argv[3].split(",") if len(sys.argv) > 3 else None,
        offset=int(sys.argv[4]) if len(sys.argv) > 4 else 0,
        n_reps=int(sys.argv[5]) if len(sys.argv) > 5 else 5)
