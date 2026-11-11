"""Profile plot: sensitivity to each non-jitter transformation (drop in AUC) with 95% CI, plus the
jitter curve (AUC retained as a function of jitter scale)."""
from __future__ import annotations
import numpy as np
import matplotlib.pyplot as plt

LABELS = {"order": "order", "duration_coupling": "duration\ncoupling", "duration_marginal": "duration\nmarginal",
          "rotation": "rotation", "duration": "duration", "anchoring": "anchoring"}


def profile(results: dict, ax=None, colors=None):
    """results: {descriptor_name: DecompositionResult}. Bars = sensitivity to every non-jitter
    transformation present in the results (order, duration coupling, duration marginal, rotation);
    line panel = jitter curve."""
    names = list(results)
    first = results[names[0]].contributions
    comps = [c for c in first.component if not str(c).startswith("resolution")]
    fig, axs = plt.subplots(1, 2, figsize=(9.6, 3.6), gridspec_kw=dict(width_ratios=[1.1, 1]))
    ax = axs[0]; w = 0.8 / len(names)
    for k, nm in enumerate(names):
        c = results[nm].contributions.set_index("component")
        vals = [c.loc[x, "contribution"] if x in c.index else np.nan for x in comps]
        lo = [c.loc[x, "contribution"] - c.loc[x, "ci_lo"] if x in c.index else 0 for x in comps]
        hi = [c.loc[x, "ci_hi"] - c.loc[x, "contribution"] if x in c.index else 0 for x in comps]
        ax.bar(np.arange(len(comps)) + k * w, vals, w, yerr=[lo, hi], capsize=3, label=nm,
               color=None if colors is None else colors[k])
    ax.axhline(0, color="k", lw=0.8); ax.set_xticks(np.arange(len(comps)) + w * (len(names) - 1) / 2)
    ax.set_xticklabels([LABELS.get(x, x) for x in comps], fontsize=8)
    ax.set_ylabel(r"sensitivity ($\mathrm{AUC}_{obs} - \mathrm{AUC}_{pert}$)")
    ax.legend(fontsize=8, frameon=True)
    ax = axs[1]
    for k, nm in enumerate(names):
        t = results[nm].table
        r = t[t.condition.str.startswith("resolution")]
        s = [float(c.split("s=")[1].rstrip(")")) for c in r.condition]
        ax.errorbar(s, r.auc_mean, yerr=r.auc_sd, marker="o", label=nm, capsize=2,
                    color=None if colors is None else colors[k])
        ax.axhline(t.loc[t.condition == "observed", "auc_mean"].item(), ls=":", lw=1,
                   color=None if colors is None else colors[k])
    ax.set_xscale("log"); ax.set_xlabel("jitter scale s"); ax.set_ylabel("AUC retained")
    ax.axhline(0.5, color="gray", ls=":", lw=1); ax.grid(alpha=0.3)
    fig.tight_layout()
    return fig
