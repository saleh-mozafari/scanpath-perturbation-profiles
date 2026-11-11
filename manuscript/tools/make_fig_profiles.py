"""Figure: three representative perturbation profiles from the real-data analyses.

(a) GazeBase video vs game: classic descriptor against the dispersion baseline (a negative order sensitivity).
(b) GazeBase first vs second half of a video: recurrence at 1u, within-fold AUC against pooled AUC.
(c) SB-SAT native vs non-native readers: recurrence at 1u, within-fold AUC against pooled AUC.

Each panel shows the four non-spatial-scale transformations as points and the jitter sweep as a curve against
scale in units of the reference scale; exact invariances are drawn as open diamonds at zero.

Usage:
    python make_fig_profiles.py RESULTS_DIR OUT_PDF
"""
from __future__ import annotations

import glob
import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

C1, C2 = "#2a78d6", "#eb6834"      # validated categorical pair (CVD dE 24.7)
INK, MUTED = "#222222", "#888888"
FIXED = ["order", "duration_coupling", "duration_marginal", "rotation"]
LABELS = ["order", "dur.\ncoupl.", "dur.\nmarg.", "rot."]


def load(res: str, folder: str, problem: str, desc: str) -> pd.DataFrame:
    """Load one profile.

    Args:
        res: Results directory.
        folder: Profile folder name.
        problem: Problem key.
        desc: Descriptor key.

    Returns:
        The profile table with a scale column in line units for the jitter rows.
    """
    f = glob.glob(os.path.join(res, folder, f"prof_{problem}_{desc}_R6b60_s42.csv"))[0]
    return pd.read_csv(f)


def matched(res: str, problem: str, line: float) -> pd.DataFrame:
    """Load the support-matched AOI-graph profile.

    Args:
        res: Results directory.
        problem: Problem key.
        line: Line spacing in px.

    Returns:
        Table with component names stripped of the [matched] tag and a scale column.
    """
    M = pd.read_csv(os.path.join(res, "support", f"matched_{problem}_R6b60_s42.csv"))
    M["component"] = M.component.str.replace("[matched]", "", regex=False)
    M["scale_lines"] = M.component.str.extract(r"s=([0-9.]+)")[0].astype(float) / line
    M["exact_invariant"] = False
    return M


def draw(ax: plt.Axes, P: pd.DataFrame, color: str, dx: float, label: str, fixed: bool = True) -> None:
    """Draw one profile: points for the fixed transformations and a curve for the jitter sweep.

    Args:
        ax: Axes.
        P: Profile table.
        color: Series colour.
        dx: Horizontal offset, so two series do not overlap.
        label: Legend label.
        fixed: Whether to draw the fixed transformations.
    """
    if fixed:
        for i, c in enumerate(FIXED):
            r = P[P.component == c]
            if r.empty:
                continue
            r = r.iloc[0]
            if bool(r.exact_invariant):
                ax.plot(i + dx, 0, marker="D", mfc="white", mec=color, ms=7, mew=1.5, zorder=3)
            else:
                ax.errorbar(i + dx, r.contribution, yerr=[[r.contribution - r.ci_lo], [r.ci_hi - r.contribution]],
                            fmt="o", color=color, ms=6, lw=1.6, capsize=0, zorder=3)
    J = P[P.component.str.startswith("resolution")].sort_values("scale_lines")
    xs = [5 + k + dx for k in range(len(J))]
    ax.fill_between(xs, J.ci_lo, J.ci_hi, color=color, alpha=0.15, lw=0)
    ax.plot(xs, J.contribution, "-o", color=color, ms=5, lw=2, label=label, zorder=3)


def panel(ax: plt.Axes, title: str, series: list) -> None:
    """Draw a panel with its axes furniture.

    Args:
        ax: Axes.
        title: Panel title.
        series: List of (table, colour, offset, label, draw fixed) tuples.
    """
    for s in series:
        draw(ax, *s)
    ax.axhline(0, color=MUTED, lw=0.8, zorder=1)
    ax.axvline(4.5, color="#dddddd", lw=0.8, zorder=1)
    ax.set_xticks(list(range(4)) + [5, 6, 7, 8])
    ax.set_xticklabels(LABELS + ["¼", "½", "1", "2"], fontsize=8)
    ax.text(6.5, -0.13, "jitter SD (units of $u$)", transform=ax.get_xaxis_transform(), ha="center", va="top",
            fontsize=8, color=INK)
    ax.set_title(title, fontsize=9.5, loc="left", color=INK)
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(colors=INK, labelsize=8)
    ax.grid(axis="y", color="#eeeeee", lw=0.6)
    ax.legend(fontsize=7.5, frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=2)


def main() -> None:
    """Build and save the figure."""
    res, out = sys.argv[1], sys.argv[2]
    fig, axes = plt.subplots(1, 3, figsize=(12, 4.2), sharey=False)
    a = load(res, "profiles_v5", "gb_vd1_blg", "classic")
    b = load(res, "profiles_v5", "gb_vd1_blg", "dispersion")
    panel(axes[0], "(a) GazeBase: video vs game", [(a, C1, -0.12, "classic", True), (b, C2, 0.12, "dispersion", True)])
    c = load(res, "profiles_v5", "gb_vd1_halves", "rqa_line")
    d = load(res, "profiles_v5_pooled", "gb_vd1_halves", "rqa_line")
    panel(axes[1], "(b) GazeBase: first vs second half, recurrence (1u)",
          [(c, C1, -0.12, "within-fold AUC", True), (d, C2, 0.12, "pooled AUC", True)])
    e = load(res, "profiles_v5", "sb_native", "rqa_line")
    f = load(res, "profiles_v5_pooled", "sb_native", "rqa_line")
    panel(axes[2], "(c) SB-SAT: native vs non-native, recurrence (1u)",
          [(e, C1, -0.12, "within-fold AUC", True), (f, C2, 0.12, "pooled AUC", True)])
    axes[0].set_ylabel("sensitivity $C_\\tau$ (AUC)", fontsize=9, color=INK)
    fig.tight_layout()
    fig.savefig(out, bbox_inches="tight")
    fig.savefig(out.replace(".pdf", ".png"), dpi=110, bbox_inches="tight")


if __name__ == "__main__":
    main()
