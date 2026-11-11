"""Figure: the Fixation Adjacency Graph on a code stimulus's token AOIs and on a lattice.

A synthetic trial (no participant's gaze) is drawn over the token AOIs of the EMIP Rectangle program
(AOI boxes from the corrected EMIP release); the stimulus image itself is not shown.

Usage:
    python make_fig_fag.py OUT_PDF
"""
from __future__ import annotations

import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import networkx as nx  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.patches import Rectangle  # noqa: E402

from scanpathdecomp.fag import StimulusTriangulation  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
RAW = os.path.join(HERE, "..", "..", "data", "raw")
STIM = "rectangle_java2.jpg"


def synthetic_trial(boxes: pd.DataFrame, rng: np.random.Generator, n: int = 45) -> tuple:
    """Draw a reading-like synthetic scanpath over token AOIs: a top-down pass with a few regressions.

    Args:
        boxes: AOI boxes with columns aoi_x, aoi_y, aoi_width, aoi_height.
        rng: Random generator.
        n: Number of fixations.

    Returns:
        (x, y, dur) arrays.
    """
    c = boxes.assign(cx=boxes.aoi_x + boxes.aoi_width / 2, cy=boxes.aoi_y + boxes.aoi_height / 2)
    c = c.sort_values(["cy", "cx"]).reset_index(drop=True)
    pos = np.linspace(0, len(c) * 0.45, n).astype(int)
    back = rng.random(n) < 0.15
    pos = np.where(back, np.maximum(pos - rng.integers(3, 12, n), 0), pos)
    x = c.cx.values[pos] + rng.normal(0, 4, n)
    y = c.cy.values[pos] + rng.normal(0, 3, n)
    return x, y, rng.gamma(4.0, 30.0, n) + 60


def graph_of(T: StimulusTriangulation, x: np.ndarray, y: np.ndarray, d: np.ndarray) -> tuple:
    """Attended triangles, their dwell and the weighted adjacency graph.

    Args:
        T: Triangulation.
        x: Fixation x.
        y: Fixation y.
        d: Fixation durations.

    Returns:
        (attended triangle indices, dwell per triangle, graph keyed by triangle index).
    """
    dw = T.dwell(x, y, d)
    att = np.where(dw > 0)[0]
    s = set(att.tolist())
    G = nx.Graph()
    G.add_nodes_from(att.tolist())
    for i in att:
        for j in T.adj[i]:
            if j in s and j > i:
                G.add_edge(int(i), int(j), weight=float(dw[i] + dw[j]))
    return att, dw, G


def main(out: str) -> None:
    """Draw and save the figure.

    Args:
        out: Output PDF path.
    """
    E = pd.read_csv(os.path.join(RAW, "emip_finalset_line_part.csv"))
    B = E[E.code_file == STIM].drop_duplicates(["aoi_x", "aoi_y", "aoi_width", "aoi_height"])
    pts = np.stack([B.aoi_x + B.aoi_width / 2, B.aoi_y + B.aoi_height / 2], 1).astype(float)
    T = StimulusTriangulation.from_points(pts)
    rng = np.random.default_rng(7)
    x, y, d = synthetic_trial(B, rng)
    att, dw, G = graph_of(T, x, y, d)
    lat = StimulusTriangulation.lattice((1920.0, 1080.0), 74.0)
    att_l, dw_l, G_l = graph_of(lat, x, y, d)

    xl = (pts[:, 0].min() - 40, pts[:, 0].max() + 40)
    yl = (pts[:, 1].max() + 30, pts[:, 1].min() - 30)
    fig, ax = plt.subplots(1, 4, figsize=(13, 4.2))
    titles = ["(a) token AOIs and anchors", "(b) Delaunay triangulation", "(c) attended triangles, dwell",
              "(d) lattice anchors, attended graph"]
    for a, t in zip(ax, titles):
        a.set_xlim(*xl)
        a.set_ylim(*yl)
        a.set_aspect("equal")
        a.set_xticks([])
        a.set_yticks([])
        a.set_title(t, fontsize=10, loc="left")
    for r in B.itertuples():
        ax[0].add_patch(Rectangle((r.aoi_x, r.aoi_y), r.aoi_width, r.aoi_height, fill=False, lw=0.5, ec="0.55"))
    ax[0].plot(pts[:, 0], pts[:, 1], ".", ms=2.5, color="k")
    ax[0].plot(x, y, "-", lw=0.6, color="tab:red", alpha=0.7)
    ax[0].scatter(x, y, s=d / 12, color="tab:red", alpha=0.6, zorder=3)
    ax[1].triplot(T.points[:, 0], T.points[:, 1], T.simplices, lw=0.35, color="0.5")
    ax[2].triplot(T.points[:, 0], T.points[:, 1], T.simplices, lw=0.25, color="0.8")
    col = plt.get_cmap("viridis")(dw[att] / dw[att].max())
    for k, c in zip(att, col):
        tri = T.points[T.simplices[k]]
        ax[2].fill(tri[:, 0], tri[:, 1], color=c, alpha=0.85, lw=0)
    mst = nx.maximum_spanning_tree(G)
    for i, j in mst.edges():
        p, q = T.centroids[i], T.centroids[j]
        ax[2].plot([p[0], q[0]], [p[1], q[1]], "-", color="k", lw=0.8)
    ax[3].triplot(lat.points[:, 0], lat.points[:, 1], lat.simplices, lw=0.3, color="0.75")
    for k in att_l:
        tri = lat.points[lat.simplices[k]]
        ax[3].fill(tri[:, 0], tri[:, 1], color="tab:blue", alpha=0.35, lw=0)
    for i, j in nx.maximum_spanning_tree(G_l).edges():
        p, q = lat.centroids[i], lat.centroids[j]
        ax[3].plot([p[0], q[0]], [p[1], q[1]], "-", color="k", lw=0.8)
    ax[3].scatter(x, y, s=4, color="tab:red", zorder=3)
    fig.tight_layout()
    fig.savefig(out, bbox_inches="tight")
    print(f"{out}: {len(pts)} AOI anchors, {len(att)} attended triangles; lattice {len(att_l)} attended")


if __name__ == "__main__":
    main(sys.argv[1])
