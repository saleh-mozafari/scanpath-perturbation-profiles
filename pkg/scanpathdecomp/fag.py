"""Undirected Fixation Adjacency Graph (FAG): a stimulus-anchored graph descriptor.

Construction (per stimulus, once): the stimulus mask is Otsu-thresholded, morphologically closed
(3x3), border-touching components removed, connected components with area >= min_area kept and
reduced to bounding-box centres; the centres are joined by a Delaunay triangulation.
Per trial: each triangle accumulates the durations of the fixations inside it; attended triangles
(dwell > 0) are the vertices; two attended triangles are adjacent if they share a vertex of the
triangulation; edge weight = dwell_i + dwell_j. Descriptors:
  attention centrality  share of total edge weight captured by the heaviest maximum spanning tree
  betweenness           maximum edge betweenness on the binary graph
  compactness           Estrada index, mean over connected components
  squareness            mean over components of dx*dy/max(dx,dy)^2 of the MST vertex bounding box
  n_components          number of connected components
Coordinates: fixations are given in the trial frame; `to_mask` maps them into mask pixels.
"""
from __future__ import annotations
import numpy as np
import networkx as nx
from scipy.spatial import Delaunay
from .trial import Trial


class StimulusTriangulation:
    def __init__(self, mask_image: np.ndarray, min_area: int = 100):
        from skimage.filters import threshold_otsu
        from skimage.segmentation import clear_border
        from skimage.measure import label, regionprops
        from skimage.morphology import closing, square
        m = mask_image[..., 0] if mask_image.ndim == 3 else mask_image
        m = 255 - m.astype(np.int32)
        bw = closing(m > threshold_otsu(m), square(3))
        lab = label(clear_border(bw))
        pts = []
        for r in regionprops(lab):
            if r.area >= min_area:
                minr, minc, maxr, maxc = r.bbox
                pts.append([(minc + maxc) / 2.0, (minr + maxr) / 2.0])
        self._build(np.array(pts))

    def _build(self, points: np.ndarray) -> None:
        """Triangulate anchor points and precompute triangle centroids and vertex-sharing adjacency.

        Args:
            points: Array of shape (n, 2) with anchor coordinates in mask pixels.
        """
        self.points = np.asarray(points, dtype=float)
        self.exact_edges = True
        self._nudge = np.zeros(2)
        self.tri = Delaunay(self.points)
        self.simplices = self.tri.simplices
        self.centroids = self.points[self.simplices].mean(1)
        # adjacency of triangles: share at least one vertex index
        by_vertex: dict = {}
        for i, s in enumerate(self.simplices):
            for v in s:
                by_vertex.setdefault(int(v), []).append(i)
        self.adj = []
        for i, s in enumerate(self.simplices):
            nb = set()
            for v in s:
                nb.update(by_vertex[int(v)])
            nb.discard(i)
            self.adj.append(sorted(nb))

    @classmethod
    def from_points(cls, points: np.ndarray) -> "StimulusTriangulation":
        """Build a triangulation from given anchor points instead of a stimulus mask.

        Anchors can be the centres of a stimulus's areas of interest (words, code tokens, glyphs)
        or the nodes of a regular lattice, which makes the descriptor applicable to any stimulus.

        Args:
            points: Array of shape (n, 2), n >= 3, of anchor coordinates in the trial frame.

        Returns:
            A StimulusTriangulation whose triangles are the Delaunay triangles of the anchors.
        """
        obj = cls.__new__(cls)
        obj._build(np.asarray(points, dtype=float))
        # point location by the fast directed walk; queries are nudged by a fixed offset far below any
        # measurement precision, so a fixation lying exactly on a shared edge is assigned deterministically
        obj.exact_edges = False
        span = float(np.ptp(obj.points, axis=0).max()) or 1.0
        obj._nudge = np.array([1.37e-9, 2.71e-9]) * span
        return obj

    @classmethod
    def lattice(cls, frame: tuple, spacing: float, cover_rotation: bool = True) -> "StimulusTriangulation":
        """Build a stimulus-independent triangulation on a square lattice.

        Args:
            frame: (width, height) of the trial frame.
            spacing: Distance between neighbouring lattice nodes, in frame units.
            cover_rotation: If True, the lattice covers the disc swept by the frame under rotation
                about its centre, so the rotation transformation never moves fixations off the support.

        Returns:
            A StimulusTriangulation on the lattice nodes.
        """
        w, h = frame
        cx, cy = w / 2.0, h / 2.0
        if cover_rotation:
            r = float(np.hypot(w, h)) / 2.0 + spacing
            xs = np.arange(cx - r, cx + r + spacing, spacing)
            ys = np.arange(cy - r, cy + r + spacing, spacing)
        else:
            xs = np.arange(0.0, w + spacing, spacing)
            ys = np.arange(0.0, h + spacing, spacing)
        gx, gy = np.meshgrid(xs, ys)
        pts = np.stack([gx.ravel(), gy.ravel()], 1)
        # tiny deterministic shear breaks the co-circularity of square cells, so the Delaunay split is unique
        pts = pts + np.stack([1e-6 * spacing * gy.ravel() / max(h, 1.0), np.zeros(pts.shape[0])], 1)
        return cls.from_points(pts)

    def dwell(self, x, y, dur):
        # bruteforce=True: assignment of points lying exactly on a shared edge must not depend on
        # query order (the default directed walk starts from the previously found simplex)
        q = np.stack([x, y], 1)
        if self.exact_edges:
            idx = self.tri.find_simplex(q, bruteforce=True)
        else:
            # a nudged point lies strictly inside one triangle, so the walk's result does not depend on query order
            idx = self.tri.find_simplex(q + self._nudge)
        d = np.zeros(len(self.simplices))
        for k, w in zip(idx, dur):
            if k >= 0:
                d[k] += w
        return d


def fixation_adjacency_graph(triangulations: dict, to_mask=lambda x, y: (x, y), stimulus_key="stimulus",
                             degenerate_attention: float = 0.25):
    """Factory. `triangulations` maps a stimulus id (from trial.meta[stimulus_key]) to a
    StimulusTriangulation; `to_mask` maps trial coordinates to mask pixels."""
    def _f(t: Trial):
        T = triangulations[t.meta[stimulus_key]]
        mx, my = to_mask(t.x, t.y)
        d = T.dwell(mx, my, t.dur if t.dur is not None else np.ones(len(t)))
        att = np.where(d > 0)[0]
        if len(att) == 0:
            return None
        G = nx.Graph(); G.add_nodes_from(range(len(att))); pos = {}
        loc = {orig: k for k, orig in enumerate(att)}
        for k, orig in enumerate(att):
            pos[k] = T.centroids[orig]
            for j in T.adj[orig]:
                if j in loc and loc[j] > k:
                    G.add_edge(k, loc[j], weight=int(d[orig] + d[j]))
        eb = nx.edge_betweenness_centrality(G)
        bt = max(eb.values()) if eb else 0.0
        ee, span, sq = [], [], []
        for comp in nx.connected_components(G):
            sub = G.subgraph(comp)
            ee.append(nx.estrada_index(sub))
            Tm = nx.maximum_spanning_tree(sub)
            w = nx.get_edge_attributes(Tm, "weight"); span.append(sum(w.values()) if w else 0.0)
            if len(sub) == 1:
                sq.append(0.0)
            else:
                nodes = sorted(set(np.array(list(w.keys())).ravel()))
                P = np.array([pos[n] for n in nodes])
                dx = np.ptp(P[:, 0]); dy = np.ptp(P[:, 1]); delta = max(dx, dy)
                sq.append((dx * dy) / delta ** 2 if delta > 0 else 0.0)
        total = sum(nx.get_edge_attributes(G, "weight").values()) or 1e-9
        attn = max(span) / total
        if attn == 0.0:
            attn = degenerate_attention      # the released pipeline recodes degenerate (edgeless) trials to 0.25
        return dict(attention=float(attn), betweenness=float(bt), estrada=float(np.mean(ee)),
                    squareness=float(np.mean(sq)), n_components=float(nx.number_connected_components(G)))
    _f.__name__ = "fixation_adjacency_graph"
    return _f
