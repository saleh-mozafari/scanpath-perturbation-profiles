"""Built-in trial-level descriptors. Each factory returns a function Trial -> dict of
features (or None if the trial is too short for the descriptor). Users may supply any
function with the same signature.
"""
from __future__ import annotations
import numpy as np
from .trial import Trial


def rqa(radius: float = 60.0, min_len: int = 5):
    """Recurrence quantification analysis (Anderson et al., 2013): recurrence rate,
    determinism, laminarity, centre of recurrence mass. Two fixations recur if their
    Euclidean distance is below `radius`."""
    def _f(t: Trial):
        N = len(t)
        if N < min_len:
            return None
        P = np.stack([t.x, t.y], 1)
        D = np.sqrt(((P[:, None, :] - P[None, :, :]) ** 2).sum(-1))
        # Strict "< radius" with a relative tolerance of 1e-9. On integer pixel coordinates some pairs lie
        # exactly at the radius (e.g. 36-48-60); after a rigid rotation their floating-point distance lands
        # a few ulps either side of it, which made recurrence look slightly rotation-sensitive. No distance
        # between integer points falls in (r(1 - 1e-9), r) for r <= 10^4 px, so observed features are unchanged.
        tri = np.triu(D < radius * (1.0 - 1e-9), 1)
        nrec = int(tri.sum())
        if nrec == 0:
            return dict(rec=0.0, det=0.0, lam=0.0, corm=0.0)
        rec = 100.0 * 2 * nrec / (N * (N - 1))
        def _runpts(lines):
            # total number of points lying on runs of length >= 2 across a list of 0/1 arrays
            tot = 0
            for a in lines:
                if a.size == 0 or not a.any():
                    continue
                b = np.concatenate([[0], a.astype(np.int8), [0]])
                d = np.diff(b); starts = np.where(d == 1)[0]; ends = np.where(d == -1)[0]
                L = ends - starts; tot += int(L[L >= 2].sum())
            return tot
        diagpts = _runpts([np.diagonal(tri, k) for k in range(1, N)])
        lampts = _runpts([tri[:, j] for j in range(N)])
        ii, jj = np.where(tri)
        return dict(rec=rec, det=100.0 * diagpts / nrec, lam=100.0 * lampts / nrec,
                    corm=100.0 * np.mean(jj - ii) / (N - 1))
    _f.__name__ = f"rqa(r={radius:g})"
    return _f


def directed_graph(tile: tuple = (60.0, 50.0), min_nodes: int = 2):
    """Directed transition graph over grid tiles: edge reciprocity, revisit tendency,
    directed transition entropy, degree asymmetry. Tiles are floor(x/tile_w), floor(y/tile_h)
    and are NOT clipped, so the descriptor is well defined after the anchoring null."""
    tw, th = tile
    def _f(t: Trial):
        cells = [(int(np.floor(x / tw)), int(np.floor(y / th))) for x, y in zip(t.x, t.y)]
        seq = [cells[0]]
        for c in cells[1:]:
            if c != seq[-1]: seq.append(c)
        nodes = sorted(set(seq))
        if len(seq) < 2 or len(nodes) < min_nodes:
            return None
        idx = {c: k for k, c in enumerate(nodes)}; n = len(nodes); A = np.zeros((n, n))
        for a, b in zip(seq[:-1], seq[1:]): A[idx[a], idx[b]] += 1
        E = {(i, j) for i in range(n) for j in range(n) if i != j and A[i, j] > 0}
        recip = np.mean([1.0 if (j, i) in E else 0.0 for (i, j) in E]) if E else 0.0
        out = A.sum(1); inn = A.sum(0); den = out + inn
        asym = np.mean(np.abs(out - inn)[den > 0] / den[den > 0]) if (den > 0).any() else 0.0
        P = A / (A.sum() or 1); nz = P[P > 0]; H = float(-(nz * np.log2(nz)).sum())
        return dict(recip=recip, revisit=1.0 - len(nodes) / len(seq), dir_tent=H, asym=asym)
    _f.__name__ = f"directed_graph(tile={tw:g}x{th:g})"
    return _f


def classic():
    """Mean fixation duration and mean saccade amplitude."""
    def _f(t: Trial):
        if len(t) < 2 or t.dur is None:
            return None
        amp = np.sqrt(np.diff(t.x) ** 2 + np.diff(t.y) ** 2)
        return dict(duration=float(t.dur.mean()), saccade=float(amp.mean()))
    _f.__name__ = "classic"
    return _f


def combine(*fs):
    """Concatenate several descriptors into one feature dict."""
    def _f(t: Trial):
        out = {}
        for f in fs:
            r = f(t)
            if r is None:
                return None
            out.update({f"{f.__name__}:{k}": v for k, v in r.items()})
        return out
    _f.__name__ = "+".join(f.__name__ for f in fs)
    return _f
