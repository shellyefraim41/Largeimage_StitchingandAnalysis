"""Tile registration and mosaic assembly.

1. Stage positions (µm) -> predicted pixel positions. The camera orientation
   relative to the stage (axis signs / swap) is detected automatically.
2. Each overlapping pair is registered by phase correlation inside the
   *predicted* overlap, then refined/validated with normalised
   cross-correlation (NCC). Pairs with low NCC fall back to the stage
   prediction (corrected by the median error of the good pairs).
3. Global positions: weighted least squares over all pairs.
4. Blending: feathered (linear ramps) or nearest-to-tile-centre.
"""
from __future__ import annotations

import itertools

import numpy as np
from skimage.registration import phase_cross_correlation


def _prep(img, offset):
    x = np.log1p(np.clip(img.astype(np.float32) - offset, 0, None))
    return x


def ncc(a, b):
    a = a - a.mean()
    b = b - b.mean()
    d = np.sqrt((a * a).sum() * (b * b).sum())
    return float((a * b).sum() / d) if d > 0 else 0.0


def overlap_views(a, b, dy, dx):
    """Views of a and b over their common area when b sits at a + (dy, dx)."""
    H, W = a.shape
    ya, yb = slice(max(0, dy), min(H, H + dy)), slice(max(0, -dy), min(H, H - dy))
    xa, xb = slice(max(0, dx), min(W, W + dx)), slice(max(0, -dx), min(W, W - dx))
    return a[ya, xa], b[yb, xb]


def candidate_pairs(pos, H, W, min_frac=0.03):
    pairs = []
    for i, j in itertools.combinations(range(len(pos)), 2):
        dy, dx = pos[j] - pos[i]
        oy, ox = H - abs(dy), W - abs(dx)
        if oy > 0 and ox > 0 and oy * ox >= min_frac * H * W:
            pairs.append((i, j))
    return pairs


def stage_to_px(stage_xy, pixel_um, orient):
    sx, sy, swap = orient
    x, y = stage_xy[:, 0] / pixel_um * sx, stage_xy[:, 1] / pixel_um * sy
    if swap:
        x, y = y, x
    p = np.stack([y, x], 1)
    return p - p.min(0)


def register_pair(a, b, pred, max_shift):
    """a, b: prepared 2D images. pred: predicted (dy, dx) of b relative to a.
    Returns (dy, dx, ncc)."""
    dy0, dx0 = int(round(pred[0])), int(round(pred[1]))
    A, B = overlap_views(a, b, dy0, dx0)
    if min(A.shape) < 32:
        return dy0, dx0, 0.0
    win = np.outer(np.hanning(A.shape[0]), np.hanning(A.shape[1])).astype(np.float32)
    s, _, _ = phase_cross_correlation((A - A.mean()) * win, (B - B.mean()) * win,
                                      normalization=None)
    cands = [(dy0, dx0)]
    if abs(s[0]) <= max_shift and abs(s[1]) <= max_shift:
        cands.insert(0, (dy0 + int(s[0]), dx0 + int(s[1])))
    best = (dy0, dx0, -1.0)
    for cy, cx in cands:  # local NCC refinement around each candidate
        for ddy in range(-3, 4):
            for ddx in range(-3, 4):
                Av, Bv = overlap_views(a, b, cy + ddy, cx + ddx)
                v = ncc(Av[::2, ::2], Bv[::2, ::2])
                if v > best[2]:
                    best = (cy + ddy, cx + ddx, v)
    return best


def detect_orientation(prep, stage_xy, pixel_um, H, W, max_shift):
    """Try the 8 axis conventions; keep the one whose predicted overlaps match best."""
    scores = {}
    for orient in itertools.product((1, -1), (1, -1), (False, True)):
        pos = stage_to_px(stage_xy, pixel_um, orient)
        pairs = candidate_pairs(pos, H, W)
        if not pairs:
            scores[orient] = -1
            continue
        # strongest-overlap pairs first
        pairs = sorted(pairs, key=lambda ij: -np.prod([H, W] - np.abs(pos[ij[1]] - pos[ij[0]])))[:3]
        vals = [register_pair(prep[i], prep[j], pos[j] - pos[i], max_shift)[2] for i, j in pairs]
        scores[orient] = float(np.mean(vals))
    best = max(scores, key=scores.get)
    return best, scores


def solve_positions(n, results, anchor_pos):
    """results: list of (i, j, dy, dx, weight). Weighted LSQ, tile 0 anchored."""
    rows, rhs, w = [], [], []
    for i, j, dy, dx, wt in results:
        r = np.zeros(n); r[j] = 1; r[i] = -1
        rows.append(r); rhs.append((dy, dx)); w.append(wt)
    r = np.zeros(n); r[0] = 1
    rows.append(r); rhs.append(tuple(anchor_pos[0])); w.append(1e3)
    A = np.array(rows) * np.array(w)[:, None]
    b = np.array(rhs, float) * np.array(w)[:, None]
    sol, *_ = np.linalg.lstsq(A, b, rcond=None)
    # tiles with no pair keep their stage position
    connected = {i for i, *_ in results} | {j for _, j, *_ in results} | {0}
    for k in range(n):
        if k not in connected:
            sol[k] = anchor_pos[k]
    sol = np.round(sol - sol.min(0)).astype(int)
    return sol


def register_tiles(reg_tiles, stage_xy, pixel_um, offset, max_shift, min_ncc):
    """reg_tiles: (P, H, W) raw MIP of the registration channel."""
    P, H, W = reg_tiles.shape
    prep = np.stack([_prep(t, offset) for t in reg_tiles])
    orient, orient_scores = detect_orientation(prep, stage_xy, pixel_um, H, W, max_shift)
    pred = stage_to_px(stage_xy, pixel_um, orient)
    pairs = candidate_pairs(pred, H, W)
    table = []
    for i, j in pairs:
        p = pred[j] - pred[i]
        dy, dx, v = register_pair(prep[i], prep[j], p, max_shift)
        table.append(dict(i=i, j=j, pred_dy=float(p[0]), pred_dx=float(p[1]),
                          dy=int(dy), dx=int(dx), ncc=float(v), ok=bool(v >= min_ncc)))
    good = [t for t in table if t["ok"]]
    if good:
        corr = np.median([[t["dy"] - t["pred_dy"], t["dx"] - t["pred_dx"]] for t in good], axis=0)
    else:
        corr = np.zeros(2)
    results = []
    for t in table:
        if t["ok"]:
            results.append((t["i"], t["j"], t["dy"], t["dx"], max(t["ncc"], 0.05)))
        else:
            t["dy"] = int(round(t["pred_dy"] + corr[0])); t["dx"] = int(round(t["pred_dx"] + corr[1]))
            results.append((t["i"], t["j"], t["dy"], t["dx"], 0.05))
    pos = solve_positions(P, results, pred)
    for t in table:  # residual of the global solution vs the pairwise measurement
        d = pos[t["j"]] - pos[t["i"]]
        t["final_dy"], t["final_dx"] = int(d[0]), int(d[1])
    return pos, table, {"orientation": orient, "orientation_scores": {str(k): v for k, v in orient_scores.items()},
                        "stage_correction_px": corr.tolist()}


def tile_weight(H, W, blend):
    y = np.minimum(np.arange(H) + 1, H - np.arange(H)).astype(np.float32) / (H / 2)
    x = np.minimum(np.arange(W) + 1, W - np.arange(W)).astype(np.float32) / (W / 2)
    return np.outer(y, x)


def assemble(tiles, pos, blend="feather"):
    """tiles: (P, C, H, W) float32 corrected. Returns mosaic (C, Hm, Wm), valid mask, source map."""
    P, C, H, W = tiles.shape
    Hm, Wm = pos[:, 0].max() + H, pos[:, 1].max() + W
    w = tile_weight(H, W, blend)
    mosaic = np.zeros((C, Hm, Wm), np.float32)
    wsum = np.zeros((Hm, Wm), np.float32)
    wbest = np.zeros((Hm, Wm), np.float32)
    source = np.full((Hm, Wm), -1, np.int16)
    for p in range(P):
        y0, x0 = pos[p]
        sl = (slice(y0, y0 + H), slice(x0, x0 + W))
        better = w > wbest[sl]
        source[sl][better] = p
        wbest[sl] = np.maximum(wbest[sl], w)
        if blend == "feather":
            mosaic[(slice(None),) + sl] += tiles[p] * w
            wsum[sl] += w
    if blend == "feather":
        valid = wsum > 0
        mosaic[:, valid] /= wsum[valid]
    else:
        valid = source >= 0
        for p in range(P):
            y0, x0 = pos[p]
            sl = (slice(y0, y0 + H), slice(x0, x0 + W))
            m = source[sl] == p
            for c in range(C):
                mosaic[c][sl][m] = tiles[p, c][m]
    return mosaic, valid, source
