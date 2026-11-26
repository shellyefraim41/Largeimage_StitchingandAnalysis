"""Illumination (flat-field) correction.

corrected = (raw - camera_offset) / F,   F normalised to mean 1 over the field.

Two ways to get F:
  * reference  - images of a uniform fluorescent sample (preferred, unbiased).
  * overlap    - fitted from tile overlaps: the same sample point seen at two
                 field positions gives  log I_a - log I_b = g(pos_a) - g(pos_b),
                 g = log F modelled as a smooth 2D polynomial.
                 Limitations (reported in QC): directions without overlaps
                 (e.g. Y in a single-row scan) are not constrained and are left
                 flat; photobleaching of the second exposure and an
                 off-centre beam are indistinguishable in a one-direction scan.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
from scipy.ndimage import gaussian_filter, uniform_filter


# ----------------------------------------------------------------- reference
def _load_reference_stack(path: Path):
    """Return (C, Y, X) float32 mean over all frames, plus channel names."""
    path = Path(path)
    if path.suffix.lower() == ".nd2":
        import nd2
        with nd2.ND2File(str(path)) as f:
            names = [c.channel.name for c in f.metadata.channels]
            acc, n = None, 0
            for k in range(len(f.loop_indices)):
                fr = f.read_frame(k).astype(np.float64)
                if fr.ndim == 2:
                    fr = fr[None]
                acc = fr if acc is None else acc + fr
                n += 1
        return (acc / n).astype(np.float32), names
    import tifffile
    arr = tifffile.imread(str(path)).astype(np.float32)
    if arr.ndim == 2:
        arr = arr[None]
    elif arr.ndim > 3:
        arr = arr.reshape(-1, *arr.shape[-2:])
        arr = arr.mean(0, keepdims=True)
    return arr, None


def from_reference(reference, channels: list[str], offset: np.ndarray,
                   shape: tuple[int, int], sigma: float):
    """reference: path (multi-channel ND2, channels matched by name) or {channel: path}."""
    F = np.ones((len(channels), *shape), np.float32)
    notes = {}
    if isinstance(reference, dict):
        for ci, ch in enumerate(channels):
            src = reference.get(ch)
            if src is None:
                # allow label-based keys ('DAPI' for 'DAPI-50')
                src = next((v for k, v in reference.items() if ch.lower().startswith(str(k).lower())), None)
            if src is None:
                notes[ch] = "no reference given -> not corrected"
                continue
            arr, _ = _load_reference_stack(src)
            F[ci] = _norm(arr.mean(0) - offset[ci], sigma)
            notes[ch] = f"reference {Path(src).name}"
    else:
        arr, names = _load_reference_stack(reference)
        for ci, ch in enumerate(channels):
            if names and ch in names:
                j = names.index(ch)
            elif names is None and arr.shape[0] == len(channels):
                j = ci
            else:
                notes[ch] = "channel not in reference -> not corrected"
                continue
            F[ci] = _norm(arr[j] - offset[ci], sigma)
            notes[ch] = f"reference {Path(reference).name}"
    return F, notes


def _norm(img, sigma):
    img = gaussian_filter(np.clip(img, 1e-3, None).astype(np.float32), sigma)
    return (img / img.mean()).astype(np.float32)


# ------------------------------------------------------------------ overlaps
def _poly_terms(degree):
    return [(a, b) for tot in range(1, degree + 1) for a in range(tot + 1) for b in [tot - a]]


def _basis(u, v, terms):
    return np.stack([u ** a * v ** b for a, b in terms], axis=1)


def overlap_samples(tiles_c, pairs, offset, saturation, smooth=9, step=6):
    """Matched intensities of the same sample points in overlapping tiles.

    tiles_c : (P, H, W) float32 raw MIP of one channel
    pairs   : list of (i, j, dy, dx): tile j sits at tile i + (dy, dx)
    returns dict with arrays ya, xa, yb, xb (tile coords), A, B (offset-subtracted)
    """
    P, H, W = tiles_c.shape
    sm = np.stack([uniform_filter(t.astype(np.float32), smooth) for t in tiles_c]) \
        - np.asarray(offset, np.float32).reshape(-1, 1, 1)  # scalar or per-tile
    sat = np.stack([uniform_filter((t >= saturation * 0.98).astype(np.float32), smooth) > 0 for t in tiles_c])
    out = {k: [] for k in ("ya", "xa", "yb", "xb", "A", "B", "pair")}
    for k, (i, j, dy, dx) in enumerate(pairs):
        y0, y1 = max(0, dy), min(H, H + dy)
        x0, x1 = max(0, dx), min(W, W + dx)
        if y1 - y0 < 20 or x1 - x0 < 20:
            continue
        yy, xx = np.mgrid[y0 + smooth:y1 - smooth:step, x0 + smooth:x1 - smooth:step]
        yy, xx = yy.ravel(), xx.ravel()
        a = sm[i][yy, xx]
        b = sm[j][yy - dy, xx - dx]
        ok = ~(sat[i][yy, xx] | sat[j][yy - dy, xx - dx])
        out["ya"].append(yy[ok]); out["xa"].append(xx[ok])
        out["yb"].append(yy[ok] - dy); out["xb"].append(xx[ok] - dx)
        out["A"].append(a[ok]); out["B"].append(b[ok]); out["pair"].append(np.full(ok.sum(), k))
    return {k: (np.concatenate(v) if v else np.zeros(0)) for k, v in out.items()}


def foreground(samples, frac=0.3, min_counts=15.0):
    """Points where both views show signal (cells), not background."""
    A, B = samples["A"], samples["B"]
    if A.size == 0:
        return np.zeros(0, bool)
    thr = max(min_counts, frac * np.percentile(np.concatenate([A, B]), 95))
    return (A > thr) & (B > thr)


def fit_overlap_model(samples, H, W, degree=4, bleach_term=False, max_points=400_000,
                      rng=np.random.default_rng(0)):
    """Fit g = log F from overlap log-ratios with a robust ridge least-squares.

    Returns (F (H,W) float32, info dict) or (None, info) when not enough data.
    """
    m = foreground(samples)
    idx = np.nonzero(m)[0]
    info = {"n_points": int(idx.size)}
    if idx.size == 0:
        return None, info
    if idx.size > max_points:
        idx = rng.choice(idx, max_points, replace=False)
    terms = _poly_terms(degree)
    nu = lambda x: (x - W / 2) / (W / 2)
    nv = lambda y: (y - H / 2) / (H / 2)
    X = _basis(nu(samples["xa"][idx]), nv(samples["ya"][idx]), terms) - \
        _basis(nu(samples["xb"][idx]), nv(samples["yb"][idx]), terms)
    if bleach_term:
        X = np.hstack([X, np.ones((len(idx), 1))])
    y = np.log(samples["A"][idx] / samples["B"][idx])
    diag = (X ** 2).sum(0)
    lam = 1e-4 * diag.max()
    w = np.ones(len(y))
    for _ in range(8):  # IRLS with Huber weights
        Xw = X * w[:, None]
        coef = np.linalg.solve(Xw.T @ X + lam * np.eye(X.shape[1]), Xw.T @ y)
        r = y - X @ coef
        s = 1.4826 * np.median(np.abs(r)) + 1e-9
        k = 1.5 * s
        w = np.where(np.abs(r) <= k, 1.0, k / np.abs(r))
    unconstrained = [terms[t] for t in range(len(terms)) if diag[t] < 100 * lam]
    info.update({
        "terms": terms, "coef": coef.tolist(), "residual_mad": float(s),
        "unconstrained_terms": unconstrained,
        "bleach_factor": float(np.exp(coef[-1])) if bleach_term else None,
    })
    yy, xx = np.mgrid[0:H, 0:W]
    g = np.zeros((H, W), np.float64)
    for c, (a, b) in zip(coef, terms):
        g += c * nu(xx) ** a * nv(yy) ** b
    F = np.exp(g - g.max())
    F = (F / F.mean()).astype(np.float32)
    return F, info


def ratio_curve(samples, F=None, axis="x", nbins=20):
    """Median ratio A/B (after optional correction by F) binned along x or y of tile a."""
    m = foreground(samples)
    if m.sum() == 0:
        return np.array([]), np.array([]), np.array([])
    A, B = samples["A"][m], samples["B"][m]
    if F is not None:
        A = A / F[samples["ya"][m], samples["xa"][m]]
        B = B / F[samples["yb"][m], samples["xb"][m]]
    pos = samples["xa"][m] if axis == "x" else samples["ya"][m]
    r = np.log(A / B)
    edges = np.linspace(pos.min(), pos.max() + 1, nbins + 1)
    k = np.digitize(pos, edges) - 1
    med = np.array([np.median(r[k == j]) if (k == j).sum() > 100 else np.nan for j in range(nbins)])
    return 0.5 * (edges[1:] + edges[:-1]), np.exp(med), np.exp(np.median(r))
