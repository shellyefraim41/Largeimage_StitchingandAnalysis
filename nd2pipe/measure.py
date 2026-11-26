"""Background, per-cell measurements and intensity-vs-X profiles.

Only pixels inside segmented cells contribute to the profiles, so empty areas
between colonies do not dilute the average. Two averages are reported per X-bin:
  per-cell  : mean over cells (each cell = its mean intensity) - default
  per-pixel : mean over all cell pixels in the bin
Background (median of cell-free pixels, estimated locally per X-bin) is
subtracted from both.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import ndimage as ndi


def local_background(mosaic, labels, valid, pixel_um, cfg):
    """Returns (edges_px, bkg (C, nbins)) of median cell-free intensity per X-bin."""
    C, H, W = mosaic.shape
    bin_px = max(1, int(round(cfg["bin_um"] / pixel_um)))
    edges = np.arange(0, W + bin_px, bin_px)
    edges[-1] = W
    free = valid & ~ndi.binary_dilation(labels > 0, iterations=cfg["mask_dilation_px"])
    bkg = np.full((C, len(edges) - 1), np.nan, np.float32)
    for b in range(len(edges) - 1):
        sl = slice(edges[b], edges[b + 1])
        m = free[:, sl]
        if m.sum() >= cfg["min_pixels"]:
            for c in range(C):
                bkg[c, b] = np.median(mosaic[c][:, sl][m])
    for c in range(C):  # fill bins without free pixels by interpolation
        v = bkg[c]
        ok = np.isfinite(v)
        if ok.any():
            centers = 0.5 * (edges[1:] + edges[:-1])
            bkg[c] = np.interp(centers, centers[ok], v[ok])
        else:
            bkg[c] = 0.0
    return edges, bkg


def bkg_at(edges, bkg_c, x):
    k = np.clip(np.searchsorted(edges, x, side="right") - 1, 0, len(edges) - 2)
    return bkg_c[k]


def cell_table(mosaic, labels, labels_names, pixel_um, edges, bkg):
    n = int(labels.max())
    ids = np.arange(1, n + 1)
    ones = np.ones(labels.shape, np.float32)
    area = ndi.sum_labels(ones, labels, ids)
    cy, cx = np.array(ndi.center_of_mass(ones, labels, ids)).reshape(-1, 2).T
    df = pd.DataFrame({"cell_id": ids, "x_px": cx, "y_px": cy, "x_um": cx * pixel_um,
                       "y_um": cy * pixel_um, "area_px": area, "area_um2": area * pixel_um ** 2})
    for c, name in enumerate(labels_names):
        mean = ndi.mean(mosaic[c], labels, ids)
        b = bkg_at(edges, bkg[c], cx)
        df[f"{name}_mean_raw"] = mean
        df[f"{name}_bkg"] = b
        df[f"{name}_mean"] = mean - b
    return df


def flag_outliers(cells, labels_names, k_mad):
    """Flag cells far above the channel's typical range (debris, aggregates).
    Flagged cells stay in the table but are excluded from the profiles."""
    counts = {}
    for name in labels_names:
        v = cells[f"{name}_mean"].to_numpy()
        if k_mad is None or len(v) < 20:
            cells[f"{name}_outlier"] = False
            continue
        med = np.median(v)
        mad = 1.4826 * np.median(np.abs(v - med)) + 1e-9
        flag = v > med + k_mad * mad
        cells[f"{name}_outlier"] = flag
        counts[name] = int(flag.sum())
    return counts


def x_profiles(cells, mosaic, labels, valid, labels_names, pixel_um, edges_bkg, bkg, cfg):
    """Binned profiles along X. Returns a long DataFrame."""
    C, H, W = mosaic.shape
    bin_px = max(1, int(round(cfg["bin_um"] / pixel_um)))
    edges = np.arange(0, W + bin_px, bin_px)
    edges[-1] = W
    nb = len(edges) - 1
    centers_um = 0.5 * (edges[1:] + edges[:-1]) * pixel_um
    inmask = labels > 0
    mask_cols = inmask.sum(0).astype(np.float64)
    valid_cols = valid.sum(0).astype(np.float64)
    kall = np.clip(np.searchsorted(edges, cells["x_px"].to_numpy(), side="right") - 1, 0, nb - 1)
    n_all = np.bincount(kall, minlength=nb)
    mpix = np.add.reduceat(mask_cols, edges[:-1])
    vpix = np.add.reduceat(valid_cols, edges[:-1])
    xs = np.arange(W)
    rows = []
    for c, name in enumerate(labels_names):
        keep = ~cells.get(f"{name}_outlier", pd.Series(False, index=cells.index)).to_numpy(bool)
        v = cells[f"{name}_mean"].to_numpy()[keep]
        kcell = kall[keep]
        n_cells = np.bincount(kcell, minlength=nb)
        s1 = np.bincount(kcell, v, nb)
        s2 = np.bincount(kcell, v * v, nb)
        with np.errstate(invalid="ignore", divide="ignore"):
            mean = s1 / n_cells
            sd = np.sqrt(np.maximum(s2 / n_cells - mean ** 2, 0) * n_cells / np.maximum(n_cells - 1, 1))
            sem = sd / np.sqrt(n_cells)
        med = np.array([np.median(v[kcell == b]) if n_cells[b] else np.nan for b in range(nb)])
        if keep.all():
            inm_c, mcols_c, mpix_c = inmask, mask_cols, mpix
        else:  # drop outlier cells' pixels from the per-pixel mean too
            ok_lut = np.ones(int(labels.max()) + 1, bool)
            ok_lut[cells["cell_id"].to_numpy()[~keep]] = False
            inm_c = inmask & ok_lut[labels]
            mcols_c = inm_c.sum(0).astype(np.float64)
            mpix_c = np.add.reduceat(mcols_c, edges[:-1])
        colsum = np.where(inm_c, mosaic[c], 0).sum(0).astype(np.float64)
        colsum -= mcols_c * bkg_at(edges_bkg, bkg[c], xs)
        with np.errstate(invalid="ignore", divide="ignore"):
            pix = np.add.reduceat(colsum, edges[:-1]) / mpix_c
        enough = n_all >= cfg["min_cells_per_bin"]
        for b in range(nb):
            rows.append(dict(channel=name, x_um=centers_um[b], x0_um=edges[b] * pixel_um,
                             x1_um=edges[b + 1] * pixel_um, n_cells=int(n_all[b]),
                             n_outliers_excluded=int(n_all[b] - n_cells[b]),
                             cell_mean=mean[b], cell_median=med[b], cell_sd=sd[b], cell_sem=sem[b],
                             pixel_mean=pix[b], cell_area_fraction=mpix[b] / vpix[b] if vpix[b] else np.nan,
                             enough_cells=bool(enough[b])))
    return pd.DataFrame(rows)
