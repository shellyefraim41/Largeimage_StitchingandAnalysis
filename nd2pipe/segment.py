"""Cellpose segmentation of a whole mosaic in overlapping blocks.

Each block is segmented with a margin; a cell is kept by the block whose *core*
contains its centroid, so cells at block borders are neither cut nor duplicated.
Intensities are normalised with experiment-wide fixed limits (not per block),
so empty blocks are not contrast-stretched into false detections.
"""
from __future__ import annotations

import numpy as np
from scipy import ndimage as ndi

_MODEL = None


def get_model(cfg):
    global _MODEL
    if _MODEL is None:
        from cellpose import models
        _MODEL = models.CellposeModel(gpu=cfg["gpu"], pretrained_model=cfg["model"])
    return _MODEL


def normalise(img, lo, hi):
    return np.clip((img - lo) / max(hi - lo, 1e-6), 0, 1).astype(np.float32)


def filter_labels(labels, seg_img_norm, pixel_um, cfg):
    """Remove objects that are too small (µm²) or too dim in the (experiment-wide
    normalised) segmentation channel - e.g. detections on background texture.
    Cheap, so it runs on cached segmentations too."""
    n = int(labels.max())
    if n == 0:
        return labels, {"too_small": 0, "too_dim": 0}
    ids = np.arange(1, n + 1)
    area = ndi.sum_labels(np.ones(labels.shape, np.float32), labels, ids) * pixel_um ** 2
    mean = ndi.mean(seg_img_norm, labels, ids)
    small = area < cfg["min_area_um2"]
    dim = (mean < cfg["min_norm_intensity"]) & ~small
    keep = ~(small | dim)
    lut = np.zeros(n + 1, np.int32)
    lut[ids[keep]] = np.arange(1, keep.sum() + 1)
    return lut[labels], {"too_small": int(small.sum()), "too_dim": int(dim.sum())}


def segment_mosaic(channels_norm: list[np.ndarray], valid: np.ndarray, cfg, log=print):
    """channels_norm: list of 2D float32 images already normalised to [0, 1]
    (first = main segmentation channel, optional second = nuclear helper)."""
    model = get_model(cfg)
    dev = str(getattr(model, "device", "?"))
    if "cuda" in dev:
        import torch
        log(f"    Cellpose on GPU ({torch.cuda.get_device_name(0)})")
    else:
        log(f"    Cellpose on CPU ({dev}) - "
            + ("no usable GPU found, this will be slow" if cfg["gpu"] else "gpu: false in config"))
    H, W = channels_norm[0].shape
    bs, mg = cfg["block_size"], cfg["block_margin"]
    labels = np.zeros((H, W), np.int32)
    next_id = 1
    blocks = [(y, x) for y in range(0, H, bs) for x in range(0, W, bs)]
    for bi, (y0, x0) in enumerate(blocks):
        cy1, cx1 = min(y0 + bs, H), min(x0 + bs, W)
        if not valid[y0:cy1, x0:cx1].any():
            continue
        ey0, ex0 = max(0, y0 - mg), max(0, x0 - mg)
        ey1, ex1 = min(H, cy1 + mg), min(W, cx1 + mg)
        if len(channels_norm) == 1:
            img = channels_norm[0][ey0:ey1, ex0:ex1]
            kw = {}
        else:
            img = np.stack([c[ey0:ey1, ex0:ex1] for c in channels_norm], axis=-1)
            kw = {"channel_axis": 2}
        masks, _, _ = model.eval(img, normalize=False, batch_size=cfg["batch_size"],
                                 diameter=cfg["diameter"], flow_threshold=cfg["flow_threshold"],
                                 cellprob_threshold=cfg["cellprob_threshold"], **kw)
        n = int(masks.max())
        if n == 0:
            continue
        ids = np.arange(1, n + 1)
        area = ndi.sum_labels(np.ones_like(masks), masks, ids)
        cy, cx = np.array(ndi.center_of_mass(np.ones_like(masks), masks, ids)).T
        gy, gx = cy + ey0, cx + ex0
        keep = (gy >= y0) & (gy < cy1) & (gx >= x0) & (gx < cx1) & \
               (area >= cfg["min_area_px"]) & (area <= cfg["max_area_px"])
        lut = np.zeros(n + 1, np.int32)
        k = np.nonzero(keep)[0]
        lut[k + 1] = np.arange(next_id, next_id + len(k))
        next_id += len(k)
        new = lut[masks]
        region = labels[ey0:ey1, ex0:ex1]
        write = (new > 0) & (region == 0)
        region[write] = new[write]
        log(f"    block {bi + 1}/{len(blocks)}: {len(k)} cells")
    labels[~valid] = 0
    # compact ids (a few cells may have lost all pixels to neighbours)
    ids = np.unique(labels)
    ids = ids[ids > 0]
    lut = np.zeros(labels.max() + 1, np.int32)
    lut[ids] = np.arange(1, len(ids) + 1)
    return lut[labels]
