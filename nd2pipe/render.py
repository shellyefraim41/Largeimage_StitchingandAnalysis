"""Saving channel / merged images with experiment-wide display settings."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import tifffile
from PIL import Image

Image.MAX_IMAGE_PIXELS = None


def to_unit(img, lo, hi):
    return np.clip((img - lo) / max(hi - lo, 1e-6), 0, 1)


def colorize(unit, rgb):
    out = np.empty(unit.shape + (3,), np.float32)
    for k in range(3):
        out[..., k] = unit * rgb[k]
    return out


def to_uint8(rgb):
    return (np.clip(rgb, 0, 1) * 255 + 0.5).astype(np.uint8)


def save_float_tiff(path, img, pixel_um):
    tifffile.imwrite(str(path), img.astype(np.float32), imagej=True,
                     resolution=(1 / pixel_um, 1 / pixel_um),
                     metadata={"unit": "um", "axes": "YX" if img.ndim == 2 else "CYX"},
                     compression="zlib")


def save_rgb(path_png, rgb8, preview_path=None, preview_width=4000):
    Image.fromarray(rgb8).save(str(path_png), optimize=False, compress_level=3)
    if preview_path is not None:
        im = Image.fromarray(rgb8)
        if im.width > preview_width:
            im = im.resize((preview_width, max(1, round(im.height * preview_width / im.width))), Image.LANCZOS)
        im.save(str(preview_path), quality=92)


def render_image(mosaic, valid, labels_names, colors_rgb, ranges, merge_sel, out_dir: Path,
                 name, pixel_um, preview_width):
    """Writes per-channel data TIFF + coloured PNG, and the merged PNG/TIFF."""
    ch_dir = out_dir / "channels"
    mg_dir = out_dir / "merged"
    ch_dir.mkdir(parents=True, exist_ok=True)
    units = {}
    for c, lab in enumerate(labels_names):
        save_float_tiff(ch_dir / f"{name}_{lab}.tif", np.where(valid, mosaic[c], 0), pixel_um)
        u = to_unit(mosaic[c], *ranges[lab])
        u[~valid] = 0
        units[lab] = u
        rgb = to_uint8(colorize(u, colors_rgb[lab]))
        save_rgb(ch_dir / f"{name}_{lab}.png", rgb, ch_dir / f"{name}_{lab}_preview.jpg", preview_width)
        gray = (u * 255 + 0.5).astype(np.uint8)
        Image.fromarray(gray).save(str(ch_dir / f"{name}_{lab}_gray8.png"))
    if merge_sel:
        mg_dir.mkdir(parents=True, exist_ok=True)
        acc = np.zeros(mosaic.shape[1:] + (3,), np.float32)
        for lab in merge_sel:
            acc += colorize(units[lab], colors_rgb[lab])
        rgb = to_uint8(acc)
        tag = "-".join(merge_sel)
        save_rgb(mg_dir / f"{name}_merge_{tag}.png", rgb, mg_dir / f"{name}_merge_{tag}_preview.jpg", preview_width)
        tifffile.imwrite(str(mg_dir / f"{name}_merge_{tag}.tif"), rgb, photometric="rgb",
                         resolution=(1 / pixel_um, 1 / pixel_um), metadata=None, compression="zlib")
