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


def nice_length_um(width_um, frac=0.12):
    """Round scale-bar length (1, 2 or 5 x 10^n µm) close to frac of the image width."""
    target = max(width_um * frac, 1e-3)
    e = 10 ** np.floor(np.log10(target))
    for m in (5, 2, 1):
        if m * e <= target:
            return float(m * e)
    return float(e)


def _font(size):
    from PIL import ImageFont
    try:
        from matplotlib import font_manager
        return ImageFont.truetype(font_manager.findfont("DejaVu Sans:bold"), size)
    except Exception:
        return ImageFont.load_default()


def bar_label(length_um):
    return f"{length_um / 1000:g} mm" if length_um >= 1000 else f"{length_um:g} µm"


def add_scale_bar(rgb8, pixel_um, sb):
    """Draw a scale bar (with thin dark outline, readable on any background) on an RGB uint8 image.
    Sizes scale with the image's short side so the bar stays visible in the downsized previews."""
    from PIL import ImageDraw
    if not sb or not sb.get("enabled", True) or not sb.get("length_um"):
        return rgb8
    H, W = rgb8.shape[:2]
    L = int(round(sb["length_um"] / pixel_um))
    s = min(H, W)
    th = max(3, round(0.012 * s))
    fs = max(12, round(0.045 * s))
    m = max(8, round(0.04 * s))
    ol = max(1, th // 5)
    color = {"white": (255, 255, 255), "black": (0, 0, 0)}.get(sb.get("color", "white"), (255, 255, 255))
    dark = (0, 0, 0) if color != (0, 0, 0) else (255, 255, 255)
    right = "right" in sb.get("position", "bottom-right")
    x0 = W - m - L if right else m
    x1, y1 = x0 + L, H - m
    y0 = y1 - th
    im = Image.fromarray(rgb8)
    d = ImageDraw.Draw(im)
    d.rectangle([x0 - ol, y0 - ol, x1 + ol, y1 + ol], fill=dark)
    d.rectangle([x0, y0, x1, y1], fill=color)
    if sb.get("label", True):
        d.text(((x0 + x1) / 2, y0 - 0.25 * fs), bar_label(sb["length_um"]), fill=color, font=_font(fs),
               anchor="md", stroke_width=max(1, fs // 14), stroke_fill=dark)
    return np.asarray(im)


def save_rgb(path_png, rgb8, preview_path=None, preview_width=4000):
    Image.fromarray(rgb8).save(str(path_png), optimize=False, compress_level=3)
    if preview_path is not None:
        im = Image.fromarray(rgb8)
        if im.width > preview_width:
            im = im.resize((preview_width, max(1, round(im.height * preview_width / im.width))), Image.LANCZOS)
        im.save(str(preview_path), quality=92)


def render_image(mosaic, valid, labels_names, colors_rgb, ranges, merge_sel, out_dir: Path,
                 name, pixel_um, preview_width, scale_bar=None):
    """Writes per-channel data TIFF + coloured PNG, and the merged PNG/TIFF.
    The scale bar goes on the coloured PNGs and JPG previews; TIFFs and gray8 PNGs stay clean."""
    ch_dir = out_dir / "channels"
    mg_dir = out_dir / "merged"
    ch_dir.mkdir(parents=True, exist_ok=True)
    units = {}
    for c, lab in enumerate(labels_names):
        save_float_tiff(ch_dir / f"{name}_{lab}.tif", np.where(valid, mosaic[c], 0), pixel_um)
        u = to_unit(mosaic[c], *ranges[lab])
        u[~valid] = 0
        units[lab] = u
        rgb = add_scale_bar(to_uint8(colorize(u, colors_rgb[lab])), pixel_um, scale_bar)
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
        save_rgb(mg_dir / f"{name}_merge_{tag}.png", add_scale_bar(rgb, pixel_um, scale_bar),
                 mg_dir / f"{name}_merge_{tag}_preview.jpg", preview_width)
        tifffile.imwrite(str(mg_dir / f"{name}_merge_{tag}.tif"), rgb, photometric="rgb",
                         resolution=(1 / pixel_um, 1 / pixel_um), metadata=None, compression="zlib")
        # small copy (no scale bar) for the image + profile figure
        im = Image.fromarray(rgb)
        fw = min(3200, im.width)
        return np.asarray(im.resize((fw, max(1, round(im.height * fw / im.width))), Image.LANCZOS))
    return None
