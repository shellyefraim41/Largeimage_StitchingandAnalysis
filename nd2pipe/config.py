"""Experiment configuration (YAML) with defaults.

Every image listed in one config is processed with identical settings, and the
display ranges / thresholds are computed once for the whole experiment.
"""
from __future__ import annotations

import copy
from pathlib import Path

import yaml

DEFAULTS = {
    "experiment_name": "experiment",
    "output_dir": "output",
    "images": [],
    # channel name in the ND2 -> {label, color}. Missing entries are auto-filled.
    "channels": {},
    # 'ask' (prompt once), 'all', 'none', or a list of channel labels / ND2 names
    "merge_channels": "ask",
    "camera_offset": "auto",          # number, or 'auto' (0.1th percentile of the data)
    "flatfield": {
        "method": "auto",             # auto | reference | overlap | none
        "reference": None,            # ND2/TIFF with all channels, or {channel: path}
        "smooth_sigma_px": 30,        # smoothing of reference images
        "poly_degree": 4,             # overlap model: 2D polynomial of log(illumination)
        "min_points": 2000,           # overlap points needed per channel
        "bleach_term": False,         # fit a constant 'second exposure is dimmer' factor
    },
    "stitching": {
        "registration_channel": None, # default: segmentation channel
        "max_shift_px": 80,           # search window around the stage prediction
        "min_ncc": 0.5,               # registrations below this fall back to stage positions
        "blend": "feather",           # feather | nearest
    },
    "segmentation": {
        "channels": None,             # e.g. [DAPI] (nuclei) or [Phalloidin, DAPI] (cells)
        "model": "cpsam",
        "diameter": None,             # px; None = Cellpose default
        "block_size": 4096,
        "block_margin": 160,
        "batch_size": 64,
        "min_area_px": 80,            # hard limit inside Cellpose blocks
        "max_area_px": 20000,
        "min_area_um2": 15,           # post-filter: smaller objects removed (nucleus ~ 100 µm²)
        "min_norm_intensity": 0.05,   # post-filter: mean of normalised seg channel (0-1)
        "flow_threshold": 0.4,
        "cellprob_threshold": 0.0,
        "norm_percentiles": [1.0, 99.8],
        "gpu": True,
    },
    "background": {
        "bin_um": 200,                # local background is estimated per X-bin of this width
        "mask_dilation_px": 8,
        "min_pixels": 2000,
    },
    "profile": {
        "bin_um": 50,
        "min_cells_per_bin": 10,
        "show_pixel_mean": True,      # also plot the per-pixel (inside masks) mean
        "statistic": "mean",          # mean (± SEM) | median of the per-cell values
        "outlier_mad": 20,            # exclude cells > median + k*MAD (debris); null = keep all
    },
    "display": {
        "low": "background",          # 'background' (pooled background median) or a percentile number
        "high_percentile": 99.5,      # of in-cell pixels, pooled over all images
        "fixed": {},                  # {label: [low, high]} overrides
        "preview_width_px": 4000,
    },
    "reuse_settings": None,           # path to experiment_settings.json of an earlier run
}

# microscopy LUT colours (images) and plot colours (validated categorical steps)
LUT_RGB = {
    "blue": (0.0, 0.0, 1.0), "green": (0.0, 1.0, 0.0), "red": (1.0, 0.0, 0.0),
    "magenta": (1.0, 0.0, 1.0), "cyan": (0.0, 1.0, 1.0), "yellow": (1.0, 1.0, 0.0),
    "gray": (1.0, 1.0, 1.0), "orange": (1.0, 0.5, 0.0),
}
PLOT_HEX = {
    "blue": "#2a78d6", "green": "#008300", "red": "#e34948", "magenta": "#d55181",
    "cyan": "#1baf7a", "yellow": "#eda100", "gray": "#52514e", "orange": "#eb6834",
}
IMAGE_SERIES_HEX = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4",
                    "#008300", "#6250d6", "#e34948"]


def _merge(base: dict, over: dict) -> dict:
    out = copy.deepcopy(base)
    for k, v in (over or {}).items():
        if isinstance(v, dict) and isinstance(out.get(k), dict) and k not in ("channels", "fixed", "reference"):
            out[k] = _merge(out[k], v)
        else:
            out[k] = v
    return out


def load_config(path: str | Path) -> dict:
    path = Path(path)
    with open(path, "r", encoding="utf-8") as fh:
        user = yaml.safe_load(fh) or {}
    cfg = _merge(DEFAULTS, user)
    base = path.parent.resolve()

    def resolve(p):
        p = Path(p)
        return p if p.is_absolute() else (base / p).resolve()

    imgs = []
    for item in cfg["images"]:
        if isinstance(item, str):
            item = {"path": item}
        item = dict(item)
        item["path"] = resolve(item["path"])
        item.setdefault("name", item["path"].stem)
        imgs.append(item)
    if not imgs:
        raise ValueError("config has no 'images'")
    names = [i["name"] for i in imgs]
    if len(set(names)) != len(names):
        raise ValueError(f"image names must be unique: {names}")
    cfg["images"] = imgs
    cfg["output_dir"] = resolve(cfg["output_dir"])
    ref = cfg["flatfield"].get("reference")
    if isinstance(ref, str):
        cfg["flatfield"]["reference"] = resolve(ref)
    elif isinstance(ref, dict):
        cfg["flatfield"]["reference"] = {k: resolve(v) for k, v in ref.items()}
    if cfg.get("reuse_settings"):
        cfg["reuse_settings"] = resolve(cfg["reuse_settings"])
    return cfg


def default_color(name: str, emission_nm: float | None) -> str:
    n = name.lower()
    if any(k in n for k in ("dapi", "hoechst", "405")):
        return "blue"
    if any(k in n for k in ("gfp", "488", "fitc", "a488")):
        return "green"
    if any(k in n for k in ("rfp", "561", "568", "mcherry", "tritc", "cy3")):
        return "red"
    if any(k in n for k in ("647", "640", "cy5", "far")):
        return "magenta"
    if emission_nm:
        if emission_nm < 480:
            return "blue"
        if emission_nm < 550:
            return "green"
        if emission_nm < 630:
            return "red"
        return "magenta"
    return "gray"


def default_label(name: str) -> str:
    # 'DAPI-50' -> 'DAPI'
    return name.split("-")[0].strip() or name
