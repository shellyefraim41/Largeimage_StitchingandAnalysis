"""Experiment pipeline.

Pass 1 (per image)   : max projection -> camera offset -> flat-field -> stitching
                       -> corrected mosaic (cached in <out>/<image>/data).
Pass 2 (experiment)  : shared segmentation normalisation (or reused settings).
Pass 3 (per image)   : Cellpose -> background -> per-cell table -> X profiles.
Pass 4 (experiment)  : shared display ranges -> channel / merged images ->
                       profile plots with shared axes -> experiment summary.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import tifffile

from . import flatfield as ffm
from . import measure, plots, qc, render, segment, stitch
from .config import IMAGE_SERIES_HEX, LUT_RGB, PLOT_HEX, default_color, default_label, load_config
from .io_nd2 import iter_mips, read_info


class Logger:
    def __init__(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        self.fh = open(path, "a", encoding="utf-8")

    def __call__(self, *msg):
        line = " ".join(str(m) for m in msg)
        stamp = time.strftime("%H:%M:%S")
        print(f"[{stamp}] {line}", flush=True)
        self.fh.write(f"[{stamp}] {line}\n")
        self.fh.flush()


# ------------------------------------------------------------------ helpers
def channel_setup(cfg, info):
    labs, colors = [], {}
    for name, em in zip(info.channels, info.emission_nm):
        user = cfg["channels"].get(name, {}) or {}
        lab = user.get("label", default_label(name))
        labs.append(lab)
        colors[lab] = user.get("color", default_color(name, em))
    return labs, colors


def resolve_selection(sel, labs, names, what):
    out = []
    for s in sel:
        if s in labs:
            out.append(s)
        elif s in names:
            out.append(labs[names.index(s)])
        else:
            raise ValueError(f"{what}: unknown channel '{s}'. Available: {labs}")
    return out


NUCLEAR_KEYS = ("dapi", "hoechst", "405", "nuc")


def nuclear_channel(labs, names, emission_nm):
    """The DAPI/Hoechst channel: by name, else the shortest emission wavelength."""
    for lab, name in zip(labs, names):
        if any(k in f"{lab} {name}".lower() for k in NUCLEAR_KEYS):
            return lab
    ems = [e if e else np.inf for e in emission_nm]
    return labs[int(np.argmin(ems))] if np.isfinite(min(ems)) else labs[0]


def segmentation_channels(cfg, labs, names, emission_nm, log=print):
    """Requested segmentation channels that exist in the file; the nuclear channel otherwise.
    Lets one config work on microscopes that name channels differently."""
    req = cfg["segmentation"]["channels"]
    if isinstance(req, str):
        req = None if req.lower() == "auto" else [req]
    found = []
    for s in req or []:
        if s in labs:
            found.append(s)
        elif s in names:
            found.append(labs[names.index(s)])
        else:
            log(f"WARNING: segmentation channel '{s}' is not in this data (channels: {labs}) -> skipped")
    if not found:
        found = [nuclear_channel(labs, names, emission_nm)]
        if req:
            log(f"WARNING: none of the segmentation channels {req} found -> using nuclear channel '{found[0]}'")
    return found


def describe_channels(config_path):
    """Print what is in each ND2 of a config (channels, labels, colours, segmentation) without running."""
    cfg = load_config(config_path)
    for it in cfg["images"]:
        info = read_info(it["path"])
        labs, colors = channel_setup(cfg, info)
        seg = segmentation_channels(cfg, labs, info.channels, info.emission_nm)
        print(f"\n{it['name']}: {it['path']}")
        print(f"  {info.kind}, {info.n_pos} position(s), Z={info.n_z}, {info.height}x{info.width} px, "
              f"{info.pixel_um:.3f} µm/px")
        for name, lab, em in zip(info.channels, labs, info.emission_nm):
            tag = "  <- segmentation" if lab in seg else ""
            print(f"    ND2 name '{name}'  ->  label {lab:<10} colour {colors[lab]:<8} "
                  f"emission {em if em else '?'} nm{tag}")


def ask_merge(cfg, labs, names, log):
    sel = cfg["merge_channels"]
    if isinstance(sel, list):
        ok = [s for s in sel if s in labs or s in names]
        for s in sel:
            if s not in ok:
                log(f"WARNING: merge channel '{s}' is not in this data (channels: {labs}) -> skipped")
        if not ok:
            log("WARNING: no merge channel found -> merging all channels")
            return list(labs)
        return resolve_selection(ok, labs, names, "merge_channels")
    sel = str(sel).strip().lower()
    if sel == "ask":
        while True:
            try:
                ans = input(f"Channels to merge {labs}  [Enter = all, comma list e.g. "
                            f"{','.join(labs[:2])}, 'none' = no merge]: ").strip()
            except (EOFError, OSError):
                log("merge_channels = ask but no input available -> merging all channels")
                return list(labs)
            if ans == "":
                return list(labs)
            if ans.lower() == "none":
                return []
            # accept any capitalisation, e.g. 'dapi,rfp'
            low = {l.lower(): l for l in labs}
            want = [a.strip() for a in ans.split(",") if a.strip()]
            bad = [a for a in want if a.lower() not in low]
            if not bad:
                return [low[a.lower()] for a in want]
            print(f"  unknown channel(s) {bad}; choose from {labs}")
    if sel == "all":
        return list(labs)
    if sel == "none":
        return []
    return resolve_selection([sel], labs, names, "merge_channels")


def auto_offset(cfg, arr):
    if cfg["camera_offset"] != "auto":
        return np.full(arr.shape[1], float(cfg["camera_offset"]), np.float32)
    out = []
    for c in range(arr.shape[1]):
        v = arr[:, c, ::7, ::7]
        v = v[v > 0]  # pre-stitched images are zero-padded outside the tiles
        out.append(np.percentile(v, 0.1) if v.size else 0.0)
    return np.array(out, np.float32)


def tile_backgrounds(mips, offset, reg_idx):
    """Median of cell-free pixels per tile and channel, (P, C).

    Subtracted BEFORE dividing by the flat-field: the MIP noise floor is not
    proportional to the illumination, so dividing it would recreate seams."""
    from scipy.ndimage import maximum_filter, uniform_filter
    from skimage.filters import threshold_otsu
    P, C = mips.shape[:2]
    sm = np.stack([uniform_filter(m[reg_idx].astype(np.float32), 15) for m in mips]) - offset[reg_idx]
    lg = np.log1p(np.clip(sm, 0, None))
    thr = threshold_otsu(lg[:, ::8, ::8])
    b = np.full((P, C), np.nan, np.float32)
    for p in range(P):
        free = ~maximum_filter(lg[p] > thr, size=31)
        if free.mean() > 0.02:
            for c in range(C):
                b[p, c] = np.median(mips[p, c][free]) - offset[c]
    for c in range(C):
        col = b[:, c]
        col[~np.isfinite(col)] = np.nanmedian(col) if np.isfinite(col).any() else 0.0
    return b


def zhists(mips, zarg, offset, n_z):
    out = []
    for c in range(mips.shape[1]):
        m = mips[:, c].astype(np.float32) - offset[c]
        thr = 0.3 * np.percentile(m[:, ::5, ::5], 99.5)
        fg = m > max(thr, 10)
        out.append(np.bincount(zarg[:, c][fg].ravel(), minlength=n_z)[:n_z])
    return out


# ------------------------------------------------------------------ pass 1
def build_mosaic(cfg, item, labs, reg_idx, log):
    name = item["name"]
    base = cfg["output_dir"] / name
    ddir, qdir = base / "data", base / "QC"
    ddir.mkdir(parents=True, exist_ok=True)
    qdir.mkdir(parents=True, exist_ok=True)
    f_mosaic, f_valid, f_meta = ddir / f"{name}_corrected.tif", ddir / f"{name}_valid.tif", ddir / "meta.json"
    if f_mosaic.exists() and f_valid.exists() and f_meta.exists() and not cfg["_force"]:
        log(f"  [{name}] using cached corrected mosaic")
        return tifffile.imread(str(f_mosaic)), tifffile.imread(str(f_valid)).astype(bool), \
            json.loads(f_meta.read_text())

    info = read_info(item["path"])
    log(f"  [{name}] {info.kind}: {info.n_pos} position(s), Z={info.n_z}, channels={info.channels}, "
        f"{info.width}x{info.height}px @ {info.pixel_um:.4f} µm")
    t0 = time.time()
    mips, zargs = [], []
    for p, mip, zarg in iter_mips(info.path):
        mips.append(mip)
        zargs.append(zarg)
    mips = np.stack(mips)
    zargs = np.stack(zargs)
    log(f"  [{name}] max projection done ({time.time() - t0:.0f}s)")
    offset = auto_offset(cfg, mips)
    log(f"  [{name}] camera offset per channel: {offset.tolist()}")
    meta = {"name": name, "source": str(info.path), "kind": info.kind, "pixel_um": info.pixel_um,
            "channels": info.channels, "labels": labs, "n_z": info.n_z, "offset": offset.tolist(),
            "saturation": info.saturation, "warnings": [], "flatfield": {}}
    sat = [float((mips[:, c] >= info.saturation).mean()) for c in range(mips.shape[1])]
    meta["saturated_fraction"] = dict(zip(labs, sat))
    for lab, s in zip(labs, sat):
        if s > 1e-4:
            meta["warnings"].append(f"{lab}: {s:.3%} of pixels saturated")
    meta["warnings"] += qc.focus(dict(zip(labs, zhists(mips, zargs, offset, info.n_z))), info.n_z,
                                 qdir / "01_focus_zplane_of_max.png")
    del zargs

    if info.kind == "stitched":
        mosaic = (mips[0].astype(np.float32) - offset[:, None, None])
        valid = mips[0].max(0) > 0
        meta["flatfield"] = {lab: "not applied: pre-stitched image has no tile layout" for lab in labs}
        meta["warnings"].append("pre-stitched image: illumination NOT corrected (see QC column profile)")
        meta["positions"] = None
    else:
        P, C, H, W = mips.shape
        scfg = cfg["stitching"]
        pos, table, sinfo = stitch.register_tiles(mips[:, reg_idx], info.stage_xy_um, info.pixel_um,
                                                  offset[reg_idx], scfg["max_shift_px"], scfg["min_ncc"])
        n_bad = sum(not t["ok"] for t in table)
        log(f"  [{name}] registered {len(table)} pairs ({n_bad} fell back to stage), "
            f"orientation {sinfo['orientation']}")
        if n_bad:
            meta["warnings"].append(f"{n_bad}/{len(table)} tile pairs could not be registered (stage position used)")
        pd.DataFrame(table).to_csv(qdir / "02_registration_pairs.csv", index=False)
        meta["positions"] = pos.tolist()
        meta["tile_shape"] = [H, W]
        meta["stitching"] = sinfo
        pairs = [(t["i"], t["j"], t["final_dy"], t["final_dx"]) for t in table]
        tbkg = tile_backgrounds(mips, offset, reg_idx)
        meta["tile_background"] = {lab: tbkg[:, c].round(2).tolist() for c, lab in enumerate(labs)}
        pd.DataFrame(tbkg, columns=labs).rename_axis("tile").to_csv(qdir / "02_tile_background.csv")
        sub = offset[None, :] + tbkg  # (P, C) subtracted before flat-field division

        # ---- flat-field
        fcfg = cfg["flatfield"]
        method = fcfg["method"]
        if method == "auto":
            method = "reference" if fcfg.get("reference") else "overlap"
        F = np.ones((C, H, W), np.float32)
        samples_all = {}
        if method == "reference":
            F, notes = ffm.from_reference(fcfg["reference"], info.channels, offset, (H, W),
                                          fcfg["smooth_sigma_px"])
            meta["flatfield"] = {labs[i]: notes.get(ch, "") for i, ch in enumerate(info.channels)}
        for c, lab in enumerate(labs):
            samples_all[lab] = ffm.overlap_samples(mips[:, c], pairs, sub[:, c], info.saturation)
            if method == "overlap":
                Fc, finfo = ffm.fit_overlap_model(samples_all[lab], H, W, fcfg["poly_degree"], fcfg["bleach_term"])
                if Fc is None or finfo["n_points"] < fcfg["min_points"]:
                    meta["flatfield"][lab] = f"overlap: too few cell points ({finfo['n_points']}) -> not corrected"
                    meta["warnings"].append(f"{lab}: flat-field not corrected (too little signal in overlaps)")
                else:
                    F[c] = Fc
                    meta["flatfield"][lab] = (f"overlap fit, {finfo['n_points']} points, "
                                              f"F range {Fc.min():.2f}-{Fc.max():.2f}")
                    if finfo.get("unconstrained_terms"):
                        meta["flatfield"][lab] += " (Y profile not constrained: no vertical neighbours)"
            else:
                finfo = {}
            if method == "none":
                meta["flatfield"][lab] = "none"
            qc.flatfield(lab, F[c], samples_all[lab], finfo, method,
                         qdir / f"03_flatfield_{c + 1}_{lab}.png")
            x, r, m_after = ffm.ratio_curve(samples_all[lab], F[c], "x")
            meta.setdefault("overlap_ratio_after", {})[lab] = float(m_after) if np.size(m_after) else None
        tifffile.imwrite(str(ddir / f"{name}_flatfield.tif"), F, imagej=True, metadata={"axes": "CYX"})
        log(f"  [{name}] flat-field ({method}): " + "; ".join(f"{k}: {v}" for k, v in meta["flatfield"].items()))

        # ---- assemble
        corr = (mips.astype(np.float32) - sub[:, :, None, None]) / F[None]
        mosaic, valid, _ = stitch.assemble(corr, pos, scfg["blend"])
        before, _, _ = stitch.assemble(mips.astype(np.float32) - sub[:, :, None, None], pos, "nearest")
        qc.layout(mosaic[reg_idx], pos, H, W, table, qdir / "02_tile_layout.png")
        qc.seams(before, mosaic, valid, pos, H, W, labs, qdir / "04_seams_before_after.png")
        del before, corr
        log(f"  [{name}] mosaic {mosaic.shape[2]}x{mosaic.shape[1]} px")

    tifffile.imwrite(str(f_mosaic), mosaic.astype(np.float32), imagej=True,
                     resolution=(1 / info.pixel_um, 1 / info.pixel_um),
                     metadata={"axes": "CYX", "unit": "um"})
    tifffile.imwrite(str(f_valid), valid.astype(np.uint8))
    f_meta.write_text(json.dumps(meta, indent=2))
    return mosaic, valid, meta


# ------------------------------------------------------------------ main
def run(config_path, force=False, force_segmentation=False):
    cfg = load_config(config_path)
    cfg["_force"] = force
    out = cfg["output_dir"]
    exp_dir = out / "experiment"
    exp_dir.mkdir(parents=True, exist_ok=True)
    log = Logger(exp_dir / "pipeline_log.txt")
    log(f"=== experiment '{cfg['experiment_name']}' : {len(cfg['images'])} image(s) -> {out}")

    # channels must match across the experiment
    infos = {it["name"]: read_info(it["path"]) for it in cfg["images"]}
    first = next(iter(infos.values()))
    labs, colors = channel_setup(cfg, first)
    for n, inf in infos.items():
        if channel_setup(cfg, inf)[0] != labs:
            raise ValueError(f"{n}: channels {inf.channels} differ from {first.channels}")
    names = first.channels
    scfg = cfg["segmentation"]
    seg_labs = segmentation_channels(cfg, labs, names, first.emission_nm, log)
    reg_lab = cfg["stitching"]["registration_channel"]
    reg_lab = resolve_selection([reg_lab], labs, names, "registration_channel")[0] if reg_lab else seg_labs[-1]
    merge_sel = ask_merge(cfg, labs, names, log)
    log(f"channels {labs} | segmentation on {seg_labs} | registration on {reg_lab} | merge {merge_sel or 'none'}")

    reused = json.loads(Path(cfg["reuse_settings"]).read_text()) if cfg.get("reuse_settings") else None
    if reused:
        log(f"reusing display/segmentation settings from {cfg['reuse_settings']}")

    # ---- pass 1
    log("--- pass 1: projection, flat-field, stitching")
    data = {}
    for it in cfg["images"]:
        mosaic, valid, meta = build_mosaic(cfg, it, labs, labs.index(reg_lab), log)
        data[it["name"]] = {"mosaic": mosaic, "valid": valid, "meta": meta}

    # ---- pass 2: experiment-wide segmentation normalisation
    if reused and "segmentation_norm" in reused:
        seg_norm = reused["segmentation_norm"]
    else:
        seg_norm = {}
        lo_p, hi_p = scfg["norm_percentiles"]
        for lab in seg_labs:
            c = labs.index(lab)
            pool = np.concatenate([d["mosaic"][c][d["valid"]][::37] for d in data.values()])
            seg_norm[lab] = [float(np.percentile(pool, lo_p)), float(np.percentile(pool, hi_p))]
    log(f"--- pass 2: segmentation normalisation {seg_norm}")

    # ---- pass 3: segmentation + measurements
    log("--- pass 3: segmentation and measurements")
    pcfg, bcfg = cfg["profile"], cfg["background"]
    for name, d in data.items():
        base = out / name
        mosaic, valid, meta = d["mosaic"], d["valid"], d["meta"]
        px = meta["pixel_um"]
        f_lab = base / "data" / f"{name}_labels.tif"
        normed = [segment.normalise(mosaic[labs.index(l)], *seg_norm[l]) for l in seg_labs]
        if f_lab.exists() and not (force or force_segmentation):
            labels = tifffile.imread(str(f_lab)).astype(np.int32)
            log(f"  [{name}] using cached segmentation ({labels.max()} cells)")
        else:
            t0 = time.time()
            labels = segment.segment_mosaic(normed, valid, scfg, log)
            tifffile.imwrite(str(f_lab), labels, compression="zlib")
            log(f"  [{name}] segmentation: {labels.max()} cells ({time.time() - t0:.0f}s)")
        labels, removed = segment.filter_labels(labels, normed[0], px, scfg)
        log(f"  [{name}] post-filter removed {removed['too_small']} small and {removed['too_dim']} dim "
            f"objects -> {labels.max()} cells")
        edges, bkg = measure.local_background(mosaic, labels, valid, px, bcfg)
        cells = measure.cell_table(mosaic, labels, labs, px, edges, bkg)
        n_out = measure.flag_outliers(cells, labs, pcfg.get("outlier_mad"))
        if any(n_out.values()):
            log(f"  [{name}] outlier cells excluded from profiles: {n_out}")
        prof = measure.x_profiles(cells, mosaic, labels, valid, labs, px, edges, bkg, pcfg)
        (base / "tables").mkdir(exist_ok=True)
        cells.to_csv(base / "tables" / f"{name}_cells.csv", index=False)
        prof.to_csv(base / "tables" / f"{name}_profile_x.csv", index=False)
        qdir = base / "QC"
        qc.segmentation(normed[0], labels, cells, qdir / "05_segmentation_examples.png")
        qc.segmentation_overview(normed[0], labels, qdir / "06_segmentation_overview.png")
        qc.background(edges, bkg, px, labs, qdir / "07_background.png")
        pos = np.array(meta["positions"]) if meta.get("positions") else None
        qc.column_profile(mosaic, labels, valid, labs, qdir / "08_background_per_column.png",
                          pos, meta["tile_shape"][1] if pos is not None else None)
        d.update(labels=labels, cells=cells, prof=prof, edges=edges, bkg=bkg)
        log(f"  [{name}] {len(cells)} cells measured, {int(prof.groupby('x_um')['enough_cells'].first().sum())}"
            f"/{prof['x_um'].nunique()} X-bins with >= {pcfg['min_cells_per_bin']} cells")

    # ---- pass 4: display ranges, images, plots
    log("--- pass 4: images and plots with experiment-wide settings")
    dcfg = cfg["display"]
    if reused and "display_ranges" in reused:
        ranges = {k: tuple(v) for k, v in reused["display_ranges"].items()}
    else:
        ranges = {}
        for c, lab in enumerate(labs):
            inc = np.concatenate([d["mosaic"][c][d["labels"] > 0][::23] for d in data.values()])
            hi = float(np.percentile(inc, dcfg["high_percentile"])) if inc.size else 1.0
            if dcfg["low"] == "background":
                lo = float(np.median(np.concatenate([d["bkg"][c] for d in data.values()])))
            else:
                pool = np.concatenate([d["mosaic"][c][d["valid"]][::37] for d in data.values()])
                lo = float(np.percentile(pool, float(dcfg["low"])))
            ranges[lab] = (lo, hi)
    for lab, v in (dcfg.get("fixed") or {}).items():
        ranges[resolve_selection([lab], labs, names, "display.fixed")[0]] = tuple(v)
    log("display ranges: " + ", ".join(f"{k}: {v[0]:.1f}-{v[1]:.1f}" for k, v in ranges.items()))

    lut = {lab: LUT_RGB.get(colors[lab], LUT_RGB["gray"]) for lab in labs}
    phex = {lab: PLOT_HEX.get(colors[lab], PLOT_HEX["gray"]) for lab in labs}

    # shared y-limits per channel (and for the overlay) across all images
    stat = pcfg.get("statistic", "mean")
    if stat not in ("mean", "median"):
        raise ValueError("profile.statistic must be 'mean' or 'median'")

    def ylim_for(sel):
        lo, hi = 0.0, 0.0
        for d in data.values():
            p = d["prof"]
            p = p[p["channel"].isin(sel) & p["enough_cells"]]
            if len(p):
                sem = p["cell_sem"] if stat == "mean" else 0
                lo = min(lo, float(np.nanmin(p[f"cell_{stat}"] - sem)))
                hi = max(hi, float(np.nanmax(p[f"cell_{stat}"] + sem)),
                         float(np.nanmax(p["pixel_mean"])) if pcfg["show_pixel_mean"] else 0)
        return (lo * 1.05, hi * 1.08 if hi > 0 else 1)

    ylims = {lab: ylim_for([lab]) for lab in labs}
    ov_sel = merge_sel if len(merge_sel) >= 2 else labs
    # one scale-bar length for the whole experiment
    sb = dict(dcfg.get("scale_bar") or {})
    if sb.get("enabled", True) and str(sb.get("length_um", "auto")).lower() == "auto":
        sb["length_um"] = reused["scale_bar_um"] if reused and reused.get("scale_bar_um") else render.nice_length_um(min(d["mosaic"].shape[2] * d["meta"]["pixel_um"]
                                                    for d in data.values()))
    if sb.get("enabled", True):
        log(f"scale bar: {render.bar_label(float(sb['length_um']))}, {sb.get('position', 'bottom-right')}")
    else:
        sb = None
    # normalisation of the multi-channel overlay (channels differ ~10x in brightness)
    norm_mode = str(pcfg.get("overlay_normalize", "experiment_mean")).lower()
    if norm_mode not in ("experiment_mean", "image_mean", "none"):
        raise ValueError("profile.overlay_normalize must be experiment_mean, image_mean or none")
    agg = np.median if stat == "median" else np.mean

    def cell_average(cell_tables, lab):
        v = np.concatenate([t.loc[~t[f"{lab}_outlier"], f"{lab}_mean"].to_numpy() for t in cell_tables])
        a = float(agg(v)) if v.size else np.nan
        return a if a > 0 else np.nan

    if norm_mode == "experiment_mean":
        if reused and reused.get("overlay_norm_factors"):
            exp_factors = {k: float(v) for k, v in reused["overlay_norm_factors"].items()}
        else:
            exp_factors = {lab: cell_average([d["cells"] for d in data.values()], lab) for lab in labs}
        factors = {n: exp_factors for n in data}
        log("overlay normalised to the experiment average cell: "
            + ", ".join(f"{k} {v:.2f}" for k, v in exp_factors.items()))
    elif norm_mode == "image_mean":
        factors = {n: {lab: cell_average([d["cells"]], lab) for lab in labs} for n, d in data.items()}
        exp_factors = None
    else:
        factors, exp_factors = None, None
    suffix = "_norm" if factors else ""
    avg_word = "median" if stat == "median" else "average"
    ov_ylabel = {"experiment_mean": f"intensity / experiment {avg_word} cell",
                 "image_mean": f"intensity / image {avg_word} cell"}.get(norm_mode, "intensity − background (a.u.)")
    for name, d in data.items():
        p = d["prof"]
        p.drop(columns=[c for c in p.columns if c.endswith("_norm")], inplace=True)
        if factors:
            f = p["channel"].map(factors[name]).astype(float)
            for col in ("cell_mean", "cell_sem", "cell_median", "pixel_mean"):
                p[f"{col}_norm"] = p[col] / f
        p.to_csv(out / name / "tables" / f"{name}_profile_x.csv", index=False)

    def ylim_overlay(sel):
        lo, hi = 0.0, 0.0
        for d in data.values():
            p = d["prof"]
            p = p[p["channel"].isin(sel) & p["enough_cells"]]
            if len(p):
                sem = p[f"cell_sem{suffix}"] if stat == "mean" else 0
                lo = min(lo, float(np.nanmin(p[f"cell_{stat}{suffix}"] - sem)))
                hi = max(hi, float(np.nanmax(p[f"cell_{stat}{suffix}"] + sem)))
        return (lo * 1.05, hi * 1.1 if hi > 0 else 1)

    ylim_ov = ylim_overlay(ov_sel)
    for name, d in data.items():
        base = out / name
        px = d["meta"]["pixel_um"]
        small = render.render_image(d["mosaic"], d["valid"], labs, lut, ranges, merge_sel, base, name, px,
                                    dcfg["preview_width_px"], sb)
        pdir = base / "profiles"
        pdir.mkdir(exist_ok=True)
        for c, lab in enumerate(labs):
            plots.plot_channel(d["prof"], lab, phex[lab], pdir / f"{name}_profile_{c + 1}_{lab}.png",
                               f"{name} - {lab}", pcfg["bin_um"], pcfg["min_cells_per_bin"], ylims[lab],
                               pcfg["show_pixel_mean"], stat)
        plots.plot_overlay(d["prof"], ov_sel, phex, pdir / f"{name}_profile_overlay_{'-'.join(ov_sel)}.png",
                           f"{name} - {' + '.join(ov_sel)}", pcfg["bin_um"], pcfg["min_cells_per_bin"], ylim_ov,
                           stat, suffix, ov_ylabel)
        if small is not None:
            H, W = d["mosaic"].shape[1:]
            plots.plot_image_with_profile(
                small, W * px, H * px, d["prof"], merge_sel, phex,
                pdir / f"{name}_merge_with_profile_{'-'.join(merge_sel)}.png",
                f"{name} - {' + '.join(merge_sel)}", pcfg["bin_um"], pcfg["min_cells_per_bin"],
                ylim_overlay(merge_sel), stat, suffix, ov_ylabel)
        log(f"  [{name}] images and plots written")

    if len(data) > 1:
        profs = {n: d["prof"] for n, d in data.items()}
        for c, lab in enumerate(labs):
            plots.plot_images_compare(profs, lab, IMAGE_SERIES_HEX, exp_dir / f"compare_images_{c + 1}_{lab}.png",
                                      f"{cfg['experiment_name']} - {lab}: all images", ylims[lab], stat)
    allp = pd.concat([d["prof"].assign(image=n) for n, d in data.items()])
    allp.to_csv(exp_dir / "all_profiles_x.csv", index=False)

    settings = {"experiment_name": cfg["experiment_name"], "channels": labs, "colors": colors,
                "merge_channels": merge_sel, "segmentation_channels": seg_labs,
                "segmentation_norm": seg_norm, "display_ranges": {k: list(v) for k, v in ranges.items()},
                "scale_bar_um": float(sb["length_um"]) if sb else None,
                "overlay_normalize": norm_mode, "overlay_norm_factors": exp_factors,
                "profile": pcfg, "background": bcfg, "flatfield": {k: str(v) for k, v in cfg["flatfield"].items()},
                "images": {n: {"source": d["meta"]["source"], "kind": d["meta"]["kind"],
                               "cells": int(d["labels"].max()), "flatfield": d["meta"]["flatfield"],
                               "overlap_ratio_after": d["meta"].get("overlap_ratio_after"),
                               "warnings": d["meta"]["warnings"]} for n, d in data.items()}}
    (exp_dir / "experiment_settings.json").write_text(json.dumps(settings, indent=2, default=str))
    lines = [f"Experiment: {cfg['experiment_name']}", ""]
    for n, s in settings["images"].items():
        lines += [f"{n} ({s['kind']}): {s['cells']} cells"]
        lines += [f"   flat-field {k}: {v}" for k, v in s["flatfield"].items()]
        if s["overlap_ratio_after"]:
            lines += ["   overlap ratio after correction (ideal 1.000): " +
                      ", ".join(f"{k} {v:.3f}" for k, v in s["overlap_ratio_after"].items() if v)]
        lines += [f"   WARNING: {w}" for w in s["warnings"]]
    (exp_dir / "summary.txt").write_text("\n".join(lines), encoding="utf-8")
    log("\n".join(lines))
    log("=== done")
