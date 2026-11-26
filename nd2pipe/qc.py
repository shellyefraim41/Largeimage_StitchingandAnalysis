"""QC figures - one folder per image, numbered in processing order."""
from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import textwrap

import numpy as np
from matplotlib.patches import Rectangle
from skimage.segmentation import find_boundaries

from .flatfield import ratio_curve
from .plots import INK, INK2, _style


def _ds(a, k):
    h, w = a.shape[0] // k * k, a.shape[1] // k * k
    return a[:h, :w].reshape(h // k, k, w // k, k).mean((1, 3))


def _disp(a, lo=None, hi=None):
    if lo is None:
        lo, hi = np.percentile(a, [1, 99.7])
    return np.clip((a - lo) / max(hi - lo, 1e-6), 0, 1)


def focus(zhists: dict, n_z, out_png):
    """zhists: {channel: counts per plane (foreground pixels)}."""
    fig, axs = plt.subplots(1, len(zhists), figsize=(4 * len(zhists), 3), squeeze=False)
    warn = []
    for ax, (lab, h) in zip(axs[0], zhists.items()):
        frac = h / max(h.sum(), 1)
        ax.bar(np.arange(n_z), frac, color="#a9a8a2", width=0.8)
        ax.set_title(f"{lab}: plane giving the max", loc="left", fontsize=10, color=INK)
        ax.set_xlabel("Z plane", fontsize=9, color=INK2)
        _style(ax)
        edge = frac[0] + frac[-1] if n_z > 2 else 0
        if edge > 0.2:
            warn.append(f"{lab}: {edge:.0%} of in-cell pixels have their max in the first/last Z plane "
                        f"(sample may extend beyond the stack)")
    axs[0][0].set_ylabel("fraction of cell pixels", fontsize=9, color=INK2)
    fig.tight_layout()
    fig.savefig(out_png, dpi=110)
    plt.close(fig)
    return warn


def layout(mosaic_reg, pos, H, W, table, out_png):
    k = max(1, int(np.ceil(max(mosaic_reg.shape) / 3000)))
    th = _ds(mosaic_reg, k)
    fig, ax = plt.subplots(figsize=(16, 16 * th.shape[0] / th.shape[1] + 1.2))
    ax.imshow(_disp(th), cmap="gray")
    for p, (y, x) in enumerate(pos):
        ax.add_patch(Rectangle((x / k, y / k), W / k, H / k, fill=False, ec="#eda100", lw=1))
        ax.text((x + W / 2) / k, (y + H / 2) / k, str(p), color="#eda100", ha="center", va="center", fontsize=10)
    for t in table:
        i, j = t["i"], t["j"]
        cx = (pos[i][1] + pos[j][1] + W) / 2 / k
        cy = (pos[i][0] + pos[j][0] + H) / 2 / k
        ax.text(cx, cy + 0.25 * H / k, f"NCC {t['ncc']:.2f}" + ("" if t["ok"] else " (stage)"),
                color="#2a78d6" if t["ok"] else "#e34948", ha="center", fontsize=8,
                bbox=dict(fc="white", ec="none", alpha=0.7, pad=1))
    ax.set_title("tile layout (registration channel). Blue = registered, red = fell back to stage position",
                 loc="left", fontsize=11)
    ax.axis("off")
    fig.savefig(out_png, dpi=100, bbox_inches="tight")
    plt.close(fig)


def flatfield(lab, F, samples, info, method, out_png):
    fig = plt.figure(figsize=(16, 8))
    gs = fig.add_gridspec(2, 3)
    ax0 = fig.add_subplot(gs[:, 0])
    im = ax0.imshow(F, cmap="magma")
    fig.colorbar(im, ax=ax0, fraction=0.046)
    ax0.set_title(f"{lab}: illumination F ({method})\nmin {F.min():.2f}  max {F.max():.2f}", loc="left", fontsize=10)
    ax0.axis("off")
    H, W = F.shape
    ax1 = fig.add_subplot(gs[0, 1])
    ax1.plot(np.arange(W), F[H // 2], color="#2a78d6", lw=2, label="along X (centre row)")
    ax1.plot(np.arange(H), F[:, W // 2], color="#eb6834", lw=2, ls="--", label="along Y (centre column)")
    ax1.set_title("illumination profile through the centre", loc="left", fontsize=10)
    ax1.set_xlabel("pixel", fontsize=9)
    ax1.legend(frameon=False, fontsize=8)
    _style(ax1)
    if samples is not None and samples["A"].size:
        for gi, axis in enumerate(("x", "y")):
            ax = fig.add_subplot(gs[gi, 2]) if gi == 0 else fig.add_subplot(gs[1, 1:])
            x0, r0, m0 = ratio_curve(samples, None, axis)
            x1, r1, m1 = ratio_curve(samples, F, axis)
            ax.plot(x0, r0, "o-", color="#a9a8a2", lw=2, ms=4, label=f"before (overall {m0:.3f})")
            ax.plot(x1, r1, "o-", color="#2a78d6", lw=2, ms=4, label=f"after (overall {m1:.3f})")
            ax.axhline(1, color=INK2, lw=0.8)
            ax.set_title(f"same cells in neighbouring tiles: intensity ratio vs {axis.upper()} "
                         f"(ideal = 1)", loc="left", fontsize=10)
            ax.set_xlabel(f"{axis} in the first tile (px)", fontsize=9)
            ax.legend(frameon=False, fontsize=8)
            _style(ax)
    notes = []
    if info.get("unconstrained_terms"):
        notes.append("Y-only terms not constrained by overlaps -> assumed flat along Y")
    if info.get("bleach_factor"):
        notes.append(f"fitted 2nd-exposure factor {info['bleach_factor']:.3f}")
    if notes:
        fig.text(0.01, 0.01, " | ".join(notes), fontsize=9, color=INK2)
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    fig.savefig(out_png, dpi=100)
    plt.close(fig)


def seams(before, after, valid, pos, H, W, labs, out_png, n=3):
    """Crops around tile boundaries: uncorrected/no blending vs final, same display range."""
    order = np.argsort(pos[:, 1] + pos[:, 0] * 1e-3)
    bounds = []
    for a, b in zip(order[:-1], order[1:]):
        if abs(pos[b][0] - pos[a][0]) < H / 2 and pos[b][1] > pos[a][1]:
            bounds.append(int((pos[a][1] + W + pos[b][1]) / 2))
    if not bounds:
        return
    sel = [bounds[i] for i in np.linspace(0, len(bounds) - 1, min(n, len(bounds))).astype(int)]
    half = int(0.45 * W)
    fig, axs = plt.subplots(len(labs) * 2, len(sel), figsize=(5 * len(sel), 5.2 * len(labs)), squeeze=False)
    for ci, lab in enumerate(labs):
        lo, hi = np.percentile(after[ci][valid][::50], [1, 99.5])
        for si, s in enumerate(sel):
            x0, x1 = max(0, s - half), min(after.shape[2], s + half)
            for ri, (img, name) in enumerate([(before, "before"), (after, "after")]):
                ax = axs[ci * 2 + ri][si]
                ax.imshow(_disp(_ds(img[ci][:, x0:x1], 3), lo, hi), cmap="gray")
                ax.axvline((s - x0) / 3, color="#eda100", lw=0.6, alpha=0.6)
                ax.set_title(f"{lab} {name}: seam at x={s}", fontsize=9, loc="left")
                ax.axis("off")
    fig.suptitle("Seams. 'before' = no flat-field, no blending. 'after' = corrected and blended. "
                 "Yellow line = tile border.", fontsize=11)
    fig.tight_layout()
    fig.savefig(out_png, dpi=90)
    plt.close(fig)


def column_profile(mosaic, labels, valid, labs, out_png, pos=None, W=None):
    """Median of cell-free pixels per column: reveals illumination ripple / seams."""
    free = valid & (labels == 0)
    fig, ax = plt.subplots(len(labs), 1, figsize=(14, 2.4 * len(labs)), squeeze=False, sharex=True)
    k = 16
    for ci, lab in enumerate(labs):
        m = mosaic[ci]
        cols = []
        for x in range(0, m.shape[1] - k + 1, k):
            v = m[:, x:x + k][free[:, x:x + k]]
            cols.append(np.median(v) if v.size > 50 else np.nan)
        a = ax[ci][0]
        a.plot(np.arange(len(cols)) * k, cols, color="#52514e", lw=1.2)
        if pos is not None:
            for p in pos:
                a.axvline(p[1], color="#eda100", lw=0.6)
                a.axvline(p[1] + W, color="#eda100", lw=0.6, ls=":")
        a.set_ylabel(lab, fontsize=9)
        _style(a)
    ax[0][0].set_title("background (cell-free pixels) per column. A repeating pattern with the tile "
                       "period = uncorrected illumination", loc="left", fontsize=10)
    ax[-1][0].set_xlabel("x (px)")
    fig.tight_layout()
    fig.savefig(out_png, dpi=100)
    plt.close(fig)


def background(edges, bkg, pixel_um, labs, out_png):
    fig, ax = plt.subplots(figsize=(11, 3.2))
    c = 0.5 * (edges[1:] + edges[:-1]) * pixel_um
    from .plots import LINESTYLES
    for k, lab in enumerate(labs):
        ax.plot(c, bkg[k], lw=2, ls=LINESTYLES[k % len(LINESTYLES)], label=lab,
                color=["#2a78d6", "#008300", "#e34948", "#d55181", "#52514e"][k % 5])
    ax.set_xlabel("X position (µm)", fontsize=9)
    ax.set_ylabel("background (a.u.)", fontsize=9)
    ax.set_title("local background subtracted from the profiles (median of cell-free pixels per bin)",
                 loc="left", fontsize=10)
    ax.legend(frameon=False, fontsize=8)
    _style(ax)
    fig.tight_layout()
    fig.savefig(out_png, dpi=110)
    plt.close(fig)


def segmentation(seg_img, labels, cells, out_png, crop=500):
    """Crops at dense, medium and sparse regions with mask outlines."""
    H, W = labels.shape
    if len(cells) == 0:
        return
    xs = cells["x_px"].to_numpy()
    hist, edges = np.histogram(xs, bins=max(4, W // crop))
    order = np.argsort(hist)
    nz = order[hist[order] > 0]
    picks = [nz[-1], nz[len(nz) // 2], nz[0], nz[int(len(nz) * 0.8)]] if len(nz) else []
    fig, axs = plt.subplots(1, len(picks), figsize=(5 * len(picks), 5.4), squeeze=False)
    for ax, b in zip(axs[0], picks):
        sub = cells[(xs >= edges[b]) & (xs < edges[b + 1])]
        cy = int(np.median(sub["y_px"]))
        cx = int(np.median(sub["x_px"]))
        y0, x0 = int(np.clip(cy - crop // 2, 0, max(0, H - crop))), int(np.clip(cx - crop // 2, 0, max(0, W - crop)))
        im = seg_img[y0:y0 + crop, x0:x0 + crop]
        lab = labels[y0:y0 + crop, x0:x0 + crop]
        rgb = np.dstack([im] * 3)
        rgb[find_boundaries(lab, mode="inner")] = (1.0, 0.85, 0.0)
        ax.imshow(rgb)
        ax.set_title(f"x={x0}..{x0 + crop}, y={y0}..{y0 + crop}  ({len(np.unique(lab)) - 1} cells)",
                     fontsize=9, loc="left")
        ax.axis("off")
    fig.suptitle("segmentation examples (dense to sparse regions), yellow = cell outlines", fontsize=11)
    fig.tight_layout()
    fig.savefig(out_png, dpi=100)
    plt.close(fig)


def segmentation_overview(seg_img, labels, out_png):
    k = max(1, int(np.ceil(max(labels.shape) / 4000)))
    im = _ds(seg_img, k)
    m = _ds((labels > 0).astype(np.float32), k)
    rgb = np.dstack([np.clip(im + 0.5 * m, 0, 1), im, im])
    fig, ax = plt.subplots(figsize=(18, 18 * rgb.shape[0] / rgb.shape[1] + 0.8))
    ax.imshow(rgb)
    ax.set_title("whole image: segmented area tinted red. Check that no colonies are missed.", loc="left", fontsize=10)
    ax.axis("off")
    fig.savefig(out_png, dpi=100, bbox_inches="tight")
    plt.close(fig)


def session_check(name, decisions, labs, out_png):
    """For each channel whose own flat-field fit failed: does this image's illumination (other,
    well-measured channels) match the rest of the session? If yes, the session fit is used."""
    rows = list(decisions.items())
    ncol = 1 + max(len(d.get("checks", {})) for _, d in rows)
    fig, axs = plt.subplots(len(rows), ncol, figsize=(4.6 * ncol, 3.6 * len(rows)), squeeze=False)
    for r, (lab, d) in enumerate(rows):
        checks = d.get("checks", {})
        lim = d.get("limits", {})
        for c, (k, v) in enumerate(checks.items()):
            ax = axs[r, c]
            x = np.arange(len(v["own_x"])) * 8
            ax.plot(x, v["session_x"], color="#a9a8a2", lw=3, label="rest of the session")
            ax.plot(x, v["own_x"], color="#2a78d6", lw=2, label="this image")
            ok = v["x_rms"] <= lim.get("x_rms", 1) and v["field_rms"] <= lim.get("field_rms", 1)
            ax.set_title(f"check {k}: X diff {v['x_rms']:.1%} (limit {lim.get('x_rms', 0):.0%}), "
                         f"field {v['field_rms']:.1%} (limit {lim.get('field_rms', 0):.0%})\n"
                         + ("PASS" if ok else "FAIL"), loc="left", fontsize=9,
                         color=INK if ok else "#e34948")
            ax.set_xlabel("x in tile (px)", fontsize=9)
            ax.set_ylabel("illumination (mean 1)", fontsize=9)
            ax.legend(frameon=False, fontsize=8)
            _style(ax)
        ax = axs[r, len(checks)]
        if d.get("session_x"):
            x = np.arange(len(d["session_x"])) * 8
            ax.plot(x, d["session_x"], color="#2a78d6" if d["source"] == "session" else "#a9a8a2", lw=2)
            ax.set_xlabel("x in tile (px)", fontsize=9)
            _style(ax)
        else:
            ax.axis("off")
        msg = (f"{lab}: USED session flat-field\nfrom {', '.join(d['members'])}" if d["source"] == "session"
               else f"{lab}: NOT corrected\n{d['reason']}")
        ax.set_title("\n".join(textwrap.wrap(msg, 55)), loc="left", fontsize=9,
                     color=INK if d["source"] == "session" else "#e34948")
        for c in range(len(checks) + 1, ncol):
            axs[r, c].axis("off")
    fig.suptitle(f"{name}: session flat-field for weak channels (same day + same optics, other channels must match)",
                 x=0.01, ha="left", fontsize=11)
    fig.tight_layout()
    fig.savefig(out_png, dpi=100)
    plt.close(fig)
