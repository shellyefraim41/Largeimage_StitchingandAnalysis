"""Intensity-vs-X profile plots.

Style: one y-axis per plot (background-subtracted intensity; the overlay of
several channels can be normalised per channel), 2px lines, SEM as a light band, cell count per bin in a panel
below. Bins with too few cells are dropped from the line and shaded grey.
Every line is direct-labelled at its end and in the legend, and channels also
differ by line style (red/green is only partly colour-blind separable).
"""
from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

# vector output that opens in Illustrator with editable text (TrueType, not outlines / Type 3)
matplotlib.rcParams["pdf.fonttype"] = 42
matplotlib.rcParams["ps.fonttype"] = 42
matplotlib.rcParams["svg.fonttype"] = "none"

INK = "#0b0b0b"
INK2 = "#52514e"
GRID = "#e4e3df"
LINESTYLES = ["-", "--", "-.", ":", (0, (5, 1)), (0, (3, 1, 1, 1)), (0, (1, 1)), (0, (5, 2, 1, 2, 1, 2))]
# colour-blind safe identity: colour + line style + marker shape (circle, square, triangle, star, ...)
MARKERS = ["o", "s", "^", "*", "D", "v", "P", "X"]
_ORDER = {}
_FORMATS = ["png"]


def set_formats(fmts):
    """File types for every profile plot, e.g. ['png', 'pdf'] (png is always written)."""
    fmts = [str(f).lower().lstrip(".") for f in ([fmts] if isinstance(fmts, str) else (fmts or []))]
    bad = [f for f in fmts if f not in ("png", "pdf", "svg")]
    if bad:
        raise ValueError(f"profile.plot_formats: unknown {bad}; use png, pdf, svg")
    _FORMATS[:] = ["png"] + [f for f in dict.fromkeys(fmts) if f != "png"]


def _save(fig, out_png, **kw):
    out_png = Path(out_png)
    for f in _FORMATS:
        fig.savefig(out_png.with_suffix("." + f), **kw)


def set_series_order(names):
    """Fix the style of each channel by its position in the experiment's channel list, so a
    channel looks the same in every plot (also when only some channels are shown)."""
    _ORDER.clear()
    _ORDER.update({n: i for i, n in enumerate(names)})


def series_style(name, k=0):
    i = _ORDER.get(name, k)
    return LINESTYLES[i % len(LINESTYLES)], MARKERS[i % len(MARKERS)]


def _ms(marker):
    return 10 if marker == "*" else 7


def _line(ax, x, y, color, name, k=0, n_series=1, lw=2, label=None, with_ls=True):
    """Line with its marker shape every ~1/12 of the length (staggered between series)."""
    ls, mk = series_style(name, k)
    step = max(1, len(x) // 12)
    off = (k * step // max(n_series, 1)) % step
    ax.plot(x, y, color=color, lw=lw, ls=ls if with_ls else "-", marker=mk, ms=_ms(mk),
            markevery=(off, step), mec="white", mew=0.8, label=name if label is None else label)


def _style(ax):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(INK2)
    ax.tick_params(colors=INK2, labelsize=9)
    ax.grid(axis="y", color=GRID, lw=0.8)
    ax.set_axisbelow(True)


def _shade_sparse(ax, d):
    for _, r in d[~d["enough_cells"]].iterrows():
        ax.axvspan(r["x0_um"], r["x1_um"], color="#f0efec", lw=0, zorder=0)


def _end_label(ax, x, y, text, color):
    ok = np.isfinite(y)
    if not ok.any():
        return
    i = np.nonzero(ok)[0][-1]
    ax.annotate(text, (x[i], y[i]), xytext=(6, 0), textcoords="offset points",
                va="center", fontsize=9, color=INK)
    mk = series_style(text)[1]
    ax.plot([x[i]], [y[i]], mk, ms=_ms(mk), color=color, mec="white", mew=1.2, zorder=5)


def _end_labels(ax, items, min_gap_pt=11):
    """items: [(x, y, text, color)]. Labels at the line ends, pushed apart vertically if they collide.
    Call after the y-limits are final."""
    pts = []
    for x, y, text, color in items:
        ok = np.isfinite(y)
        if ok.any():
            i = np.nonzero(ok)[0][-1]
            pts.append([x[i], y[i], text, color])
    if not pts:
        return
    fig = ax.figure
    to_pt = 72.0 / fig.dpi
    ypt = [ax.transData.transform((px, py))[1] * to_pt for px, py, _, _ in pts]
    order = np.argsort(ypt)
    placed = {}
    last = -np.inf
    for k in order:
        placed[k] = max(ypt[k], last + min_gap_pt)
        last = placed[k]
    shift = np.mean([placed[k] - ypt[k] for k in placed])  # keep the labels centred on their lines
    for k, (px, py, text, color) in enumerate(pts):
        ax.annotate(text, (px, py), xytext=(6, placed[k] - shift - ypt[k]), textcoords="offset points",
                    va="center", fontsize=9, color=INK)
        mk = series_style(text)[1]
        ax.plot([px], [py], mk, ms=_ms(mk), color=color, mec="white", mew=1.2, zorder=5)


def _count_panel(ax, d, bin_um, min_cells):
    ax.bar(d["x_um"], d["n_cells"], width=bin_um * 0.8, color="#a9a8a2", lw=0)
    ax.axhline(min_cells, color=INK2, lw=0.8, ls=":")
    ax.set_ylabel("cells / bin", color=INK2, fontsize=9)
    ax.set_xlabel("X position (µm)", color=INK2, fontsize=10)
    _style(ax)


def _series(d, value):
    y = d[value].to_numpy(float).copy()
    y[~d["enough_cells"].to_numpy()] = np.nan
    return d["x_um"].to_numpy(float), y


def plot_channel(prof, lab, color_hex, out_png, title, bin_um, min_cells, ylim=None, show_pixel=True, stat="mean"):
    d = prof[prof["channel"] == lab].sort_values("x_um")
    fig, (ax, axn) = plt.subplots(2, 1, figsize=(11, 5.6), sharex=True,
                                  gridspec_kw={"height_ratios": [3, 1], "hspace": 0.08})
    _shade_sparse(ax, d)
    x, y = _series(d, f"cell_{stat}")
    if stat == "mean":
        _, sem = _series(d, "cell_sem")
        ax.fill_between(x, y - sem, y + sem, color=color_hex, alpha=0.15, lw=0)
    _line(ax, x, y, color_hex, lab, with_ls=False,
          label=f"{lab}: mean of cells ± SEM" if stat == "mean" else f"{lab}: median of cells")
    _end_label(ax, x, y, lab, color_hex)
    if show_pixel:
        xp, yp = _series(d, "pixel_mean")
        ax.plot(xp, yp, color=color_hex, lw=1.2, ls="--", alpha=0.8, label=f"{lab}: mean of cell pixels")
    ax.set_ylabel("intensity − background (a.u.)", color=INK2, fontsize=10)
    ax.set_title(title, loc="left", color=INK, fontsize=12)
    if ylim:
        ax.set_ylim(*ylim)
    ax.legend(frameon=False, fontsize=9, loc="upper right")
    _style(ax)
    _count_panel(axn, d, bin_um, min_cells)
    _save(fig, out_png, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _overlay_lines(ax, prof, labs, colors_hex, stat, suffix):
    items = []
    for k, lab in enumerate(labs):
        d = prof[prof["channel"] == lab].sort_values("x_um")
        x, y = _series(d, f"cell_{stat}{suffix}")
        if stat == "mean":
            _, sem = _series(d, f"cell_sem{suffix}")
            ax.fill_between(x, y - sem, y + sem, color=colors_hex[lab], alpha=0.12, lw=0)
        _line(ax, x, y, colors_hex[lab], lab, k, len(labs))
        items.append((x, y, lab, colors_hex[lab]))
    if suffix:
        ax.axhline(1, color=INK2, lw=0.8, ls=":", zorder=1)
    return items


def plot_overlay(prof, labs, colors_hex, out_png, title, bin_um, min_cells, ylim=None, stat="mean",
                 suffix="", ylabel="intensity − background (a.u.)"):
    """All selected channels on one axis. suffix='_norm' plots the normalised columns."""
    fig, (ax, axn) = plt.subplots(2, 1, figsize=(11, 5.6), sharex=True,
                                  gridspec_kw={"height_ratios": [3, 1], "hspace": 0.08})
    d0 = prof[prof["channel"] == labs[0]].sort_values("x_um")
    _shade_sparse(ax, d0)
    items = _overlay_lines(ax, prof, labs, colors_hex, stat, suffix)
    ax.set_ylabel(ylabel, color=INK2, fontsize=10)
    ax.set_title(title, loc="left", color=INK, fontsize=12)
    if ylim:
        ax.set_ylim(*ylim)
    ax.legend(frameon=False, fontsize=9, loc="upper right", ncol=len(labs))
    _style(ax)
    _count_panel(axn, d0, bin_um, min_cells)
    _end_labels(ax, items)
    _save(fig, out_png, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_image_with_profile(rgb, width_um, height_um, prof, labs, colors_hex, out_png, title, bin_um,
                            min_cells, ylim=None, stat="mean", suffix="", ylabel="", fig_w=16.0):
    """Merged image on top, profile and cell count below, on the same X axis (µm).
    Axes are placed in inches so the image keeps its true aspect ratio and lines up with the plot."""
    left, right, top, bottom, gap = 1.0, 0.9, 0.45, 0.6, 0.12
    ax_w = fig_w - left - right
    img_h = ax_w * height_um / width_um
    plot_h, cnt_h = 3.0, 0.9
    fig_h = top + img_h + gap + plot_h + gap + cnt_h + bottom
    fig = plt.figure(figsize=(fig_w, fig_h))

    def rect(y_from_bottom, h):
        return [left / fig_w, y_from_bottom / fig_h, ax_w / fig_w, h / fig_h]

    axn = fig.add_axes(rect(bottom, cnt_h))
    ax = fig.add_axes(rect(bottom + cnt_h + gap, plot_h), sharex=axn)
    axi = fig.add_axes(rect(bottom + cnt_h + gap + plot_h + gap, img_h), sharex=axn)
    axi.imshow(rgb, extent=[0, width_um, height_um, 0], aspect="auto", interpolation="lanczos")
    axi.set_yticks([])
    axi.tick_params(labelbottom=False, bottom=False)
    for s in axi.spines.values():
        s.set_visible(False)
    axi.set_title(title, loc="left", color=INK, fontsize=12)
    d0 = prof[prof["channel"] == labs[0]].sort_values("x_um")
    _shade_sparse(ax, d0)
    items = _overlay_lines(ax, prof, labs, colors_hex, stat, suffix)
    ax.set_ylabel(ylabel, color=INK2, fontsize=10)
    if ylim:
        ax.set_ylim(*ylim)
    ax.legend(frameon=False, fontsize=9, loc="upper left", ncol=len(labs))
    ax.tick_params(labelbottom=False)
    _style(ax)
    _count_panel(axn, d0, bin_um, min_cells)
    axn.set_xlim(0, width_um)
    _end_labels(ax, items)
    _save(fig, out_png, dpi=200)
    plt.close(fig)


def plot_images_compare(profs: dict, lab, series_hex, out_png, title, ylim=None, stat="mean"):
    """Same channel, all images of the experiment on one axis."""
    fig, ax = plt.subplots(figsize=(11, 4.2))
    items = []
    saved = dict(_ORDER)
    set_series_order(list(profs))  # here the series are images, not channels
    for k, (name, prof) in enumerate(profs.items()):
        d = prof[prof["channel"] == lab].sort_values("x_um")
        x, y = _series(d, f"cell_{stat}")
        col = series_hex[k % len(series_hex)]
        _line(ax, x, y, col, name, k, len(profs))
        items.append((x, y, name, col))
    ax.set_xlabel("X position (µm)", color=INK2, fontsize=10)
    ax.set_ylabel(f"{stat} cell intensity − background (a.u.)", color=INK2, fontsize=10)
    ax.set_title(title, loc="left", color=INK, fontsize=12)
    if ylim:
        ax.set_ylim(*ylim)
    ax.legend(frameon=False, fontsize=9, loc="upper right")
    _style(ax)
    _end_labels(ax, items)
    _save(fig, out_png, dpi=150, bbox_inches="tight")
    plt.close(fig)
    _ORDER.clear()
    _ORDER.update(saved)
