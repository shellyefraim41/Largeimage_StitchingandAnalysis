"""Intensity-vs-X profile plots.

Style: one y-axis per plot (raw, background-subtracted intensity, no
normalisation), 2px lines, SEM as a light band, cell count per bin in a panel
below. Bins with too few cells are dropped from the line and shaded grey.
Every line is direct-labelled at its end and in the legend, and channels also
differ by line style (red/green is only partly colour-blind separable).
"""
from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

INK = "#0b0b0b"
INK2 = "#52514e"
GRID = "#e4e3df"
LINESTYLES = ["-", "--", "-.", ":", (0, (5, 1)), (0, (3, 1, 1, 1))]


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
    ax.plot([x[i]], [y[i]], "o", ms=5, color=color, mec="white", mew=1.5, zorder=5)


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
    ax.plot(x, y, color=color_hex, lw=2, solid_capstyle="round",
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
    fig.savefig(out_png, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_overlay(prof, labs, colors_hex, out_png, title, bin_um, min_cells, ylim=None, stat="mean"):
    fig, (ax, axn) = plt.subplots(2, 1, figsize=(11, 5.6), sharex=True,
                                  gridspec_kw={"height_ratios": [3, 1], "hspace": 0.08})
    d0 = prof[prof["channel"] == labs[0]].sort_values("x_um")
    _shade_sparse(ax, d0)
    for k, lab in enumerate(labs):
        d = prof[prof["channel"] == lab].sort_values("x_um")
        x, y = _series(d, f"cell_{stat}")
        if stat == "mean":
            _, sem = _series(d, "cell_sem")
            ax.fill_between(x, y - sem, y + sem, color=colors_hex[lab], alpha=0.12, lw=0)
        ax.plot(x, y, color=colors_hex[lab], lw=2, ls=LINESTYLES[k % len(LINESTYLES)], label=lab)
        _end_label(ax, x, y, lab, colors_hex[lab])
    ax.set_ylabel("intensity − background (a.u.)", color=INK2, fontsize=10)
    ax.set_title(title, loc="left", color=INK, fontsize=12)
    if ylim:
        ax.set_ylim(*ylim)
    ax.legend(frameon=False, fontsize=9, loc="upper right", ncol=len(labs))
    _style(ax)
    _count_panel(axn, d0, bin_um, min_cells)
    fig.savefig(out_png, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_images_compare(profs: dict, lab, series_hex, out_png, title, ylim=None, stat="mean"):
    """Same channel, all images of the experiment on one axis."""
    fig, ax = plt.subplots(figsize=(11, 4.2))
    for k, (name, prof) in enumerate(profs.items()):
        d = prof[prof["channel"] == lab].sort_values("x_um")
        x, y = _series(d, f"cell_{stat}")
        col = series_hex[k % len(series_hex)]
        ax.plot(x, y, color=col, lw=2, ls=LINESTYLES[k % len(LINESTYLES)], label=name)
        _end_label(ax, x, y, name, col)
    ax.set_xlabel("X position (µm)", color=INK2, fontsize=10)
    ax.set_ylabel(f"{stat} cell intensity − background (a.u.)", color=INK2, fontsize=10)
    ax.set_title(title, loc="left", color=INK, fontsize=12)
    if ylim:
        ax.set_ylim(*ylim)
    ax.legend(frameon=False, fontsize=9, loc="upper right")
    _style(ax)
    fig.savefig(out_png, dpi=150, bbox_inches="tight")
    plt.close(fig)
