"""Reading ND2 files: metadata, and per-position maximum-intensity projections.

Frames are read one at a time (running max over Z), so even very large
pre-stitched images never need the full Z-stack in memory.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

import nd2
import numpy as np


@dataclass
class ND2Info:
    path: Path
    kind: str                       # 'tiles' or 'stitched'
    n_pos: int
    n_z: int
    channels: list[str]
    emission_nm: list[float | None]
    pixel_um: float
    height: int
    width: int
    stage_xy_um: np.ndarray         # (P, 2) stage x, y in µm
    pos_names: list[str]
    dtype: str
    saturation: int
    extra: dict = field(default_factory=dict)


def read_info(path: str | Path) -> ND2Info:
    path = Path(path)
    with nd2.ND2File(str(path)) as f:
        sizes = dict(f.sizes)
        if sizes.get("T", 1) > 1:
            raise NotImplementedError(f"{path.name}: time-lapse (T={sizes['T']}) is not supported")
        chans, ems = [], []
        for c in f.metadata.channels or []:
            chans.append(c.channel.name)
            ems.append(c.channel.emissionLambdaNm)
        n_c = sizes.get("C", 1)
        if not chans:
            chans = [f"C{i}" for i in range(n_c)]
            ems = [None] * n_c
        n_pos = sizes.get("P", 1)
        stage, names = [], []
        for loop in f.experiment:
            if loop.type == "XYPosLoop":
                for i, p in enumerate(loop.parameters.points):
                    stage.append((p.stagePositionUm.x, p.stagePositionUm.y))
                    names.append(p.name or f"P{i:03d}")
        if len(stage) != n_pos:
            # fall back to per-frame metadata
            stage, names = [], []
            seen = set()
            for k, idx in enumerate(f.loop_indices):
                p = idx.get("P", 0)
                if p in seen:
                    continue
                seen.add(p)
                pos = f.frame_metadata(k).channels[0].position.stagePositionUm
                stage.append((pos.x, pos.y))
                names.append(f"P{p:03d}")
        bits = f.attributes.bitsPerComponentSignificant
        info = ND2Info(
            path=path,
            kind="tiles" if n_pos > 1 else "stitched",
            n_pos=n_pos,
            n_z=sizes.get("Z", 1),
            channels=chans,
            emission_nm=ems,
            pixel_um=float(f.voxel_size().x),
            height=sizes["Y"],
            width=sizes["X"],
            stage_xy_um=np.asarray(stage, dtype=np.float64).reshape(-1, 2),
            pos_names=names,
            dtype=str(f.dtype),
            saturation=int(2 ** bits - 1),
        )
        try:
            info.extra["description"] = f.text_info.get("description", "")
            info.extra["date"] = f.text_info.get("date", "")
            cap = f.text_info.get("capturing", "") or ""
            info.extra["camera"] = cap.splitlines()[0].strip() if cap else ""
            mb = re.search(r"Binning:\s*(\S+)", cap)
            info.extra["binning"] = mb.group(1) if mb else ""
        except Exception:
            pass
        try:
            mic = f.metadata.channels[0].microscope
            info.extra["objective"] = mic.objectiveName or ""
            info.extra["modality"] = ",".join(mic.modalityFlags or [])
        except Exception:
            pass
    return info


def session_key(info: ND2Info):
    """Images with the same key were taken on the same day with the same optics and camera
    settings, so they share one illumination profile. None if the date is unknown."""
    date = (info.extra.get("date") or "").strip()
    if not date:
        return None
    day = date.split()[0]
    return (day, info.extra.get("objective", ""), info.extra.get("camera", ""), info.extra.get("binning", ""),
            info.extra.get("modality", ""), round(info.pixel_um, 5), info.height, info.width, tuple(info.channels))


def iter_mips(path: str | Path, positions=None):
    """Yield (p, mip, zhist) per position.

    mip   : (C, Y, X) uint16 maximum projection over Z
    zargm : (C, Y, X) uint8 index of the plane that gave the max (for focus QC)
    """
    path = Path(path)
    with nd2.ND2File(str(path)) as f:
        by_pos: dict[int, list[int]] = {}
        for k, idx in enumerate(f.loop_indices):
            by_pos.setdefault(idx.get("P", 0), []).append(k)
        wanted = sorted(by_pos) if positions is None else positions
        for p in wanted:
            mip = zarg = None
            for z, k in enumerate(by_pos[p]):
                fr = f.read_frame(k)
                if fr.ndim == 2:
                    fr = fr[None]
                if mip is None:
                    mip = fr.copy()
                    zarg = np.zeros(fr.shape, np.uint8)
                else:
                    upd = fr > mip
                    mip[upd] = fr[upd]
                    zarg[upd] = z
            yield p, mip, zarg
