"""The 3D view's research overlays, drawn from core.research_maps and core.motion_footprint:

- heat maps: BEV feature cells per pillar (Lift-Splat / BEVFormer-style grid, ray starvation) and the stereo
  triangulation depth sigma of the best camera pair;
- lidar rings coloured by camera pixels per lidar point (fusion association density);
- the probe (a pedestrian, a car box or a ChArUco board pinned on the ground) with its rays to the cameras that see it;
- motion: the footprints and lidar rings where they are really sampled at speed and yaw rate (trigger times, rolling
  shutter), and the lidar's temporal blind wedges against each camera's exposure.

Every figure comes from the core modules; this module only turns them into ground images, lines and triangles.
"""
from __future__ import annotations

import math

import numpy as np
from PyQt6.QtGui import QColor

from . import colormaps as CM
from .theme import THEME, STATUS

# a perceptual sequential ramp (dark = little, bright = plenty) for the research maps
RESEARCH_RAMP = ['#2b1a52', '#3b4f9a', '#2f8fb0', '#3fae8c', '#9cc95e', '#f2e27a']
RESEARCH_HEAT = {
    'bevfeat': ('BEV features', 'feature cells per 0.5 m BEV pillar, stride 16 (log; grey hatched: starved < 1)'),
    'stereo': ('Stereo σ', 'triangulation depth σ of the best camera pair, mm at 0.5 px (log; bright = sharp)'),
}
_LUT = []


def lut():
    if not _LUT:
        _LUT.append(CM.ramp(RESEARCH_RAMP))
    return _LUT[0]


def _rgba(values01, ok):
    """(H, W, 4) uint8 from values in 0..1 where ok, transparent elsewhere."""
    L = lut()
    v = np.clip(np.nan_to_num(values01), 0, 1)
    idx = np.round(v * (len(L) - 1)).astype(int)
    out = np.zeros(v.shape + (4,), np.uint8)
    out[..., :3] = L[idx]
    out[..., 3] = np.where(ok, 205, 0)
    return out


def _hatch(img, mask, rgb, period=4):
    jj, ii = np.indices(mask.shape)
    h = mask & (((ii + jj) % period) == 0)
    img[h, :3] = rgb
    img[h, 3] = 230
    return img


def bev_image(bm, k=2):
    """(RGBA uint8 with row 0 = max y, extent (x0, y0, x1, y1)) of a BevMap: log2 of the summed feature cells from
    1/4 to 64; cells no camera sees in red, starved cells (some ray, < 1 feature cell) grey and hatched."""
    f = np.asarray(bm.feats, float)
    seen = np.asarray(bm.rays) > 0
    body = np.asarray(getattr(bm, 'body', np.zeros_like(seen)), bool)
    v = (np.log2(np.maximum(f, 1e-6)) + 2.0) / 8.0                 # 1/4 .. 64 feature cells
    img = _rgba(v, seen & ~body & (f >= 1))
    starved = seen & ~body & (f < 1)
    img[starved, :3] = (120, 120, 118)
    img[starved, 3] = 150
    blind = ~seen & ~body
    img[blind, :3] = np.round(CM.hex_rgb(STATUS['critical']) * 255)
    img[blind, 3] = 70
    img = np.flipud(img)
    if k > 1:
        img = np.repeat(np.repeat(img, k, 0), k, 1)
    _hatch(img, np.flipud(np.repeat(np.repeat(starved, k, 0), k, 1)) if k > 1 else np.flipud(starved),
           (210, 210, 205))
    n = bm.n
    return np.ascontiguousarray(img), (bm.x0, bm.y0, bm.x0 + n * bm.cell_m, bm.y0 + n * bm.cell_m)


def stereo_image(sm, extent, k=4):
    """(RGBA, extent) of a StereoMap on the analysis grid: log10 of sigma from 0.5 mm (bright) to 50 mm (dark);
    cells fewer than two cameras see are left clear."""
    s = np.asarray(sm.sigma_m, float) * 1000.0
    ok = np.isfinite(s) & (s > 0)
    v = 1.0 - (np.log10(np.where(ok, s, 1.0)) - math.log10(0.5)) / 2.0
    img = np.flipud(_rgba(v, ok))
    if k > 1:
        img = np.repeat(np.repeat(img, k, 0), k, 1)
    return np.ascontiguousarray(img), extent


def ratio_color(r) -> QColor:
    """Camera pixels per lidar point, log2 from 1 to 256, on the research ramp."""
    L = lut()
    v = np.clip((math.log2(max(float(r), 1e-3))) / 8.0, 0, 1)
    c = L[int(round(v * (len(L) - 1)))]
    return QColor(int(c[0]), int(c[1]), int(c[2]))


def fusion_lines(assoc, LineDraw, bins=8):
    """Each lidar ring as short pieces coloured by camera pixels per lidar point (no camera: red)."""
    out = []
    for lid in assoc or []:
        P = np.asarray(lid['points'], float)
        if not len(P):
            continue
        ring = np.asarray(lid['ring'])
        ratio = np.asarray(lid['ratio'], float)
        seen = np.asarray(lid['cam_px_m2'], float) > 0
        for ch in np.unique(ring):
            idx = np.nonzero(ring == ch)[0]
            if len(idx) < 2:
                continue
            q = P[idx]
            # order round the lidar so the pieces join up
            c = np.asarray(lid.get('pos', q.mean(axis=0)), float)
            order = np.argsort(np.arctan2(q[:, 1] - c[1], q[:, 0] - c[0]))
            idx, q = idx[order], q[order]
            b = np.where(seen[idx], np.clip(np.floor(np.log2(np.maximum(ratio[idx], 1e-3)) / 8.0 * bins), 0,
                                            bins - 1), -1).astype(int)
            start = 0
            for j in range(1, len(idx) + 1):
                if j == len(idx) or b[j] != b[start] or np.linalg.norm(q[j] - q[j - 1]) > 1.5:
                    if j - start >= 2:
                        col = QColor(STATUS['critical']) if b[start] < 0 else \
                            ratio_color(2 ** ((b[start] + 0.5) / bins * 8.0))
                        pts = np.c_[q[start:j, :2], np.full(j - start, 0.012)]
                        out.append(LineDraw(pts, col, 2.2))
                    start = j
    return out


# ---------------------------------------------------------------- the probe
def probe_lines(probe, cams_by_tag, LineDraw):
    """The probe's outline and its rays to every camera that sees it (in that camera's colour)."""
    out = []
    pr = probe['readout']
    base = np.asarray(pr['point'], float)
    kind = pr['kind']
    amber = QColor('#e8a33d')
    if kind == 'pedestrian':
        a = np.linspace(0, 2 * math.pi, 25)
        for z in (0.01, 1.75):
            out.append(LineDraw(np.c_[base[0] + 0.25 * np.cos(a), base[1] + 0.25 * np.sin(a), np.full(25, z)], amber,
                                1.8))
        for t in np.linspace(0, 2 * math.pi, 6, endpoint=False):
            out.append(LineDraw(np.array([[base[0] + 0.25 * math.cos(t), base[1] + 0.25 * math.sin(t), 0.01],
                                          [base[0] + 0.25 * math.cos(t), base[1] + 0.25 * math.sin(t), 1.75]]),
                                amber, 1.2))
    else:
        P = np.asarray(pr.get('points', [base]), float)
        lo, hi = P.min(axis=0), P.max(axis=0)
        if kind == 'board':                              # the board's outline as its sample points span it
            hi = np.maximum(hi, lo + 1e-3)
        xs, ys, zs = (lo[0], hi[0]), (lo[1], hi[1]), (lo[2], hi[2])
        for (x0, y0), (x1, y1) in (((xs[0], ys[0]), (xs[1], ys[0])), ((xs[1], ys[0]), (xs[1], ys[1])),
                                   ((xs[1], ys[1]), (xs[0], ys[1])), ((xs[0], ys[1]), (xs[0], ys[0]))):
            for z in zs:
                out.append(LineDraw(np.array([[x0, y0, z], [x1, y1, z]]), amber, 1.8))
            out.append(LineDraw(np.array([[x0, y0, zs[0]], [x0, y0, zs[1]]]), amber, 1.2))
    for row in pr['cameras']:
        if not row.get('sees'):
            continue
        cam = cams_by_tag.get(row['tag'])
        col = THEME.cam(cam.slot) if cam is not None else amber
        for C, Q in row.get('rays', [])[:6]:
            out.append(LineDraw(np.array([C, Q], float), col, 1.1, dash=True, dash_len=0.06))
    return out


def probe_card_lines(pr):
    """The probe card's text rows: per camera that sees it, its box in pixels, incidence, dr/dθ, drift shift."""
    rows = [f'{pr["about"] if "about" in pr else pr["kind"]} at {pr["point"][0]:.2f}, {pr["point"][1]:.2f} m · '
            f'seen by {len(pr["seen_by"])}']
    for r in pr['cameras']:
        if not r.get('sees'):
            continue
        w, h = r['bbox_px']
        drift = r['drift_cm']['worst']
        rows.append(f'{r["tag"]:>3}  {w:4.0f}×{h:<4.0f}px  {r["incidence_deg"]:4.1f}°  '
                    f'dr/dθ {r["dr_dtheta_px_per_rad"]:5.0f}  '
                    f'{"∞" if not math.isfinite(drift) else f"{drift:.1f}"} cm @{pr["drift_deg"]:g}°')
    if len(rows) == 1:
        rows.append('no camera sees it')
    return rows


# ---------------------------------------------------------------- motion
def motion_lines(warp, sweep, cams_by_uid, LineDraw):
    """Warped camera footprint borders (dashed, camera colour) and warped lidar rings (thin, grey)."""
    out = []
    for uid, e in (warp or {}).items():
        cam = cams_by_uid.get(uid)
        col = THEME.cam(cam.slot) if cam is not None else THEME.color('ink2')
        for pl in e.get('border_warped', []):
            pl = np.asarray(pl, float)
            if len(pl) >= 2:
                out.append(LineDraw(np.c_[pl[:, :2], np.full(len(pl), 0.014)], col, 1.8, dash=True, dash_len=0.10))
    for lid in sweep or []:
        for ring in lid.get('rings', [])[::3]:
            w = np.asarray(ring['warped'], float)
            if len(w) >= 2:
                out.append(LineDraw(np.c_[w[:, :2], np.full(len(w), 0.010)], THEME.color('ink2', 0.6), 0.9))
    return out


def wedge_tris(wedges, cams_by_uid, TriDraw, radius=12.0):
    """Each lidar–camera temporal blind wedge as a see-through sector on the ground from the lidar, in the camera's
    colour (the azimuths the lidar samples more than the threshold away from the camera's exposure)."""
    out = []
    for w in wedges or []:
        cam = cams_by_uid.get(w['camera_uid'])
        col = THEME.cam(cam.slot) if cam is not None else QColor(STATUS['warning'])
        c = np.asarray(w['lidar_pos'], float)
        for a0, a1 in w.get('wedges', []):
            span = (a1 - a0) % 360.0 or 360.0
            n = max(2, int(span / 2.0) + 1)
            aa = np.radians(a0 + np.linspace(0, span, n))
            ring = np.c_[c[0] + radius * np.cos(aa), c[1] + radius * np.sin(aa), np.full(n, 0.008)]
            ctr = np.array([c[0], c[1], 0.008])
            tris = np.array([[ctr, ring[i], ring[i + 1]] for i in range(n - 1)], float)
            out.append((tris, col))
    return out
