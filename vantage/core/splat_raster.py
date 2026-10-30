"""A differentiable rasteriser of anisotropic 3D Gaussians through Vantage's lens models, in torch (CUDA when there is
one, else the CPU): what core.splat_reproject renders a reconstructed street scene with, for the photometric
optimisation and for the re-rendered images and lidar sweeps of a rig.

The method is 3D Gaussian splatting's (Kerbl, Kopanas, Leimkühler and Drettakis, "3D Gaussian Splatting for Real-Time
Radiance Field Rendering", ACM TOG 42(4), 2023), written here in plain torch: no gsplat or CUDA extension is used.
* Each Gaussian has a centre μ, a covariance Σ = R S Sᵀ Rᵀ (R from a unit quaternion w, x, y, z; S the diagonal of its
  three scales, metres), an opacity o in [0, 1] and features (colour, range, …).
* Projection: the centre goes through the camera's projection (projector below; for a camera Vantage's lens model,
  core.lens: u = cx + fx·x·r(θ)/ρ, v = cy + fy·y·r(θ)/ρ with r(θ) the normalised image height of the projection and
  Kannala-Brandt k1..k4, ρ = √(x² + y²), θ = atan2(ρ, z)), and the covariance through the projection's Jacobian at the
  centre (the local affine approximation of EWA splatting, Zwicker et al. 2001): Σ₂ = J Σ_c Jᵀ + 0.3 I px². J is
  taken by central differences (step 10⁻³ of the range), so any projection works: a fisheye far beyond 90°, a
  rectilinear lens, the spherical grid of a lidar's beams.
* Compositing: per 16 × 16 pixel tile the Gaussians whose 3σ ellipse (or the radius where their alpha falls to 1/255,
  whichever is smaller) reaches it, sorted by range from the projection centre (not by z: a fisheye sees behind its
  own image plane), composited front to back: αᵢ = min(0.99, oᵢ·exp(−½ dᵀ Σ₂⁻¹ d)), dropped below 1/255;
  C = Σ cᵢ αᵢ Tᵢ + T_N · background, Tᵢ = Π_{j<i} (1 − αⱼ). Depth order is per Gaussian, not per pixel (as 3DGS).
  Long tile lists are composited in blocks with the transmittance carried over; without gradients a tile stops when
  every pixel's transmittance is below 10⁻⁴, with gradients a tile composites at most max_per_tile Gaussians (the
  front-most; the rest is cut and counted).
* Gradients flow to the centres, covariances (scales, rotations), opacities, features and the background through
  torch's autograd. The binning (which tiles a Gaussian touches, the depth order) is not differentiable, as in 3DGS.

Limits: the affine approximation of the projection is poor for a Gaussian that spans a large angle (one close to the
camera, on a fisheye's rim); the depth order is the centres' (overlapping Gaussians of different sizes can pop); no
view-dependent colour (spherical-harmonics degree 0).
"""
from __future__ import annotations

import math

import numpy as np

TILE = 16
LOWPASS = 0.3                       # px², added to every projected covariance (3DGS's dilation)
ALPHA_MIN = 1.0 / 255.0
ALPHA_MAX = 0.99
T_STOP = 1e-4
C0 = 0.28209479177387814            # the degree-0 spherical-harmonics constant (the .ply colour convention)


def torch_ok() -> bool:
    try:
        import torch  # noqa: F401
        return True
    except ImportError:                                   # pragma: no cover — torch ships with the app's environment
        return False


def device_for(pref: str = 'auto', mem_gb: float = 3.0):
    """The torch device: CUDA when asked for ('auto' or 'cuda'), present and with at least min(mem_gb, 1) GB free, its
    share of the GPU capped at mem_gb; else the CPU."""
    import torch
    if pref in ('auto', 'cuda') and torch.cuda.is_available():
        try:
            free, _ = torch.cuda.mem_get_info(0)
        except RuntimeError:                              # pragma: no cover — a busy or broken driver
            free = 0
        if free < min(mem_gb, 1.0) * 1e9:                 # the GPU is busy with other programs: the CPU
            return torch.device('cpu')
        try:
            total = torch.cuda.get_device_properties(0).total_memory
            torch.cuda.set_per_process_memory_fraction(min(max(mem_gb * 1e9 / total, 0.02), 0.95), 0)
        except (RuntimeError, AssertionError):            # pragma: no cover — a driver that refuses the cap
            pass
        return torch.device('cuda', 0)
    _cpu_threads(torch)
    return torch.device('cpu')


def _cpu_threads(torch):
    """At most 8 CPU threads (half the cores): torch's spinning worker threads slow to a crawl when the machine is
    shared with the user interface and other work."""
    import os
    n = max(1, min(8, (os.cpu_count() or 2) // 2))
    if torch.get_num_threads() > n:
        torch.set_num_threads(n)


# =========================================================================== geometry
def quat_to_rot(q):
    """(N, 4) quaternions w, x, y, z (normalised here) -> (N, 3, 3) rotation matrices (torch)."""
    import torch
    q = q / torch.clamp(torch.linalg.norm(q, dim=-1, keepdim=True), min=1e-12)
    w, x, y, z = q.unbind(-1)
    return torch.stack([1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y),
                        2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x),
                        2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)], -1).reshape(-1, 3, 3)


def rot_to_quat_np(R: np.ndarray) -> np.ndarray:
    """(N, 3, 3) rotation matrices -> (N, 4) unit quaternions w, x, y, z with w ≥ 0 (numpy; Shepperd's method)."""
    R = np.asarray(R, float).reshape(-1, 3, 3)
    n = len(R)
    q = np.zeros((n, 4))
    tr = R[:, 0, 0] + R[:, 1, 1] + R[:, 2, 2]
    cases = np.stack([tr, R[:, 0, 0], R[:, 1, 1], R[:, 2, 2]], 1).argmax(1)
    for c in range(4):
        m = cases == c
        if not m.any():
            continue
        r = R[m]
        if c == 0:
            s = np.sqrt(1.0 + tr[m]) * 2
            q[m] = np.stack([0.25 * s, (r[:, 2, 1] - r[:, 1, 2]) / s, (r[:, 0, 2] - r[:, 2, 0]) / s,
                             (r[:, 1, 0] - r[:, 0, 1]) / s], 1)
        elif c == 1:
            s = np.sqrt(1.0 + r[:, 0, 0] - r[:, 1, 1] - r[:, 2, 2]) * 2
            q[m] = np.stack([(r[:, 2, 1] - r[:, 1, 2]) / s, 0.25 * s, (r[:, 0, 1] + r[:, 1, 0]) / s,
                             (r[:, 0, 2] + r[:, 2, 0]) / s], 1)
        elif c == 2:
            s = np.sqrt(1.0 + r[:, 1, 1] - r[:, 0, 0] - r[:, 2, 2]) * 2
            q[m] = np.stack([(r[:, 0, 2] - r[:, 2, 0]) / s, (r[:, 0, 1] + r[:, 1, 0]) / s, 0.25 * s,
                             (r[:, 1, 2] + r[:, 2, 1]) / s], 1)
        else:
            s = np.sqrt(1.0 + r[:, 2, 2] - r[:, 0, 0] - r[:, 1, 1]) * 2
            q[m] = np.stack([(r[:, 1, 0] - r[:, 0, 1]) / s, (r[:, 0, 2] + r[:, 2, 0]) / s,
                             (r[:, 1, 2] + r[:, 2, 1]) / s, 0.25 * s], 1)
    q *= np.where(q[:, :1] < 0, -1.0, 1.0)
    return q / np.maximum(np.linalg.norm(q, axis=1, keepdims=True), 1e-12)


def quat_to_rot_np(q) -> np.ndarray:
    """(N, 4) w, x, y, z -> (N, 3, 3) (numpy)."""
    q = np.asarray(q, float).reshape(-1, 4)
    q = q / np.maximum(np.linalg.norm(q, axis=1, keepdims=True), 1e-12)
    w, x, y, z = q.T
    return np.stack([1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y),
                     2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x),
                     2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)], 1).reshape(-1, 3, 3)


def covariance(rot, scales):
    """Σ = R diag(s²) Rᵀ: (N, 3, 3) rotations and (N, 3) scales (metres) -> (N, 3, 3) covariances (torch)."""
    M = rot * scales[:, None, :]
    return M @ M.transpose(1, 2)


# =========================================================================== projections
def _r_norm(theta, projection: str, kb):
    """Normalised image height r(θ)/f of a projection (torch; core.lens.r_of_theta with f = 1)."""
    import torch
    if projection == 'rectilinear':
        return torch.tan(torch.clamp(theta, max=math.radians(89.9)))
    if projection == 'equidistant':
        return theta
    if projection == 'equisolid':
        return 2 * torch.sin(theta / 2)
    if projection == 'stereographic':
        return 2 * torch.tan(torch.clamp(theta, max=math.radians(179.0)) / 2)
    if projection == 'orthographic':
        return torch.sin(torch.clamp(theta, max=math.pi / 2))
    if projection == 'kannala_brandt':
        k1, k2, k3, k4 = (list(kb) + [0.0] * 4)[:4]
        t2 = theta * theta
        return theta * (1 + t2 * (k1 + t2 * (k2 + t2 * (k3 + t2 * k4))))
    raise ValueError(projection)


class LensProjector:
    """A camera's lens model on its image (pixel-edge coordinates: pixel (i, j) covers [i, i+1) × [j, j+1)):
    optical-frame points (x right, y down, z forward) -> (u, v, ok). fx, fy, cx, cy in image pixels; ok where
    θ ≤ max_theta (and in front of a rectilinear lens)."""

    def __init__(self, fx, fy, cx, cy, projection, kb=(0.0, 0.0, 0.0, 0.0), max_theta=math.pi):
        self.fx, self.fy, self.cx, self.cy = float(fx), float(fy), float(cx), float(cy)
        self.projection = str(projection)
        self.kb = tuple(float(k) for k in (list(kb) + [0.0] * 4)[:4])
        self.max_theta = float(max_theta)

    def __call__(self, P):
        import torch
        x, y, z = P.unbind(-1)
        rho = torch.sqrt(x * x + y * y + 1e-18)
        th = torch.atan2(rho, z)
        s = _r_norm(th, self.projection, self.kb) / rho
        u = self.cx + self.fx * x * s
        v = self.cy + self.fy * y * s
        ok = th <= self.max_theta
        if self.projection == 'rectilinear':
            ok = ok & (z > 1e-6)
        return u, v, ok


class SphereProjector:
    """A lidar's beam grid as an image: sensor-frame points (x forward, y left, z up) -> column u = (a − a_lo) / a_step,
    row v = (e_hi − e) / e_step, with a the azimuth (degrees, counter-clockwise from +x) relative to az_centre (wrapped
    to ±180°) and e the elevation. Pixel (i, j)'s centre looks along azimuth a_lo + (i + ½)·a_step, elevation
    e_hi − (j + ½)·e_step."""

    def __init__(self, az_centre, a_lo, a_step, e_hi, e_step, width, height):
        self.az_centre, self.a_lo, self.a_step = float(az_centre), float(a_lo), float(a_step)
        self.e_hi, self.e_step = float(e_hi), float(e_step)
        self.width, self.height = int(width), int(height)

    def __call__(self, P):
        import torch
        x, y, z = P.unbind(-1)
        h = torch.sqrt(x * x + y * y + 1e-18)
        a = torch.rad2deg(torch.atan2(y, x)) - self.az_centre
        a = torch.remainder(a + 180.0, 360.0) - 180.0
        e = torch.rad2deg(torch.atan2(z, h))
        u = (a - self.a_lo) / self.a_step
        v = (self.e_hi - e) / self.e_step
        margin = 0.25 * self.width
        ok = (e.abs() < 85.0) & (u > -margin) & (u < self.width + margin)
        return u, v, ok


# =========================================================================== the rasteriser
def _tile_lists(u, v, rad, keep, W, H, rows):
    """The (tile, Gaussian) pairs sorted by tile, then by the Gaussians' order (they arrive sorted by range): returns
    (tile id per pair, Gaussian per pair, tiles across)."""
    import torch
    dev = u.device
    TW = (W + TILE - 1) // TILE
    ty_lo, ty_hi = rows
    tx0 = torch.clamp(torch.floor((u - rad) / TILE), 0, TW - 1).long()
    tx1 = torch.clamp(torch.floor((u + rad) / TILE), 0, TW - 1).long()
    ty0 = torch.clamp(torch.floor((v - rad) / TILE), ty_lo, ty_hi - 1).long()
    ty1 = torch.clamp(torch.floor((v + rad) / TILE), ty_lo, ty_hi - 1).long()
    inside = keep & (u + rad >= 0) & (u - rad < W) & (v + rad >= ty_lo * TILE) & (v - rad < min(ty_hi * TILE, H))
    nw = (tx1 - tx0 + 1)
    cnt = torch.where(inside, nw * (ty1 - ty0 + 1), torch.zeros_like(nw))
    total = int(cnt.sum())
    if total == 0:
        z = torch.zeros(0, dtype=torch.long, device=dev)
        return z, z, TW
    gid = torch.repeat_interleave(torch.arange(len(u), device=dev), cnt)
    first = torch.cumsum(cnt, 0) - cnt
    k = torch.arange(total, device=dev) - torch.repeat_interleave(first, cnt)
    w = nw[gid]
    tile = (ty0[gid] + k // w) * TW + tx0[gid] + k % w
    order = torch.sort(tile, stable=True).indices       # gid ascends within a tile: the range order
    return tile[order], gid[order], TW


def splat(Pc, cov_c, feats, opacity, proj, width: int, height: int, bg=None, rows=None, near: float = 0.05,
          block: int = 512, max_per_tile: int = 4096, want_index: bool = False, chunk_elems: int | None = None,
          normals=None, depth_col: int | None = None, lowpass: float = LOWPASS) -> dict:
    """Render Gaussians already in the projector's frame.

    Pc (N, 3) centres, cov_c (N, 3, 3) covariances, feats (N, F), opacity (N,) in [0, 1] (torch, one device); proj a
    projector; bg (H·W, F) background per pixel (row-major) or None (zeros); rows (first, last + 1) tile rows to render
    (a rolling-shutter band; None: all). Returns {'image' (H·W, F): Σ fᵢ αᵢ Tᵢ + T·bg, 'alpha' (H·W,): 1 − T,
    'index' (H·W,) long: the Gaussian with the largest weight per pixel (−1 none; want_index), 'rendered' (H·W,)
    bool: pixels inside the rows rendered, 'cut' int: tile entries left out (max_per_tile, with gradients only),
    'pairs' int, 'median' (H·W,): without gradients and with depth_col, the median depth, the depth feature of the
    Gaussian at which the transmittance falls through ½ (NaN where it never does)}.
    Planar depth: with normals (N, 3) (unit, the projector's frame) and depth_col (the feature column holding each
    Gaussian's range), that feature varies across the splat as the range along each pixel's ray to the plane through
    the centre with that normal, to first order (r + ∇r·(p − μ₂), ∇r from the projection's Jacobian), instead of the
    centre's range everywhere: overlapping Gaussians of a surface seen obliquely then agree on its depth (front-to-back
    weights would otherwise pull it towards the nearer centres). lowpass: the dilation added to every projected
    covariance, px² (3DGS's 0.3 for images; a lidar's beams, each one ray, take less).""" 
    import torch
    from torch.utils.checkpoint import checkpoint
    dev = Pc.device
    W, H = int(width), int(height)
    F = feats.shape[1]
    TH = (H + TILE - 1) // TILE
    rows = (0, TH) if rows is None else (max(int(rows[0]), 0), min(int(rows[1]), TH))
    grad = torch.is_grad_enabled() and (Pc.requires_grad or cov_c.requires_grad or feats.requires_grad or
                                        opacity.requires_grad or (bg is not None and bg.requires_grad))
    if chunk_elems is None:
        cuda = dev.type == 'cuda'
        chunk_elems = (1 << 23 if cuda else 1 << 21) if grad else (1 << 24 if cuda else 1 << 22)
    bgf = bg if bg is not None else torch.zeros(H * W, F, device=dev, dtype=feats.dtype)
    out = {'image': bgf, 'alpha': torch.zeros(H * W, device=dev, dtype=feats.dtype),
           'index': torch.full((H * W,), -1, dtype=torch.long, device=dev) if want_index else None,
           'rendered': torch.zeros(H * W, dtype=torch.bool, device=dev), 'cut': 0, 'pairs': 0}
    if depth_col is not None and not grad:
        out['median'] = torch.full((H * W,), float('nan'), device=dev, dtype=feats.dtype)
    r0, r1 = rows[0] * TILE, min(rows[1] * TILE, H)
    out['rendered'].view(H, W)[r0:r1] = True
    if len(Pc) == 0 or r1 <= r0:
        return out
    # ---- centres and footprints
    rng = torch.linalg.norm(Pc, dim=-1)
    u, v, ok = proj(Pc)
    ok = ok & (rng > near) & (opacity > ALPHA_MIN)
    idx = torch.nonzero(ok).squeeze(1)
    if len(idx) == 0:
        return out
    P = Pc[idx]
    r = rng[idx]
    h = (1e-3 * r.detach()).clamp(min=1e-5)
    eye = torch.eye(3, device=dev, dtype=P.dtype)
    Q = torch.cat([P[None] + h[None, :, None] * eye[:, None, :], P[None] - h[None, :, None] * eye[:, None, :]])
    uq, vq, _ = proj(Q.reshape(-1, 3))
    uq, vq = uq.reshape(6, -1), vq.reshape(6, -1)
    J = torch.stack([torch.stack([(uq[k] - uq[k + 3]) / (2 * h) for k in range(3)], -1),
                     torch.stack([(vq[k] - vq[k + 3]) / (2 * h) for k in range(3)], -1)], 1)     # (n, 2, 3)
    S2 = J @ cov_c[idx] @ J.transpose(1, 2)
    a = S2[:, 0, 0] + lowpass
    b = S2[:, 0, 1]
    c = S2[:, 1, 1] + lowpass
    det = torch.clamp(a * c - b * b, min=1e-12)
    con = torch.stack([c / det, -b / det, a / det], -1)
    uu, vv = u[idx], v[idx]
    op = opacity[idx]
    with torch.no_grad():
        mid = 0.5 * (a + c)
        lam = mid + torch.sqrt(torch.clamp(mid * mid - det, min=0.0))
        m = torch.sqrt(torch.clamp(2.0 * torch.log(torch.clamp(op * 255.0, min=1.0 + 1e-6)), min=0.0)).clamp(max=3.0)
        rad = torch.ceil(m * torch.sqrt(lam))
        good = torch.isfinite(rad) & torch.isfinite(uu) & torch.isfinite(vv) & (rad < 4 * max(W, H))
        order = torch.argsort(r.detach(), stable=True)                  # front to back
        tile, gid_s, TW = _tile_lists(uu.detach()[order], vv.detach()[order], rad[order], good[order], W, H, rows)
    gid = order[gid_s]                                  # index into the visible set
    out['pairs'] = int(len(gid))
    if len(gid) == 0:
        return out
    with torch.no_grad():
        ntiles = TW * TH
        counts = torch.bincount(tile, minlength=ntiles)
        starts = torch.cumsum(counts, 0) - counts
        live = torch.nonzero(counts).squeeze(1)
        live = live[torch.argsort(counts[live], descending=True, stable=True)]
        cnt_live = counts[live].tolist()
    fv = feats[idx]
    dg = None
    if normals is not None and depth_col is not None:
        with torch.no_grad():
            dg = _depth_gradient(P.detach(), r.detach(), J.detach(), normals[idx].detach())
    lx = torch.arange(TILE, device=dev, dtype=P.dtype)
    gy, gx = torch.meshgrid(lx, lx, indexing='ij')
    gx, gy = gx.reshape(-1), gy.reshape(-1)                # pixel offsets within a tile (row-major)
    pieces_i, pieces_v, pieces_a, pieces_x, pieces_m = [], [], [], [], []
    i = 0
    nl = len(live)
    while i < nl:
        nmax = cnt_live[i]
        kmax = min(nmax, max_per_tile) if grad else nmax
        kb = min(block, kmax)
        n_c = max(1, min(nl - i, chunk_elems // (TILE * TILE * kb)))
        tiles = live[i:i + n_c]
        i += n_c
        tx, ty = tiles % TW, tiles // TW
        px = (tx[:, None] * TILE + gx[None]).to(P.dtype)
        py = (ty[:, None] * TILE + gy[None]).to(P.dtype)
        pix_ok = (px < W) & (py < H)
        pxc, pyc = px + 0.5, py + 0.5
        Tcur = torch.ones(len(tiles), TILE * TILE, device=dev, dtype=P.dtype)
        acc = torch.zeros(len(tiles), TILE * TILE, F, device=dev, dtype=feats.dtype)
        best_w = torch.zeros(len(tiles), TILE * TILE, device=dev, dtype=P.dtype) if want_index else None
        best_g = torch.full((len(tiles), TILE * TILE), -1, dtype=torch.long, device=dev) if want_index else None
        med = torch.full((len(tiles), TILE * TILE), float('nan'), device=dev, dtype=feats.dtype)
        tc = counts[tiles]
        if grad:
            out['cut'] += int(torch.clamp(tc - max_per_tile, min=0).sum())
            tc = torch.clamp(tc, max=max_per_tile)
        ts = starts[tiles]
        for k0 in range(0, kmax, kb):
            kk = min(kb, kmax - k0)
            ar = torch.arange(k0, k0 + kk, device=dev)
            valid = ar[None, :] < tc[:, None]
            pos = torch.clamp(ts[:, None] + ar[None, :], max=len(gid) - 1)
            g = gid[pos]
            if grad:
                add, Tb_last = checkpoint(_block, pxc, pyc, uu, vv, con, op, fv, g, valid, Tcur, False, dg, depth_col,
                                          use_reentrant=False)
            else:
                add, Tb_last, wgt, md, mh = _block(pxc, pyc, uu, vv, con, op, fv, g, valid, Tcur, True, dg,
                                                   depth_col)
                if md is not None:
                    med = torch.where(mh & torch.isnan(med), md, med)
                if want_index:
                    bw, bk = wgt.max(-1)
                    better = bw > best_w
                    best_w = torch.where(better, bw, best_w)
                    best_g = torch.where(better, torch.gather(g, 1, bk), best_g)
            acc = acc + add
            Tcur = Tcur * Tb_last
            if not grad and bool((Tcur < T_STOP).all()):
                break
        flat = (py * W + px).long()
        pieces_i.append(flat[pix_ok])
        pieces_v.append(acc[pix_ok])
        pieces_a.append(Tcur[pix_ok])
        if want_index:
            pieces_x.append(best_g[pix_ok])
        pieces_m.append(med[pix_ok])
    pi = torch.cat(pieces_i)
    T = torch.cat(pieces_a)
    img = bgf.index_copy(0, pi, torch.cat(pieces_v) + T[:, None] * bgf[pi])
    out['image'] = img
    out['alpha'] = out['alpha'].index_copy(0, pi, 1.0 - T)
    if depth_col is not None and not grad:
        out['median'] = out['median'].index_copy(0, pi, torch.cat(pieces_m))
    if want_index:
        bx = torch.cat(pieces_x)
        out['index'] = out['index'].index_copy(0, pi, torch.where(bx >= 0, idx[bx.clamp(min=0)], bx))
    return out


def _depth_gradient(P, r, J, n):
    """(n, 2): the derivative of the range along a pixel's ray to the plane through P with normal n, with respect to
    the pixel coordinates, at P's own pixel. A ray direction d(p) moves by T (J T)⁻¹ / r per pixel (T a basis of the
    plane across the ray, J the projection's Jacobian at P); the range is (n·P) / (n·d), so ∇r = −r (n·∂d/∂p) /
    (n·d) at d = P / r. Kept within ±r/4 per pixel (a plane seen edge-on)."""
    import torch
    d = P / r[:, None].clamp(min=1e-9)
    a = torch.where(d[:, 2:3].abs() < 0.9, torch.tensor([0.0, 0.0, 1.0], device=P.device, dtype=P.dtype),
                    torch.tensor([1.0, 0.0, 0.0], device=P.device, dtype=P.dtype))
    t1 = torch.linalg.cross(d, a)
    t1 = t1 / torch.linalg.norm(t1, dim=1, keepdim=True).clamp(min=1e-12)
    t2 = torch.linalg.cross(d, t1)
    T = torch.stack([t1, t2], 2)                                       # (n, 3, 2)
    JT = J @ T                                                         # (n, 2, 2)
    det = JT[:, 0, 0] * JT[:, 1, 1] - JT[:, 0, 1] * JT[:, 1, 0]
    ok = det.abs() > 1e-12
    inv = torch.stack([torch.stack([JT[:, 1, 1], -JT[:, 0, 1]], 1), torch.stack([-JT[:, 1, 0], JT[:, 0, 0]], 1)], 1)
    inv = inv / torch.where(ok, det, torch.ones_like(det))[:, None, None]
    dd = (T @ inv) / r[:, None, None].clamp(min=1e-9)                  # (n, 3, 2): ∂d/∂p
    nd = (n * d).sum(1)
    nd = torch.where(nd.abs() < 0.02, torch.full_like(nd, 0.02) * torch.sign(nd + 1e-12), nd)
    g = -r[:, None] * torch.einsum('ni,nij->nj', n, dd) / nd[:, None]
    lim = (r / 4.0)[:, None]
    return torch.where(ok[:, None], torch.maximum(torch.minimum(g, lim), -lim), torch.zeros_like(g))


def _block(pxc, pyc, uu, vv, con, op, fv, g, valid, Tcur, keep_w=False, dg=None, depth_col=None):
    """One block of a chunk of tiles: the Gaussians g (tiles × k, front to back) composited over the tiles' pixels
    with the transmittance Tcur carried in. Returns (Σ f α T, the block's own transmittance product[, the weights])."""
    import torch
    dx = pxc[:, :, None] - uu[g][:, None, :]
    dy = pyc[:, :, None] - vv[g][:, None, :]
    cg = con[g]
    power = -0.5 * (cg[:, None, :, 0] * dx * dx + cg[:, None, :, 2] * dy * dy) - cg[:, None, :, 1] * dx * dy
    al = torch.clamp(op[g][:, None, :] * torch.exp(torch.clamp(power, max=0.0)), max=ALPHA_MAX)
    al = torch.where(valid[:, None, :] & (al >= ALPHA_MIN), al, torch.zeros_like(al))
    Tb = torch.cumprod(1.0 - al, -1)
    Tex = torch.cat([torch.ones_like(Tb[..., :1]), Tb[..., :-1]], -1) * Tcur[..., None]
    wgt = al * Tex
    add = torch.einsum('cpk,ckf->cpf', wgt, fv[g])
    lin = None
    if dg is not None:
        lin = dg[g][:, None, :, 0] * dx + dg[g][:, None, :, 1] * dy
        extra = (wgt * lin).sum(-1)
        add = torch.cat([add[..., :depth_col], (add[..., depth_col] + extra)[..., None], add[..., depth_col + 1:]], -1)
    if keep_w:
        med = has = None
        if depth_col is not None:
            # the median depth: the depth of the Gaussian at which the transmittance falls through ½
            cross = (Tex >= 0.5) & (Tex * (1.0 - al) < 0.5)
            has = cross.any(-1)
            k = cross.to(torch.int8).argmax(-1, keepdim=True)
            dep = fv[g][:, None, :, depth_col].expand_as(al)
            if lin is not None:
                dep = dep + lin
            med = torch.gather(dep, 2, k)[..., 0]
        return add, Tb[..., -1], wgt, med, has
    return add, Tb[..., -1]


# =========================================================================== image quality
def ssim_map(a, b, mask=None):
    """SSIM (Wang, Bovik, Sheikh and Simoncelli 2004: 11 × 11 Gaussian window, σ 1.5, K1 0.01, K2 0.03, values in
    [0, 1]) of two (H, W, C) torch images: the per-pixel map averaged over channels, (H, W)."""
    import torch
    import torch.nn.functional as Fn
    x = a.permute(2, 0, 1)[None]
    y = b.permute(2, 0, 1)[None]
    C = x.shape[1]
    g = torch.exp(-(torch.arange(11, device=a.device, dtype=a.dtype) - 5) ** 2 / (2 * 1.5 ** 2))
    g = g / g.sum()
    w = (g[:, None] * g[None, :])[None, None].repeat(C, 1, 1, 1)

    def f(z):
        return Fn.conv2d(Fn.pad(z, (5, 5, 5, 5), mode='replicate'), w, groups=C)
    mx, my = f(x), f(y)
    sxx = f(x * x) - mx * mx
    syy = f(y * y) - my * my
    sxy = f(x * y) - mx * my
    c1, c2 = 0.01 ** 2, 0.03 ** 2
    s = ((2 * mx * my + c1) * (2 * sxy + c2)) / ((mx * mx + my * my + c1) * (sxx + syy + c2))
    return s[0].mean(0)


def psnr(a: np.ndarray, b: np.ndarray, mask=None) -> float | None:
    """PSNR in dB of two images in [0, 1] over the mask's pixels (all when None)."""
    d = (np.asarray(a, float) - np.asarray(b, float)) ** 2
    if mask is not None:
        d = d[np.asarray(mask, bool)]
    if not d.size:
        return None
    mse = float(d.mean())
    return 99.0 if mse <= 1e-10 else float(10 * math.log10(1.0 / mse))
