"""3D surround view: the bowl the cameras' images are projected onto, as the in-car display renders it.

The bowl is a flat floor around the vehicle that turns up into a curved wall. Its footprint is a super-ellipse fitted
to the vehicle outline (squareness 2 = ellipse, larger = boxier), the floor reaches floor_m beyond the body, and the
wall rises height_m over reach_m (profile z ~ s^power). Things standing on the ground near the car stay true on the
floor; things further out land on the wall, which is what a surround-view ECU does and why the bowl shape is tuned.

Points on the bowl are parametrised by azimuth around the body centre (columns) and by s in [0, 1] from the centre
out to the rim (rows). The same remap machinery as the bird's-eye view (stitch.lut_for_points) gives, per camera, where
to sample each bowl point and with what weight; stitch() then fills the bowl texture, exposure gains and multi-band
blending included.
"""
from __future__ import annotations

import io
import json
import math
from dataclasses import dataclass, asdict

import numpy as np

from .project import Project
from .stitch import StitchLUT, lut_for_points, stitch, estimate_gains, file_tags

FLOOR_S = 0.55          # share of the radial parameter given to the floor


@dataclass
class BowlShape:
    floor_m: float = 1.6        # flat floor beyond the vehicle outline
    reach_m: float = 4.5        # horizontal extent of the wall
    height_m: float = 2.6       # wall height at the rim
    power: float = 2.0          # wall profile z = height * u^power
    squareness: float = 4.0     # super-ellipse exponent of the footprint

    def to_dict(self):
        return asdict(self)

    @staticmethod
    def from_dict(d):
        return BowlShape(**{k: float(v) for k, v in (d or {}).items() if k in BowlShape.__dataclass_fields__})


def footprint(result, project: Project | None = None):
    """(centre x, centre y, half length, half width) of the vehicle on the ground."""
    g = result.grid
    if g.body.any():
        xs = g.xs[g.body.any(axis=0)]
        ys = g.ys[g.body.any(axis=1)]
        hx = (xs.max() - xs.min()) / 2 + g.cell / 2
        hy = (ys.max() - ys.min()) / 2 + g.cell / 2
    else:
        hx, hy = 2.0, 0.9
    cx, cy = result.centre
    return float(cx), float(cy), float(hx), float(hy)


def _radius(a, ex, ey, k):
    ca, sa = np.abs(np.cos(a)), np.abs(np.sin(a))
    return 1.0 / ((ca / ex) ** k + (sa / ey) ** k) ** (1.0 / k)


def bowl_points(result, shape: BowlShape, n_az=512, n_rad=192, wrap=False):
    """World points (n_rad, n_az(+1), 3) and texture coordinates (same shape, 2) of the bowl.
    Rows go from the centre (s = 0) to the rim (s = 1); columns sample azimuth -pi..pi at cell centres, or with
    wrap=True on the edges with the first column repeated at the end (a closed mesh)."""
    cx, cy, hx, hy = footprint(result)
    if wrap:
        a = np.linspace(-math.pi, math.pi, n_az + 1)
    else:
        a = -math.pi + (np.arange(n_az) + 0.5) * 2 * math.pi / n_az
    s = np.linspace(0.0, 1.0, n_rad) if wrap else (np.arange(n_rad) + 0.5) / n_rad
    ex, ey = hx + shape.floor_m, hy + shape.floor_m
    rf = _radius(a, ex, ey, max(1.0, shape.squareness))           # floor edge per azimuth
    S, A = np.meshgrid(s, a, indexing='ij')
    RF = np.broadcast_to(rf, S.shape)
    floor = S <= FLOOR_S
    u = np.clip((S - FLOOR_S) / (1 - FLOOR_S), 0, 1)
    r = np.where(floor, S / FLOOR_S * RF, RF + u * shape.reach_m)
    z = np.where(floor, 0.0, shape.height_m * u ** max(0.2, shape.power))
    P = np.stack([cx + r * np.cos(A), cy + r * np.sin(A), z], -1)
    uv = np.stack([(A + math.pi) / (2 * math.pi), S], -1)
    return P, uv


def _body_mask(result, P):
    g = result.grid
    gi = np.clip(((P[..., 0] - g.x0) / g.cell).astype(int), 0, g.nx - 1)
    gj = np.clip(((P[..., 1] - g.y0) / g.cell).astype(int), 0, g.ny - 1)
    inside = (P[..., 0] >= g.x0) & (P[..., 0] < g.x0 + g.nx * g.cell) & (P[..., 1] >= g.y0) & \
        (P[..., 1] < g.y0 + g.ny * g.cell)
    return g.body[gj, gi] & inside & (P[..., 2] < 1e-6)


def bowl_lut(project: Project, engine, result, shape: BowlShape, n_az=512, n_rad=192, poses=None, wrap=False) -> StitchLUT:
    """Remap tables onto the bowl (rows = s from the centre, columns = azimuth)."""
    P, uv = bowl_points(result, shape, n_az, n_rad, wrap)
    lut = lut_for_points(project, engine, result, P, _body_mask(result, P), poses=poses)
    lut.points = P
    lut.uv = uv
    return lut


def bowl_texture(lut: StitchLUT, images: dict, balance=True, blend='linear') -> np.ndarray:
    """RGBA texture (rows = s from the centre, columns = azimuth) of the camera images projected onto the bowl.
    Alpha fades out where no camera sees the bowl (the top of the wall, mostly), so the background shows there."""
    import cv2
    gains = estimate_gains(lut, images) if balance and len(images) > 1 else None
    rgb = stitch(lut, images, gains=gains, blend=blend)
    a = (lut.covered | lut.body).astype(np.float32)
    a = cv2.GaussianBlur(a, (0, 0), 1.5) * a
    return np.dstack([rgb, np.clip(a * 255, 0, 255).astype(np.uint8)])


@dataclass
class BowlMesh:
    vertices: np.ndarray        # (V, 3) float32 world
    uv: np.ndarray              # (V, 2) float32, u = azimuth 0..1, v = s 0..1 (texture row)
    faces: np.ndarray           # (F, 3) int32
    rows: int
    cols: int


def bowl_mesh(result, shape: BowlShape, n_az=144, n_rad=56) -> BowlMesh:
    """Closed triangle mesh of the bowl (the centre row collapses to a point; the azimuth seam is duplicated so the
    texture wraps)."""
    P, uv = bowl_points(result, shape, n_az, n_rad, wrap=True)
    R, C = P.shape[:2]
    idx = np.arange(R * C).reshape(R, C)
    a, b = idx[:-1, :-1].ravel(), idx[:-1, 1:].ravel()
    c, d = idx[1:, :-1].ravel(), idx[1:, 1:].ravel()
    faces = np.concatenate([np.stack([a, c, d], 1), np.stack([a, d, b], 1)])
    # drop the degenerate triangles of the centre row
    V = P.reshape(-1, 3)
    area = np.linalg.norm(np.cross(V[faces[:, 1]] - V[faces[:, 0]], V[faces[:, 2]] - V[faces[:, 0]]), axis=1)
    faces = faces[area > 1e-10]
    return BowlMesh(V.astype(np.float32), uv.reshape(-1, 2).astype(np.float32), faces.astype(np.int32), R, C)


def mesh_lut(project: Project, engine, result, shape: BowlShape, mesh: BowlMesh, poses=None) -> StitchLUT:
    """Per-vertex remap tables of a bowl mesh (what an ECU renderer interpolates across the triangles)."""
    P = mesh.vertices.reshape(mesh.rows, mesh.cols, 3).astype(np.float64)
    return lut_for_points(project, engine, result, P, _body_mask(result, P), poses=poses)


def export_files(project: Project, engine, result, shape: BowlShape, texture: np.ndarray | None = None,
                 n_az=144, n_rad=56) -> list[tuple[str, bytes]]:
    """[(name, bytes)]: bowl.obj (+ bowl.mtl and bowl.png when a texture is given), bowl_lut.npz and bowl_lut.json.
    The LUT holds, per vertex and camera, the sensor pixel to sample (x, y, -1 = not seen) and a uint8 weight; the
    weights of all cameras sum to 255 where the bowl is seen."""
    mesh = bowl_mesh(result, shape, n_az, n_rad)
    lut = mesh_lut(project, engine, result, shape, mesh)
    files = []
    lines = ['# Vantage 3D surround-view bowl', f'# shape {json.dumps(shape.to_dict())}',
             '# frame: ISO 8855, rear-axle origin on the ground; x forward, y left, z up; metres']
    if texture is not None:
        lines.append('mtllib bowl.mtl')
    lines += [f'v {x:.5f} {y:.5f} {z:.5f}' for x, y, z in mesh.vertices]
    lines += [f'vt {u:.6f} {v:.6f}' for u, v in mesh.uv]          # the PNG is stored rim-up, so t = s
    if texture is not None:
        lines.append('usemtl bowl')
    lines += [f'f {a + 1}/{a + 1} {b + 1}/{b + 1} {c + 1}/{c + 1}' for a, b, c in mesh.faces]
    files.append(('bowl.obj', ('\n'.join(lines) + '\n').encode('utf-8')))
    if texture is not None:
        import cv2
        bgr = texture[::-1, :, [2, 1, 0, 3]] if texture.shape[2] == 4 else texture[::-1, :, ::-1]   # row 0 = rim
        ok, png = cv2.imencode('.png', np.ascontiguousarray(bgr))
        files.append(('bowl.mtl', b'newmtl bowl\nKa 1 1 1\nKd 1 1 1\nillum 0\nmap_Kd bowl.png\n'))
        files.append(('bowl.png', png.tobytes()))
    arrays = {'vertices': mesh.vertices, 'uv': mesh.uv, 'faces': mesh.faces}
    cams = []
    V = len(mesh.vertices)
    for uid, tag, safe in zip(lut.uids, lut.tags, file_tags(lut.uids, lut.tags, project.variant)):
        arrays[f'{safe}_map'] = np.stack([lut.map_x[uid].reshape(V), lut.map_y[uid].reshape(V)], -1).astype(np.float32)
        arrays[f'{safe}_weight'] = np.round(lut.weight[uid].reshape(V) * 255).astype(np.uint8)
        y, pi, r, x, yy, z = lut.poses[uid]
        cams.append({'tag': tag, 'map': f'{safe}_map', 'weight': f'{safe}_weight', 'sensor_width': lut.sensor[uid][0],
                     'sensor_height': lut.sensor[uid][1],
                     'pose': {'x': round(x, 4), 'y': round(yy, 4), 'z': round(z, 4), 'yaw': round(y, 3),
                              'pitch': round(pi, 3), 'roll': round(r, 3)}})
    buf = io.BytesIO()
    np.savez_compressed(buf, **arrays)
    files.append(('bowl_lut.npz', buf.getvalue()))
    cx, cy, hx, hy = footprint(result)
    meta = {'schema': 'vantage.bowl/1', 'project': project.name, 'variant': project.variant.name,
            'frame': 'ISO 8855, rear-axle origin on the ground; x forward, y left, z up; metres',
            'shape': shape.to_dict(), 'centre': [round(cx, 4), round(cy, 4)], 'vehicle_half_extent': [round(hx, 4), round(hy, 4)],
            'vertices': V, 'faces': int(len(mesh.faces)), 'grid': {'rows': mesh.rows, 'cols': mesh.cols},
            'map_format': 'float32 (V, 2): x, y in SENSOR pixels (pixel centre = i + 0.5), -1 where the camera does not see '
                          'the vertex. Scale by image_width / sensor_width for other resolutions.',
            'weight_format': 'uint8 (V,), 0..255; weights of all cameras sum to 255 where the bowl is seen',
            'render': 'per fragment: colour = sum_c weight_c * texture_c(map_c / sensor_size) / 255, map and weight '
                      'interpolated across each triangle from its vertices',
            'cameras': cams}
    files.append(('bowl_lut.json', json.dumps(meta, indent=1).encode('utf-8')))
    return files
