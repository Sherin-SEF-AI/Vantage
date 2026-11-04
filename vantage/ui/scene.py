"""Scene provider for the main 3D viewport (and the bowl view): what to draw and how tools behave."""
from __future__ import annotations

import math
import re

import numpy as np
from PyQt6.QtCore import QObject, pyqtSignal, Qt, QPointF, QRectF, QLineF
from PyQt6.QtGui import QColor, QPen, QPainter, QFont, QPolygonF, QBrush, QPainterPath
from PyQt6.QtWidgets import QMenu

from ..core.analysis import surface_frame, camera_frames
from ..core.lens import resolve
from ..core.mathutil import rot_zyx, BODY_TO_OPTICAL, wrap180, yaw_pitch_from_dir
from ..core.mesh import ROLE_GLASS, ROLE_WHEELS, ROLE_IGNORED
from . import guides, maps, ops, icons, overlays3d as O3
from .theme import THEME, mono, sans, STATUS
from .viewport3d import MeshDraw, GroundDraw, LineDraw, TriDraw
from .context import OVERLAYS
from .widgets import paint_chip

MARKER = 7


def _qc(hexs, a=1.0):
    c = QColor(hexs)
    c.setAlphaF(a)
    return c


def mesh_colors(mesh, shading):
    body = np.array(QColor(THEME.tok['body']).getRgbF()[:3])
    glass = np.array(QColor(THEME.tok['glass']).getRgbF()[:3])
    tyre = np.array([0.16, 0.16, 0.16])
    roles = [p.role for p in mesh.parts]
    names = [p.name.lower() for p in mesh.parts]
    base = np.tile(body, (mesh.n_tris, 1))
    alpha = np.ones(mesh.n_tris)
    for i, r in enumerate(roles):
        m = mesh.face_part == i
        if r == ROLE_GLASS or 'glass' in names[i]:
            base[m] = glass
        elif re.search(r'\brims?\b', names[i]) or 'alloy' in names[i]:     # ('trim' is not a rim)
            base[m] = (0.70, 0.71, 0.73)
        elif r == ROLE_WHEELS:
            base[m] = tyre
        elif 'tail lamp' in names[i] or 'brake' in names[i]:
            base[m] = (0.62, 0.05, 0.04)
        elif 'head lamp' in names[i] or 'headlight' in names[i]:
            base[m] = (0.30, 0.32, 0.36)                # a smoked lens over the reflectors
        elif 'black trim' in names[i]:
            base[m] = (0.07, 0.07, 0.07)
        elif r == ROLE_IGNORED:
            alpha[m] = 0.0
    if shading == 'xray':
        alpha = np.minimum(alpha, 0.28)
    elif shading == 'photo':
        alpha = np.minimum(alpha, 0.5)        # see the photo through the model to judge the match
    keep = alpha > 0
    return base, alpha, keep


def mesh_materials_for(mesh) -> np.ndarray:
    """Per-triangle material ids for the shader renderer (viewport3d MAT_*): its guess from the parts' roles and names
    (glass, tyres, lights, trim, rig metal), and faces in a mount zone named like glass are glass."""
    from . import viewport3d as V3
    fn = getattr(V3, 'mesh_materials', None)
    out = np.asarray(fn(mesh), np.int8).copy() if fn is not None else np.zeros(mesh.n_tris, np.int8)
    fz = getattr(mesh, 'face_zone', None)
    if fz is not None and len(fz) == mesh.n_tris:
        for zi, z in enumerate(getattr(mesh, 'zones', ()) or ()):
            if any(k in z.name.lower() for k in ('glass', 'window', 'windscreen', 'windshield')):
                out[np.asarray(fz) == zi] = getattr(V3, 'MAT_GLASS', 1)
    return out


class FrustumCache:
    def __init__(self):
        self.d = {}

    def get(self, project, cam, length=0.32):
        intr = resolve(project.module_for(cam), project.lens_for(cam))
        key = (cam.x, cam.y, cam.z, cam.yaw, cam.pitch, cam.roll, intr.width, intr.height, intr.f_mm, intr.projection,
               tuple(intr.kb), round(length, 3))
        if key in self.d:
            return self.d[key]
        R, W2O = camera_frames(cam)
        O2W = W2O.T
        n = 18
        t = np.linspace(0, 1, n, endpoint=False)
        w, h = intr.width, intr.height
        us = np.concatenate([t * w, np.full(n, w), (1 - t) * w, np.zeros(n)])
        vs = np.concatenate([np.zeros(n), t * h, np.full(n, h), (1 - t) * h])
        d, th = intr.unproject(us, vs)
        th = np.minimum(th, intr.max_theta)
        dw = d @ O2W.T
        pos = cam.pos()
        rim = pos + dw * length
        rim = np.vstack([rim, rim[:1]])
        corners = [0, n, 2 * n, 3 * n]
        rays = np.stack([np.repeat(pos[None], 4, 0), rim[corners]], 1)
        axis = np.stack([pos, pos + (O2W @ np.array([0, 0, 1.0])) * length * 1.1])
        if len(self.d) >= 200:                  # a long drag passes through many poses: start again, keep this one
            self.d.clear()
        out = self.d[key] = (rim, rays, axis)
        return out


SNAP_M, SNAP_DEG = 0.01, 5.0           # Ctrl while dragging a handle: 1 cm steps, 5° steps


def _pos(obj) -> np.ndarray:
    """A camera's, sensor's or rig part's position."""
    return np.array([obj.x, obj.y, obj.z], float)


class MainScene(QObject):
    changed = pyqtSignal()

    def __init__(self, ctx, mode='scene'):
        super().__init__()
        self.ctx = ctx
        self.mode = mode                    # 'scene' | 'bowl' | 'pinned'
        self._mesh_key = None
        self._mesh = None
        self._ground_key = None
        self._ground = None
        self._fr = FrustumCache()
        self.hover_info = None
        self.hover_world = None
        self.place_preview = None
        self.measure = []
        self.drag = None
        self.gizmo_hot = None               # the handle under the pointer: ('axis', k), ('plane',), ('ring', k)
        self.result_override = None         # pinned results (tolerance worst case)
        self.cameras_override = None
        self.show_tooltip = True
        self.photo_frame = None
        self.label_prefix = ''
        self.show_volume = False            # the blind volume regardless of the overlay (its own page)
        self._cube_hint = {}                # uid -> occlusion map from the camera's last analysed position
        self._lights = (None, [])
        self._bowl_tris = (None, None)
        self.bowl = None
        if mode == 'bowl':
            from .bowlview import BowlController
            self.bowl = BowlController(ctx)
            self.bowl.ready.connect(self._emit)
        st = ctx.store
        # bound methods (not lambdas) so Qt drops the connections when this scene is deleted
        st.changed.connect(self._emit)
        st.selection_changed.connect(self._emit)
        st.project_replaced.connect(self._reset)
        ctx.analysis.finished.connect(self._emit)
        ctx.analysis.stale_changed.connect(self._emit)
        ctx.overlays_changed.connect(self._emit)
        ctx.tool_changed.connect(self._on_tool)
        THEME.changed.connect(self._on_theme)

    def _emit(self, *_):
        self.changed.emit()

    def _on_tool(self, *_):
        self.measure = []
        self.changed.emit()

    def _on_theme(self, *_):
        self._reset()
        self.changed.emit()

    def _reset(self):
        self._mesh_key = None
        self._ground_key = None

    # ------------------------------------------------------------------ data
    @property
    def project(self):
        return self.ctx.store.project

    @property
    def result(self):
        return self.result_override if self.result_override is not None else self.ctx.analysis.result

    def cameras(self):
        if self.cameras_override is not None:
            return self.cameras_override
        return self.project.variant.cameras

    def background_color(self):
        return THEME.color('viewport')

    def background_image(self, vp):
        if self.photo_frame is None or not self.ctx.photo_shading:
            return None
        img = self.ctx.window.photo_image(self.photo_frame) if self.ctx.window else None
        if img is None or img.isNull():
            return None
        r = vp.rect()
        s = min(r.width() / img.width(), r.height() / img.height())
        w, h = img.width() * s, img.height() * s
        return img, QRectF((r.width() - w) / 2, (r.height() - h) / 2, w, h)

    def mesh(self):
        m = self.project.vehicle.mesh
        if m is None or m.n_tris == 0:
            return None
        sh = self.ctx.shading
        from .viewport3d import GLViewport
        use_full = m.n_tris <= 400_000
        key = (id(m), m.n_tris, sh, THEME.name, tuple(p.role for p in m.parts), self.ctx.window.soft_3d if self.ctx.window else False)
        if key != self._mesh_key:
            src = m
            if self.ctx.window is not None and self.ctx.window.soft_3d:
                vg = self.ctx.analysis.engine.vehicle_geom(self.project)
                src = vg.body if vg and m.n_tris > 12000 else m
            elif not use_full:
                vg = self.ctx.analysis.engine.vehicle_geom(self.project)
                src = vg.body
            base, alpha, keep = mesh_colors(src, sh)
            self._mesh = MeshDraw(src.tri[keep].astype(np.float32), src.normals[keep], base[keep], alpha[keep],
                                  version=id(src), wire=(sh == 'wire'))
            if 'materials' in getattr(MeshDraw, '__dataclass_fields__', {}):
                try:
                    self._mesh.materials = mesh_materials_for(src)[keep]
                except Exception as ex:          # a mesh without parts or zones: all paint
                    print('mesh materials:', ex)
            self._mesh_key = key
        return self._mesh

    def ground(self):
        res = self.result
        if res is None or (self.bowl is not None and self.bowl.draw is not None):
            return None
        if self._road() is not None:
            return None                     # on a road the heat map is drawn on its surface (_road_draw)
        heat = self.ctx.heatmap if self.mode == 'scene' else 'bev'
        fp = self.ctx.overlays.get('footprints', True)
        mode = heat if heat != 'none' else ('footprints' if fp else 'blank')
        if self.mode == 'bowl':
            mode = 'blank'
        iso = frozenset(self.ctx.isolate) if self.ctx.isolate else None
        sres = getattr(getattr(self.ctx, 'stitched', None), 'result', None) if mode == 'stitched' else None
        if mode == 'stitched' and sres is None:
            mode = 'blank'
        a = self.project.analysis
        thr = (a.min_density, a.target_density) if mode == 'density' else None
        key = (id(res), mode, THEME.name, iso, id(sres), thr)
        if key != self._ground_key and mode in ('bevfeat', 'stereo'):
            from . import research_overlays as RO
            try:
                if mode == 'bevfeat':
                    img, ext = RO.bev_image(self.research('bev'))
                else:
                    img, ext = RO.stereo_image(self.research('stereo'), maps.extent(res))
            except Exception as ex:                       # a research map failing must not take the view down
                print(f'{mode} map:', ex)
                img, ext = maps.make(res, 'blank', k=4), maps.extent(res)
            self._ground = GroundDraw(maps.to_qimage(img), ext, 0.002, nearest=True, version=hash(key))
            self._ground_key = key
            return self._ground
        if key != self._ground_key:
            if sres is not None:
                img, ext = maps.stitched(sres, k=4), maps.stitched_extent(sres)
            else:
                img, ext = maps.make(res, mode, k=4, isolate=iso, thresholds=thr), maps.extent(res)
            self._ground = GroundDraw(maps.to_qimage(img), ext, 0.002,       # density is drawn smooth, the rest crisp
                                      nearest=mode in ('redundancy', 'coverage', 'stitched'), version=hash(key))
            self._ground_key = key
        return self._ground

    def render_style(self, vp):
        from .render_gl import Style
        dark = THEME.name != 'light'
        bg = np.array(THEME.color('viewport').getRgbF()[:3])
        ink = np.array(THEME.color('ink').getRgbF()[:3])
        photo = self.ctx.shading == 'photo' or vp.cam.custom is not None
        lit = bool(self.ctx.overlays.get('shadows', True)) and not photo
        # the floor and the backdrop stay with lighting off: it only drops the shadow and sky-occlusion maps
        plane = self.mode == 'scene' and not photo and self._road() is None
        grid = self._shader_grid(vp)
        o = self.project.origin_offset()
        return Style(shadows=lit and self.mode == 'scene', ao=lit and self.mode == 'scene',
                     ground_plane=plane,
                     ground_rgb=tuple(bg * 0.9 + ink * 0.1), sky_rgb=(0.62, 0.66, 0.72) if dark else (0.86, 0.88, 0.91),
                     bounce_rgb=(0.24, 0.23, 0.22) if dark else (0.50, 0.48, 0.45),
                     grid=grid, axes=grid, origin=(float(o[0]), float(o[1])),
                     grid_minor=(*ink, 0.07 if dark else 0.10), grid_major=(*ink, 0.15 if dark else 0.20))

    def _shader_grid(self, vp) -> bool:
        """The floor grid is drawn by the shader renderer's ground (anti-aliased, fading with distance) instead of
        as lines: a perspective scene view on the plain floor, with a shader renderer."""
        return (getattr(vp, 'renderer', None) is not None and self.mode == 'scene' and bool(self.ctx.overlays.get('grid', True))
                and self.ctx.shading != 'photo' and vp.cam.custom is None and self._road() is None
                and self.bowl_draw(vp) is None)

    def camera_lights(self, vp):
        """Cameras for the camera-light overlay (what each camera sees, painted on the body and the ground)."""
        if self.mode != 'scene' or not self.ctx.overlays.get('camlight') or self.ctx.shading == 'photo':
            return []
        from ..core.visibility import cam_views
        cams = [c for c in self.cameras() if c.visible and (not self.ctx.isolate or c.uid in self.ctx.isolate)][:8]
        if not cams:
            return []
        eng = self.ctx.analysis.engine
        views = cam_views(self.project, eng, cams, [THEME.cam(c.slot).getRgbF()[:3] for c in cams], 'cached',
                          self._cube_hint)
        for v in views:
            if v.cube is not None:
                self._cube_hint[v.uid] = v.cube
        key = tuple((v.uid, v.pos.tobytes(), v.W2O.tobytes(), v.intr, id(v.cube), v.color) for v in views)
        if self._lights[0] != key:
            self._lights = (key, views)
        return self._lights[1]

    def bowl_draw(self, vp):
        return self.bowl.draw if self.bowl is not None and self.bowl.source != 'sectors' else None

    def _bowl_soft(self):
        """The textured bowl as flat-coloured triangles for the software renderer."""
        d = self.bowl.draw
        if self._bowl_tris[0] is not d:
            tex = d.texture
            H, W = tex.shape[:2]
            uv = d.uv[d.faces].mean(axis=1)
            tri = d.vertices[d.faces]
            # sample the texture at each face's centre (the seam column wraps)
            col = tex[np.clip((uv[:, 1] * H).astype(int), 0, H - 1), np.clip((uv[:, 0] % 1.0 * W).astype(int), 0, W - 1)]
            keep = col[:, 3] > 8
            rgba = col[keep].astype(np.float32) / 255.0
            self._bowl_tris = (d, TriDraw(tri[keep], rgba, depth=True))
        return self._bowl_tris[1]

    def leave_custom_view(self, vp):
        if self.photo_frame is not None and self.ctx.window is not None:
            self.ctx.window.leave_photo_view()

    def _ov3d(self):
        """The 3D overlays' background work (lidar sweeps, sky views), told what the shown overlays need."""
        c = O3.controller(self.ctx)
        if not getattr(self, '_ov3d_hooked', False):
            c.ready.connect(self._emit)
            self._ov3d_hooked = True
        bc = getattr(self.ctx, 'brackets_ctl', None)
        if bc is not None and getattr(self, '_br_hooked', None) is not bc:
            bc.changed.connect(self._emit)          # brackets generated or gone stale: redraw them
            self._br_hooked = bc
        ov = self.ctx.overlays
        c.want(bool(ov.get('lidarpts')), bool(ov.get('gnsssky')))
        return c

    def points(self, vp):
        """Point sprites: one simulated sweep per visible lidar (View ▸ Overlays ▸ Lidar points)."""
        if self.mode != 'scene' or not self.ctx.overlays.get('lidarpts') or not self.project.variant.sensors:
            return []
        return self._ov3d().point_draws()

    def tris_extra(self, vp):
        out, late = [], []                      # late: translucent draws, after every opaque one
        p = self.project
        gl = getattr(vp, 'renderer', None) is not None
        # camera modules at their real size, the selected ones outlined
        sel = set(self.ctx.store.sel_cams) if self.cameras_override is None else set()
        cams = [c for c in self.cameras() if c.visible and not (self.ctx.isolate and c.uid not in self.ctx.isolate)]
        out += O3.camera_draws(p, cams, sel, gl, smooth=not self.ctx.store.dragging)
        # calibration targets
        if self.ctx.overlays.get('targets') and p.targets and self.mode == 'scene':
            tri, col = [], []
            dark, light = [0.08, 0.08, 0.08, 1], [0.80, 0.79, 0.74, 1]
            for t in p.targets:
                w, h = t.size_m
                sq = t.square_mm / 1000
                c, s = math.cos(math.radians(t.yaw)), math.sin(math.radians(t.yaw))
                for i in range(t.cols):
                    for j in range(t.rows):
                        u0, v0 = -w / 2 + i * sq, -h / 2 + j * sq
                        pts = [(u0, v0), (u0 + sq, v0), (u0 + sq, v0 + sq), (u0, v0 + sq)]
                        P = np.array([[t.x + u * c - v * s, t.y + u * s + v * c, 0.004] for u, v in pts])
                        cc = dark if (i + j) % 2 == 0 else light
                        tri += [P[[0, 1, 2]], P[[0, 2, 3]]]
                        col += [cc, cc]
            out.append(TriDraw(np.array(tri), np.array(col)))
        late += self.research_tris()                 # the lidar timing wedges, see-through
        vc = getattr(self.ctx, 'volume', None)
        if vc is not None and self.mode == 'scene' and (self.show_volume or self.ctx.overlays.get('volume')):
            td = vc.tri_draw(gl=getattr(vp, 'renderer', None) is not None)
            if td is not None:
                out.append(td)
        if self.mode == 'bowl' and self.result is not None:
            if self.bowl_draw(vp) is None:
                out.append(self._bowl())
            elif not getattr(vp, 'gl_bowl', False):
                out.append(self._bowl_soft())
        if self.mode in ('scene', 'bowl'):                      # the rig rides on the car in the bowl view too
            from .rigview import scene_draws as rig_draws
            from .sensorview import scene_draws as sensor_draws
            out += rig_draws(self.ctx, gl)
            out += sensor_draws(self.ctx, gl)
        road = self._road()
        if road is not None:
            out.append(self._road_draw(vp, road))
        if self.mode == 'scene':
            ov = self.ctx.overlays
            if ov.get('brackets', True):
                out += O3.bracket_draws(self.ctx, gl)
            if ov.get('sensors', True) and p.variant.sensors:
                from .sensorview import radar_draws
                late += radar_draws(self.ctx, gl, vp=vp)
            if ov.get('gnsssky') and p.variant.sensors:
                late += self._ov3d().dome_draws(gl)[0]
            if ov.get('critical'):                      # View ▸ Overlays ▸ Critical targets
                from .workspaces import critical_ws as CW
                if not isinstance(self, CW.CriticalScene):   # the page's own 3D view draws them always
                    late += CW.scene_tris(self.ctx, gl)
        out += guides.scene_tris(self, vp)
        return out + late

    # ------------------------------------------------------------------ the road (core.terrain)
    def _road(self):
        """The project's road when it has one switched on (main 3D view only), else None: flat ground."""
        if self.mode != 'scene':
            return None
        from ..core import terrain as TR
        try:
            return TR.active(self.project)
        except Exception as ex:                    # a broken road must not cost the view
            print('terrain:', ex)
            return None

    @staticmethod
    def _drape(road, pts, lift=0.0, base=None):
        """World points (…, 3) on flat ground (z above it) moved onto the road (z above it, or above `base`) and into
        the body frame of the car standing on it; unchanged without a road."""
        if road is None:
            return pts
        P = np.array(pts, float)
        flat = P.reshape(-1, 3)
        flat[:, 2] += (road.height(flat[:, 0], flat[:, 1]) if base is None else base) + lift
        pose = road.pose()
        if pose is not None:
            flat[:] = (flat - pose[1]) @ pose[0]
        return P

    def _road_edges(self, road):
        """Each primitive's edges as lines on the road's surface (a kerb's face solid, the rest dashed)."""
        ref = road.ref
        x0, x1 = ref['rear_x'] - 15.0, ref['front_x'] + 25.0
        out = []
        for f in road.features:
            for k, b in enumerate(f.breaks()):
                if f.kind == 'kerb' and k > 0:
                    continue
                c = f.origin + f.sign * b
                t = np.linspace(x0, x1, 161) if f.axis == 1 else np.linspace(-8.0, 8.0, 65)
                pts = np.c_[t, np.full_like(t, c), np.zeros_like(t)] if f.axis == 1 else \
                    np.c_[np.full_like(t, c), t, np.zeros_like(t)]
                face = f.kind == 'kerb'
                if face:                                   # on top of the face, where it shows
                    pts[:, f.axis] += f.sign * 0.012
                out.append(LineDraw(self._drape(road, pts, 0.006), THEME.color('ink', 0.85 if face else 0.6),
                                    1.6 if face else 1.0, dash=not face, dash_len=0.12))
        return out

    def _road_draw(self, vp, road):
        """The road's surface as triangles (the body frame of the car standing on it), lit, with the coverage heat
        map on it: fine over the analysis grid, coarse around it, with the primitives' edges as break lines."""
        gl = getattr(vp, 'renderer', None) is not None
        res = self.result
        heat = self.ctx.heatmap
        fp = self.ctx.overlays.get('footprints', True)
        mode = heat if heat != 'none' else ('footprints' if fp else 'blank')
        iso = frozenset(self.ctx.isolate) if self.ctx.isolate else None
        sres = getattr(getattr(self.ctx, 'stitched', None), 'result', None) if mode == 'stitched' else None
        a = self.project.analysis
        thr = (a.min_density, a.target_density) if mode == 'density' else None
        key = (road.key, mode, THEME.name, iso, thr, gl)
        cache = getattr(self, '_road_cache', (None, None, None, None))
        if cache[0] == key and cache[2] is res and cache[3] is sres:     # the objects, not their ids (reused)
            return cache[1]
        from ..core.terrain import FACE_M
        ref = road.ref
        if res is not None:
            fx0, fy0, fx1, fy1 = maps.extent(res)
        else:
            fx0, fy0, fx1, fy1 = ref['rear_x'] - 2.0, -3.0, ref['front_x'] + 2.0, 3.0
        ox0, ox1, oy0, oy1 = min(ref['rear_x'] - 15.0, fx0), max(ref['front_x'] + 25.0, fx1), min(-8.0, fy0), \
            max(8.0, fy1)
        fine, coarse = (0.1, 0.5) if gl else (0.3, 1.5)
        bx, by = [], []
        for f in road.features:
            for b in f.breaks():
                c = f.origin + f.sign * b
                (bx if f.axis == 0 else by).extend([c, c + f.sign * FACE_M] if f.kind == 'kerb' else [c])
            if f.kind == 'bump':
                (bx if f.axis == 0 else by).extend(f.origin + f.sign * np.linspace(0, f.d['length'], 9))

        def axis(o0, f0, f1, o1, extra):
            v = np.r_[np.arange(o0, f0, coarse), np.arange(f0, f1, fine), np.arange(f1, o1 + 1e-9, coarse), extra]
            v = np.unique(np.round(v[(v >= o0) & (v <= o1)], 4))
            return v
        xs, ys = axis(ox0, fx0, fx1, ox1, bx), axis(oy0, fy0, fy1, oy1, by)
        X, Y = np.meshgrid(xs, ys)
        Z = road.height(X, Y)
        gx, gy = road.gradient(X, Y)
        N = np.stack([-gx, -gy, np.ones_like(gx)], -1)
        N /= np.linalg.norm(N, axis=-1, keepdims=True)
        V = np.stack([X, Y, Z], -1)
        q = [(slice(None, -1), slice(None, -1)), (slice(None, -1), slice(1, None)), (slice(1, None), slice(1, None)),
             (slice(1, None), slice(None, -1))]
        c00, c01, c11, c10 = (V[a_, b_].reshape(-1, 3) for a_, b_ in q)
        n00, n01, n11, n10 = (N[a_, b_].reshape(-1, 3) for a_, b_ in q)
        tri = np.concatenate([np.stack([c00, c01, c11], 1), np.stack([c00, c11, c10], 1)])
        nrm = np.concatenate([np.stack([n00, n01, n11], 1), np.stack([n00, n11, n10], 1)])
        dark = THEME.name != 'light'
        base = np.array([0.30, 0.30, 0.29] if dark else [0.70, 0.70, 0.68])
        rgb = np.tile(base, (len(tri), 1))
        if res is not None:
            if sres is not None:
                img, ext = maps.stitched(sres, k=2), maps.stitched_extent(sres)
            else:
                img, ext = maps.make(res, mode, k=2, isolate=iso, thresholds=thr), maps.extent(res)
            if img is not None:
                h_, w_ = img.shape[:2]
                cen = tri.mean(axis=1)
                u = ((cen[:, 0] - ext[0]) / (ext[2] - ext[0]) * w_).astype(int)
                v = ((ext[3] - cen[:, 1]) / (ext[3] - ext[1]) * h_).astype(int)
                inside = (u >= 0) & (u < w_) & (v >= 0) & (v < h_)
                px = np.zeros((len(tri), 4))
                px[inside] = img[v[inside], u[inside]].astype(float) / 255.0
                al = px[:, 3:4]
                rgb = rgb * (1 - al) + px[:, :3] * al
        if not gl:                                     # the software renderer does not light triangles: shade here
            light = np.array([0.35, 0.25, 0.9]) / np.linalg.norm([0.35, 0.25, 0.9])
            rgb = rgb * (0.55 + 0.45 * np.clip(nrm.mean(axis=1) @ light, 0, 1))[:, None]
        pose = road.pose()
        if pose is not None:
            tri = (tri - pose[1]) @ pose[0]
            nrm = nrm @ pose[0]
        td = TriDraw(tri.astype(np.float32), np.c_[rgb, np.ones(len(tri))].astype(np.float32), depth=True,
                     normals=nrm.astype(np.float32) if gl else None)
        self._road_cache = (key, td, res, sres)
        return td

    def _bowl(self):
        res = self.result
        cx, cy = res.centre
        r0 = self.project.roi.distance + 1.0
        lo, hi = res.grid.x0, res.grid.x0 + res.grid.nx * res.grid.cell
        ry = (res.grid.ny * res.grid.cell) / 2
        rx = (hi - lo) / 2
        na, nr = 72, 14
        ang = np.linspace(-math.pi, math.pi, na + 1)
        rad = np.linspace(0.0, 1.0, nr + 1)
        # super-ellipse footprint matching the ROI box, then rising
        def pt(a, t):
            ex, ey = rx * t * 1.6, ry * t * 1.6
            ca, sa = math.cos(a), math.sin(a)
            k = 4.0
            r = 1 / ((abs(ca) / max(ex, 1e-6)) ** k + (abs(sa) / max(ey, 1e-6)) ** k) ** (1 / k) if t > 0 else 0
            x, y = cx + r * ca, cy + r * sa
            flat = max(0.0, t - 0.62) / 0.38 * 0.62 * max(rx, ry)
            return np.array([x, y, 0.42 * flat ** 2 / max(rx, ry) * 2])
        P = np.array([[pt(a, t) for a in ang] for t in rad])
        tri, col = [], []
        sec = res.sector
        g = res.grid
        for i in range(nr):
            for j in range(na):
                q = [P[i, j], P[i + 1, j], P[i + 1, j + 1], P[i, j + 1]]
                mid = np.mean(q, axis=0)
                a = math.atan2(mid[1] - cy, mid[0] - cx)
                # sector owner by azimuth from the seams
                owner = self._sector_at(a)
                if owner is None:
                    rgb = [0.3, 0.3, 0.3]
                else:
                    c = QColor(THEME.cam_hex(owner))
                    rgb = [c.redF(), c.greenF(), c.blueF()]
                shade = 0.55 + 0.45 * (i / nr)
                cc = [rgb[0] * shade, rgb[1] * shade, rgb[2] * shade, 0.42]
                tri += [np.array([q[0], q[1], q[2]]), np.array([q[0], q[2], q[3]])]
                col += [cc, cc]
        self._bowl_grid = P
        return TriDraw(np.array(tri), np.array(col), depth=True)

    def _sector_at(self, a):
        res = self.result
        if not res or not res.seams:
            return None
        cams = {c.uid: c for c in self.cameras()}
        seams = res.seams
        n = len(seams)
        angs = [math.radians(s.angle) for s in seams]
        for k in range(n):
            s0, s1 = angs[k], angs[(k + 1) % n]
            span = (s1 - s0) % (2 * math.pi)
            if (a - s0) % (2 * math.pi) < span:
                c = cams.get(seams[(k + 1) % n].a) or cams.get(seams[k].b)
                return c.slot if c else None
        return None

    # ------------------------------------------------------------------ lines
    RANGE_RINGS = (1.0, 2.0, 5.0, 10.0)

    def _range_rings(self):
        """[(distance m, (K, 3) ground points)]: rounded rectangles RANGE_RINGS metres out from the body's footprint
        (its plan-view bounding box, mirrors included), for reading near-field coverage off the ground."""
        m = self.project.vehicle.mesh
        key = (id(m), m.n_tris)
        if getattr(self, '_rings_key', None) == key:
            return self._rings_cache
        P = m.vertices
        x0, y0 = P[:, 0].min(), P[:, 1].min()
        x1, y1 = P[:, 0].max(), P[:, 1].max()
        out = []
        for d in self.RANGE_RINGS:
            pts = []
            for cx, cy, a0 in ((x1, y1, 0.0), (x0, y1, 90.0), (x0, y0, 180.0), (x1, y0, 270.0)):
                for a in np.radians(np.linspace(a0, a0 + 90.0, 13)):
                    pts.append((cx + d * math.cos(a), cy + d * math.sin(a), 0.006))
            pts.append(pts[0])
            out.append((d, np.array(pts)))
        self._rings_key, self._rings_cache = key, out
        self._ring_labels = [(x1 + d, 0.0, 0.006, f'{d:g} m') for d in self.RANGE_RINGS] + \
            [(0.5 * (x0 + x1), y1 + d, 0.006, f'{d:g} m') for d in self.RANGE_RINGS]
        return out

    def lines(self, vp):
        out = []
        p = self.project
        ov = self.ctx.overlays
        res = self.result
        textured = self.bowl_draw(vp) is not None
        road = self._road()
        if ov.get('grid', True) and not textured and road is None and not self._shader_grid(vp):
            out += self._grid()
        if road is not None:
            out += self._road_edges(road)
        if res is not None:
            if ov.get('roi', True) and not textured:
                for b in res.grid.boundary:
                    pts = self._drape(road, np.c_[b, np.full(len(b), 0.004)])
                    out.append(LineDraw(pts, THEME.color('ink', 0.9), 1.2, dash=True, dash_len=0.09))
            if ov.get('footprints', True) and self.mode == 'scene':
                for c in res.cams:
                    if self.ctx.isolate and c.uid not in self.ctx.isolate:
                        continue
                    if len(c.segments):
                        sel = c.uid in self.ctx.store.sel_cams
                        segs = self._drape(road, np.concatenate([c.segments, np.full(c.segments.shape[:2] + (1,), 0.005)],
                                                                axis=2))
                        out.append(LineDraw(segs, THEME.cam(c.slot), 1.6 if sel else 1.1, segments=True))
            if ov.get('seams', True) and res.seams:
                cx, cy = res.centre
                g = res.grid
                rmax = max(g.nx, g.ny) * g.cell
                for s in res.seams:
                    a = math.radians(s.angle)
                    rr = np.linspace(0, rmax, 120)
                    pts = np.c_[cx + rr * math.cos(a), cy + rr * math.sin(a), np.full(len(rr), 0.006)]
                    ii = ((pts[:, 0] - g.x0) / g.cell).astype(int)
                    jj = ((pts[:, 1] - g.y0) / g.cell).astype(int)
                    ok = (ii >= 0) & (ii < g.nx) & (jj >= 0) & (jj < g.ny)
                    ok[ok] &= g.roi[jj[ok], ii[ok]]
                    if ok.sum() > 1:
                        idx = np.nonzero(ok)[0]
                        out.append(LineDraw(self._drape(road, pts[idx[0]:idx[-1] + 1]), THEME.color('ink', 0.75), 1.0,
                                            dash=True, dash_len=0.06))
            if ov.get('occlusion') and self.mode == 'scene':
                out += self._occlusion_rays()
        if ov.get('frustums', True):
            sel = set(self.ctx.store.sel_cams)
            for c in self.cameras():
                if not c.visible or (self.ctx.isolate and c.uid not in self.ctx.isolate):
                    continue
                rim, rays, axis = self._fr.get(p, c, 0.32 if c.uid not in sel else 0.42)
                col = THEME.cam(c.slot)
                w = 2.0 if c.uid in sel else 1.0
                out.append(LineDraw(rim, col, w, depth=True))
                out.append(LineDraw(rays, col, w, segments=True, depth=True))
        for ob in p.obstacles:
            if self.mode != 'scene':
                break
            a = np.linspace(0, 2 * math.pi, 17)
            ring = np.c_[ob.x + ob.radius * np.cos(a), ob.y + ob.radius * np.sin(a), np.zeros(17)]
            base = float(road.height(ob.x, ob.y)) if road is not None else 0.0      # it stands on the road
            out.append(LineDraw(self._drape(road, ring, 0.0, base), THEME.color('ink2'), 1.0))
            out.append(LineDraw(self._drape(road, np.array([[ob.x, ob.y, 0], [ob.x, ob.y, ob.height]]), 0.0, base),
                                THEME.color('ink2'), 1.5))
            out.append(LineDraw(self._drape(road, ring + np.array([0, 0, ob.height]), 0.0, base), THEME.color('ink2'),
                                1.0))
        if ov.get('dimensions') and p.vehicle.mesh is not None and self.mode == 'scene':
            out += self._dimensions()
        if ov.get('rings') and p.vehicle.mesh is not None and self.mode == 'scene' and not textured:
            for d, pts in self._range_rings():
                out.append(LineDraw(self._drape(road, pts), THEME.color('ink2', 0.75 if d < 5 else 0.5),
                                    1.2 if d in (1.0, 5.0) else 0.9, dash=d not in (1.0, 5.0), dash_len=0.12))
        if self.measure:
            pts = np.array(self.measure)
            if len(pts) >= 2:
                out.append(LineDraw(pts[:2], THEME.color('ink'), 1.5, depth=False))
        if self.mode == 'bowl' and hasattr(self, '_bowl_grid') and not textured:
            P = self._bowl_grid
            for i in range(0, P.shape[0], 2):
                out.append(LineDraw(P[i], THEME.color('ink', 0.18), 1.0))
            for j in range(0, P.shape[1], 6):
                out.append(LineDraw(P[:, j], THEME.color('ink', 0.18), 1.0))
        if self.mode == 'scene' and self.ctx.overlays.get('sensors', True) and self.project.variant.sensors:
            from .sensorview import scene_lines
            out += scene_lines(self.ctx)
        if self.mode == 'scene' and ov.get('gnsssky') and self.project.variant.sensors:
            out += self._ov3d().dome_draws(getattr(vp, 'renderer', None) is not None)[1]
        if self.mode == 'scene' and ov.get('critical'):
            from .workspaces import critical_ws as CW
            if not isinstance(self, CW.CriticalScene):
                out += CW.scene_lines(self.ctx)
        if self.mode == 'scene' and ov.get('harness'):  # View ▸ Overlays ▸ Cable harness: the routes as lines
            from .harnessview import scene_draws as harness_draws
            out += harness_draws(self.ctx, getattr(vp, 'renderer', None) is not None)
        if self.mode == 'scene' and ov.get('wipers'):  # View ▸ Overlays ▸ Wiper sweep (Analysis ▸ Materials)
            from .workspaces.materials_ws import scene_lines as wiper_lines
            out += wiper_lines(self.ctx)
        if self.mode == 'scene' and ov.get('clocktree'):  # View ▸ Overlays ▸ Timestamp error (Analysis ▸ Clock tree)
            out += self._clock_tree_lines()
        out += self.research_lines()
        out += guides.scene_lines(self, vp)
        return out

    def _grid(self):
        res = self.result
        if res is not None:
            x0, y0, x1, y1 = maps.extent(res)
            cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
        else:
            cx, cy = 1.2, 0.0
        R = 14.0
        lines_a, lines_b = [], []
        for k in np.arange(-R, R + 1e-6, 0.5):
            major = abs(k - round(k)) < 1e-6
            tgt = lines_b if major else lines_a
            x = round(cx) + k
            tgt.append([[x, cy - R, 0], [x, cy + R, 0]])
            y = round(cy) + k
            tgt.append([[cx - R, y, 0], [cx + R, y, 0]])
        return [LineDraw(np.array(lines_a), THEME.color('grid1'), 1.0, segments=True),
                LineDraw(np.array(lines_b), THEME.color('grid2'), 1.0, segments=True)]

    def _occlusion_rays(self):
        c = self.ctx.store.selected
        res = self.result
        if c is None or res is None:
            return []
        i = res.cam_index(c.uid)
        if i < 0:
            return []
        ce = res.cams[i]
        g = res.grid
        # cells inside the image but blocked by the body
        fld = ce.field
        blocked = (fld < 0) & g.roi & ~ce.vis
        jj, ii = np.nonzero(blocked)
        if len(jj) == 0:
            return []
        sel = np.linspace(0, len(jj) - 1, min(40, len(jj))).astype(int)
        road = self._road()
        segs = []
        for k in sel:
            P = self._drape(road, np.array([[g.xs[ii[k]], g.ys[jj[k]], 0.0]]), 0.0)[0]
            segs.append([c.pos(), P])
        return [LineDraw(np.array(segs), _qc(STATUS['critical'], 0.55), 1.0, segments=True)]

    CLOCK_SCALE = 50.0         # the timestamp-error overlay magnifies millimetres to see them

    def _clock_tree_lines(self):
        """The reference lidar's points moved by each camera's timestamp error, × CLOCK_SCALE, in the camera's colour
        (core.clock_tree.overlay_segments): from the Clock tree page's last result when there is one, else the study's
        slip setting. Recomputed only when the project or that result changes."""
        from ..core import clock_tree as CT
        w = self.ctx.window
        pg = getattr(w, '_page_widgets', {}).get('clock_tree') if w is not None else None
        res = getattr(pg, 'res', None)
        key = (self.ctx.store.revision, id(self.project), id(res))
        cache = getattr(self, '_ct_cache', None)
        if cache is None or cache[0] != key:
            try:
                segs = CT.overlay_segments(self.project, res, scale=self.CLOCK_SCALE)
            except Exception:                  # a half-edited rig: draw nothing rather than break the view
                segs = []
            out = []
            by_col = {}
            for a, b, rgba in segs:
                by_col.setdefault(rgba, []).append((a, b))
            for rgba, ab in by_col.items():
                col = QColor.fromRgbF(*rgba)
                out.append(LineDraw(np.array(ab, float), col, 1.2, segments=True))
                ends = np.array([b for _, b in ab], float)
                tick = np.stack([ends - [0, 0, 0.02], ends + [0, 0, 0.02]], axis=1)   # a short mark at the slipped end
                out.append(LineDraw(tick, col, 2.4, segments=True))
            self._ct_cache = cache = (key, out)
        return cache[1]

    def _dimensions(self):
        m = self.project.vehicle.measured()
        lo, hi = self.project.vehicle.mesh.bbox
        y = lo[1] - 0.35
        out = [LineDraw(np.array([[lo[0], y, 0.01], [hi[0], y, 0.01]]), THEME.color('ink2'), 1.0, depth=False),
               LineDraw(np.array([[hi[0] + 0.35, lo[1], 0.01], [hi[0] + 0.35, hi[1], 0.01]]), THEME.color('ink2'), 1.0, depth=False),
               LineDraw(np.array([[lo[0] - 0.25, 0, 0], [lo[0] - 0.25, 0, hi[2]]]), THEME.color('ink2'), 1.0, depth=False)]
        self._dim_labels = [((lo[0] + hi[0]) / 2, y - 0.12, 0.0, f'L {m.get("Length", 0):.3f} m'),
                            (hi[0] + 0.5, 0, 0.0, f'W {m.get("Width", 0):.3f} m'),
                            (lo[0] - 0.35, 0, hi[2] / 2, f'H {m.get("Height", 0):.3f} m')]
        return out

    # ------------------------------------------------------------------ overlay
    def hud_label(self, vp):
        if self.label_prefix:
            return self.label_prefix
        if self.mode == 'bowl':
            if self.bowl is not None and self.bowl.source != 'sectors':
                sh = self.bowl.shape
                txt = (f'3D surround view · floor {sh.floor_m:.1f} m past the body, wall {sh.height_m:.1f} m high over '
                       f'{sh.reach_m:.1f} m')
                return txt + (f' · {self.bowl.status}' if self.bowl.status and self.bowl.draw is None else '')
            r = self.project.roi.distance
            return f'Bowl view · camera sectors · flat to {r:.1f} m'
        names = {'solid': 'Solid', 'wire': 'Wire', 'xray': 'X-ray', 'photo': 'Photo-matched'}
        view = 'Perspective' if not vp.cam.ortho else ('Top' if vp.cam.el > 80 else 'Side' if abs(vp.cam.az - 90) < 1 else 'Front' if abs(vp.cam.az) < 1 else 'Rear' if abs(abs(vp.cam.az) - 180) < 1 else 'Ortho')
        if vp.cam.custom is not None:
            view = 'Perspective · matched to photo'
        heat = {'none': '', 'coverage': ' · coverage', 'density': ' · density', 'redundancy': ' · redundancy',
                'stitched': ' · objects in the stitched view'}.get(self.ctx.heatmap, '')
        vc = getattr(self.ctx, 'volume', None)
        if vc is not None and vc.result is not None and (self.show_volume or self.ctx.overlays.get('volume')):
            heat += f' · blind space up to {vc.result.z[vc.cut_level() - 1] + vc.result.dz / 2:.2f} m'
        return f'{view} · {names.get(self.ctx.shading, "Solid")}{heat}'

    # ------------------------------------------------------------------ view bar
    SHADING_NAMES = {'solid': 'Solid', 'wire': 'Wire', 'xray': 'X-ray', 'photo': 'Photo-matched'}
    HEAT_NAMES = {'coverage': 'Coverage', 'density': 'Density', 'redundancy': 'Redundancy', 'stitched': 'Objects',
                  'bevfeat': 'BEV features', 'stereo': 'Stereo σ'}
    from .research_overlays import RESEARCH_HEAT

    def view_bar(self, vp):
        """The view bar's buttons, [(key, text, tooltip, icon)]: the main window's 3D scene views only."""
        w = self.ctx.window
        if self.mode != 'scene' or self.label_prefix or w is None or not hasattr(w, 'actions_by_id'):
            return []
        view = self.hud_label(vp).split(' · ')[0]
        if vp.cam.custom is not None:
            view = 'Matched to photo'
        n_on = sum(1 for k, *_ in OVERLAYS if self.ctx.overlays.get(k))
        heat = self.HEAT_NAMES.get(self.ctx.heatmap)
        return [('view', view, 'View: perspective, top, front, side, rear or through a camera; frame; render', 'cube'),
                ('shading', self.SHADING_NAMES.get(self.ctx.shading, 'Solid'), 'Shading: solid, wire, X-ray or matched '
                 'to a reference photo', 'shading'),
                ('overlays', f'Overlays {n_on}', 'Show or hide what is drawn over the scene (several at once)', 'layers'),
                ('heat', f'Heat map · {heat}' if heat else 'Heat map', 'Colour the ground by coverage, density, '
                 'redundancy or the objects in the stitched view', 'density')]

    def view_bar_note(self, vp):
        """What follows the view bar: the parts of the plain label the buttons do not show (the blind space's cut)."""
        vc = getattr(self.ctx, 'volume', None)
        if vc is not None and vc.result is not None and (self.show_volume or self.ctx.overlays.get('volume')):
            return f'blind space up to {vc.result.z[vc.cut_level() - 1] + vc.result.dz / 2:.2f} m'
        return ''

    def view_bar_menu(self, vp, key):
        """The menu a view bar button opens: the main window's own actions, so it stays in step with the View menu."""
        from .navigation import StayOpenMenu
        w = self.ctx.window
        a = getattr(w, 'actions_by_id', None)
        if not a:
            return None
        ids = {'view': ['view_persp', 'view_top', 'view_front', 'view_side', 'view_rear', 'view_through', '-',
                        'frame_all', 'frame_sel', '-', 'render_image', 'turntable'],
               'shading': ['shade_solid', 'shade_wire', 'shade_xray', 'shade_photo'],
               'overlays': [f'ov_{k}' for k, *_ in OVERLAYS],
               'heat': [f'heat_{k}' for k in ('none', *self.HEAT_NAMES)]}.get(key)
        if ids is None:
            return None
        m = StayOpenMenu(vp) if key == 'overlays' else QMenu(vp)
        m.setAttribute(Qt.WidgetAttribute.WA_DeleteOnClose, True)
        for i in ids:
            if i == '-':
                m.addSeparator()
            elif i in a:
                m.addAction(a[i])
        return m





    # ------------------------------------------------------------------ research maps, probes, motion
    def research(self, what):
        """The current analysis's research map (core.research_maps / core.motion_footprint), cached per result and
        motion: 'bev', 'stereo', 'fusion', 'warp', 'sweep', 'wedges'."""
        from ..core import research_maps as RM
        from ..core import motion_footprint as MF
        res = self.result
        mo = self.ctx.motion
        key = (what, id(res), mo['speed'], mo['yaw'] if what in ('warp', 'sweep') else 0)
        cache = self.__dict__.setdefault('_research', {})
        if key in cache:
            return cache[key]
        eng = self.ctx.analysis.engine
        p = self.project
        if what == 'bev':
            v = RM.bev_feature_map(p, eng, result=res)
        elif what == 'stereo':
            v = RM.stereo_depth_sigma(p, eng, result=res)
        elif what == 'fusion':
            v = RM.lidar_camera_association(p, eng, result=res)
        elif what == 'warp':
            v = MF.warp_footprints(p, eng, mo['speed'], mo['yaw'], result=res)
        elif what == 'sweep':
            v = MF.lidar_sweep(p, eng, mo['speed'], mo['yaw'])
        elif what == 'wedges':
            v = MF.temporal_wedges(p, eng)
        else:
            raise KeyError(what)
        for k in [k for k in cache if k[1] != id(res)]:
            cache.pop(k)                                  # results move on; keep only the current one's maps
        cache[key] = v
        return v

    def add_probe(self, xy, kind=None):
        """Pin a probe (the Probe tool's object) at a ground point: its readout from core.research_maps."""
        from ..core import research_maps as RM
        kind = kind or self.ctx.probe_kind
        pr = RM.probe_readout(self.project, self.ctx.analysis.engine, (float(xy[0]), float(xy[1])), kind=kind,
                              result=self.result)
        self.probes = (getattr(self, 'probes', []) + [{'point': (float(xy[0]), float(xy[1]), 0.0), 'kind': kind,
                                                       'readout': pr, 'res': id(self.result)}])[-4:]
        self.changed.emit()
        return pr

    def clear_probes(self):
        self.probes = []
        self.changed.emit()

    def _probes_current(self):
        """The probes, their readouts refreshed when the analysis has moved on (a camera was moved)."""
        from ..core import research_maps as RM
        out = []
        for pr in getattr(self, 'probes', []):
            if pr['res'] != id(self.result):
                try:
                    pr = dict(pr, readout=RM.probe_readout(self.project, self.ctx.analysis.engine, pr['point'][:2],
                                                           kind=pr['kind'], result=self.result), res=id(self.result))
                except Exception as ex:
                    print('probe:', ex)
            out.append(pr)
        self.probes = out
        return out

    def research_lines(self):
        from . import research_overlays as RO
        out = []
        if self.mode != 'scene' or self.result is None:
            return out
        try:
            if self.ctx.overlays.get('fusion') and self.project.variant.sensors:
                out += RO.fusion_lines(self.research('fusion'), LineDraw)
            probes = self._probes_current()
            if probes:
                tags = {c.tag: c for c in self.cameras()}
                for pr in probes:
                    out += RO.probe_lines(pr, tags, LineDraw)
            mo = self.ctx.motion
            if abs(mo['speed']) > 1e-6 or abs(mo['yaw']) > 1e-6:
                uids = {c.uid: c for c in self.cameras()}
                sweep = self.research('sweep') if self.project.variant.sensors else []
                out += RO.motion_lines(self.research('warp'), sweep, uids, LineDraw)
        except Exception as ex:
            print('research overlays:', ex)
        return out

    def research_tris(self):
        from . import research_overlays as RO
        if self.mode != 'scene' or not self.ctx.overlays.get('wedges') or not self.project.variant.sensors:
            return []
        try:
            uids = {c.uid: c for c in self.cameras()}
            out = []
            for tris, col in RO.wedge_tris(self.research('wedges'), uids, TriDraw):
                rgba = np.tile([col.redF(), col.greenF(), col.blueF(), 0.13], (len(tris), 1)).astype(np.float32)
                out.append(TriDraw(tris.astype(np.float32), rgba, depth=True))
            return out
        except Exception as ex:
            print('timing wedges:', ex)
            return []

    def _paint_probes(self, p, vp):
        from . import research_overlays as RO
        for pr in getattr(self, 'probes', []):
            q, _, ok = vp.project([[pr['point'][0], pr['point'][1], 1.0]])
            if ok[0]:
                rows = RO.probe_card_lines(pr['readout'])
                p.setFont(mono(10))
                wcard = max(p.fontMetrics().horizontalAdvance(r) for r in rows) + 28
                x = q[0][0] + 14 if q[0][0] + 14 + wcard < vp.width() - 8 else q[0][0] - 14 - wcard
                y = min(max(q[0][1] - 10, 40.0), vp.height() - 16.0 * len(rows) - 40)
                self._label_box(p, QPointF(max(x, 8.0), y), rows)

    # ------------------------------------------------------------------ Alt-drag seams and guideline markers
    def _vanish_m2(self):
        sres = getattr(getattr(self.ctx, 'stitched', None), 'result', None)
        m = getattr(sres, 'metrics', None)
        return float(m['vanish_m2']) if isinstance(m, dict) and 'vanish_m2' in m else None

    def _alt_pick(self, vp, pos):
        """With Alt held: the stitch seam (radial line from the stitch centre) or the guideline distance marker
        under the pointer, as a drag dict; None when neither is within reach."""
        g = vp.ground_at(pos.x(), pos.y())
        res = self.result
        if g is None or res is None:
            return None
        if self.ctx.overlays.get('seams', True) and res.seams:
            cx, cy = res.centre
            r = math.hypot(g[0] - cx, g[1] - cy)
            ang = math.degrees(math.atan2(g[1] - cy, g[0] - cx))
            best = None
            for s in res.seams:
                off = abs(wrap180(ang - s.angle)) * math.pi / 180 * r        # metres off the seam at that range
                if r > 0.3 and off < 0.12 and (best is None or off < best[0]):
                    best = (off, s)
            if best is not None:
                s = best[1]
                return {'kind': 'seam', 'uid': f'{s.a}|{s.b}', 'a0': s.angle, 'mouse': pos, 'at': pos,
                        'readout': '', 'vanish0': self._vanish_m2(), 'tags': self._seam_tags(s)}
        if self.ctx.overlays.get('guidelines'):
            from . import guides
            try:
                gs = guides.controller(self.ctx).ground()
                st = guides.controller(self.ctx).stored()
            except Exception:
                return None
            for ln in gs.lines:
                if ln.kind != 'distance' or not len(ln.pts):
                    continue
                dd = np.hypot(ln.pts[:, 0] - g[0], ln.pts[:, 1] - g[1]).min()
                if dd < 0.10:
                    ds = list(st.distances)
                    k = next((i for i, d in enumerate(ds) if f'{d:g}' in ln.label), None)
                    if k is None:
                        continue
                    sign = -1.0 if st.direction == 'reverse' else 1.0
                    return {'kind': 'guide', 'uid': ln.id, 'index': k, 'd0': ds[k], 'g0': (float(g[0]), float(g[1])),
                            'sign': sign, 'mouse': pos, 'at': pos, 'readout': ''}
        return None

    def _seam_tags(self, s):
        cams = {c.uid: c.tag for c in self.cameras()}
        return f'{cams.get(s.a, "?")}|{cams.get(s.b, "?")}'

    def _alt_move(self, vp, d, pos):
        st = self.ctx.store
        g = vp.ground_at(pos.x(), pos.y())
        if g is None:
            return
        d['at'] = pos
        if d['kind'] == 'seam':
            cx, cy = self.result.centre
            ang = round(math.degrees(math.atan2(g[1] - cy, g[0] - cx)), 1)
            key = d['uid']

            def fn(pj, key=key, ang=ang):
                pj.variant.seams[key] = ang
            st.edit('variant', fn, f'Moved seam {d["tags"]}', merge_key=('seam', key))
            parts = [f'seam {d["tags"]} {wrap180(ang - d["a0"]):+.1f}°']
            if self.ctx.analysis.drag_live_ms() is None:
                parts.append('blind and vanishing area update on release')      # never a stale +0 figure
            else:
                base = d.get('hud0')
                mt = self.hud_metrics()
                if base and mt:
                    parts.append(f'blind {mt["blind_m2"] - base["blind_m2"]:+.3f} m²')
                v0, v1 = d.get('vanish0'), self._vanish_m2()
                if v0 and v1 is not None:
                    parts.append(f'vanishing child area {100 * (v1 - v0) / v0:+.0f} %')
            d['readout'] = '  |  '.join(parts)
        else:
            from . import guides
            dx = (g[0] - d['g0'][0]) * d['sign']
            nd = max(0.1, round(d['d0'] + dx, 2))
            gc = guides.controller(self.ctx)
            ds = list(gc.stored().distances)
            ds[d['index']] = nd
            gc.set(_merge=True, distances=sorted(ds))
            d['index'] = sorted(ds).index(nd)
            d['readout'] = f'marker {d["d0"]:g} m → {nd:g} m  ({nd - d["d0"]:+.2f} m)'
        self.changed.emit()

    # ------------------------------------------------------------------ the camera picture-in-picture
    PIP_W, PIP_H = 256, 205

    def _pip_uid(self):
        """The camera the picture-in-picture shows: the pinned one, else the one under the pointer."""
        return getattr(self, 'pip_pinned', None) or getattr(self, 'pip_hover', None)

    def _label_at(self, pos):
        for r, uid in getattr(self, '_label_rects', []):
            if r.contains(pos):
                return uid
        return None

    def pip_image(self, cam):
        """(QImage, intrinsics) of a camera's simulated view (camview.render_camera: the real lens, the body, the
        ground grid, the ROI and seams), cached per camera and analysis."""
        from .camview import render_camera
        res = self.result
        key = (cam.uid, id(res), cam.x, cam.y, cam.z, cam.yaw, cam.pitch, cam.roll, cam.lens_id, THEME.name)
        cache = getattr(self, '_pip_cache', None)
        if cache is not None and cache[0] == key:
            return cache[1], cache[2]
        m = self.project.module_for(cam)
        w = self.PIP_W
        h = int(round(w * m.height_px / max(m.width_px, 1)))
        img, _ = render_camera(self.project, self.ctx.analysis.engine, cam, w, h, res)
        intr = resolve(m, self.project.lens_for(cam))
        self._pip_cache = (key, img, intr)
        return img, intr

    def pip_point(self, cam, intr, world, w, h):
        """Where a world point lands in the camera's image, scaled to a w × h thumbnail; None outside the lens."""
        d = np.asarray(world, float) - cam.pos()
        R, W2O = camera_frames(cam)
        opt = (W2O @ d)[None]
        u, v, _, ok = intr.project_dirs(opt)
        if not ok[0]:
            return None
        m = self.project.module_for(cam)
        x, y = float(u[0]) * w / m.width_px, float(v[0]) * h / m.height_px
        return (x, y) if 0 <= x <= w and 0 <= y <= h else None

    def _paint_pip(self, p, vp):
        self._pip_close = self._pip_pin = None
        uid = self._pip_uid()
        if uid is None or self.mode != 'scene' or self.label_prefix:
            return
        cam = self.ctx.store.cam(uid)
        if cam is None:
            self.pip_pinned = self.pip_hover = None
            return
        try:
            img, intr = self.pip_image(cam)
        except Exception as e:                              # a camera without a lens model: nothing to show
            print('picture-in-picture:', e)
            return
        w, h = img.width(), img.height()
        pinned = getattr(self, 'pip_pinned', None) == uid
        x0, y0 = vp.width() - w - 64.0, 44.0                 # top right, clear of the navigation controls
        p.save()
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        frame = QRectF(x0 - 1, y0 - 21, w + 2, h + 22)
        p.setPen(QPen(THEME.cam(cam.slot), 1.5))
        p.setBrush(THEME.color('panel', 0.92))
        p.drawRoundedRect(frame, 5, 5)
        p.drawImage(QRectF(x0, y0, w, h), img)
        p.setFont(sans(10, QFont.Weight.DemiBold))
        p.setPen(THEME.color('ink'))
        lens = self.project.lens_for(cam)
        p.drawText(QPointF(x0 + 6, y0 - 6), f'{cam.tag} · through the lens' + (f' · {lens.name}' if lens else ''))
        # pin and close
        p.setFont(sans(10))
        self._pip_pin = QRectF(x0 + w - 40, y0 - 19, 18, 16)
        self._pip_close = QRectF(x0 + w - 20, y0 - 19, 18, 16)
        p.setPen(THEME.color('ink') if pinned else THEME.color('ink2'))
        p.drawText(self._pip_pin, Qt.AlignmentFlag.AlignCenter, '●' if pinned else '○')
        p.setPen(THEME.color('ink2'))
        p.drawText(self._pip_close, Qt.AlignmentFlag.AlignCenter, '×')
        # the 3D cursor's ground point (or the probe) as a crosshair in the image
        g = getattr(self, '_cursor_g', None)
        pts = [g] if g is not None else []
        pts += [pr['point'] for pr in getattr(self, 'probes', [])]
        for k, wp in enumerate(pts):
            q = self.pip_point(cam, intr, wp, w, h)
            if q is None:
                continue
            cx, cy = x0 + q[0], y0 + q[1]
            p.setPen(QPen(QColor('#ffffff') if k == 0 and g is not None else QColor('#e8a33d'), 1.3))
            p.drawLine(QPointF(cx - 9, cy), QPointF(cx - 3, cy))
            p.drawLine(QPointF(cx + 3, cy), QPointF(cx + 9, cy))
            p.drawLine(QPointF(cx, cy - 9), QPointF(cx, cy - 3))
            p.drawLine(QPointF(cx, cy + 3), QPointF(cx, cy + 9))
        if not pinned:
            p.setFont(sans(9))
            p.setPen(THEME.color('muted'))
            p.drawText(QPointF(x0 + 4, y0 + h + 14), 'click the label to pin')
        p.restore()

    def pip_click(self, pos) -> bool:
        """Pin, unpin or close the picture-in-picture; a click on a camera's label pins that camera."""
        if getattr(self, '_pip_close', None) is not None and self._pip_close.contains(pos):
            self.pip_pinned = self.pip_hover = None
            self.changed.emit()
            return True
        if getattr(self, '_pip_pin', None) is not None and self._pip_pin.contains(pos):
            uid = self._pip_uid()
            self.pip_pinned = None if getattr(self, 'pip_pinned', None) == uid else uid
            self.changed.emit()
            return True
        uid = self._label_at(pos)
        if uid is not None:
            self.pip_pinned = uid
            self.ctx.store.select_cams([uid])
            self.changed.emit()
            return True
        return False

    # ------------------------------------------------------------------ the legend and live metrics card
    def hud_metrics(self):
        """The card's live figures from the current analysis: blind ROI (m², %), covered %, seen by 2+ cameras %,
        minimum / median best-camera density (px/cm), the analysis time (ms). None without a result."""
        res = self.result
        if res is None:
            return None
        g, m = res.grid, res.metrics
        cov = g.roi & (res.count > 0)
        d = res.best_dens[cov]
        roi_a = max(float(m.get('roi_area', 0.0)), 1e-9)
        return {'blind_m2': float(m['blind_m2']), 'blind_pct': 100.0 * float(m['blind_m2']) / roi_a,
                'covered_pct': float(m['covered_pct']), 'multi_pct': float(m['multi_pct']),
                'min_density': float(m['min_density']), 'median_density': float(np.median(d)) if d.size else 0.0,
                'ms': float(getattr(res, 'ms', 0.0))}

    def hud_legend(self):
        """[(kind, colour or colours, text)] for the overlays that are on: what each mark on the scene means."""
        ov, out = self.ctx.overlays, []
        res = self.result
        cams = [c for c in self.cameras() if c.visible] if res is not None else []
        heat = self.ctx.heatmap if self.mode == 'scene' else 'none'
        if heat == 'none' and ov.get('footprints', True) and cams:
            out.append(('swatches', [(THEME.cam(c.slot), c.tag) for c in cams], 'ground each camera sees (tint)'))
        if heat == 'density':
            a = self.project.analysis
            out.append(('ramp', 'density', f'best-camera density, px/cm (min {a.min_density:g}, '
                                           f'target {a.target_density:g}; grey: below the minimum)'))
        elif heat == 'redundancy':
            out.append(('ramp', 'redundancy', 'cameras seeing the ground: 1, 2, 3+'))
        elif heat == 'coverage':
            out.append(('swatches', [(THEME.cam(c.slot), c.tag) for c in cams], 'best camera at each cell'))
        elif heat in getattr(self, 'RESEARCH_HEAT', {}):
            out.append(('ramp', heat, self.RESEARCH_HEAT[heat][1]))
        if res is not None:
            out.append(('hatch', QColor(STATUS['critical']), 'blind: no camera sees the ground'))
        if ov.get('frustums', True) and cams:
            out.append(('fill', THEME.color('ink2', 0.35), 'camera fields on the ground (overlaps add up)'))
        if ov.get('roi', True):
            roi = self.project.roi
            out.append(('dash', THEME.color('ink', 0.9), f'ROI, {roi.distance:g} m from the body'
                        if roi.mode == 'offset' else 'ROI (your polygon)'))
        if ov.get('seams', True) and res is not None and res.seams:
            out.append(('dash', THEME.color('ink2', 0.9), 'stitch seams between cameras'))
        if ov.get('guidelines'):
            out.append(('swatches', [(QColor('#d33a2c'), 'near'), (QColor('#e0b429'), 'mid'),
                                     (QColor('#3aa655'), 'far')], 'parking guidelines (distance boxes)'))
        if ov.get('rings'):
            out.append(('dash', THEME.color('ink2', 0.75), 'range rings 1, 2, 5, 10 m from the body'))
        if ov.get('sensors', True) and self.project.variant.sensors:
            out.append(('line', THEME.color('ink2'), 'lidar rings on the ground · radar fields'))
        if ov.get('dimensions'):
            out.append(('line', THEME.color('ink'), 'vehicle and camera dimensions'))
        if ov.get('fusion') and self.project.variant.sensors:
            out.append(('ramp', 'bevfeat', 'lidar rings by camera px per lidar point (log2 1…256; red: no camera)'))
        if ov.get('wedges') and self.project.variant.sensors:
            out.append(('fill', THEME.color('ink2', 0.3), 'lidar azimuths sampled > 10 ms from each camera\'s exposure'))
        mo = self.ctx.motion
        if abs(mo['speed']) > 1e-6 or abs(mo['yaw']) > 1e-6:
            out.append(('dash', THEME.color('ink2'), f'where each camera samples at {mo["speed"]:g} m/s, '
                                                     f'{mo["yaw"]:g}°/s (trigger + rolling shutter)'))
        if getattr(self, 'probes', None):
            out.append(('line', QColor('#e8a33d'), 'probe rays to the cameras that see it'))
        return out

    def _hud_open(self):
        st = getattr(self.ctx, 'settings', None)
        try:
            return bool(st.value('view/hud_open', True, type=bool)) if st is not None else True
        except Exception:
            return True

    def _paint_hud(self, p, vp):
        """The bottom-left card: live metrics (with the change since a drag began) and the legend of what is on."""
        self._hud_head = None
        if self.mode != 'scene' or not self.ctx.overlays.get('hud', True) or self.label_prefix:
            return
        mt = self.hud_metrics()
        if mt is None:
            return
        if self.drag is not None:
            self.drag.setdefault('hud0', mt)
        base = self.drag.get('hud0') if self.drag is not None else None
        is_open = self._hud_open()
        W = 300
        pad, lh = 9, 16
        rows = []
        if is_open:
            def delta(k, fmt, unit=''):
                if base is None or abs(mt[k] - base[k]) < 1e-9:
                    return ''
                return f'  ({mt[k] - base[k]:+{fmt}}{unit})'
            rows = [('Blind ROI', f'{mt["blind_m2"]:.3f} m² · {mt["blind_pct"]:.2f} %' + delta('blind_m2', '.3f', ' m²'),
                     mt['blind_m2'] > 0.0005),
                    ('Covered · 2+ cams', f'{mt["covered_pct"]:.1f} % · {mt["multi_pct"]:.1f} %', False),
                    ('Min / median px/cm', f'{mt["min_density"]:.1f} / {mt["median_density"]:.1f}'
                     + delta('median_density', '.1f'), mt['min_density'] < self.project.analysis.min_density),
                    ('Analysis', f'{mt["ms"]:.0f} ms' + ((' · live while dragging'
                                                          if self.ctx.analysis.drag_live_ms() is not None else
                                                          ' · updates on release') if base is not None else ''),
                     False)]
        legend = self.hud_legend() if is_open else []
        h = pad * 2 + lh + (len(rows) * lh + 6 if rows else 0) + (len(legend) * lh + 8 if legend else 0)
        x0, y0 = 10.0, vp.height() - h - 48.0                # above the axis triad
        p.save()
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        p.setPen(QPen(THEME.color('strong'), 1))
        bg = THEME.color('panel')
        bg.setAlphaF(0.88)
        p.setBrush(bg)
        p.drawRoundedRect(QRectF(x0, y0, W, h), 6, 6)
        y = y0 + pad + 11
        p.setFont(sans(11, QFont.Weight.DemiBold))
        p.setPen(THEME.color('ink'))
        p.drawText(QPointF(x0 + pad, y), 'Legend and live metrics')
        p.setPen(THEME.color('ink2'))
        p.drawText(QPointF(x0 + W - pad - 12, y), '▾' if is_open else '▸')
        self._hud_head = QRectF(x0, y0, W, pad + lh + 4)
        y += 6
        p.setFont(mono(10))
        for name, val, bad in rows:
            y += lh
            p.setPen(THEME.color('ink2'))
            p.drawText(QPointF(x0 + pad, y), name)
            p.setPen(QColor(STATUS['critical']) if bad else THEME.color('ink'))
            p.drawText(QPointF(x0 + pad + 128, y), val)
        if legend:
            y += 8
            p.setFont(sans(10))
            for kind, col, text in legend:
                y += lh
                sx = x0 + pad
                if kind == 'swatches':
                    for c, tag in col[:8]:
                        p.setPen(Qt.PenStyle.NoPen)
                        p.setBrush(c)
                        p.drawRect(QRectF(sx, y - 9, 9, 9))
                        sx += 11
                    tx = x0 + pad + max(11 * min(len(col), 8), 40) + 4
                elif kind == 'ramp':
                    grad = [QColor(h_) for h_ in self._ramp_hex(col)]
                    for i, c in enumerate(grad):
                        p.setPen(Qt.PenStyle.NoPen)
                        p.setBrush(c)
                        p.drawRect(QRectF(sx + i * 40 / len(grad), y - 9, 40 / len(grad) + 0.5, 9))
                    tx = sx + 44
                elif kind == 'hatch':
                    p.setPen(Qt.PenStyle.NoPen)
                    p.setBrush(QBrush(col, Qt.BrushStyle.BDiagPattern))
                    p.drawRect(QRectF(sx, y - 9, 26, 9))
                    tx = sx + 30
                elif kind == 'fill':
                    p.setPen(Qt.PenStyle.NoPen)
                    p.setBrush(col)
                    p.drawRect(QRectF(sx, y - 9, 26, 9))
                    tx = sx + 30
                else:                                          # dash, line
                    pen = QPen(col, 1.5)
                    if kind == 'dash':
                        pen.setStyle(Qt.PenStyle.DashLine)
                    p.setPen(pen)
                    p.drawLine(QPointF(sx, y - 4), QPointF(sx + 26, y - 4))
                    tx = sx + 30
                p.setPen(THEME.color('ink2'))
                fm = p.fontMetrics()
                p.drawText(QPointF(tx, y), fm.elidedText(text, Qt.TextElideMode.ElideRight, int(x0 + W - pad - tx)))
        p.restore()

    @staticmethod
    def _ramp_hex(kind):
        from . import colormaps as CM
        th = 'dark' if THEME.name != 'light' else 'light'
        if kind in ('bevfeat', 'stereo'):
            from .research_overlays import RESEARCH_RAMP
            return list(RESEARCH_RAMP)
        if kind == 'density':
            stops = CM.DENSITY[th]
        elif kind == 'redundancy':
            stops = CM.REDUNDANCY[th]
        else:
            stops = getattr(CM, 'BLUE', CM.DENSITY[th])
        return [s if isinstance(s, str) else CM.rgb_hex(s) if hasattr(CM, 'rgb_hex') else '#888888' for s in stops][:10]

    def hud_click(self, pos) -> bool:
        """A click on the card's header folds or unfolds it (remembered)."""
        r = getattr(self, '_hud_head', None)
        if r is None or not r.contains(pos):
            return False
        st = getattr(self.ctx, 'settings', None)
        if st is not None:
            st.setValue('view/hud_open', not self._hud_open())
        self.changed.emit()
        return True

    def overlay(self, p: QPainter, vp):
        if self.project.vehicle.mesh is None and self.mode == 'scene':
            p.setFont(sans(12, QFont.Weight.DemiBold))
            p.setPen(THEME.color('ink'))
            p.drawText(QPointF(12, 44), 'No vehicle loaded')
            p.setFont(sans(12))
            p.setPen(THEME.color('ink2'))
            p.drawText(QPointF(12, 64), 'File ▸ Import ▸ Vehicle model…   Ctrl+I')
            p.drawText(QPointF(12, 82), 'or drop .glb .gltf .obj .stl .fbx .ply .dae .step on this view')
            p.drawText(QPointF(12, 100), 'or File ▸ New from template')
        sel = set(self.ctx.store.sel_cams)
        cams = [c for c in self.cameras() if c.visible and not (self.ctx.isolate and c.uid not in self.ctx.isolate)]
        labels = self.mode == 'scene' and bool(self.ctx.overlays.get('labels', True))
        placed, tags = [], []           # label plates already drawn (nearer and selected first), camera tags to draw
        if cams:
            P = np.array([c.pos() for c in cams])
            sc, dep, ok = vp.project(P)
            order = np.argsort(-dep)
            for k in order:
                if not ok[k]:
                    continue
                c = cams[k]
                x, y = sc[k]
                s = MARKER + (2 if c.uid in sel else 0)
                if self._module_px(vp, c) > 30:          # the module model itself shows: a small marker on it
                    s = 4 + (1 if c.uid in sel else 0)
                r = QRectF(x - s / 2, y - s / 2, s, s)
                p.setPen(QPen(QColor(0, 0, 0, 160), 1))
                p.setBrush(THEME.cam(c.slot))
                p.drawRect(r)
                if c.uid in sel:
                    p.setPen(QPen(THEME.color('ink'), 1))
                    p.setBrush(Qt.BrushStyle.NoBrush)
                    p.drawRect(r.adjusted(-2.5, -2.5, 2.5, 2.5))
                if labels:
                    tags.append((dep[k], x + s / 2 + 5, y, c))
                mi = self.ctx.analysis.mounts.get(c.uid) if self.cameras_override is None else None
                if mi is not None and mi.floating:
                    p.setFont(sans(11))
                    txt = f'Critical · {c.tag} is {mi.gap_mm:.0f} mm off the body'
                    fw = p.fontMetrics().horizontalAdvance(txt)
                    p.setPen(Qt.PenStyle.NoPen)
                    p.setBrush(QColor(0, 0, 0, 150))
                    p.drawRect(QRectF(x + 10, y - 22, fw + 26, 18))
                    p.drawPixmap(int(x + 14), int(y - 19), icons.pixmap('critical', STATUS['critical'], 12))
                    p.setPen(QColor(240, 239, 236))
                    p.drawText(QPointF(x + 30, y - 9), txt)
        # camera tags: selected cameras first, then the nearer ones; a tag that would cover another is left out
        self._label_rects = []
        for _, x, y, c in sorted(tags, key=lambda t: (t[3].uid not in sel, t[0])):
            r = self._chip(p, x, y, c.tag, accent=THEME.cam(c.slot), placed=placed, strong=c.uid in sel)
            if r is not None:
                self._label_rects.append((r, c.uid))
        # gizmo
        tg = self._gizmo_target()
        if tg is not None:
            c = tg[2]
            if self.ctx.tool in ('move', 'select'):
                self._gizmo_move(p, vp, c, faint=self.ctx.tool != 'move' and not self.drag and not self.gizmo_hot)
            elif self.ctx.tool == 'rotate':
                self._gizmo_rotate(p, vp, c, rings=self._rings(tg))
        # target labels
        if self.ctx.overlays.get('targets') and self.mode == 'scene':
            st = {s.uid: s for s in self.ctx.analysis.targets}
            for t in self.project.targets:
                q, _, ok = vp.project([[t.x, t.y, 0.0]])
                if ok[0]:
                    s = st.get(t.uid)
                    bad = s is not None and not s.ok
                    if labels:
                        self._chip(p, q[0][0] + 10, q[0][1] - 10, t.name, placed=placed,
                                   ink=QColor(STATUS['critical']) if bad else None,
                                   swatch=QColor(STATUS['critical']) if bad else None)
                    else:
                        p.setFont(mono(11))
                        p.setPen(THEME.color('ink') if not bad else QColor(STATUS['critical']))
                        p.drawText(QPointF(q[0][0] + 10, q[0][1] - 10), t.name)
        if self.ctx.overlays.get('rings') and getattr(self, '_ring_labels', None) and self.mode == 'scene':
            p.setFont(mono(10))
            for x, y, z, txt in self._ring_labels:
                q, _, ok = vp.project([[x, y, z]])
                if ok[0]:
                    p.setPen(THEME.color('ink2'))
                    p.drawText(QPointF(q[0][0] + 3, q[0][1] - 3), txt)
        if self.ctx.overlays.get('dimensions') and hasattr(self, '_dim_labels') and self.mode == 'scene':
            p.setFont(mono(11))
            p.setPen(THEME.color('ink'))
            for x, y, z, txt in self._dim_labels:
                q, _, ok = vp.project([[x, y, z]])
                if ok[0]:
                    if labels:
                        self._chip(p, q[0][0] - 30, q[0][1], txt)
                    else:
                        p.setFont(mono(11))
                        p.setPen(THEME.color('ink'))
                        p.drawText(QPointF(q[0][0] - 30, q[0][1] + 4), txt)
        # place preview
        if self.ctx.tool == 'place' and self.place_preview is not None:
            self._paint_disc(p, vp, *self.place_preview)
        if self.measure:
            self._paint_measure(p, vp)
        self._paint_ov3d(p, vp)
        self._paint_probes(p, vp)
        self._paint_hud(p, vp)
        self._paint_pip(p, vp)
        if self.show_tooltip and self.hover_info and self.ctx.tool in ('select', 'move') and self.drag is None:
            self._paint_tooltip(p, vp)
        if self.drag and self.drag.get('readout') and self.drag.get('at') is not None:
            at = self.drag['at']
            self._label_box(p, QPointF(at.x() + 18, at.y() - 12),
                            [self.drag['readout'], 'Ctrl snaps · Shift fine · Esc cancels'])

    def _module_px(self, vp, c):
        """Length of a camera's module on screen, pixels."""
        L = max(self.project.module_for(c).body_mm) / 1000.0
        return vp.px_per_m(c.pos()) * L

    def _plate(self, p, r):
        p.setPen(QPen(THEME.color('strong'), 1))
        p.setBrush(THEME.color('raised', 0.92))
        p.drawRoundedRect(r, 4, 4)

    def _paint_ov3d(self, p, vp):
        """Labels of the 3D overlays: each sky dome's open share, and the lidar points' range scale with the share
        of each lidar's beams the vehicle blocks."""
        if self.mode != 'scene' or not self.project.variant.sensors:
            return
        ov = self.ctx.overlays
        c = getattr(self.ctx, 'overlays3d', None)
        if c is None:
            return
        if ov.get('gnsssky'):
            p.setFont(mono(10))
            fm = p.fontMetrics()
            placed = []
            skies = c.fresh_skies()
            tops = [np.asarray(sv.pos, float) + np.asarray(sv.frame, float)[:, 2] * (O3.SKY_R + 0.05) for sv in skies]
            for sv, top in sorted(zip(skies, tops), key=lambda st: -vp.project([st[1]])[1][0]):   # far first
                q, _, ok = vp.project([top])
                if not ok[0]:
                    continue
                txt = f'{sv.tag} · {sv.visible * 100:.0f} % sky'
                w = fm.horizontalAdvance(txt) + 10
                r = QRectF(q[0][0] - w / 2, q[0][1] - 20, w, 16)
                while any(r.intersects(o) for o in placed):        # stack labels that would overlap
                    r.translate(0, -18)
                placed.append(r)
                self._plate(p, r)
                p.setPen(QColor(STATUS['critical']) if sv.no_sky else THEME.color('ink'))
                p.drawText(r, Qt.AlignmentFlag.AlignCenter, txt)
        if not ov.get('lidarpts') or not any(s.visible and s.kind == 'lidar' for s in self.project.variant.sensors):
            return
        sweeps = c.fresh_sweeps()
        far = sweeps[0].far if sweeps else O3.POINTS_FAR
        x0, h = 46.0, vp.height()
        p.setFont(sans(10))
        fm = p.fontMetrics()
        blocked = '  '.join(f'{sw.tag} {sw.blocked * 100:.0f} %' for sw in sweeps[:4])
        lines = 'Beams the vehicle blocks: ' + blocked if sweeps else 'Tracing the lidar sweeps…'
        bar_w = 150.0
        w = max(bar_w + 74, fm.horizontalAdvance(lines) + 30)
        r = QRectF(x0, h - 52, w, 44)
        self._plate(p, r)
        p.setPen(THEME.color('ink2'))
        p.drawText(QPointF(x0 + 8, h - 38), 'Lidar range')
        dark = THEME.name != 'light'
        from PyQt6.QtGui import QLinearGradient
        g = QLinearGradient(x0 + 70, 0, x0 + 70 + bar_w, 0)
        for f, rgb in zip(np.linspace(0, 1, 9), O3.ramp(np.linspace(0, 1, 9), dark)):
            g.setColorAt(float(f), QColor.fromRgbF(*[float(v) for v in rgb]))
        bar = QRectF(x0 + 70, h - 46, bar_w, 8)
        p.setPen(Qt.PenStyle.NoPen)
        p.setBrush(QBrush(g))
        p.drawRect(bar)
        p.setPen(THEME.color('muted'))
        p.setFont(mono(9))
        p.drawText(QRectF(bar.left() - 20, bar.bottom() + 1, 40, 11), Qt.AlignmentFlag.AlignCenter, '0')
        p.drawText(QRectF(bar.right() - 30, bar.bottom() + 1, 60, 11), Qt.AlignmentFlag.AlignCenter, f'{far:g} m')
        p.setPen(Qt.PenStyle.NoPen)
        p.setBrush(QColor(STATUS['critical']))
        p.drawEllipse(QPointF(x0 + 12, h - 16), 3.5, 3.5)
        p.setFont(sans(10))
        p.setPen(THEME.color('ink'))
        p.drawText(QPointF(x0 + 21, h - 12), lines)

    # ------------------------------------------------------------------ move and rotate handles
    def _gizmo_target(self):
        """What the handles move: the selected camera, sensor or rig part, as (kind, uid, object) with kind 'cam',
        'sensor' or 'rig'; None when nothing movable is selected (or this is not the editable scene)."""
        if self.mode != 'scene' or self.cameras_override is not None:
            return None
        st = self.ctx.store
        c = st.selected
        if c is not None:
            return ('cam', c.uid, c)
        so = st.sel_obj
        if len(so) == 2 and so[0] == 'sensor':
            sn = st.variant.sensor(so[1])
            if sn is not None and sn.visible:
                return ('sensor', sn.uid, sn)
        if len(so) == 2 and so[0] == 'rig':
            part = next((q for q in self.project.rig if q.uid == so[1]), None)
            if part is not None and part.visible:
                return ('rig', part.uid, part)
        return None

    def _obj(self, kind, uid):
        if kind == 'cam':
            return self.ctx.store.cam(uid)
        if kind == 'sensor':
            return self.ctx.store.variant.sensor(uid)
        return next((q for q in self.project.rig if q.uid == uid), None)

    @staticmethod
    def _rings(tg):
        """The rotate rings a target has: a rig part turns about z only; cameras and sensors also tilt."""
        return (0,) if tg[0] == 'rig' else (0, 1)

    def _gizmo_set(self, kind, uid, **props):
        """Write a handle drag's values, merging the steps of one drag into one undo step."""
        st = self.ctx.store
        if kind == 'cam':
            st.set_cam(uid, merge=True, **props)
            return
        moved = bool(set(props) & {'x', 'y', 'z'})
        if kind == 'sensor':
            from .sensorview import set_props
            sn = st.variant.sensor(uid)
            if sn is not None:
                set_props(self.ctx, uid, f'{"Moved" if moved else "Turned"} {sn.tag}', merge=True, **props)
            return
        part = next((q for q in self.project.rig if q.uid == uid), None)
        if part is None or all(getattr(part, k) == v for k, v in props.items()):
            return

        def fn(pr):
            for q in pr.rig:
                if q.uid == uid:
                    for k, v in props.items():
                        setattr(q, k, v)
        st.edit('rig', fn, f'{"Moved" if moved else "Turned"} {part.name}', merge_key=(uid, tuple(sorted(props))))

    def _gizmo_axes(self, vp, c):
        pos = _pos(c)
        L = 64.0 / max(vp.px_per_m(pos), 1e-3)
        return pos, L, [np.array([1.0, 0, 0]), np.array([0, 1.0, 0]), np.array([0, 0, 1.0])]

    def _plane_quad(self, vp, c):
        """The square handle between the x and y arrows that moves in the horizontal plane: its screen corners, or
        None when it is seen edge on."""
        pos, L, _ = self._gizmo_axes(vp, c)
        a, b = 0.2 * L, 0.42 * L
        q, _, ok = vp.project([pos + [a, a, 0], pos + [b, a, 0], pos + [b, b, 0], pos + [a, b, 0]])
        if not ok.all():
            return None
        poly = QPolygonF([QPointF(*x) for x in q])
        r = poly.boundingRect()
        return poly if min(r.width(), r.height()) > 3 else None

    def _gizmo_move(self, p, vp, c, faint=False):
        pos, L, axes = self._gizmo_axes(vp, c)
        cols = [QColor('#e0533f'), QColor('#3fae49'), QColor('#3d86e6')]
        o, _, ok = vp.project([pos])
        if not ok[0]:
            return
        d = self.drag or {}
        quad = self._plane_quad(vp, c)
        if quad is not None:
            hot = d.get('plane') or self.gizmo_hot == ('plane',)
            col = QColor(cols[2])
            col.setAlphaF(0.75 if hot else (0.3 if faint else 0.45))
            p.setPen(QPen(QColor(cols[2]), 1.4 if hot else 1.0))
            p.setBrush(col)
            p.drawPolygon(quad)
        for k, a in enumerate(axes):
            e, _, ok2 = vp.project([pos + a * L])
            if not ok2[0]:
                continue
            col = QColor(cols[k])
            hot = d.get('axis') == k or self.gizmo_hot == ('axis', k)
            if hot:
                col = col.lighter(120)
            elif faint:
                col.setAlphaF(0.6)
            p.setPen(QPen(col, 3.4 if hot else 2.0))
            p.drawLine(QPointF(*o[0]), QPointF(*e[0]))
            dv = e[0] - o[0]
            n = np.linalg.norm(dv)
            if n > 1:
                dv /= n
                perp = np.array([-dv[1], dv[0]])
                tip = e[0] + dv * 3
                w = 5.0 if hot else 4.0
                poly = QPolygonF([QPointF(*tip), QPointF(*(tip - dv * 11 + perp * w)), QPointF(*(tip - dv * 11 - perp * w))])
                p.setBrush(col)
                p.setPen(Qt.PenStyle.NoPen)
                p.drawPolygon(poly)
        p.setPen(QPen(THEME.color('ink'), 1.2))
        p.setBrush(THEME.color('panel'))
        p.drawEllipse(QPointF(*o[0]), 3.0, 3.0)

    def _ring(self, vp, c, which):
        pos = _pos(c)
        L = 56.0 / max(vp.px_per_m(pos), 1e-3)
        a = np.linspace(0, 2 * math.pi, 64)
        if which == 0:      # yaw ring around world z
            pts = pos + np.c_[np.cos(a), np.sin(a), np.zeros_like(a)] * L
        else:               # pitch ring around the camera's lateral axis
            R = rot_zyx(c.yaw, 0, 0)
            fwd, up = R[:, 0], R[:, 2]
            pts = pos + (np.outer(np.cos(a), fwd) + np.outer(np.sin(a), up)) * L
        return pts

    def _gizmo_rotate(self, p, vp, c, rings=(0, 1)):
        cols = [QColor('#3d86e6'), QColor('#3fae49')]
        for k in rings:
            pts = self._ring(vp, c, k)
            sc, _, ok = vp.project(pts)
            hot = (self.drag or {}).get('ring') == k or self.gizmo_hot == ('ring', k)
            p.setPen(QPen(cols[k].lighter(120) if hot else cols[k], 3.4 if hot else 2.0))
            p.setBrush(Qt.BrushStyle.NoBrush)
            p.drawPolyline(QPolygonF([QPointF(*q) for q, o in zip(sc, ok) if o]))
        # aim direction
        R = rot_zyx(c.yaw, getattr(c, 'pitch', 0.0), getattr(c, 'roll', 0.0))
        pos = _pos(c)
        L = 70.0 / max(vp.px_per_m(pos), 1e-3)
        sc, _, ok = vp.project([pos, pos + R[:, 0] * L])
        if ok.all():
            p.setPen(QPen(THEME.color('ink'), 1.2, Qt.PenStyle.DashLine))
            p.drawLine(QPointF(*sc[0]), QPointF(*sc[1]))

    def _paint_disc(self, p, vp, pos, normal, allowed, surface):
        n = np.asarray(normal)
        t1 = np.cross([0, 0, 1.0], n)
        if np.linalg.norm(t1) < 1e-6:
            t1 = np.array([1.0, 0, 0])
        t1 /= np.linalg.norm(t1)
        t2 = np.cross(n, t1)
        a = np.linspace(0, 2 * math.pi, 40)
        r = 0.04
        pts = pos + n * 0.003 + (np.outer(np.cos(a), t1) + np.outer(np.sin(a), t2)) * r
        sc, _, ok = vp.project(pts)
        if not ok.all():
            return
        poly = QPolygonF([QPointF(*q) for q in sc])
        col = THEME.color('ink') if allowed else QColor(STATUS['critical'])
        p.setPen(QPen(col, 1.5))
        if allowed:
            c2 = QColor(col)
            c2.setAlphaF(0.18)
            p.setBrush(c2)
        else:
            p.setBrush(QBrush(col, Qt.BrushStyle.BDiagPattern))
        p.drawPolygon(poly)
        q, _, _ = vp.project([pos])
        p.setFont(sans(11))
        p.setPen(THEME.color('ink'))
        alt = ' · Alt-click: mirrored pair' if allowed else ''
        p.drawText(QPointF(q[0][0] + 14, q[0][1] - 10), f'{surface}{alt}' if allowed else f'{surface} · forbidden')

    def _paint_measure(self, p, vp):
        pts = np.array(self.measure)
        sc, _, ok = vp.project(pts)
        p.setPen(QPen(THEME.color('ink'), 1))
        p.setBrush(THEME.color('ink'))
        for q, o in zip(sc, ok):
            if o:
                p.drawEllipse(QPointF(*q), 3, 3)
        if len(pts) >= 2 and ok[:2].all():
            d = pts[1] - pts[0]
            mid = (sc[0] + sc[1]) / 2
            txt = f'{np.linalg.norm(d):.3f} m   Δx {d[0]:+.3f}  Δy {d[1]:+.3f}  Δz {d[2]:+.3f}'
            self._label_box(p, QPointF(mid[0] + 8, mid[1] - 8), [txt])

    def _chip(self, p, x, y, text, swatch=None, ink=None, placed=None, strong=False, accent=None):
        """A small label plate, its left edge at x and centred on y, with an optional colour swatch before the text or
        accent bar along its left edge. With placed (the plates drawn so far), a plate that would cover one of them is
        left out (None); else it joins them."""
        p.setFont(mono(10, QFont.Weight.DemiBold))
        fm = p.fontMetrics()
        sw = 7 if swatch is not None else 0
        w = fm.horizontalAdvance(text) + 10 + (sw + 4 if sw else 0) + (2 if accent is not None else 0)
        h = fm.height() + 3
        r = QRectF(x, y - h / 2, w, h)
        if placed is not None:
            if any(r.intersects(q) for q in placed):
                return None
            placed.append(r.adjusted(-2, -2, 2, 2))
        p.setPen(QPen(THEME.color('ink', 0.75) if strong else THEME.color('strong'), 1))
        p.setBrush(THEME.color('panel', 0.84))
        p.drawRoundedRect(r, 3, 3)
        tx = r.left() + 5
        if accent is not None:
            p.setPen(Qt.PenStyle.NoPen)
            p.setBrush(accent)
            p.drawRoundedRect(QRectF(r.left() + 1.5, r.top() + 2.5, 2.5, r.height() - 5), 1.2, 1.2)
            tx += 2
        if sw:
            p.setPen(Qt.PenStyle.NoPen)
            p.setBrush(swatch)
            p.drawRoundedRect(QRectF(tx, y - sw / 2, sw, sw), 1.5, 1.5)
            tx += sw + 4
        p.setPen(ink or THEME.color('ink'))
        p.drawText(QPointF(tx, y + fm.ascent() / 2 - 1.5), text)
        return r

    def _label_box(self, p, at, lines, mono_lines=True):
        p.setFont(mono(11) if mono_lines else sans(11))
        fm = p.fontMetrics()
        w = max(fm.horizontalAdvance(t) for t in lines) + 16
        h = len(lines) * 16 + 8
        r = QRectF(at.x(), at.y() - h, w, h)
        p.setPen(QPen(THEME.color('strong'), 1))
        p.setBrush(THEME.color('raised'))
        p.drawRoundedRect(r, 4, 4)
        p.setPen(THEME.color('ink'))
        for i, t in enumerate(lines):
            p.drawText(QPointF(r.left() + 8, r.top() + 16 + i * 16), t)

    def _paint_tooltip(self, p, vp):
        info = self.hover_info
        pos = vp.hover_pos
        if pos is None or info is None:
            return
        paint_ground_tooltip(p, pos, info, self.project, self.cameras())

    # ------------------------------------------------------------------ interaction
    def _pick_camera(self, vp, pos, radius=10):
        cams = [c for c in self.cameras() if c.visible and (not self.ctx.isolate or c.uid in self.ctx.isolate)]
        if not cams:
            return None
        sc, dep, ok = vp.project(np.array([c.pos() for c in cams]))
        d = np.hypot(sc[:, 0] - pos.x(), sc[:, 1] - pos.y())
        d[~ok] = 1e9
        k = int(np.argmin(d))
        return cams[k] if d[k] < radius else None

    def _mesh_hit(self, vp, pos):
        vg = self.ctx.analysis.engine.vehicle_geom(self.project)
        if vg is None:
            return None
        o, d = vp.ray(pos.x(), pos.y())
        m = vg.occ_mesh
        t, f = m.raycast(o[None], d[None], faces_mask=m.occluder_faces())
        if f[0] < 0:
            return None
        hit = o + d * t[0]
        n = m.normals[f[0]]
        if n @ d > 0:
            n = -n
        zi = int(m.face_zone[f[0]])
        if 0 <= zi < len(m.zones):
            z = m.zones[zi]
            return hit, n, z.kind != 'forbidden', z.name
        part = m.parts[int(m.face_part[f[0]])]
        allowed = 'not mountable' not in part.role
        return hit, n, allowed, part.name

    GIZMO_TOL = 10.0                    # px from a handle that still grabs it

    def _gizmo_pick(self, vp, pos, c, rings=(0, 1)):
        """The handle of object c under screen position pos: ('axis', k), ('plane',) or ('ring', k); None."""
        if self.ctx.tool == 'rotate':
            best = None
            for k in rings:
                sc, _, ok = vp.project(self._ring(vp, c, k))
                if not ok.any():
                    continue
                dmin = float(np.hypot(sc[:, 0] - pos.x(), sc[:, 1] - pos.y())[ok].min())
                if dmin < self.GIZMO_TOL and (best is None or dmin < best[1]):
                    best = (k, dmin)
            return ('ring', best[0]) if best else None
        if self.ctx.tool not in ('move', 'select'):
            return None
        p0, L, axes = self._gizmo_axes(vp, c)
        o, _, ok = vp.project([p0])
        best = None
        for k, a in enumerate(axes):
            e, _, ok2 = vp.project([p0 + a * L])
            if not (ok[0] and ok2[0]):
                continue
            A, B = o[0], e[0] + (e[0] - o[0]) * 0.08          # the arrow head reaches a little past the end
            AB = B - A
            if AB @ AB < 16:                                    # seen end on: the plane handle and the others
                continue
            t = np.clip(((pos.x() - A[0]) * AB[0] + (pos.y() - A[1]) * AB[1]) / max(AB @ AB, 1e-9), 0, 1)
            q = A + AB * t
            dist = math.hypot(pos.x() - q[0], pos.y() - q[1])
            if dist < self.GIZMO_TOL and t > 0.12 and (best is None or dist < best[1]):
                best = (k, dist)
        if best:
            return ('axis', best[0])
        quad = self._plane_quad(vp, c)
        if quad is not None and quad.containsPoint(pos, Qt.FillRule.OddEvenFill):
            return ('plane',)
        return None

    def _object_under(self, vp, pos, kind, uid) -> bool:
        """Whether the sensor or rig part uid is under screen position pos."""
        o, d = vp.ray(pos.x(), pos.y())
        if kind == 'sensor':
            from .sensorview import pick as pick_sensor
            return pick_sensor(self.project, o, d) == uid
        if kind == 'rig':
            from .rigview import pick as pick_rig
            return pick_rig(self.project, self.ctx.analysis.engine.vehicle_geom(self.project), o, d) == uid
        return False

    def home_box(self, vp):
        """What the 3D view's frame button shows: the vehicle and the ground round it, as Home does."""
        win = self.ctx.window
        if win is None or not hasattr(win, 'viewarea'):
            return None
        lo, hi = win.viewarea._bbox()
        return lo, hi, 0.82

    def pick_point(self, vp, pos):
        """The 3D point under screen position pos that the view orbits about and zooms towards: a camera, the body or
        a rig part where the ray meets it, a sensor's housing; None over the ground or the sky."""
        c = self._pick_camera(vp, pos)
        if c is not None:
            return c.pos()
        o, d = vp.ray(pos.x(), pos.y())
        best = None
        vg = self.ctx.analysis.engine.vehicle_geom(self.project)
        if vg is not None:
            t, f = vg.occ_mesh.raycast(o[None], d[None])
            if f[0] >= 0:
                best = float(t[0])
        if self.project.variant.sensors:
            from .sensorview import pick as pick_sensor
            uid = pick_sensor(self.project, o, d)
            sn = self.project.variant.sensor(uid) if uid else None
            if sn is not None:
                ts = float((sn.pos() - o) @ d)
                if ts > 0 and (best is None or ts < best):
                    best = ts
        return o + d * best if best is not None else None

    def mouse_press(self, vp, e):
        if e.button() != Qt.MouseButton.LeftButton or self.mode != 'scene' or self.cameras_override is not None:
            return False
        pos = e.position()
        if self.hud_click(pos) or self.pip_click(pos):
            return True
        if e.modifiers() & Qt.KeyboardModifier.AltModifier:
            d = self._alt_pick(vp, pos)
            if d is not None:
                d['hud0'] = self.hud_metrics()
                self.drag = d
                self.ctx.store.dragging = True
                return True
        tool = self.ctx.tool
        st = self.ctx.store
        c = st.selected
        if tool == 'aim':
            g = vp.ground_at(pos.x(), pos.y())
            if g is not None and st.sel_cams:
                for u in list(st.sel_cams):
                    ops.aim_at(self.ctx, u, (float(g[0]), float(g[1]), 0.0))
                self.ctx.set_tool('select')
            return True
        tg = self._gizmo_target()
        if tg is not None and tool in ('move', 'rotate', 'select'):
            kind, uid, obj = tg
            g = self._gizmo_pick(vp, pos, obj, self._rings(tg))
            if g is None and tool in ('move', 'select') and kind != 'cam' and self._pick_camera(vp, pos) is None \
                    and self._object_under(vp, pos, kind, uid):
                g = ('plane',)                  # pressing on the selected sensor or rig part drags it about
            if g and not getattr(obj, 'locked', False):
                base = {'kind': kind, 'uid': uid, 'start': _pos(obj), 'mouse': pos, 'at': pos, 'readout': ''}
                if g[0] == 'axis':
                    self.drag = dict(base, axis=g[1])
                elif g[0] == 'plane':
                    self.drag = dict(base, plane=True)
                else:
                    self.drag = dict(base, ring=g[1], yaw=obj.yaw, pitch=getattr(obj, 'pitch', 0.0), turn=0.0,
                                     last=None)
                st.dragging = True             # (gizmo_hot stays: the hover after the release resets the cursor)
                return True
        if tool == 'place':
            hit = self._mesh_hit(vp, pos)
            if hit is not None:
                if not hit[2]:
                    self.ctx.msg(f'{hit[3]} is a forbidden mount zone.')
                    return True
                ops.add_camera(self.ctx, hit[0], hit[1], pair=bool(e.modifiers() & Qt.KeyboardModifier.AltModifier))
                return True
            return False
        if tool == 'measure':
            hit = self._mesh_hit(vp, pos)
            pt = hit[0] if hit is not None else vp.ground_at(pos.x(), pos.y())
            if pt is not None:
                if len(self.measure) >= 2:
                    self.measure = []
                self.measure.append(np.asarray(pt))
                if len(self.measure) == 2:
                    d = self.measure[1] - self.measure[0]
                    self.ctx.msg(f'Distance {np.linalg.norm(d):.3f} m · Δx {d[0]:+.3f} Δy {d[1]:+.3f} Δz {d[2]:+.3f}', 6000)
                self.changed.emit()
            return True
        hitc = self._pick_camera(vp, pos)
        if hitc is not None:
            add = bool(e.modifiers() & (Qt.KeyboardModifier.ShiftModifier | Qt.KeyboardModifier.ControlModifier))
            if not (hitc.uid in st.sel_cams and not add):
                st.select_cams([hitc.uid], add=add)
            if tool in ('move', 'select') and not hitc.locked and not add:
                self.drag = {'surface': True, 'kind': 'cam', 'uid': hitc.uid, 'start': hitc.pos(), 'mouse': pos,
                             'at': pos, 'readout': ''}
                st.dragging = True
            return True
        return False

    def mouse_move(self, vp, e):
        d = self.drag
        if not d:
            return
        if d.get('kind') in ('seam', 'guide'):
            self._alt_move(vp, d, e.position())
            return
        st = self.ctx.store
        kind = d.get('kind', 'cam')
        c = self._obj(kind, d['uid'])
        if c is None:
            return
        pos = e.position()
        d['at'] = pos
        fine = 0.1 if e.modifiers() & Qt.KeyboardModifier.ShiftModifier else 1.0
        snap = bool(e.modifiers() & Qt.KeyboardModifier.ControlModifier)
        set_ = lambda **kw: self._gizmo_set(kind, c.uid, **kw)
        if 'axis' in d:
            k = d['axis']
            a = np.zeros(3)
            a[k] = 1.0
            o, r = vp.ray(pos.x(), pos.y())
            o0, r0 = vp.ray(d['mouse'].x(), d['mouse'].y())
            P = d['start']

            def closest(o, r):
                w0 = P - o
                b = a @ r
                den = 1 - b * b
                if abs(den) < 1e-6:
                    return 0.0
                return (b * (r @ w0) - (a @ w0)) / den
            delta = (closest(o, r) - closest(o0, r0)) * fine
            if snap:
                delta = round(delta / SNAP_M) * SNAP_M
            elif self.ctx.snap == 'grid':
                delta = round(delta / 0.05) * 0.05
            new = d['start'] + a * delta
            set_(x=round(float(new[0]), 4), y=round(float(new[1]), 4), z=round(float(new[2]), 4))
            d['readout'] = f'Δ{"xyz"[k]} {delta * 1000:+.0f} mm  ·  {"xyz"[k]} {new[k]:.3f} m'
        elif d.get('plane'):
            g = vp.ground_at(pos.x(), pos.y(), d['start'][2])
            g0 = vp.ground_at(d['mouse'].x(), d['mouse'].y(), d['start'][2])
            if g is not None and g0 is not None:
                dv = (g - g0) * fine
                if snap:
                    dv[:2] = np.round(dv[:2] / SNAP_M) * SNAP_M
                newp = d['start'] + dv
                if not snap and self.ctx.snap == 'grid':
                    newp[:2] = np.round(newp[:2] / 0.05) * 0.05
                set_(x=round(float(newp[0]), 4), y=round(float(newp[1]), 4))
                dd = (newp - d['start']) * 1000
                d['readout'] = f'Δx {dd[0]:+.0f}  Δy {dd[1]:+.0f} mm'
        elif 'ring' in d:
            pc, _, _ = vp.project([_pos(c)])
            if d['last'] is None:
                d['last'] = math.atan2(d['mouse'].y() - pc[0][1], d['mouse'].x() - pc[0][0])
            a1 = math.atan2(pos.y() - pc[0][1], pos.x() - pc[0][0])
            # accumulate wrapped steps: the raw angle jumps by 360° when the cursor passes left of the camera
            d['turn'] += (math.degrees(a1 - d['last']) + 180.0) % 360.0 - 180.0
            d['last'] = a1
            da = d['turn'] * fine
            if d['ring'] == 0:
                # screen y points down: dragging counter-clockwise on screen turns the camera left when seen from above
                sign = -1 if vp.cam.view()[2, 2] > 0 else 1
                yaw = d['yaw'] + sign * da
                if snap:
                    yaw = round(yaw / SNAP_DEG) * SNAP_DEG
                yaw = round(wrap180(yaw), 1)
                set_(yaw=yaw)
                d['readout'] = f'yaw {yaw:.1f}°  ·  {wrap180(yaw - d["yaw"]):+.1f}°'
            else:
                pitch = d['pitch'] - da
                if snap:
                    pitch = round(pitch / SNAP_DEG) * SNAP_DEG
                pitch = round(max(-89.0, min(89.0, pitch)), 1)
                set_(pitch=pitch)
                d['readout'] = f'pitch {pitch:.1f}°  ·  {pitch - d["pitch"]:+.1f}°'
        elif d.get('surface'):
            if self.ctx.snap == 'surface':
                hit = self._mesh_hit(vp, pos)
                if hit is not None:
                    n = hit[1]
                    newp = hit[0] + n * (c.standoff_mm / 1000)
                    set_(x=round(float(newp[0]), 4), y=round(float(newp[1]), 4), z=round(float(newp[2]), 4))
                    dd = (newp - d['start']) * 1000
                    d['readout'] = f'on {hit[3]}  ·  Δx {dd[0]:+.0f}  Δy {dd[1]:+.0f}  Δz {dd[2]:+.0f} mm'
            else:
                g = vp.ground_at(pos.x(), pos.y(), d['start'][2])
                g0 = vp.ground_at(d['mouse'].x(), d['mouse'].y(), d['start'][2])
                if g is not None and g0 is not None:
                    newp = d['start'] + (g - g0) * fine
                    if snap:
                        newp[:2] = d['start'][:2] + np.round((newp[:2] - d['start'][:2]) / SNAP_M) * SNAP_M
                    elif self.ctx.snap == 'grid':
                        newp[:2] = np.round(newp[:2] / 0.05) * 0.05
                    set_(x=round(float(newp[0]), 4), y=round(float(newp[1]), 4))
                    dd = (newp - d['start']) * 1000
                    d['readout'] = f'Δx {dd[0]:+.0f}  Δy {dd[1]:+.0f} mm'
        self.changed.emit()

    def mouse_release(self, vp, e):
        if self.drag:
            self.drag = None
            self.ctx.store.dragging = False
            self.ctx.analysis.request()
            self.changed.emit()

    def cancel_drag(self) -> bool:
        """Esc during a handle drag: put the object back where the drag started, leaving no undo step."""
        if not self.drag:
            return False
        self.ctx.store.cancel_gesture()
        self.drag = None
        self.ctx.store.dragging = False
        self.ctx.analysis.request()
        self.changed.emit()
        return True

    def click(self, vp, e):
        if self.mode != 'scene' or self.cameras_override is not None:
            return
        pos = e.position()
        if self.ctx.tool == 'probe':
            g = vp.ground_at(pos.x(), pos.y())
            if g is not None and self.result is not None:
                try:
                    self.add_probe((g[0], g[1]))
                except Exception as ex:
                    print('probe:', ex)
            return
        st = self.ctx.store
        add = bool(e.modifiers() & (Qt.KeyboardModifier.ShiftModifier | Qt.KeyboardModifier.ControlModifier))
        c = self._pick_camera(vp, pos)
        if c is not None:
            st.select_cams([c.uid], add=add)
            return
        if self.project.variant.sensors:
            from .sensorview import pick as pick_sensor
            uid = pick_sensor(self.project, *vp.ray(pos.x(), pos.y()))
            if uid is not None:
                st.select_obj(('sensor', uid))
                return
        if self.project.rig:
            from .rigview import pick
            uid = pick(self.project, self.ctx.analysis.engine.vehicle_geom(self.project), *vp.ray(pos.x(), pos.y()))
            if uid is not None:
                st.select_obj(('rig', uid))
                return
        g = vp.ground_at(pos.x(), pos.y())
        if g is not None and self.ctx.overlays.get('targets'):
            for t in self.project.targets:
                w, h = t.size_m
                ca, sa = math.cos(math.radians(t.yaw)), math.sin(math.radians(t.yaw))
                u = (g[0] - t.x) * ca + (g[1] - t.y) * sa              # into the board's own axes
                v = -(g[0] - t.x) * sa + (g[1] - t.y) * ca
                if abs(u) < w / 2 and abs(v) < h / 2:
                    st.select_obj(('target', t.uid))
                    return
        if g is not None and self.hover_info and self.hover_info.get('cams') and self.ctx.tool == 'select':
            best = self.hover_info['cams'][0][0]
            st.select_cams([best.uid])
            return
        if not add:
            st.select_cams([])

    def double_click(self, vp, e):
        c = self._pick_camera(vp, e.position())
        if c is not None and self.ctx.window:
            self.ctx.window.frame_selection()

    def hover(self, vp, pos):
        hot = None
        tg = self._gizmo_target() if pos is not None and not self.drag else None
        if tg is not None and not getattr(tg[2], 'locked', False):
            hot = self._gizmo_pick(vp, pos, tg[2], self._rings(tg))
        if hot != self.gizmo_hot:
            self.gizmo_hot = hot
            if hot is None:
                vp.unsetCursor()
            else:
                vp.setCursor(Qt.CursorShape.SizeAllCursor if hot[0] != 'ring' else Qt.CursorShape.OpenHandCursor)
            self.changed.emit()
        if pos is None:
            self.hover_info = None
            self.place_preview = None
            self.ctx.cursor_world.emit(None)
            self.changed.emit()
            return
        if self.ctx.tool == 'place':
            hit = self._mesh_hit(vp, pos)
            self.place_preview = hit
            if hit is not None:
                self.ctx.cursor_world.emit(tuple(hit[0]))
            self.changed.emit()
            return
        g = vp.ground_at(pos.x(), pos.y())
        under = self._pick_camera(vp, pos)
        hov = under.uid if under is not None else self._label_at(pos)
        moved_pip = hov != getattr(self, 'pip_hover', None)
        self.pip_hover = hov
        self._cursor_g = (float(g[0]), float(g[1]), 0.0) if g is not None else None
        if self._pip_uid() is not None or moved_pip:
            self.changed.emit()
        res = self.result
        info = None
        if g is not None:
            self.ctx.cursor_world.emit((g[0], g[1], 0.0))
            if res is not None and self._pick_camera(vp, pos) is None:
                q = res.query(g[0], g[1])
                if q and q['in_roi']:
                    info = dict(q)
                    info['point'] = (g[0], g[1])
                    info['extra'] = self._hover_extra(res, g, q)
        else:
            self.ctx.cursor_world.emit(None)
        prev = self.hover_info
        self.hover_info = info
        if info or prev:
            self.changed.emit()

    def _hover_extra(self, res, g, q):
        rows = []
        cams = {c.uid: c for c in self.cameras()}
        for ce, dens in q['cams'][:4]:
            cam = cams.get(ce.uid)
            if cam is None:
                continue
            d = np.array([g[0], g[1], 0.0]) - cam.pos()
            rng = float(np.linalg.norm(d))
            R, W2O = camera_frames(cam)
            opt = W2O @ d
            off = math.degrees(math.atan2(math.hypot(opt[0], opt[1]), opt[2]))
            rows.append((cam, dens, rng, off))
        return rows

    def key_press(self, vp, e):
        if e.key() == Qt.Key.Key_Escape:
            if self.cancel_drag():
                vp.cancel_provider_drag()
                return True
            if self.measure:
                self.measure = []
                self.changed.emit()
                return True
            if self.ctx.tool != 'select':
                self.ctx.set_tool('select')
                return True
        return False

    def context_menu(self, vp, pos, gpos):
        if self.mode != 'scene' or not self.ctx.window:
            return
        c = self._pick_camera(vp, pos)
        if c is not None:
            if c.uid not in self.ctx.store.sel_cams:
                self.ctx.store.select_cams([c.uid])
            self.ctx.window.camera_menu(self.ctx.store.sel_cams, gpos)
            return
        g = vp.ground_at(pos.x(), pos.y())
        self.ctx.window.ground_menu(g, gpos)

    # drag & drop from the library
    def drag_enter(self, vp, e):
        return e.mimeData().hasFormat('application/x-vantage-lens')

    def drag_move(self, vp, e):
        hit = self._mesh_hit(vp, e.position())
        self.place_preview = hit
        self.changed.emit()

    def drop(self, vp, e):
        data = bytes(e.mimeData().data('application/x-vantage-lens')).decode()
        mid, lid = data.split('|')
        hit = self._mesh_hit(vp, e.position())
        self.place_preview = None
        if hit is None:
            self.ctx.msg('Drop the module onto the body.')
            return
        if not hit[2]:
            self.ctx.msg(f'{hit[3]} is a forbidden mount zone.')
            return
        ops.add_camera(self.ctx, hit[0], hit[1], mid, lid)
        e.acceptProposedAction()


def paint_ground_tooltip(p: QPainter, pos, info, project, cameras):
    """'Ground point' hover panel used by the 3D and plan views."""
    rows = info.get('extra') or []
    gx, gy = info['point']
    o = project.origin_offset()
    lines_h = 18
    W = 236
    H = 44 + lines_h * max(len(rows), 1) + 16
    stitched = info.get('stitched')
    if stitched is not None:
        H += 44
    x, y = pos.x() + 16, pos.y() + 14
    vw = p.device().width() / max(p.device().devicePixelRatioF(), 1) if p.device() else 2000
    vh = p.device().height() / max(p.device().devicePixelRatioF(), 1) if p.device() else 2000
    if x + W > vw - 4:
        x = pos.x() - W - 12
    if y + H > vh - 4:
        y = pos.y() - H - 10
    r = QRectF(x, y, W, H)
    p.setPen(QPen(THEME.color('strong'), 1))
    p.setBrush(THEME.color('raised'))
    p.drawRoundedRect(r, 4, 4)
    p.setFont(sans(12, QFont.Weight.Medium))
    p.setPen(THEME.color('ink'))
    p.drawText(QPointF(x + 8, y + 17), 'Ground point')
    p.setFont(mono(11))
    p.drawText(QRectF(x, y + 5, W - 8, 16), Qt.AlignmentFlag.AlignRight, f'{gx - o[0]:.3f}  {gy - o[1]:.3f} m')
    p.setFont(sans(10))
    p.setPen(THEME.color('muted'))
    yy = y + 34
    cols = [x + 8, x + 106, x + 164, x + 226]
    p.drawText(QPointF(cols[0], yy), 'cam')
    for txt, cx in (('px/cm', cols[1]), ('range m', cols[2]), ('off-axis °', cols[3])):
        p.drawText(QRectF(cx - 70, yy - 11, 70, 14), Qt.AlignmentFlag.AlignRight, txt)
    if not rows:
        p.setFont(sans(11))
        p.setPen(QColor(STATUS['critical']))
        p.drawText(QPointF(cols[0], yy + 17), 'Blind: no camera sees this point')
    for i, (cam, dens, rng, off) in enumerate(rows):
        yy2 = yy + 17 + i * lines_h
        paint_chip(p, cols[0], yy2 - 4, cam.slot, cam.tag, size=7)
        p.setFont(mono(11))
        p.setPen(THEME.color('ink'))
        for v, cx in ((f'{dens:.1f}', cols[1]), (f'{rng:.2f}', cols[2]), (f'{off:.1f}', cols[3])):
            p.drawText(QRectF(cx - 60, yy2 - 12, 60, 16), Qt.AlignmentFlag.AlignRight, v)
    if stitched is not None:
        _paint_stitched_rows(p, x, y + H - 60, W, stitched)
    if rows:
        p.setFont(sans(10))
        p.setPen(THEME.color('muted'))
        p.drawText(QPointF(x + 8, y + H - 6), f'Click selects {rows[0][0].tag} · best density')


def _paint_stitched_rows(p: QPainter, x, y, W, stitched):
    """Ground tooltip, Objects heatmap: how much of the object the stitched view shows here, and by which cameras."""
    (shown, raw, by), sres, cams = stitched
    st = sres.settings
    p.setPen(QPen(THEME.color('strong'), 1))
    p.drawLine(QPointF(x + 8, y), QPointF(x + W - 8, y))
    p.setFont(sans(11))
    if raw < st.threshold:
        txt, col = f'{st.label}: no camera sees it', QColor(STATUS['critical'])
    elif shown < st.threshold:
        txt, col = f'{st.label}: {shown * 100:.0f} % shown · lost', QColor(STATUS['serious'])
    else:
        txt, col = f'{st.label}: {shown * 100:.0f} % shown', THEME.color('ink')
    p.setPen(col)
    p.drawText(QPointF(x + 8, y + 16), txt)
    byc = {c.uid: c for c in cams}
    xx = x + 8
    p.setFont(mono(11))
    for uid, f in sorted(by.items(), key=lambda t: -t[1])[:4]:
        c = byc.get(uid)
        if c is None:
            continue
        end = paint_chip(p, xx, y + 30, c.slot, c.tag, size=7)
        p.setFont(mono(11))
        p.setPen(THEME.color('ink'))
        txt = f'{f * 100:.0f} %'
        p.drawText(QPointF(end + 5, y + 34), txt)
        xx = end + 5 + p.fontMetrics().horizontalAdvance(txt) + 12
    if not by:
        p.setPen(THEME.color('muted'))
        p.setFont(sans(10))
        p.drawText(QPointF(x + 8, y + 34), 'no camera\'s smear lands in its own sector')
