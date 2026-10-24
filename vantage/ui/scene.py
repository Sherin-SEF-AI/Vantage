"""Scene provider for the main 3D viewport (and the bowl view): what to draw and how tools behave."""
from __future__ import annotations

import math

import numpy as np
from PyQt6.QtCore import QObject, pyqtSignal, Qt, QPointF, QRectF, QLineF
from PyQt6.QtGui import QColor, QPen, QPainter, QFont, QPolygonF, QBrush, QPainterPath

from ..core.analysis import surface_frame, camera_frames
from ..core.lens import resolve
from ..core.mathutil import rot_zyx, BODY_TO_OPTICAL, wrap180, yaw_pitch_from_dir
from ..core.mesh import ROLE_GLASS, ROLE_WHEELS, ROLE_IGNORED
from . import guides, maps, ops, icons, overlays3d as O3
from .theme import THEME, mono, sans, STATUS
from .viewport3d import MeshDraw, GroundDraw, LineDraw, TriDraw
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
        elif r == ROLE_WHEELS:
            base[m] = tyre
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
        return Style(shadows=lit and self.mode == 'scene', ao=lit and self.mode == 'scene',
                     ground_plane=self.mode == 'scene' and not photo and lit,
                     ground_rgb=tuple(bg * 0.9 + ink * 0.1), sky_rgb=(0.62, 0.66, 0.72) if dark else (0.86, 0.88, 0.91),
                     bounce_rgb=(0.24, 0.23, 0.22) if dark else (0.50, 0.48, 0.45))

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
        if self.mode == 'scene':
            ov = self.ctx.overlays
            if ov.get('brackets', True):
                out += O3.bracket_draws(self.ctx, gl)
            if ov.get('sensors', True) and p.variant.sensors:
                from .sensorview import radar_draws
                late += radar_draws(self.ctx, gl, vp=vp)
            if ov.get('gnsssky') and p.variant.sensors:
                late += self._ov3d().dome_draws(gl)[0]
        out += guides.scene_tris(self, vp)
        return out + late

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
    def lines(self, vp):
        out = []
        p = self.project
        ov = self.ctx.overlays
        res = self.result
        textured = self.bowl_draw(vp) is not None
        if ov.get('grid', True) and not textured:
            out += self._grid()
        if res is not None:
            if ov.get('roi', True) and not textured:
                for b in res.grid.boundary:
                    pts = np.c_[b, np.full(len(b), 0.004)]
                    out.append(LineDraw(pts, THEME.color('ink', 0.9), 1.2, dash=True, dash_len=0.09))
            if ov.get('footprints', True) and self.mode == 'scene':
                for c in res.cams:
                    if self.ctx.isolate and c.uid not in self.ctx.isolate:
                        continue
                    if len(c.segments):
                        sel = c.uid in self.ctx.store.sel_cams
                        segs = np.concatenate([c.segments, np.full(c.segments.shape[:2] + (1,), 0.005)], axis=2)
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
                        out.append(LineDraw(pts[idx[0]:idx[-1] + 1], THEME.color('ink', 0.75), 1.0, dash=True, dash_len=0.06))
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
            out.append(LineDraw(ring, THEME.color('ink2'), 1.0))
            out.append(LineDraw(np.array([[ob.x, ob.y, 0], [ob.x, ob.y, ob.height]]), THEME.color('ink2'), 1.5))
            out.append(LineDraw(ring + np.array([0, 0, ob.height]), THEME.color('ink2'), 1.0))
        if ov.get('dimensions') and p.vehicle.mesh is not None and self.mode == 'scene':
            out += self._dimensions()
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
        segs = []
        for k in sel:
            P = np.array([g.xs[ii[k]], g.ys[jj[k]], 0.0])
            segs.append([c.pos(), P])
        return [LineDraw(np.array(segs), _qc(STATUS['critical'], 0.55), 1.0, segments=True)]

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

    def overlay(self, p: QPainter, vp):
        if self.project.vehicle.mesh is None and self.mode == 'scene':
            p.setFont(sans(12, QFont.Weight.DemiBold))
            p.setPen(THEME.color('ink'))
            p.drawText(QPointF(12, 44), 'No vehicle loaded')
            p.setFont(sans(12))
            p.setPen(THEME.color('ink2'))
            p.drawText(QPointF(12, 64), 'File ▸ Import ▸ Vehicle model…   Ctrl+I')
            p.drawText(QPointF(12, 82), 'or drop .glb .gltf .obj .stl .fbx .ply .step on this view')
            p.drawText(QPointF(12, 100), 'or File ▸ New from template')
        sel = set(self.ctx.store.sel_cams)
        cams = [c for c in self.cameras() if c.visible and not (self.ctx.isolate and c.uid not in self.ctx.isolate)]
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
                    p.setFont(mono(11))
                    p.setPen(THEME.color('ink') if not s or s.ok else QColor(STATUS['critical']))
                    p.drawText(QPointF(q[0][0] + 10, q[0][1] - 10), t.name)
        if self.ctx.overlays.get('dimensions') and hasattr(self, '_dim_labels') and self.mode == 'scene':
            p.setFont(mono(11))
            p.setPen(THEME.color('ink'))
            for x, y, z, txt in self._dim_labels:
                q, _, ok = vp.project([[x, y, z]])
                if ok[0]:
                    p.drawText(QPointF(q[0][0] - 30, q[0][1] + 4), txt)
        # place preview
        if self.ctx.tool == 'place' and self.place_preview is not None:
            self._paint_disc(p, vp, *self.place_preview)
        if self.measure:
            self._paint_measure(p, vp)
        self._paint_ov3d(p, vp)
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
