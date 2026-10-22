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
from . import maps, ops, icons
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
        self.d[key] = (rim, rays, axis)
        if len(self.d) > 200:
            self.d.clear()
        return self.d[key]


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
        self.result_override = None         # pinned results (tolerance worst case)
        self.cameras_override = None
        self.show_tooltip = True
        self.photo_frame = None
        self.label_prefix = ''
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
                src = vg.occ_mesh if vg and m.n_tris > 12000 else m
            elif not use_full:
                vg = self.ctx.analysis.engine.vehicle_geom(self.project)
                src = vg.occ_mesh
            base, alpha, keep = mesh_colors(src, sh)
            self._mesh = MeshDraw(src.tri[keep].astype(np.float32), src.normals[keep], base[keep], alpha[keep],
                                  version=id(src), wire=(sh == 'wire'))
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
        key = (id(res), mode, THEME.name, iso)
        if key != self._ground_key:
            img = maps.make(res, mode, k=4, isolate=iso)
            self._ground = GroundDraw(maps.to_qimage(img), maps.extent(res), 0.002,
                                      nearest=mode in ('density', 'redundancy', 'coverage'), version=hash(key))
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

    def tris_extra(self, vp):
        out = []
        p = self.project
        # camera mount blocks
        boxes, cols = [], []
        sel = set(self.ctx.store.sel_cams)
        for c in self.cameras():
            if not c.visible or (self.ctx.isolate and c.uid not in self.ctx.isolate):
                continue
            R = rot_zyx(c.yaw, c.pitch, c.roll)
            s = np.array([0.045, 0.04, 0.04])
            corners = np.array([[x, y, z] for x in (-1, 1) for y in (-1, 1) for z in (-1, 1)]) * s / 2
            corners[:, 0] -= 0.02
            V = c.pos() + corners @ R.T
            F = [[0, 1, 3], [0, 3, 2], [4, 6, 7], [4, 7, 5], [0, 4, 5], [0, 5, 1], [2, 3, 7], [2, 7, 6],
                 [0, 2, 6], [0, 6, 4], [1, 5, 7], [1, 7, 3]]
            body = np.array(QColor(THEME.tok['body']).getRgbF()[:3]) * (1.05 if c.uid in sel else 0.95)
            for f in F:
                boxes.append(V[f])
                cols.append([*np.clip(body, 0, 1), 1.0])
        if boxes:
            out.append(TriDraw(np.array(boxes), np.array(cols)))
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
        if self.mode == 'bowl' and self.result is not None:
            if self.bowl_draw(vp) is None:
                out.append(self._bowl())
            elif not getattr(vp, 'gl_bowl', False):
                out.append(self._bowl_soft())
        return out

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
        heat = {'none': '', 'coverage': ' · coverage', 'density': ' · density', 'redundancy': ' · redundancy'}[self.ctx.heatmap]
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
        c = self.ctx.store.selected
        if c is not None and self.mode == 'scene' and self.cameras_override is None:
            if self.ctx.tool == 'move' or (self.ctx.tool == 'select' and c is not None):
                self._gizmo_move(p, vp, c, faint=self.ctx.tool != 'move')
            elif self.ctx.tool == 'rotate':
                self._gizmo_rotate(p, vp, c)
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
        if self.show_tooltip and self.hover_info and self.ctx.tool in ('select', 'move') and self.drag is None:
            self._paint_tooltip(p, vp)

    def _gizmo_axes(self, vp, c):
        pos = c.pos()
        L = 64.0 / max(vp.px_per_m(pos), 1e-3)
        return pos, L, [np.array([1.0, 0, 0]), np.array([0, 1.0, 0]), np.array([0, 0, 1.0])]

    def _gizmo_move(self, p, vp, c, faint=False):
        pos, L, axes = self._gizmo_axes(vp, c)
        cols = [QColor('#e0533f'), QColor('#3fae49'), QColor('#3d86e6')]
        o, _, ok = vp.project([pos])
        if not ok[0]:
            return
        for k, a in enumerate(axes):
            e, _, ok2 = vp.project([pos + a * L])
            if not ok2[0]:
                continue
            col = QColor(cols[k])
            if faint:
                col.setAlphaF(0.55)
            hot = self.drag and self.drag.get('axis') == k
            p.setPen(QPen(col, 2.2 if hot else 1.6))
            p.drawLine(QPointF(*o[0]), QPointF(*e[0]))
            d = e[0] - o[0]
            n = np.linalg.norm(d)
            if n > 1:
                d /= n
                perp = np.array([-d[1], d[0]])
                tip = e[0]
                poly = QPolygonF([QPointF(*tip), QPointF(*(tip - d * 8 + perp * 3.5)), QPointF(*(tip - d * 8 - perp * 3.5))])
                p.setBrush(col)
                p.setPen(Qt.PenStyle.NoPen)
                p.drawPolygon(poly)

    def _ring(self, vp, c, which):
        pos = c.pos()
        L = 56.0 / max(vp.px_per_m(pos), 1e-3)
        a = np.linspace(0, 2 * math.pi, 64)
        if which == 0:      # yaw ring around world z
            pts = pos + np.c_[np.cos(a), np.sin(a), np.zeros_like(a)] * L
        else:               # pitch ring around the camera's lateral axis
            R = rot_zyx(c.yaw, 0, 0)
            fwd, up = R[:, 0], R[:, 2]
            pts = pos + (np.outer(np.cos(a), fwd) + np.outer(np.sin(a), up)) * L
        return pts

    def _gizmo_rotate(self, p, vp, c):
        cols = [QColor('#3d86e6'), QColor('#3fae49')]
        for k in (0, 1):
            pts = self._ring(vp, c, k)
            sc, _, ok = vp.project(pts)
            hot = self.drag and self.drag.get('ring') == k
            p.setPen(QPen(cols[k], 2.2 if hot else 1.5))
            p.setBrush(Qt.BrushStyle.NoBrush)
            p.drawPolyline(QPolygonF([QPointF(*q) for q, o in zip(sc, ok) if o]))
        # aim direction
        R = rot_zyx(c.yaw, c.pitch, c.roll)
        pos = c.pos()
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

    def _gizmo_pick(self, vp, pos, c):
        if self.ctx.tool == 'rotate':
            for k in (0, 1):
                sc, _, ok = vp.project(self._ring(vp, c, k))
                d = np.hypot(sc[:, 0] - pos.x(), sc[:, 1] - pos.y())
                if ok.any() and d[ok].min() < 6:
                    return ('ring', k)
            return None
        if self.ctx.tool not in ('move', 'select'):
            return None
        p0, L, axes = self._gizmo_axes(vp, c)
        o, _, ok = vp.project([p0])
        best = None
        for k, a in enumerate(axes):
            e, _, ok2 = vp.project([p0 + a * L])
            if not (ok[0] and ok2[0]):
                continue
            A, B = o[0], e[0]
            AB = B - A
            t = np.clip(((pos.x() - A[0]) * AB[0] + (pos.y() - A[1]) * AB[1]) / max(AB @ AB, 1e-9), 0, 1)
            q = A + AB * t
            dist = math.hypot(pos.x() - q[0], pos.y() - q[1])
            if dist < 6 and t > 0.25 and (best is None or dist < best[1]):
                best = (k, dist)
        return ('axis', best[0]) if best else None

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
        if c is not None and tool in ('move', 'rotate', 'select'):
            g = self._gizmo_pick(vp, pos, c)
            if g and not c.locked:
                if g[0] == 'axis':
                    self.drag = {'axis': g[1], 'uid': c.uid, 'start': c.pos(), 'mouse': pos}
                else:
                    self.drag = {'ring': g[1], 'uid': c.uid, 'yaw': c.yaw, 'pitch': c.pitch, 'mouse': pos, 'turn': 0.0,
                                 'last': None}
                st.dragging = True
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
                self.drag = {'surface': True, 'uid': hitc.uid, 'start': hitc.pos(), 'mouse': pos}
                st.dragging = True
            return True
        return False

    def mouse_move(self, vp, e):
        d = self.drag
        if not d:
            return
        st = self.ctx.store
        c = st.cam(d['uid'])
        if c is None:
            return
        pos = e.position()
        fine = 0.1 if e.modifiers() & Qt.KeyboardModifier.ShiftModifier else 1.0
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
            if self.ctx.snap == 'grid':
                delta = round(delta / 0.05) * 0.05
            new = d['start'] + a * delta
            st.set_cam(c.uid, merge=True, x=round(float(new[0]), 4), y=round(float(new[1]), 4), z=round(float(new[2]), 4))
        elif 'ring' in d:
            pc, _, _ = vp.project([c.pos()])
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
                st.set_cam(c.uid, merge=True, yaw=round(wrap180(d['yaw'] + sign * da), 1))
            else:
                st.set_cam(c.uid, merge=True, pitch=round(max(-89.0, min(89.0, d['pitch'] - da)), 1))
        elif d.get('surface'):
            if self.ctx.snap == 'surface':
                hit = self._mesh_hit(vp, pos)
                if hit is not None:
                    n = hit[1]
                    newp = hit[0] + n * (c.standoff_mm / 1000)
                    st.set_cam(c.uid, merge=True, x=round(float(newp[0]), 4), y=round(float(newp[1]), 4), z=round(float(newp[2]), 4))
            else:
                g = vp.ground_at(pos.x(), pos.y(), d['start'][2])
                g0 = vp.ground_at(d['mouse'].x(), d['mouse'].y(), d['start'][2])
                if g is not None and g0 is not None:
                    newp = d['start'] + (g - g0) * fine
                    if self.ctx.snap == 'grid':
                        newp[:2] = np.round(newp[:2] / 0.05) * 0.05
                    st.set_cam(c.uid, merge=True, x=round(float(newp[0]), 4), y=round(float(newp[1]), 4))
        self.changed.emit()

    def mouse_release(self, vp, e):
        if self.drag:
            self.drag = None
            self.ctx.store.dragging = False
            self.ctx.analysis.request()
            self.changed.emit()

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
    if rows:
        p.setFont(sans(10))
        p.setPen(THEME.color('muted'))
        p.drawText(QPointF(x + 8, y + H - 6), f'Click selects {rows[0][0].tag} · best density')
