"""The cable harness in the UI: the controller that keeps a routed harness for the project (worked out on a thread
after every edit while the harness page shows it or the 3D overlay asks for it), the page's view of the routes on the
body (from above, from the side, or orbiting in 3D: the body software-rendered with a z-buffer, core.meshing.render,
the routes drawn over it, dashed where they run inside or behind the body) and the drawings the main views can add
(scene_draws for the 3D viewport, paint_plan for the plan view)."""
from __future__ import annotations

import json
import math
import threading
import traceback

import numpy as np
from PyQt6.QtCore import QObject, QPointF, QRectF, Qt, QTimer, pyqtSignal
from PyQt6.QtGui import QColor, QFont, QImage, QPainter, QPen, QPolygonF
from PyQt6.QtWidgets import QWidget

from ..core import harness as HN
from ..core import meshing as M
from .theme import THEME, STATUS, mono, sans

HOME_3D = (-55.0, 28.0)                 # azimuth, elevation of the 3D view (degrees)
VIEWS = {'top': (-90.0, 90.0), 'side': (-90.0, 0.0)}      # forward to the right in both
PICK_PX = 7.0
FAST_TRIS = 2500                        # the body while orbiting


class _Sig(QObject):
    done = pyqtSignal(int, object)
    failed = pyqtSignal(int, str)


def controller(ctx) -> 'HarnessController':
    c = getattr(ctx, 'harness_ctl', None)
    if c is None:
        c = ctx.harness_ctl = HarnessController(ctx)
    return c


def inputs_key(project) -> str:
    """Everything the harness reads, as text: a result made from other inputs is stale."""
    m = project.vehicle.mesh
    v = project.variant
    return json.dumps({'mesh': [id(m), getattr(m, 'n_tris', 0), id(getattr(m, 'face_zone', None))],
                       'parts': [p.to_dict() for p in m.parts] if m is not None else [],
                       'zones': [z.to_dict() for z in m.zones] if m is not None else [],
                       'variant': v.to_dict(), 'rig': [r.to_dict() for r in project.rig], 'ecu': project.ecu.to_dict(),
                       'harness': project.analysis.harness, 'power': project.analysis.power,
                       'library': project.section('library')}, sort_keys=True, default=str)


def display_mesh(project, res: HN.HarnessResult | None):
    """What the view draws: {'mesh': the routing surface's closed body (simplified as the router's) with the visible
    rig parts (a core.mesh.Mesh, for the hidden-route tests), 'full': (V, F, class per face: 0 body, 1 glass,
    2 wheels, 3 rig), 'fast': the same coarser, for orbiting}; None without a body."""
    from ..core.mesh import ROLE_IGNORED, ROLE_WHEELS
    from ..core.rig import merge, rig_mesh
    m = project.vehicle.mesh
    if m is None or not m.n_tris:
        return None
    g = HN.route_graph(project, res.settings if res is not None else None)
    body = g.solid if g is not None and g.solid is not None else m
    full = merge(body, rig_mesh([r for r in project.rig if r.visible]))

    def arrays(mesh):
        parts = mesh.parts
        keep = np.array([p.role != ROLE_IGNORED or p.material == 'rig' for p in parts] or [True])[mesh.face_part]
        fk = HN._face_kinds(mesh)
        cls = np.zeros(mesh.n_tris, np.uint8)
        cls[(fk & HN.GLASS) > 0] = 1
        cls[np.array([p.role == ROLE_WHEELS for p in parts] or [False])[mesh.face_part]] = 2
        cls[np.array([p.material == 'rig' for p in parts] or [False])[mesh.face_part]] = 3
        return mesh.vertices, mesh.faces[keep], cls[keep]

    out = {'mesh': full, 'full': arrays(full)}
    out['fast'] = arrays(full.decimate(FAST_TRIS)[0]) if full.n_tris > FAST_TRIS * 1.5 else out['full']
    return out


class HarnessController(QObject):
    """The project's routed harness: worked out on a thread after edits (debounced) while the page shows it or an
    overlay asks for it; quick once the routing surface is built (cached per body, rig and costs)."""
    changed = pyqtSignal()

    def __init__(self, ctx):
        super().__init__()
        self.ctx = ctx
        self.result: HN.HarnessResult | None = None
        self.display = None              # display_mesh() for the result's project
        self.stale = False
        self.busy = False
        self.status = ''
        self.active = False
        self.overlay = False
        self._gen = 0
        self._for = None                 # (project, inputs key) of the result
        self.sig = _Sig()
        self.sig.done.connect(self._done)
        self.sig.failed.connect(self._failed)
        self.timer = QTimer(self)
        self.timer.setSingleShot(True)
        self.timer.setInterval(200)
        self.timer.timeout.connect(self.run)
        ctx.store.changed.connect(self._on_edit)
        ctx.store.project_replaced.connect(self._reset)
        ctx.overlays_changed.connect(self._kick)        # View ▸ Overlays ▸ Cable harness switched on

    # ---- settings (saved with the project; every edit is one undo step)
    @property
    def settings(self) -> HN.HarnessSettings:
        return HN.settings_of(self.ctx.store.project)

    def store(self, st: HN.HarnessSettings, text) -> bool:
        new = HN.HarnessSettings.from_dict(st.to_dict()).to_dict()
        if new == self.settings.to_dict():
            return False

        def fn(p):
            p.analysis.harness = new
        self.ctx.store.edit('analysis', fn, f'Cable harness · {text}')
        return True

    def set(self, text, **kw) -> bool:
        st = self.settings
        for k, v in kw.items():
            setattr(st, k, v)
        return self.store(st, text)

    def defaults(self) -> bool:
        if not self.ctx.store.project.analysis.harness:
            return False

        def fn(p):
            p.analysis.harness = {}
        self.ctx.store.edit('analysis', fn, 'Cable harness · defaults')
        return True

    # ---- state
    def wanted(self) -> bool:
        return self.active or self.overlay or bool(self.ctx.overlays.get('harness'))

    def set_active(self, on: bool):
        self.active = on
        self._kick()

    def want_overlay(self, on: bool):
        """The main views draw the routes (View ▸ Overlays): keep the result up to date while they do."""
        self.overlay = on
        self._kick()

    def _kick(self, *_):
        if self.wanted() and (self.result is None or self.stale or self._for is None or
                              self._for[0] is not self.ctx.store.project):
            self.timer.start()

    def fresh(self) -> bool:
        return (self.result is not None and not self.stale and self._for is not None and
                self._for[0] is self.ctx.store.project)

    def _on_edit(self, *_):
        if self.result is None and not self.wanted():
            return
        if self._for is None or inputs_key(self.ctx.store.project) != self._for[1]:
            if not self.stale and self.result is not None:
                self.stale = True
                self.changed.emit()
            if self.wanted():
                self.timer.start()
        elif self.stale:                       # an undo back to the inputs of the result
            self.stale = False
            self.changed.emit()

    def _reset(self, *_):
        self._gen += 1
        self.result, self.display, self._for, self.stale, self.busy, self.status = None, None, None, False, False, ''
        self.changed.emit()
        if self.wanted():
            self.timer.start()

    def run(self) -> bool:
        p = self.ctx.store.project
        project = p.snapshot()
        key = (p, inputs_key(p))
        self._gen += 1
        gen = self._gen
        self.busy = True
        if self.result is None:
            self.status = 'Routing the cables…'
        self.changed.emit()

        def thread():
            try:
                res = HN.run(project, cancel=lambda: gen != self._gen)
                disp = display_mesh(project, res) if res is not None else None
                self.sig.done.emit(gen, (key, res, disp))
            except RuntimeError:            # the window closed while routing
                pass
            except Exception as e:
                try:
                    self.sig.failed.emit(gen, f'{type(e).__name__}: {e}\n{traceback.format_exc()}')
                except RuntimeError:
                    pass
        threading.Thread(target=thread, daemon=True).start()
        return True

    def _done(self, gen, out):
        if gen != self._gen:
            return
        self.busy = False
        key, res, disp = out
        if res is None or key[0] is not self.ctx.store.project:
            self.changed.emit()
            return
        self.result, self.display, self._for = res, disp, key
        self.stale = key[1] != inputs_key(self.ctx.store.project)
        t = res.totals
        g = res.metrics.get('graph') or {}
        self.status = (f'{t["runs"]} cables · {t["length_m"]:.1f} m · routed on {g.get("vertices", 0):,} surface '
                       f'points in {res.metrics["seconds"]:.2f} s'.replace(',', ' ') if res.routed else
                       f'{t["runs"]} cables · {t["length_m"]:.1f} m · Manhattan estimate (no body)')
        self.changed.emit()
        if self.ctx.overlays.get('harness'):            # the main views draw the routes: repaint them
            self.ctx.overlays_changed.emit()
        if self.stale and self.wanted():
            self.timer.start()

    def _failed(self, gen, msg):
        if gen != self._gen:
            return
        self.busy = False
        self.status = 'Failed: ' + msg.splitlines()[0]
        self.changed.emit()


# =========================================================================== colours
def run_qcolor(r, alpha=None) -> QColor:
    """A camera's colour slot; a sensor its kind's colour (as the plan and 3D views draw it)."""
    c = THEME.cam(r['slot']) if r['kind'] == 'camera' else QColor(HN.KIND_COLORS.get(r['kind'], '#888888'))
    if alpha is not None:
        c.setAlphaF(alpha)
    return c


def _rgb(c: QColor):
    return (c.red(), c.green(), c.blue())


# =========================================================================== the page's view
class HarnessView(QWidget):
    """The routes on the body, orthographic: 'top' (forward to the right), 'side' (from the right, forward to the
    right) or '3d' (drag to orbit). Wheel to zoom, right-drag to pan, double-click to reset. Hover a route for its
    figures; click it to select its run (run_clicked), an entry point to select it (entry_clicked)."""
    run_clicked = pyqtSignal(str)
    entry_clicked = pyqtSignal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumSize(320, 220)
        self.setMouseTracking(True)
        self.res: HN.HarnessResult | None = None
        self.display = None
        self.message = 'Routing the cables…'
        self.mode = 'top'
        self.yaw, self.elev = HOME_3D
        self.zoom = 1.0
        self.pan = np.zeros(2)
        self.selected = ''               # run id
        self.selected_entry = ''
        self.hover = None                # ('run', id) | ('entry', uid) | None
        self._img = None
        self._key = None
        self._hid = {}                   # (path id, view direction) -> hidden flags
        self._drag = None
        self._dragging = False
        self._screen = {}                # run id -> list of (QPolygonF parts, hidden flags), for picking
        self._settle = QTimer(self)
        self._settle.setSingleShot(True)
        self._settle.setInterval(160)
        self._settle.timeout.connect(self._refine)
        THEME.changed.connect(self._retheme)

    def _retheme(self, *_):
        self._key = None
        self.update()

    # ---- content
    def set_result(self, res, display, message=''):
        if display is not self.display:
            self._key = None
        if res is not self.res:
            self._hid = {}
        self.res, self.display = res, display
        self.message = message
        self.update()

    def set_mode(self, mode):
        if mode in ('top', 'side', '3d') and mode != self.mode:
            self.mode = mode
            self.zoom, self.pan = 1.0, np.zeros(2)
            self._key = None
            self.update()

    def set_selected(self, rid='', entry=''):
        self.selected, self.selected_entry = rid or '', entry or ''
        self.update()

    def reset_view(self):
        self.yaw, self.elev = HOME_3D
        self.zoom, self.pan = 1.0, np.zeros(2)
        self._key = None
        self.update()

    # ---- projection
    def _angles(self):
        return VIEWS.get(self.mode, (self.yaw, self.elev))

    def _frame(self, W, H):
        """(basis, scale px/m, centre) for the current view."""
        B = M.view_basis(*self._angles())
        pts = []
        if self.display is not None:
            pts.append(self.display['full'][0])
        if self.res is not None:
            pts += [r['path'] for r in self.res.runs]
            pts.append(np.array([self.res.compute['pos']]))
        if not pts:
            return B, 100.0, np.zeros(3)
        P = np.vstack([np.asarray(q, float).reshape(-1, 3) for q in pts if len(q)])
        lo, hi = P.min(axis=0), P.max(axis=0)
        centre = (lo + hi) / 2
        q = (P - centre) @ B.T
        ext = np.abs(q[:, :2]).max(axis=0) * 2 + 0.2
        scale = min((W - 40) / max(ext[0], 0.1), (H - 50) / max(ext[1], 0.1)) * self.zoom
        centre = centre - (B[0] * self.pan[0] - B[1] * self.pan[1]) / scale
        return B, scale, centre

    def _project(self, P, B, scale, centre, W, H):
        q = (np.asarray(P, float) - centre) @ B.T
        return np.stack([W / 2 + q[..., 0] * scale, H / 2 - q[..., 1] * scale], -1), -q[..., 2]

    # ---- body image and hidden routes
    def _render(self, W, H, B, scale, centre, fast):
        """The body drawn far to near (painter's algorithm), flat shaded, into a cached image."""
        V, F, cls = self.display['fast' if fast else 'full']
        img = QImage(W, H, QImage.Format.Format_RGB32)
        img.fill(THEME.color('viewport'))
        q = (np.asarray(V, float) - centre) @ B.T
        xy = np.stack([W / 2 + q[:, 0] * scale, H / 2 - q[:, 1] * scale], -1)
        tri = q[F]
        n = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
        nn = np.linalg.norm(n, axis=1)
        keep = (n[:, 2] > 0) & (nn > 0)                 # faces turned to the viewer
        T = xy[F[keep]]
        on = (T[:, :, 0].max(1) >= 0) & (T[:, :, 0].min(1) < W) & (T[:, :, 1].max(1) >= 0) & (T[:, :, 1].min(1) < H)
        idx = np.nonzero(keep)[0][on]
        order = idx[np.argsort(tri[idx, :, 2].mean(axis=1))]          # far first
        light = np.array([0.35, 0.55, 0.75])
        light /= np.linalg.norm(light)
        lam = 0.55 + 0.45 * np.clip((n[order] / nn[order, None]) @ light, 0, 1)
        dark = THEME.color('viewport').lightness() < 128
        tones = np.array([(176, 178, 182) if dark else (206, 207, 210), (92, 104, 120) if dark else (150, 160, 172),
                          (58, 58, 60) if dark else (96, 96, 98), (118, 122, 128) if dark else (150, 154, 160)], float)
        rgb = np.clip(tones[cls[order]] * lam[:, None], 0, 255).astype(int)
        p = QPainter(img)
        p.setRenderHint(QPainter.RenderHint.Antialiasing, False)
        P = xy[F[order]].tolist()
        for (a, b, c), (r, g, bl) in zip(P, rgb.tolist()):
            col = QColor(r, g, bl)
            p.setPen(QPen(col, 1.0))
            p.setBrush(col)
            p.drawPolygon(QPolygonF([QPointF(*a), QPointF(*b), QPointF(*c)]))
        p.end()
        self._img = img

    def _hidden(self, P, B):
        """Per point of the polyline P (n, 3): hidden from the viewer behind the body or the rig (a ray from the
        point towards the viewer meets them more than 2 cm away). Cached per route and view; none while orbiting."""
        if self._dragging or self.display is None or len(P) == 0:
            return np.zeros(len(P), bool)
        key = (id(P), tuple(np.round(B[2], 5)))
        hit = self._hid.get(key)
        if hit is not None:
            return hit
        mesh = self.display['mesh']
        d = np.broadcast_to(B[2], (len(P), 3))
        t, f = mesh.raycast(np.asarray(P, float) + d * 0.004, d)
        out = (f >= 0) & (t > 0.02)
        if len(self._hid) > 400:
            self._hid.clear()
        self._hid[key] = out
        return out

    # ---- painting
    def paintEvent(self, e):
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        W, H = max(self.width(), 2), max(self.height(), 2)
        p.fillRect(self.rect(), THEME.color('viewport'))
        if self.res is None and self.display is None:
            p.setFont(sans(12))
            p.setPen(THEME.color('muted'))
            p.drawText(QRectF(20, 0, W - 40, H), Qt.AlignmentFlag.AlignCenter | Qt.TextFlag.TextWordWrap,
                       self.message or 'No harness')
            p.end()
            return
        B, scale, centre = self._frame(W, H)
        if self.display is not None:
            key = (W, H, self.mode, round(self.yaw, 2), round(self.elev, 2), round(self.zoom, 4),
                   tuple(np.round(self.pan, 1)), self._dragging, id(self.display), THEME.name)
            if key != self._key or self._img is None:
                self._render(W, H, B, scale, centre, self._dragging)
                self._key = key
            p.drawImage(0, 0, self._img)
        self._screen = {}
        if self.res is not None:
            self._paint_routes(p, B, scale, centre, W, H)
            self._paint_marks(p, B, scale, centre, W, H)
        self._paint_hud(p, W, H)
        p.end()

    def _paint_routes(self, p, B, scale, centre, W, H):
        runs = [r for r in self.res.runs if r['what'] != 'power']
        order = sorted(runs, key=lambda r: (r['id'] == self.selected, self.hover == ('run', r['id'])))
        for r in order:
            xy, _ = self._project(r['path'], B, scale, centre, W, H)
            hid = self._hidden(r['path'], B)
            span = r['inside_span']
            sel = r['id'] == self.selected or self.hover == ('run', r['id'])
            dim = bool(self.selected) and not sel
            col = run_qcolor(r, 0.35 if dim else 1.0)
            width = 3.2 if sel else 2.0
            segs = []
            for i in range(len(xy) - 1):
                inside = span is not None and span[0] <= i < span[1]
                segs.append((QPointF(*xy[i]), QPointF(*xy[i + 1]), inside or (hid[i] and hid[i + 1])))
            if sel:                                   # a halo under the selected route
                p.setPen(QPen(THEME.color('viewport', 0.85), width + 3, Qt.PenStyle.SolidLine,
                              Qt.PenCapStyle.RoundCap, Qt.PenJoinStyle.RoundJoin))
                for a, b, _ in segs:
                    p.drawLine(a, b)
            for a, b, dashed in segs:
                pen = QPen(col, width if not dashed else max(width - 0.6, 1.2), Qt.PenStyle.SolidLine,
                           Qt.PenCapStyle.RoundCap, Qt.PenJoinStyle.RoundJoin)
                if dashed:
                    pen.setStyle(Qt.PenStyle.CustomDashLine)
                    pen.setDashPattern([3, 2.5])
                p.setPen(pen)
                p.drawLine(a, b)
            self._screen[r['id']] = xy
            p.setPen(Qt.PenStyle.NoPen)
            p.setBrush(col)
            p.drawEllipse(QPointF(*xy[0]), 3.5 if sel else 2.6, 3.5 if sel else 2.6)
            if sel or len(runs) <= 24:
                p.setFont(mono(10, QFont.Weight.DemiBold))
                p.setPen(col)
                p.drawText(QPointF(xy[0][0] + 5, xy[0][1] - 5), r['tag'])

    def _paint_marks(self, p, B, scale, centre, W, H):
        res = self.res
        p.setFont(sans(10))
        for e in res.entries:
            xy, depth = self._project(np.array([e['pos']]), B, scale, centre, W, H)
            q = QPointF(*xy[0])
            sel = e['uid'] == self.selected_entry or self.hover == ('entry', e['uid'])
            col = THEME.color('ink' if e['enabled'] else 'muted')
            p.setBrush(THEME.color('viewport', 0.9))
            p.setPen(QPen(QColor(STATUS['warning']) if sel else col, 2.2 if sel else 1.4))
            p.drawEllipse(q, 5.5 if sel else 4.5, 5.5 if sel else 4.5)
            if not e['enabled']:
                p.drawLine(QPointF(q.x() - 4, q.y() + 4), QPointF(q.x() + 4, q.y() - 4))
            if sel or (e['cables'] and self.mode == 'top'):
                p.setPen(col)
                p.drawText(QPointF(q.x() + 7, q.y() + 12), e['name'])
        xy, _ = self._project(np.array([res.compute['pos']]), B, scale, centre, W, H)
        q = QPointF(*xy[0])
        p.setBrush(THEME.color('ink'))
        p.setPen(QPen(THEME.color('viewport'), 1.5))
        p.drawRect(QRectF(q.x() - 5.5, q.y() - 5.5, 11, 11))
        p.setPen(THEME.color('ink'))
        p.drawText(QPointF(q.x() + 9, q.y() - 7), res.compute['name'])

    def _paint_hud(self, p, W, H):
        p.setFont(sans(11))
        p.setPen(THEME.color('muted'))
        how = {'top': 'From above, forward to the right', 'side': 'From the right side, forward to the right',
               '3d': 'Drag to orbit'}[self.mode]
        p.drawText(QRectF(10, H - 20, W - 20, 16), Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter,
                   f'{how} · wheel to zoom · right-drag to pan · double-click to reset · dashed: inside or hidden')
        if self.message:
            p.setPen(THEME.color('ink2'))
            p.drawText(QRectF(10, 6, W - 20, 16), Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter,
                       self.message)
        tip = self.tip_text()
        if tip:
            p.setFont(sans(11))
            fm = p.fontMetrics()
            lines = tip.split('\n')
            w = max(fm.horizontalAdvance(t) for t in lines) + 16
            h = fm.height() * len(lines) + 10
            r = QRectF(W - w - 10, 26, w, h)
            p.setPen(QPen(THEME.color('strong'), 1))
            p.setBrush(THEME.color('raised'))
            p.drawRoundedRect(r, 4, 4)
            p.setPen(THEME.color('ink'))
            for k, t in enumerate(lines):
                p.drawText(QPointF(r.left() + 8, r.top() + 5 + fm.ascent() + k * fm.height()), t)

    def tip_text(self) -> str:
        """The hovered route's or entry point's figures (painted in the corner), '' without one."""
        if self.res is None or self.hover is None:
            return ''
        kind, key = self.hover
        if kind == 'run':
            r = self.res.run(key)
            if r is None:
                return ''
            lim = f' of {r["limit_m"]:g} m' if r['limit_m'] else ''
            via = f' via {r["entry_name"]}' if r['entry_name'] else (' inside' if r['mode'] == 'in_in' else '')
            out = [f'{r["tag"]} → {r["to"]}{via}', f'{r["length_m"]:.2f} m{lim} of {r["cable_label"]}']
            if r['margin_db'] is not None:
                out.append(f'RF margin {r["margin_db"]:.1f} dB ({r["loss_db"]:.1f} dB lost)')
            out += [m for s, k, m in r['problems'] if s != 'note'][:2]
            return '\n'.join(out)
        e = self.res.entry(key)
        if e is None:
            return ''
        return f'{e["name"]} · {len(e["cables"])} cable{"s" * (len(e["cables"]) != 1)}' + \
            ('' if e['enabled'] else ' · switched off')

    # ---- picking
    def pick(self, pos) -> tuple | None:
        """('entry', uid) or ('run', id) under the screen point pos (QPointF), or None."""
        if self.res is None:
            return None
        W, H = self.width(), self.height()
        B, scale, centre = self._frame(W, H)
        x, y = pos.x(), pos.y()
        for e in self.res.entries:
            xy, _ = self._project(np.array([e['pos']]), B, scale, centre, W, H)
            if math.hypot(xy[0][0] - x, xy[0][1] - y) <= PICK_PX:
                return ('entry', e['uid'])
        best = None
        for rid, xy in self._screen.items():
            if len(xy) < 2:
                continue
            a, b = xy[:-1], xy[1:]
            ab = b - a
            t = np.clip(((x - a[:, 0]) * ab[:, 0] + (y - a[:, 1]) * ab[:, 1]) /
                        np.maximum((ab ** 2).sum(axis=1), 1e-9), 0, 1)
            d = np.hypot(a[:, 0] + t * ab[:, 0] - x, a[:, 1] + t * ab[:, 1] - y).min()
            if d <= PICK_PX and (best is None or d < best[0]):
                best = (d, rid)
        return ('run', best[1]) if best else None

    # ---- interaction
    def mousePressEvent(self, e):
        if e.button() == Qt.MouseButton.LeftButton:
            self._drag = ('orbit', e.position(), self.yaw, self.elev, self.pan.copy())
        elif e.button() in (Qt.MouseButton.RightButton, Qt.MouseButton.MiddleButton):
            self._drag = ('pan', e.position(), self.yaw, self.elev, self.pan.copy())

    def mouseMoveEvent(self, e):
        if self._drag is not None:
            how, pos, yaw, elev, pan = self._drag
            d = e.position() - pos
            if how == 'pan':
                self.pan = pan + np.array([d.x(), d.y()])
            elif self.mode == '3d':
                self.yaw = yaw - d.x() * 0.5
                self.elev = max(-89.0, min(89.0, elev + d.y() * 0.5))
            else:
                return
            if abs(d.x()) + abs(d.y()) > 2:
                self._dragging = True
            self.update()
            return
        hit = self.pick(e.position())
        if hit != self.hover:
            self.hover = hit
            self.update()

    def mouseReleaseEvent(self, e):
        drag, self._drag = self._drag, None
        if drag is None:
            return
        if self._dragging:
            self._settle.start()
            return
        if drag[0] == 'orbit':
            hit = self.pick(e.position())
            if hit is not None and hit[0] == 'run':
                self.selected = hit[1]
                self.run_clicked.emit(hit[1])
            elif hit is not None:
                self.selected_entry = hit[1]
                self.entry_clicked.emit(hit[1])
            else:
                self.selected = ''
                self.run_clicked.emit('')
            self.update()

    def _refine(self):
        self._dragging = False
        self._key = None
        self.update()

    def leaveEvent(self, e):
        if self.hover is not None:
            self.hover = None
            self.update()

    def wheelEvent(self, e):
        step = e.angleDelta().y() / 120.0
        self.zoom = max(0.3, min(12.0, self.zoom * math.pow(1.15, step)))
        self.update()

    def mouseDoubleClickEvent(self, e):
        self.reset_view()


# =========================================================================== the main views (wired by the window)
def scene_draws(ctx, gl=None) -> list:
    """The routed cables in the main 3D view (View ▸ Overlays ▸ Cable harness): every data and RF run as a line in
    its device's colour, the inside part dashed, while the harness is up to date for this project and variant."""
    from .viewport3d import LineDraw
    ctl = controller(ctx)
    if not ctl.fresh():
        ctl._kick()                                     # routed on a thread; drawn when it is done
        return []
    res = ctl.result
    if res.variant_uid != ctx.store.project.variant.uid:
        return []
    out = []
    for r in res.runs:
        if r['what'] == 'power':
            continue
        P = np.asarray(r['path'], float)
        sp = r['inside_span']
        col = run_qcolor(r)
        if sp is None:
            out.append(LineDraw(P, col, 2.0))
            continue
        a, b = sp
        if a > 0:
            out.append(LineDraw(P[:a + 1], col, 2.0))
        out.append(LineDraw(P[a:b + 1], run_qcolor(r, 0.8), 1.4, dash=True, dash_len=0.05))
        if b < len(P) - 1:
            out.append(LineDraw(P[b:], col, 2.0))
    return out


def paint_plan(view, p: QPainter):
    """The routed cables seen from above in a plan view (view.w2s_np, view.ctx), the inside parts dashed, the entry
    points as rings, while the harness is up to date."""
    ctl = controller(view.ctx)
    if not ctl.fresh():
        ctl._kick()
        return
    if ctl.result.variant_uid != view.ctx.store.project.variant.uid:
        return
    res = ctl.result
    p.save()
    p.setRenderHint(QPainter.RenderHint.Antialiasing, True)
    for r in res.runs:
        if r['what'] == 'power':
            continue
        S = view.w2s_np(np.asarray(r['path'], float)[:, :2])
        sp = r['inside_span']
        col = run_qcolor(r)
        for i in range(len(S) - 1):
            inside = sp is not None and sp[0] <= i < sp[1]
            pen = QPen(col, 1.6 if not inside else 1.2, Qt.PenStyle.SolidLine, Qt.PenCapStyle.RoundCap)
            if inside:
                pen.setStyle(Qt.PenStyle.CustomDashLine)
                pen.setDashPattern([3, 3])
            p.setPen(pen)
            p.drawLine(QPointF(*S[i]), QPointF(*S[i + 1]))
    p.setBrush(Qt.BrushStyle.NoBrush)
    for e in res.entries:
        if not e['cables']:
            continue
        q = view.w2s_np(np.asarray(e['pos'], float)[None, :2])[0]
        p.setPen(QPen(THEME.color('ink'), 1.2))
        p.drawEllipse(QPointF(*q), 4, 4)
    p.restore()
