"""Lidars and radars in the UI: adding them, the inspector page of a sensor, drawing them in the 3D and plan views and
picking them with the mouse. The library and the geometry live in core.sensors."""
from __future__ import annotations

import copy
import math

import numpy as np
from PyQt6.QtCore import QPointF, Qt
from PyQt6.QtGui import QColor, QPainter, QPen, QPolygonF
from PyQt6.QtWidgets import QComboBox, QFileDialog, QHBoxLayout, QLineEdit, QWidget

from ..core import sensors as S
from ..core.project import Sensor
from .inspector import Page, _table, _cell, _fit_table
from .theme import THEME, STATUS
from .viewport3d import LIGHT, LineDraw, TriDraw
from .widgets import Section, ScrubSpinBox, StatusLine, ToggleSwitch, button, label

COLORS = {'lidar': '#0aa5b8', 'radar': '#a0743c'}       # apart from the eight camera colours in both themes
SEL_RGB = np.array(QColor(STATUS['warning']).getRgbF()[:3])
RING_FAR = 15.0                  # ground rings and radar edges are drawn this far out in 3D
MAX_RINGS = 16                   # at most this many of a lidar's channels are drawn as ground rings


def color(kind: str, alpha: float = 1.0) -> QColor:
    c = QColor(COLORS.get(kind, COLORS['lidar']))
    c.setAlphaF(alpha)
    return c


def selected_uid(ctx):
    o = ctx.store.sel_obj
    return o[1] if o and o[0] == 'sensor' and len(o) > 1 else None


def visible(project):
    return [s for s in project.variant.sensors if s.visible]


# ------------------------------------------------------------------ edits
def add_sensor(ctx, kind: str, model_id: str | None = None) -> Sensor:
    """Add a lidar or radar at its default place (lidar on the highest rig part or the roof, radar on the front bumper)
    as one undo step and select it."""
    st = ctx.store
    p = st.project
    mid = model_id if model_id in p.sensor_models else S.default_model_id(kind)
    if mid not in p.sensor_models:
        mid = next(k for k, m in p.sensor_models.items() if m.kind == kind)
    model = p.sensor_models[mid]
    pose = S.default_pose(kind, model, p.vehicle.mesh, p.rig)
    s = Sensor(kind=kind, model_id=mid, tag=S.next_tag(kind, p.variant.tags()),
               name=f'{model.vendor} {model.name}'.strip(), **pose)
    st.edit('variant', lambda pj: pj.variant.sensors.append(s), f'Added {kind} {s.tag} ({model.name})')
    st.select_obj(('sensor', s.uid))
    return s


def delete_sensors(ctx, uids):
    uids = set(uids)
    gone = [s.tag for s in ctx.store.variant.sensors if s.uid in uids]
    if not gone:
        return
    ctx.store.edit('variant', lambda p: setattr(p.variant, 'sensors', [s for s in p.variant.sensors if s.uid not in uids]),
                   f'Deleted {", ".join(gone)}')
    ctx.store.select_obj(('sensors',))


def set_props(ctx, uid, text=None, merge=False, **props):
    s = ctx.store.variant.sensor(uid)
    if s is None or all(getattr(s, k) == v for k, v in props.items()):
        return                         # focus left a field without a change: no undo step

    def fn(p):
        q = p.variant.sensor(uid)
        for k, v in props.items():
            setattr(q, k, v)
    ctx.store.edit('variant', fn, text or f'Changed {", ".join(props)} of {s.tag}',
                   merge_key=('sensor', uid, tuple(sorted(props))) if merge else None)


def set_visible(ctx, uid, on):
    s = ctx.store.variant.sensor(uid)
    if s is not None:
        set_props(ctx, uid, f'{"Showed" if on else "Hid"} {s.tag}', visible=on)


def own_model(project, s: Sensor):
    """The sensor's model, copied into an edited model of this project first when it is a built-in one, so editing
    figures never changes the library for other sensors or projects. Call inside an edit covering 'library' and
    'variant'; returns the editable model (s.model_id is switched to it)."""
    m = project.sensor_model_for(s)
    if not m.builtin:
        return m
    new_id, n = f'{m.id}_edited', 2
    while new_id in project.sensor_models:
        new_id, n = f'{m.id}_edited{n}', n + 1
    e = copy.deepcopy(m)
    e.id, e.builtin, e.name = new_id, False, f'{m.name} (edited)'
    project.sensor_models[new_id] = e
    s.model_id = new_id                # only this sensor: others keep the library model
    return e


# ------------------------------------------------------------------ 3D
def _housing(s: Sensor, model) -> tuple[np.ndarray, np.ndarray]:
    """Triangles (K, 3, 3) and face normals (K, 3) of the housing: a cylinder or a box."""
    c, half, R, rnd = S.body_box(s, model)
    if rnd:
        n = 20
        t = np.linspace(0, 2 * math.pi, n, endpoint=False)
        ring = np.c_[np.cos(t) * half[0], np.sin(t) * half[1]]
        bot = np.c_[ring, np.full(n, -half[2])]
        top = np.c_[ring, np.full(n, half[2])]
        tri = []
        for i in range(n):
            j = (i + 1) % n
            tri += [[bot[i], bot[j], top[j]], [bot[i], top[j], top[i]], [[0, 0, half[2]], top[i], top[j]],
                    [[0, 0, -half[2]], bot[j], bot[i]]]
        T = np.array(tri, float)
    else:
        q = np.array([[-1, -1, -1], [1, -1, -1], [1, 1, -1], [-1, 1, -1], [-1, -1, 1], [1, -1, 1], [1, 1, 1],
                      [-1, 1, 1]], float) * half
        F = [(0, 2, 1), (0, 3, 2), (4, 5, 6), (4, 6, 7), (0, 1, 5), (0, 5, 4), (1, 2, 6), (1, 6, 5), (2, 3, 7),
             (2, 7, 6), (3, 0, 4), (3, 4, 7)]
        T = q[np.array(F)]
    T = T @ R.T + c
    fn = np.cross(T[:, 1] - T[:, 0], T[:, 2] - T[:, 0])
    fn /= np.maximum(np.linalg.norm(fn, axis=1, keepdims=True), 1e-12)
    out = np.einsum('ij,ij->i', fn, T.mean(axis=1) - c) < 0           # wind outward (convex solids)
    T[out] = T[out][:, ::-1]
    fn[out] *= -1
    return T, fn


def scene_tris(ctx, gl: bool) -> TriDraw | None:
    """The visible sensors' housings, lit; the selected one tinted."""
    p = ctx.store.project
    sensors = visible(p)
    if not sensors:
        return None
    sel = selected_uid(ctx)
    tri, col, nrm = [], [], []
    for s in sensors:
        T, fn = _housing(s, p.sensor_model_for(s))
        base = np.array(color(s.kind).getRgbF()[:3])
        if s.uid == sel:
            base = base * 0.4 + SEL_RGB * 0.6
        rgb = np.tile(base, (len(T), 1)) if gl else base[None] * (0.55 + 0.45 * np.clip(fn @ LIGHT, 0, 1))[:, None]
        tri.append(T)
        col.append(np.c_[np.clip(rgb, 0, 1), np.ones(len(T))])
        nrm.append(np.repeat(fn[:, None], 3, axis=1))
    return TriDraw(np.concatenate(tri).astype(np.float32), np.concatenate(col).astype(np.float32), depth=True,
                   normals=np.concatenate(nrm).astype(np.float32) if gl else None)


def _ring_channels(model, n=MAX_RINGS):
    """Indices of the downward channels drawn as ground rings: all of them, or an even pick."""
    e = model.elevations()
    down = np.nonzero(e < -0.5)[0]
    if len(down) <= n:
        return set(down.tolist())
    return set(down[np.round(np.linspace(0, len(down) - 1, n)).astype(int)].tolist())


def scene_lines(ctx, far: float = RING_FAR) -> list:
    """Each lidar's ground rings (where its downward beams meet the ground, out to `far`) and each radar's field edges
    and arc at its own height, out to `far` or its range."""
    p = ctx.store.project
    sel = selected_uid(ctx)
    out = []
    for s in visible(p):
        m = p.sensor_model_for(s)
        w = 1.6 if s.uid == sel else 1.0
        if m.is_lidar:
            keep = _ring_channels(m)
            for ch, P in S.ground_rings(s, m, 1.0, far=min(far, m.range_m)):
                if ch in keep:
                    out.append(LineDraw(np.c_[P[:, :2], np.full(len(P), 0.008)], color('lidar', 0.55 if s.uid == sel
                                                                                       else 0.35), w))
        else:
            for poly in S.radar_wedges(s, m):
                d = poly - [s.x, s.y]
                r = np.linalg.norm(d, axis=1, keepdims=True)
                poly = np.where(r > far, [s.x, s.y] + d / np.maximum(r, 1e-9) * far, poly)
                P = np.c_[poly, np.full(len(poly), s.z)]
                out.append(LineDraw(np.vstack([P, P[:1]]), color('radar', 0.8 if s.uid == sel else 0.5), w, dash=True,
                                    dash_len=0.12))
    return out


def pick(project, o, d) -> str | None:
    """uid of the sensor whose housing the ray (origin o, direction d) passes nearest to its centre, within the
    housing's bounding sphere; the nearest along the ray wins."""
    best = None
    d = np.asarray(d, float) / max(np.linalg.norm(d), 1e-12)
    for s in visible(project):
        c, half, _, _ = S.body_box(s, project.sensor_model_for(s))
        v = c - np.asarray(o, float)
        t = float(v @ d)
        if t <= 0:
            continue
        miss = np.linalg.norm(v - t * d)
        if miss <= max(np.linalg.norm(half), 0.04) and (best is None or t < best[0]):
            best = (t, s.uid)
    return best[1] if best else None


# ------------------------------------------------------------------ plan view
def paint_plan(view, p: QPainter, fields: bool = True):
    """Every sensor's footprint and tag seen from above, and with fields, the radar fields and lidar ground rings."""
    project = view.project
    sensors = visible(project)
    if not sensors:
        return
    sel = selected_uid(view.ctx)
    p.save()
    p.setRenderHint(QPainter.RenderHint.Antialiasing, True)
    for s in sensors if fields else ():
        m = project.sensor_model_for(s)
        on = s.uid == sel
        if m.is_lidar:
            pen = QPen(color('lidar', 0.6 if on else 0.32), 1.2 if on else 0.8)
            p.setPen(pen)
            p.setBrush(Qt.BrushStyle.NoBrush)
            keep = _ring_channels(m)
            for ch, P in S.ground_rings(s, m, 1.0, far=min(60.0, m.range_m)):
                if ch in keep:
                    p.drawPolyline(QPolygonF([QPointF(*q) for q in view.w2s_np(P[:, :2])]))
        else:
            for poly in S.radar_wedges(s, m):
                fill = color('radar', 0.10 if on else 0.06)
                p.setPen(QPen(color('radar', 0.85 if on else 0.55), 1.3 if on else 0.9, Qt.PenStyle.DashLine))
                p.setBrush(fill)
                p.drawPolygon(QPolygonF([QPointF(*q) for q in view.w2s_np(poly)]))
    font = p.font()
    font.setPointSizeF(max(font.pointSizeF() - 1, 7))
    p.setFont(font)
    for s in sorted(sensors, key=lambda q: q.uid == sel):
        m = project.sensor_model_for(s)
        poly = QPolygonF([QPointF(*q) for q in view.w2s_np(S.footprint(s, m))])
        on = s.uid == sel
        c = QColor(STATUS['warning']) if on else color(s.kind)
        fill = QColor(c)
        fill.setAlphaF(0.75)
        p.setPen(QPen(c.darker(140), 1.6 if on else 1.0))
        p.setBrush(fill)
        p.drawPolygon(poly)
        a = math.radians(S.heading(s))
        q0 = view.w2s(s.x, s.y)
        q1 = view.w2s(s.x + 0.25 * math.cos(a), s.y + 0.25 * math.sin(a))
        p.setPen(QPen(c, 1.4))
        p.drawLine(q0, q1)
        p.setPen(THEME.color('ink'))
        p.drawText(QPointF(poly.boundingRect().right() + 3, poly.boundingRect().top() + 1), s.tag)
    p.restore()


def plan_pick(view, pos, radius_px: float = 9.0) -> Sensor | None:
    """The sensor under the screen position (its footprint, or within radius_px of its centre)."""
    best = None
    for s in visible(view.project):
        q = view.w2s(s.x, s.y)
        d = math.hypot(q.x() - pos.x(), q.y() - pos.y())
        poly = QPolygonF([QPointF(*v) for v in view.w2s_np(S.footprint(s, view.project.sensor_model_for(s)))])
        if (d <= radius_px or poly.containsPoint(pos, Qt.FillRule.OddEvenFill)) and (best is None or d < best[0]):
            best = (d, s)
    return best[1] if best else None


# ------------------------------------------------------------------ inspector
_LIDAR_FIELDS = [('channels', 'Channels', 'number', 0, 1), ('v_min', 'Lowest beam °', 'angle', 2, 0.5),
                 ('v_max', 'Highest beam °', 'angle', 2, 0.5), ('h_fov', 'Horizontal field °', 'angle', 1, 1),
                 ('h_res', 'Azimuth step °', 'angle', 4, 0.01), ('rate_hz', 'Frame rate Hz', 'number', 1, 1),
                 ('range_m', 'Range @10% m', 'number', 1, 1), ('range_min_m', 'Nearest range m', 'number', 2, 0.05),
                 ('points_per_s', 'Points/s', 'number', 0, 1000), ('bytes_per_point', 'Bytes/point', 'number', 2, 0.25)]
_RADAR_FIELDS = [('rate_hz', 'Cycle rate Hz', 'number', 1, 1), ('range_min_m', 'Nearest range m', 'number', 2, 0.05),
                 ('data_mbps', 'Data Mbit/s', 'number', 2, 0.5)]


class SensorPage(Page):
    """A lidar or radar, or with the Sensors group selected, the variant's sensors."""
    title = 'Sensor'

    def __init__(self, ctx):
        super().__init__(ctx)
        self._busy = False
        s = self.s_one = Section('Sensor')
        self.name = QLineEdit()
        self.name.editingFinished.connect(lambda: self._set(name=self.name.text().strip() or 'Sensor'))
        s.row('Name', self.name)
        self.tag = QLineEdit()
        self.tag.setMaxLength(12)
        self.tag.editingFinished.connect(self._set_tag)
        s.row('Tag', self.tag, QWidget())
        self.model = QComboBox()
        self.model.activated.connect(self._set_model)
        s.row('Model', self.model)
        self.px, self.py, self.pz = (ScrubSpinBox(a, 'length', 3) for a in ('X', 'Y', 'Z'))
        s.row('Optical centre m', self.px, self.py, self.pz)
        self.yaw, self.pitch, self.roll = (ScrubSpinBox(a, 'angle', 1) for a in ('ψ', 'θ', 'φ'))
        s.row('Yaw pitch roll °', self.yaw, self.pitch, self.roll,
              tip='Intrinsic Z-Y-X like a camera; pitch positive down. Yaw 0 faces forward.')
        for i, (w, k) in enumerate(((self.px, 'x'), (self.py, 'y'), (self.pz, 'z'), (self.yaw, 'yaw'),
                                    (self.pitch, 'pitch'), (self.roll, 'roll'))):
            w.committed.connect(lambda v, k=k, i=i: self._set(**{k: self._value(i, v)}))
            w.scrubbed.connect(lambda v, k=k, i=i: self._set(merge=True, **{k: self._value(i, v)}))
            w.scrub_started.connect(lambda: setattr(ctx.store, 'dragging', True))
            w.scrub_finished.connect(self._scrub_done)
        self.phase = ScrubSpinBox('', 'angle', 1, '°')
        self.phase.committed.connect(lambda v: self._set(phase_deg=round(float(v) % 360.0, 2)))
        self.row_phase = s.row('Sync phase', self.phase, QWidget(),
                               tip='Azimuth (its own frame) the lidar faces at the top of each sync period (PTP '
                                   'phase lock). Analysis ▸ Sync and timing… suggests one.')
        self.port = QLineEdit()
        self.port.setPlaceholderText('e.g. eth1 or switch port 3')
        self.port.editingFinished.connect(lambda: self._set(port=self.port.text().strip()))
        s.row('Port', self.port, tip='Where it plugs into the recorder (for the recording budget).')
        self.lock = ToggleSwitch()
        self.lock.toggled.connect(lambda on: self._set(locked=on) if not self._busy else None)
        s.row('Locked', self.lock, QWidget(), tip='A locked sensor is not dragged in the top view.')
        self.info = label('', muted=True, small=True, wrap=True)
        s.add(self.info)
        bl = QHBoxLayout()
        self.b_beams = button('Import beam angles…', None, 'import',
                              tip='Measured channel elevations: Ouster metadata JSON, Hesai angle-correction CSV, '
                                  'Velodyne ROS calibration YAML or one angle per line')
        self.b_beams.clicked.connect(self._import_beams)
        bl.addWidget(self.b_beams)
        b = button('Delete', 'destructive', 'critical')
        b.clicked.connect(lambda: delete_sensors(ctx, [selected_uid(ctx)]))
        bl.addWidget(b)
        bl.addStretch(1)
        s.blay.addLayout(bl)
        self.lay.addWidget(s)
        # the model's own figures (editing a built-in model makes an edited copy for this sensor)
        f = self.s_fig = Section('Model figures')
        self.fields = {}
        for k, text, kind, dec, step in _LIDAR_FIELDS + _RADAR_FIELDS:
            if k in self.fields:
                continue
            w = ScrubSpinBox('', kind, dec, step=step)
            w.committed.connect(lambda v, k=k: self._set_figure(k, v))
            self.fields[k] = (w, f.row(text, w, QWidget()))
        self.zones = _table(['Zone', 'Range m', '±Az °', '±El °'], 0)
        self.zones.setEditTriggers(self.zones.EditTrigger.DoubleClicked | self.zones.EditTrigger.EditKeyPressed)
        self.zones.itemChanged.connect(self._zone_edited)
        f.add(self.zones)
        self.fig_note = label('', muted=True, small=True, wrap=True)
        f.add(self.fig_note)
        self.lay.addWidget(f)
        g = self.s_all = Section('Sensors')
        self.t = _table(['Tag', 'Model', 'Kind', 'Height'], 1)
        self.t.cellClicked.connect(lambda r, c: ctx.store.select_obj(('sensor', self.project.variant.sensors[r].uid)))
        g.add(self.t)
        self.empty = StatusLine('good', 'No lidar or radar in this variant. A lidar goes on the highest rig part or the '
                                        'roof, a radar on the front bumper; then move them here or in the top view.')
        g.add(self.empty)
        r = QHBoxLayout()
        for kind in ('lidar', 'radar'):
            b = button(f'Add {kind}', None, 'plus')
            b.clicked.connect(lambda _=False, k=kind: add_sensor(ctx, k))
            r.addWidget(b)
        r.addStretch(1)
        g.blay.addLayout(r)
        self.lay.addWidget(g)
        self.lay.addStretch(1)

    # ------------------------------------------------------------------
    def _sensor(self):
        return self.project.variant.sensor(selected_uid(self.ctx))

    def _value(self, i, v):
        """A field's value in the vehicle frame (positions are shown from the display origin)."""
        if i < 3:
            return round(float(v + self.project.origin_offset()[i]), 4)
        return round(float(v), 2)

    def _set(self, merge=False, **props):
        s = self._sensor()
        if s is not None:
            set_props(self.ctx, s.uid, merge=merge, **props)

    def _set_tag(self):
        s = self._sensor()
        t = self.tag.text().strip()
        if s is None or not t or t == s.tag:
            return
        if t in self.project.variant.tags():
            self.ctx.store.status.emit(f'Tag {t} is already used in this variant.', 3000)
            self.tag.setText(s.tag)
            return
        set_props(self.ctx, s.uid, f'Renamed {s.tag} to {t}', tag=t)

    def _set_model(self, i):
        s = self._sensor()
        mid = self.model.itemData(i)
        if s is not None and mid and mid != s.model_id:
            m = self.project.sensor_models[mid]
            set_props(self.ctx, s.uid, f'{s.tag} is now a {m.name}', model_id=mid)

    def _scrub_done(self):
        self.ctx.store.dragging = False
        self.ctx.analysis.request()

    def _set_figure(self, k, v):
        s = self._sensor()
        if s is None:
            return
        m = self.project.sensor_model_for(s)
        v = float(v)
        if not math.isfinite(v):
            return
        if k == 'channels':
            v = int(max(1, min(512, round(v))))
        elif k in ('v_min', 'v_max'):
            v = max(-90.0, min(90.0, v))
        elif k == 'h_fov':
            v = max(1.0, min(360.0, v))
        else:
            v = max(v, 0.0) if k in ('points_per_s', 'range_min_m', 'data_mbps', 'bytes_per_point') else max(v, 1e-3)
        if getattr(m, k) == v:
            return
        uid = s.uid

        def fn(p):
            e = own_model(p, p.variant.sensor(uid))
            setattr(e, k, v)
            if k == 'channels':
                e.beams = []                   # a channel count spreads the beams evenly again
            if e.v_min > e.v_max:
                e.v_min, e.v_max = e.v_max, e.v_min
        self.ctx.store.edit(['library', 'variant'], fn, f'Changed {k} of {m.name}')

    def _zone_edited(self, item):
        if self._busy:
            return
        s = self._sensor()
        if s is None or item.column() == 0:
            return
        try:
            v = float(item.text())
        except ValueError:
            self.refresh()
            return
        r, c = item.row(), item.column() - 1
        m = self.project.sensor_model_for(s)
        if r >= len(m.zones) or not math.isfinite(v) or v <= 0 or m.zones[r][c] == v:
            self.refresh()
            return
        v = min(v, 180.0) if c else v
        uid = s.uid

        def fn(p):
            e = own_model(p, p.variant.sensor(uid))
            e.zones[r][c] = v
        self.ctx.store.edit(['library', 'variant'], fn, f'Changed radar zone {r + 1} of {m.name}')

    def _import_beams(self):
        s = self._sensor()
        if s is None:
            return
        path, _ = QFileDialog.getOpenFileName(self, 'Import beam angles', '',
                                              'Lidar calibration (*.json *.csv *.yaml *.yml *.txt);;All files (*)')
        if path:
            self.import_beams(path)

    def import_beams(self, path):
        """Use the beam elevations in a lidar's own calibration file for the selected lidar (one undo step)."""
        s = self._sensor()
        if s is None or s.kind != 'lidar':
            return False
        try:
            beams = S.read_beam_angles(path)
        except (OSError, ValueError, KeyError, TypeError) as ex:
            self.ctx.store.status.emit(f'Could not read beam angles: {ex}', 5000)
            return False
        uid = s.uid

        def fn(p):
            e = own_model(p, p.variant.sensor(uid))
            e.beams = beams
            e.channels = len(beams)
            e.v_min, e.v_max = beams[0], beams[-1]
        self.ctx.store.edit(['library', 'variant'], fn, f'Imported {len(beams)} beam angles for {s.tag}')
        return True

    # ------------------------------------------------------------------
    def refresh(self):
        s = self._sensor()
        self.s_one.setVisible(s is not None)
        self.s_fig.setVisible(s is not None)
        self.s_all.setVisible(s is None)
        self._busy = True
        try:
            if s is not None:
                self._refresh_one(s)
        finally:
            self._busy = False
        v = self.project.variant
        self.t.setRowCount(len(v.sensors))
        for i, q in enumerate(v.sensors):
            m = self.project.sensor_model_for(q)
            self.t.setItem(i, 0, _cell(q.tag, True, muted=not q.visible))
            self.t.setItem(i, 1, _cell(m.name, muted=not q.visible))
            self.t.setItem(i, 2, _cell(q.kind, True))
            self.t.setItem(i, 3, _cell(f'{q.z:.2f}', True, True))
        _fit_table(self.t)
        self.t.setVisible(bool(v.sensors))
        self.empty.setVisible(not v.sensors)

    def _refresh_one(self, s: Sensor):
        p = self.project
        m = p.sensor_model_for(s)
        self.title = 'Lidar' if s.kind == 'lidar' else 'Radar'
        self.s_one.header.title.setText(f'{self.title} {s.tag}')
        if not self.name.hasFocus():
            self.name.setText(s.name)
        if not self.tag.hasFocus():
            self.tag.setText(s.tag)
        self.model.blockSignals(True)
        self.model.clear()
        for mid, mm in p.sensor_models.items():
            if mm.kind == s.kind:
                self.model.addItem(f'{mm.vendor} {mm.name}'.strip(), mid)
        self.model.setCurrentIndex(max(self.model.findData(m.id), 0))
        self.model.blockSignals(False)
        o = p.origin_offset()
        for w, val in ((self.px, s.x - o[0]), (self.py, s.y - o[1]), (self.pz, s.z - o[2]), (self.yaw, s.yaw),
                       (self.pitch, s.pitch), (self.roll, s.roll)):
            w.set_value(val)
        self.row_phase.setVisible(m.spinning)
        self.phase.set_value(s.phase_deg)
        if not self.port.hasFocus():
            self.port.setText(s.port)
        self.lock.setChecked(s.locked)
        self.b_beams.setVisible(m.is_lidar)
        lo, hi = m.v_fov()
        lines = [m.summary()]
        if m.is_lidar:
            lines.append(f'{S.SCAN_LABEL.get(m.scan, m.scan)} · {m.v_step():.2f}° between beams · '
                         f'{m.data_bps() / 1e6:.0f} Mbit/s · {m.interface} · {m.time_sync}'
                         + (f' · spins {"clockwise" if m.rotation == "cw" else "counter-clockwise"} from above'
                            if m.spinning else ''))
            if m.beams:
                lines.append(f'Beam angles imported ({len(m.beams)} channels, {lo:.2f}° to {hi:.2f}°).')
            rings = S.ground_rings(s, m, 2.0)
            if rings:
                near = min(float(np.hypot(P[:, 0] - s.x, P[:, 1] - s.y).min()) for _, P in rings)
                lines.append(f'Nearest ground ring {near:.1f} m from the lidar (before the body blocks any beam).')
        else:
            lines.append(f'{m.interface} · {m.time_sync}')
        if m.note:
            lines.append(m.note)
        self.info.setText('\n'.join(lines))
        for k, (w, row) in self.fields.items():
            show = any(k == f[0] for f in (_LIDAR_FIELDS if m.is_lidar else _RADAR_FIELDS))
            row.setVisible(show)
            if show:
                w.set_value(float(getattr(m, k)))
        self.fields['channels'][0].setEnabled(not m.beams)
        self.zones.setVisible(not m.is_lidar)
        if not m.is_lidar:
            self.zones.setRowCount(len(m.zones))
            for i, z in enumerate(m.zones):
                self.zones.setItem(i, 0, _cell(f'{i + 1}', True))
                for j in range(3):
                    self.zones.setItem(i, j + 1, _cell(f'{z[j]:g}', True, True))
            _fit_table(self.zones)
        self.fig_note.setText('Editing a built-in model makes an edited copy for this sensor; the library stays as it '
                              'was.' if m.builtin else 'This project\'s own model: edits apply to every sensor using it.')
