"""The data-collection rig in the UI: adding presets, the inspector page of a rig part, drawing it in the 3D and plan
views and picking it with the mouse. The geometry itself lives in core.rig."""
from __future__ import annotations

import numpy as np
from PyQt6.QtCore import QPointF, Qt
from PyQt6.QtGui import QColor, QPainter, QPen, QPolygonF
from PyQt6.QtWidgets import QHBoxLayout, QLineEdit, QWidget

from ..core import rig as R
from ..core.project import RigPart
from .inspector import Page, _table, _cell, _fit_table
from .theme import THEME, STATUS
from .viewport3d import LIGHT, TriDraw
from .widgets import Section, ScrubSpinBox, SegmentedControl, StatusLine, ToggleSwitch, button, label

KINDS = [('box', 'Box'), ('cylinder', 'Cylinder'), ('bar', 'Bar')]
SEL_RGB = np.array(QColor(STATUS['warning']).getRgbF()[:3])


def selected_uid(ctx):
    o = ctx.store.sel_obj
    return o[1] if o and o[0] == 'rig' and len(o) > 1 else None


def add_preset(ctx, kind: str) -> list[RigPart]:
    """Add a preset sized from the vehicle's roof as one undo step and select its first part. It is fitted with the parts
    already there (core.rig.fit_plan): a mast stands on a roof rack under it, and a rack fitted under a mast lifts the
    mast with its head and the cameras and sensors mounted on them (in every variant)."""
    st = ctx.store
    plan = R.fit_plan(st.project, kind)
    parts, dz, moved = plan
    text = f'Added {dict(R.PRESETS)[kind].lower()} to the rig'
    masts = [q for q in st.project.rig if q.uid in dz and q.kind == 'cylinder']
    if masts:
        devs = len({d.uid for _, d, _ in moved})
        text += (f' · {", ".join(q.name for q in masts)} lifted onto it by {max(dz.values()) * 1000:.0f} mm'
                 + (f' with {devs} device{"s" if devs > 1 else ""} on it' if devs else ''))
    keys = ['rig'] + [f'variant:{u}' for u in dict.fromkeys(vu for vu, _, _ in moved)]
    st.edit(keys, lambda p: R.apply_fit(p, plan), text)
    st.select_obj(('rig', parts[0].uid))
    return parts


def delete_parts(ctx, uids):
    uids = set(uids)
    n = sum(1 for p in ctx.store.project.rig if p.uid in uids)
    if not n:
        return
    ctx.store.edit('rig', lambda p: setattr(p, 'rig', [q for q in p.rig if q.uid not in uids]),
                   f'Deleted {n} rig part{"s" if n > 1 else ""}')
    ctx.store.select_obj(('rig',))


def set_visible(ctx, uid, on):
    part = next((q for q in ctx.store.project.rig if q.uid == uid), None)
    if part is None or part.visible == on:
        return

    def fn(p):
        for q in p.rig:
            if q.uid == uid:
                q.visible = on
    ctx.store.edit('rig', fn, f'{"Showed" if on else "Hid"} {part.name}')


# ------------------------------------------------------------------ 3D
def _corner_normals(p: RigPart, T: np.ndarray, fn: np.ndarray) -> np.ndarray:
    """Per-corner normals (K, 3, 3): flat on boxes and caps, radial on the round side of cylinders and bars."""
    N = np.repeat(fn[:, None], 3, axis=1)
    if p.kind == 'box':
        return N
    a = R._rot(p.yaw) @ (np.array([0.0, 0.0, 1.0]) if p.kind == 'cylinder' else np.array([0.0, 1.0, 0.0]))
    side = np.abs(fn @ a) < 0.5
    d = T[side] - np.array([p.x, p.y, p.z])
    rad = d - (d @ a)[..., None] * a
    N[side] = rad / np.maximum(np.linalg.norm(rad, axis=-1, keepdims=True), 1e-12)
    return N


_cache = {}


def scene_tris(ctx, gl: bool) -> TriDraw | None:
    """The visible rig parts as lit triangles; parts that block nothing see-through. The selected part is tinted
    (where the renderer draws outlines, scene_draws gives it its own draw with the selection outline instead)."""
    draws = scene_draws(ctx, gl, outline=False)
    return draws[0] if draws else None


def scene_draws(ctx, gl: bool, outline: bool | None = None) -> list:
    """The visible rig parts: one draw, or with the renderer's outlines, the selected part apart with the selection
    outline. Smooth corner normals (and so shadows in the shader renderer) except while something is dragged."""
    parts = [p for p in ctx.store.project.rig if p.visible]
    if not parts:
        return []
    sel = selected_uid(ctx)
    from . import overlays3d as O3
    split = (O3.has_outline() if outline is None else outline) and any(p.uid == sel for p in parts)
    smooth = gl and not ctx.store.dragging       # while dragging the rig casts no shadows (maps redone per frame)
    key = (R.rig_key(parts), sel, gl, smooth, THEME.name, split)
    if _cache.get('key') == key:
        return _cache['draws']
    groups = {False: ([], [], []), True: ([], [], [])}
    for p in parts:
        V, F = R.part_geometry(p)
        T = V[F]
        fn = np.cross(T[:, 1] - T[:, 0], T[:, 2] - T[:, 0])
        fn /= np.maximum(np.linalg.norm(fn, axis=1, keepdims=True), 1e-12)
        base = np.array(R.COLOR) * (1.25 if THEME.name == 'light' else 1.0)
        if p.uid == sel and not split:
            base = base * 0.45 + SEL_RGB * 0.55
        rgb = np.tile(base, (len(T), 1)) if gl else base[None] * (0.55 + 0.45 * np.clip(fn @ LIGHT, 0, 1))[:, None]
        tri, col, nrm = groups[split and p.uid == sel]
        tri.append(T)
        col.append(np.c_[np.clip(rgb, 0, 1), np.full(len(T), 1.0 if p.occludes else 0.35)])
        nrm.append(_corner_normals(p, T, fn))
    draws = []
    for chosen, (tri, col, nrm) in groups.items():
        if tri:
            draws.append(O3.tri_draw(np.concatenate(tri), np.concatenate(col),
                                     normals=np.concatenate(nrm) if smooth else None,
                                     outline=O3.sel_outline() if chosen else None))
    _cache.update(key=key, draws=draws)
    return draws


def pick(project, vg, o, d) -> str | None:
    """uid of the rig part the ray (origin o, direction d) meets first, unless the body is in front of it."""
    parts = [p for p in project.rig if p.visible]
    m = R.rig_mesh(parts)
    if m is None:
        return None
    t, f = m.raycast(o[None], d[None])
    if f[0] < 0:
        return None
    if vg is not None:
        tb, fb = vg.body.raycast(o[None], d[None], faces_mask=vg.body.occluder_faces())
        if fb[0] >= 0 and tb[0] < t[0] - 1e-4:
            return None
    return parts[int(m.face_part[f[0]])].uid


# ------------------------------------------------------------------ plan view
def paint_plan(view, p: QPainter):
    """Outlines of the visible rig parts seen from above; the selected one tinted."""
    parts = [q for q in view.project.rig if q.visible]
    if not parts:
        return
    sel = selected_uid(view.ctx)
    p.save()
    p.setRenderHint(QPainter.RenderHint.Antialiasing, True)
    for part in sorted(parts, key=lambda q: (q.uid == sel, q.z)):
        poly = QPolygonF([QPointF(*q) for q in view.w2s_np(R.footprint(part))])
        # the rig's own dark grey (the body under it is light in both themes)
        c = QColor(STATUS['warning']) if part.uid == sel else QColor.fromRgbF(*(np.array(R.COLOR) * 0.75))
        fill = QColor(c) if part.uid == sel else QColor.fromRgbF(*R.COLOR)
        fill.setAlphaF((0.45 if part.uid == sel else 0.7) if part.occludes else 0.15)
        pen = QPen(c, 1.6 if part.uid == sel else 1.1)
        if not part.occludes:
            pen.setStyle(Qt.PenStyle.DashLine)
        p.setPen(pen)
        p.setBrush(fill)
        p.drawPolygon(poly)
    p.restore()


# ------------------------------------------------------------------ inspector
class RigPage(Page):
    """A rig part, or with the Rig group selected, the rig's parts and the presets."""
    title = 'Rig'

    def __init__(self, ctx):
        super().__init__(ctx)
        self._busy = False
        s = self.s_part = Section('Rig part')
        self.name = QLineEdit()
        self.name.editingFinished.connect(lambda: self._set('name', self.name.text().strip() or 'Rig part'))
        s.row('Name', self.name)
        self.kind = SegmentedControl(KINDS, min_w=56)
        self.kind.changed.connect(lambda k: self._set('kind', k))
        s.row('Kind', self.kind, QWidget())
        self.px, self.py, self.pz = (ScrubSpinBox(a, 'length', 3) for a in ('X', 'Y', 'Z'))
        s.row('Centre m', self.px, self.py, self.pz)
        # extents along the part's own x, y, z; a round side shows its diameter and is set by the radius
        self.sx, self.sy, self.sz = (ScrubSpinBox(a, 'length', 3, step=0.005) for a in ('X', 'Y', 'Z'))
        s.row('Size m', self.sx, self.sy, self.sz)
        self.sr = ScrubSpinBox('R', 'length', 3, step=0.001)
        self.row_r = s.row('Radius m', self.sr, QWidget())
        self.yaw = ScrubSpinBox('ψ', 'angle', 1)
        s.row('Yaw °', self.yaw, QWidget())
        for i, (w, k) in enumerate(((self.px, 'x'), (self.py, 'y'), (self.pz, 'z'), (self.sx, 'size_x'),
                                    (self.sy, 'size_y'), (self.sz, 'size_z'), (self.sr, 'radius'), (self.yaw, 'yaw'))):
            w.committed.connect(lambda v, k=k, i=i: self._set(self._field(k), self._model(i, v)))
            w.scrubbed.connect(lambda v, k=k, i=i: self._set(self._field(k), self._model(i, v), merge=True))
            w.scrub_started.connect(lambda: setattr(ctx.store, 'dragging', True))
            w.scrub_finished.connect(self._scrub_done)
        self.occ = ToggleSwitch()
        self.occ.toggled.connect(lambda on: self._set('occludes', on) if not self._busy else None)
        s.row('Occludes', self.occ, QWidget(), tip='Cameras cannot see through it (ground, blind volume, images).')
        self.mnt = ToggleSwitch()
        self.mnt.toggled.connect(lambda on: self._set('mountable', on) if not self._busy else None)
        s.row('Mountable', self.mnt, QWidget(), tip='Cameras can be placed on it and snap to it when dragged.')
        self.note = label('', muted=True, small=True, wrap=True)
        s.add(self.note)
        bl = QHBoxLayout()
        b = button('Delete part', 'destructive', 'critical')
        b.clicked.connect(lambda: delete_parts(ctx, [selected_uid(ctx)]))
        bl.addWidget(b)
        bl.addStretch(1)
        s.blay.addLayout(bl)
        self.lay.addWidget(s)
        g = self.s_all = Section('Rig')
        self.t = _table(['Part', 'Kind', 'Blocks'], 0)
        self.t.cellClicked.connect(lambda r, c: ctx.store.select_obj(('rig', self.project.rig[r].uid)))
        g.add(self.t)
        self.empty = StatusLine('good', 'No rig on the vehicle. Add a roof rack, a mast or an enclosure: it blocks '
                                        'cameras like the body and can carry them.')
        g.add(self.empty)
        for k, text in R.PRESETS:
            r = QHBoxLayout()
            b = button(f'Add {text.lower()}', None, 'plus')
            b.clicked.connect(lambda _=False, k=k: add_preset(ctx, k))
            r.addWidget(b)
            r.addStretch(1)
            g.blay.addLayout(r)
        self.lay.addWidget(g)
        self.lay.addStretch(1)

    def _part(self):
        uid = selected_uid(self.ctx)
        return next((q for q in self.project.rig if q.uid == uid), None)

    def _field(self, k):
        """The RigPart field an inspector field edits (the size fields depend on the kind)."""
        part = self._part()
        if k == 'size_x':
            return 'length'
        if k == 'size_y':
            return 'length' if part is not None and part.kind == 'bar' else 'width'
        return 'height' if k == 'size_z' else k

    def _model(self, i, v):
        """A field's value in the vehicle frame (positions are shown from the display origin)."""
        if i < 3:
            return round(float(v + self.project.origin_offset()[i]), 4)
        if i < 6:
            return max(0.005, round(float(v), 4))
        if i == 6:
            return max(0.002, round(float(v), 4))
        return round(float(v), 2)

    def _set(self, k, v, merge=False):
        part = self._part()
        if part is None or getattr(part, k) == v:
            return                         # focus left a field without a change: no undo step
        uid = part.uid

        def fn(p):
            for q in p.rig:
                if q.uid == uid:
                    setattr(q, k, v)
        self.ctx.store.edit('rig', fn, f'Changed {k} of {part.name}', merge_key=(uid, k) if merge else None)

    def _scrub_done(self):
        self.ctx.store.dragging = False
        self.ctx.analysis.request()

    def refresh(self):
        part = self._part()
        self.s_part.setVisible(part is not None)
        self.s_all.setVisible(part is None)
        self._busy = True
        try:
            if part is not None:
                o = self.project.origin_offset()
                if not self.name.hasFocus():
                    self.name.setText(part.name)
                self.kind.set_value(part.kind)
                d = 2 * part.radius
                ext = {'box': (part.length, part.width, part.height), 'cylinder': (d, d, part.height),
                       'bar': (d, part.length, d)}[part.kind]
                for w, v in ((self.px, part.x - o[0]), (self.py, part.y - o[1]), (self.pz, part.z - o[2]),
                             (self.sx, ext[0]), (self.sy, ext[1]), (self.sz, ext[2]),
                             (self.sr, part.radius), (self.yaw, part.yaw)):
                    w.set_value(v)
                self.sx.setEnabled(part.kind == 'box')
                self.sy.setEnabled(part.kind != 'cylinder')
                self.sz.setEnabled(part.kind != 'bar')
                self.row_r.setVisible(part.kind != 'box')
                self.occ.setChecked(part.occludes)
                self.mnt.setChecked(part.mountable)
                self.mnt.setEnabled(part.occludes)
                self.note.setText({'box': 'Size along its own X, Y and Z (turned by the yaw).',
                                   'cylinder': 'Upright: set its height (Z) and radius.',
                                   'bar': 'Runs across the vehicle: set its length (Y) and radius.'}[part.kind]
                                  + ('' if part.visible else ' Hidden: it blocks nothing until shown.'))
        finally:
            self._busy = False
        rig = self.project.rig
        self.t.setRowCount(len(rig))
        for i, q in enumerate(rig):
            self.t.setItem(i, 0, _cell(q.name, muted=not q.visible))
            self.t.setItem(i, 1, _cell(q.kind, True))
            self.t.setItem(i, 2, _cell('yes' if q.occludes else 'no', True, True))
        _fit_table(self.t)
        self.t.setVisible(bool(rig))
        self.empty.setVisible(not rig)
