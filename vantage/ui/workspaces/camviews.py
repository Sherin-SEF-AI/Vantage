"""B1 · Camera views, stitched bird's-eye and bowl view."""
from __future__ import annotations

import math

from PyQt6.QtCore import Qt, QTimer
from PyQt6.QtGui import QFont
from PyQt6.QtWidgets import (QWidget, QHBoxLayout, QVBoxLayout, QGridLayout, QSplitter, QTableWidget, QTableWidgetItem,
                             QHeaderView, QAbstractItemView, QComboBox, QCheckBox)

from ..camview import CamImageView
from ..guides import GuideBar
from ..planview import PlanView
from ..scene import MainScene
from ..theme import THEME, mono
from ..viewport3d import make_viewport
from ..widgets import label, ROW_H


class CamViewsPage(QWidget):
    def __init__(self, ctx):
        super().__init__()
        self.ctx = ctx
        self.setObjectName('base')
        lay = QHBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        split = QSplitter(Qt.Orientation.Horizontal)
        split.setHandleWidth(2)
        left = QWidget()
        left.setObjectName('base')
        ll = QVBoxLayout(left)
        ll.setContentsMargins(0, 0, 0, 0)
        ll.setSpacing(2)
        self.guide_bar = GuideBar(ctx)
        ll.addWidget(self.guide_bar)
        self.grid_w = QWidget()
        self.grid_w.setObjectName('base')
        self.grid = QGridLayout(self.grid_w)
        self.grid.setContentsMargins(0, 0, 0, 0)
        self.grid.setSpacing(2)
        ll.addWidget(self.grid_w, 1)
        seams = QWidget()
        seams.setObjectName('panel')
        sl = QVBoxLayout(seams)
        sl.setContentsMargins(8, 4, 8, 4)
        sl.setSpacing(2)
        head = QHBoxLayout()
        head.addWidget(label('Stitch seams'))
        head.addStretch(1)
        self.seam_note = label('', muted=True, small=True)
        head.addWidget(self.seam_note)
        sl.addLayout(head)
        sl.addLayout(self._objects_bar())
        self.table = QTableWidget(5, 1)
        self.table.verticalHeader().setDefaultSectionSize(ROW_H)
        self.table.horizontalHeader().setFixedHeight(ROW_H)
        self.table.setShowGrid(False)
        self.table.setEditTriggers(QAbstractItemView.EditTrigger.DoubleClicked)
        self.table.verticalHeader().setFixedWidth(158)
        self.table.setVerticalHeaderLabels(['Angle °', 'Blend m', 'Density at seam px/cm', 'Object vanishes m²',
                                            'Advised angle °'])
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        self.table.setFixedHeight(ROW_H * 6 + 4)
        self.table.itemChanged.connect(self._edited)
        sl.addWidget(self.table)
        ll.addWidget(seams)
        split.addWidget(left)
        right = QSplitter(Qt.Orientation.Vertical)
        right.setHandleWidth(2)
        self.bev = PlanView(ctx, 'bev')
        self.bev.legend = False
        self.bev.show_objects = self.obj_check.isChecked()
        self.bev.seamMoved.connect(lambda: None)
        right.addWidget(self.bev)
        self.bowl_scene = MainScene(ctx, mode='bowl')
        self.bowl = make_viewport(self.bowl_scene)
        self.bowl.cam.az, self.bowl.cam.el = 235.0, 38.0
        bw = QWidget()
        bw.setObjectName('panel')
        bl = QVBoxLayout(bw)
        bl.setContentsMargins(0, 0, 0, 0)
        bl.setSpacing(0)
        bl.addWidget(self._bowl_bar())
        bl.addWidget(self.bowl, 1)
        right.addWidget(bw)
        split.addWidget(right)
        split.setSizes([1230, 770])
        lay.addWidget(split)
        self.views = []
        self._uids = None
        self._busy = False
        ctx.store.changed.connect(lambda *_: self.rebuild())
        ctx.store.project_replaced.connect(self.rebuild)
        ctx.analysis.finished.connect(lambda *_: self.refresh())

    # ------------------------------------------------------------------ standing objects in the stitched view
    def _objects_bar(self):
        from ...core.stitched_objects import OBJECTS
        from ..widgets import button
        ctl = self.ctx.stitched
        row = QHBoxLayout()
        row.setSpacing(6)
        self.obj_check = QCheckBox('Standing objects')
        self.obj_check.setToolTip('Show in the bird’s-eye view where a standing object vanishes at a seam (hatched) or '
                                  'is cut off at the view’s edge (dotted), with the smears of a pole at the worst places')
        try:
            on = bool(self.ctx.settings.value('camviews/objects', False, type=bool))
        except Exception:
            on = False
        self.obj_check.setChecked(on)
        self.obj_check.toggled.connect(self._show_objects)
        row.addWidget(self.obj_check)
        self.obj_combo = QComboBox()
        for _, name, h in OBJECTS:
            self.obj_combo.addItem(f'{name} {h:g} m', h)
        self.obj_combo.addItem('Custom height…', None)
        self.obj_combo.setToolTip('The object stood on every point of the ROI (a pole 0.3 m across)')
        self.obj_combo.activated.connect(self._object_picked)
        row.addWidget(self.obj_combo)
        self.thr_combo = QComboBox()
        for f in (0.10, 0.25, 0.50):
            self.thr_combo.addItem(f'lost below {f * 100:.0f} % shown', f)
        self.thr_combo.setToolTip('An object counts as vanished where the stitched view shows less than this share of '
                                  'its height although a camera sees it')
        self.thr_combo.activated.connect(lambda i: ctl.set_settings(threshold=self.thr_combo.itemData(i)))
        row.addWidget(self.thr_combo)
        self.b_suggest = button('Suggest seams', None, None, 'Look for seam angles that make the object vanish on less '
                                                              'ground without more ground below the minimum density')
        self.b_suggest.clicked.connect(ctl.suggest)
        row.addWidget(self.b_suggest)
        self.b_apply = button('Apply', None, None, 'Move the seams to the advised angles (one undo step)')
        self.b_apply.clicked.connect(ctl.apply_advice)
        self.b_apply.setEnabled(False)
        row.addWidget(self.b_apply)
        self.adv_note = label('', muted=True, small=True)
        row.addWidget(self.adv_note, 1)
        ctl.ready.connect(self.refresh)
        ctl.advice_ready.connect(self._advice_changed)
        self.ctx.store.changed.connect(lambda *_: self._sync_objects())
        self.ctx.store.project_replaced.connect(self._sync_objects)
        self._sync_objects()
        return row

    def _sync_objects(self, *_):
        st = self.ctx.stitched.settings          # follows undo, redo and opening another project
        i = next((k for k in range(self.obj_combo.count() - 1) if abs(self.obj_combo.itemData(k) - st.height) < 1e-6), None)
        if i is None:
            self.obj_combo.setItemText(self.obj_combo.count() - 1, f'Custom {st.height:g} m…')
            i = self.obj_combo.count() - 1
        else:
            self.obj_combo.setItemText(self.obj_combo.count() - 1, 'Custom height…')
        self.obj_combo.setCurrentIndex(i)
        j = min(range(self.thr_combo.count()), key=lambda k: abs(self.thr_combo.itemData(k) - st.threshold))
        self.thr_combo.setCurrentIndex(j)

    def _object_picked(self, i):
        ctl = self.ctx.stitched
        h = self.obj_combo.itemData(i)
        if h is None:
            from PyQt6.QtWidgets import QInputDialog
            v, ok = QInputDialog.getDouble(self, 'Standing object', 'Height of the object, m', ctl.settings.height, 0.05,
                                           5.0, 2)
            if not ok:
                self._sync_objects()
                return
            h = v
        ctl.set_settings(height=h)
        self._sync_objects()

    def _show_objects(self, on):
        self.bev.show_objects = on
        try:
            self.ctx.settings.setValue('camviews/objects', on)
        except Exception:
            pass
        self.bev.update()

    def _seam_objects(self, s):
        """(object vanish m² at this seam, advised angle) as table text."""
        ctl = self.ctx.stitched
        van = '—'
        if ctl.result is not None:
            for x in ctl.result.seams:
                if x['a'] == s.a and x['b'] == s.b:
                    van = f'{x["vanish_m2"]:.2f}'
        adv = '—'
        if ctl.advice is not None:
            for a in ctl.advice.seams:
                if a.a == s.a and a.b == s.b and a.moved:
                    adv = f'{a.best:.1f}'
        return van, adv

    def _advice_changed(self):
        ctl = self.ctx.stitched
        a = ctl.advice
        if ctl.advising:
            self.adv_note.setText('looking for better seam angles…')
        elif a is None:
            self.adv_note.setText('')
        elif a.changes:
            pct = 100 * (a.vanish_after - a.vanish_now) / max(a.vanish_now, 1e-9)
            n = len(a.changes)
            self.adv_note.setText(f'{n} seam{"s" if n != 1 else ""} · vanish area {pct:+.0f} %'.replace('-', '−'))
            self.adv_note.setToolTip(f'On a 10 cm map: the object vanishes on {a.vanish_now:.2f} → {a.vanish_after:.2f} m²; '
                                     f'ground below the minimum density {a.ground_low_now:.2f} → {a.ground_low_after:.2f} m²')
        else:
            self.adv_note.setText('no seam angle within ±15° does better')
        self.b_apply.setEnabled(bool(a is not None and a.changes))
        self.b_suggest.setEnabled(not ctl.advising)
        self.refresh()

    # ------------------------------------------------------------------ 3D surround view
    def _bowl_bar(self):
        from ..bowlview import SOURCES
        from ..widgets import ScrubSpinBox, button
        ctl = self.bowl_scene.bowl
        w = QWidget()
        w.setObjectName('panel')
        g = QGridLayout(w)
        g.setContentsMargins(8, 4, 8, 4)
        g.setHorizontalSpacing(6)
        g.setVerticalSpacing(3)
        g.addWidget(label('3D surround view'), 0, 0)
        self.bowl_src = QComboBox()
        for k, t in SOURCES:
            self.bowl_src.addItem(t, k)
        self.bowl_src.setToolTip('What is projected onto the bowl: a synthetic parking scene rendered through every camera, '
                                 'the installation check\'s captures, or just each camera\'s sector')
        self.bowl_src.currentIndexChanged.connect(lambda _=0: ctl.set_source(self.bowl_src.currentData()))
        g.addWidget(self.bowl_src, 0, 1, 1, 2)
        self.bowl_views = QComboBox()
        for k in ('Rear left', 'Rear right', 'Front left', 'Front right', 'Behind', 'Ahead', 'Top'):
            self.bowl_views.addItem(k)
        self.bowl_views.setToolTip('Typical in-car display viewpoints')
        self.bowl_views.activated.connect(lambda i: self.bowl_view(self.bowl_views.itemText(i)))
        g.addWidget(self.bowl_views, 0, 3)
        self.bowl_balance = QCheckBox('Balance exposure')
        self.bowl_balance.setChecked(True)
        self.bowl_balance.toggled.connect(lambda on: ctl.set_options(balance=on))
        g.addWidget(self.bowl_balance, 0, 4)
        b = button('Export…', None, 'export', 'Bowl mesh (OBJ), its per-vertex remap tables and the texture')
        b.clicked.connect(self.export_bowl)
        g.addWidget(b, 0, 5)
        self.bowl_fields = {}
        sh = ctl.shape
        for i, (key, text, unit, step) in enumerate((('floor_m', 'Floor', 'm', 0.1), ('height_m', 'Wall', 'm', 0.1),
                                                     ('reach_m', 'Reach', 'm', 0.1), ('power', 'Profile', '', 0.1),
                                                     ('squareness', 'Square', '', 0.5))):
            f = ScrubSpinBox('', 'plain', 1, unit, step=step, width=74)
            f.set_value(getattr(sh, key))
            f.setToolTip({'floor_m': 'Flat floor beyond the vehicle outline', 'height_m': 'Wall height at the rim',
                          'reach_m': 'How far out the wall reaches', 'power': 'Wall curvature: 1 = straight cone, 2 = parabola',
                          'squareness': 'Footprint: 2 = ellipse, higher = closer to the vehicle\'s rectangle'}[key])
            f.committed.connect(lambda v, k=key: ctl.set_shape(**{k: max(0.0 if k == 'floor_m' else 0.2, v)}))
            box = QWidget()
            hb = QHBoxLayout(box)
            hb.setContentsMargins(0, 0, 0, 0)
            hb.setSpacing(4)
            hb.addWidget(label(text, ink2=True))
            hb.addWidget(f)
            g.addWidget(box, 1, i)
            self.bowl_fields[key] = f
        self.bowl_status = label('', muted=True, small=True)
        g.addWidget(self.bowl_status, 1, 5)
        ctl.ready.connect(lambda: self.bowl_status.setText(ctl.status))
        self.ctx.store.changed.connect(lambda *_: self._sync_bowl_fields())
        self.ctx.store.project_replaced.connect(self._sync_bowl_fields)
        return w

    def _sync_bowl_fields(self, *_):
        sh = self.bowl_scene.bowl.shape          # follows undo, redo and opening another project
        for key, f in self.bowl_fields.items():
            if not f.hasFocus():
                f.set_value(getattr(sh, key))

    def bowl_view(self, name):
        """Orbit the bowl camera to a typical display viewpoint (eye behind and above the car, looking at it)."""
        az, el, d = {'Rear left': (205.0, 28.0, 1.0), 'Rear right': (155.0, 28.0, 1.0), 'Front left': (-25.0, 28.0, 1.0),
                     'Front right': (25.0, 28.0, 1.0), 'Behind': (180.0, 22.0, 0.9), 'Ahead': (0.0, 22.0, 0.9),
                     'Top': (-90.0, 89.0, 1.3)}[name]
        res = self.ctx.analysis.result
        tgt = None
        if res is not None:
            tgt = (res.centre[0], res.centre[1], 0.4)
        self.bowl.animate_to(az=az, el=el, dist=7.5 * d, target=tgt, ortho=False)

    def export_bowl(self, out_dir=None):
        from PyQt6.QtWidgets import QFileDialog
        import os
        from ...core import bowl as bowl_core
        res = self.ctx.analysis.result
        if res is None:
            self.ctx.msg('Run the analysis first (F5).')
            return None
        if not out_dir:
            out_dir = QFileDialog.getExistingDirectory(self, 'Write the bowl into')
        if not out_dir:
            return None
        ctl = self.bowl_scene.bowl
        files = bowl_core.export_files(self.ctx.store.project, self.ctx.window._shared_engine(), res, ctl.shape, ctl.texture)
        d = os.path.join(out_dir, 'bowl')
        os.makedirs(d, exist_ok=True)
        paths = []
        for n, data in files:
            pth = os.path.join(d, n)
            with open(pth, 'wb') as fh:
                fh.write(data)
            paths.append(pth)
        self.ctx.msg(f'Wrote {len(paths)} files to {d}', 6000)
        return paths

    def showEvent(self, e):
        super().showEvent(e)
        self.bowl_scene.bowl.set_active(True)

    def hideEvent(self, e):
        self.bowl_scene.bowl.set_active(False)
        super().hideEvent(e)

    def activate(self):
        self.rebuild()
        self.bev.fit()
        res = self.ctx.analysis.result
        if res is not None:
            import numpy as np
            g = res.grid
            lo = np.array([g.x0, g.y0, 0.0])
            hi = np.array([g.x0 + g.nx * g.cell, g.y0 + g.ny * g.cell, 1.0])
            QTimer.singleShot(30, lambda: self.bowl.frame_box(lo, hi, 0.95))

    def rebuild(self):
        cams = self.ctx.store.variant.cameras
        uids = [c.uid for c in cams]
        if uids == self._uids:
            self.refresh()
            return
        self._uids = uids
        for v in self.views:
            self.grid.removeWidget(v)
            v.deleteLater()
        self.views = []
        cols = 3 if len(cams) <= 6 else 4
        for i, c in enumerate(cams):
            v = CamImageView(self.ctx, uid=c.uid, header=True, legend=False)
            self.grid.addWidget(v, i // cols, i % cols)
            self.views.append(v)
        self.refresh()

    def refresh(self):
        res = self.ctx.analysis.result
        n = len(self.ctx.store.variant.cameras)
        self.bev.label = f'Bird’s-eye · stitched from {n} cameras · drag a handle to move a seam'
        self.bev.update()
        if res is None:
            return
        self._busy = True
        st = self.ctx.store
        seams = res.seams
        self.table.setColumnCount(len(seams))
        heads = []
        for j, s in enumerate(seams):
            a, b = st.cam(s.a), st.cam(s.b)
            heads.append(f'{a.tag if a else "?"} | {b.tag if b else "?"}')
            van, adv = self._seam_objects(s)
            for i, val in enumerate((f'{s.angle:.1f}', f'{st.variant.blend_m:.2f}', f'{s.density:.1f}', van, adv)):
                it = QTableWidgetItem(val.replace('-', '−'))
                it.setFont(mono(12))
                it.setTextAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
                if i >= 2:
                    it.setFlags(it.flags() & ~Qt.ItemFlag.ItemIsEditable)
                if i == 4 and val != '—':
                    it.setForeground(THEME.color('ink'))
                    it.setFont(mono(12, QFont.Weight.DemiBold))
                self.table.setItem(i, j, it)
        self.table.setHorizontalHeaderLabels(heads)
        cx, cy = res.centre
        o = self.ctx.store.project.origin_offset()
        self.seam_note.setText(f'angles from the body centre ({cx - o[0]:.3f}, {cy - o[1]:.3f}) · drag the ■ handles in the bird’s-eye view')
        self._busy = False

    def _edited(self, it):
        if self._busy:
            return
        res = self.ctx.analysis.result
        if res is None or it.column() >= len(res.seams):
            return
        s = res.seams[it.column()]
        try:
            v = float(it.text().replace('−', '-'))
        except ValueError:
            v = math.nan
        if not math.isfinite(v):
            self.refresh()
            return
        st = self.ctx.store
        if it.row() == 0:
            st.edit('variant', lambda p: p.variant.seams.__setitem__(f'{s.a}|{s.b}', round(v, 1)), 'Seam angle')
        elif it.row() == 1:
            st.edit('variant', lambda p: setattr(p.variant, 'blend_m', max(0.0, v)), 'Seam blend width')
        self.ctx.analysis.request()
