"""Inspector pages for reference photos and the calibration target planner."""
from __future__ import annotations

import json
import math

import numpy as np
from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (QWidget, QHBoxLayout, QVBoxLayout, QLabel, QComboBox, QCheckBox, QMenu, QPushButton,
                             QAbstractItemView, QFileDialog)

from ..core import photos as ph
from ..core.mathutil import parse_number
from ..core.project import Target
from ..core.targets import suggest, auto_place, evaluate_targets
from . import icons
from .inspector import Page, _table, _fit_table, _cell, QColorHex
from .theme import THEME, STATUS, mono
from .widgets import Section, ValueLabel, StatusLine, ScrubSpinBox, label, button, hline, ROW_H, LABEL_W


def triangulate(ps, name):
    """Least-squares intersection of the viewing rays of one landmark over all solved frames."""
    A = np.zeros((3, 3))
    b = np.zeros(3)
    n = 0
    for k, fr in enumerate(ps.frames):
        pose = ph.frame_pose(ps, k)
        if pose is None:
            continue
        for nm, u, v in fr.points:
            if nm != name:
                continue
            R, t = pose
            C = -R.T @ t
            x = (u - ps.cx) / ps.f_px
            y = (v - ps.cy) / ps.f_px
            d = R.T @ np.array([x, y, 1.0])
            d /= np.linalg.norm(d)
            P = np.eye(3) - np.outer(d, d)
            A += P
            b += P @ C
            n += 1
    if n < 2:
        return None
    try:
        return np.linalg.solve(A, b)
    except np.linalg.LinAlgError:
        return None


class PhotosPage(Page):
    title = 'Reference photos'

    def __init__(self, ctx):
        super().__init__(ctx)
        s = Section('Calibration')
        self.v = {}
        for k in ('Frames', 'RMS residual', 'Worst frame', 'Photo focal', 'Distortion', 'Wheelbase', 'Method', 'Solved'):
            self.v[k] = ValueLabel()
            s.row(k, self.v[k])
        bl = QHBoxLayout()
        b1 = button('Load spin set…')
        b1.clicked.connect(lambda: ctx.window.load_photos())
        b2 = button('Calibrate')
        b2.clicked.connect(lambda: ctx.window.calibrate_photos())
        bl.addWidget(b1)
        bl.addWidget(b2)
        bl.addStretch(1)
        s.blay.addLayout(bl)
        self.lay.addWidget(s)
        self.lay.addWidget(hline())
        self.s_c = Section('Frame correspondences')
        self.t = _table(['Point', 'Error px'], 0)
        self.t.horizontalHeader().setStretchLastSection(False)
        from PyQt6.QtWidgets import QHeaderView
        self.t.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeMode.Stretch)
        self.s_c.add(self.t)
        self.s_c.add(label('In the photo pane: pick a landmark below, then click its position in the photo. '
                           'Right-click a point in the photo to remove it.', muted=True, small=True, wrap=True))
        self.lm = QComboBox()
        self.lm.currentTextChanged.connect(lambda t: setattr(ctx, 'photo_landmark', t))
        self.s_c.row('Landmark', self.lm)
        bl = QHBoxLayout()
        b3 = button('Add landmark from model…')
        b3.clicked.connect(lambda: ctx.window.add_landmark())
        bl.addWidget(b3)
        bl.addStretch(1)
        self.s_c.blay.addLayout(bl)
        self.lay.addWidget(self.s_c)
        self.lay.addWidget(hline())
        so = Section('Display')
        self.show_pane = QCheckBox('Show photo pane')
        self.show_pane.toggled.connect(lambda on: ctx.window.show_photo_page(on) if not self._busy else None)
        self.matched = QCheckBox('Photo-matched shading in 3D')
        self.matched.toggled.connect(self._matched)
        so.add(self.show_pane)
        so.add(self.matched)
        self.lay.addWidget(so)
        self.lay.addStretch(1)
        self._busy = False
        ctx.photo_landmark = ''

    def _matched(self, on):
        if self._busy:
            return
        self.ctx.photo_shading = on
        self.ctx.overlays_changed.emit()

    def refresh(self):
        self._busy = True
        ps = self.project.photos
        n = len(ps.frames)
        self.v['Frames'].set(str(n), '', None, '' if ps.source else 'no source loaded')
        self.v['RMS residual'].set(f'{ps.rms:.2f}', 'px')
        k, w = ph.worst_frame(ps)
        self.v['Worst frame'].set(f'{k:02d} · {w:.2f}' if k >= 0 else '—', 'px' if k >= 0 else '')
        self.v['Photo focal'].set(f'{ph.photo_focal_35mm(ps):.0f}' if ps.width else '—', 'mm eq.' if ps.width else '')
        self.v['Distortion'].set(f'k1 {ps.k1:+.3f}'.replace('-', '−'))
        a, b = triangulate(ps, 'Wheel centre FL'), triangulate(ps, 'Wheel centre RL')
        if a is None or b is None:
            a, b = triangulate(ps, 'Wheel centre FR'), triangulate(ps, 'Wheel centre RR')
        if a is not None and b is not None:
            wb = abs(a[0] - b[0])
            spec = self.project.vehicle.spec.get('Wheelbase')
            self.v['Wheelbase'].set(f'{wb:.3f}', 'm', 'good' if spec and abs(wb - spec) < 0.03 else 'warning' if spec else None)
            self.v['Wheelbase'].setToolTip(f'Scale check: wheelbase triangulated from the photos, spec {spec:.3f} m' if spec else '')
        else:
            self.v['Wheelbase'].set('—', '', None, 'needs 2+ views per wheel')
        meth = ps.method or '—'
        self.v['Method'].set('turntable' if meth.lower().startswith('turntable') else 'landmarks' if meth != '—' else '—')
        self.v['Method'].setToolTip(meth)
        self.v['Solved'].set(ps.solved_at or '—')
        cur = ps.current if 0 <= ps.current < n else 0
        self.s_c.header.title.setText(f'Frame {cur:02d} correspondences')
        errs = ph.point_errors(ps, cur) if n else []
        self.t.setRowCount(len(errs))
        for i, (name, u, v, q, e) in enumerate(errs):
            self.t.setItem(i, 0, _cell(name))
            self.t.setItem(i, 1, _cell(f'{e:.2f}' if math.isfinite(e) else '—', True, True))
        _fit_table(self.t)
        self.s_c.header.summary.setText(str(len(errs)))
        cur_lm = self.lm.currentText()
        self.lm.blockSignals(True)
        self.lm.clear()
        self.lm.addItems(sorted(ps.landmarks.keys()))
        if cur_lm:
            self.lm.setCurrentText(cur_lm)
        self.lm.blockSignals(False)
        self.ctx.photo_landmark = self.lm.currentText()
        self.show_pane.setChecked(self.ctx.window is not None and self.ctx.window.page_name() == 'photo')
        self.matched.setChecked(self.ctx.photo_shading)
        self._busy = False


class TargetsPage(Page):
    title = 'Target planner'

    def __init__(self, ctx):
        super().__init__(ctx)
        self.sugg = []
        top = QWidget()
        top.setFixedHeight(32)
        h = QHBoxLayout(top)
        h.setContentsMargins(8, 4, 8, 4)
        h.setSpacing(4)
        self.b_add = QPushButton(' Add target')
        self.b_add.setIcon(icons.icon('plus', 12))
        m = QMenu(self.b_add)
        for kind, lab in (('checkerboard', 'Checkerboard'), ('charuco', 'ChArUco'), ('aprilgrid', 'AprilGrid')):
            m.addAction(lab, lambda k=kind: self._add(k))
        self.b_add.setMenu(m)
        b_auto = button('Auto-place')
        b_auto.clicked.connect(self._auto)
        b_check = button('Check')
        b_check.clicked.connect(self._check)
        h.addWidget(self.b_add)
        h.addWidget(b_auto)
        h.addWidget(b_check)
        h.addStretch(1)
        self.lay.addWidget(top)
        s = Section('Pattern')
        self.pattern = QComboBox()
        for k, t in (('checkerboard', 'Checkerboard'), ('charuco', 'ChArUco'), ('aprilgrid', 'AprilGrid')):
            self.pattern.addItem(t, k)
        self.pattern.activated.connect(lambda i: self._ts('pattern', self.pattern.itemData(i)))
        self.rows = ScrubSpinBox('', 'plain', 0, '×', step=1)
        self.rows.committed.connect(lambda v: self._ts('rows', int(max(2, v))))
        self.cols = ScrubSpinBox('', 'plain', 0, '', step=1)
        self.cols.committed.connect(lambda v: self._ts('cols', int(max(2, v))))
        s.row('Pattern', self.pattern, self.rows, self.cols)
        self.square = ScrubSpinBox('', 'plain', 0, 'mm', step=5)
        self.square.committed.connect(lambda v: self._ts('square_mm', max(10.0, v)))
        s.row('Square', self.square)
        self.cpt = ScrubSpinBox('≥', 'plain', 0, '', step=1)
        self.cpt.committed.connect(lambda v: self._ts('cams_per_target', int(max(1, v))))
        s.row('Cameras per target', self.cpt)
        self.dob = ScrubSpinBox('≥', 'plain', 1, 'px/cm', step=0.1)
        self.dob.committed.connect(lambda v: self._ts('min_density', max(0.1, v)))
        s.row('Density on board', self.dob)
        self.lay.addWidget(s)
        self.t = _table(['#', 'X m', 'Y m', 'Seen by', ''], 3)
        self.t.cellClicked.connect(self._row_clicked)
        self.t.setEditTriggers(QAbstractItemView.EditTrigger.DoubleClicked)
        self.t.itemChanged.connect(self._edited)
        tw = QWidget()
        tl = QVBoxLayout(tw)
        tl.setContentsMargins(8, 4, 8, 4)
        tl.addWidget(self.t)
        self.status = StatusLine('good', '')
        tl.addWidget(self.status)
        self.lay.addWidget(tw)
        ss = Section('Suggested')
        self.sugg_box = QVBoxLayout()
        self.sugg_box.setSpacing(0)
        ss.blay.addLayout(self.sugg_box)
        self.lay.addWidget(ss)
        self.lay.addStretch(1)
        bl = QHBoxLayout()
        bl.setContentsMargins(8, 0, 8, 6)
        b_exp = button('Export positions…')
        b_exp.clicked.connect(self._export)
        self.b_apply = button('Apply suggestions', 'primary')
        self.b_apply.clicked.connect(self._apply)
        bl.addStretch(1)
        bl.addWidget(b_exp)
        bl.addWidget(self.b_apply)
        self.lay.addLayout(bl)
        self._busy = False

    def _ts(self, k, v):
        def fn(p):
            setattr(p.target_settings, k, v)
            if k in ('rows', 'cols', 'square_mm', 'pattern'):
                for t in p.targets:
                    setattr(t, k, v)
        self.ctx.store.edit('targets', fn, f'Target {k}')
        self.ctx.analysis.request()

    def _add(self, kind):
        p = self.project
        st = p.target_settings
        n = len(p.targets) + 1
        res = self.ctx.analysis.result
        x = (res.grid.x0 + res.grid.nx * res.grid.cell) - 0.8 if res else 4.0

        def fn(pj):
            pj.targets.append(Target(name=f'T{n}', x=round(x, 2), y=0.0, pattern=kind, rows=st.rows, cols=st.cols, square_mm=st.square_mm))
        self.ctx.store.edit('targets', fn, f'Added target T{n}')
        self.ctx.analysis.request()

    def _auto(self):
        res = self.ctx.analysis.result
        if res is None:
            return
        with self.ctx.analysis._lock:
            ts = auto_place(self.ctx.analysis.engine, self.project.snapshot(), res)
        if not ts:
            self.ctx.msg('No spot found where two neighbouring cameras both see a board.')
            return
        self.ctx.store.edit('targets', lambda p: setattr(p, 'targets', ts), f'Auto-placed {len(ts)} targets')
        self.ctx.analysis.request()

    def _check(self):
        res = self.ctx.analysis.result
        if res is None:
            return
        with self.ctx.analysis._lock:
            snap = self.project.snapshot()
            stt = evaluate_targets(self.ctx.analysis.engine, snap)
            self.sugg = suggest(self.ctx.analysis.engine, snap, res, stt)
        self.ctx.analysis.targets = stt
        if self.ctx.window:
            self.ctx.window.set_target_suggestions(self.sugg)
        self.refresh()

    def _apply(self):
        sugg = list(self.sugg)
        if not sugg:
            return

        def fn(p):
            for uid, x, y, _ in sugg:
                for t in p.targets:
                    if t.uid == uid:
                        t.x, t.y = x, y
        self.ctx.store.edit('targets', fn, f'Applied {len(sugg)} target suggestions')
        self.sugg = []
        if self.ctx.window:
            self.ctx.window.set_target_suggestions([])
        self.ctx.analysis.request()

    def _row_clicked(self, r, c):
        if 0 <= r < len(self.project.targets):
            self.ctx.store.select_obj(('target', self.project.targets[r].uid))

    def _edited(self, it):
        if self._busy or it.column() not in (1, 2):
            return
        r = it.row()
        try:
            v = parse_number(it.text(), 'length')        # rejects nan / inf
        except ValueError:
            self.refresh()
            return
        o = self.project.origin_offset()
        k = 'x' if it.column() == 1 else 'y'
        uid = self.project.targets[r].uid

        def fn(p):
            for t in p.targets:
                if t.uid == uid:
                    setattr(t, k, round(v + o[0 if k == 'x' else 1], 3))
        self.ctx.store.edit('targets', fn, f'Moved {self.project.targets[r].name}')
        self.ctx.analysis.request()

    def _export(self):
        path, _ = QFileDialog.getSaveFileName(self, 'Export target positions', 'targets.csv', 'CSV (*.csv);;JSON (*.json)')
        if not path:
            return
        p = self.project
        o = p.origin_offset()
        st = {s.uid: s for s in self.ctx.analysis.targets}
        rows = []
        for t in p.targets:
            s = st.get(t.uid)
            rows.append({'name': t.name, 'x': round(t.x - o[0], 3), 'y': round(t.y - o[1], 3), 'yaw': t.yaw, 'pattern': t.pattern,
                         'rows': t.rows, 'cols': t.cols, 'square_mm': t.square_mm,
                         'seen_by': [x[1] for x in s.seen_by] if s else []})
        if path.endswith('.json'):
            with open(path, 'w') as fh:
                json.dump({'frame': p.frame_label, 'targets': rows}, fh, indent=1)
        else:
            with open(path, 'w') as fh:
                fh.write('name,x_m,y_m,yaw_deg,pattern,rows,cols,square_mm,seen_by\n')
                for r in rows:
                    fh.write(f"{r['name']},{r['x']:.3f},{r['y']:.3f},{r['yaw']:.1f},{r['pattern']},{r['rows']},{r['cols']},"
                             f"{r['square_mm']:.0f},{' '.join(r['seen_by'])}\n")
        self.ctx.msg(f'Exported {len(rows)} target positions to {path}')

    def refresh(self):
        self._busy = True
        p = self.project
        st = p.target_settings
        self.pattern.setCurrentIndex(max(0, self.pattern.findData(st.pattern)))
        self.rows.set_value(st.rows)
        self.cols.set_value(st.cols)
        self.square.set_value(st.square_mm)
        self.cpt.set_value(st.cams_per_target)
        self.dob.set_value(st.min_density)
        o = p.origin_offset()
        stt = {s.uid: s for s in self.ctx.analysis.targets}
        self.t.setRowCount(len(p.targets))
        sel = self.ctx.store.sel_obj
        n_ok = 0
        bad = []
        for i, t in enumerate(p.targets):
            s = stt.get(t.uid)
            name = _cell(t.name, True)
            name.setFlags(name.flags() & ~Qt.ItemFlag.ItemIsEditable)
            self.t.setItem(i, 0, name)
            self.t.setItem(i, 1, _cell(f'{t.x - o[0]:.3f}', True, True))
            self.t.setItem(i, 2, _cell(f'{t.y - o[1]:.3f}', True, True))
            seen = ' '.join(x[1] for x in s.seen_by) if s else '…'
            c = _cell(seen, True)
            c.setFlags(c.flags() & ~Qt.ItemFlag.ItemIsEditable)
            self.t.setItem(i, 3, c)
            ok = s.ok if s else True
            n_ok += ok
            if s and not ok:
                bad.append(t.name)
            ic = _cell('')
            ic.setIcon(icons.status_icon('good' if ok else 'critical', 12))
            ic.setFlags(ic.flags() & ~Qt.ItemFlag.ItemIsEditable)
            self.t.setItem(i, 4, ic)
            if sel == ('target', t.uid):
                self.t.selectRow(i)
        _fit_table(self.t)
        if p.targets:
            if bad:
                self.status.set('critical', f'Critical · {n_ok} of {len(p.targets)} targets seen by {st.cams_per_target} cameras. '
                                            f'{", ".join(bad)} {"is" if len(bad) == 1 else "are"} seen by fewer cameras, so extrinsics '
                                            'for those pairs can\'t be chained.')
            else:
                self.status.set('good', f'All {len(p.targets)} targets are seen by {st.cams_per_target}+ cameras.')
        else:
            self.status.set('warning', 'No targets yet. Use Auto-place or Add target.')
        while self.sugg_box.count():
            w = self.sugg_box.takeAt(0).widget()
            if w:
                w.hide()
                w.setParent(None)
                w.deleteLater()
        names = {t.uid: t.name for t in p.targets}
        for uid, x, y, pair in self.sugg:
            r = QWidget()
            r.setFixedHeight(ROW_H)
            h = QHBoxLayout(r)
            h.setContentsMargins(0, 0, 0, 0)
            a = label(names.get(uid, '?'), mono_=True)
            a.setFixedWidth(30)
            b = label(f'{x - o[0]:.3f}  {y - o[1]:.3f}', mono_=True)
            c = label(f'{pair[0]} {pair[1]}', mono_=True)
            h.addWidget(a)
            h.addWidget(b, 1)
            h.addWidget(c)
            self.sugg_box.addWidget(r)
        if not self.sugg:
            self.sugg_box.addWidget(label('Press Check to compute suggestions.', muted=True, small=True))
        self.b_apply.setText(f'Apply {len(self.sugg)} suggestion{"s" if len(self.sugg) != 1 else ""}')
        self.b_apply.setEnabled(bool(self.sugg))
        self._busy = False
