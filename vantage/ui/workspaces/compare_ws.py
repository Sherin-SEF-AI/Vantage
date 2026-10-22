"""B4 · Compare variants."""
from __future__ import annotations

import threading

import numpy as np
from PyQt6.QtCore import Qt, QTimer, pyqtSignal, QObject
from PyQt6.QtGui import QPainter, QColor, QFont
from PyQt6.QtWidgets import (QWidget, QHBoxLayout, QVBoxLayout, QSplitter, QComboBox, QTableWidget, QTableWidgetItem,
                             QHeaderView, QAbstractItemView, QFileDialog)

from ..planview import PlanView
from ..theme import THEME, mono, sans, STATUS, DIVERGING
from .. import icons
from ..widgets import label, button, ROW_H, StatusLine


class _Sig(QObject):
    ready = pyqtSignal(str, object)
    worst = pyqtSignal(str, object)


class DivergingLegend(QWidget):
    def __init__(self):
        super().__init__()
        self.setFixedHeight(30)

    def paintEvent(self, e):
        p = QPainter(self)
        n = len(DIVERGING)
        w = self.width() - 20
        for i, c in enumerate(DIVERGING):
            col = QColor(c if not (THEME.name == 'light' and i == 3) else '#e1e0d9')
            p.fillRect(int(10 + i * w / n), 4, int(w / n) + 1, 7, col)
        p.setFont(mono(11))
        p.setPen(THEME.color('muted'))
        p.drawText(10, 26, 'worse −4')
        p.drawText(int(self.width() / 2 - 30), 26, 'no change')
        p.drawText(self.width() - 80, 26, 'better +4')
        p.end()


class ComparePage(QWidget):
    def __init__(self, ctx):
        super().__init__()
        self.ctx = ctx
        self.setObjectName('base')
        self.results = {}
        self.worsts = {}
        self._keys = {}
        self.sig = _Sig()
        self.sig.ready.connect(self._ready)
        self.sig.worst.connect(self._worst)
        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(2)
        top = QSplitter(Qt.Orientation.Horizontal)
        top.setHandleWidth(2)
        self.cols = []
        for k in range(3):
            w = QWidget()
            w.setObjectName('panel')
            l = QVBoxLayout(w)
            l.setContentsMargins(0, 0, 0, 0)
            l.setSpacing(0)
            head = QWidget()
            head.setFixedHeight(26)
            h = QHBoxLayout(head)
            h.setContentsMargins(8, 0, 8, 0)
            combo = QComboBox()
            combo.setFixedWidth(190)
            tags = label('', muted=True, small=True)
            status = label('', mono_=True)
            h.addWidget(combo)
            h.addWidget(tags)
            h.addStretch(1)
            h.addWidget(status)
            l.addWidget(head)
            pv = PlanView(ctx, 'compare', interactive=False)
            pv.legend = False
            l.addWidget(pv, 1)
            top.addWidget(w)
            col = {'combo': combo, 'tags': tags, 'status': status, 'plan': pv, 'uid': None}
            pv.result_fn = (lambda c=col: self.results.get(c['uid']))
            pv.cameras_fn = (lambda c=col: (self.ctx.store.project.variant_by_uid(c['uid']).cameras
                                             if self.ctx.store.project.variant_by_uid(c['uid']) else []))
            combo.activated.connect(lambda i, c=col: self._choose(c, c['combo'].itemData(i)))
            self.cols.append(col)
        lay.addWidget(top, 3)
        bottom = QSplitter(Qt.Orientation.Horizontal)
        bottom.setHandleWidth(2)
        dl = QWidget()
        dl.setObjectName('panel')
        dll = QVBoxLayout(dl)
        dll.setContentsMargins(0, 0, 0, 4)
        dh = QHBoxLayout()
        dh.setContentsMargins(8, 4, 8, 0)
        self.delta_title = label('')
        dh.addWidget(self.delta_title)
        dh.addStretch(1)
        dh.addWidget(label('Δ best-camera density, px/cm', muted=True, small=True))
        dll.addLayout(dh)
        self.delta = PlanView(ctx, 'delta', interactive=False)
        self.delta.legend = False
        self.delta.show_tooltip = False
        self.delta.delta_fn = lambda: (self.results.get(self.cols[0]['uid']), self.results.get(self.cols[1]['uid']))
        self.delta.result_fn = lambda: self.results.get(self.cols[0]['uid']) if self.results.get(self.cols[1]['uid']) is not None else None
        self.delta.cameras_fn = lambda: (self.ctx.store.project.variant_by_uid(self.cols[1]['uid']).cameras
                                         if self.ctx.store.project.variant_by_uid(self.cols[1]['uid']) else [])
        dll.addWidget(self.delta, 1)
        dll.addWidget(DivergingLegend())
        bottom.addWidget(dl)
        tw = QWidget()
        tw.setObjectName('panel')
        tl = QVBoxLayout(tw)
        tl.setContentsMargins(8, 4, 8, 6)
        th = QHBoxLayout()
        th.addWidget(label('Difference table'))
        th.addStretch(1)
        self.ref_lbl = label('', muted=True, small=True)
        th.addWidget(self.ref_lbl)
        tl.addLayout(th)
        self.table = QTableWidget(0, 8)
        self.table.setHorizontalHeaderLabels(['Variant', 'Cameras', 'Blind in ROI m²', 'Δ vs ref m²', '≥ 2 cams %', 'Min px/cm',
                                              'Worst 1-cam error m²', 'GMSL ports'])
        self.table.verticalHeader().setVisible(False)
        self.table.verticalHeader().setDefaultSectionSize(ROW_H + 1)
        self.table.horizontalHeader().setFixedHeight(ROW_H)
        self.table.setShowGrid(False)
        self.table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        hh = self.table.horizontalHeader()
        hh.setSectionResizeMode(0, QHeaderView.ResizeMode.Stretch)
        for c in range(1, 8):
            hh.setSectionResizeMode(c, QHeaderView.ResizeMode.ResizeToContents)
        self.table.cellDoubleClicked.connect(self._row_ref)
        tl.addWidget(self.table)
        self.note = StatusLine('good', '')
        tl.addWidget(self.note)
        self.note2 = label('', muted=True, small=True, wrap=True)
        tl.addWidget(self.note2)
        tl.addStretch(1)
        bl = QHBoxLayout()
        bl.addStretch(1)
        b1 = button('Export comparison…')
        b1.clicked.connect(self.export)
        self.b_active = button('Make reference active', 'primary')
        self.b_active.clicked.connect(self.make_active)
        bl.addWidget(b1)
        bl.addWidget(self.b_active)
        tl.addLayout(bl)
        bottom.addWidget(tw)
        bottom.setSizes([660, 1340])
        lay.addWidget(bottom, 2)
        self.ref_uid = None
        ctx.store.changed.connect(lambda *_: self.activate() if self.isVisible() else None)
        ctx.store.project_replaced.connect(self._project_replaced)

    def _project_replaced(self):
        self.results.clear()
        self._keys.clear()
        self.worsts.clear()
        if self.isVisible():
            self.activate()            # the combos list the new project's variants

    # ------------------------------------------------------------------
    def activate(self):
        p = self.ctx.store.project
        vs = p.variants
        if self.ref_uid not in [v.uid for v in vs]:
            self.ref_uid = p.active_uid
        default = [v.uid for v in vs if v.uid != self.ref_uid][:1] + [self.ref_uid] + [v.uid for v in vs if v.uid != self.ref_uid][1:2]
        for i, col in enumerate(self.cols):
            combo = col['combo']
            combo.blockSignals(True)
            combo.clear()
            for v in vs:
                combo.addItem(v.name, v.uid)
            if col['uid'] not in [v.uid for v in vs]:
                col['uid'] = default[i] if i < len(default) else vs[0].uid
            combo.setCurrentIndex(max(0, combo.findData(col['uid'])))
            combo.blockSignals(False)
        for v in vs:
            self._compute(v.uid)
        self.refresh()

    def _choose(self, col, uid):
        col['uid'] = uid
        self._compute(uid)
        col['plan']._img_key = None
        self.refresh()

    def _vkey(self, v):
        """Everything a variant's analysis depends on: a result is reused only while this is unchanged."""
        import json
        p = self.ctx.store.project
        m = p.vehicle.mesh
        vehicle = [id(m), m.n_tris, [pt.role for pt in m.parts]] if m is not None else None
        return json.dumps([v.to_dict()['cameras'], v.seams, v.blend_m, p.roi.to_dict(), p.analysis.to_dict(), p.section('library'),
                           [o.to_dict() for o in p.obstacles], vehicle, p.vehicle.occluder_target, p.vehicle.wheel_centres],
                          sort_keys=True, default=str)

    def _compute(self, uid):
        p = self.ctx.store.project
        v = p.variant_by_uid(uid)
        if v is None:
            return
        key = self._vkey(v)
        if self._keys.get(uid) == key and uid in self.results:
            return
        self._keys[uid] = key
        snap = p.snapshot()
        lock = self.ctx.analysis._lock
        eng = self.ctx.analysis.engine

        def run():
            from ...core.tolerance import worst_single_camera
            vv = snap.variant_by_uid(uid)
            with lock:
                res = eng.analyse(snap, variant=vv)
            self.sig.ready.emit(uid, (key, res))
            if res is None:
                return                     # no vehicle yet: nothing to perturb
            snap.active_uid = uid
            with lock:
                w = worst_single_camera(eng, snap, snap.analysis.tol_deg, snap.analysis.tol_mm / 1000)
            self.sig.worst.emit(uid, (key, w))
        threading.Thread(target=run, daemon=True).start()

    def _ready(self, uid, out):
        key, res = out
        if self._keys.get(uid) != key:
            return                         # computed for inputs that have changed since (or another project)
        self.results[uid] = res
        for c in self.cols:
            if c['uid'] == uid:
                c['plan']._img_key = None
                c['plan'].fit()
        self.delta._img_key = None
        self.delta.fit()
        self.refresh()

    def _worst(self, uid, out):
        key, w = out
        if self._keys.get(uid) != key:
            return
        self.worsts[uid] = w
        self.refresh()

    def refresh(self):
        p = self.ctx.store.project
        st = self.ctx.store
        for col in self.cols:
            v = p.variant_by_uid(col['uid'])
            if v is None:
                continue
            col['tags'].setText(' · '.join(c.tag for c in v.cameras) + ('   reference' if v.uid == self.ref_uid else ''))
            res = self.results.get(v.uid)
            if res:
                b = res.metrics['blind_m2']
                col['status'].setText(f'blind {b:.3f} m²')
                col['status'].setStyleSheet(f'color: {STATUS["good"] if b < 5e-4 else STATUS["critical"]};')
            else:
                col['status'].setText('…')
            col['plan'].update()
        a, b = p.variant_by_uid(self.cols[0]['uid']), p.variant_by_uid(self.cols[1]['uid'])
        self.delta_title.setText(f'{a.name if a else "?"} → {b.name if b else "?"}')
        self.delta.update()
        ref = p.variant_by_uid(self.ref_uid)
        rres = self.results.get(self.ref_uid)
        self.ref_lbl.setText(f'reference: {ref.name if ref else "—"} · ROI {p.roi.distance:.1f} m · {p.analysis.cell_m * 100:.0f} cm cells · double-click a row to set')
        self.b_active.setText(f'Make {ref.name if ref else "reference"} active')
        self.table.setRowCount(len(p.variants))
        crit = []
        for i, v in enumerate(p.variants):
            res = self.results.get(v.uid)
            m = res.metrics if res else None
            w = self.worsts.get(v.uid)
            ports = sum(1 for c in v.cameras if c.port)
            vals = [v.name + ('   reference' if v.uid == self.ref_uid else ''), str(len(v.cameras)),
                    f'{m["blind_m2"]:.3f}' if m else '…',
                    ('—' if v.uid == self.ref_uid else f'{m["blind_m2"] - rres.metrics["blind_m2"]:+.3f}') if m and rres else '…',
                    f'{m["multi_pct"]:.1f}' if m else '…', f'{m["min_density"]:.1f}' if m else '…',
                    f'{w[0]:.3f}' if w else '…', f'{ports} / {p.ecu.ports}']
            for j, val in enumerate(vals):
                it = QTableWidgetItem(val.replace('-', '−') if j else val)
                if j:
                    it.setFont(mono(12))
                    it.setTextAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
                if j == 2 and m and m['blind_m2'] >= 5e-4:
                    it.setIcon(icons.status_icon('critical', 12))
                if v.uid == self.ref_uid:
                    it.setBackground(THEME.color('sel'))
                self.table.setItem(i, j, it)
            if m and m['blind_m2'] >= 5e-4:
                from ...core.analysis import describe_location
                g = res.grid
                bl = g.roi & (res.count == 0)
                jj, ii = np.nonzero(bl)
                where = describe_location(float(g.xs[ii].mean()), float(g.ys[jj].mean()), p) if len(ii) else ''
                crit.append(f'{v.name} loses {m["blind_m2"]:.3f} m², mostly {where}')
        fit = 22 * (len(p.variants) + 1) + 6
        self.table.setFixedHeight(min(fit, 260))
        if crit:
            self.note.set('critical', 'Critical · blind ground inside the ROI. ' + '; '.join(crit) + '.')
        else:
            self.note.set('good', 'No blind ground in the ROI in any variant.')
        worst_v = [(self.worsts[u][0], p.variant_by_uid(u).name) for u in self.worsts if p.variant_by_uid(u)]
        if worst_v:
            wv = max(worst_v)
            self.note2.setText(f'Largest single-camera tolerance case: {wv[1]}, {wv[0]:.3f} m² when one camera is '
                               f'{p.analysis.tol_deg:g}° or {p.analysis.tol_mm / 10:g} cm off.')

    def _row_ref(self, r, c):
        p = self.ctx.store.project
        if 0 <= r < len(p.variants):
            self.ref_uid = p.variants[r].uid
            self.refresh()

    def make_active(self):
        if self.ref_uid:
            self.ctx.window.switch_variant(self.ref_uid)
            self.ctx.msg('Reference variant is now active.')

    def export(self):
        path, _ = QFileDialog.getSaveFileName(self, 'Export comparison', 'comparison.csv', 'CSV (*.csv)')
        if not path:
            return
        p = self.ctx.store.project
        with open(path, 'w') as fh:
            fh.write('variant,cameras,blind_m2,multi_pct,min_density_px_cm,worst_single_m2,gmsl_ports\n')
            for v in p.variants:
                res = self.results.get(v.uid)
                m = res.metrics if res else {}
                w = self.worsts.get(v.uid)
                fh.write(f'"{v.name}",{len(v.cameras)},{m.get("blind_m2", float("nan")):.4f},{m.get("multi_pct", float("nan")):.2f},'
                         f'{m.get("min_density", float("nan")):.2f},{w[0] if w else float("nan"):.4f},{sum(1 for c in v.cameras if c.port)}\n')
        self.ctx.msg(f'Exported the comparison to {path}')
