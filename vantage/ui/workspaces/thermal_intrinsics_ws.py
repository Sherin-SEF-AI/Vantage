"""Calibrate ▸ Thermal intrinsics: each camera unit's focal length, principal point and distortion against its
temperature, fitted from lens calibrations taken at several temperatures (imported, or a simulated sweep from the
material model), the intrinsics compensated for a capture's temperature, and the curves exported with the
calibration files. Fitting is cheap and runs on every edit while the page shows; a simulated sweep renders and
calibrates checkerboards, so it runs as a background job."""
from __future__ import annotations

import math
import threading
import traceback

import numpy as np
from PyQt6.QtCore import QObject, QPointF, QRectF, Qt, pyqtSignal
from PyQt6.QtGui import QColor, QFont, QPainter, QPainterPath, QPen
from PyQt6.QtWidgets import (QAbstractItemView, QComboBox, QFileDialog, QGridLayout, QHBoxLayout, QHeaderView,
                             QScrollArea, QSplitter, QTableWidget, QTableWidgetItem, QVBoxLayout, QWidget)

from ...core import calib as CL
from ...core import thermal_intrinsics as TI
from ..charts import ChartBase
from ..theme import THEME, mono, sans
from ..widgets import ScrubSpinBox, SegmentedControl, StatusLine, button, fit_width, hline, label

JOB = 'thermal_sweep'
SEV = {'warning': 'warning', 'note': 'good', 'good': 'good', 'serious': 'serious', 'critical': 'critical'}


class _Sig(QObject):
    progress = pyqtSignal(int, float, str)
    done = pyqtSignal(int, object)
    failed = pyqtSignal(int, str)


# ---------------------------------------------------------------------- chart
class CurveChart(ChartBase):
    """One parameter's change from its reference value against temperature: the fitted curve, its prediction band
    and the measured calibrations; hovering a point shows its figures."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.title = 'Change against temperature'
        self.curve = None
        self.param = 'fx'
        self.at = None                       # the compensation temperature, marked
        self._pts = []
        self.setMinimumHeight(220)

    def set_data(self, curve, param, at=None):
        self.curve, self.param, self.at = curve, param, at
        self.tip = None
        self.update()

    def _frame(self):
        c = self.curve
        lo, hi = c.range_c
        if self.at is not None:
            lo, hi = min(lo, self.at), max(hi, self.at)
        pad = max(5.0, 0.08 * (hi - lo))
        return lo - pad, hi + pad

    def paintEvent(self, e):
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        c, q = self.curve, self.param
        self.paint_title(p, f'{q} − {q}(T_ref) {TI.UNIT[q]}'.strip())
        W, H = float(self.width()), float(self.height())
        L, R, T, B = 52.0, 12.0, 26.0, 26.0
        if c is None or c.n < 2:
            p.setFont(sans(11))
            p.setPen(THEME.color('muted'))
            p.drawText(QRectF(0, T, W, H - T), Qt.AlignmentFlag.AlignCenter,
                       'Two calibrations at different temperatures or more are needed for a curve.')
            p.end()
            return
        lo, hi = self._frame()
        ts = np.linspace(lo, hi, 90)
        ref = c.value(q, c.reference_c)
        band = [c.interval(q, t) for t in ts]
        ys = [b['value'] - ref for b in band]
        pl = [b['pi'][0] - ref for b in band if math.isfinite(b['pi'][0])]
        ph = [b['pi'][1] - ref for b in band if math.isfinite(b['pi'][1])]
        meas = list(np.asarray(c.params[q]['y']) - ref)
        span = max([abs(v) for v in ys + pl + ph + meas] + [1e-9]) * 1.1
        X = lambda t: L + (t - lo) / (hi - lo) * (W - L - R)                    # noqa: E731
        Y = lambda v: T + (H - T - B) / 2 - v / span * (H - T - B) / 2         # noqa: E731
        p.setPen(QPen(THEME.color('hair'), 1))
        p.drawLine(QPointF(L, Y(0)), QPointF(W - R, Y(0)))
        p.setFont(mono(10))
        p.setPen(THEME.color('muted'))
        step = 10 if hi - lo <= 80 else 20 if hi - lo <= 200 else 50
        for t in np.arange(math.ceil(lo / step) * step, hi + 1e-9, step):
            p.drawText(QRectF(X(t) - 20, H - B + 4, 40, 14), Qt.AlignmentFlag.AlignHCenter, f'{t:g}°')
        dp = 2 if span < 10 else 1 if span < 100 else 0
        if q.startswith('k'):
            dp = 5
        for v in (-span / 1.1, 0.0, span / 1.1):
            p.drawText(QRectF(0, Y(v) - 7, L - 6, 14), Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter,
                       f'{v:+.{dp}f}' if v else '0')
        col = THEME.cam(0)
        if len(pl) == len(ts):
            path = QPainterPath(QPointF(X(ts[0]), Y(ph[0])))
            for t, v in zip(ts[1:], ph[1:]):
                path.lineTo(QPointF(X(t), Y(v)))
            for t, v in zip(ts[::-1], pl[::-1]):
                path.lineTo(QPointF(X(t), Y(v)))
            path.closeSubpath()
            fill = QColor(col)
            fill.setAlphaF(0.16)
            p.fillPath(path, fill)
        p.setPen(QPen(col, 2))
        for i in range(len(ts) - 1):
            p.drawLine(QPointF(X(ts[i]), Y(ys[i])), QPointF(X(ts[i + 1]), Y(ys[i + 1])))
        clo, chi = c.range_c
        p.setPen(QPen(THEME.color('strong'), 1, Qt.PenStyle.DashLine))
        for t in (clo, chi):
            p.drawLine(QPointF(X(t), T), QPointF(X(t), H - B))
        if self.at is not None:
            p.setPen(QPen(THEME.color('ink2'), 1, Qt.PenStyle.DotLine))
            p.drawLine(QPointF(X(self.at), T), QPointF(X(self.at), H - B))
        self._pts = []
        p.setPen(QPen(THEME.color('panel'), 1.5))
        p.setBrush(col)
        for t, v in zip(c.temps, meas):
            pt = QPointF(X(t), Y(v))
            self._pts.append((pt, t, v))
            p.drawEllipse(pt, 4, 4)
        self.paint_tip(p)
        p.end()

    def mouseMoveEvent(self, e):
        pos = e.position()
        best = min(self._pts, key=lambda a: (a[0] - pos).manhattanLength(), default=None)
        if best is not None and (best[0] - pos).manhattanLength() < 14:
            pt, t, v = best
            dp = 6 if self.param.startswith('k') else 2
            self.set_tip(pt, f'{v:+.{dp}f} {TI.UNIT[self.param]}'.strip(), f'calibration at {t:g} °C',
                         f'{self.param} change from {self.curve.reference_c:g} °C')
        else:
            self.tip = None
        self.update()

    def leaveEvent(self, e):
        self.tip = None
        self.update()


# ---------------------------------------------------------------------- the page
class ThermalIntrinsicsPage(QWidget):
    def __init__(self, ctx):
        super().__init__()
        self.ctx = ctx
        self.result = None
        self.cam_uid = ''
        self.param = 'fx'
        self.busy = False
        self._gen = 0
        self._dirty = True
        self.sig = _Sig()
        self.sig.progress.connect(self._progress)
        self.sig.done.connect(self._thread_done)
        self.sig.failed.connect(self._failed)
        lay = QHBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        split = QSplitter(Qt.Orientation.Horizontal)
        split.addWidget(self._left())
        split.addWidget(self._right())
        split.setSizes([360, 1240])
        lay.addWidget(split)
        ctx.store.changed.connect(self._on_change)
        ctx.store.project_replaced.connect(self._on_change)
        THEME.changed.connect(self._on_change)

    # ------------------------------------------------------------------ panels
    def _left(self):
        w = QWidget()
        w.setObjectName('panel')
        ll = QVBoxLayout(w)
        ll.setContentsMargins(8, 6, 8, 8)
        ll.setSpacing(4)
        top = QHBoxLayout()
        top.addWidget(label('Thermal intrinsics'))
        top.addStretch(1)
        top.addWidget(label('focal drift per °C', muted=True, small=True))
        ll.addLayout(top)
        ll.addWidget(hline())
        ll.addWidget(label('Lens calibrations of one camera taken at several temperatures give its focal length, '
                           'principal point and distortion as curves of temperature; a capture is then compensated '
                           'for its own temperature.', muted=True, small=True, wrap=True))
        self.cam = QComboBox()
        self.cam.setToolTip('The camera unit whose temperature sweep is shown')
        self.cam.currentIndexChanged.connect(self._pick_cam)
        row = QHBoxLayout()
        row.addWidget(label('Camera', ink2=True))
        row.addWidget(self.cam, 1)
        ll.addLayout(row)
        # ---- records
        ll.addSpacing(4)
        ll.addWidget(label('Calibrations', bold=True))
        self.recs = QTableWidget(0, 5)
        self.recs.setHorizontalHeaderLabels(['°C', 'fx px', 'cx px', 'cy px', 'Source'])
        self.recs.verticalHeader().hide()
        self.recs.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.recs.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.recs.horizontalHeader().setSectionResizeMode(4, QHeaderView.ResizeMode.Stretch)
        self.recs.setMinimumHeight(150)
        ll.addWidget(self.recs)
        bl = QHBoxLayout()
        self.b_import = button('Import…', None, 'import', 'Calibration files (OpenCV YAML/XML, ROS camera_info, '
                                                          'Kalibr, JSON) with a temperature field each')
        self.b_import.clicked.connect(lambda: self.import_files())
        self.b_remove = button('Remove', 'ghost', None, 'Remove the selected calibrations')
        self.b_remove.clicked.connect(self._remove)
        self.b_clear = button('Clear', 'ghost', None, 'Remove every calibration of this camera')
        self.b_clear.clicked.connect(lambda: self.set_records([], 'cleared'))
        for b in (self.b_import, self.b_remove, self.b_clear):
            bl.addWidget(b)
        bl.addStretch(1)
        ll.addLayout(bl)
        self.imp_temp = ScrubSpinBox('', 'plain', 1, '°C', step=1.0, width=90)
        self.imp_temp.setToolTip('Temperature given to imported files that state none')
        self.imp_temp.set_value(20.0)
        row = QHBoxLayout()
        row.addWidget(label('Files without a temperature', ink2=True, small=True))
        row.addStretch(1)
        row.addWidget(self.imp_temp)
        ll.addLayout(row)
        # ---- fit settings
        ll.addSpacing(6)
        ll.addWidget(label('Fit', bold=True))
        g = QGridLayout()
        g.setVerticalSpacing(2)
        g.setColumnStretch(1, 1)
        self.degree = SegmentedControl([('auto', 'Auto'), ('linear', 'Linear'), ('quadratic', 'Quadratic')], 'auto')
        self.degree.setToolTip('Auto: per parameter by leave-one-out cross-validation (a quadratic must predict '
                               '10 % better and its curvature stand 3 σ out)')
        self.degree.changed.connect(lambda k: self._set(f'fit {k}', degree=k))
        g.addWidget(label('Degree', ink2=True), 0, 0)
        g.addWidget(self.degree, 0, 1)
        self.fields = {}
        for i, (key, text, unit, dec, step, tip) in enumerate((
                ('reference_c', 'Reference', '°C', 1, 1.0, 'The temperature the reference intrinsics refer to'),
                ('warn_px', 'Drift limit', 'px', 2, 0.1, 'A drift over the operating range above this is flagged'),
                ('operating_lo_c', 'Operating from', '°C', 0, 5.0, 'Used when the module states no range'),
                ('operating_hi_c', 'Operating to', '°C', 0, 5.0, 'Used when the module states no range')), 1):
            f = ScrubSpinBox('', 'plain', dec, unit, step=step, width=100)
            f.setToolTip(tip)
            f.committed.connect(lambda v, k=key, t=text, u=unit: self._set(f'{t.lower()} {v:g} {u}', **{k: v}))
            g.addWidget(label(text, ink2=True), i, 0)
            g.addWidget(f, i, 1)
            self.fields[key] = f
        ll.addLayout(g)
        # ---- the simulated sweep
        ll.addSpacing(6)
        ll.addWidget(label('Simulated sweep', bold=True))
        ll.addWidget(label('Synthetic data, not a measurement: rendered checkerboards through intrinsics moved by a '
                           'material model (SCHOTT glass data, Apollo Optical Systems polymer data, Jamieson 1981), '
                           'each calibrated as real images would be. Real temperature sweeps of your units are needed '
                           'before a curve is trusted.', muted=True, small=True, wrap=True))
        g = QGridLayout()
        g.setVerticalSpacing(2)
        g.setColumnStretch(1, 1)
        self.lens_mat = QComboBox()
        for k, v in TI.LENS_MATERIALS.items():
            self.lens_mat.addItem(v['label'], k)
            self.lens_mat.setItemData(self.lens_mat.count() - 1, v['source'], Qt.ItemDataRole.ToolTipRole)
        self.lens_mat.activated.connect(lambda _: self._set('lens material', lens_material=self.lens_mat.currentData()))
        self.housing = QComboBox()
        for k, v in TI.HOUSING_MATERIALS.items():
            self.housing.addItem(v['label'], k)
            self.housing.setItemData(self.housing.count() - 1, v['source'], Qt.ItemDataRole.ToolTipRole)
        self.housing.activated.connect(lambda _: self._set('holder material',
                                                           housing_material=self.housing.currentData()))
        self.scale_ctl = SegmentedControl([('housing', 'Fixed focus'), ('lens', 'Stays in focus')], 'housing')
        self.scale_ctl.setToolTip('Fixed focus: the image scale follows the holder between lens and sensor. Stays in '
                                  'focus (athermalised or refocused): it follows the lens\'s own focal length')
        self.scale_ctl.changed.connect(lambda k: self._set('image scale', image_scale=k))
        g.addWidget(label('Lens', ink2=True), 0, 0)
        g.addWidget(self.lens_mat, 0, 1)
        g.addWidget(label('Holder', ink2=True), 1, 0)
        g.addWidget(self.housing, 1, 1)
        g.addWidget(self.scale_ctl, 2, 0, 1, 2)
        for i, (key, text, unit, dec, step, tip) in enumerate((
                ('mount_ecc_mm', 'Eccentricity', 'mm', 1, 0.5, 'Heuristic: how far off the optical axis the holder is '
                                                               'fixed (moves the principal point)'),
                ('sweep_lo_c', 'From', '°C', 0, 5.0, 'Lowest temperature of the sweep'),
                ('sweep_hi_c', 'To', '°C', 0, 5.0, 'Highest temperature of the sweep'),
                ('sweep_steps', 'Temperatures', '', 0, 1.0, 'Calibrations in the sweep'),
                ('views', 'Views each', '', 0, 1.0, 'Checkerboard views per calibration'),
                ('seed', 'Seed', '', 0, 1.0, 'The unit and its board poses')), 3):
            f = ScrubSpinBox('', 'plain', dec, unit, step=step, width=100)
            f.setToolTip(tip)
            f.committed.connect(lambda v, k=key, t=text, u=unit: self._set(f'{t.lower()} {v:g} {u}'.strip(), **{k: v}))
            g.addWidget(label(text, ink2=True), i, 0)
            g.addWidget(f, i, 1)
            self.fields[key] = f
        ll.addLayout(g)
        bl = QHBoxLayout()
        self.b_sim = button('Simulate a sweep', None, 'sparkle', 'Render and calibrate the camera at each '
                                                                 'temperature (replaces its calibrations)')
        self.b_sim.clicked.connect(lambda: self.simulate())
        bl.addWidget(self.b_sim)
        bl.addStretch(1)
        ll.addLayout(bl)
        ll.addStretch(1)
        self.status = label('', muted=True, small=True, wrap=True)
        ll.addWidget(self.status)
        bl = QHBoxLayout()
        self.b_defaults = button('Defaults', 'ghost', None, 'Back to the default settings (calibrations kept)')
        self.b_defaults.clicked.connect(self._defaults)
        bl.addWidget(self.b_defaults)
        bl.addStretch(1)
        self.b_export = button('Export…', None, 'export', 'The curves as JSON and CSV, OpenCV YAML per camera with the '
                                                          'thermal fields, a Kalibr camchain with a thermal block')
        self.b_export.clicked.connect(lambda: self.export())
        bl.addWidget(self.b_export)
        ll.addLayout(bl)
        sc = QScrollArea()
        sc.setWidgetResizable(True)
        sc.setWidget(w)
        fit_width(sc)
        sc.setMinimumWidth(320)
        return sc

    def _right(self):
        w = QWidget()
        rl = QVBoxLayout(w)
        rl.setContentsMargins(10, 8, 10, 8)
        rl.setSpacing(6)
        self.headline = label('', bold=True)
        rl.addWidget(self.headline)
        self.sub = label('', muted=True, small=True, wrap=True)
        rl.addWidget(self.sub)
        self.pick = SegmentedControl([(p, p) for p in TI.PARAMS], 'fx')
        self.pick.setToolTip('The parameter the chart shows')
        self.pick.changed.connect(self._pick_param)
        rl.addWidget(self.pick)
        self.chart = CurveChart()
        rl.addWidget(self.chart, 2)
        self.table = QTableWidget(0, 8)
        self.table.setHorizontalHeaderLabels(['Parameter', 'At T_ref', 'per K', '± σ', 'per K²', 'Residual σ',
                                              'LOO lin · quad', 'Model'])
        self.table.verticalHeader().hide()
        self.table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.table.horizontalHeader().setSectionResizeMode(7, QHeaderView.ResizeMode.Stretch)
        self.table.setMinimumHeight(250)
        rl.addWidget(self.table, 2)
        # ---- compensation
        box = QHBoxLayout()
        box.addWidget(label('Compensate for', ink2=True))
        self.comp_t = ScrubSpinBox('', 'plain', 1, '°C', step=1.0, width=100)
        self.comp_t.setToolTip('The capture\'s temperature (the camera\'s sensor reading from its metadata, or typed)')
        self.comp_t.set_value(60.0)
        self.comp_t.committed.connect(lambda _: self._show_comp())
        box.addWidget(self.comp_t)
        self.comp = label('', mono_=True, small=True, wrap=True)
        box.addWidget(self.comp, 1)
        self.b_use = button('Use as unit calibration', None, 'lens', 'Make the curve\'s reference intrinsics, with '
                                                                    'the curve, this camera\'s unit calibration')
        self.b_use.clicked.connect(self.use_as_calibration)
        box.addWidget(self.b_use)
        rl.addLayout(box)
        self.find_host = QWidget()
        self.find_lay = QVBoxLayout(self.find_host)
        self.find_lay.setContentsMargins(0, 0, 0, 0)
        self.find_lay.setSpacing(4)
        self.find_lay.addStretch(1)
        sc = QScrollArea()
        sc.setWidgetResizable(True)
        sc.setWidget(self.find_host)
        sc.setMinimumHeight(90)
        rl.addWidget(sc, 1)
        self.findings = []
        return w

    # ------------------------------------------------------------------ settings and records
    @property
    def settings(self) -> TI.ThermalSettings:
        return TI.settings_of(self.ctx.store.project)

    def _store(self):
        d = self.ctx.store.project.analysis.thermal_intrinsics
        return d if isinstance(d, dict) else {}

    def _set(self, text, **kw):
        st = self.settings
        for k, v in kw.items():
            setattr(st, k, v)
        new = TI.ThermalSettings.from_dict(st.to_dict()).to_dict()
        if new == self.settings.to_dict():
            self._sync_fields()
            return False
        sweeps = self._store().get('sweeps')

        def fn(p):
            p.analysis.thermal_intrinsics = dict(new, **({'sweeps': sweeps} if sweeps else {}))
        self.ctx.store.edit('analysis', fn, f'Thermal intrinsics · {text}')
        return True

    def _defaults(self):
        d = self._store()
        if any(k != 'sweeps' for k in d):
            sweeps = d.get('sweeps')

            def fn(p):
                p.analysis.thermal_intrinsics = {'sweeps': sweeps} if sweeps else {}
            self.ctx.store.edit('analysis', fn, 'Thermal intrinsics · defaults')

    def records(self) -> list:
        return TI.sweeps_of(self.ctx.store.project).get(self.cam_uid, [])

    def set_records(self, recs, text):
        if not self.cam_uid:
            return False
        cam = self._camera()
        new = TI.with_sweep(self._store(), self.cam_uid, recs)

        def fn(p):
            p.analysis.thermal_intrinsics = new
        self.ctx.store.edit('analysis', fn, f'Thermal intrinsics · {cam.tag if cam else ""} {text}'.strip())
        return True

    def import_files(self, paths=None, temp_c=None):
        """Add the calibrations in these files to the camera's sweep (a file dialog when paths is None). A record
        without a temperature gets temp_c (the field's value by default)."""
        if not self.cam_uid:
            return 0
        if paths is None:
            paths, _ = QFileDialog.getOpenFileNames(self, 'Import calibrations taken at known temperatures', '',
                                                    'Calibrations (*.yaml *.yml *.xml *.json);;All files (*)')
            if not paths:
                return 0
        t_def = float(self.imp_temp.value()) if temp_c is None else float(temp_c)
        add, bad = [], []
        for path in paths:
            try:
                recs = CL.read_file(path)
            except Exception as e:                    # noqa: BLE001  (an unreadable file is reported, not raised)
                bad.append(f'{path.rsplit("/", 1)[-1]}: {e}')
                continue
            for r in recs:
                r = CL.clean_record(r)
                if CL._temp(r.get('temp_c')) is None:
                    r['temp_c'] = t_def
                add.append(r)
        if add:
            self.set_records(self.records() + add, f'{len(add)} calibration{"s" * (len(add) != 1)} imported')
        self.ctx.msg(f'Imported {len(add)} calibration{"s" * (len(add) != 1)}'
                     + (f'; {len(bad)} file(s) not read: {bad[0]}' if bad else ''), 6000)
        return len(add)

    def _remove(self):
        rows = sorted({i.row() for i in self.recs.selectedIndexes()})
        if not rows:
            return
        recs = sorted(self.records(), key=lambda r: float(r['temp_c']))
        keep = [r for i, r in enumerate(recs) if i not in rows]
        self.set_records(keep, f'{len(rows)} calibration{"s" * (len(rows) != 1)} removed')

    def _camera(self):
        var = self.ctx.store.project.variant
        return next((c for c in var.cameras if c.uid == self.cam_uid), None)

    def use_as_calibration(self):
        cam = self._camera()
        cur = self.result.curve(self.cam_uid) if self.result else None
        if cam is None or cur is None or cur.n < 2:
            return False
        rec = CL.clean_record(cur.reference_record())
        rec['date'] = ''
        uid = cam.uid

        def fn(p):
            c = next(c for c in p.variant.cameras if c.uid == uid)
            c.calib = rec
        self.ctx.store.edit('variant', fn, f'{cam.tag}: unit calibration from the thermal curve at '
                                           f'{cur.reference_c:g} °C')
        self.ctx.msg(f'{cam.tag} now uses the curve\'s intrinsics at {cur.reference_c:g} °C (with the curve)', 5000)
        return True

    # ------------------------------------------------------------------ the simulated sweep
    def simulate(self, wait=False):
        cam = self._camera()
        if cam is None or self.busy:
            return False
        p = self.ctx.store.project
        m, lens = p.module_for(cam), p.nominal_lens_for(cam)
        st = self.settings
        base = dict(cam.calib) if cam.calib else None
        self._gen += 1
        gen, uid = self._gen, cam.uid
        self.busy = True
        self.b_sim.setEnabled(False)
        self.status.setText(f'Simulating a sweep of {cam.tag}: {st.sweep_steps} calibrations…')

        def work(progress, cancel):
            return uid, TI.simulate_sweep(m, lens, st, base_rec=base, progress=progress, cancel=cancel,
                                          serial=f'{cam.tag}-SIM{st.seed}')
        win = self.ctx.window
        if not wait and win is not None and hasattr(win, 'run_job'):
            if win.run_job(JOB, f'Thermal sweep · {cam.tag}', work, lambda out: self._thread_done(gen, out),
                           lambda msg: self._failed(gen, msg)):
                return True

        def thread():
            try:
                out = work(lambda f, t='': self.sig.progress.emit(gen, f, t), lambda: gen != self._gen)
                self.sig.done.emit(gen, out)
            except Exception as e:                    # noqa: BLE001
                try:
                    self.sig.failed.emit(gen, f'{e}\n{traceback.format_exc()}')
                except RuntimeError:
                    pass
        th = threading.Thread(target=thread, daemon=True)
        th.start()
        self._thread = th
        return True

    def _progress(self, gen, f, text):
        if gen == self._gen and text:
            self.status.setText(f'{f * 100:.0f} % · {text}')

    def _thread_done(self, gen, out):
        if gen != self._gen:
            return
        self.busy = False
        self.b_sim.setEnabled(True)
        uid, sim = out if out else (None, None)
        if sim is None:
            self.status.setText('Cancelled')
            return
        if not sim['records']:
            self.status.setText('No calibration of the sweep converged.')
            return
        keep, self.cam_uid = self.cam_uid, uid
        self.set_records(sim['records'], f'simulated sweep ({len(sim["records"])} temperatures)')
        self.cam_uid = keep
        self.sim_truth = (uid, sim)
        self.ctx.msg(f'Simulated {len(sim["records"])} calibrations', 5000)

    def _failed(self, gen, msg):
        if gen != self._gen:
            return
        self.busy = False
        self.b_sim.setEnabled(True)
        self.status.setText(f'The sweep failed: {msg.splitlines()[0]}')

    # ------------------------------------------------------------------ state
    def _on_change(self, *_):
        if self.isVisible():
            self.refresh()
        else:
            self._dirty = True

    def showEvent(self, e):
        super().showEvent(e)
        if self._dirty:
            self.refresh()

    def activate(self):
        if self._dirty:
            self.refresh()

    def _pick_cam(self, i):
        uid = self.cam.itemData(i)
        if uid and uid != self.cam_uid:
            self.cam_uid = uid
            self.refresh()

    def _pick_param(self, k):
        self.param = k
        self._show_curve()

    def _sync_fields(self):
        st = self.settings.to_dict()
        for k, f in self.fields.items():
            f.set_value(float(st[k]))
        self.degree.set_value(st['degree'])
        self.scale_ctl.set_value(st['image_scale'])
        for combo, key in ((self.lens_mat, 'lens_material'), (self.housing, 'housing_material')):
            i = combo.findData(st[key])
            if i >= 0 and i != combo.currentIndex():
                combo.blockSignals(True)
                combo.setCurrentIndex(i)
                combo.blockSignals(False)

    def refresh(self):
        self._dirty = False
        p = self.ctx.store.project
        cams = [c for c in p.variant.cameras if c.visible]
        if self.cam_uid not in {c.uid for c in cams}:
            have = TI.sweeps_of(p)
            self.cam_uid = next((c.uid for c in cams if c.uid in have), cams[0].uid if cams else '')
        self.cam.blockSignals(True)
        self.cam.clear()
        n_of = {u: len(r) for u, r in TI.sweeps_of(p).items()}
        for c in cams:
            n = n_of.get(c.uid, 0)
            self.cam.addItem(f'{c.tag} · {n} calibration{"s" * (n != 1)}' if n else f'{c.tag} · no sweep', c.uid)
        i = self.cam.findData(self.cam_uid)
        if i >= 0:
            self.cam.setCurrentIndex(i)
        self.cam.blockSignals(False)
        self._sync_fields()
        self.result = TI.run(p)
        # ---- records table
        recs = sorted(self.records(), key=lambda r: float(r['temp_c']))
        m = None
        cam = self._camera()
        if cam is not None:
            m = p.module_for(cam)
        self.recs.setRowCount(len(recs))
        for i, r in enumerate(recs):
            rr = TI.rescale(r, m.width_px, m.height_px) if m is not None else r
            for j, t in enumerate((f'{r["temp_c"]:g}', f'{rr["fx"]:.2f}', f'{rr["cx"]:.2f}', f'{rr["cy"]:.2f}',
                                   r.get('source', ''))):
                it = QTableWidgetItem(t)
                if j < 4:
                    it.setFont(mono(12))
                    it.setTextAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
                self.recs.setItem(i, j, it)
        for b in (self.b_remove, self.b_clear):
            b.setEnabled(bool(recs))
        self.b_sim.setEnabled(cam is not None and not self.busy)
        self._show_curve()
        # ---- findings
        for w in self.findings:
            self.find_lay.removeWidget(w)
            w.hide()
            w.deleteLater()
        self.findings = []
        res = self.result
        items = res.findings if res.findings else (
            [{'severity': 'good', 'message': f'Every fitted camera drifts less than {res.settings.warn_px:g} px over its '
                                             'operating range.'}] if res.curves else [])
        for f in items:
            w = StatusLine(SEV.get(f['severity'], 'warning'), f['message'])
            self.find_lay.insertWidget(self.find_lay.count() - 1, w)
            self.findings.append(w)
        mt = res.metrics
        if not self.busy:
            self.status.setText(f'{mt["with_sweep"]} of {mt["cameras"]} cameras have a temperature sweep'
                                + (f' · largest drift {mt["worst_drift_px"]:.2f} px ({mt["worst_tag"]})'
                                   if mt['worst_drift_px'] is not None else ''))
        self.b_export.setEnabled(bool(res.curves))

    def _show_curve(self):
        res = self.result
        cur = res.curve(self.cam_uid) if res else None
        cam = self._camera()
        tag = cam.tag if cam else ''
        self.chart.set_data(cur, self.param, float(self.comp_t.value()) if cur is not None else None)
        self.b_use.setEnabled(cur is not None and cur.n >= 2)
        if cur is None:
            self.headline.setText(f'{tag}: no calibrations at known temperatures' if tag else 'No camera')
            self.sub.setText('Import calibration files that state the camera\'s temperature, or simulate a sweep from '
                             'the material model to see the method.')
            self.table.setRowCount(0)
            self.comp.setText('')
            return
        lo, hi = cur.range_c
        e = next((x for x in res.cameras if x['uid'] == self.cam_uid), {})
        sim = all(str(r.get('source', '')).startswith('simulated') for r in self.records())
        self.headline.setText(f'{tag}: {cur.n} calibrations, {lo:g} … {hi:g} °C'
                              + (' · simulated' if sim else ''))
        opr = e.get('operating')
        self.sub.setText(f'fx {cur.slope("fx"):+.4f} px/K · cx {cur.slope("cx"):+.4f} px/K · cy {cur.slope("cy"):+.4f} '
                         f'px/K at {cur.reference_c:g} °C'
                         + (f' · up to {e["drift_px"]:.2f} px over {opr[0]:g} … {opr[1]:g} °C'
                            if e.get('drift_px') is not None and opr else '')
                         + ('. Synthetic data from a material model: not a measurement.' if sim else ''))
        self.table.setRowCount(len(TI.PARAMS))
        right = Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter
        for i, q in enumerate(TI.PARAMS):
            f = cur.params[q]
            px = q in TI.PX_PARAMS
            se1 = math.sqrt(f['cov'][1, 1]) if np.isfinite(f['cov'][1, 1]) else float('nan')
            fmt = (lambda v, d=4: '—' if not math.isfinite(v) else f'{v:+.{d}f}')            # noqa: E731
            press = f['press']
            cells = [q, f'{f["coef"][0]:.{3 if px else 6}f}', fmt(f['coef'][1], 4 if px else 8), fmt(se1, 4 if px else 8),
                     fmt(f['coef'][2], 6 if px else 10) if f['degree'] == 2 else '—',
                     fmt(f['sigma'], 3 if px else 6).lstrip('+'),
                     ' · '.join('—' if not math.isfinite(press.get(d, float('nan'))) else
                                f'{press[d]:.{3 if px else 6}f}' for d in (1, 2)),
                     f['reason']]
            for j, t in enumerate(cells):
                it = QTableWidgetItem(t)
                if 0 < j < 7:
                    it.setFont(mono(12))
                    it.setTextAlignment(right)
                if j == 7:
                    it.setToolTip(t)
                self.table.setItem(i, j, it)
        self._show_comp()

    def _show_comp(self):
        res = self.result
        cur = res.curve(self.cam_uid) if res else None
        if cur is None:
            self.comp.setText('')
            return
        t = float(self.comp_t.value())
        self.chart.set_data(cur, self.param, t)
        parts = []
        for q in TI.PX_PARAMS:
            iv = cur.interval(q, t)
            pm = f' ±{iv["se_pred"]:.2f}' if math.isfinite(iv['se_pred']) else ''
            parts.append(f'{q} {iv["value"]:.2f}{pm}')
        lo, hi = cur.range_c
        out = ' · '.join(parts) + ' px'
        if not lo - 1e-9 <= t <= hi + 1e-9:
            out += f' · extrapolated (calibrated {lo:g} … {hi:g} °C)'
        self.comp.setText(out)
        self.comp.setToolTip('Prediction ± its standard deviation (the curve\'s uncertainty and the scatter of one '
                             'calibration about it), at the curve\'s resolution')

    # ------------------------------------------------------------------ export
    def export(self, folder=None):
        res = self.result
        if res is None or not res.curves:
            return []
        if folder is None:
            folder = QFileDialog.getExistingDirectory(self, 'Export the thermal curves into')
            if not folder:
                return []
        from ...core import provenance as PV
        paths = PV.stamp_paths(TI.export(res, self.ctx.store.project, folder), self.ctx.store.project, folder)
        self.ctx.msg(f'Exported the thermal curves to {folder}', 5000)
        return paths
