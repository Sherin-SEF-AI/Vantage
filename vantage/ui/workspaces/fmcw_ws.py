"""Analysis ▸ FMCW lidar: the radial velocity of every lidar return (core.fmcw_lidar): the 4D cloud, the ego velocity
fitted from one sweep, the moving points found by Doppler alone, and what the sun and other lidars do to a
time-of-flight lidar against an FMCW one.

Left: the source (the scripted urban or highway scene at an ego speed, or a CARLA recording with ground truth), the
Doppler figures (the FMCW model a time-of-flight lidar is compared with at its pose, the moving threshold), the
receivers under the sun and the other test cars. Centre: the 4D cloud in 3D over the car, coloured by radial velocity,
by the points' own velocity (the ego's motion removed) or by moving and static. Right: per lidar the point rate,
velocity range, ego-velocity error and moving-point recall; the objects of the shown sweep; ToF against FMCW per sun
condition; cross-talk; the recording's cross-checks. The study runs in a background thread after every edit while the
page shows (a scene in well under a second, a recording in seconds)."""
from __future__ import annotations

import os
import threading
import traceback

import numpy as np
from PyQt6.QtCore import QObject, Qt, QTimer, pyqtSignal
from PyQt6.QtWidgets import (QAbstractItemView, QComboBox, QFileDialog, QHBoxLayout, QHeaderView, QProgressBar,
                             QScrollArea, QSplitter, QTableWidget, QTableWidgetItem, QVBoxLayout, QWidget)

from ...core import fmcw_lidar as FM
from ..scene import MainScene
from ..theme import THEME
from ..viewport3d import PointDraw
from ..widgets import ROW_H, ScrubSpinBox, Section, SegmentedControl, ToggleSwitch, button, hline, label

AUTO_DELAY_MS = 400
FIELDS = (('ego_speed_kmh', 'Ego speed', 0, 'km/h', 5.0, 'The car\'s speed in the scripted scene'),
          ('yaw_rate_dps', 'Yaw rate', 1, '°/s', 1.0, 'The car turning (positive to the left)'),
          ('max_sweeps', 'Sweeps', 0, '', 10.0, 'A recording\'s sweeps evaluated, spread evenly over the drive'),
          ('moving_mps', 'Moving above', 2, 'm/s', 0.05, 'A point moves when its speed over the ground exceeds this; '
                                                          'Doppler flags it when its own velocity along the ray does'),
          ('velocity_sigma_mps', 'Velocity σ', 3, 'm/s', 0.005, 'Radial-velocity noise (1 σ); 0 = the datasheet\'s '
                                                                 '(3 cm/s where none is stated)'),
          ('crosstalk_cars', 'Other test cars', 0, '', 1.0, 'Test cars nearby with a spinning time-of-flight lidar '
                                                            'each'),
          ('crosstalk_distance_m', 'Apart', 0, 'm', 5.0, 'How far ahead (and behind) the other cars are'),
          ('tof_filter_nm', 'ToF filter', 0, 'nm', 5.0, 'The time-of-flight receiver\'s optical bandpass'),
          ('tof_ifov_mrad', 'ToF pixel field', 2, 'mrad', 0.1, 'The time-of-flight receiver\'s pixel field of view'))
FIELD_LABEL = {k: t for k, t, *_ in FIELDS}
FIELD_LABEL.update(source='source', scene='scene', walls='roadside walls', counterpart='FMCW model',
                   tof_receiver='ToF receiver', tof_reference='ToF datasheet range', recording='recording',
                   recording_tag='recording lidar')
COLOUR_MODES = (('radial', 'Radial velocity'), ('own', 'Own velocity'), ('moving', 'Moving'))


class _Sig(QObject):
    done = pyqtSignal(int, object)
    failed = pyqtSignal(int, str)
    progress = pyqtSignal(int, float)


def _table(cols):
    t = QTableWidget(0, len(cols))
    t.verticalHeader().setVisible(False)
    t.verticalHeader().setDefaultSectionSize(ROW_H)
    t.horizontalHeader().setFixedHeight(ROW_H)
    t.setShowGrid(False)
    t.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
    t.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
    for c, (head, tip) in enumerate(cols):
        it = QTableWidgetItem(head)
        it.setToolTip(tip)
        t.setHorizontalHeaderItem(c, it)
    hh = t.horizontalHeader()
    hh.setSectionResizeMode(0, QHeaderView.ResizeMode.Stretch)
    for c in range(1, len(cols)):
        hh.setSectionResizeMode(c, QHeaderView.ResizeMode.ResizeToContents)
    return t


def _fill(t, rows):
    t.setRowCount(len(rows))
    for i, r in enumerate(rows):
        for c, v in enumerate(r):
            it = QTableWidgetItem(str(v))
            it.setToolTip(str(v))
            if c:
                it.setTextAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
            t.setItem(i, c, it)


def _pct(v, nd=0):
    return '—' if v is None else f'{v * 100:.{nd}f} %'


def _num(v, fmt='{:.2f}', none='—'):
    return none if v is None else fmt.format(v)


def _rate(v):
    if v is None:
        return '—'
    if v >= 1e6:
        return f'{v / 1e6:.2f} M'
    if v >= 1e3:
        return f'{v / 1e3:.1f} k'
    return f'{v:.3g}'


def cloud_colours(cl, mode, dark=True) -> np.ndarray:
    """(n, 4) colours of a 4D cloud: by measured radial velocity, by the points' own velocity along the ray, or moving
    (warning colour) against static (grey)."""
    if mode == 'moving':
        rgba = np.tile(np.array([0.55, 0.55, 0.55, 0.85], np.float32), (len(cl.P), 1))
        rgba[cl.moving] = (0.95, 0.55, 0.10, 1.0)
        return rgba
    v = cl.v_r if mode == 'radial' else cl.v_comp
    fin = np.isfinite(v)
    scale = max(float(np.percentile(np.abs(v[fin]), 98)) if fin.any() else 1.0, 1.0)
    return FM.velocity_rgba(v, scale)


class FmcwScene(MainScene):
    """The car with the shown lidar's 4D cloud."""

    def __init__(self, ctx, page):
        super().__init__(ctx)
        self.page = page

    def points(self, vp):
        cl = self.page.shown_cloud()
        if cl is None or not len(cl.P):
            return []
        rgba = cloud_colours(cl, self.page.c_mode.value() or 'radial', THEME.name != 'light')
        return [PointDraw(cl.P.astype(np.float32), rgba, 3.0, True)]


class FmcwPage(QWidget):
    def __init__(self, ctx):
        super().__init__()
        self.ctx = ctx
        self.result: FM.FmcwResult | None = None
        self.busy = False
        self.stale = False
        self._gen = 0
        self._for = None
        self._syncing = False
        self.status_text = ''
        self.view3d = None
        self.sig = _Sig()
        self.sig.done.connect(self._done)
        self.sig.failed.connect(self._failed)
        self.sig.progress.connect(lambda g, f: self.bar.setValue(int(f * 1000)) if g == self._gen else None)
        self.setObjectName('base')
        lay = QHBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        split = QSplitter(Qt.Orientation.Horizontal)
        split.setHandleWidth(2)
        split.addWidget(self._left())
        split.addWidget(self._centre())
        split.addWidget(self._right())
        split.setSizes([300, 620, 640])
        lay.addWidget(split)
        self.timer = QTimer(self)
        self.timer.setSingleShot(True)
        self.timer.setInterval(AUTO_DELAY_MS)
        self.timer.timeout.connect(lambda: self.run(quiet=True))
        ctx.store.changed.connect(self._on_store)
        ctx.store.project_replaced.connect(self._replaced)
        THEME.changed.connect(lambda *_: self.refresh())
        self._sync_fields()
        self.refresh()

    # ------------------------------------------------------------------ layout
    def _left(self):
        outer = QWidget()
        outer.setObjectName('panel')
        outer.setMinimumWidth(280)
        ol = QVBoxLayout(outer)
        ol.setContentsMargins(0, 0, 0, 0)
        ol.setSpacing(0)
        head = QWidget()
        hl = QVBoxLayout(head)
        hl.setContentsMargins(8, 6, 8, 4)
        hl.setSpacing(4)
        hl.addWidget(label('FMCW lidar'))
        hl.addWidget(hline())
        hl.addWidget(label('The radial velocity of every return: the ego velocity and the moving points from one '
                           'sweep, and the sun and other lidars against a time-of-flight lidar.',
                           muted=True, small=True, wrap=True))
        ol.addWidget(head)
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QScrollArea.Shape.NoFrame)
        body = QWidget()
        body.setObjectName('panel')
        bl = QVBoxLayout(body)
        bl.setContentsMargins(0, 0, 0, 0)
        bl.setSpacing(0)
        self.fields, self.segs = {}, {}

        def fld(sec, key):
            k, text, dec, unit, step, tip = next(f for f in FIELDS if f[0] == key)
            f = ScrubSpinBox('', 'plain', dec, unit, step=step, width=90)
            f.setToolTip(tip)
            f.committed.connect(lambda v, kk=k: self.set_option(kk, v))
            self.fields[k] = f
            sec.row(text, f, tip=tip)

        s = Section('Source')
        seg = SegmentedControl([('scene', 'Scene'), ('recording', 'Recording')])
        seg.setToolTip('A scripted scene around the car, or a CARLA drive with ground truth (its lidar carried as each '
                       'FMCW lidar)')
        seg.changed.connect(lambda k: self.set_option('source', k))
        self.segs['source'] = seg
        s.row('From', seg)
        seg = SegmentedControl([(k, v['label']) for k, v in FM.SCENES.items()])
        seg.setToolTip('Traffic around the car: urban (50 km/h) or highway (120 km/h); choosing one sets its speed')
        seg.changed.connect(self.set_scene)
        self.segs['scene'] = seg
        s.row('Scene', seg)
        fld(s, 'ego_speed_kmh')
        fld(s, 'yaw_rate_dps')
        self.walls = ToggleSwitch()
        self.walls.setToolTip('Walls along the road (buildings, barriers): static points beside the road')
        self.walls.toggled.connect(lambda on: self.set_option('walls', bool(on)))
        s.row('Roadside walls', self.walls)
        self.rec_lbl = label('', muted=True, small=True, wrap=True)
        self.b_rec = button('Choose…', None, 'folder', 'A CARLA drive folder written by Vantage (nuscenes/, '
                                                     'ground_truth/, rosbag/)')
        self.b_rec.clicked.connect(lambda: self.choose_recording())
        s.row('Recording', self.b_rec)
        s.add(self.rec_lbl)
        self.rec_tag = QComboBox()
        self.rec_tag.setToolTip('The recording\'s lidar')
        self.rec_tag.activated.connect(lambda i: self.set_option('recording_tag', self.rec_tag.itemData(i) or ''))
        s.row('Its lidar', self.rec_tag)
        fld(s, 'max_sweeps')
        bl.addWidget(s)
        s = Section('Doppler')
        self.counterpart = QComboBox()
        self.counterpart.setToolTip('The FMCW model a time-of-flight lidar of the rig is evaluated as, at its pose')
        for m in FM.library():
            self.counterpart.addItem(f'{m.vendor} {m.name}', m.id)
        self.counterpart.activated.connect(lambda i: self.set_option('counterpart', self.counterpart.itemData(i)))
        s.row('ToF lidars as', self.counterpart)
        fld(s, 'moving_mps')
        fld(s, 'velocity_sigma_mps')
        bl.addWidget(s)
        s = Section('Sun and cross-talk')
        seg = SegmentedControl([('spad', 'SPAD'), ('apd', 'APD')])
        seg.setToolTip('The time-of-flight receiver: photon-counting SPAD (most current 905 nm lidars) or a '
                       'linear-mode APD (amplifier-noise floor)')
        seg.changed.connect(lambda k: self.set_option('tof_receiver', k))
        self.segs['tof_receiver'] = seg
        s.row('ToF receiver', seg)
        seg = SegmentedControl([('auto', 'Auto'), ('dark', 'Dark'), ('sun', 'In sun')])
        seg.setToolTip('Where a time-of-flight datasheet range holds: in 100 klx sunlight when its note says so '
                       '(Auto), in the dark, or in sunlight')
        seg.changed.connect(lambda k: self.set_option('tof_reference', k))
        self.segs['tof_reference'] = seg
        s.row('Datasheet range', seg)
        fld(s, 'tof_filter_nm')
        fld(s, 'tof_ifov_mrad')
        fld(s, 'crosstalk_cars')
        fld(s, 'crosstalk_distance_m')
        bl.addWidget(s)
        bl.addStretch(1)
        scroll.setWidget(body)
        ol.addWidget(scroll, 1)
        foot = QWidget()
        fl = QVBoxLayout(foot)
        fl.setContentsMargins(8, 4, 8, 8)
        self.bar = QProgressBar()
        self.bar.setRange(0, 1000)
        self.bar.setTextVisible(False)
        self.bar.setFixedHeight(4)
        fl.addWidget(self.bar)
        self.status = label('', muted=True, small=True, wrap=True)
        fl.addWidget(self.status)
        row = QHBoxLayout()
        self.b_run = button('Run', 'primary', 'play', 'Run the study now')
        self.b_run.clicked.connect(lambda: self.run())
        self.b_export = button('Export…', None, 'export', 'JSON, CSV (lidars, sun), the 4D clouds as PCD and PLY with '
                                                          'velocity, and a ROS 2 bag of them')
        self.b_export.clicked.connect(lambda: self.export())
        row.addWidget(self.b_run)
        row.addStretch(1)
        row.addWidget(self.b_export)
        fl.addLayout(row)
        ol.addWidget(foot)
        return outer

    def _centre(self):
        w = QWidget()
        w.setObjectName('panel')
        cl = QVBoxLayout(w)
        cl.setContentsMargins(8, 4, 8, 6)
        self.head = label('', ink2=True, wrap=True)
        cl.addWidget(self.head)
        row = QHBoxLayout()
        self.c_lidar = QComboBox()
        self.c_lidar.setToolTip('The lidar whose 4D cloud is shown')
        self.c_lidar.activated.connect(lambda _: self.refresh())
        row.addWidget(self.c_lidar)
        self.c_mode = SegmentedControl(list(COLOUR_MODES), 'radial')
        self.c_mode.setToolTip('Radial velocity as measured (closing blue, receding red); the points\' own velocity '
                               'along the ray (the car\'s motion removed: static points grey); moving points by '
                               'Doppler')
        self.c_mode.changed.connect(lambda _: self._redraw())
        row.addWidget(self.c_mode)
        row.addStretch(1)
        cl.addLayout(row)
        self.view_box = QVBoxLayout()
        cl.addLayout(self.view_box, 1)
        self.legend = label('', muted=True, small=True, wrap=True)
        cl.addWidget(self.legend)
        return w

    def _right(self):
        right = QSplitter(Qt.Orientation.Vertical)
        right.setHandleWidth(2)
        a = QWidget()
        a.setObjectName('panel')
        al = QVBoxLayout(a)
        al.setContentsMargins(8, 6, 8, 6)
        al.addWidget(label('Lidars', bold=True))
        self.t_lid = _table((('Lidar', 'Tag, model (and the FMCW model a ToF lidar is evaluated as)'),
                             ('Points/s', 'Its point rate'), ('Velocity', 'Radial-velocity range and precision'),
                             ('Ego error', 'Mean error of the sensor velocity fitted from one sweep'),
                             ('Moving found', 'Moving points Doppler flags (recall)'),
                             ('Along ray', 'Recall of the moving points whose motion has a component along the ray'),
                             ('Precision', 'Flagged points that move')))
        self.t_lid.itemSelectionChanged.connect(self._lidar_row)
        al.addWidget(self.t_lid)
        al.addWidget(label('Objects of the shown sweep', bold=True))
        self.t_obj = _table((('Object', 'Name and class'), ('Speed', 'Over the ground, m/s'),
                             ('Points', 'Returns on it'), ('Found', 'Share flagged moving'),
                             ('Across', 'Share moving across the ray (Doppler cannot see it move)'),
                             ('Own v_r', 'Mean own radial velocity, m/s')))
        al.addWidget(self.t_obj)
        right.addWidget(a)
        b = QWidget()
        b.setObjectName('panel')
        bl = QVBoxLayout(b)
        bl.setContentsMargins(8, 6, 8, 6)
        bl.addWidget(label('Sun: time of flight against FMCW', bold=True))
        self.t_sun = _table((('Condition', 'Sun and sky (klx approximate)'),
                             ('ToF range', 'Range on a 10 % target, median over the beams near the horizon '
                                           '(worst azimuth) at a constant false-alarm rate'),
                             ('ToF false/s', 'False returns per second with the threshold fixed at the reference'),
                             ('FMCW range', 'The same for a coherent receiver'),
                             ('FMCW false/s', 'False returns per second of the coherent receiver'),
                             ('On the sun', 'Points per frame whose pixel holds the sun\'s disc')))
        bl.addWidget(self.t_sun, 1)
        self.xt = label('', small=True, wrap=True)
        self.xt.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        bl.addWidget(self.xt)
        self.checks = label('', small=True, wrap=True)
        self.checks.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        bl.addWidget(self.checks)
        self.assume = label(FM.ASSUMPTIONS, muted=True, small=True, wrap=True)
        bl.addWidget(self.assume)
        right.addWidget(b)
        right.setSizes([420, 420])
        return right

    def _ensure_view(self):
        if self.view3d is None:
            from ..viewport3d import make_viewport
            self.scene3d = FmcwScene(self.ctx, self)
            self.view3d = make_viewport(self.scene3d)
            self.view3d.cam.az, self.view3d.cam.el = 200.0, 32.0
            self.view_box.addWidget(self.view3d, 1)
            self.view3d.frame_box(np.array([-30.0, -25.0, 0.0]), np.array([45.0, 25.0, 4.0]), 1.0)
        return self.view3d

    def _engine(self):
        win = getattr(self.ctx, 'window', None)
        try:
            if win is not None and hasattr(win, '_shared_engine'):
                return win._shared_engine()
            return self.ctx.analysis.engine.share()
        except Exception:                                        # noqa: BLE001 — the scene without the body
            return None

    # ------------------------------------------------------------------ settings
    @property
    def settings(self) -> FM.FmcwSettings:
        return FM.settings_of(self.ctx.store.project)

    def set_option(self, key, value) -> bool:
        return self.set_options({key: value})

    def set_options(self, upd: dict) -> bool:
        """Change settings in one undo step (values clamped by FmcwSettings.from_dict); False when nothing changes."""
        if self._syncing:
            return False
        old = self.settings.to_dict()
        new = FM.FmcwSettings.from_dict(dict(old, **upd)).to_dict()
        if new == old:
            self._sync_fields()
            return False
        key = next(iter(upd))
        shown = new.get(key)
        shown = f'{shown:g}' if isinstance(shown, float) else (os.path.basename(shown) if key == 'recording'
                                                               else shown)

        def fn(p):
            p.analysis.fmcw = new
        self.ctx.store.edit('analysis', fn, f'FMCW lidar · {FIELD_LABEL.get(key, key)} {shown}')
        return True

    def set_scene(self, key) -> bool:
        sc = FM.SCENES.get(key)
        if sc is None:
            return False
        return self.set_options({'scene': key, 'ego_speed_kmh': sc['ego_kmh']})

    def choose_recording(self, folder=None) -> bool:
        if folder is None:
            folder = QFileDialog.getExistingDirectory(self, 'A CARLA drive folder with ground truth')
            if not folder:
                return False
        if not FM.recording_lidars(folder):
            self.ctx.msg('That folder holds no nuScenes lidar (nuscenes/v1.0-*/ with a LIDAR_ channel)', 6000)
            return False
        return self.set_options({'recording': folder, 'source': 'recording', 'recording_tag': ''})

    def _sync_fields(self):
        st = self.settings
        self._syncing = True
        try:
            for k, f in self.fields.items():
                f.set_value(float(getattr(st, k)))
            for k, sg in self.segs.items():
                sg.set_value(getattr(st, k))
            self.walls.setChecked(st.walls)
            i = self.counterpart.findData(st.counterpart)
            if i >= 0:
                self.counterpart.setCurrentIndex(i)
            rec = st.source == 'recording'
            self.rec_lbl.setText(st.recording or 'No recording chosen.')
            tags = FM.recording_lidars(st.recording) if st.recording and os.path.isdir(st.recording) else []
            self.rec_tag.clear()
            for t in tags:
                self.rec_tag.addItem(t, t)
            j = self.rec_tag.findData(st.recording_tag)
            self.rec_tag.setCurrentIndex(j if j >= 0 else 0)
            for w in (self.rec_tag, self.fields['max_sweeps']):
                w.setEnabled(rec)
            for w in (self.segs['scene'], self.fields['ego_speed_kmh'], self.fields['yaw_rate_dps'], self.walls):
                w.setEnabled(not rec)
        finally:
            self._syncing = False

    # ------------------------------------------------------------------ running
    def _key(self):
        return FM.cache_key(self.ctx.store.project)

    def _on_store(self, *_):
        self._sync_fields()
        if self.result is not None or self.isVisible():
            if self._for != self._key():
                self.stale = True
                self.refresh_status()
                if self.isVisible():
                    self.timer.start()

    def _replaced(self, *_):
        self._gen += 1
        self.result, self._for, self.stale, self.busy = None, None, False, False
        self._sync_fields()
        self.refresh()
        if self.isVisible():
            self.timer.start()

    def showEvent(self, e):
        super().showEvent(e)
        self._ensure_view()
        if self.result is None or self.stale:
            self.timer.start()

    def activate(self):
        self._ensure_view()
        if self.result is None or self.stale:
            self.run(quiet=True)

    def run(self, quiet=False) -> bool:
        project = self.ctx.store.project.snapshot()
        key = self._key()
        st = FM.settings_of(project)
        self._gen += 1
        gen = self._gen
        self.busy = True
        self.status_text = 'Reading the recording…' if st.source == 'recording' else 'Casting the sweeps…'
        self.bar.setValue(0)
        self.refresh_status()
        eng = self._engine() if st.source == 'scene' else None

        def thread():
            try:
                res = FM.run(project, eng, settings=st, progress=lambda f: self.sig.progress.emit(gen, f),
                             cancel=lambda: gen != self._gen)
                self.sig.done.emit(gen, (key, res))
            except RuntimeError:
                pass
            except Exception as e:                               # noqa: BLE001
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
        key, res = out
        if res is None:
            self.status_text = 'Cancelled'
            self.refresh_status()
            return
        self.result, self._for = res, key
        self.stale = key != self._key()
        n = sum(len(c.P) for c in res.clouds.values() if c is not None)
        self.status_text = f'{res.variant}: {len(res.lidars)} lidar(s), {n:,} points shown · {res.seconds:.1f} s'
        self.bar.setValue(1000)
        self.refresh()
        if self.stale and self.isVisible():
            self.timer.start()

    def _failed(self, gen, msg):
        if gen != self._gen:
            return
        self.busy = False
        self.status_text = 'Failed: ' + msg.splitlines()[0]
        self.refresh_status()

    # ------------------------------------------------------------------ views
    def shown_tag(self) -> str:
        return self.c_lidar.currentData() or ''

    def shown_cloud(self):
        res = self.result
        if res is None:
            return None
        return res.clouds.get(self.shown_tag())

    def _redraw(self):
        if self.view3d is not None:
            self.scene3d.changed.emit()
        self._legend()

    def _lidar_row(self):
        rows = self.t_lid.selectionModel().selectedRows() if self.t_lid.selectionModel() else []
        if rows and self.result is not None:
            tag = self.result.lidars[rows[0].row()]['tag']
            i = self.c_lidar.findData(tag)
            if i >= 0 and i != self.c_lidar.currentIndex():
                self.c_lidar.setCurrentIndex(i)
                self.refresh()

    def refresh_status(self):
        txt = self.status_text
        if self.stale and not self.busy and self.result is not None:
            txt = (txt + ' · ' if txt else '') + 'out of date'
        self.status.setText(txt)
        self.b_export.setEnabled(self.result is not None)

    def _legend(self):
        cl = self.shown_cloud()
        mode = self.c_mode.value() or 'radial'
        if cl is None:
            self.legend.setText('')
            return
        if mode == 'moving':
            self.legend.setText('Orange: moving by Doppler alone (own velocity along the ray above the threshold); '
                                'grey: static.')
            return
        v = cl.v_r if mode == 'radial' else cl.v_comp
        fin = np.isfinite(v)
        sc = max(float(np.percentile(np.abs(v[fin]), 98)) if fin.any() else 1.0, 1.0)
        self.legend.setText(f'Blue: closing · red: receding · full colour at ±{sc:.1f} m/s · '
                            f'{(~fin).sum():,} returns outside the velocity range (dark). ' + FM.SIGN + '.')

    def refresh(self):
        self.refresh_status()
        res = self.result
        cur = self.shown_tag()
        self.c_lidar.blockSignals(True)
        self.c_lidar.clear()
        for r in (res.lidars if res else []):
            self.c_lidar.addItem(f'{r["tag"]} · {r["name"]}', r['tag'])
        i = self.c_lidar.findData(cur)
        self.c_lidar.setCurrentIndex(i if i >= 0 else 0)
        self.c_lidar.blockSignals(False)
        for t in (self.t_lid, self.t_obj, self.t_sun):
            t.setRowCount(0)
        if res is None:
            self.head.setText('Runs on its own when the page shows: every lidar of the variant sees the scene (or the '
                              'recording), each return gets its radial velocity, and the ego motion and the moving '
                              'points are found from one sweep.')
            self.xt.setText('')
            self.checks.setText('')
            self._redraw()
            return
        if not res.lidars:
            self.head.setText('The variant has no visible lidar: add one (an FMCW model such as Aeva Aeries II, or any '
                              'lidar to compare it with its FMCW counterpart).')
            self._redraw()
            return
        rows = []
        for r in res.lidars:
            sm = r['summary']
            vm = r['velocity_max_mps']
            rows.append([f'{r["tag"]} · {r["name"]}' + ('' if r['fmcw'] else f' (as {r["doppler_name"]})'),
                         _rate(r['doppler_points_per_s']),
                         (f'±{vm:g} m/s' if vm else 'range not stated') + f', σ {r["spec"]["velocity_sigma"] * 100:.1f}'
                                                                          ' cm/s',
                         _num(sm.get('ego_error_mean_mps'), '{:.4f} m/s'), _pct(sm.get('recall'), 1),
                         _pct(sm.get('recall_along_ray'), 1), _pct(sm.get('precision'), 1)])
        _fill(self.t_lid, rows)
        row = res.lidar(self.shown_tag()) or res.lidars[0]
        met = row.get('metrics') or {}
        sm = row['summary']
        src = res.source
        where = (f'{src.get("label", "")} scene at {src.get("ego_speed_kmh", 0):g} km/h' if 'scene' in src else
                 f'recording {src.get("folder", "")} ({src.get("sweeps", 0)} sweeps of {src.get("lidar", "")}; '
                 f'velocities: {src.get("velocities", "")}; labels: {src.get("labels", "")})')
        as_ = '' if row['fmcw'] else f' evaluated as {row["doppler_name"]} at its pose'
        self.head.setText(
            f'{row["tag"]}{as_}, {where}: ego velocity within {_num(sm.get("ego_error_mean_mps"), "{:.4f}")} m/s '
            f'(mean over {sm.get("sweeps", 1)} sweep(s)); {_pct(sm.get("recall"), 1)} of the moving points found by '
            f'Doppler alone ({_pct(sm.get("recall_along_ray"), 1)} of those moving along the ray), precision '
            f'{_pct(sm.get("precision"), 1)}.')
        _fill(self.t_obj, [[f'{o["name"]} ({o["class"]})', f'{o["speed_mps"]:.1f}', o['points'],
                            _pct(o['share_found']), _pct(o['across_ray']),
                            _num(o.get('mean_own_v_r'), '{:+.2f}')] for o in met.get('objects', [])])
        sun_rows = []
        for x in row['sun']:
            t, f = x['tof'], x['fmcw']
            klx = f' ({x["klx_approx"]:g} klx)' if x['klx_approx'] else ''
            sun_rows.append([x['label'] + klx, f'{t["range_m"]:.0f} m ({t["range_min_m"]:.0f})',
                             _rate(t['false_per_s']),
                             f'{f["range_m"]:.0f} m ({f["range_min_m"]:.0f})', _rate(f['false_per_s']),
                             x['sun_points_per_frame']])
        _fill(self.t_sun, sun_rows)
        ct = row['crosstalk']
        ref = row['sun'][0]['tof_reference'] if row['sun'] else {}
        own = 'FMCW' if row['fmcw'] else 'time of flight'
        self.xt.setText(
            f'{row["tag"]} is {own}. ToF datasheet range taken '
            f'{"in 100 klx sunlight" if ref.get("condition") == "noon_ahead" else "in the dark"} '
            f'({ref.get("why", "")}). Cross-talk from {len(ct["interferers"])} other test car(s), '
            f'{ct["in_mutual_view"]} in mutual view: up to {_rate(ct["tof_points_per_s"])} interference points/s for '
            f'a time-of-flight lidar, none for FMCW ({ct["basis"]}).')
        chk = res.checks or {}
        parts = []
        rd = chk.get('radar_doppler') or {}
        if rd.get('n'):
            parts.append(f'CARLA\'s own radar Doppler against this model on the same truth: median |Δv_r| '
                         f'{rd["median_abs_mps"] * 100:.2f} cm/s over {rd["n"]:,} detections, signs agree '
                         f'{_pct(rd.get("sign_agreement"), 1)}')
        dv = chk.get('differenced_velocity') or {}
        if dv.get('n'):
            parts.append(f'boxes differenced against CARLA\'s velocities: median {dv["median_mps"] * 100:.1f} cm/s, '
                         f'95 % {dv["p95_mps"] * 100:.1f} cm/s ({dv["n"]} moving boxes)')
        self.checks.setText(('Cross-checks: ' + '; '.join(parts) + '.') if parts else '')
        self._redraw()

    def export(self, folder=None):
        res = self.result
        if res is None:
            return []
        if folder is None:
            folder = QFileDialog.getExistingDirectory(self, 'Export the FMCW lidar study into')
            if not folder:
                return []
        from ...core import provenance as PV
        p = self.ctx.store.project
        paths = PV.stamp_paths(FM.export(res, p, folder), p, folder)
        try:
            bag = FM.write_bag(res, folder)
            if bag:
                paths.append(bag)
        except Exception as ex:                                  # noqa: BLE001 — the files are written already
            self.ctx.msg(f'The ROS 2 bag was not written: {ex}', 6000)
        self.ctx.msg(f'Exported the FMCW lidar study to {folder}', 5000)
        return paths
