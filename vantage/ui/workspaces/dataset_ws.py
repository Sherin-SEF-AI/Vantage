"""Tools ▸ Sample dataset: write a short synthetic drive recorded by exactly the planned rig (cameras, lidars, radars,
GNSS and IMU) as a nuScenes version folder and a ROS 2 bag, to test ingestion, labelling tools and training loaders
before any hardware exists. The settings are saved with the project (undoable); the preview renders one instant in the
background; generating is a background job with progress and Cancel."""
from __future__ import annotations

import os
import threading
import traceback

from PyQt6.QtCore import QObject, Qt, QTimer, pyqtSignal
from PyQt6.QtWidgets import (QApplication, QFileDialog, QGridLayout, QHBoxLayout, QPlainTextEdit, QProgressBar,
                             QScrollArea, QSlider, QSplitter, QVBoxLayout, QWidget)

from ...core import dataset as DSET
from ...core.timing import plan_outline
from ..datasetview import CameraGrid, TopView
from ..theme import THEME, mono
from ..widgets import ScrubSpinBox, SegmentedControl, ToggleSwitch, button, hline, label
from .recording_ws import Tile

JOB = 'dataset'
PREVIEW_WIDTH = 320                        # px across each camera thumbnail
# (group, key, label, unit, decimals, step, scale, tip); scale: stored value × scale = shown value
FIELDS = (
    ('Drive', 'duration_s', 'Duration', 's', 1, 0.5, 1.0, 'Length of the recorded drive'),
    ('Drive', 'speed_kmh', 'Speed', 'km/h', 0, 5.0, 1.0,
     'The ego\'s set speed: it starts at 80 % of it and reaches it smoothly within the first 3 s'),
    ('Drive', 'seed', 'Seed', '', 0, 1.0, 1.0, 'The scene\'s layout and traffic: the same seed gives the same drive'),
    ('Cameras', 'image_scale', 'Image size', '%', 0, 5.0, 100.0,
     'Images at this share of each sensor\'s width and height (100 % = full size; the intrinsics are scaled to it)'),
    ('Cameras', 'jpeg_quality', 'JPEG quality', '', 0, 5.0, 1.0, 'JPEG quality of the written images (30 to 100)'),
    ('Lidar', 'lidar_noise_cm', 'Range noise', 'cm', 1, 0.5, 1.0, 'Gaussian range noise of the lidar returns (1 σ)'),
    ('Output', 'keyframe_hz', 'Keyframes', 'Hz', 1, 0.5, 1.0,
     'nuScenes samples (annotated keyframes) per second; the frames between are sweeps'),
    ('Position', 'origin_lat', 'Latitude', '°', 6, 0.001, 1.0,
     'WGS84 latitude of the scene\'s origin (GNSS fixes and the global frame are relative to it)'),
    ('Position', 'origin_lon', 'Longitude', '°', 6, 0.001, 1.0, 'WGS84 longitude of the scene\'s origin'),
    ('Position', 'origin_alt', 'Height', 'm', 1, 1.0, 1.0, 'Ellipsoidal height (WGS84) of the scene\'s origin'),
)
TOGGLES = {
    'Cameras': (('rolling_shutter', 'Rolling shutter',
                 'Each band of rows at its own exposure time over the readout (core.timing\'s readout and exposure); '
                 'off: every row at the middle row\'s time'),
                ('undistort', 'Rectified images',
                 'Write pinhole images whose nuScenes camera_intrinsic is exact (the devkit projects onto them); off: '
                 'the real fisheye images, with the Kannala-Brandt coefficients in vantage_distortion')),
    'Lidar': (('drop_self_hits', 'Drop hits on the own vehicle',
               'Leave out the returns from the own body and rig (a real lidar records them)'),),
    'Output': (('nuscenes', 'nuScenes', 'Write the nuScenes version folder v1.0-vantage'),
               ('rosbag', 'ROS 2 bag', 'Write a rosbag2 bag (MCAP) with every sensor\'s topics, /tf and the ground '
                                       'truth')),
}


class _Sig(QObject):
    preview = pyqtSignal(int, object)
    preview_failed = pyqtSignal(int, str)
    progress = pyqtSignal(float, str)
    done = pyqtSignal(object)
    failed = pyqtSignal(str)


def _fmt_bytes(n) -> str:
    return f'{n / 1e9:.2f} GB' if n >= 1e9 else f'{n / 1e6:.1f} MB' if n >= 1e5 else f'{n / 1e3:.0f} kB'


def short_path(path, n=60) -> str:
    """A folder for display: the home folder as ~, a long path as its first two and last two parts."""
    home = os.path.expanduser('~')
    if path == home or path.startswith(home + os.sep):
        path = '~' + path[len(home):]
    parts = path.split(os.sep)
    if len(path) <= n or len(parts) <= 5:
        return path
    return os.sep.join(parts[:3] + ['…'] + parts[-2:])


def _fmt_secs(s) -> str:
    return f'{s / 60:.1f} min' if s >= 90 else f'{s:.0f} s'


class DatasetPage(QWidget):
    def __init__(self, ctx):
        super().__init__()
        self.ctx = ctx
        self.setObjectName('base')
        self.sig = _Sig()
        self.sig.preview.connect(self._preview_done)
        self.sig.preview_failed.connect(self._preview_failed)
        self.sig.progress.connect(self._progress)
        self.sig.done.connect(self._done)
        self.sig.failed.connect(self._failed)
        self.result: DSET.DatasetResult | None = None
        self.preview_data = None
        self.busy = False                  # a generation is running
        self._cancel = threading.Event()
        self._pgen = 0
        self._pbusy = False
        self._pagain = False
        self._dirty = True
        self._pstale = True
        self._syncing = False
        self.timer = QTimer(self)
        self.timer.setSingleShot(True)
        self.timer.setInterval(400)
        self.timer.timeout.connect(self.run_preview)
        lay = QHBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        split = QSplitter(Qt.Orientation.Horizontal)
        split.setHandleWidth(2)
        split.addWidget(self._left())
        split.addWidget(self._right())
        split.setSizes([330, 1270])
        lay.addWidget(split)
        ctx.store.changed.connect(self._on_change)
        ctx.store.project_replaced.connect(self._on_replaced)
        THEME.changed.connect(self._on_theme)
        self.refresh()

    # ------------------------------------------------------------------ settings panel
    def _left(self):
        w = QWidget()
        w.setObjectName('panel')
        ll = QVBoxLayout(w)
        ll.setContentsMargins(8, 6, 8, 8)
        ll.setSpacing(4)
        top = QHBoxLayout()
        top.addWidget(label('Sample dataset'))
        top.addStretch(1)
        top.addWidget(label('nuScenes · ROS 2 bag', muted=True, small=True))
        ll.addLayout(top)
        ll.addWidget(hline())
        ll.addWidget(label('A short synthetic drive recorded by exactly this rig: its cameras, lidars, radars, GNSS '
                           'and IMU, their frames, timing and calibration, written as a nuScenes version folder and a '
                           'ROS 2 bag to test ingestion, labelling tools and loaders before the hardware exists.',
                           muted=True, small=True, wrap=True))
        self.fields, self.toggles = {}, {}
        group, grid = None, None

        def start(g):
            nonlocal grid
            if grid is not None:
                ll.addLayout(grid)
            ll.addSpacing(6)
            ll.addWidget(label(g, bold=True))
            if g == 'Drive':
                self.scene = SegmentedControl([(k, DSET.SCENE_LABEL[k]) for k in DSET.SCENES], 'urban')
                self.scene.setToolTip('Urban: a street with a bend, parked cars, cyclists and pedestrians. Highway: a '
                                      'dual carriageway with faster traffic')
                self.scene.changed.connect(lambda k: self._set(f'{DSET.SCENE_LABEL[k].lower()} scene', scene=k))
                ll.addWidget(self.scene)
            elif g == 'Output':
                self.names = SegmentedControl([('tags', 'Rig tags'), ('nuscenes', 'nuScenes names')], 'tags')
                self.names.setToolTip('nuScenes channel names. Rig tags: CAM_<TAG>, LIDAR_<TAG>, RADAR_<TAG> as '
                                      'the nuscenes_calib export. nuScenes names: CAM_FRONT … by heading, the '
                                      'highest lidar LIDAR_TOP, RADAR_FRONT … by corner, for loaders and devkit '
                                      'helpers that expect them')
                self.names.changed.connect(lambda k: self._set(
                    'channels named ' + ('as nuScenes' if k == 'nuscenes' else 'by rig tags'), channel_names=k))
                ll.addWidget(self.names)
            elif g == 'Cameras':
                self.rate = SegmentedControl([('lidar', 'Lidar rate'), ('module', 'Module fps')], 'lidar')
                self.rate.setToolTip('Lidar rate: one frame per lidar rotation, at the sync plan\'s trigger time. '
                                     'Module fps: every frame the module delivers (on its trigger line\'s pulses)')
                self.rate.changed.connect(lambda k: self._set(
                    'cameras at the ' + ('lidar rate' if k == 'lidar' else 'module fps'), camera_rate=k))
                ll.addWidget(self.rate)
            for key, text, tip in TOGGLES.get(g, ()):
                t = ToggleSwitch(text)
                t.setToolTip(tip)
                t.clicked.connect(lambda on, k=key, x=text: self._set(f'{x.lower()} {"on" if on else "off"}',
                                                                      **{k: on}))
                ll.addWidget(t)
                self.toggles[key] = t
            grid = QGridLayout()
            grid.setVerticalSpacing(2)
            grid.setHorizontalSpacing(6)
            grid.setColumnStretch(2, 1)
            return grid
        for g, key, text, unit, dec, step, scale, tip in FIELDS:
            if g != group:
                group = g
                grid = start(g)
            f = ScrubSpinBox('', 'plain', dec, unit, step=step, width=118)
            f.setToolTip(tip)
            f.committed.connect(lambda v, k=key, s=scale, t=text, u=unit: self._set(f'{t.lower()} {v:g} {u}'.strip(),
                                                                                  **{k: v / s}))
            a = label(text, ink2=True)
            a.setFixedWidth(104)
            a.setToolTip(tip)
            i = grid.rowCount()
            grid.addWidget(a, i, 0)
            grid.addWidget(f, i, 1)
            self.fields[key] = (f, scale)
        ll.addLayout(grid)
        ll.addSpacing(6)
        ll.addWidget(label('Limits', bold=True))
        self.limits = label(DSET.ASSUMPTIONS, muted=True, small=True, wrap=True)
        ll.addWidget(self.limits)
        ll.addStretch(1)
        bl = QHBoxLayout()
        b_def = self.b_defaults = button('Defaults', 'ghost', None, 'Back to the default dataset settings')
        b_def.clicked.connect(self._defaults)
        bl.addWidget(b_def)
        bl.addStretch(1)
        ll.addLayout(bl)
        sc = QScrollArea()
        sc.setWidgetResizable(True)
        sc.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        sc.setWidget(w)
        sc.setMinimumWidth(300)
        return sc

    # ------------------------------------------------------------------ preview, generate, result
    def _right(self):
        right = QSplitter(Qt.Orientation.Vertical)
        right.setHandleWidth(2)
        tw = QWidget()
        tw.setObjectName('panel')
        tl = QVBoxLayout(tw)
        tl.setContentsMargins(10, 8, 10, 8)
        tl.setSpacing(6)
        tiles = QHBoxLayout()
        tiles.setSpacing(8)
        self.tiles = {}
        for key, caption, tip in (
                ('frames', 'Frames', 'Camera images, lidar sweeps and radar scans the drive records'),
                ('samples', 'Keyframes', 'nuScenes samples: annotated keyframes'),
                ('size', 'Size', 'Approximate size of what is written (both outputs)'),
                ('time', 'Time', 'Approximate time to generate on this machine')):
            t = self.tiles[key] = Tile(caption, tip)
            tiles.addWidget(t, 1)
        tl.addLayout(tiles)
        head = QHBoxLayout()
        head.addWidget(label('Preview', bold=True))
        self.t_label = label('t = 0.0 s', mono_=True)
        head.addWidget(self.t_label)
        self.slider = QSlider(Qt.Orientation.Horizontal)
        self.slider.setToolTip('The instant of the drive the preview shows')
        self.slider.valueChanged.connect(lambda v: self.t_label.setText(f't = {v / 10:.1f} s'))
        self.slider.sliderReleased.connect(self._slider_released)
        self.slider.setMinimumWidth(160)
        head.addWidget(self.slider, 1)
        self.reach = SegmentedControl([('20', '20 m'), ('40', '40 m'), ('80', '80 m')], '40')
        self.reach.setToolTip('How far the top view reaches from the vehicle')
        self.reach.changed.connect(lambda v: self.top.set_reach(float(v)))
        head.addWidget(self.reach)
        self.b_preview = button('Preview', 'ghost', 'refresh', 'Render this instant of the drive again')
        self.b_preview.clicked.connect(self.run_preview)
        head.addWidget(self.b_preview)
        tl.addLayout(head)
        body = QSplitter(Qt.Orientation.Horizontal)
        body.setHandleWidth(2)
        self.grid = CameraGrid()
        self.grid.setToolTip('The cameras at the preview instant, as written (fisheye or rectified), at preview size')
        self.top = TopView()
        self.top.setToolTip('Lidar points (coloured by height), radar detections and the ground-truth boxes at the '
                            'preview instant, in the vehicle frame')
        body.addWidget(self.grid)
        body.addWidget(self.top)
        body.setSizes([760, 420])
        tl.addWidget(body, 1)
        self.p_status = label('', muted=True, small=True, wrap=True)
        tl.addWidget(self.p_status)
        right.addWidget(tw)
        gw = QWidget()
        gw.setObjectName('panel')
        gl = QVBoxLayout(gw)
        gl.setContentsMargins(10, 6, 10, 8)
        gl.setSpacing(4)
        row = QHBoxLayout()
        self.b_generate = button('Generate…', 'primary', 'export',
                                 'Choose a folder and write the drive there (nuscenes/, rosbag/, dataset.json)')
        self.b_generate.clicked.connect(lambda: self.generate())
        row.addWidget(self.b_generate)
        self.bar = QProgressBar()
        self.bar.setRange(0, 1000)
        self.bar.setTextVisible(False)
        self.bar.setFixedHeight(6)
        row.addWidget(self.bar, 1)
        self.b_cancel = button('Cancel', 'ghost', 'stop', 'Stop writing; what was written is removed')
        self.b_cancel.clicked.connect(self.cancel)
        row.addWidget(self.b_cancel)
        self.b_copy = button('Copy snippet', 'ghost', 'copy', 'Copy the devkit snippet and the ros2 bag command')
        self.b_copy.clicked.connect(self._copy)
        row.addWidget(self.b_copy)
        gl.addLayout(row)
        self.g_status = label('', muted=True, small=True, wrap=True)
        gl.addWidget(self.g_status)
        self.out = QPlainTextEdit()
        self.out.setReadOnly(True)
        self.out.setLineWrapMode(QPlainTextEdit.LineWrapMode.WidgetWidth)     # long paths wrap, never clipped
        self.out.setFont(mono(11))
        self.out.setPlaceholderText('What was written appears here, with the snippet to load it.')
        gl.addWidget(self.out, 1)
        right.addWidget(gw)
        right.setSizes([640, 240])
        self._update_buttons()
        return right

    # ------------------------------------------------------------------ settings
    @property
    def settings(self) -> DSET.DatasetSettings:
        return DSET.DatasetSettings.from_dict(self.ctx.store.project.analysis.dataset)

    def _set(self, text, **kw):
        """Change settings as one undo step (saved with the project); False when nothing changes."""
        if self._syncing:
            return False
        st = self.settings
        for k, v in kw.items():
            setattr(st, k, v)
        new = DSET.DatasetSettings.from_dict(st.to_dict()).to_dict()
        if new == self.settings.to_dict():
            self._sync_fields()                # a clamped value: show what is stored
            return False

        def fn(p):
            p.analysis.dataset = new
        self.ctx.store.edit('analysis', fn, f'Sample dataset · {text}')
        return True

    def _defaults(self):
        if self.ctx.store.project.analysis.dataset:
            def fn(p):
                p.analysis.dataset = {}
            self.ctx.store.edit('analysis', fn, 'Sample dataset · defaults')

    def _sync_fields(self):
        self._syncing = True
        try:
            st = self.settings.to_dict()
            for key, (f, scale) in self.fields.items():
                f.set_value(st[key] * scale)
            self.scene.set_value(st['scene'])
            self.rate.set_value(st['camera_rate'])
            self.names.set_value(st['channel_names'])
            for key, t in self.toggles.items():
                t.blockSignals(True)
                t.setChecked(bool(st[key]))
                t.blockSignals(False)
            v = self.slider.value()
            self.slider.blockSignals(True)
            self.slider.setRange(0, int(round(st['duration_s'] * 10)))
            self.slider.setValue(min(v, self.slider.maximum()))
            self.slider.blockSignals(False)
            self.t_label.setText(f't = {self.slider.value() / 10:.1f} s')
        finally:
            self._syncing = False

    # ------------------------------------------------------------------ state
    def _on_change(self, *_):
        self._pstale = True
        if self.isVisible():
            self.refresh()
            self.timer.start()
        else:
            self._dirty = True

    def _on_replaced(self, *_):
        self._pgen += 1                        # a preview still rendering is for the old project
        self._pbusy = False
        self.preview_data = None
        self.result = None
        self.out.setPlainText('')
        self.g_status.setText('')
        self.grid.set_empty('No preview yet.')
        self.top.set_empty('No preview yet.')
        self._on_change()

    def _on_theme(self, *_):
        self.grid.update()
        self.top.update()

    def showEvent(self, e):
        super().showEvent(e)
        self.activate()

    def activate(self):
        if self._dirty:
            self.refresh()
        if self._pstale or self.preview_data is None:
            self.timer.start()

    def refresh(self):
        """Fields and the estimate (cheap)."""
        self._dirty = False
        self._sync_fields()
        p = self.ctx.store.project
        est = DSET.estimate(p, self.settings)
        self.estimate = est
        if not est['ok']:
            for k in self.tiles:
                self.tiles[k].set('—', '', '')
            self.g_status.setText(est['reason'])
        else:
            st = self.settings
            self.tiles['frames'].set(f'{est["images"] + est["sweeps"] + est["scans"]:,}', 'frames',
                                     f'{est["images"]} images · {est["sweeps"]} sweeps · {est["scans"]} scans')
            self.tiles['samples'].set(f'{est["samples"]}', 'samples', f'at {est["keyframe_hz"]:g} Hz')
            outs = [n for n, on in (('nuScenes', st.nuscenes), ('ROS 2 bag', st.rosbag)) if on]
            self.tiles['size'].set(_fmt_bytes(est['bytes']).split()[0], _fmt_bytes(est['bytes']).split()[1],
                                   ' + '.join(outs) or 'nothing selected', None if outs else 'warning')
            self.tiles['time'].set(f'~{_fmt_secs(est["seconds"]).split()[0]}', _fmt_secs(est['seconds']).split()[1],
                                   'approximate, this machine')
            if not self.busy and not self.result:
                self.g_status.setText(' '.join(est['notes']))
        self._update_buttons()

    def _update_buttons(self):
        ok = getattr(self, 'estimate', {}).get('ok', False)
        st = self.settings
        self.b_generate.setEnabled(ok and not self.busy and (st.nuscenes or st.rosbag))
        self.b_cancel.setEnabled(self.busy)
        self.b_copy.setEnabled(self.result is not None)
        self.b_preview.setEnabled(ok)

    def _engine(self):
        win = self.ctx.window
        return win._shared_engine() if win is not None and hasattr(win, '_shared_engine') else \
            self.ctx.analysis.engine.share()

    # ------------------------------------------------------------------ preview
    def _slider_released(self):
        self._pstale = True
        self.run_preview()

    def run_preview(self):
        """Render the preview instant on a thread; a request while one renders runs again after it."""
        if not getattr(self, 'estimate', {}).get('ok', False):
            self.grid.set_empty(self.estimate.get('reason', 'Nothing to record.'))
            self.top.set_empty('')
            return False
        if self._pbusy:
            self._pagain = True
            return False
        p = self.ctx.store.project
        snap = p.snapshot()
        st = self.settings
        t = self.slider.value() / 10.0
        eng = self._engine()
        self._pgen += 1
        gen = self._pgen
        self._pbusy, self._pagain, self._pstale = True, False, False
        self.p_status.setText(f'Rendering the drive at {t:.1f} s…')

        def work():
            try:
                data = DSET.preview(snap, eng, st, t, width=PREVIEW_WIDTH)
                self.sig.preview.emit(gen, (data, plan_outline(snap.vehicle.mesh)))
            except RuntimeError:               # the window closed while rendering
                pass
            except Exception as e:
                try:
                    self.sig.preview_failed.emit(gen, f'{type(e).__name__}: {e}\n{traceback.format_exc()}')
                except RuntimeError:
                    pass
        threading.Thread(target=work, daemon=True).start()
        return True

    def _preview_done(self, gen, out):
        if gen != self._pgen:
            return
        self._pbusy = False
        data, outline = out
        self.preview_data = data
        self.grid.set_cameras(data['cameras'])
        if not data['cameras']:
            self.grid.set_empty('No visible camera.')
        self.top.set_data(data, outline)
        nb = len(data['boxes'])
        self.p_status.setText(f'{data["t"]:.1f} s · {data["speed_kmh"]:.0f} km/h · {len(data["lidar"]):,} lidar '
                              f'points · {sum(len(r["points"]) for r in data["radar"])} radar detections · {nb} '
                              f'annotated object{"s" * (nb != 1)} within {DSET.DS.ANNOT_RANGE_M:g} m')
        if self._pagain or self._pstale:
            self.timer.start()

    def _preview_failed(self, gen, msg):
        if gen != self._pgen:
            return
        self._pbusy = False
        self.p_status.setText('Preview failed: ' + msg.splitlines()[0])
        if self._pagain:
            self.timer.start()

    # ------------------------------------------------------------------ generate
    def generate(self, folder=None) -> bool:
        """Write the dataset into folder (asks for one when None) as a background job."""
        if self.busy:
            return False
        if folder is None:
            folder = QFileDialog.getExistingDirectory(self, 'Write the sample dataset into')
            if not folder:
                return False
        p = self.ctx.store.project
        st = self.settings
        try:
            DSET.check_out_dir(folder, st, p)
        except ValueError as e:
            self.g_status.setText(str(e))
            self.ctx.msg(str(e), 6000)
            return False
        snap = p.snapshot()
        eng = self._engine()
        self._for_project = p
        self._cancel.clear()
        self.busy = True
        self.result = None
        self.bar.setValue(0)
        self.out.setPlainText('')
        self.g_status.setText(f'Writing the sample dataset into {short_path(folder)}…')
        self._update_buttons()
        cancel_me = self._cancel

        def work(progress, cancel):
            def prog(f, text=''):
                progress(f, text)
                self.sig.progress.emit(f, text)
            return DSET.generate(snap, eng, st, folder, prog, lambda: cancel() or cancel_me.is_set())
        win = self.ctx.window
        if win is not None and hasattr(win, 'run_job'):
            if win.run_job(JOB, 'Sample dataset', work, self._done, self._failed):
                return True
            self.busy = False
            self.g_status.setText('A sample dataset is already being written.')
            self._update_buttons()
            return False

        def thread():
            try:
                self.sig.done.emit(work(lambda *a: None, lambda: False))
            except RuntimeError:
                pass
            except Exception as e:
                try:
                    self.sig.failed.emit(f'{type(e).__name__}: {e}\n{traceback.format_exc()}')
                except RuntimeError:
                    pass
        threading.Thread(target=thread, daemon=True).start()
        return True

    def cancel(self):
        if self.busy:
            self._cancel.set()
            self.g_status.setText('Cancelling…')

    def _progress(self, f, text):
        if self.busy:
            self.bar.setValue(int(f * 1000))
            if not self._cancel.is_set():
                self.g_status.setText(f'Writing · {f * 100:.0f} %' + (f' · {text}' if text else ''))

    def _done(self, res):
        self.busy = False
        self.bar.setValue(0 if res is None else 1000)
        if res is None:
            self.g_status.setText('Cancelled: what was written so far was removed.')
            self.ctx.msg('Sample dataset cancelled', 4000)
            self._update_buttons()
            return
        self.result = res
        self.out.setPlainText(self._result_text(res))
        self.g_status.setText(f'Written in {res.seconds:.0f} s · {_fmt_bytes(res.sizes["total"])}')
        if getattr(self, '_for_project', None) is self.ctx.store.project:     # not a project opened meanwhile
            self.ctx.store.project.add_log('dataset', f'Sample dataset written to {res.out_dir} · '
                                                      f'{res.counts["samples"]} samples · '
                                                      f'{_fmt_bytes(res.sizes["total"])}')
        self.ctx.msg(f'Sample dataset written to {short_path(res.out_dir)}', 6000)
        self.out.setToolTip(res.out_dir)
        self._update_buttons()

    def _failed(self, msg):
        self.busy = False
        self.bar.setValue(0)
        self.g_status.setText('Failed: ' + msg.splitlines()[0])
        self._update_buttons()

    @staticmethod
    def _result_text(res) -> str:
        c = res.counts
        L = [f'Written to {short_path(res.out_dir)} in {res.seconds:.0f} s · {_fmt_bytes(res.sizes["total"])}']
        if res.nuscenes_dir:
            L.append(f'nuScenes   {os.path.relpath(res.nuscenes_dir, res.out_dir)}/ ({DSET.VERSION}): '
                     f'{c["samples"]} samples, {c["sample_data"]} '
                     f'sample_data, {c["annotations"]} annotations of {c["instances"]} instances · '
                     f'{_fmt_bytes(res.sizes["nuscenes"])}')
        if res.bag_dir:
            topics = len((res.bag or {}).get('topics', {}))
            L.append(f'ROS 2 bag  {os.path.relpath(res.bag_dir, res.out_dir)}/: {c["bag_messages"]} messages on '
                     f'{topics} topics · '
                     f'{_fmt_bytes(res.sizes["rosbag"])}')
        L += ['', 'Load it:', DSET.load_snippet(res), '']
        L += [f'  {ch["channel"] or ch["topic"]:<14} {ch["kind"]:<7} {ch["frames"]:>5} frames'
              + (f' · {ch["keyframes"]} keyframes' if ch['channel'] else '') for ch in res.channels]
        if res.notes:
            L += ['', 'Notes:'] + [f'  {n}' for n in res.notes]
        return '\n'.join(L)

    def _copy(self):
        if self.result is not None:
            QApplication.clipboard().setText(DSET.load_snippet(self.result))
            self.ctx.msg('Copied the load snippet', 3000)
