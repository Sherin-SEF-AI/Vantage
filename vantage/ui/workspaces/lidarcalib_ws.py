"""Calibrate ▸ Lidar–camera calibration: where a lidar sits relative to a camera (and so in the vehicle), measured from
a checkerboard held up at several poses both see, against the plan. Plan the board poses, capture them synthetically
(with an injected mounting error to see what the solve recovers) or load a folder of real image + point-cloud pairs,
solve, check the overlay of the lidar's points on the image, and apply the measured pose to the variant or as a new
variant. The capture and the solve run as a background job."""
from __future__ import annotations

import json
import math
import threading
import traceback

from PyQt6.QtCore import QObject, Qt, QTimer, pyqtSignal
from PyQt6.QtGui import QColor
from PyQt6.QtWidgets import (QAbstractItemView, QComboBox, QFileDialog, QGridLayout, QHBoxLayout, QHeaderView,
                             QProgressBar, QScrollArea, QSplitter, QTableWidget, QTableWidgetItem, QVBoxLayout,
                             QWidget)

from ...core import lidar_calib as LC
from ..lidarcalibview import OverlayView, PlanView
from ..theme import THEME, STATUS, mono
from ..widgets import ROW_H, ScrubSpinBox, SegmentedControl, StatusLine, ToggleSwitch, button, hline, label
from .recording_ws import Tile

JOB = 'lidar_calib'
VERDICT_WORD = {'good': 'Within tolerance', 'warning': 'Adjust', 'serious': 'Re-mount',
                'unobservable': 'Not determined', 'failed': 'Failed'}
STATUS_WORD = {'used': 'Used', 'outlier': 'Set aside', 'no_camera': 'Not in the image', 'no_lidar': 'Not in the cloud',
               'unreadable': 'Unreadable'}
# (group, key, label, unit, decimals, step, tip)
FIELDS = (
    ('Board', 'cols', 'Inner corners across', '', 0, 1.0, 'Inner corners along the board\'s width (squares − 1)'),
    ('Board', 'rows', 'Inner corners down', '', 0, 1.0, 'Inner corners along the board\'s height (squares − 1)'),
    ('Board', 'square_mm', 'Square', 'mm', 0, 5.0, 'Side of one printed square'),
    ('Board', 'margin_mm', 'White margin', 'mm', 0, 5.0, 'The white border round the checker, the same all round: the '
                                                          'lidar sees the board\'s outer edge'),
    ('Poses', 'poses', 'Poses', '', 0, 1.0, 'Board poses in the plan'),
    ('Poses', 'near_m', 'Nearest', 'm', 1, 0.5, 'The nearest board, from the camera'),
    ('Poses', 'far_m', 'Farthest', 'm', 1, 0.5, 'The farthest board, from the camera (kept nearer when the lidar\'s '
                                                'channels or the image would be too coarse there)'),
    ('Synthetic capture', 'range_sigma_mm', 'Range noise', 'mm', 1, 1.0, 'The lidar\'s range noise, 1σ'),
    ('Synthetic capture', 'image_noise', 'Image noise', 'grey', 1, 0.5, 'The camera\'s image noise, 1σ, grey levels'),
    ('Synthetic capture', 'seed', 'Seed', '', 0, 1.0, 'Seed of the synthetic noise'),
    ('Injected mounting error', 'err_x_mm', 'Δx', 'mm', 1, 1.0, 'The lidar mounted this far forward of its planned '
                                                               'place (vehicle frame) in the synthetic capture'),
    ('Injected mounting error', 'err_y_mm', 'Δy', 'mm', 1, 1.0, 'The lidar mounted this far left of its planned place'),
    ('Injected mounting error', 'err_z_mm', 'Δz', 'mm', 1, 1.0, 'The lidar mounted this far above its planned place'),
    ('Injected mounting error', 'err_yaw_deg', 'Δyaw', '°', 2, 0.05, 'Added to the lidar\'s yaw'),
    ('Injected mounting error', 'err_pitch_deg', 'Δpitch', '°', 2, 0.05, 'Added to the lidar\'s pitch (positive down)'),
    ('Injected mounting error', 'err_roll_deg', 'Δroll', '°', 2, 0.05, 'Added to the lidar\'s roll'),
    ('Tolerance', 'pos_tol_mm', 'Position', 'mm', 1, 1.0, 'The lidar\'s position may be this far off the plan'),
    ('Tolerance', 'ang_tol_deg', 'Angle', '°', 2, 0.05, 'Its orientation this far'),
)
POSE_COLS = (('Pose', 'The capture'), ('Status', 'Whether the board was found by both and used'),
             ('Corner px', 'RMS of the board\'s corners against the solved board pose, image pixels'),
             ('Points', 'Lidar points on the board'),
             ('Plane mm', 'RMS of the lidar\'s board points about the board plane'),
             ('Off mm', 'The lidar\'s board centre off the plane the camera measured, through the solved extrinsic'),
             ('Off °', 'Angle between the lidar\'s and the camera\'s board planes'),
             ('Edge mm', 'RMS by which the board\'s edge brackets break the camera-measured outline'))


class _Sig(QObject):
    done = pyqtSignal(int, object)
    failed = pyqtSignal(int, str)
    progress = pyqtSignal(int, float, str)
    planned = pyqtSignal(int, object)


def inputs_key(project) -> str:
    """Everything a calibration reads, as text: a result made from other inputs is stale."""
    lib = project.section('library')
    return json.dumps({'variant': project.variant.to_dict(), 'settings': project.analysis.lidar_calib,
                       'sensors': lib['sensors'], 'lenses': lib['lenses'], 'modules': lib['modules']},
                      sort_keys=True, default=str)


def _num(text, tip=None, color=None):
    it = QTableWidgetItem(text)
    it.setFont(mono(12))
    it.setTextAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
    if tip:
        it.setToolTip(tip)
    if color is not None:
        it.setForeground(color)
    return it


def _f(v, nd=1, signed=False):
    if v is None or not isinstance(v, (int, float)) or not math.isfinite(v):
        return '—'
    return (f'{v:+.{nd}f}' if signed else f'{v:.{nd}f}').replace('-', '−')


class LidarCalibPage(QWidget):
    def __init__(self, ctx):
        super().__init__()
        self.ctx = ctx
        self.result: LC.LidarCalibResult | None = None
        self._for = None                    # (project, inputs key) of the result
        self.busy = False
        self.status_text = ''
        self.plan = None                    # {'cam', 'lidar', 'poses', 'notes', 'predicted', 'pairs', 'error'}
        self._gen = 0
        self._plan_gen = 0
        self._dirty = True
        self._syncing = False
        self.sig = _Sig()
        self.sig.done.connect(self._done)
        self.sig.failed.connect(self._failed)
        self.sig.progress.connect(lambda g, f, t: self._progress(f, t) if g == self._gen else None)
        self.sig.planned.connect(self._planned)
        self._plan_timer = QTimer(self)
        self._plan_timer.setSingleShot(True)
        self._plan_timer.setInterval(250)
        self._plan_timer.timeout.connect(self._start_plan)
        self.setObjectName('base')
        lay = QHBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        split = QSplitter(Qt.Orientation.Horizontal)
        split.setHandleWidth(2)
        split.addWidget(self._left())
        split.addWidget(self._centre())
        split.addWidget(self._right())
        split.setSizes([340, 760, 560])
        lay.addWidget(split)
        ctx.store.changed.connect(self._on_change)
        ctx.store.project_replaced.connect(self._on_replaced)
        THEME.changed.connect(self._retheme)
        self._sync_fields()
        self.refresh()

    # ================================================================== panels
    def _left(self):
        w = QWidget()
        w.setObjectName('panel')
        ll = QVBoxLayout(w)
        ll.setContentsMargins(8, 6, 8, 8)
        ll.setSpacing(4)
        top = QHBoxLayout()
        top.addWidget(label('Lidar–camera calibration'))
        top.addStretch(1)
        top.addWidget(label('a board both sensors see', muted=True, small=True))
        ll.addLayout(top)
        ll.addWidget(hline())
        ll.addWidget(label('Where the lidar sits relative to the camera, and so in the vehicle, measured from a '
                           'checkerboard held up at several poses: the camera finds its corners, the lidar its plane, '
                           'its edge and its squares. Compared with the plan.', muted=True, small=True, wrap=True))
        ll.addSpacing(4)
        ll.addWidget(label('Pair', bold=True))
        g = QGridLayout()
        g.setVerticalSpacing(2)
        g.setHorizontalSpacing(6)
        self.cam_box, self.lid_box = QComboBox(), QComboBox()
        self.cam_box.setToolTip('The camera: by default the one whose image the lidar covers most at 5–10 m')
        self.lid_box.setToolTip('The lidar to calibrate')
        for i, (t, box) in enumerate((('Camera', self.cam_box), ('Lidar', self.lid_box))):
            a = label(t, ink2=True)
            a.setFixedWidth(118)
            g.addWidget(a, i, 0)
            g.addWidget(box, i, 1)
            box.activated.connect(lambda _=0, b=box: self._pair_chosen(b))
        ll.addLayout(g)
        self.fields = {}
        group, grid = None, None
        for grp, key, text, unit, dec, step, tip in FIELDS:
            if grp != group:
                if grid is not None:
                    ll.addLayout(grid)
                group = grp
                ll.addSpacing(6)
                head = label(grp, bold=True)
                ll.addWidget(head)
                if grp == 'Board':
                    self.board_size = label('', muted=True, small=True)
                    ll.addWidget(self.board_size)
                if grp == 'Synthetic capture':
                    self.source = SegmentedControl([('synthetic', 'Synthetic'), ('folder', 'Folder')], 'synthetic')
                    self.source.setToolTip('Synthetic: render the planned poses through the lens model and the '
                                           'lidar\'s beams. Folder: real image + point-cloud pairs')
                    self.source.changed.connect(lambda k: self._set(f'source {k}', source=k))
                    ll.insertWidget(ll.count() - 1, label('Captures', bold=True))
                    ll.insertWidget(ll.count() - 1, self.source)
                    fl = QHBoxLayout()
                    self.folder_label = label('', muted=True, small=True, wrap=True)
                    self.b_folder = button('Folder…', 'ghost', 'folder', 'A folder of pairs matched by name: '
                                                                         'pose01.png + pose01.pcd (PCD, KITTI .bin, '
                                                                         'nuScenes .pcd.bin, PLY, .npy, CSV)')
                    self.b_folder.clicked.connect(lambda: self.choose_folder())
                    fl.addWidget(self.folder_label, 1)
                    fl.addWidget(self.b_folder)
                    ll.insertLayout(ll.count() - 1, fl)
                    self.bin_box = SegmentedControl([('auto', 'Auto'), ('4', 'KITTI ×4'), ('5', 'nuScenes ×5')], 'auto')
                    self.bin_box.setToolTip('Floats per point of a .bin cloud (auto: .pcd.bin is nuScenes, else by '
                                            'the file size)')
                    self.bin_box.changed.connect(lambda k: self._set(f'.bin clouds {k}', bin_fields=k))
                    ll.insertWidget(ll.count() - 1, self.bin_box)
                    self.scene = SegmentedControl([('plain', 'Floor and wall'), ('parking', 'Parking')], 'plain')
                    self.scene.setToolTip('What the synthetic camera and lidar see round the boards')
                    self.scene.changed.connect(lambda k: self._set(f'scene {k}', scene=k))
                    ll.addWidget(self.scene)
                if grp == 'Tolerance':
                    ll.insertWidget(ll.count() - 1, label('Solve', bold=True))
                    self.t_edges = ToggleSwitch('Board edges')
                    self.t_edges.setToolTip('Hold the board\'s outline: each board point at its edge inside the '
                                            'outline the camera measured, each beam that just missed it outside')
                    self.t_edges.clicked.connect(lambda on: self._set('board edges ' + ('on' if on else 'off'),
                                                                      edges=on))
                    self.t_refl = ToggleSwitch('Checker in the reflectivity')
                    self.t_refl.setToolTip('When the lidar\'s reflectivity shows the squares (dark print), each '
                                           'clearly dark point in a dark square, each bright one in a white one')
                    self.t_refl.clicked.connect(lambda on: self._set('checker in the reflectivity ' +
                                                                     ('on' if on else 'off'), reflectivity=on))
                    ll.insertWidget(ll.count() - 1, self.t_edges)
                    ll.insertWidget(ll.count() - 1, self.t_refl)
                grid = QGridLayout()
                grid.setVerticalSpacing(2)
                grid.setHorizontalSpacing(6)
                grid.setColumnStretch(2, 1)
                if grp == 'Injected mounting error':
                    self.err_grid = grid
            f = ScrubSpinBox('', 'plain', dec, unit, step=step, width=110)
            f.setToolTip(tip)
            f.committed.connect(lambda v, k=key, t=text, u=unit: self._set(f'{t} {v:g} {u}'.strip(), **{k: v}))
            a = label(text, ink2=True)
            a.setFixedWidth(118)
            a.setToolTip(tip)
            i = grid.rowCount()
            grid.addWidget(a, i, 0)
            grid.addWidget(f, i, 1)
            self.fields[key] = f
        ll.addLayout(grid)
        ll.addStretch(1)
        self.status = label('', muted=True, small=True, wrap=True)
        ll.addWidget(self.status)
        self.bar = QProgressBar()
        self.bar.setTextVisible(False)
        self.bar.setRange(0, 1000)
        self.bar.hide()
        ll.addWidget(self.bar)
        bl = QHBoxLayout()
        self.b_defaults = button('Defaults', 'ghost', None, 'Back to the default calibration settings')
        self.b_defaults.clicked.connect(self._defaults)
        bl.addWidget(self.b_defaults)
        bl.addStretch(1)
        self.b_run = button('Capture and solve', 'primary', None, 'Capture the planned poses (or read the folder) and '
                                                                  'solve the lidar\'s pose relative to the camera')
        self.b_run.clicked.connect(lambda: self.run())
        bl.addWidget(self.b_run)
        ll.addLayout(bl)
        sc = QScrollArea()
        sc.setWidgetResizable(True)
        sc.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        sc.setWidget(w)
        sc.setMinimumWidth(320)
        return sc

    def _centre(self):
        w = QWidget()
        w.setObjectName('panel')
        cl = QVBoxLayout(w)
        cl.setContentsMargins(8, 4, 8, 6)
        cl.setSpacing(4)
        head = QHBoxLayout()
        self.view = SegmentedControl([('plan', 'Pose plan'), ('overlay', 'Overlay')], 'plan')
        self.view.setToolTip('The planned board poses from above, or the lidar\'s points in the camera image')
        self.view.changed.connect(self._set_view)
        head.addWidget(self.view)
        head.addStretch(1)
        self._pose_head = head
        self.pose_pick = SegmentedControl([('0', '1')], '0')
        self.pose_pick.changed.connect(lambda k: self._pick_pose(int(k)))
        head.addWidget(self.pose_pick)
        head.addSpacing(8)
        self.mode = SegmentedControl([('plan', 'Plan'), ('solved', 'Solved')], 'solved')
        self.mode.setToolTip('Draw the lidar\'s points through the planned or the solved extrinsic')
        self.mode.changed.connect(lambda m: self.overlay.set_mode(m))
        head.addWidget(self.mode)
        self.zoom = SegmentedControl([('board', 'Board'), ('image', 'Image')], 'board')
        self.zoom.setToolTip('Round the board or the whole image')
        self.zoom.changed.connect(lambda z: self.overlay.set_zoom(z))
        head.addWidget(self.zoom)
        cl.addLayout(head)
        self.plan_view = PlanView()
        self.plan_view.picked.connect(self._pick_pose)
        self.overlay = OverlayView()
        cl.addWidget(self.plan_view, 1)
        cl.addWidget(self.overlay, 1)
        self.plan_note = label('', muted=True, small=True, wrap=True)
        cl.addWidget(self.plan_note)
        self._set_view('plan')
        return w

    def _right(self):
        right = QSplitter(Qt.Orientation.Vertical)
        right.setHandleWidth(2)
        tw = QWidget()
        tw.setObjectName('panel')
        tl = QVBoxLayout(tw)
        tl.setContentsMargins(10, 8, 10, 8)
        tl.setSpacing(8)
        tiles = QGridLayout()
        tiles.setSpacing(8)
        self.tiles = {}
        for i, (key, caption, tip) in enumerate((
                ('verdict', 'Verdict', 'The lidar\'s measured pose against the plan and the tolerance'),
                ('pos', 'Position off the plan', 'The largest of the lidar\'s Δx, Δy, Δz in the vehicle frame'),
                ('ang', 'Angle off the plan', 'The rotation between the planned and the measured orientation'),
                ('sigma', 'Uncertainty (1σ)', 'From the board planes and corners; the edge and checker brackets '
                                               'only narrow it further, so this is on the safe side'))):
            t = self.tiles[key] = Tile(caption, tip)
            tiles.addWidget(t, i // 2, i % 2)
        tl.addLayout(tiles)
        tl.addWidget(label('The lidar in the camera\'s frame', bold=True))
        self.matrix = label('', mono_=True)
        self.matrix.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        self.matrix.setToolTip('T_camera_lidar maps lidar points into the camera\'s optical frame (OpenCV: x right, '
                               'y down, z forward); the lidar\'s pose in the vehicle follows from the camera\'s')
        tl.addWidget(self.matrix)
        tl.addWidget(label('Poses', bold=True))
        self.table = QTableWidget(0, len(POSE_COLS))
        for c, (h, tip) in enumerate(POSE_COLS):
            it = QTableWidgetItem(h)
            it.setToolTip(tip)
            self.table.setHorizontalHeaderItem(c, it)
        self.table.verticalHeader().setVisible(False)
        self.table.verticalHeader().setDefaultSectionSize(ROW_H)
        self.table.horizontalHeader().setFixedHeight(ROW_H)
        self.table.setShowGrid(False)
        self.table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.table.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        self.table.cellClicked.connect(lambda r, c: self._pick_pose(r))
        hh = self.table.horizontalHeader()
        for c in range(len(POSE_COLS)):
            hh.setSectionResizeMode(c, QHeaderView.ResizeMode.ResizeToContents)
        hh.setStretchLastSection(True)              # every header stays whole, empty or filled
        self.table.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        tl.addWidget(self.table, 1)
        bl = QHBoxLayout()
        self.b_export = button('Export…', None, 'export', 'The calibration as JSON (both directions, 1σ, Δ against '
                                                          'the plan, every pose) and an overlay PNG per pose')
        self.b_export.clicked.connect(lambda: self.export())
        self.b_save = button('Save captures…', 'ghost', None, 'The synthetic captures as image + point-cloud pairs '
                                                              '(the folder format real data uses)')
        self.b_save.clicked.connect(lambda: self.save_captures())
        self.b_variant = button('As a new variant', None, None, 'A copy of the variant with the lidar at its measured '
                                                                'pose')
        self.b_variant.clicked.connect(self.apply_as_variant)
        self.b_apply = button('Apply', 'primary', None, 'Move the lidar to its measured pose in this variant')
        self.b_apply.clicked.connect(self.apply_measured)
        bl.addWidget(self.b_export)
        bl.addWidget(self.b_save)
        bl.addStretch(1)
        bl.addWidget(self.b_variant)
        bl.addWidget(self.b_apply)
        tl.addLayout(bl)
        right.addWidget(tw)
        fw = QWidget()
        fw.setObjectName('panel')
        fl = QVBoxLayout(fw)
        fl.setContentsMargins(10, 6, 10, 8)
        fl.setSpacing(4)
        fl.addWidget(label('Findings', bold=True))
        self.find_host = QWidget()
        self.find_lay = QVBoxLayout(self.find_host)
        self.find_lay.setContentsMargins(0, 0, 0, 0)
        self.find_lay.setSpacing(4)
        self.find_lay.addStretch(1)
        sc = QScrollArea()
        sc.setWidgetResizable(True)
        sc.setWidget(self.find_host)
        fl.addWidget(sc, 1)
        self.findings: list[StatusLine] = []
        right.addWidget(fw)
        right.setSizes([640, 240])
        return right

    # ================================================================== settings
    @property
    def settings(self) -> LC.LidarCalibSettings:
        return LC.LidarCalibSettings.from_dict(self.ctx.store.project.analysis.lidar_calib)

    def _set(self, text, **kw):
        """Change settings as one undo step (saved with the project); False when nothing changes."""
        st = self.settings
        for k, v in kw.items():
            setattr(st, k, v)
        new = LC.LidarCalibSettings.from_dict(st.to_dict()).to_dict()
        if new == self.settings.to_dict():
            self._sync_fields()                # a clamped value: show what is stored
            return False

        def fn(p):
            p.analysis.lidar_calib = new
        self.ctx.store.edit('analysis', fn, f'Lidar calibration · {text}')
        return True

    def _defaults(self):
        if self.ctx.store.project.analysis.lidar_calib:
            def fn(p):
                p.analysis.lidar_calib = {}
            self.ctx.store.edit('analysis', fn, 'Lidar calibration · defaults')

    def _pair_chosen(self, box):
        tag = box.currentData() or ''
        if box is self.cam_box:
            self._set(f'camera {tag or "best"}', camera=tag)
        else:
            self._set(f'lidar {tag or "best"}', lidar=tag)

    def choose_folder(self, folder=None):
        if folder is None:
            folder = QFileDialog.getExistingDirectory(self, 'Image + point-cloud pairs')
            if not folder:
                return False
        return self._set('folder', folder=folder, source='folder')

    def _sync_fields(self):
        st = self.settings
        d = st.to_dict()
        self._syncing = True
        try:
            for key, f in self.fields.items():
                f.set_value(float(d[key]))
            self.source.set_value(st.source)
            self.scene.set_value(st.scene)
            self.bin_box.set_value(st.bin_fields)
            for t, on in ((self.t_edges, st.edges), (self.t_refl, st.reflectivity)):
                t.blockSignals(True)
                t.setChecked(on)
                t.blockSignals(False)
            W, H = st.board_size()
            self.board_size.setText(f'{st.cols + 1} × {st.rows + 1} squares · the board {W * 1000:.0f} × '
                                    f'{H * 1000:.0f} mm')
            self.folder_label.setText(st.folder or 'No folder chosen')
            synth = st.source == 'synthetic'
            for k in ('range_sigma_mm', 'image_noise', 'seed') + LC.ERR_KEYS:
                self.fields[k].setEnabled(synth)
            self.scene.setEnabled(synth)
            self.bin_box.setEnabled(not synth)
            self.b_run.setText('Capture and solve' if synth else 'Solve')
            self._fill_pairs()
        finally:
            self._syncing = False

    def _fill_pairs(self):
        v = self.ctx.store.project.variant
        st = self.settings
        share = {}
        if self.plan is not None:
            for r in self.plan.get('pairs', []):          # a camera's best overlap with any lidar
                share[r['camera']] = max(share.get(r['camera'], 0.0), r['share'])
        for box, items, cur in ((self.cam_box, [(c.tag, c.tag) for c in v.cameras], st.camera),
                                (self.lid_box, [(s.tag, f'{s.tag} · {self.ctx.store.project.sensor_model_for(s).name}')
                                                for s in v.sensors if s.kind == 'lidar'], st.lidar)):
            box.blockSignals(True)
            box.clear()
            box.addItem('Best overlap', '')
            for tag, text in items:
                pct = share.get(tag)
                box.addItem(text + (f' · {pct * 100:.0f} %' if pct is not None and box is self.cam_box else ''), tag)
            i = box.findData(cur)
            if i < 0 and cur:
                box.addItem(f'{cur} (not in this variant)', cur)
                i = box.count() - 1
            box.setCurrentIndex(max(i, 0))
            box.blockSignals(False)

    # ================================================================== state
    def _on_change(self, sections=()):
        if self.isVisible():
            self._sync_fields()
            self._plan_timer.start()
            self.refresh()
        else:
            self._dirty = True

    def _on_replaced(self, *_):
        self._gen += 1                       # a run still going belongs to the old project
        self._plan_gen += 1
        self.result, self._for, self.plan, self.busy = None, None, None, False
        self.status_text = ''
        if self.isVisible():
            self._sync_fields()
            self._plan_timer.start()
            self.refresh()
        else:
            self._dirty = True

    def _retheme(self, *_):
        self.refresh()

    def showEvent(self, e):
        super().showEvent(e)
        if self._dirty:
            self.activate()

    def activate(self):
        self._dirty = False
        self._sync_fields()
        self._start_plan()
        self.refresh()

    @property
    def stale(self) -> bool:
        p = self.ctx.store.project
        return self.result is not None and (self._for is None or self._for[0] is not p or
                                            self._for[1] != inputs_key(p))

    def _engine(self):
        win = self.ctx.window
        return win._shared_engine() if win is not None and hasattr(win, '_shared_engine') else \
            self.ctx.analysis.engine.share()

    # ---- the plan (a quick background thread: the pair ranking and the pose plan look into the occlusion cubes)
    def _start_plan(self):
        self._plan_gen += 1
        gen = self._plan_gen
        project = self.ctx.store.project.snapshot()
        st = self.settings
        eng = self._engine()

        def work():
            out = {'pairs': [], 'poses': [], 'notes': [], 'predicted': {}, 'cam': None, 'lidar': None, 'error': ''}
            try:
                out['pairs'] = LC.pairs(project, None, eng)
                cam, lid, _ = LC.pick_pair(project, None, st, eng)
                out['cam'], out['lidar'] = cam, lid
                world = LC.make_world(project, st.scene, st.far_m) if st.scene == 'parking' else None
                out['poses'], out['notes'] = LC.plan_poses(project, cam, lid, st, eng, world)
                out['predicted'] = LC.predicted_sigma(project, cam, lid, out['poses'], st)
            except ValueError as e:
                out['error'] = str(e)
            except Exception as e:              # noqa: BLE001  (shown, never raised into Qt)
                out['error'] = f'{type(e).__name__}: {e}'
            out['project'] = project
            try:
                self.sig.planned.emit(gen, out)
            except RuntimeError:                # the page went while planning
                pass
        threading.Thread(target=work, daemon=True).start()

    def _planned(self, gen, out):
        if gen != self._plan_gen:
            return
        self.plan = out
        self._fill_pairs()
        self.refresh()

    # ---- the run (a main-window job with Cancel)
    def run(self):
        if self.busy:
            return False
        p = self.ctx.store.project
        project = p.snapshot()
        st = self.settings
        if st.source == 'folder' and not st.folder:
            self.ctx.msg('Choose a folder of image + point-cloud pairs first.', 5000)
            return False
        eng = self._engine()
        poses = None
        if st.source == 'synthetic' and self.plan and self.plan.get('poses') and self.plan.get('project') and \
                inputs_key(self.plan['project']) == inputs_key(project):
            poses = self.plan['poses']                       # the plan on show is the one captured
        key = (p, inputs_key(p))
        self._gen += 1
        gen = self._gen
        self.busy = True
        self.status_text = 'Capturing…' if st.source == 'synthetic' else 'Reading the folder…'

        def work(progress, cancel):
            return key, LC.run(project, eng, st, progress=progress, cancel=cancel, poses=poses)
        win = self.ctx.window
        self.bar.setValue(0)
        if win is not None and hasattr(win, 'run_job'):
            ok = win.run_job(JOB, 'Lidar–camera calibration', lambda prog, cancel: work(
                lambda f, t='': (prog(f, t), self.sig.progress.emit(gen, f, t)), cancel),
                lambda out: self._done(gen, out), lambda m: self._failed(gen, m))
            if not ok:
                self.busy = False
                self.ctx.msg('A lidar–camera calibration is already running.', 4000)
            self.refresh()
            return ok

        def thread():
            try:
                out = work(lambda f, t='': self.sig.progress.emit(gen, f, t), lambda: gen != self._gen)
                self.sig.done.emit(gen, out)
            except RuntimeError:
                pass
            except Exception as e:              # noqa: BLE001
                try:
                    self.sig.failed.emit(gen, f'{type(e).__name__}: {e}\n{traceback.format_exc()}')
                except RuntimeError:
                    pass
        threading.Thread(target=thread, daemon=True).start()
        self.refresh()
        return True

    def _progress(self, frac, text):
        self.bar.setValue(int(frac * 1000))
        if text:
            self.status_text = text
            self.status.setText(text)

    def _done(self, gen, out):
        if gen != self._gen:
            return
        self.busy = False
        key, res = out if out is not None else (None, None)
        if res is None or key[0] is not self.ctx.store.project:
            self.status_text = 'Cancelled' if res is None else ''
            self.refresh()
            return
        self.result, self._for = res, key
        self.status_text = f'{res.message} ({res.seconds:.1f} s to solve)'
        self.overlay.set_result(self.ctx.store.project, res)
        self.view.set_value('overlay')
        self._set_view('overlay')
        self.ctx.msg(f'Lidar–camera calibration: {res.message}', 6000)
        self.refresh()

    def _failed(self, gen, msg):
        if gen != self._gen:
            return
        self.busy = False
        self.status_text = msg.splitlines()[0]
        self.refresh()

    # ================================================================== show
    def _set_view(self, v):
        overlay = v == 'overlay'
        self.plan_view.setVisible(not overlay)
        self.overlay.setVisible(overlay)
        for w in (self.mode, self.zoom):
            w.setVisible(overlay)

    def _pick_pose(self, k):
        n = len(self.result.captures) if self.result is not None else len((self.plan or {}).get('poses', []))
        if not 0 <= k < n:
            return
        self.pose_pick.set_value(str(k))
        self.overlay.set_pose(k)
        self.plan_view.set_selected(k)
        if self.table.rowCount() > k:
            self.table.selectRow(k)

    def _pose_options(self, n):
        opts = [(str(i), str(i + 1)) for i in range(max(n, 1))]
        if [(k, b.text()) for k, b in self.pose_pick.buttons.items()] == opts:
            return
        cur = self.pose_pick.value() or '0'
        old = self.pose_pick
        self.pose_pick = SegmentedControl(opts, cur if cur in dict(opts) else '0')
        self.pose_pick.setToolTip('The pose shown')
        self.pose_pick.changed.connect(lambda k: self._pick_pose(int(k)))
        self._pose_head.replaceWidget(old, self.pose_pick)
        old.hide()
        old.setParent(None)
        old.deleteLater()

    def refresh(self):
        p = self.ctx.store.project
        st = self.settings
        res = self.result
        lids = [s for s in p.variant.sensors if s.kind == 'lidar']
        plan = self.plan or {}
        # ---- status
        if not p.variant.cameras or not lids:
            text = (f'{p.variant.name} has no ' + ('camera' if not p.variant.cameras else 'lidar') +
                    (': add one with Vehicle ▸ Add lidar.' if p.variant.cameras else '.'))
        else:
            text = self.status_text or plan.get('error', '') or 'Plan the poses, then capture and solve.'
        if self.stale:
            text += ' · stale: the design or the settings changed since'
        self.status.setText(text)
        win = self.ctx.window
        running = self.busy or (win is not None and JOB in getattr(win, '_jobs', {}))
        self.bar.setVisible(self.busy)
        self.b_run.setEnabled(bool(p.variant.cameras and lids) and not running)
        # ---- the plan
        cam, lid = plan.get('cam'), plan.get('lidar')
        W, H = st.board_size()
        self.plan_view.set_data(plan.get('project'), cam, lid, plan.get('poses', []), (W, H), (st.near_m, st.far_m),
                                plan.get('error', '') or 'Planning…')
        pr = plan.get('predicted') or {}
        notes = list(plan.get('notes', []))
        if pr:
            notes.insert(0, f'{len(plan.get("poses", []))} poses for {cam.tag} and {lid.tag}. Their planes alone would '
                            f'pin {lid.tag} to ±{pr["x_mm"]:.1f} / ±{pr["y_mm"]:.1f} / ±{pr["z_mm"]:.1f} mm (x / y / '
                            f'z) and ±{max(pr["roll_deg"], pr["pitch_deg"], pr["yaw_deg"]):.2f}° (predicted from '
                            f'the points expected on each board at {st.range_sigma_mm:g} mm range noise; the edges '
                            'and the checker narrow it further).')
        self.plan_note.setText(' '.join(notes))
        n = len(res.captures) if res is not None else len(plan.get('poses', []))
        self._pose_options(n)
        # ---- the result
        if self.overlay.result is not res:
            self.overlay.set_result(p, res)
        can_apply = res is not None and res.T is not None and res.verdict not in ('failed', 'unobservable') and \
            res.variant_uid == p.active_uid and p.variant.sensor(res.lidar_uid) is not None
        self.b_apply.setEnabled(can_apply)
        self.b_apply.setText(f'Apply to {res.lidar_tag}' if res is not None else 'Apply')
        self.b_variant.setEnabled(can_apply)
        self.b_export.setEnabled(res is not None)
        self.b_save.setEnabled(res is not None and res.source == 'synthetic')
        self._fill_result(res)

    def _fill_result(self, res):
        t = self.tiles
        if res is None:
            for k in t:
                t[k].set('—', '', '')
            self.matrix.setText('')
            self.table.setRowCount(0)
            self._fill_findings([])
            return
        sev = LC.VERDICT_SEVERITY.get(res.verdict, 'critical')
        t['verdict'].set(VERDICT_WORD.get(res.verdict, res.verdict), '', f'{len(res.used)} of {len(res.poses)} poses '
                         f'· {res.lidar_tag} in {res.camera_tag}', sev)
        if res.T is not None:
            d, sg = res.delta, res.sigma
            dv = d['vehicle']
            t['pos'].set(_f(d['position_mm'], 1), 'mm', f'x {_f(dv["x_mm"], 1, True)} · y {_f(dv["y_mm"], 1, True)} · '
                         f'z {_f(dv["z_mm"], 1, True)}', 'good' if d['position_mm'] <= res.settings.pos_tol_mm
                         else 'warning' if d['position_mm'] <= 3 * res.settings.pos_tol_mm else 'serious')
            t['ang'].set(_f(d['rotation_deg'], 2), '°', f'yaw {_f(dv["yaw_deg"], 2, True)} · pitch '
                         f'{_f(dv["pitch_deg"], 2, True)} · roll {_f(dv["roll_deg"], 2, True)}',
                         'good' if d['rotation_deg'] <= res.settings.ang_tol_deg else
                         'warning' if d['rotation_deg'] <= 3 * res.settings.ang_tol_deg else 'serious')
            t['sigma'].set(f'±{_f(sg["position_mm"], 1)}', 'mm', f'±{_f(sg["angle_deg"], 3)}° · residual variance '
                           f'{res.fit.get("residual_variance", float("nan")):.2f}')
            T = res.T
            q = LC.quat_from_matrix(T[:3, :3])
            m = res.measured
            rows = '\n'.join('  '.join(f'{v:+.5f}'.replace('-', '−') for v in T[i]) for i in range(3))
            self.matrix.setText(f'T_camera_lidar\n{rows}\nq (x y z w)  ' + '  '.join(f'{v:+.5f}' for v in q)
                                .replace('-', '−') + f'\n{res.lidar_tag} in the vehicle  x {m["x"]:.4f}  '
                                f'y {m["y"]:.4f}  z {m["z"]:.4f} m\n  yaw {m["yaw"]:.3f}°  pitch {m["pitch"]:.3f}°  '
                                f'roll {m["roll"]:.3f}°'.replace('-', '−'))
        else:
            for k in ('pos', 'ang', 'sigma'):
                t[k].set('—', '', '')
            self.matrix.setText('')
        tb = self.table
        tb.setRowCount(len(res.poses))
        muted = THEME.color('muted')
        for r, o in enumerate(res.poses):
            st = o.stats or {}
            tb.setItem(r, 0, QTableWidgetItem(o.name))
            it = QTableWidgetItem(STATUS_WORD.get(o.status, o.status))
            it.setForeground(QColor(STATUS['good']) if o.status == 'used' else QColor(STATUS['warning']))
            it.setToolTip(o.message or 'Found by both and used in the solve')
            tb.setItem(r, 1, it)
            col = None if o.status == 'used' else muted
            tb.setItem(r, 2, _num(_f(st.get('corner_rms_px', o.cam.rms_px if o.cam else None), 3), color=col))
            tb.setItem(r, 3, _num(str(len(o.lid.points)) if o.lid is not None else '—', color=col))
            tb.setItem(r, 4, _num(_f(o.lid.rms_m * 1000 if o.lid is not None else None, 1), color=col))
            tb.setItem(r, 5, _num(_f(st.get('offset_mm'), 1, True), color=col))
            tb.setItem(r, 6, _num(_f(st.get('angle_deg'), 2), color=col))
            tb.setItem(r, 7, _num(_f(st.get('edge_mm'), 1), color=col))
        self._fill_findings(res.findings)

    def _fill_findings(self, findings):
        for w in self.findings:
            self.find_lay.removeWidget(w)
            w.hide()
            w.deleteLater()
        self.findings = []
        for f in findings:
            w = StatusLine(f['severity'], f['message'])
            self.find_lay.insertWidget(self.find_lay.count() - 1, w)
            self.findings.append(w)

    # ================================================================== apply and export
    def apply_measured(self):
        res = self.result
        p = self.ctx.store.project
        if res is None or res.T is None or res.variant_uid != p.active_uid:
            return False
        lid = p.variant.sensor(res.lidar_uid)
        if lid is None:
            return False

        def fn(q):
            LC.apply_to_variant(q.variant, res)
        self.ctx.store.edit('variant', fn, f'{res.lidar_tag} to its measured pose (lidar–camera calibration with '
                                           f'{res.camera_tag})')
        self.ctx.msg(f'Moved {res.lidar_tag} to its measured pose', 5000)
        return True

    def apply_as_variant(self):
        res = self.result
        p = self.ctx.store.project
        if res is None or res.T is None:
            return None
        name = f'{p.variant.name} · lidar calibrated'
        v = self.ctx.store.edit('variants', lambda q: LC.calibrated_variant(q, res), f'Created variant {name}')
        win = self.ctx.window
        if win is not None and hasattr(win, '_fill_variant_combo'):
            win._fill_variant_combo()
        if v is not None:
            self.ctx.msg(f'Created variant {v.name} with {res.lidar_tag} at its measured pose', 6000)
        return v

    def export(self, folder=None):
        res = self.result
        if res is None:
            return []
        if folder is None:
            folder = QFileDialog.getExistingDirectory(self, 'Export the lidar–camera calibration into')
            if not folder:
                return []
        paths = LC.export(res, self.ctx.store.project, folder)
        self.ctx.msg(f'Exported the lidar–camera calibration to {folder}', 5000)
        return paths

    def save_captures(self, folder=None):
        res = self.result
        if res is None:
            return []
        if folder is None:
            folder = QFileDialog.getExistingDirectory(self, 'Save the captures as image + point-cloud pairs into')
            if not folder:
                return []
        paths = LC.save_captures(res.captures, folder)
        self.ctx.msg(f'Saved {len(paths) // 2} image + point-cloud pairs to {folder}', 5000)
        return paths
