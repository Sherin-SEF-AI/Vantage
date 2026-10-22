"""Main window: menus, toolbar, docks, status bar, workspaces and background jobs."""
from __future__ import annotations

import json
import math
import os
import threading
import time
import traceback

import numpy as np
from PyQt6.QtCore import Qt, QSize, QTimer, pyqtSignal, QObject, QByteArray, QBuffer, QIODevice, QPoint
from PyQt6.QtGui import QAction, QActionGroup, QKeySequence, QIcon, QImage, QGuiApplication, QPixmap, QShortcut
from PyQt6.QtWidgets import (QMainWindow, QDockWidget, QToolBar, QWidget, QHBoxLayout, QVBoxLayout, QLabel, QComboBox,
                             QLineEdit, QToolButton, QMenu, QStatusBar, QStackedWidget, QFileDialog, QMessageBox,
                             QProgressBar, QSizePolicy, QApplication, QWidgetAction, QCheckBox, QPushButton)

from ..core.project import Project, Variant, FRAME_PRESETS, MAX_CAMERAS, new_uid, Obstacle, Sensor
from ..core import samples
from . import icons, ops
from .context import AppContext, OVERLAYS, HEATMAPS, TOOLS
from .stitchedview import StitchedController
from .store import ProjectStore
from .theme import THEME, mono, sans, STATUS
from .widgets import DockTitle, SegmentedControl, tool_button, label, button, hline
from .viewarea import ViewArea
from .outliner import Outliner, LibraryPanel
from .inspector import Inspector
from .bottom.panels import CameraDock, BottomPanel


class JobSignals(QObject):
    progress = pyqtSignal(str, float, str)
    done = pyqtSignal(str, object)
    failed = pyqtSignal(str, str)


class MainWindow(QMainWindow):
    def __init__(self, project: Project, soft_3d=False):
        super().__init__()
        self.soft_3d = soft_3d
        self.store = ProjectStore(project)
        self.ctx = AppContext(self.store)
        self.ctx.window = self
        self.ctx.stitched = StitchedController(self.ctx)      # objects in the stitched view, kept up to date
        from .assistant import AssistantController
        self.ctx.assistant = AssistantController(self.ctx)    # the chat about the design (Preferences ▸ Assistant)
        self.actions_by_id: dict[str, QAction] = {}
        self.palette_extra = []
        self._photo_cache = {}
        self._jobs = {}
        self.jobs = JobSignals()
        self.jobs.progress.connect(self._job_progress)
        self.jobs.done.connect(self._job_done)
        self.jobs.failed.connect(self._job_failed)
        self._maximised_state = None
        self.setWindowTitle('Vantage')
        self.setAcceptDrops(True)
        self.resize(1920, 1080)
        self.setDockNestingEnabled(True)
        self.setCorner(Qt.Corner.BottomLeftCorner, Qt.DockWidgetArea.BottomDockWidgetArea)
        self.setCorner(Qt.Corner.BottomRightCorner, Qt.DockWidgetArea.BottomDockWidgetArea)

        self._build_central()
        self._build_docks()
        self._build_actions()
        self._build_menus()
        self._build_toolbar()
        self._build_statusbar()
        self._wire()
        self.set_project_meta()
        QTimer.singleShot(0, self._first_run)

    # ------------------------------------------------------------------ layout
    def _build_central(self):
        self.pages = QStackedWidget()
        # pages can be squeezed; without this the stack takes the largest page minimum (hidden pages too)
        self.pages.setMinimumSize(360, 260)
        self.viewarea = ViewArea(self.ctx)
        self.pages.addWidget(self.viewarea)
        self._page_widgets = {'viewports': self.viewarea}
        self.setCentralWidget(self.pages)

    def page(self, name):
        if name in self._page_widgets:
            return self._page_widgets[name]
        w = None
        if name == 'camviews':
            from .workspaces.camviews import CamViewsPage
            w = CamViewsPage(self.ctx)
        elif name == 'tolerance':
            from .workspaces.tolerance_ws import TolerancePage
            w = TolerancePage(self.ctx)
        elif name == 'optimiser':
            from .workspaces.optimiser_ws import OptimiserPage
            w = OptimiserPage(self.ctx)
        elif name == 'compare':
            from .workspaces.compare_ws import ComparePage
            w = ComparePage(self.ctx)
        elif name == 'photo':
            from .workspaces.photo_ws import PhotoPage
            w = PhotoPage(self.ctx)
        elif name == 'targets':
            from .workspaces.targets_ws import TargetsPage
            w = TargetsPage(self.ctx)
        elif name == 'install':
            from .workspaces.install_ws import InstallPage
            w = InstallPage(self.ctx)
        elif name == 'volume':
            from .workspaces.volume_ws import VolumePage
            w = VolumePage(self.ctx)
        elif name == 'perception':
            from .workspaces.perception_ws import PerceptionPage
            w = PerceptionPage(self.ctx)
        elif name == 'rig':
            from .workspaces.rig_ws import RigPage
            w = RigPage(self.ctx)
        elif name == 'coverage':
            from .workspaces.coverage_ws import SensorCoveragePage
            w = SensorCoveragePage(self.ctx)
        elif name == 'timing':
            from .workspaces.timing_ws import TimingPage
            w = TimingPage(self.ctx)
        elif name == 'recording':
            from .workspaces.recording_ws import RecordingPage
            w = RecordingPage(self.ctx)
        self.pages.addWidget(w)
        self._page_widgets[name] = w
        return w

    def page_name(self):
        cur = self.pages.currentWidget()
        for k, w in self._page_widgets.items():
            if w is cur:
                return k
        return 'viewports'

    PAGE_DOCKS = {
        'viewports': ('outliner', 'library', 'inspector', 'camview', 'bottom'),
        'camviews': ('bottom',),
        'tolerance': ('bottom',),
        'optimiser': ('bottom',),
        'compare': (),
        'photo': ('outliner', 'library', 'inspector', 'bottom'),
        'targets': ('outliner', 'library', 'inspector', 'bottom'),
        'install': (),
        'volume': (),
        'perception': (),
        'rig': (),
        'coverage': (),
        'timing': (),
        'recording': (),
    }

    def set_page(self, name):
        prev = self.page_name()
        if name == prev:
            return
        w = self.page(name)
        w.setMinimumSize(0, 0)
        w.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Ignored)
        self.pages.setCurrentWidget(w)
        want = self.PAGE_DOCKS.get(name, ())
        for k, d in self.docks.items():
            if k != 'assistant':                            # the conversation stays open across pages
                d.setVisible(k in want)
        QTimer.singleShot(30, self._restore_dock_sizes)
        if name == 'targets':
            self.store.select_obj(('targets',))
            self.ctx.set_overlay('targets', True)
        if name == 'photo':
            self.store.select_obj(('photos',))
        if hasattr(w, 'activate'):
            w.activate()
        self.sync_view_buttons()

    def _dock(self, key, title, widget, area, right=''):
        d = QDockWidget(title, self)
        d.setObjectName(f'dock_{key}')
        body = QWidget()
        body.setObjectName('dockBody')
        l = QVBoxLayout(body)
        l.setContentsMargins(0, 0, 0, 0)
        l.setSpacing(0)
        l.addWidget(widget)
        d.setWidget(body)
        tb = DockTitle(d, title, right)
        d.setTitleBarWidget(tb)
        d.title_widget = tb
        d.setFeatures(QDockWidget.DockWidgetFeature.DockWidgetClosable | QDockWidget.DockWidgetFeature.DockWidgetMovable |
                      QDockWidget.DockWidgetFeature.DockWidgetFloatable)
        self.addDockWidget(area, d)
        self.docks[key] = d
        return d

    def _build_docks(self):
        self.docks = {}
        self.outliner = Outliner(self.ctx)
        self.library = LibraryPanel(self.ctx)
        self.inspector = Inspector(self.ctx)
        self.camdock = CameraDock(self.ctx)
        self.bottom = BottomPanel(self.ctx)
        L, R, B = Qt.DockWidgetArea.LeftDockWidgetArea, Qt.DockWidgetArea.RightDockWidgetArea, Qt.DockWidgetArea.BottomDockWidgetArea
        self._dock('outliner', 'Outliner', self.outliner, L)
        self._dock('library', 'Library', self.library, L)
        self._dock('inspector', 'Inspector', self.inspector, R, 'Camera')
        self._dock('camview', 'Camera view', self.camdock, R)
        bd = self._dock('bottom', 'Analysis', self.bottom, B)
        bd.setTitleBarWidget(QWidget())
        from .assistant import AssistantPanel
        self.assistant_panel = AssistantPanel(self.ctx)
        ad = self._dock('assistant', 'Assistant', self.assistant_panel, R)
        self.addDockWidget(R, ad, Qt.Orientation.Horizontal)  # a full-height column of its own, not under the inspector
        ad.setTitleBarWidget(QWidget())                      # the panel has its own header
        ad.setMinimumWidth(340)
        ad.hide()
        self.splitDockWidget(self.docks['outliner'], self.docks['library'], Qt.Orientation.Vertical)
        self.splitDockWidget(self.docks['inspector'], self.docks['camview'], Qt.Orientation.Vertical)
        self.resizeDocks([self.docks['outliner'], self.docks['inspector']], [250, 312], Qt.Orientation.Horizontal)
        self.resizeDocks([self.docks['outliner'], self.docks['library']], [560, 240], Qt.Orientation.Vertical)
        self.resizeDocks([self.docks['inspector'], self.docks['camview']], [600, 200], Qt.Orientation.Vertical)
        self.resizeDocks([bd], [210], Qt.Orientation.Vertical)
        self.inspector.titleChanged.connect(lambda t: self.docks['inspector'].title_widget.right.setText(t))
        self.camdock.titleChanged.connect(lambda t: self.docks['camview'].title_widget.right.setText(t))

    def _restore_dock_sizes(self):
        vis = [k for k in ('outliner', 'inspector') if self.docks[k].isVisible()]
        if vis:
            self.resizeDocks([self.docks[k] for k in vis], [250 if k == 'outliner' else 312 for k in vis], Qt.Orientation.Horizontal)

    def set_dock_count(self, key, text):
        d = self.docks.get(key)
        if d is not None:
            d.title_widget.right.setText(text)

    # ------------------------------------------------------------------ actions
    def act(self, aid, text, slot=None, shortcut=None, icon=None, checkable=False, category='', tip='', checked=False):
        a = QAction(text, self)
        if shortcut:
            a.setShortcut(QKeySequence(shortcut))
            a.setShortcutContext(Qt.ShortcutContext.WindowShortcut)
        if icon:
            a.setIcon(icons.icon(icon, 16))
        a.setCheckable(checkable)
        if checkable:
            a.setChecked(checked)
        if slot is not None:
            if checkable:
                a.toggled.connect(slot)
            else:
                a.triggered.connect(lambda _=False, s=slot: s())
        a.setData({'id': aid, 'category': category, 'icon': icon, 'tip': tip})
        sc = f'  {QKeySequence(shortcut).toString(QKeySequence.SequenceFormat.NativeText)}' if shortcut else ''
        a.setToolTip(f'{text.replace("…", "").replace("&", "")}{sc}' + (f'\n{tip}' if tip else ''))
        self.addAction(a)
        self.actions_by_id[aid] = a
        return a

    def _build_actions(self):
        A = self.act
        c = self.ctx
        st = self.store
        # file
        A('new', 'New project', self.new_project, 'Ctrl+N', 'plus', category='File')
        A('open', 'Open…', self.open_project, 'Ctrl+O', 'folder', category='File')
        A('save', 'Save', self.save, 'Ctrl+S', 'file', category='File')
        A('saveas', 'Save as…', self.save_as, 'Ctrl+Shift+S', category='File')
        A('import_vehicle', 'Vehicle model…', self.import_vehicle, 'Ctrl+I', 'import', category='File ▸ Import')
        A('import_photos', 'Reference photo set…', self.load_photos, None, 'photo-spin', category='File ▸ Import')
        A('import_module', 'Camera module…', self.import_module, None, 'camera-wide', category='File ▸ Import')
        A('import_layout', 'Layout from CSV/JSON…', self.import_layout, None, 'report', category='File ▸ Import')
        A('project_settings', 'Project settings…', self.project_settings, category='File')
        A('prefs', 'Preferences…', self.preferences, 'Ctrl+,', 'gear', category='File')
        A('assistant', 'Assistant', self.toggle_assistant, 'Ctrl+J', 'sparkle', category='Window',
          tip='Ask questions about this design; changes are proposed, you apply them')
        A('start', 'Start window…', self.show_start, category='File')
        A('quit', 'Quit', self.close, 'Ctrl+Q', category='File')
        # edit
        A('undo', 'Undo', st.undo.undo, 'Ctrl+Z', 'undo', category='Edit')
        A('redo', 'Redo', st.undo.redo, 'Ctrl+Shift+Z', 'redo', category='Edit')
        A('duplicate', 'Duplicate', lambda: ops.duplicate_cameras(c, list(st.sel_cams)), 'Ctrl+D', 'copy', category='Edit')
        A('delete', 'Delete', self.delete_selection, 'Del', 'trash', category='Edit')
        A('mirror', 'Mirror to other side', lambda: ops.mirror_to_other_side(c, list(st.sel_cams)), 'Ctrl+M', 'mirror', category='Edit')
        A('link', 'Link / unlink L–R pair', lambda: ops.toggle_link(c, st.selected.uid) if st.selected else None, None, 'link', category='Edit')
        A('sel_all', 'All cameras', lambda: st.select_cams([x.uid for x in st.variant.cameras]), 'Ctrl+A', category='Edit ▸ Select')
        A('sel_left', 'Left side', lambda: st.select_cams([x.uid for x in st.variant.cameras if x.y > 0.05]), category='Edit ▸ Select')
        A('sel_right', 'Right side', lambda: st.select_cams([x.uid for x in st.variant.cameras if x.y < -0.05]), category='Edit ▸ Select')
        A('sel_invert', 'Invert', lambda: st.select_cams([x.uid for x in st.variant.cameras if x.uid not in st.sel_cams]), category='Edit ▸ Select')
        A('find', 'Find object…', lambda: self.command_palette('obj '), 'Ctrl+F', 'search', category='Edit')
        A('palette', 'Command palette', self.command_palette, 'Ctrl+K', 'search', category='Edit')
        # view
        A('split1', 'Single', lambda: self.set_split('single'), None, 'split-1', True, 'View ▸ Viewport layout', checked=True)
        A('split2', 'Side by side', lambda: self.set_split('side'), None, 'split-2', True, 'View ▸ Viewport layout')
        A('split4', '2×2', lambda: self.set_split('quad'), None, 'split-4', True, 'View ▸ Viewport layout')
        g = QActionGroup(self)
        for k in ('split1', 'split2', 'split4'):
            self.actions_by_id[k].toggled.disconnect()
            g.addAction(self.actions_by_id[k])
        self.actions_by_id['split1'].triggered.connect(lambda: self.set_split('single'))
        self.actions_by_id['split2'].triggered.connect(lambda: self.set_split('side'))
        self.actions_by_id['split4'].triggered.connect(lambda: self.set_split('quad'))
        A('view_persp', 'Perspective', lambda: self.set_view('persp'), None, 'cube', True, 'View ▸ Camera', tip='Orbit view of the vehicle')
        A('view_top', 'Top', lambda: self.set_view('top'), None, 'car-top', True, 'View ▸ Camera', tip='Plan view with coverage maps')
        A('view_front', 'Front', lambda: self.set_view('front'), None, 'car-front', True, 'View ▸ Camera')
        A('view_side', 'Side', lambda: self.set_view('side'), None, 'car-side', True, 'View ▸ Camera')
        A('view_rear', 'Rear', lambda: self.set_view('rear'), None, None, True, 'View ▸ Camera')
        A('view_through', 'Through selected camera', lambda: self.set_view('through'), None, 'through', True, 'View ▸ Camera',
          tip='Simulated image of the selected camera')
        g2 = QActionGroup(self)
        for k in ('view_persp', 'view_top', 'view_front', 'view_side', 'view_rear', 'view_through'):
            a = self.actions_by_id[k]
            a.toggled.disconnect()
            g2.addAction(a)
        for k, v in (('view_persp', 'persp'), ('view_top', 'top'), ('view_front', 'front'), ('view_side', 'side'),
                     ('view_rear', 'rear'), ('view_through', 'through')):
            self.actions_by_id[k].triggered.connect(lambda _=False, v=v: self.set_view(v))
        A('look_through', 'Look through', lambda: self.set_view('through'), 'Ctrl+0', 'through', category='Camera')
        A('camviews', 'Camera views, bird’s-eye, bowl', lambda: self.set_page('camviews'), 'Ctrl+Shift+B', 'bev', category='View')
        A('render_image', 'Render image…', self.render_image, 'Ctrl+Shift+R', 'photo-spin', category='View',
          tip='Save the active 3D view at up to 4× its size')
        A('turntable', 'Turntable video…', self.turntable, None, 'rotate', category='View',
          tip='Orbit the active 3D view once around the vehicle and save it as a video')
        A('back', 'Back to viewports', lambda: self.set_page('viewports'), category='View')
        for i, (key, text, ic, _) in enumerate(OVERLAYS):
            sc = {'frustums': 'Alt+1', 'footprints': 'Alt+2', 'camlight': 'Alt+3', 'shadows': 'Alt+0', 'occlusion': 'Alt+4',
                  'seams': 'Alt+5', 'roi': 'Alt+6', 'targets': 'Alt+7', 'dimensions': 'Alt+8', 'grid': 'Alt+9',
                  'guidelines': 'Alt+G'}.get(key)
            A(f'ov_{key}', text, lambda on, k=key: c.set_overlay(k, on), sc, ic, True, 'View ▸ Overlays', checked=c.overlays[key])
        for key, text in HEATMAPS:
            A(f'heat_{key}', text, None, None, None, True, 'View ▸ Overlays ▸ Heatmap', checked=key == 'none')
        g3 = QActionGroup(self)
        for key, _ in HEATMAPS:
            a = self.actions_by_id[f'heat_{key}']
            g3.addAction(a)
            a.triggered.connect(lambda _=False, k=key: self.set_heatmap(k))
        for key, text in (('solid', 'Solid'), ('wire', 'Wire'), ('xray', 'X-ray'), ('photo', 'Photo-matched')):
            A(f'shade_{key}', text, None, None, None, True, 'View ▸ Shading', checked=key == 'solid')
        g4 = QActionGroup(self)
        for key in ('solid', 'wire', 'xray', 'photo'):
            a = self.actions_by_id[f'shade_{key}']
            g4.addAction(a)
            a.triggered.connect(lambda _=False, k=key: self.set_shading(k))
        A('photo_prev', 'Previous matched photo', lambda: self.photo_step(-1), '[', None, category='View ▸ Shading')
        A('photo_next', 'Next matched photo', lambda: self.photo_step(1), ']', None, category='View ▸ Shading')
        A('frame_sel', 'Frame selection', self.frame_selection, 'F', 'frame', category='View')
        A('frame_all', 'Frame all', self.frame_all, 'Home', 'maximise', category='View')
        A('units_m', 'm', lambda: self.set_units('m'), None, None, True, 'View ▸ Units', checked=True)
        A('units_mm', 'mm', lambda: self.set_units('mm'), None, None, True, 'View ▸ Units')
        g5 = QActionGroup(self)
        for k in ('units_m', 'units_mm'):
            self.actions_by_id[k].toggled.disconnect()
            g5.addAction(self.actions_by_id[k])
        self.actions_by_id['units_m'].triggered.connect(lambda: self.set_units('m'))
        self.actions_by_id['units_mm'].triggered.connect(lambda: self.set_units('mm'))
        g6 = QActionGroup(self)
        for k, v in FRAME_PRESETS.items():
            a = A(f'frame_{k}', v.replace('…', '') + ('…' if k == 'custom' else ''), None, None, None, k != 'custom', 'View ▸ Coordinate frame')
            if k != 'custom':
                g6.addAction(a)
                a.triggered.connect(lambda _=False, k=k: self.set_frame(k))
            else:
                a.triggered.connect(self.custom_frame)
        for k, t in (('dark', 'Dark'), ('light', 'Light'), ('system', 'System')):
            A(f'theme_{k}', t, None, None, None, True, 'View ▸ Theme', checked=k == THEME.name)
        g7 = QActionGroup(self)
        for k in ('dark', 'light', 'system'):
            a = self.actions_by_id[f'theme_{k}']
            g7.addAction(a)
            a.triggered.connect(lambda _=False, k=k: self.set_theme(k))
        A('maximise', 'Maximise viewport', self.toggle_maximise, 'Ctrl+Space', 'maximise', category='View')
        # tools
        g8 = QActionGroup(self)
        for key, text, ic, sc in TOOLS:
            a = A(f'tool_{key}', text, None, sc, ic, True, 'Tools', checked=key == 'select',
                  tip={'select': 'Select cameras, targets and ground points.',
                       'move': 'Drag the gizmo axes, or drag a camera along the body.',
                       'rotate': 'Drag the rings to change yaw and pitch.',
                       'place': 'Click the body to place. Alt-click places a mirrored L/R pair.',
                       'measure': 'Click two points on the body or ground.'}[key])
            g8.addAction(a)
            a.triggered.connect(lambda _=False, k=key: c.set_tool(k))
        # vehicle
        A('veh_replace', 'Replace model…', self.import_vehicle, None, 'import', category='Vehicle')
        A('veh_align', 'Align, scale & origin…', lambda: self.import_vehicle(step=1, reuse=True), None, 'axes', category='Vehicle')
        A('veh_wheels', 'Pick wheel centres', lambda: self.import_vehicle(step=2, reuse=True), None, 'axle', category='Vehicle')
        A('veh_ground', 'Ground plane…', lambda: self.import_vehicle(step=2, reuse=True), None, 'ground', category='Vehicle')
        A('zone_allow', 'Paint allowed', lambda: self.import_vehicle(step=3, reuse=True, paint='allowed'), None, 'mount-zone', category='Vehicle ▸ Mount zones')
        A('zone_forbid', 'Paint forbidden', lambda: self.import_vehicle(step=3, reuse=True, paint='forbidden'), None, 'blind', category='Vehicle ▸ Mount zones')
        A('zone_glass', 'Auto-detect glass', self.auto_glass, None, None, category='Vehicle ▸ Mount zones')
        A('zone_clear', 'Clear', self.clear_zones, None, None, category='Vehicle ▸ Mount zones')
        A('occ_full', 'Full mesh', lambda: self.set_occluder(10 ** 7), None, None, category='Vehicle ▸ Occluder')
        A('occ_hull', 'Simplified hull', lambda: self.set_occluder(8000), None, None, category='Vehicle ▸ Occluder')
        A('occ_level', 'Simplification level…', lambda: self.store.select_obj(('vehicle',)), None, None, category='Vehicle ▸ Occluder')
        A('photo_load', 'Load spin set…', self.load_photos, None, 'photo-spin', category='Vehicle ▸ Reference photos')
        A('photo_cal', 'Calibrate…', self.calibrate_photos, None, None, category='Vehicle ▸ Reference photos')
        A('photo_pane', 'Show photo pane', lambda: self.show_photo_page(True), None, None, category='Vehicle ▸ Reference photos')
        A('spec', 'Spec check…', self.spec_check, None, 'report', category='Vehicle')
        from ..core.rig import PRESETS as RIG_PRESETS
        from .rigview import add_preset
        for k, t in RIG_PRESETS:
            A(f'rig_{k}', t, lambda _=False, k=k: add_preset(self.ctx, k), None, 'cylinder' if k == 'sensor_mast' else 'cube',
              category='Vehicle ▸ Add rig part')
        from .sensorview import add_sensor
        A('add_lidar', 'Add lidar', lambda: add_sensor(self.ctx, 'lidar'), None, 'sensor', category='Vehicle',
          tip='A lidar on the highest rig part or the roof (Ouster OS1-128 by default; pick another in the Inspector)')
        A('add_radar', 'Add radar', lambda: add_sensor(self.ctx, 'radar'), None, 'sensor', category='Vehicle',
          tip='A radar in the middle of the front bumper, facing forward')
        # camera
        A('add_cam', 'Add from library…', self.add_from_library, 'Shift+A', 'plus', category='Camera')
        A('place', 'Place on surface', lambda: self.set_tool('place'), None, 'place', category='Camera')
        A('aim_ground', 'At ground point…', self.start_aim, None, 'target', category='Camera ▸ Aim')
        A('aim_normal', 'Along surface normal', lambda: [ops.aim_along_normal(c, u) for u in list(st.sel_cams)], None, None, category='Camera ▸ Aim')
        A('aim_level', 'Level roll', lambda: ops.level_roll(c, list(st.sel_cams)), None, None, category='Camera ▸ Aim')
        A('lock', 'Lock transform', lambda: ops.set_lock(c, list(st.sel_cams)), 'Ctrl+L', 'lock', category='Camera')
        A('hide', 'Hide', lambda: ops.set_visible(c, list(st.sel_cams), False), 'H', 'eye-off', category='Camera')
        A('unhide', 'Show all', lambda: ops.set_visible(c, [x.uid for x in st.variant.cameras], True), 'Alt+H', 'eye', category='Camera')
        A('isolate', 'Isolate', self.toggle_isolate, '/', 'isolate', category='Camera')
        A('assign_port', 'Assign GMSL port…', self.assign_port_dialog, None, 'cable', category='Camera')
        A('library', 'Camera library…', lambda: self.docks['library'].show(), None, 'lens', category='Camera')
        A('edit_module', 'Edit module…', lambda: self.edit_module(*(st.selected.module_id, st.selected.lens_id) if st.selected else (None, None)), None, 'sensor', category='Camera')
        A('copy_pose', 'Copy pose as JSON', self.copy_pose, 'Ctrl+Shift+C', 'copy', category='Camera')
        A('snap_surface', 'Snap to surface', lambda: ops.snap_to_surface(c, list(st.sel_cams)), None, 'snap-surface', category='Camera')
        # layout
        A('var_new', 'New variant', self.new_variant, None, 'plus', category='Layout')
        A('var_dup', 'Duplicate variant', self.duplicate_variant, None, 'copy', category='Layout')
        A('var_rename', 'Rename', lambda: self.store.select_obj(('variant',)), None, None, category='Layout')
        A('var_delete', 'Delete', self.delete_variant, None, 'trash', category='Layout')
        for n in (4, 6, 8):
            A(f'preset_{n}', f'{n} cameras', lambda n=n: self.apply_preset(n), None, 'camera-wide', category='Layout ▸ Presets')
        A('preset_6f', '6 front-heavy', lambda: self.apply_preset('6f'), None, 'camera-wide', category='Layout ▸ Presets')
        A('compare', 'Compare variants…', lambda: self.set_page('compare'), None, 'compare', category='Layout')
        A('optimise', 'Optimise placement…', lambda: self.set_page('optimiser'), None, 'optimise', category='Layout')
        # analysis
        A('run', 'Run now', self.ctx.analysis.run_now, 'F5', 'refresh', category='Analysis')
        A('live', 'Live analysis', self.set_live, None, None, True, 'Analysis', checked=self.store.project.analysis.live)
        A('roi', 'Region of interest…', lambda: self.focus_roi(), None, 'roi', category='Analysis')
        for cm in (5, 2, 1):
            A(f'res_{cm}', f'{cm} cm', None, None, None, True, 'Analysis ▸ Ground resolution', checked=cm == 5)
        g9 = QActionGroup(self)
        for cm in (5, 2, 1):
            a = self.actions_by_id[f'res_{cm}']
            g9.addAction(a)
            a.triggered.connect(lambda _=False, cm=cm: self.set_cell(cm / 100))
        for k, t in (('ground', 'Ground only'), ('pole05', 'Vertical poles 0.5 m'), ('pole10', 'Vertical poles 1.0 m'), ('child', 'Child cylinder')):
            A(f'obs_{k}', t, None, None, None, True, 'Analysis ▸ Obstacles', checked=k == 'ground')
        g10 = QActionGroup(self)
        for k in ('ground', 'pole05', 'pole10', 'child'):
            a = self.actions_by_id[f'obs_{k}']
            g10.addAction(a)
            a.triggered.connect(lambda _=False, k=k: self.set_obstacle_mode(k))
        A('obs_add', 'Add pole', self.add_pole, None, 'pole', category='Analysis ▸ Obstacles')
        A('thresholds', 'Density thresholds…', lambda: self.focus_roi(), None, 'density', category='Analysis')
        A('tolerance', 'Tolerance study…', lambda: self.set_page('tolerance'), None, 'tolerance', category='Analysis')
        A('sensitivity', 'Sensitivity…', lambda: self.set_page('tolerance'), None, 'tolerance', category='Analysis')
        A('load_cases', 'Load cases…', self.load_cases, None, 'axle', category='Analysis')
        A('blind_volume', 'Blind volume…', lambda: self.set_page('volume'), None, 'blind', category='Analysis',
          tip='Where around the vehicle something can stand unseen by every camera, in 3D')
        A('perception_range', 'Perception range…', lambda: self.set_page('perception'), None, 'target',
          category='Analysis', tip='How far out the rig resolves cars, pedestrians, cyclists and signs in each direction')
        A('sensor_coverage', 'Lidar and radar coverage…', lambda: self.set_page('coverage'), None, 'sensor',
          category='Analysis', tip='Where the lidars and radars reach, what they hit on the vehicle, how far they detect '
                                   'each object, and how much of each camera image they cover')
        A('sync_plan', 'Sync and timing…', lambda: self.set_page('timing'), None, 'sync', category='Analysis',
          tip='Camera trigger offsets against the lidar sweep, trigger groups, motion blur and rolling shutter at speed')
        A('recording_budget', 'Recording budget…', lambda: self.set_page('recording'), None, 'ecu', category='Analysis',
          tip='Data rate of every sensor, the network and storage it needs, and hours of driving per disk set')
        A('an_settings', 'Settings…', self.preferences, None, 'gear', category='Analysis')
        # calibrate
        A('targets', 'Target planner…', lambda: self.set_page('targets'), None, 'checker', category='Calibrate')
        for k, t in (('checkerboard', 'Checkerboard'), ('charuco', 'ChArUco'), ('aprilgrid', 'AprilGrid')):
            A(f'target_{k}', t, lambda k=k: self._add_target(k), None, 'checker', category='Calibrate ▸ Add target')
        A('target_auto', 'Auto-place targets', lambda: (self.set_page('targets'), self.inspector.pages['targets']._auto()), None, None, category='Calibrate')
        A('target_check', 'Check visibility', lambda: (self.set_page('targets'), self.inspector.pages['targets']._check()), None, None, category='Calibrate')
        A('target_export', 'Export target positions…', lambda: self.inspector.pages['targets']._export(), None, 'export', category='Calibrate')
        A('install', 'Installation check…', lambda: self.set_page('install'), 'Ctrl+Shift+I', 'target', category='Calibrate')
        A('live_rig', 'Live rig…', lambda: self.set_page('rig'), None, 'camera-wide', category='Calibrate',
          tip='Every camera\'s stream at once: frame rate, drops, exposure, sharpness, sync and drift from the plan')
        A('lens_calib', 'Lens calibration…', self.lens_calibration, 'Ctrl+Shift+L', 'lens', category='Calibrate')
        A('calib_import', 'Import unit calibrations…', self.import_calibrations, None, 'file', category='Calibrate')
        A('calib_export', 'Export unit calibrations…', self.export_calibrations, None, 'export', category='Calibrate')
        A('calib_clear', 'Clear unit calibration', self.clear_calibration, None, None, category='Calibrate')
        # export
        A('export', 'Export…', self.export_dialog, 'Ctrl+E', 'export', category='Export')
        for fid, text in (('mount_sheet_pdf', 'Mount sheet (PDF)'), ('report_html', 'Report (HTML)'), ('report_pdf', 'Report (PDF)'),
                          ('extrinsics_json', 'JSON'), ('extrinsics_yaml', 'YAML'), ('extrinsics_csv', 'CSV'),
                          ('opencv_yaml', 'OpenCV calibration YAML'), ('ros2', 'ROS 2 URDF + static TF'), ('jetson', 'Jetson camera config'),
                          ('img_viewport', 'Viewport'), ('img_bev', 'Bird’s-eye'), ('img_cams', 'All camera views'),
                          ('dxf', 'DXF'), ('step', 'STEP')):
            cat = 'Export ▸ Extrinsics' if fid.startswith('extrinsics') else 'Export ▸ Images' if fid.startswith('img') else \
                'Export ▸ Mount points' if fid in ('dxf', 'step') else 'Export'
            A(f'exp_{fid}', text, lambda f=fid: self.export_dialog(preselect=f), None, None, category=cat)
        # window
        A('ws_placement', 'Placement', lambda: self.workspace('placement'), None, None, category='Window ▸ Workspaces')
        A('ws_analysis', 'Analysis', lambda: self.workspace('analysis'), None, None, category='Window ▸ Workspaces')
        A('ws_review', 'Review', lambda: self.workspace('review'), None, None, category='Window ▸ Workspaces')
        A('ws_save', 'Save current…', self.save_workspace, None, None, category='Window ▸ Workspaces')
        A('ws_reset', 'Reset layout', lambda: self.workspace('placement'), None, None, category='Window')
        # help
        A('shortcuts', 'Keyboard shortcuts', self.shortcuts, '?', 'keyboard', category='Help')
        A('help_frames', 'Coordinate frames', lambda: self.help_topic('frames'), None, 'axes', category='Help')
        A('help_lens', 'Lens projection models', lambda: self.help_topic('lens'), None, 'lens', category='Help')
        A('about', 'About Vantage', self.about, None, 'app', category='Help')

    def _build_menus(self):
        mb = self.menuBar()
        a = self.actions_by_id

        def menu(title):
            return mb.addMenu(title)

        def sub(parent, title, ids):
            m = parent.addMenu(title)
            for i in ids:
                if i == '-':
                    m.addSeparator()
                else:
                    m.addAction(a[i])
            return m
        f = menu('File')
        f.addActions([a['new']])
        self.tpl_menu = f.addMenu('New from template')
        for key, name in (('tigor', 'Tata Tigor EV (photo-measured)'), ('sedan', 'Sedan'), ('hatchback', 'Hatchback'), ('suv', 'SUV'),
                          ('van', 'Van'), ('truck', 'Truck + trailer')):
            self.tpl_menu.addAction(name, lambda k=key: self.new_from_template(k))
        f.addAction(a['open'])
        self.recent_menu = f.addMenu('Open recent')
        self.recent_menu.aboutToShow.connect(self._fill_recent)
        f.addSeparator()
        f.addActions([a['save'], a['saveas']])
        f.addSeparator()
        sub(f, 'Import', ['import_vehicle', 'import_photos', 'import_module', 'import_layout'])
        sub(f, 'Export', ['export', '-', 'exp_mount_sheet_pdf', 'exp_report_html', 'exp_extrinsics_json', 'exp_ros2'])
        f.addSeparator()
        f.addActions([a['project_settings'], a['prefs'], a['start']])
        f.addSeparator()
        f.addAction(a['quit'])
        e = menu('Edit')
        e.addActions([a['undo'], a['redo']])
        e.addSeparator()
        e.addActions([a['duplicate'], a['delete']])
        e.addSeparator()
        e.addActions([a['mirror'], a['link']])
        sub(e, 'Select', ['sel_all', 'sel_left', 'sel_right', 'sel_invert'])
        e.addSeparator()
        e.addActions([a['find'], a['palette']])
        v = menu('View')
        sub(v, 'Viewport layout', ['split1', 'split2', 'split4'])
        sub(v, 'Camera', ['view_persp', 'view_top', 'view_front', 'view_side', 'view_rear', 'view_through'])
        sub(v, 'Shading', ['shade_solid', 'shade_wire', 'shade_xray', 'shade_photo', '-', 'photo_prev', 'photo_next'])
        ov = v.addMenu('Overlays')
        for key, _, _, _ in OVERLAYS[:2]:
            ov.addAction(a[f'ov_{key}'])
        hm = ov.addMenu('Heatmap')
        for key, _ in HEATMAPS:
            hm.addAction(a[f'heat_{key}'])
        for key, _, _, _ in OVERLAYS[2:]:
            ov.addAction(a[f'ov_{key}'])
        v.addSeparator()
        v.addActions([a['camviews'], a['back']])
        v.addSeparator()
        v.addActions([a['render_image'], a['turntable']])
        v.addSeparator()
        v.addActions([a['frame_sel'], a['frame_all']])
        v.addSeparator()
        sub(v, 'Units', ['units_m', 'units_mm'])
        sub(v, 'Coordinate frame', [f'frame_{k}' for k in FRAME_PRESETS])
        sub(v, 'Theme', ['theme_dark', 'theme_light', 'theme_system'])
        v.addSeparator()
        v.addAction(a['maximise'])
        ve = menu('Vehicle')
        ve.addActions([a['veh_replace'], a['veh_align'], a['veh_wheels'], a['veh_ground']])
        sub(ve, 'Mount zones', ['zone_allow', 'zone_forbid', 'zone_glass', 'zone_clear'])
        sub(ve, 'Occluder', ['occ_full', 'occ_hull', 'occ_level'])
        sub(ve, 'Reference photos', ['photo_load', 'photo_cal', 'photo_pane'])
        sub(ve, 'Add rig part', ['rig_roof_rack', 'rig_sensor_mast', 'rig_enclosure', 'rig_bar'])
        for kind in ('lidar', 'radar'):
            sm = ve.addMenu(f'Add {kind}')
            sm.setIcon(icons.icon('sensor'))
            sm.aboutToShow.connect(lambda sm=sm, kind=kind: self._fill_sensor_menu(sm, kind))
        ve.addSeparator()
        ve.addAction(a['spec'])
        ca = menu('Camera')
        ca.addActions([a['add_cam'], a['place']])
        self.mod_menu = ca.addMenu('Change module')
        self.lens_menu = ca.addMenu('Change lens')
        self.mod_menu.aboutToShow.connect(lambda: self._fill_lens_menu(self.mod_menu, True))
        self.lens_menu.aboutToShow.connect(lambda: self._fill_lens_menu(self.lens_menu, False))
        sub(ca, 'Aim', ['aim_ground', 'aim_normal', 'aim_level'])
        ca.addSeparator()
        ca.addActions([a['lock'], a['hide'], a['unhide'], a['isolate'], a['snap_surface'], a['assign_port']])
        ca.addSeparator()
        ca.addActions([a['library'], a['edit_module'], a['copy_pose']])
        la = menu('Layout')
        la.addActions([a['var_new'], a['var_dup'], a['var_rename'], a['var_delete']])
        self.var_menu = la.addMenu('Switch to')
        self.var_menu.aboutToShow.connect(self._fill_variants)
        sub(la, 'Presets', ['preset_4', 'preset_6', 'preset_6f', 'preset_8'])
        la.addSeparator()
        la.addActions([a['compare'], a['optimise']])
        an = menu('Analysis')
        an.addActions([a['run'], a['live']])
        an.addSeparator()
        an.addAction(a['roi'])
        sub(an, 'Ground resolution', ['res_5', 'res_2', 'res_1'])
        sub(an, 'Obstacles', ['obs_ground', 'obs_pole05', 'obs_pole10', 'obs_child', '-', 'obs_add'])
        an.addSeparator()
        an.addActions([a['thresholds'], a['tolerance'], a['sensitivity'], a['load_cases'], a['blind_volume'],
                       a['perception_range']])
        an.addSeparator()
        an.addActions([a['sensor_coverage'], a['sync_plan'], a['recording_budget']])
        an.addSeparator()
        an.addAction(a['an_settings'])
        cb = menu('Calibrate')
        cb.addAction(a['targets'])
        sub(cb, 'Add target', ['target_checkerboard', 'target_charuco', 'target_aprilgrid'])
        cb.addActions([a['target_auto'], a['target_check'], a['target_export']])
        cb.addSeparator()
        cb.addActions([a['lens_calib'], a['calib_import'], a['calib_export'], a['calib_clear']])
        cb.addSeparator()
        cb.addActions([a['install'], a['live_rig']])
        ex = menu('Export')
        ex.addAction(a['export'])
        ex.addSeparator()
        ex.addActions([a['exp_mount_sheet_pdf'], a['exp_report_html'], a['exp_report_pdf']])
        sub(ex, 'Extrinsics', ['exp_extrinsics_json', 'exp_extrinsics_yaml', 'exp_extrinsics_csv'])
        ex.addActions([a['exp_opencv_yaml'], a['exp_ros2'], a['exp_jetson']])
        sub(ex, 'Images', ['exp_img_viewport', 'exp_img_bev', 'exp_img_cams'])
        sub(ex, 'Mount points', ['exp_dxf', 'exp_step'])
        w = menu('Window')
        self.win_menu = w
        w.aboutToShow.connect(self._fill_window_menu)
        h = menu('Help')
        h.addActions([a['shortcuts'], a['help_frames'], a['help_lens']])
        h.addSeparator()
        h.addAction(a['about'])
        corner = QWidget()
        cl = QHBoxLayout(corner)
        cl.setContentsMargins(0, 0, 10, 0)
        cl.setSpacing(8)
        self.file_label = QLabel('')
        self.file_label.setFont(mono(11))
        self.file_label.setProperty('ink2', True)
        self.saved_label = QLabel('')
        self.saved_label.setFont(mono(11))
        self.saved_label.setProperty('muted', True)
        cl.addWidget(self.file_label)
        cl.addWidget(self.saved_label)
        mb.setCornerWidget(corner, Qt.Corner.TopRightCorner)

    def _build_toolbar(self):
        tb = QToolBar('Tools')
        tb.setObjectName('toolbar')
        tb.setIconSize(QSize(16, 16))
        tb.setMovable(False)
        tb.setFloatable(False)
        tb.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonIconOnly)
        tb.setFixedHeight(32)
        a = self.actions_by_id
        for k, *_ in TOOLS:
            tb.addAction(a[f'tool_{k}'])
        tb.addSeparator()
        for k in ('view_persp', 'view_top', 'view_side', 'view_through'):
            tb.addAction(a[k])
        tb.addSeparator()
        # shading menu button
        sb = QToolButton()
        sb.setIcon(icons.icon('shading', 16))
        sb.setToolTip('Shading\nSolid, wire, X-ray or photo-matched')
        sb.setPopupMode(QToolButton.ToolButtonPopupMode.InstantPopup)
        sm = QMenu(sb)
        for k in ('solid', 'wire', 'xray', 'photo'):
            sm.addAction(a[f'shade_{k}'])
        sb.setMenu(sm)
        sb.setFixedWidth(36)
        tb.addWidget(sb)
        ob = QToolButton()
        ob.setIcon(icons.icon('layers', 16))
        ob.setToolTip('Overlays\nFrustums, footprints, heatmap, seams, ROI…')
        ob.setPopupMode(QToolButton.ToolButtonPopupMode.InstantPopup)
        om = QMenu(ob)
        wa = QWidgetAction(om)
        wa.setDefaultWidget(self._overlay_popover())
        om.addAction(wa)
        ob.setMenu(om)
        ob.setFixedWidth(36)
        self.overlay_button = ob
        tb.addWidget(ob)
        tb.addSeparator()
        for k in ('split1', 'split2', 'split4'):
            tb.addAction(a[k])
        spacer = QWidget()
        spacer.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred)
        tb.addWidget(spacer)
        # variant combo
        vw = QWidget()
        vl = QHBoxLayout(vw)
        vl.setContentsMargins(0, 0, 0, 0)
        vl.setSpacing(0)
        self.variant_combo = QComboBox()
        self.variant_combo.setFixedWidth(160)
        self.variant_combo.setToolTip('Layout variant')
        self.variant_combo.activated.connect(self._variant_chosen)
        vl.addWidget(self.variant_combo)
        tb.addWidget(vw)
        tb.addWidget(QLabel('  '))
        # analysis state
        self.state_btn = QToolButton()
        self.state_btn.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonTextBesideIcon)
        self.state_btn.setFont(sans(12))
        self.state_btn.setToolTip('Analysis state\nClick to run now (F5). Right-click for settings.')
        self.state_btn.clicked.connect(self.ctx.analysis.run_now)
        self.state_btn.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self.state_btn.customContextMenuRequested.connect(lambda pos: self._state_menu(pos))
        self.state_btn.setProperty('text-button', True)
        tb.addWidget(self.state_btn)
        tb.addWidget(QLabel(' '))
        tb.addAction(self.actions_by_id['assistant'])
        tb.addWidget(QLabel(' '))
        self.search = QLineEdit()
        self.search.setPlaceholderText('Search')
        self.search.setFixedWidth(180)
        self.search.addAction(icons.icon('search', 12), QLineEdit.ActionPosition.LeadingPosition)
        hint = QAction('Ctrl K', self.search)
        self.search.setToolTip('Command palette  Ctrl+K')
        self.search.installEventFilter(self)
        tb.addWidget(self.search)
        self.addToolBar(Qt.ToolBarArea.TopToolBarArea, tb)
        self.toolbar = tb
        # 2 px progress line under the toolbar
        self.progress_line = QProgressBar(self)
        self.progress_line.setTextVisible(False)
        self.progress_line.setRange(0, 1000)
        self.progress_line.hide()

    def _overlay_popover(self):
        w = QWidget()
        w.setObjectName('panel')
        w.setMinimumWidth(300)
        lay = QVBoxLayout(w)
        lay.setContentsMargins(10, 8, 10, 8)
        lay.setSpacing(4)
        lay.addWidget(label('Heatmap · one at a time', muted=True, small=True))
        self.heat_seg = SegmentedControl(HEATMAPS, self.ctx.heatmap)
        self.heat_seg.changed.connect(self.set_heatmap)
        lay.addWidget(self.heat_seg)
        lay.addSpacing(4)
        self.ov_checks = {}
        for key, text, ic, _ in OVERLAYS:
            row = QHBoxLayout()
            cb = QCheckBox(text)
            cb.setChecked(self.ctx.overlays[key])
            cb.toggled.connect(lambda on, k=key: self.actions_by_id[f'ov_{k}'].setChecked(on))
            il = QLabel()
            il.setPixmap(icons.pixmap(ic, THEME.tok['ink2'], 12))
            sc = QLabel(self.actions_by_id[f'ov_{key}'].shortcut().toString())
            sc.setFont(mono(11))
            sc.setProperty('muted', True)
            row.addWidget(cb)
            row.addStretch(1)
            row.addWidget(sc)
            lay.addLayout(row)
            self.ov_checks[key] = cb
        return w

    def _build_statusbar(self):
        sb = QStatusBar()
        sb.setSizeGripEnabled(False)
        self.setStatusBar(sb)
        self.frame_btn = QToolButton()
        self.frame_btn.setIcon(icons.icon('axes', 12))
        self.frame_btn.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonTextBesideIcon)
        self.frame_btn.setPopupMode(QToolButton.ToolButtonPopupMode.InstantPopup)
        fm = QMenu(self.frame_btn)
        for k in FRAME_PRESETS:
            fm.addAction(self.actions_by_id[f'frame_{k}'])
        fm.addSeparator()
        fm.addAction(self.actions_by_id['units_m'])
        fm.addAction(self.actions_by_id['units_mm'])
        self.frame_btn.setMenu(fm)
        self.coord_label = QLabel('')
        self.coord_label.setFont(mono(11))
        self.coord_label.setMinimumWidth(200)
        self.snap_btn = QToolButton()
        self.snap_btn.setIcon(icons.icon('snap-surface', 12))
        self.snap_btn.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonTextBesideIcon)
        self.snap_btn.setToolTip('Snapping while dragging cameras: surface → grid 0.05 m → off')
        self.snap_btn.clicked.connect(self._cycle_snap)
        self.sel_label = QLabel('')
        for w in (self.frame_btn, self.coord_label, self.snap_btn, self.sel_label):
            sb.addPermanentWidget(w) if False else sb.addWidget(w)
            sep = QLabel()
            sep.setFixedSize(1, 14)
            sep.setStyleSheet(f'background: {THEME.tok["hair"]};')
            sb.addWidget(sep)
        self.job_label = QLabel('')
        self.job_bar = QProgressBar()
        self.job_bar.setProperty('inline', True)
        self.job_bar.setFixedWidth(120)
        self.job_bar.setTextVisible(False)
        self.job_bar.setRange(0, 1000)
        self.job_pct = QLabel('')
        self.job_pct.setFont(mono(11))
        self.job_cancel = tool_button('close', 'Cancel', size=16, icon_size=12)
        self.job_cancel.clicked.connect(self.cancel_jobs)
        for w in (self.job_label, self.job_bar, self.job_pct, self.job_cancel):
            sb.addPermanentWidget(w)
            w.hide()
        self.time_label = QLabel('')
        self.time_label.setFont(mono(11))
        sb.addPermanentWidget(self.time_label)
        self.msg_label = QLabel('')
        sb.insertWidget(0, self.msg_label)
        self.msg_label.hide()
        self._msg_timer = QTimer(self)
        self._msg_timer.setSingleShot(True)
        self._msg_timer.timeout.connect(lambda: self.msg_label.hide())

    def _wire(self):
        st = self.store
        c = self.ctx
        st.changed.connect(self._on_changed)
        st.project_replaced.connect(self._on_project_replaced)
        st.selection_changed.connect(self._on_selection)
        st.dirty_changed.connect(lambda d: self.set_project_meta())
        st.undo.indexChanged.connect(lambda *_: self.ctx.analysis.request())
        c.analysis.started.connect(self._on_analysis_started)
        c.analysis.finished.connect(self._on_analysis_done)
        c.analysis.stale_changed.connect(lambda s: self._update_state())
        c.analysis.failed.connect(lambda m: (self.msg(f'Analysis failed: {m.splitlines()[0]}', 8000), print(m)))
        c.cursor_world.connect(self._cursor)
        c.message.connect(self.msg)
        c.tool_changed.connect(self._sync_tool)
        c.overlays_changed.connect(self._sync_overlay_ui)
        THEME.changed.connect(lambda *_: self._retheme())

    def _first_run(self):
        self._fill_variant_combo()
        self.sync_view_buttons()
        self.ctx.analysis.request(force=True)
        self._update_state()
        self._on_selection()
        self.viewarea.frame_all()
        self.set_project_meta()
        self._cycle_snap(init=True)

    # ------------------------------------------------------------------ events
    def eventFilter(self, obj, e):
        from PyQt6.QtCore import QEvent
        if obj is self.search and e.type() in (QEvent.Type.MouseButtonPress, QEvent.Type.FocusIn):
            QTimer.singleShot(0, lambda: (self.search.clearFocus(), self.command_palette()))
            return True
        return super().eventFilter(obj, e)

    def resizeEvent(self, e):
        super().resizeEvent(e)
        self._place_progress()

    def dragEnterEvent(self, e):
        from ..core.loaders import SUPPORTED_EXTS
        if e.mimeData().hasUrls() and any(u.toLocalFile().lower().endswith(SUPPORTED_EXTS + ('.vantage',))
                                          for u in e.mimeData().urls()):
            e.acceptProposedAction()

    def dropEvent(self, e):
        from ..core.loaders import SUPPORTED_EXTS
        for u in e.mimeData().urls():
            path = u.toLocalFile()
            if path.lower().endswith('.vantage'):
                self.open_project(path)
                return
            if path.lower().endswith(SUPPORTED_EXTS):
                QTimer.singleShot(0, lambda p=path: self.import_vehicle(path=p))
                return

    def _place_progress(self):
        tb = self.toolbar
        y = tb.geometry().bottom() - 1
        self.progress_line.setGeometry(0, y, self.width(), 2)

    def keyPressEvent(self, e):
        if e.key() == Qt.Key.Key_Escape and getattr(self, '_preview_restore', None):
            self._preview_restore()
            self._preview_restore = None
            self.set_page('optimiser')
            return
        if e.key() == Qt.Key.Key_Escape and self.page_name() != 'viewports':
            self.set_page('viewports')
            return
        if e.key() == Qt.Key.Key_0 and e.modifiers() & Qt.KeyboardModifier.KeypadModifier:
            self.set_view('through')
            return
        super().keyPressEvent(e)

    def closeEvent(self, e):
        if not self.store.undo.isClean():
            r = QMessageBox.question(self, 'Unsaved changes', 'Save changes to this project before closing?',
                                     QMessageBox.StandardButton.Save | QMessageBox.StandardButton.Discard |
                                     QMessageBox.StandardButton.Cancel, QMessageBox.StandardButton.Save)
            if r == QMessageBox.StandardButton.Cancel:
                e.ignore()
                return
            if r == QMessageBox.StandardButton.Save and not self.save():
                e.ignore()
                return
        self.ctx.settings.setValue('window/state', self.saveState())
        self.cancel_jobs()
        if 'install' in self._page_widgets:
            self._page_widgets['install'].stop_live()
        if 'rig' in self._page_widgets:
            self._page_widgets['rig'].stop()
        # the undo stack and analysis keep emitting while child widgets are torn down
        for obj in (self.store.undo, self.store, self.ctx.analysis, self.ctx):
            obj.blockSignals(True)
        e.accept()

    # ------------------------------------------------------------------ state sync
    def _on_changed(self, sections):
        self.ctx.analysis.request()
        self.set_project_meta()
        if 'variants' in sections or 'variant' in sections:
            self._fill_variant_combo()
        if 'frame' in sections:
            self._sync_frame()
        if 'analysis' in sections:
            self._sync_analysis_menu()

    def _sync_analysis_menu(self):
        """Radio items that mirror project settings follow undo, redo and opening a project."""
        an = self.store.project.analysis
        cm = min((5, 2, 1), key=lambda k: abs(k / 100 - an.cell_m))
        for key in (f'res_{cm}', f'obs_{an.obstacle_mode}'):
            a = self.actions_by_id.get(key)
            if a is not None and not a.isChecked():
                a.setChecked(True)
        if self.actions_by_id['live'].isChecked() != an.live:
            self.actions_by_id['live'].setChecked(an.live)

    def _on_project_replaced(self):
        self.ctx.isolate.clear()
        self._photo_cache.clear()
        self._fill_variant_combo()
        self.set_project_meta()
        self._sync_frame()
        self._sync_analysis_menu()
        self.ctx.analysis.result = None
        self.ctx.analysis.request(force=True)
        QTimer.singleShot(50, self.viewarea.frame_all)

    def _on_selection(self):
        st = self.store
        cams = st.selected_cams()
        if cams:
            t = ', '.join(c.tag for c in cams)
            self.sel_label.setText(f'{len(st.variant.cameras)} cameras · {t} selected')
        elif st.sel_obj:
            names = {'roi': 'ROI selected', 'target': 'target selected', 'vehicle': 'vehicle selected', 'photos': 'reference photos',
                     'targets': f'{len(st.project.targets)} targets', 'ecu': 'compute', 'variant': 'variant'}
            o = st.sel_obj
            if o[0] == 'target':
                t = next((t for t in st.project.targets if t.uid == o[1]), None)
                self.sel_label.setText(f'{len(st.project.targets)} targets · {t.name if t else ""} selected')
            else:
                self.sel_label.setText(names.get(o[0], o[0]))
        else:
            self.sel_label.setText(f'{len(st.variant.cameras)} cameras')
        has = bool(cams)
        for k in ('duplicate', 'mirror', 'lock', 'hide', 'isolate', 'copy_pose', 'snap_surface', 'look_through'):
            self.actions_by_id[k].setEnabled(has)

    def _on_analysis_started(self):
        self._update_state(running=True)

    def _on_analysis_done(self, res):
        self._update_state()
        if res is not None:
            self.time_label.setText(f'analysed {res.ms:.0f} ms')

    def _update_state(self, running=False):
        live = self.store.project.analysis.live
        a = self.ctx.analysis
        if self._jobs:
            name, frac, text = next(iter(self._jobs.values()))['state']
            dot, txt, rest = STATUS['good'], 'Running', f'{name} {frac * 100:.0f} %'
        elif running:
            dot, txt, rest = STATUS['warning'], 'Analysing', ''
        elif a.stale:
            dot, txt, rest = STATUS['warning'], 'Stale', 'F5 to run'
        else:
            ms = a.result.ms if a.result else 0
            dot, txt, rest = (STATUS['good'] if live else THEME.tok['muted']), ('Live' if live else 'Manual'), f'{ms:.0f} ms'
        self.state_btn.setIcon(icons.icon('dot', 8, dot) if 'dot' in icons.ICONS else _dot_icon(dot))
        self.state_btn.setText(f'{txt}  {rest}')

    def _state_menu(self, pos):
        m = QMenu(self)
        a = self.actions_by_id
        m.addActions([a['run'], a['live']])
        m.addSeparator()
        m.addAction(a['roi'])
        r = m.addMenu('Ground resolution')
        r.addActions([a['res_5'], a['res_2'], a['res_1']])
        o = m.addMenu('Obstacles')
        o.addActions([a['obs_ground'], a['obs_pole05'], a['obs_pole10'], a['obs_child']])
        m.addSeparator()
        m.addActions([a['thresholds'], a['tolerance'], a['an_settings']])
        m.exec(self.state_btn.mapToGlobal(pos))

    def _cursor(self, w):
        if w is None:
            self.coord_label.setText('')
            return
        p = self.store.project
        o = p.origin_offset()
        k = 1000 if p.units == 'mm' else 1
        dp = 0 if p.units == 'mm' else 3
        x, y, z = (w[0] - o[0]) * k, (w[1] - o[1]) * k, (w[2] - o[2]) * k
        self.coord_label.setText(f'x {x:.{dp}f}  y {y:.{dp}f}  z {z:.{dp}f}'.replace('-', '−'))

    def _cycle_snap(self, init=False):
        order = ['surface', 'grid', 'off']
        if not init:
            self.ctx.snap = order[(order.index(self.ctx.snap) + 1) % 3]
        self.snap_btn.setText({'surface': 'snap: surface', 'grid': 'snap: grid 0.05 m', 'off': 'snap: off'}[self.ctx.snap])
        self.snap_btn.setIcon(icons.icon('snap-surface' if self.ctx.snap == 'surface' else 'snap-grid', 12))

    def _sync_frame(self):
        p = self.store.project
        self.frame_btn.setText(f'{p.frame_label} · {p.units}')
        a = self.actions_by_id.get(f'frame_{p.frame}')
        if a and a.isCheckable():
            a.setChecked(True)
        self.actions_by_id['units_mm' if p.units == 'mm' else 'units_m'].setChecked(True)

    def _sync_overlay_ui(self):
        for key, *_ in OVERLAYS:
            a = self.actions_by_id[f'ov_{key}']
            if a.isChecked() != self.ctx.overlays[key]:
                a.blockSignals(True)
                a.setChecked(self.ctx.overlays[key])
                a.blockSignals(False)
            cb = self.ov_checks.get(key)
            if cb and cb.isChecked() != self.ctx.overlays[key]:
                cb.blockSignals(True)
                cb.setChecked(self.ctx.overlays[key])
                cb.blockSignals(False)
        self.heat_seg.set_value(self.ctx.heatmap)
        self.actions_by_id[f'heat_{self.ctx.heatmap}'].setChecked(True)

    def sync_view_buttons(self):
        pane = self.viewarea.active
        k = pane.kind if self.page_name() == 'viewports' else ''
        aid = {'persp': 'view_persp', 'top': 'view_top', 'front': 'view_front', 'side': 'view_side', 'rear': 'view_rear',
               'through': 'view_through'}.get(k)
        for x in ('view_persp', 'view_top', 'view_front', 'view_side', 'view_rear', 'view_through'):
            self.actions_by_id[x].setChecked(x == aid)
        m = self.viewarea.mode
        self.actions_by_id[{'single': 'split1', 'side': 'split2', 'quad': 'split4'}[m]].setChecked(self.page_name() == 'viewports')

    def set_project_meta(self):
        p = self.store.project
        name = os.path.basename(p.path) if p.path else f'{p.name}.vantage'
        self.file_label.setText(name)
        dirty = not self.store.undo.isClean()
        if p.path and not dirty and p.saved_at:
            self.saved_label.setText('saved ' + time.strftime('%H:%M', time.localtime(p.saved_at)))
        else:
            self.saved_label.setText('unsaved' if dirty or not p.path else '')
        self.setWindowTitle(f'Vantage — {p.name}' + (' •' if dirty else ''))
        self._sync_frame()

    def _fill_variant_combo(self):
        cb = self.variant_combo
        cb.blockSignals(True)
        cb.clear()
        p = self.store.project
        for v in p.variants:
            cb.addItem(f'Variant  {v.name}', v.uid)
        cb.setCurrentIndex(max(0, cb.findData(p.active_uid)))
        cb.blockSignals(False)

    def _variant_chosen(self, i):
        uid = self.variant_combo.itemData(i)
        self.switch_variant(uid)

    def switch_variant(self, uid):
        p = self.store.project
        if uid == p.active_uid:
            return
        name = p.variant_by_uid(uid).name
        self.store.edit('variants', lambda pj: setattr(pj, 'active_uid', uid), f'Switched to {name}')
        cams = p.variant.cameras
        self.store.select_cams([cams[0].uid] if cams else [])
        self._fill_variant_combo()

    def _retheme(self):
        for key, act in self.actions_by_id.items():
            d = act.data() or {}
            if d.get('icon'):
                act.setIcon(icons.icon(d['icon'], 16))
        self.frame_btn.setIcon(icons.icon('axes', 12))
        self._cycle_snap(init=True)
        self.overlay_button.setIcon(icons.icon('layers', 16))
        self._update_state()
        self.update()

    def msg(self, text, ms=3000):
        self.msg_label.setText(text)
        self.msg_label.show()
        self._msg_timer.start(ms)

    # ------------------------------------------------------------------ view commands
    def set_split(self, mode):
        if self.page_name() != 'viewports':
            self.set_page('viewports')
        self.viewarea.set_mode(mode)
        self.sync_view_buttons()
        QTimer.singleShot(30, self.viewarea.frame_all)

    def leave_photo_view(self):
        """Back from a photo-matched view to a free one: photo, label and shading all go."""
        self.photo_unmatch_view()
        if self.ctx.shading == 'photo':
            self.ctx.set_shading('solid')
            self.actions_by_id['shade_solid'].setChecked(True)

    def set_view(self, kind):
        if self.ctx.shading == 'photo':
            self.leave_photo_view()
        if self.page_name() != 'viewports':
            self.set_page('viewports')
        pane = self.viewarea.active
        pane.set_kind(kind)
        if kind in ('persp',):
            pane.vp3d.frame_box(*self.viewarea._bbox())
        self.sync_view_buttons()

    def set_tool(self, t):
        self.ctx.set_tool(t)

    def _sync_tool(self, t):
        a = self.actions_by_id.get(f'tool_{t}')
        if a is not None:
            a.setChecked(True)

    def start_aim(self):
        cams = self.store.selected_cams()
        if not cams:
            self.msg('Select the camera to aim first.')
            return
        if self.page_name() != 'viewports':
            self.set_page('viewports')
        self.ctx.set_tool('aim')
        self.msg(f'Click the ground to aim {", ".join(c.tag for c in cams)} there · Esc cancels', 8000)

    def set_heatmap(self, k):
        self.ctx.set_heatmap(k)
        if k != 'none' and self.page_name() == 'viewports' and self.viewarea.mode == 'single' and self.viewarea.active.kind == 'persp':
            pass
        for p in self.viewarea.panes:
            p._labels()

    def set_shading(self, k):
        if k == 'photo':
            if not self.photo_match_view():
                self.actions_by_id[f'shade_{self.ctx.shading}'].setChecked(True)
                return
        else:
            self.photo_unmatch_view()
        self.ctx.set_shading(k)

    def photo_match_view(self, k=None) -> bool:
        """Look through the calibrated reference photo in the active 3D view, the model drawn translucent over it."""
        from ..core import photos as ph
        from .workspaces.photo_ws import gl_from_photo
        ps = self.store.project.photos
        if not ps.frames:
            self.msg('Load a reference photo set first (Vehicle ▸ Reference photos).', 6000)
            return False
        order = [ps.current if k is None else k] + list(range(len(ps.frames)))
        m = None
        for j in order:
            m = gl_from_photo(ps, j)
            if m is not None:
                k = j
                break
        if m is None:
            self.msg('No photo has a solved pose yet. Add landmarks and press Calibrate on the photo page.', 6000)
            return False
        ps.current = k
        if self.page_name() != 'viewports':
            self.set_page('viewports')
        pane = self.viewarea.active
        if pane.stack.currentWidget() is not pane.vp3d:
            pane.set_kind('persp')
        pane.scene.photo_frame = k
        pane.vp3d.cam.custom = m
        self.ctx.photo_shading = True
        pane.scene.label_prefix = (f'Perspective · matched to photo {k:02d} / {len(ps.frames)} · '
                                   f'f {ph.photo_focal_35mm(ps):.0f} mm (35 mm eq.) · [ ] step · choose a view to leave')
        pane.vp3d.update()
        return True

    def photo_unmatch_view(self):
        for pane in self.viewarea.panes:
            if pane.scene.photo_frame is not None or pane.vp3d.cam.custom is not None:
                pane.scene.photo_frame = None
                pane.vp3d.cam.custom = None
                pane.scene.label_prefix = ''
                pane.vp3d.update()

    def photo_step(self, d):
        ps = self.store.project.photos
        if self.ctx.shading != 'photo' or not ps.frames:
            return
        n = len(ps.frames)
        for i in range(1, n + 1):
            j = (ps.current + d * i) % n
            if ps.frames[j].pose is not None and self.photo_match_view(j):
                return

    def frame_selection(self):
        cams = self.store.selected_cams()
        if not cams:
            return self.frame_all()
        P = np.array([c.pos() for c in cams])
        lo, hi = P.min(axis=0) - 0.9, P.max(axis=0) + 0.9
        lo[2] = 0
        for pane in self.viewarea.visible_panes():
            pane.vp3d.frame_box(lo, hi, 1.1)
            pane.plan.fit((lo[0], lo[1], hi[0], hi[1]))

    def frame_point(self, xy):
        x, y = xy
        lo, hi = np.array([x - 1.2, y - 1.2, 0]), np.array([x + 1.2, y + 1.2, 0.5])
        for pane in self.viewarea.visible_panes():
            pane.vp3d.frame_box(lo, hi, 1.1)
            pane.plan.fit((lo[0], lo[1], hi[0], hi[1]))
            pane.plan.pinned_marker = (x, y)

    def frame_all(self):
        self.viewarea.frame_all()

    def toggle_maximise(self):
        if self._maximised_state is None:
            self._maximised_state = {k: d.isVisible() for k, d in self.docks.items()}
            for d in self.docks.values():
                d.hide()
        else:
            for k, d in self.docks.items():
                d.setVisible(self._maximised_state.get(k, True))
            self._maximised_state = None

    def toggle_isolate(self):
        if self.ctx.isolate:
            self.ctx.isolate.clear()
        else:
            self.ctx.isolate.update(self.store.sel_cams)
        self.ctx.overlays_changed.emit()

    def set_units(self, u):
        self.store.edit('frame', lambda p: setattr(p, 'units', u), f'Units set to {u}')

    def set_frame(self, k):
        self.store.edit('frame', lambda p: setattr(p, 'frame', k), f'Frame set to {FRAME_PRESETS[k]}')

    def set_theme(self, k):
        THEME.apply(QApplication.instance(), k)
        self.ctx.settings.setValue('ui/theme', k)

    def set_live(self, on):
        self.store.project.analysis.live = on
        self._update_state()
        if on:
            self.ctx.analysis.request(force=True)

    def set_cell(self, cell):
        self.store.edit('analysis', lambda p: setattr(p.analysis, 'cell_m', cell), f'Ground resolution {cell * 100:.0f} cm')
        self.ctx.analysis.request(force=True)

    def set_obstacle_mode(self, k):
        self.store.edit('analysis', lambda p: setattr(p.analysis, 'obstacle_mode', k), 'Obstacle test mode')

    def set_occluder(self, n):
        self.store.edit('vehicle', lambda p: setattr(p.vehicle, 'occluder_target', n), f'Occluder target {n} tris')

    def focus_roi(self):
        if self.page_name() != 'viewports':
            self.set_page('viewports')
        self.store.select_obj(('roi',))
        self.docks['inspector'].show()

    def edit_roi_on_map(self, on):
        if on:
            if self.store.project.roi.mode != 'polygon':
                self.inspector.pages['roi']._mode('polygon')
            self.viewarea.active.set_kind('top')
        for p in self.viewarea.panes:
            p.plan.edit_polygon = on
            p.plan.update()
        self.sync_view_buttons()

    def set_target_suggestions(self, sugg):
        if 'targets' in self._page_widgets:
            self._page_widgets['targets'].plan.suggestions = sugg
            self._page_widgets['targets'].plan.update()
        for p in self.viewarea.panes:
            p.plan.suggestions = sugg
            p.plan.update()

    # ------------------------------------------------------------------ menus for views
    def camera_menu(self, uids, gpos):
        st = self.store
        cams = [st.cam(u) for u in uids if st.cam(u)]
        if not cams:
            return
        c = cams[0]
        a = self.actions_by_id
        m = QMenu(self)
        m.addAction(a['frame_sel'])
        m.addAction(a['isolate'])
        m.addAction(a['hide'])
        m.addAction(a['lock'])
        m.addSeparator()
        m.addAction(a['mirror'])
        if c.link_uid and st.cam(c.link_uid):
            m.addAction(f'Unlink from {st.cam(c.link_uid).tag}', lambda: ops.toggle_link(self.ctx, c.uid))
        else:
            m.addAction('Link with mirror camera', lambda: ops.toggle_link(self.ctx, c.uid))
        m.addAction(a['duplicate'])
        m.addSeparator()
        aim = m.addMenu('Aim')
        aim.addActions([a['aim_normal'], a['aim_level']])
        lm = m.addMenu('Change lens')
        self._fill_lens_menu(lm, False)
        pm = m.addMenu('Assign GMSL port')
        p = st.project
        for k in range(1, p.ecu.ports + 1):
            other = next((x.tag for x in p.variant.cameras if x.port == k and x.uid != c.uid), '')
            act = pm.addAction(f'Port {k}' + (f' (swap with {other})' if other else ''), lambda k=k: ops.assign_port(self.ctx, c.uid, k))
            act.setCheckable(True)
            act.setChecked(c.port == k)
        m.addAction(a['snap_surface'])
        m.addSeparator()
        m.addAction(a['look_through'])
        m.addAction(a['copy_pose'])
        m.addAction(a['edit_module'])
        m.addSeparator()
        m.addAction(a['delete'])
        m.exec(gpos)

    def ground_menu(self, g, gpos):
        m = QMenu(self)
        st = self.store
        if g is not None:
            o = st.project.origin_offset()
            hdr = m.addAction(f'Ground {g[0] - o[0]:.3f}, {g[1] - o[1]:.3f} m')
            hdr.setEnabled(False)
            if st.sel_cams:
                m.addAction('Aim selected here', lambda: [ops.aim_at(self.ctx, u, g) for u in list(st.sel_cams)])
            m.addAction('Add pole here', lambda: self.add_pole(g))
            m.addAction('Add target here', lambda: self._add_target('checkerboard', g))
            m.addSeparator()
        m.addAction(self.actions_by_id['frame_all'])
        m.addAction(self.actions_by_id['add_cam'])
        m.addAction(self.actions_by_id['roi'])
        m.exec(gpos)

    def object_menu(self, kind, ident, gpos):
        m = QMenu(self)
        a = self.actions_by_id
        if kind in ('vehicle-part', 'group') and ident in ('vehicle', 'wheels', 'zones'):
            m.addActions([a['veh_replace'], a['veh_align'], a['veh_wheels'], a['spec']])
        elif ident == 'photos':
            m.addActions([a['photo_load'], a['photo_cal'], a['photo_pane']])
        elif kind == 'group' and ident == 'cameras':
            m.addActions([a['add_cam'], a['sel_all'], a['unhide']])
            pm = m.addMenu('Presets')
            pm.addActions([a['preset_4'], a['preset_6'], a['preset_6f'], a['preset_8']])
        elif kind in ('group',) and ident == 'targets' or kind == 'target':
            m.addActions([a['targets'], a['target_auto'], a['target_check'], a['target_export']])
            if kind == 'target':
                m.addSeparator()
                m.addAction('Delete target', lambda: self._delete_target(ident))
        elif kind == 'roi':
            m.addActions([a['roi'], a['thresholds']])
        elif kind in ('obstacle', 'group2'):
            m.addAction(a['obs_add'])
            if kind == 'obstacle':
                m.addAction('Delete obstacle', lambda: self.inspector.pages['obstacle']._delete())
        elif kind == 'ecu':
            m.addAction(a['assign_port'])
        elif kind == 'sensor' or (kind == 'group' and ident == 'sensors'):
            m.addActions([a['add_lidar'], a['add_radar']])
            if kind == 'sensor':
                from .sensorview import delete_sensors
                m.addSeparator()
                m.addAction('Delete sensor', lambda: delete_sensors(self.ctx, [ident]))
        elif kind == 'rig' or (kind == 'group' and ident == 'rig'):
            m.addActions([a['rig_roof_rack'], a['rig_sensor_mast'], a['rig_enclosure'], a['rig_bar']])
            if kind == 'rig':
                from .rigview import delete_parts
                m.addSeparator()
                m.addAction('Delete rig part', lambda: delete_parts(self.ctx, [ident]))
        else:
            return
        m.exec(gpos)

    def _fill_sensor_menu(self, menu, kind):
        """Vehicle ▸ Add lidar / Add radar: every model of that kind in the project's library."""
        from .sensorview import add_sensor
        menu.clear()
        for mid, m in self.store.project.sensor_models.items():
            if m.kind == kind:
                act = menu.addAction(f'{m.vendor} {m.name}'.strip(), lambda mid=mid: add_sensor(self.ctx, kind, mid))
                act.setToolTip(m.summary())
        menu.setToolTipsVisible(True)

    def _fill_lens_menu(self, menu, modules=False):
        menu.clear()
        st = self.store
        p = st.project
        c = st.selected
        if modules:
            for mid, mo in p.modules.items():
                act = menu.addAction(mo.name, lambda mid=mid: self.inspector.pages['camera']._module(self.inspector.pages['camera'].module.findData(mid)))
                act.setCheckable(True)
                act.setChecked(bool(c and c.module_id == mid))
                act.setEnabled(bool(c))
            return
        for lid, l in p.lenses.items():
            mid = l.module_id or (c.module_id if c else next(iter(p.modules)))
            act = menu.addAction(f'{l.name} · {l.label}' if l.label else l.name, lambda mid=mid, lid=lid: ops.set_lens(self.ctx, list(st.sel_cams), mid, lid))
            act.setCheckable(True)
            act.setChecked(bool(c and c.lens_id == lid))
            act.setEnabled(bool(st.sel_cams))
        menu.addSeparator()
        menu.addAction('Edit module…', lambda: self.edit_module(c.module_id, c.lens_id) if c else None)

    def _fill_recent(self):
        self.recent_menu.clear()
        for path in self.recent_files():
            self.recent_menu.addAction(path, lambda p=path: self.open_project(p))
        if not self.recent_files():
            a = self.recent_menu.addAction('No recent projects')
            a.setEnabled(False)

    def _fill_variants(self):
        self.var_menu.clear()
        p = self.store.project
        for v in p.variants:
            a = self.var_menu.addAction(v.name, lambda u=v.uid: self.switch_variant(u))
            a.setCheckable(True)
            a.setChecked(v.uid == p.active_uid)

    def _fill_window_menu(self):
        w = self.win_menu
        w.clear()
        names = {'outliner': 'Outliner', 'library': 'Library', 'inspector': 'Inspector', 'camview': 'Camera view',
                 'bottom': 'Analysis panels', 'assistant': 'Assistant'}
        for k, d in self.docks.items():
            a = w.addAction(names[k])
            a.setCheckable(True)
            a.setChecked(d.isVisible())
            a.toggled.connect(lambda on, d=d: d.setVisible(on))
        w.addSeparator()
        for key, title in (('coverage', 'Coverage'), ('cameras', 'Cameras'), ('tolerance', 'Tolerance'), ('issues', 'Issues'),
                           ('history', 'History'), ('log', 'Log')):
            w.addAction(title, lambda k=key: (self.docks['bottom'].show(), self.bottom.show_tab(k)))
        w.addSeparator()
        ws = w.addMenu('Workspaces')
        ws.addActions([self.actions_by_id['ws_placement'], self.actions_by_id['ws_analysis'], self.actions_by_id['ws_review'],
                       self.actions_by_id['ws_save']])
        for name in self.ctx.settings.value('workspaces/names', [], type=list) or []:
            ws.addAction(name, lambda n=name: self.workspace(n))
        w.addAction(self.actions_by_id['ws_reset'])

    def workspace(self, name):
        if name == 'placement':
            self.set_page('viewports')
            for k, d in self.docks.items():
                if k != 'assistant':
                    d.show()
            self.set_split('single')
            self.set_view('persp')
        elif name == 'analysis':
            self.set_page('viewports')
            for k, d in self.docks.items():
                if k != 'assistant':
                    d.show()
            self.set_split('quad')
            self.bottom.show_tab('coverage')
        elif name == 'review':
            self.set_page('compare')
        else:
            st = self.ctx.settings.value(f'workspaces/{name}')
            if st is not None:
                self.restoreState(st)

    def save_workspace(self):
        from .dialogs.misc import ask_text
        name = ask_text(self, 'Save workspace', 'Name', 'My workspace')
        if not name:
            return
        names = self.ctx.settings.value('workspaces/names', [], type=list) or []
        if name not in names:
            names.append(name)
        self.ctx.settings.setValue('workspaces/names', names)
        self.ctx.settings.setValue(f'workspaces/{name}', self.saveState())
        self.msg(f'Saved workspace {name}')

    # ------------------------------------------------------------------ edit commands
    def delete_selection(self):
        st = self.store
        if st.sel_cams:
            ops.delete_cameras(self.ctx, list(st.sel_cams))
        elif st.sel_obj and st.sel_obj[0] == 'target':
            self._delete_target(st.sel_obj[1])
        elif st.sel_obj and st.sel_obj[0] == 'obstacle':
            self.inspector.pages['obstacle']._delete()
        elif st.sel_obj and st.sel_obj[0] == 'rig' and len(st.sel_obj) > 1:
            from .rigview import delete_parts
            delete_parts(self.ctx, [st.sel_obj[1]])
        elif st.sel_obj and st.sel_obj[0] == 'sensor' and len(st.sel_obj) > 1:
            from .sensorview import delete_sensors
            delete_sensors(self.ctx, [st.sel_obj[1]])

    def _delete_target(self, uid):
        self.store.edit('targets', lambda p: setattr(p, 'targets', [t for t in p.targets if t.uid != uid]), 'Deleted target')
        self.store.select_obj(('targets',))

    def _add_target(self, kind, at=None):
        from ..core.project import Target
        p = self.store.project
        st = p.target_settings
        n = len(p.targets) + 1
        x, y = (at[0], at[1]) if at is not None else (p.vehicle.measured().get('front_x', 3) + 1.0, 0.0)

        def fn(pj):
            pj.targets.append(Target(name=f'T{n}', x=round(float(x), 3), y=round(float(y), 3), pattern=kind, rows=st.rows,
                                     cols=st.cols, square_mm=st.square_mm))
        self.store.edit('targets', fn, f'Added target T{n}')
        self.ctx.set_overlay('targets', True)

    def add_pole(self, at=None):
        p = self.store.project
        n = len(p.obstacles) + 1
        x, y = (at[0], at[1]) if at is not None else (p.vehicle.measured().get('front_x', 3) + 0.8, 1.0)
        self.store.edit('obstacles', lambda pj: pj.obstacles.append(Obstacle(name=f'Pole {n}', x=round(float(x), 3), y=round(float(y), 3), height=1.0)),
                        'Added pole')

    def copy_pose(self):
        c = self.store.selected
        if c:
            QGuiApplication.clipboard().setText(ops.pose_json(c, self.store.project))
            self.msg(f'Copied {c.tag} pose as JSON')

    def assign_port_dialog(self):
        c = self.store.selected
        if not c:
            self.msg('Select a camera first.')
            return
        self.store.select_cams([c.uid])
        self.docks['inspector'].show()
        pg = self.inspector.pages['camera']
        pg.s_link.set_expanded(True)
        pg.port.showPopup()

    def add_from_library(self):
        self.docks['library'].show()
        self.library.kind.setCurrentIndex(0)
        self.ctx.set_tool('place')
        mid, lid = self.ctx.place_lens
        self.msg(f'Click the body to place {self.store.project.lenses[lid].name}. Alt-click places a mirrored pair.', 5000)

    def auto_glass(self):
        from .dialogs.import_wizard import detect_glass
        m = self.store.project.vehicle.mesh
        if m is None:
            return

        def fn(p):
            detect_glass(p.vehicle.mesh)
        self.store.edit('vehicle', fn, 'Auto-detected glass')

    def clear_zones(self):
        if self.store.project.vehicle.mesh is None:
            self.msg('There is no vehicle model.')
            return

        def fn(p):
            mesh = p.vehicle.mesh
            mesh.zones = []
            mesh.face_zone = np.full(mesh.n_tris, -1, np.int16)
        self.store.edit('vehicle', fn, 'Cleared mount zones')

    # ------------------------------------------------------------------ variants
    def new_variant(self):
        n = len(self.store.project.variants) + 1

        def fn(p):
            v = Variant(name=f'Variant {n}')
            p.variants.append(v)
            p.active_uid = v.uid
        self.store.edit('variants', fn, f'New variant {n}')
        self._fill_variant_combo()

    def duplicate_variant(self):
        import copy
        cur = self.store.project.variant

        def fn(p):
            v = Variant.from_dict(cur.to_dict())
            v.uid = new_uid()
            base = cur.name
            import re
            mo = re.match(r'(.*?)(\d+)$', base)
            v.name = f'{mo.group(1)}{int(mo.group(2)) + 1}' if mo else f'{base} copy'
            uid_map = {}
            for c in v.cameras:
                nu = new_uid()
                uid_map[c.uid] = nu
                c.uid = nu
            for c in v.cameras:
                c.link_uid = uid_map.get(c.link_uid, '')
            v.seams = {}
            p.variants.append(v)
            p.active_uid = v.uid
        self.store.edit('variants', fn, f'Duplicated variant {cur.name}')
        self.store.select_cams([self.store.variant.cameras[0].uid] if self.store.variant.cameras else [])
        self._fill_variant_combo()

    def delete_variant(self):
        p = self.store.project
        if len(p.variants) <= 1:
            self.msg('A project needs at least one variant.')
            return
        from .dialogs.misc import confirm
        v = p.variant
        if not confirm(self, f'Delete variant {v.name}', f'The variant and its {len(v.cameras)} cameras are removed. '
                                                          'You can undo this from History.', 'Delete variant'):
            return

        def fn(pj):
            pj.variants = [x for x in pj.variants if x.uid != v.uid]
            pj.active_uid = pj.variants[0].uid
        self.store.edit('variants', fn, f'Deleted variant {v.name}')
        self._fill_variant_combo()

    def apply_preset(self, n):
        p = self.store.project
        m = p.vehicle.mesh
        if m is None:
            self.msg('Import a vehicle first.')
            return
        from ..core.bodygen import template_specs, tigor_spec
        if p.vehicle.template == 'tigor' and n in (4, 6, '6f'):
            vs = samples.tigor_variants(m)
            v = {6: vs[0], 4: vs[1], '6f': vs[3]}[n]
            cams = v.cameras
        else:
            spec = template_specs().get(p.vehicle.template) or _spec_from_vehicle(p.vehicle)
            nn = 6 if n == '6f' else n
            cams = samples.template_layout(m, spec, nn)
        vv = Variant(name=f'{len(cams)}-cam preset')
        vv.cameras = cams
        vv.sensors = [Sensor.from_dict(x.to_dict()) for x in self.store.variant.sensors]   # the lidars and radars stay
        samples.link_pairs(vv)

        def fn(pj):
            pj.variants.append(vv)
            pj.active_uid = vv.uid
        self.store.edit('variants', fn, f'Added preset {vv.name}')
        self.store.select_cams([cams[0].uid])
        self._fill_variant_combo()

    # ------------------------------------------------------------------ files
    def recent_files(self):
        r = self.ctx.settings.value('recent', [], type=list) or []
        return [x for x in r if os.path.exists(x)]

    def add_recent(self, path):
        r = [x for x in (self.ctx.settings.value('recent', [], type=list) or []) if x != path]
        r.insert(0, path)
        self.ctx.settings.setValue('recent', r[:12])

    def _maybe_save(self):
        if self.store.undo.isClean():
            return True
        r = QMessageBox.question(self, 'Unsaved changes', 'Save changes to the current project first?',
                                 QMessageBox.StandardButton.Save | QMessageBox.StandardButton.Discard | QMessageBox.StandardButton.Cancel,
                                 QMessageBox.StandardButton.Save)
        if r == QMessageBox.StandardButton.Cancel:
            return False
        if r == QMessageBox.StandardButton.Save:
            return self.save()
        return True

    def new_project(self):
        if not self._maybe_save():
            return
        self.store.set_project(samples.blank_project())
        self.msg('New project. Import a vehicle model (Ctrl+I) or start from a template.', 6000)

    def new_from_template(self, key):
        if not self._maybe_save():
            return
        self.store.set_project(samples.template_project(key))

    def open_project(self, path=None):
        if not self._maybe_save():          # every route here (dialog, recent menu, start window, drop) replaces the project
            return
        if not path:
            path, _ = QFileDialog.getOpenFileName(self, 'Open project', self.ctx.settings.value('dir/projects', os.path.expanduser('~')),
                                                  'Vantage project (*.vantage)')
            if not path:
                return
        try:
            p = Project.load(path)
        except Exception as e:
            QMessageBox.critical(self, 'Could not open project', f'{os.path.basename(path)}: {e}')
            return
        self.ctx.settings.setValue('dir/projects', os.path.dirname(path))
        self.store.set_project(p)
        self.add_recent(path)
        self.msg(f'Opened {os.path.basename(path)}')

    def thumbnail_png(self) -> bytes:
        from .planview import PlanView
        try:
            pv = PlanView(self.ctx, 'coverage', interactive=False)
            pv.legend = False
            pv.show_tooltip = False
            pv.resize(240, 180)
            pv.fit()
            img = pv.grab().toImage()
            pv.deleteLater()
            buf = QBuffer()
            buf.open(QIODevice.OpenModeFlag.WriteOnly)
            img.save(buf, 'PNG')
            return bytes(buf.data())
        except Exception:
            return b''

    def save(self):
        p = self.store.project
        if not p.path:
            return self.save_as()
        try:
            p.save(p.path, self.thumbnail_png())
        except Exception as e:
            QMessageBox.critical(self, 'Could not save', str(e))
            return False
        self.store.undo.setClean()
        self.add_recent(p.path)
        self.set_project_meta()
        self.msg(f'Saved {os.path.basename(p.path)}')
        return True

    def save_as(self):
        p = self.store.project
        start = p.path or os.path.join(self.ctx.settings.value('dir/projects', os.path.expanduser('~')), f'{p.name}.vantage')
        path, _ = QFileDialog.getSaveFileName(self, 'Save project as', start, 'Vantage project (*.vantage)')
        if not path:
            return False
        if not path.endswith('.vantage'):
            path += '.vantage'
        p.path = path
        p.name = os.path.splitext(os.path.basename(path))[0]
        self.ctx.settings.setValue('dir/projects', os.path.dirname(path))
        return self.save()

    def import_layout(self):
        path, _ = QFileDialog.getOpenFileName(self, 'Import layout', os.path.expanduser('~'), 'Layouts (*.csv *.json)')
        if not path:
            return
        from .dialogs.misc import import_layout_file
        try:
            cams = import_layout_file(path, self.store.project)
        except Exception as e:
            QMessageBox.critical(self, 'Could not import layout', str(e))
            return
        v = Variant(name=os.path.splitext(os.path.basename(path))[0])
        v.cameras = cams
        samples.link_pairs(v)

        def fn(pj):
            pj.variants.append(v)
            pj.active_uid = v.uid
        self.store.edit('variants', fn, f'Imported layout {v.name} ({len(cams)} cameras)')
        self._fill_variant_combo()

    def project_settings(self):
        from .dialogs.misc import ProjectSettingsDialog
        ProjectSettingsDialog(self.ctx, self).exec()

    def toggle_assistant(self):
        d = self.docks['assistant']
        if d.isVisible() and self.assistant_panel.composer.edit.hasFocus():
            d.hide()
            return
        self.show_assistant()

    def show_assistant(self):
        d = self.docks['assistant']
        if not d.isVisible():
            d.show()
            self.resizeDocks([d], [420], Qt.Orientation.Horizontal)
        d.raise_()
        self.assistant_panel.composer.edit.setFocus()

    def ask_assistant(self, text):
        """Open the assistant and ask (from the Issues list, the command palette)."""
        self.show_assistant()
        self.assistant_panel.ask(text)

    def preferences(self):
        from .dialogs.misc import PreferencesDialog
        PreferencesDialog(self.ctx, self).exec()

    def show_start(self):
        from .dialogs.start import StartWindow
        w = StartWindow(self.ctx.settings, parent=self)
        if w.exec() and w.choice:
            kind, val = w.choice
            if kind == 'open':
                self.open_project(val)
            elif kind == 'template':
                self.new_from_template(val)
            elif kind == 'new':
                self.new_project()
            elif kind == 'model':
                self.new_project()
                self.import_vehicle(path=val)

    # ------------------------------------------------------------------ vehicle / modules / dialogs
    def import_vehicle(self, path=None, step=0, reuse=False, paint=None):
        from .dialogs.import_wizard import ImportWizard
        wiz = ImportWizard(self.ctx, self, path=path, step=step, reuse=reuse, paint=paint)
        wiz.exec()

    def spec_check(self):
        from .dialogs.misc import SpecDialog
        SpecDialog(self.ctx, self).exec()

    def edit_module(self, mid, lid):
        if not mid or not lid:
            self.msg('Select a camera or a library lens first.')
            return
        from .dialogs.module_editor import ModuleEditor
        ModuleEditor(self.ctx, mid, lid, self).exec()

    # ------------------------------------------------------------------ renders
    def _render_target(self):
        page = self.page_name()
        if page == 'camviews' and 'camviews' in self._page_widgets:
            return self._page_widgets['camviews'].bowl
        pane = self.viewarea.active
        return pane.vp3d if pane is not None else None

    def render_image(self, path=None, scale=None, transparent=False):
        vp = self._render_target()
        if vp is None:
            self.msg('No 3D view to render.')
            return None
        if scale is None:
            from PyQt6.QtWidgets import QInputDialog
            items = [f'{k}× · {vp.width() * k} × {vp.height() * k} px' for k in (1, 2, 3, 4)]
            ch, ok = QInputDialog.getItem(self, 'Render image', 'Size', items, 1, False)
            if not ok:
                return None
            scale = items.index(ch) + 1
        if not path:
            path, _ = QFileDialog.getSaveFileName(self, 'Render image', f'{self.store.project.name}-3d.png',
                                                  'PNG (*.png);;JPEG (*.jpg)')
        if not path:
            return None
        img = vp.render_image(scale, transparent=transparent)
        if not img.save(path):
            QMessageBox.critical(self, 'Render image', f'Could not write {path}')
            return None
        self.msg(f'Rendered {img.width()} × {img.height()} px to {path}', 6000)
        return path

    def turntable(self, path=None, frames=None, scale=1.0, fps=30, progress=None):
        """Orbit the active 3D view once around its target and write the frames as an MP4 (with ffmpeg) or PNGs."""
        import shutil
        import subprocess
        import tempfile
        vp = self._render_target()
        if vp is None:
            return None
        if not path:
            path, _ = QFileDialog.getSaveFileName(self, 'Turntable video', f'{self.store.project.name}-turntable.mp4',
                                                  'MP4 video (*.mp4);;PNG frames (folder) (*)')
        if not path:
            return None
        n = int(frames or 8 * fps)
        cam0 = vp.cam.copy()
        vp.cam.custom = None
        tmp = tempfile.mkdtemp(prefix='vantage-turntable-', dir=os.path.dirname(os.path.abspath(path)))
        try:
            for i in range(n):
                vp.cam.az = cam0.az + 360.0 * i / n
                img = vp.render_image(scale)
                w, h = img.width() // 2 * 2, img.height() // 2 * 2       # even sizes for H.264
                img.copy(0, 0, w, h).save(os.path.join(tmp, f'f{i:05d}.png'))
                if progress:
                    progress((i + 1) / n)
                QApplication.processEvents()
            ff = shutil.which('ffmpeg')
            if path.lower().endswith('.mp4') and ff:
                r = subprocess.run([ff, '-y', '-loglevel', 'error', '-framerate', str(fps), '-i', os.path.join(tmp, 'f%05d.png'),
                                    '-c:v', 'libx264', '-pix_fmt', 'yuv420p', '-crf', '18', '-movflags', '+faststart', path],
                                   capture_output=True, text=True)
                if r.returncode != 0:
                    raise RuntimeError(r.stderr.strip()[-300:])
                out = path
            else:
                out = os.path.splitext(path)[0] + '-frames'
                if os.path.exists(out):
                    shutil.rmtree(out)
                shutil.move(tmp, out)
                tmp = None
        except Exception as e:
            QMessageBox.critical(self, 'Turntable video', str(e))
            return None
        finally:
            vp.cam = cam0
            vp.update()
            if tmp and os.path.isdir(tmp):
                shutil.rmtree(tmp, ignore_errors=True)
        self.msg(f'Wrote {out} ({n} frames)', 6000)
        return out

    # ------------------------------------------------------------------ unit lens calibration
    def lens_calibration(self, uid=None):
        from .dialogs.lens_calib_dialog import LensCalibDialog
        if not self.store.variant.cameras:
            self.msg('This variant has no cameras.')
            return None
        d = LensCalibDialog(self.ctx, uid, self)
        d.show()
        self._calib_dialog = d
        return d

    def import_calibrations(self, paths=None):
        from .dialogs.lens_calib_dialog import ImportCalibDialog
        if not paths:
            paths, _ = QFileDialog.getOpenFileNames(self, 'Import unit calibrations', os.path.expanduser('~'),
                                                    'Calibration files (*.yaml *.yml *.xml *.json);;All files (*)')
        if not paths:
            return None
        d = ImportCalibDialog(self.ctx, paths, self)
        d.show()
        self._import_calib_dialog = d
        return d

    def export_calibrations(self, path=None):
        from ..core.calib import to_json
        cams = [c for c in self.store.variant.cameras if c.calib]
        if not cams:
            self.msg('No camera in this variant has a unit calibration.')
            return None
        if not path:
            path, _ = QFileDialog.getSaveFileName(self, 'Export unit calibrations', 'unit-calibrations.json', 'JSON (*.json)')
        if not path:
            return None
        with open(path, 'w', encoding='utf-8') as fh:
            fh.write(to_json(self.store.project, cams))
        self.msg(f'Wrote {len(cams)} unit calibration(s) to {path}')
        return path

    def clear_calibration(self):
        from .dialogs.lens_calib_dialog import apply_calibrations
        cams = [c for c in self.store.selected_cams() if c.calib] if self.store.selected else []
        if not cams:
            self.msg('Select a camera that has a unit calibration.')
            return
        apply_calibrations(self.ctx, {c.uid: {} for c in cams},
                           f'Cleared the unit calibration of {", ".join(c.tag for c in cams)}')
        self.msg(f'{", ".join(c.tag for c in cams)} use the library lens again')

    def import_module(self):
        from .dialogs.module_editor import ModuleEditor
        p = self.store.project
        mid = next(iter(p.modules))
        lid = next(iter(p.lenses))
        dlg = ModuleEditor(self.ctx, mid, lid, self, start_import=True)
        dlg.exec()

    def delete_lens(self, lid):
        p = self.store.project
        used = [c.tag for v in p.variants for c in v.cameras if c.lens_id == lid]
        if used:
            self.msg(f'In use by {", ".join(used[:6])}. Change those cameras first.')
            return
        self.store.edit('library', lambda pj: pj.lenses.pop(lid, None), 'Deleted lens')

    def edit_route(self, uid):
        from .dialogs.misc import RouteDialog
        RouteDialog(self.ctx, uid, self).exec()

    def custom_frame(self):
        from .dialogs.misc import CustomFrameDialog
        CustomFrameDialog(self.ctx, self).exec()

    def export_dialog(self, preselect=None):
        from .dialogs.export_dialog import ExportDialog
        ExportDialog(self.ctx, self, preselect=preselect).exec()

    def command_palette(self, prefix=''):
        from .dialogs.palette import CommandPalette
        CommandPalette(self.ctx, self, prefix).exec()

    def shortcuts(self):
        from .dialogs.misc import ShortcutsDialog
        ShortcutsDialog(self, self).exec()

    def help_topic(self, topic):
        from .dialogs.misc import HelpDialog
        HelpDialog(topic, self).exec()

    def about(self):
        from .dialogs.misc import AboutDialog
        AboutDialog(self).exec()

    # ------------------------------------------------------------------ photos
    def load_photos(self):
        from ..core import photos as ph
        path = QFileDialog.getExistingDirectory(self, 'Choose a folder of photos (Cancel to pick a .zip)', os.path.expanduser('~'))
        if not path:
            path, _ = QFileDialog.getOpenFileName(self, 'Choose a .zip of photos', os.path.expanduser('~'), 'Zip archives (*.zip)')
        if not path:
            return
        try:
            names = ph.list_frames(path)
        except Exception as e:
            QMessageBox.critical(self, 'Could not read photos', str(e))
            return
        if not names:
            QMessageBox.warning(self, 'No photos', 'No images (.jpg .png .bmp .webp .tif) were found there.')
            return
        data = ph.read_frame_bytes(path, names[0])
        img = QImage.fromData(data) if data else QImage()
        from ..core.project import PhotoSet, PhotoFrame

        def fn(p):
            old = p.photos
            ps = PhotoSet(source=path, width=img.width(), height=img.height(), f_px=1.2 * max(img.width(), 1), cx=img.width() / 2,
                          cy=img.height() / 2, method='', landmarks=dict(old.landmarks) or {k: list(v) for k, v in p.vehicle.landmarks.items()})
            same = [f.name for f in old.frames] == names
            ps.frames = old.frames if same and old.frames else [PhotoFrame(n) for n in names]
            if same:
                ps.f_px, ps.cx, ps.cy, ps.k1, ps.rms, ps.method, ps.solved_at = old.f_px, old.cx, old.cy, old.k1, old.rms, old.method, old.solved_at
            p.photos = ps
        self.store.edit('photos', fn, f'Loaded {len(names)} reference photos')
        self._photo_cache.clear()
        self.show_photo_page(True)

    def photo_image(self, k) -> QImage | None:
        from ..core import photos as ph
        ps = self.store.project.photos
        if not ps.frames or not (0 <= k < len(ps.frames)) or not ps.source:
            return None
        key = (ps.source, ps.frames[k].name)
        if key in self._photo_cache:
            return self._photo_cache[key]
        data = ph.read_frame_bytes(ps.source, ps.frames[k].name)
        img = QImage.fromData(data) if data else QImage()
        if len(self._photo_cache) > 40:
            self._photo_cache.clear()
        self._photo_cache[key] = img if not img.isNull() else None
        return self._photo_cache[key]

    def calibrate_photos(self):
        from ..core import photos as ph
        ps = self.store.project.photos
        if not ps.frames:
            self.msg('Load a photo spin set first.')
            return
        img = self.photo_image(ps.current)
        size = (img.width(), img.height()) if img is not None else None

        def fn(p):
            return ph.calibrate(p.photos, size)
        try:
            msg = self.store.edit('photos', fn, 'Calibrated reference photos')
        except Exception as e:
            QMessageBox.critical(self, 'Calibration failed', str(e))
            return
        self.msg(msg, 6000)

    def show_photo_page(self, on):
        if on:
            self.set_page('photo')
        elif self.page_name() == 'photo':
            self.set_page('viewports')

    def add_landmark(self):
        if self.page_name() != 'photo':
            self.set_page('photo')
        self.page('photo').start_landmark_pick()

    def load_cases(self):
        from .dialogs.loadcases_dialog import LoadCasesDialog
        d = LoadCasesDialog(self.ctx, self)
        d.show()
        self._load_dialog = d
        return d

    # ------------------------------------------------------------------ jobs (tolerance / optimiser)
    def run_tolerance(self, settings):
        from ..core.tolerance import run_monte_carlo
        from ..core.analysis import Engine
        if 'tolerance' in self._jobs:
            return
        project = self.store.project.snapshot()
        eng = self._shared_engine()
        cancel = threading.Event()
        self._jobs['tolerance'] = {'cancel': cancel, 'state': ('tolerance', 0.0, '')}
        self._show_job('Tolerance study', 0.0)

        def run():
            try:
                res = run_monte_carlo(eng, project, settings, progress=lambda f: self.jobs.progress.emit('tolerance', f, ''),
                                      cancel=cancel.is_set)
                self.jobs.done.emit('tolerance', res)
            except Exception as e:
                self.jobs.failed.emit('tolerance', f'{e}\n{traceback.format_exc()}')
        threading.Thread(target=run, daemon=True).start()

    def run_optimiser(self, settings, on_progress):
        from ..core.optimiser import Optimiser
        if 'optimiser' in self._jobs:
            return None
        project = self.store.project.snapshot()
        eng = self._shared_engine()
        opt = Optimiser(eng, project, settings)
        cancel = threading.Event()
        self._jobs['optimiser'] = {'cancel': cancel, 'state': ('optimiser', 0.0, ''), 'cb': on_progress, 'opt': opt}
        self._show_job('Optimiser', 0.0)

        def prog(o):
            self.jobs.progress.emit('optimiser', o.generation / max(settings.generations, 1), f'generation {o.generation} / {settings.generations}')

        def run():
            try:
                opt.run(progress=prog, cancel=cancel.is_set)
                self.jobs.done.emit('optimiser', opt)
            except Exception as e:
                self.jobs.failed.emit('optimiser', f'{e}\n{traceback.format_exc()}')
        threading.Thread(target=run, daemon=True).start()
        return opt

    def run_job(self, kind, title, fn, on_done, on_failed=None):
        """A background job with the job bar and Cancel: fn(progress, cancel) -> result runs on a thread, then
        on_done(result) here (on_failed(message) when it raised)."""
        if kind in self._jobs:
            return False
        cancel = threading.Event()
        self._jobs[kind] = {'cancel': cancel, 'state': (kind, 0.0, ''), 'title': title, 'on_done': on_done,
                            'on_failed': on_failed}
        self._show_job(title, 0.0)

        def run():
            try:
                res = fn(lambda f, text='': self.jobs.progress.emit(kind, f, text), cancel.is_set)
                self.jobs.done.emit(kind, res)
            except Exception as e:
                self.jobs.failed.emit(kind, f'{e}\n{traceback.format_exc()}')
        threading.Thread(target=run, daemon=True).start()
        return True

    def _shared_engine(self):
        e = self.ctx.analysis.engine.share()
        e.vehicle_geom(self.store.project)      # built here (or found), so the worker thread starts on it
        return e

    def _show_job(self, name, frac, text=''):
        self.job_label.setText(f'{name}' + (f' · {text}' if text else ''))
        self.job_bar.setValue(int(frac * 1000))
        self.job_pct.setText(f'{frac * 100:.0f} %')
        for w in (self.job_label, self.job_bar, self.job_pct, self.job_cancel):
            w.show()
        self.progress_line.show()
        self.progress_line.raise_()
        self._place_progress()
        self.progress_line.setValue(int(frac * 1000))
        self._update_state()

    def _job_progress(self, kind, frac, text):
        j = self._jobs.get(kind)
        if not j:
            return
        j['state'] = (kind, frac, text)
        self._show_job(j.get('title', kind.capitalize()), frac, text)
        cb = j.get('cb')
        if cb:
            cb(j.get('opt'))

    def _job_done(self, kind, result):
        job = self._jobs.pop(kind, None)
        self._hide_job()
        if job and job.get('on_done'):
            job['on_done'](result)
        elif kind == 'tolerance':
            self.page('tolerance').show_result(result)
            self.bottom.tolerance.show_result(result)
            n = len(result.blind)
            if getattr(result, 'cancelled', False):
                total = getattr(getattr(result, 'settings', None), 'samples', None) or '?'
                text = f'Tolerance study cancelled after {n} of {total} samples'
            else:
                text = f'Tolerance study finished · {n} samples in {result.seconds:.1f} s'
            self.store.project.add_log('tolerance', f'{text} · max {result.blind.max():.3f} m²')
            self.msg(text, 6000)
        elif kind == 'optimiser':
            self.page('optimiser').finished(result)
            self.store.project.add_log('optimiser', f'{result.evals} evaluations · {len(result.front)} on the Pareto front')
            self.msg(f'Optimiser finished · {result.evals} layouts evaluated', 6000)
        self._update_state()

    def _job_failed(self, kind, m):
        job = self._jobs.pop(kind, None)
        self._hide_job()
        title = job.get('title', kind.capitalize()) if job else kind.capitalize()
        if job and job.get('on_failed'):
            job['on_failed'](m)
        if job and job.get('cancel') is not None and job['cancel'].is_set():
            self.msg(f'{title} cancelled')          # a cancel before the first sample is not a failure
        else:
            print(m)
            QMessageBox.critical(self, f'{title} failed', m.splitlines()[0])
        self._update_state()

    def _hide_job(self):
        if not self._jobs:
            for w in (self.job_label, self.job_bar, self.job_pct, self.job_cancel):
                w.hide()
            self.progress_line.hide()

    def cancel_jobs(self):
        for j in self._jobs.values():
            j['cancel'].set()


def _dot_icon(color):
    from PyQt6.QtGui import QPainter, QColor
    pm = QPixmap(8, 8)
    pm.fill(Qt.GlobalColor.transparent)
    p = QPainter(pm)
    p.setRenderHint(QPainter.RenderHint.Antialiasing)
    p.setBrush(QColor(color))
    p.setPen(Qt.PenStyle.NoPen)
    p.drawEllipse(1, 1, 6, 6)
    p.end()
    return QIcon(pm)


def _spec_from_vehicle(v):
    from ..core.bodygen import template_specs
    s = template_specs()['sedan']
    m = v.measured()
    s.length = m.get('Length', s.length)
    s.width = m.get('Width', s.width)
    s.height = m.get('Height', s.height)
    s.rear_overhang = -m.get('rear_x', -s.rear_overhang)
    return s
