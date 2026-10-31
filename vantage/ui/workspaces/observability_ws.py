"""Calibrate ▸ Calibration observability: how firmly the planned calibration pins every camera's pose (the boards of
Calibrate ▸ Targets), each camera pair's relative pose, the lidar–camera pairs (the lidar–camera calibration's board
plan) and the radars (the corner reflectors placed here); why a direction stays weak and what moving or adding a board
would gain (Apply carries it out, undoably); a Monte Carlo check of the predictions with the solves that will solve the
real captures; and the camera–lidar fusion density (lidar points per object and per camera pixel). The study runs in
the background (a second or two) and again by itself after edits while the page shows; the Monte Carlo on demand."""
from __future__ import annotations

import json
import math
import threading
import traceback

import numpy as np
from PyQt6.QtCore import QObject, QPointF, QRectF, Qt, QTimer, pyqtSignal
from PyQt6.QtGui import QColor, QFont, QImage, QPainter, QPen, QPolygonF
from PyQt6.QtWidgets import (QAbstractItemView, QFileDialog, QGridLayout, QHBoxLayout, QHeaderView, QLabel,
                             QProgressBar, QPushButton, QScrollArea, QSplitter, QTableWidget, QTableWidgetItem,
                             QTabWidget, QToolTip, QVBoxLayout, QWidget)

from ...core import observability as OB
from ...core import sensor_coverage as SC
from ...core.mathutil import parse_number
from ..theme import SEQ, STATUS, THEME, mono, sans
from ..widgets import fit_width, ROW_H, ScrubSpinBox, StatusLine, ToggleSwitch, button, hline, label
from .recording_ws import Tile

JOB = 'observability'
JOB_MC = 'observability_mc'
AUTO_MAX_S = 3.0                       # runs again by itself after an edit when the last run took less
EMPTY = 'Plan calibration targets in Calibrate ▸ Targets, then Run.'
SEV_WORD = {'good': 'OK', 'warning': 'Warning', 'serious': 'Serious', 'critical': 'Critical'}
# (group, key, label, unit, decimals, step, tip)
FIELDS = (
    ('Corners', 'pixel_sigma', 'Corner noise', 'px', 2, 0.05,
     'How far a detected corner scatters, 1σ per image axis: typical sub-pixel refinement gives 0.1–0.5 px'),
    ('Corners', 'min_square_px', 'Smallest square', 'px', 0, 1.0,
     'A board (an AprilGrid tag) whose squares are narrower than this in the image is not found'),
    ('Boards', 'board_sigma_mm', 'Placement ±', 'mm', 0, 1.0,
     'How far a board laid by hand may lie off its planned place (the installation check\'s joint solve allows the '
     'same)'),
    ('Boards', 'board_sigma_deg', 'Turned ±', '°', 2, 0.1, 'How far a board may be turned off its planned yaw'),
    ('Requirement (1σ)', 'need_deg', 'Angles', '°', 3, 0.01, 'The 1σ every camera angle must reach'),
    ('Requirement (1σ)', 'need_mm', 'Position', 'mm', 1, 0.5, 'The 1σ every camera position must reach'),
    ('Radars', 'radar_range_sigma_m', 'Range noise', 'm', 3, 0.01,
     'A radar\'s range noise on a reflector, 1σ (typical 77 GHz: about 0.1 m)'),
    ('Radars', 'radar_az_sigma_deg', 'Azimuth noise', '°', 2, 0.05,
     'A radar\'s azimuth noise, 1σ (typical 77 GHz: 0.1–0.3° near boresight)'),
    ('Radars', 'reflector_sigma_mm', 'Survey ±', 'mm', 1, 1.0, 'How well the reflectors\' places are surveyed, 1σ'),
    ('Radars', 'radar_need_deg', 'Need yaw', '°', 2, 0.05, 'The 1σ a radar\'s yaw must reach'),
    ('Radars', 'radar_need_mm', 'Need position', 'mm', 0, 1.0, 'The 1σ a radar\'s x and y must reach'),
    ('Fusion and check', 'fusion_at_m', 'Scene at', 'm', 0, 5.0,
     'The image map of lidar points per pixel is taken on a surface this far out (on the ground where it is nearer)'),
    ('Fusion and check', 'mc_trials', 'Monte Carlo trials', '', 0, 5.0,
     'Simulated captures per check: more is tighter and slower (each solves every camera together)'),
)
TOGGLES = (('boards_placed', 'Boards placed by hand', 'Each board may lie off its plan by the placement ±: the joint '
                                                      'solve\'s model. Off: the boards taken as exact'),
           ('suggest', 'Suggest board moves', 'Try moving, turning and adding boards for the weak cameras'),
           ('lidar_pairs', 'Lidar–camera pairs', 'The lidar–camera calibration\'s board plan for every overlapping '
                                                 'camera–lidar pair (Calibrate ▸ Lidar–camera)'),
           ('fusion', 'Camera–lidar fusion', 'Lidar points per object and per camera pixel in every camera\'s view'))
CAM_COLS = (('Camera', ''), ('Boards', 'The boards counted (whole and with squares big enough) and their corners'),
            ('Yaw °', '1σ of the yaw'), ('Pitch °', '1σ of the pitch'), ('Roll °', '1σ of the roll'),
            ('x mm', '1σ along x (forward)'), ('y mm', '1σ along y (left)'), ('z mm', '1σ along z (up)'),
            ('Worst', 'The worst axis in units of the requirement'),
            ('κ', 'Condition number of the information, rotations as the shift they make at the boards'))
REL_COLS = (('Pair', 'Two cameras that count a common board'), ('Through', 'The boards they share'),
            ('Roll °', 'About camera a\'s optical (x) axis'), ('Pitch °', 'About a\'s y axis'),
            ('Yaw °', 'About a\'s z axis'), ('x mm', 'Along a\'s x (forward)'), ('y mm', 'Along a\'s y'),
            ('z mm', 'Along a\'s z'), ('Worst', 'In units of the requirement'))
LIDAR_COLS = (('Pair', 'Camera–lidar'), ('Poses', 'Board poses planned by the lidar–camera calibration'),
              ('Yaw °', 'The lidar\'s yaw, 1σ (vehicle frame)'), ('Pitch °', ''), ('Roll °', ''),
              ('x mm', 'The lidar\'s position, 1σ'), ('y mm', ''), ('z mm', ''),
              ('Worst', 'In units of the lidar–camera tolerances'))
RADAR_COLS = (('Radar', ''), ('Reflectors', 'The reflectors in its field'), ('Yaw °', '1σ'), ('x mm', '1σ'),
              ('y mm', '1σ'), ('Worst', 'In units of the radar requirement'))
REFL_COLS = (('name', 'Name', None), ('x', 'x m', 2), ('y', 'y m', 2), ('z', 'z m', 2), ('edge_mm', 'Edge mm', 0),
             ('rcs', 'RCS m²', 1))
MC_COLS = (('What', ''), ('Axis', ''), ('Predicted', 'The predicted 1σ'),
           ('Joint solve', 'The spread of the installation check\'s joint solve (or the lidar–camera / radar solve)'),
           ('Optimum', 'The same answers carried to the optimum of the least squares (where the joint solve stopped '
                       'short of it)'),
           ('Ratio', 'Empirical over predicted'))


def _table(cols, select=False, stretch=1):
    t = QTableWidget(0, len(cols))
    for c, (h, tip) in enumerate(cols):
        it = QTableWidgetItem(h)
        if tip:
            it.setToolTip(tip)
        t.setHorizontalHeaderItem(c, it)
    t.verticalHeader().setVisible(False)
    t.verticalHeader().setDefaultSectionSize(ROW_H)
    t.horizontalHeader().setFixedHeight(ROW_H)
    t.setShowGrid(False)
    t.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
    if select:
        t.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        t.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
    else:
        t.setSelectionMode(QAbstractItemView.SelectionMode.NoSelection)
    hh = t.horizontalHeader()
    for c in range(len(cols)):
        hh.setSectionResizeMode(c, QHeaderView.ResizeMode.Stretch if c == stretch else
                                QHeaderView.ResizeMode.ResizeToContents)
    return t


def _num(text, tip=None, color=None):
    it = QTableWidgetItem(text)
    it.setFont(mono(12))
    it.setTextAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
    if tip:
        it.setToolTip(tip)
    if color is not None:
        it.setForeground(color)
    return it


def _f(v, nd=2):
    return '—' if v is None or not math.isfinite(float(v)) else f'{float(v):.{nd}f}'


def _times(r):
    return '—' if r is None or not math.isfinite(float(r)) else f'{float(r):.2f}×'


def _tone(sev) -> QColor:
    return QColor(STATUS.get(sev, STATUS['critical']))


# ============================================================================ the plan
class ObsPlan(QWidget):
    """Top view (forward right, left up): the body outline, the boards (shaded by how many cameras count them), each
    camera with its verdict, lines to the boards it counts, the selected camera's suggested moves, the radars and the
    corner reflectors. Click a camera to pick it."""
    picked = pyqtSignal(str)

    def __init__(self):
        super().__init__()
        self.setMinimumSize(320, 260)
        self.setMouseTracking(True)
        self.res = None
        self.project = None
        self.sel = ''
        self.highlight = -1             # index of the hovered suggestion of the selected camera
        self.message = EMPTY
        self._cams = []                 # (screen point, uid, text)
        self._tf = None
        THEME.changed.connect(self._theme_changed)

    def _theme_changed(self, *_):
        self.update()

    def set_data(self, res, project, sel=''):
        self.res, self.project, self.sel = res, project, sel
        self.update()

    def set_selected(self, uid):
        self.sel = uid
        self.update()

    def _bounds(self):
        """The body, the boards, the cameras and the suggested places (reflectors far out are drawn at the edge)."""
        res = self.res
        m = res._model
        pts = [o for o in res.outline] + [b.centre[None, :2] for b in m.boards] + \
              [cm.pos[None, :2] for cm in m.cams]
        for c in res.cameras:
            for s in c['suggestions']:
                if 'x' in s:
                    pts.append(np.array([[s['x'], s['y']]]))
        P = np.vstack(pts) if pts else np.zeros((1, 2))
        return P.min(0) - 0.8, P.max(0) + 0.8

    def w2s(self, x, y):
        k, ox, oy = self._tf
        return QPointF(ox + x * k, oy - y * k)

    def paintEvent(self, e):
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        p.fillRect(self.rect(), THEME.color('viewport'))
        res = self.res
        self._cams = []
        if res is None:
            p.setPen(THEME.color('muted'))
            p.setFont(sans(12))
            p.drawText(self.rect(), Qt.AlignmentFlag.AlignCenter, self.message)
            p.end()
            return
        m = res._model
        lo, hi = self._bounds()
        W, H = self.width(), self.height() - 22
        k = min((W - 30) / max(hi[0] - lo[0], 1e-3), (H - 30) / max(hi[1] - lo[1], 1e-3))
        cx, cy = (lo + hi) / 2
        self._tf = (k, W / 2 - cx * k, 22 + H / 2 + cy * k)
        # body
        body = QColor(THEME.color('body'))
        body.setAlphaF(0.22)
        for o in res.outline:
            poly = QPolygonF([self.w2s(x, y) for x, y in o])
            p.setPen(QPen(THEME.color('strong'), 1))
            p.setBrush(body)
            p.drawPolygon(poly)
        tags = {c['uid']: c for c in res.cameras}
        sel_i = next((i for i, cm in enumerate(m.cams) if cm.cam.uid == self.sel), -1)
        # sight lines
        for o in m.seen:
            mine = o.cam == sel_i
            col = THEME.cam(m.cams[o.cam].cam.slot) if mine else THEME.color('hair')
            p.setPen(QPen(col, 1.4 if mine else 1.0))
            p.drawLine(self.w2s(*m.cams[o.cam].pos[:2]), self.w2s(*m.boards[o.board].centre[:2]))
        # boards
        p.setFont(sans(10))
        for bi, b in enumerate(m.boards):
            n = sum(1 for o in m.seen if o.board == bi)
            fill = QColor(SEQ[2] if n == 1 else SEQ[7]) if n else THEME.color('strong')
            fill.setAlphaF(0.85)
            corners = OB._outline(b.t)
            poly = QPolygonF([self.w2s(x, y) for x, y, _ in corners])
            mine = any(o.cam == sel_i and o.board == bi for o in m.seen)
            p.setPen(QPen(THEME.cam(m.cams[sel_i].cam.slot), 2) if mine else QPen(THEME.color('strong'), 1))
            p.setBrush(fill)
            p.drawPolygon(poly)
            br = poly.boundingRect()
            p.setPen(THEME.color('ink2'))
            p.drawText(QRectF(br.right() + 3, br.top() - 13, 80, 14), Qt.AlignmentFlag.AlignLeft, b.t.name)
        # the selected camera's suggestions
        if sel_i >= 0:
            sel = tags.get(self.sel)
            for j, s in enumerate(sel['suggestions'] if sel else []):
                if s['kind'] not in ('move', 'add'):
                    continue
                hot = j == self.highlight
                col = QColor(STATUS['warning'])
                pen = QPen(col, 2 if hot else 1.3, Qt.PenStyle.DashLine)
                t = None
                if s['kind'] == 'move':
                    b = next((b for b in m.boards if b.t.uid == s['board']), None)
                    if b is None:
                        continue
                    t = type(b.t)(**{**b.t.to_dict(), 'x': s['x'], 'y': s['y'], 'yaw': s['yaw']})
                    p.setPen(pen)
                    p.drawLine(self.w2s(*b.centre[:2]), self.w2s(s['x'], s['y']))
                else:
                    from ...core.project import Target
                    t = Target(x=s['x'], y=s['y'], yaw=s['yaw'], rows=s['rows'], cols=s['cols'],
                               square_mm=s['square_mm'])
                p.setPen(pen)
                p.setBrush(Qt.BrushStyle.NoBrush)
                p.drawPolygon(QPolygonF([self.w2s(x, y) for x, y, _ in OB._outline(t)]))
        # radars and reflectors
        brown = QColor(SC.MODALITY_COLORS['radar'])
        for s in m.v.sensors:
            if s.visible and s.kind == 'radar':
                q = self.w2s(s.x, s.y)
                p.setPen(Qt.PenStyle.NoPen)
                p.setBrush(brown)
                p.drawRect(QRectF(q.x() - 3, q.y() - 3, 6, 6))
        box = QRectF(8, 26, W - 16, H - 12)
        for r in res.settings.reflectors:
            q = self.w2s(r['x'], r['y'])
            name = r['name']
            if not box.contains(q):              # far out: at the edge, towards it, with its distance
                c0 = QPointF(W / 2, 22 + H / 2)
                d = q - c0
                f = min(abs((box.width() / 2) / d.x()) if abs(d.x()) > 1e-9 else 1e9,
                        abs((box.height() / 2) / d.y()) if abs(d.y()) > 1e-9 else 1e9, 1.0)
                q = c0 + d * f
                name += f' at {r["x"]:.0f}, {r["y"]:.0f} m'
            p.setPen(Qt.PenStyle.NoPen)
            p.setBrush(brown)
            p.drawPolygon(QPolygonF([QPointF(q.x(), q.y() - 6), QPointF(q.x() - 5, q.y() + 4),
                                     QPointF(q.x() + 5, q.y() + 4)]))
            p.setPen(THEME.color('ink2'))
            if q.x() < W - 110:
                p.drawText(QRectF(q.x() + 7, q.y() - 8, 110, 14), Qt.AlignmentFlag.AlignLeft, name)
            else:
                p.drawText(QRectF(q.x() - 117, q.y() - 8, 110, 14), Qt.AlignmentFlag.AlignRight, name)
        # cameras
        p.setFont(mono(11, QFont.Weight.DemiBold))
        for cm in m.cams:
            c = tags.get(cm.cam.uid)
            q = self.w2s(*cm.pos[:2])
            p.setPen(QPen(_tone(c['verdict']), 2.5))
            p.setBrush(THEME.cam(cm.cam.slot))
            r = 6 if cm.cam.uid == self.sel else 5
            p.drawEllipse(q, r, r)
            p.setPen(THEME.color('ink'))
            p.drawText(QRectF(q.x() + 8, q.y() - 6, 40, 14), Qt.AlignmentFlag.AlignLeft, cm.cam.tag)
            self._cams.append((q, cm.cam.uid, f'{cm.cam.tag}: {SEV_WORD[c["verdict"]]}, worst axis '
                                               f'{_times(c["ratio"])} the requirement'))
        # caption
        p.setFont(sans(11))
        p.setPen(THEME.color('muted'))
        p.drawText(QRectF(8, 2, W - 16, 18), Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter,
                   'Top view · boards: light one camera, dark two or more, grey none · ring: verdict · dashed: '
                   'suggested')
        p.end()

    def _hit(self, pos):
        for q, uid, text in self._cams:
            if (q - pos).manhattanLength() < 12:
                return uid, text
        return None, ''

    def mouseMoveEvent(self, e):
        uid, text = self._hit(e.position())
        if uid:
            QToolTip.showText(e.globalPosition().toPoint(), text, self)
        else:
            QToolTip.hideText()

    def mousePressEvent(self, e):
        uid, _ = self._hit(e.position())
        if uid:
            self.picked.emit(uid)


# ============================================================================ the fusion maps
class FusionMaps(QWidget):
    """Per camera, the image map of lidar points per camera pixel (the scene at the fusion distance): the app's
    sequential ramp from 1/1000 to 1 point per pixel (log), black where no lidar point lands, the panel colour outside
    the lens circle and on the vehicle."""

    def __init__(self):
        super().__init__()
        self.maps = []                  # (tag, slot, QImage, support share, median points per px)
        self.setMinimumHeight(160)
        self.setMouseTracking(True)
        self._rects = []
        THEME.changed.connect(self._theme_changed)

    def _theme_changed(self, *_):
        self.set_fusion(getattr(self, '_fusion', None))

    def set_fusion(self, fusion):
        self._fusion = fusion
        self.maps = []
        for f in fusion or []:
            A = np.asarray(f['image']['points_per_px'], float)
            nv, nu = A.shape
            fin = np.isfinite(A)
            t = np.clip((np.log10(np.maximum(np.where(fin, A, 1e-9), 1e-9)) + 3.0) / 3.0, 0, 1)
            lut = np.array([QColor(c).getRgb()[:3] for c in SEQ], np.uint8)
            rgb = lut[np.rint(t * (len(SEQ) - 1)).astype(int)]
            rgb[fin & (np.where(fin, A, 0) <= 0)] = (0, 0, 0)
            rgb[~fin] = QColor(THEME.color('raised')).getRgb()[:3]
            img = np.ascontiguousarray(np.dstack([rgb, np.full((nv, nu), 255, np.uint8)]))
            q = QImage(img.data, nu, nv, 4 * nu, QImage.Format.Format_RGBA8888).copy()
            self.maps.append((f['tag'], f['slot'], q, f['image']['support_share'], f['image']['median_points_per_px'],
                              f['image']['unsupported']))
        self.update()

    def paintEvent(self, e):
        p = QPainter(self)
        p.fillRect(self.rect(), THEME.color('panel'))
        self._rects = []
        if not self.maps:
            p.setPen(THEME.color('muted'))
            p.setFont(sans(12))
            p.drawText(self.rect(), Qt.AlignmentFlag.AlignCenter,
                       'No lidar in the variant, or the fusion density is off.')
            p.end()
            return
        n = len(self.maps)
        cols = 3 if n > 4 else 2 if n > 1 else 1
        rows = int(math.ceil(n / cols))
        W, H = self.width(), self.height()
        cw, ch = (W - 8) / cols, (H - 8) / rows
        for i, (tag, slot, img, share, med, where) in enumerate(self.maps):
            r, c = divmod(i, cols)
            box = QRectF(4 + c * cw + 4, 4 + r * ch + 4, cw - 8, ch - 8)
            ar = img.width() / max(img.height(), 1)
            ih = box.height() - 18
            iw = min(box.width(), ih * ar)
            ih = iw / ar
            tgt = QRectF(box.left(), box.top() + 18, iw, ih)
            p.drawImage(tgt, img)
            p.setPen(QPen(THEME.color('strong'), 1))
            p.setBrush(Qt.BrushStyle.NoBrush)
            p.drawRect(tgt)
            p.setFont(mono(11, QFont.Weight.DemiBold))
            p.setPen(THEME.cam(slot))
            p.drawText(QRectF(box.left(), box.top(), 40, 16), Qt.AlignmentFlag.AlignVCenter, tag)
            p.setFont(sans(11))
            p.setPen(THEME.color('ink2'))
            p.drawText(QRectF(box.left() + 30, box.top(), box.width() - 30, 16), Qt.AlignmentFlag.AlignVCenter,
                       f'{100 * share:.0f} % with lidar · ' + (f'1 point per {1 / med:.0f} px' if med > 0 else
                                                              'no lidar point'))
            self._rects.append((tgt, tag, where, share))
        p.end()

    def mouseMoveEvent(self, e):
        for r, tag, where, share in self._rects:
            if r.contains(e.position()):
                QToolTip.showText(e.globalPosition().toPoint(),
                                  f'{tag}: {100 * share:.0f} % of the image has lidar support'
                                  + (f'; without: {where}' if where else ''), self)
                return
        QToolTip.hideText()


class _SuggRow(QWidget):
    """A suggestion's row: hovering it shows the move in the plan."""

    def __init__(self, on_hover):
        super().__init__()
        self.on_hover = on_hover
        self.button = None

    def enterEvent(self, e):
        self.on_hover(True)
        QWidget.enterEvent(self, e)

    def leaveEvent(self, e):
        self.on_hover(False)
        QWidget.leaveEvent(self, e)


class _Sig(QObject):
    done = pyqtSignal(int, object)
    failed = pyqtSignal(int, str)
    mc_failed = pyqtSignal(int, str)
    progress = pyqtSignal(int, float)


# ============================================================================ the page
class ObservabilityPage(QWidget):
    def __init__(self, ctx):
        super().__init__()
        self.ctx = ctx
        self.result: OB.ObservabilityResult | None = None
        self.busy = False
        self.mc_busy = False
        self.status_text = ''
        self.sel = ''                   # the camera picked
        self._gen = 0
        self._mc_gen = 0
        self._for = None                # inputs key of the result
        self._dirty = True
        self._shown = None
        self._syncing = False
        self._sugg_buttons = []
        self.sig = _Sig()
        self.sig.done.connect(self._on_thread_done)
        self.sig.failed.connect(self._failed)
        self.sig.mc_failed.connect(self._mc_failed)
        self.sig.progress.connect(lambda g, f: self.bar.setValue(int(f * 1000)) if g == self._gen else None)
        self.timer = QTimer(self)
        self.timer.setSingleShot(True)
        self.timer.setInterval(400)
        self.timer.timeout.connect(lambda: self.run(quiet=True))
        self.setObjectName('base')
        lay = QHBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        split = QSplitter(Qt.Orientation.Horizontal)
        split.setHandleWidth(2)
        split.addWidget(self._left())
        # ---------------- centre: the plan and the fusion maps
        self.view_tabs = QTabWidget()
        self.plan = ObsPlan()
        self.plan.picked.connect(self.pick)
        self.view_tabs.addTab(self.plan, 'Plan')
        fw = QWidget()
        fl = QVBoxLayout(fw)
        fl.setContentsMargins(6, 6, 6, 6)
        fl.setSpacing(4)
        self.fusion_maps = FusionMaps()
        fl.addWidget(self.fusion_maps, 3)
        fl.addWidget(label('Lidar points on each object where the camera resolves it, out to the required range '
                           '(median over its bearings) · scan lines · object pixels per lidar point', muted=True,
                           small=True, wrap=True))
        self.fusion_table = _table((('Camera', ''), ('Object', ''), ('Range m', 'From the camera'),
                                    ('Points', 'Median lidar points per frame on the object'),
                                    ('Lines', 'Median scan lines across it'),
                                    ('px / point', 'Object pixels per lidar point'),
                                    ('Enough', 'Share of the bearings with the points a detector needs')))
        fl.addWidget(self.fusion_table, 2)
        self.view_tabs.addTab(fw, 'Fusion')
        split.addWidget(self.view_tabs)
        # ---------------- right: tiles, cameras, the picked camera, the rest
        right = QSplitter(Qt.Orientation.Vertical)
        right.setHandleWidth(2)
        top = QWidget()
        top.setObjectName('panel')
        tl = QVBoxLayout(top)
        tl.setContentsMargins(10, 8, 10, 6)
        tl.setSpacing(6)
        tiles = QHBoxLayout()
        tiles.setSpacing(8)
        self.tiles = {}
        for key, caption, tip in (
                ('cams', 'Cameras', 'Cameras whose every axis meets the requirement (1σ), and the camera pairs a '
                                    'shared board ties'),
                ('worst', 'Loosest', 'The camera with the worst axis against the requirement'),
                ('lidar', 'Lidar–camera', 'The lidar–camera pairs within their tolerances, and the loosest'),
                ('mc', 'Monte Carlo', 'The last check of the predictions against simulated solves')):
            t = self.tiles[key] = Tile(caption, tip)
            t.setMinimumWidth(120)
            tiles.addWidget(t, 1)
        tl.addLayout(tiles)
        tl.addWidget(label('Cameras · 1σ of the pose the planned boards give', bold=True))
        self.cams = _table(CAM_COLS, select=True, stretch=1)
        self.cams.itemSelectionChanged.connect(self._row_picked)
        tl.addWidget(self.cams, 1)
        right.addWidget(top)
        det = QWidget()
        det.setObjectName('panel')
        dl = QVBoxLayout(det)
        dl.setContentsMargins(10, 6, 10, 6)
        dl.setSpacing(4)
        self.det_head = label('', bold=True)
        dl.addWidget(self.det_head)
        self.det_weak = label('', ink2=True, small=True, wrap=True)
        dl.addWidget(self.det_weak)
        self.det_boards = _table((('Board', ''), ('Corners', ''), ('Distance m', 'Median corner distance'),
                                  ('Square px', 'The smallest square side in the image'),
                                  ('Share', 'Its share of the camera\'s information'),
                                  ('Without', 'The camera\'s worst axis without this board, × the requirement')),
                                 stretch=0)
        self.det_boards.setFixedHeight(ROW_H * 4 + 6)
        dl.addWidget(self.det_boards)
        self.why_host = QWidget()
        self.why_lay = QVBoxLayout(self.why_host)
        self.why_lay.setContentsMargins(0, 0, 0, 0)
        self.why_lay.setSpacing(3)
        self.why_lay.addStretch(1)
        sc = QScrollArea()
        sc.setWidgetResizable(True)
        sc.setWidget(self.why_host)
        dl.addWidget(sc, 1)
        self._why = []
        right.addWidget(det)
        self.tabs = QTabWidget()
        self.pairs = _table(REL_COLS)
        self.tabs.addTab(self.pairs, 'Pairs')
        self.lidars = _table(LIDAR_COLS, stretch=0)
        self.tabs.addTab(self.lidars, 'Lidar–camera')
        self.radars = _table(RADAR_COLS)
        self.tabs.addTab(self.radars, 'Radars')
        mcw = QWidget()
        ml = QVBoxLayout(mcw)
        ml.setContentsMargins(4, 4, 4, 4)
        self.mc_note = label('Verify runs simulated captures through the real solves and compares their spread with '
                             'the prediction.', muted=True, small=True, wrap=True)
        ml.addWidget(self.mc_note)
        self.mc_table = _table(MC_COLS, stretch=0)
        ml.addWidget(self.mc_table, 1)
        self.tabs.addTab(mcw, 'Verify')
        fw2 = QWidget()
        f2 = QVBoxLayout(fw2)
        f2.setContentsMargins(4, 4, 4, 4)
        self.find_host = QWidget()
        self.find_lay = QVBoxLayout(self.find_host)
        self.find_lay.setContentsMargins(0, 0, 0, 0)
        self.find_lay.setSpacing(4)
        self.find_lay.addStretch(1)
        sc2 = QScrollArea()
        sc2.setWidgetResizable(True)
        sc2.setWidget(self.find_host)
        f2.addWidget(sc2)
        self.findings: list[StatusLine] = []
        self.tabs.addTab(fw2, 'Findings')
        right.addWidget(self.tabs)
        right.setSizes([330, 300, 260])
        split.addWidget(right)
        split.setSizes([330, 640, 700])
        lay.addWidget(split)
        ctx.store.changed.connect(self._on_change)
        ctx.store.project_replaced.connect(self._on_replaced)
        THEME.changed.connect(self._retheme)
        self._sync_fields()
        self.refresh()

    # ------------------------------------------------------------------ settings panel
    def _left(self):
        w = QWidget()
        w.setObjectName('panel')
        ll = QVBoxLayout(w)
        ll.setContentsMargins(8, 6, 8, 8)
        ll.setSpacing(4)
        top = QHBoxLayout()
        top.addWidget(label('Calibration observability'))
        top.addStretch(1)
        top.addWidget(label('boards, lidars, radars', muted=True, small=True))
        ll.addLayout(top)
        ll.addWidget(hline())
        ll.addWidget(label('How firmly the planned calibration pins each sensor\'s pose, from the Fisher information '
                           'of what the detectors will measure: the boards\' corners through each lens, the lidar\'s '
                           'points on the lidar–camera boards, the radars\' ranges and azimuths to the reflectors. '
                           'Predictions, not measurements: Verify checks them against simulated solves.',
                           muted=True, small=True, wrap=True))
        self.fields = {}
        self.toggles = {}
        for key, text, tip in TOGGLES:
            t = ToggleSwitch(text)
            t.setToolTip(tip)
            t.clicked.connect(lambda on, k=key, x=text: self._set(f'{x.lower()} {"on" if on else "off"}', **{k: on}))
            self.toggles[key] = t
        group, grid = None, None
        for g, key, text, unit, dec, step, tip in FIELDS:
            if g != group:
                if grid is not None:
                    ll.addLayout(grid)
                group = g
                ll.addSpacing(6)
                ll.addWidget(label(g, bold=True))
                if g == 'Boards':
                    ll.addWidget(self.toggles['boards_placed'])
                    ll.addWidget(self.toggles['suggest'])
                elif g == 'Radars':
                    self._reflectors(ll)
                elif g == 'Fusion and check':
                    ll.addWidget(self.toggles['lidar_pairs'])
                    ll.addWidget(self.toggles['fusion'])
                grid = QGridLayout()
                grid.setVerticalSpacing(2)
                grid.setHorizontalSpacing(6)
                grid.setColumnStretch(2, 1)
            f = ScrubSpinBox('', 'plain', dec, unit, step=step, width=100)
            f.setToolTip(tip)
            f.committed.connect(lambda v, k=key, t=text, u=unit: self._set(f'{t.lower()} {v:g} {u}'.strip(), **{k: v}))
            a = label(text, ink2=True)
            a.setFixedWidth(118)
            a.setToolTip(tip)
            i = grid.rowCount()
            grid.addWidget(a, i, 0)
            grid.addWidget(f, i, 1)
            self.fields[key] = f
        ll.addLayout(grid)
        ll.addWidget(label('Lidar–camera board sizes, poses and tolerances: Calibrate ▸ Lidar–camera. Radar figures '
                           'are typical, not a product\'s: enter yours.', muted=True, small=True, wrap=True))
        ll.addSpacing(6)
        ll.addWidget(label('Assumptions', bold=True))
        self.assumptions = label(OB.ASSUMPTIONS, muted=True, small=True, wrap=True)
        ll.addWidget(self.assumptions)
        ll.addStretch(1)
        self.status = label('', muted=True, small=True, wrap=True)
        ll.addWidget(self.status)
        self.bar = QProgressBar()
        self.bar.setTextVisible(False)
        self.bar.setRange(0, 1000)
        self.bar.hide()
        ll.addWidget(self.bar)
        bl = QHBoxLayout()
        self.b_defaults = button('Defaults', 'ghost', None, 'Back to the default settings (keeps the reflectors)')
        self.b_defaults.clicked.connect(self._defaults)
        bl.addWidget(self.b_defaults)
        bl.addStretch(1)
        self.b_export = button('Export…', None, 'export', 'Every figure as JSON and CSV, the plan and the fusion maps '
                                                           'as PNG')
        self.b_export.clicked.connect(lambda: self.export())
        bl.addWidget(self.b_export)
        ll.addLayout(bl)
        bl2 = QHBoxLayout()
        bl2.addStretch(1)
        self.b_verify = button('Verify', None, None, 'Monte Carlo: simulated captures solved by the installation '
                                                     'check\'s joint solve, the lidar–camera calibration\'s solve and '
                                                     'a radar solve, against the predicted 1σ (a few minutes)')
        self.b_verify.clicked.connect(lambda: self.verify())
        self.b_run = button('Run', 'primary')
        self.b_run.clicked.connect(lambda: self.run())
        bl2.addWidget(self.b_verify)
        bl2.addWidget(self.b_run)
        ll.addLayout(bl2)
        sc = QScrollArea()
        sc.setWidgetResizable(True)
        sc.setWidget(w)
        fit_width(sc)
        sc.setMinimumWidth(310)
        return sc

    def _reflectors(self, ll):
        ll.addWidget(label('Corner reflectors (vehicle frame; RCS of a trihedral at 77 GHz)', muted=True, small=True,
                           wrap=True))
        t = self.refl = QTableWidget(0, len(REFL_COLS))
        t.setHorizontalHeaderLabels([h for _, h, _ in REFL_COLS])
        t.verticalHeader().setVisible(False)
        t.verticalHeader().setDefaultSectionSize(ROW_H)
        t.horizontalHeader().setFixedHeight(ROW_H)
        t.setShowGrid(False)
        t.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        t.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        hh = t.horizontalHeader()
        hh.setSectionResizeMode(0, QHeaderView.ResizeMode.Stretch)
        for c in range(1, len(REFL_COLS)):
            hh.setSectionResizeMode(c, QHeaderView.ResizeMode.ResizeToContents)
        t.itemChanged.connect(self._refl_edited)
        t.setFixedHeight(ROW_H * 3 + 6)
        ll.addWidget(t)
        row = QHBoxLayout()
        self.b_add_refl = button('Add', 'ghost', 'plus', 'A reflector 8 m ahead of the first radar (or the vehicle)')
        self.b_add_refl.clicked.connect(self.add_reflector)
        self.b_del_refl = button('Remove', 'ghost', 'minus', 'Remove the selected reflector')
        self.b_del_refl.clicked.connect(self.remove_reflector)
        row.addWidget(self.b_add_refl)
        row.addWidget(self.b_del_refl)
        row.addStretch(1)
        ll.addLayout(row)

    @property
    def settings(self) -> OB.ObservabilitySettings:
        return OB.settings_of(self.ctx.store.project)

    def _store(self, st: OB.ObservabilitySettings, text) -> bool:
        new = OB.ObservabilitySettings.from_dict(st.to_dict()).to_dict()
        if new == self.settings.to_dict():
            self._sync_fields()                # a clamped value: show what is stored
            return False

        def fn(p):
            p.analysis.observability = new
        self.ctx.store.edit('analysis', fn, f'Calibration observability · {text}')
        return True

    def _set(self, text, **kw):
        st = self.settings
        for k, v in kw.items():
            setattr(st, k, v)
        return self._store(st, text)

    def _defaults(self):
        refl = self.settings.reflectors
        st = OB.ObservabilitySettings(reflectors=refl)
        self._store(st, 'defaults')

    def add_reflector(self):
        p = self.ctx.store.project
        st = self.settings
        rad = next((s for s in p.variant.sensors if s.kind == 'radar' and s.visible), None)
        if rad is not None:
            from ...core import sensors as S
            h = math.radians(S.heading(rad))
            x, y, z = rad.x + 8.0 * math.cos(h), rad.y + 8.0 * math.sin(h), rad.z
        else:
            front = p.vehicle.measured().get('front_x', 3.0) if p.vehicle.mesh is not None else 3.0
            x, y, z = front + 8.0, 0.0, 0.5
        names = {r['name'] for r in st.reflectors}
        k = len(st.reflectors) + 1
        while f'CR{k}' in names:
            k += 1
        st.reflectors = st.reflectors + [{'name': f'CR{k}', 'x': round(x, 2), 'y': round(y, 2), 'z': round(z, 2),
                                          'edge_mm': 100.0}]
        return self._store(st, f'added reflector CR{k}')

    def remove_reflector(self):
        r = self.refl.currentRow()
        st = self.settings
        if not 0 <= r < len(st.reflectors):
            return False
        name = st.reflectors[r]['name']
        st.reflectors = [x for i, x in enumerate(st.reflectors) if i != r]
        return self._store(st, f'removed reflector {name}')

    def _refl_edited(self, item):
        if self._syncing:
            return
        st = self.settings
        r, c = item.row(), item.column()
        key = REFL_COLS[c][0]
        if not 0 <= r < len(st.reflectors) or key == 'rcs':
            self._sync_fields()
            return
        rows = [dict(x) for x in st.reflectors]
        text = item.text().strip()
        if key == 'name':
            if not text:
                self._sync_fields()
                return
            rows[r]['name'] = text[:24]
        else:
            try:
                rows[r][key] = float(parse_number(text, 'plain'))
            except Exception:
                self._sync_fields()
                return
        st.reflectors = rows
        self._store(st, f'reflector {rows[r]["name"]} {REFL_COLS[c][1].split()[0]}')

    def _sync_fields(self):
        st = self.settings
        d = st.to_dict()
        self._syncing = True
        try:
            for k, f in self.fields.items():
                f.set_value(float(d[k]))
            for k, t in self.toggles.items():
                t.blockSignals(True)
                t.setChecked(bool(d[k]))
                t.blockSignals(False)
            for k in ('board_sigma_mm', 'board_sigma_deg'):
                self.fields[k].setEnabled(st.boards_placed)
            t = self.refl
            t.setRowCount(len(st.reflectors))
            for r, refl in enumerate(st.reflectors):
                for c, (key, _, dec) in enumerate(REFL_COLS):
                    v = OB.reflector_rcs(refl['edge_mm']) if key == 'rcs' else refl[key]
                    text = str(v) if dec is None else f'{v:.{dec}f}'
                    it = t.item(r, c)
                    if it is None:
                        it = QTableWidgetItem()
                        if dec is not None:
                            it.setFont(mono(12))
                            it.setTextAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
                        if key == 'rcs':
                            it.setFlags(it.flags() & ~Qt.ItemFlag.ItemIsEditable)
                            it.setForeground(THEME.color('muted'))
                        t.setItem(r, c, it)
                    if it.text() != text:
                        it.setText(text)
            self.b_del_refl.setEnabled(bool(st.reflectors))
        finally:
            self._syncing = False

    # ------------------------------------------------------------------ state
    def _inputs_key(self, p=None):
        p = p or self.ctx.store.project
        m = p.vehicle.mesh
        a = p.analysis
        return json.dumps({'v': p.variant.to_dict(), 't': [t.to_dict() for t in p.targets],
                           'ts': p.target_settings.to_dict(), 'o': a.observability, 'lc': a.lidar_calib,
                           'sc': a.sensor_coverage, 'pr': a.perception, 'h': a.housings, 'lib': p.section('library'),
                           'rig': [r.to_dict() for r in p.rig], 'mesh': [id(m), getattr(m, 'n_tris', 0)]},
                          sort_keys=True, default=str)

    @property
    def stale(self) -> bool:
        return self.result is not None and self._for is not None and self._for[1] != self._inputs_key()

    def _quick(self):
        return self.result is None or self.result.metrics.get('seconds', 99) < AUTO_MAX_S

    def _on_change(self, *_):
        if not self.isVisible():
            self._dirty = True
            return
        self._sync_fields()
        if self.stale and self._quick() and not self.busy:
            self.timer.start()
        self.refresh()

    def _on_replaced(self, *_):
        self._gen += 1                          # a run still going is for the old project
        self._mc_gen += 1
        self.result, self._for, self.busy, self.mc_busy = None, None, False, False
        self.status_text = ''
        self.sel = ''
        self._shown = False
        self._sync_fields()
        self.refresh()
        if self.isVisible():
            self.timer.start()

    def _retheme(self, *_):
        self._shown = False
        self._sync_fields()
        self.refresh()

    def showEvent(self, e):
        super().showEvent(e)
        self.activate()

    def activate(self):
        if self._dirty:
            self._dirty = False
            self._sync_fields()
        if (self.result is None or self.stale) and self._quick() and not self.busy:
            self.timer.start()
        self.refresh()

    # ------------------------------------------------------------------ run
    def _engine(self):
        win = self.ctx.window
        return win._shared_engine() if win is not None and hasattr(win, '_shared_engine') else \
            self.ctx.analysis.engine.share()

    def run(self, quiet=False) -> bool:
        win = self.ctx.window
        if self.busy:
            if quiet or (win is not None and JOB in getattr(win, '_jobs', {})):
                return False
        p = self.ctx.store.project
        project = p.snapshot()
        eng = self._engine()
        self._gen += 1
        gen = self._gen
        key = (p, self._inputs_key(p))
        self.busy = True
        self.status_text = 'Computing…'

        def work(progress, cancel):
            return key, OB.run(project, eng, OB.settings_of(project), progress=progress, cancel=cancel)
        if not quiet and win is not None and hasattr(win, 'run_job'):
            ok = win.run_job(JOB, 'Calibration observability', work, lambda out: self._done(gen, out),
                             lambda m: self._failed(gen, m))
            if not ok:
                self.busy = False
                self.status_text = ''
            self.refresh()
            return ok
        self.bar.setValue(0)

        def thread():
            try:
                out = work(lambda f, *_: self.sig.progress.emit(gen, f), lambda: gen != self._gen)
                self.sig.done.emit(gen, ('run', out))
            except RuntimeError:                # the page closed while computing
                pass
            except Exception as e:              # noqa: BLE001
                try:
                    self.sig.failed.emit(gen, f'{type(e).__name__}: {e}\n{traceback.format_exc()}')
                except RuntimeError:
                    pass
        threading.Thread(target=thread, daemon=True).start()
        self.refresh()
        return True

    def _on_thread_done(self, gen, payload):
        what, out = payload
        if what == 'run':
            self._done(gen, out)
        else:
            self._mc_done(gen, out)

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
        m = res.metrics
        self.status_text = (f'{m["good"]} of {m["cameras"]} cameras within the requirement · {m["observations"]} '
                            f'board views · {m["pairs"]} pairs · {m["lidar_pairs"]} lidar–camera · '
                            f'{m["seconds"]:.1f} s')
        if self.sel not in {c['uid'] for c in res.cameras}:
            worst = max(res.cameras, key=lambda c: c['ratio'] if math.isfinite(c['ratio']) else 1e18, default=None)
            self.sel = worst['uid'] if worst else ''
        self._shown = None
        self.refresh()
        if self.stale and self.isVisible() and self._quick():
            self.timer.start()

    def _failed(self, gen, msg):
        if gen != self._gen:
            return
        self.busy = False
        self.status_text = 'Failed: ' + msg.splitlines()[0]
        self.refresh()

    def _mc_failed(self, gen, msg):
        if gen != self._mc_gen:
            return
        self.mc_busy = False
        self.status_text = 'Monte Carlo failed: ' + msg.splitlines()[0]
        self.refresh()

    def verify(self, trials=None) -> bool:
        """The Monte Carlo check of the current result (a background job)."""
        res = self.result
        win = self.ctx.window
        if res is None or self.mc_busy or self.busy:
            return False
        self._mc_gen += 1
        gen = self._mc_gen
        self.mc_busy = True
        n = int(trials or res.settings.mc_trials)

        def work(progress, cancel):
            return res, OB.monte_carlo(res, trials=n, progress=progress, cancel=cancel)
        if win is not None and hasattr(win, 'run_job'):
            ok = win.run_job(JOB_MC, 'Calibration observability · Monte Carlo', work,
                             lambda out: self._mc_done(gen, out), lambda m: self._mc_failed(gen, m))
            if not ok:
                self.mc_busy = False
            self.refresh()
            return ok

        def thread():
            try:
                out = work(lambda f, *_: self.sig.progress.emit(self._gen, f), lambda: gen != self._mc_gen)
                self.sig.done.emit(gen, ('mc', out))
            except RuntimeError:
                pass
            except Exception as e:              # noqa: BLE001
                try:
                    self.sig.mc_failed.emit(gen, f'{type(e).__name__}: {e}\n{traceback.format_exc()}')
                except RuntimeError:
                    pass
        threading.Thread(target=thread, daemon=True).start()
        self.refresh()
        return True

    def _mc_done(self, gen, out):
        if gen != self._mc_gen:
            return
        self.mc_busy = False
        res, mc = out if out is not None else (None, None)
        if mc is None:
            self.status_text = 'Monte Carlo cancelled'
        elif res is self.result:
            self.status_text = ('Monte Carlo: the predictions hold' if mc['agree'] else
                                'Monte Carlo: the simulated spread differs from the prediction') + \
                               f' ({mc["trials"]} trials, {mc["seconds"]:.0f} s)'
            self.ctx.msg(self.status_text, 5000)
            self.tabs.setCurrentIndex(3)
        self._shown = None
        self.refresh()

    def cancel(self):
        win = self.ctx.window
        for job in (JOB, JOB_MC):
            j = getattr(win, '_jobs', {}).get(job) if win is not None else None
            if j is not None:
                j['cancel'].set()
        if win is None or not hasattr(win, 'run_job'):
            self._gen += 1
            self._mc_gen += 1
            self.busy = self.mc_busy = False
            self.status_text = 'Cancelled'
            self.refresh()

    # ------------------------------------------------------------------ show
    def refresh(self):
        win = self.ctx.window
        p = self.ctx.store.project
        running = self.busy
        self.b_run.setEnabled(not running)
        self.b_verify.setEnabled(self.result is not None and not running and not self.mc_busy)
        self.b_export.setEnabled(self.result is not None)
        self.bar.setVisible((running or self.mc_busy) and (win is None or not hasattr(win, 'run_job')))
        text = self.status_text or ('Ready: Run computes the study.' if p.targets else EMPTY)
        self.status.setText(text + (' · stale: the design or the settings changed since' if self.stale else ''))
        res = self.result
        if res is self._shown:
            return
        self._shown = res
        self.plan.message = EMPTY if not p.targets else 'Run the calibration observability'
        self.plan.set_data(res, p, self.sel)
        self.fusion_maps.set_fusion(res.fusion if res is not None else None)
        self._fill_tiles(res)
        self._fill_cameras(res)
        self._fill_detail(res)
        self._fill_pairs(res)
        self._fill_lidars(res)
        self._fill_radars(res)
        self._fill_mc(res)
        self._fill_fusion(res)
        self._fill_findings(res)

    def _fill_tiles(self, res):
        if res is None:
            for t in self.tiles.values():
                t.set('—', '', 'not run yet')
            return
        cams = res.cameras
        good = sum(1 for c in cams if c['verdict'] == 'good')
        npairs = len(res.pairs)
        self.tiles['cams'].set(f'{good}', f'of {len(cams)}', f'within · {npairs} pair{"s" * (npairs != 1)} tied',
                               'good' if cams and good == len(cams) else 'warning' if cams else None)
        if cams:
            w = max(cams, key=lambda c: c['ratio'] if math.isfinite(c['ratio']) else 1e18)
            self.tiles['worst'].set(_times(w['ratio']) if w['sigma'] else '—', '',
                                    f'{w["tag"]} · {w["worst_axis"] or "no board"}', w['verdict'])
        else:
            self.tiles['worst'].set('—', '', 'no camera')
        if res.lidar_pairs:
            okp = sum(1 for x in res.lidar_pairs if x['verdict'] == 'good')
            w = max(res.lidar_pairs, key=lambda x: x['ratio'] if math.isfinite(x['ratio']) else 1e18)
            self.tiles['lidar'].set(f'{okp}', f'of {len(res.lidar_pairs)}',
                                    f'loosest {w["camera"]}–{w["lidar"]} {_times(w["ratio"])}', w['verdict'])
        else:
            self.tiles['lidar'].set('—', '', 'no lidar–camera pair')
        mc = res.mc
        if mc is None:
            self.tiles['mc'].set('—', '', 'Verify to check')
        else:
            self.tiles['mc'].set('Holds' if mc['agree'] else 'Differs', '', f'{mc["trials"]} trials',
                                 'good' if mc['agree'] else 'warning')

    def _fill_cameras(self, res):
        t = self.cams
        rows = res.cameras if res is not None else []
        t.blockSignals(True)
        t.setRowCount(len(rows))
        for r, c in enumerate(rows):
            it = QTableWidgetItem(c['tag'])
            it.setForeground(THEME.cam(c['slot']))
            it.setData(Qt.ItemDataRole.UserRole, c['uid'])
            t.setItem(r, 0, it)
            names = ' '.join(b['name'] for b in c['boards']) or 'none'
            t.setItem(r, 1, QTableWidgetItem(f'{names} · {c["corners"]}' if c['boards'] else 'none'))
            t.item(r, 1).setToolTip('Not counted: ' + ', '.join(f'{u["name"]} ({u["reason"]})' for u in c['unseen'])
                                    if c['unseen'] else 'Every board counted')
            s, s0 = c['sigma'], c['sigma_corners']
            for j, k in enumerate(OB.SIGMA_KEYS, 2):
                if s is None:
                    t.setItem(r, j, _num('—'))
                    continue
                unit = 'deg' if k.endswith('deg') else 'mm'
                need = self.result.settings.need_deg if unit == 'deg' else self.result.settings.need_mm
                v = s[k]
                col = _tone('good') if v <= need else _tone('warning') if v <= 2 * need else _tone('serious')
                t.setItem(r, j, _num(_f(v, 3 if unit == 'deg' else 1),
                                     f'{k.split("_")[0]}: ±{_f(v, 3)} {"°" if unit == "deg" else "mm"}; from the '
                                     f'corners alone (boards exact) ±{_f(s0[k], 3)}', col))
            it = QTableWidgetItem(_times(c['ratio']) if s else SEV_WORD['critical'])
            it.setForeground(_tone(c['verdict']))
            it.setToolTip(f'{SEV_WORD[c["verdict"]]}: worst axis {c["worst_axis"]}' if s else
                          (c['reasons'][0] if c['reasons'] else 'Not measured'))
            t.setItem(r, 8, it)
            t.setItem(r, 9, _num(_f(c['condition'], 0) if c['condition'] else '—',
                                 c['weakest']['text'] if c['weakest'] else None))
            if c['uid'] == self.sel:
                t.selectRow(r)
        t.blockSignals(False)

    def _row_picked(self):
        r = self.cams.currentRow()
        it = self.cams.item(r, 0) if r >= 0 else None
        if it is not None:
            self.pick(it.data(Qt.ItemDataRole.UserRole))

    def pick(self, uid):
        """Show a camera's figures, reasons and suggestions (and its boards in the plan)."""
        if uid == self.sel:
            return
        self.sel = uid
        self.plan.set_selected(uid)
        if self.result is not None:
            for r in range(self.cams.rowCount()):
                it = self.cams.item(r, 0)
                if it is not None and it.data(Qt.ItemDataRole.UserRole) == uid and self.cams.currentRow() != r:
                    self.cams.blockSignals(True)
                    self.cams.selectRow(r)
                    self.cams.blockSignals(False)
        self._fill_detail(self.result)

    def _fill_detail(self, res):
        for w in self._why + self._sugg_buttons:
            self.why_lay.removeWidget(w)
            w.hide()                    # until the deferred delete runs it would still paint over the new rows
            w.deleteLater()
        self._why, self._sugg_buttons = [], []
        c = res.camera_by_uid(self.sel) if res is not None else None
        if c is None:
            self.det_head.setText('Pick a camera')
            self.det_weak.setText('')
            self.det_boards.setRowCount(0)
            return
        verdict = SEV_WORD[c['verdict']]
        self.det_head.setText(f'{c["tag"]} · {verdict}' + (f' · worst axis {c["worst_axis"]} {_times(c["ratio"])} '
                                                            f'the requirement' if c['sigma'] else ''))
        if c['weakest']:
            down = '' if c['euler'] else ' · angles about the camera\'s own axes (it looks straight down)'
            self.det_weak.setText(f'Weakest direction (1σ): {c["weakest"]["text"]} · condition number '
                                  f'{c["condition"]:,.0f}{down}')
        else:
            self.det_weak.setText('')
        t = self.det_boards
        t.setRowCount(len(c['boards']))
        for r, b in enumerate(c['boards']):
            t.setItem(r, 0, QTableWidgetItem(b['name']))
            t.setItem(r, 1, _num(str(b['corners'])))
            t.setItem(r, 2, _num(_f(b['distance_m'], 2)))
            t.setItem(r, 3, _num(_f(b['square_px'], 0)))
            t.setItem(r, 4, _num(f'{100 * b["share"]:.0f} %'))
            t.setItem(r, 5, _num(_times(b['ratio_without']), 'Without it the camera is not determined at all'
                                 if not math.isfinite(b['ratio_without']) else None))
        def add(w):
            self.why_lay.insertWidget(self.why_lay.count() - 1, w)
        for text in c['reasons']:
            w = StatusLine('warning' if c['verdict'] != 'good' else 'good', text)
            add(w)
            self._why.append(w)
        for j, s in enumerate(c['suggestions']):
            row = _SuggRow(lambda on, j=j: self._hover_sugg(j if on else -1))
            h = QHBoxLayout(row)
            h.setContentsMargins(0, 0, 0, 0)
            h.setSpacing(6)
            lab = QLabel(s['text'])
            lab.setWordWrap(True)
            lab.setProperty('ink2', True)
            h.addWidget(lab, 1)
            b = QPushButton('Use ±' + f'{s["board_sigma_mm"]:g} mm' if s['kind'] == 'placement' else 'Apply')
            b.setToolTip('Lay the boards to this precision: sets the placement ± here (set the installation check\'s '
                         'board placement to match)' if s['kind'] == 'placement' else
                         'Move the board (undoable); the study runs again' if s['kind'] == 'move' else
                         'Add the board to the targets (undoable); the study runs again')
            b.clicked.connect(lambda _=False, s=s: self.apply(s))
            b.setCursor(Qt.CursorShape.PointingHandCursor)
            h.addWidget(b, 0, Qt.AlignmentFlag.AlignTop)
            row.button = b
            add(row)
            self._sugg_buttons.append(row)
        if not c['reasons'] and not c['suggestions']:
            w = StatusLine('good', 'Every axis meets the requirement.' if c['verdict'] == 'good' else
                           'No board move tried helps it by 10 % or more.')
            add(w)
            self._why.append(w)

    def _hover_sugg(self, j):
        self.plan.highlight = j
        self.plan.update()

    def apply(self, s) -> bool:
        """Carry out a suggestion: a board moved or added (Calibrate ▸ Targets, undoable) or the placement precision."""
        if s['kind'] == 'placement':
            return self._set(f'boards placed to ±{s["board_sigma_mm"]:g} mm', board_sigma_mm=s['board_sigma_mm'],
                             board_sigma_deg=s['board_sigma_deg'])
        p = self.ctx.store.project
        if s['kind'] == 'move' and not any(t.uid == s['board'] for t in p.targets):
            self.ctx.msg('That board is gone: run again.', 4000)
            return False
        text = (f'Moved {s["name"]} for {s["camera"]}' if s['kind'] == 'move' else
                f'Added a board for {s["camera"]}')
        self.ctx.store.edit('targets', lambda pj: OB.apply_suggestion(pj, s), text)
        self.ctx.msg(f'{text}: {s["camera"]}\'s worst axis {_times(s["before"])} → {_times(s["after"])} the '
                     'requirement', 5000)
        return True

    def _sig_cells(self, t, r, start, sig, keys, need_deg, need_mm):
        for j, k in enumerate(keys, start):
            if sig is None:
                t.setItem(r, j, _num('—'))
                continue
            v = sig[k]
            unit = 'deg' if k.endswith('deg') else 'mm'
            need = need_deg if unit == 'deg' else need_mm
            col = _tone('good') if v <= need else _tone('warning') if v <= 2 * need else _tone('serious')
            t.setItem(r, j, _num(_f(v, 3 if unit == 'deg' else 1), None, col))

    def _fill_pairs(self, res):
        t = self.pairs
        rows = res.pairs if res is not None else []
        t.setRowCount(len(rows))
        st = res.settings if res is not None else None
        for r, pr in enumerate(rows):
            t.setItem(r, 0, QTableWidgetItem('–'.join(pr['tags'])))
            t.setItem(r, 1, QTableWidgetItem(' '.join(pr['shared'])))
            self._sig_cells(t, r, 2, pr['sigma'], OB.REL_KEYS, st.need_deg, st.need_mm)
            it = QTableWidgetItem(_times(pr['ratio']))
            it.setForeground(_tone(pr['verdict']))
            t.setItem(r, 8, it)

    def _fill_lidars(self, res):
        t = self.lidars
        rows = res.lidar_pairs if res is not None else []
        t.setRowCount(len(rows))
        for r, lp in enumerate(rows):
            it = QTableWidgetItem(f'{lp["camera"]}–{lp["lidar"]}')
            it.setToolTip(f'{100 * lp["share"]:.0f} % of {lp["camera"]}\'s image overlaps {lp["lidar"]}\'s field'
                          + (' · ' + ' '.join(lp['notes']) if lp['notes'] else ''))
            t.setItem(r, 0, it)
            t.setItem(r, 1, _num(str(len(lp['poses'])), 'Lidar points per pose: ' +
                                 ', '.join(str(n) for n in lp['points'])))
            tol = lp['tolerance']
            self._sig_cells(t, r, 2, lp['sigma'], OB.SIGMA_KEYS, tol['deg'], tol['mm'])
            it = QTableWidgetItem(_times(lp['ratio']) if lp['sigma'] else 'No plan')
            it.setForeground(_tone(lp['verdict']))
            t.setItem(r, 8, it)

    def _fill_radars(self, res):
        t = self.radars
        rows = res.radars if res is not None else []
        t.setRowCount(len(rows))
        st = res.settings if res is not None else None
        for r, rd in enumerate(rows):
            t.setItem(r, 0, QTableWidgetItem(f'{rd["tag"]} · {rd["model"]}'))
            it = QTableWidgetItem(' '.join(rd['reflectors']) or '—')
            it.setToolTip(rd['note'] or '')
            t.setItem(r, 1, it)
            self._sig_cells(t, r, 2, rd['sigma'], ('yaw_deg', 'x_mm', 'y_mm'), st.radar_need_deg, st.radar_need_mm)
            it = QTableWidgetItem(_times(rd['ratio']) if rd['sigma'] else '—')
            it.setForeground(_tone(rd['verdict']) if rd['sigma'] else THEME.color('muted'))
            it.setToolTip(rd['note'] or '')
            t.setItem(r, 5, it)

    def _fill_mc(self, res):
        t = self.mc_table
        mc = res.mc if res is not None else None
        rows = []
        if mc is not None:
            cam = mc.get('cameras') or {}
            opt = cam.get('optimum') or {}
            for group, label_ in (('cameras', ''), ('pairs', ' (pair)')):
                for name, s in (cam.get(group) or {}).items():
                    o = (opt.get(group) or {}).get(name)
                    for k, a in s['axes'].items():
                        rows.append((name + label_, k, a['predicted'], a['empirical'],
                                     o['axes'][k]['empirical'] if o else None,
                                     (o['axes'][k] if o else a)['ratio']))
            for part, group in (('lidar', 'pairs'), ('radar', 'radars')):
                for name, s in ((mc.get(part) or {}).get(group) or {}).items():
                    for k, a in s['axes'].items():
                        rows.append((name, k, a['predicted'], a['empirical'], None, a['ratio']))
            note = (f'{mc["trials"]} simulated captures per check, seed {mc["seed"]}, {mc["seconds"]:.0f} s. '
                    + ('The predictions hold: ' if mc['agree'] else 'The spread differs from the prediction: ')
                    + f'every σ within ×{math.exp(mc["band"]):.2f} of its prediction, either way (4.5 standard '
                      'errors at this many trials).')
            if cam:
                note += (f' The joint solve stopped short of its optimum in {cam["stalled"]} of {cam["solved"]} '
                         'trials' + (' (its own spread is shown beside the optimum\'s).' if cam['stalled'] else '.'))
            self.mc_note.setText(note)
        t.setRowCount(len(rows))
        for r, (name, k, pred, emp, opt_, ratio) in enumerate(rows):
            unit = '°' if k.endswith('deg') else 'mm'
            nd = 3 if unit == '°' else 2
            t.setItem(r, 0, QTableWidgetItem(name))
            t.setItem(r, 1, QTableWidgetItem(f'{k.split("_")[0]} {unit}'))
            t.setItem(r, 2, _num(_f(pred, nd)))
            t.setItem(r, 3, _num(_f(emp, nd)))
            t.setItem(r, 4, _num(_f(opt_, nd) if opt_ is not None else ''))
            ok = mc is not None and abs(math.log(max(ratio, 1e-9))) <= mc['band'] + 1e-12 \
                if math.isfinite(ratio) else False
            t.setItem(r, 5, _num(_f(ratio, 2), None, _tone('good' if ok else 'warning')))

    def _fill_fusion(self, res):
        t = self.fusion_table
        rows = []
        for f in (res.fusion if res is not None else None) or []:
            for k in f['classes']:
                for row in k['rows']:
                    if row['points_median'] is None:
                        continue
                    rows.append((f, k, row))
        t.setRowCount(len(rows))
        for r, (f, k, row) in enumerate(rows):
            it = QTableWidgetItem(f['tag'])
            it.setForeground(THEME.cam(f['slot']))
            t.setItem(r, 0, it)
            t.setItem(r, 1, QTableWidgetItem(k['name']))
            t.setItem(r, 2, _num(f'{row["range_m"]:g}', f'Over {row["bearings_deg"]:.0f}° of bearings'))
            t.setItem(r, 3, _num(_f(row['points_median'], 0), f'Fewest {_f(row["points_min"], 0)}',
                                 _tone('good') if row['points_median'] >= k['min_points'] else THEME.color('ink2')))
            t.setItem(r, 4, _num(_f(row['lines_median'], 0)))
            t.setItem(r, 5, _num(_f(row['px_per_point_median'], 0)))
            t.setItem(r, 6, _num(f'{100 * row["supported_share"]:.0f} %',
                                 f'{k["min_points"]:g} points or more (Lidar and radar coverage objects)'))

    def _fill_findings(self, res):
        for w in self.findings:
            self.find_lay.removeWidget(w)
            w.hide()
            w.deleteLater()
        self.findings = []
        for f in (res.findings if res is not None else []):
            w = StatusLine(f['severity'], f['message'])
            self.find_lay.insertWidget(self.find_lay.count() - 1, w)
            self.findings.append(w)

    # ------------------------------------------------------------------ export
    def export(self, folder=None):
        res = self.result
        if res is None:
            return []
        if folder is None:
            folder = QFileDialog.getExistingDirectory(self, 'Export the calibration observability into')
            if not folder:
                return []
        from ...core import provenance as PV          # where the JSON came from
        paths = PV.stamp_paths(OB.export(res, self.ctx.store.project, folder), self.ctx.store.project, folder)
        self.ctx.msg(f'Exported the calibration observability to {folder}', 5000)
        return paths

    def summary(self) -> dict | None:
        return OB.summary(self.result) if self.result is not None else None
