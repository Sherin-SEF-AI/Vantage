"""The ground truth of a CARLA drive, frame by frame (Simulate ▸ CARLA ▸ Ground truth): a camera and a frame of it,
one layer at a time (the camera image; the semantic labels in CARLA's palette with a legend of the classes present;
the instances, one colour per object; the range along each pixel's ray on a log scale; a lidar sweep from above in
its points' nuScenes-lidarseg classes), the objects' tight and full boxes with their classes over it, and the frame's
object table (class, id, range, truncation, occlusion). A row clicked, or an object clicked in the image, is
highlighted.

Everything comes from the drive's ground_truth/index.json (core.groundtruth.read) and is loaded when shown: the label
images of one frame when its frame is shown (a few frames kept), a lidar sweep and the nuScenes tables only when the
lidar layer is chosen. A drive without ground truth, or without a layer, says so."""
from __future__ import annotations

import math
from collections import OrderedDict

import numpy as np
from PyQt6.QtCore import QPointF, QRectF, Qt, QTimer, pyqtSignal
from PyQt6.QtGui import QColor, QIcon, QImage, QPainter, QPen, QPixmap, QPolygonF
from PyQt6.QtWidgets import (QAbstractItemView, QComboBox, QHBoxLayout, QHeaderView, QSlider, QSplitter, QTableWidget,
                             QTableWidgetItem, QVBoxLayout, QWidget)

from ..core import groundtruth as GT
from . import colormaps as CM
from .theme import THEME, mono, sans
from .widgets import ROW_H, SegmentedControl, label

LAYERS = (('image', 'Image'), ('semantic', 'Semantic'), ('instance', 'Instance'), ('range', 'Range'),
          ('lidar', 'Lidar'))
LAYER_TIP = {
    'image': 'The camera image as recorded',
    'semantic': 'A class per pixel in CARLA\'s palette (the ego cyan); the legend lists the classes in this frame',
    'instance': 'Every object\'s visible pixels in its own colour (the same in every frame); black: no object',
    'range': 'The distance along each pixel\'s own ray on a log scale (nearest lightest); grey: nothing within '
             f'{GT.RANGE_MAX_M:.0f} m',
    'lidar': 'The lidar sweep nearest this frame, from above (x forward up, the rig\'s origin in the middle), each '
             'point in its nuScenes-lidarseg class\'s colour; the frame\'s objects as their 3D boxes\' footprints',
}
BOXES = (('none', 'No boxes'), ('tight', 'Tight'), ('full', 'Full'), ('both', 'Both'))
BOX_TIP = ('Tight: the box of the object\'s visible pixels (solid). Full: its 3D box\'s edges through the lens, not '
           'clipped to the image (dashed)')
OBJ_COLS = (('Class', 'The object\'s class (its nuScenes category in the tooltip)'),
            ('Id', 'Its id in the drive: the instance image\'s value, the same in every frame'),
            ('Range', 'Camera to the centre of its 3D box'),
            ('Truncated', 'Share of its 3D box outside the image or the lens\'s field'),
            ('Occluded', 'Of the pixels whose ray meets its 3D box, the share hidden by something nearer'),
            ('Pixels', 'Its visible pixels'))
RANGE_LUT = CM.ramp(CM.BLUE[::-1])   # range: the blue scale's whole lightness, the near end lightest
LIDAR_HALF_M = 40.0          # the top view: this far from the rig's origin to each edge
LIDAR_PX = 800
LABEL_MIN_PX = 14            # boxes at least this tall (image px) are labelled; the selected one always
CACHE_FRAMES = 6
EMPTY_NO_RUN = 'Run, or Open run…, to see a drive\'s ground truth here.'
EMPTY_NO_GT = ('This drive has no ground truth: it was recorded with the Ground truth switches off. Turn them on and '
               'run again.')


def _qimage(rgb: np.ndarray) -> QImage:
    a = np.ascontiguousarray(rgb)
    fmt = QImage.Format.Format_RGBA8888 if a.shape[2] == 4 else QImage.Format.Format_RGB888
    return QImage(a.data, a.shape[1], a.shape[0], a.strides[0], fmt).copy()


def _lighter(rgb, k=0.45) -> QColor:
    """A class colour lifted towards white, so a box of a dark class (car: navy) reads on an image."""
    r, g, b = (int(round(c + (255 - c) * k)) for c in rgb)
    return QColor(r, g, b)


def outline(mask: np.ndarray) -> np.ndarray:
    """The pixels along a mask's edge, two wide (its own border pixels and those just outside)."""
    m = np.asarray(mask, bool)
    inner = m.copy()
    inner[1:] &= m[:-1]
    inner[:-1] &= m[1:]
    inner[:, 1:] &= m[:, :-1]
    inner[:, :-1] &= m[:, 1:]
    e = m & ~inner
    out = e.copy()
    out[1:] |= e[:-1]
    out[:-1] |= e[1:]
    out[:, 1:] |= e[:, :-1]
    out[:, :-1] |= e[:, 1:]
    return out & ~inner


def _swatch(rgb, size=10) -> QIcon:
    pm = QPixmap(size, size)
    pm.fill(QColor(*(int(c) for c in rgb)))
    return QIcon(pm)


class GTCanvas(QWidget):
    """The frame's layer fitted into the widget (the wheel zooms about the pointer, a double click fits it again), its
    boxes (image px) or footprints drawn over it; a click gives the image pixel under it."""
    picked = pyqtSignal(float, float)
    TOP = 22.0
    MAX_ZOOM = 12.0

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumSize(320, 220)
        self.img: QImage | None = None
        self.shapes: list = []           # {'rect' (x0, y0, x1, y1) | 'poly' [(x, y)], 'color', 'dashed', 'label', 'hi'}
        self.caption = ''
        self.empty = EMPTY_NO_RUN
        self.rings: list = []            # [(radius px, text)] around the image centre (the lidar view)
        self.zoom = 1.0                  # over the fitted size
        self.centre = None               # the image point in the middle of the view (None: the image's centre)
        THEME.changed.connect(self._theme_changed)

    def _theme_changed(self, *_):
        self.update()

    def set_view(self, img, shapes=(), caption='', rings=()):
        if img is None or self.img is None or img.size() != self.img.size():
            self.zoom, self.centre = 1.0, None               # another image size: fitted again
        self.img, self.shapes, self.caption, self.rings = img, list(shapes), caption, list(rings)
        self.update()

    def _area(self) -> QRectF:
        return QRectF(0, self.TOP, self.width(), max(self.height() - self.TOP - 4, 1))

    def image_rect(self) -> QRectF:
        a = self._area()
        if self.img is None:
            return a
        iw, ih = max(self.img.width(), 1), max(self.img.height(), 1)
        s = min(a.width() / iw, a.height() / ih) * self.zoom
        cx, cy = self.centre if self.centre is not None else (iw / 2, ih / 2)
        # the view stays on the image: a side smaller than the view is centred
        hx, hy = a.width() / (2 * s), a.height() / (2 * s)
        cx = iw / 2 if iw * s <= a.width() else min(max(cx, hx), iw - hx)
        cy = ih / 2 if ih * s <= a.height() else min(max(cy, hy), ih - hy)
        self.centre = (cx, cy) if self.zoom > 1.0 else None
        return QRectF(a.center().x() - cx * s, a.center().y() - cy * s, iw * s, ih * s)

    def zoom_to(self, x0, y0, x1, y1):
        """Show the image rectangle (image px) as large as it fits."""
        if self.img is None:
            return
        a = self._area()
        fit = min(a.width() / self.img.width(), a.height() / self.img.height())
        s = min(a.width() / max(x1 - x0, 1e-6), a.height() / max(y1 - y0, 1e-6))
        self.zoom = float(min(max(s / fit, 1.0), self.MAX_ZOOM))
        self.centre = (float(x0 + x1) / 2, float(y0 + y1) / 2)
        self.update()

    def reset_zoom(self):
        self.zoom, self.centre = 1.0, None
        self.update()

    def wheelEvent(self, e):
        if self.img is None:
            return
        r = self.image_rect()
        s = r.width() / self.img.width()
        pos = e.position()
        q = ((pos.x() - r.x()) / s, (pos.y() - r.y()) / s)           # the image point under the pointer stays there
        z = min(max(self.zoom * (1.25 ** (e.angleDelta().y() / 120.0)), 1.0), self.MAX_ZOOM)
        a = self._area()
        s2 = s * z / self.zoom
        self.zoom = z
        self.centre = (q[0] + (a.center().x() - pos.x()) / s2, q[1] + (a.center().y() - pos.y()) / s2)
        self.update()
        e.accept()

    def mouseDoubleClickEvent(self, e):
        self.reset_zoom()

    def mousePressEvent(self, e):
        if self.img is None or e.button() != Qt.MouseButton.LeftButton:
            return super().mousePressEvent(e)
        r = self.image_rect()
        p = e.position()
        if r.contains(p) and self._area().contains(p):
            s = r.width() / self.img.width()
            self.picked.emit((p.x() - r.x()) / s, (p.y() - r.y()) / s)

    def paintEvent(self, e):
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        p.fillRect(self.rect(), THEME.color('viewport'))
        p.setFont(sans(11))
        p.setPen(THEME.color('ink2'))
        p.drawText(QRectF(8, 2, self.width() - 16, 18), Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter,
                   self.caption)
        if self.img is None:
            p.setPen(THEME.color('muted'))
            p.drawText(QRectF(self.rect()).adjusted(24, 24, -24, -24),
                       Qt.AlignmentFlag.AlignCenter | Qt.TextFlag.TextWordWrap, self.empty)
            return
        r = self.image_rect()
        p.setClipRect(self._area())
        p.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform, bool(self.zoom < 2.0))   # zoomed in: the pixels
        p.drawImage(r, self.img)
        s = r.width() / self.img.width()
        ox, oy = r.x(), r.y()
        if self.rings:
            pen = QPen(THEME.color('strong'), 1, Qt.PenStyle.DotLine)
            c = r.center()
            for rad, text in self.rings:
                p.setPen(pen)
                p.setBrush(Qt.BrushStyle.NoBrush)
                p.drawEllipse(c, rad * s, rad * s)
                p.setPen(THEME.color('muted'))
                p.drawText(QPointF(c.x() + 3, c.y() - rad * s - 3), text)
            p.setPen(QPen(THEME.color('ink2'), 1.5))
            p.drawLine(QPointF(c.x(), c.y() + 7), QPointF(c.x(), c.y() - 9))      # the rig's origin, forward up
            p.drawLine(QPointF(c.x() - 5, c.y() - 4), QPointF(c.x(), c.y() - 9))
            p.drawLine(QPointF(c.x() + 5, c.y() - 4), QPointF(c.x(), c.y() - 9))
        p.setFont(sans(10))
        fm = p.fontMetrics()
        for sh in sorted(self.shapes, key=lambda x: x.get('hi', False)):   # the highlighted one on top
            col = QColor(sh['color'])
            hi = sh.get('hi', False)
            if hi:
                col = THEME.color('ink')
            pen = QPen(col, 2.6 if hi else 1.3, Qt.PenStyle.DashLine if sh.get('dashed') else Qt.PenStyle.SolidLine)
            p.setPen(pen)
            p.setBrush(Qt.BrushStyle.NoBrush)
            if 'rect' in sh:
                x0, y0, x1, y1 = sh['rect']
                box = QRectF(ox + x0 * s, oy + y0 * s, (x1 - x0) * s, (y1 - y0) * s)
                p.drawRect(box)
                at = box.topLeft()
            else:
                poly = QPolygonF([QPointF(ox + x * s, oy + y * s) for x, y in sh['poly']])
                p.drawPolygon(poly)
                at = poly.boundingRect().topLeft()
            if sh.get('label'):
                tw = fm.horizontalAdvance(sh['label']) + 6
                chip = QRectF(at.x(), at.y() - 15, tw, 14)
                if chip.top() < max(r.top(), self.TOP):
                    chip.moveTop(at.y() + 1)
                p.fillRect(chip, QColor(0, 0, 0, 170))
                p.setPen(col if not hi else QColor('#ffffff'))
                p.drawText(chip, Qt.AlignmentFlag.AlignCenter, sh['label'])


class GTLegend(QWidget):
    """What the layer's colours mean: swatches with names (classes, lidar categories, the box styles) or a colour bar
    with its log scale (range)."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.rows: list = []             # [(QColor | None, text, dashed)]
        self.bar = None                  # (lut (n, 3) uint8 far→near, ticks [(t, text)], none_color, title)
        self.note = ''
        self.setMinimumHeight(60)
        THEME.changed.connect(self._theme_changed)

    def _theme_changed(self, *_):
        self.update()

    ROWS = 8                             # rows shown; more: "and N more" (the tooltip lists them all)

    def set_rows(self, rows, note=''):
        self.rows, self.bar, self.note = list(rows), None, note
        n = min(len(self.rows), self.ROWS) + (len(self.rows) > self.ROWS)
        self.setFixedHeight(int(6 + 15 * n + (16 if note else 0)))
        self.setToolTip('\n'.join([note] * bool(note) + [t for _, t, _ in self.rows]))
        self.update()

    def set_bar(self, lut, ticks, none_color, title):
        self.rows, self.bar, self.note = [], (lut, ticks, none_color, title), ''
        self.setFixedHeight(76)
        self.setToolTip(title)
        self.update()

    def paintEvent(self, e):
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        p.setFont(sans(11))
        y = 4.0
        if self.bar is not None:
            lut, ticks, none_c, title = self.bar
            p.setPen(THEME.color('ink2'))
            p.drawText(QRectF(4, y, self.width() - 8, 14), Qt.AlignmentFlag.AlignVCenter, title)
            y += 18
            x0, x1 = 6.0, self.width() - 10.0
            n = len(lut)
            for k in range(n):
                c = lut[k]
                p.fillRect(QRectF(x0 + (x1 - x0) * k / n, y, (x1 - x0) / n + 1, 10), QColor(int(c[0]), int(c[1]),
                                                                                            int(c[2])))
            p.setPen(THEME.color('muted'))
            p.setFont(mono(10))
            for t, text in ticks:
                x = x0 + (x1 - x0) * t
                p.drawLine(QPointF(x, y + 10), QPointF(x, y + 13))
                tw = p.fontMetrics().horizontalAdvance(text)
                p.drawText(QPointF(min(max(x - tw / 2, 0), self.width() - tw - 2), y + 24), text)
            y += 32
            p.setFont(sans(11))
            p.fillRect(QRectF(6, y + 2, 10, 10), none_c)
            p.setPen(THEME.color('ink2'))
            p.drawText(QRectF(22, y, self.width() - 26, 14), Qt.AlignmentFlag.AlignVCenter,
                       f'nothing within {GT.RANGE_MAX_M:.0f} m')
            return
        for c, text, dashed in self.rows[:self.ROWS]:
            if c is not None:
                if dashed is None:
                    p.fillRect(QRectF(6, y + 2, 10, 10), c)
                else:
                    p.setPen(QPen(c, 1.5, Qt.PenStyle.DashLine if dashed else Qt.PenStyle.SolidLine))
                    p.setBrush(Qt.BrushStyle.NoBrush)
                    p.drawRect(QRectF(6.5, y + 2.5, 14, 9))
            p.setPen(THEME.color('ink2'))
            p.drawText(QRectF(26, y, self.width() - 30, 14), Qt.AlignmentFlag.AlignVCenter, text)
            y += 15
        if len(self.rows) > self.ROWS:
            p.setPen(THEME.color('muted'))
            p.drawText(QRectF(26, y, self.width() - 30, 14), Qt.AlignmentFlag.AlignVCenter,
                       f'and {len(self.rows) - self.ROWS} more (hover)')
            y += 15
        if self.note:
            p.setPen(THEME.color('muted'))
            p.drawText(QRectF(4, y + 1, self.width() - 8, 14), Qt.AlignmentFlag.AlignVCenter,
                       p.fontMetrics().elidedText(self.note, Qt.TextElideMode.ElideRight, int(self.width() - 8)))


def _num(text, v, tip=None) -> QTableWidgetItem:
    it = QTableWidgetItem(text)
    it.setData(Qt.ItemDataRole.UserRole + 1, v)
    it.setFont(mono(12))
    it.setTextAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
    if tip:
        it.setToolTip(tip)
    return it


class GroundTruthView(QWidget):
    """A drive's ground truth (set_run): camera, frame, layer, boxes, opacity; the legend; the object table."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.doc = None                  # the index (core.groundtruth.read), '_root' its drive
        self.out_dir = ''
        self.selected = None             # the highlighted object's id
        self._cache = OrderedDict()      # (tag, k, what) -> array
        self._syncing = False
        self._shown = None               # (tag, k, layer) of the frame the table lists
        self.timer = QTimer(self)
        self.timer.setSingleShot(True)
        self.timer.setInterval(30)       # a dragged slider shows the frame it rests on
        self.timer.timeout.connect(self.show_frame)
        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 4, 0, 0)
        lay.setSpacing(4)
        row = QHBoxLayout()
        row.setSpacing(6)
        self.cam = QComboBox()
        self.cam.setToolTip('The camera whose frames are shown')
        self.cam.currentIndexChanged.connect(lambda _i: self._camera_changed())
        row.addWidget(self.cam)
        self.layer = SegmentedControl(LAYERS, 'image')
        for k, b in self.layer.buttons.items():
            b.setToolTip(LAYER_TIP[k])
        self.layer.changed.connect(lambda _k: self._layer_changed())
        row.addWidget(self.layer)
        self.boxes = SegmentedControl(BOXES, 'tight')
        self.boxes.setToolTip(BOX_TIP)
        for b in self.boxes.buttons.values():
            b.setToolTip(BOX_TIP)
        self.boxes.changed.connect(lambda _k: self.show_frame())
        row.addWidget(self.boxes)
        row.addSpacing(6)
        self.op_label = label('Opacity', ink2=True)
        row.addWidget(self.op_label)
        self.opacity = QSlider(Qt.Orientation.Horizontal)
        self.opacity.setRange(0, 100)
        self.opacity.setValue(100)
        self.opacity.setFixedWidth(90)
        self.opacity.setToolTip('The label layer over the camera image: 0 % the image alone, 100 % the labels alone')
        self.opacity.valueChanged.connect(lambda _v: self.timer.start())
        row.addWidget(self.opacity)
        row.addStretch(1)
        lay.addLayout(row)
        frow = QHBoxLayout()
        frow.setSpacing(6)
        self.slider = QSlider(Qt.Orientation.Horizontal)
        self.slider.setToolTip('The frame (each camera frame of the drive has its ground truth)')
        self.slider.valueChanged.connect(lambda _v: self.timer.start())
        frow.addWidget(self.slider, 1)
        self.frame_label = label('', muted=True, small=True)
        self.frame_label.setMinimumWidth(170)
        frow.addWidget(self.frame_label)
        lay.addLayout(frow)
        split = QSplitter(Qt.Orientation.Horizontal)
        split.setHandleWidth(2)
        self.canvas = GTCanvas()
        self.canvas.picked.connect(self._picked)
        split.addWidget(self.canvas)
        side = QWidget()
        sl = QVBoxLayout(side)
        sl.setContentsMargins(4, 0, 0, 0)
        sl.setSpacing(4)
        self.legend = GTLegend()
        sl.addWidget(self.legend)
        self.table = QTableWidget(0, len(OBJ_COLS))
        for c, (h, tip) in enumerate(OBJ_COLS):
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
        hh = self.table.horizontalHeader()
        hh.setSectionResizeMode(0, QHeaderView.ResizeMode.Stretch)
        for c in range(1, len(OBJ_COLS)):
            hh.setSectionResizeMode(c, QHeaderView.ResizeMode.ResizeToContents)
        self.table.itemSelectionChanged.connect(self._row_picked)
        sl.addWidget(self.table, 1)
        self.table_note = label('', muted=True, small=True, wrap=True)
        sl.addWidget(self.table_note)
        split.addWidget(side)
        split.setSizes([720, 360])
        lay.addWidget(split, 1)
        THEME.changed.connect(self._theme_changed)
        self._enable()

    # ------------------------------------------------------------------ the drive
    def set_run(self, out_dir) -> bool:
        """Show the ground truth of the drive in out_dir ('' clears); False when it has none (said in the view)."""
        self.out_dir = out_dir or ''
        self.doc = GT.read(out_dir) if out_dir else None
        self._cache.clear()
        self.selected = None
        self._shown = None
        self._syncing = True
        try:
            self.cam.clear()
            if self.doc is not None:
                for tag, c in self.doc.get('cameras', {}).items():
                    if c.get('frames'):
                        self.cam.addItem(f'{tag}  ({len(c["frames"])} frames)', tag)
            self.slider.setValue(0)
        finally:
            self._syncing = False
        self.canvas.empty = EMPTY_NO_RUN if not out_dir else (EMPTY_NO_GT if self.doc is None else
                                                               'This drive\'s ground truth has no camera frame.')
        self._camera_changed()
        return self.doc is not None

    def has_ground_truth(self) -> bool:
        return self.doc is not None

    def cameras(self) -> list:
        return [self.cam.itemData(i) for i in range(self.cam.count())]

    def _frames(self) -> list:
        tag = self.cam.currentData()
        return GT.frames(self.doc, tag) if self.doc is not None and tag else []

    def _lidar_tag(self):
        for tag, L in (self.doc or {}).get('lidars', {}).items():
            if any(s.get('lidarseg') for s in L.get('sweeps', [])):
                return tag
        return None

    def _enable(self):
        fr = self._frames()
        f0 = fr[0] if fr else {}
        have = {'image': bool(fr), 'semantic': bool(f0.get('semantic')), 'instance': bool(f0.get('instance')),
                'range': bool(f0.get('range')), 'lidar': bool(fr) and self._lidar_tag() is not None}
        for k, b in self.layer.buttons.items():
            b.setEnabled(have[k])
            b.setToolTip(LAYER_TIP[k] if have[k] else LAYER_TIP[k] + ' (not in this drive\'s ground truth)')
        if not have.get(self.layer.value() or 'image'):
            self.layer.set_value('image')
        on = bool(fr)
        for w in (self.cam, self.slider, self.boxes, self.opacity, self.table):
            w.setEnabled(on)
        for b in self.boxes.buttons.values():
            b.setEnabled(on and bool(f0.get('objects') is not None))
        self.opacity.setEnabled(on and self.layer.value() in ('semantic', 'instance', 'range'))
        self.op_label.setEnabled(self.opacity.isEnabled())

    def _camera_changed(self):
        if self._syncing:
            return
        fr = self._frames()
        k = min(self.slider.value(), max(len(fr) - 1, 0))
        self._syncing = True
        self.slider.setRange(0, max(len(fr) - 1, 0))
        self.slider.setValue(k)
        self._syncing = False
        self.selected = None
        self._enable()
        self.show_frame()

    def _layer_changed(self):
        self._enable()
        self.show_frame()

    # ------------------------------------------------------------------ public controls (tests, the clip)
    def set_camera(self, tag) -> bool:
        i = self.cam.findData(tag)
        if i < 0:
            return False
        self.cam.setCurrentIndex(i)
        self._camera_changed()
        return True

    def set_layer(self, key) -> bool:
        b = self.layer.buttons.get(key)
        if b is None or not b.isEnabled():
            return False
        self.layer.set_value(key)
        self._layer_changed()
        return True

    def set_boxes(self, mode):
        self.boxes.set_value(mode)
        self.show_frame()

    def set_opacity(self, pct):
        self.opacity.setValue(int(pct))
        self.show_frame()

    def set_frame(self, k):
        self.slider.setValue(int(k))
        self.show_frame()

    def frame_index(self) -> int:
        return self.slider.value()

    def busiest_frame(self, min_px=150) -> int:
        """The frame of the camera holding the most objects of at least min_px visible pixels (the first of a tie)."""
        fr = self._frames()
        n = [sum(1 for o in f.get('objects', []) if (o.get('pixels') or 0) >= min_px) for f in fr]
        return int(np.argmax(n)) if n else 0

    def select_object(self, oid):
        """Highlight object oid (None: none): its row selected, its boxes and pixels drawn over the others."""
        self.selected = None if oid is None else int(oid)
        self._syncing = True
        try:
            self.table.clearSelection()
            for r in range(self.table.rowCount()):
                if self.table.item(r, 1).data(Qt.ItemDataRole.UserRole + 1) == self.selected:
                    self.table.selectRow(r)
                    self.table.scrollToItem(self.table.item(r, 0))
                    break
        finally:
            self._syncing = False
        self.show_frame()

    # ------------------------------------------------------------------ loading (lazy, a few frames kept)
    def _get(self, k, what):
        tag = self.cam.currentData()
        key = (tag, k, what)
        if key in self._cache:
            self._cache.move_to_end(key)
            return self._cache[key]
        f = self._frames()[k]
        try:
            a = {'image': GT.load_image, 'semantic': GT.load_semantic, 'instance': GT.load_instance,
                 'range': GT.load_range}[what](self.doc, f)
        except (OSError, KeyError, ValueError):
            a = None
        self._cache[key] = a
        while len(self._cache) > CACHE_FRAMES * 3:
            self._cache.popitem(last=False)
        return a

    def _lidar(self, ts):
        tag = self._lidar_tag()
        sw = GT.sweep_near(self.doc, tag, ts) if tag else None
        if sw is None:
            return tag, None, None, None
        key = ('lidar', tag, sw['sample_data_token'])
        if key not in self._cache:
            try:
                P = GT.lidar_points(self.doc, sw)
                L = GT.load_lidarseg(self.doc, sw)
                if len(L) != len(P):
                    raise ValueError('labels and points differ in number')
                self._cache[key] = (P, L)
            except (OSError, KeyError, ValueError):
                self._cache[key] = None
            while len(self._cache) > CACHE_FRAMES * 3:
                self._cache.popitem(last=False)
        got = self._cache[key]
        return tag, sw, (got[0] if got else None), (got[1] if got else None)

    # ------------------------------------------------------------------ showing a frame
    def _theme_changed(self, *_):
        if self.doc is not None:
            self.show_frame()

    def show_frame(self):
        self.timer.stop()
        fr = self._frames()
        if not fr:
            self.canvas.set_view(None, (), '')
            self.frame_label.setText('')
            self.legend.set_rows([])
            self.table.setRowCount(0)
            self.table_note.setText('')
            return
        k = min(max(self.slider.value(), 0), len(fr) - 1)
        f = fr[k]
        tag = self.cam.currentData()
        lay = self.layer.value() or 'image'
        t0 = int(fr[0]['timestamp'])
        t = (int(f['timestamp']) - t0) / 1e6
        self.frame_label.setText(f'frame {k + 1} of {len(fr)} · {t:.2f} s')
        objs = sorted(f.get('objects') or [], key=lambda o: (o.get('range_m') is None, o.get('range_m') or 0.0))
        if self._shown != (tag, k) or self.table.rowCount() != len(objs):
            self._fill_table(objs, lay)              # (a selected object out of this frame stays selected)
            self._shown = (tag, k)
        else:
            self._colour_rows(objs, lay)
        if lay == 'lidar':
            self._show_lidar(f, objs, t)
            return
        img = self._get(k, 'image')
        if img is None:
            self.canvas.set_view(None, (), f'{tag} · the image of this frame does not read')
            return
        a = self.opacity.value() / 100.0
        out = img
        if lay == 'semantic':
            sem = self._get(k, 'semantic')
            if sem is not None:
                out = self._blend(img, GT.colour_semantic(sem), a)
                cl = GT.classes_in(sem)
                self.legend.set_rows([(QColor(*c['color']), f'{c["name"].replace("_", " ")}  {100 * c["share"]:.1f} %',
                                       None) for c in cl], f'{len(cl)} classes in this frame, by share of its pixels')
        elif lay == 'instance':
            inst = self._get(k, 'instance')
            if inst is not None:
                out = self._blend(img, GT.colour_instance(inst), a)
                n_obj = len(np.unique(inst)) - (1 if (inst == 0).any() else 0)
                self.legend.set_rows([(QColor(0, 0, 0), 'no object', None)],
                                     f'{n_obj} objects with pixels in this frame, one colour each (the same in '
                                     'every frame: see the table)')
        elif lay == 'range':
            rng = self._get(k, 'range')
            if rng is not None:
                near, far = GT.range_limits(rng)
                th = CM._theme(None)
                lut = RANGE_LUT                 # the palette's blue steps, dark (far) to light (near)
                tpos = GT.range_position(rng, near, far)
                none_rgb = np.round(CM.hex_rgb(CM.BELOW_MIN[th]) * 255).astype(np.uint8)
                idx = np.clip(np.rint(np.nan_to_num(tpos, nan=0.0) * (len(lut) - 1)), 0, len(lut) - 1).astype(np.intp)
                col = lut[idx]
                col[np.isnan(tpos)] = none_rgb
                out = self._blend(img, col, a)
                lnf = math.log(far) - math.log(near)
                ticks = [(1.0 - (math.log(far) - math.log(v)) / lnf, f'{v:g}') for v in GT.scale_ticks(near, far)]
                # the bar runs near (left) to far (right): the lut reversed (its strong end is the near one)
                self.legend.set_bar(lut[::-1], [(x, s) for x, s in ticks], QColor(*(int(c) for c in none_rgb)),
                                    f'Range along each pixel\'s ray, m (log scale, {near:g}–{far:g} m)')
        else:
            self.legend.set_rows([(QColor(200, 200, 200), 'tight box: the visible pixels', False),
                                  (QColor(200, 200, 200), 'full box: the 3D box through the lens', True)],
                                 'Boxes in their class\'s colour (lightened)')
        if self.selected is not None and any(o['id'] == self.selected for o in objs):
            inst = self._get(k, 'instance')            # the highlighted object outlined: its layer colours kept
            if inst is not None and inst.shape[:2] == out.shape[:2]:
                out = out.copy()
                out[outline(inst == self.selected)] = 255
        shapes = self._box_shapes(objs)
        n = len(objs)
        sel = next((o for o in objs if o['id'] == self.selected), None)
        self.canvas.set_view(_qimage(out), shapes,
                             f'{tag} · {LAYERS_LABEL[lay].lower()} · {n} object{"s" * (n != 1)}'
                             + (f' · selected: {sel["class"].replace("_", " ")} {sel["id"]}' if sel else '')
                             + (f' · {100 * a:.0f} % over the image' if lay != 'image' and a < 1.0 else ''))

    @staticmethod
    def _blend(img, layer, a):
        if a >= 0.999:
            return layer
        if a <= 0.001:
            return img
        return (img.astype(np.float32) * (1.0 - a) + layer.astype(np.float32) * a).astype(np.uint8)

    def _box_shapes(self, objs) -> list:
        mode = self.boxes.value() or 'none'
        out = []
        if mode == 'none':
            return out
        for o in objs:
            hi = o['id'] == self.selected
            rgb = GT._PALETTE[GT.CLASS_ID.get(o.get('class'), 0)]
            col = _lighter(rgb)
            text = f'{o["class"].replace("_", " ")} {o["id"]}'
            tight = o.get('bbox')
            big = tight is not None and (tight[3] - tight[1]) >= LABEL_MIN_PX
            if mode in ('tight', 'both') and tight:
                out.append({'rect': tuple(tight), 'color': col, 'dashed': False, 'label': text if (big or hi) else '',
                            'hi': hi})
            if mode in ('full', 'both') and o.get('bbox_full'):
                full = o['bbox_full']
                lab = '' if mode == 'both' else (text if (full[3] - full[1]) >= LABEL_MIN_PX or hi else '')
                out.append({'rect': tuple(full), 'color': col, 'dashed': True, 'label': lab, 'hi': hi})
        return out

    def _show_lidar(self, f, objs, t):
        tag, sw, P, L = self._lidar(int(f['timestamp']))
        if P is None:
            self.canvas.set_view(None, (), f'{tag or "lidar"} · this sweep does not read')
            self.legend.set_rows([])
            return
        pal = GT.lidarseg_palette()
        if THEME.name != 'light':                  # CARLA's palette is dark (buildings, road): lifted on a dark view
            pal = np.round(pal + (255.0 - pal) * 0.4).astype(np.uint8)
        img = GT.topdown(P, pal[L], LIDAR_HALF_M, LIDAR_PX, dot=2)
        bg = np.empty_like(img)
        bg[...] = (*THEME.color('viewport').getRgb()[:3], 255)
        a = img[..., 3:4] > 0
        rgb = np.where(a, img, bg)
        s = LIDAR_PX / (2 * LIDAR_HALF_M)
        shapes = []
        for o in objs:
            b = o.get('box3d')
            if not b:
                continue
            fp = GT.box_footprint(b)
            poly = [(LIDAR_PX / 2 - y * s, LIDAR_PX / 2 - x * s) for x, y in fp]
            hi = o['id'] == self.selected
            rgb_c = GT._PALETTE[GT.CLASS_ID.get(o.get('class'), 0)]
            shapes.append({'poly': poly, 'color': _lighter(rgb_c), 'dashed': False, 'hi': hi,
                           'label': f'{o["class"].replace("_", " ")} {o["id"]}' if hi else ''})
        counts = np.bincount(L, minlength=len(pal))
        rows = [(QColor(*(int(c) for c in pal[i])), f'{GT.NUSC_NAME[i]}  {counts[i]:,}', None)
                for i in np.argsort(-counts, kind='stable') if counts[i]]
        self.legend.set_rows(rows, f'{len(P):,} points by nuScenes-lidarseg class; boxes: this frame\'s objects')
        ts = (int(sw['timestamp']) - int(self._frames()[0]['timestamp'])) / 1e6
        rings = [(r_m * s, f'{r_m:g} m') for r_m in (10, 20, 30, 40) if r_m <= LIDAR_HALF_M]
        self.canvas.set_view(_qimage(rgb), shapes, f'{tag} sweep at {ts:.2f} s (nearest this frame) · from above, '
                                                   f'±{LIDAR_HALF_M:g} m · {len(P):,} points', rings)

    # ------------------------------------------------------------------ the object table
    def _fill_table(self, objs, lay):
        self._syncing = True
        try:
            tb = self.table
            tb.setSortingEnabled(False)
            tb.clearSelection()
            tb.setRowCount(len(objs))
            for r, o in enumerate(objs):
                it = QTableWidgetItem(o.get('class', '').replace('_', ' '))
                it.setToolTip(f'{o.get("category") or ""}' + (f' · CARLA actor {o["actor_id"]}'
                                                               if o.get('actor_id') is not None else ''))
                tb.setItem(r, 0, it)
                tb.setItem(r, 1, _num(str(o['id']), int(o['id'])))
                rm = o.get('range_m')
                tb.setItem(r, 2, _num('—' if rm is None else f'{rm:.1f} m', rm))
                tr, oc = o.get('truncated'), o.get('occluded')
                tb.setItem(r, 3, _num('—' if tr is None else f'{100 * tr:.0f} %', tr))
                tb.setItem(r, 4, _num('—' if oc is None else f'{100 * oc:.0f} %', oc))
                tb.setItem(r, 5, _num(f'{o.get("pixels", 0):,}', o.get('pixels', 0)))
                if o['id'] == self.selected:
                    tb.selectRow(r)
            self._colour_rows(objs, lay)
            n = len(objs)
            self.table_note.setText(f'{n} object{"s" * (n != 1)} in this frame, nearest first: click one (or click it '
                                    'in the image) to highlight it.' if n else 'No object in this frame.')
        finally:
            self._syncing = False

    def _colour_rows(self, objs, lay):
        cols = GT.object_colours([o['id'] for o in objs]) if lay == 'instance' else None
        for r, o in enumerate(objs):
            it = self.table.item(r, 0)
            if it is None:
                continue
            rgb = cols[r] if cols is not None else GT._PALETTE[GT.CLASS_ID.get(o.get('class'), 0)]
            it.setIcon(_swatch(rgb))

    def _row_picked(self):
        if self._syncing:
            return
        rows = self.table.selectionModel().selectedRows()
        if not rows:
            self.selected = None
        else:
            self.selected = self.table.item(rows[0].row(), 1).data(Qt.ItemDataRole.UserRole + 1)
        self.show_frame()

    def _picked(self, x, y):
        """A click in the image: the object under it (the instance image), or none."""
        if (self.layer.value() or 'image') == 'lidar':
            return
        k = self.slider.value()
        inst = self._get(k, 'instance') if self._frames() and self._frames()[0].get('instance') else None
        if inst is None:
            return
        i, j = int(y), int(x)
        if not (0 <= i < inst.shape[0] and 0 <= j < inst.shape[1]):
            return
        oid = int(inst[i, j])
        self.select_object(oid if oid else None)


LAYERS_LABEL = dict(LAYERS)
