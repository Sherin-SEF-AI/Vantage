"""Footage of real CARLA drives with their ground truth, for the episode "Ground truth from CARLA" (gt040).

The drives are the ones Vantage 0.4.0 recorded in CARLA 0.9.16 for its ground-truth work (read only, never written):
the sample's data rig in Town10HD traffic and the built-in scenarios. They are shown frame by frame at real speed in
the compositions of the plain ground-truth video (one camera with its four label layers and the roof lidar from above;
the six fisheyes with their boxes), redrawn in the tour's look: the dark palette, IBM Plex, one accent colour, the
boxes and the lidar coloured as the app's Ground truth tab colours them (vantage.ui.gtview).

A FootageView is a plain widget the size of the frame that the recorder films (r.win = view). It paints its Layout at
the drive time its Clock gives, which follows the recording's frame counter, so a drive plays at real speed in the
video however fast the recording runs; the recorder's zooms, callouts, statements and figure tiles then work on it as
on the app. Targets for them (panels, an object's box) come from the layout in window coordinates.

The drives are found in the folder $VANTAGE_GT_DRIVES names (the recording machine's; never shown). Imported inside
scene() (it imports Qt).
"""
from __future__ import annotations

import json
import math
import os
from collections import OrderedDict

import numpy as np
from PyQt6.QtCore import QPointF, QRectF, Qt
from PyQt6.QtGui import QColor, QImage, QPainter, QPainterPath, QPen, QPolygonF
from PyQt6.QtWidgets import QWidget

import gfx
from gfx import ACCENT, BASE, INK, INK2, MUTED, col, draw_text, font, text_w
from vantage.core import groundtruth as GT
from vantage.ui import colormaps as CM

W, H = gfx.W, gfx.H
ENV = 'VANTAGE_GT_DRIVES'
CAM_NAMES = {'F': 'front', 'R': 'rear', 'ML': 'left mirror', 'MR': 'right mirror', 'QL': 'left quarter',
             'QR': 'right quarter'}
GRID = (('ML', 'F', 'MR'), ('QL', 'R', 'QR'))        # the six fisheyes as they sit on the car: front row, rear row
LABELLED = ('car', 'truck', 'bus', 'pedestrian', 'rider', 'bicycle', 'motorcycle', 'traffic_light', 'traffic_sign')
VEHICLES = ('car', 'truck', 'bus', 'motorcycle', 'bicycle')
RANGE_LUT = CM.ramp(CM.BLUE[::-1])                   # as the app: the blue scale, near lightest
NEAR_M, FAR_M = 1.0, 200.0                           # a fixed log scale, so the colours keep their meaning over time
LIDAR_HALF_M = 40.0
PANEL_FILL = '#0d0d0c'
EGO_RGB = (0, 255, 255)


def drives_root() -> str:
    root = os.environ.get(ENV, '')
    if not root or not os.path.isdir(root):
        raise RuntimeError(f'set {ENV} to the folder holding the CARLA ground-truth drives')
    return root


def lighter(rgb, k=0.45) -> QColor:
    """A class colour lifted towards white (the app's boxes: a navy car still reads on an image)."""
    r, g, b = (int(round(c + (255 - c) * k)) for c in rgb)
    return QColor(r, g, b)


def class_rgb(name) -> tuple:
    return tuple(int(v) for v in GT._PALETTE[GT.CLASS_ID.get(name, 0)])


def qimage(a: np.ndarray) -> QImage:
    a = np.ascontiguousarray(a)
    fmt = QImage.Format.Format_RGBA8888 if a.shape[2] == 4 else QImage.Format.Format_RGB888
    return QImage(a.data, a.shape[1], a.shape[0], a.strides[0], fmt).copy()


def lidar_palette() -> np.ndarray:
    """nuScenes-lidarseg colours lifted for a dark view, as the app's Ground truth tab does."""
    pal = GT.lidarseg_palette()
    return np.round(pal + (255.0 - pal) * 0.4).astype(np.uint8)


# ====================================================================== a drive
class Drive:
    """A drive's ground truth (index.json, run.json) and its layers per frame as images, loaded when first painted and
    kept in a small cache."""

    def __init__(self, name, cache_mb=700):
        self.name = name
        self.root = os.path.join(drives_root(), name)
        self.doc = GT.read(self.root)
        if self.doc is None:
            raise RuntimeError(f'{name}: no ground truth')
        with open(os.path.join(self.root, 'run.json'), encoding='utf-8') as fh:
            self.run = json.load(fh)
        cams = self.doc['cameras']
        self.tags = [t for t in ('F', 'MR', 'QR', 'R', 'QL', 'ML') if t in cams]
        self.t0 = min(int(cams[t]['frames'][0]['timestamp']) for t in self.tags)
        self.lidar = next(iter(self.doc.get('lidars') or {}), None)
        self._cache = OrderedDict()
        self._bytes = 0
        self._max = cache_mb * 2 ** 20
        self.pal = lidar_palette()

    # ---------------------------------------------------------------- frames
    def frames(self, tag):
        return GT.frames(self.doc, tag)

    def duration(self, tag='F'):
        fr = self.frames(tag)
        return (int(fr[-1]['timestamp']) - self.t0) / 1e6

    def index_at(self, tag, t) -> int:
        """The frame on show at drive time t (s from the first frame): the last captured at or before it."""
        fr = self.frames(tag)
        want = self.t0 + t * 1e6 + 1.0
        k = int(np.searchsorted([int(f['timestamp']) for f in fr], want, side='right')) - 1
        return min(max(k, 0), len(fr) - 1)

    def frame(self, tag, k):
        return self.frames(tag)[k]

    def time_of(self, tag, k) -> float:
        return (int(self.frame(tag, k)['timestamp']) - self.t0) / 1e6

    def size(self, tag):
        c = self.doc['cameras'][tag]
        return int(c['width']), int(c['height'])

    def objects(self, tag, k, min_px=40, classes=LABELLED):
        return [o for o in self.frame(tag, k)['objects'] if o['class'] in classes and o['pixels'] >= min_px]

    def objects_all(self, t, min_px=40, classes=LABELLED):
        """The objects every camera shows at drive time t, each once (their 3D boxes are the same in every
        camera's frame of a step)."""
        seen, out = set(), []
        for c in self.tags:
            for o in self.objects(c, self.index_at(c, t), min_px, classes):
                if o['id'] not in seen:
                    seen.add(o['id'])
                    out.append(o)
        return out

    # ---------------------------------------------------------------- cached layers
    def _get(self, key, make):
        """A cached value (least recently used first out, at most cache_mb in all)."""
        if key in self._cache:
            self._cache.move_to_end(key)
            return self._cache[key][0]
        v = make()
        n = _nbytes(v)
        self._cache[key] = (v, n)
        self._bytes += n
        while self._bytes > self._max and len(self._cache) > 1:
            _, (_, m) = self._cache.popitem(last=False)
            self._bytes -= m
        return v

    def release(self):
        self._cache.clear()
        self._bytes = 0

    def raw(self, tag, k, kind):
        """The frame's array: 'image' RGB, 'semantic' class ids, 'instance' object ids, 'range' metres."""
        f = self.frame(tag, k)
        load = {'image': GT.load_image, 'semantic': GT.load_semantic, 'instance': GT.load_instance,
                'range': GT.load_range}[kind]
        return self._get(('raw', tag, k, kind), lambda: load(self.doc, f))

    def layer(self, tag, k, kind) -> QImage:
        """The frame's layer in colour as a QImage: 'image', 'semantic', 'instance', 'range'."""
        def make():
            a = self.raw(tag, k, kind)
            if kind == 'semantic':
                a = GT.colour_semantic(a)
            elif kind == 'instance':
                a = colour_instance_over(a, self.raw(tag, k, 'image'))
            elif kind == 'range':
                a = colour_range(a)
            return qimage(a)
        return self._get(('img', tag, k, kind), make)

    def sweep(self, t):
        """The labelled sweep nearest drive time t: (sweep, points (n, 3) in the vehicle frame, lidarseg labels)."""
        if self.lidar is None:
            return None
        sw = GT.sweep_near(self.doc, self.lidar, self.t0 + int(round(t * 1e6)))
        if sw is None:
            return None

        def make():
            return sw, GT.lidar_points(self.doc, sw), GT.load_lidarseg(self.doc, sw)
        return self._get(('sweep', sw['sample_data_token']), make)

    def topdown(self, t, px, half_m=LIDAR_HALF_M):
        s = self.sweep(t)
        if s is None:
            return None, None
        sw, P, L = s
        img = self._get(('top', sw['sample_data_token'], px, half_m),
                        lambda: qimage(GT.topdown(P, self.pal[L], half_m, px, dot=2)))
        return img, s


def _nbytes(v) -> int:
    if isinstance(v, QImage):
        return v.sizeInBytes()
    if isinstance(v, np.ndarray):
        return v.nbytes
    if isinstance(v, tuple):
        return sum(_nbytes(x) for x in v)
    return 1024


def colour_instance_over(inst, img, dim=0.22) -> np.ndarray:
    """The instance colours (GT.colour_instance: one per object id, the same in every frame); where there is no
    object, the camera image in grey, dimmed, so the objects read in their place."""
    out = GT.colour_instance(inst)
    none = inst == 0
    grey = (img.astype(np.float32) @ np.array([0.299, 0.587, 0.114], np.float32)) * dim
    out[none] = np.repeat(grey[none][:, None], 3, axis=1).astype(np.uint8)
    return out


def colour_range(rng) -> np.ndarray:
    t = GT.range_position(rng, NEAR_M, FAR_M)
    out = np.empty(rng.shape + (3,), np.uint8)
    out[...] = (56, 56, 53)                                      # nothing within 256 m: the app's grey
    ok = np.isfinite(t)
    out[ok] = RANGE_LUT[np.round(t[ok] * 255).astype(int)]
    return out


# ====================================================================== the clock
class Clock:
    """Drive time from the recorder's frame counter: play(t) runs from t at real speed (rate 1), pause() holds."""

    def __init__(self, rec):
        self.rec = rec
        self.t_base, self.f_base, self.running, self.rate = 0.0, 0, False, 1.0

    def now(self) -> float:
        if not self.running:
            return self.t_base
        return self.t_base + (self.rec.f - self.f_base) / self.rec.fps * self.rate

    def play(self, t=None, rate=1.0):
        self.t_base = self.now() if t is None else float(t)
        self.f_base, self.running, self.rate = self.rec.f, True, rate

    def pause(self, t=None):
        self.t_base = self.now() if t is None else float(t)
        self.running = False

    def seek(self, t):
        self.t_base, self.f_base = float(t), self.rec.f


# ====================================================================== the view
class FootageView(QWidget):
    """The frame-sized widget the recorder films: it paints self.layout at the clock's drive time."""

    def __init__(self, rec):
        super().__init__()
        self.setAttribute(Qt.WidgetAttribute.WA_OpaquePaintEvent)
        self.resize(W, H)
        self.clock = Clock(rec)
        self.layout_ = None

    def use(self, layout):
        self.layout_ = layout
        layout.view = self
        self.update()
        return layout

    def t(self):
        return self.clock.now()

    def paintEvent(self, e):
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        p.setRenderHint(QPainter.RenderHint.TextAntialiasing)
        p.fillRect(QRectF(0, 0, W, H), col(BASE))
        if self.layout_ is not None:
            self.layout_.paint(p, self.t())
        p.end()


# ====================================================================== painting helpers
def panel_image(p: QPainter, r: QRectF, img: QImage | None, smooth=True, src: QRectF | None = None, radius=6.0,
                alpha=1.0):
    """An image in a rounded panel with a hairline edge (nearest-neighbour when not smooth: labels stay crisp)."""
    p.save()
    path = QPainterPath()
    path.addRoundedRect(r, radius, radius)
    p.setClipPath(path, Qt.ClipOperation.IntersectClip if p.hasClipping() else Qt.ClipOperation.ReplaceClip)
    p.fillRect(r, col(PANEL_FILL))
    if img is not None:
        p.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform, smooth)
        p.setOpacity(alpha)
        if src is None:
            p.drawImage(r, img)
        else:
            p.drawImage(r, img, src)
    p.restore()
    p.save()
    p.setPen(QPen(col('#ffffff', 0.10), 1.0))
    p.setBrush(Qt.BrushStyle.NoBrush)
    p.drawRoundedRect(r, radius, radius)
    p.restore()


def caption(p: QPainter, x, y, eyebrow, text='', alpha=1.0):
    """A panel's caption: an accent eyebrow (mono, upper case) and a line in the body type, baseline y."""
    fe = font(14, 'semibold', mono=True, spacing=1.6)
    draw_text(p, x, y, eyebrow.upper(), fe, ACCENT, alpha)
    if text:
        draw_text(p, x + text_w(fe, eyebrow.upper()) + 14, y, text, font(17, 'regular'), INK2, alpha)


def header(p: QPainter, left: str, right: str = ''):
    fe = font(15, 'semibold', mono=True, spacing=2.0)
    draw_text(p, 32, 44, left.upper(), fe, MUTED)
    if right:
        fr = font(17, 'medium', mono=True, spacing=0.5)
        draw_text(p, W - 32 - text_w(fr, right), 44, right, fr, INK2)


def map_box(r: QRectF, size, bbox) -> QRectF:
    """An image-pixel box [x0, y0, x1, y1] (pixel edges) in a panel showing the whole image."""
    sx, sy = r.width() / size[0], r.height() / size[1]
    x0, y0, x1, y1 = bbox
    return QRectF(r.x() + x0 * sx, r.y() + y0 * sy, (x1 - x0) * sx, (y1 - y0) * sy)


def draw_boxes(p: QPainter, r: QRectF, size, objs, full=False, labels=True, reveal=None, hi=None, label_px=14,
               width=1.6, clip=None):
    """The objects' tight boxes (solid) or full boxes (dashed, the 3D box through the lens) in their class colours
    (lightened, as the app), labelled "class id" when tall enough; reveal(i) -> 0..1 draws box i in (the cold open);
    hi: an object id drawn thick and white. r is where the whole image would be; clip, the panel shown (a crop)."""
    p.save()
    p.setClipRect(clip if clip is not None else r)
    fl = font(12, 'medium')
    p.setFont(fl)
    sy = r.height() / size[1]
    order = sorted(range(len(objs)), key=lambda i: -objs[i]['range_m'])          # far first: near boxes on top
    for i in order:
        o = objs[i]
        bb = o.get('bbox_full') if full else o.get('bbox')
        if not bb:
            continue
        k = 1.0 if reveal is None else reveal(i)
        if k <= 0.0:
            continue
        rect = map_box(r, size, bb)
        if k < 1.0:
            c = rect.center()
            s = 1.0 + 0.6 * (1.0 - gfx.out_cubic(k))
            rect = QRectF(c.x() - rect.width() * s / 2, c.y() - rect.height() * s / 2, rect.width() * s,
                          rect.height() * s)
        is_hi = hi is not None and o['id'] == hi
        c = QColor('#ffffff') if is_hi else lighter(class_rgb(o['class']))
        c.setAlphaF(min(1.0, k * 1.4))
        pen = QPen(c, (3.0 if is_hi else width), Qt.PenStyle.DashLine if full else Qt.PenStyle.SolidLine)
        p.setPen(pen)
        p.setBrush(Qt.BrushStyle.NoBrush)
        p.drawRect(rect)
        tall = (bb[3] - bb[1]) * sy >= label_px
        if labels and (tall or is_hi) and k >= 0.999:
            t = f'{o["class"].replace("_", " ")} {o["id"]}'
            tw = text_w(fl, t) + 8
            chip = QRectF(rect.x(), rect.y() - 17, tw, 16)
            if chip.top() < (clip if clip is not None else r).top():
                chip.moveTop(rect.y() + 1)
            p.fillRect(chip, QColor(0, 0, 0, 175))
            p.setPen(c)
            p.drawText(chip, Qt.AlignmentFlag.AlignCenter, t)
    p.restore()


def crop_rects(panel: QRectF, size, crop) -> QRectF:
    """Where the whole image lies when the crop (x, y, w, h in image px) of it fills the panel."""
    s = panel.width() / crop[2]
    return QRectF(panel.x() - crop[0] * s, panel.y() - crop[1] * s, size[0] * s, size[1] * s)


def draw_footprints(p: QPainter, r: QRectF, objs, half_m=LIDAR_HALF_M, hi=None, ego=True):
    """The objects' 3D boxes seen from above in a top view (x forward up, the rig's origin at the centre)."""
    s = r.width() / (2 * half_m)
    cx, cy = r.center().x(), r.center().y()
    p.save()
    p.setClipRect(r)
    for o in objs:
        b = o.get('box3d')
        if not b:
            continue
        fp = GT.box_footprint(b)
        poly = QPolygonF([QPointF(cx - y * s, cy - x * s) for x, y in fp])
        is_hi = hi is not None and o['id'] == hi
        p.setPen(QPen(QColor('#ffffff') if is_hi else lighter(class_rgb(o['class'])), 2.6 if is_hi else 1.6))
        p.setBrush(Qt.BrushStyle.NoBrush)
        p.drawPolygon(poly)
    if ego:            # the rig's origin, forward up (as the app's top view)
        p.setPen(QPen(col(INK2), 1.6))
        p.drawLine(QPointF(cx, cy + 8), QPointF(cx, cy - 10))
        p.drawLine(QPointF(cx - 6, cy - 4), QPointF(cx, cy - 10))
        p.drawLine(QPointF(cx + 6, cy - 4), QPointF(cx, cy - 10))
    p.restore()


def draw_rings(p: QPainter, r: QRectF, half_m=LIDAR_HALF_M, rings=(10, 20, 30, 40)):
    s = r.width() / (2 * half_m)
    c = r.center()
    p.save()
    p.setClipRect(r)
    f = font(12, 'regular', mono=True)
    for m in rings:
        if m > half_m:
            continue
        p.setPen(QPen(col('#ffffff', 0.13), 1.0, Qt.PenStyle.DotLine))
        p.setBrush(Qt.BrushStyle.NoBrush)
        p.drawEllipse(c, m * s, m * s)
        draw_text(p, c.x() + 4, c.y() - m * s - 4, f'{m} m', f, MUTED, 0.9)
    p.restore()


def draw_bev(p: QPainter, r: QRectF, drive: Drive, t, objs, half_m=LIDAR_HALF_M, hi=None, footprints=True):
    """The labelled lidar sweep nearest t from above in its nuScenes-lidarseg classes, with rings and the objects'
    footprints; returns the sweep (sweep, points, labels) or None."""
    px = int(round(r.width() * 2))
    img, s = drive.topdown(t, px, half_m)
    panel_image(p, r, None)
    draw_rings(p, r, half_m)
    if img is not None:
        p.save()
        p.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform, True)
        p.drawImage(r, img)
        p.restore()
    if footprints:
        draw_footprints(p, r, objs, half_m, hi)
    return s


def range_bar(p: QPainter, x, y, w):
    """The range colour bar, near (left, light) to far, with its log ticks (m)."""
    bar = RANGE_LUT[np.clip(((math.log(FAR_M) - np.log(np.geomspace(NEAR_M, FAR_M, max(int(w), 2)))) /
                             (math.log(FAR_M) - math.log(NEAR_M)) * 255).round().astype(int), 0, 255)]
    p.save()
    p.fillRect(QRectF(x - 10, y - 10, w + 20, 46), QColor(0, 0, 0, 150))
    for i in range(int(w)):
        p.fillRect(QRectF(x + i, y, 1.2, 9), QColor(*(int(v) for v in bar[i])))
    f = font(12, 'regular', mono=True)
    for m in (1, 2, 5, 10, 20, 50, 100, 200):
        px = x + (math.log(m) - math.log(NEAR_M)) / (math.log(FAR_M) - math.log(NEAR_M)) * (w - 1)
        s = f'{m}'
        draw_text(p, px - text_w(f, s) / 2, y + 26, s, f, INK2)
    p.restore()


def legend_rows(p: QPainter, x, y, w, rows, cols=2, row_h=26, title='', alpha=1.0):
    """Swatches with text in columns (rows: [(rgb, text)]); returns the y below them."""
    if title:
        draw_text(p, x, y, title, font(17, 'medium'), INK, alpha)
        y += 14
    f = font(15, 'regular')
    cw = w / cols
    n = len(rows)
    per = math.ceil(n / cols) if n else 0
    for i, (rgb, text) in enumerate(rows):
        c, rr = divmod(i, per) if per else (0, 0)
        cx, cy = x + c * cw, y + 12 + rr * row_h
        p.fillRect(QRectF(cx, cy, 14, 14), QColor(*rgb))
        draw_text(p, cx + 24, cy + 12, text, f, INK2, alpha)
    return y + 12 + per * row_h


def class_rows(sem, min_share=0.002, n=10):
    return [(c['color'], f'{c["name"].replace("_", " ")}  {c["share"] * 100:.1f} %')
            for c in GT.classes_in(sem) if c['share'] >= min_share][:n]


def lidar_rows(labels, pal, n=10):
    counts = np.bincount(labels, minlength=len(pal))
    return [(tuple(int(v) for v in pal[i]), f'{GT.NUSC_NAME[i]}  {counts[i]:,}')
            for i in np.argsort(-counts, kind='stable') if counts[i]][:n]


# ====================================================================== layouts
class Layout:
    """A composition painted at drive time t; rect(name) gives its parts as targets (window coordinates)."""
    view = None
    rects: dict = {}

    def rect(self, name) -> QRectF:
        return QRectF(self.rects[name])

    def tlabel(self, t, what='t = ', digits=1):
        """The drive time for the header; 'paused' while the clock holds."""
        paused = self.view is not None and not self.view.clock.running
        return ('paused · ' if paused else '') + f'{what}{t:.{digits}f} s'

    def paint(self, p, t):
        raise NotImplementedError


class SixGrid(Layout):
    """The six fisheyes as they sit on the car (front row ML F MR, rear row QL R QR) with their objects' tight boxes;
    reveal: {tag: drive time the camera's boxes start to draw in} (the cold open), else all shown."""
    CW, CH, GAP, TOP = 560, 448, 32, 108

    def __init__(self, drive: Drive, boxes=True, layer='image', reveal=None, title='', right=True, min_px=40):
        self.d, self.boxes, self.layer, self.reveal, self.title, self.right, self.min_px = \
            drive, boxes, layer, reveal, title, right, min_px
        x0 = (W - 3 * self.CW - 2 * self.GAP) / 2
        self.rects = {}
        for r_, row in enumerate(GRID):
            for c_, tag in enumerate(row):
                self.rects[tag] = QRectF(x0 + c_ * (self.CW + self.GAP), self.TOP + r_ * (self.CH + 52), self.CW,
                                         self.CH)

    def paint(self, p, t):
        d = self.d
        header(p, self.title or 'Six fisheyes · real CARLA 0.9.16 drive', self.tlabel(t) if self.right else '')
        for tag, r in self.rects.items():
            if tag not in d.tags:
                continue
            k = d.index_at(tag, t)
            panel_image(p, r, d.layer(tag, k, self.layer), smooth=self.layer == 'image')
            caption(p, r.x(), r.y() - 12, tag, CAM_NAMES[tag])
            if self.boxes:
                objs = d.objects(tag, k, self.min_px)
                rv = None
                if self.reveal is not None:
                    t0 = self.reveal.get(tag)
                    if t0 is None:
                        continue
                    rv = (lambda i, t0=t0: gfx.clamp((t - t0 - 0.05 * i) / 0.35))
                draw_boxes(p, r, d.size(tag), objs, labels=False, reveal=rv)


class Single(Layout):
    """One fisheye large; wipe: (from layer, to layer, 0..1) a vertical wipe between two layers (the cold open)."""

    def __init__(self, drive: Drive, tag, layer='image', boxes=False, title='', wipe=None, head=True, labels=True):
        self.d, self.tag, self.layer, self.boxes, self.title, self.wipe, self.head, self.labels = \
            drive, tag, layer, boxes, title, wipe, head, labels
        w_, h_ = 1250, 1000
        self.rects = {'image': QRectF((W - w_) / 2, 52, w_, h_)}

    def paint(self, p, t):
        d, tag = self.d, self.tag
        k = d.index_at(tag, t)
        r = self.rects['image']
        if self.head:
            header(p, self.title or f'{tag} · {CAM_NAMES[tag]}', self.tlabel(t))
        if self.wipe is None:
            panel_image(p, r, d.layer(tag, k, self.layer), smooth=self.layer == 'image')
        else:
            a, b, u = self.wipe(t)
            panel_image(p, r, d.layer(tag, k, a), smooth=a == 'image')
            if u > 0:
                w_ = r.width() * u
                p.save()
                p.setClipRect(QRectF(r.x(), r.y(), w_, r.height()))
                panel_image(p, r, d.layer(tag, k, b), smooth=b == 'image')
                p.restore()
                p.fillRect(QRectF(r.x() + w_ - 1, r.y(), 2, r.height()), col('#ffffff', 0.85))
        if self.boxes:
            draw_boxes(p, r, d.size(tag), d.objects(tag, k), labels=self.labels)


class Quad(Layout):
    """One camera with its four layers (the image with its tight boxes, semantic, instance, range) and the roof lidar
    from above with the objects' footprints, the classes in the frame below it."""
    PW, PH = 576, 461

    def __init__(self, drive: Drive, tag, title='', full=False, hi=None):
        self.d, self.tag, self.title, self.full, self.hi = drive, tag, title, full, hi
        x0, x1 = 32, 32 + self.PW + 24
        y0, y1 = 92, 92 + self.PH + 50
        self.rects = {'boxes': QRectF(x0, y0, self.PW, self.PH), 'semantic': QRectF(x1, y0, self.PW, self.PH),
                      'instance': QRectF(x0, y1, self.PW, self.PH), 'range': QRectF(x1, y1, self.PW, self.PH),
                      'lidar': QRectF(1232, 92, 656, 656), 'legend': QRectF(1232, 778, 656, 280)}
        self.hi_classes = None             # a set of class names drawn on the semantic panel's legend in the accent

    def frame_index(self, t):
        return self.d.index_at(self.tag, t)

    def paint(self, p, t):
        d, tag = self.d, self.tag
        k = d.index_at(tag, t)
        R = self.rects
        tt = d.time_of(tag, k)
        header(p, self.title or f'{tag} · {CAM_NAMES[tag]} camera · real CARLA 0.9.16 drive',
               f'frame {k + 1} of {len(d.frames(tag))} · ' + self.tlabel(tt))
        objs = d.objects(tag, k)
        size = d.size(tag)
        panel_image(p, R['boxes'], d.layer(tag, k, 'image'))
        draw_boxes(p, R['boxes'], size, objs, hi=self.hi)
        if self.full:
            draw_boxes(p, R['boxes'], size, objs, full=True, labels=False)
        panel_image(p, R['semantic'], d.layer(tag, k, 'semantic'), smooth=False)
        panel_image(p, R['instance'], d.layer(tag, k, 'instance'), smooth=False)
        panel_image(p, R['range'], d.layer(tag, k, 'range'), smooth=False)
        rb = R['range']
        range_bar(p, rb.right() - 300, rb.bottom() - 44, 280)
        caption(p, R['boxes'].x(), R['boxes'].y() - 12, 'Boxes', 'each object\'s tight box from its own pixels')
        caption(p, R['semantic'].x(), R['semantic'].y() - 12, 'Semantic', 'CARLA\'s classes, the ego cyan')
        caption(p, R['instance'].x(), R['instance'].y() - 12, 'Instances', 'one colour per object, every frame')
        caption(p, R['range'].x(), R['range'].y() - 12, 'Range', 'metres along each pixel\'s ray, log scale')
        caption(p, R['lidar'].x(), R['lidar'].y() - 12, 'Roof lidar', 'from above, nuScenes-lidarseg, ±40 m')
        s = draw_bev(p, R['lidar'], d, tt, objs, hi=self.hi)
        lg = R['legend']
        rows = class_rows(d.raw(tag, k, 'semantic'))
        y = legend_rows(p, lg.x(), lg.y(), lg.width(), rows, title='Classes in this frame, by share of its pixels')
        counts = {}
        for o in objs:
            counts[o['class']] = counts.get(o['class'], 0) + 1
        txt = ', '.join(f'{n} {c.replace("_", " ")}' for c, n in sorted(counts.items(), key=lambda kv: -kv[1]))
        draw_text(p, lg.x(), y + 18, f'{len(objs)} objects boxed: {txt}' if objs else 'no objects in view',
                  font(15, 'regular'), MUTED)
        if s is not None:
            draw_text(p, lg.x(), y + 42, f'lidar sweep: {len(s[2]):,} points, each labelled', font(15, 'regular'),
                      MUTED)

    def obj_rect(self, panel, oid):
        """A target: object oid's tight box in a panel at the frame on show (None when it is not in it)."""
        def f():
            t = self.view.t() if self.view is not None else 0.0
            k = self.d.index_at(self.tag, t)
            o = next((o for o in self.d.frame(self.tag, k)['objects'] if o['id'] == oid), None)
            if o is None:
                return None
            return map_box(self.rects[panel], self.d.size(self.tag), o['bbox'])
        return f


class Pair(Layout):
    """A camera's image and its semantic labels side by side, large; loupe: an image rectangle (px) shown magnified
    on both, nearest-neighbour, so the label edges are seen pixel by pixel."""
    PW, PH = 860, 688

    def __init__(self, drive: Drive, tag, title=''):
        self.d, self.tag, self.title = drive, tag, title
        x0 = (W - 2 * self.PW - 40) / 2
        y0 = (H - self.PH) / 2 + 24
        self.rects = {'image': QRectF(x0, y0, self.PW, self.PH), 'semantic': QRectF(x0 + self.PW + 40, y0, self.PW,
                                                                                     self.PH)}
        self.loupe = None                 # (x0, y0, x1, y1) image px
        self.loupe_k = None               # fn() -> 0..1 (its entrance)
        self.loupe_size = (400, 320)

    def loupe_rects(self):
        """(source rect on each panel, loupe rect on each panel) in window coordinates."""
        size = self.d.size(self.tag)
        out = {}
        for name in ('image', 'semantic'):
            r = self.rects[name]
            src = map_box(r, size, self.loupe)
            lw, lh = self.loupe_size
            lx = r.right() - lw - 20 if src.center().x() < r.center().x() else r.x() + 20
            ly = r.bottom() - lh - 20 if src.center().y() < r.center().y() else r.y() + 20
            out[name] = (src, QRectF(lx, ly, lw, lh))
        return out

    def paint(self, p, t):
        d, tag = self.d, self.tag
        k = d.index_at(tag, t)
        tt = d.time_of(tag, k)
        header(p, self.title or f'{tag} · {CAM_NAMES[tag]} camera · real CARLA 0.9.16 drive', self.tlabel(tt))
        R = self.rects
        panel_image(p, R['image'], d.layer(tag, k, 'image'))
        panel_image(p, R['semantic'], d.layer(tag, k, 'semantic'), smooth=False)
        w, h = d.size(tag)
        caption(p, R['image'].x(), R['image'].y() - 14, 'Colour image', f'{w} × {h}, through the lens\'s remap')
        caption(p, R['semantic'].x(), R['semantic'].y() - 14, 'Semantic labels',
                'the same remap, the nearest view pixel')
        if self.loupe is None:
            return
        kk = 1.0 if self.loupe_k is None else self.loupe_k()
        if kk <= 0:
            return
        x0, y0, x1, y1 = self.loupe
        src_px = QRectF(x0, y0, x1 - x0, y1 - y0)
        for name, (src, lr) in self.loupe_rects().items():
            p.save()
            p.setOpacity(kk)
            p.setPen(QPen(col(ACCENT), 2.0))
            p.setBrush(Qt.BrushStyle.NoBrush)
            p.drawRect(src)
            p.setPen(QPen(col(ACCENT, 0.6), 1.2))
            p.drawLine(src.topLeft(), lr.topLeft())
            p.drawLine(src.bottomRight(), lr.bottomRight())
            p.restore()
            p.save()
            p.setOpacity(kk)
            panel_image(p, lr, d.layer(tag, k, name), smooth=False, src=src_px, radius=4)
            p.setPen(QPen(col(ACCENT), 2.0))
            p.drawRoundedRect(lr, 4, 4)
            f = font(13, 'semibold', mono=True, spacing=1.2)
            z = lr.width() / (x1 - x0)
            s = f'{z:.0f}× · EACH SQUARE ONE PIXEL'
            p.fillRect(QRectF(lr.x(), lr.bottom() - 26, text_w(f, s) + 20, 26), QColor(0, 0, 0, 190))
            draw_text(p, lr.x() + 10, lr.bottom() - 8, s, f, INK)
            p.restore()


class LidarBig(Layout):
    """The roof lidar large from above, with the objects' footprints; beside it the front camera with its boxes and
    the sweep's points by nuScenes-lidarseg category."""

    def __init__(self, drive: Drive, tag='F', title='', half_m=LIDAR_HALF_M):
        self.d, self.tag, self.title, self.half = drive, tag, title, half_m
        self.rects = {'lidar': QRectF(900, 92, 968, 968), 'camera': QRectF(48, 92, 800, 640),
                      'legend': QRectF(48, 772, 800, 290)}
        self.hi = None
        self.facts = []                    # lines under the legend (the drive's run.json, quoted)

    def paint(self, p, t):
        d, tag = self.d, self.tag
        k = d.index_at(tag, t)
        tt = d.time_of(tag, k)
        R = self.rects
        objs = d.objects(tag, k)
        s = d.sweep(tt)
        header(p, self.title or 'Roof lidar · real CARLA 0.9.16 drive',
               self.tlabel((int(s[0]["timestamp"]) - d.t0) / 1e6, 'sweep at ') if s else '')
        panel_image(p, R['camera'], d.layer(tag, k, 'image'))
        draw_boxes(p, R['camera'], d.size(tag), objs, labels=False, hi=self.hi)
        caption(p, R['camera'].x(), R['camera'].y() - 12, tag, f'{CAM_NAMES[tag]} camera, its objects boxed')
        draw_bev(p, R['lidar'], d, tt, d.objects_all(tt, 40, VEHICLES + ('pedestrian',)), self.half, hi=self.hi)
        caption(p, R['lidar'].x(), R['lidar'].y() - 12, 'Roof lidar from above',
                f'each point in its nuScenes-lidarseg class · ±{self.half:g} m')
        if s is not None:
            lg = R['legend']
            y = legend_rows(p, lg.x(), lg.y(), lg.width(), lidar_rows(s[2], d.pal, 8), cols=2,
                            title=f'This sweep: {len(s[2]):,} points by nuScenes-lidarseg category')
            self.rects['points'] = QRectF(lg.x() - 4, lg.y() - 20, 640, 30)
            if self.facts:
                f = font(15, 'regular', mono=True)
                y += 26
                for i, line in enumerate(self.facts):
                    draw_text(p, lg.x(), y, line, f, INK2 if i else MUTED)
                    self.rects[f'fact{i}'] = QRectF(lg.x() - 4, y - 18, text_w(f, line) + 8, 26)
                    y += 26


class Scenario(Layout):
    """A scenario drive: the front camera with its boxes, the lidar from above, and the scenario's events from its
    run.json on a timeline with a play head (each event lit once passed)."""

    def __init__(self, drive: Drive, label, events, span=(0.0, 6.0), tag='F', hi=None, title=''):
        self.d, self.label, self.events, self.span, self.tag, self.hi, self.title = \
            drive, label, events, span, tag, hi, title
        self.crop = (0, 0, 960, 350)       # the fisheye's upper part: the road ahead (below it, the bonnet's shadow)
        self.rects = {'camera': QRectF(40, 92, 1100, 401), 'semantic': QRectF(40, 548, 1100, 401),
                      'lidar': QRectF(1176, 92, 704, 704), 'events': QRectF(1176, 830, 704, 150),
                      'timeline': QRectF(60, 1010, 1800, 40)}

    def paint(self, p, t):
        d, tag = self.d, self.tag
        k = d.index_at(tag, t)
        tt = d.time_of(tag, k)
        R = self.rects
        objs = d.objects(tag, k)
        header(p, self.title or f'Scenario · {self.label} · seed {d.run["scene"]["seed"]}', self.tlabel(tt, digits=2))
        cr = QRectF(*self.crop)
        panel_image(p, R['camera'], d.layer(tag, k, 'image'), src=cr)
        draw_boxes(p, self.full_rect('camera'), d.size(tag), objs, hi=self.hi, clip=R['camera'])
        caption(p, R['camera'].x(), R['camera'].y() - 12, tag, f'{CAM_NAMES[tag]} camera, the road ahead, every object '
                'boxed')
        panel_image(p, R['semantic'], d.layer(tag, k, 'semantic'), smooth=False, src=cr)
        caption(p, R['semantic'].x(), R['semantic'].y() - 12, 'Semantic', 'the same view, CARLA\'s classes')
        draw_bev(p, R['lidar'], d, tt, d.objects_all(tt, 20, VEHICLES + ('pedestrian', 'rider')), 30.0, hi=self.hi)

        caption(p, R['lidar'].x(), R['lidar'].y() - 12, 'Roof lidar', 'from above, ±30 m, footprints of the objects')
        # the events so far
        ev = R['events']
        fh = font(15, 'semibold', mono=True)
        fb = font(17, 'regular')
        y = ev.y() + 20
        for e in self.events:
            if e['t'] > t + 1e-6:
                break
            draw_text(p, ev.x(), y, f'{e["ts"]} s', fh, ACCENT)
            draw_text(p, ev.x() + 96, y, e['text'], fb, INK)
            y += 30
        # the timeline
        tl = R['timeline']
        t0, t1 = self.span
        x_of = (lambda s: tl.x() + (s - t0) / (t1 - t0) * tl.width())
        y = tl.center().y()
        p.save()
        p.fillRect(QRectF(tl.x(), y - 1.5, tl.width(), 3), col('#ffffff', 0.14))
        p.fillRect(QRectF(tl.x(), y - 1.5, max(0.0, min(tl.width(), x_of(t) - tl.x())), 3), col(ACCENT, 0.85))
        f = font(12, 'regular', mono=True)
        for s in range(int(math.ceil(t0)), int(t1) + 1):
            draw_text(p, x_of(s) - 6, y + 22, f'{s} s', f, MUTED)
            p.fillRect(QRectF(x_of(s) - 0.5, y + 4, 1, 5), col('#ffffff', 0.25))
        fe = font(13, 'semibold', mono=True)
        for e in self.events:
            x = x_of(e['t'])
            done = e['t'] <= t + 1e-6
            p.setPen(Qt.PenStyle.NoPen)
            p.setBrush(col(ACCENT if done else '#ffffff', 1.0 if done else 0.35))
            p.drawEllipse(QPointF(x, y), 6, 6)
            draw_text(p, x - text_w(fe, e['short']) / 2, y - 14, e['short'], fe, INK if done else MUTED)
        x = x_of(min(max(t, t0), t1))
        p.setBrush(col(INK))
        p.drawRoundedRect(QRectF(x - 2, y - 12, 4, 24), 2, 2)
        p.restore()

    def full_rect(self, panel):
        return crop_rects(self.rects[panel], self.d.size(self.tag), self.crop)

    def obj_rect(self, panel, oid):
        """A target: object oid's tight box in a cropped panel at the frame on show (None when not in it)."""
        def f():
            t = self.view.t() if self.view is not None else 0.0
            k = self.d.index_at(self.tag, t)
            o = next((o for o in self.d.frame(self.tag, k)['objects'] if o['id'] == oid), None)
            if o is None:
                return None
            return map_box(self.full_rect(panel), self.d.size(self.tag), o['bbox']).intersected(self.rects[panel])
        return f


class BevFull(Layout):
    """The roof lidar alone from above, large and centred, with the objects' footprints (the cold open)."""

    def __init__(self, drive: Drive, title='', half_m=LIDAR_HALF_M):
        self.d, self.title, self.half = drive, title, half_m
        self.rects = {'lidar': QRectF((W - 1000) / 2, 64, 1000, 1000)}

    def paint(self, p, t):
        d = self.d
        header(p, self.title or 'Roof lidar from above · nuScenes-lidarseg', self.tlabel(t))
        draw_bev(p, self.rects['lidar'], d, t, d.objects_all(t, 40, VEHICLES + ('pedestrian',)), self.half)


# ====================================================================== for the scenes
def start(stage, r) -> FootageView:
    """A FootageView filmed by the recorder (hidden and deleted when the segment ends); the pointer off."""
    v = FootageView(r)
    v.show()
    stage.extra.append(v)
    for _ in range(3):
        stage.app.processEvents()
    r.win = v
    r.origin = QPointF(0, 0)
    r.cursor_on = False
    return v


def play_until(r, view, t_end, rate=1.0):
    """Play the drive from where the clock is to drive time t_end at real speed (rate 1), recording; then hold there
    (paused)."""
    if view.clock.now() > t_end + 0.05:
        raise RuntimeError(f'play_until({t_end}): the drive is already at {view.clock.now():.2f} s')
    if not view.clock.running:
        view.clock.play(rate=rate)
    while view.clock.now() < t_end - 1e-6:
        r.tick()
    view.clock.pause(t_end)
    view.update()


def fmt(x, decimals=2) -> str:
    """A measured value as the narration says it: rounded half up from its decimal text (3.025 -> 3.03, not the
    binary float's 3.02)."""
    from decimal import ROUND_HALF_UP, Decimal
    return str(Decimal(str(x)).quantize(Decimal(1).scaleb(-decimals), rounding=ROUND_HALF_UP))


def render(view, layout, t, size=(W, H)) -> QImage:
    """The layout painted at drive time t as a frame-sized image (backdrops, stills), the clock left as it was."""
    from PyQt6.QtCore import Qt as _Qt
    old, running, tb, fb = view.layout_, view.clock.running, view.clock.t_base, view.clock.f_base
    view.use(layout)
    view.clock.pause(t)
    img = view.grab().toImage().scaled(size[0], size[1], _Qt.AspectRatioMode.IgnoreAspectRatio,
                                       _Qt.TransformationMode.SmoothTransformation)
    view.clock.running, view.clock.t_base, view.clock.f_base = running, tb, fb
    if old is not None:
        view.use(old)
    return img.convertToFormat(QImage.Format.Format_RGB32)
