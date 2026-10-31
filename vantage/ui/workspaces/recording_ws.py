"""Analysis ▸ Recording budget: how much data every camera, lidar and radar records, whether the links, the network
and the recorder's disks keep up, and how many hours of driving fit on one disk set. Cheap: computed again on every
edit while the page shows."""
from __future__ import annotations

from PyQt6.QtCore import QPointF, QRectF, Qt
from PyQt6.QtGui import QColor, QFont, QPainter, QPainterPath, QPen
from PyQt6.QtWidgets import (QAbstractItemView, QFileDialog, QGridLayout, QHBoxLayout, QHeaderView, QScrollArea,
                             QSplitter, QTableWidget, QTableWidgetItem, QToolTip, QVBoxLayout, QWidget)

from ...core import recording as RB
from .. import icons
from ..charts import ChartBase
from ..theme import THEME, STATUS, mono, sans
from ..widgets import (fit_width, ROW_H, ScrubSpinBox, SegmentedControl, StatusLine, ToggleSwitch, button, hline, label)

# one hue family per kind, apart from the eight camera colours; the first is the kind's colour in the 3D and plan views
# (ui.sensorview.COLORS), the others step darker and lighter so neighbouring segments stay apart
SENSOR_COLORS = {'lidar': ['#0aa5b8', '#0a6f8f', '#3fc4d3'], 'radar': ['#a0743c', '#7a5220', '#d4a05a'],
                 'imu': ['#6e7f96', '#4f5d70', '#97a6ba']}
KIND_LABEL = {'camera': 'Camera', 'lidar': 'Lidar', 'radar': 'Radar', 'imu': 'IMU'}
KIND_WORD = {'camera': 'camera', 'lidar': 'lidar', 'radar': 'radar', 'imu': 'IMU'}       # in running text
SEV_WORD = {'good': 'OK', 'warning': 'Warning', 'serious': 'Serious', 'critical': 'Critical'}

# (group, key, label, unit, decimals, step, scale, tip); scale: stored value × scale = shown value
FIELDS = (
    ('Cameras', 'raw_bpp', 'Raw', 'bit/px', 0, 1.0, 1.0,
     'Bits per pixel of a raw image: the ECU\'s pixel format (16 = YUV 4:2:2, 12 = RAW12). The GMSL link budget uses '
     'the same figure'),
    ('Cameras', 'jpeg_bpp', 'JPEG', 'bit/px', 2, 0.1, 1.0,
     'Average bits per pixel of a JPEG image: an assumption, replace it with a figure measured on your encoder'),
    ('Cameras', 'h265_bpp', 'H.265', 'bit/px', 3, 0.01, 1.0,
     'Average bits per pixel of an H.265 stream: an assumption, replace it with a figure measured on your encoder'),
    ('Lidars and radars', 'point_bytes', 'Point size', 'B', 0, 1.0, 1.0,
     'Bytes per point of a decoded point cloud (x y z intensity ring time)'),
    ('Other data', 'extra_mbps', 'GNSS · IMU · CAN', 'Mbit/s', 1, 0.5, 1.0,
     'Everything else recorded with the sensors: GNSS, IMU, the vehicle\'s CAN'),
    ('Other data', 'container_overhead', 'Container', '%', 1, 0.5, 100.0,
     'The recording container\'s own overhead (message headers, indexes) as a share of everything recorded'),
    ('Recorder', 'storage_tb', 'Disk set', 'TB', 1, 1.0, 1.0, 'Capacity of one disk set in the recorder (decimal TB)'),
    ('Recorder', 'write_mb_s', 'Sustained write', 'MB/s', 0, 50.0, 1.0,
     'What the disk set writes for hours, after any SLC cache is full (decimal MB)'),
    ('Recorder', 'write_headroom', 'Write headroom', '×', 2, 0.05, 1.0,
     'The disks must sustain this many times the data rate (bursts, file system, a disk running hot)'),
    ('Recorder', 'shift_hours', 'Shift', 'h', 1, 0.5, 1.0, 'Hours of driving in one shift'),
    ('Recorder', 'uplink_gbps', 'Uplink', 'Gbit/s', 1, 1.0, 1.0,
     'The link from the network switch to the recorder that every lidar and radar shares'),
    ('Recorder', 'link_warn', 'Link margin', '%', 0, 5.0, 100.0, 'A link or the uplink busier than this is flagged'),
)


def row_color(row) -> QColor:
    """A camera's colour slot; a lidar or radar a step of its kind's hue family."""
    if row['kind'] == 'camera':
        return THEME.cam(row['slot'])
    fam = SENSOR_COLORS.get(row['kind'], SENSOR_COLORS['lidar'])
    return QColor(fam[row['slot'] % len(fam)])


def _ink_on(c: QColor) -> QColor:
    """Text colour that reads on a fill."""
    lum = 0.2126 * c.redF() + 0.7152 * c.greenF() + 0.0722 * c.blueF()
    return QColor('#0b0b0b') if lum > 0.55 else QColor('#f0efec')


# ---------------------------------------------------------------------- tiles
class Tile(QWidget):
    """One total: a muted caption, the value in mono with its unit, a note under it and, for a checked figure, its
    status as icon and word with a stripe on the left."""

    def __init__(self, caption, tip=''):
        super().__init__()
        self.caption = caption
        self.value, self.unit, self.note, self.sev = '—', '', '', None
        self.setMinimumWidth(130)
        self.setFixedHeight(76)
        if tip:
            self.setToolTip(tip)
        THEME.changed.connect(self._theme_changed)      # a bound slot: disconnected when the widget goes

    def _theme_changed(self, *_):
        self.update()

    def set(self, value, unit='', note='', sev=None):
        self.value, self.unit, self.note, self.sev = value, unit, note, sev
        self.update()

    def paintEvent(self, e):
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        r = QRectF(0.5, 0.5, self.width() - 1, self.height() - 1)
        p.setPen(QPen(THEME.color('hair'), 1))
        p.setBrush(THEME.color('raised'))
        p.drawRoundedRect(r, 4, 4)
        if self.sev:
            p.fillRect(QRectF(1, 1, 3, self.height() - 2), QColor(STATUS[self.sev]))
        x, w = 12, self.width() - 20
        p.setFont(sans(11))
        p.setPen(THEME.color('muted'))
        p.drawText(QRectF(x, 6, w, 16), Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter, self.caption)
        if self.sev:
            word = SEV_WORD[self.sev]
            tw = p.fontMetrics().horizontalAdvance(word)
            p.setPen(THEME.color('ink2'))
            p.drawText(QRectF(x + w - tw, 6, tw, 16), Qt.AlignmentFlag.AlignVCenter, word)
            p.drawPixmap(QPointF(x + w - tw - 16, 8), icons.pixmap(self.sev, STATUS[self.sev], 12))
        p.setFont(mono(20, QFont.Weight.DemiBold))
        p.setPen(THEME.color('ink'))
        fm = p.fontMetrics()
        p.drawText(QPointF(x, 48), self.value)
        vx = x + fm.horizontalAdvance(self.value) + 5
        p.setFont(mono(11))
        p.setPen(THEME.color('muted'))
        p.drawText(QPointF(vx, 48), self.unit)
        p.setFont(sans(11))
        p.setPen(THEME.color('ink2'))
        p.drawText(QRectF(x, 54, w, 16), Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter,
                   p.fontMetrics().elidedText(self.note, Qt.TextElideMode.ElideRight, int(w)))
        p.end()


# ---------------------------------------------------------------------- share bar
class ShareBar(ChartBase):
    """Each sensor's share of the recorded data as one horizontal stacked bar (2 px gaps between segments; a segment
    is never thinner than 2 px), a legend under it, and the hovered segment's figures as a tooltip."""
    BAR_Y, BAR_H, GAP, LEG_H = 24, 20, 2, 18

    def __init__(self, parent=None):
        super().__init__(parent)
        self.title = 'Share of the recorded data'
        self.segs = []                  # (label, QColor, mbps, tooltip)
        self.total = 0.0
        self._hits = []                 # (QRectF, index)
        self.setFixedHeight(self.BAR_Y + self.BAR_H + 10 + self.LEG_H)

    def set_data(self, segs, total):
        self.segs, self.total = segs, total
        self._fit()
        self.update()

    def resizeEvent(self, e):
        super().resizeEvent(e)
        self._fit()

    def _legend(self, W):
        """Legend entries placed in rows: [(x, row, text, index)]."""
        fm_w = lambda t: self.fontMetrics().horizontalAdvance(t)          # noqa: E731
        out, x, row = [], 0.0, 0
        for i, (lab, _, mbps, _) in enumerate(self.segs):
            t = f'{lab} {100 * mbps / self.total:.1f} %' if self.total > 0 else lab
            w = 13 + fm_w(t) + 14
            if x > 0 and x + w > W:
                x, row = 0.0, row + 1
            out.append((x, row, t, i))
            x += w
        return out

    def _fit(self):
        self.setFont(sans(11))
        rows = 1 + max((r for _, r, _, _ in self._legend(max(self.width(), 100))), default=0)
        h = self.BAR_Y + self.BAR_H + 10 + rows * self.LEG_H
        if h != self.height():
            self.setFixedHeight(h)

    def _widths(self, avail):
        if self.total <= 0 or not self.segs:
            return []
        raw = [max(m, 0.0) / self.total * avail for _, _, m, _ in self.segs]
        small = [w < 2.0 for w in raw]
        rest = avail - 2.0 * sum(small)
        big = sum(w for w, s in zip(raw, small) if not s) or 1.0
        return [2.0 if s else w * rest / big for w, s in zip(raw, small)]

    def paintEvent(self, e):
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        self.paint_title(p, f'{self.total:,.0f} Mbit/s' if self.total > 0 else '')
        W = float(self.width())
        self._hits = []
        segs = self.segs
        if not segs or self.total <= 0:
            p.setFont(sans(11))
            p.setPen(THEME.color('muted'))
            p.drawText(QRectF(0, self.BAR_Y, W, self.BAR_H), Qt.AlignmentFlag.AlignVCenter,
                       'Nothing is recorded: no visible camera, lidar or radar.')
            p.end()
            return
        n = len(segs)
        widths = self._widths(W - self.GAP * (n - 1))
        x = 0.0
        p.setFont(mono(11, QFont.Weight.DemiBold))
        fm = p.fontMetrics()
        for i, ((lab, col, mbps, tip), w) in enumerate(zip(segs, widths)):
            r = QRectF(x, self.BAR_Y, w, self.BAR_H)
            c = QColor(col)
            if self.hover is not None and self.hover != i:
                c.setAlphaF(0.45)
            if i == n - 1 and w > 8:                              # a 4 px rounded data end, square at the baseline
                path = QPainterPath()
                path.addRoundedRect(r, 4, 4)
                p.fillPath(path, c)
                p.fillRect(QRectF(r.left(), r.top(), min(4.0, w), r.height()), c)
            else:
                p.fillRect(r, c)
            if fm.horizontalAdvance(lab) + 10 <= w:              # direct label only where it fits
                p.setPen(_ink_on(QColor(col)))
                p.drawText(r, Qt.AlignmentFlag.AlignCenter, lab)
            self._hits.append((QRectF(x - 1, self.BAR_Y - 3, w + 2, self.BAR_H + 6), i))
            x += w + self.GAP
        p.setFont(sans(11))
        y0 = self.BAR_Y + self.BAR_H + 10
        for lx, row, text, i in self._legend(W):
            y = y0 + row * self.LEG_H
            p.fillRect(QRectF(lx, y + 5, 8, 8), segs[i][1])
            p.setPen(THEME.color('ink' if self.hover == i else 'ink2'))
            p.drawText(QRectF(lx + 13, y, W, self.LEG_H - 2), Qt.AlignmentFlag.AlignVCenter, text)
            self._hits.append((QRectF(lx, y, p.fontMetrics().horizontalAdvance(text) + 14, self.LEG_H), i))
        p.end()

    def mouseMoveEvent(self, e):
        pos = e.position()
        hit = next((i for r, i in self._hits if r.contains(pos)), None)
        if hit != self.hover:
            self.hover = hit
            self.update()
        if hit is not None:
            QToolTip.showText(e.globalPosition().toPoint(), self.segs[hit][3], self)
        else:
            QToolTip.hideText()


# ---------------------------------------------------------------------- the page
def _table(cols):
    t = QTableWidget(0, len(cols))
    t.setHorizontalHeaderLabels(cols)
    t.verticalHeader().setVisible(False)
    t.verticalHeader().setDefaultSectionSize(ROW_H)
    t.horizontalHeader().setFixedHeight(ROW_H)
    t.setShowGrid(False)
    t.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
    t.setSelectionMode(QAbstractItemView.SelectionMode.NoSelection)
    return t


TABLE_COLS = ('Sensor', 'Kind', 'Model', 'Recorded as', 'Mbit/s', 'GB/h', 'Link', 'Load')


class RecordingPage(QWidget):
    def __init__(self, ctx):
        super().__init__()
        self.ctx = ctx
        self.result: RB.RecordingResult | None = None
        self._dirty = True
        self.setObjectName('base')
        lay = QHBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        split = QSplitter(Qt.Orientation.Horizontal)
        split.setHandleWidth(2)
        split.addWidget(self._left())
        # ---------------- right: totals, shares, the table and the findings
        right = QSplitter(Qt.Orientation.Vertical)
        right.setHandleWidth(2)
        tw = QWidget()
        tw.setObjectName('panel')
        tl = QVBoxLayout(tw)
        tl.setContentsMargins(10, 8, 10, 8)
        tl.setSpacing(8)
        tiles = QHBoxLayout()
        tiles.setSpacing(8)
        self.tiles = {}
        for key, caption, tip in (
                ('rate', 'Data rate', 'Everything recorded: the sensors, GNSS/IMU/CAN and the container overhead'),
                ('hour', 'Per hour of driving', 'Data recorded per hour, and per shift'),
                ('disk', 'One disk set', 'Hours of driving until one disk set is full'),
                ('write', 'Sustained write', 'Write speed the disks need (the data rate × the headroom) against what '
                                             'they sustain'),
                ('uplink', 'Uplink', 'The lidars\' and radars\' share of the link from the switch to the recorder')):
            t = self.tiles[key] = Tile(caption, tip)
            tiles.addWidget(t, 1)
        tl.addLayout(tiles)
        self.bar = ShareBar()
        tl.addWidget(self.bar)
        tl.addWidget(label('Every sensor', bold=True))
        self.table = _table(TABLE_COLS)
        self.table.setToolTip('Mbit/s and GB/h: what the sensor records, before the container overhead. Link and load: '
                              'what its own link carries (a camera\'s GMSL2 link always carries raw pixels; a lidar '
                              'its packets) against the link\'s rate')
        hh = self.table.horizontalHeader()
        for c in range(len(TABLE_COLS)):
            hh.setSectionResizeMode(c, QHeaderView.ResizeMode.ResizeToContents)
        hh.setSectionResizeMode(3, QHeaderView.ResizeMode.Stretch)
        tl.addWidget(self.table, 1)
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
        right.setSizes([620, 260])
        split.addWidget(right)
        split.setSizes([330, 1270])
        lay.addWidget(split)
        ctx.store.changed.connect(self._on_change)
        ctx.store.project_replaced.connect(self._on_change)
        THEME.changed.connect(self._on_change)
        self.refresh()

    # ------------------------------------------------------------------ settings panel
    def _left(self):
        w = QWidget()
        w.setObjectName('panel')
        ll = QVBoxLayout(w)
        ll.setContentsMargins(8, 6, 8, 8)
        ll.setSpacing(4)
        top = QHBoxLayout()
        top.addWidget(label('Recording budget'))
        top.addStretch(1)
        top.addWidget(label('every sensor, the recorder', muted=True, small=True))
        ll.addLayout(top)
        ll.addWidget(hline())
        ll.addWidget(label('How much data the rig records, whether the links, the network and the recorder\'s disks '
                           'keep up, and how many hours of driving fit on one disk set. Rates are decimal (1 MB = '
                           '10⁶ bytes, 1 TB = 10¹² bytes).', muted=True, small=True, wrap=True))
        self.fields = {}
        group = None
        grid = None
        for g, key, text, unit, dec, step, scale, tip in FIELDS:
            if g != group:
                if grid is not None:
                    ll.addLayout(grid)
                group = g
                ll.addSpacing(6)
                ll.addWidget(label(g, bold=True))
                if g == 'Cameras':
                    self.cam_fmt = SegmentedControl([(k, RB.FORMAT_LABEL[k]) for k in RB.CAMERA_FORMATS], 'raw')
                    self.cam_fmt.setToolTip('What the recorder stores of each camera image')
                    self.cam_fmt.changed.connect(
                        lambda k: self._set(f'cameras recorded as {RB.FORMAT_LABEL[k]}', camera_format=k))
                    ll.addWidget(self.cam_fmt)
                    ll.addWidget(label('JPEG and H.265 sizes are assumptions (typical averages for driving scenes): '
                                       'replace them with figures measured on your encoder and scenes.',
                                       muted=True, small=True, wrap=True))
                elif g == 'Lidars and radars':
                    self.lidar_fmt = SegmentedControl([(k, RB.FORMAT_LABEL[k]) for k in RB.LIDAR_FORMATS], 'packets')
                    self.lidar_fmt.setToolTip('Packets: as the lidar\'s driver delivers them (the model\'s bytes per '
                                              'point). Point cloud: decoded points of the size below')
                    self.lidar_fmt.changed.connect(
                        lambda k: self._set(f'lidars recorded as {RB.FORMAT_LABEL[k].lower()}', lidar_format=k))
                    ll.addWidget(self.lidar_fmt)
                    self.radars = ToggleSwitch('Record the radars')
                    self.radars.setToolTip('Count the radars\' output in the recording and on the uplink')
                    self.radars.clicked.connect(
                        lambda on: self._set('radars recorded' if on else 'radars not recorded', include_radars=on))
                    ll.addWidget(self.radars)
                grid = QGridLayout()
                grid.setVerticalSpacing(2)
                grid.setHorizontalSpacing(6)
                grid.setColumnStretch(2, 1)
            f = ScrubSpinBox('', 'plain', dec, unit, step=step, width=110)
            f.setToolTip(tip)
            f.committed.connect(lambda v, k=key, s=scale, t=text, u=unit: self._commit(k, v / s, f'{t} {v:g} {u}'))
            a = label(text, ink2=True)
            a.setFixedWidth(118)
            a.setToolTip(tip)
            i = grid.rowCount()
            grid.addWidget(a, i, 0)
            grid.addWidget(f, i, 1)
            self.fields[key] = (f, scale)
        ll.addLayout(grid)
        ll.addStretch(1)
        self.status = label('', muted=True, small=True, wrap=True)
        ll.addWidget(self.status)
        bl = QHBoxLayout()
        b_def = self.b_defaults = button('Defaults', 'ghost', None, 'Back to the default recording settings')
        b_def.clicked.connect(self._defaults)
        bl.addWidget(b_def)
        bl.addStretch(1)
        self.b_export = button('Export…', None, 'export', 'Every sensor\'s data rate and link, the totals and the '
                                                           'findings as JSON and CSV')
        self.b_export.clicked.connect(lambda: self.export())
        bl.addWidget(self.b_export)
        ll.addLayout(bl)
        sc = QScrollArea()
        sc.setWidgetResizable(True)
        sc.setWidget(w)
        fit_width(sc)
        sc.setMinimumWidth(300)
        return sc

    @property
    def settings(self) -> RB.RecordingSettings:
        return RB.RecordingSettings.from_dict(self.ctx.store.project.analysis.recording)

    def _set(self, text, **kw):
        """Change settings as one undo step (saved with the project); False when nothing changes."""
        st = self.settings
        for k, v in kw.items():
            setattr(st, k, v)
        new = RB.RecordingSettings.from_dict(st.to_dict()).to_dict()
        if new == self.settings.to_dict():
            self._sync_fields()                # a clamped value: show what is stored
            return False

        def fn(p):
            p.analysis.recording = new
        self.ctx.store.edit('analysis', fn, f'Recording budget · {text}')
        return True

    def _commit(self, key, value, text):
        if key != 'raw_bpp':
            return self._set(text, **{key: value})
        bpp = int(round(min(max(value, 1.0), 64.0)))
        if bpp == self.ctx.store.project.ecu.bpp:
            self._sync_fields()
            return False

        def fn(p):
            p.ecu.bpp = bpp
        self.ctx.store.edit('ecu', fn, f'Raw camera format {bpp} bit/px')
        return True

    def _defaults(self):
        if self.ctx.store.project.analysis.recording:
            def fn(p):
                p.analysis.recording = {}
            self.ctx.store.edit('analysis', fn, 'Recording budget · defaults')

    def _sync_fields(self):
        st = self.settings.to_dict()
        st['raw_bpp'] = self.ctx.store.project.ecu.bpp
        for key, (f, scale) in self.fields.items():
            f.set_value(st[key] * scale)       # leaves a field the user is typing in alone
        self.cam_fmt.set_value(st['camera_format'])
        self.lidar_fmt.set_value(st['lidar_format'])
        self.radars.blockSignals(True)
        self.radars.setChecked(st['include_radars'])
        self.radars.blockSignals(False)

    # ------------------------------------------------------------------ state
    def _on_change(self, *_):
        if self.isVisible():
            self.refresh()
        else:
            self._dirty = True                 # computed when the page shows again

    def showEvent(self, e):
        super().showEvent(e)
        if self._dirty:
            self.refresh()

    def activate(self):
        if self._dirty:
            self.refresh()

    def refresh(self):
        p = self.ctx.store.project
        res = self.result = RB.budget(p)
        self._dirty = False
        self._sync_fields()
        st, t = res.settings, res.totals
        sev = {f['kind']: f['severity'] for f in reversed(res.findings)}       # the worst of each kind
        n = t['counts']
        count = lambda k, word: f'{n[k]} {word}{"" if n[k] == 1 else "s"}'
        self.tiles['rate'].set(f'{t["mbps"]:,.0f}', 'Mbit/s', f'{t["mb_s"]:,.1f} MB/s · ' + ', '.join(
            count(k, KIND_WORD[k]) for k in RB.KINDS if n[k] or k == 'camera'))
        self.tiles['hour'].set(f'{t["gb_h"]:,.0f}', 'GB/h', f'{t["tb_shift"]:,.1f} TB per {st.shift_hours:g} h shift')
        if t['mb_s'] > 0:
            self.tiles['disk'].set(f'{t["hours_per_set"]:,.1f}', 'h', f'on {st.storage_tb:g} TB · '
                                   f'{t["sets_per_shift"]} per {st.shift_hours:g} h shift', sev.get('disk'))
            self.tiles['write'].set(f'{t["write_needed_mb_s"]:,.0f}', f'of {st.write_mb_s:,.0f} MB/s',
                                    f'{t["mb_s"]:,.0f} MB/s × {st.write_headroom:g} headroom', sev.get('write'))
        else:
            self.tiles['disk'].set('—', '', 'nothing recorded')
            self.tiles['write'].set('—', '', 'nothing recorded')
        if t['networked']:
            self.tiles['uplink'].set(f'{t["uplink_load"] * 100:.0f}', '%', f'{t["uplink_mbps"]:,.1f} of '
                                     f'{t["uplink_capacity_mbps"]:,.0f} Mbit/s', sev.get('uplink'))
        else:
            self.tiles['uplink'].set('—', '', 'no lidar or radar recorded')
        # ---- shares
        segs = [(r['tag'], row_color(r), r['mbps'],
                 f'{r["tag"]} · {KIND_WORD.get(r["kind"], r["kind"])} · {r["model"]}\n{r["format"]}\n'
                 f'{r["mbps"]:,.1f} Mbit/s · {r["gb_h"]:,.1f} GB/h · {100 * r["mbps"] / t["mbps"]:.1f} % of all')
                for r in res.rows if r['mbps'] > 0] if t['mbps'] > 0 else []
        if t['mbps'] > 0:
            if t['extra_mbps'] > 0:
                segs.append(('GNSS · IMU · CAN', THEME.color('muted'), t['extra_mbps'],
                             f'GNSS, IMU and CAN\n{t["extra_mbps"]:g} Mbit/s'))
            if t['overhead_mbps'] > 0:
                segs.append(('Overhead', THEME.color('strong'), t['overhead_mbps'],
                             f'Container overhead ({st.container_overhead * 100:g} %)\n'
                             f'{t["overhead_mbps"]:,.1f} Mbit/s'))
        self.bar.set_data(segs, t['mbps'])
        # ---- the table
        tb = self.table
        extra = [('GNSS · IMU · CAN', 'Other', '', f'{t["extra_mbps"]:g} Mbit/s', t['extra_mbps'], None),
                 ('Container', 'Other', '', f'{st.container_overhead * 100:g} % of all', t['overhead_mbps'], None),
                 ('Total', '', '', f'{t["mb_s"]:,.1f} MB/s', t['mbps'], 'total')]
        tb.setRowCount(len(res.rows) + len(extra))
        right = Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter

        def cell(r, c, text, num=False, color=None, tip=None, bold=False, swatch=None):
            it = QTableWidgetItem(text)
            if num:
                it.setFont(mono(12, QFont.Weight.DemiBold if bold else None))
                it.setTextAlignment(right)
            elif bold:
                it.setFont(sans(12, QFont.Weight.DemiBold))
            if color is not None:
                it.setForeground(color)
            if swatch is not None:
                it.setData(Qt.ItemDataRole.DecorationRole, swatch)
            if tip:
                it.setToolTip(tip)
            tb.setItem(r, c, it)
        for i, r in enumerate(res.rows):
            muted = None if r['recorded'] else THEME.color('muted')
            cell(i, 0, r['tag'], swatch=row_color(r), color=muted)
            cell(i, 1, KIND_LABEL.get(r['kind'], r['kind']), color=muted)
            cell(i, 2, r['model'], color=muted)
            cell(i, 3, r['format'], color=muted)
            cell(i, 4, f'{r["mbps"]:,.1f}', True, muted)
            cell(i, 5, f'{r["gb_h"]:,.1f}', True, muted)
            cell(i, 6, r['link'] + (f' · port {r["port"]}' if r['port'] else ''),
                 tip=f'{r["link_mbps"]:,.1f} Mbit/s on the link'
                     + (f' of {r["link_capacity_mbps"]:,.0f} Mbit/s' if r['link_capacity_mbps'] else
                        ' (its rate is not known)'))
            if r['load'] is None:
                cell(i, 7, '—', True)
            else:
                cell(i, 7, f'{r["load"] * 100:.0f} %', True, QColor(STATUS[r['link_severity']]),
                     f'{SEV_WORD[r["link_severity"]]}: {r["link_mbps"]:,.1f} of {r["link_capacity_mbps"]:,.0f} Mbit/s')
        for j, (name, kind, model, fmt, mbps, what) in enumerate(extra, len(res.rows)):
            bold = what == 'total'
            cell(j, 0, name, bold=bold)
            cell(j, 1, kind)
            cell(j, 2, model)
            cell(j, 3, fmt)
            cell(j, 4, f'{mbps:,.1f}', True, bold=bold)
            cell(j, 5, f'{mbps / 8 * 3.6:,.1f}', True, bold=bold)
            if bold and t['networked']:
                cell(j, 6, 'Uplink', tip='The link from the switch to the recorder, shared by the lidars and radars')
                cell(j, 7, f'{t["uplink_load"] * 100:.0f} %', True, QColor(STATUS[sev.get('uplink', 'good')]))
            else:
                cell(j, 6, '')
                cell(j, 7, '')
        # ---- findings
        for w in self.findings:
            self.find_lay.removeWidget(w)
            w.hide()                    # until the deferred delete runs it would still paint over the new rows
            w.deleteLater()
        self.findings = []
        for f in res.findings:
            w = StatusLine(f['severity'], f['message'])
            self.find_lay.insertWidget(self.find_lay.count() - 1, w)
            self.findings.append(w)
        worst = res.worst()
        bad = sum(f['severity'] != 'good' for f in res.findings)
        self.status.setText(f'{res.variant}: ' + ('every check passes' if worst == 'good' else
                                                   f'{bad} finding{"s" if bad > 1 else ""} to look at'))

    # ------------------------------------------------------------------ export
    def export(self, folder=None):
        res = self.result
        if res is None:
            return []
        if folder is None:
            folder = QFileDialog.getExistingDirectory(self, 'Export the recording budget into')
            if not folder:
                return []
        from ...core import provenance as PV          # where the JSON came from
        paths = PV.stamp_paths(RB.export(res, self.ctx.store.project, folder), self.ctx.store.project, folder)
        self.ctx.msg(f'Exported the recording budget to {folder}', 5000)
        return paths
