"""B7 · Export dialog with live preview."""
from __future__ import annotations

import os

from PyQt6.QtCore import Qt, QBuffer, QIODevice, QUrl, QTimer, QSize
from PyQt6.QtGui import QDesktopServices, QFont, QImage
from PyQt6.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QGridLayout, QLabel, QLineEdit, QComboBox, QCheckBox,
                             QTabBar, QPlainTextEdit, QFileDialog, QScrollArea, QFrame, QPushButton, QMessageBox)

from ...core import exporters as ex
from ...core.project import FRAME_PRESETS
from ...core.rules import blocks_export
from .. import icons
from ..theme import THEME, mono, sans, STATUS
from ..widgets import label, button, SegmentedControl, hline, vline, StatusLine, ROW_H
from .base import VDialog

MENU_TO_FMT = {'bev_lut': 'bev_lut', 'mount_sheet_pdf': 'mount_sheet_pdf', 'report_html': 'report_html', 'report_pdf': 'report_pdf',
               'extrinsics_json': 'extrinsics_json', 'extrinsics_yaml': 'extrinsics_yaml', 'extrinsics_csv': 'extrinsics_csv',
               'opencv_yaml': 'opencv_yaml', 'ros2': 'ros2', 'jetson': 'jetson_json', 'img_viewport': 'viewport_png',
               'img_bev': 'bev_png', 'img_cams': 'camera_views_png', 'dxf': 'dxf', 'step': 'step'}
ICONS = {'Documents': 'report', 'Extrinsics': 'axes', 'Calibration and robotics': 'ecu', 'Surround view': 'bowl', 'Images': 'bev',
         'Mount points': 'mount-zone'}
DEFAULT_ON = {'mount_sheet_pdf', 'report_html', 'extrinsics_json', 'ros2', 'jetson_json'}


def png_bytes(img: QImage) -> bytes:
    buf = QBuffer()
    buf.open(QIODevice.OpenModeFlag.WriteOnly)
    img.save(buf, 'PNG')
    return bytes(buf.data())


class FormatRow(QWidget):
    def __init__(self, fmt, on, dlg):
        super().__init__()
        self.fmt = fmt
        self.dlg = dlg
        self.setFixedHeight(ROW_H + 1)
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        h = QHBoxLayout(self)
        h.setContentsMargins(12, 0, 12, 0)
        h.setSpacing(8)
        self.cb = QCheckBox()
        self.cb.setChecked(on)
        self.cb.toggled.connect(lambda _: dlg.changed())
        ic = QLabel()
        ic.setPixmap(icons.pixmap(ICONS.get(fmt.group, 'file'), THEME.tok['ink2'], 12))
        self.t = QLabel(fmt.label)
        e = label(fmt.ext_label, muted=True, mono_=True, small=True)
        h.addWidget(self.cb)
        h.addWidget(ic)
        h.addWidget(self.t, 1)
        h.addWidget(e)
        self.selected = False

    def mousePressEvent(self, e):
        self.dlg.select(self.fmt.id)

    def paintEvent(self, e):
        if self.selected:
            from PyQt6.QtGui import QPainter
            p = QPainter(self)
            p.fillRect(self.rect(), THEME.color('sel'))
            p.fillRect(0, 0, 2, self.height(), THEME.color('ink'))
            p.end()


class ExportDialog(VDialog):
    def __init__(self, ctx, parent=None, preselect=None):
        p = ctx.store.project
        super().__init__(parent, 'Export', f'{p.name} · {p.variant.name}', (1210, 820))
        self.ctx = ctx
        self.images = {}
        self.lut = None
        self.current = None
        pre = MENU_TO_FMT.get(preselect) if preselect else None
        body = QVBoxLayout(self.body)
        body.setContentsMargins(0, 0, 0, 0)
        body.setSpacing(0)
        main = QHBoxLayout()
        main.setSpacing(0)
        # left: format list
        left = QWidget()
        left.setFixedWidth(355)
        ll = QVBoxLayout(left)
        ll.setContentsMargins(0, 8, 0, 8)
        ll.setSpacing(0)
        self.rows = {}
        for g in ex.GROUPS:
            gl = label(g, muted=True, small=True)
            gl.setContentsMargins(12, 8, 0, 2)
            ll.addWidget(gl)
            for f in ex.FORMATS:
                if f.group != g:
                    continue
                on = (f.id == pre) if pre else (f.id in DEFAULT_ON)
                r = FormatRow(f, on, self)
                ll.addWidget(r)
                self.rows[f.id] = r
        ll.addStretch(1)
        main.addWidget(left)
        main.addWidget(vline())
        # right: preview
        right = QWidget()
        rl = QVBoxLayout(right)
        rl.setContentsMargins(0, 0, 0, 0)
        rl.setSpacing(0)
        self.tabs = QTabBar()
        self.tabs.setExpanding(False)
        self.tabs.currentChanged.connect(self._tab)
        tw = QWidget()
        tw.setObjectName('base')
        tl = QHBoxLayout(tw)
        tl.setContentsMargins(0, 0, 0, 0)
        tl.addWidget(self.tabs)
        tl.addStretch(1)
        rl.addWidget(tw)
        ph = QHBoxLayout()
        ph.setContentsMargins(12, 4, 12, 4)
        self.path_lbl = label('', mono_=True, muted=True, small=True)
        self.size_lbl = label('', mono_=True, muted=True, small=True)
        ph.addWidget(self.path_lbl)
        ph.addWidget(self.size_lbl)
        ph.addStretch(1)
        ph.addWidget(label('Live preview · updates as options change', muted=True, small=True))
        rl.addLayout(ph)
        rl.addWidget(hline())
        self.preview = QPlainTextEdit()
        self.preview.setReadOnly(True)
        self.preview.setFont(mono(12))
        self.preview.setLineWrapMode(QPlainTextEdit.LineWrapMode.NoWrap)
        rl.addWidget(self.preview, 1)
        main.addWidget(right, 1)
        body.addLayout(main, 1)
        body.addWidget(hline())
        # options
        opts = QWidget()
        og = QGridLayout(opts)
        og.setContentsMargins(14, 8, 14, 8)
        og.setHorizontalSpacing(10)
        self.frame = QComboBox()
        for k, v in FRAME_PRESETS.items():
            if k != 'custom' or p.frame == 'custom':
                self.frame.addItem(v.replace('…', ''), k)
        self.frame.setCurrentIndex(max(0, self.frame.findData(p.frame)))
        self.frame.setFixedWidth(240)
        self.frame.currentIndexChanged.connect(lambda _: self.changed())
        self.units = SegmentedControl([('m', 'm'), ('mm', 'mm')], p.units, min_w=34)
        self.units.changed.connect(lambda _: self.changed())
        self.angles = SegmentedControl([('deg', 'deg'), ('rad', 'rad')], 'deg', min_w=40)
        self.angles.changed.connect(lambda _: self.changed())
        try:
            out = ex.default_out_dir(p)
        except ValueError:
            out = os.path.join(os.path.expanduser('~'), 'Documents', 'vantage-exports', ex._slug(p.name), ex._slug(p.variant.name))
        self.out = QLineEdit(ctx.settings.value('export/last_dir', '') or out)
        self.out.setProperty('mono', True)
        self.out.textChanged.connect(lambda _: self.changed(fast=True))
        browse = button('Browse…')
        browse.clicked.connect(self._browse)
        self.open_after = QCheckBox('Open the folder when done')
        self.open_after.setChecked(ctx.settings.value('export/open_after', True, type=bool))
        og.addWidget(label('Frame', ink2=True), 0, 0)
        og.addWidget(self.frame, 0, 1)
        og.addWidget(label('URDF and TF always use REP 103.', muted=True, small=True), 1, 1)
        og.addWidget(label('Units', ink2=True), 0, 2)
        uw = QHBoxLayout()
        uw.addWidget(self.units)
        uw.addSpacing(10)
        uw.addWidget(self.angles)
        og.addLayout(uw, 0, 3)
        og.addWidget(label('Override for this export only.', muted=True, small=True), 1, 3)
        og.addWidget(label('Output folder', ink2=True), 0, 4)
        oh = QHBoxLayout()
        oh.addWidget(self.out, 1)
        oh.addWidget(browse)
        og.addLayout(oh, 0, 5)
        og.addWidget(self.open_after, 1, 5)
        og.setColumnStretch(5, 1)
        body.addWidget(opts)
        # footer
        iss = ctx.analysis.issues
        top = [i for i in iss if i.severity in ('critical', 'serious')]
        self.block = blocks_export(iss)
        if top:
            i = top[0]
            n = len(top)
            txt = f'{i.severity.capitalize()} · {n} open issue{"s" if n > 1 else ""}: {i.message}'
            if self.block:
                txt = 'Export is blocked while a camera floats. ' + txt
            self.hint.setText('')
            sl = StatusLine(i.severity, txt, wrap=False)
            self.hint.parentWidget().layout().insertWidget(0, sl, 1)
        c = button('Cancel', 'ghost')
        c.clicked.connect(self.reject)
        self.go = button('Export', 'primary')
        self.go.clicked.connect(self.do_export)
        self.buttons.addWidget(c)
        self.buttons.addWidget(self.go)
        self.timer = QTimer(self)
        self.timer.setSingleShot(True)
        self.timer.setInterval(120)
        self.timer.timeout.connect(self._refresh_preview)
        self.select(pre or 'extrinsics_json')
        self.changed()

    # ------------------------------------------------------------------
    def opts(self, with_images=False):
        return ex.ExportOptions(frame=self.frame.currentData(), units=self.units.value(), angles=self.angles.value(),
                                out_dir=self.out.text().strip(), images=self.images if with_images else self.images,
                                issues=self.ctx.analysis.issues, worst_single=self.ctx.analysis.worst, lut=self.lut)

    def checked(self):
        return [f.id for f in ex.FORMATS if self.rows[f.id].cb.isChecked()]

    def changed(self, fast=False):
        ids = self.checked()
        n = 0
        p = self.ctx.store.project
        for fid in ids:
            n += len(ex._names(ex.format_by_id(fid), p, p.variant))
        self.go.setText(f'Export {n} file{"s" if n != 1 else ""}')
        self.go.setEnabled(n > 0 and not self.block)
        self.tabs.blockSignals(True)
        while self.tabs.count():
            self.tabs.removeTab(0)
        for fid in ids:
            names = ex._names(ex.format_by_id(fid), p, p.variant)
            self.tabs.addTab(os.path.basename(names[0]) + (f' +{len(names) - 1}' if len(names) > 1 else ''))
            self.tabs.setTabData(self.tabs.count() - 1, fid)
        self.tabs.blockSignals(False)
        if self.current in ids:
            self.tabs.setCurrentIndex(ids.index(self.current))
        self.timer.start()

    def select(self, fid):
        self.current = fid
        for k, r in self.rows.items():
            r.selected = k == fid
            r.update()
        ids = self.checked()
        if fid in ids:
            self.tabs.blockSignals(True)
            self.tabs.setCurrentIndex(ids.index(fid))
            self.tabs.blockSignals(False)
        self.timer.start()

    def _tab(self, i):
        fid = self.tabs.tabData(i)
        if fid:
            self.select(fid)

    def _browse(self):
        d = QFileDialog.getExistingDirectory(self, 'Output folder', self.out.text())
        if d:
            self.out.setText(d)

    def _ensure_images(self, ids):
        """Render the images the selected formats need (viewport, bird's-eye, coverage, camera views)."""
        need_reports = any(i in ids for i in ('report_html', 'report_pdf'))
        w = self.ctx.window
        if 'bev_lut' in ids and self.lut is None and self.ctx.analysis.result is not None:
            from ...core.stitch import build_lut
            with self.ctx.analysis._lock:
                self.lut = build_lut(self.ctx.store.project, self.ctx.analysis.engine, self.ctx.analysis.result, res=0.02)
        if ('viewport_png' in ids or need_reports) and 'viewport' not in self.images:
            pane = w.viewarea.panes[0]
            img = pane.vp3d.grab_image() if pane.vp3d.isVisible() else None
            if img is None or img.isNull() or img.width() < 50:
                from ..scene import MainScene
                from ..viewport3d import SoftViewport
                sc = MainScene(self.ctx)
                vp = SoftViewport(sc)
                vp.resize(1600, 1000)
                vp.frame_box(*w.viewarea._bbox(), margin=0.82)
                img = vp.grab().toImage()
                vp.deleteLater()
            self.images['viewport'] = png_bytes(img)
        for key, mode in (('bev', 'bev'), ('coverage', 'coverage')):
            if (key == 'bev' and 'bev_png' in ids) or need_reports:
                if key not in self.images:
                    from ..planview import PlanView
                    pv = PlanView(self.ctx, mode, interactive=False)
                    pv.show_tooltip = False
                    pv.resize(1600, 1100)
                    pv.fit()
                    pv.label = 'Bird’s-eye · stitched' if mode == 'bev' else 'Coverage by best camera'
                    self.images[key] = png_bytes(pv.grab().toImage())
                    pv.deleteLater()
        if 'camera_views_png' in ids or need_reports:
            from ..camview import render_camera
            p = self.ctx.store.project
            for c in p.variant.cameras:
                if c.visible and f'cam:{c.uid}' not in self.images:
                    m = p.module_for(c)
                    wpx = 960
                    hpx = int(wpx * m.height_px / m.width_px)
                    with self.ctx.analysis._lock:
                        img, _ = render_camera(p, self.ctx.analysis.engine, c, wpx, hpx, self.ctx.analysis.result)
                    self.images[f'cam:{c.uid}'] = png_bytes(img)

    def _refresh_preview(self):
        fid = self.current
        if fid is None:
            return
        p = self.ctx.store.project
        try:
            if fid in ('viewport_png', 'bev_png', 'camera_views_png', 'bev_lut'):
                self._ensure_images([fid])
            path, text, size = ex.preview(fid, p, self.ctx.analysis.result, self.opts())
        except Exception as e:
            path, text, size = '', f'Preview failed: {e}', 0
        self.path_lbl.setText(path)
        fmt = ex.format_by_id(fid)
        self.size_lbl.setText(f'   {ex._size_text(size)}' + ('   UTF-8 · LF' if fmt.kind == 'text' else ''))
        self.preview.setPlainText(text)

    def do_export(self):
        ids = self.checked()
        if not ids:
            return
        out = self.out.text().strip()
        if not out:
            QMessageBox.warning(self, 'No output folder', 'Choose an output folder.')
            return
        self.go.setEnabled(False)
        self.go.setText('Exporting…')
        try:
            self._ensure_images(ids)
            with self.ctx.analysis._lock:
                paths = ex.export(ids, self.ctx.store.project, self.ctx.analysis.result, self.opts(True))
        except Exception as e:
            self.go.setEnabled(True)
            self.changed()
            QMessageBox.critical(self, 'Export failed', str(e))
            return
        self.ctx.settings.setValue('export/last_dir', out)
        self.ctx.settings.setValue('export/open_after', self.open_after.isChecked())
        self.ctx.store.project.add_log('export', f'{len(paths)} files to {out}')
        self.ctx.msg(f'Exported {len(paths)} files to {out}', 6000)
        if self.open_after.isChecked():
            QDesktopServices.openUrl(QUrl.fromLocalFile(out))
        self.accept()
