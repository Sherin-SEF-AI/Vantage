"""Analysis ▸ Frame modes: the frame's tetrahedral modal analysis (core.fea). Left: the mesh, the feet, the modes
and the damping. Centre: the selected mode animated on the frame (drag to orbit, wheel to zoom). Right: the modes
(frequency, effective mass, what moves), each sensor's motion in the selected mode, its road jitter beside the
one-mode beam estimate (core.thermo_vib), the posts against their closed form, the wind load case and the findings.
The solve takes seconds to a minute: Run starts it as a background job with progress and Cancel."""
from __future__ import annotations

import math
import threading
import traceback

import numpy as np
from PyQt6.QtCore import QObject, Qt, QTimer, pyqtSignal
from PyQt6.QtGui import QImage
from PyQt6.QtWidgets import (QFileDialog, QGridLayout, QHBoxLayout, QProgressBar, QScrollArea, QSplitter, QTabWidget,
                             QVBoxLayout, QWidget)

from ...core import fea as FE
from ..theme import THEME
from ..widgets import ScrubSpinBox, SegmentedControl, ToggleSwitch, button, fit_width, hline, label
from .framing_ws import FrameView, fill, findings_box, set_findings, table

FRAMES = 16                    # animation frames per cycle
# key, label, unit, decimals, step, tip
FIELDS = (
    ('mesh_mm', 'Element size', 'mm', 0, 5.0, 'Element length along the members (smaller: more accurate, slower)'),
    ('plate_cells_mm', 'Plate cells', 'mm', 0, 5.0, 'Element size across the plates'),
    ('modes', 'Modes', '', 0, 1.0, 'How many of the lowest modes to find'),
    ('zeta', 'Damping ζ', '', 3, 0.005, 'Modal damping for the road response (bolted aluminium: 0.01-0.03)'),
    ('k_foot_z', 'Foot k vertical', 'N/mm', 0, 50.0, 'Spring feet: the roof\'s stiffness per foot, vertical (an '
                                                     'assumption: measure it)'),
    ('k_foot_xy', 'Foot k lateral', 'N/mm', 0, 25.0, 'Spring feet: the roof\'s stiffness per foot in its plane'),
)
_RGB = lambda c: (c.red(), c.green(), c.blue())       # noqa: E731


class _Sig(QObject):
    done = pyqtSignal(int, object)
    failed = pyqtSignal(int, str)
    progress = pyqtSignal(int, float)


class FEAPage(QWidget):
    def __init__(self, ctx):
        super().__init__()
        self.ctx = ctx
        self.result: FE.FEAResult | None = None
        self.busy = False
        self.stale = False
        self._gen = 0
        self._for = None
        self.mode = 0
        self.phase = 0
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
        cw = QWidget()
        cw.setObjectName('panel')
        cl = QVBoxLayout(cw)
        cl.setContentsMargins(8, 6, 8, 8)
        self.view = FrameView(message='Run the modal analysis of the frame (Analysis ▸ Frame builds it from the rig).')
        self.view.render_image = self._render
        cl.addWidget(self.view, 1)
        row = QHBoxLayout()
        self.t_anim = ToggleSwitch('Animate')
        self.t_anim.setChecked(True)
        self.t_anim.clicked.connect(lambda on: self._anim(on))
        row.addWidget(self.t_anim)
        row.addStretch(1)
        self.mode_text = label('', ink2=True)
        row.addWidget(self.mode_text)
        cl.addLayout(row)
        split.addWidget(cw)
        rw = QWidget()
        rw.setObjectName('panel')
        rl = QVBoxLayout(rw)
        rl.setContentsMargins(8, 6, 8, 8)
        self.tabs = QTabWidget()
        self.t_modes = table([('Mode', ''), ('Hz', 'Natural frequency'), ('x %', 'Effective mass fore-aft'),
                              ('y %', 'Effective mass side to side'), ('z %', 'Effective mass up and down'),
                              ('Moves most', 'The part with the largest motion'), ('How', '')], 5)
        self.t_modes.itemSelectionChanged.connect(self._mode_selected)
        self.t_dev = table([('Device', ''), ('Move mm', 'With the mode scaled to 1 mm at its largest motion'),
                            ('Turn °', 'Same scale'), ('x', ''), ('y', ''), ('z', '')], 0)
        self.t_jit = table([('Device', ''), ('FE flex ° RMS', 'Turn relative to the body on the road (this model)'),
                            ('Beam flex ° RMS', 'core.thermo_vib\'s one-mode estimate of the rig as drawn'),
                            ('FE / beam', ''), ('FE total ° RMS', 'With the body\'s pitch'),
                            ('Beam chain', 'The beam estimate\'s modes')], 5)
        self.t_beam = table([('Post', ''), ('Length mm', ''), ('Tip kg', ''), ('Beam Hz', 'Cantilever on a rigid '
                                                                                         'base (Rayleigh)'),
                             ('FE Hz', 'The nearest FE mode it moves in'), ('FE mode', '')], 0)
        self.t_wind = table([('Device', ''), ('Move mm', 'Static, in the aerodynamics page\'s load case'),
                             ('Turn °', '')], 0)
        self.tabs.addTab(self.t_modes, 'Modes')
        self.tabs.addTab(self.t_dev, 'Sensors in mode')
        self.tabs.addTab(self.t_jit, 'Road jitter')
        self.tabs.addTab(self.t_beam, 'Posts vs beam')
        self.tabs.addTab(self.t_wind, 'Wind case')
        fsc, self.find_lay = findings_box()
        self.tabs.addTab(fsc, 'Findings')
        rl.addWidget(self.tabs, 1)
        split.addWidget(rw)
        split.setSizes([300, 640, 660])
        lay.addWidget(split)
        self.timer = QTimer(self)
        self.timer.setInterval(90)
        self.timer.timeout.connect(self._tick)
        self._frames = {}
        ctx.store.changed.connect(self._on_store)
        ctx.store.project_replaced.connect(self._replaced)
        THEME.changed.connect(lambda *_: (self._frames.clear(), self.view.changed()))
        self._sync()
        self.refresh()

    # ------------------------------------------------------------------ settings panel
    def _left(self):
        w = QWidget()
        w.setObjectName('panel')
        ll = QVBoxLayout(w)
        ll.setContentsMargins(8, 6, 8, 8)
        ll.setSpacing(4)
        ll.addWidget(label('Frame modes'))
        ll.addWidget(hline())
        ll.addWidget(label('Quadratic tetrahedra of the frame, the sensors as rigid masses on their plates, the feet '
                           'fixed or on springs: its lowest modes, and the road through them to each sensor.',
                           muted=True, small=True, wrap=True))
        ll.addSpacing(4)
        ll.addWidget(label('Feet', bold=True))
        self.seg_bc = SegmentedControl([('fixed', 'Fixed'), ('spring', 'Roof springs')], 'fixed')
        self.seg_bc.setToolTip('Fixed: the feet held. Roof springs: the stiffness set below per foot (an assumption)')
        self.seg_bc.changed.connect(lambda k: self.set_option('bc', k))
        ll.addWidget(self.seg_bc)
        grid = QGridLayout()
        self.fields = {}
        for i, (key, text, unit, dec, step, tip) in enumerate(FIELDS):
            f = ScrubSpinBox('', 'plain', dec, unit, step=step, width=100)
            f.setToolTip(tip)
            f.committed.connect(lambda v, k=key: self.set_option(k, v))
            a = label(text, ink2=True)
            a.setToolTip(tip)
            grid.addWidget(a, i, 0)
            grid.addWidget(f, i, 1)
            self.fields[key] = f
        ll.addLayout(grid)
        ll.addSpacing(4)
        self.sw_conv = ToggleSwitch('Mesh convergence')
        self.sw_conv.setToolTip('Also solve at 2× and 1.4× the element size and list the frequencies')
        ll.addWidget(self.sw_conv)
        self.sw_wind = ToggleSwitch('Wind load case')
        self.sw_wind.setToolTip('Solve the static case of the aerodynamics page\'s highway speed and crosswind')
        self.sw_wind.setChecked(True)
        ll.addWidget(self.sw_wind)
        ll.addStretch(1)
        self.bar = QProgressBar()
        self.bar.setRange(0, 1000)
        self.bar.setTextVisible(False)
        self.bar.setFixedHeight(4)
        ll.addWidget(self.bar)
        self.status = label('', muted=True, small=True, wrap=True)
        ll.addWidget(self.status)
        bl = QHBoxLayout()
        self.b_run = button('Run', 'primary', 'play', 'Mesh the frame and solve its modes (a background job)')
        self.b_run.clicked.connect(lambda: self.run())
        bl.addWidget(self.b_run)
        self.b_cancel = button('Cancel', 'ghost', None, 'Stop the running solve')
        self.b_cancel.clicked.connect(self.cancel)
        bl.addWidget(self.b_cancel)
        bl.addStretch(1)
        self.b_export = button('Export…', None, 'export', 'Modes, jitter and wind as JSON and CSV, mode pictures')
        self.b_export.clicked.connect(lambda: self.export())
        bl.addWidget(self.b_export)
        ll.addLayout(bl)
        sc = QScrollArea()
        sc.setWidgetResizable(True)
        sc.setWidget(w)
        fit_width(sc)
        sc.setMinimumWidth(290)
        return sc

    @property
    def settings(self) -> FE.FEASettings:
        return FE.settings_of(self.ctx.store.project)

    def set_option(self, key, value) -> bool:
        st = self.settings.to_dict()
        d = dict(st, **{key: value})
        new = FE.FEASettings.from_dict(d).to_dict()
        if new == st:
            self._sync()
            return False

        def fn(p):
            p.analysis.fea = new
        self.ctx.store.edit('analysis', fn, f'Frame modes · {key.replace("_", " ")} {new[key]}')
        return True

    def _sync(self):
        st = self.settings
        for k, f in self.fields.items():
            f.set_value(float(getattr(st, k)))
        self.seg_bc.set_value(st.bc)
        for k in ('k_foot_z', 'k_foot_xy'):
            self.fields[k].setEnabled(st.bc == 'spring')

    # ------------------------------------------------------------------ running
    def _on_store(self, *_):
        self._sync()
        if self.result is not None and self._for is not None and \
                FE.inputs_key(self.ctx.store.project) != self._for[1]:
            self.stale = True
            self.refresh_status()

    def _replaced(self, *_):
        self._gen += 1
        self.result, self._for, self.stale, self.busy = None, None, False, False
        self._frames.clear()
        self._sync()
        self.refresh()

    def activate(self):
        self.refresh_status()

    def showEvent(self, e):
        super().showEvent(e)
        self._anim(self.t_anim.isChecked())

    def hideEvent(self, e):
        super().hideEvent(e)
        self.timer.stop()

    def run(self, quiet=False, wait=False) -> bool:
        """Start the solve in a background thread (a running one is replaced). wait: block until done (tests)."""
        p = self.ctx.store.project
        project = p.snapshot()
        key = (p, FE.inputs_key(p))
        self._gen += 1
        gen = self._gen
        self.busy = True
        self.status_text = 'Meshing the frame and solving its modes…'
        self.bar.setValue(0)
        self.refresh_status()
        conv, windc = self.sw_conv.isChecked(), self.sw_wind.isChecked()

        def work():
            try:
                wind = None
                if windc:
                    from ...core import aero as AE
                    wind = AE.run(project)
                res = FE.run(project, progress=lambda f, *_: self.sig.progress.emit(gen, f),
                             cancel=lambda: gen != self._gen, convergence=conv, wind=wind)
                self.sig.done.emit(gen, (key, res))
            except RuntimeError:
                pass
            except Exception as e:
                try:
                    self.sig.failed.emit(gen, f'{type(e).__name__}: {e}\n{traceback.format_exc()}')
                except RuntimeError:
                    pass
        th = threading.Thread(target=work, daemon=True)
        th.start()
        if wait:
            th.join()
        return True

    def cancel(self):
        if self.busy:
            self._gen += 1
            self.busy = False
            self.status_text = 'Cancelled'
            self.refresh_status()

    def _done(self, gen, out):
        if gen != self._gen:
            return
        self.busy = False
        key, res = out
        if res is None or key[0] is not self.ctx.store.project:
            self.status_text = 'Cancelled' if res is None else ''
            self.refresh_status()
            return
        self.result, self._for = res, key
        self.stale = key[1] != FE.inputs_key(self.ctx.store.project)
        self.status_text = (f'{res.variant}: {len(res.modes)} modes, {res.stats.get("dofs", 0):,} DOFs · '
                            f'{res.seconds:.1f} s')
        self.bar.setValue(1000)
        self.mode = 0
        self._frames.clear()
        self.refresh()

    def _failed(self, gen, msg):
        if gen != self._gen:
            return
        self.busy = False
        self.status_text = 'Failed: ' + msg.splitlines()[0]
        self.refresh_status()

    def refresh_status(self):
        txt = getattr(self, 'status_text', '')
        if self.stale and not self.busy and self.result is not None:
            txt = (txt + ' · ' if txt else '') + 'out of date: Run again'
        self.status.setText(txt)
        self.b_cancel.setEnabled(self.busy)
        self.b_run.setEnabled(not self.busy)
        self.b_export.setEnabled(self.result is not None and not self.busy)

    # ------------------------------------------------------------------ views
    def refresh(self):
        self.refresh_status()
        res = self.result
        if res is None:
            for t in (self.t_modes, self.t_dev, self.t_jit, self.t_beam, self.t_wind):
                t.setRowCount(0)
            set_findings(self.find_lay, [])
            self.view.changed()
            self.mode_text.setText('')
            return
        fill(self.t_modes, [[m['mode'], f'{m["f_hz"]:.2f}'] + [f'{m["eff_mass_pct"][k]:.1f}' for k in 'xyz'] +
                            [m['part'], m['motion']] for m in res.modes], numeric=(0, 1, 2, 3, 4))
        fill(self.t_jit, [[d['tag'], f'{d["flex_deg_rms"]:.5f}',
                           f'{d["beam"]["flex_deg_rms"]:.5f}' if d.get('beam') else '—',
                           f'{d["ratio_flex"]:.2f}×' if d.get('ratio_flex') else '—', f'{d["angle_deg_rms"]:.4f}',
                           '; '.join(d['beam']['chain']) if d.get('beam') else ''] for d in res.devices],
             numeric=(1, 2, 3, 4))
        fill(self.t_beam, [[b['name'], f'{b["length_mm"]:.0f}', f'{b["tip_kg"]:.2f}', f'{b["beam_f1_hz"]:.1f}',
                            f'{b["fea_hz"]:.1f}' if b['fea_hz'] else '—', b['fea_mode'] or ''] for b in res.beams],
             numeric=(1, 2, 3, 4, 5))
        wd = (res.wind or {}).get('devices', {})
        fill(self.t_wind, [[t, f'{d["move_abs_mm"]:.3f}', f'{d["turn_abs_deg"]:.4f}'] for t, d in wd.items()],
             numeric=(1, 2))
        set_findings(self.find_lay, res.findings)
        if res.modes:
            self.t_modes.selectRow(min(self.mode, len(res.modes) - 1))
        self._show_mode()

    def _mode_selected(self):
        rows = {i.row() for i in self.t_modes.selectedIndexes()}
        if rows:
            self.mode = min(rows)
            self._show_mode()

    def _show_mode(self):
        res = self.result
        if res is None or not res.modes:
            return
        m = res.modes[min(self.mode, len(res.modes) - 1)]
        self.mode_text.setText(f'Mode {m["mode"]}: {m["f_hz"]:.2f} Hz · {m["part"]} {m["motion"]}')
        self.view.overlay = f'Mode {m["mode"]} · {m["f_hz"]:.2f} Hz'
        fill(self.t_dev, [[t, f'{d["move_abs_mm"]:.3f}', f'{d["turn_abs_deg"]:.4f}'] +
                          [f'{x:+.3f}' for x in d['move_mm']] for t, d in m['devices'].items()],
             numeric=(1, 2, 3, 4, 5))
        self.view.changed()

    def _anim(self, on):
        if on and self.isVisible():
            self.timer.start()
        else:
            self.timer.stop()
            self.phase = FRAMES // 4                 # at rest: the largest deflection
            self.view.changed()

    def _tick(self):
        if self.result is None or not self.result.modes:
            return
        self.phase = (self.phase + 1) % FRAMES
        self.view.changed()

    def _render(self, w, h):
        res = self.result
        if res is None or not res.modes:
            return None
        key = (self.mode, self.phase, w, h, round(self.view.yaw, 1), round(self.view.elev, 1),
               round(self.view.zoom, 3))
        img = self._frames.get(key)
        if img is None:
            dark = THEME.color('viewport').lightness() < 128
            arr = FE.mode_image(res, self.mode, (w, h), self.view.yaw, self.view.elev,
                                math.sin(2 * math.pi * self.phase / FRAMES), zoom=self.view.zoom,
                                bg=_RGB(THEME.color('viewport')), body=(120, 122, 126) if dark else (170, 172, 176))
            arr = np.ascontiguousarray(arr)
            img = QImage(arr.data, w, h, 3 * w, QImage.Format.Format_RGB888).copy()
            if len(self._frames) > 4 * FRAMES:
                self._frames.clear()
            self._frames[key] = img
        return img

    # ------------------------------------------------------------------ export
    def export(self, folder=None):
        res = self.result
        if res is None:
            return []
        if folder is None:
            folder = QFileDialog.getExistingDirectory(self, 'Export the frame\'s modes into')
            if not folder:
                return []
        from ...core import provenance as PV
        paths = PV.stamp_paths(FE.export(res, self.ctx.store.project, folder), self.ctx.store.project, folder)
        self.ctx.msg(f'Exported the frame modes to {folder}', 5000)
        return paths
