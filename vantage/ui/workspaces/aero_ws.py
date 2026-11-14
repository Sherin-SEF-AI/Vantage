"""Analysis ▸ Aerodynamics: the drag of the roof rig and the EV range it costs, vortex shedding against the frame's
modes (lock-in) and the wind load on each sensor (core.aero). Not CFD: published bluff-body coefficients times the
rig's own areas, every one cited. Left: the speeds, the air, the vehicle and its battery. Centre: range loss and rig
drag over speed. Right: the drag of every element, the range at the two report speeds, shedding and lock-in, the
wind load case and the findings. Cheap without a frame-modes result: worked out again on every edit while the page
shows; with one (Analysis ▸ Frame modes) its modes are used for the lock-in check."""
from __future__ import annotations

from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (QFileDialog, QGridLayout, QHBoxLayout, QScrollArea, QSplitter, QTabWidget, QVBoxLayout,
                             QWidget)

from ...core import aero as AE
from ..theme import THEME
from ..widgets import ScrubSpinBox, ToggleSwitch, button, fit_width, hline, label
from .framing_ws import fill, findings_box, set_findings, table
from .thermo_vib_ws import LineChart

# group, key, label, unit, decimals, step, tip
FIELDS = (
    ('Speeds', 'speed1_kmh', 'Report speed 1', 'km/h', 0, 10.0, 'Drag and range are reported at this speed …'),
    ('Speeds', 'speed2_kmh', 'Report speed 2', 'km/h', 0, 10.0, '… and at this one'),
    ('Speeds', 'v_max_kmh', 'Top speed', 'km/h', 0, 10.0, 'Lock-in is checked up to this driving speed'),
    ('Wind load', 'load_kmh', 'Load speed', 'km/h', 0, 10.0, 'The wind load case: this speed …'),
    ('Wind load', 'crosswind_ms', 'Crosswind', 'm/s', 1, 1.0, '… with this crosswind from the left'),
    ('Air', 'air_c', 'Air', '°C', 0, 1.0, 'Air temperature (density and viscosity)'),
    ('Air', 'pressure_kpa', 'Pressure', 'kPa', 1, 0.5, 'Air pressure (lower at altitude)'),
    ('Air', 'roof_speed_factor', 'Roof speed', '×', 2, 0.05, 'Air speed over the roof as a share of the driving '
                                                            'speed (1 = the free stream; 1.1-1.3 typical near the '
                                                            'roof\'s front)'),
    ('Vehicle', 'cd_base', 'Cd', '', 3, 0.01, 'The car\'s drag coefficient (0 = typical of its body family)'),
    ('Vehicle', 'frontal_m2', 'Frontal area', 'm²', 2, 0.05, 'The car\'s frontal area (0 = 0.84 × width × height)'),
    ('Vehicle', 'mass_kg', 'Mass', 'kg', 0, 10.0, 'Driving mass (0 = kerb + 75 kg driver)'),
    ('Vehicle', 'crr', 'Rolling res.', '', 4, 0.001, 'Rolling resistance coefficient (typical 0.010)'),
    ('Vehicle', 'drive_eff', 'Drive eff.', '', 2, 0.01, 'Battery to wheel efficiency (typical 0.85)'),
    ('Vehicle', 'aux_w', 'Auxiliaries', 'W', 0, 50.0, 'HVAC and 12 V while driving'),
    ('Battery', 'battery_kwh', 'Battery', 'kWh', 1, 1.0, 'Usable energy: set the vehicle\'s'),
    ('Battery', 'consumption_wh_km', 'Measured', 'Wh/km', 0, 5.0, 'Measured consumption at report speed 2 '
                                                                  '(0 = the road-load model)'),
    ('Lock-in', 'lockin_band', 'Lock-in band', '±', 2, 0.01, 'A mode within this share of the shedding frequency '
                                                              'locks in (typical ±0.2)'),
    ('Lock-in', 'zeta', 'Damping ζ', '', 3, 0.005, 'Structural damping for the Scruton number'),
)


class AeroPage(QWidget):
    def __init__(self, ctx):
        super().__init__()
        self.ctx = ctx
        self.result: AE.AeroResult | None = None
        self._dirty = True
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
        self.headline = label('', ink2=True, wrap=True)
        cl.addWidget(self.headline)
        self.c_loss = LineChart('Range loss', unit='%', x_fmt=lambda v: f'{v:g} km/h')
        self.c_drag = LineChart('Rig drag', unit='N', x_fmt=lambda v: f'{v:g} km/h')
        cl.addWidget(self.c_loss, 1)
        cl.addWidget(self.c_drag, 1)
        cl.addWidget(label('Not CFD: drag coefficients of simple bluff bodies (cited in the export) times the rig\'s '
                           'frontal areas, with stated corrections.', muted=True, small=True, wrap=True))
        split.addWidget(cw)
        rw = QWidget()
        rw.setObjectName('panel')
        rl = QVBoxLayout(rw)
        rl.setContentsMargins(8, 6, 8, 8)
        self.tabs = QTabWidget()
        self.t_drag = table([('Element', ''), ('Kind', ''), ('Drag N 1', 'At report speed 1'),
                             ('Drag N 2', 'At report speed 2'), ('Cd', 'Its coefficient at speed 2'),
                             ('Area m²', 'Its reference area'), ('Shield', 'Share of its drag left in the wake of '
                                                                          'others')], 0)
        self.t_range = table([('km/h', ''), ('Rig drag N', ''), ('Rig CdA m²', ''), ('Base Wh/km', ''),
                              ('Added Wh/km', ''), ('Range loss %', ''), ('Range km', 'Without → with the rig')], 6)
        self.t_shed = table([('Element', ''), ('Mode', 'The structure\'s mode it moves in'), ('Mode Hz', ''),
                             ('Lock-in km/h', 'Where its shedding frequency meets the mode'),
                             ('Band km/h', 'The lock-in range'), ('Scruton', 'Mass-damping: higher, smaller motion')],
                            1)
        self.t_wind = table([('Device', ''), ('Own force N', ''), ('Move mm', 'Static, beam chain'),
                             ('Turn °', ''), ('Through', 'The mounts it rides on')], 4)
        self.t_tone = table([('Element', ''), ('D mm', 'Width across the wind'), ('Tone Hz 1', ''),
                             ('Tone Hz 2', '')], 0)
        self.tabs.addTab(self.t_range, 'Range')
        self.tabs.addTab(self.t_drag, 'Drag')
        self.tabs.addTab(self.t_shed, 'Lock-in')
        self.tabs.addTab(self.t_wind, 'Wind load')
        self.tabs.addTab(self.t_tone, 'Tones')
        fsc, self.find_lay = findings_box()
        self.tabs.addTab(fsc, 'Findings')
        rl.addWidget(self.tabs, 1)
        self.modes_from = label('', muted=True, small=True, wrap=True)
        rl.addWidget(self.modes_from)
        split.addWidget(rw)
        split.setSizes([300, 640, 660])
        lay.addWidget(split)
        ctx.store.changed.connect(self._on_change)
        ctx.store.project_replaced.connect(self._on_change)
        THEME.changed.connect(self._on_change)

    def _left(self):
        w = QWidget()
        w.setObjectName('panel')
        ll = QVBoxLayout(w)
        ll.setContentsMargins(8, 6, 8, 8)
        ll.setSpacing(4)
        ll.addWidget(label('Aerodynamics'))
        ll.addWidget(hline())
        ll.addWidget(label('The rig\'s drag and the range it costs, vortex shedding against the frame\'s modes and '
                           'the wind on each sensor.', muted=True, small=True, wrap=True))
        self.t_frame = ToggleSwitch('The frame (not the rig as drawn)')
        self.t_frame.setToolTip('Take the synthesised frame (Analysis ▸ Frame) with its members, plates and feet; '
                                'off: the rig parts as drawn')
        self.t_frame.clicked.connect(lambda on: self._set('frame' if on else 'rig as drawn', use_frame=bool(on)))
        ll.addWidget(self.t_frame)
        self.fields = {}
        group, grid = None, None
        for g, key, text, unit, dec, step, tip in FIELDS:
            if g != group:
                if grid is not None:
                    ll.addLayout(grid)
                group = g
                ll.addSpacing(4)
                ll.addWidget(label(g, bold=True))
                grid = QGridLayout()
            f = ScrubSpinBox('', 'plain', dec, unit, step=step, width=100)
            f.setToolTip(tip)
            f.committed.connect(lambda v, k=key, t=text: self._set(f'{t.lower()} {v:g}', **{k: v}))
            a = label(text, ink2=True)
            a.setToolTip(tip)
            i = grid.rowCount()
            grid.addWidget(a, i, 0)
            grid.addWidget(f, i, 1)
            self.fields[key] = f
        ll.addLayout(grid)
        ll.addStretch(1)
        self.status = label('', muted=True, small=True, wrap=True)
        ll.addWidget(self.status)
        bl = QHBoxLayout()
        b = button('Defaults', 'ghost', None, 'Back to the default settings')
        b.clicked.connect(self._defaults)
        bl.addWidget(b)
        bl.addStretch(1)
        self.b_export = button('Export…', None, 'export', 'Drag, range, lock-in and the wind case as JSON, CSV, PNG')
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
    def settings(self) -> AE.AeroSettings:
        return AE.settings_of(self.ctx.store.project)

    def _set(self, text, **kw) -> bool:
        st = self.settings
        for k, v in kw.items():
            setattr(st, k, v)
        new = AE.AeroSettings.from_dict(st.to_dict()).to_dict()
        if new == self.settings.to_dict():
            self._sync()
            return False

        def fn(p):
            p.analysis.aero = new
        self.ctx.store.edit('analysis', fn, f'Aerodynamics · {text}')
        return True

    def _defaults(self):
        if self.ctx.store.project.analysis.aero:
            def fn(p):
                p.analysis.aero = {}
            self.ctx.store.edit('analysis', fn, 'Aerodynamics · defaults')

    def _sync(self):
        st = self.settings
        for k, f in self.fields.items():
            f.set_value(float(getattr(st, k)))
        self.t_frame.blockSignals(True)
        self.t_frame.setChecked(st.use_frame)
        self.t_frame.blockSignals(False)

    def _on_change(self, *_):
        if self.isVisible():
            self.refresh()
        else:
            self._dirty = True

    def showEvent(self, e):
        super().showEvent(e)
        if self._dirty:
            self.refresh()

    def activate(self):
        if self._dirty:
            self.refresh()

    def refresh(self):
        self._dirty = False
        self._sync()
        try:
            res = self.result = AE.run(self.ctx.store.project)
        except Exception as e:                           # a broken input: say so, keep the page alive
            self.result = None
            self.status.setText(f'Failed: {type(e).__name__}: {e}')
            return
        st = res.settings
        hi = res.speeds[-1] if res.speeds else None
        self.headline.setText(
            (f'At {hi["kmh"]:g} km/h the rig adds {hi["drag_n"]:.0f} N (CdA {hi["rig_cda_m2"]:.3f} m²) and '
             f'{hi["added_wh_km"]:.0f} Wh/km: {hi["range_loss_pct"]:.1f} % less range, {hi["range_km"]:.0f} → '
             f'{hi["range_with_km"]:.0f} km on {st.battery_kwh:g} kWh. Car: Cd {res.vehicle["cd"]:.2f} × '
             f'{res.vehicle["area"]:.2f} m², {res.vehicle["mass"]:.0f} kg; rig {res.rig_kg:.1f} kg.')
            if hi and res.elements else 'Nothing above the roof.')
        c = res.curve
        self.c_loss.set_data([{'name': 'Range loss', 'x': c['kmh'], 'y': c['loss_pct'], 'color': THEME.cam(0)}],
                             vlines=[(st.speed1_kmh, f'{st.speed1_kmh:g} km/h', THEME.color('muted')),
                                     (st.speed2_kmh, f'{st.speed2_kmh:g} km/h', THEME.color('muted'))],
                             empty='Nothing above the roof')
        self.c_drag.set_data([{'name': 'Rig drag', 'x': c['kmh'], 'y': c['drag_n'], 'color': THEME.color('ink2')}],
                             empty='Nothing above the roof')
        k1, k2 = f'{st.speed1_kmh:g}', f'{st.speed2_kmh:g}'
        fill(self.t_drag, [[e['name'], e['kind'], f'{e.get("at", {}).get(k1, {}).get("drag_n", 0):.2f}',
                            f'{e.get("at", {}).get(k2, {}).get("drag_n", 0):.2f}',
                            f'{e.get("at", {}).get(k2, {}).get("cd", 0):.2f}',
                            f'{e.get("at", {}).get(k2, {}).get("area_m2", 0):.4f}',
                            f'{e.get("at", {}).get(k2, {}).get("shield", 1):.2f}'] for e in res.elements],
             numeric=(2, 3, 4, 5, 6))
        fill(self.t_range, [[f'{s["kmh"]:g}', f'{s["drag_n"]:.1f}', f'{s["rig_cda_m2"]:.3f}',
                             f'{s["base_wh_km"]:.0f}', f'{s["added_wh_km"]:.1f}', f'{s["range_loss_pct"]:.1f}',
                             f'{s["range_km"]:.0f} → {s["range_with_km"]:.0f}'] for s in res.speeds],
             numeric=(0, 1, 2, 3, 4, 5, 6))
        fill(self.t_shed, [[s['element'], s['mode'], f'{s["f_hz"]:.1f}', f'{s["lock_kmh"]:.0f}',
                            f'{s["band_kmh"][0]:.0f}-{s["band_kmh"][1]:.0f}' + ('' if s['in_range'] else ' (above)'),
                            '—' if s['scruton'] is None else f'{s["scruton"]:.0f}'] for s in res.shedding],
             numeric=(2, 3, 5))
        fill(self.t_wind, [[d['tag'], f'{d["force_n"]:.1f}', f'{d["move_abs_mm"]:.3f}', f'{d["turn_abs_deg"]:.4f}',
                            ' → '.join(d['chain'])] for d in res.load_case.get('devices', [])], numeric=(1, 2, 3))
        fill(self.t_tone, [[t['element'], f'{t["D_m"] * 1000:.0f}', f'{t["hz"].get(k1, 0):.0f}',
                            f'{t["hz"].get(k2, 0):.0f}'] for t in res.tones], numeric=(1, 2, 3))
        set_findings(self.find_lay, res.findings)
        self.modes_from.setText(f'Lock-in modes from {res.modes_source}. Wind noise: the tones are the shedding '
                                'frequencies; no level is estimated (it depends on the roof\'s transmission and the '
                                'slots: fit slot covers).')
        bad = sum(f['severity'] in ('critical', 'serious', 'warning') for f in res.findings)
        self.status.setText(f'{res.variant}: ' + (f'{bad} finding{"s" * (bad != 1)} to look at' if bad else
                                                  'nothing to flag'))

    def export(self, folder=None):
        res = self.result
        if res is None:
            return []
        if folder is None:
            folder = QFileDialog.getExistingDirectory(self, 'Export the aerodynamics into')
            if not folder:
                return []
        from ...core import provenance as PV
        paths = PV.stamp_paths(AE.export(res, self.ctx.store.project, folder), self.ctx.store.project, folder)
        self.ctx.msg(f'Exported the aerodynamics to {folder}', 5000)
        return paths
