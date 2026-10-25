"""Analysis ▸ Mount brackets: a 3D-printable bracket per camera that holds it at its planned position and angles on
the surface under it (core.brackets). The list shows at once which cameras can take a printed bracket (worked out
in the background on every edit); Generate meshes them as a job with progress and Cancel; the preview, the figures
and the checks follow the selected camera; Export writes the STL of one camera, or every STL with the JSON and the
PNG sheet."""
from __future__ import annotations

import json
import math
import os
import threading
import traceback

from PyQt6.QtCore import QObject, Qt, QTimer, pyqtSignal
from PyQt6.QtGui import QColor
from PyQt6.QtWidgets import (QAbstractItemView, QComboBox, QFileDialog, QGridLayout, QHBoxLayout, QHeaderView,
                             QProgressBar, QScrollArea, QSplitter, QTableWidget, QTableWidgetItem, QVBoxLayout,
                             QWidget)

from ...core import brackets as BR
from ...core.project import mirror_props
from ..bracketview import BracketView
from ..theme import THEME, STATUS, mono
from ..widgets import ROW_H, ScrubSpinBox, SegmentedControl, StatusLine, ToggleSwitch, button, hline, label

JOB = 'brackets'
STATUS_WORD = {'ok': 'Generated', 'warning': 'Warning', 'standoff': 'Needs standoff', 'floating': 'Too far out',
               'inside': 'Inside surface', 'face': 'Face turned away', 'skipped': 'No bracket', 'ready': 'Ready'}
STATUS_SEV = {'ok': 'good', 'warning': 'warning', 'standoff': 'serious', 'floating': 'critical', 'inside': 'critical',
              'face': 'serious', 'skipped': None, 'ready': None}

# (group, key, label, unit, decimals, step, tip)
FIELDS = (
    ('Camera fixing', 'clearance_mm', 'Hole clearance', 'mm', 2, 0.05,
     'Holes are the screw\'s nominal diameter plus this (0.4 mm suits most printers)'),
    ('Camera fixing', 'setback_mm', 'Optical centre', 'mm', 1, 0.5,
     'How far the optical centre (the lens\'s entrance pupil) sits behind the lens front: check the module\'s '
     'drawing'),
    ('Camera fixing', 'connector_mm', 'Connector hole', 'mm', 1, 0.5,
     'Back face only: a hole through the plate and support for the connector and cable (0 = none)'),
    ('Bracket', 'wall_mm', 'Wall', 'mm', 1, 0.5, 'Plate and base thickness (at least 2 mm)'),
    ('Bracket', 'margin_mm', 'Plate margin', 'mm', 1, 0.5, 'How far the plate reaches past the housing face'),
    ('Bracket', 'fillet_mm', 'Fillet', 'mm', 1, 0.5, 'Radius where the support meets the base and the plate'),
    ('Base', 'tape_mm', 'Tape', 'mm', 1, 0.1, 'VHB foam thickness: the gap under the base it fills'),
    ('Tube clamp', 'clamp_clear_mm', 'Bore clearance', 'mm', 2, 0.05, 'Clamp bore over the tube\'s radius'),
    ('Mesh and checks', 'res_mm', 'Resolution', 'mm', 2, 0.05,
     'Mesh grid: finer is smoother and larger (0.35 mm: a few hundred thousand triangles per bracket)'),
    ('Mesh and checks', 'overhang_mm', 'Lever warning', 'mm', 0, 5.0,
     'Warn when the camera sits farther than this from the base'),
    ('Mesh and checks', 'reach_mm', 'Reach', 'mm', 0, 5.0,
     'Farther than this from every surface, a camera gets no printed bracket (it needs a custom mount)'),
)
CUSTOM = (('custom_d_mm', 'Screw', 'mm', 1, 0.5, 'Screw diameter of the custom pattern'),
          ('custom_a_mm', 'Spacing', 'mm', 1, 0.5, 'Hole spacing along the face (across the back face; along the '
                                                     'optical axis on the bottom or top)'),
          ('custom_b_mm', 'Spacing 2', 'mm', 1, 0.5, 'Second spacing of a 4-hole pattern'))


def inputs_key(project) -> str:
    """Everything the brackets depend on, as text: a result made from other inputs is stale."""
    m = project.vehicle.mesh
    v = project.variant
    return json.dumps({'mesh': [id(m), getattr(m, 'n_tris', 0)], 'variant': v.uid,
                       'cams': [c.to_dict() for c in v.cameras], 'rig': [r.to_dict() for r in project.rig],
                       'settings': project.analysis.brackets,
                       'modules': {k: x.to_dict() for k, x in project.modules.items()},
                       'lenses': {k: x.to_dict() for k, x in project.lenses.items()}}, sort_keys=True, default=str)


class _Sig(QObject):
    pre = pyqtSignal(int, object)
    done = pyqtSignal(int, object)
    failed = pyqtSignal(int, str)
    progress = pyqtSignal(int, float)


def controller(ctx) -> 'BracketsController':
    c = getattr(ctx, 'brackets_ctl', None)
    if c is None:
        c = ctx.brackets_ctl = BracketsController(ctx)
    return c


class BracketsController(QObject):
    """The project's brackets: each camera's status worked out in the background after every edit while the page
    shows (cheap, no mesh), and the generated meshes (Generate: a main-window job with Cancel)."""
    changed = pyqtSignal()
    progress = pyqtSignal(float)

    def __init__(self, ctx):
        super().__init__()
        self.ctx = ctx
        self.pre: list = []              # Brackets without meshes, for the current inputs (or being refreshed)
        self._pre_key = None
        self.result: BR.BracketResult | None = None
        self._res_key = None
        self._res_project = None
        self.stale = False
        self.busy = False
        self.status = ''
        self.active = False
        self._gen = 0
        self._pgen = 0
        self.sig = _Sig()
        self.sig.pre.connect(self._pre_done)
        self.sig.done.connect(self._done)
        self.sig.failed.connect(self._failed)
        self.sig.progress.connect(lambda g, f: self.progress.emit(f) if g == self._gen else None)
        self.timer = QTimer(self)
        self.timer.setSingleShot(True)
        self.timer.setInterval(250)
        self.timer.timeout.connect(self.refresh_status)
        ctx.store.changed.connect(self._on_edit)
        ctx.store.project_replaced.connect(self._reset)

    # ---- settings (saved with the project; every edit is one undo step)
    @property
    def settings(self) -> BR.BracketSettings:
        return BR.BracketSettings.from_dict(self.ctx.store.project.analysis.brackets)

    def set(self, text, **kw) -> bool:
        st = self.settings
        for k, v in kw.items():
            setattr(st, k, v)
        new = BR.BracketSettings.from_dict(st.to_dict()).to_dict()
        if new == self.settings.to_dict():
            return False

        def fn(p):
            p.analysis.brackets = new
        self.ctx.store.edit('analysis', fn, f'Mount brackets · {text}')
        return True

    def defaults(self):
        if self.ctx.store.project.analysis.brackets:
            def fn(p):
                p.analysis.brackets = {}
            self.ctx.store.edit('analysis', fn, 'Mount brackets · defaults')

    # ---- state
    def _on_edit(self, *_):
        key = inputs_key(self.ctx.store.project)
        stale = self.result is not None and key != self._res_key        # an undo back to the inputs: fresh again
        if stale != self.stale:
            self.stale = stale
            self.changed.emit()
        if self.active and key != self._pre_key:
            self.timer.start()

    def set_active(self, on: bool):
        self.active = on
        if on and inputs_key(self.ctx.store.project) != self._pre_key:
            self.timer.start()

    def _reset(self, *_):
        self._gen += 1
        self._pgen += 1
        self.pre, self._pre_key = [], None
        self.result, self._res_key, self._res_project, self.stale, self.busy, self.status = \
            None, None, None, False, False, ''
        self.changed.emit()
        if self.active:
            self.timer.start()

    def _engine(self):
        win = self.ctx.window
        return win._shared_engine() if win is not None and hasattr(win, '_shared_engine') else \
            self.ctx.analysis.engine.share()

    def refresh_status(self):
        """Every camera's status for the current inputs, worked out on a thread."""
        p = self.ctx.store.project
        key = inputs_key(p)
        snap = p.snapshot()
        eng = self._engine()
        self._pgen += 1
        gen = self._pgen

        def thread():
            try:
                out = BR.preflight(snap, eng)
            except Exception as e:                      # a broken input: show it, never crash the page
                out = e
            try:
                self.sig.pre.emit(gen, (key, out))
            except RuntimeError:
                pass
        threading.Thread(target=thread, daemon=True).start()

    def _pre_done(self, gen, out):
        if gen != self._pgen:
            return
        key, pre = out
        if isinstance(pre, Exception):
            self.status = f'Failed: {type(pre).__name__}: {pre}'
            pre = []
        self.pre, self._pre_key = pre, key
        self.changed.emit()

    def bracket(self, uid):
        """The camera's bracket as things stand: the generated one while the result is fresh, else its status
        worked out for the current inputs (no mesh)."""
        if self.result is not None and not self.stale:
            b = self.result.by_uid(uid)
            if b is not None:
                return b
        return next((b for b in self.pre if b.uid == uid), None)

    def meshed(self, uid):
        """The camera's last generated bracket (stale or not), or None."""
        b = self.result.by_uid(uid) if self.result is not None else None
        return b if b is not None and b.generated else None

    # ---- generate
    def generate(self, quiet=False) -> bool:
        ctx = self.ctx
        p = ctx.store.project
        win = ctx.window
        if not any(c.visible for c in p.variant.cameras):
            self.status = 'No visible camera to make a bracket for.'
            self.changed.emit()
            return False
        if self.busy and win is not None and JOB in getattr(win, '_jobs', {}):
            return False
        snap = p.snapshot()
        st = BR.BracketSettings.from_dict(snap.analysis.brackets)
        eng = self._engine()
        self._gen += 1
        gen = self._gen
        self.busy = True
        self.status = 'Generating the brackets…'
        key = (p, inputs_key(p))

        def work(progress, cancel):
            return key, BR.run(snap, eng, st, progress=progress, cancel=cancel)
        if not quiet and win is not None and hasattr(win, 'run_job'):
            ok = win.run_job(JOB, 'Mount brackets', work, lambda out: self._done(gen, out),
                             lambda m: self._failed(gen, m))
            self.changed.emit()
            return ok
        self.changed.emit()

        def thread():
            try:
                out = work(lambda f, *_: self.sig.progress.emit(gen, f), lambda: gen != self._gen)
                self.sig.done.emit(gen, out)
            except RuntimeError:
                pass
            except Exception as e:
                try:
                    self.sig.failed.emit(gen, f'{type(e).__name__}: {e}\n{traceback.format_exc()}')
                except RuntimeError:
                    pass
        threading.Thread(target=thread, daemon=True).start()
        return True

    def _done(self, gen, out):
        if gen != self._gen:
            return
        self.busy = False
        key, res = out if out is not None else (None, None)
        if res is None or key[0] is not self.ctx.store.project:
            self.status = 'Cancelled' if res is None else ''
            self.changed.emit()
            return
        self.result, self._res_key, self._res_project = res, key[1], key[0]
        self.stale = key[1] != inputs_key(self.ctx.store.project)
        n = sum(b.generated for b in res.brackets)
        warn = sum(b.status == 'warning' for b in res.brackets)
        none = len(res.brackets) - n
        self.status = (f'{n} bracket{"s" * (n != 1)} generated' + (f', {warn} with warnings' if warn else '')
                       + (f' · {none} camera{"s" * (none != 1)} without' if none else '') + f' · {res.seconds:.1f} s')
        self.changed.emit()

    def _failed(self, gen, msg):
        if gen != self._gen:
            return
        self.busy = False
        self.status = 'Failed: ' + msg.splitlines()[0]
        self.changed.emit()

    # ---- moving cameras out to a printable standoff
    def movable(self, uid):
        b = self.bracket(uid)
        if b is None or b.status != 'standoff' or not math.isfinite(b.required_mm) or b.normal is None or \
                b.required_mm > self.settings.reach_mm:
            return None
        return b

    def move_out(self, uids) -> bool:
        """Move cameras out along their surface normal to the standoff a printed bracket needs (one undo step; a
        mirrored partner follows)."""
        moves = {}
        for u in uids:
            b = self.movable(u)
            if b is not None:
                moves[u] = b
        if not moves:
            return False

        def fn(p):
            for u, b in moves.items():
                c = p.variant.cam(u)
                if c is None:
                    continue
                q = [b.contact[i] + b.normal[i] * b.required_mm / 1000.0 for i in range(3)]
                c.x, c.y, c.z = (round(float(v), 4) for v in q)
                c.standoff_mm = float(b.required_mm)
                partner = p.variant.cam(c.link_uid) if c.link_uid else None
                if partner is not None and partner.uid not in moves:
                    mirror_props(c, partner, ['x', 'y', 'z', 'standoff_mm'])
        tags = ', '.join(b.tag for b in moves.values())
        mm = ', '.join(f'{b.required_mm:.0f}' for b in moves.values())
        self.ctx.store.edit('variant', fn, f'Moved {tags} out to {mm} mm for a printed bracket')
        return True


# =========================================================================== the page
def _table(cols):
    t = QTableWidget(0, len(cols))
    t.setHorizontalHeaderLabels(cols)
    t.verticalHeader().setVisible(False)
    t.verticalHeader().setDefaultSectionSize(ROW_H)
    t.horizontalHeader().setFixedHeight(ROW_H)
    t.setShowGrid(False)
    t.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
    t.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
    t.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
    return t


class BracketsPage(QWidget):
    def __init__(self, ctx):
        super().__init__()
        self.ctx = ctx
        self.ctl = controller(ctx)
        self.sel_uid = None
        self._syncing = False
        self.setObjectName('base')
        lay = QHBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        split = QSplitter(Qt.Orientation.Horizontal)
        split.setHandleWidth(2)
        split.addWidget(self._left())
        right = QSplitter(Qt.Orientation.Vertical)
        right.setHandleWidth(2)
        top = QSplitter(Qt.Orientation.Horizontal)
        top.setHandleWidth(2)
        tw = QWidget()
        tw.setObjectName('panel')
        tl = QVBoxLayout(tw)
        tl.setContentsMargins(8, 6, 8, 6)
        tl.setSpacing(4)
        tl.addWidget(label('Cameras', bold=True))
        self.table = _table(['Camera', 'Mount', 'Status', 'Standoff'])
        self.table.setToolTip('Standoff: the optical centre\'s distance from the surface under it, mm (and what a '
                              'printed bracket needs, where more)')
        hh = self.table.horizontalHeader()
        hh.setSectionResizeMode(0, QHeaderView.ResizeMode.ResizeToContents)
        hh.setSectionResizeMode(1, QHeaderView.ResizeMode.Stretch)
        hh.setSectionResizeMode(2, QHeaderView.ResizeMode.ResizeToContents)
        hh.setSectionResizeMode(3, QHeaderView.ResizeMode.ResizeToContents)
        self.table.itemSelectionChanged.connect(self._picked)
        tl.addWidget(self.table, 1)
        row = QHBoxLayout()
        self.b_move_all = button('Move all out', 'ghost', None,
                                 'Move every camera that needs it out along its surface normal to the standoff a '
                                 'printed bracket needs (one undo step; the coverage changes a little)')
        self.b_move_all.clicked.connect(self._move_all)
        row.addWidget(self.b_move_all)
        row.addStretch(1)
        tl.addLayout(row)
        top.addWidget(tw)
        self.view = BracketView()
        top.addWidget(self.view)
        top.setSizes([380, 620])
        right.addWidget(top)
        right.addWidget(self._details())
        right.setSizes([520, 380])
        split.addWidget(right)
        split.setSizes([360, 1240])
        lay.addWidget(split)
        self.ctl.changed.connect(self.refresh)
        self.ctl.progress.connect(lambda f: self.bar.setValue(int(f * 1000)))
        ctx.store.changed.connect(lambda *_: self._sync_fields())
        ctx.store.project_replaced.connect(self._sync_fields)
        THEME.changed.connect(self._theme_changed)      # a bound slot: disconnected when the widget goes
        self._sync_fields()
        self.refresh()

    # ------------------------------------------------------------------ settings panel

    def _theme_changed(self, *_):
        self.refresh()
    def _left(self):
        w = QWidget()
        w.setObjectName('panel')
        ll = QVBoxLayout(w)
        ll.setContentsMargins(8, 6, 8, 8)
        ll.setSpacing(4)
        top = QHBoxLayout()
        top.addWidget(label('Mount brackets'))
        top.addStretch(1)
        top.addWidget(label('3D-printable, per camera', muted=True, small=True))
        ll.addLayout(top)
        ll.addWidget(hline())
        ll.addWidget(label('A bracket per camera that holds it at its planned position and angles on the surface '
                           'under it: a plate on the housing, a solid support, and a screwed or taped base, or a '
                           'split clamp on a rig tube. STL in millimetres, base down.', muted=True, small=True,
                           wrap=True))
        self.fields = {}
        self.grids = {}

        def grid():
            g = QGridLayout()
            g.setVerticalSpacing(2)
            g.setHorizontalSpacing(6)
            g.setColumnStretch(2, 1)
            ll.addLayout(g)
            return g

        def num(g, key, text, unit, dec, step, tip):
            f = ScrubSpinBox('', 'plain', dec, unit, step=step, width=110)
            f.setToolTip(tip)
            f.committed.connect(lambda v, k=key, t=text, u=unit: self.ctl.set(f'{t.lower()} {v:g} {u}', **{k: v}))
            a = label(text, ink2=True)
            a.setFixedWidth(112)
            a.setToolTip(tip)
            i = g.rowCount()
            g.addWidget(a, i, 0)
            g.addWidget(f, i, 1)
            self.fields[key] = f
            return a, f

        def head(text):
            ll.addSpacing(6)
            ll.addWidget(label(text, bold=True))
        # ---- camera fixing
        head('Camera fixing')
        self.face = SegmentedControl([(k, BR.FACE_LABEL[k]) for k in BR.FACES], 'auto')
        self.face.setToolTip('The housing face the plate holds. Auto: the bottom or top face where one faces the '
                             'surface (the back keeps its connector free), else the back')
        self.face.changed.connect(lambda k: self.ctl.set(f'{BR.FACE_LABEL[k].lower()} face', face=k))
        ll.addWidget(self.face)
        g = grid()
        self.pattern = QComboBox()
        for k, p in BR.PATTERNS.items():
            self.pattern.addItem(p['label'], k)
        self.pattern.setToolTip('The camera module\'s mounting holes on that face (check its drawing)')
        self.pattern.activated.connect(lambda i: self.ctl.set(f'holes {self.pattern.itemText(i)}',
                                                              pattern=self.pattern.itemData(i)))
        a = label('Holes', ink2=True)
        a.setFixedWidth(112)
        g.addWidget(a, 0, 0)
        g.addWidget(self.pattern, 0, 1, 1, 2)
        self.custom = QWidget()
        cl = QGridLayout(self.custom)
        cl.setContentsMargins(0, 0, 0, 0)
        cl.setVerticalSpacing(2)
        cl.setHorizontalSpacing(6)
        cl.setColumnStretch(2, 1)
        self.count = SegmentedControl([('1', '1 hole'), ('2', '2 holes'), ('4', '4 holes')], '2')
        self.count.changed.connect(lambda k: self.ctl.set(f'{k} custom holes', custom_count=int(k)))
        cl.addWidget(self.count, 0, 0, 1, 3)
        for key, text, unit, dec, step, tip in CUSTOM:
            num(cl, key, text, unit, dec, step, tip)
        ll.addWidget(self.custom)
        g = grid()
        for grp, key, text, unit, dec, step, tip in FIELDS:
            if grp == 'Camera fixing':
                num(g, key, text, unit, dec, step, tip)
        # ---- bracket
        head('Bracket')
        g = grid()
        for grp, key, text, unit, dec, step, tip in FIELDS:
            if grp == 'Bracket':
                num(g, key, text, unit, dec, step, tip)
        self.material = SegmentedControl([(k, k) for k in BR.MATERIALS], 'ASA')
        self.material.setToolTip('The material changes only the notes, the mass and the stiffness estimate')
        self.material.changed.connect(lambda k: self.ctl.set(f'material {k}', material=k))
        ll.addWidget(self.material)
        self.mat_note = label('', muted=True, small=True, wrap=True)
        ll.addWidget(self.mat_note)
        # ---- base
        head('Base')
        self.fixing = SegmentedControl([(k, BR.FIXING_LABEL[k]) for k in BR.FIXINGS], 'screws2')
        self.fixing.setToolTip('How the flat base holds on the body or a rig box: countersunk screws (each placed '
                               'where a screwdriver reaches it past the camera) or VHB tape')
        self.fixing.changed.connect(lambda k: self.ctl.set(f'base {BR.FIXING_LABEL[k].lower()}', fixing=k))
        ll.addWidget(self.fixing)
        g = grid()
        self.base_screw = QComboBox()
        self.base_screw.addItems(list(BR.BASE_SCREWS))
        self.base_screw.activated.connect(lambda i: self.ctl.set(f'base screws {self.base_screw.itemText(i)}',
                                                                 base_screw=self.base_screw.itemText(i)))
        a = label('Base screws', ink2=True)
        a.setFixedWidth(112)
        g.addWidget(a, 0, 0)
        g.addWidget(self.base_screw, 0, 1)
        self.csk = ToggleSwitch('Countersunk heads')
        self.csk.setToolTip('Countersink the base\'s screw holes (flat heads flush with the base)')
        self.csk.clicked.connect(lambda on: self.ctl.set('countersunk' if on else 'not countersunk', countersink=on))
        g.addWidget(self.csk, 1, 0, 1, 3)
        for grp, key, text, unit, dec, step, tip in FIELDS:
            if grp == 'Base':
                num(g, key, text, unit, dec, step, tip)
        # ---- tube clamp
        head('Tube clamp')
        g = grid()
        for grp, key, text, unit, dec, step, tip in FIELDS:
            if grp == 'Tube clamp':
                num(g, key, text, unit, dec, step, tip)
        self.bolt = QComboBox()
        self.bolt.addItems(list(BR.BASE_SCREWS))
        self.bolt.setToolTip('Clamp bolts: in from under the cap into nuts held in the saddle')
        self.bolt.activated.connect(lambda i: self.ctl.set(f'clamp bolts {self.bolt.itemText(i)}',
                                                           clamp_bolt=self.bolt.itemText(i)))
        a = label('Bolts', ink2=True)
        a.setFixedWidth(112)
        i = g.rowCount()
        g.addWidget(a, i, 0)
        g.addWidget(self.bolt, i, 1)
        # ---- mesh and checks
        head('Mesh and checks')
        g = grid()
        for grp, key, text, unit, dec, step, tip in FIELDS:
            if grp == 'Mesh and checks':
                num(g, key, text, unit, dec, step, tip)
        ll.addSpacing(6)
        ll.addWidget(label('Assumptions', bold=True))
        ll.addWidget(label(BR.ASSUMPTIONS, muted=True, small=True, wrap=True))
        ll.addStretch(1)
        sc = QScrollArea()
        sc.setWidgetResizable(True)
        sc.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        sc.setWidget(w)
        # status, progress and the buttons stay in view under the scrolling settings
        outer = QWidget()
        outer.setObjectName('panel')
        ol = QVBoxLayout(outer)
        ol.setContentsMargins(0, 0, 0, 0)
        ol.setSpacing(0)
        ol.addWidget(sc, 1)
        foot = QWidget()
        fl = QVBoxLayout(foot)
        fl.setContentsMargins(8, 4, 8, 8)
        fl.setSpacing(4)
        fl.addWidget(hline())
        self.status = label('', muted=True, small=True, wrap=True)
        fl.addWidget(self.status)
        self.bar = QProgressBar()
        self.bar.setTextVisible(False)
        self.bar.setRange(0, 1000)
        self.bar.hide()
        fl.addWidget(self.bar)
        bl = QHBoxLayout()
        b_def = button('Defaults', 'ghost', None, 'Back to the default bracket settings')
        b_def.clicked.connect(self.ctl.defaults)
        bl.addWidget(b_def)
        bl.addStretch(1)
        self.b_gen = button('Generate', 'primary', None, 'Mesh every visible camera\'s bracket and run the checks')
        self.b_gen.clicked.connect(lambda: self.ctl.generate())
        bl.addWidget(self.b_gen)
        fl.addLayout(bl)
        ol.addWidget(foot)
        outer.setMinimumWidth(350)
        return outer

    def _details(self):
        w = QWidget()
        w.setObjectName('panel')
        dl = QHBoxLayout(w)
        dl.setContentsMargins(10, 8, 10, 8)
        dl.setSpacing(16)
        left = QVBoxLayout()
        left.setSpacing(4)
        self.head = label('', bold=True)
        left.addWidget(self.head)
        self.state = StatusLine('good', '')
        left.addWidget(self.state)
        mv = QHBoxLayout()
        self.b_move = button('Move out', 'primary', None, '')
        self.b_move.clicked.connect(self._move_sel)
        mv.addWidget(self.b_move)
        mv.addStretch(1)
        left.addLayout(mv)
        self.figs = QGridLayout()
        self.figs.setHorizontalSpacing(10)
        self.figs.setVerticalSpacing(2)
        self.fig_labels = {}
        for i, (key, cap) in enumerate((('mount', 'Mount'), ('standoff', 'Standoff'), ('base', 'Base'),
                                        ('height', 'Height'), ('angle', 'Plate angle'), ('lever', 'Lever arm'),
                                        ('fov', 'Image blocked'), ('wall', 'Thinnest wall'), ('solid', 'Solid'),
                                        ('print', 'Print'), ('stiff', 'Stiffness'))):
            a = label(cap, muted=True)
            v = label('—', mono_=True)
            v.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
            self.figs.addWidget(a, i, 0)
            self.figs.addWidget(v, i, 1)
            self.fig_labels[key] = v
        self.figs.setColumnStretch(1, 1)
        left.addLayout(self.figs)
        left.addStretch(1)
        dl.addLayout(left, 5)
        right = QVBoxLayout()
        right.setSpacing(4)
        right.addWidget(label('Checks', bold=True))
        self.check_host = QWidget()
        self.check_lay = QVBoxLayout(self.check_host)
        self.check_lay.setContentsMargins(0, 0, 0, 0)
        self.check_lay.setSpacing(4)
        self.check_lay.addStretch(1)
        sc = QScrollArea()
        sc.setWidgetResizable(True)
        sc.setWidget(self.check_host)
        right.addWidget(sc, 1)
        self.notes = label('', muted=True, small=True, wrap=True)
        right.addWidget(self.notes)
        el = QHBoxLayout()
        el.addStretch(1)
        self.b_stl = button('Export STL…', None, 'export', 'The selected camera\'s bracket as a binary STL '
                                                            '(millimetres, base down)')
        self.b_stl.clicked.connect(lambda: self.export_selected())
        self.b_all = button('Export all…', None, 'export', 'Every bracket\'s STL, the JSON with their parameters, '
                                                           'frames and checks, and the PNG sheet')
        self.b_all.clicked.connect(lambda: self.export_all())
        el.addWidget(self.b_stl)
        el.addWidget(self.b_all)
        right.addLayout(el)
        dl.addLayout(right, 6)
        self.check_lines = []
        return w

    # ------------------------------------------------------------------ fields
    def _sync_fields(self):
        st = self.ctl.settings
        d = st.to_dict()
        for key, f in self.fields.items():
            f.set_value(d[key])
        self.face.set_value(st.face)
        i = self.pattern.findData(st.pattern)
        if i >= 0 and i != self.pattern.currentIndex():
            self.pattern.setCurrentIndex(i)
        self.custom.setVisible(st.pattern == 'custom')
        self.count.set_value(str(st.custom_count))
        self.material.set_value(st.material)
        self.mat_note.setText(BR.MATERIALS[st.material]['note'])
        self.fixing.set_value(st.fixing)
        self.base_screw.setCurrentText(st.base_screw)
        self.bolt.setCurrentText(st.clamp_bolt)
        self.csk.blockSignals(True)
        self.csk.setChecked(st.countersink)
        self.csk.blockSignals(False)
        self.csk.setEnabled(st.fixing != 'vhb')
        self.base_screw.setEnabled(st.fixing != 'vhb')

    # ------------------------------------------------------------------ state
    def showEvent(self, e):
        super().showEvent(e)
        self.ctl.set_active(True)

    def hideEvent(self, e):
        super().hideEvent(e)
        self.ctl.set_active(False)

    def activate(self):
        self.ctl.set_active(True)
        if self.sel_uid is None and self.ctx.store.selected is not None:
            self.select(self.ctx.store.selected.uid)

    def _cams(self):
        return [c for c in self.ctx.store.variant.cameras if c.visible]

    def select(self, uid):
        """Show this camera's bracket (and select its row)."""
        self.sel_uid = uid
        for r in range(self.table.rowCount()):
            it = self.table.item(r, 0)
            if it is not None and it.data(Qt.ItemDataRole.UserRole) == uid:
                if self.table.currentRow() != r:
                    self.table.selectRow(r)
                break
        self._show_selected()

    def _picked(self):
        if self._syncing:
            return
        rows = self.table.selectionModel().selectedRows()
        if rows:
            it = self.table.item(rows[0].row(), 0)
            self.sel_uid = it.data(Qt.ItemDataRole.UserRole) if it is not None else None
            self._show_selected()

    def refresh(self):
        ctl = self.ctl
        cams = self._cams()
        win = self.ctx.window
        running = ctl.busy and win is not None and JOB in getattr(win, '_jobs', {})
        self.b_gen.setEnabled(bool(cams) and not running)
        self.bar.setVisible(ctl.busy)
        status = ctl.status or ('Generate to mesh the brackets.' if ctl.result is None else '')
        if ctl.result is not None and ctl.stale:
            status += ' · stale: the design or the settings changed, Generate again'
        if not cams:
            status = 'No visible camera in this variant.'
        self.status.setText(status)
        self._syncing = True
        try:
            t = self.table
            t.setRowCount(len(cams))
            for r, c in enumerate(cams):
                b = ctl.bracket(c.uid)
                it = QTableWidgetItem(c.tag)
                it.setData(Qt.ItemDataRole.UserRole, c.uid)
                it.setForeground(THEME.cam(c.slot))
                it.setFont(mono(12))
                t.setItem(r, 0, it)
                mount = '—' if b is None else (f'{b.surface} · ' + ('clamp' if b.kind == 'clamp' else 'flat')
                                               if b.surface else '—')
                t.setItem(r, 1, QTableWidgetItem(mount))
                st = '…' if b is None else ('ready' if b.status == 'ok' and not b.generated else b.status)
                sev = STATUS_SEV.get(st)
                if b is not None and b.generated and b.status == 'warning':
                    sev = b.worst()
                si = QTableWidgetItem(STATUS_WORD.get(st, 'Checking…'))
                si.setForeground(QColor(STATUS[sev]) if sev else THEME.color('muted'))
                si.setToolTip(b.message if b is not None else '')
                t.setItem(r, 2, si)
                if b is None or not math.isfinite(b.gap_mm):
                    text = '—'
                elif b.status == 'standoff' and math.isfinite(b.required_mm):
                    text = f'{b.gap_mm:.0f} → {b.required_mm:.0f}'
                else:
                    text = f'{b.gap_mm:.0f}'
                ni = QTableWidgetItem(text)
                ni.setFont(mono(12))
                ni.setTextAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
                t.setItem(r, 3, ni)
            uids = [c.uid for c in cams]
            if self.sel_uid not in uids:
                self.sel_uid = uids[0] if uids else None
            if self.sel_uid is not None:
                t.selectRow(uids.index(self.sel_uid))
        finally:
            self._syncing = False
        self.b_move_all.setEnabled(any(ctl.movable(c.uid) for c in cams))
        self.b_all.setEnabled(ctl.result is not None and any(b.generated for b in ctl.result.brackets))
        self._show_selected()

    def _show_selected(self):
        ctl = self.ctl
        c = self.ctx.store.variant.cam(self.sel_uid) if self.sel_uid else None
        b = ctl.bracket(self.sel_uid) if c is not None else None
        g = ctl.meshed(self.sel_uid) if c is not None else None
        for w in self.check_lines:
            w.setParent(None)
            w.deleteLater()
        self.check_lines = []
        for v in self.fig_labels.values():
            v.setText('—')
        self.notes.setText('')
        self.b_stl.setEnabled(g is not None)
        if c is None:
            self.head.setText('')
            self.state.set(None, 'No camera selected.')
            self.b_move.hide()
            self.view.set_bracket(None, 0, 'No visible camera')
            return
        self.head.setText(f'{c.tag} · {c.name}' + (f' — {c.location}' if c.location else ''))
        if b is None:
            self.state.set(None, 'Checking…')
            self.b_move.hide()
            self.view.set_bracket(None, c.slot, 'Checking…')
            return
        st = 'ready' if b.status == 'ok' and not b.generated else b.status
        sev = b.worst() if b.generated else STATUS_SEV.get(st)
        msg = b.message if b.generated or st != 'ready' else 'Ready to generate: press Generate.'
        if g is not None and g is not b:
            msg += ' The preview is from before the last change: Generate again.'
        self.state.set(sev, msg)
        mv = ctl.movable(c.uid)
        self.b_move.setVisible(mv is not None)
        if mv is not None:
            self.b_move.setText(f'Move out to {mv.required_mm:.0f} mm')
            self.b_move.setToolTip(f'Move {c.tag} {mv.required_mm - mv.gap_mm:.0f} mm out along the surface normal so '
                                   'a printed bracket fits (one undo step; a mirrored partner follows; the coverage '
                                   'changes a little)')
        self.view.set_bracket(g, c.slot, b.message if g is None and st != 'ready' else
                              'Press Generate to mesh this bracket.')
        self._figures(b)

    def _figures(self, b):
        f = self.fig_labels
        p = b.params
        if b.surface:
            f['mount'].setText(f'{b.surface} · ' + ('tube clamp' if b.kind == 'clamp' else 'flat base')
                               + (f' · {b.face} face' if b.face else ''))
        if math.isfinite(b.gap_mm):
            f['standoff'].setText(f'{b.gap_mm:.1f} mm' + (f' (planned {b.plan_standoff_mm:.0f})'
                                                          if abs(b.gap_mm - b.plan_standoff_mm) > 2 else '')
                                  + (f' · needs {b.required_mm:.0f}' if math.isfinite(b.required_mm) else ''))
        if b.status in BR.NOT_BUILT:
            return                                   # no figures of a bracket that cannot be made as planned
        if 'base' in p:
            bb = p['base']
            fix = 'tube clamp' if b.kind == 'clamp' else BR.FIXING_LABEL.get(bb.get('fixing'), '')
            f['base'].setText(f'{bb["length_mm"]:.1f} × {bb["width_mm"]:.1f} × {bb["thickness_mm"]:.1f} mm · {fix}')
        if 'height_mm' in p:
            f['height'].setText(f'{p["height_mm"]:.1f} mm · optical centre {p["oc_height_mm"]:.1f} mm')
            f['angle'].setText(f'{p["plate_angle_deg"]:.1f}° to the base')
            f['lever'].setText(f'{p["lever_mm"]:.0f} mm')
        if 'fov_blocked_pct' in p:
            f['fov'].setText(f'{p["fov_blocked_pct"]:.1f} %')
        wc = b.check('wall')
        if wc is not None:
            f['wall'].setText(f'{wc["value"]:.1f} mm')
        if b.generated:
            f['solid'].setText(f'{b.mesh["volume_mm3"] / 1000:.1f} cm³ · {b.mesh["mass_g"]:.0f} g · '
                               f'{b.mesh["triangles"]:,} triangles')
            pr = p.get('print') or []
            if pr:
                f['print'].setText(f'{pr[0]["orientation"]} · {pr[0]["overhang_mm2"] / 100:.1f} cm² overhang')
            s = p.get('stiffness')
            f['stiff'].setText(f'≈ {s["first_mode_hz"]:.0f} Hz first mode (beam estimate)' if s else
                               'rigid: the plate sits on the base')
        for chk in b.checks:
            line = StatusLine(chk['severity'], chk['text'])
            self.check_lay.insertWidget(self.check_lay.count() - 1, line)
            self.check_lines.append(line)
        self.notes.setText('\n'.join(b.notes))

    # ------------------------------------------------------------------ actions
    def _move_sel(self):
        if self.sel_uid and self.ctl.move_out([self.sel_uid]):
            self.ctx.msg('Moved the camera out for a printed bracket: Generate again', 4000)

    def _move_all(self):
        uids = [c.uid for c in self._cams() if self.ctl.movable(c.uid)]
        if self.ctl.move_out(uids):
            self.ctx.msg(f'Moved {len(uids)} camera{"s" * (len(uids) != 1)} out for printed brackets', 4000)

    def export_selected(self, path=None):
        """The selected camera's STL; path None asks for it. Returns the path written or None."""
        b = self.ctl.meshed(self.sel_uid) if self.sel_uid else None
        if b is None:
            return None
        if path is None:
            path, _ = QFileDialog.getSaveFileName(self, f'Export the {b.tag} bracket', f'{BR.safe_tag(b.tag)}.stl',
                                                  'STL (*.stl)')
            if not path:
                return None
        with open(path, 'wb') as fh:
            fh.write(BR.stl(b))
        self.ctx.msg(f'Exported the {b.tag} bracket to {path}', 5000)
        return path

    def export_all(self, folder=None):
        """Every STL, the JSON and the PNG sheet under folder/brackets; folder None asks. Returns the paths."""
        res = self.ctl.result
        if res is None:
            return []
        if folder is None:
            folder = QFileDialog.getExistingDirectory(self, 'Export the brackets into')
            if not folder:
                return []
        paths = BR.export(res, self.ctx.store.project, folder)
        self.ctx.msg(f'Exported {sum(p.endswith(".stl") for p in paths)} brackets to '
                     f'{os.path.join(folder, "brackets")}', 5000)
        return paths
