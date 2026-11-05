"""Analysis ▸ Risk-weighted coverage: the rig's coverage weighted by where hazards appear around the car in the seconds
before a crash (core.risk_coverage: NHTSA's pre-crash scenario statistics and the Euro NCAP test scenarios, cited).

Left: the weighting (crash frequency, cost, fatal crashes, functional years lost, or the Euro NCAP scenarios
equally), the NHTSA dataset, the coverage test (resolved, seen, redundant), the time before the conflict. Centre: the
hazard map of a class over the plan view with the rig's coverage (uncovered ground grey, the hazard density in warm
tones, the regions' rings). Right: weighted against unweighted per class, the largest gaps with the scenarios that put
them there, the scenario list with its sources and the requirement proposal. The study takes a couple of seconds: it
runs in a background thread after every edit while the page shows."""
from __future__ import annotations

import math
import threading
import traceback

import numpy as np
from PyQt6.QtCore import QObject, QPointF, QRectF, Qt, QTimer, pyqtSignal
from PyQt6.QtGui import QColor, QPainter, QPen
from PyQt6.QtWidgets import (QAbstractItemView, QComboBox, QFileDialog, QHBoxLayout, QHeaderView, QProgressBar,
                             QScrollArea, QSplitter, QTableWidget, QTableWidgetItem, QVBoxLayout, QWidget)

from ...core import risk_coverage as RC
from ..theme import THEME, mono, sans
from ..widgets import ROW_H, ScrubSpinBox, Section, SegmentedControl, ToggleSwitch, button, hline, label

AUTO_DELAY_MS = 400
FIELDS = (('ttc_min', 'From', 1, 's', 0.1, 'Earliest time before the conflict counted (time to collision)'),
          ('ttc_max', 'To', 1, 's', 0.5, 'Latest time before the conflict counted'),
          ('ttc_samples', 'Samples', 0, '', 1.0, 'Instants between the two, evenly'),
          ('area_m', 'Area out to', 0, 'm', 5.0, 'The unweighted figure counts the ground out to this range'),
          ('map_m', 'Map half width', 0, 'm', 5.0, 'The fine hazard map\'s extent around the car'),
          ('cell_m', 'Map cell', 2, 'm', 0.1, 'The fine hazard map\'s cell'))
FIELD_LABEL = {k: t for k, t, *_ in FIELDS}
FIELD_LABEL.update(weight='weighting', dataset='NHTSA data', metric='coverage test', extended='extended range')


class _Sig(QObject):
    done = pyqtSignal(int, object)
    failed = pyqtSignal(int, str)
    progress = pyqtSignal(int, float)


def _table(cols):
    t = QTableWidget(0, len(cols))
    t.verticalHeader().setVisible(False)
    t.verticalHeader().setDefaultSectionSize(ROW_H)
    t.horizontalHeader().setFixedHeight(ROW_H)
    t.setShowGrid(False)
    t.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
    t.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
    for c, (head, tip) in enumerate(cols):
        it = QTableWidgetItem(head)
        it.setToolTip(tip)
        t.setHorizontalHeaderItem(c, it)
    hh = t.horizontalHeader()
    hh.setSectionResizeMode(0, QHeaderView.ResizeMode.Stretch)
    for c in range(1, len(cols)):
        hh.setSectionResizeMode(c, QHeaderView.ResizeMode.ResizeToContents)
    return t


def _pct(v, nd=0):
    return '—' if v is None else f'{v * 100:.{nd}f} %'


# =========================================================================== the plan view
class HazardMap(QWidget):
    """The fine hazard density of a class over the plan view (x forward up, y left to the left), the ground the rig
    does not cover in grey, the body and the region grid."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.res = None
        self.cls = 'pedestrian'
        self.half = 40.0
        self.setMinimumSize(320, 320)
        self.setMouseTracking(True)

    def set_data(self, res, cls, half=None):
        self.res, self.cls = res, cls
        if half:
            self.half = half
        self.update()

    def _frame(self):
        s = min(self.width(), self.height()) - 16
        return QRectF((self.width() - s) / 2, (self.height() - s) / 2, s, s)

    def _px(self, r, x, y, o):
        k = r.width() / (2 * self.half)
        return QPointF(r.center().x() - (y - o[1]) * k, r.center().y() - (x - o[0]) * k)

    def paintEvent(self, e):
        p = QPainter(self)
        p.fillRect(self.rect(), THEME.color('panel'))
        r = self._frame()
        res = self.res
        if res is None or self.cls not in res.fine:
            p.setPen(THEME.color('muted'))
            p.setFont(sans(11))
            p.drawText(r, Qt.AlignmentFlag.AlignCenter, 'No result yet')
            p.end()
            return
        f = res.fine[self.cls]
        xe, ye = res.fine['x_edges'], res.fine['y_edges']
        o = res.grid.origin_xy
        D, cov, on = f['density'], f['covered'], f['on']
        cell = res.fine['cell_m']
        k = r.width() / (2 * self.half)
        w = max(cell * k, 1.0)
        grey = THEME.color('muted', 0.3)
        pos = D > 0
        lo = hi = 0.0
        if pos.any():
            lv = np.log10(D[pos])
            lo, hi = float(np.percentile(lv, 5)), float(lv.max())
        xs = (xe[:-1] + xe[1:]) / 2
        ys = (ye[:-1] + ye[1:]) / 2
        mx = np.abs(xs - o[0]) <= self.half
        my = np.abs(ys - o[1]) <= self.half
        ix, iy = np.nonzero(on & ~cov & mx[:, None] & my[None, :])
        for a, b in zip(ix, iy):
            q = self._px(r, xs[a], ys[b], o)
            p.fillRect(QRectF(q.x() - w / 2, q.y() - w / 2, w, w), grey)
        ix, iy = np.nonzero(pos & mx[:, None] & my[None, :])
        for a, b in zip(ix, iy):
            t = min(max((math.log10(D[a, b]) - lo) / max(hi - lo, 1e-9), 0.05), 1.0)
            q = self._px(r, xs[a], ys[b], o)
            c = QColor(255, int(170 - 120 * t), 40, int(60 + 190 * t))
            p.fillRect(QRectF(q.x() - w / 2, q.y() - w / 2, w, w), c)
        p.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        p.setPen(QPen(THEME.color('hair'), 1))
        c0 = self._px(r, o[0], o[1], o)
        for e_ in res.grid.range_edges:
            rr = e_ * k
            p.drawEllipse(c0, rr, rr)
        b = res.body
        a0 = self._px(r, b.x_front, b.width / 2, o)
        a1 = self._px(r, b.x_rear, -b.width / 2, o)
        p.setPen(QPen(THEME.color('ink'), 1.5))
        p.drawRect(QRectF(a0, a1).normalized())
        d = res.classes.get(self.cls) or {}
        p.setPen(THEME.color('ink'))
        p.setFont(sans(11))
        p.drawText(QRectF(8, 4, self.width() - 16, 18), Qt.AlignmentFlag.AlignLeft,
                   f'{RC.CLASS_LABEL[self.cls]}: {_pct(d.get("weighted"))} of the pre-crash exposure covered, '
                   f'{_pct(d.get("area"))} of the ground out to {res.settings.area_m:g} m')
        p.setPen(THEME.color('muted'))
        p.setFont(mono(10))
        p.drawText(QRectF(8, self.height() - 18, self.width() - 16, 16), Qt.AlignmentFlag.AlignLeft,
                   f'±{self.half:g} m · warm: hazard density (log) · grey: ground the rig does not cover')
        p.end()

    def wheelEvent(self, e):
        self.half = min(max(self.half * (0.85 if e.angleDelta().y() > 0 else 1.18), 8.0), 300.0)
        self.update()


# =========================================================================== the page
class RiskPage(QWidget):
    def __init__(self, ctx):
        super().__init__()
        self.ctx = ctx
        self.result: RC.RiskResult | None = None
        self.busy = False
        self.stale = False
        self._gen = 0
        self._for = None
        self._syncing = False
        self.coverage = None                     # a coverage object for tests; None: the cameras' perception test
        self.status_text = ''
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
        split.addWidget(self._centre())
        split.addWidget(self._right())
        split.setSizes([300, 640, 620])
        lay.addWidget(split)
        self.timer = QTimer(self)
        self.timer.setSingleShot(True)
        self.timer.setInterval(AUTO_DELAY_MS)
        self.timer.timeout.connect(lambda: self.run(quiet=True))
        ctx.store.changed.connect(self._on_store)
        ctx.store.project_replaced.connect(self._replaced)
        THEME.changed.connect(lambda *_: self.refresh())
        self._sync_fields()
        self.refresh()

    def _left(self):
        outer = QWidget()
        outer.setObjectName('panel')
        outer.setMinimumWidth(280)
        ol = QVBoxLayout(outer)
        ol.setContentsMargins(0, 0, 0, 0)
        ol.setSpacing(0)
        head = QWidget()
        hl = QVBoxLayout(head)
        hl.setContentsMargins(8, 6, 8, 4)
        hl.setSpacing(4)
        hl.addWidget(label('Risk-weighted coverage'))
        hl.addWidget(hline())
        hl.addWidget(label('Where the other road user is in the seconds before a crash, from NHTSA\'s pre-crash '
                           'scenarios and the Euro NCAP test scenarios, and how much of it the rig covers.',
                           muted=True, small=True, wrap=True))
        ol.addWidget(head)
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QScrollArea.Shape.NoFrame)
        body = QWidget()
        body.setObjectName('panel')
        bl = QVBoxLayout(body)
        bl.setContentsMargins(0, 0, 0, 0)
        bl.setSpacing(0)
        self.fields, self.segs = {}, {}
        s = Section('Weighting')
        seg = SegmentedControl([('frequency', 'Crashes'), ('cost', 'Cost'), ('fatal', 'Fatal'), ('fyl', 'FYL'),
                                ('ncap', 'NCAP')])
        seg.setToolTip('Crashes, cost or fatal crashes of each NHTSA pre-crash scenario; functional years lost (Najm '
                       '2007); or every Euro NCAP scenario equally')
        seg.changed.connect(lambda k: self.set_option('weight', k))
        self.segs['weight'] = seg
        s.row('By', seg)
        seg = SegmentedControl([('swanson2019', '2011–15'), ('najm2007', '2004')])
        seg.setToolTip('NHTSA data for the vehicle and cyclist rows: Swanson et al. 2019 (2011–2015 FARS and GES) or '
                       'Najm et al. 2007 (2004 GES)')
        seg.changed.connect(lambda k: self.set_option('dataset', k))
        self.segs['dataset'] = seg
        s.row('NHTSA data', seg)
        seg = SegmentedControl([('resolved', 'Resolved'), ('seen', 'Seen'), ('redundant', '2+ cams')])
        seg.setToolTip('Covered when a camera resolves the object (in view, not hidden, tall enough in pixels: the '
                       'perception range study), sees it, or two or more cameras resolve it')
        seg.changed.connect(lambda k: self.set_option('metric', k))
        self.segs['metric'] = seg
        s.row('Covered', seg)
        self.weight_note = label('', muted=True, small=True, wrap=True)
        s.add(self.weight_note)
        bl.addWidget(s)
        s = Section('Before the conflict')
        for key, text, dec, unit, step, tip in FIELDS:
            f = ScrubSpinBox('', 'plain', dec, unit, step=step, width=90)
            f.setToolTip(tip)
            f.committed.connect(lambda v, k=key: self.set_option(k, v))
            self.fields[key] = f
            s.row(text, f, tip=tip)
        bl.addWidget(s)
        s = Section('Issues below')
        self.th_fields = {}
        for c in RC.CLASSES:
            f = ScrubSpinBox('', 'plain', 0, '%', step=5.0, width=90)
            tip = (f'The Issues list flags {c}s when their risk-weighted coverage is below this (default '
                   f'{RC.ISSUE_MIN_DEFAULT[c]:g} %: round 14\'s proposal on the sample\'s Data rig); a requirements '
                   'scorecard row on risk-weighted coverage takes precedence')
            f.setToolTip(tip)
            f.committed.connect(lambda v, k=c: self.set_threshold(k, v))
            self.th_fields[c] = f
            s.row(RC.CLASS_LABEL[c], f, tip=tip)
        self.th_note = label('', muted=True, small=True, wrap=True)
        s.add(self.th_note)
        bl.addWidget(s)
        bl.addStretch(1)
        scroll.setWidget(body)
        ol.addWidget(scroll, 1)
        foot = QWidget()
        fl = QVBoxLayout(foot)
        fl.setContentsMargins(8, 4, 8, 8)
        self.bar = QProgressBar()
        self.bar.setRange(0, 1000)
        self.bar.setTextVisible(False)
        self.bar.setFixedHeight(4)
        fl.addWidget(self.bar)
        self.status = label('', muted=True, small=True, wrap=True)
        fl.addWidget(self.status)
        row = QHBoxLayout()
        self.b_run = button('Run', 'primary', 'play', 'Run the study now')
        self.b_run.clicked.connect(lambda: self.run())
        self.b_export = button('Export…', None, 'export', 'JSON (densities per region, every weight traced to its '
                                                          'table row), CSV and the maps as PNG')
        self.b_export.clicked.connect(lambda: self.export())
        row.addWidget(self.b_run)
        row.addStretch(1)
        row.addWidget(self.b_export)
        fl.addLayout(row)
        ol.addWidget(foot)
        return outer

    def _centre(self):
        w = QWidget()
        w.setObjectName('panel')
        cl = QVBoxLayout(w)
        cl.setContentsMargins(8, 4, 8, 6)
        self.head = label('', ink2=True, wrap=True)
        cl.addWidget(self.head)
        self.c_cls = SegmentedControl([(c, RC.CLASS_LABEL[c]) for c in RC.CLASSES], 'pedestrian')
        self.c_cls.changed.connect(lambda _: self.refresh())
        cl.addWidget(self.c_cls)
        self.map = HazardMap()
        cl.addWidget(self.map, 1)
        return w

    def _right(self):
        right = QSplitter(Qt.Orientation.Vertical)
        right.setHandleWidth(2)
        a = QWidget()
        a.setObjectName('panel')
        al = QVBoxLayout(a)
        al.setContentsMargins(8, 6, 8, 6)
        al.addWidget(label('Weighted against unweighted', bold=True))
        self.t_cls = _table((('Class', 'Road user'), ('Exposure', 'Share of the pre-crash exposure covered'),
                             ('Ground', 'Share of the ground out to the area range covered'),
                             ('Largest gap', 'The region with the most uncovered exposure')))
        al.addWidget(self.t_cls)
        al.addWidget(label('Largest gaps', bold=True))
        self.t_gap = _table((('Region', 'Sector · range'), ('Uncovered', 'Share of the class\'s exposure uncovered '
                                                                         'there'),
                             ('Covered', 'Covered share of the region\'s exposure'), ('Scenarios', 'What puts it '
                                                                                                  'there')))
        al.addWidget(self.t_gap)
        right.addWidget(a)
        b = QWidget()
        b.setObjectName('panel')
        bl = QVBoxLayout(b)
        bl.setContentsMargins(8, 6, 8, 6)
        bl.addWidget(label('Scenarios and sources', bold=True))
        self.t_src = _table((('Scenario', 'The kinematic template'), ('Weight', 'Its share of the class'),
                             ('Covered', 'Covered share of its exposure'), ('Source', 'Table row or protocol section')))
        bl.addWidget(self.t_src, 1)
        self.req = label('', small=True, wrap=True)
        self.req.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        bl.addWidget(self.req)
        right.addWidget(b)
        right.setSizes([420, 420])
        return right

    def _engine(self):
        """The window's shared engine (its occlusion maps are reused), else a share of the analysis engine."""
        win = getattr(self.ctx, 'window', None)
        try:
            if win is not None and hasattr(win, '_shared_engine'):
                return win._shared_engine()
            return self.ctx.analysis.engine.share()
        except Exception:                                        # noqa: BLE001 — the study makes its own
            return None

    # ------------------------------------------------------------------ settings
    @property
    def settings(self) -> RC.RiskSettings:
        return RC.settings_of(self.ctx.store.project)

    def set_option(self, key, value) -> bool:
        if self._syncing:
            return False
        old = self.settings.to_dict()
        d = dict(old)
        d[key] = value
        new = RC.RiskSettings.from_dict(d).to_dict()
        if new == old:
            self._sync_fields()
            return False
        shown = new[key]
        shown = f'{shown:g}' if isinstance(shown, float) else shown

        def fn(p):
            p.analysis.risk_coverage = new
        self.ctx.store.edit('analysis', fn, f'Risk-weighted coverage · {FIELD_LABEL.get(key, key)} {shown}')
        return True

    def set_threshold(self, cls, value) -> bool:
        """The Issues threshold of a class, % (one undo step; the Issues list follows at once)."""
        if self._syncing or cls not in RC.CLASSES:
            return False
        old = self.settings.to_dict()
        d = dict(old, issue_min=dict(old.get('issue_min') or {}, **{cls: value}))
        new = RC.RiskSettings.from_dict(d).to_dict()
        if new == old:
            self._sync_fields()
            return False

        def fn(p):
            p.analysis.risk_coverage = new
        self.ctx.store.edit('analysis', fn, f'Risk-weighted coverage · Issues below {new["issue_min"][cls]:g} % for '
                                            f'{cls}s')
        self.push_issues()
        return True

    def push_issues(self) -> bool:
        """Put this study's rows of the Issues list up to date from the cache (after a run or a threshold edit),
        without collecting every other study's issues again."""
        an = getattr(self.ctx, 'analysis', None)
        if an is None or not hasattr(an, 'secondary'):
            return False
        try:
            merged = RC.merge_issues(list(getattr(an, 'issues', []) or []), self.ctx.store.project)
            an.secondary.emit('issues', merged)
        except RuntimeError:
            return False
        return True

    def _sync_fields(self):
        st = self.settings
        self._syncing = True
        try:
            th = RC.thresholds(self.ctx.store.project, st)
            for c, f in self.th_fields.items():
                f.set_value(float(th[c]['pct']))
                f.setEnabled(not th[c]['source'].startswith('requirement'))
            req = [f'{c}s: {th[c]["source"]}' for c in RC.CLASSES if th[c]['source'].startswith('requirement')]
            self.th_note.setText('From the requirements scorecard: ' + '; '.join(req) if req else
                                 'Defaults: round 14\'s proposal on the sample\'s Data rig (measured minus 5 %).')
            for k, f in self.fields.items():
                f.set_value(float(getattr(st, k)))
            for k, sg in self.segs.items():
                sg.set_value(getattr(st, k))
            self.segs['dataset'].setEnabled(st.weight not in ('ncap', 'fyl'))
            ds = st.effective_dataset()
            self.weight_note.setText(RC.WEIGHT_LABEL[st.weight] + (
                '' if st.weight == 'ncap' else f' · vehicles and cyclists: {RC.NHTSA_SOURCES[ds]["citation"]} · '
                                               f'pedestrians: {RC.NHTSA_SOURCES["v2p2016"]["citation"]}'))
        finally:
            self._syncing = False

    # ------------------------------------------------------------------ running
    def _key(self):
        """Everything the result depends on (core.risk_coverage.cache_key: cameras, lenses, body, settings, the
        hazard tables), not the Issues thresholds."""
        return RC.cache_key(self.ctx.store.project)

    def _on_store(self, *_):
        self._sync_fields()
        if self.result is not None or self.isVisible():
            if self._for != self._key():
                self.stale = True
                self.refresh_status()
                if self.isVisible():
                    self.timer.start()

    def _replaced(self, *_):
        self._gen += 1
        self.result, self._for, self.stale, self.busy = None, None, False, False
        self._sync_fields()
        self.refresh()
        if self.isVisible():
            self.timer.start()

    def showEvent(self, e):
        super().showEvent(e)
        if self.result is None or self.stale:
            self.timer.start()

    def activate(self):
        if self.result is None or self.stale:
            self.run(quiet=True)

    def run(self, quiet=False) -> bool:
        project = self.ctx.store.project.snapshot()
        key = self._key()
        self._gen += 1
        gen = self._gen
        self.busy = True
        self.status_text = 'Placing the hazards and testing the cameras…'
        self.bar.setValue(0)
        self.refresh_status()
        cov = self.coverage
        eng = self._engine()

        def thread():
            try:
                res = RC.run(project, eng, settings=RC.settings_of(project), coverage=cov,
                             progress=lambda f: self.sig.progress.emit(gen, f), cancel=lambda: gen != self._gen)
                self.sig.done.emit(gen, (key, res))
            except RuntimeError:
                pass
            except Exception as e:                               # noqa: BLE001
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
        key, res = out
        if res is None:
            self.status_text = 'Cancelled'
            self.refresh_status()
            return
        self.result, self._for = res, key
        self.stale = key != self._key()
        if not self.stale:
            self.push_issues()
        self.status_text = f'{res.variant}: {sum(d["points"] for d in res.classes.values())} hazard points · ' \
                           f'{res.seconds:.1f} s'
        self.bar.setValue(1000)
        self.refresh()
        if self.stale and self.isVisible():
            self.timer.start()

    def _failed(self, gen, msg):
        if gen != self._gen:
            return
        self.busy = False
        self.status_text = 'Failed: ' + msg.splitlines()[0]
        self.refresh_status()

    # ------------------------------------------------------------------ views
    def refresh_status(self):
        txt = self.status_text
        if self.stale and not self.busy and self.result is not None:
            txt = (txt + ' · ' if txt else '') + 'out of date'
        self.status.setText(txt)
        self.b_export.setEnabled(self.result is not None)

    def refresh(self):
        self.refresh_status()
        res = self.result
        cls = self.c_cls.value() or 'pedestrian'
        self.map.set_data(res, cls, half=None)
        for t in (self.t_cls, self.t_gap, self.t_src):
            t.setRowCount(0)
        if res is None:
            self.head.setText('Runs on its own when the page shows: the hazards of every cited scenario are placed '
                              'around the car and tested against the cameras.')
            self.req.setText('')
            return
        d = res.classes.get(cls) or {}
        self.head.setText(d.get('text', ''))
        rows = []
        for c, x in res.classes.items():
            rows.append([RC.CLASS_LABEL[c], _pct(x['weighted'], 1), _pct(x['area'], 1),
                         x['gaps'][0]['name'] if x['gaps'] else '—'])
        _fill(self.t_cls, rows)
        _fill(self.t_gap, [[g['name'], _pct(g['uncovered_share'], 1), _pct(g['covered']),
                            ', '.join(t['template'] for t in g['top_uncovered'])] for g in d.get('gaps', [])])
        rows = []
        for it in sorted(d.get('items', []), key=lambda i: -i['weight']):
            tr = it['trace']
            g = tr.get('geometry', tr)
            src = (f'{tr["table"]}: {tr["row"]}' if 'table' in tr else '') + (
                f' · {g.get("protocol", "")} §{g.get("section", "")} {g.get("page", "")}' if g.get('section') else
                (f' · {g.get("protocol", "")}' if g.get('protocol') else ''))
            cov = (d.get('templates') or {}).get(it['template'], {}).get('covered')
            rows.append([f'{it["template"]}: {RC.TEMPLATES[it["template"]]["what"]}', _pct(it['weight'], 2),
                         _pct(cov), src.strip(' ·')])
        _fill(self.t_src, rows)
        props = RC.requirement_proposal(res)
        lo = d.get('left_out') or []
        self.req.setText('Requirement proposals: ' + '; '.join(p['text'] for p in props) +
                         (f'<br>Left out (no road user of these classes): {len(lo)} rows' if lo else '') +
                         f'<br><span style="color:{THEME.tok.get("muted", "#888")}">{d.get("basis", "")}</span>')

    def export(self, folder=None):
        res = self.result
        if res is None:
            return []
        if folder is None:
            folder = QFileDialog.getExistingDirectory(self, 'Export the risk-weighted coverage into')
            if not folder:
                return []
        from ...core import provenance as PV
        paths = PV.stamp_paths(RC.export(res, self.ctx.store.project, folder), self.ctx.store.project, folder)
        self.ctx.msg(f'Exported the risk-weighted coverage to {folder}', 5000)
        return paths


def _fill(t, rows):
    t.setRowCount(len(rows))
    for i, r in enumerate(rows):
        for c, v in enumerate(r):
            it = QTableWidgetItem(str(v))
            it.setToolTip(str(v))
            if c and c < 3:
                it.setTextAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
            t.setItem(i, c, it)
