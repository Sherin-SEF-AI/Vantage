"""Keeps the 'objects in the stitched view' map up to date (core.stitched_objects): recomputed on a worker thread after
every analysis, its warning kept in the Issues list, and seam advice on demand, applied as one undoable edit."""
from __future__ import annotations

import threading
import traceback

from PyQt6.QtCore import QObject, QTimer, pyqtSignal

from ..core import stitched_objects as so
from ..core.rules import SEVERITY_ORDER


class _Sig(QObject):
    done = pyqtSignal(int, object)
    failed = pyqtSignal(int, str)
    advice = pyqtSignal(int, object)


class StitchedController(QObject):
    ready = pyqtSignal()
    advice_ready = pyqtSignal()

    def __init__(self, ctx):
        super().__init__()
        self.ctx = ctx
        self.result: so.StitchedObjects | None = None
        self.advice: so.Advice | None = None
        self.advising = False
        self.status = ''
        self._gen = 0
        self._adv_gen = 0
        self._busy = False
        self._again = False
        self.sig = _Sig()
        self.sig.done.connect(self._done)
        self.sig.failed.connect(self._failed)
        self.sig.advice.connect(self._advice_done)
        self.timer = QTimer(self)
        self.timer.setSingleShot(True)
        self.timer.setInterval(250)
        self.timer.timeout.connect(self._start)
        ctx.analysis.finished.connect(lambda *_: self.schedule())
        ctx.analysis.secondary.connect(self._secondary)
        ctx.store.changed.connect(self._changed)
        ctx.store.project_replaced.connect(self._reset)

    # ---- settings (saved with the project; edits are undoable)
    @property
    def settings(self) -> so.Settings:
        return so.Settings.from_dict(self.ctx.store.project.analysis.stitched)

    def set_settings(self, **kw):
        st = self.settings
        for k, v in kw.items():
            setattr(st, k, float(v))
        st = so.Settings.from_dict(st.to_dict())
        if st.to_dict() == self.settings.to_dict():
            return

        def fn(p):
            p.analysis.stitched = st.to_dict()
        self.ctx.store.edit('analysis', fn, f'Stitched objects · {st.label}, lost below {st.threshold * 100:.0f} %')
        self.schedule()

    # ---- the map
    def schedule(self):
        self.timer.start()

    def _changed(self, sections):
        if 'variant' in sections or 'analysis' in sections or 'cameras' in sections:
            if self.advice is not None or self.advising:
                self._adv_gen += 1
                self.advice, self.advising = None, False
                self.advice_ready.emit()
        if 'analysis' in sections:
            self.schedule()

    def _reset(self, *_):
        self._gen += 1
        self._adv_gen += 1
        self.result, self.advice, self.advising, self.status = None, None, False, ''
        self._busy = False
        self.ready.emit()
        self.advice_ready.emit()

    def _engine(self):
        w = self.ctx.window
        if w is not None and hasattr(w, '_shared_engine'):
            return w._shared_engine()
        e = self.ctx.analysis.engine.share()
        e.vehicle_geom(self.ctx.store.project)
        return e

    def _start(self):
        res = self.ctx.analysis.result
        if res is None:
            self.result, self.status = None, 'Waiting for the analysis'
            self._merge_issue()
            self.ready.emit()
            return
        if self._busy:
            self._again = True
            return
        self._busy = True
        self._gen += 1
        gen = self._gen
        snap = self.ctx.store.project.snapshot()
        eng = self._engine()
        self.status = 'Computing…'

        def work():
            try:
                emit('done', gen, so.analyse(snap, eng, res))
            except Exception as e:
                emit('failed', gen, f'{type(e).__name__}: {e}\n{traceback.format_exc()}')

        def emit(name, *args):
            try:
                getattr(self.sig, name).emit(*args)
            except RuntimeError:            # the window closed meanwhile
                pass
        threading.Thread(target=work, daemon=True).start()

    def _done(self, gen, r):
        self._busy = False
        if gen == self._gen:
            self.result = r
            self.status = ''
            self._merge_issue()
            self.ready.emit()
        if self._again:
            self._again = False
            self.schedule()

    def _failed(self, gen, msg):
        self._busy = False
        if gen == self._gen:
            self.status = msg.splitlines()[0]
            print(msg)
            self.ready.emit()
        if self._again:
            self._again = False
            self.schedule()

    # ---- the Issues list: the analysis worker sends its own list; ours is added to it
    def _with_mine(self, issues):
        out = [i for i in issues if i.action != 'stitched']
        mine = so.issue(self.result)
        if mine is not None:
            out.append(mine)
            out.sort(key=lambda i: SEVERITY_ORDER.get(i.severity, 9))
        return out

    def _merge_issue(self):
        cur = list(self.ctx.analysis.issues)
        new = self._with_mine(cur)
        if [i.message for i in new] != [i.message for i in cur]:
            self.ctx.analysis.secondary.emit('issues', new)

    def _secondary(self, kind, val):
        if kind == 'issues' and self.result is not None:
            if so.issue(self.result) is not None and not any(i.action == 'stitched' for i in val):
                self.ctx.analysis.secondary.emit('issues', self._with_mine(val))

    # ---- seam advice
    def suggest(self):
        res = self.ctx.analysis.result
        if res is None or not res.seams:
            return False
        self._adv_gen += 1
        g = self._adv_gen
        self.advice, self.advising = None, True
        self.advice_ready.emit()
        snap = self.ctx.store.project.snapshot()
        eng = self._engine()

        def work():
            try:
                a = so.advise(snap, eng, res)
            except Exception:
                traceback.print_exc()
                a = None
            try:
                self.sig.advice.emit(g, a)
            except RuntimeError:
                pass
        threading.Thread(target=work, daemon=True).start()
        return True

    def _advice_done(self, g, a):
        if g != self._adv_gen:
            return
        self.advice, self.advising = a, False
        self.advice_ready.emit()

    def apply_advice(self) -> bool:
        ch = self.advice.changes if self.advice is not None else {}
        if not ch:
            return False

        def fn(p):
            for k, v in ch.items():
                p.variant.seams[k] = v
        self.ctx.store.edit('variant', fn, f'Apply seam advice · {len(ch)} seam{"s" if len(ch) != 1 else ""}')
        self._adv_gen += 1
        self.advice, self.advising = None, False
        self.advice_ready.emit()
        self.ctx.analysis.request()
        return True
