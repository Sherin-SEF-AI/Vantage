"""Keeps the 3D surround-view bowl up to date: camera images (a synthetic parking scene or the installation captures)
projected onto the bowl on a worker thread whenever the cameras, the lenses, the analysis or the bowl shape change."""
from __future__ import annotations

import json
import threading
import traceback

from PyQt6.QtCore import QObject, QTimer, pyqtSignal

from ..core import bowl as bowl_core
from ..core.analysis import Engine
from .render_gl import BowlDraw

SOURCES = [('synthetic', 'Parking scene'), ('captures', 'Installation captures'), ('sectors', 'Camera sectors')]
CAPTURE_SIZE = (640, 512)


class _Sig(QObject):
    done = pyqtSignal(int, object)
    failed = pyqtSignal(int, str)


class BowlController(QObject):
    ready = pyqtSignal()

    def __init__(self, ctx):
        super().__init__()
        self.ctx = ctx
        self.source = 'synthetic'
        self.balance = True
        self.blend = 'linear'
        self.draw: BowlDraw | None = None
        self.mesh = None
        self.texture = None
        self.status = ''
        self.active = False
        self._dirty = True
        self._gen = 0
        self._busy = False
        self._again = False
        self._engine = Engine()             # its own: the worker never shares caches with the analysis thread
        self._captures = {}                 # key -> synthetic capture
        self._world_key = None
        self.sig = _Sig()
        self.sig.done.connect(self._done)
        self.sig.failed.connect(self._failed)
        self.timer = QTimer(self)
        self.timer.setSingleShot(True)
        self.timer.setInterval(300)
        self.timer.timeout.connect(self._start)
        ctx.store.changed.connect(lambda *_: self.schedule())
        ctx.store.project_replaced.connect(self._reset)
        ctx.analysis.finished.connect(lambda *_: self.schedule())

    # ---- settings
    @property
    def shape(self) -> bowl_core.BowlShape:
        """The project's bowl (saved with it; edits are undoable)."""
        return bowl_core.BowlShape.from_dict(self.ctx.store.project.analysis.bowl)

    def set_active(self, on: bool):
        self.active = on
        if on and self._dirty:
            self.schedule()

    def set_shape(self, **kw):
        sh = self.shape
        for k, v in kw.items():
            setattr(sh, k, float(v))
        if sh.to_dict() == self.shape.to_dict():
            return

        def fn(p):
            p.analysis.bowl = sh.to_dict()
        self.ctx.store.edit('analysis', fn, 'Bowl shape ' + ', '.join(f'{k} {float(v):g}' for k, v in kw.items()))
        self.schedule()

    def set_source(self, src):
        self.source = src
        self.schedule()

    def set_options(self, balance=None, blend=None):
        if balance is not None:
            self.balance = bool(balance)
        if blend is not None:
            self.blend = blend
        self.schedule()

    def _reset(self, *_):
        self.draw = self.mesh = self.texture = None
        self._captures = {}
        self._engine = Engine()
        self.schedule()

    # ---- work
    def schedule(self):
        self._dirty = True
        if self.active:
            self.timer.start()

    def _images_from_install(self):
        w = self.ctx.window
        page = getattr(w, '_page_widgets', {}).get('install') if w else None
        return dict(page.images) if page is not None and page.images else {}

    def _start(self):
        res = self.ctx.analysis.result
        if self.source == 'sectors' or res is None:
            self.draw = self.mesh = self.texture = None
            self.status = '' if res is not None else 'Waiting for the analysis'
            self._dirty = False
            self.ready.emit()
            return
        if self._busy:
            self._again = True
            return
        self._busy = True
        self._dirty = False
        self._gen += 1
        gen = self._gen
        snap = self.ctx.store.project.snapshot()
        shape = bowl_core.BowlShape(**self.shape.to_dict())
        source, balance, blend = self.source, self.balance, self.blend
        given = self._images_from_install() if source == 'captures' else {}
        self.status = 'Projecting the cameras onto the bowl…'
        self.ready.emit()

        def work():
            try:
                from ..core.synth import parking_world, render_capture
                eng = self._engine
                cams = [c for c in snap.variant.cameras if c.visible]
                images = {}
                if source == 'captures':
                    images = {c.uid: given[c.uid] for c in cams if c.uid in given}
                else:
                    world = parking_world(snap)
                    wkey = json.dumps([world.markings, world.boxes, world.cylinders, world.centre], default=str)
                    if wkey != self._world_key:
                        self._captures, self._world_key = {}, wkey
                    for c in cams:
                        lens = snap.lens_for(c)
                        key = (c.uid, c.x, c.y, c.z, c.yaw, c.pitch, c.roll, json.dumps(lens.to_dict(), sort_keys=True),
                               json.dumps(snap.module_for(c).to_dict(), sort_keys=True), id(snap.vehicle.mesh))
                        img = self._captures.get(key)
                        if img is None:
                            img = render_capture(snap, eng, c, *CAPTURE_SIZE, world=world, seed=c.slot, noise=1.0)
                            self._captures[key] = img
                        images[c.uid] = img
                    live = {k for k in self._captures if any(k[0] == c.uid for c in cams)}
                    for k in list(self._captures):
                        if k not in live:
                            self._captures.pop(k, None)
                if not images:
                    emit('failed', gen, 'No camera images to project (run the installation check or pick the '
                                         'parking scene).')
                    return
                lut = bowl_core.bowl_lut(snap, eng, res, shape, 1024, 320)
                tex = bowl_core.bowl_texture(lut, images, balance=balance, blend=blend)
                mesh = bowl_core.bowl_mesh(res, shape)
                emit('done', gen, (mesh, tex, len(images)))
            except Exception as e:
                emit('failed', gen, f'{type(e).__name__}: {e}\n{traceback.format_exc()}')

        def emit(name, *args):
            try:
                getattr(self.sig, name).emit(*args)
            except RuntimeError:        # the window (and this controller) closed while the bowl was being computed
                pass
        threading.Thread(target=work, daemon=True).start()

    def _done(self, gen, out):
        self._busy = False
        mesh, tex, n = out
        if gen == self._gen:
            self.mesh, self.texture = mesh, tex
            self.draw = BowlDraw(mesh.vertices, mesh.uv, mesh.faces, tex, version=gen)
            self.status = f'{n} camera(s) on the bowl'
            self.ready.emit()
        if self._again or self._dirty:
            self._again = False
            self.schedule()

    def _failed(self, gen, msg):
        self._busy = False
        self.status = msg.splitlines()[0]
        if '\n' in msg:
            print(msg)
        self.ready.emit()
        if self._again:
            self._again = False
            self.schedule()
