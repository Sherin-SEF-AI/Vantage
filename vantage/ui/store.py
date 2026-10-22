"""Project store (undoable edits, selection, signals) and the background analysis controller."""
from __future__ import annotations

import copy
import threading
import time
import traceback

from PyQt6.QtCore import QObject, pyqtSignal, QTimer
from PyQt6.QtGui import QUndoCommand, QUndoStack

from ..core.analysis import Engine
from ..core.project import Project, Camera, Variant
from ..core import rules

_MERGE_ID = 4242


class EditCommand(QUndoCommand):
    def __init__(self, store, keys, before, after, text, merge_key=None):
        super().__init__(text)
        self.store = store
        self.keys = keys
        self.before = before
        self.after = after
        self.merge_key = merge_key
        self.first = True
        self.t = time.time()
        self.gesture = store.gesture

    def id(self):
        return _MERGE_ID if self.merge_key else -1

    def mergeWith(self, other):
        if other.merge_key is None or other.merge_key != self.merge_key or other.keys != self.keys:
            return False
        if other.gesture != self.gesture:      # a new drag is a new undo step, however soon it follows
            return False
        if other.t - self.t > 2.5:
            return False
        self.after = other.after
        self.t = other.t
        self.setText(other.text())
        return True

    def redo(self):
        if self.first:
            self.first = False
            return
        self.store._restore(self.keys, self.after)

    def undo(self):
        self.store._restore(self.keys, self.before)


class ProjectStore(QObject):
    changed = pyqtSignal(set)              # sections changed
    project_replaced = pyqtSignal()
    selection_changed = pyqtSignal()
    status = pyqtSignal(str, int)          # transient message, ms
    dirty_changed = pyqtSignal(bool)

    def __init__(self, project: Project):
        super().__init__()
        self.project = project
        self.undo = QUndoStack(self)
        self.undo.cleanChanged.connect(lambda c: self.dirty_changed.emit(not c))
        self.sel_cams: list[str] = []
        self.sel_obj: tuple = ()           # ('roi',) ('target', uid) ('vehicle',) ...
        self.gesture = 0                   # bumped when a drag starts
        self.revision = 0                  # bumped by every change to the project (edits, undo, redo, replace)
        self._dragging = False

    @property
    def dragging(self):
        return self._dragging

    @dragging.setter
    def dragging(self, on):
        if on and not self._dragging:
            self.gesture += 1
        self._dragging = bool(on)

    # ------------------------------------------------------------------ project
    def set_project(self, project: Project):
        self.revision += 1
        self.project = project
        self.undo.clear()
        self.sel_cams = [project.variant.cameras[0].uid] if project.variant.cameras else []
        self.sel_obj = ()
        self.project_replaced.emit()
        self.selection_changed.emit()

    @property
    def variant(self) -> Variant:
        return self.project.variant

    def cam(self, uid) -> Camera | None:
        return self.project.variant.cam(uid)

    @property
    def selected(self) -> Camera | None:
        for u in self.sel_cams:
            c = self.cam(u)
            if c:
                return c
        return None

    def selected_cams(self) -> list[Camera]:
        return [c for c in (self.cam(u) for u in self.sel_cams) if c]

    # ------------------------------------------------------------------ selection
    def select_cams(self, uids, add=False):
        uids = [u for u in uids if self.cam(u)]
        if add:
            for u in uids:
                if u in self.sel_cams:
                    self.sel_cams.remove(u)
                else:
                    self.sel_cams.append(u)
        else:
            self.sel_cams = list(uids)
        if self.sel_cams:
            self.sel_obj = ()
        self.selection_changed.emit()

    def select_obj(self, obj: tuple):
        self.sel_obj = obj
        if obj:
            self.sel_cams = []
        self.selection_changed.emit()

    # ------------------------------------------------------------------ edits
    def edit(self, keys, fn, text: str, merge_key=None, sections=None):
        """Run fn(project) as one undoable step covering the given section keys."""
        if isinstance(keys, str):
            keys = [keys]
        keys = [k if k != 'variant' else f'variant:{self.project.active_uid}' for k in keys]
        before = {k: copy.deepcopy(self.project.section(k)) if k != 'vehicle' else self.project.section(k) for k in keys}
        if 'vehicle' in keys:
            before['vehicle'] = {'meta': copy.deepcopy(before['vehicle']['meta']), 'vehicle': copy.copy(self.project.vehicle)}
        result = fn(self.project)
        self.revision += 1
        after = {k: copy.deepcopy(self.project.section(k)) if k != 'vehicle' else None for k in keys}
        if 'vehicle' in keys:
            after['vehicle'] = {'meta': copy.deepcopy(self.project.vehicle.meta()), 'vehicle': self.project.vehicle}
        cmd = EditCommand(self, keys, before, after, text, merge_key)
        self.undo.push(cmd)
        self.project.add_log('edit', text)
        self.changed.emit(set(sections or self._sections(keys)))
        return result

    def _sections(self, keys):
        out = set()
        for k in keys:
            out.add('variant' if k.startswith('variant:') else k)
        return out

    def _restore(self, keys, state):
        for k in keys:
            if k == 'vehicle':
                v = state[k]['vehicle']
                self.project.restore('vehicle', {'meta': state[k]['meta'], 'vehicle': v})
            else:
                self.project.restore(k, copy.deepcopy(state[k]))
        self.revision += 1
        # drop selection of removed cameras
        self.sel_cams = [u for u in self.sel_cams if self.cam(u)]
        self.changed.emit(self._sections(keys))
        self.selection_changed.emit()

    def set_cam(self, uid, merge=False, text=None, **props):
        """Set camera properties (and its linked partner, mirrored)."""
        cam = self.cam(uid)
        if cam is None:
            return
        def fn(p):
            c = p.variant.cam(uid)
            for k, v in props.items():
                setattr(c, k, v)
            if c.link_uid:
                partner = p.variant.cam(c.link_uid)
                if partner is not None and not partner.locked:      # a locked camera stays where it is
                    mirror_props(c, partner, props.keys())
        partner = self.cam(cam.link_uid) if cam.link_uid else None
        label = text or f"{'Moved' if set(props) & {'x', 'y', 'z'} else 'Changed'} {cam.tag}" + (
            f' to {cam.x if "x" not in props else props["x"]:.3f}, {props.get("y", cam.y):.3f}, {props.get("z", cam.z):.3f}'
            if set(props) & {'x', 'y', 'z'} else '') + (
            (f' · {partner.tag} locked, not mirrored' if partner.locked else f' · {partner.tag} mirrored') if partner else '')
        self.edit('variant', fn, label, merge_key=(uid, tuple(sorted(props))) if merge else None)


def mirror_props(src: Camera, dst: Camera, keys):
    for k in keys:
        if k == 'y':
            dst.y = -src.y
        elif k == 'yaw':
            dst.yaw = -src.yaw
        elif k == 'roll':
            dst.roll = -src.roll
        elif k in ('x', 'z', 'pitch', 'module_id', 'lens_id', 'standoff_mm', 'bracket', 'snap_normal', 'cable_m', 'sync'):
            setattr(dst, k, getattr(src, k))


class AnalysisController(QObject):
    """Runs the coverage analysis on a worker thread, debounced and serialised."""
    started = pyqtSignal()
    finished = pyqtSignal(object)          # AnalysisResult
    secondary = pyqtSignal(str, object)    # ('issues', list) ('worst', tuple) ('mounts', dict) ('targets', list)
    failed = pyqtSignal(str)
    stale_changed = pyqtSignal(bool)
    _rerun = pyqtSignal()                  # from the worker: a run was asked for while it was busy

    def __init__(self, store: ProjectStore):
        super().__init__()
        self.store = store
        self.engine = Engine()
        self.result = None
        self.issues = []
        self.mounts = {}
        self.worst = None
        self.targets = []
        self.stale = False
        self._pending = False
        self._busy = False
        self._gen = 0
        self._lock = threading.Lock()
        self.timer = QTimer(self)
        self.timer.setSingleShot(True)
        self.timer.setInterval(30)
        self.timer.timeout.connect(self._kick)
        self.finished.connect(self._on_finished)
        self.secondary.connect(self._on_secondary)
        self._rerun.connect(self._kick)        # queued into the main thread (a Python thread has no Qt timers)
        self._qlock = threading.Lock()

    def request(self, force=False):
        if not force and not self.store.project.analysis.live:
            self.set_stale(True)
            return
        if self.store.dragging and not force:
            self.set_stale(True)
            return
        self.timer.start()

    def set_stale(self, s):
        if s != self.stale:
            self.stale = s
            self.stale_changed.emit(s)

    def _kick(self):
        with self._qlock:
            if self._busy:
                self._pending = True
                return
            self._busy = True
        self._gen += 1
        gen = self._gen
        self.started.emit()
        project = self.store.project.snapshot()
        threading.Thread(target=self._run, args=(project, gen, self.store.revision), daemon=True).start()

    def _run(self, project, gen, revision=0):
        try:
            with self._lock:
                res = self.engine.analyse(project)
            if res is not None:
                res.revision = revision
            self.finished.emit(res)
            if res is None:
                return
            vg = self.engine.vg
            mounts = {}
            for c in project.variant.cameras:
                try:
                    mounts[c.uid] = rules.mount_info(project, c, vg)
                except Exception:
                    pass
            self.secondary.emit('mounts', mounts)
            self.secondary.emit('issues', rules.collect_issues(project, res, vg, mounts))
            if project.targets:
                from ..core.targets import evaluate_targets
                with self._lock:
                    self.secondary.emit('targets', evaluate_targets(self.engine, project))
            if not self._pending:
                from ..core.tolerance import worst_single_camera
                with self._lock:
                    w = worst_single_camera(self.engine, project, project.analysis.tol_deg, project.analysis.tol_mm / 1000.0,
                                            cell=max(project.analysis.cell_m, 0.05))
                self.secondary.emit('worst', (gen, w))
        except Exception as e:
            self.failed.emit(f'{type(e).__name__}: {e}\n{traceback.format_exc()}')
        finally:
            with self._qlock:
                self._busy = False
                again, self._pending = self._pending, False
            if again:
                self._rerun.emit()

    def _on_finished(self, res):
        self.result = res
        # an older run finishing after an edit (manual mode) is not up to date
        self.set_stale(res is not None and getattr(res, 'revision', self.store.revision) != self.store.revision)
        if res is not None:
            self.store.project.add_log('analysis', f'{len(res.cams)} cameras · {res.metrics["cells"]:,} cells at '
                                                   f'{res.cell * 100:.0f} cm · {res.ms:.0f} ms'.replace(',', ' '))

    def _on_secondary(self, kind, val):
        if kind == 'issues':
            self.issues = val
        elif kind == 'mounts':
            self.mounts = val
        elif kind == 'worst':
            gen, w = val
            self.worst = w
        elif kind == 'targets':
            self.targets = val

    def run_now(self):
        self.request(force=True)
