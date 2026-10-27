"""Export ▸ Automation: regenerate what downstream needs when the design changes and run the user's own checks.

AutomationController watches the project (on save, after a quiet period of changes, or by hand) and runs
core.automation.run on a background thread: the exports into a folder next to the project, the requirements check,
then the command kept in the user's own settings for this project file, once approved. AutomationPage shows the
settings (undoable, saved with the project), the command (this computer only), every run's output, exit code,
duration and changed files, and the history of runs.

A project file can never make Vantage run a command: the command is read only from CommandStore (QSettings, keyed by
the project file's path), set on this page, shown in full and approved by the user before its first run; changing it
withdraws the approval.
"""
from __future__ import annotations

import os
import threading
import time
import traceback
import weakref

from PyQt6.QtCore import QCoreApplication, QObject, Qt, QThread, QTimer, pyqtSignal
from PyQt6.QtGui import QColor, QTextCursor
from PyQt6.QtWidgets import (QAbstractItemView, QCheckBox, QComboBox, QFileDialog, QGridLayout, QHBoxLayout,
                             QHeaderView, QLineEdit, QPlainTextEdit, QProgressBar, QScrollArea, QSizePolicy, QSplitter,
                             QTableWidget, QTableWidgetItem, QVBoxLayout, QWidget)

from ...core import automation as AU
from ..dialogs.base import VDialog
from ..theme import STATUS, THEME, mono
from ..widgets import (ROW_H, ScrubSpinBox, SegmentedControl, StatusLine, ToggleSwitch, button, hline, label)
from .recording_ws import Tile

VERDICT_SEV = {'pass': 'good', 'fail': 'serious', 'incomplete': 'warning', 'empty': None}
CHANGE_WORD = {'added': 'New', 'changed': 'Changed', 'rewritten': 'Rewritten', 'removed': 'Removed'}
CHANGE_TIP = {'added': 'A file that was not in the folder before',
              'changed': 'Its content differs (time stamps the exports carry do not count)',
              'rewritten': 'A PDF written again: it embeds its creation time, so it cannot be compared',
              'removed': 'Gone from the folder: the previous run wrote it and this one did not (or the command '
                         'deleted it)'}


# ====================================================================== the controller
class _Bridge(QObject):
    """Signals a worker thread emits; queued to the UI thread."""
    progress = pyqtSignal(str, float, str)
    output = pyqtSignal(str, str, int)
    done = pyqtSignal(object)


class _Worker(QThread):
    """A run on a Qt thread: the PDFs (review pack, report) draw with Qt objects that need the thread's own event
    dispatcher, which a plain Python thread does not have."""

    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def run(self):
        self.fn()


class AutomationController(QObject):
    """Runs the project's automation when its trigger fires, one run at a time (a trigger during a run queues one more
    run). Lives on ctx.automation for the whole session (AutomationController.of(ctx))."""
    started = pyqtSignal(object)               # RunRecord
    progress = pyqtSignal(str, float, str)     # run id, fraction, text
    output = pyqtSignal(str, str, int)         # run id, text, characters of output so far (this text included)
    finished = pyqtSignal(object)              # RunRecord
    changed = pyqtSignal()                     # the command, its approval or the history changed

    def __init__(self, ctx, history: AU.History | None = None):
        super().__init__()
        self.ctx = ctx
        self.commands = AU.CommandStore(ctx.settings)
        self.history = history or AU.default_history()
        self.current: AU.RunRecord | None = None
        self._cancel: threading.Event | None = None
        self._thread: _Worker | None = None
        self._queued: str | None = None
        self.timer = QTimer(self)               # the quiet period after changes
        self.timer.setSingleShot(True)
        self.timer.timeout.connect(lambda: self._auto('change'))
        self._saved = self._saved_key()
        self._design = self._design_key()
        b = self._bridge = _Bridge()
        b.progress.connect(self.progress)
        b.output.connect(self.output)
        b.done.connect(self._done)
        store = ctx.store
        store.changed.connect(self._on_changed)
        store.project_replaced.connect(self._on_replaced)
        store.undo.cleanChanged.connect(self._on_clean)
        app = QCoreApplication.instance()
        if app is not None:
            app.aboutToQuit.connect(self.shutdown)

    @classmethod
    def of(cls, ctx) -> 'AutomationController':
        c = getattr(ctx, 'automation', None)
        if c is None:
            c = ctx.automation = cls(ctx)
        return c

    # ------------------------------------------------------------------ state
    @property
    def project(self):
        return self.ctx.store.project

    @property
    def settings(self) -> AU.AutomationSettings:
        return AU.settings_of(self.project)

    def command(self) -> AU.CommandConfig | None:
        return self.commands.get(self.project.path) if self.project.path else None

    @property
    def busy(self) -> bool:
        return self.current is not None

    def _saved_key(self):
        """(the project object, its file, when it was saved): a save changes the time, opening another project
        the object (which is not a save)."""
        p = self.project
        return (weakref.ref(p), AU.norm_path(p.path), p.saved_at) if p.path and p.saved_at else (weakref.ref(p),)

    def _design_key(self):
        """The project's analysis settings without the automation's own (editing those is not a design change)."""
        d = self.project.analysis.to_dict()
        d.pop('automation', None)
        return repr(sorted(d.items(), key=lambda kv: kv[0]))

    # ------------------------------------------------------------------ triggers
    def on_saved(self, path=None) -> bool:
        """The project was saved (MainWindow.save calls this; a save is also noticed when the undo stack becomes
        clean with a new saved time). Runs when automation is on with the save trigger."""
        key, old = self._saved_key(), self._saved
        self._saved = key
        if len(key) < 3 or key == old or old[0]() is not self.project:
            return False                        # nothing saved, the same save, or another project just opened
        return self._auto('save')

    def _on_clean(self, clean):
        if clean:
            self.on_saved()

    def _on_changed(self, sections):
        key = self._design_key()
        if set(sections) <= {'analysis'} and key == self._design:
            return                              # only the automation settings changed
        self._design = key
        st = self.settings
        if st.enabled and st.trigger == 'change' and self.project.path:
            self.timer.start(int(st.quiet_s * 1000))

    def _on_replaced(self):
        self.timer.stop()
        self._saved = self._saved_key()
        self._design = self._design_key()
        self.changed.emit()

    def _auto(self, trigger) -> bool:
        st = self.settings
        if not (st.enabled and st.trigger == trigger and self.project.path):
            return False
        return self.start(trigger)

    # ------------------------------------------------------------------ runs
    def start(self, trigger='manual') -> bool:
        """Regenerate and, when this computer has an approved command for the project file, run it. A run already
        in progress queues one more. False when the project has no file yet."""
        p = self.project
        if not p.path:
            self.ctx.msg('Automation: save the project first; the export folder is relative to the project file.',
                         6000)
            return False
        if self.current is not None:
            self._queued = trigger
            return True
        self.timer.stop()
        st = self.settings
        cfg = self.commands.get(p.path)
        ok = self.commands.approved_command(p.path)         # the only source of a command: this user's settings
        command, state, timeout = None, 'none', None
        if cfg is not None:
            if not st.run_command:
                state = 'off'
            elif ok is not None:
                command, state, timeout = ok.command, 'pending', ok.timeout_s
            else:
                state = 'not_approved'
        snap = p.snapshot()
        win = self.ctx.window
        eng = win._shared_engine() if win is not None and hasattr(win, '_shared_engine') else \
            self.ctx.analysis.engine.share()
        ares = self.ctx.analysis.result
        fresh = ares is not None and getattr(ares, 'revision', -1) == self.ctx.store.revision and \
            not self.ctx.analysis.stale
        try:
            prev = self.history.last(p.path, AU.resolve_out_dir(p.path, st.out_dir), complete=True)
        except ValueError:
            prev = None
        rec = AU.RunRecord.new(trigger, p)
        unsaved = not self.ctx.store.undo.isClean()
        self.current, cancel = rec, threading.Event()
        self._cancel = cancel
        b, rid = self._bridge, rec.id

        def work():
            try:
                AU.run(snap, eng, st, None, lambda f, text='': b.progress.emit(rid, f, text), cancel.is_set,
                       command=command, command_state=state, timeout_s=timeout, trigger=trigger,
                       on_output=lambda text: b.output.emit(rid, text, rec.output_n), previous=prev,
                       result=ares if fresh else None, record=rec, unsaved=unsaved)
            except Exception as e:              # noqa: BLE001  run() reports its own failures; this is a last resort
                rec.status, rec.message = 'error', f'{type(e).__name__}: {e}'
                rec.notes.append(traceback.format_exc(limit=4))
                rec.finished = time.time()
            finally:
                try:
                    b.done.emit(rec)
                except RuntimeError:            # the window closed meanwhile
                    pass
        self._thread = _Worker(work)
        self._thread.start()
        self.started.emit(rec)
        if trigger != 'manual':
            self.ctx.msg(f'Automation: regenerating ({AU.TRIGGER_LABEL.get(trigger, trigger).lower()})…', 3000)
        if state == 'not_approved':
            self.ctx.msg('Automation: the command for this project waits for your approval (Export ▸ Automation); '
                         'only the files are regenerated.', 7000)
        return True

    def _done(self, rec):
        if rec is not self.current:
            return
        th = self._thread
        if th is not None:
            th.wait(5000)                       # it returns right after emitting done
        self.current, self._cancel, self._thread = None, None, None
        if rec.project_path:
            try:
                self.history.add(rec)
            except OSError as e:
                rec.notes.append(f'The history could not be saved: {e}')
        self.finished.emit(rec)
        self.changed.emit()
        self.ctx.msg(f'Automation: {AU.STATUS_LABEL.get(rec.status, rec.status).lower()} · {rec.message}', 6000)
        queued, self._queued = self._queued, None
        if queued is not None:
            QTimer.singleShot(0, lambda: self.start(queued) if queued == 'manual' else self._auto(queued))

    def stop(self):
        """Stop the run in progress (the command and everything it started) and drop a queued one."""
        self._queued = None
        self.timer.stop()
        if self._cancel is not None:
            self._cancel.set()

    def shutdown(self, wait=5.0):
        """Stop and wait for the run to end (the app is closing: never leave the command running)."""
        self.stop()
        th = self._thread
        if th is not None and th.isRunning():
            th.wait(int(wait * 1000))

    # ------------------------------------------------------------------ the command
    def set_command(self, command: str, timeout_min: float | None = None) -> AU.CommandConfig | None:
        p = self.project
        if not p.path:
            return None
        cfg = self.commands.put(p.path, command, timeout_min)
        self.changed.emit()
        return cfg

    def approve(self, command: str) -> bool:
        ok = self.commands.approve(self.project.path, command) is not None if self.project.path else False
        self.changed.emit()
        return ok

    def remove_command(self):
        if self.project.path:
            self.commands.remove(self.project.path)
        self.changed.emit()

    def runs(self) -> list:
        """Newest first: the run in progress, then the history of this project file."""
        p = self.project
        rows = self.history.runs(p.path) if p.path else []
        cur = self.current
        if cur is not None and cur.project_path and p.path and AU.norm_path(cur.project_path) == AU.norm_path(p.path):
            rows = [cur] + [r for r in rows if r.id != cur.id]
        return rows


# ====================================================================== the approval
class ApproveDialog(VDialog):
    """The command in full, where and how it runs, and the variables it gets; Allow or Cancel."""

    def __init__(self, parent, project_path, command, out_dir, timeout_min):
        super().__init__(parent, 'Allow this command to run?', os.path.basename(project_path), (640, 470),
                         resizable=True)
        lay = QVBoxLayout(self.body)
        lay.setContentsMargins(14, 12, 14, 12)
        lay.setSpacing(8)
        folder = os.path.dirname(os.path.abspath(project_path))
        lay.addWidget(label(f'Vantage will run this command for {os.path.basename(project_path)} after it '
                            f'regenerates the files, through the system shell with your permissions, in {folder}. '
                            f'It is stopped after {timeout_min:g} min.', wrap=True))
        self.text = QPlainTextEdit(command)
        self.text.setReadOnly(True)
        self.text.setLineWrapMode(QPlainTextEdit.LineWrapMode.WidgetWidth)
        self.text.setFont(mono(12))
        self.text.setMinimumHeight(80)
        lay.addWidget(self.text, 1)
        lay.addWidget(label('It was set here, on this computer, for this project file; a project file cannot set or '
                            'change it. Allow only a command you wrote or have read. Changing it later asks again.',
                            muted=True, small=True, wrap=True))
        env = ', '.join(k for k, _ in AU.ENV_HELP)
        lay.addWidget(label(f'It gets {env}; the export folder is {out_dir or "not set yet"}.', muted=True, small=True,
                            wrap=True))
        self.cancel_b = button('Cancel', 'ghost')
        self.cancel_b.clicked.connect(self.reject)
        self.cancel_b.setDefault(True)          # Enter does not approve by accident
        self.allow_b = button('Allow', 'primary', 'check', 'Approve this command for this project file')
        self.allow_b.setAutoDefault(False)
        self.allow_b.clicked.connect(self.accept)
        self.buttons.addWidget(self.cancel_b)
        self.buttons.addWidget(self.allow_b)
        self.hint.setText('Shown in full: scroll to read all of it.')
        self.cancel_b.setFocus()                # Enter does not approve by accident


# ====================================================================== the page
def _combo(tip):
    """A combo box as narrow as the panel, whatever the length of its items."""
    c = QComboBox()
    c.setSizeAdjustPolicy(QComboBox.SizeAdjustPolicy.AdjustToMinimumContentsLengthWithIcon)
    c.setMinimumContentsLength(12)
    c.setToolTip(tip)
    return c


def _fill_combo(c, items, current):
    """Set a combo's items [(text, data)] (only when they differ: an open list stays open) and pick current."""
    have = [(c.itemText(i), c.itemData(i)) for i in range(c.count())]
    if have != items:
        c.blockSignals(True)
        c.clear()
        for text, data in items:
            c.addItem(text, data)
        c.blockSignals(False)
    c.setCurrentIndex(max(c.findData(current) if current else 0, 0))


def _title(text=''):
    """A bold heading that may be cut at the right instead of widening its pane (it carries paths)."""
    w = label(text, bold=True)
    w.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Preferred)
    w.setMinimumWidth(40)
    return w


def _table(cols):
    t = QTableWidget(0, len(cols))
    t.setHorizontalHeaderLabels(cols)
    t.verticalHeader().setVisible(False)
    t.verticalHeader().setDefaultSectionSize(ROW_H)
    t.horizontalHeader().setFixedHeight(ROW_H)
    t.setShowGrid(False)
    t.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
    return t


HISTORY_COLS = ('When', 'Trigger', 'Result', 'Exit', 'Time', 'Changed')
FILE_COLS = ('File', 'Change', 'By')


class AutomationPage(QWidget):
    def __init__(self, ctx):
        super().__init__()
        self.ctx = ctx
        self.ctl = AutomationController.of(ctx)
        self._dirty = True
        self._shown: AU.RunRecord | None = None         # the run the output and files show
        self._follow = True                             # show the newest run when one starts
        self._syncing = False
        self._cmd_for = None                            # (project file, stored command) the editor shows
        self._shown_n = 0                               # characters of the shown run's output in the view
        self._folder_editing = False                    # the folder field holds text not yet taken
        self._dialog = None                             # the approval dialog while it is open
        self.setObjectName('base')
        lay = QHBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        split = QSplitter(Qt.Orientation.Horizontal)
        split.setHandleWidth(2)
        split.addWidget(self._left())
        split.addWidget(self._right())
        split.setSizes([360, 1240])
        lay.addWidget(split)
        ctx.store.changed.connect(self._on_change)
        ctx.store.project_replaced.connect(self._on_replaced)
        c = self.ctl
        c.started.connect(self._on_started)
        c.progress.connect(self._on_progress)
        c.output.connect(self._on_output)
        c.finished.connect(self._on_finished)
        c.changed.connect(self._on_change)
        THEME.changed.connect(self._on_change)          # a bound slot: disconnected when the page goes
        self.refresh()

    # ------------------------------------------------------------------ settings panel
    def _left(self):
        w = QWidget()
        w.setObjectName('panel')
        ll = QVBoxLayout(w)
        ll.setContentsMargins(8, 6, 8, 8)
        ll.setSpacing(4)
        top = QHBoxLayout()
        top.addWidget(label('Automation'))
        top.addStretch(1)
        top.addWidget(label('regenerate · check · run', muted=True, small=True))
        ll.addLayout(top)
        ll.addWidget(hline())
        ll.addWidget(label('When the design changes, Vantage regenerates the files downstream needs into a folder next '
                           'to the project and runs your own checks. The project keeps what to regenerate; the command '
                           'stays in your own settings on this computer.', muted=True, small=True, wrap=True))
        # ---- when
        ll.addSpacing(6)
        ll.addWidget(label('When', bold=True))
        self.enabled = ToggleSwitch('Automate this project')
        self.enabled.setToolTip('Regenerate (and run your command) when the trigger below fires. Run now works '
                                'either way')
        self.enabled.clicked.connect(lambda on: self._set('on' if on else 'off', enabled=on))
        ll.addWidget(self.enabled)
        self.trigger = SegmentedControl([(k, AU.TRIGGER_LABEL[k]) for k in AU.TRIGGERS], 'save')
        self.trigger.setToolTip('On save: every time the project is saved. After changes: once nothing has changed '
                                'for the quiet period (unsaved edits included). By hand: only Run now')
        self.trigger.changed.connect(lambda k: self._set(AU.TRIGGER_LABEL[k].lower(), trigger=k))
        ll.addWidget(self.trigger)
        g = QGridLayout()
        g.setVerticalSpacing(2)
        g.setHorizontalSpacing(6)
        g.setColumnStretch(1, 1)
        self.quiet = ScrubSpinBox('', 'plain', 0, 's', step=1.0, width=90)
        self.quiet.setToolTip('After changes: run once nothing has changed for this long')
        self.quiet.committed.connect(lambda v: self._set(f'quiet period {v:g} s', quiet_s=v))
        self._row(g, 'Quiet period', self.quiet, 'After changes: run once nothing has changed for this long')
        ll.addLayout(g)
        # ---- what
        ll.addSpacing(6)
        ll.addWidget(label('What', bold=True))
        g = QGridLayout()
        g.setVerticalSpacing(2)
        g.setHorizontalSpacing(6)
        g.setColumnStretch(1, 1)
        fr = QHBoxLayout()
        fr.setSpacing(4)
        self.folder = QLineEdit()
        self.folder.setToolTip('Relative to the project file\'s folder, and never outside it')
        self.folder.editingFinished.connect(self._folder_typed)
        self.folder.textEdited.connect(lambda _: setattr(self, '_folder_editing', True))
        fr.addWidget(self.folder, 1)
        self.b_folder = button('Choose…', 'ghost', None, 'Pick a folder inside the project\'s folder')
        self.b_folder.clicked.connect(self._choose_folder)
        fr.addWidget(self.b_folder)
        fw = QWidget()
        fw.setLayout(fr)
        fr.setContentsMargins(0, 0, 0, 0)
        self._row(g, 'Folder', fw, 'Where the files go, relative to the project file')
        self.variant = _combo('The variant to export (the active one follows the window)')
        self.variant.activated.connect(self._variant_picked)
        self._row(g, 'Variant', self.variant, 'The variant to export')
        self.frame = _combo('The origin of the exported poses')
        self.frame.activated.connect(self._frame_picked)
        self._row(g, 'Frame', self.frame, 'The origin of the exported poses')
        conv = QHBoxLayout()
        conv.setSpacing(6)
        self.units = SegmentedControl([('m', 'm'), ('mm', 'mm')], 'm', min_w=40)
        self.units.changed.connect(lambda k: self._set(f'units {k}', units=k))
        self.angles = SegmentedControl([('deg', 'deg'), ('rad', 'rad')], 'deg', min_w=40)
        self.angles.changed.connect(lambda k: self._set(f'angles {k}', angles=k))
        conv.addWidget(self.units)
        conv.addWidget(self.angles)
        conv.addStretch(1)
        cw = QWidget()
        cw.setLayout(conv)
        conv.setContentsMargins(0, 0, 0, 0)
        self._row(g, 'Units', cw, 'Lengths and angles in the exported files')
        ll.addLayout(g)
        self.folder_problem = StatusLine('warning', '')         # a folder typed that cannot be used
        self.folder_problem.hide()
        ll.addWidget(self.folder_problem)
        self.stored_problem = StatusLine('warning', '')         # what the project's stored settings had to drop
        self.stored_problem.hide()
        ll.addWidget(self.stored_problem)
        ll.addWidget(label('Formats', ink2=True))
        fl = QVBoxLayout()
        fl.setContentsMargins(4, 2, 4, 2)
        fl.setSpacing(1)
        self.format_boxes: dict[str, QCheckBox] = {}
        self._fill_formats(fl)
        fl.addStretch(1)
        host = QWidget()
        host.setObjectName('panel')
        host.setLayout(fl)
        self.formats = QScrollArea()
        self.formats.setWidgetResizable(True)
        self.formats.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.formats.setWidget(host)
        self.formats.setFixedHeight(250)
        self.formats.setToolTip('The export formats to regenerate (images of the window cannot be made in the '
                                'background)')
        ll.addWidget(self.formats)
        self.check = ToggleSwitch('Run the requirements check')
        self.check.setToolTip('The requirements scorecard into <folder>/requirements (JSON, CSV, review pack); its '
                              'exit code is in VANTAGE_CHECK_EXIT')
        self.check.clicked.connect(lambda on: self._set('requirements check ' + ('on' if on else 'off'), check=on))
        ll.addWidget(self.check)
        self.prune = ToggleSwitch('Remove files the last run wrote and this one did not')
        self.prune.setToolTip('A camera removed or a format dropped leaves no stale file behind. Only files Vantage '
                              'wrote in the last run (this computer\'s history); your own files stay')
        self.prune.clicked.connect(lambda on: self._set('remove stale files ' + ('on' if on else 'off'), prune=on))
        ll.addWidget(self.prune)
        self.run_cmd = ToggleSwitch('Run my command afterwards')
        self.run_cmd.setToolTip('Run the command below (this computer only) after the files are written')
        self.run_cmd.clicked.connect(lambda on: self._set('command ' + ('on' if on else 'off'), run_command=on))
        ll.addWidget(self.run_cmd)
        # ---- the command (this computer only)
        ll.addSpacing(6)
        ll.addWidget(label('Command · this computer only', bold=True))
        ll.addWidget(label('Kept in your own settings for this project file, never in the project. It runs through the '
                           'shell in the project\'s folder, only after you approve it; changing it asks again. It is '
                           'stored as plain text: keep passwords and tokens in environment variables, not in it.',
                           muted=True, small=True, wrap=True))
        self.cmd = QPlainTextEdit()
        self.cmd.setPlaceholderText('make perception-test')
        self.cmd.setFixedHeight(62)
        self.cmd.setFont(mono(12))
        self.cmd.setToolTip('A shell command, e.g. make perception-test or ./scripts/check.sh "$VANTAGE_EXPORT_DIR"')
        ll.addWidget(self.cmd)
        g = QGridLayout()
        g.setVerticalSpacing(2)
        g.setHorizontalSpacing(6)
        g.setColumnStretch(1, 1)
        self.timeout = ScrubSpinBox('', 'plain', 1, 'min', step=1.0, width=90)
        self.timeout.committed.connect(self._timeout_set)
        self._row(g, 'Time limit', self.timeout, 'The command is stopped (with everything it started) after this long')
        ll.addLayout(g)
        self.cmd_state = StatusLine(None, '')
        ll.addWidget(self.cmd_state)
        bl = QHBoxLayout()
        self.b_save_cmd = button('Save…', None, None, 'Keep this command for this project file and approve it')
        self.b_save_cmd.clicked.connect(self.save_command)
        self.b_approve = button('Approve…', None, 'check', 'Read the command in full and allow it to run')
        self.b_approve.clicked.connect(lambda: self.approve())
        self.b_remove_cmd = button('Remove', 'ghost', None, 'Forget the command for this project file')
        self.b_remove_cmd.clicked.connect(self.remove_command)
        bl.addWidget(self.b_save_cmd)
        bl.addWidget(self.b_approve)
        bl.addStretch(1)
        bl.addWidget(self.b_remove_cmd)
        ll.addLayout(bl)
        ll.addWidget(label('It gets ' + ' · '.join(f'{k}: {t}' for k, t in AU.ENV_HELP) + '.', muted=True,
                           small=True, wrap=True))
        ll.addStretch(1)
        self.status = label('', muted=True, small=True, wrap=True)
        ll.addWidget(self.status)
        bl = QHBoxLayout()
        self.b_defaults = button('Defaults', 'ghost', None, 'Back to the default automation settings (off)')
        self.b_defaults.clicked.connect(self._defaults)
        bl.addWidget(self.b_defaults)
        bl.addStretch(1)
        self.b_ci = button('CI templates…', None, 'export', 'A GitHub Actions workflow and a GitLab CI job that run '
                                                           'the batch checks on this project in its repository')
        self.b_ci.clicked.connect(lambda: self.export_ci())
        bl.addWidget(self.b_ci)
        ll.addLayout(bl)
        sc = QScrollArea()
        sc.setWidgetResizable(True)
        sc.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        sc.setWidget(w)
        sc.setMinimumWidth(320)
        return sc

    def _row(self, grid, text, w, tip):
        a = label(text, ink2=True)
        a.setFixedWidth(84)
        a.setToolTip(tip)
        i = grid.rowCount()
        grid.addWidget(a, i, 0)
        grid.addWidget(w, i, 1)

    def _fill_formats(self, lay):
        from ...core import exporters as ex
        avail = AU.formats_available()
        for grp in ex.GROUPS:
            rows = [f for f in avail if f[1] == grp]
            if not rows:
                continue
            head = label(grp, muted=True, small=True)
            lay.addSpacing(4)
            lay.addWidget(head)
            for fid, _, lab, ext in rows:
                cb = QCheckBox(f'{lab} · {ext}')
                cb.setToolTip(f'Format id {fid}')
                cb.clicked.connect(lambda on, k=fid: self._format_toggled(k, on))
                lay.addWidget(cb)
                self.format_boxes[fid] = cb

    # ------------------------------------------------------------------ right: runs
    def _right(self):
        w = QWidget()
        w.setObjectName('panel')
        rl = QVBoxLayout(w)
        rl.setContentsMargins(10, 8, 10, 8)
        rl.setSpacing(8)
        tiles = QHBoxLayout()
        tiles.setSpacing(8)
        self.tiles = {}
        for key, caption, tip in (
                ('result', 'Last run', 'The outcome of the run shown: the command\'s exit code, or the requirements '
                                       'check when no command ran'),
                ('exit', 'Command', 'The command\'s exit code'),
                ('time', 'Duration', 'The regeneration and the command'),
                ('files', 'Files', 'Files the regeneration added, changed or removed in the folder (time stamps do '
                                   'not count)'),
                ('req', 'Requirements', 'The requirements check\'s verdict')):
            t = self.tiles[key] = Tile(caption, tip)
            tiles.addWidget(t, 1)
        rl.addLayout(tiles)
        bar = QHBoxLayout()
        bar.setSpacing(6)
        self.b_run = button('Run now', 'primary', 'play', 'Regenerate the files and run your command (asks for '
                                                          'approval first when it has none)')
        self.b_run.clicked.connect(lambda: self.run_now())
        self.b_stop = button('Stop', None, 'stop', 'Stop the run and the command with everything it started')
        self.b_stop.clicked.connect(self.ctl.stop)
        bar.addWidget(self.b_run)
        bar.addWidget(self.b_stop)
        self.bar = QProgressBar()
        self.bar.setRange(0, 1000)
        self.bar.setTextVisible(False)
        self.bar.setFixedHeight(4)
        self.bar.setFixedWidth(160)
        bar.addWidget(self.bar)
        self.run_text = label('', muted=True)
        bar.addWidget(self.run_text, 1)
        self.b_export = button('Export run…', None, 'export', 'The run shown as JSON (without the command\'s text) '
                                                              'and its output as a log')
        self.b_export.clicked.connect(lambda: self.export())
        bar.addWidget(self.b_export)
        rl.addLayout(bar)
        split = QSplitter(Qt.Orientation.Horizontal)
        split.setHandleWidth(2)
        hw = QWidget()
        hl = QVBoxLayout(hw)
        hl.setContentsMargins(0, 0, 0, 0)
        hl.setSpacing(4)
        hl.addWidget(label('History · this computer', bold=True))
        self.history = _table(HISTORY_COLS)
        self.history.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.history.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        hh = self.history.horizontalHeader()
        for c in range(len(HISTORY_COLS)):
            hh.setSectionResizeMode(c, QHeaderView.ResizeMode.ResizeToContents)
        hh.setSectionResizeMode(2, QHeaderView.ResizeMode.Stretch)
        self.history.itemSelectionChanged.connect(self._history_picked)
        hl.addWidget(self.history, 1)
        split.addWidget(hw)
        right = QSplitter(Qt.Orientation.Vertical)
        right.setHandleWidth(2)
        ow = QWidget()
        ol = QVBoxLayout(ow)
        ol.setContentsMargins(0, 0, 0, 0)
        ol.setSpacing(4)
        self.out_title = _title('Output')
        ol.addWidget(self.out_title)
        self.out = QPlainTextEdit()
        self.out.setReadOnly(True)
        self.out.setMaximumBlockCount(20000)
        self.out.setFont(mono(11))
        self.out.setLineWrapMode(QPlainTextEdit.LineWrapMode.NoWrap)
        ol.addWidget(self.out, 1)
        right.addWidget(ow)
        fw = QWidget()
        fl = QVBoxLayout(fw)
        fl.setContentsMargins(0, 0, 0, 0)
        fl.setSpacing(4)
        self.files_title = _title('Files')
        fl.addWidget(self.files_title)
        self.files = _table(FILE_COLS)
        self.files.setSelectionMode(QAbstractItemView.SelectionMode.NoSelection)
        fh = self.files.horizontalHeader()
        fh.setSectionResizeMode(0, QHeaderView.ResizeMode.Stretch)
        fh.setSectionResizeMode(1, QHeaderView.ResizeMode.ResizeToContents)
        fh.setSectionResizeMode(2, QHeaderView.ResizeMode.ResizeToContents)
        fl.addWidget(self.files, 1)
        self.notes = QWidget()
        self.notes_lay = QVBoxLayout(self.notes)
        self.notes_lay.setContentsMargins(0, 0, 0, 0)
        self.notes_lay.setSpacing(2)
        self._note_widgets: list[StatusLine] = []
        fl.addWidget(self.notes)
        right.addWidget(fw)
        right.setSizes([420, 300])
        split.addWidget(right)
        split.setSizes([460, 780])
        rl.addWidget(split, 1)
        return w

    # ------------------------------------------------------------------ settings edits
    @property
    def settings(self) -> AU.AutomationSettings:
        return AU.settings_of(self.ctx.store.project)

    def _set(self, text, **kw):
        """Change settings as one undo step (saved with the project); False when nothing changes."""
        st = self.settings
        for k, v in kw.items():
            setattr(st, k, v)
        new = AU.AutomationSettings.from_dict(st.to_dict()).to_dict()
        if new == self.settings.to_dict() and self.ctx.store.project.analysis.automation:
            self._sync()                        # a clamped or unchanged value: show what is stored
            return False

        def fn(p):
            p.analysis.automation = new
        self.ctx.store.edit('analysis', fn, f'Automation · {text}')
        return True

    def _folder_typed(self):
        if self._syncing:
            return
        text = self.folder.text()
        rel, why = AU.clean_folder(text)
        if rel is None:
            self.folder_problem.set('warning', f'Not used: {why}.')
            self.folder_problem.show()
            return                              # the text stays for the user to correct
        self._folder_editing = False
        self.folder_problem.hide()
        if rel != self.settings.out_dir:
            self._set(f'folder {rel}', out_dir=rel)
        else:
            self.folder.setText(rel)

    def _choose_folder(self):
        p = self.ctx.store.project
        if not p.path:
            self.ctx.msg('Save the project first: the folder is relative to the project file.', 5000)
            return
        base = os.path.dirname(os.path.abspath(p.path))
        try:
            start = AU.resolve_out_dir(p.path, self.settings.out_dir)
        except ValueError:
            start = base
        path = QFileDialog.getExistingDirectory(self, 'Export folder (inside the project\'s folder)',
                                                start if os.path.isdir(start) else base)
        if not path:
            return
        rel = os.path.relpath(path, base)
        clean, why = AU.clean_folder(rel)
        if clean is None:
            self.folder_problem.set('warning', f'Not used: {why}.')
            self.folder_problem.show()
            return
        self.folder_problem.hide()
        self._set(f'folder {clean}', out_dir=clean)

    def _variant_picked(self, i):
        uid = self.variant.itemData(i) or ''
        name = self.variant.itemText(i)
        self._set(f'variant {name}', variant=uid)

    def _frame_picked(self, i):
        self._set(f'frame {self.frame.itemText(i)}', frame=self.frame.itemData(i) or '')

    def _format_toggled(self, fid, on):
        if self._syncing:
            return
        fids = [k for k, cb in self.format_boxes.items() if cb.isChecked()]
        lab = self.format_boxes[fid].text().split(' · ')[0]
        self._set(f'{"+" if on else "−"} {lab}', formats=fids)

    def _defaults(self):
        if self.ctx.store.project.analysis.automation:
            def fn(p):
                p.analysis.automation = {}
            self.ctx.store.edit('analysis', fn, 'Automation · defaults')

    # ------------------------------------------------------------------ the command
    def save_command(self) -> bool:
        """Keep the command typed for this project file and, when it is new or changed, ask for approval."""
        p = self.ctx.store.project
        if not p.path:
            self.ctx.msg('Save the project first: the command is kept for the project file.', 5000)
            return False
        text = self.cmd.toPlainText().strip()
        if not text:
            self.remove_command()
            return False
        self.ctl.set_command(text, self.timeout.value())
        if not self.ctl.commands.is_approved(p.path):
            self.approve()
        else:
            self.ctx.msg('Command kept (unchanged, still approved).', 4000)
        self.refresh()
        return True

    def approve(self) -> bool:
        """Show the stored command in full and ask the user to allow it; True when approved."""
        p = self.ctx.store.project
        cfg = self.ctl.command()
        if cfg is None:
            return False
        try:
            out = AU.resolve_out_dir(p.path, self.settings.out_dir)
        except ValueError:
            out = ''
        d = ApproveDialog(self, p.path, cfg.command, out, cfg.timeout_min)
        self._dialog = d
        ok = bool(d.exec())
        self._dialog = None
        if ok:
            self.ctl.approve(cfg.command)
            self.ctx.msg('Command approved for this project file.', 4000)
        else:
            self.ctx.msg('The command is not approved: it will not run.', 5000)
        self.refresh()
        return ok

    def remove_command(self):
        self.ctl.remove_command()
        self.cmd.setPlainText('')
        self.ctx.msg('Command removed for this project file.', 4000)
        self.refresh()

    def _timeout_set(self, v):
        cfg = self.ctl.command()
        if cfg is not None:
            self.ctl.set_command(cfg.command, v)        # the same command keeps its approval
        self.refresh()

    # ------------------------------------------------------------------ runs
    def run_now(self) -> bool:
        """Run by hand; a command without approval is shown for approval first (declined: nothing runs)."""
        p = self.ctx.store.project
        if not p.path:
            self.ctx.msg('Save the project first: the export folder is relative to the project file.', 5000)
            return False
        cfg = self.ctl.command()
        if cfg is not None and self.settings.run_command and not self.ctl.commands.is_approved(p.path):
            if not self.approve():
                self.ctx.msg('Nothing ran: the command is not approved.', 5000)
                return False
        self._follow = True
        return self.ctl.start('manual')

    def _on_started(self, rec):
        if self._follow or self._shown is None:
            self._shown = rec
            self._show_output(rec)
        self._on_change()

    def _on_progress(self, rid, f, text):
        cur = self.ctl.current
        if cur is None or cur.id != rid:
            return
        self.bar.setValue(int(f * 1000))
        if text:
            self.run_text.setText(f'{text[:1].upper()}{text[1:]}…')

    def _show_output(self, rec):
        """Put a run's output in the view; later chunks of a running one are appended by _on_output."""
        self._shown_n = rec.output_n if rec is not None else 0
        self.out.setPlainText(rec.output if rec is not None else '')
        sb = self.out.verticalScrollBar()
        sb.setValue(sb.maximum())

    def _on_output(self, rid, text, n):
        if self._shown is not None and self._shown.id == rid:
            start = n - len(text)
            if n <= self._shown_n:
                return                          # already in the view (it was shown after this chunk arrived)
            text = text[max(0, self._shown_n - start):]
            self._shown_n = n
            sb = self.out.verticalScrollBar()
            at_end = sb.value() >= sb.maximum() - 2
            cur = self.out.textCursor()
            cur.movePosition(QTextCursor.MoveOperation.End)
            cur.insertText(text)
            if at_end:
                sb.setValue(sb.maximum())

    def _on_finished(self, rec):
        if self._shown is not None and self._shown.id == rec.id:
            self._shown = rec
            self._show_output(rec)
        self._on_change()

    def _history_picked(self):
        if self._syncing:
            return
        rows = self.history.selectionModel().selectedRows()
        if not rows:
            return
        rid = self.history.item(rows[0].row(), 0).data(Qt.ItemDataRole.UserRole)
        rec = next((r for r in self.ctl.runs() if r.id == rid), None)
        if rec is None:
            return
        self._follow = rows[0].row() == 0
        self._shown = rec
        self._show_output(rec)
        self._fill_run(rec)

    # ------------------------------------------------------------------ state
    def _on_change(self, *_):
        if self.isVisible():
            self.refresh()
        else:
            self._dirty = True

    def _on_replaced(self):
        self._shown = None
        self._follow = True
        self.cmd.setPlainText('')
        self._cmd_for = None
        self._on_change()

    def showEvent(self, e):
        super().showEvent(e)
        if self._dirty:
            self.refresh()

    def activate(self):
        if self._dirty:
            self.refresh()

    def _sync(self):
        """The settings fields from the project."""
        self._syncing = True
        p = self.ctx.store.project
        st = self.settings
        self.enabled.setChecked(st.enabled)
        self.trigger.set_value(st.trigger)
        self.quiet.set_value(st.quiet_s)
        self.quiet.setEnabled(st.trigger == 'change')
        if not (self._folder_editing and self.folder.hasFocus()):
            self._folder_editing = False
            self.folder.setText(st.out_dir)
        _fill_combo(self.variant, [(f'Active variant ({p.variant.name})', '')] + [(v.name, v.uid) for v in p.variants],
                    st.variant)
        from ...core.project import FRAME_PRESETS
        _fill_combo(self.frame, [(f'Project\'s ({FRAME_PRESETS.get(p.frame, p.frame)})', '')] +
                    [(t.replace('…', ''), k) for k, t in FRAME_PRESETS.items()], st.frame)
        self.units.set_value(st.units)
        self.angles.set_value(st.angles)
        on = set(st.formats)
        for fid, cb in self.format_boxes.items():
            cb.setChecked(fid in on)
        for t, v in ((self.check, st.check), (self.prune, st.prune), (self.run_cmd, st.run_command)):
            t.setChecked(v)
        probs = AU.settings_problems(p.analysis.automation)
        self.stored_problem.set('warning', 'In the project file: ' + '; '.join(probs) + '.' if probs else '')
        self.stored_problem.setVisible(bool(probs))
        self._syncing = False

    def _sync_command(self):
        p = self.ctx.store.project
        cfg = self.ctl.command()
        saved = bool(p.path)
        for w in (self.cmd, self.timeout, self.b_save_cmd):
            w.setEnabled(saved)
        if getattr(self, '_cmd_for', None) != (p.path, cfg.command if cfg else None):
            self._cmd_for = (p.path, cfg.command if cfg else None)
            self.cmd.setPlainText(cfg.command if cfg else '')
        self.timeout.set_value(cfg.timeout_min if cfg else AU.DEFAULT_TIMEOUT_MIN)
        approved = saved and self.ctl.commands.is_approved(p.path)
        self.b_approve.setVisible(cfg is not None and not approved)
        self.b_remove_cmd.setEnabled(cfg is not None)
        if not saved:
            self.cmd_state.set('warning', 'Save the project first: the command is kept for the project file.')
        elif cfg is None:
            self.cmd_state.set(None, 'No command: only the files are regenerated.')
        elif approved:
            when = time.strftime('%d %b %Y %H:%M', time.localtime(cfg.approved_at)) if cfg.approved_at else ''
            self.cmd_state.set('good', f'Approved for this project file{" on " + when if when else ""}.')
        else:
            self.cmd_state.set('warning', 'Not approved yet: it does not run until you approve it.')

    def refresh(self):
        self._dirty = False
        p = self.ctx.store.project
        st = self.settings
        self._sync()
        self._sync_command()
        busy = self.ctl.busy
        self.b_run.setEnabled(bool(p.path) and not busy)
        self.b_stop.setEnabled(busy)
        self.bar.setVisible(busy)
        if not busy:
            self.bar.setValue(0)
            self.run_text.setText('')
        cfg = self.ctl.command()
        approved = bool(p.path) and self.ctl.commands.is_approved(p.path)
        if not p.path:
            self.status.setText('Save the project to automate it: the folder and the command belong to the project '
                                'file.')
        else:
            when = AU.TRIGGER_LABEL[st.trigger] + (f' (quiet {st.quiet_s:g} s)' if st.trigger == 'change' else '') \
                if st.enabled else 'Off: runs only with Run now'
            what = f'{len(st.formats)} format{"s" if len(st.formats) != 1 else ""} into {st.out_dir}' + \
                (' · requirements check' if st.check else '')
            cmd = ('no command' if cfg is None else 'command switched off' if not st.run_command else
                   'command approved' if approved else 'command not approved')
            self.status.setText(f'{when} · {what} · {cmd}.')
        self._fill_history()
        rec = self._shown
        runs = self.ctl.runs()
        if rec is None or (self._follow and runs and runs[0].id != rec.id):
            rec = self._shown = runs[0] if runs else None
            self._show_output(rec)
        self._fill_run(rec)
        self.b_export.setEnabled(rec is not None and rec.done)

    def _fill_history(self):
        runs = self.ctl.runs()
        self._syncing = True
        t = self.history
        t.setRowCount(len(runs))
        right = Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter
        sel = None
        for i, r in enumerate(runs):
            sev = AU.STATUS_SEVERITY.get(r.status)
            items = [time.strftime('%d %b %H:%M:%S', time.localtime(r.started)),
                     AU.TRIGGER_LABEL.get(r.trigger, r.trigger), AU.STATUS_LABEL.get(r.status, r.status),
                     '—' if r.exit_code is None else str(r.exit_code),
                     f'{r.seconds:.1f} s' if r.done else '…',
                     str(r.n_changed()) if r.done else '…']
            for c, text in enumerate(items):
                it = QTableWidgetItem(text)
                if c == 0:
                    it.setData(Qt.ItemDataRole.UserRole, r.id)
                    it.setToolTip(r.message)
                if c == 2 and sev:
                    it.setForeground(QColor(STATUS[sev]))
                    it.setToolTip(r.message)
                if c in (3, 4, 5):
                    it.setFont(mono(12))
                    it.setTextAlignment(right)
                t.setItem(i, c, it)
            if self._shown is not None and r.id == self._shown.id:
                sel = i
        if sel is not None:
            t.selectRow(sel)
        else:
            t.clearSelection()
        self._syncing = False

    def _fill_run(self, rec):
        tl = self.tiles
        if rec is None:
            tl['result'].set('—', '', 'no run yet')
            tl['exit'].set('—', '', 'no run yet')
            tl['time'].set('—', '')
            tl['files'].set('—', '')
            tl['req'].set('—', '')
            self.out_title.setText('Output')
            self.files_title.setText('Files')
            self.files.setRowCount(0)
            self._fill_notes([])
            return
        when = time.strftime('%H:%M:%S', time.localtime(rec.started))
        tl['result'].set(AU.STATUS_LABEL.get(rec.status, rec.status), '',
                         f'{AU.TRIGGER_LABEL.get(rec.trigger, rec.trigger)} · {when}',
                         AU.STATUS_SEVERITY.get(rec.status))
        if rec.command_state == 'ran' and rec.exit_code is not None:
            tl['exit'].set(str(rec.exit_code), 'exit', 'your command', 'good' if rec.exit_code == 0 else 'serious')
        elif rec.command_state == 'ran':
            tl['exit'].set('—', '', 'running' if not rec.done else AU.STATUS_LABEL.get(rec.status, rec.status).lower(),
                           None if not rec.done else AU.STATUS_SEVERITY.get(rec.status))
        else:
            tl['exit'].set('—', '', AU.COMMAND_STATE_TEXT.get(rec.command_state, rec.command_state),
                           'warning' if rec.command_state == 'not_approved' else None)
        tl['time'].set(f'{rec.seconds:.1f}', 's', f'files {rec.export_s:.1f} s' +
                       (f' · command {rec.command_s:.1f} s' if rec.command_state == 'ran' else ''))
        c = rec.changed
        if rec.done and rec.status != 'error' or rec.written:
            parts = [f'{len(c[k])} {w}' for k, w in (('added', 'new'), ('removed', 'removed'),
                                                     ('rewritten', 'PDF rewritten')) if c.get(k)]
            tl['files'].set(str(rec.n_changed()), 'changed', ' · '.join(parts + [f'{len(rec.written)} written']))
        else:
            tl['files'].set('—', '', 'running' if not rec.done else 'not written')
        req = rec.requirements
        if req:
            tl['req'].set(req['verdict_text'], '', f'must {req["musts"]} · should {req["shoulds"]}',
                          VERDICT_SEV.get(req['verdict']))
        else:
            tl['req'].set('—', '', 'not run' if rec.done else 'running')
        self.out_title.setText('Output · ' + (rec.command.splitlines()[0][:80] if rec.command else
                                              AU.COMMAND_STATE_TEXT.get(rec.command_state, '')))
        # ---- the files
        rows = []
        for by, d in (('Vantage', rec.changed), ('Command', rec.command_changed)):
            for kind in ('added', 'changed', 'removed', 'rewritten'):
                rows += [(f, kind, by) for f in d.get(kind, ())]
        f = self.files
        f.setRowCount(len(rows))
        for i, (path, kind, by) in enumerate(rows):
            a = QTableWidgetItem(path)
            a.setFont(mono(11))
            a.setToolTip(os.path.join(rec.out_dir, path) if rec.out_dir else path)
            b = QTableWidgetItem(CHANGE_WORD[kind])
            b.setToolTip(CHANGE_TIP[kind])
            if kind in ('rewritten',):
                b.setForeground(THEME.color('muted'))
            f.setItem(i, 0, a)
            f.setItem(i, 1, b)
            f.setItem(i, 2, QTableWidgetItem(by))
        unchanged = c.get('unchanged', 0)
        folder = rec.out_dir or 'no folder'
        if rec.out_dir and rec.project_path:
            try:
                folder = os.path.relpath(rec.out_dir, os.path.dirname(rec.project_path))
            except ValueError:
                pass
        self.files_title.setText(f'Files · {folder}' + (
            f' · {len(rec.written)} written, {rec.n_changed()} changed, {unchanged} unchanged' if rec.written else ''))
        self.files_title.setToolTip(rec.out_dir)
        notes = []
        if rec.status in ('error', 'timeout', 'cancelled', 'failed'):
            notes.append((AU.STATUS_SEVERITY.get(rec.status) or 'warning', rec.message))
        for fid, msg in rec.format_errors.items():
            notes.append(('critical', f'{fid}: {msg}'))
        if rec.pruned:
            notes.append(('warning', f'Removed {len(rec.pruned)} file{"s" if len(rec.pruned) != 1 else ""} the '
                                     'previous run wrote and this one did not: ' + ', '.join(rec.pruned[:6])
                          + (' …' if len(rec.pruned) > 6 else '')))
        if not c.get('complete', True):
            notes.append(('warning', 'The folder holds more than 20 000 files or 4 GB: not every file was compared.'))
        notes += [('warning', n) for n in rec.notes if not n.startswith('Traceback')]
        if rec.unsaved:
            notes.append(('warning', 'Unsaved edits were exported (VANTAGE_UNSAVED=1); the project file on disk is '
                                     'older.'))
        self._fill_notes(notes)

    def _fill_notes(self, notes):
        for w in self._note_widgets:
            self.notes_lay.removeWidget(w)
            w.hide()
            w.deleteLater()
        self._note_widgets = []
        for sev, text in notes:
            w = StatusLine(sev, text)
            self.notes_lay.addWidget(w)
            self._note_widgets.append(w)

    # ------------------------------------------------------------------ export
    def export(self, folder=None):
        rec = self._shown
        if rec is None or not rec.done:
            return []
        if folder is None:
            folder = QFileDialog.getExistingDirectory(self, 'Export the run into')
            if not folder:
                return []
        from ...core import provenance as PV          # where the JSON came from
        paths = PV.stamp_paths(AU.export(rec, self.ctx.store.project, folder), self.ctx.store.project, folder)
        self.ctx.msg(f'Exported the run to {folder}', 5000)
        return paths

    def export_ci(self, folder=None):
        p = self.ctx.store.project
        if folder is None:
            start = os.path.dirname(p.path) if p.path else os.path.expanduser('~')
            folder = QFileDialog.getExistingDirectory(self, 'Write the CI templates into', start)
            if not folder:
                return []
        try:
            paths = AU.write_ci(p, folder)
        except (ValueError, OSError) as e:
            self.ctx.msg(f'Could not write the CI templates: {e}', 6000)
            return []
        self.ctx.msg(f'Wrote the CI templates to {os.path.join(folder, "ci")} (see README.md there)', 6000)
        return paths
