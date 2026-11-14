"""File ▸ Lab: the team's lab server (vantage.lab). Connect with the server's address and a token (kept in the system
keyring or a private file, never in the project), pick or register the project on the lab, see exactly which files a
publish sends, publish a version, sync (publish, pull, or merge both ways), browse the version history with the
semantic diff of each version against the one before (or against this copy), open a version, share the fleet and the
perception records with other teams, and run the pooled fleet twin of every fleet the team may see. Every network call
runs in the background; nothing here changes the project except opening a version or a sync that pulled or merged
(the project is then replaced, and saved to its file or folder when it has one)."""
from __future__ import annotations

import os
import threading
import traceback

from PyQt6.QtCore import QObject, Qt, QTimer, pyqtSignal
from PyQt6.QtWidgets import (QAbstractItemView, QComboBox, QFileDialog, QHBoxLayout, QHeaderView, QLineEdit,
                             QPlainTextEdit, QScrollArea, QSplitter, QTableWidget, QTableWidgetItem, QVBoxLayout,
                             QWidget)

from ...core import fleet_twin as FT
from ...core import project_dir as PD
from ...lab import client as LC
from ..theme import STATUS, mono
from ..widgets import ROW_H, StatusLine, ToggleSwitch, button, hline, label

URG_TONE = {'beyond': 'critical', 'due': 'serious', 'soon': 'warning', 'later': 'good', 'ok': 'good',
            'unknown': None}


class _Sig(QObject):
    done = pyqtSignal(int, str, object, object)          # generation, kind, result, error text


def _table(cols, select=True):
    t = QTableWidget(0, len(cols))
    t.setHorizontalHeaderLabels(cols)
    t.verticalHeader().setVisible(False)
    t.verticalHeader().setDefaultSectionSize(ROW_H)
    t.horizontalHeader().setFixedHeight(ROW_H)
    t.setShowGrid(False)
    t.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
    t.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
    t.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection if select else
                       QAbstractItemView.SelectionMode.NoSelection)
    hh = t.horizontalHeader()
    for c in range(len(cols)):
        hh.setSectionResizeMode(c, QHeaderView.ResizeMode.ResizeToContents)
    hh.setSectionResizeMode(len(cols) - 1, QHeaderView.ResizeMode.Stretch)
    return t


def _it(text, num=False, tip=None, color=None, data=None):
    it = QTableWidgetItem(str(text))
    it.setFlags(Qt.ItemFlag.ItemIsSelectable | Qt.ItemFlag.ItemIsEnabled)
    if num:
        it.setFont(mono(12))
        it.setTextAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
    if tip:
        it.setToolTip(tip)
    if color is not None:
        from PyQt6.QtGui import QColor
        it.setForeground(QColor(color))
    if data is not None:
        it.setData(Qt.ItemDataRole.UserRole, data)
    return it


def _size(n: int) -> str:
    for unit in ('B', 'kB', 'MB', 'GB'):
        if n < 1024 or unit == 'GB':
            return f'{n:.0f} {unit}' if unit == 'B' else f'{n:.1f} {unit}'
        n /= 1024.0
    return str(n)


class LabPage(QWidget):
    def __init__(self, ctx):
        super().__init__()
        self.ctx = ctx
        self.who = None                       # whoami of the connected token
        self.pid = ''                         # the lab project this page works on
        self.pinfo = None                     # its document (sharing, head)
        self.versions = []
        self.pool_result = None
        self.pool_doc = None
        self.busy = False
        self.status_text = ''
        self._gen = 0
        self._files_dirty = True
        self.sig = _Sig()
        self.sig.done.connect(self._done)
        self._refresh_timer = QTimer(self)
        self._refresh_timer.setSingleShot(True)
        self._refresh_timer.setInterval(400)
        self._refresh_timer.timeout.connect(self._refresh_files)
        self.setObjectName('base')
        lay = QHBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        split = QSplitter(Qt.Orientation.Horizontal)
        split.setHandleWidth(2)
        split.addWidget(self._left())
        split.addWidget(self._right())
        split.setSizes([400, 1200])
        lay.addWidget(split)
        ctx.store.changed.connect(self._on_change)
        ctx.store.project_replaced.connect(self._on_replaced)
        self.url.setText(str(ctx.settings.value('lab/url', '') or ''))
        self._token_hint()
        self._sync_buttons()

    # ------------------------------------------------------------------ layout
    def _row(self, *ws, stretch_last=True):
        h = QHBoxLayout()
        h.setSpacing(4)
        for i, w in enumerate(ws):
            h.addWidget(w, 1 if (stretch_last and i == len(ws) - 1) else 0)
        return h

    def _left(self):
        outer = QScrollArea()
        outer.setWidgetResizable(True)
        w = QWidget()
        w.setObjectName('panel')
        ll = QVBoxLayout(w)
        ll.setContentsMargins(8, 6, 8, 8)
        ll.setSpacing(4)
        top = QHBoxLayout()
        top.addWidget(label('Lab'))
        top.addStretch(1)
        top.addWidget(label('versions, shared fleets, pooled records', muted=True, small=True))
        ll.addLayout(top)
        ll.addWidget(hline())
        ll.addWidget(label('Publish this project to your team\'s lab server as immutable versions, compare them, sync '
                           'with your colleagues, and pool fleets and perception records across teams. A publish '
                           'sends exactly the files listed below; stream passwords, API keys and the lab token are '
                           'never in them.', muted=True, small=True, wrap=True))
        ll.addSpacing(4)
        ll.addWidget(label('Server', bold=True))
        self.url = QLineEdit()
        self.url.setPlaceholderText('https://lab.example.org or http://127.0.0.1:8765')
        self.url.setToolTip('The lab server\'s address (behind HTTPS when it is not on this computer)')
        self.url.editingFinished.connect(self._url_edited)
        ll.addLayout(self._row(self._lbl('Address'), self.url))
        self.token = QLineEdit()
        self.token.setEchoMode(QLineEdit.EchoMode.Password)
        self.token.setToolTip('The token your lab admin gave you. Saved in the system keyring (or a private file in '
                              'your settings folder), never in the project')
        self.b_save_token = button('Save', None, 'key', 'Keep the token for this server')
        self.b_save_token.clicked.connect(self._save_token)
        self.b_forget = button('Forget', None, None, 'Remove the saved token for this server')
        self.b_forget.clicked.connect(self._forget_token)
        ll.addLayout(self._row(self._lbl('Token'), self.token, self.b_save_token, self.b_forget, stretch_last=False))
        self.b_connect = button('Connect', 'primary', 'refresh', 'Check the token and list the lab\'s projects')
        self.b_connect.clicked.connect(self.connect_lab)
        ll.addWidget(self.b_connect)
        self.status = StatusLine(None, 'Not connected.')
        ll.addWidget(self.status)
        ll.addSpacing(6)
        ll.addWidget(label('Project on the lab', bold=True))
        self.project_box = QComboBox()
        self.project_box.setToolTip('The lab project this copy publishes to and syncs with')
        self.project_box.activated.connect(self._project_picked)
        ll.addWidget(self.project_box)
        self.new_name = QLineEdit()
        self.new_name.setPlaceholderText('name for a new lab project')
        self.b_register = button('Register', None, 'plus', 'Register a new project in your team')
        self.b_register.clicked.connect(self.register)
        ll.addLayout(self._row(self.new_name, self.b_register, stretch_last=False))
        self.share_fleet = ToggleSwitch('Share the fleet with other teams')
        self.share_fleet.setToolTip('Teams that can read this project pool its vehicles\' calibration records into '
                                    'their fleet twin')
        self.share_fleet.toggled.connect(lambda on: self._patch(share_fleet=on))
        ll.addWidget(self.share_fleet)
        self.share_records = ToggleSwitch('Share perception records with the lab')
        self.share_records.setToolTip('Opt in: this project\'s perception records (anonymised) train every team\'s '
                                      'learned rig model')
        self.share_records.toggled.connect(lambda on: self._patch(share_records=on))
        ll.addWidget(self.share_records)
        self.readers = QLineEdit()
        self.readers.setPlaceholderText('teams that may read it (comma-separated; * for every team)')
        self.readers.editingFinished.connect(self._readers_edited)
        ll.addLayout(self._row(self._lbl('Readers'), self.readers))
        ll.addSpacing(6)
        ll.addWidget(label('Publish', bold=True))
        self.message = QLineEdit()
        self.message.setPlaceholderText('what changed')
        ll.addWidget(self.message)
        self.b_publish = button('Publish', 'primary', 'export', 'Publish this copy as a new version')
        self.b_publish.clicked.connect(self.publish)
        self.b_sync = button('Sync', None, 'sync', 'Publish your changes, take your colleagues\', or merge both')
        self.b_sync.clicked.connect(self.sync)
        ll.addLayout(self._row(self.b_publish, self.b_sync, stretch_last=False))
        self.files_label = label('', muted=True, small=True, wrap=True)
        ll.addWidget(self.files_label)
        self.files_table = _table(['File', 'Size', 'SHA-256'], select=False)
        self.files_table.setMinimumHeight(160)
        ll.addWidget(self.files_table)
        ll.addSpacing(6)
        ll.addWidget(label('Perception records', bold=True))
        self.b_push = button('Send records', None, 'export', 'Send this project\'s perception records (anonymised)')
        self.b_push.clicked.connect(self.push_records)
        self.b_pull = button('Fetch records', None, 'import', 'Add every record of the lab to your local store')
        self.b_pull.clicked.connect(self.pull_records)
        ll.addLayout(self._row(self.b_push, self.b_pull, stretch_last=False))
        self.records_label = label('', muted=True, small=True, wrap=True)
        ll.addWidget(self.records_label)
        ll.addStretch(1)
        outer.setWidget(w)
        return outer

    def _lbl(self, text):
        a = label(text, ink2=True)
        a.setFixedWidth(62)
        return a

    def _right(self):
        sp = QSplitter(Qt.Orientation.Vertical)
        sp.setHandleWidth(2)
        top = QWidget()
        tl = QVBoxLayout(top)
        tl.setContentsMargins(8, 6, 8, 4)
        h = QHBoxLayout()
        h.addWidget(label('Versions', bold=True))
        h.addStretch(1)
        self.b_compare_local = button('Compare with my copy', None, 'compare',
                                      'The selected version against the project as it is now')
        self.b_compare_local.clicked.connect(self.compare_local)
        self.b_open = button('Open this version', None, 'folder', 'Replace the project in the window with it')
        self.b_open.clicked.connect(self.open_version)
        self.b_save_as = button('Save as folder…', None, 'export', 'Save the selected version as a project folder')
        self.b_save_as.clicked.connect(self.save_version_as)
        for b in (self.b_compare_local, self.b_open, self.b_save_as):
            h.addWidget(b)
        tl.addLayout(h)
        self.history = _table(['Version', 'Published', 'By', 'Message', 'Size', 'Digest'])
        self.history.itemSelectionChanged.connect(self._version_picked)
        tl.addWidget(self.history, 1)
        sp.addWidget(top)
        mid = QWidget()
        ml = QVBoxLayout(mid)
        ml.setContentsMargins(8, 4, 8, 4)
        self.diff_title = label('Changes', bold=True)
        ml.addWidget(self.diff_title)
        self.diff_text = QPlainTextEdit()
        self.diff_text.setReadOnly(True)
        self.diff_text.setFont(mono(12))
        self.diff_text.setPlaceholderText('Pick a version to see what it changed.')
        ml.addWidget(self.diff_text, 1)
        sp.addWidget(mid)
        bot = QWidget()
        bl = QVBoxLayout(bot)
        bl.setContentsMargins(8, 4, 8, 8)
        h = QHBoxLayout()
        h.addWidget(label('Pooled fleet twin', bold=True))
        h.addStretch(1)
        self.b_pool = button('Run pooled twin', None, 'fleet', 'The fleet twin of every fleet your team may pool, '
                                                               'on this project\'s plan')
        self.b_pool.clicked.connect(self.run_pool)
        h.addWidget(self.b_pool)
        bl.addLayout(h)
        self.pool_label = label('', muted=True, small=True, wrap=True)
        bl.addWidget(self.pool_label)
        self.pool_table = _table(['Vehicle', 'Sensor', 'Urgency', 'P now', 'Recalibrate by', 'Limit'], select=False)
        bl.addWidget(self.pool_table, 1)
        sp.addWidget(bot)
        sp.setSizes([320, 280, 300])
        return sp

    # ------------------------------------------------------------------ state
    def activate(self):
        if self._files_dirty:
            self._refresh_files()

    def _on_change(self, *_):
        self._files_dirty = True
        if self.isVisible():
            self._refresh_timer.start()

    def _on_replaced(self, *_):
        self._files_dirty = True
        st = LC.sync_state(self.ctx.store.project)
        if st and st.get('url') == LC.norm_url(self.url.text()) and self.who is not None:
            self._select_pid(st['project'])
        if self.isVisible():
            self._refresh_timer.start()

    def _sync_buttons(self):
        conn = self.who is not None
        member = conn and self.who.get('role') in ('member', 'admin')
        has = bool(self.pid)
        mine = has and self.pinfo is not None and (self.pinfo.get('team') == (self.who or {}).get('team')
                                                   or (self.who or {}).get('role') == 'admin')
        for b, on in ((self.b_register, member), (self.b_publish, member and mine), (self.b_sync, member and mine),
                      (self.b_push, member and mine), (self.b_pull, conn), (self.b_pool, conn),
                      (self.b_open, has), (self.b_save_as, has), (self.b_compare_local, has)):
            b.setEnabled(bool(on) and not self.busy)
        for w in (self.share_fleet, self.share_records, self.readers):
            w.setEnabled(bool(member and mine) and not self.busy)
        self.b_connect.setEnabled(not self.busy)

    def _say(self, sev, text):
        self.status_text = text
        self.status.set(sev, text)
        self.ctx.msg(text, 5000)

    def _client(self):
        return LC.LabClient(self.url.text())

    def _run(self, kind, fn):
        """fn() on a worker thread; its result (or error) back to _done on the UI thread."""
        self.busy = True
        self._gen += 1
        gen = self._gen
        self._sync_buttons()

        def work():
            try:
                res, err = fn(), None
            except LC.LabClientError as e:
                res, err = None, e.message + (f' (HTTP {e.status})' if e.status else '')
                if e.status == 409 and 'head' in e.extra:
                    err += ': Sync merges the newer version with yours'
            except Exception as e:                       # noqa: BLE001  shown, never raised into Qt
                traceback.print_exc()
                res, err = None, f'{type(e).__name__}: {e}'
            self.sig.done.emit(gen, kind, res, err)
        threading.Thread(target=work, daemon=True).start()

    def _done(self, gen, kind, res, err):
        if gen != self._gen:
            return
        self.busy = False
        if err:
            self._say('serious', f'{kind.capitalize()}: {err}')
        else:
            getattr(self, f'_got_{kind}')(res)
        self._sync_buttons()

    # ------------------------------------------------------------------ server and token
    def _url_edited(self):
        self.ctx.settings.setValue('lab/url', LC.norm_url(self.url.text()))
        self._token_hint()

    def _token_hint(self):
        u = LC.norm_url(self.url.text())
        has = bool(u) and LC.load_token(u) is not None
        self.token.setPlaceholderText('saved (keyring or private file)' if has else 'paste your token')

    def _save_token(self):
        u, t = LC.norm_url(self.url.text()), self.token.text().strip()
        if not u or not t:
            self._say('warning', 'Give the server address and the token first.')
            return
        where = LC.save_token(u, t)
        self.ctx.settings.setValue('lab/url', u)
        self.token.clear()
        self._token_hint()
        self._say('good', f'Token saved in {"the system keyring" if where == "keyring" else "a private file"}.')

    def _forget_token(self):
        LC.forget_token(self.url.text())
        self._token_hint()
        self._say(None, 'Token forgotten.')

    def connect_lab(self):
        if not LC.norm_url(self.url.text()):
            self._say('warning', 'Give the lab server\'s address.')
            return
        if self.token.text().strip():
            self._save_token()
        self.ctx.settings.setValue('lab/url', LC.norm_url(self.url.text()))
        c = self._client()
        self._run('connect', lambda: (c.whoami(), c.projects()))

    def _got_connect(self, res):
        self.who, projects = res
        self._fill_projects(projects)
        st = LC.sync_state(self.ctx.store.project)
        if st and st.get('url') == LC.norm_url(self.url.text()):
            self._select_pid(st['project'])
        self._say('good', f'Connected as {self.who.get("name") or "token " + str(self.who.get("token"))} of team '
                          f'{self.who["team"]} ({self.who["role"]}); {len(projects)} project(s) you can read.')

    def _fill_projects(self, projects, keep=None):
        keep = keep or self.pid
        self.project_box.blockSignals(True)
        self.project_box.clear()
        self.project_box.addItem('— none —', '')
        for p in projects:
            n = p['head']['number'] if p.get('head') else 0
            self.project_box.addItem(f'{p["id"]} · {n} version(s)', p['id'])
        self.project_box.blockSignals(False)
        self._projects = {p['id']: p for p in projects}
        if keep:
            self._select_pid(keep)

    def _select_pid(self, pid):
        i = self.project_box.findData(pid)
        if i < 0:
            return
        self.project_box.setCurrentIndex(i)
        self._project_picked(i)

    def _project_picked(self, i):
        self.pid = self.project_box.itemData(i) or ''
        self.pinfo = getattr(self, '_projects', {}).get(self.pid)
        self._show_sharing()
        self.versions = []
        self.history.setRowCount(0)
        self.diff_text.clear()
        if self.pid:
            c, pid = self._client(), self.pid
            self._run('history', lambda: (c.project(pid), c.versions(pid)))
        self._sync_buttons()

    def _show_sharing(self):
        p = self.pinfo or {}
        for w, v in ((self.share_fleet, bool(p.get('share_fleet'))), (self.share_records, bool(p.get('share_records')))):
            w.blockSignals(True)
            w.setChecked(v)
            w.blockSignals(False)
        self.readers.setText(', '.join(p.get('readers') or []))

    def register(self):
        name = self.new_name.text().strip() or self.ctx.store.project.name
        c = self._client()
        self._run('register', lambda: (c.register(name), c.projects()))

    def _got_register(self, res):
        doc, projects = res
        self.new_name.clear()
        self._fill_projects(projects, keep=doc['id'])
        self._say('good', f'Registered {doc["id"]}. Publish sends the files listed on the left.')

    def _patch(self, **kw):
        if not self.pid:
            return
        c, pid = self._client(), self.pid
        self._run('patch', lambda: c.patch(pid, **kw))

    def _readers_edited(self):
        if self.pid and self.pinfo is not None:
            want = [x.strip() for x in self.readers.text().split(',') if x.strip()]
            if sorted(want) != sorted(self.pinfo.get('readers') or []):
                self._patch(readers=want)

    def _got_patch(self, doc):
        self.pinfo = doc
        if hasattr(self, '_projects'):
            self._projects[doc['id']] = doc
        self._show_sharing()
        self._say('good', f'{doc["id"]}: fleet {"shared" if doc["share_fleet"] else "not shared"}, records '
                          f'{"shared" if doc["share_records"] else "not shared"}, readers '
                          f'{", ".join(doc["readers"]) or "your team only"}.')

    # ------------------------------------------------------------------ what is sent, publish, sync
    def _refresh_files(self):
        self._files_dirty = False
        try:
            files = LC.files_to_publish(self.ctx.store.project)
        except Exception as e:                            # noqa: BLE001
            self.files_label.setText(f'The project cannot be written as files: {e}')
            return
        import hashlib
        self.files_table.setRowCount(0)
        for path in sorted(files):
            r = self.files_table.rowCount()
            self.files_table.insertRow(r)
            b = files[path]
            self.files_table.setItem(r, 0, _it(path))
            self.files_table.setItem(r, 1, _it(_size(len(b)), num=True))
            self.files_table.setItem(r, 2, _it(hashlib.sha256(b).hexdigest()[:12], num=True))
        total = sum(len(b) for b in files.values())
        self.files_label.setText(f'A publish sends these {len(files)} files ({_size(total)}), the project folder\'s '
                                 'files, and nothing else. Files the lab already has are not sent again.')

    def publish(self):
        if not self.pid:
            self._say('warning', 'Pick or register the lab project first.')
            return
        p = self.ctx.store.project.snapshot()
        p.thumb_png = self.ctx.store.project.thumb_png
        st = LC.sync_state(p)
        parent = st['version'] if st and st.get('project') == self.pid and st.get('url') == LC.norm_url(
            self.url.text()) else 'auto'
        msg = self.message.text().strip()
        c, pid = self._client(), self.pid
        self._run('publish', lambda: (c.publish(p, pid, msg, parent=parent), c.versions(pid)))

    def _got_publish(self, res):
        doc, versions = res
        self.message.clear()
        self._fill_history(versions)
        if doc.get('existing'):
            self._say('good', f'The lab already has these files as version {doc["number"]}: nothing new published.')
        else:
            up = doc.get('upload', {})
            self._say('good', f'Published version {doc["number"]} of {doc["project"]} ({up.get("uploaded", 0)} '
                              f'file(s), {_size(up.get("bytes", 0))} sent).')

    def sync(self):
        if not self.pid:
            self._say('warning', 'Pick or register the lab project first.')
            return
        p = self.ctx.store.project.snapshot()
        p.thumb_png = self.ctx.store.project.thumb_png
        msg = self.message.text().strip()
        c, pid = self._client(), self.pid
        self._run('sync', lambda: (c.sync(p, pid, msg), c.versions(pid)))

    def _got_sync(self, res):
        r, versions = res
        self._fill_history(versions)
        s = r['status']
        if s in ('pulled', 'merged'):
            q = r['project']
            q.path = self.ctx.store.project.path
            if q.path:
                q.save(q.path)
            self.ctx.store.set_project(q)
        if s == 'conflicts':
            self.diff_title.setText('Conflicts: nothing was published')
            self.diff_text.setPlainText('\n'.join(
                f'{c["file"]}: {c["path"]}: yours {c.get("ours")!r}, the lab\'s {c.get("theirs")!r}'
                for c in r['conflicts']) + '\n\nChange one side so they agree, then sync again.')
            self._say('serious', f'{len(r["conflicts"])} conflict(s) with version {r["version"]}: nothing published.')
        else:
            self._say('good', {'up to date': f'Up to date with version {r["version"]}.',
                               'published': f'Published version {r["version"]}.',
                               'pulled': f'Took version {r["version"]} from the lab.',
                               'merged': f'Merged with the lab and published version {r["version"]}.'}[s])
        self.message.clear()

    # ------------------------------------------------------------------ history and diffs
    def _got_history(self, res):
        doc, versions = res
        self.pinfo = doc
        self._show_sharing()
        self._fill_history(versions)

    def _fill_history(self, versions):
        self.versions = list(reversed(versions))          # newest first
        self.history.blockSignals(True)
        self.history.setRowCount(0)
        for v in self.versions:
            r = self.history.rowCount()
            self.history.insertRow(r)
            self.history.setItem(r, 0, _it(v['number'], num=True, data=v['number']))
            self.history.setItem(r, 1, _it(str(v['created']).replace('T', ' ').replace('Z', ' UTC')))
            self.history.setItem(r, 2, _it(v['author']))
            self.history.setItem(r, 3, _it(v['message'], tip=v['message']))
            self.history.setItem(r, 4, _it(_size(int(v.get('bytes') or 0)), num=True))
            self.history.setItem(r, 5, _it(v['digest'][:12], num=True, tip=v['digest']))
        self.history.blockSignals(False)

    def _selected_version(self):
        rows = self.history.selectionModel().selectedRows() if self.history.selectionModel() else []
        if not rows:
            return None
        return self.versions[rows[0].row()] if rows[0].row() < len(self.versions) else None

    def _version_picked(self):
        v = self._selected_version()
        if v is None or not self.pid:
            return
        if v['number'] == 1:
            self.diff_title.setText(f'Version 1 · {v["message"] or "the first version"}')
            self.diff_text.setPlainText('The first version: nothing before it to compare with.')
            return
        c, pid, n = self._client(), self.pid, v['number']
        self.diff_title.setText(f'Version {n} against version {n - 1}')
        self._run('compare', lambda: c.compare(pid, n - 1, n))

    def _got_compare(self, res):
        self.diff_text.setPlainText(res['text'])

    def compare_local(self):
        v = self._selected_version()
        if v is None:
            return
        c, pid, n = self._client(), self.pid, v['number']
        mine = self.ctx.store.project.snapshot()

        def work():
            q = c.pull(pid, n)
            return PD.diff_text(PD.diff(q, mine, f'{pid}@{n}', 'this copy'))
        self.diff_title.setText(f'This copy against version {n}')
        self._run('compare_local', work)

    def _got_compare_local(self, text):
        self.diff_text.setPlainText(text)

    def open_version(self):
        v = self._selected_version()
        if v is None:
            return
        c, pid, n = self._client(), self.pid, v['number']
        self._run('open', lambda: c.pull(pid, n))

    def _got_open(self, p):
        w = self.ctx.window
        if w is not None and hasattr(w, '_maybe_save') and not w._maybe_save():
            return
        p.path = ''
        p.name = PD.name_from_path(self.pid.split('/')[-1]) + f'-v{p.lab_version["number"]}'
        self.ctx.store.set_project(p)
        self._say('good', f'Opened version {p.lab_version["number"]} of {self.pid} (not saved yet: File ▸ Save '
                          'as… keeps it).')

    def save_version_as(self):
        v = self._selected_version()
        if v is None:
            return
        start = os.path.join(os.path.expanduser('~'), f'{self.pid.split("/")[-1]}-v{v["number"]}{PD.DIR_SUFFIX}')
        path, _ = QFileDialog.getSaveFileName(self, 'Save the version as a project folder', start, PD.DIALOG_TEXT)
        if not path:
            return
        path = PD.default_dir_for(path)
        c, pid, n = self._client(), self.pid, v['number']
        self._run('save_as', lambda: (c.pull(pid, n, dest=path), path))

    def _got_save_as(self, res):
        self._say('good', f'Saved version {res[0].lab_version["number"]} as {res[1]}.')

    # ------------------------------------------------------------------ pooled fleet twin, records
    def run_pool(self):
        base = self.ctx.store.project.snapshot()
        st = FT.settings_of(base)
        c = self._client()
        self.pool_label.setText('Running the pooled fleet twin…')
        self._run('pool', lambda: c.pooled_twin(base, settings=st))

    def _got_pool(self, res):
        self.pool_result, self.pool_doc = res
        r = self.pool_result
        srcs = ', '.join(f'{s["label"]} ({s["vehicles"]} vehicles)' for s in r.metrics.get('pooled', []))
        skipped = '; '.join(f'{s["project"]}: {s["why"]}' for s in self.pool_doc.get('skipped', []))
        self.pool_label.setText(f'{r.metrics["vehicles"]} vehicles from {len(r.metrics.get("pooled", []))} '
                                f'project(s): {srcs or "none"}.' + (f' Not pooled: {skipped}.' if skipped else '')
                                + ' Each vehicle\'s measurements are re-based onto this project\'s plan, sensor by '
                                  'tag.')
        self.pool_table.setRowCount(0)
        cells = sorted(r.cells.values(), key=lambda c: (FT.URGENCY.index(c['urgency']), c['name'], c['tag']))
        for c in cells:
            row = self.pool_table.rowCount()
            self.pool_table.insertRow(row)
            tone = URG_TONE.get(c['urgency'])
            self.pool_table.setItem(row, 0, _it(c['name']))
            self.pool_table.setItem(row, 1, _it(c['tag']))
            self.pool_table.setItem(row, 2, _it(FT.URGENCY_LABEL[c['urgency']],
                                                color=STATUS.get(tone) if tone else None))
            self.pool_table.setItem(row, 3, _it(f'{c["p_now"]:.3f}', num=True))
            self.pool_table.setItem(row, 4, _it(FT.date_text(c['date_risk']) or '—'))
            self.pool_table.setItem(row, 5, _it(', '.join(sorted(set(c['limit_source'].values())))))
        self._say('good', f'Pooled fleet twin: {r.metrics["vehicles"]} vehicles, {len(cells)} sensors followed.')

    def push_records(self):
        if not self.pid:
            return
        p = self.ctx.store.project.snapshot()
        c, pid = self._client(), self.pid

        def work():
            docs = LC.project_records(p)
            if not docs:
                return {'added': 0, 'duplicates': 0, 'rejected': 0, 'errors': [], 'none': True}
            return c.push_records(pid, docs)
        self._run('push', work)

    def _got_push(self, r):
        if r.get('none'):
            self.records_label.setText('This project has no perception records in your store yet.')
            return
        self.records_label.setText(f'Sent: {r["added"]} new, {r["duplicates"]} the lab had, {r["rejected"]} refused.')
        self._say('good', f'{r["added"]} perception record(s) sent to the lab (anonymised).')

    def pull_records(self):
        c = self._client()
        from ...core import perf_model as PM
        store = PM.store_for(self.ctx.store.project)
        self._run('pull', lambda: c.pull_records(store))

    def _got_pull(self, r):
        self.records_label.setText(f'Fetched {r["received"]} record(s) from the lab: {r["added"]} new, '
                                   f'{r["duplicates"]} you had, {r["rejected"]} refused.')
        self._say('good', f'{r["added"]} new perception record(s) from the lab.')
