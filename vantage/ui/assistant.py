"""The Assistant dock: a chat with a Groq-hosted model about the open design.

The model reads the design only through vantage.assistant.tools (text summaries; never the vehicle model, images or
files) and can only *propose* camera changes: each proposal shows the change and its predicted effect, and the user
applies it as one undo step or discards it. The API key is set in Preferences ▸ Assistant."""
from __future__ import annotations

import html
import json
import math
import re
import threading
import time

from PyQt6.QtCore import QObject, QPointF, QRectF, QSize, Qt, QTimer, QUrl, pyqtSignal
from PyQt6.QtGui import QColor, QDesktopServices, QFont, QFontMetricsF, QPainter, QPen, QTextOption
from PyQt6.QtWidgets import (QApplication, QCheckBox, QComboBox, QFileDialog, QFrame, QGridLayout, QHBoxLayout,
                             QLabel, QLayout, QLineEdit, QMenu, QPlainTextEdit, QPushButton, QScrollArea,
                             QSizePolicy, QTextBrowser, QToolButton, QVBoxLayout, QWidget)

from .. import workdir
from ..assistant import agent as A
from ..assistant import client as C
from ..assistant import secrets as K
from ..assistant import tools as T
from . import icons
from .theme import THEME, mono, sans
from .widgets import SegmentedControl, button, label, tool_button

SUGGESTIONS = [
    ('Summarise this design', 'Summarise this design: coverage, weak spots and what to look at first.'),
    ('Why is any ground blind?', 'Where is ground blind or below the minimum density, and why? Name the cameras.'),
    ('Which camera matters most?', 'Which camera matters most, and what happens if it fails or is mounted badly?'),
    ('Where can a child go unseen?', 'Where could a 0.9 m child stand next to the car without any camera seeing '
                                     'them, and how could the rig be improved?'),
    ('Explain the issues', 'Explain each item in the Issues list and suggest a fix for each.'),
    ('Write a design review', 'Write a short design review of this camera rig for a colleague: strengths, risks, '
                              'and recommended changes with their measured effect.'),
]
AGENT_GOALS = [
    ('Close every blind patch', 'Get the blind ground to zero without lowering the minimum density anywhere. Keep '
                                'left/right pairs symmetric.'),
    ('Robust to mounting errors', 'Make the rig robust to the mounting tolerance: find which camera\'s error hurts '
                                  'most and reduce that worst case.'),
    ('Fewer vanishing people', 'Reduce where a 0.9 m child vanishes at the stitch seams, without creating blind ground.'),
    ('Pick the best variant', 'Compare the variants, pick the best one for coverage and robustness, and say what would '
                              'make it better still.'),
    ('Check under load', 'Check the rig under the load cases and propose changes if any case is serious.'),
]
SLASH = [  # command, what it does, completion
    ('/goal', 'Work towards a goal on its own (agent mode for this message)', '/goal '),
    ('/optimise', "Search a camera's pose, e.g. /optimise QR", '/optimise '),
    ('/compare', 'Compare all variants and recommend one', '/compare'),
    ('/study', 'Run a study: blind_volume, stitched_objects, load_cases, mounting_tolerance', '/study '),
    ('/show', 'Show something in the app, e.g. /show the blind spots', '/show '),
    ('/review', 'Write a design review', '/review'),
    ('/issues', 'Explain the issues and suggest fixes', '/issues'),
    ('/brief', 'Edit the project brief', '/brief'),
    ('/new', 'Start a new conversation', '/new'),
]


def expand_slash(text: str):
    """(prompt to send, agent mode or None, local action or None) for a message that starts with a command."""
    t = text.strip()
    if not t.startswith('/'):
        return t, None, None
    cmd, _, rest = t.partition(' ')
    rest = rest.strip()
    cmd = cmd.lower()
    if cmd in ('/new', '/clear'):
        return '', None, 'new'
    if cmd == '/brief':
        return '', None, 'brief'
    if cmd == '/goal':
        return (rest, True, None) if rest else ('', None, 'agent')
    if cmd in ('/optimise', '/optimize'):
        tag = rest.split()[0] if rest else ''
        what = f'camera {tag}' if tag else 'the camera that limits the design most'
        return (f'Optimise {what}: use optimise_camera for the least blind ground while keeping the minimum density, '
                f'check the result, and propose it only if the gain is worth it.' + (f' Notes: {rest}' if len(rest.split()) > 1
                                                                                     else '')), False, None
    if cmd == '/compare':
        return ('Compare all variants of this project with compare_variants and recommend one, with the evidence.'
                + (f' {rest}' if rest else '')), False, None
    if cmd == '/study':
        return (f'Run the {rest or "blind_volume"} study with run_study and explain what it shows for this design.',
                False, None)
    if cmd == '/show':
        return f'Show me in Vantage with show_in_app: {rest or "the weakest spot of this design"}.', False, None
    if cmd == '/review':
        return SUGGESTIONS[5][1], False, None
    if cmd == '/issues':
        return SUGGESTIONS[4][1], False, None
    return t, None, None


PAGES = {'viewports': '3D and top views', 'camviews': 'Camera views', 'tolerance': 'Tolerance study',
         'optimiser': 'Optimiser', 'compare': 'Compare variants', 'photo': 'Photo matching', 'targets': 'Targets',
         'install': 'Installation check', 'volume': 'Blind volume', 'perception': 'Perception range',
         'rig': 'Live rig', 'coverage': 'Lidar and radar coverage', 'timing': 'Sync and timing',
         'recording': 'Recording budget', 'power': 'Power budget',
         'gnss': 'GNSS and IMU', 'exposure': 'Sun glare and dirt',
         'fleet': 'Fleet', 'brackets': 'Mount brackets',
         'dataset': 'Sample dataset', 'drivelog': 'Drive-log check',
         'requirements': 'Requirements scorecard', 'lidar_calib': 'Lidar–camera calibration',
         'asbuilt': 'As-built check', 'critical': 'Critical targets', 'materials': 'Materials, glass and wipers',
         'automation': 'Automation', 'terrain': 'Road and body motion', 'harness': 'Cable harness',
         'observability': 'Calibration observability', 'experiments': 'Experiments', 'sim': 'CARLA simulator'}


# ============================================================================================ controller
class AssistantController(QObject):
    """Settings, the conversation and its worker thread, and the proposals (applied on the UI thread)."""
    event = pyqtSignal(object)          # agent.Event, on the UI thread, for the current conversation only
    changed = pyqtSignal()              # set-up, model or busy state
    reset = pyqtSignal()                # a new conversation started
    proposal_state = pyqtSignal(int)    # a proposal was applied, discarded, undone or measured
    agent_changed = pyqtSignal(bool)
    _raw = pyqtSignal(object)           # (thread id, Event) from the worker
    _ui_req = pyqtSignal(object)        # (thread id, request) from a show_in_app call

    def __init__(self, ctx):
        super().__init__()
        self.ctx = ctx
        self.s = ctx.settings
        self.key, self.key_source = K.load()
        self.session: A.Session | None = None
        self.busy = False
        self.status = ''
        self.proposals: dict = {}       # id -> {'p': Proposal, 'state': str, 'undo_index': int}
        self.last_question = ''
        self._tid = 0
        self._cancel = None
        self._mark = 0
        self._started = 0.0
        self.agent_mode = False
        self.agent_running = False
        self._raw.connect(self._on_raw)
        self._ui_req.connect(self._do_ui)
        ctx.store.project_replaced.connect(self.new_chat)

    # ---------------------------------------------------------------- settings
    @property
    def enabled(self) -> bool:
        return self.s.value('assistant/enabled', False, type=bool)

    @property
    def ready(self) -> bool:
        return self.enabled and bool(self.key)

    @property
    def model(self) -> str:
        return str(self.s.value('assistant/model', C.DEFAULT_MODEL) or C.DEFAULT_MODEL)

    @property
    def effort(self) -> str:
        return str(self.s.value('assistant/effort', 'medium') or 'medium')

    @property
    def show_reasoning(self) -> bool:
        return self.s.value('assistant/show_reasoning', True, type=bool)

    def models(self) -> list:
        v = self.s.value('assistant/models', [])
        ids = [v] if isinstance(v, str) else list(v or [])
        ids = [i for i in ids if i]
        if not ids:
            ids = [m for m in C.KNOWN]
        if self.model not in ids:
            ids.insert(0, self.model)
        return sorted(dict.fromkeys(ids), key=lambda i: (C.model_info(i).rank, i))

    def set_option(self, **kw):
        for k, v in kw.items():
            self.s.setValue(f'assistant/{k}', v)
        if self.session is not None:
            self.session.settings = self._settings()
        self.changed.emit()

    def reload(self):
        """After Preferences: re-read the key and the model."""
        self.key, self.key_source = K.load()
        if self.session is not None:
            self.session.client = C.Client(self.key or '')
            self.session.settings = self._settings()
        self.changed.emit()

    def _settings(self, agent=False) -> A.Settings:
        return A.Settings(self.model, self.effort, self.show_reasoning, agent=agent)

    def set_agent(self, on: bool):
        if on != self.agent_mode:
            self.agent_mode = bool(on)
            self.agent_changed.emit(self.agent_mode)

    # ---------------------------------------------------------------- project brief (saved with the project)
    def brief(self) -> str:
        return str((self.ctx.store.project.analysis.assistant or {}).get('brief', ''))

    def set_brief(self, text: str):
        text = text.strip()
        if text == self.brief():
            return

        def fn(p):
            p.analysis.assistant = dict(p.analysis.assistant or {}, brief=text)
        self.ctx.store.edit('analysis', fn, 'Assistant · project brief')
        self.changed.emit()

    # ---------------------------------------------------------------- conversation
    def ask(self, text: str, agent: bool | None = None) -> bool:
        text = text.strip()
        if not text or self.busy or not self.ready:
            return False
        agent = self.agent_mode if agent is None else agent
        if self.session is None:
            self.session = A.Session(self.key, self._settings(agent))
        self.session.settings = self._settings(agent)
        self.agent_running = agent
        tc = self._context()
        tc.agent = agent
        tid_next = self._tid + 1
        tc.on_ui = lambda req: self._post_ui(tid_next, req)
        self.last_question = text
        self._mark = len(self.session.messages)
        self._cancel = threading.Event()
        self._tid += 1
        tid, session, cancel = self._tid, self.session, self._cancel
        self.busy, self.status, self._started = True, '', time.perf_counter()
        self.changed.emit()

        def work():
            session.run_turn(text, tc, lambda ev: self._post(tid, ev), cancel)
        threading.Thread(target=work, daemon=True, name='assistant').start()
        return True

    def _post(self, tid, ev):
        try:
            self._raw.emit((tid, ev))
        except RuntimeError:            # the window closed while the model was answering
            pass

    def _post_ui(self, tid, req):
        try:
            self._ui_req.emit((tid, req))
        except RuntimeError:
            pass

    def _do_ui(self, item):
        """A show_in_app request, on the UI thread. Never changes the design."""
        tid, req = item
        w = self.ctx.window
        if tid != self._tid or w is None:
            return
        st, a = self.ctx.store, req.get('action')
        try:
            if a in ('frame_point', 'frame_all', 'heatmap', 'select_cameras') and w.page_name() not in ('viewports',):
                w.set_page('viewports')
            if a == 'select_cameras':
                uids = [c.uid for c in st.project.variant.cameras if c.tag in req.get('cameras', [])]
                if uids:
                    st.select_cams(uids)
                    w.frame_selection()
            elif a == 'frame_point':
                w.frame_point((req['x'], req['y']))
            elif a == 'frame_all':
                w.frame_all()
            elif a == 'heatmap':
                w.set_heatmap(req['heatmap'])
            elif a == 'overlay':
                from .context import OVERLAYS
                if req.get('overlay') in {k for k, *_ in OVERLAYS}:
                    self.ctx.set_overlay(req['overlay'], bool(req.get('on', True)))
            elif a == 'open_page':
                if req.get('page') in w.PAGE_DOCKS:
                    w.set_page(req['page'])
            elif a == 'dataset_written':
                res = req.get('result')
                w.page('dataset').show_written(res)            # the page shows it (and the project's log has it)
                if res is not None:
                    self.ctx.msg(f'The assistant wrote a sample dataset to {res.out_dir}', 8000)
            elif a == 'propose_requirements':
                w.set_page('requirements')
                w.page('requirements').show_proposals(req.get('items', []))
        except Exception as e:          # showing something must never break the conversation
            print('assistant show_in_app failed:', e)

    def _on_raw(self, item):
        tid, ev = item
        if tid != self._tid:
            return                      # from a stopped or replaced conversation
        if ev.kind == 'proposal':
            p = ev.data['proposal']
            self.proposals[p.id] = {'p': p, 'state': 'pending', 'undo_index': -1}
        elif ev.kind == 'status':
            self.status = ev.data['text']
        elif ev.kind == 'done':
            self.busy, self.status = False, ''
            ev.data['seconds'] = time.perf_counter() - self._started
        self.event.emit(ev)
        if ev.kind == 'done':
            self.changed.emit()

    def stop(self):
        """Stop at once: the conversation continues from before this question (the old thread winds down alone)."""
        if not self.busy:
            return
        self._cancel.set()
        old = self.session
        fresh = A.Session(self.key, self._settings())
        fresh.messages = list(old.messages[:self._mark])
        fresh.usage = old.usage
        self.session = fresh
        self._tid += 1
        self.busy, self.status = False, ''
        self.event.emit(A.Event('done', {'ok': False, 'stopped': True, 'usage': fresh.usage,
                                         'seconds': time.perf_counter() - self._started}))
        self.changed.emit()

    def new_chat(self, *_):
        if self.busy:
            self._cancel.set()
            self._tid += 1
            self.busy = False
        self.session = None
        self.proposals.clear()
        self.last_question = ''
        self.reset.emit()
        self.changed.emit()

    def transcript(self) -> str:
        return self.session.transcript_markdown() if self.session is not None else ''

    def usage(self):
        return self.session.usage if self.session is not None else None

    def _context(self) -> T.ToolContext:
        ctx, st, an = self.ctx, self.ctx.store, self.ctx.analysis
        w = ctx.window
        page = w.page_name() if w is not None and hasattr(w, 'page_name') else ''
        so = getattr(ctx, 'stitched', None)
        vol = getattr(ctx, 'volume', None)
        return T.ToolContext(st.project.snapshot(), an.result, an.engine.share(), list(an.issues), dict(an.mounts),
                             an.worst, getattr(so, 'result', None), getattr(so, 'advice', None),
                             getattr(vol, 'result', None), self._guidelines(), [c.tag for c in st.selected_cams()],
                             PAGES.get(page, page), st.revision, brief=self.brief(), rig_health=self._rig_health(),
                             drive_log=self._drive_log(), asbuilt=self._asbuilt(), experiment=self._experiment(),
                             working_dir=workdir.get(self.ctx.settings))

    def _rig_health(self):
        w = self.ctx.window
        rig = getattr(w, '_page_widgets', {}).get('rig') if w is not None else None
        return rig.health() if rig is not None else None

    def _drive_log(self):
        w = self.ctx.window
        pg = getattr(w, '_page_widgets', {}).get('drivelog') if w is not None else None
        return pg.summary() if pg is not None else None

    def _asbuilt(self):
        w = self.ctx.window
        pg = getattr(w, '_page_widgets', {}).get('asbuilt') if w is not None else None
        return pg.summary() if pg is not None else None

    def _experiment(self):
        w = self.ctx.window
        pg = getattr(w, '_page_widgets', {}).get('experiments') if w is not None else None
        return pg.summary() if pg is not None else None

    def _guidelines(self):
        try:
            from . import guides
            if not guides.enabled(self.ctx):
                return {'shown': False, 'note': 'turned off (View ▸ Overlays ▸ Parking guidelines, Alt+G)'}
            gc = guides.controller(self.ctx)
            gs = gc.ground()
            return {'shown': True, 'direction': gs.direction, 'road_wheel_angle_deg': round(gs.angle_deg, 1),
                    'turn_radius_m': round(gs.radius, 2) if math.isfinite(gs.radius) else None,
                    'path_m': gs.settings.path_m,
                    'obstacles_in_swept_path': [{'name': ob.name, 'touched_after_m': round(s, 2)} for ob, s in gs.hits]}
        except Exception:
            return None

    # ---------------------------------------------------------------- proposals
    def check(self, pid) -> str:
        """'' when the proposal can be applied as proposed, else why not ('missing', 'locked', 'stale')."""
        e = self.proposals.get(pid)
        if e is None:
            return 'missing'
        p = self.ctx.store.project
        prop = e['p']
        if prop.variant_uid and prop.variant_uid != p.active_uid and (prop.targets or prop.seams):
            return 'missing'
        for k, v in prop.seams_before.items():
            if k in p.variant.seams and abs(p.variant.seams[k] - v) > 1e-6:
                return 'stale'
        for k, v in prop.settings_before.items():
            if abs(float(T.SETTINGS[k][2](p)) - float(v)) > 1e-9:
                return 'stale'
        for uid, props in e['p'].before.items():
            c = p.variant.cam(uid)
            if c is None:
                return 'missing'
            if c.locked:
                return 'locked'
            for k, v in props.items():
                now = getattr(c, k)
                if (now != v) if isinstance(v, (str, bool)) else abs(float(now) - float(v)) > 1e-9:
                    return 'stale'
        return ''

    def apply(self, pid, force=False, as_variant=False) -> str:
        why = self.check(pid)
        if why in ('missing', 'locked') or (why == 'stale' and not force):
            return why
        e = self.proposals[pid]
        prop = e['p']
        st = self.ctx.store
        if as_variant:
            if prop.settings:
                return 'settings'
            self._apply_as_variant(prop)
        else:
            def fn(p):
                for uid, props in prop.targets.items():
                    c = p.variant.cam(uid)
                    if c is not None:
                        for k, v in props.items():
                            setattr(c, k, v)
                for k, ang in prop.seams.items():
                    p.variant.seams[k] = ang
                for k, v in prop.settings.items():
                    T.SETTINGS[k][3](p, v)
            keys = (['variant'] if prop.targets or prop.seams else []) + \
                   (['analysis'] if {'min_density', 'target_density'} & set(prop.settings) else []) + \
                   (['roi'] if 'roi_distance' in prop.settings else [])
            st.edit(keys, fn, f'Assistant · {prop.title}')
        e['state'], e['undo_index'], e['as_variant'] = 'applied', st.undo.index(), as_variant
        e.pop('measured', None)
        self.proposal_state.emit(pid)
        self._measure(pid)
        return ''

    def _apply_as_variant(self, prop):
        from ..core.project import Variant, new_uid
        st = self.ctx.store
        cur = st.project.variant

        def fn(p):
            v = Variant.from_dict(cur.to_dict())
            v.uid = new_uid()
            v.name = f'{cur.name} · {prop.title}'[:64]
            remap = {}
            for c in v.cameras:
                remap[c.uid] = new_uid()
                c.uid = remap[c.uid]
            for c in v.cameras:
                c.link_uid = remap.get(c.link_uid, '')
            rk = lambda k: '|'.join(remap.get(u, u) for u in k.split('|'))
            v.seams = {rk(k): a for k, a in cur.seams.items()}
            for uid, props in prop.targets.items():
                c = v.cam(remap.get(uid, ''))
                if c is not None:
                    for k, val in props.items():
                        setattr(c, k, val)
            for k, ang in prop.seams.items():
                v.seams[rk(k)] = ang
            p.variants.append(v)
            p.active_uid = v.uid
        st.edit('variants', fn, f'Assistant · new variant: {prop.title}')
        w = self.ctx.window
        if w is not None and hasattr(w, '_fill_variant_combo'):
            w._fill_variant_combo()

    def _measure(self, pid):
        """After applying: the next analysis result, next to what was predicted."""
        an = self.ctx.analysis

        def done(res, pid=pid):
            try:
                an.finished.disconnect(done)
            except TypeError:
                pass
            e = self.proposals.get(pid)
            if e is None or res is None or e['state'] != 'applied':
                return
            e['measured'] = {k: float(res.metrics.get(k, 0.0)) for k in ('blind_m2', 'covered_pct', 'multi_pct',
                                                                        'min_density', 'min_ok_pct', 'in_focus_pct')}
            self.proposal_state.emit(pid)
        an.finished.connect(done)
        an.request(force=True)

    def discard(self, pid):
        if pid in self.proposals:
            self.proposals[pid]['state'] = 'discarded'
            self.proposal_state.emit(pid)

    def can_undo(self, pid) -> bool:
        e = self.proposals.get(pid)
        try:
            return bool(e) and e['state'] == 'applied' and self.ctx.store.undo.index() == e['undo_index']
        except RuntimeError:            # the window is closing
            return False

    def undo(self, pid):
        if self.can_undo(pid):
            self.ctx.store.undo.undo()
            self.proposals[pid]['state'] = 'pending'
            self.proposal_state.emit(pid)


# ============================================================================================ markdown
_INLINE = [(re.compile(r'\*\*(?=\S)(.+?)(?<=\S)\*\*'), r'<b>\1</b>'),
           (re.compile(r'(?<![\w*])\*(?=\S)(.+?)(?<=\S)\*(?![\w*])'), r'<i>\1</i>'),
           (re.compile(r'(?<![\w])_(?=\S)(.+?)(?<=\S)_(?![\w])'), r'<i>\1</i>'),
           (re.compile(r'\[([^\]]+)\]\((https?://[^)\s]+)\)'), r'<a href="\2">\1</a>')]
_LIST = re.compile(r'^(\s*)([-*+•]|\d+[.)])\s+(.*)$')
_RULE = re.compile(r'^\s*([-*_])(\s*\1){2,}\s*$')
_SEP = re.compile(r'^\s*\|?\s*:?-{2,}:?\s*(\|\s*:?-{2,}:?\s*)*\|?\s*$')


def _inline(s: str) -> str:
    codes = []

    def keep(m):
        codes.append(m.group(1))
        return f'\x00{len(codes) - 1}\x00'
    s = re.sub(r'`([^`]+)`', keep, s)
    s = html.escape(s, quote=False)
    for rx, rep in _INLINE:
        s = rx.sub(rep, s)
    return re.sub('\x00(\\d+)\x00', lambda m: f'<code>{html.escape(codes[int(m.group(1))], quote=False)}</code>', s)


def _cells(line: str) -> list:
    t = line.strip()
    if t.startswith('|'):
        t = t[1:]
    if t.endswith('|'):
        t = t[:-1]
    return [c.strip() for c in t.split('|')]


def md_to_html(text: str) -> str:
    """The Markdown chat models write (headings, emphasis, code, lists, tables, quotes, rules) as Qt rich text.
    Tolerates half-written text while streaming."""
    lines = text.replace('\r\n', '\n').split('\n')
    out, i, n = [], 0, len(lines)
    while i < n:
        ln = lines[i]
        s = ln.strip()
        if not s:
            i += 1
            continue
        if s.startswith('```'):
            buf, i = [], i + 1
            while i < n and not lines[i].strip().startswith('```'):
                buf.append(lines[i])
                i += 1
            i += 1
            out.append('<pre>' + html.escape('\n'.join(buf), quote=False) + '</pre>')
            continue
        m = re.match(r'^(#{1,6})\s+(.*)$', s)
        if m:
            lvl = min(len(m.group(1)) + 2, 6)
            out.append(f'<h{lvl}>{_inline(m.group(2).rstrip("#").strip())}</h{lvl}>')
            i += 1
            continue
        if _RULE.match(s):
            out.append('<hr/>')
            i += 1
            continue
        if '|' in s and i + 1 < n and _SEP.match(lines[i + 1]):
            head = _cells(s)
            aligns = []
            for c in _cells(lines[i + 1]):
                aligns.append('right' if c.endswith(':') and not c.startswith(':') else
                              'center' if c.startswith(':') and c.endswith(':') else 'left')
            rows, i = [], i + 2
            while i < n and '|' in lines[i] and lines[i].strip():
                rows.append(_cells(lines[i]))
                i += 1
            al = lambda k: aligns[k] if k < len(aligns) else 'left'
            t = ['<table cellspacing="0" cellpadding="4">', '<tr>' + ''.join(
                f'<th align="{al(k)}">{_inline(c)}</th>' for k, c in enumerate(head)) + '</tr>']
            for r in rows:
                t.append('<tr>' + ''.join(f'<td align="{al(k)}">{_inline(c)}</td>'
                                          for k, c in enumerate(r[:len(head)] + [''] * (len(head) - len(r)))) + '</tr>')
            out.append(''.join(t) + '</table>')
            continue
        if _LIST.match(ln):
            stack = []                                  # [(indent, tag)]
            buf = []
            while i < n:
                m = _LIST.match(lines[i])
                if not m:
                    if lines[i].strip() and stack and lines[i].startswith(' '):   # continuation of an item
                        buf.append(' ' + _inline(lines[i].strip()))
                        i += 1
                        continue
                    break
                ind = len(m.group(1).expandtabs(4))
                tag = 'ol' if m.group(2)[0].isdigit() else 'ul'
                while stack and ind < stack[-1][0]:
                    buf.append(f'</li></{stack.pop()[1]}>')
                if not stack or ind > stack[-1][0]:
                    buf.append(f'<{tag}>')
                    stack.append((ind, tag))
                elif buf and not buf[-1].startswith('<' + tag):
                    buf.append('</li>')
                buf.append('<li>' + _inline(m.group(3)))
                i += 1
            while stack:
                buf.append(f'</li></{stack.pop()[1]}>')
            out.append(''.join(buf))
            continue
        if s.startswith('>'):
            buf = []
            while i < n and lines[i].strip().startswith('>'):
                buf.append(_inline(lines[i].strip()[1:].strip()))
                i += 1
            out.append('<blockquote>' + '<br/>'.join(buf) + '</blockquote>')
            continue
        buf = []
        while i < n and lines[i].strip() and not (lines[i].strip().startswith(('```', '#', '>')) or
                                                  _LIST.match(lines[i]) or _RULE.match(lines[i].strip()) or
                                                  ('|' in lines[i] and i + 1 < n and _SEP.match(lines[i + 1]))):
            buf.append(_inline(lines[i].strip()))
            i += 1
        out.append('<p>' + '<br/>'.join(buf) + '</p>')
    return '\n'.join(out)


def _css() -> str:
    t = THEME.tok
    return (f'p {{ margin: 0 0 7px 0; }} '
            f'h3 {{ font-size: 14px; font-weight: 600; margin: 10px 0 4px 0; }} '
            f'h4 {{ font-size: 13px; font-weight: 600; margin: 9px 0 3px 0; }} '
            f'h5, h6 {{ font-size: 12px; font-weight: 600; margin: 8px 0 2px 0; color: {t["ink2"]}; }} '
            f'ul, ol {{ margin: 0 0 7px 0; }} li {{ margin: 0 0 3px 0; }} '
            f'code {{ font-family: "IBM Plex Mono"; background-color: {t["raised"]}; }} '
            f'pre {{ font-family: "IBM Plex Mono"; background-color: {t["raised"]}; margin: 4px 0 8px 0; }} '
            f'table {{ border-collapse: collapse; margin: 3px 0 9px 0; }} '
            f'th {{ font-weight: 600; background-color: {t["raised"]}; border: 1px solid {t["strong"]}; '
            f'padding: 3px 8px; }} '
            f'td {{ border: 1px solid {t["hair"]}; padding: 3px 8px; }} '
            f'hr {{ color: {t["hair"]}; }} '
            f'a {{ color: {t["ink"]}; }} '
            f'blockquote {{ color: {t["ink2"]}; margin: 0 0 7px 10px; }}')


# ============================================================================================ small widgets
class Spinner(QWidget):
    """A small rotating arc."""

    def __init__(self, size=12, parent=None):
        super().__init__(parent)
        self.setFixedSize(size, size)
        self._a = 0
        self._t = QTimer(self)
        self._t.setInterval(50)
        self._t.timeout.connect(self._tick)
        self._t.start()

    def _tick(self):
        self._a = (self._a + 30) % 360
        self.update()

    def stop(self):
        self._t.stop()
        self.hide()

    def paintEvent(self, e):
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        r = QRectF(1.5, 1.5, self.width() - 3, self.height() - 3)
        p.setPen(QPen(THEME.color('hair'), 1.6))
        p.drawEllipse(r)
        pen = QPen(THEME.color('ink2'), 1.6)
        pen.setCapStyle(Qt.PenCapStyle.RoundCap)
        p.setPen(pen)
        p.drawArc(r, -self._a * 16, 100 * 16)


class Rich(QTextBrowser):
    """Rendered Markdown that grows to its content (no scroll bar of its own)."""

    def __init__(self, parent=None, size=12):
        super().__init__(parent)
        self.setOpenExternalLinks(True)
        self.setFrameShape(QFrame.Shape.NoFrame)
        self.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        self.setFont(sans(size))
        self.document().setDocumentMargin(0)
        self.document().setIndentWidth(18)
        self.viewport().setAutoFillBackground(False)
        self.setStyleSheet('QTextBrowser { background: transparent; border: none; font-family: "IBM Plex Sans"; }')
        self.setTextInteractionFlags(Qt.TextInteractionFlag.TextBrowserInteraction)
        self.document().contentsChanged.connect(self._fit)
        self.text = ''

    def set_markdown(self, text: str):
        self.text = text
        self.document().setDefaultStyleSheet(_css())
        self.setHtml(md_to_html(text))
        self._fit()

    def _fit(self):
        w = max(40, self.viewport().width())
        self.document().setTextWidth(w)
        h = int(math.ceil(self.document().size().height())) + 2
        if h != self.height():
            self.setFixedHeight(h)

    def resizeEvent(self, e):
        super().resizeEvent(e)
        self._fit()


def _qss_frame(name, bg, border, radius=6, left=None):
    lb = f'border-left: 3px solid {left};' if left else ''
    return (f'QFrame#{name} {{ background: {bg}; border: 1px solid {border}; border-radius: {radius}px; {lb} }}'
            f'QFrame#{name} QLabel {{ background: transparent; border: none; }}')


def _cam_dot(slot, tag):
    """'● ML' in the camera's colour."""
    return f'<span style="color:{THEME.cam_hex(slot)}">●</span>&nbsp;<b>{html.escape(tag)}</b>'


def _num(v, unit=''):
    if isinstance(v, str):
        return v
    if v is None:
        return '—'
    av = abs(v)
    d = 3 if av < 10 else 2 if av < 100 else 1
    return f'{v:.{d}f}'.rstrip('0').rstrip('.') + (f' {unit}' if unit and unit not in ('%', '°') else unit)


# ============================================================================================ message parts
class Thinking(QWidget):
    """The model's reasoning: collapsed by default, with how long it thought."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.t0 = time.perf_counter()
        self.text = ''
        v = QVBoxLayout(self)
        v.setContentsMargins(0, 0, 0, 0)
        v.setSpacing(4)
        self.head = QToolButton()
        self.head.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonTextBesideIcon)
        self.head.setCursor(Qt.CursorShape.PointingHandCursor)
        self.head.setFont(sans(11))
        self.head.setIconSize(QSize(10, 10))
        self.head.clicked.connect(self.toggle)
        row = QHBoxLayout()
        row.setContentsMargins(0, 0, 0, 0)
        row.setSpacing(6)
        self.spin = Spinner(10)
        row.addWidget(self.head)
        row.addWidget(self.spin)
        row.addStretch(1)
        v.addLayout(row)
        self.body = QLabel()
        self.body.setWordWrap(True)
        self.body.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        self.body.setFont(sans(11))
        self.body.setObjectName('thinkBody')
        self.body.hide()
        v.addWidget(self.body)
        self.open = False
        self.done = False
        self._restyle()
        self._label()

    def _restyle(self):
        t = THEME.tok
        self.head.setStyleSheet(f'QToolButton {{ color: {t["muted"]}; padding: 0; border: none; }}'
                                f'QToolButton:hover {{ color: {t["ink2"]}; }}')
        self.body.setStyleSheet(f'QLabel#thinkBody {{ color: {t["muted"]}; border-left: 2px solid {t["hair"]}; '
                                f'padding: 2px 0 2px 10px; }}')

    def _label(self):
        secs = time.perf_counter() - self.t0
        text = (f'Thought for {secs:.0f} s' if secs >= 1 else 'Thought briefly') if self.done else 'Thinking'
        self.head.setText(text)
        self.head.setIcon(icons.icon('chev-d' if self.open else 'chev-r', 10, THEME.tok['muted']))

    def add(self, text):
        self.text += text
        if self.open:
            self.body.setText(self.text.strip())

    def finish(self):
        if not self.done:
            self.done = True
            self.spin.stop()
            self._label()

    def toggle(self):
        self.open = not self.open
        self.body.setText(self.text.strip())
        self.body.setVisible(self.open)
        self._label()


def _arg_summary(name, args):
    if not isinstance(args, dict):
        return ''
    if name == 'get_camera':
        return str(args.get('camera', ''))
    if name == 'run_study':
        return str(args.get('study', '')).replace('_', ' ')
    if name == 'optimise_camera':
        bits = [str(args.get('camera', ''))]
        if args.get('vary'):
            bits.append(', '.join(map(str, args['vary'])))
        if args.get('objective'):
            bits.append(f'for {args["objective"]}')
        return ' · '.join(b for b in bits if b)
    if name == 'compare_variants':
        return ', '.join(map(str, args.get('variants') or [])) or 'all variants'
    if name == 'show_in_app':
        a = str(args.get('action', '')).replace('_', ' ')
        extra = args.get('cameras') or args.get('heatmap') or args.get('overlay') or args.get('page') or ''
        if args.get('action') == 'frame_point':
            try:
                extra = f'({float(args.get("x")):.2f}, {float(args.get("y")):.2f})'
            except (TypeError, ValueError):
                extra = ''
        return f'{a} {", ".join(extra) if isinstance(extra, list) else extra}'.strip()
    if name == 'explain_point':
        try:
            return f'({float(args.get("x", 0)):.2f}, {float(args.get("y", 0)):.2f}, {float(args.get("z", 0) or 0):.2f}) m'
        except (TypeError, ValueError):
            return ''
    if name in ('simulate_changes', 'propose_changes'):
        bits = []
        for ch in (args.get('changes') or [])[:3]:
            if not isinstance(ch, dict):
                continue
            parts = []
            for k, v in ch.items():
                if k in ('camera', 'mirror_partner'):
                    continue
                base = T.DELTAS.get(k, k)
                try:
                    f = float(v)
                    if base in ('x', 'y', 'z'):
                        txt = f'{f * 1000:+.0f} mm' if k in T.DELTAS else f'{f:.3f} m'
                    elif base in ('yaw', 'pitch', 'roll'):
                        txt = f'{f:+.1f}°' if k in T.DELTAS else f'{f:.1f}°'
                    else:
                        txt = str(v)
                except (TypeError, ValueError):
                    txt = str(v)
                parts.append(f'{base} {txt}'.replace('-', '−'))
            bits.append(f'{ch.get("camera", "?")} {", ".join(parts)}')
        return ' · '.join(bits)
    return ''


class ToolRow(QFrame):
    """One tool call: what it did, and on click exactly what the model asked and what was sent back to Groq."""
    KIND_ICON = {'read': 'search', 'simulate': 'play', 'propose': 'sparkle', 'compute': 'tolerance', 'ui': 'frame',
                 'plan': 'check'}
    NAME_ICON = {'optimise_camera': 'optimise', 'compare_variants': 'compare', 'run_study': 'report'}

    def __init__(self, data, parent=None):
        super().__init__(parent)
        self.setObjectName('toolRow')
        self.name = data['name']
        self.args = data.get('args') or {}
        self.result = ''
        v = QVBoxLayout(self)
        v.setContentsMargins(8, 4, 8, 4)
        v.setSpacing(4)
        h = QHBoxLayout()
        h.setContentsMargins(0, 0, 0, 0)
        h.setSpacing(6)
        self.ico = QLabel()
        self.ico.setPixmap(icons.pixmap(self.NAME_ICON.get(self.name) or self.KIND_ICON.get(data.get('kind'), 'search'),
                                        THEME.tok['ink2'], 12))
        h.addWidget(self.ico)
        summ = _arg_summary(self.name, self.args)
        self.title = QLabel(f'{html.escape(data["label"])}' + (f'<span style="color:{THEME.tok["muted"]}">'
                                                                 f' · {html.escape(summ)}</span>' if summ else ''))
        self.title.setFont(sans(11))
        self.title.setWordWrap(True)
        self.title.setMinimumWidth(60)
        h.addWidget(self.title, 1)
        self.spin = Spinner(10)
        h.addWidget(self.spin)
        self.meta = QLabel('')
        self.meta.setFont(mono(10))
        h.addWidget(self.meta)
        self.chev = QLabel()
        self.chev.setPixmap(icons.pixmap('chev-r', THEME.tok['muted'], 10))
        h.addWidget(self.chev)
        v.addLayout(h)
        self.progress = QLabel('')
        self.progress.setFont(sans(10))
        self.progress.setWordWrap(True)
        self.progress.hide()
        v.addWidget(self.progress)
        self.detail = QLabel()
        self.detail.setFont(mono(10))
        self.detail.setWordWrap(True)
        self.detail.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        self.detail.hide()
        v.addWidget(self.detail)
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self.setToolTip('Click to see what the model asked for and what Vantage sent back')
        self._restyle()

    def _restyle(self):
        t = THEME.tok
        self.setStyleSheet(_qss_frame('toolRow', t['panel'], t['hair'], 5) +
                           f'QFrame#toolRow:hover {{ border-color: {t["strong"]}; }}')
        self.meta.setStyleSheet(f'color: {t["muted"]};')
        self.progress.setStyleSheet(f'color: {t["muted"]};')
        self.detail.setStyleSheet(f'color: {t["ink2"]};')

    def set_progress(self, text):
        self.progress.setText(text)
        self.progress.setVisible(bool(text))

    def finish(self, data):
        self.result = data.get('result', '')
        self.spin.stop()
        self.progress.hide()
        ok = data.get('ok', True)
        size = len(self.result.encode('utf-8'))
        self.meta.setText((f'{size / 1024:.1f} kB sent' if size >= 1024 else f'{size} B sent') if ok else 'failed')
        if not ok:
            self.meta.setStyleSheet(f'color: {THEME.color("serious").name()};')

    def mousePressEvent(self, e):
        if e.button() != Qt.MouseButton.LeftButton:
            return super().mousePressEvent(e)
        open_ = not self.detail.isVisible()
        if open_:
            try:
                res = json.dumps(json.loads(self.result), indent=1, ensure_ascii=False) if self.result else ''
            except ValueError:
                res = self.result
            if len(res) > 4000:
                res = res[:4000] + '\n…'
            a = json.dumps(self.args, ensure_ascii=False) if self.args else '(no arguments)'
            mut = THEME.tok['muted']
            self.detail.setText(f'<span style="color:{mut}">The model asked</span><br/>{html.escape(a)}<br/><br/>'
                                f'<span style="color:{mut}">Sent back to Groq</span><br/>'
                                f'<pre style="white-space: pre-wrap">{html.escape(res)}</pre>')
        self.detail.setVisible(open_)
        self.chev.setPixmap(icons.pixmap('chev-d' if open_ else 'chev-r', THEME.tok['muted'], 10))


class PlanCard(QFrame):
    """The agent's plan as a live checklist."""
    MARK = {'pending': ('○', 'muted'), 'in_progress': ('◐', 'ink'), 'done': ('✓', 'good'), 'skipped': ('–', 'muted')}

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName('plan')
        self.v = QVBoxLayout(self)
        self.v.setContentsMargins(10, 8, 10, 8)
        self.v.setSpacing(4)
        head = QHBoxLayout()
        head.setSpacing(6)
        t = QLabel('Plan')
        t.setFont(sans(11, QFont.Weight.DemiBold))
        head.addWidget(t)
        self.count = label('', muted=True, small=True)
        head.addWidget(self.count, 1)
        self.v.addLayout(head)
        self.rows = []
        tok = THEME.tok
        self.setStyleSheet(_qss_frame('plan', tok['panel'], tok['strong'], 6))

    def set_steps(self, steps):
        for r in self.rows:
            r.hide()
            r.setParent(None)
            r.deleteLater()
        self.rows = []
        tok = THEME.tok
        for st in steps:
            mark, col = self.MARK.get(st['status'], self.MARK['pending'])
            colour = THEME.color(col).name() if col in ('good',) else tok[col]
            text = html.escape(st['step'])
            if st['status'] == 'skipped':
                text = f'<s>{text}</s>'
            weight = 'font-weight:600;' if st['status'] == 'in_progress' else ''
            ink = tok['muted'] if st['status'] in ('done', 'skipped') else tok['ink']
            w = QLabel(f'<table cellspacing="0" cellpadding="0"><tr><td width="16" style="color:{colour}">{mark}</td>'
                       f'<td style="color:{ink};{weight}">{text}</td></tr></table>')
            w.setWordWrap(True)
            w.setFont(sans(11))
            self.v.addWidget(w)
            self.rows.append(w)
        done = sum(1 for st in steps if st['status'] in ('done', 'skipped'))
        self.count.setText(f'{done} of {len(steps)}')


class ProposalCard(QFrame):
    """A proposed change with its predicted effect; Apply makes it one undo step."""

    def __init__(self, ctl: AssistantController, prop: T.Proposal, parent=None):
        super().__init__(parent)
        self.ctl, self.prop = ctl, prop
        self.setObjectName('proposal')
        v = QVBoxLayout(self)
        v.setContentsMargins(12, 10, 12, 10)
        v.setSpacing(8)
        head = QHBoxLayout()
        head.setSpacing(8)
        ico = QLabel()
        ico.setPixmap(icons.pixmap('sparkle', THEME.tok['ink'], 14))
        head.addWidget(ico)
        t = QLabel(html.escape(prop.title))
        t.setFont(sans(12, QFont.Weight.DemiBold))
        t.setWordWrap(True)
        head.addWidget(t, 1)
        self.badge = QLabel('Proposal')
        self.badge.setFont(sans(10, QFont.Weight.DemiBold))
        head.addWidget(self.badge, 0, Qt.AlignmentFlag.AlignTop)
        v.addLayout(head)
        if prop.rationale:
            r = QLabel(html.escape(prop.rationale))
            r.setWordWrap(True)
            r.setFont(sans(11))
            r.setProperty('ink2', True)
            v.addWidget(r)
        # the changes
        g = QGridLayout()
        g.setHorizontalSpacing(12)
        g.setVerticalSpacing(3)
        for c, text in enumerate(('Camera', 'Change', 'Now', 'Proposed')):
            h = label(text, muted=True, small=True)
            g.addWidget(h, 0, c, Qt.AlignmentFlag.AlignRight if c >= 2 else Qt.AlignmentFlag.AlignLeft)
        for i, r in enumerate(prop.rows, 1):
            kind = r.get('kind', 'camera')
            cam = QLabel(_cam_dot(r['slot'], r['camera']) if kind == 'camera' else
                         f'<b>{html.escape(r["label"] if kind == "seam" else "Project")}</b>')
            cam.setFont(sans(11))
            g.addWidget(cam, i, 0)
            what = r['prop'] + (f'  ·  mirrors {r["mirrored_from"]}' if r.get('mirrored_from') else '')
            wl = label(what, ink2=True, small=True, wrap=True)
            wl.setMinimumWidth(40)
            g.addWidget(wl, i, 1)
            for c, val in ((2, r['before']), (3, r['after'])):
                if isinstance(val, (int, float)) and r['unit'] == 'm':
                    text = f'{val:.3f} m'
                elif isinstance(val, (int, float)) and r['unit'] == '°':
                    text = f'{val:.1f}°'
                elif isinstance(val, (int, float)) and r['unit']:
                    text = f'{val:g} {r["unit"]}'
                else:
                    text = str(val)
                w = QLabel(text)
                w.setFont(mono(11))
                g.addWidget(w, i, c, Qt.AlignmentFlag.AlignRight)
            if isinstance(r['before'], (int, float)) and isinstance(r['after'], (int, float)) and r['unit']:
                d = r['after'] - r['before']
                dt = (f'{d * 1000:+.0f} mm' if r['unit'] == 'm' and r.get('kind', 'camera') == 'camera' else
                      f'{d:+.2f} m' if r['unit'] == 'm' else f'{d:+.1f}°' if r['unit'] == '°' else f'{d:+g}')
                dl = QLabel(dt)
                dl.setFont(mono(10))
                dl.setProperty('muted', True)
                g.addWidget(dl, i, 4, Qt.AlignmentFlag.AlignRight)
        g.setColumnStretch(1, 1)
        v.addLayout(g)
        v.addWidget(self._hair())
        # predicted effect
        e = QGridLayout()
        e.setHorizontalSpacing(12)
        e.setVerticalSpacing(3)
        e.addWidget(label('Predicted effect', muted=True, small=True), 0, 0)
        for c, text in ((1, 'Now'), (2, 'After')):
            e.addWidget(label(text, muted=True, small=True), 0, c, Qt.AlignmentFlag.AlignRight)
        row = 1
        for ef in prop.effects:
            if ef['better'] is None and ef['key'] not in ('blind_m2', 'min_density'):
                continue
            el = label(ef['label'], ink2=True, small=True, wrap=True)
            el.setMinimumWidth(60)
            e.addWidget(el, row, 0)
            for c, val in ((1, ef['before']), (2, ef['after'])):
                w = QLabel(_num(val, ef['unit']))
                w.setFont(mono(11))
                e.addWidget(w, row, c, Qt.AlignmentFlag.AlignRight)
            if ef['better'] is not None:
                col = THEME.color('good' if ef['better'] else 'critical').name()
                arrow = '▲' if ef['after'] > ef['before'] else '▼'
                m = QLabel(f'<span style="color:{col}">{arrow}</span>')
                e.addWidget(m, row, 3)
            row += 1
        if prop.issues_before != prop.issues_after:
            e.addWidget(label('Issues', ink2=True, small=True), row, 0)
            for c, val in ((1, prop.issues_before), (2, prop.issues_after)):
                w = QLabel(str(val))
                w.setFont(mono(11))
                e.addWidget(w, row, c, Qt.AlignmentFlag.AlignRight)
            col = THEME.color('good' if prop.issues_after < prop.issues_before else 'critical').name()
            e.addWidget(QLabel(f'<span style="color:{col}">{"▼" if prop.issues_after < prop.issues_before else "▲"}'
                               f'</span>'), row, 3)
        e.setColumnStretch(0, 1)
        v.addLayout(e)
        # actions
        self.note = label('', small=True, wrap=True)
        self.note.hide()
        v.addWidget(self.note)
        self.measured = label('', small=True, wrap=True)
        self.measured.hide()
        v.addWidget(self.measured)
        b = QHBoxLayout()
        b.setSpacing(6)
        self.state_lbl = label('', muted=True, small=True)
        b.addWidget(self.state_lbl, 1)
        self.b_discard = button('Discard', 'ghost')
        self.b_discard.clicked.connect(lambda: ctl.discard(prop.id))
        self.b_apply = button('Apply', 'primary')
        self.b_apply.setToolTip('Apply these changes as one step (Edit ▸ Undo reverts it)')
        self.b_apply.clicked.connect(self._apply)
        self.b_variant = button('As new variant', None, 'copy',
                                'Apply to a copy of this variant and switch to it; the original stays as it is '
                                '(Analysis ▸ Compare shows them side by side)')
        self.b_variant.clicked.connect(lambda: self._apply(as_variant=True))
        self.b_undo = button('Undo', 'ghost', 'undo')
        self.b_undo.clicked.connect(lambda: ctl.undo(prop.id))
        for w in (self.b_undo, self.b_discard, self.b_variant, self.b_apply):
            b.addWidget(w)
        v.addLayout(b)
        self._force = False
        ctl.proposal_state.connect(self._sync)
        ctl.ctx.store.undo.indexChanged.connect(lambda *_: self._sync(prop.id))
        self._restyle()
        self._sync(prop.id)

    def _show_measured(self, m):
        if not m:
            self.measured.hide()
            return
        pred = {ef['key']: ef['after'] for ef in self.prop.effects}
        tok = THEME.tok
        parts, off = [], []
        for key, lab, unit, tol in (('blind_m2', 'blind ground', 'm²', 0.003), ('min_density', 'minimum density', 'px/cm', 0.05),
                                    ('covered_pct', 'covered', '%', 0.05)):
            if key in m:
                parts.append(f'{lab} {_num(m[key], unit)}')
                if key in pred and abs(m[key] - pred[key]) > tol:
                    off.append(f'{lab} predicted {_num(pred[key], unit)}')
        ok = not off
        col = THEME.color('good' if ok else 'warning').name()
        verdict = 'as predicted' if ok else 'differs: ' + ', '.join(off)
        self.measured.setText(f'<span style="color:{col}">{"✓" if ok else "!"}</span> '
                              f'<span style="color:{tok["ink2"]}">Measured after applying: {", ".join(parts)} · '
                              f'{verdict}</span>')
        self.measured.show()

    @staticmethod
    def _hair():
        f = QFrame()
        f.setFixedHeight(1)
        f.setStyleSheet(f'background: {THEME.tok["hair"]};')
        return f

    def _restyle(self):
        t = THEME.tok
        self.setStyleSheet(_qss_frame('proposal', t['raised'], t['strong'], 6))
        self.badge.setStyleSheet(f'color: {t["ink2"]}; border: 1px solid {t["strong"]}; border-radius: 3px; '
                                 f'padding: 0 5px;')

    def _apply(self, as_variant=False):
        why = self.ctl.apply(self.prop.id, force=self._force, as_variant=as_variant)
        if why == 'stale':
            self._force = True
            self.note.setText('The design changed since this was proposed. Apply sets the proposed values anyway.')
            self.note.show()
            self.b_apply.setText('Apply anyway')
        elif why == 'locked':
            self.note.setText('A camera in this proposal is locked. Unlock it to apply.')
            self.note.show()
        elif why == 'missing':
            self.note.setText('A camera in this proposal is not in the current variant.')
            self.note.show()

    def _sync(self, pid):
        if pid != self.prop.id:
            return
        try:
            self._sync_now(pid)
        except RuntimeError:            # the window is closing
            pass

    def _sync_now(self, pid):
        st = self.ctl.proposals.get(pid, {}).get('state', 'discarded')
        pending = st == 'pending'
        e = self.ctl.proposals.get(pid, {})
        self.b_apply.setVisible(pending)
        self.b_discard.setVisible(pending)
        self.b_variant.setVisible(pending and not self.prop.settings and bool(self.prop.targets or self.prop.seams))
        self.b_undo.setVisible(st == 'applied' and self.ctl.can_undo(pid))
        self._show_measured(e.get('measured') if st == 'applied' else None)
        if st == 'applied':
            self.state_lbl.setText('Applied as a new variant' if e.get('as_variant') else 'Applied · one undo step')
            self.badge.setText('Applied')
            self.note.hide()
        elif st == 'discarded':
            self.state_lbl.setText('Discarded')
            self.badge.setText('Discarded')
        else:
            self.state_lbl.setText('Nothing changes until you apply it.')
            self.badge.setText('Proposal')
            self.b_apply.setText('Apply anyway' if self._force else 'Apply')
        self.setEnabled(True)


class Notice(QFrame):
    """An error or a note inside a turn, with what to do about it."""
    settings = pyqtSignal()
    retry = pyqtSignal()

    def __init__(self, kind, message, parent=None):
        super().__init__(parent)
        self.setObjectName('notice')
        h = QHBoxLayout(self)
        h.setContentsMargins(10, 8, 10, 8)
        h.setSpacing(8)
        sev = 'warning' if kind in ('rate', 'rounds', 'network') else 'critical'
        ico = QLabel()
        ico.setPixmap(icons.status_icon(sev, 14).pixmap(14, 14))
        h.addWidget(ico, 0, Qt.AlignmentFlag.AlignTop)
        m = QLabel(html.escape(message))
        m.setWordWrap(True)
        m.setFont(sans(11))
        h.addWidget(m, 1)
        if kind in ('auth', 'model', 'permission'):
            b = button('Settings…', 'ghost')
            b.clicked.connect(self.settings.emit)
            h.addWidget(b, 0, Qt.AlignmentFlag.AlignTop)
        if kind not in ('auth', 'cancelled'):
            b = button('Retry', 'ghost', 'refresh')
            b.clicked.connect(self.retry.emit)
            h.addWidget(b, 0, Qt.AlignmentFlag.AlignTop)
        t = THEME.tok
        self.setStyleSheet(_qss_frame('notice', t['raised'], t['hair'], 5, THEME.color(sev).name()))


# ============================================================================================ turns
class UserTurn(QWidget):
    def __init__(self, text, parent=None):
        super().__init__(parent)
        h = QHBoxLayout(self)
        h.setContentsMargins(36, 0, 0, 0)
        h.addStretch(1)
        self.bubble = QFrame()
        self.bubble.setObjectName('userBubble')
        b = QVBoxLayout(self.bubble)
        b.setContentsMargins(11, 7, 11, 7)
        self.lbl = QLabel(html.escape(text).replace('\n', '<br/>'))
        self.lbl.setWordWrap(True)
        self.lbl.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        self.lbl.setFont(sans(12))
        b.addWidget(self.lbl)
        h.addWidget(self.bubble)
        t = THEME.tok
        self.bubble.setStyleSheet(_qss_frame('userBubble', t['sel'], t['sel'], 9))


class AssistantTurn(QWidget):
    """One answer: reasoning, tool calls and text in the order they came, then a footer with copy and figures."""
    retry = pyqtSignal()
    settings = pyqtSignal()

    def __init__(self, ctl: AssistantController, parent=None):
        super().__init__(parent)
        self.ctl = ctl
        self.v = QVBoxLayout(self)
        self.v.setContentsMargins(0, 0, 0, 0)
        self.v.setSpacing(7)
        head = QHBoxLayout()
        head.setSpacing(6)
        ico = QLabel()
        ico.setPixmap(icons.pixmap('sparkle', THEME.tok['ink'], 14))
        head.addWidget(ico)
        name = QLabel('Assistant')
        name.setFont(sans(11, QFont.Weight.DemiBold))
        head.addWidget(name)
        self.model_lbl = label(C.model_info(ctl.model).label, muted=True, small=True)
        head.addWidget(self.model_lbl)
        head.addStretch(1)
        self.prog = label('', muted=True, small=True)
        head.addWidget(self.prog)
        self.working = Spinner(12)
        head.addWidget(self.working)
        self.v.addLayout(head)
        self.plan: PlanCard | None = None
        self._running: ToolRow | None = None
        self.thinking: Thinking | None = None
        self.text_w: Rich | None = None
        self.tools = {}
        self.texts = []
        self._pending = False
        self._timer = QTimer(self)
        self._timer.setSingleShot(True)
        self._timer.setInterval(60)
        self._timer.timeout.connect(self._render)
        self.status = label('', muted=True, small=True)
        self.status.hide()
        self.v.addWidget(self.status)
        self.footer = QWidget()
        f = QHBoxLayout(self.footer)
        f.setContentsMargins(0, 0, 0, 0)
        f.setSpacing(4)
        self.stats = label('', muted=True, small=True)
        f.addWidget(self.stats, 1)
        self.b_copy = tool_button('copy', 'Copy the answer as Markdown', size=22, icon_size=12)
        self.b_copy.clicked.connect(self._copy)
        f.addWidget(self.b_copy)
        self.footer.hide()
        self._n_tools = 0

    def _insert(self, w):
        self.v.insertWidget(self.v.indexOf(self.status), w)

    def on(self, ev: A.Event):
        k, d = ev.kind, ev.data
        if k == 'reasoning':
            if self.thinking is None or (self.text_w is not None and self.v.indexOf(self.text_w) > self.v.indexOf(self.thinking)):
                self.thinking = Thinking()
                self._insert(self.thinking)
            self.thinking.add(d['text'])
        elif k == 'content':
            if self.thinking is not None:
                self.thinking.finish()
            if self.text_w is None or self._last_is_tool():
                self.text_w = Rich()
                self.texts.append('')
                self._insert(self.text_w)
            self.texts[-1] += d['text']
            self._schedule()
        elif k == 'tool_start':
            if self.thinking is not None:
                self.thinking.finish()
            self._n_tools += 1
            if d['name'] == 'update_plan':          # the plan card shows it
                return
            row = ToolRow(d)
            self.tools[d['id']] = row
            self._running = row
            self._insert(row)
        elif k == 'tool_done':
            row = self.tools.get(d['id'])
            if row is not None:
                row.finish(d)
            self._running = None
        elif k == 'tool_status':
            if self._running is not None:
                self._running.set_progress(d['text'])
        elif k == 'plan':
            if self.plan is None:
                self.plan = PlanCard()
                self.v.insertWidget(1, self.plan)
            self.plan.set_steps(d['steps'])
        elif k == 'progress':
            if d.get('agent'):
                self.prog.setText(f'· agent · step {d["round"] + 1} of {d["max"]} · {d["elapsed"]:.0f} s')
        elif k == 'proposal':
            self._insert(ProposalCard(self.ctl, d['proposal']))
        elif k == 'status':
            self.status.setText(d['text'])
            self.status.show()
        elif k == 'error':
            n = Notice(d['kind'], d['message'])
            n.retry.connect(self.retry.emit)
            n.settings.connect(self.settings.emit)
            self._insert(n)
        elif k == 'done':
            self._finish(d)

    def _last_is_tool(self):
        i = self.v.indexOf(self.status) - 1
        w = self.v.itemAt(i).widget() if i >= 0 else None
        return w is not self.text_w

    def _schedule(self):
        if not self._timer.isActive():
            self._timer.start()

    def _render(self):
        if self.text_w is not None:
            self.text_w.set_markdown(self.texts[-1])

    def _finish(self, d):
        self._timer.stop()
        self._render()
        if self.thinking is not None:
            self.thinking.finish()
        for row in self.tools.values():
            row.spin.stop()
        self.working.stop()
        self.status.hide()
        if self.prog.text():
            self.prog.setText(self.prog.text().split(' · step')[0] + (f' · {d["seconds"]:.0f} s' if d.get('seconds') else ''))
        if d.get('stopped'):
            self._insert(label('Stopped. The conversation continues from before this question.', muted=True, small=True))
        bits = []
        if self._n_tools:
            bits.append(f'{self._n_tools} lookup' + ('s' if self._n_tools != 1 else ''))
        if d.get('seconds'):
            bits.append(f'{d["seconds"]:.1f} s')
        self.stats.setText(' · '.join(bits))
        self.b_copy.setVisible(any(t.strip() for t in self.texts))
        self.v.addWidget(self.footer)
        self.footer.show()

    def _copy(self):
        QApplication.clipboard().setText('\n\n'.join(t.strip() for t in self.texts if t.strip()))
        self.stats.setText('Copied')


# ============================================================================================ composer
class Composer(QFrame):
    send = pyqtSignal(str)
    stop = pyqtSignal()

    def __init__(self, ctl: AssistantController, parent=None):
        super().__init__(parent)
        self.ctl = ctl
        self.setObjectName('composer')
        v = QVBoxLayout(self)
        v.setContentsMargins(10, 8, 8, 6)
        v.setSpacing(4)
        self.edit = QPlainTextEdit()
        self.edit.setFrameShape(QFrame.Shape.NoFrame)
        self.edit.setPlaceholderText('Ask about this design…')
        self.edit.setFont(sans(12))
        self.edit.setTabChangesFocus(True)
        self.edit.setWordWrapMode(QTextOption.WrapMode.WrapAtWordBoundaryOrAnywhere)
        self.edit.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        self.edit.document().setDocumentMargin(2)
        self.edit.textChanged.connect(self._grow)
        self.edit.installEventFilter(self)
        v.addWidget(self.edit)
        row = QHBoxLayout()
        row.setContentsMargins(0, 0, 0, 0)
        row.setSpacing(6)
        self.agent = QPushButton('Agent')
        self.agent.setCheckable(True)
        self.agent.setCursor(Qt.CursorShape.PointingHandCursor)
        self.agent.setFont(sans(10))
        self.agent.setToolTip('Agent mode: describe a goal. The assistant plans, runs studies and searches on its '
                              'own, then proposes the best change it found. Larger budget; Esc stops it.')
        self.agent.toggled.connect(ctl.set_agent)
        row.addWidget(self.agent)
        self.ctx_lbl = QLabel('')
        self.ctx_lbl.setFont(sans(10))
        row.addWidget(self.ctx_lbl, 1)
        self.model_btn = QToolButton()
        self.model_btn.setPopupMode(QToolButton.ToolButtonPopupMode.InstantPopup)
        self.model_btn.setFont(sans(10))
        self.model_btn.setToolTip('Model and reasoning effort')
        self.model_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self.model_btn.setMenu(QMenu(self.model_btn))
        self.model_btn.menu().aboutToShow.connect(self._fill_menu)
        row.addWidget(self.model_btn)
        self.go = QToolButton()
        self.go.setFixedSize(26, 26)
        self.go.setIconSize(QSize(14, 14))
        self.go.setCursor(Qt.CursorShape.PointingHandCursor)
        self.go.clicked.connect(self._go)
        row.addWidget(self.go)
        v.addLayout(row)
        self.popup: SlashPopup | None = None
        self._grow()
        ctl.changed.connect(self.sync)
        ctl.agent_changed.connect(self._agent_changed)
        ctl.ctx.store.selection_changed.connect(self.sync)
        self.sync()

    def _agent_changed(self, on):
        self.agent.blockSignals(True)
        self.agent.setChecked(on)
        self.agent.blockSignals(False)
        self.sync()

    def eventFilter(self, obj, e):
        from PyQt6.QtCore import QEvent
        if obj is self.edit and e.type() == QEvent.Type.KeyPress:
            k, mods = e.key(), e.modifiers()
            pop = self.popup
            if pop is not None and pop.isVisible():
                if k in (Qt.Key.Key_Up, Qt.Key.Key_Down):
                    pop.step(-1 if k == Qt.Key.Key_Up else 1)
                    return True
                if k in (Qt.Key.Key_Tab, Qt.Key.Key_Return, Qt.Key.Key_Enter) and pop.current():
                    self.edit.setPlainText(pop.current())
                    self.edit.moveCursor(self.edit.textCursor().MoveOperation.End)
                    pop.hide()
                    if not pop.current().endswith(' ') and k != Qt.Key.Key_Tab:
                        self._go()
                    return True
                if k == Qt.Key.Key_Escape:
                    pop.hide()
                    return True
            if k in (Qt.Key.Key_Return, Qt.Key.Key_Enter) and not mods & Qt.KeyboardModifier.ShiftModifier:
                self._go()
                return True
            if k == Qt.Key.Key_Escape and self.ctl.busy:
                self.stop.emit()
                return True
            if k == Qt.Key.Key_Up and not self.edit.toPlainText() and self.ctl.last_question:
                self.edit.setPlainText(self.ctl.last_question)
                self.edit.moveCursor(self.edit.textCursor().MoveOperation.End)
                return True
        return super().eventFilter(obj, e)

    def _go(self):
        if self.ctl.busy:
            self.stop.emit()
            return
        text = self.edit.toPlainText().strip()
        if text:
            self.send.emit(text)

    def take(self) -> str:
        t = self.edit.toPlainText()
        self.edit.clear()
        return t

    def _grow(self):
        fm = QFontMetricsF(self.edit.font())
        lines = max(1, min(8, int(self.edit.document().size().height())))
        self.edit.setFixedHeight(int(lines * fm.lineSpacing() + 10))
        if self.popup is not None:
            t = self.edit.toPlainText()
            self.popup.filter(t if t.startswith('/') and ' ' not in t and '\n' not in t else None)
        self.sync()

    def _fill_menu(self):
        m = self.model_btn.menu()
        m.clear()
        cur = self.ctl.model
        for mid in self.ctl.models():
            info = C.model_info(mid)
            a = m.addAction(f'{info.label}    {info.note}' if info.note else info.label)
            a.setCheckable(True)
            a.setChecked(mid == cur)
            a.triggered.connect(lambda _=False, x=mid: self.ctl.set_option(model=x))
        info = C.model_info(cur)
        if info.efforts:
            m.addSeparator()
            hdr = m.addAction('Reasoning effort')
            hdr.setEnabled(False)
            for e in info.efforts:
                a = m.addAction('    ' + e.capitalize())
                a.setCheckable(True)
                a.setChecked(e == (self.ctl.effort if self.ctl.effort in info.efforts else info.default_effort))
                a.triggered.connect(lambda _=False, x=e: self.ctl.set_option(effort=x))
        m.addSeparator()
        a = m.addAction('Show the model\'s reasoning')
        a.setCheckable(True)
        a.setChecked(self.ctl.show_reasoning)
        a.toggled.connect(lambda on: self.ctl.set_option(show_reasoning=on))
        m.addAction('Assistant settings…', lambda: open_settings(self.ctl.ctx))

    def sync(self, *_):
        t = THEME.tok
        busy = self.ctl.busy
        self.go.setIcon(icons.icon('stop' if busy else 'send', 14, t['panel']))
        self.go.setToolTip('Stop  Esc' if busy else 'Send  Enter  (Shift+Enter for a new line)')
        has = bool(self.edit.toPlainText().strip())
        en = busy or (has and self.ctl.ready)
        self.go.setEnabled(en)
        bg = t['ink'] if en else t['sel']
        self.go.setStyleSheet(f'QToolButton, QToolButton:disabled {{ background: {bg}; border: none; '
                              f'border-radius: 13px; min-width: 26px; min-height: 26px; }}'
                              f'QToolButton:hover {{ background: {t["ink2"]}; }}')
        info = C.model_info(self.ctl.model)
        eff = self.ctl.effort if self.ctl.effort in info.efforts else info.default_effort
        self.model_btn.setText(info.label + (f' · {eff}' if info.efforts else '') + '  ▾')
        self.model_btn.setStyleSheet(f'QToolButton {{ color: {t["ink2"]}; border: none; padding: 2px 4px; '
                                     f'border-radius: 3px; }} QToolButton:hover {{ background: {t["sel"]}; }}'
                                     f'QToolButton::menu-indicator {{ image: none; width: 0; }}')
        sel = self.ctl.ctx.store.selected_cams()
        if sel:
            dots = ' '.join(_cam_dot(c.slot, c.tag) for c in sel[:3]) + (' …' if len(sel) > 3 else '')
            self.ctx_lbl.setText(f'<span style="color:{t["muted"]}">Context</span>&nbsp; {dots}')
            self.ctx_lbl.setToolTip('The selected cameras are mentioned with your question')
        else:
            self.ctx_lbl.setText('')
        self.edit.setEnabled(self.ctl.ready)
        on = self.ctl.agent_mode
        self.edit.setPlaceholderText(('Describe a goal, e.g. “no blind ground, at least 4 px/cm”…' if on else
                                      'Ask about this design…   / for commands') if self.ctl.ready
                                     else 'Set up the assistant first')
        self.agent.setStyleSheet(
            f'QPushButton {{ border: 1px solid {t["ink"] if on else t["strong"]}; border-radius: 10px; padding: 1px 9px; '
            f'min-height: 18px; color: {t["panel"] if on else t["ink2"]}; background: {t["ink"] if on else "transparent"}; }}'
            f'QPushButton:hover {{ border-color: {t["ink"]}; }}')
        self.setStyleSheet(f'QFrame#composer {{ background: {t["raised"]}; border: 1px solid {t["strong"]}; '
                           f'border-radius: 8px; }} QPlainTextEdit {{ background: transparent; border: none; '
                           f'font-family: "IBM Plex Sans"; }}')


class SlashPopup(QFrame):
    """Commands that match what is typed after '/'."""

    def __init__(self, parent):
        super().__init__(parent)
        self.setObjectName('slash')
        v = QVBoxLayout(self)
        v.setContentsMargins(4, 4, 4, 4)
        v.setSpacing(0)
        self.items = []
        self.idx = 0
        self.labels = []
        for cmd, what, _ in SLASH:
            l = QLabel()
            l.setFont(sans(11))
            l.setTextFormat(Qt.TextFormat.RichText)
            l.setWordWrap(True)
            v.addWidget(l)
            self.labels.append(l)
        self.hide()

    def filter(self, text):
        if text is None:
            self.hide()
            return
        self.items = [x for x in SLASH if x[0].startswith(text.lower())]
        if not self.items:
            self.hide()
            return
        self.idx = min(self.idx, len(self.items) - 1)
        self._paint()
        self.show()
        self.raise_()

    def step(self, d):
        if self.items:
            self.idx = (self.idx + d) % len(self.items)
            self._paint()

    def current(self):
        return self.items[self.idx][2] if self.items and self.isVisible() else ''

    def _paint(self):
        t = THEME.tok
        self.setStyleSheet(_qss_frame('slash', t['raised'], t['strong'], 6))
        for i, l in enumerate(self.labels):
            if i < len(self.items):
                cmd, what, _ = self.items[i]
                bg = t['sel'] if i == self.idx else 'transparent'
                l.setText(f'<table width="100%" cellpadding="3" style="background:{bg}"><tr><td><b>{cmd}</b> '
                          f'<span style="color:{t["muted"]}">{html.escape(what)}</span></td></tr></table>')
                l.show()
            else:
                l.hide()
        self.adjustSize()
        par = self.parentWidget()
        comp = par.composer
        w = comp.width()
        self.setFixedWidth(w)
        self.adjustSize()
        pos = comp.mapTo(par, comp.rect().topLeft())
        self.move(pos.x(), pos.y() - self.height() - 4)


def edit_brief(ctl: AssistantController, parent=None):
    """The project brief: requirements and constraints the assistant takes into account (saved with the project)."""
    from .dialogs.base import VDialog
    d = VDialog(parent, 'Project brief', 'saved with the project · sent with every question', (580, 440))
    v = QVBoxLayout(d.body)
    v.setContentsMargins(14, 12, 14, 12)
    v.setSpacing(8)
    v.addWidget(label('Requirements, constraints and decisions for this design, in plain language. The assistant '
                      'checks its answers and proposals against them.', ink2=True, wrap=True))
    ed = QPlainTextEdit()
    ed.setPlainText(ctl.brief())
    ed.setPlaceholderText('For example:\n- At least 4 px/cm within 1 m of the bumpers\n- The mirror cameras can move '
                          '±20 mm at most (fixed housing)\n- The rear camera stays in the tailgate handle\n- Pedestrian '
                          'detection uses the stitched view: 0.9 m children must not vanish at the seams')
    ed.setStyleSheet('QPlainTextEdit { font-family: "IBM Plex Sans"; }')
    v.addWidget(ed, 1)
    c = button('Cancel', 'ghost')
    c.clicked.connect(d.reject)
    o = button('Save', 'primary')
    o.clicked.connect(d.accept)
    d.buttons.addWidget(c)
    d.buttons.addWidget(o)
    ed.setFocus()
    if d.exec():
        ctl.set_brief(ed.toPlainText())
        return True
    return False


# ============================================================================================ panel
class FlowChips(QWidget):
    """Suggestion buttons that wrap onto as many rows as the width needs."""
    picked = pyqtSignal(str)

    def __init__(self, items, parent=None):
        super().__init__(parent)
        self.btns = []
        for text, prompt in items:
            b = QPushButton(text)
            b.setCursor(Qt.CursorShape.PointingHandCursor)
            b.setFont(sans(11))
            b.clicked.connect(lambda _=False, p=prompt: self.picked.emit(p))
            b.setParent(self)
            self.btns.append(b)
        t = THEME.tok
        self.setStyleSheet(f'QPushButton {{ border: 1px solid {t["strong"]}; border-radius: 13px; padding: 3px 11px; '
                           f'min-height: 20px; color: {t["ink2"]}; background: transparent; }}'
                           f'QPushButton:hover {{ background: {t["sel"]}; color: {t["ink"]}; }}')
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)

    def _place(self, w, apply=True):
        x = y = 0
        rowh = 0
        for b in self.btns:
            s = b.sizeHint()
            if x and x + s.width() > w:
                x, y = 0, y + rowh + 6
                rowh = 0
            if apply:
                b.setGeometry(x, y, s.width(), s.height())
            x += s.width() + 6
            rowh = max(rowh, s.height())
        return y + rowh

    def resizeEvent(self, e):
        h = self._place(self.width())
        if h != self.height():
            self.setFixedHeight(h)

    def sizeHint(self):
        return QSize(300, self._place(300, False))


class AssistantPanel(QWidget):
    def __init__(self, ctx, parent=None):
        super().__init__(parent)
        self.ctx = ctx
        self.ctl: AssistantController = ctx.assistant
        self.setObjectName('panel')
        self.cur: AssistantTurn | None = None
        v = QVBoxLayout(self)
        v.setContentsMargins(0, 0, 0, 0)
        v.setSpacing(0)
        # header
        head = QWidget()
        head.setObjectName('panel')
        head.setFixedHeight(34)
        h = QHBoxLayout(head)
        h.setContentsMargins(12, 0, 6, 0)
        h.setSpacing(4)
        self.dot = QLabel()
        h.addWidget(self.dot)
        self.title = QLabel('')
        self.title.setFont(sans(11))
        h.addWidget(self.title, 1)
        self.b_brief = QPushButton('Brief')
        self.b_brief.setCursor(Qt.CursorShape.PointingHandCursor)
        self.b_brief.setFont(sans(10))
        self.b_brief.clicked.connect(lambda: edit_brief(self.ctl, self))
        h.addWidget(self.b_brief)
        self.b_new = tool_button('plus', 'New conversation', size=24, icon_size=14)
        self.b_new.clicked.connect(self.ctl.new_chat)
        self.b_more = tool_button('more', 'More', size=24, icon_size=14)
        m = QMenu(self.b_more)
        m.addAction(icons.icon('report', 16), 'Project brief…', lambda: edit_brief(self.ctl, self))
        self.a_export = m.addAction(icons.icon('export', 16), 'Save conversation…', self._export)
        m.addSeparator()
        m.addAction(icons.icon('gear', 16), 'Assistant settings…', lambda: open_settings(self.ctx))
        self.b_more.setMenu(m)
        self.b_more.setPopupMode(QToolButton.ToolButtonPopupMode.InstantPopup)
        self.b_more.setStyleSheet('QToolButton::menu-indicator { image: none; width: 0; }')
        for b in (self.b_new, self.b_more):
            h.addWidget(b)
        v.addWidget(head)
        line = QFrame()
        line.setFixedHeight(1)
        line.setObjectName('hairline')
        v.addWidget(line)
        self.hairline = line
        # transcript
        self.area = QScrollArea()
        self.area.setWidgetResizable(True)
        self.area.setFrameShape(QFrame.Shape.NoFrame)
        self.area.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.body = QWidget()
        self.body.setObjectName('panel')
        self.col = QVBoxLayout(self.body)
        self.col.setContentsMargins(14, 14, 14, 14)
        self.col.setSpacing(16)
        self.col.addStretch(1)
        self.body.setMinimumWidth(0)
        self.area.setWidget(self.body)
        self.area.verticalScrollBar().rangeChanged.connect(self._range)
        v.addWidget(self.area, 1)
        # composer and footer
        bottom = QWidget()
        bottom.setObjectName('panel')
        b = QVBoxLayout(bottom)
        b.setContentsMargins(10, 6, 10, 8)
        b.setSpacing(5)
        self.composer = Composer(self.ctl)
        self.composer.send.connect(self._send)
        self.composer.stop.connect(self.ctl.stop)
        b.addWidget(self.composer)
        self.composer.popup = SlashPopup(self)
        self.foot = label('', muted=True, small=True, wrap=True)
        b.addWidget(self.foot)
        v.addWidget(bottom)
        self._stick = True
        self.empty = None
        self._empty_ready = None
        self.ctl.event.connect(self._event)
        self.ctl.changed.connect(self._sync)
        self.ctl.reset.connect(self._clear)
        self.ctl.agent_changed.connect(lambda *_: self._show_empty() if self.empty is not None else None)
        self.ctx.store.changed.connect(lambda *_: self._brief_btn())
        self.ctx.store.project_replaced.connect(self._brief_btn)
        THEME.changed.connect(self._theme_changed)      # a bound slot: disconnected when the widget goes
        self._restyle()
        self._clear()

    # ---------------------------------------------------------------- state

    def _theme_changed(self, *_):
        self._restyle()
    def _restyle(self):
        t = THEME.tok
        self.hairline.setStyleSheet(f'background: {t["hair"]};')
        self.area.setStyleSheet(f'QScrollArea {{ background: {t["panel"]}; }}')
        self._sync()

    def _sync(self):
        c = self.ctl
        info = C.model_info(c.model)
        if not c.enabled or not c.key:
            col, text = THEME.tok['muted'], 'Not set up'
        elif c.busy:
            col, text = THEME.color('warning').name(), f'{info.label} · working…'
        else:
            col, text = THEME.color('good').name(), f'{info.label} · Groq'
        self.dot.setText(f'<span style="color:{col}">●</span>')
        self.title.setText(text)
        self.a_export.setEnabled(bool(c.transcript()))
        self._brief_btn()
        u = c.usage()
        tok = f' · {u.prompt / 1000:.1f}k tokens in, {u.completion / 1000:.1f}k out' if u and u.prompt else ''
        self.foot.setText(('Answers are computed from Vantage\'s analysis by a model hosted on Groq and can be wrong; '
                           'check a proposal\'s effect before applying it.' if c.ready else
                           'The assistant is off until you add a Groq API key in Preferences ▸ Assistant.') + tok)
        if self.empty is not None and self._empty_ready != c.ready:
            self._show_empty()              # set up, or the key was removed: swap the welcome for the other one

    @staticmethod
    def _drop(w):
        w.hide()                            # deleteLater waits for the event loop; hide it now
        w.setParent(None)
        w.deleteLater()

    def _brief_btn(self, *_):
        t = THEME.tok
        has = bool(self.ctl.brief())
        self.b_brief.setToolTip(('Project brief (saved with the project):\n' + self.ctl.brief()[:400]) if has else
                                'Write a project brief: requirements the assistant takes into account')
        self.b_brief.setStyleSheet(f'QPushButton {{ border: 1px solid {t["strong"] if has else t["hair"]}; '
                                   f'border-radius: 10px; padding: 1px 9px; min-height: 18px; '
                                   f'color: {t["ink"] if has else t["muted"]}; background: transparent; }}'
                                   f'QPushButton:hover {{ border-color: {t["ink"]}; }}')
        self.b_brief.setText('Brief ✓' if has else 'Brief')

    def _clear(self):
        while self.col.count() > 1:
            w = self.col.takeAt(0).widget()
            if w is not None:
                self._drop(w)
        self.cur = None
        self.empty = None
        self._show_empty()
        self._sync()

    def _show_empty(self):
        if self.empty is not None:
            self.col.removeWidget(self.empty)
            self._drop(self.empty)
        self._empty_ready = self.ctl.ready
        w = QWidget()
        v = QVBoxLayout(w)
        v.setContentsMargins(4, 18, 4, 4)
        v.setSpacing(10)
        ico = QLabel()
        ico.setPixmap(icons.pixmap('sparkle', THEME.tok['ink2'], 28))
        v.addWidget(ico)
        if self.ctl.ready and self.ctl.agent_mode:
            t = QLabel('Give the agent a goal')
            t.setFont(sans(15, QFont.Weight.DemiBold))
            v.addWidget(t)
            v.addWidget(label('It plans the work, reads the figures, runs studies, searches camera poses and checks '
                              'each option, then proposes the best change with its measured effect. You stay in '
                              'charge: nothing changes until you apply a proposal, and Esc stops it.', ink2=True,
                              wrap=True))
            chips = FlowChips(AGENT_GOALS)
            chips.picked.connect(self._send)
            v.addSpacing(4)
            v.addWidget(chips)
        elif self.ctl.ready:
            t = QLabel('Ask about this design')
            t.setFont(sans(15, QFont.Weight.DemiBold))
            v.addWidget(t)
            s = label('The assistant reads Vantage\'s own analysis (coverage, density, seams, the studies), runs '
                      'studies and checks what a change would do. It proposes changes; nothing changes until you '
                      'apply one. Type / for commands, or switch on Agent for goals.', ink2=True, wrap=True)
            v.addWidget(s)
            n = len(self.ctx.analysis.issues)
            items = [(f'Explain the {n} issue' + ('s' if n != 1 else ''), prompt) if text == 'Explain the issues'
                     else (text, prompt) for text, prompt in SUGGESTIONS if n or text != 'Explain the issues']
            chips = FlowChips(items)
            chips.picked.connect(self._send)
            v.addSpacing(4)
            v.addWidget(chips)
        else:
            t = QLabel('Connect the assistant')
            t.setFont(sans(15, QFont.Weight.DemiBold))
            v.addWidget(t)
            v.addWidget(label('Ask questions about this design in plain language: why ground is blind, which camera '
                              'matters most, what a change would do. The assistant runs on a model hosted by Groq '
                              'with your own API key.', ink2=True, wrap=True))
            v.addWidget(label('When you ask a question, sensor poses, lens, module and sensor model names, analysis '
                              'figures, the fleet\'s vehicle names and offsets (no VINs) and your requirements are '
                              'sent to Groq. The vehicle model, images, point clouds and files are never sent.',
                              muted=True, small=True, wrap=True))
            b = button('Set up…', 'primary', 'key')
            b.clicked.connect(lambda: open_settings(self.ctx))
            r = QHBoxLayout()
            r.addWidget(b)
            r.addStretch(1)
            v.addSpacing(4)
            v.addLayout(r)
        v.addStretch(1)
        self.empty = w
        self.col.insertWidget(self.col.count() - 1, w)

    # ---------------------------------------------------------------- conversation
    def ask(self, text: str):
        """Ask from outside (the Issues list, the command palette)."""
        self._send(text)

    def _send(self, text):
        text = text.strip()
        if not text:
            return
        if not self.ctl.ready:
            open_settings(self.ctx)
            return
        if self.ctl.busy:
            return
        prompt, agent, local = expand_slash(text)
        if local is not None:
            self.composer.take()
            if self.composer.popup is not None:
                self.composer.popup.hide()
            if local == 'new':
                self.ctl.new_chat()
            elif local == 'brief':
                edit_brief(self.ctl, self)
            elif local == 'agent':
                self.ctl.set_agent(True)
            return
        if not prompt:
            return
        if self.empty is not None:
            self.col.removeWidget(self.empty)
            self._drop(self.empty)
            self.empty = None
        if self.composer.edit.toPlainText().strip() == text:
            self.composer.take()
        self._add(UserTurn(text))
        turn = AssistantTurn(self.ctl)
        turn.retry.connect(lambda: self._send(self.ctl.last_question))
        turn.settings.connect(lambda: open_settings(self.ctx))
        self._add(turn)
        self.cur = turn
        self._stick = True
        if self.composer.popup is not None:
            self.composer.popup.hide()
        if not self.ctl.ask(prompt, agent=agent):
            turn.on(A.Event('error', {'kind': 'internal', 'message': 'The assistant is busy or not set up.'}))
            turn.on(A.Event('done', {'ok': False}))
            self.cur = None

    def _add(self, w):
        self.col.insertWidget(self.col.count() - 1, w)

    def _event(self, ev):
        if self.cur is not None:
            self.cur.on(ev)
            if ev.kind == 'done':
                self.cur = None
        self._sync() if ev.kind in ('usage', 'done') else None

    def _range(self, lo, hi):
        if self._stick:
            self.area.verticalScrollBar().setValue(hi)

    def _export(self):
        text = self.ctl.transcript()
        if not text:
            return
        path, _ = QFileDialog.getSaveFileName(self, 'Save conversation', 'assistant.md', 'Markdown (*.md)')
        if path:
            with open(path, 'w', encoding='utf-8') as f:
                f.write(f'# Vantage assistant · {self.ctx.store.project.name}\n\n{text}')


def open_settings(ctx):
    from .dialogs.misc import PreferencesDialog
    PreferencesDialog(ctx, ctx.window, tab='assistant').exec()


# ============================================================================================ preferences
class AssistantPrefs(QWidget):
    """Preferences ▸ Assistant: turn it on, the API key (tested), the model and the reasoning effort."""
    _tested = pyqtSignal(object)

    def __init__(self, settings, parent=None):
        super().__init__(parent)
        self.s = settings
        self.key, self.source = K.load()
        self._new_key = None            # None: unchanged, '': remove
        g = QGridLayout(self)
        g.setContentsMargins(14, 12, 14, 12)
        g.setVerticalSpacing(7)
        g.setHorizontalSpacing(10)
        r = 0

        def row(text, w, span=1):
            nonlocal r
            a = label(text, ink2=True)
            a.setFixedWidth(110)
            g.addWidget(a, r, 0, Qt.AlignmentFlag.AlignTop if span > 1 else Qt.AlignmentFlag.AlignVCenter)
            g.addWidget(w, r, 1, 1, 2)
            r += 1
        self.on = QCheckBox('Use the assistant')
        self.on.setChecked(self.s.value('assistant/enabled', False, type=bool))
        row('Assistant', self.on)
        kw = QWidget()
        kh = QHBoxLayout(kw)
        kh.setContentsMargins(0, 0, 0, 0)
        kh.setSpacing(6)
        self.key_edit = QLineEdit()
        self.key_edit.setEchoMode(QLineEdit.EchoMode.Password)
        self.key_edit.setFont(mono(11))
        self.key_edit.setPlaceholderText('gsk_…  paste your Groq API key')
        self.key_edit.textEdited.connect(self._edited)
        self.eye = tool_button('eye', 'Show the key', checkable=True, size=24, icon_size=14)
        self.eye.toggled.connect(lambda on: self.key_edit.setEchoMode(QLineEdit.EchoMode.Normal if on else
                                                                      QLineEdit.EchoMode.Password))
        self.test = button('Test', None, None, 'Check the key with Groq and load the models it may use')
        self.test.clicked.connect(self._test)
        kh.addWidget(self.key_edit, 1)
        kh.addWidget(self.eye)
        kh.addWidget(self.test)
        row('API key', kw)
        self.key_note = label('', muted=True, small=True, wrap=True)
        self.key_note.setOpenExternalLinks(True)
        self.key_note.setTextFormat(Qt.TextFormat.RichText)
        g.addWidget(self.key_note, r, 1, 1, 2)
        r += 1
        self.status = label('', small=True, wrap=True)
        g.addWidget(self.status, r, 1, 1, 2)
        r += 1
        self.model = QComboBox()
        self.model.setMinimumWidth(260)
        self.model.currentIndexChanged.connect(self._model_changed)
        row('Model', self.model)
        self.effort = SegmentedControl([('low', 'Low'), ('medium', 'Medium'), ('high', 'High')],
                                       str(self.s.value('assistant/effort', 'medium')))
        self.effort_row = r
        row('Reasoning', self.effort)
        self.show_r = QCheckBox("Show the model's reasoning in the answers")
        self.show_r.setChecked(self.s.value('assistant/show_reasoning', True, type=bool))
        g.addWidget(self.show_r, r, 1, 1, 2)
        r += 1
        g.addWidget(label('What is sent', ink2=True), r, 0, Qt.AlignmentFlag.AlignTop)
        g.addWidget(label('With each question: camera poses, lens and module names, and the analysis figures the '
                          'assistant looks up (each lookup is shown in the conversation). Never the vehicle model, '
                          'images, captures or files. Groq processes them under its own terms.',
                          muted=True, small=True, wrap=True), r, 1, 1, 2)
        r += 1
        g.setRowStretch(r, 1)
        g.setColumnStretch(1, 1)
        self._tested.connect(self._on_tested)
        self._fill_models()
        self._key_ui()

    def _key_ui(self):
        a = f'style="color:{THEME.tok["ink2"]}"'
        link = f'<a {a} href="{C.KEYS_URL}">Create a key at console.groq.com ↗</a>'
        if self._new_key == '':
            self.key_edit.clear()
            self.key_note.setText('The key will be removed when you apply. · ' + link)
        elif self.source == 'environment':
            self.key_edit.setPlaceholderText(f'Set by {K.ENV}: {K.mask(self.key)}')
            self.key_note.setText(f'The {K.ENV} environment variable takes precedence over a saved key. · ' + link)
        elif self.key and self._new_key is None:
            self.key_edit.setPlaceholderText(f'Saved: {K.mask(self.key)}  (type to replace)')
            self.key_note.setText(f'Saved {K.SOURCES.get(self.source, "")}. · <a {a} href="remove">Remove the key</a>'
                                  f' · ' + link)
        else:
            self.key_note.setText('Stored in the system keyring when there is one, otherwise in a private file in '
                                  'your settings folder; never in projects. · ' + link)
        self.key_note.setOpenExternalLinks(False)
        try:
            self.key_note.linkActivated.disconnect()
        except TypeError:
            pass
        self.key_note.linkActivated.connect(self._link)

    def _link(self, href):
        if href == 'remove':
            self._new_key = ''
            self._key_ui()
        else:
            QDesktopServices.openUrl(QUrl(href))

    def _edited(self, text):
        self._new_key = text.strip() or None
        self.status.setText('')

    def current_key(self):
        if self._new_key == '':
            return None
        return self._new_key or self.key

    def _fill_models(self, ids=None):
        cur = self.model.currentData() or str(self.s.value('assistant/model', C.DEFAULT_MODEL))
        v = self.s.value('assistant/models', [])
        ids = ids or ([v] if isinstance(v, str) else list(v or [])) or list(C.KNOWN)
        if cur not in ids:
            ids = [cur] + list(ids)
        ids = sorted(dict.fromkeys(ids), key=lambda i: (C.model_info(i).rank, i))
        self.model.blockSignals(True)
        self.model.clear()
        for i in ids:
            info = C.model_info(i)
            self.model.addItem(f'{info.label}  —  {info.note}' if info.note else info.label, i)
        self.model.setCurrentIndex(max(0, self.model.findData(cur)))
        self.model.blockSignals(False)
        self._model_changed()

    def _model_changed(self, *_):
        info = C.model_info(self.model.currentData() or C.DEFAULT_MODEL)
        opts = [(e, e.capitalize()) for e in info.efforts] if info.efforts else [('medium', 'Medium')]
        cur = self.effort.value()
        lay = self.layout()
        new = SegmentedControl(opts, cur if cur in dict(opts) else (info.default_effort or opts[0][0]))
        lay.replaceWidget(self.effort, new)
        self.effort.hide()
        self.effort.setParent(None)
        self.effort.deleteLater()
        self.effort = new
        new.setEnabled(bool(info.efforts))
        new.setToolTip('' if info.efforts else 'This model does not take a reasoning effort')

    def _test(self):
        key = self.current_key()
        if not key:
            self.status.setText('Enter a key first.')
            return
        self.test.setEnabled(False)
        self.status.setText('Checking the key with Groq…')

        def work():
            try:
                ids = C.Client(key, timeout=20).models()
                self._emit(('ok', ids))
            except C.ApiError as e:
                self._emit(('err', e.message))
            except Exception as e:
                self._emit(('err', f'{type(e).__name__}: {e}'))
        threading.Thread(target=work, daemon=True).start()

    def _emit(self, v):
        try:
            self._tested.emit(v)
        except RuntimeError:
            pass

    def _on_tested(self, v):
        self.test.setEnabled(True)
        kind, val = v
        if kind == 'ok':
            col = THEME.color('good').name()
            self.status.setText(f'<span style="color:{col}">●</span> Connected · {len(val)} chat models available')
            if val:
                self.s.setValue('assistant/models', val)
                self._fill_models(val)
        else:
            col = THEME.color('critical').name()
            self.status.setText(f'<span style="color:{col}">●</span> {html.escape(val)}')

    def apply(self):
        self.s.setValue('assistant/enabled', self.on.isChecked())
        self.s.setValue('assistant/model', self.model.currentData() or C.DEFAULT_MODEL)
        self.s.setValue('assistant/effort', self.effort.value())
        self.s.setValue('assistant/show_reasoning', self.show_r.isChecked())
        if self._new_key == '':
            K.delete()
        elif self._new_key:
            K.save(self._new_key)
