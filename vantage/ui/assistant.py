"""The Assistant dock: a chat about the open design with a cloud model (Groq or Claude, chosen in Preferences ▸
Assistant), and the Copilot: plans for runs that the user approves before anything runs.

The model reads the design only through vantage.assistant.tools (text summaries, and with Claude the pictures the look
tool draws from the analysis; never the vehicle model file, photos, recordings, point clouds or files) and can only
*propose*: camera changes (each shows its predicted effect, and the
user applies it as one undo step or discards it) and plans (the plan card shows every value with where it came from,
the warnings and the estimate; Approve and run writes an approval record into the project and runs each step through
the page's own Run, which refuses a plan without its record). The API keys are set in Preferences ▸ Assistant."""
from __future__ import annotations

import copy
import hashlib
import html
import json
import math
import os
import re
import threading
import time

from PyQt6 import sip
from PyQt6.QtCore import QObject, QPointF, QRectF, QSize, Qt, QTimer, QUrl, pyqtSignal
from PyQt6.QtGui import QColor, QDesktopServices, QFont, QFontMetricsF, QPainter, QPen, QTextOption
from PyQt6.QtWidgets import (QApplication, QCheckBox, QComboBox, QDialog, QFileDialog, QFrame, QGridLayout,
                             QHBoxLayout, QLabel, QLayout, QLineEdit, QMenu, QPlainTextEdit, QPushButton, QScrollArea,
                             QSizePolicy, QTextBrowser, QToolButton, QVBoxLayout, QWidget)

from .. import workdir
from ..assistant import agent as A
from ..assistant import client as C
from ..assistant import copilot as CP
from ..assistant import memory as M
from ..assistant import providers as PR
from ..assistant import recommend as RC
from ..assistant import routing as RT
from ..assistant import secrets as K
from ..assistant import tools as T
from . import icons
from .theme import THEME, mono, sans
from .widgets import ScrubSpinBox, SegmentedControl, button, label, tool_button

SUGGESTIONS = [
    ('Summarise this design', 'Summarise this design: coverage, weak spots and what to look at first.'),
    ('Why is any ground blind?', 'Where is ground blind or below the minimum density, and why? Name the cameras.'),
    ('Which camera matters most?', 'Which camera matters most, and what happens if it fails or is mounted badly?'),
    ('Where can a child go unseen?', 'Where could a 0.9 m child stand next to the car without any camera seeing '
                                     'them, and how could the rig be improved?'),
    ('Explain the issues', 'Explain each item in the Issues list and suggest a fix for each.'),
    ('Write a design review', 'Write a short design review of this camera rig for a colleague: strengths, risks, '
                              'and recommended changes with their measured effect.'),
    ('Plan a CARLA drive', 'Plan a 10 s CARLA drive in Town10 at dusk in light rain with 30 cars and ground truth on, '
                           'for my approval.'),
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
    ('/plan', 'Plan a run for your approval, e.g. /plan a 10 s drive in Town10 in rain', '/plan '),
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
    if cmd == '/plan':
        return (f'Plan this for my approval with propose_plan: {rest}' if rest else ''), False, None
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


KIND_WORD = {'decision': 'decision:', 'rejected': '', 'constraint': 'constraint:', 'note': 'note:'}

PAGES = {'viewports': '3D and top views', 'camviews': 'Camera views', 'tolerance': 'Tolerance study',
         'optimiser': 'Optimiser', 'compare': 'Compare variants', 'photo': 'Photo matching', 'targets': 'Targets',
         'install': 'Installation check', 'volume': 'Blind volume', 'perception': 'Perception range',
         'rig': 'Live rig', 'coverage': 'Lidar and radar coverage', 'timing': 'Sync and timing',
         'clock_tree': 'Clock tree and time sync', 'isp': 'Exposure metering and HDR seams',
         'recording': 'Recording budget', 'power': 'Power budget',
         'gnss': 'GNSS and IMU', 'exposure': 'Sun glare and dirt',
         'fleet': 'Fleet', 'brackets': 'Mount brackets',
         'dataset': 'Sample dataset', 'drivelog': 'Drive-log check',
         'requirements': 'Requirements scorecard', 'lidar_calib': 'Lidar–camera calibration',
         'asbuilt': 'As-built check', 'online_calib': 'Online self-calibration', 'critical': 'Critical targets',
         'materials': 'Materials, glass and wipers',
         'automation': 'Automation', 'terrain': 'Road and body motion', 'thermo_vib': 'Thermal and vibration drift',
         'harness': 'Cable harness',
         'observability': 'Calibration observability', 'experiments': 'Experiments', 'sim': 'CARLA simulator',
         'detection': 'Detection quality', 'campaign': 'Simulation campaigns',
         'critical_carla': 'Critical targets in CARLA', 'splat': 'Log replay'}


# ============================================================================================ controller
class AssistantController(QObject):
    """Settings, the conversation and its worker thread, the proposals (applied on the UI thread) and the Copilot's
    plans (approved, recorded and run on the UI thread)."""
    event = pyqtSignal(object)          # agent.Event, on the UI thread, for the current conversation only
    changed = pyqtSignal()              # set-up, model or busy state
    reset = pyqtSignal()                # a new conversation started
    proposal_state = pyqtSignal(int)    # a proposal was applied, discarded, undone or measured
    plan_state = pyqtSignal(str)        # a plan was edited, approved, rejected, or one of its steps moved on
    agent_changed = pyqtSignal(bool)
    log_changed = pyqtSignal()          # the design log or a proposed entry changed
    cta = pyqtSignal(object)            # a recommendation was clicked (a card's or a turn's chip): the panel acts on it
    _raw = pyqtSignal(object)           # (thread id, Event) from the worker
    _ui_req = pyqtSignal(object)        # (thread id, request) from a show_in_app call

    def __init__(self, ctx):
        super().__init__()
        self.ctx = ctx
        self.s = ctx.settings
        self.keys = {p: K.load(p) for p in PR.PROVIDERS}
        self.session: A.BaseSession | None = None
        self.busy = False
        self.status = ''
        self.proposals: dict = {}       # id -> {'p': Proposal, 'state': str, 'undo_index': int}
        self.log_proposals: dict = {}   # entry id -> {'entry': dict, 'state': 'pending' | 'accepted' | 'discarded'}
        self.plans: dict = {}           # plan id -> {'plan': the latest copilot.Plan, 'versions': {n: Plan}}
        self.current_plan: CP.Plan | None = None    # the plan on screen a follow-up revises
        self.runner: PlanRunner | None = None
        self.last_question = ''
        self._tid = 0
        self._cancel = None
        self._mark = 0
        self._started = 0.0
        self.agent_mode = False
        self.agent_running = False
        self.last_route: RT.Route | None = None
        self._pending_handover = ''
        self._retired = []              # Usage of conversations handed over to the other provider (still counted)
        self._retired_md = []           # ... and their transcripts (Save conversation keeps the whole of it)
        self._said = set()              # fallback notes already shown once
        self.recommender = None
        self._raw.connect(self._on_raw)
        self._ui_req.connect(self._do_ui)
        ctx.store.project_replaced.connect(self.new_chat)

    # ---------------------------------------------------------------- settings
    @property
    def provider(self) -> str:
        v = str(self.s.value('assistant/provider', PR.GROQ) or PR.GROQ)
        return v if v in PR.PROVIDERS else PR.GROQ

    @property
    def key(self) -> str | None:
        return self.keys.get(self.provider, (None, None))[0]

    @property
    def key_source(self) -> str | None:
        return self.keys.get(self.provider, (None, None))[1]

    @property
    def enabled(self) -> bool:
        return self.s.value('assistant/enabled', False, type=bool)

    @property
    def available(self) -> bool:
        """The provider can be used here (Claude needs the anthropic SDK)."""
        return self.provider != PR.CLAUDE or PR.claude_available()

    @property
    def ready(self) -> bool:
        if self.auto:
            return self.enabled and any(RT.usable(self._keys(), PR.claude_available()).values())
        return self.enabled and bool(self.key) and self.available

    # ---------------------------------------------------------------- automatic model choice
    def _keys(self) -> dict:
        return {p: self.keys.get(p, (None, None))[0] for p in PR.PROVIDERS}

    @property
    def auto(self) -> bool:
        """Automatic model choice (Preferences ▸ Assistant): off for existing users unless they turn it on; a new
        installation starts with it on when both keys exist."""
        if self.s.contains('assistant/auto'):
            return self.s.value('assistant/auto', False, type=bool)
        fresh = not self.s.contains('assistant/provider')
        return RT.default_auto(fresh, self._keys())

    def route_for(self, text: str, agent: bool | None = None, force: RT.Route | None = None) -> RT.Route | None:
        """Who answers this request: the forced route (a "re-ask with …" chip), the automatic choice, or the provider
        and model chosen in Preferences."""
        if force is not None:
            return force
        agent = self.agent_mode if agent is None else agent
        planning = CP.is_plan_request(text) or (self.current_plan is not None and self.current_plan.status == 'draft')
        if self.auto:
            return RT.route(text, agent, planning, self._keys(), PR.claude_available())
        tier, reason = RT.classify(text, agent, planning)
        note = 'Pictures need Claude: on Groq the assistant answers from the figures alone.' \
            if reason == 'image' and self.provider == PR.GROQ else ''
        return RT.Route(self.provider, self.model, tier, 'manual', note=note)

    def alternative(self, r: RT.Route | None) -> RT.Route | None:
        if r is None or not self.auto and r.reason != 'reask':
            return None
        return RT.alternative(r, self._keys(), PR.claude_available())

    @property
    def model(self) -> str:
        if self.provider == PR.CLAUDE:
            m = str(self.s.value('assistant/claude_model', PR.CLAUDE_DEFAULT) or PR.CLAUDE_DEFAULT)
            return m if m in PR.CLAUDE_MODELS else PR.CLAUDE_DEFAULT
        return str(self.s.value('assistant/model', C.DEFAULT_MODEL) or C.DEFAULT_MODEL)

    @property
    def effort(self) -> str:
        return str(self.s.value('assistant/effort', 'medium') or 'medium')

    @property
    def show_reasoning(self) -> bool:
        return self.s.value('assistant/show_reasoning', True, type=bool)

    @property
    def fallbacks(self) -> bool:
        return self.s.value('assistant/claude_fallbacks', True, type=bool)

    @property
    def daily_cap(self) -> int:
        try:
            return max(0, int(self.s.value('assistant/daily_cap', 0) or 0))
        except (TypeError, ValueError):
            return 0

    def groq_prices(self) -> dict | None:
        try:
            i = float(self.s.value('assistant/groq_price_in', 0) or 0)
            o = float(self.s.value('assistant/groq_price_out', 0) or 0)
        except (TypeError, ValueError):
            return None
        return {'input': i, 'output': o} if i > 0 and o > 0 else None

    def info(self, model=None):
        return PR.model_info(self.provider, model or self.model)

    def models(self) -> list:
        if self.provider == PR.CLAUDE:
            return list(PR.CLAUDE_MODELS)
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
            if k == 'model' and self.provider == PR.CLAUDE:
                k = 'claude_model'
            self.s.setValue(f'assistant/{k}', v)
        if self.session is not None and not self.auto:
            if self.session.provider != self.provider:
                self.new_chat()
            else:
                self.session.settings = self._settings()
        self.changed.emit()

    def reload(self):
        """After Preferences: re-read the keys, the provider and the model (another provider: a new conversation)."""
        self.keys = {p: K.load(p) for p in PR.PROVIDERS}
        if self.session is not None:
            prov = self.session.provider
            if self.session.provider != self.provider and not self.auto:
                self.new_chat()
            elif prov == PR.GROQ:
                self.session.client = C.Client(self.keys.get(prov, (None, None))[0] or '')
                if not self.auto:
                    self.session.settings = self._settings()
            else:
                self.session = None          # a new client with the new key; the history stays with the old one
        self.changed.emit()

    def _settings(self, agent=False, planning=False, model=None) -> A.Settings:
        return A.Settings(model or self.model, self.effort, self.show_reasoning, agent=agent, planning=planning,
                          fallbacks=self.fallbacks)

    def set_agent(self, on: bool):
        if on != self.agent_mode:
            self.agent_mode = bool(on)
            self.agent_changed.emit(self.agent_mode)

    # ---------------------------------------------------------------- consent and the daily cap
    def consented(self, provider=None) -> bool:
        return self.s.value(f'assistant/consent_{provider or self.provider}', False, type=bool)

    def give_consent(self, provider=None):
        self.s.setValue(f'assistant/consent_{provider or self.provider}', True)
        self.changed.emit()

    def tokens_today(self, provider=None) -> int:
        day = time.strftime('%Y-%m-%d')
        if str(self.s.value('assistant/day', '')) != day:
            return 0
        if provider is None:
            return sum(self.tokens_today(p) for p in PR.PROVIDERS)
        try:
            return int(self.s.value(f'assistant/day_tokens_{provider}', 0) or 0)
        except (TypeError, ValueError):
            return 0

    def _count_tokens(self, provider, n):
        day = time.strftime('%Y-%m-%d')
        if str(self.s.value('assistant/day', '')) != day:
            self.s.setValue('assistant/day', day)
            for p in PR.PROVIDERS:
                self.s.setValue(f'assistant/day_tokens_{p}', 0)
        self.s.setValue(f'assistant/day_tokens_{provider}', self.tokens_today(provider) + int(n))

    def blocked(self, provider=None) -> str:
        """Why a question cannot be sent now ('' when it can): consent not given (to the provider that would answer
        it), the daily cap reached."""
        if not self.consented(provider):
            return 'consent'
        cap = self.daily_cap
        if cap and self.tokens_today() >= cap:
            return 'cap'
        return ''

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
    def ask(self, text: str, agent: bool | None = None, force: RT.Route | None = None) -> bool:
        """Send one request. The route (automatic, forced or the Preferences choice) picks the provider and model;
        consent and the daily cap are checked for that provider. A request for the other provider than the
        conversation's starts a new conversation with a hand-over summary (the two history formats never mix)."""
        text = text.strip()
        if not text or self.busy or not self.ready:
            return False
        agent = self.agent_mode if agent is None else agent
        r = self.route_for(text, agent, force)
        if r is None or self.blocked(r.provider):
            return False
        planning = CP.is_plan_request(text) or self.current_plan is not None and self.current_plan.status == 'draft'
        key = self.keys.get(r.provider, (None, None))[0]
        old = self.session
        if old is None or old.provider != r.provider:
            try:
                new = A.new_session(r.provider, key, self._settings(agent, planning, r.model))
            except C.ApiError as e:
                self._fail_now(e)
                return False
            if old is not None:
                md = old.transcript_markdown()
                if md.strip():
                    new.handover = RT.handover_text(md, old.provider)
                    self._retired_md.append(md)
                self._retired.append(old.usage)
            self.session = new
        self.session.settings = self._settings(agent, planning, r.model)
        announce = bool(r.fallback) and r.fallback not in self._said
        self._said.add(r.fallback)
        self.last_route = r
        self._pending_handover = self.session.handover
        self.event.emit(A.Event('route', {**r.to_dict(), 'announce': announce, 'auto': self.auto,
                                          'handover': bool(self.session.handover)}))
        self.agent_running = agent
        tc = self._context()
        tc.provider, tc.model = r.provider, r.model
        tc.agent = agent
        tc.request = text
        tid_next = self._tid + 1
        tc.on_ui = lambda req: self._post_ui(tid_next, req)
        self.last_question = text
        self._mark = len(self.session.messages)
        self._cancel = threading.Event()
        self._tid += 1
        tid, session, cancel = self._tid, self.session, self._cancel
        self.busy, self.status, self._started = True, '', time.perf_counter()
        self._tokens0 = session.usage.tokens
        self.changed.emit()

        def work():
            session.run_turn(text, tc, lambda ev: self._post(tid, ev), cancel)
        threading.Thread(target=work, daemon=True, name='assistant').start()
        return True

    def _fail_now(self, e):
        self.event.emit(A.Event('error', {'kind': e.kind, 'message': e.message}))
        self.event.emit(A.Event('done', {'ok': False}))

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
        elif ev.kind == 'copilot':
            self._add_plan(ev.data['plan'])
        elif ev.kind == 'log_proposal':
            self.log_proposals[ev.data['entry']['id']] = {'entry': ev.data['entry'], 'state': 'pending'}
        elif ev.kind == 'status':
            self.status = ev.data['text']
        elif ev.kind == 'done':
            self.busy, self.status = False, ''
            ev.data['seconds'] = time.perf_counter() - self._started
            if self.session is not None:
                self._count_tokens(self.session.provider, max(0, self.session.usage.tokens -
                                                              getattr(self, '_tokens0', 0)))
        self.event.emit(ev)
        if ev.kind == 'done':
            self.changed.emit()

    def stop(self):
        """Stop at once: the conversation continues from before this question (the old thread winds down alone)."""
        if not self.busy:
            return
        self._cancel.set()
        old = self.session
        try:
            fresh = A.new_session(old.provider, self.keys.get(old.provider, (None, None))[0],
                                  self._settings(model=old.settings.model))
        except C.ApiError:
            fresh = None
        if fresh is not None:
            fresh.messages = list(old.messages[:self._mark])
            fresh.usage = old.usage
            fresh.handover = self._pending_handover or old.handover     # the stopped question carried it
        self.session = fresh
        self._tid += 1
        self.busy, self.status = False, ''
        self.event.emit(A.Event('done', {'ok': False, 'stopped': True, 'usage': old.usage,
                                         'seconds': time.perf_counter() - self._started}))
        self.changed.emit()

    def new_chat(self, *_):
        if self.busy:
            self._cancel.set()
            self._tid += 1
            self.busy = False
        self.session = None
        self._retired = []
        self._retired_md = []
        self.last_route = None
        self.proposals.clear()
        self.log_proposals.clear()
        self.plans.clear()
        self.current_plan = None
        self.last_question = ''
        self.reset.emit()
        self.changed.emit()

    def transcript(self) -> str:
        cur = self.session.transcript_markdown() if self.session is not None else ''
        return '\n'.join([t for t in self._retired_md if t.strip()] + ([cur] if cur.strip() else []))

    def usage(self):
        return self.session.usage if self.session is not None else None

    def usages(self) -> dict:
        """provider -> Usage of this conversation, the parts handed over to the other provider included."""
        out = {}
        for u in self._retired + ([self.session.usage] if self.session is not None else []):
            if u is None or not u.requests:
                continue
            m = out.get(u.provider)
            if m is None:
                m = out[u.provider] = PR.Usage(provider=u.provider, model=u.model)
            for f in ('prompt', 'completion', 'reasoning', 'cache_read', 'cache_write', 'requests'):
                setattr(m, f, getattr(m, f) + getattr(u, f))
            m.seconds += u.seconds
            for k, row in u.by_model.items():
                cur = m.by_model.setdefault(k, [0, 0, 0, 0])
                for i in range(4):
                    cur[i] += row[i]
        return out

    def usage_line(self) -> str:
        us = self.usages()
        if not us:
            return ''
        if len(us) == 1:
            return next(iter(us.values())).line(self.groq_prices())
        return ' · '.join(f'{PR.LABEL[p]}: {u.line(self.groq_prices())}' for p, u in us.items())

    def last_sent(self) -> str:
        """The last request's payload as it was sent, masked (keys are never in it; machine paths shown as ~)."""
        s = self.session
        if s is None or s.last_payload is None:
            return ''
        return A.mask(json.dumps(s.last_payload, indent=1, ensure_ascii=False, default=str))

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
                             detection=self._page_summary('detection'), campaign=self._page_summary('campaign'),
                             critical_carla=self._page_summary('critical_carla'), carla_live=self._carla_live(),
                             splat=self._page_summary('splat'), online_calib=self._page_summary('online_calib'),
                             working_dir=workdir.get(self.ctx.settings), plan=self.current_plan,
                             provider=self.provider, model=self.model)

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

    def _page_summary(self, name):
        """A page's last result in brief (its summary(): no paths), or None when the page was never opened."""
        w = self.ctx.window
        pg = getattr(w, '_page_widgets', {}).get(name) if w is not None else None
        return pg.summary() if pg is not None and hasattr(pg, 'summary') else None

    def _carla_live(self):
        w = self.ctx.window
        pg = getattr(w, '_page_widgets', {}).get('sim') if w is not None else None
        live = getattr(pg, 'live', None)
        return live.summary() if live is not None else None

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
        from ..core.project import MAX_CAMERAS
        adds = [u for u, props in prop.targets.items() if '__add__' in props and p.variant.cam(u) is None]
        if adds and len(p.variant.cameras) + len(adds) > MAX_CAMERAS:
            return 'full'
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
        if why in ('missing', 'locked', 'full') or (why == 'stale' and not force):
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
                from ..core.project import Camera
                for uid, props in prop.targets.items():
                    if '__add__' in props:          # a new camera (add_camera_options)
                        if p.variant.cam(uid) is None:
                            p.variant.cameras.append(Camera.from_dict(props['__add__']))
                        continue
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
            from ..core.project import Camera
            v = Variant.from_dict(cur.to_dict())
            v.uid = new_uid()
            v.name = f'{cur.name} · {prop.title}'[:64]
            for uid, props in prop.targets.items():     # new cameras go in with their own (remapped) uids
                if '__add__' in props and v.cam(uid) is None:
                    v.cameras.append(Camera.from_dict(props['__add__']))
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
                if c is not None and '__add__' not in props:
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
        """Discard a proposal; the design log records it as rejected (one undo step), so the assistant does not offer
        the same change again."""
        if pid in self.proposals:
            e = self.proposals[pid]
            e['state'] = 'discarded'
            entry = M.rejected_entry(e['p'])
            if entry is not None:
                known = set(M.tag_to_uid(self.ctx.store.project).values())
                entry['about'] = [u for u in entry['about'] if u in known]     # a camera it would have added: none
                self.ctx.store.edit('analysis', lambda p: M.add(p, entry), f'Design log · {entry["text"][:60]}')
                e['log_id'] = entry['id']
            self.proposal_state.emit(pid)

    # ---------------------------------------------------------------- the design log (kept in the project)
    def log(self) -> list:
        return M.entries(self.ctx.store.project)

    def add_log(self, kind, text, about=(), source='user', changes=None) -> dict | None:
        """Write an entry (one undo step); None when it is empty."""
        e = M.new_entry(kind, text, about, source, changes)
        if e is None:
            return None
        self.ctx.store.edit('analysis', lambda p: M.add(p, e), f'Design log · {KIND_WORD.get(e["kind"], "")} '
                                                               f'{e["text"][:50]}')
        self.log_changed.emit()
        return e

    def edit_log(self, eid, **fields) -> bool:
        if not any(e['id'] == eid for e in self.log()):
            return False
        ok = [False]

        def fn(p):
            ok[0] = M.update(p, eid, **fields)
        self.ctx.store.edit('analysis', fn, 'Design log · edit entry')
        self.log_changed.emit()
        return ok[0]

    def delete_log(self, eid) -> bool:
        if not any(e['id'] == eid for e in self.log()):
            return False
        self.ctx.store.edit('analysis', lambda p: M.remove(p, eid), 'Design log · delete entry')
        self.log_changed.emit()
        return True

    def accept_log(self, eid) -> bool:
        """The user accepts an entry the model proposed: written with source 'assistant' (proposed and accepted)."""
        lp = self.log_proposals.get(eid)
        if lp is None or lp['state'] != 'pending':
            return False
        e = dict(lp['entry'], source='assistant')
        self.ctx.store.edit('analysis', lambda p: M.add(p, e), f'Design log · {e["text"][:60]}')
        lp['state'] = 'accepted'
        self.log_changed.emit()
        return True

    def discard_log(self, eid):
        lp = self.log_proposals.get(eid)
        if lp is not None and lp['state'] == 'pending':
            lp['state'] = 'discarded'
            self.log_changed.emit()

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

    # ---------------------------------------------------------------- the Copilot's plans
    def _add_plan(self, plan: CP.Plan):
        e = self.plans.setdefault(plan.id, {'versions': {}})
        e['versions'][plan.version] = plan
        e['plan'] = plan
        if plan.parent is not None and plan.parent in e['versions']:
            old = e['versions'][plan.parent]
            if old.status == 'draft':
                old.status = 'superseded'
                self.plan_state.emit(plan.id)
        self.current_plan = plan

    def plan_of(self, pid, version=None) -> CP.Plan | None:
        e = self.plans.get(pid)
        if e is None:
            return None
        return e['versions'].get(version) if version is not None else e.get('plan')

    def _engine(self):
        try:
            return self.ctx.analysis.engine.share()
        except Exception:                                   # noqa: BLE001
            return None

    def edit_plan(self, plan: CP.Plan, step: int, key: str, value):
        """The user changed a value on the plan card: a new version (an approved plan never changes), checked by the
        page's own rules; (the new plan, '') or (None, why not)."""
        try:
            new = CP.edit_value(self.ctx.store.project.snapshot(), plan, step, key, value,
                                workdir.get(self.ctx.settings), self._engine())
        except (IndexError, KeyError, ValueError) as e:
            return None, str(e)
        new.provider, new.model = plan.provider, plan.model
        if plan.status == 'draft':
            plan.status = 'superseded'
        self._add_plan(new)
        self.plan_state.emit(plan.id)
        return new, ''

    def reject_plan(self, plan: CP.Plan):
        if plan.status == 'draft':
            plan.status = 'rejected'
            if self.current_plan is plan:
                self.current_plan = None
            self.plan_state.emit(plan.id)

    def _apply_step(self, rec_or_plan, i, text):
        """Write a step's settings into its page's slot of the project as one undo step (nothing runs)."""
        steps = rec_or_plan['steps'] if isinstance(rec_or_plan, dict) else [s.to_dict() for s in rec_or_plan.steps]
        s = steps[i]
        slot = CP.KIND_SLOT.get(s['kind'])
        if slot is None:
            return False
        settings = copy.deepcopy(s['settings'])

        def fn(p):
            setattr(p.analysis, slot, settings)
        self.ctx.store.edit('analysis', fn, text)
        return True

    def open_on_page(self, plan: CP.Plan, step: int = 0) -> bool:
        """Fill the step's page with its settings as one undo step and show the page; runs nothing."""
        if not 0 <= step < len(plan.steps):
            return False
        s = plan.steps[step]
        page = CP.KIND_PAGE.get(s.kind)
        self._apply_step(plan, step, f'Copilot · {plan.title} · on its page')
        w = self.ctx.window
        if page and w is not None:
            w.set_page(page)
            if s.kind == 'carla_live':
                live = getattr(w.page('sim'), 'live', None)
                if live is not None:
                    live.set_cameras(s.settings.get('cameras') or [])
                    live.scale.set_value(100.0 * float(s.settings.get('scale', CP.LIVE_SCALE)))
        self.ctx.msg(f'{CP.KIND_LABEL[s.kind]} filled in from the plan (Edit ▸ Undo reverts it); nothing was run.', 6000)
        return True

    def approve_plan(self, plan: CP.Plan) -> str:
        """Approve and run: validated again now, the approval record written into the project (with the first step's
        settings, one undo step; undo never removes the record), then the steps run through their pages' own Run.
        '' when it started, else why not."""
        if self.runner is not None and self.runner.running:
            return 'Another plan is running: wait for it or stop it first.'
        p = self.ctx.store.project
        rec, errors = CP.approve(p.snapshot(), plan, plan.provider or self.provider, plan.model or self.model,
                                 workdir.get(self.ctx.settings), self._engine())
        if rec is None:
            return ' '.join(errors[:3])

        def fn(pj):
            CP.add_record(pj, rec)
        self.ctx.store.edit('analysis', fn, f'Copilot · approved plan {rec["id"]} v{rec["version"]}')
        plan.status = 'approved'
        self.plan_state.emit(plan.id)
        return self.run_record(rec['id'], rec['version'], via='app')

    def run_record(self, pid, version, via='rerun') -> str:
        """Run a plan from its approval record (the stored settings; the model is not asked)."""
        if self.runner is not None and self.runner.running:
            return 'Another plan is running: wait for it or stop it first.'
        rec = CP.find(self.ctx.store.project, pid, version)
        if rec is None or not CP.verify(rec):
            return 'No intact approval record of that plan in this project.'
        self.runner = PlanRunner(self, pid, int(version), via)
        self.runner.progress.connect(lambda pid=pid: self.plan_state.emit(pid))
        self.runner.finished.connect(lambda _st, pid=pid: self.plan_state.emit(pid))
        self.runner.start()
        return ''

    def stop_plan(self):
        if self.runner is not None and self.runner.running:
            self.runner.cancel()

    def records(self) -> list:
        return list(reversed(CP.records(self.ctx.store.project)))


# ============================================================================================ running a plan
class PlanRunner(QObject):
    """Runs an approved plan's steps one after another on the UI thread, each through its page's own Run with the
    plan's ticket (the page refuses it without the approval record), polling the page until it is done; the record's
    run and copilot_plan.json in the plan's folder say how each step went."""
    progress = pyqtSignal()
    finished = pyqtSignal(str)
    POLL_MS = 250
    SERVER_WAIT_S = 300.0

    def __init__(self, ctl, pid, version, via='app'):
        super().__init__()
        self.ctl, self.ctx = ctl, ctl.ctx
        self.id, self.version, self.via = pid, int(version), via
        self.i = -1
        self.running = False
        self.status = ''
        self._job = None
        self._phase = ''
        self._t0 = 0.0
        self._project = ctl.ctx.store.project
        self.folder = ''
        self.timer = QTimer(self)
        self.timer.setInterval(self.POLL_MS)
        self.timer.timeout.connect(self._tick)

    def rec(self):
        return CP.find(self._project, self.id, self.version)

    def start(self):
        rec = self.rec()
        self.running = True
        CP.start_run(rec, self.via)
        self.folder = CP.run_folder(self._project, rec, workdir.get(self.ctx.settings))
        self._write()
        self._next()

    def _write(self, step=None, summary=None):
        try:
            CP.write_run_file(self._project, self.rec(), self.folder, step, summary)
        except OSError as e:
            self.ctx.msg(f'The plan\'s record could not be written to its folder: {e}', 6000)

    def _step_run(self):
        return self.rec()['runs'][-1]['steps'][self.i]

    def _fail(self, why):
        st = self._step_run()
        st.update(status='failed', message=str(why)[:600])
        self._write(self.i)
        self._finish('failed')

    def _finish(self, status):
        self.timer.stop()
        self.running = False
        rec = self.rec()
        if rec is not None:
            for s in rec['runs'][-1]['steps']:
                if s['status'] in ('pending', 'running') and status != 'done':
                    s['status'] = 'not run' if s['status'] == 'pending' else status
            CP.finish_run(rec, status)
        self.status = status
        self._write()
        self.ctx.msg(f'Plan {self.id} v{self.version}: {status}', 6000)
        self.finished.emit(status)

    def cancel(self):
        if not self.running:
            return
        job = self._job or {}
        pg = job.get('page')
        try:
            if job.get('kind') == 'carla_live' and pg is not None:
                pg.stop('the plan was stopped')
            elif pg is not None and hasattr(pg, 'cancel'):
                pg.cancel()
        except Exception:                                   # noqa: BLE001 — stopping must not raise
            pass
        st = self._step_run()
        st.update(status='cancelled', message='stopped by you')
        self._finish('cancelled')

    def _next(self):
        rec = self.rec()
        if rec is None:
            self.running = False
            self.finished.emit('failed')
            return
        self.i += 1
        if self.i >= len(rec['steps']):
            self._finish('done')
            return
        step = rec['steps'][self.i]
        self._step_run().update(status='running')
        self.progress.emit()
        p = self.ctx.store.project
        if p is not self._project:
            self._fail('The project was replaced while the plan ran.')
            return
        # checked again now: the project, the server, the memory or the model may have changed since approval
        again = CP.revalidate(p.snapshot(), CP.Plan(rec['id'], rec['version'], rec['title'],
                                                    steps=[CP.Step.from_dict(step)]),
                              workdir.get(self.ctx.settings), self.ctl._engine())
        if not again.ok:
            self._fail('It does not validate now: ' + ' '.join(again.steps[0].errors[:3]))
            return
        self.ctl._apply_step(rec, self.i, f'Copilot · {rec["title"]} · step {self.i + 1}')
        tk = CP.ticket(rec, self.i)
        try:
            job = self._make_job(step['kind'], step['settings'], tk)
        except Exception as e:                              # noqa: BLE001 — a step that cannot start says why
            self._fail(f'{type(e).__name__}: {e}')
            return
        self._job = job
        self._phase = 'server' if job.get('port') else 'start'
        self._t0 = time.monotonic()
        if job.get('port'):
            from ..sim import server as SV
            if SV.port_open('127.0.0.1', job['port']):
                self._phase = 'start'
            else:
                sim = self.ctx.window.page('sim')
                if not sim.start_server():
                    self._fail(f'No CARLA server answers on port {job["port"]} and the CARLA page could not start '
                               'one: choose the CARLA folder on Simulate ▸ CARLA.')
                    return
        self._tick()
        if self.running and self._job is job:
            self.timer.start()

    def _tick(self):
        job = self._job
        if job is None or not self.running:
            return
        try:
            if self._phase == 'server':
                from ..sim import server as SV
                if SV.port_open('127.0.0.1', job['port']):
                    self._phase = 'start'
                elif time.monotonic() - self._t0 > self.SERVER_WAIT_S:
                    self._fail(f'The CARLA server did not answer on port {job["port"]} within '
                               f'{self.SERVER_WAIT_S:.0f} s.')
                    return
                else:
                    return
            if self._phase == 'start':
                ok, why = job['start']()
                if not ok:
                    self._fail(why or 'The page refused to run it.')
                    return
                self._phase = 'running'
                self._t0 = time.monotonic()
                return
            if self._phase == 'running':
                if not job['done']():
                    return
                ok, message, summary, out, files = job['result']()
                st = self._step_run()
                st.update(status='done' if ok else 'failed', message=str(message or '')[:600],
                          output=workdir.display(out, workdir.get(self.ctx.settings)) if out else '',
                          summary=self._scrub(summary))
                rec = self.rec()
                if out and os.path.isdir(out):
                    CP.stamp_outputs(self._project, rec, self.i, files, out)
                    try:
                        CP.write_run_file(self._project, rec, out, self.i, summary)
                    except OSError:
                        pass
                self._write(self.i, summary)
                self.progress.emit()
                self._job = None
                self.timer.stop()
                if not ok:
                    self._finish('failed')
                    return
                self._next()
        except Exception as e:                              # noqa: BLE001 — a failed step is said, never silent
            self._fail(f'{type(e).__name__}: {e}')

    @staticmethod
    def _scrub(summary):
        from ..core import provenance as PV
        try:
            return json.loads(json.dumps(PV.scrub(summary), default=str)) if summary is not None else None
        except (TypeError, ValueError):
            return None

    # ---------------------------------------------------------------- each kind: through its page's own Run
    def _make_job(self, kind, settings, tk) -> dict:
        w = self.ctx.window
        p = self.ctx.store.project
        if kind == 'carla_run':
            pg = w.page('sim')

            def start():
                ok = pg.run(plan=tk)
                return ok, '' if ok else pg.g_status.text()

            def result():
                doc = pg.result or {}
                out = pg.out_dir
                ok = doc.get('status') == 'ok'
                summ = {k: doc.get(k) for k in ('status', 'performance', 'scene', 'scenario', 'faults_injected',
                                                 'notes', 'error') if k in doc}
                files = [os.path.join(out, n) for n in ('run.json', 'dataset.json')] if out else []
                return ok, pg.g_status.text(), summ, out, files
            return {'kind': kind, 'page': pg, 'port': SR_port(settings), 'start': start, 'done': lambda: not pg.busy,
                    'result': result}
        if kind in ('campaign', 'detection', 'critical_carla', 'experiment'):
            name = CP.KIND_PAGE[kind]
            pg = w.page(name)
            before = pg.result
            mod = {'campaign': 'simcampaign', 'detection': 'detection_eval', 'critical_carla': 'critical_carla',
                   'experiment': 'experiments'}[kind]

            def start():
                ok = pg.run(confirm=False, plan=tk) if kind == 'experiment' else pg.run(plan=tk)
                return ok, '' if ok else (getattr(pg, 'status_text', '') or 'The page did not start it.')

            def result():
                import importlib
                M = importlib.import_module(f'..core.{mod}', __package__)
                res = pg.result
                if res is None or res is before:
                    return False, getattr(pg, 'status_text', '') or 'It did not finish.', None, '', []
                ok = bool(getattr(res, 'complete', True))
                out = ''
                if kind == 'campaign':
                    from ..core import simcampaign as SC
                    out = SC.campaign_dir(p, SC.settings_of(p), workdir.get(self.ctx.settings))
                return ok, getattr(pg, 'status_text', ''), M.summary(res), out, []
            port = None
            if kind == 'critical_carla' and pg.runner is None:
                port = SR_port(None, p)
            if kind == 'campaign' and settings.get('server', 'running') == 'running' and pg.runner is None:
                port = SR_port((settings.get('base') or {}), p)
            return {'kind': kind, 'page': pg, 'port': port, 'start': start, 'done': lambda: not pg.busy,
                    'result': result}
        if kind == 'sample_dataset':
            pg = w.page('dataset')

            def start():
                ok = pg.generate(plan=tk)
                return ok, '' if ok else pg.g_status.text()

            def result():
                from ..core import dataset as DSET
                res = pg.result
                if res is None:
                    return False, pg.g_status.text(), None, '', []
                return True, pg.g_status.text(), DSET.summary(res), res.out_dir, [os.path.join(res.out_dir,
                                                                                            'dataset.json')]
            return {'kind': kind, 'page': pg, 'start': start, 'done': lambda: not pg.busy, 'result': result}
        if kind == 'carla_live':
            pg = w.page('sim').live
            job = {'kind': kind, 'page': pg, 'faults': list(settings.get('faults') or []), 'streaming_at': None,
                   'sent': [], 'replies': []}

            def start():
                pg.set_cameras(settings.get('cameras') or [])
                pg.scale.set_value(100.0 * float(settings.get('scale', CP.LIVE_SCALE)))
                ok = pg.start(plan=tk)
                return ok, '' if ok else (pg.status_line.text.text() or 'The live rig did not start.')

            def done():
                if pg.state == 'failed' or (pg.state == 'off' and job['streaming_at'] is None and
                                            time.monotonic() - self._t0 > 2.0):
                    return True
                if pg.state == 'streaming' and job['streaming_at'] is None:
                    job['streaming_at'] = time.monotonic()
                if job['streaming_at'] is None:
                    return False
                t = time.monotonic() - job['streaming_at']
                for f in job['faults']:
                    if f not in job['sent'] and t >= float(f['at_s']):
                        job['sent'].append(f)
                        pg.inject(dict(f['fault']))
                dur = float(settings.get('duration_s') or 0.0)
                if dur and t >= dur:
                    if pg.running:
                        pg.stop('the plan\'s duration is over')
                    return not pg.running
                return len(job['sent']) == len(job['faults']) and not dur

            def result():
                ok = job['streaming_at'] is not None and pg.state != 'failed'
                summ = pg.summary() if hasattr(pg, 'summary') else None
                msg = (f'streamed {", ".join(settings.get("cameras") or [])}; {len(job["sent"])} fault'
                       + ('s' if len(job['sent']) != 1 else '') + ' injected on time'
                       + ('' if settings.get('duration_s') else '; still streaming (stop it on Simulate ▸ CARLA)'))
                return ok, msg if ok else (pg.message or 'the live rig did not start'), summ, pg.folder, []
            job.update(start=start, done=done, result=result)
            return job
        if kind == 'study':
            ok, why = CP.gate(p, tk, 'study', settings)
            if not ok:
                raise RuntimeError(why)
            box = {}
            tc = self.ctl._context()
            args = {k: v for k, v in settings.items() if v not in (None, '')}

            def work():
                try:
                    box['out'] = T.run_study(tc, **args)
                except Exception as e:                      # noqa: BLE001
                    box['out'] = {'error': f'{type(e).__name__}: {e}'}

            def start():
                threading.Thread(target=work, daemon=True, name='copilot-study').start()
                return True, ''

            def result():
                out = box.get('out') or {}
                return 'error' not in out, out.get('error', f'{settings["study"]} computed'), out, '', []
            return {'kind': kind, 'start': start, 'done': lambda: 'out' in box, 'result': result}
        if kind == 'export':
            ok, why = CP.gate(p, tk, 'export', settings)
            if not ok:
                raise RuntimeError(why)
            from ..core import exporters as ex
            base = workdir.get(self.ctx.settings)
            out = os.path.join(base, settings['folder']) if settings.get('folder') else \
                workdir.project_dir(p, base, 'exports', f'{self.id}-v{self.version}')
            box = {}

            def start():
                an = self.ctx.analysis
                opts = ex.ExportOptions(out_dir=out, frame=settings['frame'], units=settings['units'],
                                        angles=settings['angles'], issues=list(an.issues), engine=an.engine)
                try:
                    with an._lock:
                        box['paths'] = ex.export(settings['formats'], p, an.result, opts)
                except Exception as e:                      # noqa: BLE001 (shown)
                    return False, f'The export failed: {e}'
                p.add_log('export', f'{len(box["paths"])} files to {out} (Copilot plan {self.id})')
                return True, ''

            def result():
                paths = box.get('paths') or []
                return bool(paths), f'{len(paths)} files written', {'files': [os.path.basename(x) for x in paths]}, \
                    out, [x for x in paths if x.endswith('.json')]
            return {'kind': kind, 'start': start, 'done': lambda: 'paths' in box, 'result': result}
        raise KeyError(kind)


def SR_port(settings=None, project=None) -> int:
    """The CARLA port a step uses: its own settings' (a CARLA drive), else the CARLA page's."""
    from ..core import simrig as SR
    base = SR.settings_of(project).to_dict() if project is not None else {}
    return int(SR.SimSettings.from_dict(dict(base, **(settings or {}))).port)


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
    s = re.sub(r'&lt;br\s*/?&gt;', '<br>', html.escape(s, quote=False), flags=re.I)   # models put <br> in table cells
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
    """One tool call: what it did, and on click exactly what the model asked and what was sent back to the model."""
    KIND_ICON = {'read': 'search', 'simulate': 'play', 'propose': 'sparkle', 'compute': 'tolerance', 'ui': 'frame',
                 'plan': 'check', 'copilot': 'sparkle', 'look': 'eye'}
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
        self.title.setTextFormat(Qt.TextFormat.RichText)     # always: an escaped label alone (no span) is not auto-detected
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
                                f'<span style="color:{mut}">Sent back to the model</span><br/>'
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
            cam = QLabel(_cam_dot(r['slot'], r['camera']) if kind in ('camera', 'add') else
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
        elif why == 'full':
            self.note.setText('The rig already has as many cameras as the recorder takes; remove one to add this.')
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


# ============================================================================================ the Copilot's plan card
SOURCE_LABEL = {'request': 'your request', 'default': 'default', 'copilot': 'chosen by the Copilot',
                'limit': 'forced by a limit', 'unused': 'not used'}
STATUS_LABEL = {'draft': 'Plan · needs your approval', 'approved': 'Approved', 'running': 'Running', 'done': 'Done',
                'failed': 'Failed', 'cancelled': 'Stopped', 'rejected': 'Rejected', 'superseded': 'Superseded'}


def _sections(kind):
    """[(section, [keys or key prefixes])] for the plan card's table (the CARLA page's own groups for a drive)."""
    if kind == 'carla_run':
        from .workspaces.sim_ws import FIELDS, TOGGLES
        groups = {}
        for g, key, *_ in FIELDS:
            groups.setdefault(g, []).append(key)
        for g, rows in TOGGLES.items():
            groups.setdefault(g, []).extend(k for k, *_ in rows)
        groups.setdefault('Cameras', []).extend(['camera_rate', 'exposure', 'channel_names'])
        groups.setdefault('Server', []).extend(['quality', 'timeout_s'])
        out = [('World', ['town', 'weather', 'vehicle'])]
        out += [('Scenario', groups.pop('Scenario', []) + ['scenario.', 'actors'])]
        out += [('Faults', ['faults']), ('Ground truth', ['ground_truth.'])]
        out += [(g, ks) for g, ks in groups.items()]
        return out
    return [('Settings', [])]


def _field_labels(kind) -> dict:
    """{flat key: (label, unit, scale)}: the page's own labels and units where it has them."""
    if kind == 'carla_run':
        from ..sim import carla_bridge as CB
        from .workspaces.sim_ws import FIELDS, GT_TOGGLES, SCENARIO_LABELS, TOGGLES
        out = {key: (label, unit, scale) for _g, key, label, unit, _d, _st, scale, *_ in FIELDS}
        for rows in TOGGLES.values():
            for key, label, *_ in rows:
                out[key] = (label, '', 1.0)
        for key, label, _tip in GT_TOGGLES:
            out[f'ground_truth.{key}'] = (label, '', 1.0)
        for _name, (_lab, _about, params) in CB.SCENARIOS.items():
            for k, spec in params.items():
                out.setdefault(f'scenario.params.{k}', (SCENARIO_LABELS.get(k, CP._human(k).capitalize()),
                                                        spec[4] if spec[0] == 'num' else '', 1.0))
        out.update({'town': ('Town', '', 1.0), 'weather': ('Weather', '', 1.0), 'vehicle': ('Vehicle', '', 1.0),
                    'scenario.name': ('Scenario', '', 1.0), 'faults': ('Faults', '', 1.0),
                    'actors': ('Actors', '', 1.0)})
        return out
    if kind == 'experiment':
        from .workspaces.experiments_ws import DESIGN_FIELDS
        out = {key: (label, '', 1.0) for key, label, *_ in DESIGN_FIELDS}
        out.update({'name': ('Name', '', 1.0), 'variant': ('Variant', '', 1.0), 'factors': ('Factors', '', 1.0),
                    'responses': ('Responses', '', 1.0), 'design': ('Design', '', 1.0)})
        return out
    if kind == 'carla_live':
        return {'cameras': ('Cameras', '', 1.0), 'scale': ('Render', '%', 100.0), 'duration_s': ('Duration', 's', 1.0),
                'faults': ('Faults', '', 1.0)}
    return {}


def _label_of(labels, key):
    return labels.get(key, (key.split('.')[-1].replace('_', ' ').capitalize(), '', 1.0))


def _fmt(v, unit='', scale=1.0, kind='', key='', project=None) -> str:
    """A value as the card shows it: in words (copilot.value_text), never raw JSON."""
    return CP.value_text(kind, key, v, project, unit, scale)


def change_lines(plan: CP.Plan, project=None) -> list:
    """One line per changed setting in words ('Traffic 30 → 20 vehicles'; for a list, what was added or removed)."""
    out = []
    many = len(plan.steps) > 1
    for d in plan.diff:
        pre = f'Step {d["step"] + 1} · ' if many else ''
        if 'added' in d:
            out.append(f'Step {d["step"] + 1} added: {CP.KIND_LABEL.get(d["added"], d["added"])}')
            continue
        if 'removed' in d:
            out.append(f'Step {d["step"] + 1} removed: {CP.KIND_LABEL.get(d["removed"], d["removed"])}')
            continue
        st = plan.steps[d['step']] if d['step'] < len(plan.steps) else None
        kind = st.kind if st is not None else ''
        lab, unit, scale = _label_of(_field_labels(kind), d['key'])
        u = CP.unused_of(d['key'], CP.unused(kind, st.settings, project)) if st is not None else None
        out.append(pre + CP.change_text(kind, d['key'], d['before'], d['after'], project, lab, unit, scale)
                   + (f' ({u[0]})' if u else ''))
    return out


def plan_rows(step: CP.Step, project=None) -> list:
    """[(section, key, label, value text, source, reason)] grouped by section, the page's own labels where it has
    them, every value in words. Settings the step does not use (another design's, a free drive's traffic in a
    scenario) come last, in their own section, with source 'unused' and why."""
    flat = CP.flatten(step.settings)
    if step.kind == 'carla_run':
        from ..sim import carla_bridge as CB
        sc = step.settings.get('scenario') or {}
        full = CB.clean_scenario(sc.get('name'), sc.get('params'))[0]['params']
        for k, v in full.items():
            flat.setdefault(f'scenario.params.{k}', v)
        flat.pop('scenario.params', None)
    labels = _field_labels(step.kind)
    secs = _sections(step.kind)
    used, rows = set(), []
    for sec, keys in secs:
        for k in sorted(flat, key=lambda x: (next((i for i, kk in enumerate(keys) if x == kk or (kk.endswith('.') and
                                                                                             x.startswith(kk))), 99), x)):
            if k in used or not any(k == kk or (kk.endswith('.') and k.startswith(kk)) for kk in keys):
                continue
            used.add(k)
            rows.append((sec, k, flat[k]))
    for k in flat:
        if k not in used:
            sec = k.split('.')[0].replace('_', ' ').capitalize() if '.' in k else 'Settings'
            rows.append((sec, k, flat[k]))
    out, later = [], []
    un = CP.unused(step.kind, step.settings, project)
    for sec, k, v in rows:
        lab, unit, scale = _label_of(labels, k)
        src = step.sources.get(k) or step.sources.get(k.split('.')[0]) or {'source': 'default', 'reason': ''}
        text = _fmt(v, unit, scale, step.kind, k, project)
        u = CP.unused_of(k, un)
        if u is not None:
            later.append((u[0][:1].upper() + u[0][1:], k, lab, text, 'unused', u[1]))
        else:
            out.append((sec, k, lab, text, src.get('source', 'default'), src.get('reason', '')))
    return out + sorted(later, key=lambda r: r[0])


class CopilotCard(QFrame):
    """A plan: the summary, the steps, every value with where it came from (editable in place, the page's own rules
    checking it), the warnings and the estimate, the changes from the previous version; Approve and run, Open on its
    page, Reject."""

    def __init__(self, ctl: AssistantController, plan: CP.Plan, parent=None):
        super().__init__(parent)
        self.ctl, self.plan = ctl, plan
        self.setObjectName('copilot')
        self.v = QVBoxLayout(self)
        self.v.setContentsMargins(12, 10, 12, 10)
        self.v.setSpacing(7)
        head = QHBoxLayout()
        head.setSpacing(8)
        ico = QLabel()
        ico.setPixmap(icons.pixmap('sparkle', THEME.tok['ink'], 14))
        head.addWidget(ico)
        self.title = QLabel('')
        self.title.setFont(sans(12, QFont.Weight.DemiBold))
        self.title.setWordWrap(True)
        head.addWidget(self.title, 1)
        self.badge = QLabel('')
        self.badge.setFont(sans(10, QFont.Weight.DemiBold))
        head.addWidget(self.badge, 0, Qt.AlignmentFlag.AlignTop)
        self.v.addLayout(head)
        self.body = QWidget()
        self.bv = QVBoxLayout(self.body)
        self.bv.setContentsMargins(0, 0, 0, 0)
        self.bv.setSpacing(6)
        self.v.addWidget(self.body)
        self.note = label('', small=True, wrap=True)
        self.note.hide()
        self.v.addWidget(self.note)
        b = QHBoxLayout()
        b.setSpacing(6)
        self.state_lbl = label('', muted=True, small=True, wrap=True)
        b.addWidget(self.state_lbl, 1)
        self.b_reject = button('Reject', 'ghost')
        self.b_reject.clicked.connect(self.reject)
        self.b_open = button('Open on its page', 'ghost', None, 'Fill the page\'s fields with this plan as one undo '
                                                             'step; nothing runs')
        self.b_open.clicked.connect(self.open_on_page)
        self.b_stop = button('Stop', 'ghost', 'stop')
        self.b_stop.clicked.connect(ctl.stop_plan)
        self.b_approve = button('Approve and run', 'primary', None, 'Check it again, record your approval in the '
                                                                     'project and run each step on its page')
        self.b_approve.clicked.connect(self.approve)
        for w in (self.b_reject, self.b_open, self.b_stop, self.b_approve):
            b.addWidget(w)
        self.v.addLayout(b)
        self.chips_box = QWidget()
        self.chips_lay = QVBoxLayout(self.chips_box)
        self.chips_lay.setContentsMargins(0, 0, 0, 0)
        self.chips_box.hide()
        self.v.addWidget(self.chips_box)
        self.chips: NextSteps | None = None
        self._chips_sig = None
        self.tables = []
        ctl.plan_state.connect(self._sync)
        self._restyle()
        self.set_plan(plan)

    # ---------------------------------------------------------------- contents
    def set_plan(self, plan: CP.Plan):
        self.plan = plan
        while self.bv.count():
            it = self.bv.takeAt(0)
            w = it.widget()
            if w is not None:
                w.hide()
                w.setParent(None)
                w.deleteLater()
        self.tables = []
        self.title.setText(html.escape(plan.title or 'Plan') + f'  <span style="color:{THEME.tok["muted"]}">'
                           f'{plan.id} · v{plan.version}</span>')
        if plan.rationale:
            r = label(plan.rationale, ink2=True, small=True, wrap=True)
            self.bv.addWidget(r)
        if plan.diff:
            lines = change_lines(plan, self._project())
            more = [f'… and {len(lines) - 14} more'] if len(lines) > 14 else []
            self.change_items = lines
            dl = QLabel(f'Changed from v{plan.parent}:' + ''.join(f'<br>· {html.escape(x)}' for x in lines[:14] + more))
            dl.setTextFormat(Qt.TextFormat.RichText)
            dl.setWordWrap(True)
            dl.setFont(sans(10))
            dl.setToolTip(json.dumps(plan.diff, indent=1, ensure_ascii=False, default=str)[:4000])
            dl.setObjectName('diff')
            self.diff_lbl = dl
            self.bv.addWidget(dl)
        else:
            self.diff_lbl = None
            self.change_items = []
        tok = THEME.tok
        for i, st in enumerate(plan.steps):
            h = QLabel(f'<b>{i + 1} · {html.escape(CP.KIND_LABEL.get(st.kind, st.kind))}</b>'
                       + (f' · {html.escape(st.label)}' if st.label and st.label != CP.KIND_LABEL.get(st.kind) else ''))
            h.setFont(sans(11))
            h.setWordWrap(True)
            self.bv.addWidget(h)
            if st.estimate.get('text'):
                e = label('Estimate: ' + st.estimate['text'] + (f' ({st.estimate["basis"]})' if st.estimate.get('basis')
                                                                 else ''), muted=True, small=True, wrap=True)
                self.bv.addWidget(e)
            for txt, sev in [(x, 'critical') for x in st.errors] + [(x, 'warning') for x in st.warnings]:
                col = THEME.color(sev).name()
                m = QLabel(f'<span style="color:{col}">{"✕" if sev == "critical" else "!"}</span> '
                           f'<span style="color:{tok["ink2"]}">{html.escape(txt)}</span>')
                m.setWordWrap(True)
                m.setFont(sans(10))
                self.bv.addWidget(m)
            for txt in st.notes[:6]:
                self.bv.addWidget(label(txt, muted=True, small=True, wrap=True))
            self.bv.addWidget(self._table(i, st))
        self._sync(plan.id)

    def _table(self, i, st):
        from PyQt6.QtWidgets import QAbstractItemView, QHeaderView, QTableWidget, QTableWidgetItem
        rows = plan_rows(st, self._project())
        flat = CP.flatten(st.settings)
        t = QTableWidget(0, 4)
        t.setHorizontalHeaderLabels(['Setting', 'Value', 'From', 'Why'])
        t.verticalHeader().hide()
        t.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        t.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        t.setWordWrap(False)
        t.setFont(sans(10))
        keys = []
        last = None
        tok = THEME.tok
        for sec, key, lab, val, src, why in rows:
            if sec != last:
                r = t.rowCount()
                t.insertRow(r)
                it = QTableWidgetItem(sec)
                it.setFont(sans(10, QFont.Weight.DemiBold))
                it.setFlags(Qt.ItemFlag.ItemIsEnabled)
                t.setItem(r, 0, it)
                t.setSpan(r, 0, 1, 4)
                keys.append(None)
                last = sec
            r = t.rowCount()
            t.insertRow(r)
            cells = (lab, val, SOURCE_LABEL.get(src, src), why)
            raw = CP.get_path(st.settings, key, flat.get(key))
            for c, text in enumerate(cells):
                it = QTableWidgetItem(text)
                if c == 1 and isinstance(raw, (list, dict)):          # in words, then as the settings hold it
                    it.setToolTip(f'{key}: {text}\n\n' + json.dumps(raw, indent=1, ensure_ascii=False,
                                                                     default=str)[:4000])
                else:
                    it.setToolTip(f'{key}: {text}' if c < 2 else text)
                if src == 'unused':                                  # greyed: not a decision of the plan
                    it.setForeground(QColor(tok['muted']))
                elif c == 2 and src != 'default':
                    it.setForeground(QColor(THEME.color('warning' if src == 'limit' else 'good').name()))
                if c == 1:
                    it.setFont(mono(10))
                t.setItem(r, c, it)
            keys.append(key)
        t.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeMode.ResizeToContents)
        t.horizontalHeader().setSectionResizeMode(1, QHeaderView.ResizeMode.Interactive)
        t.horizontalHeader().setSectionResizeMode(2, QHeaderView.ResizeMode.ResizeToContents)
        t.horizontalHeader().setStretchLastSection(True)
        t.resizeColumnToContents(1)                 # values in words: as wide as they need, within reason
        t.setColumnWidth(1, min(max(t.columnWidth(1), 120), 320))
        t.verticalHeader().setDefaultSectionSize(20)
        t.setMinimumHeight(min(260, 26 + 20 * max(t.rowCount(), 1)))
        t.setMaximumHeight(260)
        t.keys = keys
        t.step = i
        t.cellDoubleClicked.connect(lambda r, c, t=t: self._edit_cell(t, r))
        t.setToolTip('Double-click a value to change it (a new version of the plan, checked by the page\'s rules)')
        self.tables.append(t)
        return t

    def _project(self):
        try:
            return self.ctl.ctx.store.project
        except AttributeError:
            return None

    def _edit_cell(self, t, r):
        key = t.keys[r] if 0 <= r < len(t.keys) else None
        if key is None or self.plan.status not in ('draft', 'approved', 'done', 'failed', 'cancelled'):
            return None
        st = self.plan.steps[t.step]
        cur = CP.get_path(st.settings, key, CP.flatten(st.settings).get(key))
        if cur is None and key.startswith('scenario.params.'):
            from ..sim import carla_bridge as CB
            sc = st.settings.get('scenario') or {}
            cur = CB.clean_scenario(sc.get('name'), sc.get('params'))[0]['params'].get(key.split('.')[-1])
        ed = self.editor_for(t.step, st.kind, key, cur)
        t.setCellWidget(r, 1, ed)
        ed.setFocus()
        return ed

    def editor_for(self, step, kind, key, cur):
        """The page's own kind of field for a value: a number field, a choice, a switch or JSON text."""
        from ..assistant import copilot_tools as CT
        choices = CT._choices(kind, key)
        lim = CT._limits(kind).get(key)
        if isinstance(cur, bool):
            w = QCheckBox()
            w.setChecked(cur)
            w.toggled.connect(lambda on, k=key: self.edit(step, k, bool(on)))
            return w
        if isinstance(cur, (int, float)):
            w = ScrubSpinBox('', 'plain', 0 if isinstance(cur, int) else 3, '')
            if lim:
                w.setRange(float(lim[0]), float(lim[1]))
            w.set_value(float(cur))
            w.committed.connect(lambda v, k=key, ints=isinstance(cur, int):
                                self.edit(step, k, int(round(v)) if ints else float(v)))
            return w
        if isinstance(cur, str) and choices:
            w = QComboBox()
            w.setEditable(True)
            for c in choices:
                w.addItem(str(c))
            w.setCurrentText(cur)
            w.activated.connect(lambda _i, w=w, k=key: self.edit(step, k, w.currentText()))
            return w
        w = QLineEdit(json.dumps(cur, ensure_ascii=False) if not isinstance(cur, str) else cur)
        w.setFont(mono(10))

        def commit(w=w, k=key, was_text=isinstance(cur, str)):
            txt = w.text()
            val = txt if was_text else CP.parse_value(txt)
            self.edit(step, k, val)
        w.returnPressed.connect(commit)
        return w

    def edit(self, step: int, key: str, value) -> bool:
        """Change one value: a new version of the plan (approval again), shown here with what changed."""
        new, why = self.ctl.edit_plan(self.plan, step, key, value)
        if new is None:
            self.note.setText(why)
            self.note.show()
            return False
        self.note.hide()
        self.set_plan(new)
        return True

    # ---------------------------------------------------------------- actions
    def approve(self):
        why = self.ctl.approve_plan(self.plan)
        if why:
            self.note.setText(why)
            self.note.show()
        self._sync(self.plan.id)

    def reject(self):
        self.ctl.reject_plan(self.plan)

    def open_on_page(self):
        self.ctl.open_on_page(self.plan, 0)

    def _record(self):
        return CP.find(self.ctl.ctx.store.project, self.plan.id, self.plan.version)

    def _sync(self, pid):
        if pid != self.plan.id:
            return
        try:
            self._sync_now()
        except RuntimeError:            # the window is closing
            pass

    def _sync_now(self):
        rec = self._record() if self.plan.status not in ('draft', 'rejected', 'superseded') else None
        status = rec['status'] if rec is not None else self.plan.status
        running = self.ctl.runner is not None and self.ctl.runner.running and self.ctl.runner.id == self.plan.id \
            and self.ctl.runner.version == self.plan.version
        self.badge.setText(STATUS_LABEL.get(status, status) if self.plan.ok or status != 'draft' else 'Invalid')
        draft = status == 'draft'
        self.b_approve.setVisible(draft)
        self.b_approve.setEnabled(self.plan.ok)
        self.b_approve.setToolTip('' if self.plan.ok else 'Not valid: the errors above must be fixed first')
        self.b_reject.setVisible(draft)
        self.b_open.setVisible(status not in ('rejected', 'superseded'))
        self.b_stop.setVisible(running)
        if rec is not None and rec.get('runs'):
            run = rec['runs'][-1]
            parts = []
            for i, s in enumerate(run['steps']):
                parts.append(f'{i + 1} {s["status"]}' + (f' ({s["message"]})' if s.get('message') else ''))
            self.state_lbl.setText(f'Approved {rec["approved_utc"]} · recorded in the project · ' + '; '.join(parts))
        elif draft:
            self.state_lbl.setText('Nothing runs until you approve it. Double-click a value to change it.'
                                   if self.plan.ok else 'This plan is not valid and cannot be approved.')
        elif status == 'superseded':
            self.state_lbl.setText('A newer version of this plan replaces it.')
        elif status == 'rejected':
            self.state_lbl.setText('Rejected: nothing was run.')
        else:
            self.state_lbl.setText('')
        self._sync_chips(rec)

    def _sync_chips(self, rec):
        """Revisions while it waits (from its own warnings), the checks that fit it once it ran; clicked, a revision is
        a follow-up to the Copilot (a new version, approval again), a check opens or asks."""
        recs = plan_chips(self.plan, rec) if self.plan.status not in ('rejected', 'superseded') else []
        sig = [(r.id, r.title) for r in recs]
        if sig == self._chips_sig:
            return
        self._chips_sig = sig
        if self.chips is not None:
            self.chips.hide()
            self.chips.setParent(None)
            self.chips.deleteLater()
            self.chips = None
        if recs:
            self.chips = NextSteps(recs, 'Revise it' if self.plan.status == 'draft' else 'Next')
            self.chips.act.connect(self.ctl.cta.emit)
            self.chips_lay.addWidget(self.chips)
        self.chips_box.setVisible(bool(recs))

    def _restyle(self):
        t = THEME.tok
        self.setStyleSheet(_qss_frame('copilot', t['raised'], t['strong'], 6))
        self.badge.setStyleSheet(f'color: {t["ink2"]}; border: 1px solid {t["strong"]}; border-radius: 3px; '
                                 f'padding: 0 5px;')


class ConsentCard(QFrame):
    """The first question to a provider asks once, naming it and what is sent."""

    def __init__(self, ctl: AssistantController, on_ok, parent=None, provider=None):
        super().__init__(parent)
        self.ctl = ctl
        self.provider = provider or ctl.provider
        self.setObjectName('consent')
        v = QVBoxLayout(self)
        v.setContentsMargins(12, 10, 12, 10)
        v.setSpacing(6)
        who = PR.LABEL[self.provider]
        t = QLabel(f'Send this question to {who}?')
        t.setFont(sans(12, QFont.Weight.DemiBold))
        v.addWidget(t)
        pics = (' With Claude, the pictures the assistant draws from the analysis (the coverage map, a camera\'s '
                'synthetic view) go too, inside the question.') if self.provider == PR.CLAUDE else ''
        v.addWidget(label(f'Your question, the plan settings, the design log and small tool results (camera poses, '
                          f'lens and module names, analysis figures, run summaries) go to {who} under its terms, with '
                          f'your own API key.{pics} Never sent: photos, recordings, point clouds, the vehicle model '
                          f'file, keys, stream passwords, the automation command or this computer\'s folders (shown '
                          f'as ~). Asked once for {who}.', ink2=True, small=True, wrap=True))
        r = QHBoxLayout()
        r.addStretch(1)
        self.b_no = button('Not now', 'ghost')
        self.b_ok = button(f'Allow {who}', 'primary')
        r.addWidget(self.b_no)
        r.addWidget(self.b_ok)
        v.addLayout(r)
        self.b_ok.clicked.connect(lambda: (ctl.give_consent(self.provider), self.hide(), on_ok()))
        self.b_no.clicked.connect(self.hide)
        tk = THEME.tok
        self.setStyleSheet(_qss_frame('consent', tk['raised'], tk['strong'], 6, THEME.color('warning').name()))


class SentDialog(QDialog):
    """What was sent: the last request's payload, masked, as it left the app."""

    def __init__(self, ctl: AssistantController, parent=None):
        super().__init__(parent)
        self.setWindowTitle('What was sent')
        self.resize(720, 560)
        v = QVBoxLayout(self)
        text = ctl.last_sent()
        who = PR.LABEL[ctl.session.provider] if ctl.session is not None else PR.LABEL[ctl.provider]
        v.addWidget(label(f'The last request to {who}, as sent (machine folders shown as ~; the API key goes in a '
                          'header and is never part of it).' if text else 'Nothing has been sent in this '
                                                                           'conversation yet.', ink2=True, wrap=True))
        self.text = QPlainTextEdit(text)
        self.text.setReadOnly(True)
        self.text.setFont(mono(10))
        v.addWidget(self.text, 1)
        b = QHBoxLayout()
        b.addStretch(1)
        c = button('Copy', 'ghost', 'copy')
        c.clicked.connect(lambda: QApplication.clipboard().setText(self.text.toPlainText()))
        ok = button('Close', 'primary')
        ok.clicked.connect(self.accept)
        b.addWidget(c)
        b.addWidget(ok)
        v.addLayout(b)


class PlansDialog(QDialog):
    """The project's approved plans: each with its status and runs; Run again re-runs one from its stored settings
    (the model is not asked)."""

    def __init__(self, ctl: AssistantController, parent=None):
        super().__init__(parent)
        from PyQt6.QtWidgets import QAbstractItemView, QHeaderView, QTableWidget
        self.ctl = ctl
        self.setWindowTitle('Copilot plans')
        self.resize(760, 420)
        v = QVBoxLayout(self)
        v.addWidget(label('Plans approved in this project, newest first. Run again uses the settings stored with the '
                          'approval; the model is not asked.', ink2=True, wrap=True))
        self.t = QTableWidget(0, 5)
        self.t.setHorizontalHeaderLabels(['Plan', 'Title', 'Steps', 'Status', 'Approved'])
        self.t.verticalHeader().hide()
        self.t.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.t.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.t.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        self.t.horizontalHeader().setSectionResizeMode(1, QHeaderView.ResizeMode.Stretch)
        v.addWidget(self.t, 1)
        self.info = label('', small=True, wrap=True)
        v.addWidget(self.info)
        b = QHBoxLayout()
        b.addStretch(1)
        self.b_run = button('Run again', 'primary', None, 'Run this plan again from its stored settings')
        self.b_run.clicked.connect(self.run_again)
        close = button('Close', 'ghost')
        close.clicked.connect(self.accept)
        b.addWidget(self.b_run)
        b.addWidget(close)
        v.addLayout(b)
        self.refresh()

    def refresh(self):
        from PyQt6.QtWidgets import QTableWidgetItem
        self.rows = self.ctl.records()
        self.t.setRowCount(0)
        for r in self.rows:
            i = self.t.rowCount()
            self.t.insertRow(i)
            ok = CP.verify(r)
            for c, text in enumerate((f'{r.get("id")} v{r.get("version")}', r.get('title', ''),
                                      ', '.join(CP.KIND_LABEL.get(s['kind'], s['kind']) for s in r.get('steps') or []),
                                      r.get('status', '') + ('' if ok else ' · does not verify'),
                                      r.get('approved_utc', ''))):
                self.t.setItem(i, c, QTableWidgetItem(str(text)))
        if self.rows:
            self.t.selectRow(0)
        self.b_run.setEnabled(bool(self.rows))

    def run_again(self) -> str:
        i = self.t.currentRow()
        if not 0 <= i < len(self.rows):
            return 'nothing chosen'
        r = self.rows[i]
        why = self.ctl.run_record(r['id'], r['version'], via='rerun')
        self.info.setText(why or f'Running plan {r["id"]} v{r["version"]} again from its stored settings…')
        return why


# ============================================================================================ pictures the model saw
def _pixmap(data: bytes):
    from PyQt6.QtGui import QPixmap
    pm = QPixmap()
    pm.loadFromData(data)
    return pm


class PictureThumb(QFrame):
    """A picture the model looked at (the look tool), as a thumbnail under its tool row; a click enlarges it."""
    THUMB = 220

    def __init__(self, pic, parent=None):
        super().__init__(parent)
        self.pic = pic
        self.setObjectName('thumb')
        v = QVBoxLayout(self)
        v.setContentsMargins(6, 6, 6, 6)
        v.setSpacing(4)
        self.img = QLabel()
        pm = _pixmap(pic.data)
        self.ok = not pm.isNull()
        if self.ok:
            self.img.setPixmap(pm.scaled(self.THUMB, self.THUMB, Qt.AspectRatioMode.KeepAspectRatio,
                                         Qt.TransformationMode.SmoothTransformation))
        self.img.setCursor(Qt.CursorShape.PointingHandCursor)
        v.addWidget(self.img, 0, Qt.AlignmentFlag.AlignLeft)
        kb = len(pic.data) / 1024
        self.meta = label(f'What the model saw · {pic.width}×{pic.height} px, {kb:.0f} kB, ≈ {pic.tokens:,} image '
                          f'tokens · click to enlarge', muted=True, small=True, wrap=True)
        v.addWidget(self.meta)
        self.setToolTip(pic.caption)
        self.dialog = None
        t = THEME.tok
        self.setStyleSheet(_qss_frame('thumb', t['panel'], t['hair'], 5))

    def mousePressEvent(self, e):
        if e.button() == Qt.MouseButton.LeftButton:
            self.enlarge()
        else:
            super().mousePressEvent(e)

    def enlarge(self):
        self.dialog = PictureDialog(self.pic, self.window())
        self.dialog.show()
        return self.dialog


class PictureDialog(QDialog):
    """The picture at full size with its caption: exactly what was sent to the model."""

    def __init__(self, pic, parent=None):
        super().__init__(parent)
        self.setWindowTitle('What the model saw')
        self.resize(min(pic.width + 60, 1100), min(pic.height + 160, 900))
        v = QVBoxLayout(self)
        sc = QScrollArea()
        sc.setWidgetResizable(True)
        self.img = QLabel()
        self.img.setPixmap(_pixmap(pic.data))
        self.img.setAlignment(Qt.AlignmentFlag.AlignCenter)
        sc.setWidget(self.img)
        v.addWidget(sc, 1)
        self.caption = label(pic.caption, ink2=True, small=True, wrap=True)
        self.caption.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        v.addWidget(self.caption)
        v.addWidget(label(f'{pic.media_type}, {pic.width}×{pic.height} px, {len(pic.data) / 1024:.0f} kB, ≈ '
                          f'{pic.tokens:,} image tokens. Drawn from the analysis on this machine and sent to the model '
                          f'inside your question, like any tool result.', muted=True, small=True, wrap=True))
        b = QHBoxLayout()
        b.addStretch(1)
        ok = button('Close', 'primary')
        ok.clicked.connect(self.accept)
        b.addWidget(ok)
        v.addLayout(b)


# ============================================================================================ the design log
class LogProposalCard(QFrame):
    """A design-log entry the model proposes: Accept writes it into the project (one undo step), Discard drops it."""

    def __init__(self, ctl: AssistantController, entry: dict, parent=None):
        super().__init__(parent)
        self.ctl, self.entry = ctl, entry
        self.setObjectName('logprop')
        v = QVBoxLayout(self)
        v.setContentsMargins(12, 8, 12, 8)
        v.setSpacing(5)
        head = QHBoxLayout()
        head.setSpacing(6)
        ico = QLabel()
        ico.setPixmap(icons.pixmap('history', THEME.tok['ink'], 14))
        head.addWidget(ico)
        t = label(f'Add to the design log: {M.KIND_LABEL[entry["kind"]].lower()}', bold=True)
        head.addWidget(t, 1)
        v.addLayout(head)
        tags = M.about_tags(ctl.ctx.store.project, entry)
        self.text = label(html.escape(entry['text']) + (f'<span style="color:{THEME.tok["muted"]}"> · about '
                                                        f'{html.escape(", ".join(tags))}</span>' if tags else ''),
                          wrap=True)
        self.text.setTextFormat(Qt.TextFormat.RichText)
        v.addWidget(self.text)
        b = QHBoxLayout()
        self.state = label('Kept with the project when you accept it.', muted=True, small=True)
        b.addWidget(self.state, 1)
        self.b_no = button('Discard', 'ghost')
        self.b_ok = button('Accept', 'primary')
        self.b_no.clicked.connect(lambda: ctl.discard_log(entry['id']))
        self.b_ok.clicked.connect(lambda: ctl.accept_log(entry['id']))
        b.addWidget(self.b_no)
        b.addWidget(self.b_ok)
        v.addLayout(b)
        ctl.log_changed.connect(self._sync)
        t_ = THEME.tok
        self.setStyleSheet(_qss_frame('logprop', t_['raised'], t_['strong'], 6))
        self._sync()

    def _sync(self):
        try:
            st = self.ctl.log_proposals.get(self.entry['id'], {}).get('state', 'discarded')
            self.b_ok.setVisible(st == 'pending')
            self.b_no.setVisible(st == 'pending')
            self.state.setText({'pending': 'Kept with the project when you accept it.',
                                'accepted': 'Added to the design log (Edit ▸ Undo removes it).',
                                'discarded': 'Discarded.'}.get(st, ''))
        except RuntimeError:            # the window is closing
            pass


class DesignLogDialog(QDialog):
    """The project's design log: list, add, edit and delete entries (each one undo step). Kept in the project file;
    the assistant reads it with every question and never proposes a rejected change again unasked."""

    def __init__(self, ctl: AssistantController, parent=None):
        super().__init__(parent)
        from PyQt6.QtWidgets import QAbstractItemView, QHeaderView, QTableWidget
        self.ctl = ctl
        self.setWindowTitle('Design log')
        self.resize(820, 480)
        v = QVBoxLayout(self)
        v.addWidget(label('Decisions, rejected ideas, constraints and notes, kept with the project. The assistant '
                          'reads them with every question: it respects the constraints and does not propose a '
                          'rejected change again unless you ask. Discarding a proposal adds a rejected entry by '
                          'itself.', ink2=True, wrap=True))
        self.t = QTableWidget(0, 5)
        self.t.setHorizontalHeaderLabels(['Kind', 'Entry', 'About', 'Date', 'By'])
        self.t.verticalHeader().hide()
        self.t.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.t.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.t.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        self.t.horizontalHeader().setSectionResizeMode(1, QHeaderView.ResizeMode.Stretch)
        self.t.itemSelectionChanged.connect(self._selected)
        self.t.itemDoubleClicked.connect(lambda *_: self.edit_selected())
        v.addWidget(self.t, 1)
        self.empty = label('The design log is empty. Add a decision or a constraint, or discard a proposal.',
                           muted=True, small=True, wrap=True)
        v.addWidget(self.empty)
        # the form: add a new entry, or edit the selected one
        form = QFrame()
        form.setObjectName('logform')
        f = QGridLayout(form)
        f.setContentsMargins(8, 8, 8, 8)
        f.setHorizontalSpacing(8)
        f.setVerticalSpacing(5)
        self.kind = QComboBox()
        for k in M.KINDS:
            self.kind.addItem(M.KIND_LABEL[k], k)
        self.kind.setCurrentIndex(M.KINDS.index('decision'))
        self.text = QLineEdit()
        self.text.setPlaceholderText('e.g. keep F above 0.55 m: the tow bar is in the way')
        self.text.setMaxLength(M.TEXT_MAX)
        self.text.textChanged.connect(lambda *_: self._buttons())
        self.about = QLineEdit()
        self.about.setPlaceholderText('camera or sensor tags, e.g. F, ML')
        f.addWidget(label('Kind', ink2=True, small=True), 0, 0)
        f.addWidget(self.kind, 0, 1)
        f.addWidget(label('About', ink2=True, small=True), 0, 2)
        f.addWidget(self.about, 0, 3)
        f.addWidget(label('Entry', ink2=True, small=True), 1, 0)
        f.addWidget(self.text, 1, 1, 1, 3)
        f.setColumnStretch(3, 1)
        self.note = label('', small=True, wrap=True)
        f.addWidget(self.note, 2, 0, 1, 4)
        v.addWidget(form)
        b = QHBoxLayout()
        self.b_new = button('New', 'ghost', 'plus', 'Clear the form for a new entry')
        self.b_new.clicked.connect(self.new)
        self.b_delete = button('Delete', 'ghost', 'trash', 'Delete the selected entry (Edit ▸ Undo brings it back)')
        self.b_delete.clicked.connect(self.delete_selected)
        b.addWidget(self.b_new)
        b.addWidget(self.b_delete)
        b.addStretch(1)
        self.b_save = button('Add', 'primary', None, 'Add the entry to the design log (one undo step)')
        self.b_save.clicked.connect(self.save)
        close = button('Close', 'ghost')
        close.clicked.connect(self.accept)
        b.addWidget(self.b_save)
        b.addWidget(close)
        v.addLayout(b)
        self.editing = None
        self.rows = []
        t = THEME.tok
        form.setStyleSheet(_qss_frame('logform', t['panel'], t['hair'], 5))
        ctl.log_changed.connect(self.refresh)
        ctl.ctx.store.changed.connect(self._store_changed)
        ctl.ctx.store.project_replaced.connect(self._store_changed)
        self.refresh()

    def _store_changed(self, *_):
        try:
            if self.isVisible():
                self.refresh()
        except RuntimeError:
            pass

    def refresh(self):
        from PyQt6.QtWidgets import QTableWidgetItem
        p = self.ctl.ctx.store.project
        self.rows = list(reversed(self.ctl.log()))           # newest first
        sel = self.editing
        self.t.blockSignals(True)
        self.t.setRowCount(0)
        for e in self.rows:
            i = self.t.rowCount()
            self.t.insertRow(i)
            for c, text in enumerate((M.KIND_LABEL[e['kind']], e['text'], ', '.join(M.about_tags(p, e)), e['date'],
                                      'You' if e['source'] == 'user' else 'Assistant, accepted')):
                it = QTableWidgetItem(text)
                if c == 1:
                    it.setToolTip(e['text'])
                self.t.setItem(i, c, it)
        self.t.blockSignals(False)
        self.empty.setVisible(not self.rows)
        ids = [e['id'] for e in self.rows]
        if sel in ids:
            self.t.selectRow(ids.index(sel))
        else:
            self.editing = None
        self._buttons()

    def _selected(self):
        i = self.t.currentRow()
        if not 0 <= i < len(self.rows):
            return
        e = self.rows[i]
        self.editing = e['id']
        self.kind.setCurrentIndex(M.KINDS.index(e['kind']))
        self.text.setText(e['text'])
        self.about.setText(', '.join(M.about_tags(self.ctl.ctx.store.project, e)))
        self.note.setText('')
        self._buttons()

    def edit_selected(self):
        self._selected()
        self.text.setFocus()

    def _buttons(self):
        self.b_save.setText('Save' if self.editing else 'Add')
        self.b_save.setEnabled(bool(self.text.text().strip()))
        self.b_delete.setEnabled(self.editing is not None)

    def new(self):
        self.editing = None
        self.t.clearSelection()
        self.text.clear()
        self.about.clear()
        self.note.setText('')
        self._buttons()
        self.text.setFocus()

    def _about(self):
        tags = M.tag_to_uid(self.ctl.ctx.store.project)
        names = [a.strip() for a in re.split(r'[,\s]+', self.about.text()) if a.strip()]
        bad = [a for a in names if a.lower() not in tags]
        return [tags[a.lower()] for a in names if a.lower() in tags], bad

    def save(self) -> bool:
        text = self.text.text().strip()
        if not text:
            return False
        about, bad = self._about()
        if bad:
            self.note.setText(f'Not a camera or sensor of this variant: {", ".join(bad)}')
            return False
        kind = self.kind.currentData()
        changes = M.parse_changes(text, [c.tag for c in self.ctl.ctx.store.project.variant.cameras])
        if self.editing:
            ok = self.ctl.edit_log(self.editing, kind=kind, text=text, about=about, changes=changes)
        else:
            e = self.ctl.add_log(kind, text, about, 'user', changes)
            ok = e is not None
            if ok:
                self.editing = e['id']
        self.refresh()
        self.note.setText('Saved. Edit ▸ Undo reverts it.' if ok else 'Nothing was saved.')
        return ok

    def delete_selected(self) -> bool:
        if not self.editing:
            return False
        ok = self.ctl.delete_log(self.editing)
        self.new()
        self.refresh()
        return ok


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
        sev = 'warning' if kind in ('rate', 'rounds', 'network', 'overloaded', 'refusal', 'cap', 'truncated') \
            else 'critical'
        ico = QLabel()
        ico.setPixmap(icons.status_icon(sev, 14).pixmap(14, 14))
        h.addWidget(ico, 0, Qt.AlignmentFlag.AlignTop)
        m = QLabel(html.escape(message))
        m.setWordWrap(True)
        m.setFont(sans(11))
        h.addWidget(m, 1)
        if kind in ('auth', 'model', 'permission', 'unavailable', 'billing', 'cap'):
            b = button('Settings…', 'ghost')
            b.clicked.connect(self.settings.emit)
            h.addWidget(b, 0, Qt.AlignmentFlag.AlignTop)
        if kind not in ('auth', 'cancelled', 'refusal', 'unavailable', 'billing', 'cap'):
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
    """One answer: reasoning, tool calls and text in the order they came, then a footer with copy and figures. The
    header names the model that answered and why it was chosen; the footer can re-ask the question with the other
    model."""
    retry = pyqtSignal()
    settings = pyqtSignal()
    reask = pyqtSignal(object)          # routing.Route: ask the same question again with that model

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
        self.model_lbl = label(f'{ctl.info().label} · {PR.LABEL[ctl.provider]}', muted=True, small=True)
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
        self.copilot_cards = []
        self.next_items = []            # the model's suggest_next items (recommend.Recommendation)
        self.next_steps: NextSteps | None = None
        self.route: dict | None = None
        self.pictures = []              # PictureThumb widgets: what the model looked at
        self.log_cards = []
        self.b_reask = None
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
        self.b_reask = QPushButton('')
        self.b_reask.setFont(sans(10))
        self.b_reask.setCursor(Qt.CursorShape.PointingHandCursor)
        self.b_reask.hide()
        self.b_reask.clicked.connect(self._reask)
        f.addWidget(self.b_reask)
        self.b_copy = tool_button('copy', 'Copy the answer as Markdown', size=22, icon_size=12)
        self.b_copy.clicked.connect(self._copy)
        f.addWidget(self.b_copy)
        self.footer.hide()
        self._n_tools = 0

    def _insert(self, w):
        self.v.insertWidget(self.v.indexOf(self.status), w)

    def on(self, ev: A.Event):
        k, d = ev.kind, ev.data
        if k == 'route':
            self.route = dict(d)
            self.model_lbl.setText(d['label'] if d.get('auto') or d.get('reason') == 'reask' else
                                   f'{PR.model_info(d["provider"], d["model"]).label} · {PR.LABEL[d["provider"]]}')
            self.model_lbl.setToolTip(('Chosen automatically on this machine from the request: ' if d.get('auto') else
                                       '') + f'{PR.LABEL[d["provider"]]} · {d["model"]}')
            notes = []
            if d.get('fallback') and d.get('announce'):
                notes.append(d['fallback'])
            if d.get('note'):
                notes.append(d['note'])
            if d.get('handover'):
                notes.append(f'A new conversation with {PR.LABEL[d["provider"]]}: a short summary of this one so far '
                             f'goes with the question.')
            if notes:
                self._insert(label(' '.join(notes), muted=True, small=True, wrap=True))
            return
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
            if d['name'] == 'suggest_next':         # shown as chips under the answer
                return
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
            for pic in d.get('pictures') or []:
                th = PictureThumb(pic)
                self.pictures.append(th)
                self._insert(th)
        elif k == 'log_proposal':
            card = LogProposalCard(self.ctl, d['entry'])
            self.log_cards.append(card)
            self._insert(card)
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
        elif k == 'next_steps':
            self.next_items = list(d.get('items') or [])[:RC.NEXT_MAX]
        elif k == 'proposal':
            self._insert(ProposalCard(self.ctl, d['proposal']))
        elif k == 'copilot':
            card = CopilotCard(self.ctl, d['plan'])
            self.copilot_cards.append(card)
            self._insert(card)
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
        alt = self._alternative()
        if alt is not None and d.get('ok') is not None:
            self._alt = alt
            self.b_reask.setText(f'Re-ask with {PR.model_info(alt.provider, alt.model).label}')
            self.b_reask.setToolTip(f'Ask the same question again with {PR.model_info(alt.provider, alt.model).label} '
                                    f'on {PR.LABEL[alt.provider]} (a new answer below; consent and the daily cap '
                                    f'apply)')
            t = THEME.tok
            self.b_reask.setStyleSheet(f'QPushButton {{ border: 1px solid {t["hair"]}; border-radius: 9px; '
                                       f'padding: 0 8px; min-height: 18px; color: {t["ink2"]}; background: '
                                       f'transparent; }} QPushButton:hover {{ border-color: {t["ink"]}; }}')
            self.b_reask.show()
        if d.get('ok') and not d.get('stopped'):
            self._add_next_steps()
        self.v.addWidget(self.footer)
        self.footer.show()

    def _add_next_steps(self):
        """The model's next steps, then the local recommendations worked out again on the new state (the ones about
        what the user is doing first), without duplicates; at most five chips."""
        model = list(self.next_items)
        local = []
        rc = getattr(self.ctl, 'recommender', None)
        if rc is not None:
            rc.refresh()
            norm = lambda x: re.sub(r'\W+', ' ', str(x).lower()).strip()
            taken = {norm(r.title) for r in model} | {norm(r.prompt) for r in model}
            pool = rc.strip() + [r for r in rc.recs if r.base >= RC.SEVERITY['info'] * 0.6]
            for r in pool:
                if r.id not in {x.id for x in local} and norm(r.title) not in taken and norm(r.prompt or '-') not in taken:
                    local.append(r)
                if len(local) >= 2:
                    break
        items = (model + local)[:5]
        if not items:
            return
        self.next_steps = NextSteps(items)
        self.next_steps.act.connect(self.ctl.cta.emit)
        self.v.addWidget(self.next_steps)

    def _alternative(self):
        if not self.route:
            return None
        r = RT.Route(self.route['provider'], self.route['model'], self.route['tier'], self.route['reason'])
        return self.ctl.alternative(r) if self.route.get('auto') or self.route.get('reason') == 'reask' else None

    def _reask(self):
        alt = getattr(self, '_alt', None)
        if alt is not None:
            self.reask.emit(alt)

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
        auto = self.ctl.auto
        a = m.addAction('Automatic    quick questions to a fast model, goals, plans and pictures to Claude Opus')
        a.setCheckable(True)
        a.setChecked(auto)
        a.triggered.connect(lambda on: self.ctl.set_option(auto=bool(on)))
        m.addSeparator()
        for mid in self.ctl.models():
            info = self.ctl.info(mid)
            a = m.addAction(f'{info.label}    {info.note}' if info.note else info.label)
            a.setCheckable(True)
            a.setChecked(mid == cur and not auto)
            a.triggered.connect(lambda _=False, x=mid: self.ctl.set_option(model=x, auto=False))
        info = self.ctl.info(cur)
        if info.efforts and self.ctl.provider == PR.CLAUDE:
            m.addSeparator()
            hdr = m.addAction('Thinking: adaptive · effort high when planning, medium for questions')
            hdr.setEnabled(False)
        elif info.efforts:
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
        info = self.ctl.info()
        eff = self.ctl.effort if self.ctl.effort in info.efforts else info.default_effort
        claude = self.ctl.provider == PR.CLAUDE
        if self.ctl.auto:
            self.model_btn.setText('Automatic  ▾')
            self.model_btn.setToolTip('Automatic model choice: each answer names the model that wrote it and why')
        else:
            self.model_btn.setText(info.label + (f' · {eff}' if info.efforts and not claude else '') + '  ▾')
            self.model_btn.setToolTip('Model and reasoning effort')
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


# ============================================================================================ recommendations
KIND_ICON = {'ask': 'chat', 'goal': 'optimise', 'plan': 'checklist', 'open': 'external'}


def _metrics_of(res) -> dict:
    m = res.metrics
    return {'blind_m2': float(m.get('blind_m2', 0.0)), 'min_density': float(m.get('min_density', 0.0)),
            'min_ok_pct': float(m.get('min_ok_pct', 0.0))}


class ContextTracker(QObject):
    """What the user is doing now, for the recommendations: the open page, the selection, the last edits (read off the
    undo stack) with the analysis results just before and just after each, and a job that just finished (a page's
    result that appeared or changed, a plan that ran). It reads; it never changes anything."""
    changed = pyqtSignal()
    POLL_MS = 1000
    # page -> the job its result stands for (the attribute holding the result)
    WATCH = (('drivelog', 'drivelog', 'result'), ('detection', 'detection', 'result'),
             ('campaign', 'campaign', 'result'), ('sim', 'carla_run', 'result'), ('requirements', 'requirements', 'full'),
             ('asbuilt', 'asbuilt', 'result'))

    def __init__(self, ctx, ctl, parent=None):
        super().__init__(parent)
        self.ctx, self.ctl = ctx, ctl
        st = ctx.store
        self.edits: list = []                # oldest first: {'index', 'text', 'before', 'after', 'revision', 'cmd'}
        self.job: dict | None = None
        self._metrics, self._metrics_rev = None, -1
        self._rev = st.revision
        self._plan_seen = None
        st.undo.indexChanged.connect(self._on_index)
        st.project_replaced.connect(self._reset)
        ctx.analysis.finished.connect(self._on_analysis)
        ctl.plan_state.connect(self._on_plan)
        if ctx.analysis.result is not None:
            self._metrics, self._metrics_rev = _metrics_of(ctx.analysis.result), \
                getattr(ctx.analysis.result, 'revision', st.revision)
        self._sig = self._signature()        # results already there did not "just finish"
        self.timer = QTimer(self)
        self.timer.setInterval(self.POLL_MS)
        self.timer.timeout.connect(self.poll)
        self.timer.start()

    # ---------------------------------------------------------------- edits and the analysis around them
    def _on_index(self, idx):
        try:
            st = self.ctx.store
            stack = st.undo
            if sip.isdeleted(stack):
                return
            cmd = stack.command(idx - 1) if idx > 0 else None
            known = next((e for e in self.edits if e['cmd'] is cmd), None) if cmd is not None else None
            if cmd is not None and idx == stack.count() and known is None:          # a new step
                before = self._metrics if self._metrics is not None and self._metrics_rev == self._rev else None
                self.edits.append({'index': idx, 'text': cmd.text(), 'before': before, 'after': None,
                                   'revision': st.revision, 'cmd': cmd})
                self.edits = self.edits[-6:]
            elif known is not None and known is self.edits[-1] and idx == stack.count() and \
                    known['revision'] != st.revision and known['text'] != cmd.text():
                known.update(text=cmd.text(), revision=st.revision, after=None)     # merged into it (a drag)
            self._rev = st.revision
        except RuntimeError:                 # the window is closing
            return
        self.changed.emit()

    def _on_analysis(self, res):
        if res is None:
            return
        try:
            st = self.ctx.store
            self._metrics, self._metrics_rev = _metrics_of(res), getattr(res, 'revision', st.revision)
            e = self.edits[-1] if self.edits else None
            if e is not None and e['after'] is None and e['before'] is not None and self._metrics_rev >= e['revision'] \
                    and st.undo.index() == e['index'] and st.undo.command(e['index'] - 1) is e['cmd']:
                e['revision'] = max(e['revision'], self._metrics_rev)
                e['after'] = self._metrics
            if self._metrics_rev == st.revision:
                self._rev = st.revision
        except RuntimeError:
            return
        self.changed.emit()

    def _reset(self, *_):
        self.edits, self.job = [], None
        self._metrics, self._metrics_rev, self._rev = None, -1, self.ctx.store.revision
        self._sig = self._signature()
        self.changed.emit()

    # ---------------------------------------------------------------- jobs
    def _signature(self) -> dict:
        w = self.ctx.window
        pages = getattr(w, '_page_widgets', {}) if w is not None else {}
        out = {}
        for name, job, attr in self.WATCH:
            pg = pages.get(name)
            res = getattr(pg, attr, None) if pg is not None else None
            ok = not (isinstance(res, dict) and res.get('status') not in (None, 'ok'))
            out[job] = (id(res) if res is not None else None, ok)
        pc = getattr(self.ctx, 'perception', None)
        res = getattr(pc, 'result', None)
        out['perception'] = (id(res) if res is not None else None, True)
        return out

    def poll(self):
        """A page's result appeared or changed: a job finished (checked once a second; nothing is computed)."""
        try:
            sig = self._signature()
        except RuntimeError:
            return
        new = [(k, v) for k, v in sig.items() if v[0] is not None and self._sig.get(k, (None,))[0] != v[0]]
        self._sig = sig
        if new:
            k, (_, ok) = new[-1]
            self.job = {'kind': k, 'ok': ok, 't': time.time()}
            self.changed.emit()

    def _on_plan(self, pid):
        r = self.ctl.runner
        if r is None or r.running or r.id != pid or not r.status or self._plan_seen == (r.id, r.version, r.status):
            return
        self._plan_seen = (r.id, r.version, r.status)
        self.job = {'kind': 'plan', 'ok': r.status == 'done', 't': time.time(), 'plan': pid}
        self.changed.emit()

    # ---------------------------------------------------------------- now
    def context(self) -> RC.Context:
        ctx, st = self.ctx, self.ctx.store
        w = ctx.window
        page = w.page_name() if w is not None and hasattr(w, 'page_name') else 'viewports'
        sel = [('camera', c.tag, c.tag) for c in st.selected_cams()]
        o = st.sel_obj
        if not sel and o and len(o) > 1:
            v = st.project.variant
            if o[0] == 'sensor':
                s = v.sensor(o[1]) if hasattr(v, 'sensor') else None
                if s is not None:
                    sel.append(('sensor', s.tag, s.tag))
            elif o[0] == 'rig':
                q = next((x for x in getattr(st.project, 'rig', ()) if x.uid == o[1]), None)
                if q is not None:
                    sel.append(('rig', q.uid, q.name))
            elif o[0] == 'target':
                t = next((x for x in st.project.targets if x.uid == o[1]), None)
                if t is not None:
                    sel.append(('target', t.uid, t.name))
        if page == 'requirements':
            rq = getattr(w, '_page_widgets', {}).get('requirements')
            if rq is not None and getattr(rq, 'sel_id', None):
                sel.append(('requirement', rq.sel_id, rq.sel_id))
        pl = self.ctl.current_plan
        if pl is not None and pl.status == 'draft':
            sel.append(('plan', pl.id, f'plan {pl.id}'))
        return RC.Context(page=page, page_label=PAGES.get(page, page), selection=tuple(sel),
                          edits=[{k: v for k, v in e.items() if k != 'cmd'} for e in self.edits],
                          undo_index=st.undo.index(), job=dict(self.job) if self.job else None, now=time.time())


class Recommender(QObject):
    """The recommendations, worked out locally (recommend.py) a moment after the design, the selection, the page, the
    analysis, a page's result or a plan changes (debounced), and the ones the user dismissed (per project file, in
    QSettings; never in the project). Nothing here sends anything or runs anything."""
    updated = pyqtSignal()
    _hint_done = pyqtSignal(object)      # (key, hint) from the worker that analyses copies
    DEBOUNCE_MS = 300

    def __init__(self, ctx, ctl, parent=None):
        super().__init__(parent)
        self.ctx, self.ctl = ctx, ctl
        self.hints: dict = {}            # key -> hint (or None: nothing helps); see recommend.pitch_hint / scale_hint
        self._hint_busy = None
        self._hint_next = None
        self._hint_done.connect(self._on_hint)
        self.tracker = ContextTracker(ctx, ctl, self)
        self.recs: list = []
        self.suppressed: list = []      # left out: they would suggest a change the design log records as rejected
        self.context: RC.Context | None = None
        self.ms = {'gather': 0.0, 'recommend': 0.0}
        self.refreshes = 0
        self._shown = None
        self._runs = (None, [])
        self.dismissed = RC.Dismissed()
        self._dkey = ''
        self._load_dismissed()
        self.timer = QTimer(self)
        self.timer.setSingleShot(True)
        self.timer.setInterval(self.DEBOUNCE_MS)
        self.timer.timeout.connect(self.refresh)
        st, an = ctx.store, ctx.analysis
        st.changed.connect(self.schedule)
        st.selection_changed.connect(self.schedule)
        st.project_replaced.connect(self._replaced)
        an.finished.connect(self.schedule)
        an.secondary.connect(self.schedule)
        an.stale_changed.connect(self.schedule)
        ctl.plan_state.connect(self.schedule)
        ctl.reset.connect(self.schedule)
        self.tracker.changed.connect(self.schedule)
        w = ctx.window
        if w is not None and hasattr(w, 'pages'):
            w.pages.currentChanged.connect(self.schedule)
        so = getattr(ctx, 'stitched', None)
        if so is not None and hasattr(so, 'ready'):
            so.ready.connect(self.schedule)
        self.refresh()

    def schedule(self, *_):
        self.timer.start()

    # ---------------------------------------------------------------- dismissed, per project file
    def _key(self) -> str:
        p = self.ctx.store.project
        ident = os.path.abspath(p.path) if getattr(p, 'path', '') else f'unsaved:{p.name}'
        return 'assistant/dismissed/' + hashlib.sha1(ident.encode('utf-8')).hexdigest()[:16]

    def _load_dismissed(self):
        self._dkey = self._key()
        try:
            data = json.loads(str(self.ctx.settings.value(self._dkey, '') or '{}'))
        except (TypeError, ValueError):
            data = {}
        self.dismissed = RC.Dismissed(data if isinstance(data, dict) else {})

    def _save_dismissed(self):
        self.ctx.settings.setValue(self._dkey, json.dumps(self.dismissed.data, default=str))

    def _replaced(self, *_):
        self._load_dismissed()
        self._runs = (None, [])
        self.schedule()

    def dismiss(self, rec: RC.Recommendation):
        if self._key() != self._dkey:            # saved under a new name since: its own list
            self._load_dismissed()
        self.dismissed.dismiss(rec)
        self._save_dismissed()
        self.refresh()

    # ---------------------------------------------------------------- what the app knows now
    def _page(self, name):
        w = self.ctx.window
        return getattr(w, '_page_widgets', {}).get(name) if w is not None else None

    def _carla(self, p) -> dict:
        from ..assistant import copilot_tools as CT
        from ..core import simrig as SR
        from ..sim import carla_bridge as CB
        from ..sim import server as SV
        pg = self._page('sim')
        root = str(self.ctx.settings.value('sim/carla_root', '') or '')
        installed = SV.is_carla_root(root) or (pg is not None and SV.is_carla_root(getattr(pg, 'carla_root', '')))
        port = int(SR.settings_of(p).port)
        up, render = None, None
        if pg is not None:
            txt = pg.srv_status.text.text()
            up = True if (pg.server is not None and pg.server.running) or 'answers on port' in txt else \
                False if txt.startswith('No server') else None
            rig = getattr(pg, 'rig', None)
            if rig is not None:
                d = rig.doc
                render = {'mpx': sum(c['render_mpx'] for c in d['cameras']),
                          'gt_mpx': float((d.get('ground_truth') or {}).get('gt_render_mpx', 0.0)),
                          'budget': float(CB.RENDER_BUDGET_MPX), 'scale': float(pg.settings.image_scale)}
        wd = workdir.get(self.ctx.settings)
        rdir = workdir.project_dir(p, wd, 'carla')
        try:
            key = (rdir, os.stat(rdir).st_mtime_ns)
        except OSError:
            key = (rdir, None)
        if not installed and pg is None:
            key = (rdir, None)                   # CARLA not set up and its page never opened: no runs to look for
        if self._runs[0] != key:                 # read the runs again only when the folder changed
            self._runs = (key, CT.find_runs(p, wd, limit=12) if key[1] is not None else [])
        rows = self._runs[1]
        last = None
        if pg is not None and isinstance(pg.result, dict) and pg.out_dir:
            from ..core import groundtruth as GT
            last = {'path': pg.out_dir, 'name': os.path.basename(pg.out_dir), 'status': pg.result.get('status'),
                    'error': str(pg.result.get('error') or ''),
                    'ground_truth': os.path.isfile(os.path.join(pg.out_dir, GT.INDEX))}
        elif rows:
            r = rows[0]
            last = {'path': r['path'], 'name': os.path.basename(r['path']), 'status': r['status'], 'error': '',
                    'ground_truth': bool(r['ground_truth'])}
        if last is not None:
            base = os.path.normpath(last['path'])
            dl, de = self._page('drivelog'), self._page('detection')
            dres = getattr(dl, 'result', None)
            last['checked'] = dres is not None and os.path.normpath(str(dres.path)).startswith(base)
            eres = getattr(de, 'result', None)
            last['scored'] = eres is not None and any(os.path.normpath(str(r.get('path') or '')).startswith(base)
                                                      for r in eres.recordings)
        return {'installed': installed, 'port': port, 'server_up': up, 'runs': len(rows) + (1 if last and not rows
                                                                                           else 0),
                'last_run': last, 'render': render}

    def gather(self) -> RC.State:
        ctx, ctl = self.ctx, self.ctl
        p, an = ctx.store.project, ctx.analysis
        res = an.result if an.result is not None and getattr(an.result, 'variant_uid', p.active_uid) == p.active_uid \
            else None
        worst = an.worst if res is not None and an.worst and len(an.worst) == 3 else None
        so = getattr(ctx, 'stitched', None)
        stitched = getattr(so, 'result', None)
        if stitched is not None and getattr(stitched, 'variant_uid', p.active_uid) not in ('', p.active_uid):
            stitched = None
        rq = self._page('requirements')
        lc = self._page('lidar_calib')
        rig = self._page('rig')
        pc = getattr(ctx, 'perception', None)
        try:
            carla = self._carla(p)
        except Exception:                                    # noqa: BLE001 — no CARLA facts beats no list
            carla = None
        return RC.State(p, res, list(an.issues), worst, bool(an.stale), stitched, getattr(so, 'advice', None),
                        getattr(rq, 'res', None), getattr(pc, 'result', None),
                        getattr(self._page('drivelog'), 'result', None), getattr(self._page('detection'), 'result', None),
                        getattr(self._page('campaign'), 'result', None), carla,
                        rig.health() if rig is not None else None,
                        lc is not None and (lc.result is not None or lc.plan is not None),
                        [e['plan'] for e in ctl.plans.values() if e.get('plan') is not None],
                        CP.records(p), ctl.brief())

    # ---------------------------------------------------------------- hints: copies analysed off the UI thread
    def _hint_keys(self, state, cx) -> dict:
        """The hints worth having now: a pitch for the selected camera whose minimum density is below the target
        (on an up-to-date analysis), the image scale that fits when the CARLA page's rig is over the render budget."""
        st, an, p = self.ctx.store, self.ctx.analysis, state.project
        out = {}
        par = self.parent()
        if par is None or not par.isVisible():          # only while the assistant panel is open
            return out
        res = state.result
        fresh = res is not None and not state.stale and getattr(res, 'revision', -1) == st.revision
        for sel in cx.selection[:1]:
            if sel[0] != 'camera' or not fresh:
                continue
            c = p.variant.by_tag(sel[1])
            d = res.per_cam.get(c.uid) if c is not None else None
            if d and d.get('best_m2', 0) > 0 and d['min_dens'] < p.analysis.target_density and not c.locked:
                out[f'pitch:{c.tag}'] = ('pitch', c.tag, st.revision, p.active_uid)
        r = (state.carla or {}).get('render') or {}
        if r and r['mpx'] + r['gt_mpx'] > r['budget']:
            from ..core import simrig as SR
            out['scale'] = ('scale', st.revision, p.active_uid,
                            json.dumps(SR.settings_of(p).to_dict(), sort_keys=True, default=str))
        return out

    def _start_hint(self, name, key):
        if self._hint_busy is not None:
            self._hint_next = (name, key)
            return
        self._hint_busy = key
        snap = self.ctx.store.project.snapshot()
        if key[0] == 'pitch':
            eng = self.ctx.analysis.engine.share()

            def work():
                return RC.pitch_hint(snap, eng, key[1])
        else:
            from ..core import simrig as SR
            from ..sim import carla_bridge as CB
            settings = SR.settings_of(snap).to_dict()

            def work():
                return RC.scale_hint(snap, settings, float(CB.RENDER_BUDGET_MPX))

        def run():
            try:
                h = work()
            except Exception:                                # noqa: BLE001 — no hint, the plain recommendation
                h = None
            try:
                self._hint_done.emit((key, h))
            except RuntimeError:                             # the window closed meanwhile
                pass
        threading.Thread(target=run, daemon=True, name='recommend-hint').start()

    def _on_hint(self, item):
        key, h = item
        self.hints[key] = h
        self._hint_busy = None
        nxt, self._hint_next = self._hint_next, None
        if nxt is not None and nxt[1] not in self.hints:
            self._start_hint(*nxt)
        self.schedule()

    def refresh(self):
        """Work the list out now (the debounce ends here); updated is emitted only when what is shown changes."""
        try:
            t0 = time.perf_counter()
            state = self.gather()
            cx = self.tracker.context()
            for name, key in self._hint_keys(state, cx).items():
                if key in self.hints:
                    if self.hints[key] is not None:
                        state.hints[name] = self.hints[key]
                elif key != self._hint_busy:
                    self._start_hint(name, key)
            t1 = time.perf_counter()
            every = RC.recommend(state, cx, None, [r.id for r in self.recs])
            every, self.suppressed = M.suppress(state.project, every)     # what the design log rejected
            self.dismissed.prune(every)
            recs = [r for r in every if not self.dismissed.hides(r)]
            t2 = time.perf_counter()
        except RuntimeError:                                 # the window is closing
            return
        self.ms = {'gather': (t1 - t0) * 1000, 'recommend': (t2 - t1) * 1000}
        self.refreshes += 1
        self.recs, self.context = recs, cx
        strip = self.strip()
        shown = ([(r.id, r.title, r.why, r.kind) for r in recs], RC.context_label(cx, strip), [r.id for r in strip])
        if shown != self._shown:
            self._shown = shown
            self.updated.emit()

    def strip(self) -> list:
        return RC.context_items(self.recs, self.context)


class RecRow(QFrame):
    """One recommendation: an icon by kind, the title, the evidence (muted), a button saying what a click does, ×."""
    act = pyqtSignal(object)
    dismiss = pyqtSignal(object)

    def __init__(self, rec: RC.Recommendation, compact=False, parent=None):
        super().__init__(parent)
        self.setObjectName('recrow')
        self.compact = compact
        h = QHBoxLayout(self)
        h.setContentsMargins(8, 5 if compact else 7, 4, 5 if compact else 7)
        h.setSpacing(8)
        self.ico = QLabel()
        h.addWidget(self.ico, 0, Qt.AlignmentFlag.AlignTop)
        col = QVBoxLayout()
        col.setSpacing(1)
        self.title = QLabel('')
        self.title.setWordWrap(True)
        self.title.setFont(sans(11, QFont.Weight.DemiBold if not compact else QFont.Weight.Normal))
        col.addWidget(self.title)
        self.why = label('', muted=True, small=True, wrap=True)
        col.addWidget(self.why)
        h.addLayout(col, 1)
        self.b = QPushButton('')
        self.b.setCursor(Qt.CursorShape.PointingHandCursor)
        self.b.setFont(sans(10))
        self.b.clicked.connect(lambda: self.act.emit(self.rec))
        h.addWidget(self.b, 0, Qt.AlignmentFlag.AlignTop)
        self.x = tool_button('close', 'Dismiss (it comes back when its figures change)', size=20, icon_size=10)
        self.x.clicked.connect(lambda: self.dismiss.emit(self.rec))
        h.addWidget(self.x, 0, Qt.AlignmentFlag.AlignTop)
        self.rec = None
        self.set_rec(rec)

    def set_rec(self, rec: RC.Recommendation):
        """In place: only what changed is set again (no flicker)."""
        t = THEME.tok
        old = self.rec
        self.rec = rec
        if old is None or old.kind != rec.kind:
            name = 'undo' if rec.target and 'undo' in rec.target else KIND_ICON.get(rec.kind, 'sparkle')
            self.ico.setPixmap(icons.pixmap(name, t['ink2'], 14))
            self.b.setText(rec.button)
            primary = rec.kind != 'open'
            self.b.setStyleSheet(
                f'QPushButton {{ border: 1px solid {t["ink"] if primary else t["strong"]}; border-radius: 10px; '
                f'padding: 1px 10px; min-height: 18px; color: {t["panel"] if primary else t["ink"]}; '
                f'background: {t["ink"] if primary else "transparent"}; }}'
                f'QPushButton:hover {{ background: {t["ink2"] if primary else t["sel"]}; }}')
            self.setStyleSheet(f'QFrame#recrow {{ background: {t["raised"]}; border: 1px solid {t["hair"]}; '
                               f'border-radius: 6px; }}')
        if old is None or old.title != rec.title:
            self.title.setText(html.escape(rec.title, quote=False))
        if old is None or old.why != rec.why:
            self.why.setText(html.escape(rec.why, quote=False))
            self.why.setVisible(bool(rec.why))
        tip = {'ask': 'Sends this to the assistant, as typing it would:\n\n', 'goal': 'Starts the agent on this goal:\n\n',
               'plan': 'Asks the Copilot for a plan; nothing runs until you approve it:\n\n'}.get(rec.kind, '')
        self.b.setToolTip((tip + rec.prompt) if rec.prompt else 'Opens it; runs nothing')
        self.setToolTip(f'From: {rec.source}' if rec.source else '')


class RecList(QWidget):
    """Recommendations as rows: the first few, the rest one click away (More)."""
    act = pyqtSignal(object)
    dismiss = pyqtSignal(object)

    def __init__(self, shown=RC.SHOWN, compact=False, parent=None):
        super().__init__(parent)
        self.shown, self.compact = shown, compact
        self.v = QVBoxLayout(self)
        self.v.setContentsMargins(0, 0, 0, 0)
        self.v.setSpacing(5)
        self.rows: dict = {}
        self.more = QPushButton('')
        self.more.setCursor(Qt.CursorShape.PointingHandCursor)
        self.more.setFlat(True)
        self.more.setFont(sans(10))
        self.more.clicked.connect(self.toggle)
        self.v.addWidget(self.more, 0, Qt.AlignmentFlag.AlignLeft)
        self.expanded = False
        self.recs = []

    def toggle(self):
        self.expanded = not self.expanded
        self.set_recs(self.recs)

    def set_recs(self, recs):
        """Rows kept by id and updated in place, in the new order; hidden rather than rebuilt."""
        self.recs = list(recs)
        n = len(self.recs) if self.expanded else min(self.shown, len(self.recs))
        want = [r.id for r in self.recs[:n]]
        for rid in [k for k in self.rows if k not in want]:
            w = self.rows.pop(rid)
            w.hide()
            w.setParent(None)
            w.deleteLater()
        for i, r in enumerate(self.recs[:n]):
            w = self.rows.get(r.id)
            if w is None:
                w = RecRow(r, self.compact)
                w.act.connect(self.act.emit)
                w.dismiss.connect(self.dismiss.emit)
                self.rows[r.id] = w
            else:
                w.set_rec(r)
            if self.v.indexOf(w) != i:
                self.v.insertWidget(i, w)
            w.show()
        rest = len(self.recs) - self.shown
        self.more.setVisible(rest > 0)
        self.more.setText('Fewer' if self.expanded else f'More ({rest})')
        t = THEME.tok
        self.more.setStyleSheet(f'QPushButton {{ color: {t["ink2"]}; border: none; padding: 0 2px; }}'
                                f'QPushButton:hover {{ color: {t["ink"]}; }}')


class LiveStrip(QFrame):
    """The top of the panel: 1–3 recommendations about what the user is doing now ("For QR · Coverage page"),
    updated a moment after the selection, the page or a result changes; hidden when there is none."""
    act = pyqtSignal(object)
    dismiss = pyqtSignal(object)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName('livestrip')
        v = QVBoxLayout(self)
        v.setContentsMargins(10, 6, 10, 7)
        v.setSpacing(4)
        head = QHBoxLayout()
        head.setSpacing(6)
        self.ico = QLabel()
        head.addWidget(self.ico)
        self.head = label('', muted=True, small=True)
        head.addWidget(self.head, 1)
        v.addLayout(head)
        self.list = RecList(RC.STRIP, compact=True)
        self.list.act.connect(self.act.emit)
        self.list.dismiss.connect(self.dismiss.emit)
        v.addWidget(self.list)
        self._restyle()
        self.hide()

    def _restyle(self):
        t = THEME.tok
        self.ico.setPixmap(icons.pixmap('sparkle', t['ink2'], 12))
        self.setStyleSheet(f'QFrame#livestrip {{ background: {t["panel"]}; border: none; '
                           f'border-bottom: 1px solid {t["hair"]}; }}')

    def set_items(self, recs, text):
        if not recs:
            self.hide()
            return
        self.head.setText(html.escape(text or 'Now', quote=False))
        self.list.set_recs(recs)
        self.show()


class NextSteps(QWidget):
    """Next steps under an answer: the model's suggest_next items and the local recommendations, as chips. A chip
    does what its kind says (ask, start a goal, ask for a plan, open); none runs anything by itself."""
    act = pyqtSignal(object)

    def __init__(self, recs, title='Next steps', parent=None):
        super().__init__(parent)
        v = QVBoxLayout(self)
        v.setContentsMargins(0, 2, 0, 0)
        v.setSpacing(4)
        v.addWidget(label(title, muted=True, small=True))
        self.recs = list(recs)
        self.chips = FlowChips([(r.title, r.id) for r in self.recs])
        by = {r.id: r for r in self.recs}
        self.chips.picked.connect(lambda rid: self.act.emit(by[rid]))
        for b, r in zip(self.chips.btns, self.recs):
            b.setIcon(icons.icon(KIND_ICON.get(r.kind, 'sparkle'), 12))
            b.setToolTip({'ask': 'Ask: ', 'goal': 'Start a goal: ', 'plan': 'Ask for a plan (approval needed): ',
                          'open': 'Open (runs nothing): '}.get(r.kind, '') + (r.prompt or r.why))
        v.addWidget(self.chips)


def plan_chips(plan: CP.Plan, rec: dict | None) -> list:
    """The chips under a plan card: while it waits, the revisions its own warnings suggest; after it ran, the checks
    that fit its kind; after it failed, why."""
    if plan.status == 'draft':
        return RC.plan_revisions(plan)
    if rec is None or not rec.get('runs'):
        return []
    run = rec['runs'][-1]
    out = []
    if run.get('status') == 'failed':
        st = RC.State(None, records=[rec])
        return [r for r in RC.plan_rules(st, RC.Context()) if r.id.startswith('plan:failed')]
    if run.get('status') != 'done':
        return []
    kinds = [s['kind'] for s in rec.get('steps', [])]
    gt = any(s['kind'] == 'carla_run' and any((s.get('settings') or {}).get('ground_truth', {}).values())
             for s in rec.get('steps', []) if isinstance((s.get('settings') or {}).get('ground_truth'), dict))
    if 'carla_run' in kinds:
        out.append(RC.Recommendation(f'plan:{plan.id}:check', 'Check the drive', 'Opens the CARLA page on the drive; '
                                     'its Drive-log check button runs the check.', 'open',
                                     target={'page': 'sim', 'focus': 'b_drivelog'}, source='Copilot plan'))
        if gt:
            out.append(RC.Recommendation(f'plan:{plan.id}:score', 'Score detection', 'Opens the CARLA page on the drive; '
                                         'its Detection quality button opens the scoring.', 'open',
                                         target={'page': 'sim', 'focus': 'b_detect'}, source='Copilot plan'))
    for k, title, prompt in (('campaign', 'What did the campaign find?', 'Read the campaign\'s results with '
                                                                         'get_campaign and say what it found.'),
                             ('detection', 'Where is recall weak?', 'Read the detection quality with '
                                                                    'get_detection_quality: where is recall weak?'),
                             ('critical_carla', 'Which targets were missed?', 'Read get_critical_carla: which critical '
                                                                              'targets were missed, and why?'),
                             ('experiment', 'What did the experiment show?', 'Read get_experiment: which factors '
                                                                             'matter most?'),
                             ('carla_live', 'How healthy was the live rig?', 'Read get_carla_live and get_rig_health: '
                                                                             'which faults showed up?')):
        if k in kinds:
            out.append(RC.Recommendation(f'plan:{plan.id}:{k}', title, '', 'ask', prompt=prompt, source='Copilot plan'))
    if 'sample_dataset' in kinds:
        out.append(RC.Recommendation(f'plan:{plan.id}:dataset', 'Open the sample dataset', '', 'open',
                                     target={'page': 'dataset'}, source='Copilot plan'))
    return out


class RecPopup(QFrame):
    """The header chip's list: every recommendation, the first five shown, the rest under More."""

    def __init__(self, parent=None):
        super().__init__(parent, Qt.WindowType.Popup)
        self.setObjectName('recpopup')
        v = QVBoxLayout(self)
        v.setContentsMargins(10, 10, 10, 10)
        v.setSpacing(6)
        v.addWidget(label('Recommended for this design', bold=True))
        v.addWidget(label('Worked out on this machine from Vantage\'s own results; nothing is sent until you click Ask, '
                          'Start goal or Plan it.', muted=True, small=True, wrap=True))
        self.list = RecList(RC.SHOWN)
        v.addWidget(self.list)
        t = THEME.tok
        self.setStyleSheet(f'QFrame#recpopup {{ background: {t["panel"]}; border: 1px solid {t["strong"]}; '
                           f'border-radius: 6px; }}')

    def open_at(self, anchor):
        self.setFixedWidth(max(360, min(460, anchor.window().width() // 3)))
        self.adjustSize()
        pos = anchor.mapToGlobal(anchor.rect().bottomRight())
        self.move(pos.x() - self.width(), pos.y() + 4)
        self.show()


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
        self.b_recs = QPushButton('Recommended')
        self.b_recs.setCursor(Qt.CursorShape.PointingHandCursor)
        self.b_recs.setFont(sans(10))
        self.b_recs.setToolTip('Recommended actions for this design, worked out on this machine (nothing is sent)')
        self.b_recs.clicked.connect(self.show_recommended)
        h.addWidget(self.b_recs)
        self.b_brief = QPushButton('Brief')
        self.b_brief.setCursor(Qt.CursorShape.PointingHandCursor)
        self.b_brief.setFont(sans(10))
        self.b_brief.clicked.connect(lambda: edit_brief(self.ctl, self))
        h.addWidget(self.b_brief)
        self.b_log = QPushButton('Log')
        self.b_log.setCursor(Qt.CursorShape.PointingHandCursor)
        self.b_log.setFont(sans(10))
        self.b_log.clicked.connect(self.show_log)
        h.addWidget(self.b_log)
        self.log_dialog = None
        self.b_new = tool_button('plus', 'New conversation', size=24, icon_size=14)
        self.b_new.clicked.connect(self.ctl.new_chat)
        self.b_more = tool_button('more', 'More', size=24, icon_size=14)
        m = QMenu(self.b_more)
        m.addAction(icons.icon('report', 16), 'Project brief…', lambda: edit_brief(self.ctl, self))
        m.addAction(icons.icon('history', 16), 'Design log…', self.show_log)
        self.a_export = m.addAction(icons.icon('export', 16), 'Save conversation…', self._export)
        self.a_plans = m.addAction(icons.icon('report', 16), 'Copilot plans…', self.show_plans)
        self.a_sent = m.addAction(icons.icon('eye', 16), 'What was sent…', self.show_sent)
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
        # what the user is doing now: 1–3 recommendations about it
        self.strip = LiveStrip()
        self.strip.act.connect(self.run_cta)
        self.strip.dismiss.connect(lambda r: self.recommender.dismiss(r))
        v.addWidget(self.strip)
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
        self.empty_list: RecList | None = None
        self.consent = None
        self.popup = None
        self._empty_ready = None
        self.recommender = Recommender(ctx, self.ctl, self)
        self.ctl.recommender = self.recommender
        self.recommender.updated.connect(self._recs_updated)
        self.ctl.cta.connect(self.run_cta)
        self.ctl.event.connect(self._event)
        self.ctl.changed.connect(self._sync)
        self.ctl.reset.connect(self._clear)
        self.ctl.agent_changed.connect(lambda *_: self._show_empty() if self.empty is not None else None)
        self.ctx.store.changed.connect(lambda *_: self._brief_btn())
        self.ctx.store.project_replaced.connect(self._brief_btn)
        self.ctl.log_changed.connect(self._brief_btn)
        THEME.changed.connect(self._theme_changed)      # a bound slot: disconnected when the widget goes
        self._restyle()
        self._clear()
        self._recs_updated()

    # ---------------------------------------------------------------- state

    def _theme_changed(self, *_):
        self._restyle()
    def _restyle(self):
        t = THEME.tok
        self.hairline.setStyleSheet(f'background: {t["hair"]};')
        self.strip._restyle()
        self._recs_chip()
        self.area.setStyleSheet(f'QScrollArea {{ background: {t["panel"]}; }}')
        self._sync()

    def _sync(self):
        c = self.ctl
        info = c.info()
        who = PR.LABEL[c.provider]
        if c.auto:
            ok = RT.usable(c._keys(), PR.claude_available())
            who = ' + '.join(PR.LABEL[p] for p in PR.PROVIDERS if ok[p]) or who
            name = 'Automatic'
        else:
            name = info.label
        if not c.ready:
            col, text = THEME.tok['muted'], 'Not set up' if c.available else f'{who}: the SDK is not installed'
        elif c.busy:
            r = c.last_route
            col, text = THEME.color('warning').name(), (f'{PR.model_info(r.provider, r.model).label}' if r is not None
                                                        else name) + ' · working…'
        else:
            col, text = THEME.color('good').name(), f'{name} · {who}'
        self.dot.setText(f'<span style="color:{col}">●</span>')
        self.title.setText(text)
        self.a_export.setEnabled(bool(c.transcript()))
        self.a_sent.setEnabled(bool(c.last_sent()))
        self._brief_btn()
        line = c.usage_line()
        cap = c.daily_cap
        today = f' · today {c.tokens_today() / 1000:.1f}k of {cap / 1000:.0f}k tokens' if cap else ''
        self.foot.setText((f'Answers are computed from Vantage\'s analysis by a model hosted by {who} and can be wrong; '
                           'check a proposal\'s effect before applying it, and a plan before approving it.' if c.ready
                           else PR.INSTALL if not c.available else
                           f'The assistant is off until you add a {who} API key in Preferences ▸ Assistant.')
                          + (f' · {line}' if line else '') + today)
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
        n = len(self.ctl.log())
        self.b_log.setText(f'Log ({n})' if n else 'Log')
        self.b_log.setToolTip('Design log: decisions, rejected ideas, constraints and notes kept with the project; '
                              'the assistant reads it with every question')
        self.b_log.setStyleSheet(f'QPushButton {{ border: 1px solid {t["strong"] if n else t["hair"]}; '
                                 f'border-radius: 10px; padding: 1px 9px; min-height: 18px; '
                                 f'color: {t["ink"] if n else t["muted"]}; background: transparent; }}'
                                 f'QPushButton:hover {{ border-color: {t["ink"]}; }}')

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
        self.empty_list = None
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
                      'apply one. Ask for a run and the Copilot plans it: nothing runs until you approve the plan. '
                      'Type / for commands, or switch on Agent for goals.', ink2=True, wrap=True)
            v.addWidget(s)
            recs = self.recommender.recs if getattr(self, 'recommender', None) is not None else []
            v.addSpacing(4)
            if recs:
                v.addWidget(label('Recommended for this design', bold=True))
                self.empty_list = RecList(RC.SHOWN)
                self.empty_list.act.connect(self.run_cta)
                self.empty_list.dismiss.connect(lambda r: self.recommender.dismiss(r))
                self.empty_list.set_recs(self._below_strip())
                v.addWidget(self.empty_list)
            else:                               # the static starters, when nothing is recommended
                n = len(self.ctx.analysis.issues)
                items = [(f'Explain the {n} issue' + ('s' if n != 1 else ''), prompt) if text == 'Explain the issues'
                         else (text, prompt) for text, prompt in SUGGESTIONS if n or text != 'Explain the issues']
                chips = FlowChips(items)
                chips.picked.connect(self._send)
                v.addWidget(chips)
        else:
            t = QLabel('Connect the assistant')
            t.setFont(sans(15, QFont.Weight.DemiBold))
            v.addWidget(t)
            who = PR.LABEL[self.ctl.provider]
            v.addWidget(label('Ask questions about this design in plain language: why ground is blind, which camera '
                              'matters most, what a change would do; or ask for a run ("a 10 s drive in Town10 at '
                              'dusk in rain") and the Copilot plans it for your approval. The assistant runs on a '
                              f'cloud model, Groq or Claude (Preferences ▸ Assistant), with your own API key.',
                              ink2=True, wrap=True))
            v.addWidget(label(f'When you ask a question, sensor poses, lens, module and sensor model names, analysis '
                              f'figures, plan settings, the fleet\'s vehicle names and offsets (no VINs) and your '
                              f'requirements and the design log are sent to {who}; with Claude also the pictures the '
                              f'assistant draws from the analysis. The vehicle model file, photos, recordings, point '
                              f'clouds, keys, stream passwords and files are never sent.', muted=True, small=True, wrap=True))
            if not self.ctl.available:
                v.addWidget(label(PR.INSTALL, small=True, wrap=True))
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
        self._stick = False                     # the welcome reads from its top; an answer sticks to the bottom again
        self.area.verticalScrollBar().setValue(0)

    # ---------------------------------------------------------------- conversation
    def ask(self, text: str):
        """Ask from outside (the Issues list, the command palette)."""
        self._send(text)

    def _send(self, text, agent=None, force=None):
        text = text.strip()
        if not text:
            return
        if not self.ctl.ready:
            open_settings(self.ctx)
            return
        if self.ctl.busy:
            return
        prompt, slash_agent, local = expand_slash(text)
        agent = slash_agent if agent is None else agent
        route = self.ctl.route_for(prompt or text, agent, force) if local is None else None
        why = self.ctl.blocked(route.provider if route is not None else None) if local is None else ''
        if why == 'consent':
            if self.consent is None or not self.consent.isVisible() or self.consent.provider != route.provider:
                if self.consent is not None and self.consent.isVisible():
                    self.consent.hide()
                self.consent = ConsentCard(self.ctl, lambda t=text, a=agent, f=force: self._send(t, a, f),
                                           provider=route.provider)
                self._add(self.consent)
            return
        if why == 'cap':
            n = Notice('cap', f'Today\'s token cap ({self.ctl.daily_cap:,} tokens, Preferences ▸ Assistant) is used up: '
                              'nothing was sent.')
            n.settings.connect(lambda: open_settings(self.ctx))
            self._add(n)
            return
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
            self.empty_list = None
        if self.composer.edit.toPlainText().strip() == text:
            self.composer.take()
        self._add(UserTurn(text))
        turn = AssistantTurn(self.ctl)
        turn.retry.connect(lambda: self._send(self.ctl.last_question))
        turn.settings.connect(lambda: open_settings(self.ctx))
        turn.reask.connect(lambda r, t=text, a=agent: self._send(t, a, r))
        self._add(turn)
        self.cur = turn
        self._stick = True
        if self.composer.popup is not None:
            self.composer.popup.hide()
        if not self.ctl.ask(prompt, agent=agent, force=force):
            turn.on(A.Event('error', {'kind': 'internal', 'message': 'The assistant is busy or not set up.'}))
            turn.on(A.Event('done', {'ok': False}))
            self.cur = None

    def _add(self, w):
        self.col.insertWidget(self.col.count() - 1, w)

    # ---------------------------------------------------------------- recommendations
    def _recs_chip(self):
        t = THEME.tok
        rc = getattr(self, 'recommender', None)
        n = len(rc.recs) if rc is not None else 0
        self.b_recs.setText(f'Recommended ({n})' if n else 'Recommended')
        self.b_recs.setStyleSheet(f'QPushButton {{ border: 1px solid {t["strong"] if n else t["hair"]}; '
                                  f'border-radius: 10px; padding: 1px 9px; min-height: 18px; '
                                  f'color: {t["ink"] if n else t["muted"]}; background: transparent; }}'
                                  f'QPushButton:hover {{ border-color: {t["ink"]}; }}')

    def _recs_updated(self):
        rc = self.recommender
        self._recs_chip()
        items = rc.strip()
        self.strip.set_items(items, RC.context_label(rc.context, items))
        if self.empty_list is not None and not rc.recs:
            self._show_empty()                  # every one dismissed: the static starters again
        elif self.empty_list is not None:
            self.empty_list.set_recs(self._below_strip())
        elif self.empty is not None and self.ctl.ready and not self.ctl.agent_mode and rc.recs:
            self._show_empty()                  # the first recommendations replace the starters
        if self.popup is not None and self.popup.isVisible():
            self.popup.list.set_recs(rc.recs)

    def _below_strip(self) -> list:
        """The list under the strip leaves out what the strip already shows (the header's list keeps every one)."""
        rc = self.recommender
        shown = {r.id for r in rc.strip()} if self.strip.isVisible() or rc.strip() else set()
        return [r for r in rc.recs if r.id not in shown]

    def show_recommended(self):
        """The header chip: every recommendation, any time (mid-conversation too)."""
        self.recommender.refresh()
        if self.popup is None:
            self.popup = RecPopup(self)
            self.popup.list.act.connect(lambda r: (self.popup.hide(), self.run_cta(r)))
            self.popup.list.dismiss.connect(lambda r: self.recommender.dismiss(r))
        self.popup.list.set_recs(self.recommender.recs)
        self.popup.open_at(self.b_recs)
        return self.popup

    def run_cta(self, rec: RC.Recommendation):
        """What a recommendation's button does: ask and goal send its prompt as typing would (consent and the daily cap
        apply), plan sends a planning request (the plan card needs approval), open navigates. Nothing runs by itself."""
        if rec.kind == 'open':
            return self.open_target(rec.target or {})
        if rec.kind not in ('ask', 'goal', 'plan') or not rec.prompt:
            return False
        w = self.ctx.window
        if w is not None and hasattr(w, 'show_assistant') and not self.isVisible():
            w.show_assistant()
        self._send(rec.prompt, agent=rec.kind == 'goal')
        return True

    def open_target(self, t: dict) -> bool:
        """Open a page, a panel or a plan card, or point at a button: shows, never runs or changes anything."""
        w = self.ctx.window
        if not isinstance(t, dict) or w is None:
            return False
        if t.get('plan'):
            for turn in self.findChildren(AssistantTurn):
                for card in turn.copilot_cards:
                    if card.plan.id == t['plan'] and card.isVisible():
                        self.area.ensureWidgetVisible(card, 0, 40)
                        self._stick = False
                        return True
            self.show_plans()
            return True
        if t.get('undo') is not None:
            st = self.ctx.store
            if hasattr(w, 'docks') and hasattr(w, 'bottom'):
                w.docks['bottom'].show()
                w.bottom.show_tab('history')
            what = st.undo.undoText() or t.get('text', '')
            self.ctx.msg(f'Edit ▸ Undo (Ctrl+Z) would undo: {what}. Nothing was undone.', 8000)
            return True
        page = t.get('page')
        if page and hasattr(w, 'set_page'):
            w.set_page(page)
            pg = w.page(page) if hasattr(w, 'page') else None
            run = t.get('run')
            if run and pg is not None and hasattr(pg, 'open_run') and not pg.busy and \
                    os.path.normpath(pg.out_dir or '') != os.path.normpath(run):
                pg.open_run(run)                # shows the recorded drive (reads its run.json); runs nothing
            foc = getattr(pg, t['focus'], None) if t.get('focus') and pg is not None else None
            if foc is not None:
                foc.setFocus(Qt.FocusReason.OtherFocusReason)
                self.ctx.msg(f'{foc.text()}: the page\'s own button does it when you click it.', 6000)
        if t.get('point') == 'analysis':
            b = getattr(w, 'state_btn', None)
            if b is not None:
                b.setFocus(Qt.FocusReason.OtherFocusReason)
            self.ctx.msg('The analysis is out of date: press F5 (Analysis ▸ Run now) to update it.', 6000)
        if t.get('camera'):
            c = self.ctx.store.project.variant.by_tag(t['camera'])
            if c is not None:
                self.ctx.store.select_cams([c.uid])
        return True

    def _event(self, ev):
        if self.cur is not None:
            self.cur.on(ev)
            if ev.kind == 'done':
                self.cur = None
        self._sync() if ev.kind in ('usage', 'done') else None

    def _range(self, lo, hi):
        if self._stick:
            self.area.verticalScrollBar().setValue(hi)

    def show_plans(self):
        d = PlansDialog(self.ctl, self)
        d.show()
        return d

    def show_sent(self):
        d = SentDialog(self.ctl, self)
        d.show()
        return d

    def show_log(self):
        """The design log view (list, add, edit, delete)."""
        if self.log_dialog is None:
            self.log_dialog = DesignLogDialog(self.ctl, self)
        self.log_dialog.refresh()
        self.log_dialog.show()
        self.log_dialog.raise_()
        return self.log_dialog

    def _export(self):
        text = self.ctl.transcript()
        if not text:
            return
        path, _ = QFileDialog.getSaveFileName(self, 'Save conversation', 'assistant.md', 'Markdown (*.md)')
        if path:
            log = M.to_markdown(self.ctx.store.project)
            with open(path, 'w', encoding='utf-8') as f:
                f.write(f'# Vantage assistant · {self.ctx.store.project.name}\n\n{text}' + (f'\n{log}' if log else ''))


def open_settings(ctx):
    from .dialogs.misc import PreferencesDialog
    PreferencesDialog(ctx, ctx.window, tab='assistant').exec()


# ============================================================================================ preferences
class AssistantPrefs(QWidget):
    """Preferences ▸ Assistant: turn it on, the provider (Groq or Claude), each provider's API key (tested), its model,
    the reasoning, Claude's refusal fallback, a daily token cap and the prices the cost estimate uses."""
    _tested = pyqtSignal(object)

    def __init__(self, settings, parent=None):
        super().__init__(parent)
        self.s = settings
        self.keys = {p: K.load(p) for p in PR.PROVIDERS}
        self.new_keys = {p: None for p in PR.PROVIDERS}      # None: unchanged, '': remove
        self.chosen_models = {PR.GROQ: str(self.s.value('assistant/model', C.DEFAULT_MODEL) or C.DEFAULT_MODEL),
                              PR.CLAUDE: str(self.s.value('assistant/claude_model', PR.CLAUDE_DEFAULT) or
                                             PR.CLAUDE_DEFAULT)}
        prov = str(self.s.value('assistant/provider', PR.GROQ) or PR.GROQ)
        self.prov = prov if prov in PR.PROVIDERS else PR.GROQ
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
        self.provider = SegmentedControl([(PR.GROQ, 'Groq'), (PR.CLAUDE, 'Claude')], self.prov)
        self.provider.setToolTip('The cloud model the assistant and the Copilot use; each has its own key and models')
        self.provider.changed.connect(self._provider_changed)
        row('Provider', self.provider)
        self.sdk = label('', small=True, wrap=True)
        g.addWidget(self.sdk, r, 1, 1, 2)
        r += 1
        self.fresh = not self.s.contains('assistant/provider') and not self.s.contains('assistant/auto')
        self.auto = QCheckBox('Automatic: choose the model for each request')
        self.auto.setChecked(self.s.value('assistant/auto', False, type=bool) if self.s.contains('assistant/auto')
                             else RT.default_auto(self.fresh, {p: k[0] for p, k in self.keys.items()}))
        self._auto_touched = False
        self.auto.toggled.connect(self._auto_toggled)
        row('Model choice', self.auto)
        self.auto_note = label('Quick questions go to a fast model (GPT-OSS 120B on Groq, or Claude Sonnet 5.5 without '
                               'a Groq key); goals, Copilot plans, whole-rig optimisation and questions that need a '
                               'picture go to Claude Opus 5.5 (or the best model you have a key for, said once). '
                               'Decided on this machine from the request; each answer names its model and why, and '
                               'can be re-asked with the other. Each provider\'s key, consent and the daily cap apply '
                               'to each request; switching provider mid-conversation sends a short summary of it, '
                               'never the other provider\'s history.', muted=True, small=True, wrap=True)
        g.addWidget(self.auto_note, r, 1, 1, 2)
        r += 1
        kw = QWidget()
        kh = QHBoxLayout(kw)
        kh.setContentsMargins(0, 0, 0, 0)
        kh.setSpacing(6)
        self.key_edit = QLineEdit()
        self.key_edit.setEchoMode(QLineEdit.EchoMode.Password)
        self.key_edit.setFont(mono(11))
        self.key_edit.textEdited.connect(self._edited)
        self.eye = tool_button('eye', 'Show the key', checkable=True, size=24, icon_size=14)
        self.eye.toggled.connect(lambda on: self.key_edit.setEchoMode(QLineEdit.EchoMode.Normal if on else
                                                                      QLineEdit.EchoMode.Password))
        self.test = button('Test', None, None, 'Check the key with the provider and load the models it may use')
        self.test.clicked.connect(self._test)
        kh.addWidget(self.key_edit, 1)
        kh.addWidget(self.eye)
        kh.addWidget(self.test)
        row('API key', kw)
        self.key_note = label('', muted=True, small=True, wrap=True)
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
        self.effort_note = label('', muted=True, small=True, wrap=True)
        g.addWidget(self.effort_note, r, 1, 1, 2)
        r += 1
        self.show_r = QCheckBox("Show the model's reasoning in the answers")
        self.show_r.setChecked(self.s.value('assistant/show_reasoning', True, type=bool))
        g.addWidget(self.show_r, r, 1, 1, 2)
        r += 1
        self.fallbacks = QCheckBox('When Claude declines a request, retry it on Anthropic\'s recommended model '
                                   '(server-side fallback)')
        self.fallbacks.setChecked(self.s.value('assistant/claude_fallbacks', True, type=bool))
        self.fallbacks.setToolTip('fallbacks: "default" (beta server-side-fallback-2026-07-01): a request a safety '
                                  'classifier declines is re-run on the model Anthropic recommends for that category')
        g.addWidget(self.fallbacks, r, 1, 1, 2)
        r += 1
        cw = QWidget()
        ch = QHBoxLayout(cw)
        ch.setContentsMargins(0, 0, 0, 0)
        ch.setSpacing(6)
        self.cap = ScrubSpinBox('', 'plain', 0, 'k tokens', step=10.0, width=130)
        self.cap.setRange(0, 100000)
        try:
            cap = int(self.s.value('assistant/daily_cap', 0) or 0)
        except (TypeError, ValueError):
            cap = 0
        self.cap.set_value(cap / 1000.0)
        ch.addWidget(self.cap)
        ch.addWidget(label('a day, all providers (0: no cap)', muted=True, small=True), 1)
        row('Daily cap', cw)
        pw = QWidget()
        ph = QHBoxLayout(pw)
        ph.setContentsMargins(0, 0, 0, 0)
        ph.setSpacing(6)
        self.price_in = ScrubSpinBox('', 'plain', 3, '$ in', step=0.01, width=110)
        self.price_out = ScrubSpinBox('', 'plain', 3, '$ out', step=0.01, width=110)
        for w, k in ((self.price_in, 'assistant/groq_price_in'), (self.price_out, 'assistant/groq_price_out')):
            w.setRange(0, 1000)
            try:
                w.set_value(float(self.s.value(k, 0) or 0))
            except (TypeError, ValueError):
                w.set_value(0.0)
            ph.addWidget(w)
        ph.addWidget(label('per million tokens, for the cost line (0: not estimated)', muted=True, small=True), 1)
        row('Groq prices', pw)
        self.price_note = label('Claude\'s prices are built in: ' + '; '.join(
            f'{PR.CLAUDE_MODELS[m].label} ${p["input"]:g} in, ${p["output"]:g} out, cache reads ${p["cache_read"]:g}'
            for m, p in PR.PRICES.items()) + f' per million tokens ({PR.PRICE_SOURCE[PR.CLAUDE]}).',
            muted=True, small=True, wrap=True)
        g.addWidget(self.price_note, r, 1, 1, 2)
        r += 1
        g.addWidget(label('What is sent', ink2=True), r, 0, Qt.AlignmentFlag.AlignTop)
        self.sent_note = label('', muted=True, small=True, wrap=True)
        g.addWidget(self.sent_note, r, 1, 1, 2)
        r += 1
        g.setRowStretch(r, 1)
        g.setColumnStretch(1, 1)
        self._tested.connect(self._on_tested)
        self._provider_changed(self.prov)

    def _auto_toggled(self, on):
        self._auto_touched = True

    # ---------------------------------------------------------------- the provider shown
    @property
    def key(self):
        return self.keys[self.prov][0]

    @property
    def source(self):
        return self.keys[self.prov][1]

    @property
    def _new_key(self):
        return self.new_keys[self.prov]

    def _provider_changed(self, prov):
        if self.model.count():
            self.chosen_models[self.prov] = self.model.currentData() or self.chosen_models[self.prov]
        self.prov = prov if prov in PR.PROVIDERS else PR.GROQ
        claude = self.prov == PR.CLAUDE
        self.key_edit.clear()
        self.key_edit.setPlaceholderText(PR.KEY_HINT[self.prov])
        self.status.setText('')
        ok = PR.claude_available()
        self.sdk.setVisible(claude)
        self.sdk.setText(('<span style="color:%s">●</span> Anthropic\'s Python SDK is installed.' %
                          THEME.color('good').name()) if ok else
                         ('<span style="color:%s">●</span> %s' % (THEME.color('warning').name(), html.escape(PR.INSTALL))))
        self.fallbacks.setVisible(claude)
        self.test.setEnabled(not claude or ok)
        who = PR.LABEL[self.prov]
        pics = ('With Claude also the pictures the assistant draws from the analysis with its look tool (the coverage '
                'map, a camera\'s synthetic view; shown in the conversation as what the model saw), inside a '
                'question you send, like any other tool result. ') if claude else \
            ('Pictures need Claude: Groq\'s models here do not take images, so on Groq the assistant works from '
             'the figures. ')
        self.sent_note.setText(f'With each question: your words, camera poses, lens and module names, plan settings, '
                               f'the design log and the analysis figures the assistant looks up (each lookup is shown '
                               f'in the conversation; the panel\'s More ▸ What was sent shows the last request). '
                               f'{pics}Never the vehicle model file, photos, recordings, point clouds, keys, stream '
                               f'passwords, the automation command or this computer\'s folders. {who} processes them '
                               f'under its own terms; the first question to each provider asks once.')
        self._fill_models()
        self._key_ui()

    def _key_ui(self):
        a = f'style="color:{THEME.tok["ink2"]}"'
        link = f'<a {a} href="{PR.KEYS_URL[self.prov]}">Create a key ({PR.LABEL[self.prov]}) ↗</a>'
        env = K.env_name(self.prov)
        if self._new_key == '':
            self.key_edit.clear()
            self.key_note.setText('The key will be removed when you apply. · ' + link)
        elif self.source == 'environment':
            self.key_edit.setPlaceholderText(f'Set by {env}: {K.mask(self.key)}')
            self.key_note.setText(f'The {env} environment variable takes precedence over a saved key. · ' + link)
        elif self.key and self._new_key is None:
            self.key_edit.setPlaceholderText(f'Saved: {K.mask(self.key)}  (type to replace)')
            self.key_note.setText(f'Saved {K.sources(self.prov).get(self.source, "")}. · '
                                  f'<a {a} href="remove">Remove the key</a> · ' + link)
        else:
            self.key_edit.setPlaceholderText(PR.KEY_HINT[self.prov])
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
            self.new_keys[self.prov] = ''
            self._key_ui()
        else:
            QDesktopServices.openUrl(QUrl(href))

    def _edited(self, text):
        self.new_keys[self.prov] = text.strip() or None
        self.status.setText('')

    def current_key(self):
        if self._new_key == '':
            return None
        return self._new_key or self.key

    def _fill_models(self, ids=None):
        cur = self.chosen_models[self.prov]
        if self.prov == PR.CLAUDE:
            ids = ids or list(PR.CLAUDE_MODELS)
        else:
            v = self.s.value('assistant/models', [])
            ids = ids or ([v] if isinstance(v, str) else list(v or [])) or list(C.KNOWN)
        if cur not in ids and (self.prov == PR.GROQ or cur in PR.CLAUDE_MODELS):
            ids = [cur] + list(ids)
        ids = sorted(dict.fromkeys(ids), key=lambda i: (PR.model_info(self.prov, i).rank, i))
        self.model.blockSignals(True)
        self.model.clear()
        for i in ids:
            info = PR.model_info(self.prov, i)
            self.model.addItem(f'{info.label}  —  {info.note}' if info.note else info.label, i)
        self.model.setCurrentIndex(max(0, self.model.findData(cur)))
        self.model.blockSignals(False)
        self._model_changed()

    def _model_changed(self, *_):
        mid = self.model.currentData() or PR.default_model(self.prov)
        self.chosen_models[self.prov] = mid
        info = PR.model_info(self.prov, mid)
        claude = self.prov == PR.CLAUDE
        opts = [(e, e.capitalize()) for e in info.efforts] if info.efforts and not claude else [('medium', 'Medium')]
        cur = self.effort.value()
        lay = self.layout()
        new = SegmentedControl(opts, cur if cur in dict(opts) else (info.default_effort or opts[0][0]))
        lay.replaceWidget(self.effort, new)
        self.effort.hide()
        self.effort.setParent(None)
        self.effort.deleteLater()
        self.effort = new
        new.setEnabled(bool(info.efforts) and not claude)
        new.setVisible(not claude)
        new.setToolTip('' if info.efforts else 'This model does not take a reasoning effort')
        self.effort_note.setVisible(claude)
        self.effort_note.setText('Adaptive thinking (always on): effort high when the Copilot plans a run, medium '
                                 'for questions.')

    def _test(self):
        key = self.current_key()
        if not key:
            self.status.setText('Enter a key first.')
            return
        self.test.setEnabled(False)
        prov = self.prov
        self.status.setText(f'Checking the key with {PR.LABEL[prov]}…')

        def work():
            try:
                if prov == PR.CLAUDE:
                    import anthropic
                    from ..assistant import claude as CL
                    cl = anthropic.Anthropic(api_key=key, base_url=CL.base_url(), timeout=20.0, max_retries=0)
                    ids = [m.id for m in cl.models.list()]
                    ids = [i for i in ids if i in PR.CLAUDE_MODELS] or ids
                else:
                    ids = C.Client(key, timeout=20).models()
                self._emit(('ok', prov, ids))
            except C.ApiError as e:
                self._emit(('err', prov, e.message))
            except Exception as e:
                msg = str(e)
                try:
                    from ..assistant import claude as CL
                    msg = CL.map_error(e).message if CL.anthropic is not None and \
                        isinstance(e, CL.anthropic.AnthropicError) else msg
                except Exception:                           # noqa: BLE001
                    pass
                self._emit(('err', prov, msg))
        threading.Thread(target=work, daemon=True).start()

    def _emit(self, v):
        try:
            self._tested.emit(v)
        except RuntimeError:
            pass

    def _on_tested(self, v):
        self.test.setEnabled(True)
        kind, prov, val = v
        if prov != self.prov:
            return
        if kind == 'ok':
            col = THEME.color('good').name()
            self.status.setText(f'<span style="color:{col}">●</span> Connected · {len(val)} chat models available')
            if val and prov == PR.GROQ:
                self.s.setValue('assistant/models', val)
                self._fill_models(val)
            elif val:
                self._fill_models([i for i in val if i in PR.CLAUDE_MODELS] or None)
        else:
            col = THEME.color('critical').name()
            self.status.setText(f'<span style="color:{col}">●</span> {html.escape(val)}')

    def apply(self):
        self.chosen_models[self.prov] = self.model.currentData() or self.chosen_models[self.prov]
        self.s.setValue('assistant/enabled', self.on.isChecked())
        if self._auto_touched or self.s.contains('assistant/auto'):
            auto = self.auto.isChecked()
        else:                              # untouched on a new installation: on when both keys exist after this
            after = {p: (None if self.new_keys[p] == '' else self.new_keys[p] or self.keys[p][0])
                     for p in PR.PROVIDERS}
            auto = RT.default_auto(self.fresh, after)
        self.s.setValue('assistant/auto', bool(auto))
        self.s.setValue('assistant/provider', self.prov)
        self.s.setValue('assistant/model', self.chosen_models[PR.GROQ] or C.DEFAULT_MODEL)
        self.s.setValue('assistant/claude_model', self.chosen_models[PR.CLAUDE] or PR.CLAUDE_DEFAULT)
        if self.prov == PR.GROQ:
            self.s.setValue('assistant/effort', self.effort.value())
        self.s.setValue('assistant/show_reasoning', self.show_r.isChecked())
        self.s.setValue('assistant/claude_fallbacks', self.fallbacks.isChecked())
        self.s.setValue('assistant/daily_cap', int(round(self.cap.value() * 1000)))
        self.s.setValue('assistant/groq_price_in', float(self.price_in.value()))
        self.s.setValue('assistant/groq_price_out', float(self.price_out.value()))
        for p, k in self.new_keys.items():
            if k == '':
                K.delete(p)
            elif k:
                K.save(k, p)
