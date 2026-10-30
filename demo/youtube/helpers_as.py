"""What the assistant episode's chapters (as070, files 60-69) share: the real Groq key for the recording, the window
set up for a legible chat, asking a question like a user, reading what the tools returned, the zoom on the chat, and a
record of every take.

The key (the user approved real Groq replies for this recording): loaded read-only with the app's own
vantage.assistant.secrets.load('groq') from the user's settings folder, put into GROQ_API_KEY of this recording
process only (studio.py drops it on import), the local name deleted. It is never printed, logged, written, shown or
put into a settings file: the throw-away settings hold the assistant's switches only. groq_off() removes it from the
environment at the end of the scene. Only the app's built-in sample projects are open while it is set. No Claude key
exists in the throw-away settings, so nothing goes to Anthropic.

Every take writes its transcript (the conversation as the app exports it, no key in it) and its usage (requests, tokens)
to $VANTAGE_AS_TAKES (default $VANTAGE_SCRATCH/as070-takes)/<chapter>-<time>.md and .json, so each reply can be read
after the take: a reply that gets a number wrong, invents something or is clumsy means another take, never an edit.
"""
from __future__ import annotations

import json
import os
import pwd
import time

MODEL = 'openai/gpt-oss-120b'
NOTE = 'Real replies from GPT-OSS 120B on Groq, unedited  ·  every tool and figure computed live by Vantage'
UI_SCALE = 1.3          # Preferences ▸ General ▸ UI scale 130 %: the chat legible on a phone once zoomed
DOCK_W = 660            # the assistant panel's width (window pixels)


# ---------------------------------------------------------------- the key
def groq_on():
    """GROQ_API_KEY for this process, from the user's own store (read only)."""
    from vantage.assistant import secrets as K
    for k in ('VANTAGE_ASSISTANT_URL', 'ANTHROPIC_API_KEY'):        # Groq itself; never Anthropic
        os.environ.pop(k, None)
    tmp = os.environ.get('XDG_CONFIG_HOME')
    os.environ['XDG_CONFIG_HOME'] = os.path.join(pwd.getpwuid(os.getuid()).pw_dir, '.config')
    try:
        key, _ = K.load('groq')
    finally:
        if tmp is None:
            os.environ.pop('XDG_CONFIG_HOME', None)
        else:
            os.environ['XDG_CONFIG_HOME'] = tmp
    if not key:
        raise SystemExit('no Groq key in Vantage: this chapter is recorded with real Groq replies only')
    os.environ['GROQ_API_KEY'] = key
    del key
    claude, _ = K.load('claude')                   # the throw-away settings: no Claude key, so no Anthropic call
    if claude:
        raise SystemExit('a Claude key is set: this recording must not reach Anthropic')


def groq_off():
    os.environ.pop('GROQ_API_KEY', None)


def takes_dir():
    d = os.environ.get('VANTAGE_AS_TAKES') or os.path.join(
        os.environ.get('VANTAGE_SCRATCH', os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'build',
                                                       'youtube', 'work')), 'as070-takes')
    os.makedirs(d, exist_ok=True)
    return d


# ---------------------------------------------------------------- the window
def window(stage, edit=None, rig=True, consent=True, auto=False, cap=500_000, dock_w=DOCK_W, scale=UI_SCALE,
           effort='high'):
    """The sample (its data rig by default) at UI scale 130 %, the assistant on Groq · GPT-OSS 120B in a fresh
    throw-away profile (every assistant/* setting cleared first: consent, today's token count), reasoning
    effort high (Preferences ▸ Assistant ▸ Reasoning, chosen for this video), the panel open."""
    from PyQt6.QtCore import Qt
    from vantage.ui.theme import THEME
    THEME.apply(stage.app, 'dark', scale)
    w = stage._rig_window(edit) if rig else stage.window(edit)
    s = w.ctx.settings
    s.setValue('ui/scale', str(scale))
    for k in list(s.allKeys()):
        if k.startswith('assistant/'):
            s.remove(k)
    for k, v in (('enabled', True), ('provider', 'groq'), ('model', MODEL), ('effort', effort),
                 ('show_reasoning', True), ('auto', bool(auto)), ('daily_cap', int(cap))):
        s.setValue(f'assistant/{k}', v)
    if consent:
        s.setValue('assistant/consent_groq', True)
    w.ctx.assistant.reload()
    w.show_assistant()
    w.resizeDocks([w.docks['assistant']], [dock_w], Qt.Orientation.Horizontal)
    stage.settle(3)
    return w


class Chat:
    """Asks the open window's assistant like a user and keeps what happened: the events (errors), the tool calls with
    their results, the usage, and a transcript per take."""

    def __init__(self, stage, r, w, cid):
        self.stage, self.r, self.w, self.cid = stage, r, w, cid
        self.panel = w.assistant_panel
        self.ctl = w.ctx.assistant
        self.errors = []
        self.turns = 0
        self.t0 = time.strftime('%Y%m%d-%H%M%S')
        self.ctl.event.connect(self._on)

    def _on(self, ev):
        if ev.kind == 'error':
            self.errors.append(dict(ev.data))
            print('ASSISTANT ERROR', ev.data, flush=True)

    # -------------------------------------------------------------- asking
    def type_in(self, text, cps=24.0):
        from PyQt6.QtCore import QPointF
        ed = self.panel.composer.edit
        self.r.move_to(ed, QPointF(min(ed.width() * 0.3, 160), ed.height() / 2), 0.5)
        self.r.ripple()
        self.r.type(ed, text, cps)

    def send(self):
        from PyQt6.QtCore import Qt
        from PyQt6.QtTest import QTest
        QTest.keyClick(self.panel.composer.edit, Qt.Key.Key_Return)
        self.r.tick(2)

    def consent_card(self):
        from vantage.ui.assistant import ConsentCard
        cards = [c for c in self.panel.findChildren(ConsentCard) if c.isVisible()]
        return cards[-1] if cards else None

    def wait(self, timeout=240, speed=1.0, after=6.0):
        """Record while the model and the tools work: at `speed` (1 = real time) while the narration plays, then, if
        the work goes on after the words end, as a marked time-lapse at `after` (no long silence). Raises on an API
        error, so a failed take never passes for a good one."""
        n0 = len(self.errors)
        r = self.r
        t0 = time.time()
        if r.speech_left() > 0.3:
            r.wait_for(lambda: not self.ctl.busy or r.speech_left() <= 0.3, timeout=timeout, speed=speed)
        if self.ctl.busy:
            r.wait_for(lambda: not self.ctl.busy, timeout=max(10.0, timeout - (time.time() - t0)),
                       speed=max(speed, after))
        self.turns += 1
        self.stage.app.processEvents()
        if len(self.errors) == n0 and not self.answer().strip() and not self.consent_card():
            self.errors.append({'kind': 'empty', 'message': 'the model ended its turn without an answer'})
        if len(self.errors) > n0:
            self.dump('failed')
            raise RuntimeError(f'the assistant failed: {self.errors[-1]}')
        self.stage.app.processEvents()

    def ask(self, text, cps=24.0, timeout=240, speed=1.0):
        self.type_in(text, cps)
        self.r.hold(0.25)
        self.send()
        self.wait(timeout, speed)
        return self.turn()

    # -------------------------------------------------------------- what happened
    def turn(self):
        from vantage.ui.assistant import AssistantTurn
        turns = [t for t in self.panel.findChildren(AssistantTurn) if t.isVisible()]
        return turns[-1] if turns else None

    def calls(self):
        """[(name, args, result)] of every tool call in this conversation, in order (results parsed from JSON)."""
        s = self.ctl.session
        if s is None:
            return []
        out, by_id = [], {}
        for m in s.messages:
            if m.get('role') == 'assistant':
                for tc in m.get('tool_calls') or []:
                    fn = tc.get('function', {})
                    try:
                        args = json.loads(fn.get('arguments') or '{}')
                    except ValueError:
                        args = {}
                    by_id[tc.get('id')] = len(out)
                    out.append([fn.get('name'), args, None])
            elif m.get('role') == 'tool':
                i = by_id.get(m.get('tool_call_id'))
                if i is not None:
                    try:
                        out[i][2] = json.loads(m.get('content') or 'null')
                    except ValueError:
                        out[i][2] = m.get('content')
        return [tuple(x) for x in out]

    def last(self, name):
        hits = [c for c in self.calls() if c[0] == name]
        return hits[-1] if hits else None

    def answer(self):
        """The text of the latest answer (the model's own words)."""
        t = self.turn()
        return ''.join(t.texts) if t is not None else ''

    def usage(self):
        out = {}
        for p, u in self.ctl.usages().items():
            out[p] = {'model': u.model, 'requests': u.requests, 'prompt_tokens': u.prompt,
                      'completion_tokens': u.completion, 'reasoning_tokens': u.reasoning}
        return out

    def dump(self, tag='ok'):
        """The take's transcript and usage (no key in either), for reading after the take."""
        base = os.path.join(takes_dir(), f'{self.cid}-{self.t0}-{tag}')
        with open(base + '.md', 'w', encoding='utf-8') as fh:
            fh.write(self.ctl.transcript())
        with open(base + '.json', 'w', encoding='utf-8') as fh:
            json.dump({'chapter': self.cid, 'take': self.t0, 'status': tag, 'turns': self.turns,
                       'usage': self.usage(), 'errors': self.errors,
                       'calls': [{'name': n, 'args': a, 'result': res} for n, a, res in self.calls()]}, fh, indent=1,
                      ensure_ascii=False, default=str)
        print(f'take written: {base}.md  usage {self.usage()}', flush=True)
        return base


# ---------------------------------------------------------------- the frame on the chat
def chat_rect(r, w, h=600, top=None, extra_left=None):
    """A 16:9 frame on the assistant panel: its full width plus some of the app to its left (where kinetic captions
    sit), `h` tall (zoom about 1080 / h), its bottom on the composer unless `top` (window y) is given."""
    from PyQt6.QtCore import QRectF
    d = r.wrect(w.docks['assistant'])
    ww = h * 16 / 9
    x = min(d.right() + 2 - ww, d.x() - (extra_left if extra_left is not None else 0))
    x = max(0.0, min(x, 1920 - ww))
    comp = r.wrect(w.assistant_panel.composer)
    y = (top if top is not None else comp.bottom() + 14 - h)
    y = max(0.0, min(y, 1080 - h))
    return QRectF(x, y, ww, h)


def show_widget(r, panel, widget, margin=40, settle_frames=2):
    """Scroll the transcript so `widget` is in view (top first) and stop it sticking to the bottom."""
    panel._stick = False
    panel.area.ensureWidgetVisible(widget, 0, margin)
    r.tick(settle_frames)


def scroll_to(r, panel, value, dur=1.0):
    """Ease the transcript's scroll bar to `value` (pixels) over dur seconds, recording as it goes."""
    from gfx import smoother
    sb = panel.area.verticalScrollBar()
    panel._stick = False
    a = sb.value()
    b = max(sb.minimum(), min(sb.maximum(), int(value)))
    n = max(1, round(dur * r.fps))
    for k in range(1, n + 1):
        sb.setValue(int(round(a + (b - a) * smoother(k / n))))
        r.tick()


def widget_y(panel, widget):
    """The widget's top in the transcript's own coordinates (the scroll bar's value that puts it at the top)."""
    from PyQt6.QtCore import QPoint
    return widget.mapTo(panel.body, QPoint(0, 0)).y()


def read_through(r, panel, widget, secs, lead=10):
    """Show a long answer from its top and scroll down through it slowly over `secs`, ending at the bottom."""
    sb = panel.area.verticalScrollBar()
    top = widget_y(panel, widget) - lead
    scroll_to(r, panel, top, 0.6)
    r.hold(0.6)
    end = sb.maximum()
    if end > sb.value() + 4:
        scroll_to(r, panel, end, max(0.8, secs - 1.2))
    else:
        r.hold(max(0.0, secs - 0.6))


def kinetic(r, text, accent=(), dur=3.0, at='left', px=54, eyebrow=''):
    import gfx_as
    return r.add(gfx_as.Kinetic(text, dur, at, px, accent, eyebrow))
