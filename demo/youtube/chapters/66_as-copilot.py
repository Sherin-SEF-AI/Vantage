"""66 The Copilot plans a drive (episode as070, chapter 6): the sample's data rig, the REAL model (GPT-OSS 120B on Groq,
unedited reply). A CARLA drive asked for in words -> propose_plan -> the plan card: every value with its source (your
request, a default, the Copilot's choice, a limit), warnings, the estimate. The plan is left a draft: Approve and run
is shown, never clicked. No CARLA server is started or probed on its default ports: the project's CARLA port is set
to 2300 (nothing listens there; nothing is approved, so nothing connects).

Checked with r.expect while recording: the plan is a draft CARLA drive with the requested town, duration, traffic and
ground truth; no plan runner exists at the end (nothing ran). No figure is narrated beyond the request's own words.
"""

META = {
    'id': 'as-copilot',
    'part': 3,
    'title': 'The Copilot plans a drive',
    'promise': 'A run asked for in words becomes a plan you check and approve.',
    'menu': 'Assistant ▸ propose_plan  ·  Approve and run',
}

LINES = [
    # 0
    ("Ask for a run in words, and the Copilot drafts a plan: a 10-second CARLA drive in Town 10 at dusk, in rain, "
     "with 20 cars and ground truth.",
     "Ask for a run in words, and the Copilot drafts a plan: a ten second CARLA drive in town ten at dusk, in rain, "
     "with twenty cars, and ground truth."),
    # 1
    ("Every value says where it came from: your request, a default, or the Copilot's own choice; the card also "
     "estimates the time and the disk.",
     "Every value says where it came from: your request, a default, or the Copilot's own choice; the card also "
     "estimates the time, and the disk."),
    # 2
    ("The plan stays a draft. A run needs your approval: Approve and run checks it again and records it. Here, "
     "nothing runs.",
     "The plan stays a draft. A run needs your approval: Approve and run checks it again, and records it. Here, "
     "nothing runs."),
]

Q = 'Plan a 10 s CARLA drive in Town10HD at dusk in rain with 20 cars and ground truth on. One sentence.'


def scene(stage, r):
    import helpers_as as H
    H.groq_on()
    try:
        _scene(stage, r, H)
    finally:
        H.groq_off()


def _scene(stage, r, H):
    from vantage.ui.assistant import CopilotCard

    def edit(p):
        p.analysis.sim = dict(p.analysis.sim or {}, port=2300, tm_port=8300)   # never CARLA's default ports
    w = H.window(stage, edit=edit)
    r.win = w
    chat = H.Chat(stage, r, w, 'as-copilot')
    panel = w.assistant_panel
    ctl = w.ctx.assistant
    vp = stage.vp
    vp.cam.az, vp.cam.el = -50.0, 18.0
    w.viewarea.frame_all()
    stage.settle(2)
    r.cursor_on = False
    try:
        r.lower_third()
        r.note(H.NOTE)
        r.zoom(H.chat_rect(r, w, 620), 0.01)
        # ---- 0: the request
        r.say(0)
        r.hold(0.5)
        r.cursor_on = True
        chat.type_in(Q, cps=30)
        chat.send()
        r.cursor_on = False
        chat.wait(timeout=180)
        cards = [c for c in panel.findChildren(CopilotCard) if c.isVisible()]
        if not cards:
            raise RuntimeError(f'no plan card: {[c[0] for c in chat.calls()]}')
        card = cards[-1]
        plan = card.plan
        s0 = plan.steps[0]
        print('plan', plan.title, plan.status, s0.kind, s0.settings, flush=True)
        print('sources', s0.sources, flush=True)
        print('estimate', s0.estimate, 'warnings', s0.warnings, 'errors', s0.errors, flush=True)
        st = s0.settings
        r.expect((plan.status, s0.kind), ('draft', 'carla_run'), 'a draft CARLA drive')
        r.expect((float(st.get('duration_s', 0)), str(st.get('town', '')).startswith('Town10')), (10.0, True),
                 'duration and town as asked')
        srcs = {k: v.get('source') for k, v in s0.sources.items()}
        r.expect((srcs.get('town'), srcs.get('duration_s'), srcs.get('traffic')), ('request', 'request', 'request'),
                 'the requested values are labelled as your request')
        r.finish_speech(0.1)

        # ---- 1: the values and their sources
        r.say(1)
        H.show_widget(r, panel, card, margin=10)
        r.mark('cold', shot='plan')            # the intro's cold open quotes from here
        r.zoom(lambda: r.wrect(card).adjusted(-20, -20, 20, 0) if r.wrect(card).height() < 560 else
               H.chat_rect(r, w, 620), 0.8)
        H.read_through(r, panel, card, max(3.5, r.speech_left()), lead=6)
        r.finish_speech(0.1)

        # ---- 2: still a draft; the approval gate (shown, never clicked)
        r.say(2)
        panel.area.ensureWidgetVisible(card.b_approve, 0, 60)
        r.tick(2)
        r.zoom(H.chat_rect(r, w, 620), 0.6)
        H.kinetic(r, 'Still a draft.\nA run needs approval.', accent=('approval.',), dur=3.4, px=48)
        r.cursor_on = True
        r.move_to(card.b_approve, None, 0.8)
        r.cursor_on = False
        r.callout(card.b_approve, '', dur=max(2.0, r.speech_left()), pad=4, dim=0.25)
        r.finish_speech(0.6)
        r.expect((plan.status, ctl.runner is None or not ctl.runner.running), ('draft', True), 'nothing ran')
        chat.dump('ok')
        r.zoom(None, 0.8)
        r.note(None)
        r.hold(0.5)
    except Exception:
        chat.dump('failed')
        raise
