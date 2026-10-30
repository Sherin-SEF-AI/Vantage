"""65 The design log (episode as070, chapter 5): the sample's data rig, the REAL model (GPT-OSS 120B on Groq, unedited
replies). The move of chapter 2 (F 30 mm forward and 30 mm up) asked for as a proposal and discarded: the design log
records a "rejected" entry by itself (Log (1)); asked again for another way to close the patch ahead of the bumper,
the model does not offer the rejected move (checked: no proposal of the second answer is flagged by the tools as
repeating the rejected entry; the transcript is read after the take for the words); the Design log view.

Narrated and checked with r.expect while recording: the reason given for turning the move down is the app's own
issue for it (simulate_changes on the same change, run locally: "Routed cable ... is longer than the 3.00 m module
cable"); the log holds one rejected entry about F after Discard.
"""

META = {
    'id': 'as-log',
    'part': 2,
    'title': 'The design log',
    'promise': 'What was decided and turned down, kept with the project.',
    'menu': 'Assistant panel ▸ Log  ·  More ▸ Design log…',
}

LINES = [
    # 0
    ("Suppose the team turns down the move from before: the cable to the front camera would be longer than the "
     "module's own. Ask for it as a proposal…",
     "Suppose the team turns down the move from before: the cable to the front camera would be longer than the "
     "module's own. Ask for it as a proposal,"),
    # 1
    ("…and discard it. The design log records it by itself, as rejected, with the exact change, in the project file.",
     "and discard it. The design log records it by itself, as rejected, with the exact change, in the project file."),
    # 2
    ("The model reads the log with every question, even in a new conversation. Ask again for a way to close the "
     "patch: the rejected move is not offered again.",
     "The model reads the log with every question, even in a new conversation. Ask again for a way to close the "
     "patch: the rejected move is not offered again."),
    # 3
    ("The Log button opens the log itself: decisions, constraints, rejected ideas and notes, each edit one undo step.",
     "The log button opens the log itself: decisions, constraints, rejected ideas, and notes, each edit one undo "
     "step."),
]

Q1 = 'Propose moving F 3 cm forward and 3 cm up. One sentence.'
Q2 = 'Suggest another fix for the blind patch ahead of the front bumper, checked with the tools, in one sentence.'


def scene(stage, r):
    import helpers_as as H
    H.groq_on()
    try:
        _scene(stage, r, H)
    finally:
        H.groq_off()


def _scene(stage, r, H):
    from vantage.assistant import memory as M
    from vantage.assistant import tools as T
    from vantage.ui.assistant import ProposalCard

    w = H.window(stage)
    r.win = w
    chat = H.Chat(stage, r, w, 'as-log')
    panel = w.assistant_panel
    ctl = w.ctx.assistant
    st = w.store
    # the reason, from the app itself: the same change simulated locally (no model involved)
    sim = T.BY_NAME['simulate_changes'].fn(ctl._context(), changes=[{'camera': 'F', 'dx': 0.03, 'dz': 0.03}])
    cable = [s for s in sim['issues']['new'] if s.startswith('Routed cable') and 'longer than the 3.00 m module' in s]
    print('the move\'s new issues:', sim['issues']['new'], flush=True)
    r.expect(bool(cable), True, 'the move makes the front camera\'s cable too long (the app\'s own issue)')
    vp = stage.vp
    vp.cam.az, vp.cam.el = 24.0, 22.0
    w.viewarea.frame_all()
    stage.settle(2)
    r.cursor_on = False
    try:
        r.lower_third()
        r.note(H.NOTE)
        r.zoom(H.chat_rect(r, w, 620), 0.01)
        # ---- 0: the proposal again
        r.say(0)
        r.hold(0.5)
        r.cursor_on = True
        chat.type_in(Q1, cps=28)
        chat.send()
        r.cursor_on = False
        chat.wait(timeout=120)
        cards = [c for c in panel.findChildren(ProposalCard) if c.isVisible()]
        if not cards:
            raise RuntimeError(f'no proposal card: {[c[0] for c in chat.calls()]}')
        card = cards[-1]
        print('proposal', card.prop.title, card.prop.rows, flush=True)
        H.show_widget(r, panel, card, margin=12)
        r.finish_speech(0.1)

        # ---- 1: discard -> a rejected entry
        r.say(1)
        r.cursor_on = True
        r.click(card.b_discard, None, 0.6)
        stage.settle(1)
        rej = [e for e in M.entries(st.project) if e['kind'] == 'rejected']
        print('log:', M.entries(st.project), flush=True)
        r.expect((len(rej), [st.variant.cam(u).tag for u in rej[0]['about']]), (1, ['F']), 'one rejected entry about F')
        r.cursor_on = False
        r.until_word(1, 'records it')
        r.zoom(lambda: r.wrect(panel.b_log).adjusted(-420, -14, 40, 330), 0.7, margin=10)
        r.callout(panel.b_log, '', dur=max(1.8, r.speech_left() - 0.2), pad=3, dim=0.25)
        r.finish_speech(0.1)

        # ---- 2: ask again: not re-suggested
        r.say(2)
        r.zoom(H.chat_rect(r, w, 620), 0.7, wait=False)
        r.cursor_on = True
        r.until_word(2, 'even in a new')
        chat.dump('part1')                          # the first conversation's transcript, before it is cleared
        r.click(panel.b_new, None, 0.5)             # a new conversation: only the design log remembers
        stage.settle(1)
        n0 = len(chat.calls())
        chat.type_in(Q2, cps=30)
        chat.send()
        r.cursor_on = False
        chat.wait(timeout=240, speed=3.0)
        later = chat.calls()[n0:]
        print('second turn calls:', [(n, a) for n, a, _ in later], flush=True)
        repeats = [(n, a) for n, a, res in later if n in ('simulate_changes', 'propose_changes')
                   and isinstance(res, dict) and res.get('design_log')]
        print('second-turn calls the tools flagged as repeating the rejected change:', repeats, flush=True)
        r.expect([n for n, _ in repeats if n == 'propose_changes'], [], 'no proposal repeats the rejected change')
        import re
        ans = chat.answer()
        if re.search(r'\b(F|front camera)\b[^.]*\b(forward|up|raise|higher)', ans, re.I):
            raise RuntimeError(f'the answer seems to offer the rejected move again: {ans[:300]}')
        turn = chat.turn()
        H.read_through(r, panel, turn, max(2.5, r.speech_left() + 0.8), lead=8)
        r.finish_speech(0.1)

        # ---- 3: the Design log view
        r.say(3)
        r.cursor_on = True
        r.click(panel.b_log, None, 0.6)
        dlg = panel.log_dialog
        stage.extra.append(dlg)
        stage.settle(1)
        r.cursor_on = False
        r.zoom(lambda: r.wrect(dlg).adjusted(-20, -20, 20, 20) if dlg.isVisible() else None, 0.9, margin=10)
        r.hold(max(1.5, r.speech_left() + 0.4))
        r.finish_speech(0.2)
        dlg.hide()
        chat.dump('ok')
        r.zoom(None, 0.7)
        r.note(None)
        r.hold(0.6)
    except Exception:
        chat.dump('failed')
        raise
