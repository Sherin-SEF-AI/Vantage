"""61 Ask the design (episode as070, chapter 1): Window ▸ Assistant on the sample's data rig (six cameras, a roof lidar
OS1-128, five radars), talking to the REAL model, GPT-OSS 120B on Groq (the user's own key, approved for this
recording; see helpers_as.py): the replies are the model's, unedited, and the note on screen says so. A fresh profile:
the first question shows the consent card, which is accepted on camera. "Where is this rig blind, and why?" ->
find_blind_regions (live) -> the model's answer; the lookup opened; a follow-up about a point behind the car ->
explain_point (live); the footer's token count and daily cap.

Figures narrated, read from the live tool results while recording and checked with r.expect: blind ground 0.24 m², the
largest patch 0.23 m² ahead of the front bumper, F's view of it "hidden by radar RD1's housing". The tiles show the
tool results (blind total; the best density at the point behind the car, from explain_point). The model's own wording
is not narrated; if its answer contradicts the tools, the take is redone.
"""

META = {
    'id': 'as-ask',
    'part': 1,
    'title': 'Ask the design',
    'promise': 'Plain questions, answered from the app’s own analysis.',
    'menu': 'Window ▸ Assistant  ·  Ctrl+J',
}

LINES = [
    # 0
    ("Window ▸ Assistant opens a chat beside the design: here the sample's data rig, with six cameras, a roof lidar "
     "and five radars.",
     "Window, assistant, opens a chat beside the design: here, the sample's data rig, with six cameras, a roof lidar, "
     "and five radars."),
    # 1
    ("The first question to Groq asks for consent and says what goes: poses, lens and module names, figures. Never "
     "the vehicle model, photos or recordings.",
     "The first question to Grok asks for consent, and says what goes: poses, lens and module names, figures. Never "
     "the vehicle model, photos, or recordings."),
    # 2
    ("The model reads the design only through Vantage's tools, and every lookup is listed as it runs.",
     "The model reads the design only through Vantage's tools, and every lookup is listed as it runs."),
    # 3
    ("The analysis finds 0.24 m² of blind ground. The largest patch, 0.23 m², lies just ahead of the front bumper, "
     "where radar RD1's housing hides it from the front camera.",
     "The analysis finds zero point two four square metres of blind ground. The largest patch, zero point two three, "
     "lies just ahead of the front bumper, where radar R D one's housing hides it from the front camera."),
    # 4
    ("Open a lookup to see exactly what was asked, and what came back.",
     "Open a lookup to see exactly what was asked, and what came back."),
    # 5
    ("A follow-up keeps the context: which camera sees a point behind the car, and how finely.",
     "A follow-up keeps the context: which camera sees a point behind the car, and how finely."),
    # 6
    ("Nothing is sent until you ask. The footer counts today's tokens against the daily cap set in Preferences.",
     "Nothing is sent until you ask. The footer counts today's tokens, against the daily cap set in preferences."),
]

Q1 = "Where is the largest blind patch, and why can't the front camera see it? One sentence."
Q2 = 'Which camera sees the ground 1.5 m behind the rear axle, and at what pixel density? One sentence.'


def scene(stage, r):
    import helpers_as as H
    H.groq_on()
    try:
        _scene(stage, r, H)
    finally:
        H.groq_off()


def _scene(stage, r, H):
    from PyQt6.QtCore import QPointF, QRectF, Qt
    from vantage.ui.assistant import ToolRow

    w = H.window(stage, consent=False)
    r.win = w
    chat = H.Chat(stage, r, w, 'as-ask')
    panel = w.assistant_panel
    rows = w.bottom.coverage.rows
    r.expect(rows['Blind in ROI'].v.text(), '0.240', 'blind ground of the data rig')
    w.docks['assistant'].hide()
    w.docks['bottom'].hide()                   # room for the conversation: the analysis panels are not used here
    vp = stage.vp
    vp.cam.az, vp.cam.el = -38.0, 22.0
    w.viewarea.frame_all()
    stage.settle(2)
    r.cursor_on = False
    try:
        # ---- 0: Window ▸ Assistant
        r.lower_third()
        r.say(0)
        stage.orbit(r, vp, 16, 2.0, d_el=2)
        r.cursor_on = True
        r.menu('Window', 'Assistant')
        w.resizeDocks([w.docks['assistant']], [H.DOCK_W], Qt.Orientation.Horizontal)
        stage.settle(1)
        r.crossfade(0.3)
        r.note(H.NOTE)
        r.hold(0.6)
        r.zoom(H.chat_rect(r, w, 640), 1.0)
        r.finish_speech(0.1)

        # ---- 1: the question; the consent card
        r.say(1)
        chat.type_in(Q1, cps=28)
        chat.send()
        r.hold(0.3)
        card = chat.consent_card()
        if card is None:
            raise RuntimeError('no consent card on the first question')
        H.show_widget(r, panel, card)
        r.callout(card, '', dur=max(2.5, r.speech_left() - 0.4), pad=6, dim=0.22)
        r.until(r.speech_end - 0.6)
        r.click(card.b_ok, None, 0.6)
        r.mark('cold', shot='tool')            # the intro's cold open quotes from here
        r.say(2)
        r.cursor_on = False
        chat.wait(timeout=120)
        trow = next(t for t in panel.findChildren(ToolRow) if t.isVisible() and t.name == 'find_blind_regions')
        r.callout(trow, '', dur=max(1.6, r.speech_left()), pad=4, dim=0.2)
        r.finish_speech(0.1)

        # ---- 3: the answer, the figures from the tool
        call = chat.last('find_blind_regions')
        if call is None:
            raise RuntimeError(f'the model did not call find_blind_regions: {[c[0] for c in chat.calls()]}')
        res = call[2]
        regs = res['blind_regions']
        print('find_blind_regions:', str(res)[:400], flush=True)
        r.expect(res['blind_total_m2'], 0.24, 'blind total (live tool)')
        r.expect((regs[0]['area_m2'], regs[0]['where']), (0.23, 'front'), 'largest patch (live tool)')
        r.expect(regs[0]['per_camera_at_sample']['F'], "hidden by radar RD1's housing", 'why F misses it')
        turn = chat.turn()
        r.say(3)
        r.figure('Blind ground', res['blind_total_m2'], 'm²', 2, dur=3.2, src=trow, note='find_blind_regions · live')
        H.read_through(r, panel, turn, max(3.0, r.speech_left() - 0.3), lead=8)
        r.finish_speech(0.1)

        # ---- 4: open the lookup
        r.say(4)
        H.show_widget(r, panel, trow, margin=10)
        r.cursor_on = True
        r.click(trow, QPointF(trow.width() * 0.35, 14), 0.6)
        stage.settle(1)
        H.scroll_to(r, panel, H.widget_y(panel, trow) - 8, 0.5)
        r.zoom(H.chat_rect(r, w, 640, top=r.wrect(panel.area).y() - 6), 0.8)
        r.hold(max(1.5, r.speech_left() - 0.2))
        r.click(trow, QPointF(trow.width() * 0.35, 14), 0.4)
        stage.settle(1)
        panel._stick = True
        panel.area.verticalScrollBar().setValue(panel.area.verticalScrollBar().maximum())
        r.zoom(H.chat_rect(r, w, 640), 0.8, wait=False)
        r.finish_speech(0.1)

        # ---- 5: a follow-up about one point
        r.say(5)
        chat.type_in(Q2, cps=30)
        r.hold(0.2)
        chat.send()
        r.cursor_on = False
        chat.wait(timeout=120)
        call = chat.last('explain_point')
        if call is None:
            raise RuntimeError(f'the follow-up did not call explain_point: {[c[0] for c in chat.calls()]}')
        ep = call[2]
        print('explain_point:', str(ep)[:300], flush=True)
        r.expect((ep['best_camera'], ep['seen_by']), ('R', ['R']), 'the point behind the car (live tool)')
        trow2 = [t for t in panel.findChildren(ToolRow) if t.isVisible() and t.name == 'explain_point'][-1]
        turn2 = chat.turn()
        r.figure(f'{ep["best_camera"]} at the point', ep['best_density_px_cm'], 'px/cm', 2, dur=3.0, src=trow2,
                 note='explain_point · only R sees it')
        H.read_through(r, panel, turn2, max(2.6, r.speech_left() + 0.6), lead=8)
        r.finish_speech(0.2)

        # ---- 6: only on a request; the footer and the cap
        r.say(6)
        H.kinetic(r, 'Sent only\nwhen you ask.', accent=('ask',), dur=r.line(6) * 0.55, px=52)
        r.until_word(6, 'The footer')
        r.callout(panel.foot, '', dur=max(1.8, r.speech_left() + 0.2), pad=4, dim=0.2)
        r.finish_speech(0.3)
        chat.dump('ok')
        r.zoom(None, 0.9)
        r.note(None)
        r.hold(0.8)
    except Exception:
        chat.dump('failed')
        raise
