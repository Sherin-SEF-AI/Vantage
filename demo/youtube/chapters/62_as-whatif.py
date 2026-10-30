"""62 What-ifs and proposals (episode as070, chapter 2): the sample's data rig, the REAL model (GPT-OSS 120B on Groq,
unedited replies). A what-if in words ("What if F moved 3 cm forward and 3 cm up? ...") -> simulate_changes on a copy
(live) -> the design untouched (the coverage panel still reads 0.240 m²); "Propose it." -> propose_changes -> the
proposal card (each change now / proposed, the predicted effect); Apply (one undo step) -> the measured result beside
the prediction; the coverage panel agrees; the card's Undo puts F back (0.240 m² again).

Figures narrated, read from the app while recording and checked with r.expect: blind ground 0.24 m² now (the coverage
panel and the tool's "before"), 0.025 m² predicted (simulate_changes and the card's predicted effect) and measured
after Apply (the coverage panel, 0.025). Moving F 30 mm forward and 30 mm up was measured beforehand on the same
sample (the same tools) to be worth showing; the model's reply is its own.
"""

META = {
    'id': 'as-whatif',
    'part': 1,
    'title': 'What-ifs and proposals',
    'promise': 'A change simulated on a copy, proposed on a card, and nothing applied until you say so.',
    'menu': 'Assistant ▸ simulate_changes · propose_changes',
}

LINES = [
    # 0
    ("A what-if runs Vantage's analysis on a copy of the design. Ask what moving the front camera 3 cm forward and "
     "3 cm up would do.",
     "A what-if runs Vantage's analysis on a copy of the design. Ask what moving the front camera three centimetres "
     "forward, and three up, would do."),
    # 1
    ("Simulated, the blind ground falls from 0.24 to 0.025 m². In the design nothing has moved: the coverage panel "
     "still reads 0.24.",
     "Simulated, the blind ground falls from zero point two four, to zero point zero two five square metres. In the "
     "design, nothing has moved: the coverage panel still reads zero point two four."),
    # 2
    ("Ask for it as a proposal, and a card lists each change, now and proposed, with its predicted effect.",
     "Ask for it as a proposal, and a card lists each change, now and proposed, with its predicted effect."),
    # 3
    ("Nothing changes until you accept. Apply is one undo step, and the card then puts the measured result beside "
     "the prediction.",
     "Nothing changes until you accept. Apply is one undo step, and the card then puts the measured result beside "
     "the prediction."),
    # 4
    ("The coverage agrees: 0.025 m². The card's Undo puts the camera back.",
     "The coverage agrees: zero point zero two five square metres. The card's undo puts the camera back."),
]

Q1 = "What if F moved 3 cm forward and 3 cm up? Don't change anything; two sentences."
Q2 = 'Propose it.'


def scene(stage, r):
    import helpers_as as H
    H.groq_on()
    try:
        _scene(stage, r, H)
    finally:
        H.groq_off()


def _scene(stage, r, H):
    from PyQt6.QtCore import QRectF
    from vantage.ui.assistant import ProposalCard, ToolRow

    w = H.window(stage)
    r.win = w
    chat = H.Chat(stage, r, w, 'as-whatif')
    panel = w.assistant_panel
    ctl = w.ctx.assistant
    rows = w.bottom.coverage.rows
    blind = lambda: rows['Blind in ROI'].v.text()                         # noqa: E731
    r.expect(blind(), '0.240', 'blind ground before')
    vp = stage.vp
    vp.cam.az, vp.cam.el = 20.0, 26.0
    w.viewarea.frame_all()
    stage.settle(2)
    r.cursor_on = False
    try:
        # ---- 0: the what-if, asked in words
        r.lower_third()
        r.note(H.NOTE)
        r.zoom(H.chat_rect(r, w, 620), 0.01)
        r.say(0)
        r.hold(0.8)
        r.cursor_on = True
        chat.type_in(Q1, cps=26)
        chat.send()
        r.cursor_on = False
        chat.wait(timeout=120)
        call = chat.last('simulate_changes')
        if call is None:
            raise RuntimeError(f'no simulate_changes: {[c[0] for c in chat.calls()]}')
        args, res = call[1], call[2]
        print('simulate_changes', args, str(res)[:300], flush=True)
        eff = res['effect']['blind_m2']
        r.expect((round(eff['before'], 3), round(eff['after'], 3)), (0.24, 0.025), 'blind before / after (simulated, live)')
        if chat.last('propose_changes') is not None:
            raise RuntimeError('the what-if made a proposal')
        trow = next(t for t in panel.findChildren(ToolRow) if t.isVisible() and t.name == 'simulate_changes')
        turn = chat.turn()
        r.finish_speech(0.1)

        # ---- 1: the simulated effect; the design untouched
        r.say(1)
        r.figure('Blind, simulated', eff['after'], 'm²', 3, dur=3.4, src=trow, start=eff['before'],
                 note='simulate_changes · on a copy')
        H.read_through(r, panel, turn, max(2.8, r.at_word(1, 'In the design') - r.now()), lead=8)
        r.zoom(None, 0.8)
        r.expect(blind(), '0.240', 'the design is unchanged after the what-if')
        r.callout(rows['Blind in ROI'], '', dur=max(1.6, r.speech_left() + 0.2), pad=4, dim=0.25)
        r.finish_speech(0.1)

        # ---- 2: propose it
        r.say(2)
        r.zoom(H.chat_rect(r, w, 620), 0.8, wait=False)
        r.cursor_on = True
        chat.type_in(Q2, cps=18)
        chat.send()
        r.cursor_on = False
        chat.wait(timeout=120)
        cards = [c for c in panel.findChildren(ProposalCard) if c.isVisible()]
        if not cards:
            raise RuntimeError(f'no proposal card: {[c[0] for c in chat.calls()]}')
        card = cards[-1]
        pe = {e['key']: e for e in card.prop.effects}
        print('proposal', card.prop.title, card.prop.rows, pe.get('blind_m2'), flush=True)
        r.expect((round(pe['blind_m2']['before'], 3), round(pe['blind_m2']['after'], 3)), (0.24, 0.025),
                 'the card\'s predicted blind')
        H.show_widget(r, panel, card, margin=12)
        r.zoom(lambda: r.wrect(card).adjusted(-24, -30, 24, 24), 1.0, margin=10)
        r.mark('cold', shot='proposal')        # the intro's cold open and the thumbnail quote from here
        r.finish_speech(0.1)

        # ---- 3: nothing changes until you accept; Apply
        r.say(3)
        H.kinetic(r, 'Nothing changes\nuntil you accept.', accent=('accept.',), dur=3.4, px=50)
        r.until_word(3, 'Apply is one')
        r.cursor_on = True
        stage.mark_result()
        r.click(card.b_apply, None, 0.6)
        r.wait_for(lambda: bool(ctl.proposals.get(card.prop.id, {}).get('measured')), timeout=60)
        stage.settle(2)
        H.show_widget(r, panel, card, margin=12)
        r.cursor_on = False
        print('measured:', card.measured.text(), flush=True)
        r.expect('as predicted' in card.measured.text(), True, 'measured as predicted')
        r.callout(card.measured, '', dur=max(1.8, r.speech_left() + 0.3), pad=4, dim=0.2)
        r.finish_speech(0.1)

        # ---- 4: the coverage agrees; Undo
        r.say(4)
        r.zoom(None, 0.8)
        r.expect(blind(), '0.025', 'blind after applying')
        r.figure('Blind in ROI', float(blind()), 'm²', 3, dur=2.4, src=rows['Blind in ROI'], start=0.24,
                 note='measured after Apply')
        r.until_word(4, "The card's")
        r.zoom(lambda: r.wrect(card).adjusted(-24, -30, 24, 24), 0.8, margin=10)
        r.cursor_on = True
        r.click(card.b_undo, None, 0.5)
        stage.settle(2)
        r.cursor_on = False
        r.hold(0.6)
        r.zoom(None, 0.8)
        r.expect(blind(), '0.240', 'blind after the undo')
        r.callout(rows['Blind in ROI'], '', dur=1.6, pad=4, dim=0.25)
        r.finish_speech(0.8)
        chat.dump('ok')
        r.note(None)
        r.hold(0.4)
    except Exception:
        chat.dump('failed')
        raise
