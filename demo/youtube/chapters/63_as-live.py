"""63 Recommendations as you work (episode as070, chapter 3): the sample's data rig. The live strip at the top of the
assistant panel follows the selection and the last edit, worked out on this machine: R selected in the Outliner ->
"R alone sees 4.07 m²: check what its loss or a mounting error costs"; R tilted 10° up in the Inspector (pitch 40 ->
30°) -> "Your last change added 0.062 m² blind ground: undo it" / "Ask why your last change added 0.062 m² blind
ground"; the header's Recommended (n) list, worked out locally (checked: no request before Ask). Then the Ask button of
one listed recommendation, the Issues list's "Camera F: radar RD1's housing covers 14 % of its image": its prompt
("Explain this issue and suggest the smallest fix, checked with simulate_changes ...") goes to the REAL model
(GPT-OSS 120B on Groq), which checks a fix with simulate_changes (checked) and answers, unedited.

Figures narrated, read from the strip while recording and checked with r.expect: 4.07 m² (R's unique ground, the
strip's title), 0.062 m² (the edit's added blind ground, the strip's title; the coverage panel goes 0.240 -> 0.303).
"""

META = {
    'id': 'as-live',
    'part': 1,
    'title': 'Recommendations as you work',
    'promise': 'Advice about what you are doing now, worked out locally.',
    'menu': 'Assistant panel ▸ live strip  ·  Recommended',
}

LINES = [
    # 0
    ("The assistant also works without being asked. The strip at the top of the panel follows what you do, worked "
     "out on this machine.",
     "The assistant also works without being asked. The strip at the top of the panel follows what you do, worked "
     "out on this machine."),
    # 1
    ("Select the rear camera: R alone sees 4.07 m² of the ground, so the strip suggests checking what its loss, or a "
     "mounting error, would cost.",
     "Select the rear camera: R alone sees four point zero seven square metres of the ground, so the strip suggests "
     "checking what its loss, or a mounting error, would cost."),
    # 2
    ("Tilt it up ten degrees in the Inspector, and the strip notices: your last change added 0.062 m² of blind "
     "ground. Undo it, or ask why.",
     "Tilt it up ten degrees in the inspector, and the strip notices: your last change added zero point zero six two "
     "square metres of blind ground. Undo it, or ask why."),
    # 3
    ("Recommended lists them all; each button says what it does. Working them out is local and instant, and sends "
     "nothing.",
     "Recommended lists them all; each button says what it does. Working them out is local and instant, and sends "
     "nothing."),
    # 4
    ("Only Ask sends. Here: the front radar's housing covers part of the front camera's image; the model explains "
     "it and checks a fix.",
     "Only ask sends. Here: the front radar's housing covers part of the front camera's image; the model explains "
     "it, and checks a fix."),
]


def scene(stage, r):
    import helpers_as as H
    H.groq_on()
    try:
        _scene(stage, r, H)
    finally:
        H.groq_off()


def _scene(stage, r, H):
    from PyQt6.QtCore import QPointF, QRectF
    from vantage.ui.assistant import ProposalCard, RecRow
    from vantage.ui.outliner import ROLE_ID, ROLE_KIND

    w = H.window(stage)
    r.win = w
    chat = H.Chat(stage, r, w, 'as-live')
    panel = w.assistant_panel
    st = w.store
    rows = w.bottom.coverage.rows
    rc = panel.recommender
    R = st.variant.by_tag('R')
    r.expect(rows['Blind in ROI'].v.text(), '0.240', 'blind before the edit')
    vp = stage.vp
    vp.cam.az, vp.cam.el = 150.0, 24.0               # the rear of the car
    w.viewarea.frame_all()
    stage.settle(2)
    r.cursor_on = False

    def strip_row(rid_start):
        for row in panel.strip.findChildren(RecRow):
            if row.isVisible() and row.rec is not None and row.rec.id.startswith(rid_start):
                return row
        return None

    def outliner_row(uid):
        tree = w.outliner

        def find():
            stack = [tree.topLevelItem(i) for i in range(tree.topLevelItemCount())]
            while stack:
                it = stack.pop(0)
                if it.data(0, ROLE_KIND) == 'camera' and it.data(0, ROLE_ID) == uid:
                    rr = tree.visualItemRect(it)
                    tl = r.wpos(tree.viewport(), QPointF(0, rr.top()))
                    return QRectF(tl.x(), tl.y(), tree.viewport().width(), rr.height())
                stack += [it.child(k) for k in range(it.childCount())]
            return None
        return find

    insp = lambda: r.wrect(w.docks['inspector'])                           # noqa: E731
    dock = lambda: r.wrect(w.docks['assistant'])                           # noqa: E731

    def work_rect():                                 # the Inspector and the panel's top, side by side
        a, d = insp(), dock()
        return QRectF(a.x() - 10, d.y(), d.right() - a.x() + 10, 600)

    try:
        r.lower_third()
        r.note(H.NOTE)
        # ---- 0: the strip
        r.say(0)
        r.zoom(work_rect, 1.2)
        r.callout(panel.strip if panel.strip.isVisible() else panel.hairline, '', dur=max(1.6, r.speech_left() - 0.6),
                  pad=2, dim=0.2)
        r.finish_speech(0.1)

        # ---- 1: select R
        r.say(1)
        r.zoom(None, 0.7)
        r.cursor_on = True
        tgt = outliner_row(R.uid)
        rr = tgt()
        r.move(QPointF(rr.x() + 60, rr.center().y()), 0.6)
        r.ripple()
        st.select_cams([R.uid])
        r.wait_for(lambda: strip_row('unique:R') is not None, timeout=20)
        r.cursor_on = False
        row = strip_row('unique:R')
        print('strip:', panel.strip.head.text(), '|', row.rec.title, flush=True)
        r.expect(row.rec.title.split(' ')[3], '4.07', 'R alone sees (the strip)')
        r.zoom(work_rect, 0.9)
        r.callout(row, '', dur=max(2.0, r.speech_left()), pad=3, dim=0.25)
        r.finish_speech(0.1)

        # ---- 2: tilt R up 10° in the Inspector
        r.say(2)
        page = w.inspector.pages['camera']
        r.cursor_on = True
        stage.mark_result()
        r.enter(page.rp, f'{R.pitch - 10:g}', cps=8)
        r.cursor_on = False
        stage.wait_analysis(r)
        r.wait_for(lambda: strip_row('edit:') is not None, timeout=20)
        stage.app.processEvents()
        und = next(x for x in panel.strip.findChildren(RecRow) if x.isVisible() and x.rec.id.endswith(':undo'))
        why = next(x for x in panel.strip.findChildren(RecRow) if x.isVisible() and x.rec.id.endswith(':why'))
        print('strip after the edit:', und.rec.title, '|', why.rec.title, '| blind', rows['Blind in ROI'].v.text(),
              flush=True)
        r.expect(und.rec.title, 'Your last change added 0.062 m² blind ground: undo it', 'the strip after the edit')
        r.expect(rows['Blind in ROI'].v.text(), '0.303', 'blind after the edit')
        r.mark('cold', shot='strip')           # the intro's cold open quotes from here
        r.callout(und, '', dur=2.2, pad=3, dim=0.25)
        r.until_word(2, 'Undo it')
        r.callout(why, '', dur=max(1.4, r.speech_left() + 0.3), pad=3, dim=0.25)
        r.finish_speech(0.1)

        # ---- 3: Recommended (n): local, nothing sent
        r.say(3)
        r.cursor_on = True
        r.click(panel.b_recs, None, 0.6, real=False)
        pop = panel.show_recommended()
        stage.extra.append(pop)
        r.hold(0.3)
        r.zoom(lambda: r.wrect(pop).adjusted(-260, -60, 30, 30), 0.8, margin=10)
        r.cursor_on = False
        r.until_word(3, 'Working them out')
        H.kinetic(r, 'Local. Instant.\nNothing sent.', accent=('local.', 'nothing', 'sent.'), dur=3.0, px=54)
        r.finish_speech(0.1)
        r.expect(sum(u['requests'] for u in chat.usage().values()), 0, 'no request before Ask')

        # ---- 4: Ask on the housing issue: its prompt goes to the model
        r.say(4)

        def issue_rows():
            return [x for x in pop.findChildren(RecRow) if x.isVisible() and x.rec is not None
                    and x.rec.id.startswith('issue:') and 'RD1' in x.rec.title]
        if not issue_rows() and pop.list.more.isVisible():        # past the first five: More shows the rest
            r.cursor_on = True
            r.click(pop.list.more, None, 0.5)
            stage.app.processEvents()
            pop.adjustSize()
            r.tick(3)
        rows_ = issue_rows()
        if not rows_:
            raise RuntimeError(f'no RD1 housing issue in Recommended: {[x.rec.title for x in pop.findChildren(RecRow) if x.rec]}')
        row = rows_[0]
        print('issue rec:', row.rec.title, '|', row.rec.prompt, flush=True)
        r.callout(row, '', dur=2.0, pad=3, dim=0.25)
        r.cursor_on = True
        r.move_to(row.b, None, 0.8)
        r.until_word(4, 'the model explains')
        r.click(row.b, None, 0.2)
        r.cursor_on = False
        pop.hide()
        r.zoom(H.chat_rect(r, w, 620), 0.9, wait=False)
        chat.wait(timeout=240, speed=1.0, after=8.0)
        print('calls:', [(c[0], c[1]) for c in chat.calls()], flush=True)
        if chat.last('simulate_changes') is None:
            raise RuntimeError('the model did not check a fix with simulate_changes')
        turn = chat.turn()
        H.read_through(r, panel, turn, max(3.0, r.speech_left() + 0.8), lead=8)
        r.finish_speech(0.4)
        chat.dump('ok')
        r.zoom(None, 0.8)
        r.note(None)
        r.hold(0.5)
    except Exception:
        chat.dump('failed')
        raise
