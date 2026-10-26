"""26 The assistant: Window ▸ Assistant (Ctrl+J) on the sample with its quarter cameras tilted 12° up (an edit made for
this chapter, said on screen), talking to tests/fake_groq.py: the model's replies are SCRIPTED for this video (the note
says so on screen all the time the assistant is visible); every tool it calls runs live in Vantage and every figure
comes from those tools. Never a real key, never the network.

Ask mode: "Why is the ground beside the rear wheels blind?" -> find_blind_regions (live) -> a scripted answer whose
figures are checked against the live tool result (r.expect): blind total 0.053 m², the largest region 0.028 m² at
(0.0, -0.89) with QR "outside the image frame", the next 0.013 m² with QL "outside the image frame". The tool row
opened to show what was asked and sent back. Agent mode: "Close every blind patch" -> a plan, find_blind_regions,
optimise_camera QR (live: 22 trials, best change dpitch +4°, mirrored, blind 0.053 -> 0.010 m², minimum density
3.12 px/cm unchanged), propose_changes (live prediction), a scripted summary checked against the tool results; Apply;
the measured result next to the prediction (coverage panel: blind 0.010 m²).
"""

META = {
    'id': 'assistant',
    'part': 3,
    'title': 'The assistant',
    'promise': 'Ask about the design; let an agent search for a fix. It only ever proposes.',
    'menu': 'Window ▸ Assistant  ·  Ctrl+J',
}

LINES = [
    ("The assistant is a chat about the open design, with a model hosted on Groq. In this video its replies are "
     "scripted; the tools it calls, and every figure they return, are computed live by Vantage.",
     "The assistant is a chat about the open design, with a model hosted on Grok. In this video, its replies are "
     "scripted. The tools it calls, and every figure they return, are computed live by Vantage."),
    ("It is set up in Preferences with your own Groq key, kept in the system keyring or a private file, never in a "
     "project; you choose the model and its reasoning effort.",
     "It is set up in preferences, with your own Grok key, kept in the system keyring or a private file, never in a "
     "project. You choose the model, and its reasoning effort."),
    ("For this chapter the quarter cameras are tilted twelve degrees up, and 0.053 m² of the region is blind. Ask "
     "why.",
     "For this chapter, the quarter cameras are tilted twelve degrees up, and zero point zero five three square metres "
     "of the region is blind. Ask why."),
    ("The model sees the design only through Vantage's tools. It asks for the blind regions: the largest, 0.028 m², "
     "lies against the right side at the rear axle, just outside the tilted right quarter camera's image.",
     "The model sees the design only through Vantage's tools. It asks for the blind regions: the largest, zero point "
     "zero two eight square metres, lies against the right side at the rear axle, just outside the tilted right "
     "quarter camera's image."),
    ("Every lookup is listed: open it to see exactly what the model asked for, and what was sent back.",
     "Every lookup is listed: open it, to see exactly what the model asked for, and what was sent back."),
    ("Agent mode works towards a goal. It plans a checklist, then works through it: find the blind ground, search a "
     "better pose, check it, and propose.",
     "Agent mode works towards a goal. It plans a checklist, then works through it: find the blind ground, search a "
     "better pose, check it, and propose."),
    ("The search tried 22 poses of the right quarter camera: tilted four degrees down, with its mirrored partner, "
     "the blind ground falls from 0.053 to 0.010 m², and the minimum density stays put.",
     "The search tried twenty-two poses of the right quarter camera. Tilted four degrees down, with its mirrored "
     "partner, the blind ground falls from zero point zero five three, to zero point zero one square metres, and the "
     "minimum density stays put."),
    ("The assistant can only propose. The card shows the change and its predicted effect: apply it as one undo step, "
     "as a new variant, or discard it.",
     "The assistant can only propose. The card shows the change and its predicted effect: apply it as one undo step, "
     "as a new variant, or discard it."),
    ("Applied, the card puts the measured result next to the prediction, and the coverage agrees: 0.010 m² of blind "
     "ground.",
     "Applied, the card puts the measured result next to the prediction, and the coverage agrees: zero point zero one "
     "square metres of blind ground."),
    ("What leaves the machine: poses, lens and model names, figures and requirements. Never the vehicle model, "
     "images, point clouds, files or stream addresses.",
     "What leaves the machine: poses, lens and model names, figures, and requirements. Never the vehicle model, "
     "images, point clouds, files, or stream addresses."),
]

NOTE = 'Assistant replies scripted for this video; the tools and figures are computed live'

Q1 = 'Why is the ground beside the rear wheels blind?'
A1 = ('**The largest blind patch, 0.028 m², lies against the right side at the rear axle** (x 0.0, y −0.89 m).\n\n'
      '- Five cameras have it outside their lens field of view.\n'
      '- QR, tilted up, has it just outside its image frame.\n'
      '- Its mirror image on the left, 0.013 m², is just outside QL\'s frame.\n\n'
      'In all, 0.053 m² of the region is blind.')
Q2 = 'Close every blind patch'
STEPS = ['Find the blind ground and why', 'Search a better pose for the quarter cameras',
         'Check it and propose the best option']
A2 = ('**Proposed: tilt QR and QL down 4°.** The blind ground falls from 0.053 to 0.010 m²; the minimum density '
      'stays at 3.1 px/cm.\n\n'
      '- 22 poses of QR tried; yaw did not help.\n'
      '- Nothing changes until you apply it.')


def scene(stage, r):
    import json
    import os
    import numpy as np
    from PyQt6.QtCore import QPointF, QRectF, Qt
    from PyQt6.QtTest import QTest
    from fake_groq import FakeGroq
    from vantage.assistant import secrets as K
    from vantage.ui.assistant import ProposalCard, ToolRow, PlanCard

    srv = FakeGroq().start()                 # a local stand-in for Groq on 127.0.0.1: nothing leaves the machine
    os.environ['VANTAGE_ASSISTANT_URL'] = srv.url
    try:
        K.save('test-key')

        def tilt(p):
            for tag in ('QR', 'QL'):
                p.variant.by_tag(tag).pitch -= 12
        w = stage.window(edit=tilt)
        r.win = w
        st = w.store
        rows = w.bottom.coverage.rows
        r.expect(rows['Blind in ROI'].v.text(), '0.053', 'blind with the quarter cameras tilted up')
        w.ctx.settings.setValue('assistant/enabled', True)
        w.ctx.assistant.reload()
        ctl = w.ctx.assistant
        srv.delay = 0.025
        r.cursor_on = False
        vp = stage.vp
        vp.cam.az, vp.cam.el = -120.0, 30.0
        w.viewarea.frame_all()
        stage.settle(2)
        r.cur = np.array([1150.0, 560.0])

        def tool_results(i0=0):
            out = {}
            for req in srv.requests[i0:]:
                for m in req['messages']:
                    if m['role'] == 'tool':
                        out[m.get('tool_call_id', '')] = json.loads(m['content'])
            return out

        # ---- 0: the opener; the note; the panel from the toolbar (Ctrl+J)
        r.lower_third()
        r.note(NOTE)
        r.say(0)
        stage.orbit(r, vp, 24, 2.6, d_el=2)
        r.cursor_on = True
        r.key_combo('Ctrl+J', dur=1.4)
        r.toolbar('assistant')
        stage.settle(2)
        panel = w.assistant_panel
        r.crossfade(0.4)
        r.finish_speech(0.1)
        dock = lambda: r.wrect(w.docks['assistant'])                                  # noqa: E731

        def dock_zoom(extra=260):
            d = dock()
            return QRectF(d.x() - extra, d.y(), d.width() + extra, d.height())

        def dock_bottom():                     # the newest part of the conversation and the composer
            d = dock()
            return QRectF(d.x() - 30, d.bottom() - 600, d.width() + 40, 590)

        # ---- 1: set up in Preferences (not filmed: no key ever shows); the model in the composer
        r.say(1)
        r.zoom(dock_bottom(), 1.0)
        r.until_word(1, 'You choose the model')
        r.callout(panel.composer.model_btn, '', dur=max(2.0, r.speech_left() + 0.4), pad=4, dim=0.2)
        r.finish_speech(0.2)

        # ---- 2: the tilted quarter cameras; ask why
        r.say(2)
        r.zoom(None, 0.8)
        r.figure('Blind in ROI', float(rows['Blind in ROI'].v.text()), 'm²', 3, dur=2.8, src=rows['Blind in ROI'],
                 note='quarter cameras tilted 12° up')
        r.hold(2.6)
        srv.script += [srv.tool_call('find_blind_regions', {}, 'q1', reasoning='Look up the blind regions first.'),
                       srv.answer(A1, reasoning='The largest region is beside the rear axle on the right; QR has it '
                                                'outside its image frame.')]
        r.zoom(dock_bottom(), 0.8, wait=False)
        r.move_to(panel.composer.edit, None, 0.6)
        r.type(panel.composer.edit, Q1, cps=22)
        r.until_word(2, 'Ask why')
        r.hold(0.3)
        n0 = len(srv.requests)
        QTest.keyClick(panel.composer.edit, Qt.Key.Key_Return)
        r.finish_speech(0.1)

        # ---- 3: the tools; the answer (scripted text, live figures checked)
        r.say(3)
        r.wait_for(lambda: not ctl.busy, timeout=120)
        res = tool_results(n0)['q1']
        regs = res['blind_regions']
        print('find_blind_regions:', json.dumps(res)[:600], flush=True)
        r.expect(res['blind_total_m2'], 0.053, 'blind total (live tool)')
        r.expect((regs[0]['area_m2'], regs[0]['centre'], regs[0]['per_camera_at_sample']['QR']),
                 (0.028, [-0.0, -0.89], 'outside the image frame'), 'largest region (live tool)')
        r.expect(sum(1 for v in regs[0]['per_camera_at_sample'].values() if v == 'outside the lens field of view'), 5,
                 'cameras with the region outside their field of view')
        r.expect((regs[1]['area_m2'], regs[1]['per_camera_at_sample']['QL']), (0.013, 'outside the image frame'),
                 'second region (live tool)')
        r.hold(max(0.5, r.speech_left() - 1.0))
        r.finish_speech(0.2)

        # ---- 4: open the lookup
        r.say(4)
        trow = panel.findChildren(ToolRow)[-1]
        panel.area.ensureWidgetVisible(trow)
        r.tick(2)
        r.click(trow, QPointF(trow.width() * 0.4, 12), 0.7)
        stage.settle(1)
        panel.area.ensureWidgetVisible(trow)
        r.tick(2)
        r.zoom(lambda: r.wrect(trow).adjusted(-10, -10, 10, min(0, 360 - r.wrect(trow).height())), 1.0, margin=12)
        r.hold(max(1.4, r.speech_left() - 0.6))
        r.click(trow, QPointF(trow.width() * 0.4, 12), 0.5)
        stage.settle(1)
        r.zoom(dock_bottom(), 0.8, wait=False)
        r.finish_speech(0.1)

        # ---- 5: agent mode
        r.say(5)

        def plan(states, cid):
            return srv.tool_call('update_plan', {'steps': [{'step': a, 'status': b} for a, b in zip(STEPS, states)]},
                                 cid)
        srv.script += [
            plan(['in_progress', 'pending', 'pending'], 'p1'),
            srv.tool_call('find_blind_regions', {}, 'a1', reasoning='Start with where the ground is blind and why.'),
            plan(['done', 'in_progress', 'pending'], 'p2'),
            srv.tool_call('optimise_camera', {'camera': 'QR', 'vary': ['pitch', 'yaw'], 'objective': 'blind',
                                              'min_density': 3}, 'a2'),
            plan(['done', 'done', 'in_progress'], 'p3'),
            srv.tool_call('propose_changes', {'changes': [{'camera': 'QR', 'dpitch': 4, 'mirror_partner': True}],
                                              'title': 'Tilt the quarter cameras down 4°',
                                              'rationale': 'Brings the ground beside the rear wheels back into QR '
                                                           'and QL.'}, 'a3'),
            plan(['done', 'done', 'done'], 'p4'),
            srv.answer(A2),
        ]
        r.click(panel.composer.agent, None, 0.7)
        r.move_to(panel.composer.edit, None, 0.5)
        r.type(panel.composer.edit, Q2, cps=18)
        r.hold(0.3)
        n1 = len(srv.requests)
        QTest.keyClick(panel.composer.edit, Qt.Key.Key_Return)
        r.hold(0.8)
        cards = panel.findChildren(PlanCard)
        if cards:
            panel.area.ensureWidgetVisible(cards[-1])
        r.finish_speech(0.1)

        # ---- 6: the search (live) and its figures
        r.say(6)
        r.wait_for(lambda: not ctl.busy, timeout=180)
        tr = tool_results(n1)
        opt = tr['a2']
        print('optimise_camera:', json.dumps(opt)[:700], flush=True)
        r.expect(opt['search']['trials'], 22, 'poses tried (live tool)')
        r.expect(opt['best_change'], {'camera': 'QR', 'dpitch': 4.0, 'mirror_partner': True}, 'best change (live)')
        eff = opt['checked_at_project_resolution']
        r.expect((eff['blind_m2']['before'], eff['blind_m2']['after']), (0.053, 0.01), 'blind before/after (live)')
        r.expect(round(eff['min_density']['after'], 1), round(eff['min_density']['before'], 1), 'minimum density kept')
        r.expect(f"{eff['min_density']['after']:.1f}", '3.1', 'minimum density (live)')
        cards = panel.findChildren(ProposalCard)
        card = cards[-1]
        panel.area.ensureWidgetVisible(card)
        r.tick(2)
        r.zoom(lambda: r.wrect(card).adjusted(-20, -40, 20, 20), 1.0, margin=10)
        r.hold(max(1.0, r.speech_left() + 0.3))
        r.finish_speech(0.2)

        # ---- 7: propose only; apply
        r.say(7)
        r.callout(card.b_apply, '', dur=1.6, pad=4, dim=0.2)
        r.until_word(7, 'as a new variant')
        r.callout(card.b_variant, '', dur=1.4, pad=4, dim=0.2)
        r.until_word(7, 'or discard it')
        r.callout(card.b_discard, '', dur=1.4, pad=4, dim=0.2)
        r.finish_speech(0.1)
        stage.mark_result()
        r.click(card.b_apply, None, 0.7)
        r.wait_for(lambda: bool(ctl.proposals.get(1, {}).get('measured')), timeout=60)
        stage.settle(2)

        # ---- 8: the measured result; the coverage agrees
        r.say(8)
        panel.area.ensureWidgetVisible(card.measured)
        r.tick(2)
        print('measured:', card.measured.text(), flush=True)
        r.expect('as predicted' in card.measured.text(), True, 'measured as predicted')
        r.callout(card.measured, '', dur=2.6, pad=4, dim=0.2)
        r.hold(2.4)
        r.until_word(8, 'the coverage agrees')
        r.zoom(None, 0.8)
        r.expect(rows['Blind in ROI'].v.text(), '0.010', 'blind after applying')
        r.figure('Blind in ROI', float(rows['Blind in ROI'].v.text()), 'm²', 3, dur=max(2.6, r.speech_left() + 0.6),
                 src=rows['Blind in ROI'], start=0.053, note='applied')
        r.finish_speech(0.2)

        # ---- 9: what leaves the machine
        r.say(9)
        r.cursor_on = False
        r.zoom(dock_zoom(120), 1.0)
        r.hold(max(1.0, r.speech_left() - 1.2))
        r.zoom(None, 0.8)
        r.finish_speech(0.8)
        r.note(None)
        r.hold(0.4)
    finally:
        os.environ.pop('VANTAGE_ASSISTANT_URL', None)
        K.delete()
        srv.stop()
