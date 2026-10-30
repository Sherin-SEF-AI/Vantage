"""64 Whole-rig design (episode as070, chapter 4): the sample's data rig with its quarter cameras QL and QR tilted 12°
up (pitch 50 -> 38°, an edit made for this chapter and said on screen, so the search has something to find: on the
shipped rig it honestly finds no improvement). The REAL model (GPT-OSS 120B on Groq, unedited replies):
"Optimise the whole rig: 4 to 6 cameras, at least 3 px/cm, as little blind ground as possible." -> optimise_rig (the
app's placement search over every camera, a time budget, live; shown as a marked time-lapse) -> options with their
trade-offs; then "Where would one more camera close the patch in front of the bumper?" -> add_camera_options (live).

Figures narrated, read from the app while recording and checked with r.expect: 0.283 m² blind with the quarter cameras
tilted (the coverage panel). The option figures are shown as tiles read from the tool results (not narrated: the
search runs to a time budget, so how far it gets depends on the machine). Narrated without a figure and checked: the
search returned options; add_camera_options' best spot is on the front bumper (its mount
location starts with "Front bumper").
"""

META = {
    'id': 'as-rig',
    'part': 2,
    'title': 'Whole-rig design',
    'promise': 'Targets in, complete rig options out, with their trade-offs.',
    'menu': 'Assistant ▸ optimise_rig · add_camera_options',
}

LINES = [
    # 0
    ("For this chapter the quarter cameras are tilted 12° up, off their best pose: 0.283 m² of the ground is blind.",
     "For this chapter, the quarter cameras are tilted twelve degrees up, off their best pose: zero point two eight "
     "three square metres of the ground is blind."),
    # 1
    ("Ask for the whole rig: four to six cameras, at least 3 px/cm, as little blind ground as possible.",
     "Ask for the whole rig: four to six cameras, at least three pixels per centimetre, as little blind ground as "
     "possible."),
    # 2
    ("optimise_rig runs the app's placement search over every camera within a time budget, and scores each complete "
     "rig under the same mounting errors.",
     "Optimise rig runs the app's placement search over every camera, within a time budget, and scores each complete "
     "rig under the same mounting errors."),
    # 3
    ("Each option comes back re-checked at full resolution: what it changes, what it costs, and whether it meets "
     "every target.",
     "Each option comes back re-checked at full resolution: what it changes, what it costs, and whether it meets "
     "every target."),
    # 4
    ("Moving cameras cannot reach the patch hidden by the front radar. add_camera_options looks for where one more "
     "camera would close it:",
     "Moving cameras cannot reach the patch hidden by the front radar. Add camera options looks for where one more "
     "camera would close it:"),
    # 5
    ("mountable spots on the body, each aimed at the gap and checked with the camera added. The best is on the front "
     "bumper. Nothing is changed until you accept.",
     "mountable spots on the body, each aimed at the gap, and checked with the camera added. The best is on the front "
     "bumper. Nothing is changed until you accept."),
]

Q1 = 'Optimise the whole rig: 4 to 6 cameras, at least 3 px/cm, least blind ground. What would the best option do? One sentence.'
Q2 = 'Where would one more camera close the patch in front of the bumper? One sentence.'


def scene(stage, r):
    import helpers_as as H
    H.groq_on()
    try:
        _scene(stage, r, H)
    finally:
        H.groq_off()


def _scene(stage, r, H):
    from vantage.ui.assistant import ToolRow

    def tilt(p):
        for tag in ('QL', 'QR'):
            p.variant.by_tag(tag).pitch -= 12
    w = H.window(stage, edit=tilt)
    r.win = w
    chat = H.Chat(stage, r, w, 'as-rig')
    panel = w.assistant_panel
    rows = w.bottom.coverage.rows
    r.expect(rows['Blind in ROI'].v.text(), '0.283', 'blind with the quarter cameras tilted up')
    vp = stage.vp
    vp.cam.az, vp.cam.el = -70.0, 30.0
    w.viewarea.frame_all()
    stage.settle(2)
    r.cursor_on = False
    try:
        r.lower_third()
        r.note(H.NOTE)
        # ---- 0: the set-up, said on screen
        r.say(0)
        stage.orbit(r, vp, 18, 2.2, d_el=2)
        r.figure('Blind in ROI', float(rows['Blind in ROI'].v.text()), 'm²', 3, dur=3.0, src=rows['Blind in ROI'],
                 note='QL and QR tilted 12° up for this chapter')
        r.finish_speech(0.1)

        # ---- 1: targets in words
        r.say(1)
        r.zoom(H.chat_rect(r, w, 620), 0.9)
        r.cursor_on = True
        chat.type_in(Q1, cps=30)
        chat.send()
        r.cursor_on = False
        r.finish_speech(0.1)

        # ---- 2: the search (a marked time-lapse)
        r.say(2)
        r.mark('cold', shot='search')          # the intro's cold open quotes from here
        chat.wait(timeout=300, speed=8.0, after=8.0)     # x8: the recorder leaves the search the CPU
        call = chat.last('optimise_rig')
        if call is None:
            raise RuntimeError(f'no optimise_rig: {[c[0] for c in chat.calls()]}')
        args, res = call[1], call[2]
        print('optimise_rig args', args, flush=True)
        print('optimise_rig', {k: res.get(k) for k in ('current', 'search', 'result', 'targets')}, flush=True)
        for o in res.get('options', []):
            print('  option', o['option'], o['trade_off'], o['meets_all_targets'], flush=True)
        r.expect(bool(res.get('options')), True, 'the search returned options')
        trow = next(t for t in panel.findChildren(ToolRow) if t.isVisible() and t.name == 'optimise_rig')
        turn = chat.turn()
        r.finish_speech(0.1)

        # ---- 3: the options
        r.say(3)
        best = res['options'][0]
        r.figure(f'Option {best["option"]} · blind', best['metrics']['blind_m2'][1], 'm²', 3, dur=3.2, src=trow,
                 start=best['metrics']['blind_m2'][0], note=f'{res["search"]["rigs_scored"]} rigs scored · live')
        H.read_through(r, panel, turn, max(3.0, r.speech_left() + 0.6), lead=8)
        r.finish_speech(0.1)

        # ---- 4 / 5: one more camera
        r.say(4)
        r.cursor_on = True
        chat.type_in(Q2, cps=30)
        chat.send()
        r.cursor_on = False
        chat.wait(timeout=200, speed=2.0)
        call = chat.last('add_camera_options')
        if call is None:
            raise RuntimeError(f'no add_camera_options: {[c[0] for c in chat.calls()]}')
        add = call[2]
        for o in add.get('options', []):
            print('  add', o['option'], o['adds'], o['mount'], o['trade_off'], flush=True)
        a0 = add['options'][0]
        r.expect(a0['mount']['location'].startswith('Front bumper'), True, 'the best spot is on the front bumper')
        trow2 = [t for t in panel.findChildren(ToolRow) if t.isVisible() and t.name == 'add_camera_options'][-1]
        turn2 = chat.turn()
        r.finish_speech(0.1)
        r.say(5)
        r.figure(f'+{a0["adds"][0]} · blind', a0['metrics']['blind_m2'][1], 'm²', 3, dur=3.2, src=trow2,
                 start=a0['metrics']['blind_m2'][0], note=a0['mount']['location'] + ' · live')
        H.read_through(r, panel, turn2, max(4.0, r.speech_left() - 0.5), lead=8)
        r.finish_speech(0.6)
        chat.dump('ok')
        r.zoom(None, 0.8)
        r.note(None)
        r.hold(0.5)
    except Exception:
        chat.dump('failed')
        raise
