"""47 A more accurate calibration (episode v020, chapter 7): what was wrong with 0.1.1's installation-check joint solve
and lidar–camera calibration, how it shows, what 0.2.0 changed, and the before/after figures.

On the sample (6-cam · v3): Calibrate ▸ Installation check, a synthetic capture with the page's simulated mounting
error, Run check (a marked time-lapse): the joint solve's line and board table. Calibrate ▸ Calibration observability,
Verify (30 simulated captures, seed 1) first with 0.1.1's joint solve - read for this comparison from the git tag v0.1.1
(`git show`, nothing in the repository changes) into the throw-away home and put in place of vantage.core.joint's
joint_solve in this process only, said on screen - then with 0.2.0's. Then the data rig (Variant), Calibrate ▸
Lidar–camera calibration with F and L1 (seed 4, an injected error of 15 mm forward, 8 mm up, 0.6 deg of yaw): the plan
note's prediction, Capture and solve (a marked time-lapse), the findings. Cards with the release's before/after figures
measured for this video by demo/youtube/calib_measure.py (demo/build/youtube/data/calib_measure.json, read here and
checked against the narration), and a real batch run of --observability --verify 30 in the throw-away home.

Figures read from the app while recording (r.expect): the install check's joint line (6 cameras, 8 boards, 150 shared
corners); Verify with 0.1.1's solve: stopped short in 6 of 30, F yaw predicted 0.131 deg, joint solve 0.298, optimum
0.129, ML z 1.13 / 4.13 / 1.02 mm; with 0.2.0's: 0 of 30, F yaw 0.128 both; the plan note ±4.2 / ±6.3 / ±16.4 mm;
the findings: L1 15 mm / 0.61 deg from its plan, determined to ±2.1 mm / ±0.020 deg, the solve 2.2 mm / 0.021 deg from
the true mounting. 0.1.1's plan note (±0.9 / ±1.7 / ±3.4 mm) is quoted from the first tour's recording of 0.1.1 and is
0.1.1's predicted_sigma in calib_measure.json.
"""

META = {
    'id': 'new-calib',
    'part': 3,
    'title': 'A more accurate calibration',
    'promise': 'The joint solve and the lidar–camera calibration: what was wrong, and the figures now.',
    'menu': 'Calibrate ▸ Installation check  ·  Calibration observability  ·  Lidar–camera calibration',
}

LINES = [
    # 0
    ("Version 0.2.0 makes two calibrations more accurate: the installation check's joint solve, and the lidar–camera "
     "calibration. The pages work as before; what changed is that their answers, and the precision they state, hold.",
     "Version zero point two makes two calibrations more accurate: the installation check's joint solve, and the "
     "lidar camera calibration. The pages work as before. What changed is that their answers, and the precision they "
     "state, hold."),
    # 1
    ("First, the joint solve. The installation check measures each camera from the floor boards it sees, then solves "
     "all six cameras together with each board's place on the floor.",
     "First, the joint solve. The installation check measures each camera from the floor boards it sees, then solves "
     "all six cameras together, with each board's place on the floor."),
    # 2
    ("Here on synthetic captures with a simulated mounting error: 6 cameras, 8 boards and 150 corners shared between "
     "cameras, and each board's estimated offset from where it was meant to lie.",
     "Here, on synthetic captures with a simulated mounting error: six cameras, eight boards, and a hundred and fifty "
     "corners shared between cameras, and each board's estimated offset from where it was meant to lie."),
    # 3
    ("One check cannot show whether the solve reached its best answer: its errors are within the noise either way. "
     "That shows only when the same capture is simulated many times.",
     "One check cannot show whether the solve reached its best answer: its errors are within the noise either way. "
     "That shows only when the same capture is simulated many times."),
    # 4
    ("Calibration observability predicts how precisely the boards pin each camera. Verify simulates 30 captures, with "
     "noisy corners and boards laid off their marks, and solves each with the installation check's own joint solve.",
     "Calibration observability predicts how precisely the boards pin each camera. Verify simulates thirty captures, "
     "with noisy corners, and boards laid off their marks, and solves each with the installation check's own joint "
     "solve."),
    # 5
    ("First with 0.1.1's joint solve, read from that release's source for this comparison: the solve that could stop "
     "before its best answer.",
     "First, with zero point one point one's joint solve, read from that release's source for this comparison: the "
     "solve that could stop before its best answer."),
    # 6
    ("It stopped short of its optimum in some of the 30 trials, the note says how many. How to read the table: "
     "Predicted is the scatter the boards allow; Joint solve, the scatter the solve really had; Optimum, the same "
     "answers carried to their best.",
     "It stopped short of its optimum in some of the thirty trials: the note says how many. How to read the table: "
     "Predicted is the scatter the boards allow; Joint solve, the scatter the solve really had; Optimum, the same "
     "answers carried to their best."),
    # 7
    ("F's yaw should scatter by 0.13 degrees: the old solve scattered more than twice as much, while its optimum "
     "stays at 0.13. ML's height scattered nearly four times its prediction.",
     "F's yaw should scatter by zero point one three degrees: the old solve scattered more than twice as much, while "
     "its optimum stays at zero point one three. M L's height scattered nearly four times its prediction."),
    # 8
    ("Now 0.2.0's solve, on the same 30 captures. It reaches its optimum every time: the Joint solve column is the "
     "Optimum column, and every row matches its prediction.",
     "Now, zero point two's solve, on the same thirty captures. It reaches its optimum every time: the joint solve "
     "column is the optimum column, and every row matches its prediction."),
    # 9
    ("The cause was in the solver. It found each step iteratively, and those steps barely moved along the shallow "
     "directions, where every camera and board shifts together: it ran out of evaluations first.",
     "The cause was in the solver. It found each step iteratively, and those steps barely moved along the shallow "
     "directions, where every camera and board shifts together: it ran out of evaluations first."),
    # 10
    ("0.2.0 solves each step exactly: plain least squares first, then the robust loss for a stray corner. And as a "
     "square grid looks the same turned a quarter round, a board may now move ten times its placement tolerance at "
     "most, and turn 30 degrees.",
     "Zero point two solves each step exactly: plain least squares first, then the robust loss, for a stray corner. "
     "And as a square grid looks the same turned a quarter round, a board may now move ten times its placement "
     "tolerance at most, and turn thirty degrees."),
    # 11
    ("Measured for this video on 40 simulated checks, against an independent reference solve: 0.1.1 stopped short "
     "in nearly all of them, and scattered up to 3.3 times its stated uncertainty.",
     "Measured for this video, on forty simulated checks, against an independent reference solve: zero point one "
     "point one stopped short in nearly all of them, and scattered up to three point three times its stated "
     "uncertainty."),
    # 12
    ("0.2.0 stopped short in none, its scatter is what its uncertainty says, and a joint solve is more than twenty "
     "times faster.",
     "Zero point two stopped short in none, its scatter is what its uncertainty says, and a joint solve is more than "
     "twenty times faster."),
    # 13
    ("Second, the lidar–camera calibration, on the data rig: the front camera F and the roof lidar L1. It had two "
     "problems.",
     "Second, the lidar camera calibration, on the data rig: the front camera F, and the roof lidar L one. It had "
     "two problems."),
    # 14
    ("The first was its predicted precision. The plan's boards alone now pin L1 to 4.2, 6.3 and 16.4 millimetres. "
     "0.1.1 said 0.9, 1.7 and 3.4: three to five times too optimistic.",
     "The first was its predicted precision. The plan's boards alone now pin L one to four point two, six point "
     "three, and sixteen point four millimetres. Zero point one point one said zero point nine, one point seven, and "
     "three point four: three to five times too optimistic."),
    # 15
    ("It counted the lidar's range noise, but not how roughly the camera fixes each board's plane: from 4 to 8 "
     "metres, a fisheye pins a board's distance only to millimetres, not fractions of one.",
     "It counted the lidar's range noise, but not how roughly the camera fixes each board's plane: from four to "
     "eight metres, a fisheye pins a board's distance only to millimetres, not fractions of one."),
    # 16
    ("The prediction now carries the camera's corner noise through its lens model too. Against 300 simulated "
     "calibrations it matches the real scatter within 7 per cent on every axis.",
     "The prediction now carries the camera's corner noise through its lens model too. Against three hundred "
     "simulated calibrations, it matches the real scatter within seven per cent, on every axis."),
    # 17
    ("That is the cautious figure, from the planes alone. The full calibration also uses each board's edges and its "
     "squares in the lidar's reflectivity, and does much better. Capture and solve.",
     "That is the cautious figure, from the planes alone. The full calibration also uses each board's edges, and its "
     "squares in the lidar's reflectivity, and does much better. Capture, and solve."),
    # 18
    ("L1 is 15 millimetres and 0.61 degrees from its plan, as injected, and determined to 2.1 millimetres; the solve "
     "lands 2.2 millimetres from the true mounting.",
     "L one is fifteen millimetres, and zero point six one degrees, from its plan, as injected, and determined to two "
     "point one millimetres. The solve lands two point two millimetres from the true mounting."),
    # 19
    ("The second problem was a small bias. The solve measured each lidar point's distance to the board from the noisy "
     "point itself; part of the noise counted as signal, and the answer leaned by 3.4 millimetres in x.",
     "The second problem was a small bias. The solve measured each lidar point's distance to the board from the noisy "
     "point itself. Part of the noise counted as signal, and the answer leaned by three point four millimetres in x."),
    # 20
    ("Now each point's error is measured along its own beam, where the noise really lies. Over the same 300 "
     "calibrations the lean is gone, and the solve is several times faster.",
     "Now each point's error is measured along its own beam, where the noise really lies. Over the same three hundred "
     "calibrations, the lean is gone, and the solve is several times faster."),
    # 21
    ("Everything built on these solves follows: the requirements scorecard names the lidar metric for what it is, "
     "from the board planes alone; batch mode, the exports and the assistant give the corrected figures.",
     "Everything built on these solves follows. The requirements scorecard names the lidar metric for what it is, "
     "from the board planes alone. Batch mode, the exports, and the assistant give the corrected figures."),
    # 22
    ("In CI, observability with --verify runs the same Monte Carlo check on your own rig, and every build of Vantage "
     "now tests both solves against an independent reference.",
     "In C I, observability with verify runs the same Monte Carlo check on your own rig, and every build of Vantage "
     "now tests both solves against an independent reference."),
    # 23
    ("Limits: the board positions the joint solve estimates are good to about a centimetre; a board laid further off "
     "than its reach must be re-laid; and the lidar prediction stays deliberately cautious.",
     "Limits. The board positions the joint solve estimates are good to about a centimetre. A board laid further off "
     "than its reach must be re-laid. And the lidar prediction stays deliberately cautious."),
]


def scene(stage, r):
    import importlib.util
    import json
    import os
    import site
    import subprocess
    import sys
    from PyQt6.QtCore import QRectF
    from PyQt6.QtWidgets import QScrollArea
    import helpers_tourc as H
    import studio
    import common
    import vantage.core.joint as JOINT

    # ---- the measured before/after figures (calib_measure.py)
    mpath = os.path.join(common.BUILD, 'data', 'calib_measure.json')
    if not os.path.exists(mpath):
        raise SystemExit(f'{mpath} is missing: run demo/youtube/calib_measure.py first')
    M = json.load(open(mpath, encoding='utf-8'))
    jo, jn = M['joint']['0.1.1'], M['joint']['0.2.0']
    lo, ln = M['lidar']['0.1.1'], M['lidar']['0.2.0']
    r.expect(jo['stalled'] >= 0.9 * jo['trials'] and jn['stalled'] == 0, True, 'stopped short: nearly all, then none')
    r.expect(f'{jo["ratio_max"]:.1f}', '3.3', '0.1.1 scatter over its uncertainty')
    r.expect(0.6 < jn['ratio_min'] and jn['ratio_max'] < 1.3, True, '0.2.0 scatter matches its uncertainty')
    sp = jo['seconds_per_solve'] / jn['seconds_per_solve']
    r.expect(20 < sp < 40, True, f'joint speed-up more than twenty ({sp:.1f})')
    ax = ('x_mm', 'y_mm', 'z_mm')
    r.expect(' / '.join(f'{lo["axes"][k]["predicted"]:.1f}' for k in ax), '0.9 / 1.7 / 3.4', '0.1.1 prediction')
    r.expect(' / '.join(f'{ln["axes"][k]["predicted"]:.1f}' for k in ax), '4.2 / 6.3 / 16.4', '0.2.0 prediction')
    r.expect(all(3.0 < lo['axes'][k]['ratio'] < 5.0 for k in ax), True, 'three to five times too optimistic')
    r.expect(max(round(abs(a['ratio'] - 1) * 100) for a in ln['axes'].values()), 7, 'within 7 per cent')
    r.expect(f'{lo["axes"]["x_mm"]["bias"]:.1f}', '-3.4', 'the lean in x')
    r.expect(all(abs(a['bias_se']) < 1.0 for a in ln['axes'].values()), True, 'the lean is gone')
    sl = lo['seconds_per_solve'] / ln['seconds_per_solve']
    r.expect(sl > 3, True, f'lidar solve several times faster ({sl:.1f})')

    # ---- 0.1.1's joint solve, from the tag, for the comparison (this process only)
    home = stage.home
    src = subprocess.run(['git', 'show', 'v0.1.1:vantage/core/joint.py'], cwd=studio.ROOT, check=True,
                         capture_output=True).stdout
    old_path = os.path.join(home, 'joint_011.py')
    with open(old_path, 'wb') as fh:
        fh.write(src)
    spec = importlib.util.spec_from_file_location('vantage.core.joint_011', old_path)
    old = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = old
    spec.loader.exec_module(old)
    new_solve = JOINT.joint_solve

    # ---- a real batch run (not filmed): the same Monte Carlo check from the command line
    work = os.path.join(home, 'projects')
    bin_ = os.path.join(home, 'bin')
    os.makedirs(work, exist_ok=True)
    os.makedirs(bin_, exist_ok=True)
    from vantage.core import samples
    q = samples.tigor_sample_project()
    q.photos.source = ''
    q.save(os.path.join(work, 'tigor.vantage'), b'')
    with open(os.path.join(bin_, 'vantage'), 'w') as fh:
        fh.write(f'#!/bin/sh\nexec {sys.executable} {os.path.join(studio.ROOT, "run_vantage.py")} "$@"\n')
    os.chmod(os.path.join(bin_, 'vantage'), 0o755)
    env = {k: v for k, v in os.environ.items() if k not in ('QT_SCALE_FACTOR',)}
    env.update(PATH=bin_ + os.pathsep + env.get('PATH', ''), HOME=home, PYTHONUSERBASE=site.USER_BASE,
               QT_QPA_PLATFORM='offscreen')
    noise = (lambda ln_: not ln_.startswith(('This plugin does not', '[ WARN', 'QFont')))
    b_cmd = 'vantage tigor.vantage --observability obs/ --verify 30'
    c_b, l_b, s_b = H.run_timed(b_cmd, work, env, home)
    q_cmd = "jq '.monte_carlo | .joint_agree, .cameras.stalled' obs/observability.json"
    c_q, l_q, _ = H.run_timed(q_cmd, work, env, home)
    print('batch:', c_b, round(s_b, 1), [x for _, x in l_b], [x for _, x in l_q], flush=True)
    r.expect(c_b, 0, 'batch exit code')
    r.expect([x for _, x in l_q], ['true', '0'], 'batch: the joint solve agrees, none stopped short')

    # ---- the window: the sample, with the lidar calibration's settings for later
    def setup(p):
        p.analysis.lidar_calib = {'camera': 'F', 'lidar': 'L1', 'seed': 4, 'err_x_mm': 15, 'err_z_mm': 8,
                                  'err_yaw_deg': 0.6}
    w = stage.window(setup)
    r.win = w
    st = w.store
    vp = stage.vp
    stage.tilde(w.msg_label)

    # 0 - the subject, over the rig and its floor boards
    r.lower_third()
    r.say(0)
    r.cursor_on = False
    stage.orbit(r, vp, 24, max(3.0, r.speech_left() + 0.3), d_el=5)
    r.finish_speech(0.2)

    # 1 - the installation check: Run check (a time-lapse)
    r.say(1)
    r.cursor_on = True
    r.menu('Calibrate', 'Installation check')
    pg = w.page('install')
    stage.settle(3)
    r.crossfade(0.4)
    r.until_word(1, 'then solves')
    r.click(pg.b_run, real=False)
    pg.b_run.click()
    r.move(r.wpos(pg.tile_host, None), 0.8)
    r.cursor_on = False
    H.wait_job(r, lambda: not pg.busy and bool(pg.results), timeout=900, fast=16.0)
    stage.settle(1)
    r.finish_speech(0.2)

    # 2 - the joint solve's line and the boards it moved
    jl = pg.joint_lbl.text()
    print('joint:', jl, flush=True)
    r.expect('6 cameras, 8 boards, 150 shared corners' in jl, True, 'the joint line')
    bt = pg.btable
    boards = [[bt.item(i, j).text() for j in range(bt.columnCount())] for i in range(bt.rowCount())]
    print('boards:', boards, flush=True)
    r.say(2)
    jr = H.union(r.wrect(pg.joint_lbl), r.wrect(pg.btable))
    r.zoom(QRectF(jr.x() - 20, jr.y() - 30, jr.width() + 40, jr.height() + 60), 1.0)
    r.callout(pg.joint_lbl, dur=max(2.4, r.at_word(2, "and each board's") - r.now()), pad=4, dim=0.2)
    r.until_word(2, "and each board's")
    r.callout(lambda: H.rows_rect(r, bt), dur=max(2.0, r.speech_left() + 0.3), pad=3, dim=0.2)
    r.finish_speech(0.2)

    # 3 - one check cannot tell
    r.say(3)
    r.zoom(None, 0.9)
    r.hold(max(1.0, r.speech_left() - 0.2))
    r.finish_speech(0.2)

    # 4 - Calibration observability, Verify
    r.say(4)
    r.cursor_on = True
    r.menu('Calibrate', 'Calibration observability')
    ob = w.page('observability')
    stage.settle(2)
    r.crossfade(0.4)
    if ob.result is None and not ob.busy:
        ob.run()
    r.wait_for(lambda: ob.result is not None and not ob.busy, timeout=120)
    stage.settle(1)
    r.expect(ob.result.settings.mc_trials, 30, 'Monte Carlo trials')
    sa = ob.b_verify.parentWidget()
    while sa is not None and not isinstance(sa, QScrollArea):
        sa = sa.parentWidget()
    if sa is not None:
        sa.ensureWidgetVisible(ob.b_verify)
        stage.settle(0.3)
    r.until_word(4, 'Verify simulates')
    vb = r.wrect(ob.b_verify)
    r.zoom(QRectF(vb.x() + vb.width() - 620, vb.y() - 330, 660, 380), 0.9)
    r.callout(ob.b_verify, dur=max(2.0, r.at_word(4, 'and solves each') - r.now()), pad=4)
    r.until_word(4, 'and solves each')

    # 5 (9, 10) - Verify with 0.1.1's joint solve; the cause and the fix told while it runs (in real time, then a
    # marked time-lapse for the rest)
    JOINT.joint_solve = old.joint_solve
    note = r.note("0.1.1's joint solve, read from the v0.1.1 source for this comparison")
    r.click(ob.b_verify, real=False)
    ob.verify()
    mc_done = (lambda: not ob.mc_busy and ob.result.mc is not None)

    def real_time_until_free():
        r.wait_for(lambda: mc_done() or r.now() >= r.speech_end + 0.4, timeout=600, speed=1)
    real_time_until_free()
    r.say(5)
    r.zoom(None, 0.8)
    r.cursor_on = False
    real_time_until_free()
    r.say(9)
    r.zoom(ob.plan, dur=r.line(9) + r.line(10) + 1.0, wait=False, margin=60)
    r.statement('It ran out of evaluations', 'its steps barely moved where all cameras and boards shift together',
                dur=r.line(9) + 0.2, eyebrow='The cause', scrim=0.92)
    real_time_until_free()
    r.say(10)
    r.statement('Each step solved exactly', 'least squares, then the robust loss · boards held near their marks',
                dur=r.line(10) + 0.2, eyebrow='The fix', scrim=0.92)
    r.wait_for(lambda: mc_done() or r.now() >= r.speech_end + 0.2, timeout=600, speed=1)
    if not mc_done():
        r.wait_for(mc_done, timeout=1500, speed=80.0)
    JOINT.joint_solve = new_solve
    stage.settle(1)
    mc = ob.result.mc
    cam = mc['cameras']
    print('mc 0.1.1:', ob.mc_note.text(), flush=True)
    t = ob.mc_table
    rows = {(t.item(i, 0).text(), t.item(i, 1).text()): [t.item(i, j).text() for j in range(2, 6)]
            for i in range(t.rowCount())}
    print('F yaw', rows[('F', 'yaw °')], 'ML z', rows[('ML', 'z mm')], flush=True)
    fy = [float(x) for x in rows[('F', 'yaw °')][:3]]
    mz = [float(x) for x in rows[('ML', 'z mm')][:3]]
    r.expect((cam['solved'], 0 < cam['stalled'] < 30), (30, True), 'some stopped short with 0.1.1')
    r.expect((f'{fy[0]:.2f}', f'{fy[2]:.2f}', fy[1] / fy[0] > 2.0), ('0.13', '0.13', True), 'F yaw, 0.1.1')
    r.expect(3.4 <= mz[1] / mz[0] < 4.0, True, f'ML z nearly four times ({mz[1] / mz[0]:.2f})')
    r.finish_speech(0.2)

    # 6 - the note; how to read the table
    r.say(6)
    mn = r.wrect(ob.mc_note)
    tr = r.wrect(t)
    r.zoom(QRectF(tr.x() - 10, mn.y() - 20, tr.width() + 20, min(tr.bottom(), mn.y() + 420) - mn.y() + 40), 1.0)
    r.figure('Stopped short', cam['stalled'], f'of {cam["solved"]}', 0, dur=max(2.4, r.at_word(6, 'How to read') - r.now()),
             src=ob.mc_note)
    hh = t.horizontalHeader()

    def head(col):
        def f():
            x = hh.sectionViewportPosition(col)
            p0 = r.wpos(hh, None)
            g = r.wrect(hh)
            return QRectF(g.x() + x, g.y(), hh.sectionSize(col), g.height())
        return f
    for col, frag in ((2, 'Predicted is'), (3, 'Joint solve, the'), (4, 'Optimum, the')):
        r.until_word(6, frag)
        r.callout(head(col), dur=2.0, pad=2, dim=0.2)
    r.finish_speech(0.2)

    # 7 - two rows
    r.say(7)
    r.callout(H.cell(r, t, 0), dur=max(3.0, r.at_word(7, "M L's height") - r.now()), pad=2, dim=0.3)
    r.until_word(7, "M L's height")
    ml_row = next(i for i in range(t.rowCount()) if t.item(i, 0).text() == 'ML' and t.item(i, 1).text() == 'z mm')
    t.scrollToItem(t.item(ml_row, 0))
    stage.settle(0.3)
    r.callout(H.cell(r, t, ml_row), dur=max(2.0, r.speech_left() + 0.3), pad=2, dim=0.3)
    r.finish_speech(0.2)
    r.close_layer(note)
    t.scrollToTop()

    # 8 - with 0.2.0's solve, the same captures (real time while it runs)
    r.say(8)
    r.zoom(None, 0.6)
    ob.verify()
    H.wait_job(r, lambda: not ob.mc_busy and ob.result.mc is not None and ob.result.mc['cameras'].get('stalled') == 0
               and 'in 0 of 30' in ob.mc_note.text(), timeout=600, fast=4.0)
    stage.settle(1)
    cam = ob.result.mc['cameras']
    rows = {(t.item(i, 0).text(), t.item(i, 1).text()): [t.item(i, j).text() for j in range(2, 6)]
            for i in range(t.rowCount())}
    print('mc 0.2.0:', ob.mc_note.text(), rows[('F', 'yaw °')], flush=True)
    r.expect((cam['stalled'], cam['solved']), (0, 30), 'stopped short with 0.2.0')
    r.expect(all(abs(float(v[1]) - float(v[2])) <= 0.01 * float(v[2]) + 1e-9 for k, v in rows.items()
                 if '(pair)' not in k[0]), True, 'joint = optimum')
    r.zoom(QRectF(tr.x() - 10, mn.y() - 20, tr.width() + 20, min(tr.bottom(), mn.y() + 420) - mn.y() + 40), 0.8)
    r.callout(ob.mc_note, dur=2.2, pad=3, dim=0.25)
    r.until_word(8, 'the joint solve')
    r.callout(lambda: H.union(head(3)(), head(4)()).adjusted(0, 0, 0, 260), dur=max(2.0, r.speech_left() + 0.2),
              pad=2, dim=0.2)
    r.finish_speech(0.2)

    r.zoom(None, 0.8)

    # 11, 12 - the measured before/after
    r.say(11)
    _card(r, 'The joint solve, before and after',
                 f'{jo["trials"]} simulated installation checks of the sample, the same captures, against an independent '
                 'reference solve · measured for this video',
                 ['', '0.1.1', '0.2.0'],
                 [['Stopped short of its optimum', f'{jo["stalled"]} of {jo["trials"]}', f'{jn["stalled"]} of {jn["trials"]}'],
                  ['Scatter over its stated uncertainty', f'{jo["ratio_min"]:.2f}–{jo["ratio_max"]:.2f}×',
                   f'{jn["ratio_min"]:.2f}–{jn["ratio_max"]:.2f}×'],
                  ['Time per joint solve', f'{jo["seconds_per_solve"]:.2f} s', f'{jn["seconds_per_solve"]:.2f} s']],
                 foot='cameras up to 1° and 10 mm off their plan · boards laid ±20 mm, ±1° · 0.5 px corner noise',
                 col_w=[0.46, 0.27, 0.27], colors=[None, '#ec835a', '#86b6ef'])
    r.crossfade(0.5)
    jw = [0.46, 0.27, 0.27]
    r.until_word(11, 'stopped short in nearly all')
    r.callout(_cell(jw, 0, 1), dur=2.4, pad=8, dim=0.3)
    r.until_word(11, 'and scattered up to')
    r.callout(_cell(jw, 1, 1), dur=max(2.0, r.speech_left()), pad=8, dim=0.3)
    r.finish_speech(0.2)
    r.say(12)
    r.callout(_cell(jw, (0, 1), 2), dur=max(2.4, r.at_word(12, 'and a joint solve') - r.now()), pad=8, dim=0.3)
    r.until_word(12, 'and a joint solve')
    r.callout(_cell(jw, 2, None), dur=max(2.0, r.speech_left() + 0.4), pad=8, dim=0.3)
    r.hold(max(1.0, r.speech_left() + 0.6))
    r.card = None

    # 13 - the data rig; the lidar–camera page
    r.say(13)
    r.crossfade(0.5)
    r.cursor_on = True
    rig_i = next(i for i in range(w.variant_combo.count()) if 'Data rig' in w.variant_combo.itemText(i))
    r.choose(w.variant_combo, rig_i)
    stage.settle(4)
    r.expect(st.variant.name, 'Data rig · 6-cam + lidar + radars', 'the data rig')
    r.until_word(13, 'and the roof lidar')
    r.menu('Calibrate', 'Lidar–camera calibration')
    lp = w.page('lidar_calib')
    r.wait_for(lambda: lp.plan is not None and bool(lp.plan.get('poses')), timeout=180)
    stage.settle(1)
    r.crossfade(0.4)
    r.expect(lp.cam_box.currentText(), 'F · 15 %', 'the camera picked')

    # 14 - the prediction now, and 0.1.1's
    pn = lp.plan_note.text()
    print('plan note:', pn, flush=True)
    r.expect('±4.2 / ±6.3 / ±16.4 mm' in pn, True, 'the prediction')
    r.say(14)
    nr = r.wrect(lp.plan_note)
    r.zoom(QRectF(nr.x() - 20, nr.y() - 120, nr.width() + 40, nr.height() + 150), 1.0)
    r.callout(lp.plan_note, dur=max(2.6, r.at_word(14, 'Zero point one point one said') - r.now()), pad=4, dim=0.25)
    r.until_word(14, 'Zero point one point one said')
    r.zoom(None, 0)                      # the card is not zoomed: its callouts are in frame coordinates
    _card(r, 'Lidar–camera: the predicted precision',
                 f'F and L1 on the data rig, the plan\'s 6 boards, from their planes alone · {lo["trials"]} simulated '
                 'calibrations measured for this video',
                 ['Axis (1σ)', '0.1.1 predicted', '0.2.0 predicted', 'Real scatter'],
                 [[k.split('_')[0] + ' mm', f'±{lo["axes"][k]["predicted"]:.1f}', f'±{ln["axes"][k]["predicted"]:.1f}',
                   f'{ln["axes"][k]["scatter"]:.1f}'] for k in ax] +
                 [['roll / pitch / yaw °', ' / '.join(f'{lo["axes"][k]["predicted"]:.2f}' for k in ('roll_deg', 'pitch_deg', 'yaw_deg')),
                   ' / '.join(f'{ln["axes"][k]["predicted"]:.2f}' for k in ('roll_deg', 'pitch_deg', 'yaw_deg')),
                   ' / '.join(f'{ln["axes"][k]["scatter"]:.2f}' for k in ('roll_deg', 'pitch_deg', 'yaw_deg'))]],
                 foot='0.1.1\'s figures are what its page showed: ±0.9 / ±1.7 / ±3.4 mm',
                 col_w=[0.28, 0.24, 0.24, 0.24], colors=[None, '#ec835a', '#86b6ef', '#f0efec'])
    r.crossfade(0.5)
    lw = [0.28, 0.24, 0.24, 0.24]
    r.hold(1.2)
    r.callout(_cell(lw, (0, 2), 1), dur=max(2.0, r.speech_left() + 0.2), pad=8, dim=0.3)
    r.finish_speech(0.2)
    lidar_card = r.card

    # 15 - why: the plan's boards, 4 to 8 m from the camera
    r.say(15)
    r.card = None
    r.crossfade(0.5)
    r.zoom(lp.plan_view, 1.2, margin=20)
    r.hold(max(1.0, r.speech_left()))
    r.finish_speech(0.2)

    # 16 - the check against 300 calibrations: back to the card
    r.say(16)
    r.zoom(None, 0)
    r.card = lidar_card
    r.crossfade(0.5)
    r.until_word(16, 'it matches the real scatter')
    r.callout(_cell(lw, (0, 3), (2, 3)), dur=max(2.0, r.speech_left() + 0.3), pad=8, dim=0.3)
    r.finish_speech(0.2)

    # 17 - the full calibration: Capture and solve (a time-lapse)
    r.say(17)
    r.card = None
    r.crossfade(0.5)
    r.zoom(None, 0.6)
    r.until_word(17, 'Capture, and solve')
    r.cursor_on = True
    r.click(lp.b_run, real=False)
    lp.b_run.click()
    r.cursor_on = False
    H.wait_job(r, lambda: lp.result is not None and not lp.busy, timeout=600, fast=10.0)
    stage.settle(1)
    find = [f.text.text() for f in lp.findings]
    print('findings:', find, flush=True)
    r.expect(lp.result.message, 'L1 is 15 mm / 0.61° from its planned pose: adjust the bracket, or apply the measured '
                                'pose to the variant.', 'the verdict')
    r.expect('Determined to ±2.1 mm / ±0.020° (1σ) from 6 poses.' in find, True, 'the uncertainty')
    r.expect('Synthetic: the solve is 2.2 mm / 0.021° from the true mounting.' in find, True, 'against the truth')
    r.finish_speech(0.2)

    # 18 - the findings
    r.say(18)
    fr = r.wrect(lp.find_host)
    t0, t1 = r.wrect(lp.tiles['verdict']), r.wrect(lp.tiles['sigma'])
    area = H.union(QRectF(t0.x(), t0.y(), t1.right() - t0.x(), t1.bottom() - t0.y()), QRectF(fr.x(), fr.y(), fr.width(), 160))
    r.zoom(area.adjusted(-20, -30, 20, 30), 1.0)
    r.callout(lp.tiles['pos'], dur=2.2, pad=3, dim=0.2)
    r.until_word(18, 'and determined')
    r.callout(lp.tiles['sigma'], dur=2.0, pad=3, dim=0.2)
    r.until_word(18, 'The solve lands')
    r.callout(lambda: r.wrect(lp.findings[-1]), dur=max(2.0, r.speech_left() + 0.3), pad=3, dim=0.25)
    r.finish_speech(0.2)

    # 19, 20 - the bias, and the fix (a card)
    r.say(19)
    r.zoom(None, 0.8)
    r.statement('A lean of 3.4 mm in x', 'the noisy point itself measured its distance to the board',
                dur=r.line(19) + 0.2, eyebrow='The bias', scrim=0.92)
    r.finish_speech(0.2)
    r.say(20)
    _card(r, 'Lidar–camera: the lean, before and after',
                 f'{ln["trials"]} simulated calibrations of F and L1 from the board planes · measured for this video',
                 ['', '0.1.1', '0.2.0'],
                 [['Mean error in x', f'{lo["axes"]["x_mm"]["bias"]:+.2f} mm ({lo["axes"]["x_mm"]["bias_se"]:+.1f} s.e.)',
                   f'{ln["axes"]["x_mm"]["bias"]:+.2f} mm ({ln["axes"]["x_mm"]["bias_se"]:+.1f} s.e.)'],
                  ['Mean error in z', f'{lo["axes"]["z_mm"]["bias"]:+.2f} mm ({lo["axes"]["z_mm"]["bias_se"]:+.1f} s.e.)',
                   f'{ln["axes"]["z_mm"]["bias"]:+.2f} mm ({ln["axes"]["z_mm"]["bias_se"]:+.1f} s.e.)'],
                  ['Time per solve', f'{lo["seconds_per_solve"]:.2f} s', f'{ln["seconds_per_solve"]:.2f} s']],
                 foot='s.e.: standard errors of the mean over the trials',
                 col_w=[0.30, 0.35, 0.35], colors=[None, '#ec835a', '#86b6ef'])
    r.crossfade(0.5)
    bw = [0.30, 0.35, 0.35]
    r.until_word(20, 'the lean is gone')
    r.callout(_cell(bw, (0, 1), 2), dur=2.4, pad=8, dim=0.3)
    r.until_word(20, 'several times faster')
    r.callout(_cell(bw, 2, None), dur=max(1.8, r.speech_left() + 0.4), pad=8, dim=0.3)
    r.hold(max(1.0, r.speech_left() + 0.6))
    r.finish_speech(0.2)

    # 21 - what follows the fix
    r.say(21)
    H.table_card(r, 'Everything built on the solves follows', 'same flags, same keys: the corrected figures',
                 ['Where', 'What', 'Now'],
                 [['Requirements', 'lidar_calib.planned_sigma_mm', 'named "from the board planes alone"; target 20 mm'],
                  ['Batch mode', '--install-check  --no-joint  --board-tol', '--lidar-calib  ·  --observability --verify N'],
                  ['Exports and assistant', 'the joint poses, the lidar transform', 'with the uncertainty they really have'],
                  ['Tests', 'tests/test_calib_accuracy.py', 'both solves against an independent reference']],
                 col_w=[0.22, 0.4, 0.38])
    r.crossfade(0.5)
    r.until_word(21, 'The requirements scorecard')
    r.callout(QRectF(190, 380, 1540, 92), dur=max(2.0, r.at_word(21, 'Batch mode') - r.now()), pad=8, dim=0.3)
    r.until_word(21, 'Batch mode')
    r.callout(QRectF(190, 490, 1540, 202), dur=max(2.0, r.speech_left() + 0.3), pad=8, dim=0.3)
    r.hold(max(1.0, r.speech_left() + 0.4))
    r.finish_speech(0.2)

    # 22 - the batch check (the real run, replayed)
    r.say(22)
    # the run's output at its real pace, unless that outlasts the line: then sped up, and marked as a time-lapse
    out_s = l_b[-1][0] if l_b else 0.0
    k = max(1.0, out_s / max(3.0, r.line(22) - 6.5))
    span = H.terminal(r, [('cmd', b_cmd)] + H.timed_output(l_b, speed=k, keep=noise, tones=True) +
                      [('wait', 0.3), ('cmd', q_cmd)] + H.timed_output(l_q), title='~/projects')
    if k > 1.05:
        r.timelapse(k)
    r.crossfade(0.4)
    r.hold(max(span + 0.2, r.speech_left() + 0.2))
    r.timelapse(None)

    # 23 - limits, over the page
    r.say(23)
    r.card = None
    r.crossfade(0.6)
    r.statement('Limits', 'boards to about a centimetre · a board beyond its reach is re-laid · the lidar prediction '
                          'is cautious', dur=r.line(23) + 0.3, eyebrow='Calibration', scrim=0.92)
    r.cursor_on = False
    r.finish_speech(1.0)


def _cell(col_w, rows, cols):
    """A rectangle on _card's table: rows / cols an index, an inclusive (first, last) pair, or None for all columns
    (rows None: the header and every row)."""
    from PyQt6.QtCore import QRectF
    x0, w = 190.0, 1920 - 380.0
    c0, c1 = (0, len(col_w) - 1) if cols is None else (cols, cols) if isinstance(cols, int) else cols
    r0, r1 = (rows, rows) if isinstance(rows, int) else rows
    x = x0 + w * sum(col_w[:c0])
    return QRectF(x, 382 + 100 * r0, w * sum(col_w[c0:c1 + 1]) - 16, 100 * (r1 - r0) + 80)


def _card(r, title, sub, header, rows, foot='', col_w=None, colors=None):
    """A full-frame before/after table: a title, a subtitle, a header and rows; the first column a label, the others
    figures in mono, each column in its own colour (colors[j]); rows fade in one after another."""
    from PyQt6.QtCore import QRectF, Qt
    from PyQt6.QtGui import QColor, QImage, QPainter
    import gfx
    f0 = r.f
    n = len(header)
    col_w = col_w or [1.0 / n] * n
    colors = colors or [gfx.INK] * n

    def card(rec):
        t = (rec.f - f0) / rec.fps
        img = QImage(gfx.W, gfx.H, QImage.Format.Format_RGB32)
        p = QPainter(img)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        p.setRenderHint(QPainter.RenderHint.TextAntialiasing)
        p.fillRect(QRectF(0, 0, gfx.W, gfx.H), QColor(gfx.BASE))
        gfx.glow(p, gfx.W * 0.5, gfx.H * 0.4, gfx.W * 0.55, gfx.ACCENT, 0.06)
        gfx.vignette(p, 0.5)
        x0, w = 190.0, gfx.W - 380.0
        p.setOpacity(gfx.smoother(t / 0.5))
        p.setFont(gfx.font(54, 'semibold'))
        p.setPen(gfx.col(gfx.INK))
        p.drawText(QRectF(x0, 150, w, 70), Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter, title)
        p.setFont(gfx.font(24))
        p.setPen(gfx.col(gfx.MUTED))
        p.drawText(QRectF(x0, 222, w, 40), Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter, sub)
        y = 320.0
        x = x0
        for j, (h, cw) in enumerate(zip(header, col_w)):
            p.setFont(gfx.font(19, 'semibold', mono=j > 0, spacing=1.0 if j else 0.0))
            p.setPen(gfx.col(colors[j] if j else gfx.MUTED))
            p.drawText(QRectF(x, y, w * cw - 20, 34), Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter, h)
            x += w * cw
        p.fillRect(QRectF(x0, y + 42, w, 1), gfx.col(gfx.HAIR))
        y += 62
        for i, row in enumerate(rows):
            ra = gfx.smoother((t - 0.5 - 0.35 * i) / 0.5)
            if ra <= 0:
                break
            p.setOpacity(ra)
            x = x0
            for j, (txt, cw) in enumerate(zip(row, col_w)):
                p.setFont(gfx.font(26, 'semibold') if j == 0 else gfx.font(30, 'medium', mono=True))
                p.setPen(gfx.col(gfx.INK if j == 0 else colors[j]))
                p.drawText(QRectF(x, y, w * cw - 24, 80), Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter
                           | Qt.TextFlag.TextWordWrap, txt)
                x += w * cw
            p.setOpacity(ra * 0.6)
            p.fillRect(QRectF(x0, y + 88, w, 1), gfx.col(gfx.HAIR))
            y += 100
        if foot:
            p.setOpacity(gfx.smoother((t - 0.8 - 0.35 * len(rows)) / 0.5))
            p.setFont(gfx.font(21, mono=True))
            p.setPen(gfx.col(gfx.MUTED))
            p.drawText(QRectF(x0, y + 24, w, 40), Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter, foot)
        p.end()
        return img
    r.card = card
    return card
