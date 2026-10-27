"""48 Automation and CI (episode v020, chapter 8): Export ▸ Automation on the sample (six cameras), saved in the
throw-away home as ~/rig-design/tigor.vantage beside a Makefile whose `test` target runs a short Python check of the
regenerated files (as demo/clip_automation.py does: every camera has its OpenCV and camera_info file; what changed; the
requirements check's verdict), with the plan's requirements the six-camera rig meets. Every control of the page; the
approval dialog for the harmless command `make test`; a save that regenerates the calibration files, runs the
requirements check and the command; F moved 5 cm and saved again (only F's files change); ROS camera_info unticked and
run by hand (the stale files removed, the check fails, exit 1); then batch mode's --watch and the CI templates, from
real runs made while recording (not filmed) in a second folder of the throw-away home, replayed in terminal cards.

Figures read from the app while recording and checked with r.expect: the files written by the first run, the files
changed by moving F, the requirements verdict, the exit codes, the files removed by unticking camera_info.
"""

META = {
    'id': 'new-automation',
    'part': 3,
    'title': 'Automation and CI',
    'promise': 'The calibration files and your own checks, in step with every change.',
    'menu': 'Export ▸ Automation  ·  vantage --watch  ·  CI templates',
}

LINES = [
    # 0
    ("A rig design keeps changing: a camera moves five centimetres, a lens is swapped. Everything downstream has to "
     "follow: the calibration files, and the data pipeline's own tests.",
     "A rig design keeps changing: a camera moves five centimetres, a lens is swapped. Everything downstream has to "
     "follow: the calibration files, and the data pipeline's own tests."),
    # 1
    ("Automation keeps them in step. Export ▸ Automation: when the design changes, Vantage regenerates the files and "
     "runs your own checks. It is off until you turn it on.",
     "Automation keeps them in step. Export, Automation. When the design changes, Vantage regenerates the files, and "
     "runs your own checks. It is off until you turn it on."),
    # 2
    ("Three triggers: on save; after changes, once nothing has changed for a quiet period, unsaved edits included; or "
     "only by hand, with Run now.",
     "Three triggers: on save; after changes, once nothing has changed for a quiet period, unsaved edits included; or "
     "only by hand, with Run now."),
    # 3
    ("What to regenerate is kept in the project: a folder next to the project file, never outside its folder; the "
     "variant, the frame, the units and the angles.",
     "What to regenerate is kept in the project: a folder next to the project file, never outside its folder; the "
     "variant, the frame, the units, and the angles."),
    # 4
    ("And the formats. By default the calibration files a data pipeline reads: extrinsics, OpenCV and ROS camera "
     "info per camera, URDF and static TF, nuScenes, KITTI and Kalibr.",
     "And the formats. By default, the calibration files a data pipeline reads: extrinsics, open C V and ROS camera "
     "info per camera, U R D F and static T F, new scenes, KITTI, and Kalibr."),
    # 5
    ("Two switches: the requirements check writes the scorecard and the review pack; and files the last run wrote "
     "and this one did not are removed, so nothing stale is left. Your own files are never touched.",
     "Two switches. The requirements check writes the scorecard and the review pack. And files the last run wrote, "
     "and this one did not, are removed, so nothing stale is left. Your own files are never touched."),
    # 6
    ("Then your own command. Here make test: a short Python script beside the project that checks the regenerated "
     "files. It is kept in your settings on this computer, for this project file, never in the project.",
     "Then your own command. Here, make test: a short Python script beside the project that checks the regenerated "
     "files. It is kept in your settings on this computer, for this project file, never in the project."),
    # 7
    ("Save asks for approval. The dialog shows the command in full, the folder it runs in, the time limit and the "
     "variables it gets. Cancel is the default; Allow approves this exact text.",
     "Save asks for approval. The dialog shows the command in full, the folder it runs in, the time limit, and the "
     "variables it gets. Cancel is the default. Allow approves this exact text."),
    # 8
    ("Change one character and it asks again. And a project file you are sent can never set or run a command: one "
     "that names a command is ignored, and the Issues list says so.",
     "Change one character, and it asks again. And a project file you are sent can never set or run a command: one "
     "that names a command is ignored, and the Issues list says so."),
    # 9
    ("Now save the project. Vantage regenerates the calibration files, runs the requirements check, then the "
     "command, whose output streams in as it runs.",
     "Now, save the project. Vantage regenerates the calibration files, runs the requirements check, then the "
     "command, whose output streams in as it runs."),
    # 10
    ("How to read the tiles: the run passed; the command's exit code, 0; the duration; 25 files new, plus the run's "
     "own summary; and the requirements: pass, all 10 musts.",
     "How to read the tiles: the run passed; the command's exit code, zero; the duration; twenty-five files new, plus "
     "the run's own summary; and the requirements: pass, all ten musts."),
    # 11
    ("The command gets the export folder, the project, the variant, the requirements check's exit code, and "
     "VANTAGE_SUMMARY: a JSON of the run, every file with its SHA-256, what changed and the verdict.",
     "The command gets the export folder, the project, the variant, the requirements check's exit code, and Vantage "
     "summary: a Jason file of the run, with every file and its S H A two fifty-six, what changed, and the verdict."),
    # 12
    ("That is what the example script reads. Now move the front camera 5 centimetres back, and save again.",
     "That is what the example script reads. Now, move the front camera five centimetres back, and save again."),
    # 13
    ("This run lists what changed: 10 files. F's OpenCV file, the files that hold the whole rig, and the scorecard. "
     "The other 14, every other camera's among them, are unchanged.",
     "This run lists what changed: ten files. F's open C V file, the files that hold the whole rig, and the "
     "scorecard. The other fourteen, every other camera's among them, are unchanged."),
    # 14
    ("Files are compared by content: the time stamps exports carry do not count, so a save with no change reports "
     "nothing changed. A PDF embeds its creation time, so it is listed as rewritten.",
     "Files are compared by content: the time stamps exports carry do not count, so a save with no change reports "
     "nothing changed. A P D F embeds its creation time, so it is listed as rewritten."),
    # 15
    ("Cause and effect: untick ROS camera info and run by hand. The six stale files are removed, the test finds them "
     "missing, and make fails with exit code 2.",
     "Cause and effect: untick ROS camera info, and run by hand. The six stale files are removed, the test finds them "
     "missing, and make fails, with exit code two."),
    # 16
    ("The history keeps every run on this computer: trigger, result, exit code, time and files changed. Stop, or "
     "the time limit, ends the command with everything it started; Export run writes the run as JSON, without the "
     "command.",
     "The history keeps every run on this computer: trigger, result, exit code, time, and files changed. Stop, or the "
     "time limit, ends the command with everything it started. Export run writes the run as Jason, without the "
     "command."),
    # 17
    ("Without a window, on a CI box or a Jetson: vantage --watch regenerates at the start and after every change of "
     "the file, runs the command given on the command line, and exits with its code.",
     "Without a window, on a C I box, or a Jetson: vantage watch regenerates at the start, and after every change of "
     "the file, runs the command given on the command line, and exits with its code."),
    # 18
    ("CI templates writes a GitHub Actions workflow and a GitLab CI job. They install this version, run export and "
     "the requirements check on the project in its repository, and keep the calibration files and the review pack.",
     "C I templates writes a GitHub Actions workflow, and a GitLab C I job. They install this version, run export, "
     "and the requirements check, on the project in its repository, and keep the calibration files and the review "
     "pack."),
    # 19
    ("The job fails as batch mode does: exit code 2 on a critical issue in the plan, 5 when a must requirement "
     "fails. The assistant can read the automation, but never the command's text, and cannot run it.",
     "The job fails as batch mode does: exit code two on a critical issue in the plan, five when a must requirement "
     "fails. The assistant can read the automation, but never the command's text, and it cannot run it."),
    # 20
    ("Limits: the command runs with your permissions, unsandboxed; approving its exact text is the gate. It is "
     "stored as plain text: keep secrets in environment variables.",
     "Limits. The command runs with your permissions, unsandboxed: approving its exact text is the gate. It is stored "
     "as plain text, so keep secrets in environment variables."),
    # 21
    ("Images of the window cannot be regenerated in the background, the first run on a computer removes nothing, "
     "and the CI templates were checked locally, not on hosted runners.",
     "Images of the window cannot be regenerated in the background, the first run on a computer removes nothing, "
     "and the C I templates were checked locally, not on hosted runners."),
]

MAKEFILE = '''# the checks to run after Vantage regenerates the calibration files
test:
\tpython3 check_calibration.py
'''

CHECK = '''"""Check the calibration files Vantage regenerated (VANTAGE_EXPORT_DIR) against its run summary."""
import json
import os
import sys

summary = json.load(open(os.environ['VANTAGE_SUMMARY']))
files = {f['path'] for f in summary['files']}
changed = summary['changed']
print(f"run {summary['run_id']} on {summary['trigger']}, variant {summary['variant']}")
print(f"{len(files)} files: {len(changed['added'])} new, {len(changed['changed'])} changed, "
      f"{len(changed['removed'])} removed")
for name in changed['changed']:
    print(f"  changed  {name}")
tags = sorted(os.path.splitext(os.path.basename(f))[0] for f in files if f.startswith('opencv/'))
missing = [t for t in tags if f'camera_info/{t}.yaml' not in files]
print(f"cameras with OpenCV files: {' '.join(tags)}")
print(f"camera_info files missing: {' '.join(missing)}" if missing else "every camera has its camera_info file")
req = summary['requirements'] or {}
print(f"requirements: {req.get('verdict_text', 'not run')} (must {req.get('musts', '-')}), "
      f"exit {os.environ.get('VANTAGE_CHECK_EXIT', '')}")
ok = not missing and os.environ.get('VANTAGE_CHECK_EXIT') in ('0', '')
print('OK' if ok else 'FAILED')
sys.exit(0 if ok else 1)
'''

# the plan's requirements (Requirements ▸ Add from the plan) that the six-camera rig fails are the requirements
# chapter's subject; this one keeps the rest, so the check passes and the run shows its verdict (as the clip does)
LEAVE_OUT = ('No critical issue', 'No blind ground', 'resolved to 80 m in the front sector', 'h of driving per disk')


def scene(stage, r):
    import os
    import site
    import sys
    from PyQt6.QtCore import QPointF, QRectF
    from PyQt6.QtTest import QTest
    from PyQt6.QtWidgets import QLabel
    from vantage.core import requirements as RQ, samples
    from vantage.ui.workspaces import automation_ws as AW
    import helpers_tourc as H
    import studio

    home = stage.home
    # ---- the project beside its Makefile and check script (the clip's set-up)
    folder = os.path.join(home, 'rig-design')
    os.makedirs(folder, exist_ok=True)
    for name, text in (('Makefile', MAKEFILE), ('check_calibration.py', CHECK)):
        with open(os.path.join(folder, name), 'w') as fh:
            fh.write(text)
    path = os.path.join(folder, 'tigor.vantage')

    def reqs(p):
        items = [i for i in RQ.from_plan(p, p.variant) if not any(s in i['text'] for s in LEAVE_OUT)]
        p.analysis.requirements = RQ.RequirementsSettings.from_dict({'items': items}).to_dict()

    # ---- the batch runs (not filmed): a second copy in ~/ci-box, as on a CI machine, so its runs never show in the
    # page's history; --watch for one run with the same command, and the CI templates written for it
    box = os.path.join(home, 'ci-box')
    bin_ = os.path.join(home, 'bin')
    os.makedirs(box, exist_ok=True)
    os.makedirs(bin_, exist_ok=True)
    for name, text in (('Makefile', MAKEFILE), ('check_calibration.py', CHECK)):
        with open(os.path.join(box, name), 'w') as fh:
            fh.write(text)
    q = samples.tigor_sample_project()
    q.photos.source = ''
    reqs(q)
    q.analysis.automation = {'enabled': True}
    q.save(os.path.join(box, 'tigor.vantage'), b'')
    with open(os.path.join(bin_, 'vantage'), 'w') as fh:
        fh.write(f'#!/bin/sh\nexec {sys.executable} {os.path.join(studio.ROOT, "run_vantage.py")} "$@"\n')
    os.chmod(os.path.join(bin_, 'vantage'), 0o755)
    env = {k: v for k, v in os.environ.items() if k not in ('QT_SCALE_FACTOR',)}
    env.update(PATH=bin_ + os.pathsep + env.get('PATH', ''), HOME=home, PYTHONUSERBASE=site.USER_BASE,
               QT_QPA_PLATFORM='offscreen')

    def run(cmd):
        code, lines, secs = H.run_timed(cmd, box, env, home)
        print(f'$ {cmd}  -> {code} in {secs:.1f} s', flush=True)
        for _, ln in lines:
            print('   ', ln, flush=True)
        return code, lines, secs
    noise = (lambda ln: not ln.startswith(('This plugin does not', '[ WARN', 'QFont')))
    c_w, l_w, _ = run('vantage --watch tigor.vantage --on-change "make test" --watch-runs 1')
    r.expect(c_w, 0, 'exit code of --watch')
    c_ci, l_ci, _ = run('vantage tigor.vantage --ci-templates .')
    c_ls, l_ls, _ = run('ls ci ci/github ci/gitlab')
    c_cat, l_cat, _ = run('grep -n -E "name:|on:|push|run: |vantage |path" ci/github/vantage.yml | head -24')

    # ---- the window
    def setup(p):
        reqs(p)
        p.save(path)
        p.path = path
    w = stage.window(setup)
    r.win = w
    st = w.store
    ctl = w.ctx.automation
    vp = stage.vp
    stage.tilde(w.msg_label)

    # the approval dialog is modal (exec): shown here without blocking, read, and allowed with the pointer
    def exec_(d):
        stage.tilde(*d.findChildren(QLabel))
        d.adjustSize()
        g = w.geometry()
        d.move(g.x() + (g.width() - d.width()) // 2, g.y() + (g.height() - d.height()) // 2)
        d.show()
        r.hold(0.4)
        r.zoom(r.wrect(d), 0.7, margin=30)
        r.callout(d.text, dur=2.6, pad=4, dim=0.2)
        r.until_word(7, 'the time limit')
        r.callout(lambda: r.wrect(d.findChildren(QLabel)[0]), dur=2.2, pad=4, dim=0.2)
        r.until_word(7, 'Cancel is the default')
        r.callout(d.cancel_b, dur=1.6, pad=4, dim=0.2)
        r.until_word(7, 'Allow approves')
        r.click(d.allow_b, None, 0.6, real=False)
        r.finish_speech(0.2)
        d.hide()
        r.zoom(None, 0.6)
        return 1
    real_exec = AW.ApproveDialog.exec
    AW.ApproveDialog.exec = exec_
    try:
        _drive(stage, r, w, st, ctl, vp, path, H, QPointF, QRectF, QTest, QLabel,
               (c_w, l_w), (c_ci, l_ci), (c_ls, l_ls), (c_cat, l_cat), noise)
    finally:
        AW.ApproveDialog.exec = real_exec


def _save(stage, r, w, ctl, what='Ctrl+S'):
    """Ctrl+S, as the user presses it: the window saves, and automation runs on the save (recorded in real time)."""
    n0 = len(ctl.runs())
    r.key_combo(what)
    r.hold(0.3)
    w.save()
    r.wait_for(lambda: not ctl.busy and len(ctl.runs()) > n0, timeout=180)
    stage.settle(1)


def _drive(stage, r, w, st, ctl, vp, path, H, QPointF, QRectF, QTest, QLabel, watch, ci, ls, cat, noise):
    # 0 - the problem, over the rig
    r.lower_third()
    r.say(0)
    r.cursor_on = False
    stage.orbit(r, vp, 26, max(3.0, r.speech_left() + 0.2), d_el=4)
    r.finish_speech(0.2)

    # 1 - the page; off until turned on
    r.say(1)
    r.cursor_on = True
    r.until_word(1, 'Export, Automation')
    r.menu('Export', 'Automation')
    pg = w.page('automation')
    stage.tilde(*pg.findChildren(QLabel))
    stage.settle(1)
    r.crossfade(0.4)
    left = r.wrect(pg.enabled.parentWidget())
    r.until_word(1, 'It is off')
    r.expect(pg.enabled.isChecked(), False, 'automation off at first')
    r.zoom(QRectF(left.x(), left.y(), left.width() + 260, 420), 0.9)
    r.callout(pg.enabled, dur=1.8, pad=4)
    r.click(pg.enabled, QPointF(12, pg.enabled.height() / 2))
    r.finish_speech(0.2)

    # 2 - the triggers
    r.say(2)
    r.callout(pg.trigger, dur=max(2.0, r.at_word(2, 'after changes') - r.now()), pad=4)
    r.until_word(2, 'quiet period')
    r.callout(pg.quiet, dur=2.4, pad=4)
    r.until_word(2, 'by hand')
    r.callout(pg.trigger, dur=max(1.6, r.speech_left()), pad=4)
    r.finish_speech(0.2)

    # 3 - what: folder, variant, frame, units
    r.say(3)
    r.expect(pg.folder.text(), 'exports/automation', 'the default folder')
    r.callout(pg.folder, dur=2.6, pad=4)
    r.until_word(3, 'the variant')
    r.callout(lambda: H.union(r.wrect(pg.variant), r.wrect(pg.units.parentWidget())), dur=max(2.0, r.speech_left()),
              pad=4)
    r.finish_speech(0.2)

    # 4 - the formats
    r.say(4)
    checked = [cb for cb in pg.format_boxes.values() if cb.isChecked()]
    sb = pg.formats.verticalScrollBar()
    sb.setValue(max(0, min(cb.y() for cb in checked) - 40))         # the calibration groups, their headers too
    stage.settle(0.3)
    fr = r.wrect(pg.formats)
    r.zoom(QRectF(fr.x() - 10, fr.y() - 30, fr.width() + 260, fr.height() + 60), 0.9)
    on = [fid for fid, cb in pg.format_boxes.items() if cb.isChecked()]
    print('formats on:', on, flush=True)
    r.expect(sorted(on), sorted(['extrinsics_json', 'opencv_yaml', 'ros_camera_info', 'ros2', 'nuscenes_calib',
                                 'kitti_calib', 'kalibr_camchain']), 'the default formats')
    r.callout(lambda: H.union(*[r.wrect(cb) for cb in checked if cb.visibleRegion().boundingRect().height() > 4]),
              dur=max(2.4, r.speech_left()), pad=3, dim=0.25)
    r.finish_speech(0.2)

    # 5 - requirements check, remove stale files
    r.say(5)
    r.expect((pg.check.isChecked(), pg.prune.isChecked()), (True, True), 'check and prune on')
    cr = H.union(r.wrect(pg.check), r.wrect(pg.run_cmd))
    r.zoom(QRectF(cr.x() - 10, cr.y() - 60, cr.width() + 260, cr.height() + 140), 0.9)
    r.callout(pg.check, dur=2.6, pad=3)
    r.until_word(5, 'And files the last run')
    r.callout(pg.prune, dur=max(2.4, r.speech_left()), pad=3)
    r.finish_speech(0.2)

    # 6 - the command
    r.say(6)
    cm = H.union(r.wrect(pg.cmd), r.wrect(pg.b_remove_cmd))
    r.zoom(QRectF(cm.x() - 10, cm.y() - 70, cm.width() + 260, cm.height() + 120), 0.9)
    r.click(pg.cmd, None, 0.6)
    for ch in 'make test':
        QTest.keyClicks(pg.cmd, ch)
        r.tick(3)
    r.until_word(6, 'It is kept')
    r.callout(lambda: r.wrect(pg.cmd), dur=max(2.0, r.speech_left()), pad=4)
    r.finish_speech(0.2)

    # 7 - Save… → the approval dialog (exec_) → Allow
    r.say(7)
    r.click(pg.b_save_cmd)
    if not ctl.commands.is_approved(path):
        raise RuntimeError('the command was not approved')
    stage.settle(1)

    # 8 - approval bound to the text; a project file cannot carry a command
    r.say(8)
    r.zoom(QRectF(cm.x() - 10, cm.y() - 70, cm.width() + 260, cm.height() + 120), 0.8)
    r.callout(pg.cmd_state, dur=max(2.4, r.at_word(8, 'And a project') - r.now()), pad=3)
    r.finish_speech(0.2)
    r.zoom(None, 0.6)

    # 9 - a save: regenerate, check, command
    r.say(9)
    _save(stage, r, w, ctl)
    rec = ctl.runs()[0]
    print('run 1:', rec.status, rec.exit_code, len(rec.written), rec.changed.get('added', [])[:3], rec.seconds,
          rec.requirements, flush=True)
    o = r.wrect(pg.out)
    r.zoom(QRectF(o.x() - 10, o.y() - 40, o.width() + 20, o.height() + 50), 0.8)
    r.finish_speech(0.4)

    # 10 - the tiles
    tl = pg.tiles
    vals = {k: (t.value, t.unit, t.note) for k, t in tl.items()}
    print('tiles:', vals, flush=True)
    r.expect(tl['result'].value, 'Passed', 'the run')
    r.expect(tl['exit'].value, '0', 'exit code')
    r.expect((tl['files'].value, len(rec.changed['added']), len(rec.written)), ('25', 25, 26), 'files new')
    r.expect(sorted(set(rec.written) - set(rec.changed['added'])), ['vantage_summary.json'], 'the 26th file')
    r.expect((tl['req'].value, rec.requirements['musts']), ('Pass', '10/10'), 'requirements verdict')
    r.say(10)
    trow = H.union(r.wrect(tl['result']), r.wrect(tl['req']))
    r.zoom(QRectF(trow.x() - 10, trow.y() - 20, trow.width() + 20, trow.height() + 40), 0.8)
    r.callout(tl['result'], dur=1.6, pad=3, dim=0.2)
    r.until_word(10, "the command's exit code")
    r.callout(tl['exit'], dur=1.8, pad=3, dim=0.2)
    r.until_word(10, 'the duration')
    r.callout(tl['time'], dur=1.4, pad=3, dim=0.2)
    r.until_word(10, 'twenty-five files')
    r.figure('Files', 25, 'new', 0, dur=2.6, src=tl['files'], note='+ vantage_summary.json')
    r.until_word(10, 'and the requirements')
    r.callout(tl['req'], dur=max(1.6, r.speech_left() + 0.3), pad=3, dim=0.2)
    r.finish_speech(0.2)

    # 11 - the variables
    r.say(11)
    env_lbl = next(lb for lb in pg.findChildren(QLabel) if lb.text().startswith('It gets '))
    er = r.wrect(env_lbl)
    r.zoom(QRectF(er.x() - 10, er.y() - 30, er.width() + 260, er.height() + 60), 0.9)
    r.callout(env_lbl, dur=max(3.0, r.speech_left() - 0.6), pad=3, dim=0.25)
    r.finish_speech(0.2)
    r.zoom(None, 0.6)

    # 12 - the script reads it; move F 5 cm back and save
    r.say(12)
    r.zoom(QRectF(o.x() - 10, o.y() - 40, o.width() + 20, o.height() + 50), 0.8)
    r.until_word(12, 'Now, move the front')
    r.zoom(None, 0.6)
    f = st.variant.by_tag('F')
    st.set_cam(f.uid, text='Moved F', x=round(f.x - 0.05, 4))
    stage.settle(2)
    r.until_word(12, 'and save again')
    # the save; line 13 starts as soon as line 12 ends, while the run finishes (real time)
    n0 = len(ctl.runs())
    r.key_combo('Ctrl+S')
    r.hold(0.3)
    w.save()
    run_done = (lambda: not ctl.busy and len(ctl.runs()) > n0)
    r.wait_for(lambda: run_done() or r.now() >= r.speech_end + 0.4, timeout=180)
    r.say(13)
    r.wait_for(run_done, timeout=180)
    stage.settle(1)
    rec2 = ctl.runs()[0]
    ch = rec2.changed
    print('run 2:', rec2.status, rec2.exit_code, {k: v for k, v in ch.items()}, flush=True)
    print('run 2 written:', sorted(rec2.written), flush=True)

    # 13 - what changed
    r.expect((len(ch['changed']), ch.get('unchanged'), len(ch['added']), len(ch['removed'])), (10, 14, 0, 0),
             'files changed by moving F')
    r.expect(sorted(ch['changed']), ['extrinsics.json', 'kalibr/camchain.yaml', 'kitti/calib_cam_to_cam.txt',
                                     'kitti/calib_cams_to_vehicle.txt', 'nuscenes/calibrated_sensor.json',
                                     'opencv/F.yaml', 'requirements/requirements.csv', 'requirements/requirements.json',
                                     'static_tf.launch.py', 'vantage_cameras.urdf'], 'the files that changed')
    fl = r.wrect(pg.files)
    r.zoom(QRectF(fl.x() - 10, fl.y() - 40, fl.width() + 20, fl.height() + 50), 0.8)
    r.figure('Files changed', 10, 'of 24', 0, dur=3.0, src=pg.files_title)
    ftab = pg.files

    def rows_of(names):
        idx = [i for i in range(ftab.rowCount()) if ftab.item(i, 0).text() in names]
        def rect():
            rs = [x for x in (H.cell(r, ftab, i)() for i in idx) if x is not None]
            return H.union(*rs) if rs else r.wrect(ftab)
        return rect
    r.until_word(13, "F's open C V file")
    r.callout(rows_of({'opencv/F.yaml'}), dur=1.8, pad=2, dim=0.3)
    r.until_word(13, 'the files that hold the whole rig')
    # the rig files sit in two runs of the sorted list: two outlines, nothing dimmed
    r.callout(rows_of({'extrinsics.json', 'kalibr/camchain.yaml', 'kitti/calib_cam_to_cam.txt',
                       'kitti/calib_cams_to_vehicle.txt', 'nuscenes/calibrated_sensor.json'}), dur=2.0, pad=2, dim=0.0)
    r.callout(rows_of({'static_tf.launch.py', 'vantage_cameras.urdf'}), dur=2.0, pad=2, dim=0.0)
    r.until_word(13, 'and the scorecard')
    r.callout(rows_of({'requirements/requirements.csv', 'requirements/requirements.json'}), dur=1.6, pad=2, dim=0.3)
    r.until_word(13, 'The other fourteen')
    r.callout(pg.files_title, dur=max(1.8, r.speech_left() + 0.3), pad=3, dim=0.3)
    r.finish_speech(0.2)

    # 14 - by content; PDFs rewritten
    rw = ch.get('rewritten', [])
    print('rewritten:', rw, flush=True)
    r.say(14)
    r.hold(max(1.5, r.at_word(14, 'A P D F') - r.now()))
    if rw:
        row = next(i for i in range(pg.files.rowCount()) if pg.files.item(i, 1).text().lower().startswith('rewritten'))
        r.callout(H.cell(r, pg.files, row), dur=max(2.0, r.speech_left()), pad=2, dim=0.25)
    r.finish_speech(0.2)
    r.zoom(None, 0.6)

    # 15 - untick ROS camera_info, run by hand: pruned, the check fails
    r.say(15)
    cb = pg.format_boxes['ros_camera_info']
    pg.formats.ensureWidgetVisible(cb)
    stage.settle(0.5)
    r.zoom(QRectF(fr.x() - 10, fr.y() - 30, fr.width() + 260, fr.height() + 60), 0.8)
    r.click(cb, QPointF(10, cb.height() / 2))
    stage.settle(0.5)
    r.zoom(None, 0.6)
    n0 = len(ctl.runs())
    r.click(pg.b_run)
    r.wait_for(lambda: not ctl.busy and len(ctl.runs()) > n0, timeout=180)
    stage.settle(1)
    rec3 = ctl.runs()[0]
    print('run 3:', rec3.status, rec3.exit_code, rec3.pruned, rec3.changed.get('removed'), flush=True)
    r.expect((rec3.exit_code, len(rec3.pruned), tl['exit'].value, rec3.status), (2, 6, '2', 'failed'),
             'run by hand without camera_info')
    rt = H.union(r.wrect(tl['result']), r.wrect(pg.out))
    r.zoom(QRectF(rt.x() - 10, rt.y() - 20, rt.width() + 20, rt.height() + 40), 0.8)
    r.callout(tl['exit'], dur=max(2.0, r.speech_left() + 0.4), pad=3, dim=0.2)
    r.finish_speech(0.3)

    # 16 - history, Stop, Export run
    r.say(16)
    hr = r.wrect(pg.history)
    r.zoom(QRectF(hr.x() - 10, hr.y() - 40, hr.width() + 20, min(hr.height(), 300) + 50), 0.8)
    r.callout(lambda: H.rows_rect(r, pg.history), dur=3.0, pad=2, dim=0.2)
    r.until_word(16, 'Stop, or the')
    r.zoom(None, 0.6)
    r.callout(pg.b_stop, dur=2.2, pad=3)
    r.until_word(16, 'Export run')
    r.callout(pg.b_export, dur=max(1.6, r.speech_left()), pad=3)
    r.finish_speech(0.2)

    # 17 - batch --watch (the real run, replayed)
    c_w, l_w = watch
    r.say(17)
    span = H.terminal(r, [('cmd', 'vantage --watch tigor.vantage --on-change "make test" --watch-runs 1')] +
                      H.timed_output(l_w, keep=noise, tones=True) + [('wait', 0.4), ('cmd', 'echo $?'),
                                                                     ('out', str(c_w), 'ok' if c_w == 0 else 'warn')],
                      title='~/ci-box')
    r.crossfade(0.5)
    r.hold(max(span + 0.6, r.speech_left() + 0.2))
    r.finish_speech(0.2)

    # 18 - the CI templates (written for real), and what the workflow runs
    (c_ci, l_ci), (c_ls, l_ls), (c_cat, l_cat) = ci, ls, cat
    r.say(18)
    span = H.terminal(r, [('cmd', 'vantage tigor.vantage --ci-templates .')] + H.timed_output(l_ci, keep=noise) +
                      [('wait', 0.3), ('cmd', 'ls ci ci/github ci/gitlab')] + H.timed_output(l_ls) +
                      [('wait', 0.3), ('cmd', 'grep -n -E "name:|on:|push|run: |vantage |path" ci/github/vantage.yml')]
                      + H.timed_output(l_cat), title='~/ci-box')
    r.crossfade(0.4)
    r.hold(max(span + 0.6, r.speech_left() + 0.3))
    r.finish_speech(0.2)

    # 19 - exit codes; the assistant reads, never runs
    r.say(19)
    r.card = None
    r.crossfade(0.6)
    r.statement('Exit 2 · exit 5', 'a critical issue in the plan · a must requirement that fails',
                dur=max(2.0, r.at_word(19, 'The assistant can read') - r.now() - 0.2), eyebrow='In CI', scrim=0.92)
    r.until_word(19, 'The assistant can read')
    r.statement('Read, never run', 'the assistant sees the automation and its runs, never the command',
                dur=max(2.0, r.speech_left() + 0.2), eyebrow='The assistant', scrim=0.92)
    r.finish_speech(0.2)

    # 20, 21 - limits, over the page
    r.say(20)
    r.statement('Limits', 'the command runs with your permissions; approval of its exact text is the gate',
                dur=r.line(20) + 0.3, eyebrow='Automation', scrim=0.9)
    r.finish_speech(0.2)
    r.say(21)
    r.statement('Limits', 'no window images · the first run removes nothing · CI checked locally',
                dur=r.line(21) + 0.3, eyebrow='Automation', scrim=0.9)
    r.cursor_on = False
    r.finish_speech(1.0)
