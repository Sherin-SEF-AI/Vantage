"""Website clip 'automation': Export ▸ Automation on the sample (six cameras), with a harmless example command.

The sample is saved in the throw-away home folder (~/rig-design/tigor.vantage) beside a Makefile whose `test` target
runs a short Python check of the regenerated files (every camera has its OpenCV and camera_info file; what changed;
the requirements check's verdict), and the plan's requirements the six-camera rig meets. The clip turns automation
on (on save), types `make test` as the command, Save… shows the approval dialog (the command in full) and Allow
approves it; Ctrl+S saves the project, and the run that follows regenerates the 26 calibration files, runs the
requirements check and then the command, whose output streams in. Camera F moved 5 cm back and saved again: the run
lists the 10 files that changed.
"""
from __future__ import annotations

import os

from PyQt6.QtCore import QPointF, QRectF
from PyQt6.QtTest import QTest
from PyQt6.QtWidgets import QLabel

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
print(f"cameras with OpenCV and camera_info files: {' '.join(tags)}" + (f"; missing {missing}" if missing else ''))
req = summary['requirements'] or {}
print(f"requirements: {req.get('verdict_text', 'not run')} (must {req.get('musts', '-')}), "
      f"exit {os.environ.get('VANTAGE_CHECK_EXIT', '')}")
ok = not missing and os.environ.get('VANTAGE_CHECK_EXIT') in ('0', '')
print('OK' if ok else 'FAILED')
sys.exit(0 if ok else 1)
'''

# the plan's requirements (Requirements ▸ Add from the plan) that the six-camera rig fails are the requirements
# clip's subject; this one keeps the rest, so the check passes and the run shows its verdict
LEAVE_OUT = ('No critical issue', 'No blind ground', 'resolved to 80 m in the front sector', 'h of driving per disk')


def clip(stage, r):
    from vantage.core import requirements as RQ
    from vantage.ui.workspaces import automation_ws as AW
    folder = os.path.join(stage.home, 'rig-design')
    os.makedirs(folder, exist_ok=True)
    with open(os.path.join(folder, 'Makefile'), 'w') as fh:
        fh.write(MAKEFILE)
    with open(os.path.join(folder, 'check_calibration.py'), 'w') as fh:
        fh.write(CHECK)
    path = os.path.join(folder, 'tigor.vantage')

    def setup(p):
        items = [i for i in RQ.from_plan(p, p.variant) if not any(s in i['text'] for s in LEAVE_OUT)]
        p.analysis.requirements = RQ.RequirementsSettings.from_dict({'items': items}).to_dict()
        p.save(path)
        p.path = path
    w = stage.window(setup)
    r.win = w

    # the approval dialog is modal (exec): shown here, read, and allowed with the pointer instead
    def exec_(d):
        stage.tilde(*d.findChildren(QLabel))
        d.adjustSize()
        g = w.geometry()
        d.move(g.x() + (g.width() - d.width()) // 2, g.y() + (g.height() - d.height()) // 2)
        d.show()
        r.hold(0.4)
        r.zoom(r.wrect(d), 0.6, margin=30)
        r.move_to(d.text, None, 0.6)
        r.hold(2.2)
        r.click(d.allow_b, None, 0.6, real=False)
        d.hide()
        r.zoom(None, 0.5)
        return 1
    real_exec = AW.ApproveDialog.exec
    AW.ApproveDialog.exec = exec_
    try:
        _drive(stage, r, w, path)
    finally:
        AW.ApproveDialog.exec = real_exec


def _save(stage, r, w):
    """Ctrl+S, as the user presses it: the window saves, and automation runs on the save."""
    ctl = w.ctx.automation
    n0 = len(ctl.runs())
    r.key_badge('Ctrl+S')
    r.hold(0.3)
    w.save()
    r.wait_for(lambda: not ctl.busy and len(ctl.runs()) > n0, timeout=120)
    stage.settle(1)
    r.hold(0.4)


def _drive(stage, r, w, path):
    st = w.store
    ctl = w.ctx.automation
    r.chapter('Automation and CI', 'The calibration files stay in step with the design')
    r.menu('Export', 'Automation')
    pg = w.page('automation')
    stage.tilde(*pg.findChildren(QLabel))
    stage.settle(1)
    r.hold(0.4)
    # on (on save), and the command: make test
    r.click(pg.enabled, QPointF(12, pg.enabled.height() / 2))
    r.hold(0.4)
    r.click(pg.cmd, None, 0.6)
    for ch in 'make test':
        QTest.keyClicks(pg.cmd, ch)
        r.tick(3)
    r.hold(0.3)
    r.click(pg.b_save_cmd)                      # → the approval dialog (exec_ above) → Allow
    r.hold(0.5)
    if not ctl.commands.is_approved(path):
        raise RuntimeError('the command was not approved')
    # saved: the calibration files regenerated, the requirements checked, then the command
    _save(stage, r, w)
    o = r.wrect(pg.out)
    tl = r.wrect(pg.tiles['result']).united(r.wrect(pg.tiles['req']))
    r.zoom(tl.united(o), 0.8, margin=8)
    r.hold(2.6)
    r.zoom(None, 0.6)
    # F moved 5 cm back and saved again: the run lists what changed
    f = st.variant.by_tag('F')
    st.set_cam(f.uid, text='Moved F', x=round(f.x - 0.05, 4))
    stage.settle(2)
    _save(stage, r, w)
    fr = r.wrect(pg.files)
    r.zoom(QRectF(fr.x(), fr.y() - 30, fr.width(), fr.height() + 30).united(o), 0.8, margin=8)
    r.hold(3.0)
    r.zoom(None, 0.6)
    r.cursor_on = False
    r.hold(0.4)
