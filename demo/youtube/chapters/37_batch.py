"""37 Batch mode and CI: real batch runs, made while recording (not filmed, about 20 s) in the throw-away home with the
same program (a `vantage` launcher in ~/bin that runs this checkout's run_vantage.py, as pip's entry point does): the
sample saved as tigor.vantage, and its data rig with the requirements its own plan implies (core.requirements
.from_plan, as Add from the plan does) saved as data-rig.vantage. The terminal cards replay each command's real output
at the pace it arrived (paths in the throw-away home shown from ~); the review pack's first two pages are rendered from
the PDF the run wrote.

Figures checked against the runs (r.expect): --analyse exits 2 (the sample plan has a critical issue: 0.010 m² of
blind ground, mostly at the right rear wheel arch); --check on the data rig: 'Fail · must 10/14 · should 5/8', exit 5.
"""

META = {
    'id': 'batch',
    'part': 5,
    'title': 'Batch mode and CI',
    'promise': 'Everything without a window: for CI pipelines and the end of the line.',
    'menu': 'vantage project.vantage --analyse · --check · --export …',
}

LINES = [
    ("Everything in the window also runs without one: batch mode, for CI pipelines and the end of the production line. "
     "The same program and the same project file, with flags.",
     "Everything in the window also runs without one: batch mode, for C I pipelines, and the end of the production "
     "line. The same program, and the same project file, with flags."),
    ("Analyse writes the coverage figures, seams, issues and link budget as JSON, and its exit code tells a script "
     "what it needs to know: 2, because the plan has a critical issue.",
     "Analyse writes the coverage figures, seams, issues, and link budget as Jason, and its exit code tells a script "
     "what it needs to know: two, because the plan has a critical issue."),
    ("The report names it: 0.01 square metres of blind ground, mostly at the right rear wheel arch.",
     "The report names it: zero point zero one square metres of blind ground, mostly at the right rear wheel arch."),
    ("Check runs the requirements scorecard, with every study a requirement needs. On the data rig, with the "
     "requirements its own plan implies, 10 of 14 musts pass: exit code 5, and a CI job stops.",
     "Check runs the requirements scorecard, with every study a requirement needs. On the data rig, with the "
     "requirements its own plan implies, ten of fourteen musts pass: exit code five, and a C I job stops."),
    ("It writes the scorecard as JSON and CSV, and a design-review pack as a PDF: the verdict, then the evidence for "
     "each requirement.",
     "It writes the scorecard as Jason and C S V, and a design review pack as a P D F: the verdict, then the evidence "
     "for each requirement."),
    ("Export writes formats by their ids: here the rig's calibration in the four dataset layouts, for the data "
     "pipeline.",
     "Export writes formats by their ids: here, the rig's calibration in the four dataset layouts, for the data "
     "pipeline."),
    ("And every study has a flag of its own, from the recording budget to the drive-log and as-built checks, each "
     "with exit codes a script can act on.",
     "And every study has a flag of its own, from the recording budget to the drive log and as-built checks, each "
     "with exit codes a script can act on."),
]

ROOT_RUN = 'run_vantage.py'


def scene(stage, r):
    import os
    import site
    import subprocess
    import sys
    from PyQt6.QtGui import QImage
    from vantage.core import samples, requirements as RQ
    import helpers_tourc as H

    # ---- the real runs (not filmed)
    home = stage.home
    import studio
    root = studio.ROOT
    bin_ = os.path.join(home, 'bin')
    work = os.path.join(home, 'projects')
    os.makedirs(bin_, exist_ok=True)
    os.makedirs(work, exist_ok=True)
    with open(os.path.join(bin_, 'vantage'), 'w') as fh:
        fh.write(f'#!/bin/sh\nexec {sys.executable} {os.path.join(root, ROOT_RUN)} "$@"\n')
    os.chmod(os.path.join(bin_, 'vantage'), 0o755)
    env = {k: v for k, v in os.environ.items() if k not in ('QT_SCALE_FACTOR',)}
    env.update(PATH=bin_ + os.pathsep + env.get('PATH', ''), HOME=home, PYTHONUSERBASE=site.USER_BASE,
               QT_QPA_PLATFORM='offscreen')
    p = samples.tigor_sample_project()
    p.photos.source = ''
    p.save(os.path.join(work, 'tigor.vantage'), b'')
    q = samples.tigor_sample_project()
    q.photos.source = ''
    q.active_uid = next(v.uid for v in q.variants if v.sensors)
    q.analysis.requirements = {'items': RQ.from_plan(q, q.variant)}
    q.save(os.path.join(work, 'data-rig.vantage'), b'')

    def run(cmd):
        code, lines, secs = H.run_timed(cmd, work, env, home)
        print(f'$ {cmd}  -> {code} in {secs:.1f} s', flush=True)
        for _, ln in lines:
            print('   ', ln, flush=True)
        return code, lines, secs
    noise = (lambda ln: not ln.startswith(('This plugin does not', '[ WARN')))
    c_ver, l_ver, _ = run('vantage --version')
    c_hlp, l_hlp, _ = run('vantage --help | head -18')
    c_an, l_an, _ = run('vantage tigor.vantage --analyse report.json')
    r.expect(c_an, 2, 'exit code of --analyse')
    c_jq, l_jq, _ = run("jq -r '.issues[0].message' report.json")
    r.expect(l_jq[0][1], '0.010 m² of ground in the ROI is blind, mostly right rear wheel arch.', 'the critical issue')
    c_ck, l_ck, _ = run('vantage data-rig.vantage --check review/')
    r.expect(c_ck, 5, 'exit code of --check')
    r.expect(l_ck[-1][1].strip(), 'requirements: Fail · must 10/14 · should 5/8', 'the scorecard')
    c_ls, l_ls, _ = run('ls review/')
    c_ex, l_ex, _ = run('vantage tigor.vantage --export calib/ --formats '
                        'nuscenes_calib,kitti_calib,ros_camera_info,kalibr_camchain')
    c_rb, l_rb, _ = run('vantage data-rig.vantage --recording-budget budget.json')
    pages = os.path.join(home, 'pack')
    subprocess.run(['pdftoppm', '-r', '110', '-f', '1', '-l', '2', '-png', os.path.join(work, 'review', 'review_pack.pdf'),
                    pages], check=True)
    pngs = sorted(f for f in os.listdir(home) if f.startswith('pack') and f.endswith('.png'))
    page_imgs = [QImage(os.path.join(home, f)) for f in pngs]

    def term(cmd, code, lines, show_code=True, tones=True):
        out = [('cmd', cmd)] + H.timed_output(lines, keep=noise, tones=tones)
        if show_code:
            out += [('wait', 0.4), ('cmd', 'echo $?'), ('out', str(code), 'warn' if code else 'ok')]
        return out

    # 0 - the same program, with flags
    w = stage.window()
    r.win = w
    vp = stage.vp
    r.lower_third()
    r.say(0)
    stage.orbit(r, vp, 16, 2.6, d_el=3)
    span = H.terminal(r, [('cmd', 'vantage --version')] + H.timed_output(l_ver) + [('wait', 0.6)] +
                      [('cmd', 'vantage --help | head -18')] + H.timed_output(l_hlp), title='~/projects')
    r.crossfade(0.6)
    r.hold(max(span + 0.5, r.speech_left() + 0.3))
    r.finish_speech(0.2)

    # 1 - analyse; exit code 2
    r.say(1)
    span = H.terminal(r, term('vantage tigor.vantage --analyse report.json', c_an, l_an), title='~/projects')
    r.crossfade(0.4)
    r.hold(max(span + 0.6, r.speech_left() + 0.2))
    r.finish_speech(0.2)

    # 2 - the issue in the report
    r.say(2)
    span = H.terminal(r, term('vantage tigor.vantage --analyse report.json', c_an, l_an) +
                      [('wait', 0.3)] + term("jq -r '.issues[0].message' report.json", c_jq, l_jq, show_code=False),
                      title='~/projects')
    r.hold(max(span + 0.4, r.speech_left() + 0.4))
    r.finish_speech(0.2)

    # 3 - the requirements scorecard; exit code 5
    r.say(3)
    span = H.terminal(r, term('vantage data-rig.vantage --check review/', c_ck, l_ck), title='~/projects')
    r.crossfade(0.4)
    r.hold(max(span + 0.8, r.speech_left() + 0.2))
    r.finish_speech(0.2)

    # 4 - the review pack
    r.say(4)
    span = H.terminal(r, [('cmd', 'ls review/')] + H.timed_output(l_ls), title='~/projects')
    r.crossfade(0.4)
    r.hold(max(1.6, span + 0.6))
    H.pages_card(r, page_imgs, 'review/review_pack.pdf · the first two pages, as the batch run wrote them')
    r.crossfade(0.5)
    r.hold(max(2.4, r.speech_left() + 1.0))
    r.finish_speech(0.2)

    # 5 - export by ids
    r.say(5)
    span = H.terminal(r, term('vantage tigor.vantage --export calib/ --formats '
                              'nuscenes_calib,kitti_calib,ros_camera_info,kalibr_camchain', c_ex, l_ex, show_code=False),
                      title='~/projects')
    r.crossfade(0.4)
    r.hold(max(span + 0.8, r.speech_left() + 0.2))
    r.finish_speech(0.2)

    # 6 - a flag per study
    r.say(6)
    span = H.terminal(r, term('vantage data-rig.vantage --recording-budget budget.json', c_rb, l_rb), title='~/projects')
    r.crossfade(0.4)
    r.hold(max(span + 1.0, r.speech_left() + 0.8))
    r.finish_speech(0.6)
    r.card = None
    r.crossfade(0.6)
    r.hold(0.8)
