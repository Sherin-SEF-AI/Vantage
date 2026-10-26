"""38 Downloads, self-test and running on the Jetson: the release downloads (README's Install table: the file names
and what each needs), the Linux install steps and the checksum file, a real `vantage --self-test` run while recording
(from this checkout, through a `vantage` launcher in the throw-away home; its output replayed at the pace it arrived,
x4 and marked; Qt's and OpenCV's stderr chatter about the offscreen platform left out), then the Live rig page with a
V4L2 device and a Jetson GStreamer pipeline typed in as camera sources (not started: no camera here).

Figures: the self-test's own lines, read from the run (r.expect: it ends with 'self-test passed'); the downloads and
requirements are the README's. Nothing about the release page is fetched (no network).
"""

META = {
    'id': 'install-anywhere',
    'part': 5,
    'title': 'Downloads, self-test and running on the Jetson',
    'promise': 'Windows, macOS, Linux and the Jetson Orin, checked by a self-test.',
    'menu': 'github.com/Sherin-SEF-AI/Vantage/releases  ·  vantage --self-test',
}

LINES = [
    ("Vantage is free and open source, under the GPL. Each release has a download for every system, with its own "
     "Python and libraries inside, so there is nothing else to install.",
     "Vantage is free and open source, under the G P L. Each release has a download for every system, with its own "
     "python and libraries inside, so there is nothing else to install."),
    ("On Windows, an installer that needs no administrator rights; on macOS, the app in a zip; on Linux, a tarball "
     "for x86 machines, and one for arm64.",
     "On Windows, an installer that needs no administrator rights. On mac O S, the app in a zip. On Linux, a tarball "
     "for x eighty-six machines, and one for arm sixty-four."),
    ("On Linux: unpack, run install.sh for a menu entry, the .vantage file type and the vantage command. Every release "
     "lists each file's SHA-256 to check the download against.",
     "On Linux: unpack, and run install dot S H, for a menu entry, the dot vantage file type, and the vantage command. "
     "Every release lists each file's S H A two fifty-six, to check the download against."),
    ("The self-test checks an installation without opening a window. Here it runs on this machine, from source.",
     "The self-test checks an installation without opening a window. Here it runs on this machine, from source."),
    ("It analyses the sample, checks the camera readers and runs a GStreamer test pipeline, loads Qt's plugins, writes "
     "and reads ROS 2 bags, draws with OpenGL, builds every page, and writes every export format.",
     "It analyses the sample, checks the camera readers, and runs a G streamer test pipeline, loads Q T's plugins, "
     "writes and reads ross two bags, draws with open G L, builds every page, and writes every export format."),
    ("A warning names what is missing on the machine, and how to fix it. The run ends with: self-test passed.",
     "A warning names what is missing on the machine, and how to fix it. The run ends with: self-test passed."),
    ("The arm64 build runs on a Jetson Orin with JetPack 6, so the live rig health, the drive-log check and the "
     "as-built check run on the data-collection computer itself, right after a drive.",
     "The arm sixty-four build runs on a Jetson Orin with Jet Pack six, so the live rig health, the drive log check, "
     "and the as-built check run on the data collection computer itself, right after a drive."),
    ("There, cameras open as V4L2 devices, the way e-con's GMSL drivers expose them, or as GStreamer pipelines, which "
     "Vantage runs through the system's gst-launch with NVIDIA's plug-ins.",
     "There, cameras open as V four L two devices, the way e-con's G M S L drivers expose them, or as G streamer "
     "pipelines, which Vantage runs through the system's G S T launch, with NVIDIA's plug-ins."),
    ("With Python 3.10 or later, it also installs from source with pip. Either way, the project file is the same.",
     "With python three point ten or later, it also installs from source with pip. Either way, the project file is the "
     "same."),
]

SPEED = 4.0


def scene(stage, r):
    import os
    import site
    import sys
    import time
    from PyQt6.QtCore import QRectF
    import helpers_tourc as H
    import studio

    # ---- the real self-test (not filmed): from this checkout, through a `vantage` launcher in the throw-away home
    home = stage.home
    bin_ = os.path.join(home, 'bin')
    os.makedirs(bin_, exist_ok=True)
    with open(os.path.join(bin_, 'vantage'), 'w') as fh:
        fh.write(f'#!/bin/sh\nexec {sys.executable} {os.path.join(studio.ROOT, "run_vantage.py")} "$@"\n')
    os.chmod(os.path.join(bin_, 'vantage'), 0o755)
    env = {k: v for k, v in os.environ.items() if k not in ('QT_SCALE_FACTOR',)}
    env.update(PATH=bin_ + os.pathsep + env.get('PATH', ''), HOME=home, PYTHONUSERBASE=site.USER_BASE,
               QT_QPA_PLATFORM='offscreen')
    code, lines, secs = H.run_timed('vantage --self-test', home, env, home, timeout=600)
    for _, ln in lines:
        print('   ', ln, flush=True)
    print(f'self-test: exit {code} in {secs:.1f} s', flush=True)
    r.expect(code, 0, 'self-test exit code')
    keep = (lambda ln: not ln.startswith(('This plugin does not', '[ WARN')))
    shown = [ln for _, ln in lines if keep(ln)]
    r.expect(shown[-1].startswith('self-test passed'), True, 'the self-test passed')
    warn = [ln for ln in shown if ' warning ' in ln[:30]]
    print('warnings:', warn, flush=True)

    # 0-1 - the downloads
    r.lower_third()
    r.say(0)
    rows = [('Windows', 'Vantage-0.1.1-windows-x64-setup.exe\n…-windows-x64.zip', 'Windows 10 or 11, x64; installs per user'),
            ('macOS', 'Vantage-0.1.1-macos-arm64.zip', 'macOS 14 or later, Apple silicon'),
            ('Linux', 'Vantage-0.1.1-linux-x86_64.tar.gz', 'x86_64, glibc 2.35 or later (Ubuntu 22.04, Debian 12, '
                                                          'Fedora 36 and newer)'),
            ('Linux arm64', 'Vantage-0.1.1-linux-arm64.tar.gz', 'NVIDIA Jetson Orin with JetPack 6, Ubuntu 22.04 '
                                                               'arm64 and newer')]
    H.table_card(r, 'Downloads', 'github.com/Sherin-SEF-AI/Vantage/releases  ·  free and open source, GPL-3.0-or-later',
                 ('System', 'File', 'Needs'), rows, foot='Each carries its own Python and libraries: nothing else to '
                                                        'install.', col_w=(0.17, 0.43, 0.40))
    r.mark('music', db=-4, dur=2)
    r.hold(max(2.0, r.speech_left() + 0.2))
    r.say(1)
    r.finish_speech(0.3)

    # 2 - Linux install, checksums
    r.say(2)
    steps = [('Unpack', 'tar xf Vantage-*-linux-x86_64.tar.gz', 'or …-linux-arm64.tar.gz on a Jetson'),
             ('Install', 'cd Vantage-*-linux-*/ && ./install.sh', 'per user: a menu entry, the .vantage file type, '
                                                                   '~/.local/bin/vantage'),
             ('Check', 'sha256sum -c SHA256SUMS.txt --ignore-missing', 'every file of the release, against its '
                                                                      'SHA-256'),
             ('Test', 'vantage --self-test', 'checks the installation without a window')]
    H.table_card(r, 'Linux', 'from the README', ('Step', 'Command', 'What it does'), steps, col_w=(0.13, 0.50, 0.37))
    r.crossfade(0.5)
    r.hold(max(2.0, r.speech_left() + 0.3))
    r.finish_speech(0.2)

    # 3-5 - the self-test, replayed at the pace it ran (x4, marked)
    r.say(3)
    script = [('cmd', 'vantage --self-test')] + H.timed_output(lines, speed=SPEED, keep=keep, width=112, indent=24,
                                                               tones=True)
    span = H.terminal(r, script, title='~')
    r.crossfade(0.5)
    t0 = r.now()
    r.hold(0.9)
    r.timelapse(SPEED)
    r.finish_speech(0.2)
    r.say(4)
    r.until(t0 + span + 0.3)
    r.timelapse(None)
    r.finish_speech(0.2)
    # the warning line on the card: gfx.terminal draws the last 21 lines from y = 222, 32 px apart
    items = [x for x in script if x[0] != 'wait']
    vis = items[-21:]
    k = next((i for i, x in enumerate(vis) if x[0] == 'out' and x[2] == 'warn'), None)
    n = 1
    while k is not None and k + n < len(vis) and vis[k + n][1].startswith(' ' * 10):
        n += 1
    r.say(5)
    if k is not None:
        r.callout((226, 222 + 32 * k - 25, 1478, 32 * n + 4), dur=max(2.6, r.speech_left() - 2.0), pad=2, dim=0.3)
    r.until_word(5, 'The run ends')
    r.callout((226, 222 + 32 * (len(vis) - 1) - 25, 1478, 36), dur=max(1.8, r.speech_left() + 0.3), pad=2, dim=0.3)
    r.finish_speech(0.4)

    # 6-7 - on the Jetson: Live rig sources
    w = stage.window()
    r.card = None
    r.win = w
    vp = stage.vp
    r.say(6)
    r.crossfade(0.6)
    stage.orbit(r, vp, 18, 2.6, d_el=3)
    r.cursor_on = True
    r.menu('Calibrate', 'Live rig')
    pg = w.page('rig')
    stage.settle(2)
    r.crossfade(0.4)
    r.finish_speech(0.2)
    r.say(7)
    st = w.store
    F, R = st.variant.by_tag('F'), st.variant.by_tag('R')
    g = H.union(r.wrect(pg.rows[F.uid][0]), r.wrect(pg.rows[st.variant.cameras[-1].uid][1]))
    r.zoom(QRectF(g.x() - 60, g.y() - 60, g.width() + 120, g.height() + 140), 1.0)
    for cam, spec in ((F, '/dev/video0'), (R, 'nvarguscamerasrc sensor-id=1 ! nvvidconv ! video/x-raw,format=BGRx ! '
                                             'appsink')):
        combo, edit, _ = pg.rows[cam.uid]
        r.choose(combo, combo.findData('stream'))
        r.enter(edit, spec, cps=40)
        if st.project.streams.get(cam.tag) != spec:
            edit.editingFinished.emit()
        r.hold(0.3)
    r.expect(st.project.streams.get('R', '').startswith('nvarguscamerasrc'), True, "R's pipeline")
    stage.settle(1)
    th = H.union(r.wrect(pg.tiles[F.uid]), r.wrect(pg.tiles[R.uid]))      # the tiles name each source in full
    r.cursor_on = False
    r.zoom(QRectF(th.x() - 10, th.y() - 30, th.width() + 20, 200), 1.0)
    r.hold(max(1.6, r.speech_left() - 0.4))
    r.finish_speech(0.2)
    r.zoom(None, 0.8)

    # 8 - from source
    r.say(8)
    r.cursor_on = False
    H.table_card(r, 'From source', 'Python 3.10 or later', ('Step', 'Command', ''),
                 [('Install', 'python3 -m pip install .', '[step] STEP import, [bags] compressed bags, [keyring]'),
                  ('Run', 'vantage', 'the window; vantage project.vantage --analyse … for batch mode')],
                 col_w=(0.13, 0.42, 0.45))
    r.crossfade(0.6)
    r.hold(max(2.0, r.speech_left() + 0.8))
    r.finish_speech(0.6)
