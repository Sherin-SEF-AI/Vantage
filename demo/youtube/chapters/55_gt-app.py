"""55 In the app (episode gt040, chapter 5): Simulate ▸ CARLA on the sample's data rig; Open run… on a real 4 s CARLA
drive recorded before the video (headless, with --carla-run, ground truth on), as the website's carla clip does
(demo/clip_carla.py: the drive's folder linked into the throw-away working folder as the project's CARLA runs, the
CARLA folder linked as ~/carla, no server started, CARLA's default ports never probed); its report; its Ground truth
tab (the frame with the most labelled objects, zoomed to them: boxes, semantic labels, instances, range, the lidar
from above; a vehicle picked in the object table and outlined); then a terminal: the drive's folders on disk and a
real batch run of --ground-truth-summary on it.

Figures narrated, read from the app while recording and checked with r.expect: the report's line "Ground truth: 242
camera frames, 3835 objects, 40 lidar sweeps labelled". The folders and the batch output on the terminal are the real
ones (run in the throw-away home, shown from ~).
"""
import os

META = {
    'id': 'gt-app',
    'part': 2,
    'title': 'In the app',
    'promise': 'Open a drive, look through its labels, and hand them on.',
    'menu': 'Simulate ▸ CARLA ▸ Open run…  ·  Ground truth',
}

LINES = [
    # 0
    ("In the app, Simulate ▸ CARLA ▸ Open run… shows a drive recorded before, with its report and its checks.",
     "In the app, simulate, CARLA, open run, shows a drive recorded before, with its report and its checks."),
    # 1
    ("This one: 242 camera frames with ground truth, 3,835 objects, 40 labelled lidar sweeps.",
     "This one: two hundred and forty-two camera frames with ground truth, three thousand, eight hundred and "
     "thirty-five objects, forty labelled lidar sweeps."),
    # 2
    ("The Ground truth tab steps through the boxes, semantic labels, instances, range and lidar; a row of the object "
     "table is outlined in the image.",
     "The ground truth tab steps through the boxes, semantic labels, instances, range, and lidar; a row of the "
     "object table is outlined in the image."),
    # 3
    ("On disk, it is a nuScenes dataset with lidarseg and panoptic labels, plus COCO files; --ground-truth-summary "
     "reads it back.",
     "On disk, it is a new scenes dataset, with lidar seg and panoptic labels, plus coco files; and ground truth "
     "summary reads it back."),
]


REPORT = 'Ground truth: 242 camera frames, 3835 objects, 40 lidar sweeps labelled'


def scene(stage, r):
    import clip_carla as CC
    from vantage.sim import server as SV
    if CC.CARLA_PORT in CC.FORBIDDEN or CC.CARLA_TM_PORT in CC.FORBIDDEN:
        raise RuntimeError('refusing CARLA\'s default ports')
    runs = CC.need('VANTAGE_CLIP_RUNS', 'the folder holding the drive made by `vantage PROJECT --carla-run`')
    if not CC.finished_run(runs):
        raise RuntimeError('no CARLA drive with ground truth in VANTAGE_CLIP_RUNS')
    undo = CC.guard_ports(SV)
    links = CC.link_carla(stage.home)
    if not links:
        undo()
        raise RuntimeError('no CARLA installation found (set VANTAGE_CARLA_ROOT)')
    held = {'links': links}
    try:
        _scene(stage, r, held)
    finally:
        for x in held['links']:
            try:
                os.unlink(x)
            except OSError:
                pass
        undo()


def _scene(stage, r, held):
    import site
    import sys

    from PyQt6.QtCore import QPointF, QRectF
    from PyQt6.QtGui import QTextCursor

    import clip_carla as CC
    import helpers_tourc as H
    import studio
    from vantage.core import groundtruth as GT

    w = stage._rig_window(CC.setup)
    r.win = w
    orig_msg = w.ctx.msg                       # status-bar messages show paths in the throw-away home from ~
    import clips
    w.ctx.msg = lambda text, *a, **k: orig_msg(clips._from_home(text), *a, **k)

    # ---- 0: Simulate ▸ CARLA; Open run…
    r.lower_third()
    r.cursor_on = True
    r.say(0)
    r.menu('Simulate', 'CARLA')
    pg = w.page('sim')
    held['links'].append(CC.link_runs(w.store.project, pg.working()))
    pg._show_where()
    # the page checks the CARLA folder and its Python (a few seconds in a subprocess): set-up, not the subject, so
    # it is waited for without recording (the video's time stands still meanwhile)
    import time
    t_c = time.time()
    while not (pg.rig is not None and pg._check is not None) and time.time() - t_c < 90:
        stage.app.processEvents()
        time.sleep(0.02)
    print(f'the CARLA check took {time.time() - t_c:.1f} s (not filmed)', flush=True)
    if pg.rig is None or not (pg._check or {}).get('ok'):
        raise RuntimeError(f'the CARLA plan or its Python is not ready: {pg._check}')
    st = pg.settings
    if (st.port, st.tm_port) != (CC.CARLA_PORT, CC.CARLA_TM_PORT) or pg.server is not None:
        raise RuntimeError(f'the page is set to port {st.port} / {st.tm_port}')
    drive = CC.finished_run(pg.runs_folder())       # through the link: shown from ~
    r.crossfade(0.3)
    r.hold(0.3)
    r.until_word(0, 'open run')
    r.click(pg.b_open, dur=0.6, real=False)
    if not pg.open_run(drive):
        raise RuntimeError(f'Open run… refused the drive: {pg.g_status.text()}')
    print('opened:', pg.g_status.text(), flush=True)
    r.hold(0.4)
    r.cursor_on = False
    r.until_word(0, 'its report')
    a, o = r.wrect(pg.b_drivelog), r.wrect(pg.out)
    r.zoom(QRectF(o.x(), a.y() - 8, o.width() * 0.66, o.bottom() - a.y() + 16), 0.8)
    r.callout(pg.b_drivelog.parentWidget(), '', dur=max(1.4, r.speech_left()), pad=4, dim=0.25)
    r.finish_speech(0.1)

    # ---- 1: the report's ground truth line
    text = pg.out.toPlainText()
    line = next(ln for ln in text.splitlines() if ln.startswith('Ground truth:'))
    print('report:', line, flush=True)
    r.expect(line.split(' (')[0], REPORT, 'the report\'s ground truth line')
    doc = GT.read(drive)
    summ = GT.summary(doc)
    r.expect((sum(summ['cameras'].values()), summ['objects'], sum(summ['lidars'].values())), (242, 3835, 40),
             'the index agrees')
    blk = pg.out.document().find('Ground truth:')

    def line_rect():
        cur = QTextCursor(blk)
        cur.movePosition(QTextCursor.MoveOperation.StartOfBlock)
        rr = pg.out.cursorRect(cur)
        vp = pg.out.viewport()
        tl = r.wpos(vp, QPointF(rr.x(), rr.y()))
        fm = pg.out.fontMetrics()
        return QRectF(tl.x() - 2, tl.y() - 2, fm.horizontalAdvance(line) + 8, rr.height() + 4)
    pg.out.setTextCursor(QTextCursor(blk))
    pg.out.ensureCursorVisible()
    stage.app.processEvents()
    r.say(1)
    lr = line_rect()
    r.zoom(QRectF(lr.x() - 20, lr.y() - 160, max(lr.width() + 60, 900), 330), 0.8)
    r.figure('Objects labelled', summ['objects'], '', 0, dur=max(3.0, r.speech_left() + 0.2), src=line_rect,
             note='242 camera frames · 40 lidar sweeps', pos='tr')
    r.finish_speech(0.1)
    r.zoom(None, 0.6, wait=False)

    # ---- 2: the Ground truth tab: the layers; a row picked
    gt = pg.gt
    tag, k = CC.best_frame(doc)                # the frame with the most labelled objects, set before the tab opens
    frame = GT.frames(doc, tag)[k]
    gt.set_camera(tag)
    gt.set_boxes('tight')
    gt.slider.setValue(k)
    gt.show_frame()
    box = CC.objects_box(frame)
    r.say(2)
    r.cursor_on = True
    CC.click_tab(r, pg.out_tabs, 'Ground truth')
    gt.show_frame()
    gt.canvas.zoom_to(*box)
    stage.app.processEvents()
    view = r.wrect(gt)
    r.zoom(QRectF(view.x(), view.y(), view.width(), view.height()), 0.5, margin=4)
    print(f'frame {tag} {k}: {gt.canvas.caption}', flush=True)
    t_last = r.now()
    for word, key in (('semantic', 'semantic'), ('instances', 'instance'), ('range', 'range'), ('and lidar', 'lidar')):
        r.until(max(r.at_word(2, word) - 0.5, t_last + 0.6))    # with the word; each layer on screen 1.2 s or more
        r.click(gt.layer.buttons[key], dur=0.3)
        t_last = r.now()
        if key == 'lidar':
            gt.canvas.reset_zoom()
    r.until(max(r.at_word(2, 'a row of the object table') - 0.5, t_last + 0.8))
    r.click(gt.layer.buttons['image'], dur=0.3)
    gt.canvas.zoom_to(*box)
    car = max((o for o in frame['objects'] if o['class'] in ('car', 'truck', 'bus')), key=lambda o: o['pixels'])
    row = next(i for i in range(gt.table.rowCount()) if gt.table.item(i, 1).text() == str(car['id']))
    gt.table.scrollToItem(gt.table.item(row, 0))
    cell = gt.table.visualItemRect(gt.table.item(row, 0))
    r.click(gt.table.viewport(), QPointF(cell.center()), 0.4)
    if gt.selected != car['id']:
        raise RuntimeError(f'the table row did not select {car["class"]} {car["id"]}')
    print('selected', car['class'], car['id'], flush=True)
    r.cursor_on = False
    def in_image():                            # the picked object's box in the image, as the canvas shows it now
        ir = gt.canvas.image_rect()
        k_ = ir.width() / max(gt.canvas.img.width(), 1)
        x0, y0, x1, y1 = car['bbox']
        tl = r.wpos(gt.canvas, QPointF(ir.x() + x0 * k_, ir.y() + y0 * k_))
        return QRectF(tl.x(), tl.y(), (x1 - x0) * k_, (y1 - y0) * k_)
    co = r.callout(in_image, '', dur=2.0, pad=6, dim=0.25)
    r.hold(1.4)
    r.close_layer(co)                          # gone before the terminal: its outline would stay on the card

    # ---- 3: on disk, and --ground-truth-summary (real commands, run now in the throw-away home)
    home = stage.home
    bin_ = os.path.join(home, 'bin')
    os.makedirs(bin_, exist_ok=True)
    with open(os.path.join(bin_, 'vantage'), 'w') as fh_:
        fh_.write(f'#!/bin/sh\nexec {sys.executable} {os.path.join(studio.ROOT, "run_vantage.py")} "$@"\n')
    os.chmod(os.path.join(bin_, 'vantage'), 0o755)
    work = os.path.join(home, 'drives')
    os.makedirs(work, exist_ok=True)
    os.symlink(os.path.realpath(drive), os.path.join(work, 'drive'))
    held['links'].append(os.path.join(work, 'drive'))
    env = {k_: v_ for k_, v_ in os.environ.items() if k_ not in ('QT_SCALE_FACTOR',)}
    env.update(PATH=bin_ + os.pathsep + env.get('PATH', ''), HOME=home, PYTHONUSERBASE=site.USER_BASE,
               QT_QPA_PLATFORM='offscreen')
    cmds = ['ls drive/nuscenes drive/ground_truth/coco',
            'vantage --ground-truth-summary drive/ gt.json',
            "jq -c '.objects_by_class' gt.json"]
    script = []
    for cmd in cmds:
        code, lines_, secs = H.run_timed(cmd, work, env, home)
        print(f'$ {cmd}  -> exit {code} in {secs:.1f} s', flush=True)
        for _, ln in lines_:
            print('   ', ln, flush=True)
        r.expect(code, 0, f'exit code of {cmd}')
        noise = (lambda ln: ln.strip() and not ln.startswith(('This plugin does not', '[ WARN', 'qt.')))
        script += [('cmd', cmd)] + H.timed_output(lines_, keep=noise) + [('wait', 0.3)]
    out_all = '\n'.join(t for it in script if it[0] != 'cmd' and it[0] != 'wait' for t in [it[1]])
    r.expect('ground truth: 242 camera frames, 3835 objects' in out_all, True, 'the batch summary line')
    for leak in (home, '/home/', '/tmp/'):
        r.expect(leak in out_all, False, 'no machine path on the terminal')
    r.say(3)
    span = H.terminal(r, script, title='~/drives')
    print(f'terminal span {span:.1f} s, line {r.line(3):.1f} s', flush=True)
    r.crossfade(0.5)
    r.hold(max(span + 1.0, r.speech_left() + 0.3))
    r.finish_speech(1.0)
