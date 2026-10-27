"""40 The intro of "What's new in Vantage 0.2.0" (episode v020): a cold open on the beat made of the most striking
moments of this episode's own chapter recordings (the app on the sample, recorded for this video: the excerpts below,
played at their recorded speed), the logo (v0.2.0), the title, then a narrated overview of the eight chapters in their
three parts and whom each part helps, over slowly drifting stills of those chapters.

The music starts with the video at 90 BPM (a beat is 2/3 s): the cuts fall on beats and the logo lands on the downbeat
of bar 5 (10.67 s), as in the first tour's intro. Record it after the chapters it quotes (their final recordings; a
draft run falls back to their drafts). Nothing in the cold open is a claim: it is the app's own rendering of its sample.

The shots: the sample standing on a road (filmed live here); the sample dataset's camera images at night with the
headlamps (43 Weather and night); the harness routed along the body in 3D (45 Cable harness); the wiper's sweep over
the windscreen camera's view (44 Materials, glass and wipers). Behind the overview: the night images (part 1), the
harness in 3D (part 2), the lidar–camera overlay of a solved board (part 3, 47).
"""

META = {
    'id': 'intro-v020',
    'part': 0,
    'title': "What's new in Vantage 0.2.0",
    'promise': '',
    'menu': '',
}

LINES = [
    # 0
    ("Vantage plans the sensor rigs of data-collection vehicles, and proves in simulation what they will record, "
     "before any hardware is fitted.",
     "Vantage plans the sensor rigs of data collection vehicles, and proves in simulation what they will record, "
     "before any hardware is fitted."),
    # 1
    ("Version 0.2.0 adds seven studies for the rig on the road and in the workshop, and makes its calibration more "
     "accurate. This video explains each one, on the app's Tata Tigor EV sample.",
     "Version zero point two adds seven studies for the rig on the road and in the workshop, and makes its "
     "calibration more accurate. This video explains each one, on the app's Tata Tee-gor E V sample."),
    # 2
    ("On the road: the critical targets a rig must resolve, and how far each camera may move before it loses one; "
     "the road itself, and the body's motion on it;",
     "On the road: the critical targets a rig must resolve, and how far each camera may move before it loses one; the "
     "road itself, and the body's motion on it;"),
    # 3
    ("and what rain, fog, snow and night take from each sensor. These three are for whoever decides where the sensors "
     "go.",
     "and what rain, fog, snow, and night take from each sensor. These three are for whoever decides where the "
     "sensors go."),
    # 4
    ("In the workshop: what each sensor looks through, from a painted bumper to the windscreen and its wiper; every "
     "cable, routed along the body; and how firmly the calibration boards pin each sensor.",
     "In the workshop: what each sensor looks through, from a painted bumper to the windscreen and its wiper; every "
     "cable, routed along the body; and how firmly the calibration boards pin each sensor."),
    # 5
    ("These are for the people who build the rig and calibrate it.",
     "These are for the people who build the rig, and calibrate it."),
    # 6
    ("In the pipeline: a calibration that now reaches its best answer and states its precision honestly, and "
     "automation that keeps the calibration files and your own tests in step with every change.",
     "In the pipeline: a calibration that now reaches its best answer, and states its precision honestly; and "
     "automation that keeps the calibration files, and your own tests, in step with every change."),
    # 7
    ("These two are for the data team. Everything here is recorded from the running app: the figures are its own "
     "output on the sample, and typical values are called typical.",
     "These two are for the data team. Everything here is recorded from the running app: the figures are its own "
     "output on the sample, and typical values are called typical."),
]

BEAT = 60.0 / 90.0
LOGO_AT = 16 * BEAT           # 10.67 s: the downbeat of bar 5
LOGO_DUR = 5.6
TITLE_DUR = 3.4
VERSION = '0.2.0'

# the cold open: (segment id, second of its recording, cut at this beat); 'road' is filmed live: the sample standing on
# a road (a 15 % garage ramp ahead, a speed bump behind, a kerb on the left) in a chromeless 3D view, turning slowly.
# The logo follows at beat 16
ROAD = [{'kind': 'garage'}, {'kind': 'bump', 'where': 'behind'}, {'kind': 'kerb', 'side': 'left'}]
COLD = [('road', 0.0, 6), ('new-weather', 194.4, 9), ('new-harness', 202.5, 12), ('new-materials', 288.6, 16)]
THUMB = 0                     # the still for the thumbnail: from this shot, at its middle
# behind each part of the overview: (segment id, second of its recording)
BACKDROPS = {1: ('new-weather', 195.5), 2: ('new-harness', 203.0), 3: ('new-calib', 241.0)}
AGENDA = {1: ('Part 1 · On the road', [('01', 'Critical targets and mounting envelopes'), ('02', 'Road and body motion'),
                                       ('03', 'Weather and night')]),
          2: ('Part 2 · In the workshop', [('04', 'Materials, glass and wipers'), ('05', 'Cable harness'),
                                           ('06', 'Calibration observability and fusion')]),
          3: ('Part 3 · In the pipeline', [('07', 'A more accurate calibration'), ('08', 'Automation and CI')])}


def _video(cid):
    """The recording of a chapter of this episode: the final one, else its draft (a draft run of the intro)."""
    import os
    import common
    final = os.environ.get('QT_SCALE_FACTOR') == '2'          # record.py: a final recording grabs at 2x
    for root in (common.SEG_DIR,) + (() if final else (os.path.join(common.DRAFT_DIR, 'seg'),)):
        p = os.path.join(root, cid, 'video.mp4')
        if os.path.exists(p):
            return p
    raise SystemExit(f'the intro quotes {cid}, which is not recorded yet')


class Clip:
    """Frames of a recording from a given second, decoded as they are shown (1920 x 1080, 30 fps), with a slow push
    in; frame(f) returns the frame for the recorder's frame f (the same one when asked twice)."""

    def __init__(self, cid, t0, fps, push=0.05, dur=4.0):
        import subprocess
        import gfx
        self.W, self.H = gfx.W, gfx.H
        self.proc = subprocess.Popen(['ffmpeg', '-v', 'error', '-ss', f'{t0:.3f}', '-i', _video(cid), '-vf',
                                      f'scale={gfx.W}:{gfx.H}:flags=bicubic,fps={fps}', '-f', 'rawvideo', '-pix_fmt',
                                      'bgra', '-'], stdout=subprocess.PIPE)
        self.f0, self.k, self.img, self.push, self.n = None, -1, None, push, max(1, round(dur * fps))

    def frame(self, f):
        import numpy as np
        import gfx
        from PyQt6.QtCore import QRectF
        from PyQt6.QtGui import QImage
        if self.f0 is None:
            self.f0 = f
        want = f - self.f0
        size = self.W * self.H * 4
        while self.k < want:
            raw = self.proc.stdout.read(size)
            if len(raw) == size:
                self.img = np.frombuffer(raw, np.uint8).reshape(self.H, self.W, 4)
            self.k += 1
        s = 1.0 + self.push * gfx.smoother(want / self.n)
        out = QImage(self.W, self.H, QImage.Format.Format_RGB32)
        p = gfx.painter(out)
        p.setRenderHint(gfx.QPainter.RenderHint.SmoothPixmapTransform)
        src = gfx.from_np(self.img)
        p.drawImage(QRectF(self.W * (1 - s) / 2, self.H * (1 - s) / 2, self.W * s, self.H * s), src)
        p.end()
        return out

    def close(self):
        try:
            self.proc.stdout.close()
            self.proc.kill()
            self.proc.wait()
        except Exception:                        # noqa: BLE001
            pass


def _still(cid, t):
    """One frame of a recording as a QImage (1920 x 1080)."""
    import subprocess
    import numpy as np
    import gfx
    raw = subprocess.run(['ffmpeg', '-v', 'error', '-ss', f'{t:.3f}', '-i', _video(cid), '-frames:v', '1', '-vf',
                          f'scale={gfx.W}:{gfx.H}:flags=bicubic', '-f', 'rawvideo', '-pix_fmt', 'bgra', '-'],
                         capture_output=True, check=True).stdout
    return gfx.from_np(np.frombuffer(raw[:gfx.W * gfx.H * 4], np.uint8).reshape(gfx.H, gfx.W, 4))


def scene(stage, r):
    import gfx
    from PyQt6.QtCore import QRectF
    from PyQt6.QtGui import QImage, QPainter

    fps = stage.fps
    W, H = gfx.W, gfx.H

    # ---- set-up, not filmed: the sample on a road, and a chromeless 3D view of it
    def setup(p):
        p.analysis.terrain = {'enabled': True, 'features': ROAD}
    w = stage.window(setup)
    a = w.ctx.analysis
    import time
    t_ = time.time()
    while (a.result is None or a._busy or a.timer.isActive()) and time.time() - t_ < 180:
        stage.app.processEvents()
        time.sleep(0.05)
    stage.settle(3)
    lo, hi = stage.car_box(w)
    road = stage.full_view(stage.vp.provider)
    road.cam.az, road.cam.el = -62.0, 13.0
    road.frame_box(lo, hi, 0.92)
    road.cam.dist *= 1.05

    def frames_until(t_end):
        return max(0, round(t_end * fps) - r.f)

    # ---- the cold open, music alone (the mixer plays it louder here)
    r.mark('music', db=-2.5, dur=LOGO_AT + LOGO_DUR + TITLE_DUR + 0.4)
    r.cursor_on = False
    t_prev = 0.0
    for i, (cid, t0, beat) in enumerate(COLD):
        t_end = beat * BEAT - (0.25 if i == len(COLD) - 1 else 0.0)
        clip = None
        if cid == 'road':
            r.card = None
            r.win = road
        else:
            clip = Clip(cid, t0, fps, push=0.06, dur=t_end - t_prev)
            r.card = lambda rec, c=clip: c.frame(rec.f)
        if i:
            r.crossfade(0.22)
        n = frames_until(t_end)
        az0, d0 = road.cam.az, road.cam.dist
        for k in range(n):
            if cid == 'road':
                u = (k + 1) / n
                road.cam.az = az0 + 26 * u
                road.cam.dist = d0 * (1 - 0.08 * gfx.smoother(u))
                road.update()
            r.tick()
            if i == THUMB and k == n // 2:
                r.still('thumb')
        if i == len(COLD) - 1:
            r.mark('whoosh')                         # its hit lands on the logo
            r.tick(frames_until(LOGO_AT))
        if clip is not None:
            clip.close()
        t_prev = t_end

    # ---- the logo, then the title
    f0 = r.f
    r.mark('logo')

    def logo(rec):
        img = QImage(W, H, QImage.Format.Format_RGB32)
        img.fill(gfx.col(gfx.BASE))
        tt = (rec.f - f0) / rec.fps
        p = gfx.painter(img)
        gfx.vignette(p, 0.4)
        p.end()
        gfx.logo_reveal(img, tt, LOGO_DUR, 1.0 - gfx.smoother((tt - (LOGO_DUR - 0.7)) / 0.7), version=VERSION)
        return img
    r.crossfade(0.18)
    r.card = logo
    r.hold(LOGO_DUR)
    f1 = r.f

    def title(rec):
        img = QImage(W, H, QImage.Format.Format_RGB32)
        img.fill(gfx.col(gfx.BASE))
        tt = (rec.f - f1) / rec.fps
        out = 1 - gfx.in_cubic(gfx.clamp((tt - (TITLE_DUR - 0.6)) / 0.6))
        p = gfx.painter(img)
        gfx.vignette(p, 0.4)
        gfx.glow(p, W / 2, H / 2 - 30, 800, gfx.ACCENT, 0.08 * gfx.out_cubic(gfx.window(tt, 0, 1.0)) * out)
        k = gfx.out_cubic(gfx.window(tt, 0.1, 0.5))
        f_e = gfx.font(22, 'semibold', mono=True, spacing=5.0)
        s = f'VANTAGE {VERSION}'
        gfx.draw_text(p, W / 2 - gfx.text_w(f_e, s) / 2, H / 2 - 120 + (1 - k) * 12, s, f_e, gfx.ACCENT, k * out)
        k = gfx.out_cubic(gfx.window(tt, 0.25, 0.6))
        f_t = gfx.font(124, 'semibold')
        s = "What's new"
        gfx.draw_text(p, W / 2 - gfx.text_w(f_t, s) / 2, H / 2 + 20 + (1 - k) * 22, s, f_t, gfx.INK, k * out)
        k = gfx.out_quint(gfx.window(tt, 0.5, 0.7))
        p.fillRect(QRectF(W / 2 - 80 * k, H / 2 + 70, 160 * k, 3), gfx.col(gfx.ACCENT, out))
        k = gfx.out_cubic(gfx.window(tt, 0.65, 0.6))
        f_s = gfx.font(34)
        s = 'Seven new studies, and a more accurate calibration'
        gfx.draw_text(p, W / 2 - gfx.text_w(f_s, s) / 2, H / 2 + 150 + (1 - k) * 12, s, f_s, gfx.INK2, k * out)
        p.end()
        return img
    r.card = title
    r.hold(TITLE_DUR)

    # ---- the overview: each part over a drifting still of one of its chapters
    # softened: a little blurred and darkened, so the agenda reads over them
    stills = {pt: gfx.from_np(gfx.Backdrop(gfx.to_np(_still(cid, t))).at(0.55, 0.45))
              for pt, (cid, t) in BACKDROPS.items()}
    cur = {'pt': None, 'f': r.f}

    def backdrop(rec):
        im = stills[cur['pt']]
        tt = (rec.f - cur['f']) / rec.fps
        s = 1.02 + 0.05 * gfx.smoother(tt / 22.0)
        img = QImage(W, H, QImage.Format.Format_RGB32)
        p = gfx.painter(img)
        p.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform)
        p.drawImage(QRectF(W * (1 - s) / 2 + 40, H * (1 - s) / 2, W * s, H * s), im)
        p.end()
        return img

    def part(pt, lines_):
        cur['pt'], cur['f'] = pt, r.f
        r.card = backdrop
        r.crossfade(0.8)
        ttl, items = AGENDA[pt]
        ag = r.agenda(items, ttl, dur=None)
        return ag

    # 0, 1 - what Vantage is; what 0.2.0 brings (over the first backdrop, no agenda yet)
    cur['pt'], cur['f'] = 1, r.f
    r.card = backdrop
    r.crossfade(0.9)
    r.hold(0.4)
    r.say(0)
    r.statement('Sensor rigs for data-collection vehicles', 'planned and proven in simulation',
                dur=r.line(0) + 0.2)
    r.finish_speech(0.2)
    r.say(1)
    r.statement(f'What\'s new in {VERSION}', 'seven studies · a more accurate calibration', dur=r.line(1) + 0.2)
    r.finish_speech(0.4)

    # 2, 3 - part 1
    ag = part(1, (2, 3))
    r.say(2)
    for i, frag in enumerate(('the critical targets', 'the road itself')):
        r.until_word(2, frag)
        ag.reveal(r, i)
    r.finish_speech(0.2)
    r.say(3)
    ag.reveal(r, 2)
    r.finish_speech(0.6)
    ag.close(r, 0.0)

    # 4, 5 - part 2
    ag = part(2, (4, 5))
    r.say(4)
    for i, frag in enumerate(('what each sensor', 'every cable', 'and how firmly')):
        r.until_word(4, frag)
        ag.reveal(r, i)
    r.finish_speech(0.2)
    r.say(5)
    r.finish_speech(0.6)
    ag.close(r, 0.0)

    # 6, 7 - part 3, and the honesty note
    ag = part(3, (6, 7))
    r.say(6)
    for i, frag in enumerate(('a calibration that', 'and automation')):
        r.until_word(6, frag)
        ag.reveal(r, i)
    r.finish_speech(0.4)
    r.say(7)
    r.until_word(7, 'Everything here')
    ag.close(r, 0.0)
    r.note(f'Recorded from Vantage v{VERSION}  ·  Tata Tigor EV sample project', dur=r.speech_left() + 0.8)
    r.finish_speech(1.2)
