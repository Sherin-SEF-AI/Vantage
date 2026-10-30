"""60 The intro of "An AI assistant for sensor-rig design · Vantage" (episode as070): a cold open on the beat made of
this episode's own chapter recordings (a tool row ticking as the model works, the proposal card, the live strip after
an edit, the whole-rig search, the Copilot's plan card: each shot found by the 'cold' mark its chapter sets while
recording, played at its recorded speed), the logo (no version on screen), the title, then a narrated overview over a
drifting, softened frame of the proposal card.

The music (the episode's own bed, music.bed_as) starts with the video at 90 BPM (a beat is 2/3 s): the cuts fall on
beats and the logo lands on the downbeat of bar 5 (10.67 s), as in the other episodes. Record it after the chapters it
quotes (their final recordings; a draft run falls back to their drafts). No figure is narrated here; nothing in the
cold open is a claim beyond what it shows. The thumbnail's shot (stills/thumb.png) is the proposal card's frame.
"""

META = {
    'id': 'intro-as070',
    'part': 0,
    'title': 'An AI assistant for sensor-rig design',
    'promise': '',
    'menu': '',
}

LINES = [
    # 0
    ("Vantage designs the sensor rigs of data-collection cars. Its assistant answers from the app's own analysis, "
     "and proposes changes you accept or reject.",
     "Vantage designs the sensor rigs of data collection cars. Its assistant answers from the app's own analysis, "
     "and proposes changes you accept, or reject."),
    # 1
    ("Every reply in this video is the real model's, GPT-OSS 120B on Groq, unedited. Every figure comes from "
     "Vantage's tools.",
     "Every reply in this video is the real model's, G P T O S S one twenty B on Grok, unedited. Every figure comes "
     "from Vantage's tools."),
]

BEAT = 60.0 / 90.0
LOGO_AT = 16 * BEAT           # 10.67 s: the downbeat of bar 5
LOGO_DUR = 5.4
TITLE_DUR = 3.6
# the cold open: (chapter, its 'cold' mark, seconds after the mark, cut on this beat); the logo follows at beat 16
COLD = [('as-ask', 'tool', -0.4, 3), ('as-whatif', 'proposal', 0.2, 6), ('as-live', 'strip', 0.0, 9),
        ('as-rig', 'search', 0.6, 12), ('as-copilot', 'plan', 0.3, 16)]
THUMB = ('as-whatif', 'proposal', -0.3)      # the thumbnail's shot: the proposal card
BACKDROP = ('as-whatif', 'proposal', -0.3)
BACKDROP2 = ('as-copilot', 'plan', 1.0)


def _seg(cid):
    import os
    import common
    final = os.environ.get('QT_SCALE_FACTOR') == '2'
    for root in (common.SEG_DIR,) + (() if final else (os.path.join(common.DRAFT_DIR, 'seg'),)):
        if os.path.exists(os.path.join(root, cid, 'video.mp4')):
            return os.path.join(root, cid)
    raise SystemExit(f'the intro quotes {cid}, which is not recorded yet')


def _mark(cid, name):
    import json
    import os
    tl = json.load(open(os.path.join(_seg(cid), 'timeline.json'), encoding='utf-8'))
    for e in tl.get('events', []):
        if e.get('kind') == 'mark' and e.get('name') == 'cold' and e.get('shot') == name:
            return e['start']
    raise SystemExit(f'{cid} has no cold mark {name!r}: record it again')


def scene(stage, r):
    import importlib.util
    import os

    import gfx
    import gfx_as
    import helpers_as as HA
    from PyQt6.QtCore import QRectF
    from PyQt6.QtGui import QImage, QPainter

    spec = importlib.util.spec_from_file_location('intro_v020', os.path.join(os.path.dirname(__file__),
                                                                             '40_intro-v020.py'))
    V = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(V)                  # its Clip and _still, pointed at this episode's recordings
    V._video = lambda cid: os.path.join(_seg(cid), 'video.mp4')

    fps = stage.fps
    W, H = gfx.W, gfx.H

    def frames_until(t_end):
        return max(0, round(t_end * fps) - r.f)

    # ---- the cold open, music alone (the mixer plays it louder here)
    r.mark('music', db=-2.0, dur=LOGO_AT + LOGO_DUR + TITLE_DUR + 0.4)
    r.cursor_on = False
    t_prev = 0.0
    for i, (cid, shot, dt, beat) in enumerate(COLD):
        t_end = beat * BEAT - (0.25 if i == len(COLD) - 1 else 0.0)
        clip = V.Clip(cid, max(0.0, _mark(cid, shot) + dt), fps, push=0.07, dur=t_end - t_prev)
        r.card = lambda rec, c=clip: c.frame(rec.f)
        if i:
            r.crossfade(0.2)
            r.mark('sfx', sound='click')
        r.tick(frames_until(t_end))
        if i == len(COLD) - 1:
            r.mark('whoosh')                         # its hit lands on the logo
            r.tick(frames_until(LOGO_AT))
        clip.close()
        t_prev = t_end
    cid, shot, dt = THUMB
    V._still(cid, _mark(cid, shot) + dt).save(os.path.join(r.out, 'stills', 'thumb.png'))

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
        gfx_as.logo_reveal(img, tt, LOGO_DUR, 1.0 - gfx.smoother((tt - (LOGO_DUR - 0.7)) / 0.7),
                           line='Free and open source  ·  Windows · macOS · Linux')
        return img
    r.crossfade(0.18)
    r.card = logo
    r.zoom(None, 0)
    r.hold(LOGO_DUR)
    f1 = r.f

    def title(rec):
        img = QImage(W, H, QImage.Format.Format_RGB32)
        img.fill(gfx.col(gfx.BASE))
        gfx_as.title_card(img, (rec.f - f1) / rec.fps, TITLE_DUR, 'VANTAGE', 'An AI assistant',
                          'for sensor-rig design', chips=('ASK', 'SIMULATE', 'PROPOSE', 'PLAN', 'MEASURE'))
        return img
    r.card = title
    r.hold(TITLE_DUR)

    # ---- the overview over a drifting, softened frame of the proposal card
    softs = [gfx.from_np(gfx.Backdrop(gfx.to_np(V._still(c, _mark(c, sh) + d))).at(0.55, 0.45))
             for c, sh, d in (BACKDROP, BACKDROP2)]
    cur = {'i': 0, 'f': r.f}

    def backdrop(rec):
        tt = (rec.f - cur['f']) / rec.fps
        s = 1.02 + 0.06 * gfx.smoother(tt / 12.0)
        img = QImage(W, H, QImage.Format.Format_RGB32)
        p = gfx.painter(img)
        p.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform)
        dx = 40 - 50 * gfx.smoother(tt / 12.0)
        p.drawImage(QRectF(W * (1 - s) / 2 + dx, H * (1 - s) / 2, W * s, H * s), softs[cur['i']])
        p.end()
        return img
    r.card = backdrop
    r.crossfade(0.8)
    r.hold(0.3)
    r.say(0)
    HA.kinetic(r, 'It proposes.\nYou decide.', accent=('decide.',), dur=r.line(0) + 0.1, px=84,
              eyebrow='answers from the app’s own analysis')
    r.finish_speech(0.1)
    cur['i'], cur['f'] = 1, r.f
    r.crossfade(0.6)
    r.say(1)
    HA.kinetic(r, 'Real replies.\nUnedited.', accent=('real', 'unedited.'), dur=r.line(1) + 0.3, px=84,
              eyebrow='GPT-OSS 120B on Groq')
    r.finish_speech(1.0)
