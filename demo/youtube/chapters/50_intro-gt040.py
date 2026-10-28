"""50 The intro of "Ground truth from CARLA · Vantage 0.4.0" (episode gt040): a cold open on the beat made of real CARLA
drives with their ground truth (the six fisheyes, their boxes drawing in one camera after another; a fisheye wiping
from its image to its semantic labels, another to its range; the roof lidar from above in its classes), the logo
(v0.4.0), the title, then a narrated overview of the five chapters over a drifting, softened frame of the four layers.

The music starts with the video at 90 BPM (a beat is 2/3 s): the cuts fall on beats and the logo lands on the downbeat
of bar 5 (10.67 s), as in the other episodes' intros. The footage is the drive gt_drive_walkers (CARLA 0.9.16,
Town10HD, 30 cars, 30 walkers, seed 1) in $VANTAGE_GT_DRIVES, painted by gtfootage.py at real speed; nothing in the
cold open is a claim beyond what it shows. No figure is narrated here.
"""

META = {
    'id': 'intro-gt040',
    'part': 0,
    'title': 'Ground truth from CARLA',
    'promise': '',
    'menu': '',
}

LINES = [
    # 0
    ("Vantage plans sensor rigs and drives them in CARLA. Version 0.4.0 labels every camera frame and every lidar "
     "point the rig records there.",
     "Vantage plans sensor rigs, and drives them in CARLA. Version zero point four labels every camera frame, and "
     "every lidar point, the rig records there."),
]


BEAT = 60.0 / 90.0
LOGO_AT = 16 * BEAT           # 10.67 s: the downbeat of bar 5
LOGO_DUR = 5.6
TITLE_DUR = 3.4
VERSION = '0.4.0'
DRIVE = 'gt_drive_walkers'
# the cold open: (kind, camera, drive time at the cut, the beat it ends on). 'six': the six fisheyes, their boxes
# drawing in camera by camera on the beats; 'wipe': one fisheye wiping from its image to a label layer; 'bev': the
# roof lidar from above, pushed in slowly. The logo follows at beat 16
COLD = [('six', None, 1.0, 6), ('wipe', ('F', 'image', 'semantic'), 5.0, 9),
        ('wipe', ('MR', 'image', 'range'), 4.0, 12), ('bev', None, 6.0, 16)]
REVEAL = ('F', 'MR', 'QR', 'R', 'QL', 'ML')       # the order the six cameras' boxes draw in, a beat apart
THUMB = ('F', 5.6, 0.42)      # the thumbnail's still: F at this drive time, wiped this far to its semantic labels
BACKDROP = ('MR', 2.6)        # behind the overview: the four layers of MR at this drive time, softened


def scene(stage, r):
    import os

    import gfx
    import gtfootage as G
    from PyQt6.QtCore import QRectF
    from PyQt6.QtGui import QImage, QPainter

    fps = stage.fps
    W, H = gfx.W, gfx.H
    v = G.start(stage, r)
    d = G.Drive(DRIVE)
    scene_ = d.run['scene']
    r.expect((scene_['carla_version'], scene_['town'].startswith('Town10HD')), ('0.9.16', True), 'the drive')

    def frames_until(t_end):
        return max(0, round(t_end * fps) - r.f)

    # ---- the cold open, music alone (the mixer plays it louder here)
    r.mark('music', db=-2.5, dur=LOGO_AT + LOGO_DUR + TITLE_DUR + 0.4)
    t_prev = 0.0
    for i, (kind, what, t0, beat) in enumerate(COLD):
        t_end = beat * BEAT - (0.25 if i == len(COLD) - 1 else 0.0)
        span = t_end - t_prev
        if kind == 'six':
            lay = G.SixGrid(d, title=f'Real CARLA {scene_["carla_version"]} drive · ground truth',
                            reveal={tag: t0 + (k + 0.5) * BEAT for k, tag in enumerate(REVEAL)})
        elif kind == 'wipe':
            tag, a, b = what
            lay = G.Single(d, tag, title=f'{tag} · {G.CAM_NAMES[tag]} camera · {a} to {b}',
                           wipe=(lambda t, t0=t0, a=a, b=b, span=span:
                                 (a, b, gfx.smoother((t - t0 - 0.15) / max(span - 0.5, 0.1)))))
        else:
            lay = G.BevFull(d)
        v.use(lay)
        v.clock.play(t0)
        if i:
            r.crossfade(0.22)
        if kind == 'bev':
            rr = lay.rect('lidar')
            r.zoom(QRectF(rr.center().x() - 330, rr.center().y() - 190, 660, 380), dur=span, wait=False)
        n = frames_until(t_end)
        r.tick(n)
        if i == len(COLD) - 1:
            r.mark('whoosh')                         # its hit lands on the logo
            r.tick(frames_until(LOGO_AT))
        t_prev = t_end
    v.clock.pause()
    tag, t_th, u = THUMB                   # the thumbnail's still (assemble.py reads stills/thumb.png): no header
    G.render(v, G.Single(d, tag, boxes=True, labels=False, head=False, wipe=lambda t: ('image', 'semantic', u)),
             t_th).save(
        os.path.join(r.out, 'stills', 'thumb.png'))

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
    r.zoom(None, 0)
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
        f_t = gfx.font(116, 'semibold')
        s = 'Ground truth from CARLA'
        gfx.draw_text(p, W / 2 - gfx.text_w(f_t, s) / 2, H / 2 + 20 + (1 - k) * 22, s, f_t, gfx.INK, k * out)
        k = gfx.out_quint(gfx.window(tt, 0.5, 0.7))
        p.fillRect(QRectF(W / 2 - 80 * k, H / 2 + 70, 160 * k, 3), gfx.col(gfx.ACCENT, out))
        k = gfx.out_cubic(gfx.window(tt, 0.65, 0.6))
        f_s = gfx.font(34)
        s = 'Every camera frame and lidar point, labelled'
        gfx.draw_text(p, W / 2 - gfx.text_w(f_s, s) / 2, H / 2 + 150 + (1 - k) * 12, s, f_s, gfx.INK2, k * out)
        p.end()
        return img
    r.card = title
    r.hold(TITLE_DUR)

    # ---- the overview over a drifting, softened frame of the four layers
    tag, t_b = BACKDROP
    still = G.render(v, G.Quad(d, tag), t_b)
    soft = gfx.from_np(gfx.Backdrop(gfx.to_np(still)).at(0.55, 0.45))
    d.release()
    fb = r.f

    def backdrop(rec):
        tt = (rec.f - fb) / rec.fps
        s = 1.02 + 0.05 * gfx.smoother(tt / 16.0)
        img = QImage(W, H, QImage.Format.Format_RGB32)
        p = gfx.painter(img)
        p.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform)
        p.drawImage(QRectF(W * (1 - s) / 2 + 40, H * (1 - s) / 2, W * s, H * s), soft)
        p.end()
        return img
    r.card = backdrop
    r.crossfade(0.9)
    r.hold(0.4)
    r.say(0)
    r.statement('Labels for every frame', 'every camera frame · every lidar point · from CARLA', dur=r.line(0) + 0.2)
    r.until_word(0, 'Version zero point four')
    r.note(f'Real CARLA {scene_["carla_version"]} drives, recorded by Vantage {VERSION} · shown at real speed',
           dur=r.speech_left() + 1.0)
    r.finish_speech(1.2)
