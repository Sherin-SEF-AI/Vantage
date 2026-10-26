"""00 The intro: a cold open on the beat (the shader-rendered data rig turning with its lidar points and camera light,
the 3D surround view, the density map), the logo, then a narrated overview of what the tour covers.

The music starts with the video at 90 BPM (a beat is 2/3 s): the cuts fall on beats 6 and 11 and the logo lands on
the downbeat of bar 5 (10.67 s). Nothing in the cold open is a claim: it is the app's own rendering of its sample.
"""

META = {
    'id': 'intro',
    'part': 0,
    'title': 'Vantage',
    'promise': '',
    'menu': '',
}

LINES = [
    ("Vantage designs the sensor rigs of data-collection vehicles: the cars that record the camera data self-driving "
     "models are trained on.",
     "Vantage designs the sensor rigs of data collection vehicles: the cars that record the camera data that "
     "self-driving models are trained on."),
    ("You place the cameras, lidars and radars on a 3D model of the vehicle, and prove in simulation what the rig will "
     "record, before any hardware is fitted.",
     "You place the cameras, lidars, and radars on a 3D model of the vehicle, and prove in simulation what the rig "
     "will record, before any hardware is fitted."),
    ("Then you calibrate the real rig, check it as built against the plan, and hand its calibration to the data "
     "pipeline.",
     "Then you calibrate the real rig, check it as built against the plan, and hand its calibration to the data "
     "pipeline."),
    ("This tour shows all of it, in five parts: designing the rig, proving what it sees, the studies, calibrating and "
     "verifying, and the hand-off to the data.",
     "This tour shows all of it, in five parts: designing the rig, proving what it sees, the studies, calibrating "
     "and verifying, and the hand-off to the data."),
    ("Everything here is recorded from the running app, on its Tata Tigor EV sample.",
     "Everything here is recorded from the running app, on its Tata Tee-gor E V sample."),
]

BEAT = 60.0 / 90.0
LOGO_AT = 16 * BEAT           # 10.67 s: the downbeat of bar 5
LOGO_DUR = 5.6


def scene(stage, r):
    import time
    import gfx
    from PyQt6.QtGui import QImage
    from chapters import PARTS

    fps = stage.fps
    # ---- set-up, not filmed: the data rig with its lidar points and camera light; the bowl; a density map
    w = stage._rig_window()
    w.ctx.set_overlay('camlight', True)
    w.ctx.set_overlay('lidarpts', True)
    w.ctx.set_overlay('frustums', False)
    stage.settle(10)
    vp = stage.vp
    lo, hi = stage.car_box(w)
    hero = stage.full_view(vp.provider)
    hero.cam.az, hero.cam.el = -48.0, 17.0
    hero.frame_box(lo, hi, 0.92)
    hero.cam.dist *= 0.74
    w.set_page('camviews')
    pg = w._page_widgets['camviews']
    ctl = pg.bowl_scene.bowl
    t = time.time()
    while ctl.draw is None and time.time() - t < 180:
        stage.app.processEvents()
        time.sleep(0.05)
    stage.settle(3)
    bowl = stage.full_view(pg.bowl_scene, cam_from=pg.bowl.cam)
    bowl.cam.dist *= 0.80
    bowl.cam.az, bowl.cam.el = 150.0, 33.0
    w.set_page('viewports')
    stage.settle(2)
    from vantage.ui.planview import PlanView
    plan = PlanView(w.ctx, 'density', interactive=False)
    plan.legend = False
    plan.resize(gfx.W, gfx.H)
    plan.show()
    stage.extra.append(plan)
    for _ in range(10):
        stage.app.processEvents()
    plan.fit()
    s0 = plan.scale
    stage.app.processEvents()

    def frames_until(t_end):
        return max(0, round(t_end * fps) - r.f)

    # ---- cold open, music alone (the mixer plays it louder here)
    r.mark('music', db=-2.5, dur=LOGO_AT + LOGO_DUR + 0.4)
    r.win = hero
    r.cursor_on = False
    n = frames_until(6 * BEAT)
    az0, d0 = hero.cam.az, hero.cam.dist
    for k in range(n):
        u = (k + 1) / n
        hero.cam.az = az0 + 34 * u
        hero.cam.dist = d0 * (1 - 0.07 * gfx.smoother(u))
        hero.update()
        r.tick()
        if k == int(n * 0.55):
            r.still('thumb')
    r.crossfade(0.22)
    r.win = bowl
    n = frames_until(11 * BEAT)
    az0 = bowl.cam.az
    for k in range(n):
        bowl.cam.az = az0 - 40 * (k + 1) / n
        bowl.update()
        r.tick()
    r.crossfade(0.22)
    r.win = plan
    n = frames_until(LOGO_AT - 0.25)
    for k in range(n):
        plan.scale = s0 * (1.0 + 0.16 * gfx.smoother((k + 1) / n))
        plan._user_view = True
        plan.update()
        r.tick()
    r.mark('whoosh')                                         # its hit lands on the logo
    r.tick(frames_until(LOGO_AT))

    # ---- the logo
    f0 = r.f
    r.mark('logo')

    def logo(rec):
        img = QImage(gfx.W, gfx.H, QImage.Format.Format_RGB32)
        img.fill(gfx.col(gfx.BASE))
        tt = (rec.f - f0) / rec.fps
        p = gfx.painter(img)
        gfx.vignette(p, 0.4)
        p.end()
        gfx.logo_reveal(img, tt, LOGO_DUR, 1.0 - gfx.smoother((tt - (LOGO_DUR - 0.7)) / 0.7))
        return img
    r.crossfade(0.18)
    r.card = logo
    r.hold(LOGO_DUR)

    # ---- the overview over the turning rig, the density map and the bowl
    r.card = None
    plan.scale = s0 * 1.04
    bowl.cam.az, bowl.cam.el = 200.0, 30.0
    hero.cam.az, hero.cam.el = 128.0, 19.0
    hero.frame_box(lo, hi, 0.92)
    hero.cam.dist *= 0.8
    hero.cam.target = hero.cam.target + __import__('numpy').array([0.0, -0.9, 0.0])
    r.win = hero
    r.crossfade(0.9)

    views = {'hero': hero, 'bowl': bowl, 'plan': plan}
    cur = ['hero']
    plan_s = [plan.scale]

    def show(name):
        if name != cur[0]:
            cur[0] = name
            r.win = views[name]
            r.crossfade(0.7)

    def turn(sec):
        n = max(1, round(sec * fps))
        for _ in range(n):
            v = views[cur[0]]
            if cur[0] == 'hero':
                v.cam.az += 6.0 / fps
            elif cur[0] == 'bowl':
                v.cam.az -= 5.0 / fps
            else:
                plan_s[0] *= 1 - 0.02 / fps
                v.scale = plan_s[0]
                v._user_view = True
            v.update()
            r.tick()
    turn(0.5)
    r.say(0)
    r.statement('Sensor rigs for data-collection vehicles', 'cameras \u00b7 lidars \u00b7 radars \u00b7 GNSS \u00b7 IMU',
                dur=r.line(0) + 0.2)
    turn(r.speech_left() + 0.4)
    show('plan')
    r.say(1)
    r.statement('Placed and proven in simulation', 'before any hardware is fitted', dur=r.line(1) + 0.2)
    turn(r.speech_left() + 0.4)
    show('bowl')
    r.say(2)
    r.statement('Calibrated, checked as built, handed off', 'to OpenCV, ROS 2, Kalibr, nuScenes, KITTI and more',
                dur=r.line(2) + 0.2)
    turn(r.speech_left() + 0.4)
    show('hero')
    r.say(3)
    ag = r.agenda([(i, PARTS[i][0]) for i in range(1, 6)], 'In this tour', dur=None)
    for i, frag in enumerate(('designing the rig', 'proving what it sees', 'the studies', 'calibrating',
                              'the hand-off')):
        target = r.at_word(3, frag)
        turn(max(0.0, target - r.now()))
        ag.reveal(r, i)
    turn(r.speech_left() + 0.8)
    r.say(4)
    ag.close(r, 0.0)
    r.note('Recorded from Vantage v0.1.1  \u00b7  Tata Tigor EV sample project', dur=r.line(4) + 0.6)
    turn(r.speech_left() + 1.2)
