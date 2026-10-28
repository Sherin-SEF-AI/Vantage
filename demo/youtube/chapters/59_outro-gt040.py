"""59 The outro and end screen of "Ground truth from CARLA · Vantage 0.4.0" (episode gt040): the 0.4.0 release page, the
website and its documentation, the source and the licence. The right half stays clear for YouTube's end-screen
elements (two videos and the subscribe button).
"""

META = {
    'id': 'outro-gt040',
    'part': 3,
    'title': 'Get Vantage 0.4.0',
    'promise': '',
    'menu': '',
}

LINES = [
    # 0
    ("Vantage 0.4.0 is free and open source, for Windows, macOS and Linux, including the Jetson.",
     "Vantage zero point four is free and open source, for Windows, mac O S, and Linux, including the Jetson."),
    # 1
    ("The website documents every layer, with its limits. Thanks for watching.",
     "The website documents every layer, with its limits. Thanks for watching."),
]

END_DUR = 13.5
VERSION = '0.4.0'
ROWS = [('DOWNLOAD 0.4.0', 'github.com/Sherin-SEF-AI/Vantage/releases/tag/v0.4.0',
         'Windows · macOS · Linux x86-64 and arm64 (Jetson)'),
        ('WEBSITE AND DOCS', 'sherin-sef-ai.github.io/Vantage', 'ground truth, faults, scenarios: docs and limits'),
        ('SOURCE', 'github.com/Sherin-SEF-AI/Vantage', 'GPL-3.0-or-later · free and open source')]
FOOT = f'v{VERSION}  ·  REAL CARLA 0.9.16 DRIVES AND THE APP  ·  VOICE: KOKORO-82M, LOCAL'


def scene(stage, r):
    import gfx
    from PyQt6.QtGui import QImage

    f0 = r.f

    def card(rec):
        img = QImage(gfx.W, gfx.H, QImage.Format.Format_RGB32)
        img.fill(gfx.col(gfx.BASE))
        p = gfx.painter(img)
        gfx.vignette(p, 0.35)
        p.end()
        gfx.end_card(img, (rec.f - f0) / rec.fps, END_DUR, version=VERSION, rows=ROWS, foot=FOOT)
        return img
    r.card = card
    r.cursor_on = False
    r.hold(1.0)
    r.say(0)
    r.finish_speech(0.1)
    r.say(1)
    r.finish_speech(0.2)
    r.mark('music', db=-7.0, dur=max(1.0, END_DUR - r.now()))
    r.until(END_DUR)
