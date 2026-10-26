"""99 The outro and end screen: where to download it, the website and documentation, the source and the licence.
The right half stays clear for YouTube's end-screen elements (two videos and the subscribe button), which can sit
there for the last 20 s.
"""

META = {
    'id': 'outro',
    'part': 6,
    'title': 'Get Vantage',
    'promise': '',
    'menu': '',
}

LINES = [
    ("Vantage is free and open source, under the GPL. Installers for Windows, macOS and Linux, including arm64 for "
     "the Jetson, are on the GitHub releases page.",
     "Vantage is free and open source, under the G P L. Installers for Windows, mac O S, and Linux, including arm "
     "sixty-four for the Jetson, are on the GitHub releases page."),
    ("The website has a short video for every feature and the full documentation. Thanks for watching.",
     "The website has a short video for every feature, and the full documentation. Thanks for watching."),
]

END_DUR = 20.0


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
        gfx.end_card(img, (rec.f - f0) / rec.fps, END_DUR)
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
