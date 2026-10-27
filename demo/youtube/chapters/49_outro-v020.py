"""49 The outro and end screen of "What's new in Vantage 0.2.0" (episode v020): the 0.2.0 release page, the website and
its documentation, the source and the licence. The right half stays clear for YouTube's end-screen elements (two
videos and the subscribe button), which can sit there for the last 20 s.
"""

META = {
    'id': 'outro-v020',
    'part': 4,
    'title': 'Get Vantage 0.2.0',
    'promise': '',
    'menu': '',
}

LINES = [
    ("Vantage 0.2.0 is free and open source, under the GPL. Installers for Windows, macOS and Linux, including arm64 "
     "for the Jetson, are on its release page.",
     "Vantage zero point two is free and open source, under the G P L. Installers for Windows, mac O S, and Linux, "
     "including arm sixty-four for the Jetson, are on its release page."),
    ("The website documents every new study with its limits, and the full tour shows everything that came before. "
     "Thanks for watching.",
     "The website documents every new study, with its limits, and the full tour shows everything that came before. "
     "Thanks for watching."),
]

END_DUR = 20.0
VERSION = '0.2.0'
ROWS = [('DOWNLOAD 0.2.0', 'github.com/Sherin-SEF-AI/Vantage/releases/tag/v0.2.0',
         'Windows · macOS · Linux x86-64 and arm64 (Jetson)'),
        ('WEBSITE AND DOCS', 'sherin-sef-ai.github.io/Vantage', 'every study, its documentation and limits, batch mode'),
        ('SOURCE', 'github.com/Sherin-SEF-AI/Vantage', 'GPL-3.0-or-later · free and open source')]


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
        gfx.end_card(img, (rec.f - f0) / rec.fps, END_DUR, version=VERSION, rows=ROWS)
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
