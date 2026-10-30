"""69 The outro and end screen of "An AI assistant for sensor-rig design · Vantage" (episode as070): the website and its
documentation of the assistant, the releases page, the source and the licence. No version number: the features shown
may ship in a later release. The right half stays clear for YouTube's end-screen elements.
"""

META = {
    'id': 'outro-as070',
    'part': 4,
    'title': 'Get Vantage',
    'promise': '',
    'menu': '',
}

LINES = [
    # 0
    ("Vantage is free and open source, for Windows, macOS and Linux, including the Jetson.",
     "Vantage is free and open source, for Windows, mac O S, and Linux, including the Jetson."),
    # 1
    ("The website documents the assistant: what it sends, what it can do, and its limits. Thanks for watching.",
     "The website documents the assistant: what it sends, what it can do, and its limits. Thanks for watching."),
]

END_DUR = 13.0
ROWS = [('WEBSITE AND DOCS', 'sherin-sef-ai.github.io/Vantage', 'the assistant, the Copilot, what is sent, the limits'),
        ('DOWNLOAD', 'github.com/Sherin-SEF-AI/Vantage/releases', 'Windows · macOS · Linux x86-64 and arm64 (Jetson)'),
        ('SOURCE', 'github.com/Sherin-SEF-AI/Vantage', 'GPL-3.0-or-later · free and open source')]
FOOT = 'REAL GROQ REPLIES, UNEDITED  ·  RECORDED FROM THE APP  ·  VOICE: KOKORO-82M, LOCAL'


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
        gfx.end_card(img, (rec.f - f0) / rec.fps, END_DUR, rows=ROWS, foot=FOOT)
        return img
    r.card = card
    r.cursor_on = False
    r.hold(0.8)
    r.say(0)
    r.finish_speech(0.1)
    r.say(1)
    r.finish_speech(0.2)
    r.mark('music', db=-7.0, dur=max(1.0, END_DUR - r.now()))
    r.until(END_DUR)
