"""Draw the Vantage application icon (a car seen from above with a lidar on its roof, inside its four camera fans and
the lidar's scan rings: the sensor rig) and write it as
vantage/resources/icons/vantage.png (512 px, used for the window and the Linux menu entry), packaging/icons/vantage.ico
(Windows, 16-256 px) and packaging/icons/vantage.icns (macOS, 16-1024 px). Both containers hold PNG images, which
Windows Vista+ and macOS 10.7+ read.

    QT_QPA_PLATFORM=offscreen python3 packaging/make_icon.py
"""
import os
import struct
import sys

from PyQt6.QtCore import QBuffer, QByteArray, QIODevice, QPointF, QRectF, Qt
from PyQt6.QtGui import QColor, QGuiApplication, QImage, QLinearGradient, QPainter, QPainterPath, QPen, QRadialGradient

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BASE, BODY, GLASS, RIM = '#121211', '#c8c7c2', '#3b3c3b', '#383835'
TOP, LIDAR = '#1f1f1d', '#0aa5b8'          # the tile's lighter top edge; the lidar's colour in the app
FANS = [('#3987e5', 90), ('#d95926', 270), ('#199e70', 180), ('#c98500', 0)]     # front, rear, left, right (Qt angles)


def draw(n: int) -> QImage:
    img = QImage(n, n, QImage.Format.Format_ARGB32_Premultiplied)
    img.fill(Qt.GlobalColor.transparent)
    p = QPainter(img)
    p.setRenderHint(QPainter.RenderHint.Antialiasing)
    p.scale(n / 512.0, n / 512.0)
    tile = QRectF(16, 16, 480, 480)
    g = QLinearGradient(0, 16, 0, 496)
    g.setColorAt(0.0, QColor(TOP))
    g.setColorAt(1.0, QColor(BASE))
    p.setPen(QPen(QColor(RIM), 6))
    p.setBrush(g)
    p.drawRoundedRect(tile, 104, 104)
    clip = QPainterPath()
    clip.addRoundedRect(tile.adjusted(3, 3, -3, -3), 101, 101)
    p.setClipPath(clip)
    # the camera fans: wide wedges from each camera, overlapping at the corners like a surround view, fading outwards
    p.setPen(Qt.PenStyle.NoPen)
    origin = {90: QPointF(256, 146), 270: QPointF(256, 366), 180: QPointF(212, 236), 0: QPointF(300, 236)}
    for col, ang in FANS:
        o = origin[ang]
        r = 300
        rg = QRadialGradient(o, r)
        for t, a in ((0.0, 0.80), (0.55, 0.62), (1.0, 0.34)):
            c = QColor(col)
            c.setAlphaF(a)
            rg.setColorAt(t, c)
        p.setBrush(rg)
        p.drawPie(QRectF(o.x() - r, o.y() - r, 2 * r, 2 * r), int((ang - 62) * 16), int(124 * 16))
    # the lidar's scan rings on the ground (left out where they would only blur a small icon)
    centre = QPointF(256, 272)
    if n >= 48:
        for r, a in ((150, 0.70), (196, 0.55), (242, 0.40)):
            c = QColor(LIDAR)
            c.setAlphaF(a)
            p.setPen(QPen(c, 7))
            p.setBrush(Qt.BrushStyle.NoBrush)
            p.drawEllipse(centre, r, r * 0.92)
    p.setClipping(False)
    # the car from above: body, windscreen and rear window
    p.setPen(QPen(QColor(BASE), 10))
    p.setBrush(QColor(BODY))
    p.drawRoundedRect(QRectF(196, 130, 120, 252), 48, 48)
    p.setPen(Qt.PenStyle.NoPen)
    p.setBrush(QColor(GLASS))
    p.drawRoundedRect(QRectF(212, 196, 88, 46), 14, 14)
    p.drawRoundedRect(QRectF(214, 304, 84, 34), 12, 12)
    # the lidar on the roof, between the windows
    p.setPen(QPen(QColor(BASE), 7))
    p.setBrush(QColor(LIDAR))
    p.drawEllipse(centre, 25, 25)
    p.setPen(Qt.PenStyle.NoPen)
    p.setBrush(QColor(255, 255, 255, 150))
    p.drawEllipse(QPointF(centre.x() - 7, centre.y() - 8), 7, 7)
    p.end()
    return img


def png(img: QImage) -> bytes:
    ba = QByteArray()
    buf = QBuffer(ba)
    buf.open(QIODevice.OpenModeFlag.WriteOnly)
    img.save(buf, 'PNG')
    buf.close()
    return bytes(ba)


def ico(sizes) -> bytes:
    data = [png(draw(s)) for s in sizes]
    head = struct.pack('<HHH', 0, 1, len(sizes))
    off = 6 + 16 * len(sizes)
    entries = b''
    for s, d in zip(sizes, data):
        entries += struct.pack('<BBBBHHII', s % 256, s % 256, 0, 0, 1, 32, len(d), off)
        off += len(d)
    return head + entries + b''.join(data)


def icns() -> bytes:
    body = b''
    for kind, s in ((b'icp4', 16), (b'icp5', 32), (b'icp6', 64), (b'ic07', 128), (b'ic08', 256), (b'ic09', 512),
                    (b'ic10', 1024)):
        d = png(draw(s))
        body += kind + struct.pack('>I', 8 + len(d)) + d
    return b'icns' + struct.pack('>I', 8 + len(body)) + body


def main():
    QGuiApplication.instance() or QGuiApplication(sys.argv[:1])
    res = os.path.join(ROOT, 'vantage', 'resources', 'icons')
    out = os.path.join(ROOT, 'packaging', 'icons')
    os.makedirs(res, exist_ok=True)
    os.makedirs(out, exist_ok=True)
    with open(os.path.join(res, 'vantage.png'), 'wb') as fh:
        fh.write(png(draw(512)))
    with open(os.path.join(out, 'vantage.ico'), 'wb') as fh:
        fh.write(ico([16, 24, 32, 48, 64, 128, 256]))
    with open(os.path.join(out, 'vantage.icns'), 'wb') as fh:
        fh.write(icns())
    print('wrote', res, out)


if __name__ == '__main__':
    main()
