"""Draw the Vantage application icon (a car seen from above inside its four camera fans) and write it as
vantage/resources/icons/vantage.png (512 px, used for the window and the Linux menu entry), packaging/icons/vantage.ico
(Windows, 16-256 px) and packaging/icons/vantage.icns (macOS, 16-1024 px). Both containers hold PNG images, which
Windows Vista+ and macOS 10.7+ read.

    QT_QPA_PLATFORM=offscreen python3 packaging/make_icon.py
"""
import os
import struct
import sys

from PyQt6.QtCore import QBuffer, QByteArray, QIODevice, QPointF, QRectF, Qt
from PyQt6.QtGui import QColor, QGuiApplication, QImage, QPainter, QPainterPath, QPen

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BASE, BODY, GLASS, RIM = '#121211', '#c8c7c2', '#3b3c3b', '#383835'
FANS = [('#3987e5', 90), ('#d95926', 270), ('#199e70', 180), ('#c98500', 0)]     # front, rear, left, right (Qt angles)


def draw(n: int) -> QImage:
    img = QImage(n, n, QImage.Format.Format_ARGB32_Premultiplied)
    img.fill(Qt.GlobalColor.transparent)
    p = QPainter(img)
    p.setRenderHint(QPainter.RenderHint.Antialiasing)
    p.scale(n / 512.0, n / 512.0)
    tile = QRectF(16, 16, 480, 480)
    p.setPen(QPen(QColor(RIM), 6))
    p.setBrush(QColor(BASE))
    p.drawRoundedRect(tile, 104, 104)
    clip = QPainterPath()
    clip.addRoundedRect(tile.adjusted(3, 3, -3, -3), 101, 101)
    p.setClipPath(clip)
    # the camera fans: wide wedges from each camera, overlapping at the corners like a surround view
    p.setPen(Qt.PenStyle.NoPen)
    origin = {90: QPointF(256, 146), 270: QPointF(256, 366), 180: QPointF(212, 236), 0: QPointF(300, 236)}
    for col, ang in FANS:
        c = QColor(col)
        c.setAlphaF(0.62)
        p.setBrush(c)
        o = origin[ang]
        r = 300
        p.drawPie(QRectF(o.x() - r, o.y() - r, 2 * r, 2 * r), int((ang - 62) * 16), int(124 * 16))
    p.setClipping(False)
    # the car from above: body, windscreen and rear window
    p.setPen(QPen(QColor(BASE), 10))
    p.setBrush(QColor(BODY))
    p.drawRoundedRect(QRectF(196, 130, 120, 252), 48, 48)
    p.setPen(Qt.PenStyle.NoPen)
    p.setBrush(QColor(GLASS))
    p.drawRoundedRect(QRectF(212, 196, 88, 46), 14, 14)
    p.drawRoundedRect(QRectF(214, 304, 84, 34), 12, 12)
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
