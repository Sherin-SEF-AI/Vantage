"""Render the Vantage application icon from the logo kit in branding/ (the Surround mark on a dark tile; see
packaging/make_brand.py) and write it as
vantage/resources/icons/vantage.png (512 px, used for the window and the Linux menu entry), packaging/icons/vantage.ico
(Windows, 16-256 px), packaging/icons/vantage.icns (macOS, 16-1024 px) and the website's favicon, header logo and
Apple touch icon (site/assets/img). Both containers hold PNG images, which Windows Vista+ and macOS 10.7+ read.
Each size uses the mark drawn for it: the 16 px and 32 px versions have heavier strokes and wider gaps.

    QT_QPA_PLATFORM=offscreen python3 packaging/make_icon.py
"""
import os
import struct
import sys

from PyQt6.QtCore import QBuffer, QByteArray, QIODevice, QRectF, Qt
from PyQt6.QtGui import QColor, QGuiApplication, QImage, QPainter
from PyQt6.QtSvg import QSvgRenderer

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BRAND = os.path.join(ROOT, 'branding')
TILE = '#0A0B0B'


def _tile_file(n: int) -> str:
    return os.path.join(BRAND, 'app-icon-16-dark.svg' if n <= 20 else 'app-icon-32-dark.svg' if n <= 40
                        else 'app-icon-dark.svg')


def _mark_file(n: int) -> str:
    return os.path.join(BRAND, 'mark-16-white.svg' if n <= 20 else 'mark-32-white.svg' if n <= 40
                        else 'mark-white.svg')


def draw(n: int, square: bool = False) -> QImage:
    """The icon at n px: the rounded dark tile with the mark (a small margin from 64 px up, as desktop icons have);
    square=True: a full-bleed square tile (the Apple touch icon, which the system rounds itself)."""
    img = QImage(n, n, QImage.Format.Format_ARGB32_Premultiplied)
    img.fill(Qt.GlobalColor.transparent)
    p = QPainter(img)
    p.setRenderHint(QPainter.RenderHint.Antialiasing)
    p.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform)
    if square:
        p.fillRect(QRectF(0, 0, n, n), QColor(TILE))
        k = 0.6 * n
        QSvgRenderer(_mark_file(n)).render(p, QRectF((n - k) / 2, (n - k) / 2, k, k))
    else:
        m = n * 16 / 512 if n >= 64 else 0
        QSvgRenderer(_tile_file(n)).render(p, QRectF(m, m, n - 2 * m, n - 2 * m))
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
    site = os.path.join(ROOT, 'site', 'assets', 'img')
    for d in (res, out, site):
        os.makedirs(d, exist_ok=True)
    with open(os.path.join(res, 'vantage.png'), 'wb') as fh:
        fh.write(png(draw(512)))
    with open(os.path.join(out, 'vantage.ico'), 'wb') as fh:
        fh.write(ico([16, 24, 32, 48, 64, 128, 256]))
    with open(os.path.join(out, 'vantage.icns'), 'wb') as fh:
        fh.write(icns())
    for name, n, sq in (('favicon.png', 64, False), ('logo.png', 96, False), ('apple-touch-icon.png', 180, True)):
        with open(os.path.join(site, name), 'wb') as fh:
            fh.write(png(draw(n, square=sq)))
    print('wrote', res, out, site)


if __name__ == '__main__':
    main()
