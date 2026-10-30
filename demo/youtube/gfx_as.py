"""Motion graphics of the assistant episode (as070), on top of gfx.py and in its look: kinetic captions, the logo
without a version line, the title card and the thumbnail. Additive: nothing here changes what the other episodes
draw (they never import it).

Kinetic(text, ...) is an overlay layer like gfx's (r.add(Kinetic(...))): a short key statement whose words rise in one
after another on a frosted panel, the accented words in the accent colour with a line drawn under them, and leave
together. It sits in frame coordinates (after the zoom), so it stays put while the frame moves.
"""
from __future__ import annotations

import os

from PyQt6.QtCore import QPointF, QRectF, Qt
from PyQt6.QtGui import QImage, QLinearGradient, QBrush, QPainter, QPen

import gfx
from gfx import (ACCENT, ACCENT_SOFT, BASE, H, INK, INK2, MUTED, W, clamp, col, draw_text, font, glow, in_cubic,
                 out_cubic, out_quint, painter, panel, smoother, text_w, vignette, window)

SITE = gfx.SITE
RELEASES = gfx.RELEASES
REPO = gfx.REPO


class Kinetic(gfx.Layer):
    """A key statement, word by word. `text` may hold '\\n' for line breaks; words in `accent` (compared without
    punctuation, case-insensitive) are drawn in the accent colour and underlined. `at` is the top-left corner in frame
    coordinates, or 'left' (the left part of a frame zoomed onto the assistant panel) / 'bottom' (centred low)."""
    z = 33
    t_in = 0.3
    t_out = 0.45

    def __init__(self, text, dur=3.2, at='left', px=56, accent=(), eyebrow='', stagger=0.075):
        super().__init__(dur)
        self.lines = [ln.split() for ln in text.split('\n')]
        self.accent = {a.lower() for a in accent}
        self.px, self.eyebrow, self.stagger = px, eyebrow, stagger
        self.f = font(px, 'semibold')
        self.f_eye = font(17, 'semibold', mono=True, spacing=3.0)
        self.lh = px * 1.22
        self.space = text_w(self.f, ' ')
        self.w = max(sum(text_w(self.f, wd) for wd in ln) + self.space * (len(ln) - 1) for ln in self.lines)
        self.h = self.lh * len(self.lines) + (34 if eyebrow else 0)
        pad_x, pad_y = 40, 30
        if at == 'left':
            x, y = 86, H * 0.5 - (self.h + 2 * pad_y) / 2
        elif at == 'bottom':
            x, y = (W - self.w - 2 * pad_x) / 2, H - 150 - (self.h + 2 * pad_y)
        else:
            x, y = at
        self.box = QRectF(x, y, self.w + 2 * pad_x, self.h + 2 * pad_y)
        self.pad = (pad_x, pad_y)

    @staticmethod
    def _bare(word):
        return word.strip('.,;:!?…"“”‘’()').lower()

    def frost(self, rec, f):
        k = out_cubic(window(self.t(f), 0.0, 0.35)) * self.a_out(f)
        return [(self._panel_rect(f), 18.0, k)]

    def _panel_rect(self, f):
        k = out_quint(window(self.t(f), 0.0, 0.42))
        r = self.box
        return QRectF(r.x(), r.y(), max(8.0, r.width() * (0.35 + 0.65 * k)), r.height())

    def paint(self, p, rec, f):
        t = self.t(f)
        a_out = self.a_out(f)
        slide = -18 * in_cubic(1 - a_out)
        p.save()
        p.setOpacity(a_out)
        pr = self._panel_rect(f).translated(slide, 0)
        panel(p, pr, a=out_cubic(window(t, 0.0, 0.3)), radius=14, fill=col('#101010', 0.74))
        bar = out_cubic(window(t, 0.05, 0.4))
        p.fillRect(QRectF(pr.x(), pr.y() + pr.height() * (1 - bar), 4, pr.height() * bar), col(ACCENT))
        x0 = self.box.x() + self.pad[0] + slide
        y = self.box.y() + self.pad[1]
        if self.eyebrow:
            k = out_cubic(window(t, 0.12, 0.4))
            draw_text(p, x0, y + 18 + (1 - k) * 8, self.eyebrow.upper(), self.f_eye, ACCENT, k)
            y += 34
        i = 0
        for ln in self.lines:
            x = x0
            base = y + self.px * 0.98
            for wd in ln:
                t0 = 0.18 + i * self.stagger
                k = out_quint(window(t, t0, 0.42))
                ww = text_w(self.f, wd)
                acc = self._bare(wd) in self.accent
                draw_text(p, x, base + (1 - k) * 22, wd, self.f, ACCENT_SOFT if acc else INK, k)
                if acc:
                    u = out_cubic(window(t, t0 + 0.25, 0.4))
                    p.fillRect(QRectF(x, base + 10, ww * u, 3), col(ACCENT, 0.9 * k))
                x += ww + self.space
                i += 1
            y += self.lh
        p.restore()


def logo_reveal(img: QImage, t: float, dur: float, a_out: float = 1.0, line='Free and open source'):
    """gfx.logo_reveal without the version: the same icon, wordmark and tagline, and `line` under them."""
    import cv2
    import numpy as np
    from gfx import TAGLINE, icon_image, to_np
    p = painter(img)
    glow(p, W / 2, H / 2 - 40, 760, ACCENT, 0.09 * out_cubic(window(t, 0.0, 1.2)) * a_out)
    ic = icon_image()
    size = 196
    k = out_quint(window(t, 0.0, 1.1))
    s = size * (0.86 + 0.14 * k)
    f_word = font(132, 'semibold')
    ww = text_w(f_word, 'Vantage')
    slide = out_quint(window(t, 0.55, 0.9))
    total = size + 46 + ww
    cx_pair = W / 2 - total / 2
    icon_x = W / 2 - size / 2 + (cx_pair - (W / 2 - size / 2)) * slide
    icon_y = H / 2 - 150
    blur = (1 - k) * 18
    src = ic.scaled(int(s * 2), int(s * 2), Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation)
    if blur > 0.5:
        a = to_np(src.convertToFormat(QImage.Format.Format_ARGB32))
        a = cv2.GaussianBlur(a, (0, 0), blur)
        src = QImage(np.ascontiguousarray(a).data, a.shape[1], a.shape[0], a.shape[1] * 4,
                     QImage.Format.Format_ARGB32).copy()
    p.setOpacity(k * a_out)
    p.drawImage(QRectF(icon_x + (size - s) / 2, icon_y + (size - s) / 2, s, s), src)
    tx = icon_x + size + 46
    p.save()
    p.setClipRect(QRectF(icon_x + size + 10, 0, W, H))
    p.setOpacity(slide * a_out)
    draw_text(p, tx - (1 - slide) * 80, icon_y + size / 2 + 46, 'Vantage', f_word, INK)
    p.restore()
    k2 = out_cubic(window(t, 1.25, 0.7))
    f_tag = font(36, 'regular')
    tw = text_w(f_tag, TAGLINE)
    draw_text(p, W / 2 - tw / 2, icon_y + size + 112 + (1 - k2) * 12, TAGLINE, f_tag, INK2, k2 * a_out)
    k3 = out_cubic(window(t, 1.6, 0.7))
    f_l = font(21, 'medium', mono=True, spacing=1.0)
    lw = text_w(f_l, line)
    p.setOpacity(1.0)
    draw_text(p, W / 2 - lw / 2, icon_y + size + 170 + (1 - k3) * 10, line, f_l, MUTED, k3 * a_out)
    k4 = out_quint(window(t, 1.5, 0.8))
    p.fillRect(QRectF(W / 2 - 60 * k4, icon_y + size + 64, 120 * k4, 2), col(ACCENT, a_out))
    p.end()


def title_card(img: QImage, t: float, dur: float, eyebrow: str, title: str, sub: str, chips=()):
    """The title after the logo: an eyebrow, the title, an accent rule, a subtitle and a row of chips, each rising in
    after the other, all leaving together over the last 0.6 s."""
    out = 1 - in_cubic(clamp((t - (dur - 0.6)) / 0.6))
    p = painter(img)
    vignette(p, 0.4)
    glow(p, W / 2, H / 2 - 30, 820, ACCENT, 0.09 * out_cubic(window(t, 0, 1.0)) * out)
    k = out_cubic(window(t, 0.1, 0.5))
    f_e = font(22, 'semibold', mono=True, spacing=5.0)
    draw_text(p, W / 2 - text_w(f_e, eyebrow) / 2, H / 2 - 150 + (1 - k) * 12, eyebrow, f_e, ACCENT, k * out)
    k = out_cubic(window(t, 0.22, 0.6))
    f_t = font(112, 'semibold')
    draw_text(p, W / 2 - text_w(f_t, title) / 2, H / 2 - 10 + (1 - k) * 24, title, f_t, INK, k * out)
    k = out_quint(window(t, 0.5, 0.7))
    p.fillRect(QRectF(W / 2 - 90 * k, H / 2 + 40, 180 * k, 3), col(ACCENT, out))
    k = out_cubic(window(t, 0.62, 0.6))
    f_s = font(38)
    draw_text(p, W / 2 - text_w(f_s, sub) / 2, H / 2 + 118 + (1 - k) * 12, sub, f_s, INK2, k * out)
    if chips:
        f_c = font(21, 'medium', mono=True, spacing=1.2)
        gap = 18
        ws = [text_w(f_c, c) + 44 for c in chips]
        x = W / 2 - (sum(ws) + gap * (len(ws) - 1)) / 2
        for i, (c, cw) in enumerate(zip(chips, ws)):
            k = out_cubic(window(t, 0.9 + 0.12 * i, 0.5)) * out
            r = QRectF(x, H / 2 + 190 + (1 - k) * 10, cw, 46)
            p.setOpacity(k)
            p.setPen(QPen(col(ACCENT, 0.55), 1.2))
            p.setBrush(col('#101010', 0.6))
            p.drawRoundedRect(r, 23, 23)
            draw_text(p, r.x() + 22, r.center().y() + 7.5, c, f_c, INK)
            p.setOpacity(1.0)
            x += cw + gap
    p.end()


def thumbnail(shot: QImage | None, path: str):
    """1280 x 720, under 2 MB: the real proposal card from the episode (`shot`, a frame of the app zoomed on it) on the
    right, level and lit; the title large on the left."""
    from gfx import base_np, from_np, icon_image
    img = from_np(base_np())
    p = painter(img)
    p.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform)
    if shot is not None and not shot.isNull():
        if shot.height() > 400:                     # below the note chip at the top of the frame
            k = shot.height() / 1080
            shot = shot.copy(0, int(140 * k), shot.width(), shot.height() - int(140 * k))
        glow(p, 1420, 520, 900, ACCENT, 0.22)
        r = QRectF(760, 120, 1120, 1120 * shot.height() / max(1, shot.width()))
        p.save()                                    # kept level: a tilt shifts the card's rows against each other
        for k, a in ((60, 0.18), (30, 0.22), (10, 0.3)):
            p.setPen(Qt.PenStyle.NoPen)
            p.setBrush(col('#000000', a))
            p.drawRoundedRect(r.adjusted(-k / 2, -k / 4, k / 2, k), 24, 24)
        p.setClipRect(r)
        p.drawImage(r, shot)
        p.setClipping(False)
        p.setPen(QPen(col(ACCENT, 0.9), 4))
        p.setBrush(Qt.BrushStyle.NoBrush)
        p.drawRoundedRect(r, 18, 18)
        p.restore()
    g = QLinearGradient(0, 0, 1920, 0)
    g.setColorAt(0.0, col('#0b0b0a', 1.0))
    g.setColorAt(0.36, col('#0b0b0a', 0.96))
    g.setColorAt(0.5, col('#0b0b0a', 0.25))
    g.setColorAt(0.62, col('#0b0b0a', 0.0))
    p.fillRect(QRectF(0, 0, 1920, 1080), QBrush(g))
    p.drawImage(QRectF(110, 110, 130, 130), icon_image())
    draw_text(p, 270, 200, 'VANTAGE', font(52, 'semibold', mono=True, spacing=7), ACCENT)
    draw_text(p, 100, 470, 'AI assistant', font(168, 'semibold'), INK)
    draw_text(p, 106, 600, 'for sensor-rig design', font(84, 'semibold'), INK2)
    p.fillRect(QRectF(110, 650, 260, 8), col(ACCENT))
    f_c = font(44, 'semibold')
    for i, s in enumerate(('It proposes.', 'You decide.')):
        draw_text(p, 110, 770 + i * 74, s, f_c, INK if i == 0 else ACCENT_SOFT)
    draw_text(p, 112, 1000, 'REAL ANSWERS · GPT-OSS 120B ON GROQ', font(30, 'medium', mono=True, spacing=3), MUTED)
    p.end()
    small = img.scaled(1280, 720, transformMode=Qt.TransformationMode.SmoothTransformation)
    small.save(path, 'PNG')
    if os.path.getsize(path) > 2_000_000:
        small.save(path, 'PNG', 1)
    return path


def terminal(img: QImage, t: float, script, title='bash', cps=44.0, px=27, lh=39):
    """gfx.terminal with larger type (legible on a phone): [('cmd', text) | ('out', text[, tone])]. Returns the time
    the script finishes (s)."""
    from PyQt6.QtGui import QFontMetricsF, QPainterPath
    from gfx import HAIR, PANEL, STRONG
    p = painter(img)
    r = QRectF(110, 140, W - 220, H - 236)
    for k, a in ((30, 0.10), (14, 0.16), (5, 0.22)):
        p.setPen(Qt.PenStyle.NoPen)
        p.setBrush(col('#000000', a))
        p.drawRoundedRect(r.adjusted(-k / 2, -k / 4, k / 2, k), 16, 16)
    p.setBrush(col('#0d0d0c'))
    p.setPen(QPen(col(STRONG), 1))
    p.drawRoundedRect(r, 14, 14)
    bar = QPainterPath()
    bar.addRoundedRect(QRectF(r.x(), r.y(), r.width(), 52), 14, 14)
    p.fillPath(bar, col(PANEL))
    p.fillRect(QRectF(r.x(), r.y() + 38, r.width(), 14), col(PANEL))
    p.fillRect(QRectF(r.x(), r.y() + 52, r.width(), 1), col(HAIR))
    for i in range(3):
        p.setPen(Qt.PenStyle.NoPen)
        p.setBrush(col(STRONG))
        p.drawEllipse(QPointF(r.x() + 30 + i * 24, r.y() + 26), 7, 7)
    p.setFont(font(19, 'regular', mono=True))
    p.setPen(col(MUTED))
    p.drawText(QRectF(r.x(), r.y(), r.width(), 52), Qt.AlignmentFlag.AlignCenter, title)
    f = font(px, 'regular', mono=True)
    p.setFont(f)
    x0, y = r.x() + 40, r.y() + 52 + lh + 10
    clock = 0.3
    tones = {'ok': '#5fc35f', 'warn': '#fab219', 'bad': '#ec835a', '': INK2, 'dim': MUTED}
    shown = []
    for item in script:
        kind, s = item[0], item[1]
        if t < clock:
            break
        if kind == 'cmd':
            n = int(max(0, min(len(s), (t - clock) * cps)))
            shown.append(('cmd', s[:n], n < len(s)))
            clock += len(s) / cps + 0.4
            if n < len(s):
                break
        else:
            shown.append(('out', s, item[2] if len(item) > 2 else ''))
            clock += item[3] if len(item) > 3 else 0.09
    maxl = int((r.height() - 52 - 40) / lh)
    for kind, s, extra in shown[-maxl:]:
        if kind == 'cmd':
            p.setPen(col(ACCENT))
            p.drawText(QPointF(x0, y), '$')
            p.setPen(col(INK))
            p.drawText(QPointF(x0 + 32, y), s)
            if extra and int(t * 2.2) % 2 == 0:
                cw = QFontMetricsF(f).horizontalAdvance(s)
                p.fillRect(QRectF(x0 + 34 + cw, y - px + 3, px * 0.5, px + 4), col(INK))
        else:
            p.setPen(col(tones.get(extra, INK2)))
            p.drawText(QPointF(x0, y), s)
        y += lh
    p.end()
    return 0.3 + sum(len(it[1]) / cps + 0.4 if it[0] == 'cmd' else (it[3] if len(it) > 3 else 0.09) for it in script)


def scorecard(img: QImage, t: float, title: str, rows, overall: float, sub: str, foot=(), a_out=1.0):
    """The benchmark's result as a card: the overall score counting up on the left, one bar per category on the right
    (rows: [(label, questions, score 0-1)]), drawn in one after another; `foot` lines under it."""
    from gfx import fmt_number
    p = painter(img)
    vignette(p, 0.45)
    glow(p, 430, 470, 620, ACCENT, 0.10 * a_out)
    p.setOpacity(a_out)
    k = out_cubic(window(t, 0.0, 0.5))
    draw_text(p, 130, 250 + (1 - k) * 10, title, font(20, 'semibold', mono=True, spacing=3.0), ACCENT, k)
    u = out_cubic(window(t, 0.2, 1.6))
    f_big = font(220, 'semibold')
    s = fmt_number(overall * u, 0)
    draw_text(p, 120, 530, s, f_big, INK, out_cubic(window(t, 0.1, 0.4)))
    draw_text(p, 128 + text_w(f_big, s), 530, '%', font(90, 'semibold'), INK2, out_cubic(window(t, 0.3, 0.4)))
    k = out_cubic(window(t, 0.5, 0.6))
    draw_text(p, 130, 600 + (1 - k) * 8, sub, font(30), INK2, k)
    x, y0, bw, lh = 820, 240, 760, 66
    f_l = font(25)
    f_v = font(25, 'semibold', mono=True)
    for i, (lab, n, sc) in enumerate(rows):
        k = out_cubic(window(t, 0.4 + 0.12 * i, 0.5))
        g = out_quint(window(t, 0.55 + 0.12 * i, 0.9))
        y = y0 + i * lh
        draw_text(p, x, y + 4 + (1 - k) * 8, f'{lab}', f_l, INK, k)
        draw_text(p, x + bw + 30, y + 30, f'{round(sc * 100):d} %', f_v, INK if sc >= 0.9 else '#fab219', k)
        p.fillRect(QRectF(x, y + 18, bw, 10), col('#ffffff', 0.06 * k))
        p.fillRect(QRectF(x, y + 18, bw * sc * g, 10), col(ACCENT if sc >= 0.9 else '#c98f1a', 0.95 * k))
        draw_text(p, x + bw - text_w(font(18), f'{n} q') , y + 4 + (1 - k) * 8, f'{n} q', font(18), MUTED, k)
    k = out_cubic(window(t, 1.8, 0.6))
    for j, ln in enumerate(foot):
        draw_text(p, 130, 700 + j * 44 + (1 - k) * 8, ln, font(24), MUTED if j else INK2, k)
    p.end()
