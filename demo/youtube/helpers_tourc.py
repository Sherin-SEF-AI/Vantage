"""Small helpers shared by tour-c's chapters (27 to 38): events sent straight to a widget, finding buttons by their
text, and table cells as targets. Imported inside scene() (it imports Qt)."""
from __future__ import annotations

import numpy as np
from PyQt6.QtCore import QEvent, QPointF, QRectF, Qt
from PyQt6.QtGui import QMouseEvent
from PyQt6.QtWidgets import QAbstractButton, QApplication

from gfx import smoother


def _mouse(widget, kind, pt, button, buttons):
    pt = QPointF(pt)
    g = widget.mapToGlobal(pt.toPoint())
    QApplication.sendEvent(widget, QMouseEvent(kind, pt, QPointF(g), button, buttons, Qt.KeyboardModifier.NoModifier))


def drag(r, widget, a, b, dur=1.2, approach=0.6):
    """Press at a, drag to b and release, with the events sent straight to the widget (works without mouse tracking
    and without a real cursor), the pointer following."""
    a, b = QPointF(a), QPointF(b)
    r.move_to(widget, a, approach)
    r.tick(3)
    r.ripple()
    L, N = Qt.MouseButton.LeftButton, Qt.MouseButton.NoButton
    _mouse(widget, QEvent.Type.MouseButtonPress, a, L, L)
    n = max(2, round(dur * r.fps))
    for k in range(1, n + 1):
        t = smoother(k / n)
        p = QPointF(a.x() + (b.x() - a.x()) * t, a.y() + (b.y() - a.y()) * t)
        _mouse(widget, QEvent.Type.MouseMove, p, N, L)
        q = r.wpos(widget, p)
        r.cur = np.array([q.x(), q.y()])
        r.tick()
    _mouse(widget, QEvent.Type.MouseButtonRelease, b, L, N)
    r.app.processEvents()
    r.tick()


def button(parent, text):
    """The button under parent whose text (without & and …) starts with text."""
    for b in parent.findChildren(QAbstractButton):
        t = b.text().replace('&', '').strip()
        if t.startswith(text) and b.isVisible():
            return b
    for b in parent.findChildren(QAbstractButton):
        if b.text().replace('&', '').strip().startswith(text):
            return b
    raise LookupError(f'no button {text!r}')


def cell(r, table, row, col=None):
    """A table row (col None) or cell, as a target in window coordinates (evaluated when used)."""
    def f():
        if not table.isVisible() or row >= table.rowCount():
            return None
        if col is None:
            y = table.rowViewportPosition(row)
            rect = QRectF(0, y, table.horizontalHeader().length(), table.rowHeight(row))
        else:
            rect = QRectF(table.columnViewportPosition(col), table.rowViewportPosition(row), table.columnWidth(col),
                          table.rowHeight(row))
        tl = r.wpos(table.viewport(), rect.topLeft())
        return QRectF(tl.x(), tl.y(), rect.width(), rect.height())
    return f


def rows_rect(r, table, first=0, last=None, cols=None):
    """Header plus rows first..last of a table (all columns, or the first `cols`), as a rectangle now."""
    last = table.rowCount() - 1 if last is None else last
    g = r.wrect(table)
    hh = table.horizontalHeader().height() if table.horizontalHeader().isVisible() else 0
    width = (table.columnViewportPosition(cols - 1) + table.columnWidth(cols - 1)) if cols else \
        table.horizontalHeader().length()
    y0 = g.y() + hh + table.rowViewportPosition(first) if first else g.y()
    y1 = g.y() + hh + table.rowViewportPosition(last) + table.rowHeight(last)
    return QRectF(g.x(), y0, min(width + 4, g.width()), y1 - y0 + 2)


def union(*rects):
    out = None
    for x in rects:
        x = QRectF(x)
        out = x if out is None else out.united(x)
    return out


# ---------------------------------------------------------------------- real commands on a terminal card
def run_timed(cmd, cwd, env=None, home=None, timeout=900):
    """Run a shell command for real; returns (exit code, [(seconds since start, line)], seconds). stdout and stderr
    are read together, line by line, with the time each line arrived; the throw-away home is shown as ~."""
    import os
    import subprocess
    import time
    t0 = time.time()
    p = subprocess.Popen(cmd, shell=True, cwd=cwd, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                         text=True, bufsize=1)
    lines = []
    for ln in p.stdout:
        ln = ln.rstrip('\n')
        if home:
            ln = ln.replace(home + os.sep, '~' + os.sep).replace(home, '~')
        lines.append((time.time() - t0, ln))
    code = p.wait(timeout=timeout)
    return code, lines, time.time() - t0


def terminal(r, script, title='bash', cps=38.0):
    """A terminal card like gfx.terminal, with ('wait', seconds) items: commands type at cps, output lines appear
    0.12 s apart or after the waits given. Returns the seconds it needs; r.card = None ends it."""
    from PyQt6.QtCore import QRectF, Qt
    from PyQt6.QtGui import QColor, QImage, QPainter
    import gfx
    flat = []                        # (kind, text, tone, time it appears)
    clock = 0.3
    for item in script:
        kind = item[0]
        if kind == 'wait':
            clock += float(item[1])
            continue
        flat.append((kind, item[1], item[2] if len(item) > 2 else '', clock))
        clock += len(item[1]) / cps + 0.45 if kind == 'cmd' else 0.12
    total = clock
    f0 = r.f

    def card(rec):
        t = (rec.f - f0) / rec.fps
        img = QImage(gfx.W, gfx.H, QImage.Format.Format_RGB32)
        p = QPainter(img)
        p.fillRect(QRectF(0, 0, gfx.W, gfx.H), QColor(gfx.BASE))
        gfx.vignette(p, 0.5)
        p.end()
        sub = []
        for kind, text, tone, at in flat:
            if t < at:
                break
            if kind == 'cmd':
                sub.append(('cmd', text))
            else:
                sub.append(('out', text, tone))
        # the commands still typing: gfx.terminal types a command from its own clock, so give it the partial time
        last = flat[len(sub) - 1] if sub else None
        tt = 0.3 + sum(len(x[1]) / cps + 0.45 if x[0] == 'cmd' else 0.12 for x in sub[:-1]) if sub else 0.0
        if last is not None and last[0] == 'cmd':
            tt += min(t - last[3], len(last[1]) / cps + 0.44)
        else:
            tt = 0.3 + sum(len(x[1]) / cps + 0.45 if x[0] == 'cmd' else 0.12 for x in sub) + 0.01
        gfx.terminal(img, tt, sub, title, cps)
        return img
    r.card = card
    return total


def wrap(line, width=112, indent=None):
    """A long output line as several, continued under its text (indent: the column it continues at)."""
    if len(line) <= width:
        return [line]
    if indent is None:
        indent = len(line) - len(line.lstrip())
    out, cur = [], line
    while len(cur) > width:
        k = cur.rfind(' ', indent + 10, width)
        k = k if k > 0 else width
        out.append(cur[:k])
        cur = ' ' * indent + cur[k:].lstrip()
    out.append(cur)
    return out


def tone_of(line):
    s = line.strip()
    if s.startswith(('fail', 'warning', 'bad')) or ' warning ' in f' {s[:40]} ':
        return 'warn'
    if ' ok ' in f' {s[:28]} ' or s.startswith(('pass', 'good')) or 'self-test passed' in s:
        return 'ok'
    return ''


def timed_output(lines, speed=1.0, keep=None, width=112, indent=None, tones=False):
    """[(t, line)] as terminal items with ('wait', dt / speed) between them (keep(line) filters lines; long lines
    wrap)."""
    out, prev = [], 0.0
    for t, ln in lines:
        if keep is not None and not keep(ln):
            continue
        dt = max(0.0, t - prev) / speed
        if dt > 0.13:
            out.append(('wait', dt - 0.12))
        tone = tone_of(ln) if tones else ''
        for part in wrap(ln, width, indent):
            out.append(('out', part, tone))
        prev = t
    return out


# ---------------------------------------------------------------------- cards
def table_card(r, title, sub, header, rows, foot='', col_w=None):
    """A full-frame card: a title, a subtitle and a table (header + rows of strings), drawn in the tour's look; it
    fades its rows in. Returns the layer function set as r.card."""
    from PyQt6.QtCore import QRectF, Qt
    from PyQt6.QtGui import QColor, QImage, QPainter, QPen
    import gfx
    f0 = r.f
    col_w = col_w or [1.0 / len(header)] * len(header)

    def card(rec):
        t = (rec.f - f0) / rec.fps
        img = QImage(gfx.W, gfx.H, QImage.Format.Format_RGB32)
        p = QPainter(img)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        p.setRenderHint(QPainter.RenderHint.TextAntialiasing)
        p.fillRect(QRectF(0, 0, gfx.W, gfx.H), QColor(gfx.BASE))
        gfx.glow(p, gfx.W * 0.5, gfx.H * 0.4, gfx.W * 0.55, gfx.ACCENT, 0.06)
        gfx.vignette(p, 0.5)
        x0, w = 190.0, gfx.W - 380.0
        a = gfx.smoother(t / 0.5)
        p.setOpacity(a)
        p.setFont(gfx.font(54, 'semibold'))
        p.setPen(gfx.col(gfx.INK))
        p.drawText(QRectF(x0, 150, w, 70), Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter, title)
        p.setFont(gfx.font(24))
        p.setPen(gfx.col(gfx.MUTED))
        p.drawText(QRectF(x0, 222, w, 40), Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter, sub)
        y = 320.0
        p.setFont(gfx.font(19, 'semibold'))
        p.setPen(gfx.col(gfx.MUTED))
        x = x0
        for h, cw in zip(header, col_w):
            p.drawText(QRectF(x, y, w * cw - 20, 34), Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter, h)
            x += w * cw
        p.fillRect(QRectF(x0, y + 42, w, 1), gfx.col(gfx.HAIR))
        y += 60
        for i, row in enumerate(rows):
            ra = gfx.smoother((t - 0.5 - 0.35 * i) / 0.5)
            if ra <= 0:
                break
            p.setOpacity(ra)
            x = x0
            for j, (txt, cw) in enumerate(zip(row, col_w)):
                p.setFont(gfx.font(22, 'semibold' if j == 0 else 'regular', mono=(j == 1)))
                p.setPen(gfx.col(gfx.INK if j != 2 else gfx.INK2))
                p.drawText(QRectF(x, y, w * cw - 24, 92), Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignTop
                           | Qt.TextFlag.TextWordWrap, txt)
                x += w * cw
            p.setOpacity(ra * 0.6)
            p.fillRect(QRectF(x0, y + 96, w, 1), gfx.col(gfx.HAIR))
            y += 110
        if foot:
            p.setOpacity(gfx.smoother((t - 0.8 - 0.35 * len(rows)) / 0.5))
            p.setFont(gfx.font(22, mono=True))
            p.setPen(gfx.col(gfx.ACCENT_SOFT))
            p.drawText(QRectF(x0, y + 20, w, 40), Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter, foot)
        p.end()
        return img
    r.card = card
    return card


def pages_card(r, images, caption=''):
    """A full-frame card with document pages (QImages) side by side, dropped in one after the other."""
    from PyQt6.QtCore import QRectF, Qt
    from PyQt6.QtGui import QColor, QImage, QPainter
    import gfx
    f0 = r.f

    def card(rec):
        t = (rec.f - f0) / rec.fps
        img = QImage(gfx.W, gfx.H, QImage.Format.Format_RGB32)
        p = QPainter(img)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        p.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform)
        p.fillRect(QRectF(0, 0, gfx.W, gfx.H), QColor(gfx.BASE))
        gfx.glow(p, gfx.W * 0.5, gfx.H * 0.45, gfx.W * 0.6, gfx.ACCENT, 0.06)
        gfx.vignette(p, 0.5)
        ph = 880.0
        n = len(images)
        pw = [ph * im.width() / im.height() for im in images]
        gap = 60.0
        x = (gfx.W - sum(pw) - gap * (n - 1)) / 2
        for k, (im, w_) in enumerate(zip(images, pw)):
            a = gfx.smoother((t - 0.45 * k) / 0.6)
            if a > 0:
                dy = (1 - a) * 40
                rect = QRectF(x, 70 + dy, w_, ph)
                p.setOpacity(a)
                for kk, aa in ((26, 0.10), (12, 0.16), (4, 0.24)):
                    p.fillRect(rect.adjusted(-kk / 2, kk / 3, kk / 2, kk), gfx.col('#000000', aa))
                p.drawImage(rect, im)
            x += w_ + gap
        if caption:
            p.setOpacity(gfx.smoother((t - 0.6) / 0.6))
            p.setFont(gfx.font(22))
            p.setPen(gfx.col(gfx.MUTED))
            p.drawText(QRectF(0, 970, gfx.W, 40), Qt.AlignmentFlag.AlignCenter, caption)
        p.end()
        return img
    r.card = card
    return card


def wait_job(r, cond, timeout=600, fast=6.0, grace=0.3):
    """Record a background job in real time while the current line is spoken (and `grace` seconds after it); if it
    is still running then, carry on as a marked time-lapse (x fast, the factor measured): the job is never shown
    faster than it ran without the badge, and there is no long silence."""
    t_end = r.speech_end + grace
    r.wait_for(lambda: cond() or r.now() >= t_end, timeout=timeout, speed=1)
    if not cond():
        r.wait_for(cond, timeout=timeout, speed=fast)


def light_wait(r, seconds, compose_hz=5.0):
    """Record `seconds` of wall time in real time (the video keeps pace with the clock), composing a new frame at
    most compose_hz times a second and repeating it in between: the recorder then leaves the computer to what the
    app is measuring in real time (live streams), at the price of a lower picture rate on screen meanwhile."""
    import time
    t0, f0 = time.time(), r.f
    t_end = t0 + seconds
    img = None
    while True:
        now = time.time()
        if now >= t_end and img is not None:
            break
        r.app.processEvents()
        r._advance()
        img = r.compose()
        want = int((time.time() - t0) * r.fps)
        for _ in range(max(1, want - (r.f - f0))):
            r._write(img)
            r.f += 1
        nxt = now + 1.0 / compose_hz
        while time.time() < min(nxt, t_end):
            r.app.processEvents()
            time.sleep(0.01)
    for _ in range(int(r.fps * 3)):                      # catch up with the clock, writes included (bounded)
        if int((time.time() - t0) * r.fps) <= r.f - f0:
            break
        r._write(img)
        r.f += 1
    r.events.append({'kind': 'wait', 'start': round(f0 / r.fps, 3), 'video_s': round((r.f - f0) / r.fps, 3),
                     'wall_s': round(time.time() - t0, 3), 'speed': 1.0})
