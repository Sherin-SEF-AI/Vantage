"""Helpers for tour-a's chapters (02-07, 10-16): files the scenes prepare in the throw-away home folder.

No Qt at import time: chapter scenes import this inside scene().
"""
from __future__ import annotations

import os

import numpy as np


def write_body_obj(path, source=None):
    """Write the sample's own body as a Wavefront OBJ the way many modelling tools export a car: millimetres, Y up,
    the nose towards -Z, the origin at the middle of the body; each wheel its own group, the mirrors one, the glass
    with a see-through material (.mtl beside it). Returns the path. The import chapter brings it back in with the
    wizard: nothing about it is invented, it is the sample's mesh in another file's conventions."""
    from vantage.core import samples
    from vantage.core.mesh import ROLE_GLASS, ROLE_WHEELS
    p = source or samples.tigor_sample_project()
    m = p.vehicle.mesh
    v = np.asarray(m.vertices, float)
    f = np.asarray(m.faces, np.int64)
    fp = np.asarray(m.face_part)
    lo, hi = v.min(axis=0), v.max(axis=0)
    c = (lo + hi) / 2
    c[2] = 0.0
    # vehicle (x forward, y left, z up) -> file (X = -y, Y = z, Z = -x), millimetres
    q = v - c
    fv = np.stack([-q[:, 1], q[:, 2], -q[:, 0]], axis=1) * 1000.0
    groups = []                        # (group, material, face indices)
    cen = v[f].mean(axis=1)
    xm = (lo[0] + hi[0]) / 2
    for i, part in enumerate(m.parts):
        sel = np.nonzero(fp == i)[0]
        if not len(sel):
            continue
        name = part.name.lower()
        if part.role == ROLE_WHEELS:
            for tag, sx, sy in (('front_left', 1, 1), ('front_right', 1, -1), ('rear_left', -1, 1),
                                ('rear_right', -1, -1)):
                s = sel[(np.sign(cen[sel, 0] - xm) == sx) & (np.sign(cen[sel, 1]) == sy)]
                groups.append((f'wheel_{tag}', 'tyre_rubber', s))
        elif name.startswith('mirror'):
            groups.append(('mirrors', 'body_paint', sel))
        elif part.role == ROLE_GLASS:
            groups.append(('glass', 'glass_tinted', sel))
        else:
            groups.append(('body', 'body_paint', sel))
    os.makedirs(os.path.dirname(path), exist_ok=True)
    mtl = os.path.splitext(path)[0] + '.mtl'
    with open(mtl, 'w') as fh:
        fh.write('# materials\n'
                 'newmtl body_paint\nKd 0.78 0.78 0.76\nd 1.0\n\n'
                 'newmtl glass_tinted\nKd 0.18 0.19 0.20\nd 0.35\n\n'
                 'newmtl tyre_rubber\nKd 0.13 0.13 0.13\nd 1.0\n')
    with open(path, 'w') as fh:
        fh.write(f'# body exported in millimetres, Y up\nmtllib {os.path.basename(mtl)}\n')
        fh.write(''.join(f'v {a:.1f} {b:.1f} {cc:.1f}\n' for a, b, cc in fv))
        for g, mat, sel in groups:
            fh.write(f'g {g}\nusemtl {mat}\n')
            fh.write(''.join(f'f {a + 1} {b + 1} {cc + 1}\n' for a, b, cc in f[sel]))
    return path


def drag_sent(r, widget, a, b, dur=1.0, button=None, path=None, mods=None, during=None, hold_end=0.0, kick=False):
    """Press, drag and release on a widget with the moves sent straight to it (QTest.mouseMove does not reach widgets
    without mouse tracking offscreen): for 3D views (orbit, pan, painting) and sliders. `path(t) -> QPointF` replaces
    the straight line from a to b. `mods`: keyboard modifiers held on every event (Ctrl snaps, Shift fine); `during`:
    (fraction, fn) calls fn() once the drag is that far (Esc mid-drag); hold_end: seconds held still before the release
    (the read-out stays up); kick: a first move 6 px towards b right after the press, so a 3D view orbits at once
    instead of showing its ground read-out for the first few pixels."""
    from PyQt6.QtCore import QEvent, QPointF, Qt
    from PyQt6.QtGui import QMouseEvent
    from PyQt6.QtWidgets import QApplication
    btn = button or Qt.MouseButton.LeftButton
    km = mods if mods is not None else Qt.KeyboardModifier.NoModifier

    def send(kind, pt, held):
        g = widget.mapToGlobal(QPointF(pt).toPoint())
        ev = QMouseEvent(kind, QPointF(pt), QPointF(g), btn if kind != QEvent.Type.MouseMove else Qt.MouseButton.NoButton,
                         held, km)
        QApplication.sendEvent(widget, ev)
    r.move_to(widget, a, 0.6)
    r.tick(3)
    r.ripple()
    send(QEvent.Type.MouseButtonPress, a, btn)
    if kick:
        dx, dy = b.x() - a.x(), b.y() - a.y()
        d = max(1e-6, (dx * dx + dy * dy) ** 0.5)
        send(QEvent.Type.MouseMove, QPointF(a.x() + 6 * dx / d, a.y() + 6 * dy / d), btn)
    n = max(2, round(dur * r.fps))
    done = during is None
    for k in range(1, n + 1):
        t = k / n
        if not done and t >= during[0]:
            during[1]()
            done = True
        t = t * t * t * (t * (t * 6 - 15) + 10)
        p = path(t) if path is not None else QPointF(a.x() + (b.x() - a.x()) * t, a.y() + (b.y() - a.y()) * t)
        send(QEvent.Type.MouseMove, p, btn)
        q = r.wpos(widget, p)
        r.cur = np.array([q.x(), q.y()])
        r.tick()
    end = path(1.0) if path is not None else b
    if hold_end > 0:
        r.hold(hold_end)
    send(QEvent.Type.MouseButtonRelease, end, Qt.MouseButton.NoButton)
    r.tick(3)


def wheel(r, widget, pt, clicks, dur=1.2):
    """Turn a mouse wheel over a point of a widget: `clicks` notches (positive zooms in), spread over dur seconds."""
    from PyQt6.QtCore import QPoint, QPointF, Qt
    from PyQt6.QtGui import QWheelEvent
    from PyQt6.QtWidgets import QApplication
    r.move_to(widget, pt, 0.6)
    n = max(1, round(dur * r.fps))
    sign = 1 if clicks > 0 else -1
    sent = 0
    for k in range(1, n + 1):
        want = round(abs(clicks) * k / n)
        while sent < want:
            g = widget.mapToGlobal(QPointF(pt).toPoint())
            ev = QWheelEvent(QPointF(pt), QPointF(g), QPoint(0, 0), QPoint(0, 120 * sign), Qt.MouseButton.NoButton,
                             Qt.KeyboardModifier.NoModifier, Qt.ScrollPhase.NoScrollPhase, False)
            QApplication.sendEvent(widget, ev)
            sent += 1
        r.tick()


def click_sent(r, widget, pt, dur=0.6, mods=None):
    """Move to a point of a widget and click there with events sent straight to it (3D views, the nav control)."""
    from PyQt6.QtCore import QEvent, QPointF, Qt
    from PyQt6.QtGui import QMouseEvent
    from PyQt6.QtWidgets import QApplication
    km = mods if mods is not None else Qt.KeyboardModifier.NoModifier
    r.move_to(widget, pt, dur)
    r.tick(2)
    r.ripple()
    g = widget.mapToGlobal(QPointF(pt).toPoint())
    for kind, b, held in ((QEvent.Type.MouseButtonPress, Qt.MouseButton.LeftButton, Qt.MouseButton.LeftButton),
                          (QEvent.Type.MouseButtonRelease, Qt.MouseButton.LeftButton, Qt.MouseButton.NoButton)):
        QApplication.sendEvent(widget, QMouseEvent(kind, QPointF(pt), QPointF(g), b, held, km))
    r.tick()


def project_pt(r, vp, xyz):
    """A world point as a point in the 3D view's own coordinates (QPointF), or None behind the camera."""
    from PyQt6.QtCore import QPointF
    s, _, ok = vp.project([np.asarray(xyz, float)])
    return QPointF(*s[0]) if ok[0] else None


def edit_cell(r, view, row, col, text, cps=10.0):
    """Double-click a table cell (QTableView or QTableWidget), type `text` into its editor and press Enter."""
    from PyQt6.QtCore import Qt
    from PyQt6.QtTest import QTest
    from PyQt6.QtWidgets import QApplication, QLineEdit
    idx = view.model().index(row, col)
    view.scrollTo(idx)
    rect = view.visualRect(idx)
    r.move_to(view.viewport(), rect.center(), 0.6)
    r.ripple()
    r.tick(2)
    r.ripple()
    view.setCurrentIndex(idx)
    view.edit(idx)
    for _ in range(3):
        QApplication.processEvents()
    eds = [e for e in view.viewport().findChildren(QLineEdit) if e.isVisible()]
    if not eds:
        raise RuntimeError('no cell editor opened')
    ed = eds[-1]
    ed.selectAll()
    r.hold(0.2)
    for ch in text:
        QTest.keyClicks(ed, ch)
        r.hold(1.0 / cps)
    r.hold(0.2)
    QTest.keyClick(ed, Qt.Key.Key_Return)
    QApplication.processEvents()
    r.tick()

