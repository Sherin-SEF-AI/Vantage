"""Website clip 'library': the e-con Systems camera library and the module editor.

The clip: the Library dock (the modules grouped, each with its lenses), Camera ▸ Edit module… on the front camera's
STURDeCAM31 + H05R2, the editor's Preset list (every library lens under its module), STURDeCAM88 with its wide H04R1
lens picked: the datasheet fields e-con's page gives, the Source block (the page, the date it was read, the figures it
does not give) and the datasheet-vs-model field of view check. Nothing is applied (the editor is cancelled).

Camera ▸ Edit module… runs the editor modally (exec), which would stop the script: the clip walks the menu like a user
and then shows the same editor itself.
"""
from __future__ import annotations

from PyQt6.QtCore import QPoint, QPointF, QRectF
from PyQt6.QtWidgets import QApplication, QMenu, QScrollArea

PICK = 'sturdecam88_h04r1'


def _walk_menu(r, title, *path):
    """Open a menu-bar menu and walk to an item without triggering it (Recorder.menu without the trigger)."""
    mb = r.win.menuBar()
    m = next(a.menu() for a in mb.actions() if a.menu() and a.text().replace('&', '') == title)
    rect = mb.actionGeometry(m.menuAction())
    r.move_to(mb, QPointF(rect.center()), 0.7)
    r.ripple()
    m.popup(mb.mapToGlobal(rect.bottomLeft()))
    r.hold(0.35)
    for name in path:
        act = next(a for a in m.actions() if a.text().replace('&', '').startswith(name))
        ar = m.actionGeometry(act)
        r.move_to(m, QPointF(ar.center().x() - ar.width() * 0.2, ar.center().y()), 0.55)
        m.setActiveAction(act)
        r.hold(0.25)
    r.ripple()
    r.hold(0.15)
    for w in QApplication.topLevelWidgets():
        if isinstance(w, QMenu) and w.isVisible():
            w.hide()


def clip(stage, r):
    from vantage.ui.dialogs.module_editor import ModuleEditor
    w = stage.window()
    r.win = w
    st = w.store
    cam = st.variant.by_tag('F')
    st.select_cams([cam.uid])
    stage.settle(2)
    r.chapter('Camera library', 'e-con Systems modules, each figure with its source')
    r.hold(0.6)
    lib = r.wrect(w.docks['library'])
    r.zoom(lib, 0.7, margin=8)
    r.hold(1.6)
    r.zoom(None, 0.5)
    _walk_menu(r, 'Camera', 'Edit module')
    d = ModuleEditor(w.ctx, cam.module_id, cam.lens_id, w)
    d.resize(min(1500, w.width() - 60), min(860, w.height() - 30))
    d.show()
    for _ in range(6):
        stage.app.processEvents()
    # placed after show (showing places it again); a dialog wider than the window keeps its left side in view
    d.move(w.mapToGlobal(QPoint(max(0, (w.width() - d.width()) // 2), max(0, (w.height() - d.height()) // 2))))
    for _ in range(4):
        stage.app.processEvents()
    print(f'  module editor {d.width()} x {d.height()} at {d.x() - w.x()}, {d.y() - w.y()}', flush=True)
    r.hold(1.4)
    # the Preset list: every library lens under its module
    r.click(d.preset, dur=0.6, real=False)
    d.preset.showPopup()
    r.hold(2.4)
    i = d.preset.findData(PICK)
    d.preset.hidePopup()
    d.preset.setCurrentIndex(i)
    d.apply_preset(PICK)
    r.hold(2.2)                                      # the datasheet fields e-con's page gives
    # scroll the datasheet column down to the Source block: the page, the date it was read, what it does not give
    sa = d.source.parentWidget()
    while sa is not None and not isinstance(sa, QScrollArea):
        sa = sa.parentWidget()
    if sa is not None:
        bar = sa.verticalScrollBar()
        v0, v1 = bar.value(), bar.maximum()
        r.move_to(sa.viewport(), QPointF(sa.viewport().width() * 0.6, sa.viewport().height() * 0.7), 0.5)
        n = round(0.8 * stage.fps)
        for k in range(1, n + 1):
            t = k / n
            bar.setValue(round(v0 + (v1 - v0) * t * t * (3 - 2 * t)))
            r.tick()
    src = r.wrect(d.source)
    r.zoom(QRectF(src.x() - 10, src.y() - 40, src.width() + 20, src.height() + 50), 0.7, margin=8)
    r.hold(2.6)
    r.zoom(None, 0.5)
    # the stated field of view against the model's
    r.zoom(r.wrect(d.circle).united(r.wrect(d.cmp)), 0.7, margin=12)
    r.hold(2.0)
    r.zoom(None, 0.5)
    r.move_to(d.ok, None, 0.6)                       # the button now offers to switch the cameras: not pressed
    r.hold(1.4)
    d.reject()
    r.cursor_on = False
    r.hold(0.8)
