"""Regression tests for UI bugs found in review: undo granularity, stale state after File > New / Open or a variant
switch, unit handling in imports and table cells, and fields that a refresh or a focus change must not rewrite."""
import json
import os
import sys
import tempfile
import time
import traceback
import types
import zipfile

os.environ['QT_QPA_PLATFORM'] = 'offscreen'
os.environ['XDG_CONFIG_HOME'] = tempfile.mkdtemp(prefix='vantage-test-cfg-')

import numpy as np
import pytest
from PyQt6.QtCore import Qt
from PyQt6.QtTest import QTest
from PyQt6.QtWidgets import QApplication, QMessageBox, QFileDialog


@pytest.fixture(scope='module')
def app():
    from vantage.ui.viewport3d import configure_default_format
    configure_default_format()
    a = QApplication.instance() or QApplication([])
    from vantage.ui.theme import THEME, load_fonts
    load_fonts()
    THEME.apply(a, 'dark')
    return a


def pump(app, s=0.3):
    t = time.time()
    while time.time() - t < s:
        app.processEvents()
        time.sleep(0.01)


def settle(app, w, timeout=60):
    pump(app, 0.2)
    t = time.time()
    while (w.ctx.analysis._busy or w.ctx.analysis.timer.isActive()) and time.time() - t < timeout:
        pump(app, 0.05)
    pump(app, 0.2)


def _wait(app, cond, timeout=120):
    t = time.time()
    while not cond() and time.time() - t < timeout:
        pump(app, 0.05)
    assert cond()


@pytest.fixture(autouse=True)
def no_modals(monkeypatch):
    """Message boxes and file dialogs never block; exceptions raised in Qt slots fail the test."""
    shown = []

    def box(*a, **k):
        shown.append(a[1:3])
        return QMessageBox.StandardButton.Discard
    for n in ('question', 'warning', 'critical', 'information'):
        monkeypatch.setattr(QMessageBox, n, staticmethod(box))
    for n in ('getOpenFileName', 'getSaveFileName'):
        monkeypatch.setattr(QFileDialog, n, staticmethod(lambda *a, **k: ('', '')))
    monkeypatch.setattr(QFileDialog, 'getExistingDirectory', staticmethod(lambda *a, **k: ''))
    errors = []
    monkeypatch.setattr(sys, 'excepthook', lambda t, v, tb: errors.append(''.join(traceback.format_exception(t, v, tb))))
    yield shown
    assert not errors, errors[0]


@pytest.fixture()
def win(app):
    from vantage.core import samples
    from vantage.ui.mainwindow import MainWindow
    w = MainWindow(samples.tigor_sample_project(), soft_3d=True)
    w.resize(1600, 1000)
    w.show()
    t = time.time()
    while w.ctx.analysis.result is None and time.time() - t < 60:
        pump(app, 0.05)
    settle(app, w)
    yield w
    w.store.undo.setClean()
    w.cancel_jobs()
    w.close()
    w.deleteLater()
    pump(app, 0.1)


def _type(app, box, text):
    """Type into a ScrubSpinBox like a user (focus, select all, keys) without finishing the edit."""
    box.window().activateWindow()
    box.setFocus()
    pump(app, 0.05)
    box.lineEdit().selectAll()
    QTest.keyClicks(box.lineEdit(), text)


# ------------------------------------------------------------------------------------------------ inspector
def test_multi_select_edit_reaches_cameras_after_linked_pair(app, win):
    """#4: a linked pair early in a multi-selection stopped the edit before the remaining cameras."""
    st = win.store
    ml, mr, f = (st.variant.by_tag(t) for t in ('ML', 'MR', 'F'))
    before = [(c.y, c.z) for c in (ml, mr, f)]
    st.select_cams([ml.uid, mr.uid, f.uid])
    settle(app, win)
    i0 = st.undo.index()
    page = win.inspector.pages['camera']
    page.pz.lineEdit().setText('1.5')
    page.pz._on_finished()
    settle(app, win)
    assert [st.cam(u).z for u in (ml.uid, mr.uid, f.uid)] == [1.5, 1.5, 1.5]
    assert st.undo.index() == i0 + 1
    page.py.lineEdit().setText('0.5')                # Y of a linked pair stays mirrored
    page.py._on_finished()
    settle(app, win)
    assert st.cam(ml.uid).y == 0.5 and st.cam(mr.uid).y == -0.5 and st.cam(f.uid).y == 0.5
    st.undo.undo()
    st.undo.undo()
    settle(app, win)
    assert [(st.cam(u).y, st.cam(u).z) for u in (ml.uid, mr.uid, f.uid)] == before


def test_scrub_with_two_cameras_is_one_undo_step(app, win):
    """#24: scrubbing with two cameras selected pushed an undo entry per camera and step."""
    st = win.store
    f, r = st.variant.by_tag('F'), st.variant.by_tag('R')
    z0 = (f.z, r.z)
    st.select_cams([f.uid, r.uid])
    settle(app, win)
    page = win.inspector.pages['camera']
    i0 = st.undo.index()
    st.dragging = True
    for k in range(10):
        page._set('z', 1.0 + 0.01 * k, merge=True)
    st.dragging = False
    assert st.undo.index() == i0 + 1
    assert st.cam(f.uid).z == st.cam(r.uid).z == 1.09
    st.undo.undo()
    settle(app, win)
    assert (st.cam(f.uid).z, st.cam(r.uid).z) == z0


def test_refresh_keeps_the_text_being_typed(app, win):
    """#22: an inspector refresh (another edit, an analysis finishing) overwrote a field the user was typing in."""
    st = win.store
    f = st.variant.by_tag('F')
    st.select_cams([f.uid])
    settle(app, win)
    page = win.inspector.pages['camera']
    _type(app, page.py, '0.25')
    st.set_cam(st.variant.by_tag('R').uid, z=0.9)
    win.inspector.refresh()
    assert page.py.lineEdit().text() == '0.25'
    QTest.keyClick(page.py.lineEdit(), Qt.Key.Key_Return)
    settle(app, win)
    assert st.cam(f.uid).y == 0.25


def test_typing_the_shown_rounded_value_applies_it(app, win):
    """#23: typing exactly the (rounded) value a field shows did nothing; leaving a field untouched must not round."""
    st = win.store
    f = st.variant.by_tag('F')
    st.set_cam(f.uid, yaw=23.4567, x=3.23456)
    st.select_cams([f.uid])
    settle(app, win)
    page = win.inspector.pages['camera']
    assert page.ry.lineEdit().text() == '23.5' and page.px.lineEdit().text() == '3.235'
    i0 = st.undo.index()
    win.activateWindow()
    page.px.setFocus()                                # in and out without typing: no edit
    page.pz.setFocus()
    pump(app, 0.05)
    assert st.cam(f.uid).x == 3.23456 and st.undo.index() == i0
    _type(app, page.ry, '23.5')
    QTest.keyClick(page.ry.lineEdit(), Qt.Key.Key_Return)
    settle(app, win)
    _type(app, page.px, '3.235')
    QTest.keyClick(page.px.lineEdit(), Qt.Key.Key_Return)
    settle(app, win)
    assert st.cam(f.uid).yaw == 23.5 and st.cam(f.uid).x == 3.235


def test_zone_delete_undo_restores_face_zones(app, win, tmp_path):
    """#9: undoing 'Delete zone' brought the zone back but not which faces belong to it (also Clear mount zones)."""
    st = win.store
    m = st.project.vehicle.mesh
    fz0 = m.face_zone.copy()
    names = [z.name for z in m.zones]

    def delete(p, i=0):                               # what ZonesPage's 'Delete zone' runs
        mesh = p.vehicle.mesh
        mesh.face_zone = np.where(mesh.face_zone == i, -1, np.where(mesh.face_zone > i, mesh.face_zone - 1,
                                                                     mesh.face_zone)).astype(np.int16)
        mesh.zones.pop(i)
    st.edit('vehicle', delete, 'Deleted zone')
    after = st.project.vehicle.mesh.face_zone.copy()
    st.undo.undo()
    m = st.project.vehicle.mesh
    assert [z.name for z in m.zones] == names and np.array_equal(m.face_zone, fz0)
    st.undo.redo()
    m = st.project.vehicle.mesh
    assert len(m.zones) == len(names) - 1 and np.array_equal(m.face_zone, after)
    st.undo.undo()
    # the undo state is not written into project.json; the zones still save and load
    path = str(tmp_path / 'z.vantage')
    st.project.save(path)
    with zipfile.ZipFile(path) as z:
        assert 'face_zone' not in json.loads(z.read('project.json'))['vehicle']
    from vantage.core.project import Project
    assert np.array_equal(Project.load(path).vehicle.mesh.face_zone, fz0)


def test_non_finite_seam_angle_and_target_position_are_rejected(app, win):
    """#16: 'nan' / 'inf' in the seam angle or target X/Y cells were stored and broke the analysis."""
    st = win.store
    st.select_obj(('variant',))
    settle(app, win)
    vp = win.inspector.pages['variant']
    i0 = st.undo.index()
    vp.t.item(0, 1).setText('nan')
    settle(app, win)
    assert not st.variant.seams and st.undo.index() == i0
    st.select_obj(('targets',))
    settle(app, win)
    tp = win.inspector.pages['targets']
    tp.refresh()
    x0 = st.project.targets[0].x
    tp.t.item(0, 1).setText('inf')
    tp.t.item(0, 2).setText('nan')
    settle(app, win)
    assert st.project.targets[0].x == x0 and np.isfinite(st.project.targets[0].y) and st.undo.index() == i0


def test_target_setting_change_is_one_undo_step(app, win):
    """#25: changing the board rows made two undo steps (settings, then every target)."""
    st = win.store
    st.select_obj(('targets',))
    settle(app, win)
    rows0 = st.project.target_settings.rows
    i0 = st.undo.index()
    win.inspector.pages['targets']._ts('rows', 9)
    assert st.undo.index() == i0 + 1
    assert st.project.target_settings.rows == 9 and all(t.rows == 9 for t in st.project.targets)
    st.undo.undo()
    assert st.project.target_settings.rows == rows0 and all(t.rows == rows0 for t in st.project.targets)


def test_leaving_text_fields_unchanged_adds_no_undo_step(app, win):
    """#26: focus leaving ECU name / location, an obstacle name or the variant note pushed an undo entry."""
    st = win.store
    st.undo.setClean()
    i0 = st.undo.index()
    st.select_obj(('ecu',))
    settle(app, win)
    ep = win.inspector.pages['ecu']
    ep.name.editingFinished.emit()
    ep.loc.editingFinished.emit()
    st.select_obj(('obstacle', st.project.obstacles[0].uid))
    settle(app, win)
    win.inspector.pages['obstacle'].name.editingFinished.emit()
    st.select_obj(('variant',))
    settle(app, win)
    win.inspector.pages['variant'].note.editingFinished.emit()
    assert st.undo.index() == i0 and st.undo.isClean()
    ep.name.setText('Orin A')                          # a real change still is one
    ep.name.editingFinished.emit()
    assert st.undo.index() == i0 + 1 and st.project.ecu.name == 'Orin A'


# ------------------------------------------------------------------------------------------------ cameras table
def _paste(app, win, text):
    QApplication.clipboard().setText(text)
    win.bottom.cameras.paste()
    settle(app, win)


def test_cameras_paste_is_one_undo_step_and_skips_locked_and_unknown(app, win):
    """#10: a paste made one undo step per cell, moved locked cameras, and a row with a tag this variant does not
    have was written onto whichever camera sat at that row position."""
    from vantage.ui import ops
    st = win.store
    win.bottom.show_tab('cameras')
    pump(app)
    f0, r0 = (st.variant.by_tag(t) for t in ('F', 'R'))
    f0, r0 = types.SimpleNamespace(**vars(f0)), types.SimpleNamespace(**vars(r0))      # values, not the live cameras
    i0 = st.undo.index()
    _paste(app, win, 'Tag\tName\tX\tY\tZ\tYaw °\tPitch °\nF\tFront\t3.25\t0.01\t0.63\t1\t31\nR\tRear\t-0.8\t0\t0.6\t180\t35\n')
    assert st.undo.index() == i0 + 1
    assert st.variant.by_tag('F').pitch == 31 and st.variant.by_tag('R').pitch == 35
    st.undo.undo()
    settle(app, win)
    f, r = st.variant.by_tag('F'), st.variant.by_tag('R')
    assert (f.x, f.pitch, r.x, r.pitch) == (f0.x, f0.pitch, r0.x, r0.pitch)
    ops.set_lock(win.ctx, [f.uid], True)
    i1 = st.undo.index()
    _paste(app, win, 'Tag\tName\tX\nF\tFront\t2.0\nX9\tOther\t1.5\n')
    assert st.variant.by_tag('F').x == f0.x
    assert all(c.x != 1.5 for c in st.variant.cameras)
    assert st.undo.index() == i1                       # nothing applied, nothing pushed


def test_port_cell_accepts_only_valid_ports(app, win):
    """#15: the Port cell took any integer, and 0 swapped the camera's port onto another unassigned camera."""
    from vantage.ui.bottom.panels import COLS
    st = win.store
    model = win.bottom.cameras.model
    col = [k for k, _ in COLS].index('port')
    row = [c.tag for c in st.variant.cameras].index('F')
    i0 = st.undo.index()
    for txt in ('12', '-3', '2.5'):
        assert not model.set_cell(row, col, txt)
    assert st.variant.by_tag('F').port == 1 and st.undo.index() == i0
    qr = st.variant.by_tag('QR')
    st.set_cam(qr.uid, port=0)
    assert model.set_cell(row, col, '0')
    ports = {c.tag: c.port for c in st.variant.cameras}
    assert ports['F'] == 0 and ports['QR'] == 0
    assert model.set_cell(row, col, '2')               # a valid port still swaps with its holder
    assert st.variant.by_tag('F').port == 2 and st.variant.by_tag('R').port == 0


# ------------------------------------------------------------------------------------------------ dialogs
def test_module_import_add_to_library_keeps_the_existing_module(app, win, tmp_path, monkeypatch):
    """#1: 'Add to library' after importing a module JSON wrote the import over the module in use."""
    from vantage.ui.dialogs.module_editor import ModuleEditor
    st = win.store
    p = st.project
    path = str(tmp_path / 'other.json')
    with open(path, 'w') as fh:
        json.dump({'module': {'name': 'Other cam', 'width_px': 1280, 'height_px': 960, 'pixel_um': 3.75},
                   'lens': {'name': 'Other lens', 'focal_mm': 2.1}}, fh)
    monkeypatch.setattr(QFileDialog, 'getOpenFileName', staticmethod(lambda *a, **k: (path, '')))
    mid, lid = next(iter(p.modules)), next(iter(p.lenses))
    before = p.modules[mid].to_dict()
    dlg = ModuleEditor(win.ctx, mid, lid, win, start_import=True)
    assert dlg.ok.text() == 'Add to library'
    dlg.apply()
    settle(app, win)
    p = st.project
    assert p.modules[mid].to_dict() == before and len(p.modules) == 2
    new_lens = next(l for l in p.lenses.values() if l.name == 'Other lens')
    new_mod = p.modules[new_lens.module_id]
    assert new_mod.id != mid and (new_mod.name, new_mod.width_px) == ('Other cam', 1280)
    assert p.module_for(st.variant.by_tag('F')).width_px == before['width_px']
    st.undo.undo()
    assert list(st.project.modules) == [mid]


def test_module_editor_fy_cell_and_image_circle(app, win):
    """Suspected #5: the fy px cell changed the focal length, and any field edit froze lens.max_half_deg to the
    rounded derived value."""
    from vantage.ui.dialogs.module_editor import ModuleEditor
    p = win.store.project
    f = win.store.variant.by_tag('F')
    dlg = ModuleEditor(win.ctx, f.module_id, f.lens_id, win)
    l = dlg.lens
    focal, mh = l.focal_mm, l.max_half_deg
    dlg.f['l_name'].setText('Renamed lens')
    dlg.read_fields()
    assert l.max_half_deg == mh == 0.0
    fx = float(dlg.kb.item(0, 4).text())
    dlg.kb.item(0, 5).setText(f'{fx * 1.01:.3f}')
    assert l.focal_mm == focal and l.aspect == pytest.approx(1.01, abs=1e-4)
    assert float(dlg.kb.item(0, 5).text()) == pytest.approx(fx * 1.01, abs=0.05)
    dlg.f['l_maxhalf'].set_value(88.0)                 # editing the field itself still sets the limit
    dlg.f['l_maxhalf'].committed.emit(88.0)
    assert l.max_half_deg == 88.0
    dlg.reject()


def test_project_rename_can_be_undone(app, win):
    """#17: the name changed inside an 'analysis' edit, so undo kept the new name."""
    from vantage.ui.dialogs.misc import ProjectSettingsDialog
    st = win.store
    old = st.project.name
    d = ProjectSettingsDialog(win.ctx, win)
    d.name.setText('Renamed project')
    d.apply()
    assert st.project.name == 'Renamed project'
    st.undo.undo()
    assert st.project.name == old
    st.undo.redo()
    assert st.project.name == 'Renamed project'


def test_import_layout_reads_units_angles_and_frame_of_the_export(app, tmp_path):
    """#3: importing the app's own extrinsics export ignored its units (mm), angles (rad) and frame."""
    from vantage.core import exporters as ex, samples
    from vantage.ui.dialogs.misc import import_layout_file
    p = samples.tigor_sample_project()
    ref = {c.tag: c for c in p.variant.cameras}
    for frame in ('rear_axle', 'front_axle'):
        for units, angles in (('mm', 'rad'), ('m', 'deg'), ('mm', 'deg')):
            out = str(tmp_path / f'{frame}_{units}_{angles}')
            opts = ex.ExportOptions(frame=frame, units=units, angles=angles, out_dir=out, images={}, issues=[])
            paths = ex.export(['extrinsics_json', 'extrinsics_csv'], p, None, opts)
            for fp in [x for x in paths if os.path.basename(x) in ('extrinsics.json', 'extrinsics.csv')]:
                cams = import_layout_file(fp, p)
                assert [c.tag for c in cams] == list(ref), fp
                for c in cams:
                    r = ref[c.tag]
                    assert np.allclose(c.pos(), r.pos(), atol=1.5e-3), (fp, c.tag, c.pos(), r.pos())
                    assert (c.yaw, c.pitch, c.roll) == pytest.approx((r.yaw, r.pitch, r.roll), abs=0.01), (fp, c.tag)
                    assert c.port == r.port
    # a hand-written CSV without units is metres and degrees in the display frame, as before
    path = tmp_path / 'hand.csv'
    path.write_text('tag,name,x,y,z,yaw,pitch\nF,Front,3.2,0,0.6,0,40\n')
    c = import_layout_file(str(path), p)[0]
    assert (c.x, c.z, c.pitch) == (3.2, 0.6, 40.0)


def test_import_wizard_reuse_keeps_other_landmarks(app, win):
    """#11: re-running the wizard on the current body kept only the wheel-centre landmarks."""
    from vantage.ui.dialogs.import_wizard import ImportWizard
    st = win.store
    lm0 = dict(st.project.vehicle.landmarks)
    wiz = ImportWizard(win.ctx, win, step=2, reuse=True)
    wiz.show()
    pump(app, 0.3)
    wiz.go(4)
    pump(app, 0.1)
    wiz.finish()
    settle(app, win)
    lm = st.project.vehicle.landmarks
    assert set(lm) == set(lm0)
    for k in lm0:
        assert np.allclose(lm[k], lm0[k], atol=1e-3), k
    st.undo.undo()
    settle(app, win)


def test_load_cases_drop_columns_read_units(app, win):
    """#12: '2 cm' in a millimetre column was read as metres."""
    d = win.load_cases()
    pump(app, 0.1)
    d.table.item(1, 1).setText('2 cm')
    d.table.item(1, 2).setText('0.5 in')
    assert d.table.item(1, 1).text() == '20' and d.table.item(1, 2).text() == '12.7'
    assert d.cases()[1][1:] == (20.0, 12.7)
    d.close()


def test_load_cases_dialog_follows_the_project(app, win):
    """#13: the open dialog kept the old project's cases after File > Open and Run wrote them into the new one."""
    from vantage.core import samples
    st = win.store
    d = win.load_cases()
    pump(app, 0.1)
    d.table.item(1, 2).setText('55')
    newp = samples.tigor_sample_project()
    newp.analysis.load_cases = [['Kerb', 0, 0], ['Driver only', 5, 3]]
    st.set_project(newp)
    settle(app, win)
    assert d.table.rowCount() == 2 and d.table.item(1, 2).text() == '3'
    _wait(app, lambda: win.ctx.analysis.result is not None)
    d.run()
    _wait(app, lambda: not d.busy and d.results is not None)
    assert st.project.analysis.load_cases == [['Kerb', 0, 0], ['Driver only', 5, 3]]
    assert len(d.results) == 2
    d.close()


def test_load_cases_without_a_result_says_so(app, win):
    """#21: Run with no analysis result (no vehicle) failed with a traceback in the worker."""
    from vantage.core import samples
    win.store.set_project(samples.blank_project())
    settle(app, win)
    d = win.load_cases()
    pump(app, 0.1)
    d.run()
    pump(app, 0.2)
    assert not d.busy and d.results is None
    assert 'analysis result' in d.sub_lbl.text()
    d.close()


def test_export_dialog_does_not_reuse_another_variants_folder(app, win, tmp_path):
    """#14: the last export folder was proposed for every project and variant, so a second export overwrote the
    first one's files."""
    from vantage.core import exporters as ex
    from vantage.ui.dialogs.export_dialog import ExportDialog
    st = win.store
    v1, v2 = st.project.variants[0], st.project.variants[1]

    def export_to(path):
        d = ExportDialog(win.ctx, win)
        d.open_after.setChecked(False)
        for fid, r in d.rows.items():
            r.cb.setChecked(fid == 'extrinsics_json')
        d.out.setText(path)
        d.do_export()
    first = str(tmp_path / 'exp' / ex._slug(v1.name))
    export_to(first)
    assert os.path.exists(os.path.join(first, 'extrinsics.json'))
    win.switch_variant(v2.uid)
    settle(app, win)
    d = ExportDialog(win.ctx, win)
    assert d.out.text() == str(tmp_path / 'exp' / ex._slug(v2.name))
    d.close()
    custom = str(tmp_path / 'custom')
    export_to(custom)
    d = ExportDialog(win.ctx, win)
    assert d.out.text() == custom                      # the same variant again: its own folder
    d.close()
    win.switch_variant(v1.uid)
    settle(app, win)
    d = ExportDialog(win.ctx, win)
    assert d.out.text() != custom
    d.close()


def test_export_dialog_with_every_camera_hidden(app, win):
    """#20: a checked format without files (every camera hidden) raised IndexError in the dialog."""
    from vantage.core import exporters as ex
    from vantage.ui import ops
    from vantage.ui.dialogs.export_dialog import ExportDialog
    ops.set_visible(win.ctx, [c.uid for c in win.store.variant.cameras], False)
    settle(app, win)
    d = ExportDialog(win.ctx, win, preselect='img_cams')
    assert 'camera_views_png' not in d.tab_ids
    for f in ex.FORMATS:
        d.rows[f.id].cb.setChecked(True)
    assert d.tabs.count() == len(d.tab_ids) and d.tab_ids
    d.select(d.tab_ids[-1])
    assert d.tabs.currentIndex() == len(d.tab_ids) - 1
    d.timer.stop()
    d.close()


def test_lens_calibration_result_stays_with_its_camera(app, win):
    """#6: a calibration was applied to whichever camera the combo showed when it finished."""
    st = win.store
    f, r = st.variant.by_tag('F'), st.variant.by_tag('R')
    d = win.lens_calibration(f.uid)
    pump(app, 0.1)
    d.source.buttons['synthetic'].click()
    d.syn_n.set_value(10)
    d.render_synthetic()
    _wait(app, lambda: not d.busy and len(d.views) == 10)
    d.run()
    assert d.busy and not d.cam.isEnabled()
    d.cam.setCurrentIndex(d.cam.findData(r.uid))      # e.g. the list rebuilt while it ran
    _wait(app, lambda: not d.busy)
    assert d.result is None and not d.b_apply.isEnabled() and d.cam.isEnabled()
    d.apply()
    assert not st.cam(r.uid).calib and not st.cam(f.uid).calib
    d.cam.setCurrentIndex(d.cam.findData(f.uid))
    d.run()
    _wait(app, lambda: not d.busy)
    assert d.result is not None and d.result.ok and d.b_apply.text() == 'Apply to F'
    d.apply()
    pump(app, 0.1)
    assert st.cam(f.uid).calib and not st.cam(r.uid).calib
    d.close()


def test_lens_calibration_dialog_follows_file_new(app, win):
    """#19: after File > New the camera list still showed the old cameras and Render views crashed."""
    from vantage.core import samples
    d = win.lens_calibration(win.store.variant.by_tag('F').uid)
    pump(app, 0.1)
    win.store.set_project(samples.blank_project())
    settle(app, win)
    assert d.cam.count() == 0
    d.source.buttons['synthetic'].click()
    d.render_synthetic()
    d.run()
    d.apply()
    d.save('unused.yaml')
    assert not d.busy and not d.b_run.isEnabled() and not d.b_apply.isEnabled()
    win.store.set_project(samples.tigor_sample_project())
    settle(app, win)
    assert d.cam.count() == len(win.store.variant.cameras) and d.cam.currentData() == win.store.variant.cameras[0].uid
    d.close()


# ------------------------------------------------------------------------------------------------ workspaces
def test_cancelled_tolerance_study_keeps_only_evaluated_samples(app, win):
    """#5: a cancelled study counted the samples it never ran as 'zero blind' and showed made-up statistics."""
    from vantage.core.tolerance import TolSettings, run_monte_carlo
    eng = win._shared_engine()
    p = win.store.project.snapshot()
    st = TolSettings(samples=120, seed=3)
    calls = [0]

    def cancel():
        calls[0] += 1
        return calls[0] > 25
    res = run_monte_carlo(eng, p, st, cancel=cancel, threads=1)
    assert res.cancelled and len(res.blind) == len(res.min_density) == len(res.seam_shift) == 25
    full = run_monte_carlo(eng, p, TolSettings(samples=25, seed=3), threads=1)
    assert not full.cancelled and np.allclose(res.blind, full.blind) and np.allclose(res.min_density, full.min_density)
    assert all(c.index < 25 for c in res.worst)
    win.page('tolerance').show_result(res)
    win.bottom.tolerance.show_result(res)
    assert 'cancelled' in win.page('tolerance').count.text() and '25 / 120' in win.page('tolerance').count.text()
    assert 'cancelled after 25 of 120 samples' in win.bottom.tolerance.summary.text()
    with pytest.raises(RuntimeError):
        run_monte_carlo(eng, p, st, cancel=lambda: True, threads=1)


def test_optimiser_candidates_do_not_survive_a_new_project(app, win):
    """#7: candidates of the old project stayed after File > New / Open and Apply added them to the new one."""
    from vantage.core import samples
    st = win.store
    win.set_page('optimiser')
    pump(app, 0.2)
    op = win.page('optimiser')
    old_opt = types.SimpleNamespace(to_variant=lambda cand, name: pytest.fail('applied a stale candidate'))
    op.opt, op.cands, op.sel = old_opt, [types.SimpleNamespace(label='C1')], 0
    st.set_project(samples.blank_project())
    settle(app, win)
    assert op.opt is None and not op.cands and op.sel is None and op.table.rowCount() == 0
    n = len(st.project.variants)
    op.apply()
    op.finished(old_opt)                               # a run of the old project finishing late
    assert len(st.project.variants) == n and not op.cands
    win.set_page('viewports')


def test_compare_follows_library_changes_and_new_projects(app, win):
    """#8, #27: the compare cache ignored the lens library (and vehicle, obstacles, settings), and the variant combos
    kept the old project's variants after File > Open."""
    from vantage.core import samples
    st = win.store
    win.set_page('compare')
    cp = win.page('compare')
    nv = len(st.project.variants)
    _wait(app, lambda: sum(1 for v in st.project.variants if v.uid in cp.results) == nv, 240)
    uid1 = st.project.variants[1].uid
    b0 = cp.results[uid1].metrics['blind_m2']

    def narrow(p):
        for l in p.lenses.values():
            l.focal_mm *= 2.5
            l.projection = 'rectilinear'
    st.edit('library', narrow, 'Narrow lenses')
    _wait(app, lambda: cp.results.get(uid1) is not None and cp.results[uid1].metrics['blind_m2'] > b0 + 1.0, 240)
    newp = samples.tigor_sample_project()
    newp.variants = newp.variants[:1]
    newp.variants[0].name = 'Only variant'
    st.set_project(newp)
    settle(app, win)
    combos = [c['combo'] for c in cp.cols]
    assert all([cb.itemText(i) for i in range(cb.count())] == ['Only variant'] for cb in combos)
    assert all(st.project.variant_by_uid(c['uid']) is not None for c in cp.cols)
    win.set_page('viewports')
    settle(app, win)


def test_install_apply_as_built_belongs_to_its_variant(app, win, monkeypatch):
    """#18: 'Apply as-built' stayed enabled after a variant switch and counted cameras of another variant."""
    st = win.store
    win.set_page('install')
    pump(app, 0.3)
    pg = win.page('install')
    cams = st.variant.cameras

    def pose(c):
        return types.SimpleNamespace(pos=c.pos() + np.array([0.01, 0, 0]), yaw=c.yaw + 1, pitch=c.pitch, roll=c.roll,
                                     status='good', message='', boards=[], n_corners=0, rms_px=0.1, dpos_mm=np.zeros(3),
                                     dang_deg=np.zeros(3), single=None, joint=False)
    pg.results = {c.uid: pose(c) for c in cams}
    pg.results['not-a-camera'] = pose(cams[0])
    pg.results_variant = st.project.active_uid
    pg._sync_buttons()
    assert pg.b_apply.isEnabled()
    win.switch_variant(st.project.variants[1].uid)
    settle(app, win)
    assert not pg.b_apply.isEnabled() and not pg.results
    n = len(st.project.variants)
    pg.apply_as_built()
    assert len(st.project.variants) == n
    st.undo.undo()                                     # back to the measured variant
    settle(app, win)
    pg.results = {c.uid: pose(c) for c in st.variant.cameras[:2]}
    pg.results['not-a-camera'] = pose(cams[0])
    pg.results_variant = st.project.active_uid
    msgs = []
    monkeypatch.setattr(win.ctx, 'msg', lambda m, *a: msgs.append(m))
    pg.apply_as_built()
    assert len(st.project.variants) == n + 1 and 'with 2 measured camera pose(s)' in msgs[-1]
    win.set_page('viewports')


# ------------------------------------------------------------------------------------------------ project file
def test_resave_keeps_the_stored_thumbnail(app, tmp_path):
    """Project.save(path) without a thumbnail (batch --save / --as-built) dropped the one the file had."""
    from vantage.core import samples
    from vantage.core.project import Project
    png = b'\x89PNG\r\n\x1a\nnot-really-a-png'
    a, b = str(tmp_path / 'a.vantage'), str(tmp_path / 'b.vantage')
    samples.tigor_sample_project().save(a, png)
    p = Project.load(a)
    p.save(b)
    with zipfile.ZipFile(b) as z:
        assert z.read('thumb.png') == png
