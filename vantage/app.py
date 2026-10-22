"""Vantage entry point."""
from __future__ import annotations

import argparse
import os
import sys


APP_ICON = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'resources', 'icons', 'vantage.png')


def make_app(theme=None, scale=None):
    """The QApplication with Vantage's fonts, theme and icon; returns (app, settings)."""
    from PyQt6.QtCore import Qt, QSettings
    from PyQt6.QtGui import QIcon
    from PyQt6.QtWidgets import QApplication
    from .ui.viewport3d import configure_default_format
    configure_default_format()
    QApplication.setHighDpiScaleFactorRoundingPolicy(Qt.HighDpiScaleFactorRoundingPolicy.PassThrough)
    app = QApplication(sys.argv[:1])
    app.setApplicationName('Vantage')
    app.setOrganizationName('Vantage')
    app.setApplicationDisplayName('Vantage')
    app.setDesktopFileName('vantage')              # Wayland matches windows to vantage.desktop by this id
    from .ui.theme import THEME, load_fonts
    from .ui import icons
    load_fonts()
    settings = QSettings('Vantage', 'Vantage')
    THEME.apply(app, theme or settings.value('ui/theme', 'dark'), scale or float(settings.value('ui/scale', 1.0)))
    app.setWindowIcon(QIcon(APP_ICON) if os.path.exists(APP_ICON) else icons.icon('app', 16, THEME.tok['ink']))
    return app, settings


def main(argv=None):
    from . import __version__
    ap = argparse.ArgumentParser(prog='vantage', description='Vantage · sensor rig planner for autonomous-driving data collection')
    ap.add_argument('--version', action='version', version=f'vantage {__version__}')
    ap.add_argument('--self-test', action='store_true',
                    help='check this installation (analysis, OpenCV, fonts, OpenGL, main window) without a window')
    ap.add_argument('path', nargs='?', help='a .vantage project, or a vehicle model to import')
    ap.add_argument('--sample', action='store_true', help='open the Tata Tigor EV sample project')
    ap.add_argument('--template', help='start from a template: tigor, sedan, hatchback, suv, van, truck')
    ap.add_argument('--theme', choices=['dark', 'light', 'system'])
    ap.add_argument('--scale', type=float, help='UI scale, e.g. 1.15')
    ap.add_argument('--software-3d', action='store_true', help='draw the 3D view with QPainter instead of OpenGL')
    ap.add_argument('--no-start', action='store_true', help='skip the start window')
    from . import batch
    batch.add_arguments(ap)
    args = ap.parse_args(argv)
    if args.self_test:
        from . import selftest
        return selftest.run()
    if batch.wanted(args):
        return batch.run(args)

    if args.software_3d:
        os.environ['VANTAGE_SOFTWARE_3D'] = '1'
    app, settings = make_app(args.theme, args.scale)
    from .ui.viewport3d import gl_available

    from .core import samples
    from .core.project import Project
    from .ui.mainwindow import MainWindow
    soft = not gl_available()

    project = None
    model_path = None
    if args.sample:
        project = samples.tigor_sample_project()
    elif args.template:
        project = samples.template_project(args.template)
    elif args.path:
        if args.path.lower().endswith('.vantage'):
            project = Project.load(args.path)
        else:
            project = samples.blank_project()
            model_path = args.path
    elif not args.no_start:
        from .ui.dialogs.start import StartWindow
        sw = StartWindow(settings)
        if not sw.exec() or not sw.choice:
            return 0
        kind, val = sw.choice
        if kind == 'open':
            try:
                project = Project.load(val)
            except Exception as e:
                from PyQt6.QtWidgets import QMessageBox
                QMessageBox.critical(None, 'Could not open project', str(e))
                project = samples.blank_project()
        elif kind == 'template':
            project = samples.template_project(val)
        elif kind == 'model':
            project = samples.blank_project()
            model_path = val
        else:
            project = samples.blank_project()
    if project is None:
        project = samples.tigor_sample_project()
    w = MainWindow(project, soft_3d=soft)
    st = settings.value('window/state')
    w.show()
    if model_path:
        from PyQt6.QtCore import QTimer
        QTimer.singleShot(100, lambda: w.import_vehicle(path=model_path))
    if project.path:
        w.add_recent(project.path)
    return app.exec()


if __name__ == '__main__':
    sys.exit(main())
