"""Vantage entry point."""
from __future__ import annotations

import argparse
import os
import sys


def main(argv=None):
    ap = argparse.ArgumentParser(prog='vantage', description='Vantage · surround-view camera planner')
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
    if batch.wanted(args):
        return batch.run(args)

    if args.software_3d:
        os.environ['VANTAGE_SOFTWARE_3D'] = '1'
    from PyQt6.QtCore import Qt, QSettings
    from PyQt6.QtWidgets import QApplication
    from .ui.viewport3d import configure_default_format, gl_available
    configure_default_format()
    QApplication.setHighDpiScaleFactorRoundingPolicy(Qt.HighDpiScaleFactorRoundingPolicy.PassThrough)
    app = QApplication(sys.argv[:1])
    app.setApplicationName('Vantage')
    app.setOrganizationName('Vantage')
    app.setApplicationDisplayName('Vantage')
    from .ui.theme import THEME, load_fonts
    from .ui import icons
    load_fonts()
    settings = QSettings('Vantage', 'Vantage')
    theme = args.theme or settings.value('ui/theme', 'dark')
    scale = args.scale or float(settings.value('ui/scale', 1.0))
    THEME.apply(app, theme, scale)
    app.setWindowIcon(icons.icon('app', 16, THEME.tok['ink']))

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
