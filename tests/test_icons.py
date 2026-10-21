"""Icon set checks: every required icon exists, parses, renders, and follows the markup rules."""
import os
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from vantage.ui.icon_data import ICONS  # noqa: E402

REQUIRED = [
    # tools
    "select", "move", "rotate", "place", "measure",
    # cameras / optics
    "camera-wide", "camera-narrow", "frustum", "lens", "sensor", "through",
    # vehicle
    "car-top", "car-side", "car-front", "axle", "ground", "mount-zone", "mirror-mount", "bumper-mount", "mesh",
    # analysis
    "roi", "blind", "seam", "coverage", "redundancy", "density", "occlusion", "bev", "bowl", "tolerance",
    "optimise", "compare",
    # editing
    "mirror", "link", "unlink", "lock", "unlock", "eye", "eye-off", "isolate", "snap-surface", "snap-grid",
    # hardware
    "cable", "ecu", "sync",
    # calibration / photos
    "checker", "photo-spin", "play", "pause", "step", "step-back",
    # view
    "split-1", "split-2", "split-4", "maximise", "frame", "cube", "axes",
    # output / status
    "export", "import", "report", "good", "warning", "serious", "critical",
    # chrome
    "chev-r", "chev-d", "chev-u", "chev-l", "caret", "close", "float", "more", "search", "plus", "minus",
    "check", "check-inv", "dash", "layers", "shading", "history", "folder", "file", "refresh", "stop", "pin",
    "filter", "cylinder", "pole", "gear", "trash", "copy", "undo", "redo", "keyboard", "help", "drag", "target",
    # templates
    "sedan", "hatchback", "suv", "van", "truck",
    # app
    "app",
]

ALLOWED_TAGS = {"path", "line", "rect", "circle", "ellipse", "polyline", "polygon"}

WRAP = ('<svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 16 16" fill="none" '
        'stroke="currentColor" stroke-width="1.25" stroke-linecap="round" stroke-linejoin="round">{inner}</svg>')


def _svg(name, colour="#f0efec"):
    return WRAP.format(inner=ICONS[name]).replace("currentColor", colour).encode()


def test_required_names_present():
    missing = [n for n in REQUIRED if n not in ICONS]
    assert not missing, f"missing icons: {missing}"


def test_values_are_nonempty_strings():
    for name, inner in ICONS.items():
        assert isinstance(name, str) and isinstance(inner, str) and inner.strip(), name


@pytest.mark.parametrize("name", sorted(ICONS))
def test_markup_rules(name):
    inner = ICONS[name]
    root = ET.fromstring(f'<g xmlns="http://www.w3.org/2000/svg">{inner}</g>')  # well-formed XML
    elements = list(root)
    assert elements, name
    for el in elements:
        tag = el.tag.rsplit("}", 1)[-1]
        assert tag in ALLOWED_TAGS, f"{name}: <{tag}> not allowed"
        assert len(el) == 0, f"{name}: nested elements not allowed"
        for attr in ("fill", "stroke"):
            if attr in el.attrib:
                assert el.attrib[attr] in ("currentColor", "none"), f"{name}: hard-coded {attr}"
    assert "#" not in inner, f"{name}: hard-coded colour"


@pytest.fixture(scope="module")
def qt_app():
    QtGui = pytest.importorskip("PyQt6.QtGui")
    pytest.importorskip("PyQt6.QtSvg")
    app = QtGui.QGuiApplication.instance() or QtGui.QGuiApplication([])
    yield app


@pytest.mark.parametrize("name", sorted(ICONS))
def test_parses_and_renders(qt_app, name):
    from PyQt6.QtCore import QByteArray, QRectF, Qt
    from PyQt6.QtGui import QImage, QPainter
    from PyQt6.QtSvg import QSvgRenderer

    renderer = QSvgRenderer(QByteArray(_svg(name)))
    assert renderer.isValid(), f"{name}: QSvgRenderer could not parse"

    img = QImage(16, 16, QImage.Format.Format_ARGB32_Premultiplied)
    img.fill(Qt.GlobalColor.transparent)
    painter = QPainter(img)
    painter.setRenderHint(QPainter.RenderHint.Antialiasing)
    renderer.render(painter, QRectF(0, 0, 16, 16))
    painter.end()
    lit = sum(1 for y in range(16) for x in range(16) if img.pixelColor(x, y).alpha() > 0)
    assert lit > 0, f"{name}: renders empty at 16 px"
