"""One QApplication for the whole session: Qt objects created while an earlier application was alive (the theme
manager, cached icons) do not survive that application being destroyed between test modules."""
import os

os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')

import pytest


@pytest.fixture(scope='session', autouse=True)
def _qt_app():
    from PyQt6.QtWidgets import QApplication
    app = QApplication.instance() or QApplication([])
    yield app
