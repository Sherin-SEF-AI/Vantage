"""One QApplication for the whole session: Qt objects created while an earlier application was alive (the theme
manager, cached icons) do not survive that application being destroyed between test modules. Settings go to a
throw-away config folder."""
import os
import tempfile

os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')
# never read or write the user's real settings (~/.config/Vantage) from a test
os.environ['XDG_CONFIG_HOME'] = tempfile.mkdtemp(prefix='vantage-test-cfg-')
# the assistant: never the system keyring, never a real key from the environment (tests talk to tests/fake_groq.py)
os.environ['VANTAGE_NO_KEYRING'] = '1'
os.environ.pop('GROQ_API_KEY', None)
os.environ.pop('VANTAGE_ASSISTANT_URL', None)

import pytest


@pytest.fixture(scope='session', autouse=True)
def _qt_app():
    from PyQt6.QtWidgets import QApplication
    app = QApplication.instance() or QApplication([])
    yield app
