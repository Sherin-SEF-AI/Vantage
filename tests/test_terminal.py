"""The Terminal panel (vantage.ui.terminal): keys as an xterm sends them, a real shell drawn through pyte, and Claude
started there working on its own window (VANTAGE_MCP_PID)."""
import json
import os
import time

import pytest
from PyQt6.QtCore import Qt

from vantage.ui import terminal as TM

K, M = Qt.Key, Qt.KeyboardModifier


def test_keys_are_what_an_xterm_sends():
    none = M.NoModifier
    assert TM.key_bytes(K.Key_Up, none, '') == b'\x1b[A'
    assert TM.key_bytes(K.Key_Up, none, '', app_cursor=True) == b'\x1bOA'          # DECCKM
    assert TM.key_bytes(K.Key_Left, M.ControlModifier, '') == b'\x1b[1;5D'
    assert TM.key_bytes(K.Key_C, M.ControlModifier, '\x03') == b'\x03'
    assert TM.key_bytes(K.Key_Return, none, '\r') == b'\r'
    assert TM.key_bytes(K.Key_Return, M.ShiftModifier, '\r') == b'\x1b\r'            # a new line in Claude's prompt
    assert TM.key_bytes(K.Key_Backspace, none, '\x08') == b'\x7f'
    assert TM.key_bytes(K.Key_Backtab, M.ShiftModifier, '') == b'\x1b[Z'
    assert TM.key_bytes(K.Key_B, M.AltModifier, 'b') == b'\x1bb'
    assert TM.key_bytes(K.Key_unknown, none, 'é') == 'é'.encode()
    assert TM.key_bytes(K.Key_Shift, M.ShiftModifier, '') == b''


def test_claude_in_a_windows_terminal_works_on_that_window(tmp_path, monkeypatch):
    from vantage.mcp import session as SE
    from vantage.mcp.server import Router
    monkeypatch.setenv('XDG_CONFIG_HOME', str(tmp_path))
    os.makedirs(SE.bridge_dir())
    for pid, started in ((os.getpid(), 1), (os.getppid(), 2)):                      # ours is the older one
        with open(os.path.join(SE.bridge_dir(), f'{pid}.json'), 'w') as f:
            json.dump({'pid': pid, 'port': 1, 'token': 't', 'path': '', 'started': started}, f)
    assert Router().session().pid == os.getppid()                                    # the newest, by default
    monkeypatch.setenv('VANTAGE_MCP_PID', str(os.getpid()))
    assert Router().session().pid == os.getpid()                                     # the window it started in


def _pump(app, s):
    from PyQt6.QtCore import QEventLoop
    t = time.time()
    while time.time() - t < s:
        app.processEvents(QEventLoop.ProcessEventsFlag.AllEvents, 20)
        time.sleep(0.005)


def _wait(app, cond, s=20):
    t = time.time()
    while not cond() and time.time() - t < s:
        _pump(app, 0.05)
    return cond()


@pytest.mark.skipif(bool(TM.available()), reason=TM.available() or 'available')
def test_a_real_shell_runs_and_is_drawn(tmp_path):
    from PyQt6.QtWidgets import QApplication
    app = QApplication.instance() or QApplication([])
    env = dict(os.environ, PS1='$ ', TERM='xterm-256color', VANTAGE_MCP_PID='4242', HOME=str(tmp_path))
    v = TM.TerminalView(str(tmp_path), env, argv=['/bin/sh'])
    v.resize(800, 300)
    v.show()
    assert _wait(app, lambda: (v.rows, v.cols) != (24, 80) or True)
    v.send(b'printf "hello \\033[31mred\\033[0m %s\\n" "$VANTAGE_MCP_PID"; pwd\r')
    assert _wait(app, lambda: 'hello red 4242' in v.screen_text()), v.screen_text()
    assert str(tmp_path) in v.screen_text()                                           # it starts in the folder
    row = next(r for r in range(v.rows) if 'hello red' in ''.join(v.screen.buffer[r][c].data for c in range(v.cols)))
    line = v.screen.buffer[row]
    text = ''.join(line[c].data for c in range(v.cols))
    at = text.index('hello red') + 6                                                  # (the prompt may come first)
    assert [line[c].fg for c in range(at, at + 3)] == ['red'] * 3                    # colours arrive as attributes
    assert line[at - 2].fg == 'default' and line[at + 4].fg == 'default'
    v.grab()                                                                          # paints without error
    v.send(b'stty size\r')
    assert _wait(app, lambda: f'{v.rows} {v.cols}' in v.screen_text())               # the shell knows the size
    from PyQt6.QtTest import QTest
    v.setFocus()
    QTest.keyClicks(v, 'echo typed-by-keys')                                         # real key presses, not send()
    QTest.keyClick(v, K.Key_Return)
    assert _wait(app, lambda: v.screen_text().count('typed-by-keys') >= 2), v.screen_text()
    v.send(b'sleep 30; echo slept\r')
    _pump(app, 0.5)
    t0 = time.time()
    v.send(b'\x03')                                                                 # Ctrl+C: its own terminal
    v.send(b'echo after-ctrl-c\r')
    assert _wait(app, lambda: v.screen_text().count('after-ctrl-c') >= 2, 10) and time.time() - t0 < 10
    assert 'slept' not in v.screen_text().replace('echo slept', '')
    v.send(b'exit\r')
    assert _wait(app, lambda: v.ended)
    v.close_shell()


@pytest.mark.skipif(bool(TM.available()), reason=TM.available() or 'available')
def test_the_panel_opens_with_the_window_and_starts_claude_there(tmp_path, monkeypatch):
    from PyQt6.QtWidgets import QApplication
    from vantage.core import samples
    from vantage.ui.mainwindow import MainWindow
    app = QApplication.instance() or QApplication([])
    w = MainWindow(samples.tigor_sample_project(), soft_3d=True)
    w.show()
    assert _wait(app, lambda: w.ctx.analysis.result is not None, 120)
    w.toggle_terminal()
    pn = w.terminal_panel
    assert w.docks['terminal'].isVisible() and pn.view is not None and not pn.view.ended
    assert pn.env()['VANTAGE_MCP_PID'] == str(os.getpid()) and pn.env()['TERM'] == 'xterm-256color'
    sent = []
    monkeypatch.setattr(pn.view, 'send', lambda b: sent.append(b))
    monkeypatch.setattr(TM.shutil, 'which', lambda *a, **k: '/usr/bin/claude')
    pn.start_claude()
    assert sent == [b'claude\r']
    assert _wait(app, lambda: not (w.ctx.analysis._busy or w.ctx.analysis.timer.isActive()), 120)
    _pump(app, 0.3)                                    # never close a window while its analysis still emits
    w.store.undo.setClean()
    w.close()
    assert pn.view.ended                                                              # the shell ends with the window
    w.deleteLater()
    _pump(app, 0.2)


def test_without_pyte_the_panel_says_what_it_needs(monkeypatch):
    from PyQt6.QtWidgets import QApplication
    QApplication.instance() or QApplication([])
    monkeypatch.setattr(TM, 'pyte', None)
    assert 'pip install' in TM.available() or not TM.POSIX

    class Ctx:
        settings = None
    pn = TM.TerminalPanel(Ctx())
    assert pn.note is not None and not pn.b_claude.isEnabled() and pn.ensure() is None
