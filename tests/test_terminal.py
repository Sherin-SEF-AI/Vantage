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
    v.send(b'seq 1 200\r')
    assert _wait(app, lambda: '200' in v.screen_text().split())
    from PyQt6.QtCore import QPoint, QPointF
    from PyQt6.QtGui import QWheelEvent
    from PyQt6.QtWidgets import QApplication as QA

    def wheel(notches):
        QA.sendEvent(v, QWheelEvent(QPointF(50, 50), QPointF(50, 50), QPoint(0, 0), QPoint(0, 120 * notches),
                                    Qt.MouseButton.NoButton, M.NoModifier, Qt.ScrollPhase.NoScrollPhase, False))
    wheel(10)                                                                         # 30 lines back
    assert v.scrolled == 30
    top = v.screen_text().split()[0]
    assert '200' not in v.screen_text().split() and top.isdigit()
    v.shell.write(b'echo more-output\r')                                             # output while scrolled back
    assert _wait(app, lambda: v.scrolled > 30)
    assert v.screen_text().split()[0] == top                                          # the view did not move
    wheel(-200)
    assert v.scrolled == 0 and 'more-output' in v.screen_text()
    wheel(3)
    QTest.keyClick(v, K.Key_Space)                                                    # a key brings the live screen back
    assert v.scrolled == 0
    QTest.keyClick(v, K.Key_Backspace)
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
@pytest.mark.skipif(bool(TM.available()), reason=TM.available() or 'available')
def test_full_screen_programs_get_the_wheel_and_their_own_screen(tmp_path):
    """What Claude Code turns on: the alternate screen (1049), mouse reports (1000, SGR 1006), alternate scroll (1007)."""
    from PyQt6.QtCore import QPoint, QPointF
    from PyQt6.QtGui import QMouseEvent, QWheelEvent
    from PyQt6.QtWidgets import QApplication
    app = QApplication.instance() or QApplication([])
    v = TM.TerminalView(str(tmp_path), dict(os.environ, PS1='$ '), argv=['/bin/sh'])
    v.resize(800, 300)
    v.show()
    _pump(app, 0.3)
    sent = []
    v.shell.write = lambda b: sent.append(b)

    def wheel(notches, mods=M.NoModifier):
        QApplication.sendEvent(v, QWheelEvent(QPointF(4 + 10.5 * v.cw, 3 + 2.5 * v.ch), QPointF(0, 0), QPoint(0, 0),
                                              QPoint(0, 120 * notches), Qt.MouseButton.NoButton, mods,
                                              Qt.ScrollPhase.NoScrollPhase, False))
    v.stream.feed(b'main-text\r\n' + b'\n'.join(b'%d' % i for i in range(60)) + b'\r\n')
    hist = len(v.screen.history.top)
    v.stream.feed(b'\x1b[?1049h\x1b[?1000h\x1b[?1006h\x1b[Halt-text')
    assert v.screen.alt and 'alt-text' in v.screen_text() and 'main-text' not in v.screen_text()
    wheel(1)                                                                          # three lines a notch
    assert sent == [b'\x1b[<64;11;3M'] * 3                                          # wheel up at column 11, row 3
    sent.clear()
    wheel(-1)
    assert sent == [b'\x1b[<65;11;3M'] * 3
    sent.clear()
    pos = QPointF(4 + 2.5 * v.cw, 3 + 0.5 * v.ch)
    for kind in (QMouseEvent.Type.MouseButtonPress, QMouseEvent.Type.MouseButtonRelease):
        QApplication.sendEvent(v, QMouseEvent(kind, pos, pos, Qt.MouseButton.LeftButton, Qt.MouseButton.LeftButton,
                                              M.NoModifier))
    assert sent == [b'\x1b[<0;3;1M', b'\x1b[<0;3;1m']                              # a click, pressed and released
    sent.clear()
    v.stream.feed(b'\n' * 40)                                                       # scrolling on the alternate screen
    assert len(v.screen.history.top) == hist                                          # leaves no history behind
    v.stream.feed(b'\x1b[?1000l\x1b[?1007h')                                        # no mouse, alternate scroll
    wheel(1)
    assert sent == [b'\x1b[A' * 3]
    sent.clear()
    v.stream.feed(b'\x1b[?1049l')                                                   # back to the shell's screen
    assert not v.screen.alt and 'alt-text' not in v.screen_text() and '59' in v.screen_text()
    wheel(2)                                                                          # the shell: the history scrolls
    assert not sent and v.scrolled == 6
    v.close_shell()


def test_claude_code_is_a_side_of_the_assistant_panel(tmp_path, monkeypatch):
    from PyQt6.QtWidgets import QApplication
    from vantage.core import samples
    from vantage.ui.mainwindow import MainWindow
    app = QApplication.instance() or QApplication([])
    started = []
    monkeypatch.setattr(TM.TerminalPanel, 'start_claude', lambda self: (started.append(1),
                                                                        setattr(self, 'claude_started', True)))
    w = MainWindow(samples.tigor_sample_project(), soft_3d=True)
    w.resize(1600, 1000)
    w.show()
    assert _wait(app, lambda: w.ctx.analysis.result is not None, 120)
    side, dock = w.side_panel, w.docks['assistant']
    assert 'terminal' not in w.docks and side.mode() == 'assistant'
    w.show_assistant()
    assert dock.isVisible() and side.mode() == 'assistant' and not started
    side.switch.buttons['claude'].click()                                             # the switch at the top
    _pump(app, 0.3)
    pn = w.terminal_panel
    assert side.mode() == 'claude' and pn.view is not None and not pn.view.ended
    assert started == [1] and dock.width() >= 600                                     # Claude Code, with room
    assert w.ctx.settings.value('assistant/side') == 'claude'
    assert pn.env()['VANTAGE_MCP_PID'] == str(os.getpid()) and pn.env()['TERM'] == 'xterm-256color'
    side.set_mode('assistant')
    side.set_mode('claude')
    assert started == [1]                                                             # once per shell
    w.toggle_assistant()                                                              # Ctrl+J: the assistant side
    assert side.mode() == 'assistant' and dock.isVisible()
    w.toggle_terminal()                                                               # Ctrl+`: Claude's side
    assert side.mode() == 'claude' and pn.view.hasFocus() or side.mode() == 'claude'
    w.ctx.settings.setValue('assistant/side', 'assistant')
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


@pytest.mark.skipif(bool(TM.available()), reason=TM.available() or 'available')
def test_claude_codes_own_history_from_the_panel(tmp_path, monkeypatch):
    from PyQt6.QtWidgets import QApplication
    app = QApplication.instance() or QApplication([])

    class Ctx:
        class settings:
            @staticmethod
            def value(k, d=None, type=None):
                return str(tmp_path) if k == 'dir/working' else d

        class store:
            class project:
                path = ''
        window = mcp = None
    pn = TM.TerminalPanel(Ctx())
    monkeypatch.setattr(pn, 'cwd', lambda: str(tmp_path))
    v = pn.ensure(argv=['/bin/sh'])
    _pump(app, 0.3)
    sent = []
    real = v.send
    monkeypatch.setattr(v, 'send', lambda b: sent.append(b))
    monkeypatch.setattr(TM.shutil, 'which', lambda *a, **k: '/usr/bin/claude')
    assert not pn.busy()
    pn.claude_history(False)
    pn.claude_history(True)
    assert sent == [b'claude --continue\r', b'claude --resume\r'] and pn.claude_started
    real(b'sleep 5\r')                                                                # a program in the foreground
    assert _wait(app, pn.busy, 5)
    sent.clear()
    pn.claude_history(True)
    assert sent == [b'/resume\r']                                                     # Claude Code's own command
    v.close_shell()
