"""The Terminal panel (Window ▸ Terminal, Ctrl+`): the user's shell inside the window, for Claude Code above all.

A real shell runs on a pseudo-terminal (Linux, macOS) in the project's folder; pyte (optional, LGPL-3.0: `pip install
pyte`) keeps the screen the shell draws, and TerminalView paints it in the theme's colours. Claude started here (the
Claude button, or `claude` typed) finds this window's MCP bridge first (VANTAGE_MCP_PID), so its changes land in
this window as undo steps while the conversation stays beside the 3D view.

Keys go to the shell while the terminal has focus (the app's shortcuts too: Ctrl+Z is the shell's here), except
Ctrl+Shift+C / Ctrl+Shift+V (copy, paste), Shift+PageUp/PageDown (scrollback) and Ctrl+` (back to the window).
Windows has no pseudo-terminal in Python's standard library: the panel says so there."""
from __future__ import annotations

import os
import shutil
import signal
import struct
import sys

from PyQt6.QtCore import QRect, QSocketNotifier, Qt, QTimer, pyqtSignal
from PyQt6.QtGui import QColor, QFontMetricsF, QGuiApplication, QPainter
from PyQt6.QtWidgets import QHBoxLayout, QLabel, QVBoxLayout, QWidget

from .theme import THEME, mono
from .widgets import button, label

try:
    import pyte
except ImportError:                                  # optional: the panel explains how to get it
    pyte = None

POSIX = os.name == 'posix'
HISTORY = 5000
NAMES = ('black', 'red', 'green', 'brown', 'blue', 'magenta', 'cyan', 'white')
ANSI = {   # theme -> 16 colours (normal, then bright), readable on that theme's panel colour
    'dark': ['#1a1a19', '#e5534b', '#57ab5a', '#c69026', '#539bf5', '#b083f0', '#39c5cf', '#c3c2b7',
             '#636e7b', '#ff938a', '#6bc46d', '#daaa3f', '#6cb6ff', '#dcbdfb', '#56d4dd', '#f0efec'],
    'light': ['#0b0b0b', '#c4314b', '#1a7f37', '#9a6700', '#0969da', '#8250df', '#1b7c83', '#6e7781',
              '#57606a', '#e0525e', '#2da44e', '#bf8700', '#218bff', '#a475f9', '#3192aa', '#8c959f']}
# keys without text: what an xterm sends (cursor keys change with DECCKM, the application cursor mode)
KEYS = {Qt.Key.Key_Home: b'\x1b[H', Qt.Key.Key_End: b'\x1b[F', Qt.Key.Key_PageUp: b'\x1b[5~',
        Qt.Key.Key_PageDown: b'\x1b[6~', Qt.Key.Key_Insert: b'\x1b[2~', Qt.Key.Key_Delete: b'\x1b[3~',
        Qt.Key.Key_Backtab: b'\x1b[Z', Qt.Key.Key_Escape: b'\x1b', Qt.Key.Key_Backspace: b'\x7f',
        Qt.Key.Key_Return: b'\r', Qt.Key.Key_Enter: b'\r', Qt.Key.Key_Tab: b'\t',
        Qt.Key.Key_F1: b'\x1bOP', Qt.Key.Key_F2: b'\x1bOQ', Qt.Key.Key_F3: b'\x1bOR', Qt.Key.Key_F4: b'\x1bOS',
        Qt.Key.Key_F5: b'\x1b[15~', Qt.Key.Key_F6: b'\x1b[17~', Qt.Key.Key_F7: b'\x1b[18~',
        Qt.Key.Key_F8: b'\x1b[19~', Qt.Key.Key_F9: b'\x1b[20~', Qt.Key.Key_F10: b'\x1b[21~',
        Qt.Key.Key_F11: b'\x1b[23~', Qt.Key.Key_F12: b'\x1b[24~'}
ARROWS = {Qt.Key.Key_Up: b'A', Qt.Key.Key_Down: b'B', Qt.Key.Key_Right: b'C', Qt.Key.Key_Left: b'D'}
BRACKETED_PASTE = 2004 << 5                          # pyte keeps private modes shifted by 5
DECCKM = 1 << 5                                      # application cursor keys (pyte 0.8.0 has no name for it)
# box drawing, drawn as lines through the whole cell (a font's glyph leaves gaps between rows): the arms each
# character has (left, right, up, down), heavy or not; rounded corners are drawn as arcs
_L, _R, _U, _D = 1, 2, 4, 8
BOX = {'─': _L | _R, '│': _U | _D, '┌': _R | _D, '┐': _L | _D, '└': _R | _U, '┘': _L | _U, '├': _U | _D | _R,
       '┤': _U | _D | _L, '┬': _L | _R | _D, '┴': _L | _R | _U, '┼': _L | _R | _U | _D, '╴': _L, '╶': _R, '╵': _U,
       '╷': _D, '╭': _R | _D, '╮': _L | _D, '╰': _R | _U, '╯': _L | _U}
HEAVY = {'━': _L | _R, '┃': _U | _D, '┏': _R | _D, '┓': _L | _D, '┗': _R | _U, '┛': _L | _U}
ROUND = '╭╮╰╯'


def available() -> str:
    """'' when the terminal can run here, else why not."""
    if not POSIX:
        return 'The terminal panel needs a pseudo-terminal: Linux or macOS for now.'
    if pyte is None:
        return 'The terminal panel needs the pyte package: pip install --user pyte (then reopen the panel).'
    return ''


def key_bytes(key, mods, text, app_cursor=False) -> bytes:
    """What a key press sends to the shell (an xterm's sequences); b'' for none."""
    ctrl = bool(mods & Qt.KeyboardModifier.ControlModifier)
    alt = bool(mods & Qt.KeyboardModifier.AltModifier)
    shift = bool(mods & Qt.KeyboardModifier.ShiftModifier)
    if key in ARROWS:
        n = 1 + (1 if shift else 0) + (2 if alt else 0) + (4 if ctrl else 0)
        if n > 1:
            return b'\x1b[1;%d' % n + ARROWS[key]
        return (b'\x1bO' if app_cursor else b'\x1b[') + ARROWS[key]
    if key in (Qt.Key.Key_Return, Qt.Key.Key_Enter) and (shift or alt):
        return b'\x1b\r'                              # a new line in Claude Code's prompt, not send
    if key in KEYS:
        out = KEYS[key]
        return b'\x1b' + out if alt and key in (Qt.Key.Key_Backspace,) else out
    if ctrl and not alt:
        if Qt.Key.Key_A <= key <= Qt.Key.Key_Z:
            return bytes([key - Qt.Key.Key_A + 1])
        special = {Qt.Key.Key_Space: b'\x00', Qt.Key.Key_BracketLeft: b'\x1b', Qt.Key.Key_Backslash: b'\x1c',
                   Qt.Key.Key_BracketRight: b'\x1d', Qt.Key.Key_AsciiCircum: b'\x1e', Qt.Key.Key_Underscore: b'\x1f',
                   Qt.Key.Key_Slash: b'\x1f'}
        if key in special:
            return special[key]
    if text:
        b = text.encode('utf-8')
        return b'\x1b' + b if alt else b
    return b''


class Shell:
    """A shell on a pseudo-terminal: bytes in and out, its size, whether it still runs."""

    def __init__(self, cwd: str, env: dict, argv=None):
        self.argv = argv or [os.environ.get('SHELL') or shutil.which('bash') or '/bin/sh']
        if sys.platform.startswith('linux'):
            self._spawn(cwd, env)
        else:
            import pty
            self.pid, self.fd = pty.fork()
            if self.pid == 0:                         # the child: become the shell
                try:
                    os.chdir(cwd)
                except OSError:
                    pass
                os.execvpe(self.argv[0], self.argv, env)
        os.set_blocking(self.fd, False)

    def _spawn(self, cwd, env):
        """posix_spawn, not fork: the app has many threads, and a forked child of a threaded process may hang before
        it execs. In a new session the shell opens the terminal by name, which makes it its controlling terminal
        (Ctrl+C, job control); a small sh changes to the folder and execs the shell."""
        self.fd, slave = os.openpty()
        name = os.ttyname(slave)
        os.close(slave)
        acts = [(os.POSIX_SPAWN_OPEN, 0, name, os.O_RDWR, 0), (os.POSIX_SPAWN_DUP2, 0, 1),
                (os.POSIX_SPAWN_DUP2, 0, 2)]
        argv = ['/bin/sh', '-c', 'cd -- "$0" 2>/dev/null; exec "$@"', cwd, *self.argv]
        self.pid = os.posix_spawn('/bin/sh', argv, env, file_actions=acts, setsid=True)

    def read(self) -> bytes:
        try:
            return os.read(self.fd, 1 << 16)
        except BlockingIOError:
            return b''
        except OSError:
            return b''                                # the shell has gone (EIO)

    def write(self, data: bytes):
        while data:
            try:
                n = os.write(self.fd, data)
            except BlockingIOError:
                import select
                select.select([], [self.fd], [], 1.0)
                continue
            except OSError:
                return
            data = data[n:]

    def resize(self, rows, cols):
        import fcntl
        import termios
        try:
            fcntl.ioctl(self.fd, termios.TIOCSWINSZ, struct.pack('HHHH', rows, cols, 0, 0))
        except OSError:
            pass

    def alive(self) -> bool:
        try:
            pid, _ = os.waitpid(self.pid, os.WNOHANG)
        except ChildProcessError:
            return False
        return pid == 0

    def close(self):
        if self.alive():
            try:
                os.kill(self.pid, signal.SIGHUP)
            except OSError:
                pass
        try:
            os.close(self.fd)
        except OSError:
            pass
        try:
            os.waitpid(self.pid, os.WNOHANG)
        except ChildProcessError:
            pass


class _Screen(pyte.HistoryScreen if pyte is not None else object):
    """pyte's screen with a way back to the shell for the terminal's answers (cursor position, device attributes)."""

    def __init__(self, cols, rows, reply):
        super().__init__(cols, rows, history=HISTORY, ratio=0.5)
        self._reply = reply

    def write_process_input(self, data):
        self._reply(data.encode('utf-8') if isinstance(data, str) else data)


class TerminalView(QWidget):
    """Paints a pyte screen and sends keys, pastes and resizes to the shell."""
    exited = pyqtSignal()                              # the shell ended
    restart = pyqtSignal()                             # Enter after it ended: start another
    title = pyqtSignal(str)

    def __init__(self, cwd: str, env: dict, parent=None, argv=None):
        super().__init__(parent)
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        self.setAttribute(Qt.WidgetAttribute.WA_OpaquePaintEvent)
        self.setAttribute(Qt.WidgetAttribute.WA_InputMethodEnabled)
        self.setCursor(Qt.CursorShape.IBeamCursor)
        self.font_px = 13
        self._metrics()
        self.rows, self.cols = 24, 80
        self.screen = _Screen(self.cols, self.rows, self._reply)
        self.stream = pyte.ByteStream(self.screen)
        self.shell = Shell(cwd, env, argv)
        self.shell.resize(self.rows, self.cols)
        self.notifier = QSocketNotifier(self.shell.fd, QSocketNotifier.Type.Read, self)
        self.notifier.activated.connect(self._readable)
        self._paint = QTimer(self)                     # a burst of output is drawn once, not once per chunk
        self._paint.setSingleShot(True)
        self._paint.setInterval(12)
        self._paint.timeout.connect(self.update)
        self.scrolled = 0                              # lines scrolled back into the history
        self.sel = None                                # ((row, col), (row, col)) in screen cells
        self._anchor = None
        self.ended = False
        THEME.changed.connect(lambda *_: self.update())

    # ---------------------------------------------------------------- geometry
    def _metrics(self):
        self.font = mono(self.font_px)
        fm = QFontMetricsF(self.font)
        self.cw = max(1.0, fm.horizontalAdvance('M'))
        self.ch = max(1.0, fm.height())
        self.ascent = fm.ascent()

    def set_font_px(self, px):
        self.font_px = max(8, min(28, int(px)))
        self._metrics()
        self._fit()
        self.update()

    def resizeEvent(self, e):
        super().resizeEvent(e)
        self._fit()

    def _fit(self):
        cols = max(20, int((self.width() - 8) / self.cw))
        rows = max(4, int((self.height() - 6) / self.ch))
        if (rows, cols) != (self.rows, self.cols):
            self.rows, self.cols = rows, cols
            self.screen.resize(rows, cols)
            self.shell.resize(rows, cols)
            self.update()

    # ---------------------------------------------------------------- the shell
    def _reply(self, data: bytes):
        self.shell.write(data)

    def send(self, data: bytes):
        if self.ended or not data:
            return
        if self.scrolled:
            self._scroll_to(0)
        self.sel = None
        self.shell.write(data)

    def _readable(self, *_):
        data = self.shell.read()
        if data:
            self.stream.feed(data)
            if self.screen.title:
                self.title.emit(self.screen.title)
            if not self._paint.isActive():
                self._paint.start()
            return
        if not self.shell.alive():
            self.notifier.setEnabled(False)
            self.ended = True
            self.update()
            self.exited.emit()

    def close_shell(self):
        self.notifier.setEnabled(False)
        self.shell.close()
        self.ended = True

    # ---------------------------------------------------------------- scrollback
    def _scroll_to(self, n):
        n = max(0, n)
        while self.scrolled < n and self.screen.history.top:
            self.screen.prev_page()
            self.scrolled += 1
        while self.scrolled > n:
            self.screen.next_page()
            self.scrolled -= 1
        self.update()

    def scroll_lines(self, pages: int):
        """Shift+PageUp/PageDown and the wheel: whole half-pages back or forward through the history."""
        self._scroll_to(self.scrolled + pages)

    def wheelEvent(self, e):
        steps = e.angleDelta().y() / 120
        if steps:
            self.scroll_lines(int(round(steps)) or (1 if steps > 0 else -1))

    # ---------------------------------------------------------------- keys
    def event(self, e):
        from PyQt6.QtCore import QEvent
        if e.type() == QEvent.Type.ShortcutOverride:   # the app's shortcuts do not steal the shell's keys
            mods = e.modifiers()
            ctrl_shift = Qt.KeyboardModifier.ControlModifier | Qt.KeyboardModifier.ShiftModifier
            if e.key() == Qt.Key.Key_QuoteLeft and mods & Qt.KeyboardModifier.ControlModifier:
                return super().event(e)                # Ctrl+`: the window's toggle
            if (mods & ctrl_shift) == ctrl_shift and e.key() in (Qt.Key.Key_C, Qt.Key.Key_V):
                e.accept()
                return True
            e.accept()
            return True
        if e.type() == QEvent.Type.KeyPress and e.key() in (Qt.Key.Key_Tab, Qt.Key.Key_Backtab):
            self.keyPressEvent(e)                      # never focus-chain navigation
            return True
        return super().event(e)

    def keyPressEvent(self, e):
        mods, key = e.modifiers(), e.key()
        ctrl_shift = Qt.KeyboardModifier.ControlModifier | Qt.KeyboardModifier.ShiftModifier
        if (mods & ctrl_shift) == ctrl_shift and key == Qt.Key.Key_C:
            return self.copy()
        if (mods & ctrl_shift) == ctrl_shift and key == Qt.Key.Key_V:
            return self.paste()
        if mods & Qt.KeyboardModifier.ShiftModifier and key in (Qt.Key.Key_PageUp, Qt.Key.Key_PageDown):
            return self.scroll_lines(1 if key == Qt.Key.Key_PageUp else -1)
        if self.ended and key in (Qt.Key.Key_Return, Qt.Key.Key_Enter):
            self.restart.emit()
            return
        app_cursor = DECCKM in self.screen.mode
        self.send(key_bytes(key, mods, e.text(), app_cursor))

    def inputMethodEvent(self, e):
        if e.commitString():
            self.send(e.commitString().encode('utf-8'))

    # ---------------------------------------------------------------- copy and paste
    def paste(self):
        text = QGuiApplication.clipboard().text()
        if not text:
            return
        data = text.replace('\r\n', '\r').replace('\n', '\r').encode('utf-8')
        if BRACKETED_PASTE in self.screen.mode:
            data = b'\x1b[200~' + data + b'\x1b[201~'
        self.send(data)

    def selected_text(self) -> str:
        if self.sel is None:
            return ''
        (r0, c0), (r1, c1) = sorted(self.sel)
        lines = []
        for r in range(r0, r1 + 1):
            row = self.screen.buffer[r]
            a = c0 if r == r0 else 0
            b = c1 if r == r1 else self.cols - 1
            lines.append(''.join(row[c].data for c in range(a, b + 1)).rstrip())
        return '\n'.join(lines)

    def copy(self):
        t = self.selected_text()
        if t:
            QGuiApplication.clipboard().setText(t)

    def screen_text(self) -> str:
        """The visible screen as text (tests, and copy of the whole screen)."""
        return '\n'.join(''.join(self.screen.buffer[r][c].data for c in range(self.cols)).rstrip()
                         for r in range(self.rows))

    def _cell(self, pos):
        return (max(0, min(self.rows - 1, int((pos.y() - 3) / self.ch))),
                max(0, min(self.cols - 1, int((pos.x() - 4) / self.cw))))

    def mousePressEvent(self, e):
        self.setFocus()
        if e.button() == Qt.MouseButton.LeftButton:
            self._anchor = self._cell(e.position())
            self.sel = None
            self.update()
        elif e.button() == Qt.MouseButton.MiddleButton:
            text = QGuiApplication.clipboard().text(QGuiApplication.clipboard().Mode.Selection)
            if text:
                self.send(text.encode('utf-8'))

    def mouseMoveEvent(self, e):
        if self._anchor is not None:
            self.sel = (self._anchor, self._cell(e.position()))
            self.update()

    def mouseReleaseEvent(self, e):
        if self.sel is not None and self.sel[0] != self.sel[1]:
            cb = QGuiApplication.clipboard()
            if cb.supportsSelection():
                cb.setText(self.selected_text(), cb.Mode.Selection)
        else:
            self.sel = None
        self._anchor = None

    # ---------------------------------------------------------------- painting
    def _colour(self, name, fg, bold=False):
        pal = ANSI['light' if THEME.name == 'light' else 'dark']
        if name == 'default':
            return QColor(THEME.tok['ink'] if fg else THEME.tok['panel'])
        if name in NAMES:
            return QColor(pal[NAMES.index(name) + (8 if bold and fg else 0)])
        if name.startswith('bright') and name[6:] in NAMES:
            return QColor(pal[8 + NAMES.index(name[6:])])
        if len(name) == 6:
            return QColor('#' + name)
        return QColor(THEME.tok['ink'] if fg else THEME.tok['panel'])

    def paintEvent(self, e):
        p = QPainter(self)
        p.fillRect(self.rect(), QColor(THEME.tok['panel']))
        p.setFont(self.font)
        sel = sorted(self.sel) if self.sel is not None else None
        selc = QColor(THEME.tok['sel'])
        buf = self.screen.buffer
        for r in range(self.rows):
            y = 3 + r * self.ch
            row = buf[r]
            c = 0
            while c < self.cols:                       # a run of cells that look alike is one drawText
                ch0 = row[c]
                attrs = (ch0.fg, ch0.bg, ch0.bold, ch0.italics, ch0.underscore, ch0.reverse, ch0.strikethrough)
                start = c
                text = []
                while c < self.cols:
                    ch = row[c]
                    if (ch.fg, ch.bg, ch.bold, ch.italics, ch.underscore, ch.reverse, ch.strikethrough) != attrs:
                        break
                    if sel is not None and self._in_sel(sel, r, c) != self._in_sel(sel, r, start):
                        break
                    text.append(ch.data or ' ')
                    c += 1
                fg = self._colour(ch0.fg, True, ch0.bold)
                bg = self._colour(ch0.bg, False)
                if ch0.reverse:
                    fg, bg = bg, fg
                x = 4 + start * self.cw
                rect = QRect(int(x), int(y), int((c - start) * self.cw + 1), int(self.ch + 1))
                if sel is not None and self._in_sel(sel, r, start):
                    p.fillRect(rect, selc)
                elif ch0.bg != 'default' or ch0.reverse:
                    p.fillRect(rect, bg)
                s = ''.join(text)
                if s.strip():
                    f = self.font
                    if ch0.bold or ch0.italics or ch0.underscore or ch0.strikethrough:
                        f = mono(self.font_px)
                        f.setBold(ch0.bold)
                        f.setItalic(ch0.italics)
                        f.setUnderline(ch0.underscore)
                        f.setStrikeOut(ch0.strikethrough)
                    p.setFont(f)
                    p.setPen(fg)
                    for i, chx in enumerate(s):        # cell by cell: wide glyphs and fallback fonts stay on the grid
                        if chx == ' ':
                            continue
                        if chx in BOX or chx in HEAVY:
                            self._box(p, chx, x + i * self.cw, y, fg)
                        else:
                            p.drawText(int(x + i * self.cw), int(y + self.ascent), chx)
                    p.setFont(self.font)
        cur = self.screen.cursor
        if not cur.hidden and not self.scrolled and not self.ended:
            rect = QRect(int(4 + cur.x * self.cw), int(3 + cur.y * self.ch), int(self.cw), int(self.ch))
            ink = QColor(THEME.tok['ink'])
            if self.hasFocus():
                ink.setAlpha(150)
                p.fillRect(rect, ink)
            else:
                p.setPen(ink)
                p.drawRect(rect.adjusted(0, 0, -1, -1))
        if self.scrolled:
            p.setPen(QColor(THEME.tok['muted']))
            p.drawText(self.rect().adjusted(0, 2, -8, 0), Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignTop,
                       f'history · Shift+PageDown')
        p.end()

    def _box(self, p, ch, x, y, colour):
        from PyQt6.QtCore import QPointF, QRectF
        from PyQt6.QtGui import QPainterPath, QPen
        arms = BOX.get(ch) or HEAVY[ch]
        pen = QPen(colour)
        pen.setWidthF(max(1.0, self.font_px / 13.0) * (2 if ch in HEAVY else 1))
        pen.setCapStyle(Qt.PenCapStyle.FlatCap)
        p.save()
        p.setPen(pen)
        cx, cy = x + self.cw / 2, y + self.ch / 2
        x1, y1 = x + self.cw, y + self.ch
        if ch in ROUND:
            p.setRenderHint(QPainter.RenderHint.Antialiasing)
            r = min(self.cw, self.ch) / 2
            ex = x1 if arms & _R else x                # the arm along the row
            ey = y1 if arms & _D else y                # the arm along the column
            path = QPainterPath(QPointF(ex, cy))
            sx = 1 if arms & _R else -1
            sy = 1 if arms & _D else -1
            path.lineTo(cx + sx * r, cy)
            path.quadTo(QPointF(cx, cy), QPointF(cx, cy + sy * r))
            path.lineTo(cx, ey)
            p.drawPath(path)
        else:
            if arms & _L:
                p.drawLine(QPointF(x, cy), QPointF(cx, cy))
            if arms & _R:
                p.drawLine(QPointF(cx, cy), QPointF(x1, cy))
            if arms & _U:
                p.drawLine(QPointF(cx, y), QPointF(cx, cy))
            if arms & _D:
                p.drawLine(QPointF(cx, cy), QPointF(cx, y1))
        p.restore()

    @staticmethod
    def _in_sel(sel, r, c):
        (r0, c0), (r1, c1) = sel
        return (r0, c0) <= (r, c) <= (r1, c1)


class TerminalPanel(QWidget):
    """The dock's content: a header (Claude, new shell, font size) over the terminal."""

    def __init__(self, ctx, parent=None):
        super().__init__(parent)
        self.ctx = ctx
        self.view = None
        v = QVBoxLayout(self)
        v.setContentsMargins(0, 0, 0, 0)
        v.setSpacing(0)
        bar = QWidget()
        bar.setObjectName('panel')
        h = QHBoxLayout(bar)
        h.setContentsMargins(8, 3, 8, 3)
        h.setSpacing(6)
        self.where = label('Terminal', ink2=True, small=True)
        h.addWidget(self.where, 1)
        self.b_claude = button('Claude', 'ghost')
        self.b_claude.setToolTip('Start Claude Code here: it works on this window\'s project over MCP (its changes '
                                 'are undo steps "Claude · …")')
        self.b_claude.clicked.connect(self.start_claude)
        h.addWidget(self.b_claude)
        self.b_new = button('New shell', 'ghost')
        self.b_new.clicked.connect(self.restart)
        h.addWidget(self.b_new)
        for text, d in (('A−', -1), ('A+', 1)):
            b = button(text, 'ghost')
            b.setFixedWidth(34)
            b.clicked.connect(lambda _=False, d=d: self.view and self.view.set_font_px(self.view.font_px + d))
            h.addWidget(b)
        v.addWidget(bar)
        self.body = QVBoxLayout()
        self.body.setContentsMargins(0, 0, 0, 0)
        v.addLayout(self.body, 1)
        self.note = None
        why = available()
        if why:
            self.note = QLabel(why)
            self.note.setWordWrap(True)
            self.note.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
            self.note.setContentsMargins(14, 12, 14, 12)
            self.note.setAlignment(Qt.AlignmentFlag.AlignTop | Qt.AlignmentFlag.AlignLeft)
            self.body.addWidget(self.note, 1)
            for b in (self.b_claude, self.b_new):
                b.setEnabled(False)

    # ---------------------------------------------------------------- the shell
    def cwd(self) -> str:
        p = self.ctx.store.project
        if p.path and os.path.isdir(os.path.dirname(p.path)):
            return os.path.dirname(p.path)
        from .. import workdir
        d = workdir.get(self.ctx.settings)
        return d if os.path.isdir(d) else os.path.expanduser('~')

    def env(self) -> dict:
        from ..sysenv import system_env
        env = system_env()
        env.update(TERM='xterm-256color', COLORTERM='truecolor', VANTAGE_MCP_PID=str(os.getpid()))
        return env

    def ensure(self, argv=None):
        """Start the shell the first time the panel is shown (or after it ended)."""
        if available():
            return None
        if self.view is None or self.view.ended:
            if self.view is not None:
                self.body.removeWidget(self.view)
                self.view.close_shell()
                self.view.deleteLater()
            self.view = TerminalView(self.cwd(), self.env(), self, argv)
            self.view.exited.connect(self._exited)
            self.view.restart.connect(self.restart)
            self.view.title.connect(lambda t: self.where.setText(f'Terminal · {t}'[:90]))
            self.body.addWidget(self.view, 1)
            self.where.setText(f'Terminal · {self._short(self.cwd())}')
        return self.view

    @staticmethod
    def _short(path):
        home = os.path.expanduser('~')
        return '~' + path[len(home):] if path.startswith(home) else path

    def _exited(self):
        self.where.setText('Terminal · the shell ended (Enter or New shell starts another)')

    def restart(self):
        if self.view is not None:
            self.view.close_shell()
            self.view.ended = True
        v = self.ensure()
        if v is not None:
            v.setFocus()

    def start_claude(self):
        v = self.ensure()
        if v is None:
            return
        v.setFocus()
        if shutil.which('claude', path=self.env().get('PATH')) is None:
            v.send(b"echo 'Claude Code is not installed: https://claude.com/claude-code'\r")
            return
        if not getattr(self.ctx, 'mcp', None) or not self.ctx.mcp.running:
            w = self.ctx.window
            if w is not None:
                w.msg('Claude can read files here, but Preferences ▸ General ▸ Claude is off: it cannot work in this '
                      'window', 6000)
        v.send(b'claude\r')

    def shutdown(self):
        if self.view is not None:
            self.view.close_shell()
