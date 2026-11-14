"""The optional alarm output: one GPIO line for a buzzer or a lamp, on (or beeping) while a fault is raised.

Backends (config alerts.gpio.backend):
- 'gpiod': the libgpiod Python bindings (python3-libgpiod: the v1 API on Ubuntu 22.04 / JetPack 6, v2 on newer
  systems; both are handled), chip and line offset;
- 'gpioset': the libgpiod command-line tool, run once per change (`gpioset <chip> <line>=<0|1>`, the v1 tool's form;
  the v2 tool keeps running by default and needs `-t0`, which is added when `gpioset --version` reports v2);
- 'file': writes 0 or 1 into a file: a sysfs GPIO value file (/sys/class/gpio/gpioN/value, exported beforehand), or
  any file for a test;
- 'none' (the default).
'auto' picks gpiod when it imports, else gpioset when it is on the PATH, else none.

The line is an output only and signals people; the watchdog never drives anything that moves.
"""
from __future__ import annotations

import os
import shutil
import subprocess
import threading


class _Line:
    def __init__(self, cfg, log):
        self.cfg, self.log = cfg, log or (lambda m: None)
        self.active_high = bool(cfg.get('active_high', True))
        self.pattern = cfg.get('pattern', 'beep')        # 'beep' (0.2 s on, 0.8 s off) or 'steady'
        self.alarm = False
        self.level = None
        self.writes = 0
        self.error = ''
        self._stop = threading.Event()
        self._wake = threading.Event()
        self._t = threading.Thread(target=self._run, name='edge-gpio', daemon=True)
        self._t.start()

    def set_alarm(self, on: bool):
        if bool(on) != self.alarm:
            self.alarm = bool(on)
            self._wake.set()

    def _run(self):
        self._write(False)
        phase = False
        while not self._stop.is_set():
            if not self.alarm:
                self._write(False)
                self._wake.wait()
                self._wake.clear()
                continue
            if self.pattern == 'steady':
                self._write(True)
                self._wake.wait(1.0)
                self._wake.clear()
                continue
            phase = not phase
            self._write(phase)
            self._wake.wait(0.2 if phase else 0.8)
            self._wake.clear()
        self._write(False)

    def _write(self, on: bool):
        lvl = int(on) if self.active_high else int(not on)
        if lvl == self.level:
            return
        try:
            self.write(lvl)
            self.level = lvl
            self.writes += 1
        except Exception as e:                            # noqa: BLE001 - a lamp never stops the watchdog
            if str(e) != self.error:
                self.log(f'GPIO: {e}')
            self.error = str(e)

    def close(self):
        self.alarm = False
        self._stop.set()
        self._wake.set()
        self._t.join(timeout=2.0)
        self.release()

    def release(self):
        pass

    def describe(self) -> dict:
        return {'backend': self.name, 'line': self.where(), 'alarm': self.alarm, 'level': self.level,
                'writes': self.writes, 'error': self.error}


class FileLine(_Line):
    name = 'file'

    def __init__(self, cfg, log):
        self.path = cfg.get('path') or ''
        if not self.path:
            raise ValueError('the file GPIO backend needs a path')
        super().__init__(cfg, log)

    def write(self, lvl):
        with open(self.path, 'w') as fh:
            fh.write(f'{lvl}\n')

    def where(self):
        return self.path


class GpiodLine(_Line):
    name = 'gpiod'

    def __init__(self, cfg, log):
        import gpiod
        self.g = gpiod
        chip = str(cfg.get('chip') or 'gpiochip0')
        self.offset = int(cfg.get('line', -1))
        if self.offset < 0:
            raise ValueError('no GPIO line offset configured')
        self.chip_name = chip
        if hasattr(gpiod, 'request_lines'):                       # libgpiod v2 bindings
            from gpiod.line import Direction, Value
            self._v = (Value.INACTIVE, Value.ACTIVE)
            path = chip if chip.startswith('/dev/') else f'/dev/{chip}'
            self.req = gpiod.request_lines(path, consumer='vantage-edge',
                                           config={self.offset: gpiod.LineSettings(direction=Direction.OUTPUT)})
            self._set = lambda lvl: self.req.set_value(self.offset, self._v[lvl])
        else:                                                     # v1
            self.chip = gpiod.Chip(chip)
            self.req = self.chip.get_line(self.offset)
            self.req.request(consumer='vantage-edge', type=gpiod.LINE_REQ_DIR_OUT)
            self._set = lambda lvl: self.req.set_value(lvl)
        super().__init__(cfg, log)

    def write(self, lvl):
        self._set(lvl)

    def release(self):
        try:
            self.req.release()
        except Exception:                                         # noqa: BLE001
            pass

    def where(self):
        return f'{self.chip_name}:{self.offset}'


class GpiosetLine(_Line):
    name = 'gpioset'

    def __init__(self, cfg, log):
        self.exe = shutil.which('gpioset')
        if not self.exe:
            raise ValueError('gpioset is not installed (gpiod package)')
        self.chip = str(cfg.get('chip') or 'gpiochip0')
        self.offset = int(cfg.get('line', -1))
        if self.offset < 0:
            raise ValueError('no GPIO line offset configured')
        try:
            ver = subprocess.run([self.exe, '--version'], capture_output=True, text=True, timeout=3).stdout
        except (OSError, subprocess.TimeoutExpired):
            ver = ''
        self.v2 = 'v2' in ver or ' 2.' in ver
        super().__init__(cfg, log)

    def write(self, lvl):
        args = ([self.exe, '-t0', '-c', self.chip, f'{self.offset}={lvl}'] if self.v2 else
                [self.exe, self.chip, f'{self.offset}={lvl}'])
        r = subprocess.run(args, capture_output=True, text=True, timeout=3)
        if r.returncode:
            raise RuntimeError((r.stderr or r.stdout).strip() or f'gpioset exited {r.returncode}')

    def where(self):
        return f'{self.chip}:{self.offset}'


def make(cfg: dict, log=None):
    """The configured alarm line, or None ('none', or a backend that cannot start: logged)."""
    b = str(cfg.get('backend') or 'none').lower()
    if b == 'none':
        return None
    try:
        if b == 'file':
            return FileLine(cfg, log)
        if b == 'gpiod':
            return GpiodLine(cfg, log)
        if b == 'gpioset':
            return GpiosetLine(cfg, log)
        if b == 'auto':
            if int(cfg.get('line', -1)) < 0:
                return None
            try:
                import gpiod  # noqa: F401
                return GpiodLine(cfg, log)
            except ImportError:
                return GpiosetLine(cfg, log) if shutil.which('gpioset') else None
    except Exception as e:                                        # noqa: BLE001
        if log:
            log(f'GPIO {b}: {e}; no alarm line')
        return None
    if log:
        log(f'unknown GPIO backend {b!r}; no alarm line')
    return None


def available() -> dict:
    """What this computer offers: the gpiod module, the gpioset tool and the GPIO chips."""
    try:
        import gpiod  # noqa: F401
        mod = True
    except ImportError:
        mod = False
    chips = sorted(f for f in os.listdir('/dev') if f.startswith('gpiochip')) if os.path.isdir('/dev') else []
    return {'gpiod_module': mod, 'gpioset': bool(shutil.which('gpioset')), 'chips': chips}
