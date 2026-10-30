"""Finding CARLA, starting and stopping its server, and running the bridge, from Vantage (standard library only).

The server runs in its own systemd user scope with a memory cap (MemoryMax, no swap) so it can never take the machine
down: `systemd-run --user --scope -p MemoryMax=<cap> -p MemorySwapMax=0 CarlaUE4-Linux-Shipping CarlaUE4
-RenderOffScreen -carla-rpc-port=<port> -nosound [-quality-level=Low]`; without systemd-run it is not started. With
no_render the server starts with -nullrhi (no GPU at all: lidar, radar, GNSS and IMU work, cameras do not) and the
Vulkan loader is pointed at no driver, so it cannot touch a GPU even by mistake. Rendering, on a machine with an
NVIDIA driver loaded, the loader is pointed at NVIDIA's Vulkan driver only (unless VK_ICD_FILENAMES or VK_DRIVER_FILES
is set already): Unreal could otherwise pick an integrated GPU, the one that usually drives the desktop.

The bridge (vantage.sim.carla_bridge) runs in CARLA's own Python (a venv with the carla wheel, numpy, OpenCV and
PyYAML) with Vantage's source folder on PYTHONPATH, in a capped scope of its own. SUMO traffic (vantage.sim.sumo_cosim)
needs eclipse-sumo, traci and sumolib in that Python: check_sumo asks it (in a child process, cached), and SUMO runs as
the bridge's child in the bridge's scope.
"""
from __future__ import annotations

import glob
import json
import os
import shutil
import signal
import socket
import subprocess
import sys
import time

BINARY = os.path.join('CarlaUE4', 'Binaries', 'Linux', 'CarlaUE4-Linux-Shipping')
CLIENT_MODULES = ('carla', 'numpy', 'cv2', 'yaml')
BRIDGE_MEMORY_GB = 4.0
ICD_DIRS = ('/etc/vulkan/icd.d', '/usr/share/vulkan/icd.d', '/usr/local/share/vulkan/icd.d')


def nvidia_vulkan(proc='/proc/driver/nvidia/version', dirs=ICD_DIRS) -> str:
    """NVIDIA's Vulkan driver manifest when its kernel driver is loaded, else ''."""
    if not os.path.exists(proc):
        return ''
    for d in dirs:
        f = os.path.join(d, 'nvidia_icd.json')
        if os.path.isfile(f):
            return f
    return ''


def is_carla_root(path) -> bool:
    return bool(path) and os.path.isfile(os.path.join(path, BINARY))


def version(root) -> str:
    try:
        with open(os.path.join(root, 'VERSION'), encoding='utf-8') as fh:
            return fh.read().strip()
    except OSError:
        return ''


def candidate_roots() -> list:
    """CARLA folders in the usual places (a folder holding CarlaUE4/Binaries/Linux/CarlaUE4-Linux-Shipping)."""
    home = os.path.expanduser('~')
    pats = [os.path.join(home, p) for p in ('carla', 'CARLA*', 'carla/CARLA*', 'Documents/carla/CARLA*',
                                            'Documents/*/carla/CARLA*', 'Documents/*/carla', 'opt/carla*')]
    pats += ['/opt/carla*', '/opt/carla-simulator']
    out = []
    for p in pats:
        for d in sorted(glob.glob(p)):
            d = os.path.abspath(d)
            if is_carla_root(d) and d not in out:
                out.append(d)
    return out


def client_python(root) -> str:
    """The Python of CARLA's client environment next to the CARLA folder (<root>/../venv or <root>/venv), or ''."""
    if not root:
        return ''
    for d in (os.path.join(os.path.dirname(root), 'venv'), os.path.join(root, 'venv')):
        py = os.path.join(d, 'bin', 'python')
        if os.path.isfile(py) and os.access(py, os.X_OK):
            return py
    return ''


def check_client(python, timeout=60.0) -> dict:
    """Whether CARLA's Python imports what the bridge needs: {'ok', 'carla_version', 'python', 'missing', 'text'}."""
    if not python or not os.path.isfile(python):
        return {'ok': False, 'missing': list(CLIENT_MODULES), 'text': 'No Python for CARLA\'s client chosen.'}
    code = ('import importlib, json, sys\nout = {"python": sys.version.split()[0], "missing": []}\n'
            f'for m in {CLIENT_MODULES!r}:\n'
            '    try:\n        mod = importlib.import_module(m)\n        if m == "carla":\n'
            '            out["carla_file"] = getattr(mod, "__file__", "")\n'
            '    except Exception:\n        out["missing"].append(m)\n'
            'try:\n    from importlib.metadata import version\n    out["carla_version"] = version("carla")\n'
            'except Exception:\n    out["carla_version"] = ""\nprint(json.dumps(out))\n')
    try:
        r = subprocess.run([python, '-I', '-c', code], capture_output=True, text=True, timeout=timeout,
                           env=_clean_env())
    except (OSError, subprocess.TimeoutExpired) as e:
        return {'ok': False, 'missing': list(CLIENT_MODULES), 'text': f'{python} did not run ({e}).'}
    try:
        out = json.loads(r.stdout.strip().splitlines()[-1])
    except (ValueError, IndexError):
        return {'ok': False, 'missing': list(CLIENT_MODULES), 'text': (r.stderr or r.stdout).strip()[-400:]}
    out['ok'] = not out['missing']
    out['text'] = (f'Python {out["python"]}, carla {out.get("carla_version") or "?"}' if out['ok'] else
                   f'Missing in CARLA\'s Python: {", ".join(out["missing"])} (pip install carla==0.9.16 numpy '
                   'opencv-python-headless pyyaml zstandard)')
    return out


SUMO_MODULES = ('sumo', 'traci', 'sumolib', 'lxml')
SUMO_PACKAGES = ('eclipse-sumo==1.23.1', 'traci==1.23.1', 'sumolib==1.23.1')   # (sumo_cosim.PIP_PACKAGES)
_SUMO_CHECKS = {}                       # (python, its environment's stamp) -> check_sumo's answer


def sumo_install_command(python) -> str:
    """The command that installs SUMO into CARLA's Python (shown as text; Vantage never runs it)."""
    return 'uv pip install --python ' + (python or "<CARLA's python>") + ' ' + ' '.join(SUMO_PACKAGES)


def _env_stamp(python) -> tuple:
    """What changes when packages are installed into the Python's environment: its site-packages folders' times."""
    venv = os.path.dirname(os.path.dirname(os.path.abspath(python)))
    out = []
    for d in sorted(glob.glob(os.path.join(venv, 'lib', 'python*', 'site-packages'))):
        try:
            out.append((d, os.stat(d).st_mtime_ns))
        except OSError:
            pass
    try:
        out.append((python, os.stat(python).st_mtime_ns))
    except OSError:
        pass
    return tuple(out)


def check_sumo(python, timeout=60.0, refresh=False) -> dict:
    """Whether CARLA's Python has SUMO for the co-simulation (sumo, traci, sumolib, lxml), asked of that Python in a
    child process (never imported into Vantage) and cached until its environment changes: {'ok', 'version' (the
    eclipse-sumo package), 'traci', 'sumolib', 'sumo_home', 'binary', 'missing', 'text', 'install'}."""
    install = sumo_install_command(python)
    if not python or not os.path.isfile(python):
        return {'ok': False, 'missing': list(SUMO_MODULES), 'install': install,
                'text': 'No Python for CARLA\'s client chosen.'}
    key = (os.path.abspath(python), _env_stamp(python))
    if not refresh and key in _SUMO_CHECKS:
        return dict(_SUMO_CHECKS[key])
    code = ('import importlib, json, os\nout = {"missing": []}\n'
            f'for m in {SUMO_MODULES!r}:\n'
            '    try:\n        mod = importlib.import_module(m)\n'
            '        if m == "sumo":\n            out["sumo_home"] = getattr(mod, "SUMO_HOME", "")\n'
            '    except Exception:\n        out["missing"].append(m)\n'
            'from importlib.metadata import version\n'
            'for k, d in (("version", "eclipse-sumo"), ("traci", "traci"), ("sumolib", "sumolib")):\n'
            '    try:\n        out[k] = version(d)\n    except Exception:\n        out[k] = ""\n'
            'b = os.path.join(out.get("sumo_home") or "", "bin", "sumo")\n'
            'out["binary"] = b if os.path.isfile(b) or os.path.isfile(b + ".exe") else ""\n'
            'print(json.dumps(out))\n')
    try:
        r = subprocess.run([python, '-I', '-c', code], capture_output=True, text=True, timeout=timeout,
                           env=_clean_env())
        out = json.loads(r.stdout.strip().splitlines()[-1])
    except (OSError, subprocess.TimeoutExpired, ValueError, IndexError) as e:
        return {'ok': False, 'missing': list(SUMO_MODULES), 'install': install,
                'text': f'{python} did not answer ({type(e).__name__}).'}
    if not out.get('missing') and not out.get('binary'):
        out['missing'] = ['sumo binary']
    out['ok'] = not out['missing']
    out['install'] = install
    out['text'] = (f'SUMO {out.get("version") or "?"} in CARLA\'s Python (traci {out.get("traci") or "?"})' if out['ok']
                   else f'SUMO is not in CARLA\'s Python (missing {", ".join(out["missing"])}): install it with '
                        + install)
    _SUMO_CHECKS[key] = dict(out)
    return out


def cosim_folder(root) -> str:
    """CARLA's SUMO co-simulation folder (Co-Simulation/Sumo with sumo_integration/) in a CARLA folder, or ''."""
    d = os.path.join(root or '', 'Co-Simulation', 'Sumo')
    return d if root and os.path.isfile(os.path.join(d, 'sumo_integration', 'sumo_simulation.py')) else ''


def sumo_bundled_towns(root) -> list:
    """The towns CARLA bundles a SUMO net for (Co-Simulation/Sumo/examples/net/<town>.net.xml)."""
    d = os.path.join(cosim_folder(root), 'examples', 'net') if cosim_folder(root) else ''
    try:
        return sorted(n[:-8] for n in os.listdir(d) if n.endswith('.net.xml')) if d else []
    except OSError:
        return []


def _clean_env(extra=None) -> dict:
    env = {k: v for k, v in os.environ.items() if not k.startswith(('PYTHON', 'QT_', 'VIRTUAL_ENV'))}
    env.update(extra or {})
    return env


def port_open(host='127.0.0.1', port=2000, timeout=0.5) -> bool:
    try:
        socket.create_connection((host, int(port)), timeout).close()
        return True
    except OSError:
        return False


def vantage_source() -> str:
    """The folder holding Vantage's `vantage` package as plain source files (what the bridge's PYTHONPATH needs), or
    '' (a frozen build: the bridge then needs a source checkout)."""
    roots = [os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))]
    if getattr(sys, 'frozen', False) and getattr(sys, '_MEIPASS', ''):
        roots.insert(0, sys._MEIPASS)                 # a bundle that carries the bridge's sources as data files
    for root in roots:
        pkg = os.path.join(root, 'vantage')
        if all(os.path.isfile(os.path.join(pkg, *f)) for f in (('__init__.py',), ('core', '__init__.py'),
                                                                ('core', 'rosbag.py'), ('core', 'mathutil.py'),
                                                                ('core', 'groundtruth.py'),
                                                                ('sim', 'carla_bridge.py'), ('sim', 'record.py'),
                                                                ('sim', 'frames.py'), ('sim', 'tonemap.py'))):
            return root
    return ''


def scoped(cmd: list, memory_gb: float, unit: str = '') -> list:
    """cmd in a systemd user scope capped at memory_gb (no swap). Raises RuntimeError without systemd-run."""
    sr = shutil.which('systemd-run')
    if not sr:
        raise RuntimeError('systemd-run is not available: CARLA is only started under a memory cap.')
    pre = [sr, '--user', '--scope', '-q', '-p', f'MemoryMax={max(int(memory_gb * 1024), 512)}M',
           '-p', 'MemorySwapMax=0']
    if unit:
        pre += ['--unit', unit]
    return pre + list(cmd)


class Server:
    """A CARLA server this process started (its own scope, capped)."""

    def __init__(self, root, port=2000, memory_gb=12.0, quality='Epic', no_render=False, log_path=''):
        if not is_carla_root(root):
            raise ValueError(f'{root} is not a CARLA folder ({BINARY} is missing).')
        self.root, self.port, self.memory_gb = root, int(port), float(memory_gb)
        self.quality, self.no_render = quality, bool(no_render)
        self.log_path = log_path
        self.proc = None
        self.env = None                  # what start() ran the server with
        self.started = 0.0
        self.unit = f'vantage-carla-{self.port}-{os.getpid()}'

    def command(self) -> list:
        args = [os.path.join(self.root, BINARY), 'CarlaUE4', f'-carla-rpc-port={self.port}', '-nosound']
        if self.no_render:
            args.append('-nullrhi')                     # (-quality-level=Low crashes UE4 with -nullrhi)
        else:
            args.append('-RenderOffScreen')
            if self.quality == 'Low':
                args.append('-quality-level=Low')
        return scoped(args, self.memory_gb, self.unit)

    def start(self):
        if self.running:
            return
        if port_open('127.0.0.1', self.port):
            raise RuntimeError(f'Port {self.port} is already in use (another CARLA server?).')
        env = _clean_env()
        env.pop('DISPLAY', None)
        env.pop('WAYLAND_DISPLAY', None)
        if self.no_render:                              # no Vulkan driver at all: it cannot reach a GPU
            env['VK_DRIVER_FILES'] = env['VK_ICD_FILENAMES'] = '/nonexistent.json'
        elif not env.get('VK_ICD_FILENAMES') and not env.get('VK_DRIVER_FILES'):
            icd = nvidia_vulkan()                       # the discrete GPU, never the desktop's integrated one
            if icd:
                env['VK_DRIVER_FILES'] = env['VK_ICD_FILENAMES'] = icd
        self.env = env
        log = open(self.log_path, 'w') if self.log_path else subprocess.DEVNULL
        self.proc = subprocess.Popen(self.command(), stdout=log, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL,
                                     env=env, start_new_session=True, cwd=self.root)
        self.started = time.time()

    @property
    def running(self) -> bool:
        return self.proc is not None and self.proc.poll() is None

    def ready(self) -> bool:
        return self.running and port_open('127.0.0.1', self.port)

    def stop(self, timeout=20.0):
        """SIGTERM to the server's process group, SIGKILL after timeout; the scope goes with it."""
        p = self.proc
        if p is None:
            return
        if p.poll() is None:
            try:
                os.killpg(p.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
            try:
                p.wait(timeout)
            except subprocess.TimeoutExpired:
                try:
                    os.killpg(p.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                try:
                    p.wait(5)
                except subprocess.TimeoutExpired:
                    pass
        self.proc = None


def bridge_command(python, rig_path, out_dir, args=(), memory_gb=BRIDGE_MEMORY_GB, scope=True) -> tuple[list, dict]:
    """(command, environment) running the bridge in CARLA's Python with Vantage's source on PYTHONPATH."""
    src = vantage_source()
    if not src:
        raise RuntimeError('The bridge needs Vantage\'s source folder (a frozen build has none): run it from a '
                           'source checkout.')
    cmd = [python, '-m', 'vantage.sim.carla_bridge', rig_path, out_dir, '--progress-json'] + list(args)
    if scope:
        cmd = scoped(cmd, memory_gb)
    return cmd, _clean_env({'PYTHONPATH': src, 'PYTHONUNBUFFERED': '1'})


def tidy_command(python, port, host='127.0.0.1', memory_gb=1.0, scope=True) -> tuple[list, dict]:
    """(command, environment) tidying the server at port after a killed bridge (carla_bridge --tidy)."""
    src = vantage_source()
    if not src:
        raise RuntimeError('Tidying the server needs Vantage\'s source folder.')
    cmd = [python, '-m', 'vantage.sim.carla_bridge', '--tidy', '--host', host, '--port', str(int(port))]
    if scope:
        cmd = scoped(cmd, memory_gb)
    return cmd, _clean_env({'PYTHONPATH': src, 'PYTHONUNBUFFERED': '1'})


def parse_progress(line: str):
    """(fraction, text) of a --progress-json line, or None."""
    try:
        d = json.loads(line)
    except ValueError:
        return None
    if isinstance(d, dict) and 'progress' in d:
        return float(d['progress']), str(d.get('text', ''))
    return None


if __name__ == '__main__':                              # a quick look: python -m vantage.sim.server
    for r in candidate_roots():
        print(r, version(r), client_python(r) or '(no venv next to it)')
    sys.exit(0)
