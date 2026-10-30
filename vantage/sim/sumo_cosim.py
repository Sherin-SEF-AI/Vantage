"""SUMO traffic co-simulated with CARLA: Eclipse SUMO drives the background vehicles of a CARLA drive in lockstep, the
way CARLA's own co-simulation (CARLA_ROOT/Co-Simulation/Sumo, run_synchronization.py) does it, so a drive gets SUMO's
demand models, routes and, optionally, its traffic-light control.

Two halves:

* the contract (stdlib only, imported by the app: core.simrig, the CARLA page): the settings block (`default`,
  `clean_sumo`, `problems`), where CARLA's co-simulation lives (`find_cosim`) and the install command for CARLA's own
  Python (`install_command`). Nothing here imports traci or sumolib.
* `CoSim`, run by the bridge (vantage.sim.carla_bridge) in CARLA's Python: it reuses CARLA's `sumo_integration`
  package (BridgeHelper, SumoSimulation, CarlaSimulation; its folder is put on sys.path from the CARLA root) and does
  per synchronous CARLA step what run_synchronization.py's SimulationSynchronization.tick does, in two halves around
  the bridge's own World.tick (the bridge ticks the world itself, with its sensors):
    before the tick: SUMO steps once (its step = CARLA's fixed_delta_seconds when that is a whole number of
    milliseconds, SUMO's resolution; else SUMO steps every k ticks, k·dt whole milliseconds, and its vehicles move on
    along their heading at SUMO's speed in between: sumo_substeps); vehicles SUMO inserted are spawned in
    CARLA (physics off, BridgeHelper's blueprint for their vType), those that arrived are destroyed, every SUMO vehicle
    is moved to SUMO's pose (BridgeHelper.get_carla_transform); with tls 'sumo', CARLA's lights take SUMO's states;
    after the tick: the CARLA vehicles SUMO did not make (the ego, a scenario's vehicles) are mirrored into SUMO
    (traci moveToXY) so its traffic reacts to them; with tls 'carla', SUMO's lights take CARLA's states.
  Every set is walked in sorted order and every random choice is seeded (SUMO's --seed = the drive's seed,
  randomTrips' --seed too, the vehicle types by numpy from the seed, BridgeHelper's colour choice by numpy seeded
  with the seed and the SUMO vehicle's id), so a seed gives the same SUMO vehicles on the same paths.

The net: 'auto' takes CARLA's bundled net for Town01, Town04 and Town05 (Co-Simulation/Sumo/examples/net) and for any
other town generates one from the map's OpenDRIVE (World.get_map().to_opendrive()) with CARLA's
util/netconvert_carla.py (SUMO's netconvert, then CARLA's traffic-light landmarks), cached by the OpenDRIVE's SHA-256;
'generate' generates it for a bundled town too; a path takes the user's .net.xml or .sumocfg. Measured on 0.9.16: the
bundled nets (SUMO net version 1.6) carry no traffic lights, so SUMO's vehicles there do not stop at CARLA's lights
whoever runs them; a generated net has CARLA's lights as SUMO traffic lights linked to CARLA's landmarks.

The demand: randomTrips.py (SUMO's tools) on the net with --seed, trips over passenger edges checked by duarouter
(--validate), each trip's vType drawn from CARLA's measured vTypes (Co-Simulation/Sumo/examples/carlavtypes.rou.xml)
of the chosen classes that the server has as blueprints (data/vtypes.json maps vClass to blueprint). 'count': the
trips come in fast and SUMO's --max-num-vehicles holds `traffic` vehicles in the network at once (plus the mirrored
CARLA vehicles); 'period': one trip every period_s, at most `traffic` at once. A user .sumocfg brings its own demand.

Limits (said in run.json): walkers stay on CARLA's walker AI (CARLA's co-simulation does not synchronise SUMO's
pedestrians) and SUMO's vehicles do not see them; SUMO's vehicles are kinematic in CARLA (physics off, moved each step:
no suspension, wheels or skids); a vehicle SUMO teleports jumps; SUMO's vehicles appear where their trip starts and
vanish where it ends.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import shutil
import subprocess
import sys
import time
import xml.etree.ElementTree as ET
import zlib

SUMO_VERSION = '1.23.1'                  # the eclipse-sumo release the co-simulation was built and measured with
PIP_PACKAGES = (f'eclipse-sumo=={SUMO_VERSION}', f'traci=={SUMO_VERSION}', f'sumolib=={SUMO_VERSION}')
SOURCES = ('carla', 'sumo')
TLS = ('carla', 'sumo')
DEMANDS = ('count', 'period')
NET_CHOICES = ('auto', 'generate')
NET_SUFFIXES = ('.net.xml', '.net.xml.gz', '.sumocfg')
CLASSES = ('passenger', 'evehicle', 'authority', 'truck', 'motorcycle', 'bicycle')
DEFAULT_CLASSES = ('passenger', 'evehicle', 'authority', 'truck')      # CARLA's four-wheeled cars, as its own traffic
BUNDLED_TOWNS = ('Town01', 'Town04', 'Town05')                         # Co-Simulation/Sumo/examples/net in 0.9.16
LIMITS = {'period_s': (0.1, 60.0)}
COSIM_DIR = os.path.join('Co-Simulation', 'Sumo')
ROLE_SUMO = 'vantage_sumo'               # role name of the CARLA vehicles SUMO drives (carla_bridge.ROLE_PREFIX)
MAX_SUBSTEPS = 16                        # SUMO steps at least every this many simulation steps
MAX_TRIPS = 20000                        # the most trips a 'count' demand writes (the period is lengthened beyond)
START_TIMEOUT_S = 60.0                   # SUMO must listen on its port within this
NET_TIMEOUT_S = 600.0                    # netconvert on a town's OpenDRIVE (Town10HD: 0.5 s measured here)
TRIPS_TIMEOUT_S = 300.0
WALKERS_NOTE = ('Walkers stay on CARLA\'s walker AI: CARLA\'s SUMO co-simulation does not synchronise SUMO\'s '
                'pedestrians, and SUMO\'s vehicles do not see CARLA\'s walkers.')
LIMITS_TEXT = ('SUMO\'s vehicles are kinematic in CARLA (physics off, moved to SUMO\'s pose every step): no suspension, '
               'wheel or skid; a vehicle SUMO teleports (a jam it gives up on) jumps; vehicles appear where their trip '
               'starts and vanish where it ends. ' + WALKERS_NOTE)


# =========================================================================== the contract (app side, stdlib only)
def default() -> dict:
    """The settings block's defaults (SimSettings.sumo)."""
    return {'net': 'auto', 'demand': 'count', 'period_s': 2.0, 'classes': list(DEFAULT_CLASSES), 'tls': 'carla',
            'binary': 'auto'}


def _net_ok(v) -> bool:
    return isinstance(v, str) and (v in NET_CHOICES or (v.strip().lower().endswith(NET_SUFFIXES) and len(v) < 1024))


def clean_sumo(d) -> tuple[dict, list]:
    """(settings, problems): the SUMO block kept to what is valid (numbers clamped, words checked, unknown keys
    dropped); what was dropped is said in problems."""
    out = default()
    bad = []
    d = d if isinstance(d, dict) else {}
    net = d.get('net')
    if net is not None:
        if _net_ok(net):
            out['net'] = net.strip() if net not in NET_CHOICES else net
        else:
            bad.append(f'SUMO net {net!r}: auto, generate, or a .net.xml or .sumocfg file')
    for k, opts in (('demand', DEMANDS), ('tls', TLS)):
        if k in d:
            if d[k] in opts:
                out[k] = d[k]
            else:
                bad.append(f'SUMO {k} {d[k]!r}: one of {", ".join(opts)}')
    if 'period_s' in d and not isinstance(d['period_s'], bool):
        try:
            x = float(d['period_s'])
        except (TypeError, ValueError):
            x = float('nan')
        if math.isfinite(x):
            lo, hi = LIMITS['period_s']
            out['period_s'] = min(max(x, lo), hi)
        else:
            bad.append(f'SUMO period_s {d["period_s"]!r}: a number of seconds')
    if 'classes' in d:
        c = d['classes']
        c = [c] if isinstance(c, str) else c
        got = [x for x in CLASSES if isinstance(c, (list, tuple)) and x in c]
        if got:
            out['classes'] = got
        else:
            bad.append(f'SUMO classes {d["classes"]!r}: some of {", ".join(CLASSES)}')
        unknown = [x for x in (c if isinstance(c, (list, tuple)) else []) if x not in CLASSES]
        if unknown and got:
            bad.append(f'SUMO classes: {", ".join(map(str, unknown))} not known (of {", ".join(CLASSES)})')
    b = d.get('binary')
    if b is not None:
        if isinstance(b, str) and b.strip() and len(b) < 1024:
            out['binary'] = b.strip()
        else:
            bad.append(f'SUMO binary {b!r}: auto or a path')
    return out, bad


def net_kind(sumo: dict, town: str) -> str:
    """What the net will be for the town: 'bundled', 'generated', 'user' or 'user_config'."""
    net = sumo.get('net', 'auto')
    if net == 'auto':
        return 'bundled' if _town(town) in BUNDLED_TOWNS else 'generated'
    if net == 'generate':
        return 'generated'
    return 'user_config' if str(net).lower().endswith('.sumocfg') else 'user'


def problems(sumo: dict) -> list:
    """What refuses a SUMO run before anything starts: a user net or SUMO binary that is not there."""
    out = []
    net = sumo.get('net', 'auto')
    if net not in NET_CHOICES and not os.path.isfile(os.path.expanduser(str(net))):
        out.append(f'The SUMO net {net} does not exist.')
    b = sumo.get('binary', 'auto')
    if b != 'auto' and not (os.path.isfile(os.path.expanduser(b)) and os.access(os.path.expanduser(b), os.X_OK)):
        out.append(f'The SUMO binary {b} does not exist or cannot be run.')
    return out


def install_command(python: str) -> str:
    """The command that puts SUMO into CARLA's own Python (shown to the user, never run by Vantage)."""
    return 'uv pip install --python ' + (python or "<CARLA's python>") + ' ' + ' '.join(PIP_PACKAGES)


def _town(name: str) -> str:
    t = str(name or '').rsplit('/', 1)[-1]
    return t[:-4] if t.endswith('_Opt') else t


def is_cosim(path) -> bool:
    return bool(path) and os.path.isfile(os.path.join(path, 'sumo_integration', 'sumo_simulation.py'))


def find_cosim(carla_root: str = '', python: str = '') -> str:
    """CARLA's co-simulation folder (Co-Simulation/Sumo, holding sumo_integration/): under carla_root, else under the
    CARLA folder next to the Python's environment (the layout server.client_python reads the other way round: a
    venv beside or inside the CARLA folder), else ''."""
    roots = []
    if carla_root:
        roots.append(carla_root)
    env = os.environ.get('CARLA_ROOT', '')
    if env:
        roots.append(env)
    py = python or sys.executable
    venv = os.path.dirname(os.path.dirname(os.path.abspath(py)))
    for base in (os.path.dirname(venv), venv):
        try:
            names = sorted(os.listdir(base))
        except OSError:
            continue
        roots += [os.path.join(base, n) for n in names if n.lower().startswith('carla')]
        roots.append(base)
    for r in roots:
        for cand in (os.path.join(r, COSIM_DIR), r):
            if is_cosim(cand):
                return os.path.abspath(cand)
    return ''


def bundled_nets(cosim: str) -> list:
    """The towns CARLA bundles a SUMO net for in this co-simulation folder."""
    d = os.path.join(cosim or '', 'examples', 'net')
    try:
        return sorted(n[:-8] for n in os.listdir(d) if n.endswith('.net.xml'))
    except OSError:
        return []


def default_cache() -> str:
    """Where generated nets are cached when the caller names no folder: the default working folder's sumo-nets
    (vantage.workdir.default(); the app passes the working folder it uses)."""
    return os.path.join(os.path.expanduser('~'), 'Documents', 'Vantage', 'sumo-nets')


def sha256_file(path) -> str:
    h = hashlib.sha256()
    with open(path, 'rb') as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def sumo_substeps(dt: float) -> int:
    """How many simulation steps one SUMO step spans: SUMO's time is whole milliseconds (its --step-length must be a
    multiple of 0.001 s; measured: asked for 0.0125 it ran 0.013), so the smallest k for which k·dt is a whole number
    of milliseconds (80 Hz: 2, a 25 ms SUMO step), else the k nearest a whole one within MAX_SUBSTEPS."""
    ms = float(dt) * 1000.0
    best = (1e9, 1)
    for k in range(1, MAX_SUBSTEPS + 1):
        err = abs(k * ms - round(k * ms))
        if err < 1e-6 and round(k * ms) >= 1:
            return k
        best = min(best, (err / (k * ms), k))
    return best[1]


def _tail(path, n=12) -> str:
    try:
        with open(path, encoding='utf-8', errors='replace') as fh:
            lines = [x.rstrip() for x in fh.readlines() if x.strip()]
    except OSError:
        return ''
    return ' | '.join(lines[-n:])


# =========================================================================== the co-simulation (CARLA's Python)
class SumoMissing(RuntimeError):
    pass


def preflight(sumo: dict, carla_root: str = '', python: str = '') -> dict:
    """Before CARLA is touched: SUMO, traci, sumolib and lxml import in this Python, CARLA's co-simulation folder
    exists, the user's net and binary are there. Returns {'cosim', 'sumo_home'}; raises SumoMissing (what is missing
    and the install command) or ValueError."""
    missing = []
    for m in ('sumo', 'traci', 'sumolib', 'lxml'):
        try:
            __import__(m)
        except Exception:                                # noqa: BLE001 (any import failure is "missing")
            missing.append(m)
    if missing:
        raise SumoMissing(f'SUMO traffic needs {", ".join(missing)} in CARLA\'s Python ({python or sys.executable}): '
                          + install_command(python or sys.executable))
    cosim = find_cosim(carla_root, python)
    if not cosim:
        raise SumoMissing('CARLA\'s SUMO co-simulation (Co-Simulation/Sumo with sumo_integration/) was not found next '
                          'to CARLA\'s Python; give the CARLA folder (--carla-root).')
    bad = problems(sumo)
    if bad:
        raise ValueError(' '.join(bad))
    import sumo as _sumo
    return {'cosim': cosim, 'sumo_home': getattr(_sumo, 'SUMO_HOME', '') or os.environ.get('SUMO_HOME', '')}


class CoSim:
    """SUMO beside a CARLA drive, stepped by the bridge: start(), then before_tick() / after_tick() around every
    synchronous World.tick, close() on every exit path. on_spawn(actor) / on_destroy(actor_id) tell the bridge about
    the CARLA vehicles SUMO drives (its ground truth, its tidy-up); carla_vehicles() lists the bridge's own CARLA
    vehicles (the ego, a scenario's) to mirror into SUMO; check() is the bridge's cancel check (it raises)."""

    def __init__(self, carla, client, world, sumo: dict, *, seed: int, dt: float, vehicles: int, horizon_s: float,
                 work_dir: str, cache_dir: str = '', carla_root: str = '', on_spawn=None, on_destroy=None,
                 carla_vehicles=None, check=None, note=None):
        self.carla, self.client, self.world = carla, client, world
        self.opts = dict(sumo)
        self.seed, self.dt, self.n = int(seed), float(dt), max(int(vehicles), 0)
        self.k = sumo_substeps(self.dt)   # simulation steps per SUMO step (SUMO's time is whole milliseconds)
        self.sumo_dt = round(self.k * self.dt, 3)
        self.ticks = 0                    # before_tick calls: SUMO steps on every k-th
        self.base = {}                    # SUMO vehicle id -> its CARLA transform at SUMO's last step
        self.horizon_s = float(horizon_s)
        self.work_dir = os.path.abspath(work_dir)
        self.cache_dir = os.path.abspath(cache_dir) if cache_dir else default_cache()
        self.carla_root = carla_root
        self.on_spawn = on_spawn or (lambda a: None)
        self.on_destroy = on_destroy or (lambda aid: None)
        self.carla_vehicles = carla_vehicles or (lambda: [])
        self.check = check or (lambda: None)
        self.note = note or (lambda text: None)
        self.town = _town(world.get_map().name)
        self.sumo = None                  # sumo_integration's SumoSimulation
        self.side = None                  # sumo_integration's CarlaSimulation on the bridge's client
        self.proc = None
        self.port = None
        self.log_path = os.path.join(self.work_dir, 'sumo.log')
        self.sumo2carla = {}              # SUMO vehicle id -> CARLA actor id (vehicles SUMO drives)
        self.carla2sumo = {}              # CARLA actor id -> SUMO vehicle id (the bridge's vehicles mirrored)
        self.speed = {}                   # SUMO vehicle id -> its speed this step (m/s)
        self.rows = {}                    # SUMO vehicle id -> its row in run.json traffic_source.actors
        self.mirrored = []
        self.common_tls = []
        self.frozen = False               # CARLA's lights frozen for SUMO to set them
        self.closed = False
        self.started = False
        self.t = 0.0                      # SUMO's time after the last step
        self.recording = False
        self.st = {'spawned': 0, 'unsupported': 0, 'spawn_failed': 0, 'teleports': 0, 'collisions': 0,
                   'arrived': 0, 'peak': 0, 'sum': 0, 'steps': 0, 'at_start': None, 'mirror_errors': 0}
        self.timing = {'net_s': 0.0, 'demand_s': 0.0, 'launch_s': 0.0, 'step_s': 0.0, 'sync_s': 0.0}
        self.info = {}
        self.versions = {}

    # ------------------------------------------------------------------ setup
    def _imports(self):
        """CARLA's sumo_integration on sys.path (from the CARLA root, never copied), SUMO_HOME from the sumo package."""
        self.cosim = find_cosim(self.carla_root)
        if not self.cosim:
            raise SumoMissing('CARLA\'s SUMO co-simulation (Co-Simulation/Sumo) was not found.')
        import sumo as _sumo
        home = getattr(_sumo, 'SUMO_HOME', '') or os.environ.get('SUMO_HOME', '')
        if home:
            os.environ['SUMO_HOME'] = home                 # sumolib.checkBinary and SUMO's tools read it
        self.sumo_home = home
        tools = os.path.join(home, 'tools')
        if os.path.isdir(tools) and tools not in sys.path:
            sys.path.append(tools)                         # as run_synchronization.py does
        if self.cosim not in sys.path:
            sys.path.insert(0, self.cosim)
        import sumolib
        import traci
        from sumo_integration import constants as K
        from sumo_integration import bridge_helper as BHM
        from sumo_integration.bridge_helper import BridgeHelper
        from sumo_integration.carla_simulation import CarlaSimulation
        from sumo_integration.sumo_simulation import SumoSimulation
        self.traci, self.sumolib, self.K = traci, sumolib, K
        self.BH, self.SumoSimulation = BridgeHelper, SumoSimulation
        self.BHM = BHM                    # its `random` (BridgeHelper's colour choice): a seeded draw per vehicle

        class _CarlaSide(CarlaSimulation):
            """CARLA's CarlaSimulation on the bridge's own client and world: its spawn, move, destroy and light
            methods as they are; not its constructor, which opens a second client with a 2 s timeout, nor its tick
            (the bridge ticks the world)."""

            def __init__(self, client, world, step_length):           # noqa: super().__init__ deliberately not called
                self.client, self.world = client, world
                self.blueprint_library = world.get_blueprint_library()
                self.step_length = step_length
                self._active_actors, self.spawned_actors, self.destroyed_actors = set(), set(), set()
                self._tls = {}
                m = world.get_map()
                for lm in (m.get_all_landmarks_of_type('1000001') if hasattr(m, 'get_all_landmarks_of_type') else []):
                    if lm.id != '':
                        tl = world.get_traffic_light(lm)
                        if tl is not None:
                            self._tls[lm.id] = tl
        self._CarlaSide = _CarlaSide
        try:
            from importlib.metadata import version
            self.versions['eclipse_sumo'] = version('eclipse-sumo')
        except Exception:                                  # noqa: BLE001
            self.versions['eclipse_sumo'] = getattr(_sumo, '__version__', None)

    def _run(self, cmd, timeout, what, cwd=None):
        """A SUMO tool as a child process (inside the bridge's memory-capped scope), polled so a cancel stops it."""
        env = dict(os.environ)
        if self.sumo_home:
            env['SUMO_HOME'] = self.sumo_home
            env['PATH'] = os.path.join(self.sumo_home, 'bin') + os.pathsep + env.get('PATH', '')
        log = os.path.join(self.work_dir, what.replace(' ', '_') + '.log')
        with open(log, 'w', encoding='utf-8') as fh:
            p = subprocess.Popen(cmd, stdout=fh, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL, env=env,
                                 cwd=cwd or self.work_dir)
            t0 = time.monotonic()
            try:
                while p.poll() is None:
                    self.check()
                    if time.monotonic() - t0 > timeout:
                        raise RuntimeError(f'{what} took more than {timeout:g} s')
                    time.sleep(0.2)
            finally:
                if p.poll() is None:
                    p.kill()
                    p.wait(10)
        if p.returncode != 0:
            raise RuntimeError(f'{what} failed (exit {p.returncode}): {_tail(log, 6)}')
        return log

    def _net(self) -> dict:
        """The SUMO net: {'kind', 'file', 'path', 'sha256', 'cached', 'seconds', 'config'}."""
        t0 = time.perf_counter()
        net = self.opts.get('net', 'auto')
        kind = net_kind(self.opts, self.town)
        out = {'kind': kind, 'town': self.town, 'cached': None, 'config': None}
        if kind in ('user', 'user_config'):
            path = os.path.abspath(os.path.expanduser(net))
            if not os.path.isfile(path):
                raise ValueError(f'The SUMO net {net} does not exist.')
            if kind == 'user_config':
                out['config'] = path
                tag = ET.parse(path).getroot().find('.//net-file')
                if tag is None:
                    raise ValueError(f'{os.path.basename(path)} names no net-file.')
                path = os.path.join(os.path.dirname(path), tag.get('value'))
            out['path'] = path
        elif kind == 'bundled':
            path = os.path.join(self.cosim, 'examples', 'net', f'{self.town}.net.xml')
            if not os.path.isfile(path):                   # not in this CARLA package: generate it
                kind = out['kind'] = 'generated'
            else:
                out['path'] = path
        if kind == 'generated':
            xodr = self.world.get_map().to_opendrive()
            h = hashlib.sha256(xodr.encode('utf-8')).hexdigest()
            os.makedirs(self.cache_dir, exist_ok=True)
            path = os.path.join(self.cache_dir, f'{self.town}-{h[:16]}.net.xml')
            out['opendrive_sha256'] = h
            if os.path.isfile(path):
                out['cached'] = True
            else:
                out['cached'] = False
                src = os.path.join(self.cache_dir, f'{self.town}-{h[:16]}.xodr')
                with open(src, 'w', encoding='utf-8') as fh:
                    fh.write(xodr)
                tmp = path + f'.{os.getpid()}.tmp.net.xml'
                tool = os.path.join(self.cosim, 'util', 'netconvert_carla.py')
                self.note(f'Generating the SUMO net of {self.town} from its OpenDRIVE (CARLA\'s netconvert_carla.py); '
                          'cached for the next run.')
                try:
                    self._run([sys.executable, tool, src, '--output', tmp], NET_TIMEOUT_S, 'netconvert')
                    os.replace(tmp, path)                  # atomic: two runs at once never read half a net
                finally:
                    for f in (tmp, src):
                        if os.path.exists(f):
                            os.remove(f)
            out['path'] = path
        out['file'] = os.path.basename(out['path'])
        out['sha256'] = sha256_file(out['path'])
        out['seconds'] = round(time.perf_counter() - t0, 3)
        self.timing['net_s'] = out['seconds']
        return out

    def _vtypes(self) -> list:
        """[(vType id, its XML attributes)] SUMO may insert: CARLA's measured vTypes (examples/carlavtypes.rou.xml) of
        the chosen classes (data/vtypes.json's vClass) that the server has as blueprints, by id."""
        with open(os.path.join(self.cosim, 'data', 'vtypes.json'), encoding='utf-8') as fh:
            known = json.load(fh).get('carla_blueprints', {})
        have = {b.id for b in self.world.get_blueprint_library().filter('vehicle.*')}
        classes = set(self.opts.get('classes') or DEFAULT_CLASSES)
        out = []
        root = ET.parse(os.path.join(self.cosim, 'examples', 'carlavtypes.rou.xml')).getroot()
        for vt in root.iter('vType'):
            vid = vt.get('id')
            if vid in have and known.get(vid, {}).get('vClass') in classes and vt.get('vClass') in classes:
                out.append((vid, dict(vt.attrib)))
        out.sort(key=lambda r: r[0])
        if not out:
            raise RuntimeError(f'No CARLA blueprint of the classes {", ".join(sorted(classes))} has a SUMO vType.')
        return out

    def _demand(self, net_path) -> dict:
        """randomTrips.py on the net (seeded), the trips' vTypes drawn from the seed: demand.rou.xml."""
        import numpy as np
        t0 = time.perf_counter()
        mode = self.opts.get('demand', 'count')
        end = max(self.horizon_s, 1.0)
        if mode == 'count':
            period = max(1.0 / max(self.n, 1), end / MAX_TRIPS)
        else:
            period = float(self.opts.get('period_s', 2.0))
        trips = os.path.join(self.work_dir, 'trips.xml')
        tool = os.path.join(self.sumo_home, 'tools', 'randomTrips.py')
        cmd = [sys.executable, tool, '-n', net_path, '-o', trips, '--seed', str(self.seed), '-b', '0',
               '-e', f'{end:.3f}', '-p', f'{period:.6g}', '--vclass', 'passenger', '--validate',
               '--remove-loops', '--prefix', 'veh', '--min-distance', '100',
               '--trip-attributes', 'departLane="best" departSpeed="max" departPos="random"']
        self._run(cmd, TRIPS_TIMEOUT_S, 'randomTrips')
        types = self._vtypes()
        rng = np.random.default_rng([self.seed, 5381])
        tree = ET.parse(trips)
        root = tree.getroot()
        items = [e for e in root if e.tag in ('trip', 'vehicle')]
        for e in items:
            e.set('type', types[int(rng.integers(len(types)))][0])
        out = ET.Element('routes')
        for vid, attrs in types:
            ET.SubElement(out, 'vType', attrs)
        for e in items:
            out.append(e)
        path = os.path.join(self.work_dir, 'demand.rou.xml')
        ET.ElementTree(out).write(path, encoding='utf-8', xml_declaration=True)
        for f in os.listdir(self.work_dir):                # randomTrips' and duarouter's own files
            if f.startswith('trips.') or f == 'routes.rou.xml':
                os.remove(os.path.join(self.work_dir, f))
        self.timing['demand_s'] = round(time.perf_counter() - t0, 3)
        return {'mode': mode, 'vehicles': self.n, 'period_s': round(period, 6), 'trips': len(items),
                'classes': list(self.opts.get('classes') or DEFAULT_CLASSES), 'vtypes': [t[0] for t in types],
                'routes_file': 'demand.rou.xml', 'routes_sha256': sha256_file(path),
                'generator': f'randomTrips.py --seed {self.seed} -b 0 -e {end:.3f} -p {period:.6g} --vclass passenger '
                             '--validate --remove-loops --min-distance 100; vTypes drawn by numpy from the seed',
                'seconds': self.timing['demand_s']}

    def _config(self, net) -> tuple[str, dict]:
        """The SUMO configuration run: the user's .sumocfg as it is, else run.sumocfg (the net, demand.rou.xml)."""
        if net['kind'] == 'user_config':
            return net['config'], {'mode': 'user', 'vehicles': self.n, 'config': os.path.basename(net['config']),
                                   'config_sha256': sha256_file(net['config'])}
        demand = self._demand(net['path'])
        cfg = os.path.join(self.work_dir, 'run.sumocfg')
        root = ET.Element('configuration')
        inp = ET.SubElement(root, 'input')
        ET.SubElement(inp, 'net-file', value=net['path'])
        ET.SubElement(inp, 'route-files', value='demand.rou.xml')
        ET.ElementTree(root).write(cfg, encoding='utf-8', xml_declaration=True)
        return cfg, demand

    def _binary(self) -> str:
        b = self.opts.get('binary', 'auto')
        if b and b != 'auto':
            return os.path.abspath(os.path.expanduser(b))
        return self.sumolib.checkBinary('sumo')           # SUMO_HOME/bin/sumo: headless, never sumo-gui

    def start(self):
        """Net, demand, SUMO started headless with its own TraCI port, connected through sumo_integration."""
        os.makedirs(self.work_dir, exist_ok=True)
        self._imports()
        self.check()
        net = self._net()
        self.check()
        cfg, demand = self._config(net)
        self.check()
        t0 = time.perf_counter()
        mirrored = len(self.carla_vehicles())
        cap = self.n + mirrored                            # SUMO counts the mirrored vehicles as its own
        self.port = int(self.sumolib.miscutils.getFreeSocketPort())
        cmd = [self._binary(), '-c', cfg, '--remote-port', str(self.port), '--num-clients', '1',
               '--seed', str(self.seed), '--step-length', f'{self.sumo_dt:.3f}', '--begin', '0',
               '--lateral-resolution', '0.25', '--collision.check-junctions',      # (as SumoSimulation starts SUMO)
               '--max-num-vehicles', str(cap), '--ignore-route-errors', 'true', '--no-step-log', 'true',
               '--duration-log.disable', 'true']
        self.cmd = cmd
        log = open(self.log_path, 'w', encoding='utf-8')
        try:
            self.proc = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL,
                                         cwd=os.path.dirname(cfg))
        finally:
            log.close()
        deadline = time.monotonic() + START_TIMEOUT_S
        while True:                                        # (traci.init would retry for a minute on a dead SUMO)
            self.check()
            if self.proc.poll() is not None:
                raise RuntimeError(f'SUMO did not start (exit {self.proc.returncode}): {_tail(self.log_path)}')
            up = listening(self.port)
            if up or up is None:                           # (None: no /proc to read: traci.init's retries wait)
                break
            if time.monotonic() > deadline:
                raise RuntimeError(f'SUMO did not listen on port {self.port} within {START_TIMEOUT_S:g} s')
            time.sleep(0.1)
        self.sumo = self.SumoSimulation(cfg, self.dt, '127.0.0.1', self.port)
        self.started = True
        traci = self.traci
        BH = self.BH
        BH.blueprint_library = self.world.get_blueprint_library()
        BH.offset = self.sumo.get_net_offset()
        self.side = self._CarlaSide(self.client, self.world, self.dt)
        self.common_tls = sorted(self.sumo.traffic_light_ids & self.side.traffic_light_ids)
        tls = self.opts.get('tls', 'carla')
        if tls == 'carla':
            self.sumo.switch_off_traffic_lights()          # CARLA's lights rule; SUMO's take their states each step
        elif self.common_tls:
            self.side.switch_off_traffic_lights()          # frozen (green); SUMO's states set each step
            self.frozen = True
        if not self.common_tls:
            self.note(f'The SUMO net ({net["file"]}) has no traffic lights linked to CARLA\'s: SUMO\'s vehicles do not '
                      'stop at CARLA\'s lights' + (' (CARLA\'s bundled nets have none; net "generate" builds one '
                                                   'with them)' if net['kind'] == 'bundled' else '')
                      + ('; CARLA keeps running its own lights.' if tls == 'sumo' else '.'))
        try:
            ver = traci.getVersion()
        except Exception:                                  # noqa: BLE001
            ver = (None, '')
        self.versions.update(traci_api=ver[0], sumo=str(ver[1]).replace('SUMO ', '').strip() or None)
        try:
            step = float(traci.simulation.getDeltaT())
        except Exception:                                  # noqa: BLE001
            step = self.dt
        self.timing['launch_s'] = round(time.perf_counter() - t0, 3)
        self.info = {'net': net, 'demand': demand, 'cap': cap, 'step_s': step}
        if abs(step - self.sumo_dt) > 1e-9:
            raise RuntimeError(f'SUMO steps {step} s, not the {self.sumo_dt} s asked for.')
        if self.k > 1:
            self.note(f'SUMO keeps time in whole milliseconds: it steps every {self.k} simulation steps '
                      f'({1000 * self.sumo_dt:g} ms for CARLA\'s {1000 * self.dt:g} ms); in between its vehicles move '
                      'on along their heading at SUMO\'s speed (dead reckoning, exact on a straight road).')

    # ------------------------------------------------------------------ every step
    def _fatal(self, e, when):
        code = None
        if self.proc is not None:
            try:
                code = self.proc.wait(2.0)                 # (a SUMO that crashed: its exit code)
            except subprocess.TimeoutExpired:
                code = None
        return RuntimeError(f'SUMO stopped {when} at {self.t:.3f} s of its time'
                            + (f' (exit {code})' if code is not None else '') + f': {e}'
                            + (f'; its log: {_tail(self.log_path, 6)}' if _tail(self.log_path, 6) else ''))

    def before_tick(self):
        """Every k-th simulation step (k = 1 when CARLA's step is whole milliseconds) SUMO steps once and its vehicles
        are spawned, moved and destroyed in CARLA, CARLA's lights set by SUMO's; on the steps between, SUMO's vehicles
        move on along their heading at SUMO's speed."""
        traci = self.traci
        self.ticks += 1
        if self.ticks % self.k:
            t0 = time.perf_counter()
            self._dead_reckon(self.ticks % self.k)
            self.timing['sync_s'] += time.perf_counter() - t0
            return
        t0 = time.perf_counter()
        try:
            self.sumo.tick()
            self.t = float(traci.simulation.getTime())
            self.st['teleports'] += len(traci.simulation.getStartingTeleportIDList())
            self.st['collisions'] += len(traci.simulation.getCollisions())
        except (traci.exceptions.FatalTraCIError, traci.exceptions.TraCIException, OSError) as e:
            raise self._fatal(e, 'stepping') from None
        t1 = time.perf_counter()
        self.timing['step_s'] += t1 - t0
        BH, side = self.BH, self.side
        try:
            mine = set(self.carla2sumo.values())
            for sid in sorted(self.sumo.spawned_actors - mine):
                self._spawn_in_carla(sid)
            for sid in sorted(self.sumo.destroyed_actors):
                if sid in self.sumo2carla:
                    cid = self.sumo2carla.pop(sid)
                    self.speed.pop(sid, None)
                    self.base.pop(sid, None)
                    self.rows[sid]['arrived_t_s'] = round(self.t, 4)
                    self.st['arrived'] += 1
                    self.on_destroy(cid)
                    side.destroy_actor(cid)
            for sid in sorted(self.sumo2carla):
                sa = self.sumo.get_actor(sid)
                res = traci.vehicle.getSubscriptionResults(sid) or {}
                self.speed[sid] = float(res.get(traci.constants.VAR_SPEED, 0.0) or 0.0)
                tr = BH.get_carla_transform(sa.transform, sa.extent)
                self.base[sid] = tr
                side.synchronize_vehicle(self.sumo2carla[sid], tr, None)
            if self.opts.get('tls') == 'sumo' and self.frozen:
                for lid in self.common_tls:
                    st = self.sumo.get_traffic_light_state(lid)
                    if st is not None:
                        side.synchronize_traffic_light(lid, BH.get_carla_traffic_light_state(st))
        except traci.exceptions.FatalTraCIError as e:
            raise self._fatal(e, 'while its vehicles were read') from None
        self.timing['sync_s'] += time.perf_counter() - t1

    def _dead_reckon(self, r):
        """SUMO's vehicles r simulation steps after SUMO's last step: moved along their heading at SUMO's speed."""
        C = self.carla
        for sid in sorted(self.sumo2carla):
            tr = self.base.get(sid)
            if tr is None:
                continue
            d = self.speed.get(sid, 0.0) * r * self.dt
            y = math.radians(tr.rotation.yaw)
            loc = C.Location(tr.location.x + d * math.cos(y), tr.location.y + d * math.sin(y), tr.location.z)
            self.side.synchronize_vehicle(self.sumo2carla[sid], C.Transform(loc, tr.rotation), None)

    def _spawn_in_carla(self, sid):
        BH, side = self.BH, self.side
        self.sumo.subscribe(sid)
        sa = self.sumo.get_actor(sid)
        self.BHM.random = _Draw(self.seed, sid)            # BridgeHelper's colour / driver choice, per vehicle
        bp = BH.get_carla_blueprint(sa, False)
        row = {'sumo_id': sid, 'sumo_type': sa.type_id, 'vclass': getattr(sa.vclass, 'value', str(sa.vclass)),
               'actor_id': None, 'blueprint': None, 'spawned_t_s': round(self.t, 4), 'arrived_t_s': None}
        self.rows[sid] = row
        if bp is None:
            self.sumo.unsubscribe(sid)
            self.st['unsupported'] += 1                  # (no CARLA blueprint for its vType: actor_id stays null)
            return
        if bp.has_attribute('role_name'):
            bp.set_attribute('role_name', ROLE_SUMO)
        cid = side.spawn_actor(bp, BH.get_carla_transform(sa.transform, sa.extent))
        if cid == self.K.INVALID_ACTOR_ID:
            self.sumo.unsubscribe(sid)
            self.st['spawn_failed'] += 1                 # (CARLA refused the spawn)
            return
        self.sumo2carla[sid] = cid
        self.st['spawned'] += 1
        row.update(actor_id=int(cid), blueprint=bp.id)
        actor = self.world.get_actor(cid)
        if actor is not None:
            self.on_spawn(actor, sid)

    def after_tick(self, recording: bool = False):
        """The bridge's own CARLA vehicles (the ego, a scenario's) mirrored into SUMO; SUMO's lights set by CARLA's;
        how many SUMO vehicles are in CARLA (while recording)."""
        traci = self.traci
        BH, sumo = self.BH, self.sumo
        t0 = time.perf_counter()
        if (self.ticks + 1) % self.k:                      # SUMO does not step before the next tick: nothing to tell it
            self._count(recording)
            return
        try:
            cur = list(self.carla_vehicles())
            ids = {int(a.id) for a in cur}
            for aid in sorted(set(self.carla2sumo) - ids):
                try:
                    sumo.destroy_actor(self.carla2sumo.pop(aid))
                except traci.exceptions.TraCIException:
                    pass
            for a in cur:
                aid = int(a.id)
                if aid in self.carla2sumo:
                    continue
                type_id = BH.get_sumo_vtype(a)
                if type_id is None:
                    continue
                sid = sumo.spawn_actor(type_id, None)
                if sid != self.K.INVALID_ACTOR_ID:
                    self.carla2sumo[aid] = sid
                    sumo.subscribe(sid)
                    self.mirrored.append({'actor_id': aid, 'sumo_id': sid, 'blueprint': a.type_id,
                                          'role': _role(a), 'sumo_type': type_id})
            for a in cur:
                sid = self.carla2sumo.get(int(a.id))
                if sid is None:
                    continue
                try:
                    sumo.synchronize_vehicle(sid, BH.get_sumo_transform(a.get_transform(), a.bounding_box.extent),
                                             None)
                except traci.exceptions.TraCIException as e:
                    self.st['mirror_errors'] += 1
                    if self.st['mirror_errors'] == 1:
                        self.note(f'SUMO did not take a CARLA vehicle\'s pose ({e}): it keeps its own there.')
            if self.opts.get('tls') == 'carla':
                for lid in self.common_tls:
                    st = self.side.get_traffic_light_state(lid)
                    if st is not None:
                        sumo.synchronize_traffic_light(lid, BH.get_sumo_traffic_light_state(st))
        except traci.exceptions.FatalTraCIError as e:
            raise self._fatal(e, 'while CARLA\'s vehicles were mirrored') from None
        self.timing['sync_s'] += time.perf_counter() - t0
        self._count(recording)

    def _count(self, recording):
        if recording:
            n = len(self.sumo2carla)
            if self.st['at_start'] is None:
                self.st['at_start'] = n
            self.st['peak'] = max(self.st['peak'], n)
            self.st['sum'] += n
            self.st['steps'] += 1

    # ------------------------------------------------------------------ what the bridge reads
    def is_sumo(self, actor_id) -> bool:
        return int(actor_id) in self._carla_ids()

    def _carla_ids(self) -> set:
        return set(self.sumo2carla.values())

    def sumo_id(self, actor_id):
        for sid, cid in self.sumo2carla.items():
            if cid == int(actor_id):
                return sid
        return None

    def velocity(self, actor_id, yaw_deg: float):
        """A SUMO vehicle's velocity in CARLA's axes (m/s) from SUMO's speed along its heading, or None."""
        sid = self.sumo_id(actor_id)
        if sid is None:
            return None
        v = self.speed.get(sid, 0.0)
        y = math.radians(yaw_deg)
        return (v * math.cos(y), v * math.sin(y), 0.0)

    def report(self) -> dict:
        """run.json traffic_source."""
        net = dict(self.info.get('net') or {})
        net.pop('path', None)
        net.pop('config', None)
        steps = self.st['steps']
        return {
            'source': 'sumo', 'sumo_version': self.versions.get('sumo'),
            'package_version': self.versions.get('eclipse_sumo'), 'traci_api': self.versions.get('traci_api'),
            'binary': os.path.basename(self.cmd[0]) if getattr(self, 'cmd', None) else None,
            'net': net, 'demand': self.info.get('demand'), 'seed': self.seed,
            'step_s': self.info.get('step_s', self.sumo_dt), 'steps_per_sumo_step': self.k,
            'max_vehicles': self.info.get('cap'),
            'tls': {'owner': self.opts.get('tls', 'carla'), 'linked': len(self.common_tls),
                    'frozen_in_carla': self.frozen},
            'vehicles_spawned': self.st['spawned'],
            'vehicles_in_carla': {'at_start': self.st['at_start'] or 0, 'peak': self.st['peak'],
                                  'mean': round(self.st['sum'] / steps, 3) if steps else 0.0,
                                  'note': 'SUMO\'s vehicles in CARLA at each step from t = 0 to the end'},
            'arrived': self.st['arrived'], 'not_spawned': self.st['unsupported'] + self.st['spawn_failed'],
            'teleports': self.st['teleports'], 'collisions': self.st['collisions'],
            'mirrored': list(self.mirrored),
            'actors': [self.rows[k] for k in sorted(self.rows, key=_natural)],
            'timing': {k: round(v, 3) for k, v in self.timing.items()},
            'walkers': WALKERS_NOTE, 'limits': LIMITS_TEXT,
        }

    # ------------------------------------------------------------------ the end
    def close(self, ok: bool = True):
        """Always (also after an error or a cancel): CARLA's lights given back (unfrozen), the TraCI connection
        closed, SUMO stopped (killed when it does not end), SUMO's files removed after a failed run (its log kept).
        The CARLA vehicles SUMO drove are the bridge's to destroy (it owns them)."""
        if self.closed:
            return
        self.closed = True
        if self.frozen and self.side is not None:
            for tl in list(getattr(self.side, '_tls', {}).values()):
                try:
                    tl.freeze(False)
                except Exception:                          # noqa: BLE001
                    pass
            try:
                self.side.close()                          # CarlaSimulation.close: every light unfrozen
            except Exception:                              # noqa: BLE001
                pass
        if self.started:
            try:
                self.traci.close()
            except Exception:                              # noqa: BLE001 (SUMO gone already)
                pass
        self.kill(wait=5.0)
        if not ok:
            for f in ('run.sumocfg', 'demand.rou.xml', 'trips.xml', 'randomTrips.log', 'netconvert.log'):
                p = os.path.join(self.work_dir, f)
                if os.path.exists(p):
                    os.remove(p)

    def kill(self, wait: float = 0.0):
        """SUMO's process ended: terminated, killed when it does not go within wait seconds."""
        p = self.proc
        if p is None or p.poll() is not None:
            return
        try:
            p.terminate()
            p.wait(max(wait, 0.1))
        except subprocess.TimeoutExpired:
            p.kill()
            try:
                p.wait(5)
            except subprocess.TimeoutExpired:
                pass
        except OSError:
            pass


class _Draw:
    """Stands in for the random module inside CARLA's bridge_helper (it calls random.choice for a blueprint's colour
    and driver): draws from numpy's generator seeded by the drive's seed and the SUMO vehicle's id, so a vehicle's
    colour depends on neither the order of spawns nor Python's global generator."""

    def __init__(self, seed: int, sid: str):
        import numpy as np
        self.rng = np.random.default_rng([int(seed), zlib.crc32(str(sid).encode('utf-8'))])

    def choice(self, seq):
        seq = list(seq)
        return seq[int(self.rng.integers(len(seq)))]


def listening(port: int):
    """Whether a socket listens on the local TCP port (Linux: /proc/net/tcp*), None when that cannot be read. (A test
    connection would not do: SUMO takes the first client that connects as its TraCI client and ends when it leaves.)"""
    seen = False
    for f in ('/proc/net/tcp', '/proc/net/tcp6'):
        try:
            with open(f, encoding='ascii') as fh:
                rows = fh.read().splitlines()[1:]
        except OSError:
            continue
        seen = True
        for r in rows:
            p = r.split()
            if len(p) > 3 and p[3] == '0A' and int(p[1].rsplit(':', 1)[1], 16) == int(port):
                return True
    return False if seen else None


def _role(actor) -> str:
    try:
        return str((getattr(actor, 'attributes', None) or {}).get('role_name', ''))
    except Exception:                                      # noqa: BLE001
        return ''


def _natural(s):
    """'veh2' before 'veh10'."""
    import re
    return [int(x) if x.isdigit() else x for x in re.split(r'(\d+)', str(s))]


def remove_tree(path):
    shutil.rmtree(path, ignore_errors=True)
