"""A fake TraCI client (the part of SUMO 1.23's traci the CARLA co-simulation uses) for tests without SUMO.

With tests/fake_sumo on sys.path, `import traci` gets this. traci.init connects to the fake SUMO process
(tests/fake_sumo/sumo_home/bin/sumo) on its --remote-port, which sends its command line and then only holds the socket
open: the simulation itself runs here, from that command line (-c run.sumocfg, --seed, --step-length,
--max-num-vehicles) and the files it names:

* the net (tests/fake_sumo nets: straight edges along CARLA's fake road, a net offset, optionally one traffic light
  linked to CARLA's landmark '1000');
* the demand: vTypes and trips (from, depart, type, departPos="random"); a trip departs at its time when fewer than
  --max-num-vehicles are in the network (the mirrored CARLA vehicles count), at a random place of its edge (drawn from
  --seed, at least 8 m from the vehicles there), and drives along its edge at a speed drawn from --seed until the edge
  ends (it arrives);
* vehicles added by traci.vehicle.add (the mirrored CARLA vehicles) depart at the next step and go where moveToXY puts
  them.

Positions are SUMO's: the front bumper's centre, angle in degrees clockwise from north. When the fake SUMO process
dies, the next call that talks to it raises FatalTraCIError, as the real client does. LOG records every command."""
from __future__ import annotations

import json
import math
import os
import random
import socket
import time
import xml.etree.ElementTree as ET

from . import constants, exceptions
from .exceptions import FatalTraCIError, TraCIException

LOG = []
_S = None
DEFAULT_TYPES = {'DEFAULT_VEHTYPE': {'vClass': 'passenger', 'length': 5.0, 'width': 1.8, 'height': 1.5},
                 'DEFAULT_BIKETYPE': {'vClass': 'bicycle', 'length': 1.6, 'width': 0.65, 'height': 1.7}}


def _opts(argv):
    out = {}
    i = 0
    while i < len(argv):
        a = argv[i]
        if a.startswith('-') and i + 1 < len(argv) and not argv[i + 1].startswith('--'):
            out[a.lstrip('-')] = argv[i + 1]
            i += 2
        else:
            out[a.lstrip('-')] = True
            i += 1
    return out


def read_net(path):
    """{'offset': (x, y), 'edges': {id: {'shape', 'speed', 'allow'}}, 'tls': {id: {'phases', 'params'}}}."""
    root = ET.parse(path).getroot()
    loc = root.find('location')
    off = tuple(float(v) for v in (loc.get('netOffset') if loc is not None else '0,0').split(','))
    edges, tls = {}, {}
    for e in root.iter('edge'):
        if e.get('function') == 'internal':
            continue
        lane = e.find('lane')
        shape = [tuple(float(v) for v in p.split(',')) for p in lane.get('shape').split()]
        edges[e.get('id')] = {'shape': shape, 'speed': float(lane.get('speed', 13.89)),
                              'allow': (lane.get('allow') or '').split()}
    for t in root.iter('tlLogic'):
        tls[t.get('id')] = {'phases': [p.get('state') for p in t.iter('phase')],
                            'durations': [float(p.get('duration')) for p in t.iter('phase')],
                            'params': {p.get('key'): p.get('value') for p in t.iter('param')}}
    return {'offset': off, 'edges': edges, 'tls': tls}


class _Sim:
    def __init__(self, sock, argv):
        self.sock, self.argv = sock, argv
        o = _opts(argv[1:])
        self.opts = o
        cfg = o['c']
        croot = ET.parse(cfg).getroot()
        d = os.path.dirname(cfg)
        net = croot.find('.//net-file').get('value')
        self.net = read_net(os.path.join(d, net))
        self.types = {k: dict(v) for k, v in DEFAULT_TYPES.items()}
        self.pending = []
        rf = croot.find('.//route-files')
        for f in (rf.get('value').split(',') if rf is not None else []):
            r = ET.parse(os.path.join(d, f.strip())).getroot()
            for vt in r.iter('vType'):
                self.types[vt.get('id')] = {'vClass': vt.get('vClass', 'passenger'),
                                            'length': float(vt.get('length', 5.0)),
                                            'width': float(vt.get('width', 1.8)),
                                            'height': float(vt.get('height', 1.5))}
            for t in r:
                if t.tag in ('trip', 'vehicle'):
                    self.pending.append({'id': t.get('id'), 'depart': float(t.get('depart')), 'edge': t.get('from'),
                                         'type': t.get('type', 'DEFAULT_VEHTYPE')})
        self.pending.sort(key=lambda p: p['depart'])
        self.dt = float(o.get('step-length', 1.0))
        self.t = float(o.get('begin', 0.0))
        self.cap = int(o.get('max-num-vehicles', 10 ** 9))
        self.rng = random.Random(int(o.get('seed', 23423)))
        self.veh = {}                     # id -> {'type', 'edge', 's', 'speed', 'x', 'y', 'angle', 'mirrored'}
        self.added = []                   # traci.vehicle.add: depart at the next step
        self.placed = {}                  # an added vehicle's moveToXY before it departed
        self.departed, self.arrived = [], []
        self.subs, self.tl_subs = {}, set()
        self.tl_set = {}                  # (tlid, link) -> state set by setLinkState
        self.routes = {}

    def alive(self):
        try:
            self.sock.setblocking(False)
            data = self.sock.recv(1)
        except BlockingIOError:
            return
        except OSError as e:
            raise FatalTraCIError(f'Connection closed by SUMO ({e}).')
        finally:
            try:
                self.sock.setblocking(True)
            except OSError:
                pass
        if data == b'':
            raise FatalTraCIError('Connection closed by SUMO.')

    def _place(self, v):
        e = self.net['edges'][v['edge']]
        (x0, y0), (x1, y1) = e['shape'][0], e['shape'][-1]
        L = math.hypot(x1 - x0, y1 - y0)
        f = min(v['s'] / L, 1.0)
        v['x'], v['y'] = x0 + f * (x1 - x0), y0 + f * (y1 - y0)
        v['angle'] = (90.0 - math.degrees(math.atan2(y1 - y0, x1 - x0))) % 360.0
        return L

    def step(self):
        self.alive()
        self.steps = getattr(self, 'steps', 0) + 1
        crash = os.environ.get('FAKE_SUMO_CRASH_AT_STEP')
        if crash and self.steps >= int(crash):          # SUMO dies (the fake process exits 139): the socket closes
            self.sock.sendall(b'x')
            self.sock.recv(1)
            raise FatalTraCIError('Connection closed by SUMO.')
        self.t = round(self.t + self.dt, 9)
        self.departed, self.arrived = [], []
        for vid in sorted(self.veh):
            v = self.veh[vid]
            if v['mirrored']:
                continue
            v['s'] += v['speed'] * self.dt
            if v['s'] >= self._place(v):
                self.arrived.append(vid)
        for vid in self.arrived:
            del self.veh[vid]
            self.subs.pop(vid, None)
        for vid, tid in self.added:
            self.veh[vid] = dict({'type': tid, 'edge': None, 's': 0.0, 'speed': 0.0, 'x': 0.0, 'y': 0.0, 'angle': 90.0,
                                  'mirrored': True}, **self.placed.pop(vid, {}))
            self.departed.append(vid)
        self.added = []
        while self.pending and self.pending[0]['depart'] <= self.t + 1e-9 and len(self.veh) < self.cap:
            p = self.pending.pop(0)
            if p['edge'] not in self.net['edges']:
                continue
            L = math.hypot(*(a - b for a, b in zip(self.net['edges'][p['edge']]['shape'][-1],
                                                   self.net['edges'][p['edge']]['shape'][0])))
            on = [w['s'] for w in self.veh.values() if w['edge'] == p['edge']]
            s = None
            for _ in range(10):
                c = self.rng.uniform(0.0, 0.7 * L)
                if all(abs(c - q) > 8.0 for q in on):
                    s = c
                    break
            if s is None:
                continue
            v = {'type': p['type'], 'edge': p['edge'], 's': s, 'speed': self.rng.uniform(6.0, 12.0), 'mirrored': False}
            self._place(v)
            self.veh[p['id']] = v
            self.departed.append(p['id'])

    def results(self, vid):
        v = self.veh.get(vid)
        if v is None or vid not in self.subs:
            return {}
        t = self.types.get(v['type'], DEFAULT_TYPES['DEFAULT_VEHTYPE'])
        C = constants
        return {C.VAR_TYPE: v['type'], C.VAR_VEHICLECLASS: t['vClass'], C.VAR_COLOR: (255, 255, 0, 255),
                C.VAR_LENGTH: t['length'], C.VAR_WIDTH: t['width'], C.VAR_HEIGHT: t['height'],
                C.VAR_POSITION3D: (v['x'], v['y'], 0.0), C.VAR_ANGLE: v['angle'], C.VAR_SLOPE: 0.0,
                C.VAR_SPEED: v['speed'], C.VAR_SPEED_LAT: 0.0, C.VAR_SIGNALS: 0}

    def phase(self, tlid):
        tl = self.net['tls'][tlid]
        cyc = sum(tl['durations'])
        t = self.t % cyc
        for i, d in enumerate(tl['durations']):
            if t < d:
                return i
            t -= d
        return 0


def _sim():
    if _S is None:
        raise FatalTraCIError('Not connected.')
    return _S


# ------------------------------------------------------------------ the connection
def init(port=8813, numRetries=60, host='localhost', label='default', proc=None, doSwitch=True, traceFile=None,
         traceGetters=True):
    global _S
    if _S is not None:
        raise TraCIException(f'Connection {label!r} is already active.')
    sock = None
    for _ in range(max(int(numRetries), 1) * 20):
        try:
            sock = socket.create_connection((host, int(port)), 1.0)
            break
        except OSError:
            time.sleep(0.05)
    if sock is None:
        raise FatalTraCIError(f'Could not connect to {host}:{port}.')
    buf = b''
    while not buf.endswith(b'\n'):
        chunk = sock.recv(65536)
        if not chunk:
            raise FatalTraCIError('Connection closed by SUMO.')
        buf += chunk
    argv = json.loads(buf.decode())
    _S = _Sim(sock, argv)
    LOG.append(('init', int(port), argv))
    return 21, 'SUMO 1.23.1'


def start(cmd, port=None, **kw):
    raise TraCIException('the fake traci only connects (traci.init) to a fake SUMO already started')


def close(wait=True):
    global _S
    if _S is not None:
        try:
            _S.sock.close()
        except OSError:
            pass
    _S = None
    LOG.append(('close',))


def setOrder(order):
    LOG.append(('order', order))


def getVersion():
    _sim()
    return 21, 'SUMO 1.23.1'


def simulationStep(step=0.0):
    s = _sim()
    s.step()
    LOG.append(('step', s.t))


# ------------------------------------------------------------------ domains
class _Simulation:
    def getDepartedIDList(self):
        return list(_sim().departed)

    def getArrivedIDList(self):
        return list(_sim().arrived)

    def getStartingTeleportIDList(self):
        _sim()
        return []

    def getCollisions(self):
        _sim()
        return []

    def getTime(self):
        return _sim().t

    def getDeltaT(self):
        return _sim().dt

    def getMinExpectedNumber(self):
        s = _sim()
        return len(s.veh) + len(s.pending)


class _Vehicle:
    def subscribe(self, vid, varIDs=(), begin=None, end=None):
        _sim().subs[vid] = list(varIDs)

    def unsubscribe(self, vid):
        _sim().subs.pop(vid, None)

    def getSubscriptionResults(self, vid):
        return _sim().results(vid)

    def getIDList(self):
        return sorted(_sim().veh)

    def add(self, vehID, routeID, typeID='DEFAULT_VEHTYPE', **kw):
        s = _sim()
        if vehID in s.veh or any(v == vehID for v, _ in s.added):
            raise TraCIException(f'The vehicle {vehID!r} to add already exists.')
        s.added.append((vehID, typeID))
        LOG.append(('add', vehID, routeID, typeID))

    def remove(self, vehID, reason=3):
        s = _sim()
        if vehID not in s.veh:
            raise TraCIException(f'Vehicle {vehID!r} is not known.')
        del s.veh[vehID]
        s.subs.pop(vehID, None)
        LOG.append(('remove', vehID))

    def moveToXY(self, vehID, edgeID, laneIndex, x, y, angle=-1073741824.0, keepRoute=1, matchThreshold=100):
        s = _sim()
        v = s.veh.get(vehID)
        if v is None and any(a == vehID for a, _ in s.added):      # added, not yet departed: it departs there
            v = s.placed.setdefault(vehID, {})
        if v is None:
            raise TraCIException(f'Vehicle {vehID!r} is not known.')
        v['x'], v['y'], v['angle'] = float(x), float(y), float(angle)
        LOG.append(('move', vehID, round(float(x), 4), round(float(y), 4), round(float(angle), 4), keepRoute))

    def setSignals(self, vehID, signals):
        LOG.append(('signals', vehID, signals))

    def setColor(self, vehID, color):
        LOG.append(('color', vehID, tuple(color)))

    def getSpeed(self, vehID):
        return _sim().veh[vehID]['speed']


class _VehicleType:
    def getIDList(self):
        return sorted(_sim().types)

    def copy(self, origTypeID, newTypeID):
        s = _sim()
        s.types[newTypeID] = dict(s.types[origTypeID])
        LOG.append(('vtype_copy', origTypeID, newTypeID))

    def getVehicleClass(self, typeID):
        return _sim().types[typeID]['vClass']

    def setVehicleClass(self, typeID, clazz):
        _sim().types[typeID]['vClass'] = clazz

    def setShapeClass(self, typeID, shape):
        _sim().types[typeID]['shape'] = shape

    def getShapeClass(self, typeID):
        return _sim().types[typeID].get('shape', '')

    def setColor(self, typeID, color):
        _sim().types[typeID]['color'] = tuple(color)

    def getColor(self, typeID):
        return _sim().types[typeID].get('color', (255, 255, 0, 255))

    def setLength(self, typeID, v):
        _sim().types[typeID]['length'] = float(v)

    def setWidth(self, typeID, v):
        _sim().types[typeID]['width'] = float(v)

    def setHeight(self, typeID, v):
        _sim().types[typeID]['height'] = float(v)

    def getLength(self, typeID):
        return _sim().types[typeID]['length']

    def getWidth(self, typeID):
        return _sim().types[typeID]['width']

    def getHeight(self, typeID):
        return _sim().types[typeID]['height']


class _Route:
    def add(self, routeID, edges):
        s = _sim()
        if routeID in s.routes:
            raise TraCIException(f'Could not add route {routeID!r}.')
        s.routes[routeID] = list(edges)
        LOG.append(('route', routeID, list(edges)))


class _Phase:
    def __init__(self, state, duration):
        self.state, self.duration = state, duration


class _Logic:
    def __init__(self, tl):
        self.programID = '0'
        self._tl = tl

    def getPhases(self):
        return [_Phase(s, d) for s, d in zip(self._tl['phases'], self._tl['durations'])]

    def getParameters(self):
        return dict(self._tl['params'])


class _TrafficLight:
    def getIDList(self):
        return sorted(_sim().net['tls'])

    def subscribe(self, tlid, varIDs=(), begin=None, end=None):
        _sim().tl_subs.add(tlid)

    def unsubscribe(self, tlid):
        _sim().tl_subs.discard(tlid)

    def getAllProgramLogics(self, tlid):
        return [_Logic(_sim().net['tls'][tlid])]

    def getProgram(self, tlid):
        return '0'

    def getPhase(self, tlid):
        return _sim().phase(tlid)

    def getSubscriptionResults(self, tlid):
        s = _sim()
        return {constants.TL_CURRENT_PROGRAM: '0', constants.TL_CURRENT_PHASE: s.phase(tlid)}

    def setLinkState(self, tlid, tlsLinkIndex, state):
        _sim().tl_set[(tlid, int(tlsLinkIndex))] = state
        LOG.append(('tl_link', tlid, int(tlsLinkIndex), state))


simulation = _Simulation()
vehicle = _Vehicle()
vehicletype = _VehicleType()
route = _Route()
trafficlight = _TrafficLight()
