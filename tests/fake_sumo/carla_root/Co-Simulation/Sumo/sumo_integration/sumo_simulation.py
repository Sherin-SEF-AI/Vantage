"""SumoSimulation as CARLA's: traci.init on a running SUMO, the net from the configuration, departed / arrived vehicles
per step, subscriptions, a traffic-light manager keyed by CARLA landmark ids (the nets' linkSignalID parameters)."""
import collections
import enum
import os
import xml.etree.ElementTree as ET

import carla  # noqa: F401 (as CARLA's: the transforms are carla's)
import sumolib
import traci

from .constants import INVALID_ACTOR_ID


class SumoSignalState(object):
    RED = 'r'
    YELLOW = 'y'
    GREEN = 'G'
    GREEN_WITHOUT_PRIORITY = 'g'
    GREEN_RIGHT_TURN = 's'
    RED_YELLOW = 'u'
    OFF_BLINKING = 'o'
    OFF = 'O'


class SumoVehSignal(object):
    BLINKER_RIGHT = 1 << 0
    BLINKER_LEFT = 1 << 1
    BLINKER_EMERGENCY = 1 << 2
    BRAKELIGHT = 1 << 3


class SumoActorClass(enum.Enum):
    PASSENGER = 'passenger'
    EVEHICLE = 'evehicle'
    AUTHORITY = 'authority'
    TRUCK = 'truck'
    MOTORCYCLE = 'motorcycle'
    BICYCLE = 'bicycle'


SumoActor = collections.namedtuple('SumoActor', 'type_id vclass transform signals extent color')


class SumoTLManager(object):
    def __init__(self):
        self._tls = {}
        self._phase = {}
        for tlid in traci.trafficlight.getIDList():
            traci.trafficlight.subscribe(tlid, [traci.constants.TL_CURRENT_PROGRAM, traci.constants.TL_CURRENT_PHASE])
            logic = traci.trafficlight.getAllProgramLogics(tlid)[0]
            states = [p.state for p in logic.getPhases()]
            links = {}
            for k, lm in logic.getParameters().items():
                links.setdefault(lm, []).append(int(k.split(':')[1]))
            self._tls[tlid] = (states, links)
            self._phase[tlid] = traci.trafficlight.getPhase(tlid)
        self._off = False

    def get_all_landmarks(self):
        return {lm for _, links in self._tls.values() for lm in links}

    def get_state(self, landmark_id):
        states = set()
        for tlid, (st, links) in self._tls.items():
            for i in links.get(landmark_id, []):
                states.add(st[self._phase[tlid]][i])
        if len(states) == 1:
            return states.pop()
        return SumoSignalState.RED if states else None

    def set_state(self, landmark_id, state):
        for tlid, (st, links) in self._tls.items():
            for i in links.get(landmark_id, []):
                traci.trafficlight.setLinkState(tlid, i, state)
        return True

    def switch_off(self):
        for tlid, (st, links) in self._tls.items():
            for i in range(len(st[0]) if st else 0):
                traci.trafficlight.setLinkState(tlid, i, SumoSignalState.OFF)
        self._off = True

    def tick(self):
        if not self._off:
            for tlid in traci.trafficlight.getIDList():
                self._phase[tlid] = traci.trafficlight.getSubscriptionResults(tlid)[traci.constants.TL_CURRENT_PHASE]


def _get_sumo_net(cfg_file):
    cfg_file = os.path.join(os.getcwd(), cfg_file)
    tag = ET.parse(cfg_file).find('.//net-file')
    if tag is None:
        return None
    return sumolib.net.readNet(os.path.join(os.path.dirname(cfg_file), tag.get('value')))


class SumoSimulation(object):
    def __init__(self, cfg_file, step_length, host=None, port=None, sumo_gui=False, client_order=1):
        if host is None or port is None:
            raise RuntimeError('the stand-in only connects to a running SUMO')
        traci.init(host=host, port=port)
        traci.setOrder(client_order)
        self.net = _get_sumo_net(cfg_file)
        self._routes = set()
        self._sequential_id = 0
        self.spawned_actors = set()
        self.destroyed_actors = set()
        self.traffic_light_manager = SumoTLManager()

    @property
    def traffic_light_ids(self):
        return self.traffic_light_manager.get_all_landmarks()

    @staticmethod
    def subscribe(actor_id):
        traci.vehicle.subscribe(actor_id, [
            traci.constants.VAR_TYPE, traci.constants.VAR_VEHICLECLASS, traci.constants.VAR_COLOR,
            traci.constants.VAR_LENGTH, traci.constants.VAR_WIDTH, traci.constants.VAR_HEIGHT,
            traci.constants.VAR_POSITION3D, traci.constants.VAR_ANGLE, traci.constants.VAR_SLOPE,
            traci.constants.VAR_SPEED, traci.constants.VAR_SPEED_LAT, traci.constants.VAR_SIGNALS])

    @staticmethod
    def unsubscribe(actor_id):
        traci.vehicle.unsubscribe(actor_id)

    def get_net_offset(self):
        return (0, 0) if self.net is None else self.net.getLocationOffset()

    @staticmethod
    def get_actor(actor_id):
        r = traci.vehicle.getSubscriptionResults(actor_id)
        C = traci.constants
        loc = list(r[C.VAR_POSITION3D])
        tr = carla.Transform(carla.Location(loc[0], loc[1], loc[2]),
                             carla.Rotation(r[C.VAR_SLOPE], r[C.VAR_ANGLE], 0.0))
        extent = carla.Vector3D(r[C.VAR_LENGTH] / 2.0, r[C.VAR_WIDTH] / 2.0, r[C.VAR_HEIGHT] / 2.0)
        return SumoActor(r[C.VAR_TYPE], SumoActorClass(r[C.VAR_VEHICLECLASS]), tr, r[C.VAR_SIGNALS], extent,
                         r[C.VAR_COLOR])

    def spawn_actor(self, type_id, color=None):
        actor_id = 'carla' + str(self._sequential_id)
        try:
            vclass = traci.vehicletype.getVehicleClass(type_id)
            if vclass not in self._routes:
                edges = [e for e in self.net.getEdges() if e.allows(vclass)]
                if not edges:
                    return INVALID_ACTOR_ID
                traci.route.add('carla_route_{}'.format(vclass), [edges[0].getID()])
                self._routes.add(vclass)
            traci.vehicle.add(actor_id, 'carla_route_{}'.format(vclass), typeID=type_id)
        except traci.exceptions.TraCIException:
            return INVALID_ACTOR_ID
        self._sequential_id += 1
        return actor_id

    @staticmethod
    def destroy_actor(actor_id):
        traci.vehicle.remove(actor_id)

    def get_traffic_light_state(self, landmark_id):
        return self.traffic_light_manager.get_state(landmark_id)

    def switch_off_traffic_lights(self):
        self.traffic_light_manager.switch_off()

    def synchronize_vehicle(self, vehicle_id, transform, signals=None):
        traci.vehicle.moveToXY(vehicle_id, '', 0, transform.location.x, transform.location.y,
                               angle=transform.rotation.yaw, keepRoute=2)
        return True

    def synchronize_traffic_light(self, landmark_id, state):
        self.traffic_light_manager.set_state(landmark_id, state)

    def tick(self):
        traci.simulationStep()
        self.traffic_light_manager.tick()
        self.spawned_actors = set(traci.simulation.getDepartedIDList())
        self.destroyed_actors = set(traci.simulation.getArrivedIDList())

    @staticmethod
    def close():
        traci.close()
