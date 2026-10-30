"""BridgeHelper as CARLA's: SUMO <-> CARLA transforms (front bumper / centre, the net offset, y mirrored, SUMO's angle
clockwise from north), blueprints by vType id (else one of the vClass, random), vTypes for CARLA actors, light
states."""
import json
import logging
import math
import os
import random

import carla
import traci

from .sumo_simulation import SumoSignalState


class BridgeHelper(object):
    blueprint_library = []
    offset = (0, 0)
    _vtypes_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), '..', 'data', 'vtypes.json')
    with open(_vtypes_path) as f:
        _VTYPES = json.load(f)['carla_blueprints']

    @staticmethod
    def get_carla_transform(in_sumo_transform, extent):
        offset = BridgeHelper.offset
        loc, rot = in_sumo_transform.location, in_sumo_transform.rotation
        yaw = -1 * rot.yaw + 90
        pitch = rot.pitch
        out = (loc.x - math.cos(math.radians(yaw)) * extent.x, loc.y - math.sin(math.radians(yaw)) * extent.x,
               loc.z - math.sin(math.radians(pitch)) * extent.x)
        out = (out[0] - offset[0], out[1] - offset[1], out[2])
        return carla.Transform(carla.Location(out[0], -out[1], out[2]),
                               carla.Rotation(rot.pitch, rot.yaw - 90, rot.roll))

    @staticmethod
    def get_sumo_transform(in_carla_transform, extent):
        offset = BridgeHelper.offset
        loc, rot = in_carla_transform.location, in_carla_transform.rotation
        yaw = -1 * rot.yaw
        pitch = rot.pitch
        out = (loc.x + math.cos(math.radians(yaw)) * extent.x, loc.y - math.sin(math.radians(yaw)) * extent.x,
               loc.z - math.sin(math.radians(pitch)) * extent.x)
        out = (out[0] + offset[0], out[1] - offset[1], out[2])
        return carla.Transform(carla.Location(out[0], -out[1], out[2]),
                               carla.Rotation(rot.pitch, rot.yaw + 90, rot.roll))

    @staticmethod
    def get_carla_blueprint(sumo_actor, sync_color=False):
        lib = BridgeHelper.blueprint_library
        type_id = sumo_actor.type_id
        if type_id in [bp.id for bp in lib]:
            blueprint = lib.filter(type_id)[0]
        else:
            vclass = sumo_actor.vclass.value
            bps = [bp for bp in lib if bp.id in BridgeHelper._VTYPES and
                   BridgeHelper._VTYPES[bp.id]['vClass'] == vclass]
            if not bps:
                logging.error('sumo vtype %s not supported', type_id)
                return None
            blueprint = random.choice(bps)
        if blueprint.has_attribute('color'):
            blueprint.set_attribute('color', random.choice(['255,0,0', '0,255,0', '0,0,255']))
        blueprint.set_attribute('role_name', 'sumo_driver')
        return blueprint

    @staticmethod
    def get_sumo_vtype(carla_actor):
        type_id = carla_actor.type_id
        if not type_id.startswith('vehicle'):
            return None
        if type_id in traci.vehicletype.getIDList():
            return type_id
        attrs = carla_actor.attributes
        extent = carla_actor.bounding_box.extent
        traci.vehicletype.copy('DEFAULT_BIKETYPE' if int(attrs['number_of_wheels']) == 2 else 'DEFAULT_VEHTYPE',
                               type_id)
        if 'vClass' in BridgeHelper._VTYPES.get(type_id, {}):
            traci.vehicletype.setVehicleClass(type_id, BridgeHelper._VTYPES[type_id]['vClass'])
        traci.vehicletype.setLength(type_id, 2.0 * extent.x)
        traci.vehicletype.setWidth(type_id, 2.0 * extent.y)
        traci.vehicletype.setHeight(type_id, 2.0 * extent.z)
        return type_id

    @staticmethod
    def get_carla_traffic_light_state(sumo_tl_state):
        if sumo_tl_state in (SumoSignalState.RED, SumoSignalState.RED_YELLOW):
            return carla.TrafficLightState.Red
        if sumo_tl_state == SumoSignalState.YELLOW:
            return carla.TrafficLightState.Yellow
        if sumo_tl_state in (SumoSignalState.GREEN, SumoSignalState.GREEN_WITHOUT_PRIORITY):
            return carla.TrafficLightState.Green
        if sumo_tl_state == SumoSignalState.OFF:
            return carla.TrafficLightState.Off
        return carla.TrafficLightState.Unknown

    @staticmethod
    def get_sumo_traffic_light_state(carla_tl_state):
        if carla_tl_state == carla.TrafficLightState.Red:
            return SumoSignalState.RED
        if carla_tl_state == carla.TrafficLightState.Yellow:
            return SumoSignalState.YELLOW
        if carla_tl_state == carla.TrafficLightState.Green:
            return SumoSignalState.GREEN
        return SumoSignalState.OFF
