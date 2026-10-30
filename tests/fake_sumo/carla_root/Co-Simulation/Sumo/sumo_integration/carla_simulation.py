"""CarlaSimulation as CARLA's: spawn with physics off (a batch, raised SPAWN_OFFSET_Z), move, destroy, traffic lights by
landmark id."""
import carla

from .constants import INVALID_ACTOR_ID, SPAWN_OFFSET_Z


class CarlaSimulation(object):
    def __init__(self, host, port, step_length):
        self.client = carla.Client(host, port)
        self.client.set_timeout(2.0)
        self.world = self.client.get_world()
        self.blueprint_library = self.world.get_blueprint_library()
        self.step_length = step_length
        self._active_actors, self.spawned_actors, self.destroyed_actors = set(), set(), set()
        self._tls = {}
        for landmark in self.world.get_map().get_all_landmarks_of_type('1000001'):
            if landmark.id != '':
                tl = self.world.get_traffic_light(landmark)
                if tl is not None:
                    self._tls[landmark.id] = tl

    def get_actor(self, actor_id):
        return self.world.get_actor(actor_id)

    @property
    def traffic_light_ids(self):
        return set(self._tls.keys())

    def get_traffic_light_state(self, landmark_id):
        return self._tls[landmark_id].state if landmark_id in self._tls else None

    def switch_off_traffic_lights(self):
        for actor in self.world.get_actors():
            if actor.type_id == 'traffic.traffic_light':
                actor.freeze(True)
                actor.set_state(carla.TrafficLightState.Green)

    def spawn_actor(self, blueprint, transform):
        transform = carla.Transform(transform.location + carla.Location(0, 0, SPAWN_OFFSET_Z), transform.rotation)
        batch = [carla.command.SpawnActor(blueprint, transform).then(
            carla.command.SetSimulatePhysics(carla.command.FutureActor, False))]
        response = self.client.apply_batch_sync(batch, False)[0]
        if response.error:
            return INVALID_ACTOR_ID
        return response.actor_id

    def destroy_actor(self, actor_id):
        actor = self.world.get_actor(actor_id)
        if actor is not None:
            return actor.destroy()
        return False

    def synchronize_vehicle(self, vehicle_id, transform, lights=None):
        vehicle = self.world.get_actor(vehicle_id)
        if vehicle is None:
            return False
        vehicle.set_transform(transform)
        return True

    def synchronize_traffic_light(self, landmark_id, state):
        if landmark_id not in self._tls:
            return False
        self._tls[landmark_id].set_state(state)
        return True

    def tick(self):
        self.world.tick()

    def close(self):
        for actor in self.world.get_actors():
            if actor.type_id == 'traffic.traffic_light':
                actor.freeze(False)
