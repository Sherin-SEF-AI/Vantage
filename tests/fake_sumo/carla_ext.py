"""What CARLA's co-simulation calls of the carla module beyond what tests/fake_carla has, added to it for a test (through
pytest's monkeypatch, so it is undone after): Client.apply_batch_sync with command.SpawnActor(...).then(
command.SetSimulatePhysics(FutureActor, False)), TrafficLightState, the map's traffic-light landmarks (one, id '1000',
the fake world's light), BlueprintLibrary[i], World.get_traffic_light, TrafficLight state / set_state / freeze, Map.to_opendrive. Spawns go
through the fake World.spawn_actor (logged there); the rest is logged in the fake's LOG as ('batch', ...),
('tl_state', ...), ('tl_freeze', ...)."""
import types

LANDMARK = '1000'


class _State:
    Red, Yellow, Green, Off, Unknown = 'Red', 'Yellow', 'Green', 'Off', 'Unknown'


class _FutureActor:
    pass


class _SpawnActor:
    def __init__(self, blueprint, transform):
        self.blueprint, self.transform, self.after = blueprint, transform, []

    def then(self, cmd):
        self.after.append(cmd)
        return self


class _SetSimulatePhysics:
    def __init__(self, actor, enabled):
        self.actor, self.enabled = actor, enabled


class _Response:
    def __init__(self, actor_id=0, error=''):
        self.actor_id, self.error = actor_id, error

    def has_error(self):
        return bool(self.error)


class _Landmark:
    def __init__(self, lid):
        self.id, self.name, self.road_id, self.s = lid, 'light', 1, 10.0


def extend(fc, monkeypatch, no_landmarks=False):
    def apply_batch_sync(self, batch, do_tick=False):
        w = self.get_world()
        out = []
        for c in batch:
            try:
                a = w.spawn_actor(c.blueprint, c.transform)
            except RuntimeError as e:
                out.append(_Response(0, str(e)))
                continue
            for after in c.after:
                if isinstance(after, _SetSimulatePhysics):
                    a.set_simulate_physics(after.enabled)
            fc.LOG.append(('batch', a.id, c.blueprint.id))
            out.append(_Response(a.id, ''))
        return out

    def landmarks(self, type_id):
        return [] if no_landmarks or str(type_id) != '1000001' else [_Landmark(LANDMARK)]

    def get_traffic_light(self, landmark):
        return self._light if getattr(landmark, 'id', None) == LANDMARK else None

    def set_state(self, state):
        self.state = state
        fc.LOG.append(('tl_state', self.id, state))

    def freeze(self, on):
        self.frozen = bool(on)
        fc.LOG.append(('tl_freeze', self.id, bool(on)))

    def to_opendrive(self):
        return f'<OpenDRIVE><header name="{self.name.rsplit("/", 1)[-1]}"/></OpenDRIVE>'
    cmd = types.SimpleNamespace(SpawnActor=_SpawnActor, SetSimulatePhysics=_SetSimulatePhysics,
                                FutureActor=_FutureActor)
    monkeypatch.setattr(fc, 'command', cmd, raising=False)
    monkeypatch.setattr(fc.BlueprintLibrary, '__getitem__', lambda self, i: list(self)[i], raising=False)
    monkeypatch.setattr(fc, 'TrafficLightState', _State, raising=False)
    monkeypatch.setattr(fc.Client, 'apply_batch_sync', apply_batch_sync, raising=False)
    monkeypatch.setattr(fc.Map, 'get_all_landmarks_of_type', landmarks, raising=False)
    monkeypatch.setattr(fc.Map, 'to_opendrive', to_opendrive, raising=False)
    monkeypatch.setattr(fc.World, 'get_traffic_light', get_traffic_light, raising=False)
    monkeypatch.setattr(fc.TrafficLight, 'state', _State.Red, raising=False)
    monkeypatch.setattr(fc.TrafficLight, 'set_state', set_state, raising=False)
    monkeypatch.setattr(fc.TrafficLight, 'freeze', freeze, raising=False)
