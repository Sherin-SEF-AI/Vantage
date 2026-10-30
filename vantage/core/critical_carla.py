"""Critical targets as CARLA scenes: every critical target of the project (core.critical) stood around the still ego in
CARLA as an actor of its kind, recorded with ground truth (the bridge's actors scene), and what each camera sees of it
there compared, camera by camera, with what the critical-targets study predicts.

Actors. A target becomes the CARLA 0.9.16 actor nearest it in kind and size (ACTORS.json, vantage.sim.carla_bridge):
* child: a child walker (blueprint attribute age = "child"), the one whose rendered height is nearest the target's;
  pedestrian: an adult walker likewise. Walkers stand still on the ground, facing the vehicle.
* car: the car of simrig.CARLA_VEHICLES nearest in height and width (a CARLA box's width includes the mirrors: 92 % of
  it is compared), at the target's heading.
* sign: CARLA has no road-sign prop of a sign plate's size (its three street-sign props are boards of 1.06 × 1.68,
  2.44 × 1.96 and 1.16 × 2.20 m; the map's signs are level geometry): the board nearest in size stands in, raised or
  lowered so its board's bottom is at the target's height above the ground, facing the vehicle (said per target).
* traffic_light: CARLA 0.9.16 has no traffic-light blueprint or prop that can be spawned (the map's lights are level
  actors): not spawned, unless a stand-in prop is asked for (the prop nearest in size, its bottom at the head's height).
* object: by its name (a cyclist or bicycle → the bicycle with its rider nearest in size, a motorcycle, a truck or a
  car → that kind, a child or a person → a walker, a cone, a barrier), else the static prop nearest in size.
* kerb, lane: part of the road: an actor cannot add one (not spawned, said per target).
The sizes the choice uses were measured on a 0.9.16 server (Town10HD, 2026-10-27; WALKERS, PROPS, BOARDS,
TWO_WHEELERS: each blueprint spawned alone, its own pixels in an instance and a depth camera turned into 3D points):
a walker's rendered height is not its box's (the adults' boxes are 1.86 m, they render 1.80–1.87 m; the children's
1.10 and 1.30 m boxes render 1.095–1.15 and 1.24 m; the "teenagers" render 1.19–1.20 m in 1.86 m boxes). Each actor's
size as spawned (run.json actors) is reported against the target's nominal size.

Scenes. Targets share a drive when no camera sees two of them overlap (their rectangles' image boxes, at the larger of
the nominal and the actor's size, with a margin) and none stand within a metre of each other: the study looks at each
target alone, CARLA renders them together. Every scene is driven at each spawn point asked for (different
backgrounds and light), the ego parked on a straight road, no traffic, ground truth instance + semantic + range.

Measured in CARLA, per target and camera, from the last frames of each drive: the actor's rectangle (its box centre,
bottom and height; its box's width across the line of sight from that camera; a board's panel for a sign) sampled as
the study samples a standing target (stations across, nodes up, square to the camera's line of sight, every node a
fraction of a pixel apart) and every node looked up in the ground-truth images at its pixel (the rig's lens model, the
camera's actual mount): the actor's own pixel (the instance image, by its actor id) is the target seen; a pixel whose
range is shorter than the node's by more than a tolerance is the node hidden, by what the semantic image says is there
(class 'ego' is CARLA's own car, told apart by its actor id); else the line of sight is clear. Then, as the study
defines them: in view (the share of nodes inside the image and the lens circle), seen (inside and not hidden), and the
pixels across its height: the actor's own pixels' extent in height (from the lowest to the highest node of any
station on them, the stations reaching 15 % beyond the rectangle so a taller or nearer actor is measured whole),
measured along the projected centre line as the study measures a station, in sensor pixels: a car's wheels and its roof
are not in one column, and gaps in a silhouette (between a walker's legs, in a bicycle's frame) are spanned as a box
around it would span them. The ground truth's own object figures are kept
beside them (visible pixels, tight and full boxes, truncation, occlusion, range).

Predicted by Vantage: the study's look at the nominal target (core.critical.run, no envelopes), and the same
measurement of a target of the actor's size at the actor's measured place (core.critical's own look: the project's
body, rig and housings, the lens model): the prediction scaled for the size difference.

Compared: a camera meets the target in CARLA under the study's rule (seen ≥ min share, pixels ≥ min px, and the focus
range by distance, which CARLA does not render: applied alike). Agreement is counted over (target, camera, scene)
where either side has the target in view, with a Wilson interval; the pixel error over pairs both see. Each
disagreement is explained from the measurements: the camera inside CARLA's car, CARLA's car (not the project's body)
hiding a different share (the ego's own pixels over the target's), something of CARLA's town hiding it, the image's
edge, the actor's placement, a figure on the threshold, or the actor's silhouette not filling its rectangle.
"""
from __future__ import annotations

import copy
import csv
import io
import json
import math
import os
import re
import shutil
import signal
import subprocess
import time
from dataclasses import asdict, dataclass, field

import numpy as np

from . import critical as CR
from . import groundtruth as GT
from .mathutil import BODY_TO_OPTICAL, rot_zyx

SCHEMA = 'vantage.critical_carla/1'
FRAME = 'vehicle (ISO 8855, rear axle)'
CARLA_VERSION = '0.9.16'
MEASURED_ON = 'a CARLA 0.9.16 server, Town10HD, 2026-10-27'
# walkers: suffix of walker.pedestrian.NNNN -> (age attribute, box height m, rendered height m, rendered width m
# facing the camera), measured (MEASURED_ON)
WALKERS = {
    '0001': ('adult', 1.860, 1.847, 0.516), '0002': ('adult', 1.860, 1.842, 0.625),
    '0003': ('adult', 1.860, 1.842, 0.625), '0004': ('adult', 1.860, 1.842, 0.625),
    '0005': ('adult', 1.860, 1.847, 0.516), '0006': ('adult', 1.860, 1.847, 0.516),
    '0007': ('adult', 1.860, 1.847, 0.516), '0008': ('adult', 1.860, 1.847, 0.516),
    '0009': ('child', 1.100, 1.095, 0.340), '0010': ('child', 1.100, 1.095, 0.340),
    '0011': ('child', 1.100, 1.095, 0.340), '0012': ('child', 1.300, 1.244, 0.387),
    '0013': ('child', 1.300, 1.244, 0.381), '0014': ('child', 1.300, 1.244, 0.381),
    '0015': ('adult', 1.860, 1.845, 0.573), '0016': ('adult', 1.860, 1.841, 0.636),
    '0017': ('adult', 1.860, 1.841, 0.636), '0018': ('adult', 1.860, 1.836, 0.636),
    '0019': ('adult', 1.860, 1.845, 0.573), '0020': ('adult', 1.860, 1.841, 0.568),
    '0021': ('adult', 1.860, 1.841, 0.573), '0022': ('adult', 1.860, 1.842, 0.581),
    '0023': ('adult', 1.860, 1.842, 0.581), '0024': ('adult', 1.860, 1.870, 0.641),
    '0025': ('adult', 1.860, 1.870, 0.620), '0026': ('adult', 1.860, 1.836, 0.641),
    '0027': ('adult', 1.860, 1.847, 0.648), '0028': ('adult', 1.860, 1.847, 0.648),
    '0029': ('adult', 1.860, 1.847, 0.648), '0030': ('adult', 1.860, 1.843, 0.642),
    '0031': ('adult', 1.860, 1.852, 0.547), '0032': ('adult', 1.860, 1.857, 0.581),
    '0033': ('adult', 1.860, 1.874, 0.581), '0034': ('adult', 1.860, 1.815, 0.710),
    '0035': ('adult', 1.860, 1.847, 0.713), '0036': ('adult', 1.860, 1.847, 0.713),
    '0037': ('adult', 1.860, 1.847, 0.713), '0038': ('adult', 1.860, 1.815, 0.710),
    '0039': ('adult', 1.860, 1.857, 0.674), '0040': ('adult', 1.860, 1.850, 0.581),
    '0041': ('adult', 1.860, 1.850, 0.581), '0042': ('adult', 1.860, 1.845, 0.673),
    '0043': ('adult', 1.860, 1.845, 0.673), '0044': ('adult', 1.860, 1.845, 0.673),
    '0045': ('adult', 1.860, 1.848, 0.608), '0046': ('adult', 1.860, 1.848, 0.608),
    '0047': ('adult', 1.860, 1.848, 0.608), '0048': ('child', 1.100, 1.142, 0.360),
    '0049': ('child', 1.100, 1.149, 0.374), '0050': ('teenager', 1.860, 1.203, 0.380),
    '0051': ('teenager', 1.860, 1.190, 0.380), '0052': ('adult', 1.860, 1.801, 0.603),
}
# static props that stand upright on their box bottom (rendered extent within 15 % of the box; trees, debris and props
# that sink or fall left out): static.prop.<name> -> box length (x), width (y), height m (MEASURED_ON)
PROPS = {
    'advertisement': (1.548, 0.291, 2.443), 'atm': (0.711, 0.876, 2.276), 'barbeque': (0.944, 0.501, 1.264),
    'barrel': (0.458, 0.473, 0.807), 'bench01': (1.542, 0.740, 0.970), 'bench02': (1.562, 0.549, 0.509),
    'bench03': (1.582, 0.517, 0.513), 'bin': (0.546, 0.640, 1.059), 'box01': (0.650, 0.654, 0.692),
    'box02': (0.647, 0.650, 0.649), 'box03': (0.659, 0.672, 0.649), 'busstop': (3.871, 1.904, 2.739),
    'chainbarrier': (1.494, 0.243, 0.889), 'chainbarrierend': (0.238, 0.239, 0.889),
    'clothcontainer': (1.234, 1.890, 1.810), 'constructioncone': (0.343, 0.345, 0.586),
    'container': (1.929, 1.018, 1.714), 'doghouse': (1.076, 1.275, 1.094), 'gardenlamp': (0.296, 0.282, 0.648),
    'glasscontainer': (1.949, 2.014, 1.788), 'gnome': (0.366, 0.377, 0.883), 'haybalelb': (1.244, 1.197, 1.195),
    'kiosk_01': (1.621, 1.659, 3.245), 'mailbox': (0.525, 0.440, 1.157), 'maptable': (1.571, 0.621, 1.181),
    'plantpot01': (1.088, 0.479, 0.557), 'plantpot02': (1.050, 0.434, 0.487), 'plantpot05': (0.399, 0.401, 0.280),
    'plantpot06': (1.589, 1.598, 0.850), 'plantpot07': (0.493, 0.477, 0.557), 'plantpot08': (0.529, 0.532, 0.489),
    'plasticchair': (0.728, 0.752, 1.271), 'streetbarrier': (1.214, 0.375, 1.069),
    'streetfountain': (0.285, 0.671, 1.259), 'table': (2.150, 2.158, 0.846), 'trafficcone01': (0.880, 0.885, 1.133),
    'trafficcone02': (0.455, 0.396, 1.183), 'trashbag': (0.377, 0.431, 0.702), 'trashcan01': (0.634, 0.570, 0.848),
    'trashcan02': (0.522, 0.515, 1.045), 'trashcan03': (0.628, 0.547, 0.934), 'trashcan04': (0.787, 0.583, 1.016),
    'vendingmachine': (0.870, 1.104, 2.108), 'warningaccident': (1.302, 1.059, 1.860),
}
# the street-sign props are boards: the board's band above the prop's bottom (where the profile is ≥ 90 % of its widest)
# and its width, with the prop's box (the board spans its x axis; its face looks along y). MEASURED_ON, facing.
BOARDS = {
    'streetsign': {'box': (1.064, 0.128, 2.154), 'z0': 0.48, 'z1': 2.16, 'width': 1.058},
    'streetsign01': {'box': (2.470, 0.310, 3.878), 'z0': 1.92, 'z1': 3.88, 'width': 2.444},
    'streetsign04': {'box': (1.170, 0.128, 2.773), 'z0': 0.28, 'z1': 2.48, 'width': 1.162},
}
# two-wheelers with their rider (one actor: the rider is part of the vehicle's mesh, tagged rider): base type, box
# length, width, height, rendered height (MEASURED_ON)
TWO_WHEELERS = {
    'vehicle.bh.crossbike': ('bicycle', 1.509, 0.866, 1.612, 1.633),
    'vehicle.diamondback.century': ('bicycle', 1.656, 0.582, 1.620, 1.613),
    'vehicle.gazelle.omafiets': ('bicycle', 1.843, 0.659, 1.776, 1.750),
    'vehicle.harley-davidson.low_rider': ('motorcycle', 2.350, 0.766, 1.649, 1.598),
    'vehicle.kawasaki.ninja': ('motorcycle', 2.044, 0.797, 1.523, 1.522),
    'vehicle.vespa.zx125': ('motorcycle', 1.817, 0.866, 1.590, 1.582),
    'vehicle.yamaha.yzf': ('motorcycle', 2.191, 0.866, 1.530, 1.488),
}
CHOICES = ('auto', 'standin', 'none')        # per target; or 'blueprint:<id>'
CHOICE_LABEL = {'auto': 'Automatic', 'standin': 'Stand-in prop', 'none': 'Not in CARLA'}
NOT_SPAWNABLE = {
    'kerb': 'a kerb is part of the road: an actor cannot add one (CARLA\'s town has its own kerbs where its map has '
            'them)',
    'lane': 'a lane marking is painted on the road: an actor cannot add one (CARLA\'s town has its own markings)',
    'traffic_light': 'CARLA 0.9.16 has no traffic-light blueprint or prop that can be spawned (the map\'s own lights '
                     'are level actors); choose a stand-in prop to compare the geometry',
}
GROUND_CLASSES = frozenset((1, 2, 10, 23, 24, 25, 27))   # road, sidewalk, terrain, water, road line, ground, rail
EGO = GT.EGO_CLASS
REASONS = {
    'inside': 'the camera sits inside CARLA\'s car',
    'body': 'CARLA\'s car is not the project\'s body',
    'scene': 'CARLA\'s town hides it',
    'view': 'the image edge or lens circle',
    'placement': 'the actor stands elsewhere',
    'threshold': 'on the threshold',
    'shape': 'the actor\'s silhouette',
    'other': 'not explained by the checks',
}
LIMITS = {'duration_s': (1.0, 10.0), 'image_scale': (0.1, 1.0), 'frames': (1, 20), 'seed': (0, 999999)}
INTS = ('frames', 'seed')
MAX_SPAWN_POINTS = 8
MIN_SEP_M = 1.0                              # targets nearer each other than this never share a scene
OVERLAP_PAD = (0.1, 10.0)                    # image boxes grown by 10 % and 10 sensor px before the overlap test
BRIDGE_MEMORY_GB = 8.0
MAX_CROPS = 400                              # crops kept and exported (every compared view up to this)
OWN_MARGIN = 0.15                            # the actor's own pixels looked for this share of its height beyond it
SECONDS_PER_DRIVE = 20.0                     # wall seconds of one drive (1 s, images at half size, the sample's six
                                             # fisheyes, lidar and radars; RTX 5080, 2026-10-27: 17.8–21.0 s, nine drives)
ASSUMPTIONS = (
    'Each target is the CARLA actor nearest it in kind and size (child and adult walkers by their measured rendered '
    'height, cars by height and width, props by size; street-sign boards stand in for sign plates), standing still '
    'around the still ego on a straight road at the target\'s place in the vehicle frame (walkers facing the vehicle, '
    'vehicles at the target\'s heading). CARLA renders its own car (the vehicle nearest the project\'s by size), its '
    'own town and light. A node of the target\'s rectangle is seen when the actor\'s own pixel or a farther surface '
    'is at its pixel, hidden when a nearer surface is (by more than 5 cm + 1 % of the distance; 30 cm + 3 % for the '
    'ground); the study\'s rule then gives the verdict, with its focus range applied by distance alike (CARLA renders '
    'no defocus). The prediction at the actor\'s size re-measures the study\'s look with the actor\'s box (its height, '
    'its width across the line of sight, its bottom) at its measured place.')
LIMITS_TEXT = (
    'CARLA\'s actors are not the targets: a child walker is at least 1.10 m tall, an adult 1.80 m, and a sign is one '
    'of three information boards; the comparison at the actor\'s size takes that out of the pixel figures, not out of '
    'the silhouette (legs, arms, a rider\'s outline are not a rectangle). Traffic lights, kerbs and lane markings are '
    'not spawned. The actor is placed by its box (the bridge reads it from CARLA; 0.9.16 gives a static prop\'s box '
    'aligned with the world\'s axes, so a turned prop\'s box is larger than the prop). One instant per frame with a '
    'manual exposure; the last frames of each drive are measured and their spread stated. The scene is CARLA\'s town '
    'at a spawn point: what hides a target there (a pole, a parked car) is not in the study, and is reported as such.')
METHOD = {
    'seen': 'share of the rectangle\'s nodes inside the image and the lens circle and not hidden (the study\'s '
            'definition, measured on the range image)',
    'px': 'the actor\'s own pixels\' extent in height (the lowest to the highest node of any station on them), '
          'measured along the projected centre line of its rectangle, in sensor pixels (the study\'s pixels across '
          'the target\'s height)',
    'hidden': 'share of the in-view nodes hidden, by the class at the pixel (ego: CARLA\'s car)',
    'agreement': 'the camera\'s verdict (meets the target or not) the same in CARLA and in the prediction at the '
                 'actor\'s size, over (target, camera, scene) with the target in view on either side; Wilson 95 % '
                 'interval',
    'px_error': 'CARLA − prediction at the actor\'s size, sensor pixels, over pairs where both put pixels on it'}


# =========================================================================== settings
def _num(v, default, lim=None):
    try:
        x = float(v)
    except (TypeError, ValueError):
        return default
    if not math.isfinite(x):
        return default
    return min(max(x, lim[0]), lim[1]) if lim else x


def clean_choice(v) -> str:
    """'auto', 'standin', 'none' or 'blueprint:<id>' (a CARLA vehicle, walker or static prop id), else 'auto'."""
    if v in CHOICES:
        return v
    if isinstance(v, str) and v.startswith('blueprint:'):
        bid = v.split(':', 1)[1].strip()
        if re.fullmatch(r'(vehicle|walker|static)\.[A-Za-z0-9_.\-]{1,80}', bid):
            return 'blueprint:' + bid
    return 'auto'


@dataclass
class CriticalCarlaSettings:
    town: str = ''                     # '' = the CARLA page's town
    weather: str = 'ClearNoon'         # a CARLA preset
    spawn_points: list = field(default_factory=lambda: [-1])    # −1: chosen by the seed on a straight road
    seed: int = 1
    duration_s: float = 1.0
    image_scale: float = 0.5           # against the sensor's size (1.0: the study's own pixels; measured: six full-size
                                       # fisheyes' ground truth outgrow the bridge's 8 GB cap, half size fits)
    frames: int = 3                    # the last frames of each drive measured (their spread is stated)
    group: bool = True                 # targets that cannot see each other share a drive
    vehicle: str = 'auto'              # the ego: 'auto' = the CARLA page's choice
    actors: dict = field(default_factory=dict)   # target uid -> CHOICES or 'blueprint:<id>'

    def to_dict(self):
        return asdict(self)

    @staticmethod
    def from_dict(d) -> 'CriticalCarlaSettings':
        from . import simrig as SR
        s = CriticalCarlaSettings()
        d = d if isinstance(d, dict) else {}
        if isinstance(d.get('town'), str):
            s.town = d['town'].strip()[:64]
        if d.get('weather') in SR.CARLA_WEATHERS:
            s.weather = d['weather']
        sp = d.get('spawn_points')
        if isinstance(sp, (int, float)) and not isinstance(sp, bool):
            sp = [sp]
        if isinstance(sp, str):
            sp = [x for x in re.split(r'[,\s;]+', sp) if x]
        if isinstance(sp, list):
            out = []
            for x in sp:
                try:
                    k = int(float(x))
                except (TypeError, ValueError):
                    continue
                k = min(max(k, -1), 100000)
                if k not in out:
                    out.append(k)
            s.spawn_points = out[:MAX_SPAWN_POINTS] or [-1]
        for k, lim in LIMITS.items():
            if k in d and not isinstance(d[k], bool):
                v = _num(d[k], getattr(s, k), lim)
                setattr(s, k, int(round(v)) if k in INTS else v)
        g = d.get('group')
        if isinstance(g, bool):
            s.group = g
        v = d.get('vehicle')
        if isinstance(v, str) and (v == 'auto' or re.fullmatch(r'vehicle\.[A-Za-z0-9_.\-]{1,80}', v)):
            s.vehicle = v
        a = d.get('actors')
        if isinstance(a, dict):
            s.actors = {str(k)[:40]: clean_choice(c) for k, c in list(a.items())[:CR.MAX_TARGETS * 4]
                        if clean_choice(c) != 'auto'}
        return s


def settings_of(project) -> CriticalCarlaSettings:
    return CriticalCarlaSettings.from_dict(getattr(project.analysis, 'critical_carla', None))


# =========================================================================== the actors
def _walker_id(k: str) -> str:
    return f'walker.pedestrian.{k}'


def _score(h, w, H, Wd) -> float:
    return abs(h - H) / max(H, 1e-6) + 0.5 * abs(w - Wd) / max(Wd, 1e-6)


def _pick_walker(age: str, h: float, w: float):
    rows = [(k, r) for k, r in sorted(WALKERS.items()) if r[0] == age]
    k, r = min(rows, key=lambda kr: (_score(kr[1][2], kr[1][3], h, w), kr[0]))
    return _walker_id(k), r


def _pick_prop(h: float, w: float, names=None):
    cands = [(k, r) for k, r in sorted(PROPS.items()) if names is None or k in names]
    k, r = min(cands, key=lambda kr: (_score(kr[1][2], kr[1][0], h, w), kr[0]))
    return f'static.prop.{k}', r


def _pick_vehicle(base_types, h: float, w: float):
    from . import simrig as SR
    rows = [(k, r) for k, r in sorted(SR.CARLA_VEHICLES.items()) if r[0] in base_types]
    k, r = min(rows, key=lambda kr: (_score(kr[1][3], kr[1][2] * 0.92, h, w), kr[0]))
    return k, r


def _pick_two_wheeler(base: str, h: float, w: float):
    rows = [(k, r) for k, r in sorted(TWO_WHEELERS.items()) if r[0] == base]
    k, r = min(rows, key=lambda kr: (_score(kr[1][4], kr[1][2], h, w), kr[0]))
    return k, r


# an object target's name -> the kind of actor (whole words, first match)
OBJECT_WORDS = ((r'cyclists?|bicycles?|bikes?|riders?', 'bicycle'),
                (r'motorcycl\w*|motorbikes?|scooters?|mopeds?', 'motorcycle'), (r'trucks?|lorr(y|ies)', 'truck'),
                (r'vans?', 'van'), (r'bus(es)?', 'bus'), (r'cars?|vehicles?', 'car'),
                (r'child(ren)?|kids?|toddlers?', 'child'),
                (r'pedestrians?|persons?|people|adults?|walkers?|man|woman|men|women', 'adult'),
                (r'cones?', 'cone'), (r'barriers?|barricades?', 'barrier'))


def _object_kind(name: str) -> str:
    n = (name or '').lower()
    for pat, kind in OBJECT_WORDS:
        if re.search(r'\b(' + pat + r')\b', n):
            return kind
    return 'prop'


def _bearing_deg(frm, to) -> float:
    return math.degrees(math.atan2(to[1] - frm[1], to[0] - frm[0]))


def actor_for(project, t: dict, choice: str = 'auto', box: dict | None = None) -> dict:
    """The CARLA actor that stands in for target t: {'spawnable', 'why', 'kind' (ACTORS.json kind: always
    'blueprint:<id>' here, so the scene is the same on every run), 'blueprint', 'family' (walker, vehicle, two_wheeler,
    prop, board), 'label', 'stand_in', 'expected': {'length', 'width', 'height', 'rendered_height'} (the measured
    tables'), 'nominal': {'height', 'width', 'base'}, 'pose': {'x', 'y', 'z', 'yaw'} (vehicle frame: the box's bottom
    centre and heading), 'board': the board's band for a sign, 'note'}."""
    b = box or CR.vehicle_box(project)
    centre = (b['mid_x'], 0.0)
    kind = t['kind']
    h, w, base = float(t['height']), float(t['width']), float(t['base'])
    out = {'spawnable': False, 'why': '', 'kind': None, 'blueprint': None, 'family': None, 'label': '',
           'stand_in': False, 'expected': None, 'nominal': {'height': h, 'width': w, 'base': base}, 'pose': None,
           'board': None, 'note': '', 'choice': choice}
    to_vehicle = _bearing_deg((t['x'], t['y']), centre)
    if choice == 'none':
        out['why'] = 'left out (Not in CARLA)'
        return out
    if kind in ('kerb', 'lane'):
        out['why'] = NOT_SPAWNABLE[kind]
        return out
    fam = bid = None
    exp = None
    yaw = float(t['yaw'])
    z = 0.0
    note = ''
    stand_in = False
    if choice.startswith('blueprint:'):
        bid = choice.split(':', 1)[1]
        if bid.startswith('walker.'):
            fam = 'walker'
            r = WALKERS.get(bid.rsplit('.', 1)[-1])
            exp = {'length': None, 'width': r[3], 'height': r[1], 'rendered_height': r[2]} if r else None
        elif bid in TWO_WHEELERS:
            fam = 'two_wheeler'
            r = TWO_WHEELERS[bid]
            exp = {'length': r[1], 'width': r[2], 'height': r[3], 'rendered_height': r[4]}
        elif bid.startswith('vehicle.'):
            from . import simrig as SR
            fam = 'vehicle'
            r = SR.CARLA_VEHICLES.get(bid)
            exp = {'length': r[1], 'width': r[2], 'height': r[3], 'rendered_height': None} if r else None
        else:
            name = bid.split('static.prop.', 1)[-1]
            if name in BOARDS:
                fam = 'board'
            else:
                fam = 'prop'
                r = PROPS.get(name)
                exp = {'length': r[0], 'width': r[1], 'height': r[2], 'rendered_height': None} if r else None
        note = 'chosen'
    elif choice == 'standin':
        bid, r = _pick_prop(h, w)
        fam = 'prop'
        stand_in = True
        exp = {'length': r[0], 'width': r[1], 'height': r[2], 'rendered_height': None}
        note = (f'a stand-in prop of about the target\'s size ({r[0]:.2f} × {r[2]:.2f} m), not a '
                f'{CR.KIND_LABEL[kind].lower()}')
    elif kind == 'traffic_light':
        out['why'] = NOT_SPAWNABLE['traffic_light']
        return out
    elif kind in ('child', 'pedestrian') or (kind == 'object' and _object_kind(t['name']) in ('child', 'adult')):
        age = 'child' if kind == 'child' or (kind == 'object' and _object_kind(t['name']) == 'child') else 'adult'
        bid, r = _pick_walker(age, h, w)
        fam = 'walker'
        exp = {'length': None, 'width': r[3], 'height': r[1], 'rendered_height': r[2]}
        note = f'the {age} walker nearest in rendered height ({r[2]:.2f} m; its box {r[1]:.2f} m)'
    elif kind == 'car':
        bid, r = _pick_vehicle(('car',), h, w)
        fam = 'vehicle'
        exp = {'length': r[1], 'width': r[2], 'height': r[3], 'rendered_height': None}
        note = f'the car nearest in height and width ({r[3]:.2f} m × {r[2]:.2f} m with its mirrors)'
    elif kind == 'sign':
        name, bd = min(sorted(BOARDS.items()), key=lambda kv: _score(kv[1]['z1'] - kv[1]['z0'], kv[1]['width'],
                                                                         h, w))
        bid, fam, stand_in = f'static.prop.{name}', 'board', True
        out['board'] = dict(bd)
        exp = {'length': bd['box'][0], 'width': bd['box'][1], 'height': bd['box'][2], 'rendered_height': None,
               'board_width': bd['width'], 'board_height': round(bd['z1'] - bd['z0'], 3)}
        z = base - bd['z0']
        note = (f'CARLA has no road-sign prop of this size: the information board {name} ({bd["width"]:.2f} × '
                f'{bd["z1"] - bd["z0"]:.2f} m) stands in, its board\'s bottom {base:.2f} m above the ground')
    else:                                                        # an object
        ok = _object_kind(t['name'])
        if ok in ('bicycle', 'motorcycle'):
            bid, r = _pick_two_wheeler(ok, h, w)
            fam = 'two_wheeler'
            exp = {'length': r[1], 'width': r[2], 'height': r[3], 'rendered_height': r[4]}
            note = f'the {ok} with its rider nearest in size'
        elif ok in ('truck', 'van', 'bus', 'car'):
            bid, r = _pick_vehicle((ok,), h, w)
            fam = 'vehicle'
            exp = {'length': r[1], 'width': r[2], 'height': r[3], 'rendered_height': None}
            note = f'the {ok} nearest in height and width'
        else:
            names = {'cone': ('constructioncone', 'trafficcone01', 'trafficcone02'),
                     'barrier': ('streetbarrier', 'chainbarrier')}.get(ok)
            bid, r = _pick_prop(h, w, names)
            fam = 'prop'
            exp = {'length': r[0], 'width': r[1], 'height': r[2], 'rendered_height': None}
            note = f'the static prop nearest in size ({r[0]:.2f} × {r[2]:.2f} m)'
    if fam == 'board' and out['board'] is None:
        bd = BOARDS[bid.split('static.prop.', 1)[-1]]
        out['board'] = dict(bd)
        exp = {'length': bd['box'][0], 'width': bd['box'][1], 'height': bd['box'][2], 'rendered_height': None,
               'board_width': bd['width'], 'board_height': round(bd['z1'] - bd['z0'], 3)}
        z = base - bd['z0']
    if fam == 'walker':
        yaw = to_vehicle                                  # walkers face the vehicle
        if base > 0.01:
            note += f'; a walker stands on the ground (the target is {base:.2f} m up)'
    elif fam in ('prop', 'board'):
        yaw = to_vehicle - 90.0                           # its x axis across the line of sight: its face to the car
        if fam == 'prop':
            z = base
    yaw = (yaw + 180.0) % 360.0 - 180.0
    out.update(spawnable=True, kind=f'blueprint:{bid}', blueprint=bid, family=fam, expected=exp, stand_in=stand_in,
               note=note, pose={'x': round(float(t['x']), 4), 'y': round(float(t['y']), 4), 'z': round(z, 4),
                                'yaw': round(yaw, 3)},
               label=bid.split('.', 1)[1] if '.' in bid else bid)
    return out


# =========================================================================== the scenes
@dataclass
class ScenePlan:
    targets: list                    # [{'uid', 'name', 'kind', 'place', 'actor': actor_for(...)}]
    groups: list                     # [[uid, …]]: the targets that share a drive
    drives: list                     # [{'key', 'group', 'spawn_point'}]
    notes: list = field(default_factory=list)

    def target(self, uid):
        return next((r for r in self.targets if r['uid'] == uid), None)

    def actors_json(self, group) -> list:
        """ACTORS.json of a group (the bridge's --actors list; each actor's id is its target's uid)."""
        out = []
        for uid in self.groups[group]:
            a = self.target(uid)['actor']
            out.append({'id': uid, 'kind': a['kind'], 'pose': dict(a['pose'])})
        return out


def _cams(p, v):
    return [CR._Cam(p, c) for c in v.cameras if c.visible and CR._finite_cam(c)]


def _image_box(cm, t) -> tuple | None:
    """The image box (sensor px) of target t's rectangle in camera cm, or None when none of it is in view."""
    sh = CR.shape(t, cm.pos, fine=False)
    W2O = BODY_TO_OPTICAL @ cm.R.T
    u, v, _, ok = cm.intr.project_dirs((sh.flat - cm.pos) @ W2O.T)
    if not ok.any():
        return None
    return float(u[ok].min()), float(v[ok].min()), float(u[ok].max()), float(v[ok].max())


def _grown(bx):
    w, h = bx[2] - bx[0], bx[3] - bx[1]
    gx, gy = OVERLAP_PAD[0] * w + OVERLAP_PAD[1], OVERLAP_PAD[0] * h + OVERLAP_PAD[1]
    return bx[0] - gx, bx[1] - gy, bx[2] + gx, bx[3] + gy


def _overlap(a, b) -> bool:
    a, b = _grown(a), _grown(b)
    return a[0] < b[2] and b[0] < a[2] and a[1] < b[3] and b[1] < a[3]


def _extent_target(t, a) -> dict:
    """t at the larger of its nominal size and its actor's (for the overlap test)."""
    e = a.get('expected') or {}
    h = max(float(t['height']), float(e.get('board_height') or e.get('rendered_height') or e.get('height') or 0.0))
    w = max(float(t['width']), float(e.get('board_width') or max(e.get('length') or 0.0, e.get('width') or 0.0)))
    return dict(t, height=h, width=w)


def plan(project, settings: CriticalCarlaSettings | None = None, variant=None, critical_settings=None) -> ScenePlan:
    """The actors, the scenes (targets that share a drive) and the drives (scenes × spawn points). Cheap: no CARLA."""
    st = settings or settings_of(project)
    p = CR._project_for(project, variant)
    v = p.variant
    cst = critical_settings or CR.settings_of(p)
    box = CR.vehicle_box(p)
    rows = []
    for t in cst.targets:
        if not t['enabled']:
            continue
        a = actor_for(p, t, st.actors.get(t['uid'], 'auto'), box)
        rows.append({'uid': t['uid'], 'name': t['name'], 'kind': t['kind'], 'place': CR.describe_place(p, t, box),
                     'actor': a, 'target': t})
    live = [r for r in rows if r['actor']['spawnable']]
    cams = _cams(p, v)
    boxes = {}
    for r in live:
        ext = _extent_target(r['target'], r['actor'])
        for cm in cams:
            bx = _image_box(cm, ext)
            if bx is not None:
                boxes[(r['uid'], cm.uid)] = bx

    def clash(a, b) -> bool:
        ta, tb = a['target'], b['target']
        if math.hypot(ta['x'] - tb['x'], ta['y'] - tb['y']) < MIN_SEP_M:
            return True
        return any((a['uid'], cm.uid) in boxes and (b['uid'], cm.uid) in boxes and
                   _overlap(boxes[(a['uid'], cm.uid)], boxes[(b['uid'], cm.uid)]) for cm in cams)
    groups = []
    for r in live:
        if st.group:
            for g in groups:
                if not any(clash(r, plan_row) for plan_row in g):
                    g.append(r)
                    break
            else:
                groups.append([r])
        else:
            groups.append([r])
    gids = [[r['uid'] for r in g] for g in groups]
    drives = []
    for gi in range(len(gids)):
        for sp in st.spawn_points:
            drives.append({'key': f'scene{gi + 1}-sp{"auto" if sp < 0 else sp}', 'group': gi, 'spawn_point': int(sp)})
    notes = []
    if len(live) < len(rows):
        notes.append(f'{len(rows) - len(live)} of {len(rows)} targets are not stood in CARLA (see each target).')
    if st.group and len(groups) > 1:
        notes.append(f'{len(live)} targets in {len(groups)} scenes: targets that a camera would see overlap, or that '
                     f'stand within {MIN_SEP_M:g} m of each other, are driven apart.')
    for r in rows:
        r.pop('target')
    return ScenePlan(rows, gids, drives, notes)


def scene_settings(project, st: CriticalCarlaSettings, actors: list, spawn_point: int, sim_settings=None):
    """The CARLA page's settings (server, ports, vehicle) for one drive of an actors scene: no traffic, the ego still,
    camera ground truth (instance, semantic, range; no lidar labels), nuScenes only (the images), a short drive."""
    from . import simrig as SR
    base = sim_settings or SR.settings_of(project)
    d = base.to_dict()
    d.update(town=st.town or d['town'], weather=st.weather, duration_s=st.duration_s, traffic=0, seed=st.seed,
             image_scale=st.image_scale, spawn_point=int(spawn_point), actors=actors, faults=[],
             scenario={'name': 'free_drive', 'params': {}}, nuscenes=True, rosbag=False,
             warmup_s=min(float(d.get('warmup_s', 2.0)), 1.0),
             ground_truth={'semantic': True, 'instance': True, 'range': True, 'lidar_labels': False})
    if st.vehicle != 'auto':
        d['vehicle'] = st.vehicle
    return SR.SimSettings.from_dict(d)


# =========================================================================== running a drive
def carla_python() -> str:
    """CARLA's client Python: the CARLA page's choice (kept per machine), else one found next to a CARLA folder."""
    try:
        from PyQt6.QtCore import QSettings
        saved = str(QSettings('Vantage', 'Vantage').value('sim/python', '') or '')
    except Exception:                                 # noqa: BLE001 (no Qt settings here: look for it)
        saved = ''
    if saved and os.path.isfile(saved):
        return saved
    from ..sim import server as SV
    return next((py for py in (SV.client_python(r) for r in SV.candidate_roots()) if py), '')


def _load_json(path):
    try:
        with open(path, encoding='utf-8') as fh:
            return json.load(fh)
    except (OSError, ValueError):
        return None


def _same_rig(a, b) -> bool:
    """Two sim_rig.json describe the same run (everything but when they were written) and their remap tables match."""
    da, db = _load_json(a), _load_json(b)
    if not isinstance(da, dict) or not isinstance(db, dict):
        return False
    da.pop('created', None)
    db.pop('created', None)
    if json.dumps(da, sort_keys=True) != json.dumps(db, sort_keys=True):
        return False
    ra, rb = (os.path.join(os.path.dirname(x), 'sim_remap.npz') for x in (a, b))
    try:
        with np.load(ra) as za, np.load(rb) as zb:
            return set(za.files) == set(zb.files) and all(np.array_equal(za[k], zb[k]) for k in za.files)
    except (OSError, ValueError):
        return not os.path.exists(ra) and not os.path.exists(rb)


def drive(project, variant, sim, out_dir, python=None, progress=None, cancel=None, runner=None,
          memory_gb=BRIDGE_MEMORY_GB) -> dict | None:
    """One drive: the rig into out_dir/rig, the bridge in CARLA's Python on the server answering on sim.port (or
    runner(rig_json, out_dir, progress, cancel), the tests' in-process bridge). A drive already in out_dir from the
    same rig that ended well with ground truth is reused (a cancelled run resumes). Returns {'dir', 'reused',
    'seconds', 'run'}; None when cancelled. Raises RuntimeError when no server answers, CARLA's Python is missing or the
    drive fails (run.json says why)."""
    from . import simrig as SR
    from ..sim import server as SV
    stop = cancel or (lambda: False)

    def prog(f, text=''):
        if progress is not None:
            progress(min(max(float(f), 0.0), 1.0), text)
    t0 = time.perf_counter()
    out_dir = os.path.abspath(out_dir)
    os.makedirs(out_dir, exist_ok=True)
    rig = SR.build(project, sim, variant=variant, cancel=cancel)
    if rig is None:
        return None
    new_dir = os.path.join(out_dir, '.rig-new')
    if os.path.isdir(new_dir):
        shutil.rmtree(new_dir)
    SR.export(rig, project, new_dir)
    rig_dir = os.path.join(out_dir, 'rig')
    doc = SR.read_run(out_dir)
    reuse = (_same_rig(os.path.join(new_dir, 'sim_rig.json'), os.path.join(rig_dir, 'sim_rig.json')) and
             doc is not None and doc.get('status') == 'ok' and GT.read(out_dir) is not None)
    if reuse:
        shutil.rmtree(new_dir, ignore_errors=True)
        prog(1.0, 'reused')
        return {'dir': out_dir, 'reused': True, 'seconds': time.perf_counter() - t0, 'run': doc}
    for name in ('rig', 'nuscenes', 'ground_truth', 'rosbag', 'dataset.json', 'README.md', 'run.json'):
        path = os.path.join(out_dir, name)
        if os.path.isdir(path):
            shutil.rmtree(path)
        elif os.path.exists(path):
            os.remove(path)
    os.replace(new_dir, rig_dir)
    rig_json = os.path.join(rig_dir, 'sim_rig.json')
    code = None
    if runner is not None:
        try:
            runner(rig_json, out_dir, prog, stop)
        except Exception:                          # noqa: BLE001 (a cancel inside the runner is a cancel)
            if stop():
                return None
            raise
    else:
        py = python or carla_python()
        if not py or not os.path.isfile(py):
            raise RuntimeError('No Python for CARLA\'s client: choose it on Simulate ▸ CARLA.')
        if not SV.port_open('127.0.0.1', sim.port):
            raise RuntimeError(f'No CARLA server answers on port {sim.port}: start one on Simulate ▸ CARLA.')
        cmd, env = SV.bridge_command(py, rig_json, out_dir, ['--port', str(sim.port), '--tm-port', str(sim.tm_port),
                                                             '--timeout', f'{sim.timeout_s:g}'], memory_gb=memory_gb)
        with open(os.path.join(out_dir, 'bridge.log'), 'w', encoding='utf-8') as log:
            proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=log, text=True, env=env, cwd=out_dir,
                                    start_new_session=True)
            asked = False
            try:
                for line in proc.stdout:
                    pr = SV.parse_progress(line)
                    if pr is not None:
                        prog(pr[0], pr[1])
                    else:
                        log.write(line)
                    if stop() and not asked:
                        _signal(proc, signal.SIGINT)        # the bridge tidies CARLA up and removes its output
                        asked = True
                proc.wait()
            finally:
                if proc.poll() is None:
                    _signal(proc, signal.SIGINT)
                    try:
                        proc.wait(60)
                    except subprocess.TimeoutExpired:
                        _signal(proc, signal.SIGKILL)
                        proc.wait(10)
            code = proc.returncode
    if stop():
        return None
    doc = SR.read_run(out_dir)
    if doc is None:                                # killed or aborted without its report: nothing half-written kept
        for name in ('nuscenes', 'ground_truth', 'rosbag', 'dataset.json', 'README.md'):
            path = os.path.join(out_dir, name)
            if os.path.isdir(path):
                shutil.rmtree(path, ignore_errors=True)
            elif os.path.exists(path):
                os.remove(path)
        why = (f'the bridge was killed (signal {-code}: its memory cap of {memory_gb:g} GB, or the server stopping)'
               if code is not None and code < 0 else f'the bridge exited with {code} and wrote no report'
               if code is not None else 'it wrote no report')
        doc = {'schema': SR.RUN_SCHEMA, 'status': 'failed', 'error': f'{why}; see bridge.log', 'exit_code': code}
        with open(os.path.join(out_dir, 'run.json'), 'w', encoding='utf-8') as fh:
            json.dump(doc, fh, indent=1)
    if doc.get('status') != 'ok':
        raise RuntimeError(f'The CARLA drive failed: {str(doc.get("error") or "see bridge.log").splitlines()[0]}')
    if GT.read(out_dir) is None:
        raise RuntimeError('The CARLA drive wrote no ground truth (it needs cameras and a rendering server).')
    return {'dir': out_dir, 'reused': False, 'seconds': time.perf_counter() - t0, 'run': doc}


def _signal(proc, sig):
    try:
        os.killpg(proc.pid, sig)
    except (ProcessLookupError, PermissionError, OSError):
        try:
            proc.send_signal(sig)
        except (ProcessLookupError, OSError):
            pass


# =========================================================================== measuring in CARLA
def apparent_width(length: float, width: float, yaw_deg: float, cam_xy, centre_xy) -> float:
    """A box's extent across the line of sight from cam_xy to its centre (length along its heading yaw)."""
    phi = math.atan2(centre_xy[1] - cam_xy[1], centre_xy[0] - cam_xy[0])
    a = math.radians(yaw_deg) - phi
    return abs(length * math.sin(a)) + abs(width * math.cos(a))


def rect_nodes(cx, cy, base, height, width, cam_pos, nc: int, n: int, yaw_deg: float = 0.0) -> np.ndarray:
    """(nc, n, 3) nodes of a standing rectangle as core.critical.shape builds them: stations across its width square
    to the camera's line of sight (cell centres), nodes from its bottom to its top."""
    G = np.array([cx, cy], float)
    los = G - np.asarray(cam_pos, float)[:2]
    L = float(np.linalg.norm(los))
    if L > 1e-6:
        perp = np.array([-los[1], los[0]]) / L
    else:
        a = math.radians(yaw_deg + 90.0)
        perp = np.array([math.cos(a), math.sin(a)])
    u = ((np.arange(nc) + 0.5) / nc - 0.5) * width
    z = base + np.linspace(0.0, height, n)
    xy = G[None, :] + u[:, None] * perp[None, :]
    return np.concatenate([np.repeat(xy[:, None, :], n, 1), np.broadcast_to(z[None, :, None], (nc, n, 1))], 2)


def rect_outline(rect: dict, cam_pos, n: int = 40) -> np.ndarray:
    """(4 n, 3) points around a standing rectangle's edges (bottom left to right, up, top right to left, down), square
    to the line of sight from cam_pos as rect_nodes builds it."""
    G = np.array([rect['x'], rect['y']], float)
    los = G - np.asarray(cam_pos, float)[:2]
    L = float(np.linalg.norm(los))
    perp = np.array([-los[1], los[0]]) / L if L > 1e-6 else np.array([0.0, 1.0])
    a = G + perp * rect['width'] / 2
    b = G - perp * rect['width'] / 2
    z0, z1 = rect['base'], rect['base'] + rect['height']
    s = np.linspace(0.0, 1.0, n)[:, None]
    A, B = np.r_[a, z0], np.r_[b, z0]
    C, D = np.r_[b, z1], np.r_[a, z1]
    return np.concatenate([A + s * (B - A), B + s * (C - B), C + s * (D - C), D + s * (A - D)])


def with_actual_mounts(rig: dict, run: dict | None) -> dict:
    """The rig with each faulted sensor's actual mount as 'pose_actual'. The bridge moves the mounts of an in-memory
    copy and never writes it back, so sim_rig.json on disk states the plan; run.json faults_injected carries the
    actual pose (cumulative: the last row of a sensor is where it ended up)."""
    actual = {}
    for f in (run or {}).get('faults_injected') or []:
        if isinstance(f, dict) and f.get('kind') in ('rotation', 'translation') and f.get('actual_pose') and f.get('tag'):
            actual[f['tag']] = f['actual_pose']
    if not actual or not isinstance(rig, dict):
        return rig
    rig = json.loads(json.dumps(rig))
    for key in ('cameras', 'lidars', 'radars', 'imus', 'gnss'):
        for d in rig.get(key, []):
            if d.get('tag') in actual:
                d['pose_actual'] = dict(actual[d['tag']])
    return rig


def cam_pose(cam: dict) -> tuple[np.ndarray, np.ndarray]:
    """(R body -> vehicle, position) of a rig camera's actual mount (with_actual_mounts puts it there)."""
    pa = cam.get('pose_actual') or cam['pose_vehicle']
    return rot_zyx(pa['yaw'], pa['pitch'], pa['roll']), np.array([pa['x'], pa['y'], pa['z']], float)


def _tol(cls: np.ndarray, d: np.ndarray) -> np.ndarray:
    ground = np.isin(cls, list(GROUND_CLASSES))
    return np.where(ground, 0.3 + 0.03 * d, 0.05 + 0.01 * d)


def classify(cam: dict, rect: dict, inst, sem, rng, oid, n_max: int = 1500) -> dict:
    """The target's rectangle in one camera frame of CARLA's ground truth. cam: the rig's camera (lens block, mount),
    rect {'x', 'y', 'base', 'height', 'width'} (vehicle frame), inst/sem/rng: the frame's instance (object ids), semantic
    (class ids) and range (m) images, oid: the actor's object id in this frame (None: not in it). Returns in_view,
    seen, hidden {class name: share of in-view nodes} (over the rectangle's nodes, as the study counts them), px_image
    (the actor's own pixels' extent in height, from the lowest to the highest node of any station on them, measured
    along the projected centre line; image px), own_share (rectangle nodes on the actor), z_own (that extent, m), the
    rectangle's image box, the stations and nodes used. The nodes reach OWN_MARGIN of the height beyond the rectangle
    so an actor taller than it, or nearer than its plane, is measured whole."""
    R, pos = cam_pose(cam)
    lens = cam['lens']
    h, w = inst.shape
    V2O = BODY_TO_OPTICAL @ R.T
    H = float(rect['height'])
    m = float(rect['margin']) if 'margin' in rect else max(OWN_MARGIN * H, 0.05)   # a little beyond the rectangle
    z0, z1 = rect['base'] - m, rect['base'] + H + m
    probe = rect_nodes(rect['x'], rect['y'], z0, z1 - z0, 0.0, pos, 1, 2)[0]
    u0, v0, _, _ = GT.project(lens, (probe - pos) @ V2O.T)
    est = float(np.hypot(u0[1] - u0[0], v0[1] - v0[0])) if np.all(np.isfinite(u0)) else 0.0
    n = int(min(max(math.ceil(est * 2.0) + 2, 41), n_max))     # nodes at most half a pixel apart
    ends = rect_outline(rect, pos, 2)[[0, 1]]                    # the bottom edge's two corners
    uw, vw, _, _ = GT.project(lens, (ends - pos) @ V2O.T)
    wpx = float(np.hypot(uw[1] - uw[0], vw[1] - vw[0])) if np.all(np.isfinite(uw)) else 0.0
    nc = int(min(max(math.ceil(wpx / 2.0), 9), 81))             # a station every 2 px across, 9 to 81 of them
    P = rect_nodes(rect['x'], rect['y'], z0, z1 - z0, rect['width'], pos, nc, n)
    flat = P.reshape(-1, 3)
    zz = flat[:, 2]
    core = (zz >= rect['base'] - 1e-9) & (zz <= rect['base'] + H + 1e-9)      # the rectangle the study measures
    u, v, _, ok = GT.project(lens, (flat - pos) @ V2O.T)
    inside = ok & np.isfinite(u) & np.isfinite(v) & (u >= 0) & (u < w) & (v >= 0) & (v < h)
    iu = np.clip(np.floor(np.nan_to_num(u)).astype(np.int64), 0, w - 1)
    iv = np.clip(np.floor(np.nan_to_num(v)).astype(np.int64), 0, h - 1)
    d = np.linalg.norm(flat - pos, axis=1)
    ids = inst[iv, iu].astype(np.int64)
    cls = sem[iv, iu].astype(np.int64)
    r = rng[iv, iu].astype(np.float64)
    own = inside & (ids == int(oid)) if oid is not None else np.zeros(len(flat), bool)
    hidden = inside & ~own & (r > 0) & (r < d - _tol(cls, d)) & core
    in_core = inside & core
    n_core = max(int(core.sum()), 1)
    n_in = int(in_core.sum())
    hid = {}
    if n_in:
        for k in np.unique(cls[hidden]):
            hid[GT.CLASS_NAMES.get(int(k), 'unlabelled')] = float((hidden & (cls == k)).sum()) / n_in
    # the pixels across its height: the actor's own pixels' extent in height (over every station: a car's wheels and
    # its roof are not in one column), measured along its projected centre line as the study measures a station
    zo = zz[own]
    px = 0.0
    z_ext = None
    if len(zo) >= 2 and zo.max() > zo.min():
        z_ext = [float(zo.min()), float(zo.max())]
        line = rect_nodes(rect['x'], rect['y'], z0, z1 - z0, 0.0, pos, 1, n)[0]
        lu, lv, _, lok = GT.project(lens, (line - pos) @ V2O.T)
        seg = np.hypot(np.diff(lu), np.diff(lv))
        seg = np.where(np.isfinite(seg) & lok[1:] & lok[:-1], seg, 0.0)
        cum = np.r_[0.0, np.cumsum(seg)]
        px = float(np.interp(z_ext[1], line[:, 2], cum) - np.interp(z_ext[0], line[:, 2], cum))
    bx = None
    if in_core.any():
        bx = [float(u[in_core].min()), float(v[in_core].min()), float(u[in_core].max()), float(v[in_core].max())]
    return {'in_view': n_in / n_core, 'seen': float((in_core & ~hidden).sum()) / n_core, 'hidden': hid,
            'px_image': px, 'own_share': float((own & core).sum()) / n_core, 'z_own': z_ext,
            'image_box': bx, 'stations': nc, 'nodes': n}


def _row_box(row: dict) -> dict | None:
    """The actor's box in the vehicle frame from its run.json row: centre, size (l, w, h), yaw, bottom z."""
    pv, bb = row.get('pose_vehicle'), row.get('bbox')
    if not pv or not bb:
        return None
    L, W, H = (float(x) for x in bb['size_lwh'])
    return {'x': float(pv['x']), 'y': float(pv['y']), 'bottom': float(pv['z']), 'yaw': float(pv.get('yaw', 0.0)),
            'size': [L, W, H]}


def actor_rect(a: dict, box: dict, cam_xy) -> dict:
    """The rectangle a camera at cam_xy measures of an actor: its board for a sign, else its box (a walker's or a
    two-wheeler's height its rendered one, from the measured table), its width across the line of sight."""
    exp = a.get('expected') or {}
    L, W, H = box['size']
    if a.get('family') in ('prop', 'board', 'walker', 'two_wheeler') and exp.get('length') and exp.get('width'):
        L, W = float(exp['length']), float(exp['width'])  # (0.9.16 gives a static prop's box in world axes)
    if a.get('family') in ('prop', 'board') and exp.get('height'):
        H = float(exp['height'])
    if a.get('family') == 'walker':
        L, W = box['size'][0], box['size'][1]
    if a.get('board'):                            # the board only: its legs or post are not the sign (no margin)
        bd = a['board']
        return {'x': box['x'], 'y': box['y'], 'base': box['bottom'] + bd['z0'], 'height': bd['z1'] - bd['z0'],
                'width': apparent_width(bd['width'], W, box['yaw'], cam_xy, (box['x'], box['y'])), 'margin': 0.0}
    if a.get('family') in ('walker', 'two_wheeler') and exp.get('rendered_height'):
        H = float(exp['rendered_height'])
    return {'x': box['x'], 'y': box['y'], 'base': box['bottom'], 'height': H,
            'width': apparent_width(L, W, box['yaw'], cam_xy, (box['x'], box['y']))}


def _median(xs):
    xs = [x for x in xs if x is not None]
    return float(np.median(xs)) if xs else None


def measure_drive(drive_dir: str, plan_rows: dict, frames: int = 3, focus=None) -> dict:
    """Every target of a drive in every camera: {'run', 'rig', 'gt', 'actors': {uid: run.json row}, 'looks': {(uid,
    tag): carla dict}}; focus: {camera uid: (near, far)} for the verdict (the lens's focus range)."""
    from . import simrig as SR
    rig = _load_json(os.path.join(drive_dir, 'rig', 'sim_rig.json'))
    run = SR.read_run(drive_dir)
    gt = GT.read(drive_dir)
    if not isinstance(rig, dict) or run is None or gt is None:
        raise ValueError(f'{drive_dir} is not a CARLA drive with ground truth.')
    rig = with_actual_mounts(rig, run)
    acts = {r['id']: r for r in run.get('actors') or [] if isinstance(r, dict) and 'id' in r}
    mounts = {m['tag']: m for m in run.get('mounts') or [] if m.get('kind') == 'camera'}
    looks = {}
    for cam in rig.get('cameras', []):
        tag = cam['tag']
        fr = GT.frames(gt, tag)[-max(int(frames), 1):]
        loaded = []
        for f in fr:
            try:
                loaded.append((f, GT.load_instance(gt, f), GT.load_semantic(gt, f), GT.load_range(gt, f)))
            except OSError:
                continue
        lens = cam['lens']
        scale = float(lens['sensor_size'][0]) / float(lens['image_size'][0]) if lens.get('sensor_size') else 1.0
        R, pos = cam_pose(cam)
        for uid, pr in plan_rows.items():
            row = acts.get(uid)
            if row is None or row.get('actor_id') is None:
                continue
            box = _row_box(row)
            if box is None:
                continue
            rect = actor_rect(pr['actor'], box, pos[:2])
            per = []
            for f, inst, sem, rng in loaded:
                obj = next((o for o in f.get('objects', []) if o.get('actor_id') == row['actor_id']), None)
                m = classify(cam, rect, inst, sem, rng, obj['id'] if obj else None)
                m['object'] = {k: obj.get(k) for k in ('id', 'pixels', 'bbox', 'bbox_full', 'truncated', 'occluded',
                                                        'range_m', 'class', 'category')} if obj else None
                m['frame'] = f
                per.append(m)
            if not per:
                continue
            px = [m['px_image'] * scale for m in per]
            last = per[-1]
            hid = {}
            for m in per:
                for k, s in m['hidden'].items():
                    hid.setdefault(k, []).append(s)
            hidden = {k: float(np.median(v + [0.0] * (len(per) - len(v)))) for k, v in hid.items()}
            dist = float(np.linalg.norm(np.array([rect['x'], rect['y'], rect['base'] + rect['height'] / 2]) - pos))
            near, far = (focus or {}).get(cam.get('uid'), (0.0, math.inf))
            in_focus = (dist >= near) and (dist <= far)
            obj = next((m['object'] for m in reversed(per) if m['object']), None)
            looks[(uid, tag)] = {
                'in_view': _median([m['in_view'] for m in per]), 'seen': _median([m['seen'] for m in per]),
                'px': _median(px), 'px_spread': float(max(px) - min(px)), 'hidden': hidden,
                'ego_hidden': hidden.get('ego', 0.0), 'own_share': _median([m['own_share'] for m in per]),
                'z_own': last['z_own'], 'dist': dist, 'focus': bool(in_focus),
                'pixels': (obj or {}).get('pixels', 0) if obj else 0, 'object': obj, 'frames': len(per),
                'rect': rect, 'image_box': last['image_box'], 'scale': scale,
                'frame': {k: last['frame'].get(k) for k in ('image', 'instance', 'semantic', 'range', 'timestamp')},
                'object_id': last['object']['id'] if last['object'] else None,
                'camera_inside_m': (mounts.get(tag) or {}).get('inside_body_m') or 0.0}
    return {'run': run, 'rig': rig, 'gt': gt, 'actors': acts, 'looks': looks}


def verdict(look: dict, t: dict) -> tuple[bool, str]:
    """(met, first failed criterion) under the study's rule for a standing target."""
    ok_seen = look['seen'] >= t['min_share'] - 1e-9
    ok_px = look['px'] >= t['min_px']
    ok_focus = bool(look.get('focus', True)) or not t.get('focus', True)
    met = ok_seen and ok_px and ok_focus
    return met, '' if met else 'seen' if not ok_seen else 'pixels' if not ok_px else 'focus'


# =========================================================================== comparing
def wilson(k: int, n: int, z: float = 1.959964) -> tuple | None:
    """Wilson score interval of k successes in n."""
    if n <= 0:
        return None
    p = k / n
    den = 1 + z * z / n
    c = (p + z * z / (2 * n)) / den
    hw = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return max(0.0, c - hw), min(1.0, c + hw)


def _look_dict(lk) -> dict:
    return {'met': bool(lk.met), 'px': float(lk.px), 'seen': float(lk.seen), 'in_view': float(lk.in_view),
            'hidden': {k: float(v) for k, v in lk.hidden.items()}, 'fail': lk.fail, 'dist': float(lk.dist),
            'focus': bool(lk.focus)}


def explain(row: dict, t: dict) -> tuple[str, str, list]:
    """(reason code, text, [(code, weight, text)] every reason found) for a pair whose verdicts differ (or whose
    pixels differ by more than 15 %)."""
    v, c = row['vantage_actor'], row['carla']
    out = []
    tag = row['camera']
    if c.get('camera_inside_m', 0.0) > 0.005:
        out.append(('inside', 2.0, f'{tag} sits {100 * c["camera_inside_m"]:.1f} cm inside CARLA\'s car '
                                   f'({row.get("ego", "")}): it sees the inside of the car'))
    vb = float(sum(v['hidden'].values()))
    ce = float(c['hidden'].get('ego', 0.0))
    if abs(ce - vb) >= 0.1:
        out.append(('body', abs(ce - vb), f'CARLA\'s car ({row.get("ego", "its ego")}) hides {100 * ce:.0f} % of it '
                                          f'in view, the project\'s body and rig {100 * vb:.0f} % (checked on the '
                                          'ego\'s own pixels over the target)'))
    other = {k: s for k, s in c['hidden'].items() if k != 'ego'}
    co = float(sum(other.values()))
    if co >= 0.1:
        top = max(other, key=other.get)
        out.append(('scene', co, f'{top.replace("_", " ")} of CARLA\'s town hides {100 * co:.0f} % of it in view '
                                 '(the study has no surroundings)'))
    dv = abs(c['in_view'] - v['in_view'])
    if dv >= 0.1:
        out.append(('view', dv, f'{100 * c["in_view"]:.0f} % of it is in CARLA\'s image, {100 * v["in_view"]:.0f} % '
                                'in the study\'s'))
    pl = row.get('placement') or {}
    if (pl.get('error_m') or 0.0) > 0.1 or abs(pl.get('error_z_m') or 0.0) > 0.05:
        out.append(('placement', 0.3, f'the actor stands {pl.get("error_m") or 0:.2f} m from its place and '
                                      f'{pl.get("error_z_m") or 0:+.2f} m off its height'))
    need = float(t['min_px'])
    if v['px'] > 0 and c['px'] > 0:
        rel = c['px'] / v['px'] - 1.0
        if abs(rel) >= 0.15:
            zo = c.get('z_own')
            h = f'; its own pixels reach {zo[1] - zo[0]:.2f} m of the {c["rect"]["height"]:.2f} m measured' if zo else ''
            out.append(('shape', 0.5 * abs(rel), f'its own pixels span {c["px"]:.0f} px where its rectangle spans '
                                                 f'{v["px"]:.0f} px ({100 * rel:+.0f} %): the actor does not fill its '
                                                 f'rectangle the way a flat target does{h}'))
    near_px = min(abs(c['px'] - need), abs(v['px'] - need)) <= 0.15 * need
    near_seen = min(abs(c['seen'] - t['min_share']), abs(v['seen'] - t['min_share'])) <= 0.05
    if (near_px and (c['px'] >= need) != (v['px'] >= need)) or \
            (near_seen and (c['seen'] >= t['min_share']) != (v['seen'] >= t['min_share'])):
        out.append(('threshold', 0.08, f'on the threshold: CARLA {c["px"]:.0f} px and {100 * c["seen"]:.0f} % seen, '
                                       f'the prediction {v["px"]:.0f} px and {100 * v["seen"]:.0f} % (needs '
                                       f'{need:g} px, {100 * t["min_share"]:.0f} %)'))
    if not out:
        out.append(('other', 0.0, f'CARLA {c["px"]:.0f} px, {100 * c["seen"]:.0f} % seen; the prediction '
                                  f'{v["px"]:.0f} px, {100 * v["seen"]:.0f} %'))
    out.sort(key=lambda r: -r[1])
    return out[0][0], out[0][2], out


@dataclass
class CriticalCarlaResult:
    settings: CriticalCarlaSettings
    plan: ScenePlan
    drives: list                     # [{'key', 'dir', 'group', 'spawn_point', 'reused', 'seconds', 'town', 'start',
                                     #   'ego', 'weather', 'status'}]
    rows: list                       # one per (target, camera, drive)
    targets: list                    # one per target: actor, sizes, resolved per drive
    metrics: dict
    findings: list
    variant: str = ''
    variant_uid: str = ''
    seconds: float = 0.0
    notes: list = field(default_factory=list)
    study: object = None             # core.critical's result (no envelopes)
    crops: dict = field(default_factory=dict)   # (uid, camera, drive) -> BGR crop (make_crops)

    def rows_of(self, uid=None, camera=None, drive=None, counted=None) -> list:
        return [r for r in self.rows if (uid is None or r['uid'] == uid) and (camera is None or r['camera'] == camera)
                and (drive is None or r['drive'] == drive) and (counted is None or r['counted'] == counted)]

    def target(self, key):
        return next((t for t in self.targets if t['uid'] == key or t['name'] == key), None)


def analyse(project, drives: list, settings: CriticalCarlaSettings | None = None, variant=None, engine=None,
            study=None, scene_plan: ScenePlan | None = None, progress=None, cancel=None) -> CriticalCarlaResult | None:
    """Compare the project's critical targets with CARLA drives of them (drives: [{'dir', 'key', 'group',
    'spawn_point'}] or folder paths; each drive's run.json names its actors by target uid). None when cancelled."""
    from .analysis import Engine
    t0 = time.perf_counter()
    st = settings or settings_of(project)
    p = CR._project_for(project, variant)
    v = p.variant
    cst = CR.settings_of(p)
    pl = scene_plan or plan(p, st, v, cst)
    eng = engine or Engine()
    if study is None:
        study = CR.run(p, eng, settings=cst, variant=v, cancel=cancel, envelopes=False)
        if study is None:
            return None
    vg = eng.vehicle_geom(p, v)
    occ = CR.Occluders(vg, p, v)
    cams = {c.uid: CR._Cam(p, c) for c in v.cameras if c.visible and CR._finite_cam(c)}
    focus = {uid: (cm.intr.focus_near if cm.intr.focus_near > 0 else 0.0, cm.intr.focus_far)
             for uid, cm in cams.items()}
    targets = {t['uid']: t for t in cst.targets if t['enabled']}
    plan_all = {r['uid']: r for r in pl.targets}
    vbox = CR.vehicle_box(p)
    rows, drv = [], []
    ds = [dict(d) if isinstance(d, dict) else {'dir': d, 'key': None, 'group': None, 'spawn_point': None}
          for d in drives]
    if any(not d.get('key') for d in ds) or len({d['key'] for d in ds}) < len(ds):
        # folders given (or two drives keyed alike, as per-seed subfolders named the same are): keyed by their path
        # under their common parent, so two drives never share a key
        for d, k in zip(ds, drive_keys([d['dir'] for d in ds])):
            d['key'] = k
    for di, dd in enumerate(ds):
        if cancel is not None and cancel():
            return None
        if progress is not None:
            progress(di / max(len(ds), 1), f'Measuring {dd["key"]}')
        from . import simrig as SR
        rows_d = {}                    # the actor each target had in this drive (the plan's, else by its blueprint)
        for arow in (SR.read_run(dd['dir']) or {}).get('actors') or []:
            uid = arow.get('id') if isinstance(arow, dict) else None
            if uid not in targets:
                continue
            a = (plan_all.get(uid) or {}).get('actor')
            bp = arow.get('blueprint')
            if bp and (a is None or a.get('blueprint') != bp):
                a = actor_for(p, targets[uid], 'blueprint:' + bp, vbox)
            if a is not None:
                rows_d[uid] = {'actor': a}
        m = measure_drive(dd['dir'], rows_d, st.frames, focus)
        run = m['run']
        ego = (run.get('vehicle') or {}).get('blueprint', '')
        info = dict(dd, town=run.get('town'), start=run.get('start'), ego=ego,
                    weather=(run.get('weather') or {}).get('preset') or (run.get('weather') or {}).get('vantage'),
                    status=run.get('status'), actors=run.get('actors') or [],
                    mounts_inside=[x['tag'] for x in run.get('mounts') or [] if x.get('kind') == 'camera' and
                                   (x.get('inside_body_m') or 0) > 0.005],
                    notes=list(run.get('notes') or []))
        drv.append(info)
        for uid, arow in m['actors'].items():
            if uid not in targets or uid not in rows_d:
                continue
            t = targets[uid]
            a = rows_d[uid]['actor']
            box = _row_box(arow) if arow.get('actor_id') is not None else None
            tr = study.target(uid)
            for cm in cams.values():
                nominal = next((lk for lk in tr.looks if lk.uid == cm.uid), None) if tr is not None else None
                base = {'uid': uid, 'target': t['name'], 'kind': t['kind'], 'camera': cm.tag, 'camera_uid': cm.uid,
                        'drive': dd['key'], 'drive_dir': dd['dir'], 'spawn_point': (run.get('start') or {}).get(
                            'spawn_point', dd.get('spawn_point')), 'ego': ego,
                        'vantage': _look_dict(nominal) if nominal is not None else None,
                        'placement': {k: arow.get(k) for k in ('error_m', 'error_z_m', 'error_yaw_deg',
                                                               'ground_gap_m')}}
                if box is None:
                    rows.append(dict(base, spawned=False, why=arow.get('error') or 'not spawned', vantage_actor=None,
                                     carla=None, counted=False, agree=None, agree_nominal=None, px_error=None,
                                     reason='', reasons=[]))
                    continue
                c = m['looks'].get((uid, cm.tag))
                rect = actor_rect(a, box, cm.pos[:2])
                ta = dict(t, x=rect['x'], y=rect['y'], base=rect['base'], height=max(rect['height'], 0.01),
                          width=max(rect['width'], 0.01))
                va = _look_dict(CR._look(cm, occ, ta)[0])
                if c is None:
                    c = {'in_view': 0.0, 'seen': 0.0, 'px': 0.0, 'px_spread': 0.0, 'hidden': {}, 'ego_hidden': 0.0,
                         'own_share': 0.0, 'z_own': None, 'dist': va['dist'], 'focus': va['focus'],
                         'pixels': 0, 'object': None, 'frames': 0, 'rect': rect, 'image_box': None, 'scale': 1.0,
                         'frame': None, 'object_id': None, 'camera_inside_m': 0.0}
                met, fail = verdict(c, t)
                c = dict(c, met=met, fail=fail)
                counted = va['in_view'] > 0 or c['in_view'] > 0 or (base['vantage'] or {}).get('in_view', 0) > 0
                row = dict(base, spawned=True, why='', vantage_actor=va, carla=c, counted=bool(counted),
                           agree=(va['met'] == met) if counted else None,
                           agree_nominal=(base['vantage']['met'] == met) if counted and base['vantage'] else None,
                           px_error=(c['px'] - va['px']) if (c['px'] > 0 and va['px'] > 0) else None,
                           reason='', reasons=[])
                if counted and (not row['agree'] or (row['px_error'] is not None and va['px'] > 0 and
                                                     abs(row['px_error']) / va['px'] >= 0.15)):
                    code, text, allr = explain(row, t)
                    row.update(reason=code, why=text, reasons=[{'code': k, 'text': x} for k, _, x in allr])
                rows.append(row)
    tg = _targets_table(pl, rows, drv, targets)
    res = CriticalCarlaResult(st, pl, drv, rows, tg, {}, [], v.name, v.uid, 0.0, list(pl.notes), study)
    res.metrics = _metrics(res)
    res.findings = _findings(res)
    res.seconds = time.perf_counter() - t0
    if progress is not None:
        progress(1.0, 'Done')
    return res


def _targets_table(pl: ScenePlan, rows, drives, targets) -> list:
    out = []
    for r in pl.targets:
        t = targets.get(r['uid'])
        if t is None:
            continue
        a = r['actor']
        mine = [x for x in rows if x['uid'] == r['uid']]
        spawned = []
        for d in drives:
            row = next((x for x in d.get('actors', []) if x.get('id') == r['uid']), None)
            if row is not None:
                spawned.append({'drive': d['key'], 'actor_id': row.get('actor_id'), 'blueprint': row.get('blueprint'),
                                'size_lwh': (row.get('bbox') or {}).get('size_lwh'), 'error_m': row.get('error_m'),
                                'error_z_m': row.get('error_z_m'), 'ground_gap_m': row.get('ground_gap_m'),
                                'error': row.get('error')})
        res_c, res_v, res_n = {}, {}, {}
        for x in mine:
            if not x.get('spawned') or x['carla'] is None:
                continue
            res_c.setdefault(x['drive'], False)
            res_v.setdefault(x['drive'], False)
            res_c[x['drive']] |= bool(x['carla']['met'])
            res_v[x['drive']] |= bool(x['vantage_actor']['met'])
        for x in mine:
            if x['vantage'] is not None:
                res_n[x['drive']] = res_n.get(x['drive'], False) or bool(x['vantage']['met'])
        out.append({'uid': r['uid'], 'name': r['name'], 'kind': r['kind'], 'place': r['place'],
                    'nominal': {'height': t['height'], 'width': t['width'], 'base': t['base'],
                                'min_px': t['min_px'], 'min_share': t['min_share']},
                    'actor': a, 'spawned': spawned, 'actor_height': actor_height(a, spawned),
                    'resolved_study': any(res_n.values()) if res_n else None,
                    'resolved_actor': res_v, 'resolved_carla': res_c,
                    'agree': {k: res_v[k] == res_c[k] for k in res_c}})
    return out


def actor_height(a: dict, spawned=()) -> float | None:
    """The height of the actor that the comparison measures: a board's band, a walker's or a two-wheeler's rendered
    height (measured alone, MEASURED_ON), else its box's (as spawned, else the table's)."""
    e = a.get('expected') or {}
    h = e.get('board_height') or e.get('rendered_height')
    if h:
        return float(h)
    box = next((s['size_lwh'] for s in spawned if s.get('size_lwh')), None)
    if box and a.get('family') not in ('prop', 'board'):
        return float(box[2])
    return float(e['height']) if e.get('height') else None


def _stats(xs) -> dict | None:
    xs = np.asarray([x for x in xs if x is not None and math.isfinite(x)], float)
    if not len(xs):
        return None
    return {'n': int(len(xs)), 'median': float(np.median(xs)), 'mean': float(xs.mean()),
            'rms': float(math.sqrt(float((xs ** 2).mean()))), 'min': float(xs.min()), 'max': float(xs.max())}


def _metrics(res: CriticalCarlaResult) -> dict:
    cnt = res.rows_of(counted=True)
    k = sum(1 for r in cnt if r['agree'])
    kn = sum(1 for r in cnt if r['agree_nominal'])
    nn = sum(1 for r in cnt if r['agree_nominal'] is not None)
    both = [r for r in cnt if r['px_error'] is not None]
    rel = [r['px_error'] / r['vantage_actor']['px'] for r in both if r['vantage_actor']['px'] > 0]
    clear = [r for r in both if r['carla']['seen'] >= 0.9 and r['vantage_actor']['seen'] >= 0.9]
    per_t = [(t['agree'].get(d), d) for t in res.targets for d in t['agree']]
    kt = sum(1 for a, _ in per_t if a)
    reasons = {}
    for r in cnt:
        if not r['agree']:
            reasons[r['reason']] = reasons.get(r['reason'], 0) + 1
    seen_err = [r['carla']['seen'] - r['vantage_actor']['seen'] for r in cnt if r['carla']['in_view'] > 0 and
                r['vantage_actor']['in_view'] > 0]
    ego = [r['carla']['ego_hidden'] - float(sum(r['vantage_actor']['hidden'].values())) for r in cnt
           if r['carla']['in_view'] > 0]
    return {'pairs': len(cnt), 'agree': k, 'agreement': k / len(cnt) if cnt else None, 'agreement_ci': wilson(k, len(cnt)),
            'pairs_nominal': nn, 'agree_nominal': kn, 'agreement_nominal': kn / nn if nn else None,
            'agreement_nominal_ci': wilson(kn, nn), 'targets_scenes': len(per_t), 'targets_agree': kt,
            'target_agreement': kt / len(per_t) if per_t else None, 'target_agreement_ci': wilson(kt, len(per_t)),
            'px_error': _stats([r['px_error'] for r in both]), 'px_error_rel': _stats(rel),
            'px_error_clear': _stats([r['px_error'] for r in clear]),
            'px_error_rel_clear': _stats([r['px_error'] / r['vantage_actor']['px'] for r in clear]),
            'seen_error': _stats(seen_err), 'ego_minus_body': _stats(ego), 'disagreements_by_reason': reasons,
            'drives': len(res.drives), 'targets': len(res.targets),
            'spawned': sum(1 for t in res.targets if t['actor']['spawnable']),
            'frames_spread_px': _stats([r['carla']['px_spread'] for r in cnt if r['carla']['frames'] > 1])}


def _findings(res: CriticalCarlaResult) -> list:
    out = []
    m = res.metrics
    if m['pairs']:
        ci = m['agreement_ci']
        out.append({'severity': 'good' if m['agreement'] >= 0.9 else 'warning', 'kind': 'agreement', 'target': '',
                    'message': f'CARLA agrees with the study\'s prediction at the actor\'s size in {m["agree"]} of '
                               f'{m["pairs"]} camera views ({100 * m["agreement"]:.0f} %, 95 % interval '
                               f'{100 * ci[0]:.0f}–{100 * ci[1]:.0f} %).'})
    for r in res.rows_of(counted=True):
        if r['agree']:
            continue
        c, va = r['carla'], r['vantage_actor']
        optimistic = va['met'] and not c['met']
        out.append({'severity': 'serious' if optimistic else 'warning', 'kind': 'disagree', 'target': r['uid'],
                    'camera': r['camera'], 'drive': r['drive'], 'reason': r['reason'],
                    'message': f'"{r["target"]}" in {r["camera"]} ({r["drive"]}): the study says it is '
                               f'{"met" if va["met"] else "not met"} ({va["px"]:.0f} px, {100 * va["seen"]:.0f} % '
                               f'seen), CARLA {"meets" if c["met"] else "does not meet"} it ({c["px"]:.0f} px, '
                               f'{100 * c["seen"]:.0f} % seen): {r["why"]}.'})
    for t in res.targets:
        if not t['actor']['spawnable']:
            out.append({'severity': 'info', 'kind': 'not_spawned', 'target': t['uid'],
                        'message': f'"{t["name"]}" is not stood in CARLA: {t["actor"]["why"]}.'})
        elif t['actor'].get('stand_in'):
            out.append({'severity': 'info', 'kind': 'stand_in', 'target': t['uid'],
                        'message': f'"{t["name"]}": {t["actor"]["note"]}.'})
    order = {'serious': 0, 'warning': 1, 'good': 2, 'info': 3}
    out.sort(key=lambda f: order.get(f['severity'], 9))
    return out


def run(project, engine=None, settings: CriticalCarlaSettings | None = None, variant=None, progress=None,
        cancel=None, sim_settings=None, out_dir=None, python=None, runner=None,
        memory_gb=BRIDGE_MEMORY_GB) -> CriticalCarlaResult | None:
    """The whole comparison: the study (no envelopes), the plan, every drive (scenes × spawn points) through the
    bridge on the CARLA server answering on the CARLA page's port (sim_settings to override), then the measurements and
    the comparison. out_dir holds one folder per drive (default <working folder>/<project>/carla/critical/<variant>);
    drives already there from the same rig are reused. None when cancelled; ValueError when no target can be stood in
    CARLA; RuntimeError when no server answers or a drive fails."""
    from . import simrig as SR
    from .. import workdir as WD
    from .analysis import Engine
    t0 = time.perf_counter()
    st = settings or settings_of(project)
    p = CR._project_for(project, variant)
    v = p.variant

    def prog(f, text=''):
        if progress is not None:
            progress(min(max(float(f), 0.0), 1.0), text)
    eng = engine or Engine()
    cst = CR.settings_of(p)
    if not any(t['enabled'] for t in cst.targets):
        raise ValueError('The project has no critical target: add them on Analysis ▸ Critical targets.')
    if not any(c.visible and CR._finite_cam(c) for c in v.cameras):
        raise ValueError(f'The variant {v.name} has no visible camera: nothing to compare in CARLA.')
    prog(0.0, 'The critical-targets study')
    study = CR.run(p, eng, settings=cst, variant=v, cancel=cancel, envelopes=False)
    if study is None:
        return None
    pl = plan(p, st, v, cst)
    if not pl.drives:
        raise ValueError('None of the critical targets can be stood in CARLA (kerbs, lane markings and traffic lights '
                         'cannot be spawned; choose a stand-in for a traffic light).')
    sim0 = sim_settings or SR.settings_of(p)
    root = os.path.abspath(out_dir or WD.project_dir(p, None, 'carla', 'critical', WD.slug(v.name)))
    done = []
    n = len(pl.drives)
    for i, d in enumerate(pl.drives):
        if cancel is not None and cancel():
            return None
        sim = scene_settings(p, st, pl.actors_json(d['group']), d['spawn_point'], sim0)

        def sub(f, text='', i=i, d=d):
            prog(0.05 + 0.85 * (i + f) / n, f'{d["key"]} ({i + 1} of {n}) · {text}')
        r = drive(p, v, sim, os.path.join(root, d['key']), python, sub, cancel, runner, memory_gb)
        if r is None:
            return None
        done.append(dict(d, dir=r['dir'], reused=r['reused'], seconds=round(r['seconds'], 2)))
    res = analyse(p, done, st, v, eng, study, pl, lambda f, t='': prog(0.9 + 0.1 * f, t), cancel)
    if res is None:
        return None
    res.seconds = time.perf_counter() - t0
    res.notes.append(f'Drives in {root}.')
    return res


# =========================================================================== output
def _r(v, nd=3):
    return None if v is None or not isinstance(v, (int, float)) or not math.isfinite(float(v)) else round(float(v), nd)


def _pct(v):
    return None if v is None else round(100.0 * float(v), 2)


def _look_doc(d: dict | None) -> dict | None:
    if d is None:
        return None
    return {'met': d['met'], 'px': _r(d['px'], 2), 'seen_pct': _pct(d['seen']), 'in_view_pct': _pct(d['in_view']),
            'hidden_pct': {k: _pct(x) for k, x in d['hidden'].items()}, 'fails_on': d.get('fail') or None,
            'distance_m': _r(d.get('dist'), 3)}


def _carla_doc(c: dict | None) -> dict | None:
    if c is None:
        return None
    o = c.get('object') or {}
    return {'met': c['met'], 'px': _r(c['px'], 2), 'px_spread_frames': _r(c['px_spread'], 2),
            'seen_pct': _pct(c['seen']), 'in_view_pct': _pct(c['in_view']),
            'hidden_pct': {k: _pct(x) for k, x in c['hidden'].items()}, 'ego_hidden_pct': _pct(c['ego_hidden']),
            'fails_on': c.get('fail') or None, 'distance_m': _r(c['dist'], 3), 'frames': c['frames'],
            'own_z_m': [_r(x, 3) for x in c['z_own']] if c.get('z_own') else None,
            'rectangle': {k: _r(x, 4) for k, x in c['rect'].items()},
            'ground_truth': {'object_id': o.get('id'), 'visible_pixels': o.get('pixels'), 'bbox': o.get('bbox'),
                             'bbox_full': o.get('bbox_full'), 'truncated': o.get('truncated'),
                             'occluded': o.get('occluded'), 'range_m': o.get('range_m'), 'class': o.get('class'),
                             'category': o.get('category')} if o else None,
            'image': (c.get('frame') or {}).get('image'), 'image_scale': _r(1.0 / c['scale'], 6) if c['scale'] else None,
            'camera_inside_car_m': _r(c.get('camera_inside_m'), 4)}


def summary(res: CriticalCarlaResult) -> dict:
    """Compact figures for the assistant and the batch log."""
    m = res.metrics
    return {'variant': res.variant, 'targets': m['targets'], 'stood_in_carla': m['spawned'], 'drives': m['drives'],
            'camera_views_compared': m['pairs'],
            'agreement_pct': _pct(m['agreement']), 'agreement_ci_pct': [_pct(x) for x in m['agreement_ci']]
            if m['agreement_ci'] else None,
            'agreement_with_nominal_study_pct': _pct(m['agreement_nominal']),
            'target_agreement_pct': _pct(m['target_agreement']),
            'px_error_median': _r((m['px_error'] or {}).get('median'), 2),
            'px_error_rel_median_pct': _pct((m['px_error_rel'] or {}).get('median')),
            'px_error_rel_median_clear_pct': _pct((m['px_error_rel_clear'] or {}).get('median')),
            'disagreements_by_reason': m['disagreements_by_reason'],
            'per_target': [{'name': t['name'], 'kind': t['kind'], 'actor': t['actor']['blueprint'],
                            'stood_in': t['actor']['spawnable'], 'why_not': t['actor']['why'] or None,
                            'resolved_study': t['resolved_study'], 'resolved_carla': t['resolved_carla'],
                            'agree': t['agree']} for t in res.targets],
            'disagreements': [{'target': r['target'], 'camera': r['camera'], 'drive': r['drive'],
                               'study': r['vantage_actor']['met'], 'carla': r['carla']['met'], 'why': r['why']}
                              for r in res.rows_of(counted=True) if not r['agree']][:20],
            'assumptions': ASSUMPTIONS}


def to_doc(res: CriticalCarlaResult, project=None) -> dict:
    """The JSON export: the settings, the drives, every target's actor and sizes, every (target, camera, drive) with
    the study's look, the prediction at the actor's size and CARLA's measurement, the metrics and the findings."""
    m = res.metrics

    def st_doc(s):
        return None if s is None else {k: (_r(x, 4) if isinstance(x, float) else x) for k, x in s.items()}
    tg = []
    for t in res.targets:
        a = t['actor']
        tg.append({'name': t['name'], 'kind': t['kind'], 'where': t['place'], 'nominal_m': t['nominal'],
                   'actor': {'stood_in': a['spawnable'], 'why_not': a['why'] or None, 'blueprint': a['blueprint'],
                             'family': a['family'], 'stand_in': a['stand_in'], 'choice': a['choice'],
                             'note': a['note'] or None, 'pose_vehicle': a['pose'], 'expected_m': a['expected'],
                             'board': a['board']},
                   'spawned': t['spawned'], 'actor_height_m': _r(t['actor_height'], 3),
                   'resolved': {'study_nominal': t['resolved_study'], 'study_at_actor_size': t['resolved_actor'],
                                'carla': t['resolved_carla'], 'agree': t['agree']}})
    rows = []
    for r in res.rows:
        rows.append({'target': r['target'], 'kind': r['kind'], 'camera': r['camera'], 'drive': r['drive'],
                     'spawn_point': r['spawn_point'], 'stood_in': r['spawned'], 'counted': r['counted'],
                     'study_nominal': _look_doc(r['vantage']), 'study_at_actor_size': _look_doc(r['vantage_actor']),
                     'carla': _carla_doc(r['carla']), 'agree': r['agree'], 'agree_with_nominal': r['agree_nominal'],
                     'px_error': _r(r['px_error'], 2), 'reason': r['reason'] or None, 'why': r['why'] or None,
                     'reasons': r['reasons'], 'placement': r['placement']})
    dr = [{'key': d['key'], 'folder': os.path.basename(d['dir']), 'spawn_point': d.get('spawn_point'),
           'start': d.get('start'), 'town': d.get('town'), 'weather': d.get('weather'), 'ego': d.get('ego'),
           'reused': d.get('reused'), 'seconds': d.get('seconds'), 'cameras_inside_car': d.get('mounts_inside'),
           'notes': d.get('notes')} for d in res.drives]
    st = res.settings
    return {'schema': SCHEMA, 'project': getattr(project, 'name', ''), 'variant': res.variant, 'frame': FRAME,
            'carla_version': CARLA_VERSION,
            'units': {'lengths': 'm', 'px': 'sensor pixels (the module\'s full resolution)', 'shares': 'percent'},
            'settings': st.to_dict(), 'drives': dr, 'targets': tg, 'rows': rows,
            'metrics': {k: (st_doc(x) if isinstance(x, dict) and k != 'disagreements_by_reason' else
                            [_r(y, 4) for y in x] if isinstance(x, tuple) else _r(x, 4) if isinstance(x, float)
                            else x) for k, x in m.items()},
            'findings': [{k: f[k] for k in ('severity', 'kind', 'message') if k in f} for f in res.findings],
            'method': METHOD, 'assumptions': ASSUMPTIONS, 'limits': LIMITS_TEXT, 'measured_on': MEASURED_ON,
            'reasons': REASONS, 'notes': res.notes}


CSV_COLUMNS = ('target', 'kind', 'camera', 'drive', 'spawn_point', 'stood_in', 'counted', 'study_met', 'study_px',
               'study_seen_pct', 'actor_size_met', 'actor_size_px', 'actor_size_seen_pct', 'carla_met', 'carla_px',
               'carla_px_spread', 'carla_seen_pct', 'carla_in_view_pct', 'carla_ego_hidden_pct',
               'carla_other_hidden_pct', 'carla_visible_pixels', 'carla_truncated', 'carla_occluded', 'distance_m',
               'agree', 'px_error', 'reason', 'why')


def to_csv(res: CriticalCarlaResult) -> str:
    """One row per target, camera and drive."""
    buf = io.StringIO()
    w = csv.writer(buf, lineterminator='\n')
    w.writerow(CSV_COLUMNS)
    b = lambda x: '' if x is None else x                       # noqa: E731
    for r in res.rows:
        v, va, c = r['vantage'] or {}, r['vantage_actor'] or {}, r['carla'] or {}
        o = c.get('object') or {}
        w.writerow([r['target'], r['kind'], r['camera'], r['drive'], b(r['spawn_point']), r['spawned'], r['counted'],
                    b(v.get('met')), b(_r(v.get('px'), 1)), b(_pct(v.get('seen'))), b(va.get('met')),
                    b(_r(va.get('px'), 1)), b(_pct(va.get('seen'))), b(c.get('met')), b(_r(c.get('px'), 1)),
                    b(_r(c.get('px_spread'), 1)), b(_pct(c.get('seen'))), b(_pct(c.get('in_view'))),
                    b(_pct(c.get('ego_hidden'))),
                    b(_pct(sum(x for k, x in (c.get('hidden') or {}).items() if k != 'ego')) if c else None),
                    b(o.get('pixels')), b(o.get('truncated')), b(o.get('occluded')), b(_r(c.get('dist'), 2)),
                    b(r['agree']), b(_r(r['px_error'], 1)), r['reason'], r['why']])
    return buf.getvalue()


TARGET_COLUMNS = ('target', 'kind', 'where', 'nominal_height_m', 'nominal_width_m', 'above_ground_m', 'stood_in',
                  'why_not', 'blueprint', 'stand_in', 'actor_box_l_m', 'actor_box_w_m', 'actor_box_h_m',
                  'actor_height_m', 'height_vs_nominal_pct', 'resolved_study', 'resolved_carla', 'agree')


def targets_csv(res: CriticalCarlaResult) -> str:
    """One row per target: its actor, its size as spawned against the nominal, resolved in the study and in CARLA."""
    buf = io.StringIO()
    w = csv.writer(buf, lineterminator='\n')
    w.writerow(TARGET_COLUMNS)
    for t in res.targets:
        a = t['actor']
        sz = next((s['size_lwh'] for s in t['spawned'] if s.get('size_lwh')), None) or [None] * 3
        hh = t['actor_height']
        res_c = t['resolved_carla']
        w.writerow([t['name'], t['kind'], t['place'], t['nominal']['height'], t['nominal']['width'],
                    t['nominal']['base'], a['spawnable'], a['why'], a['blueprint'] or '', a['stand_in'],
                    *(_r(x, 3) if x is not None else '' for x in sz), _r(hh, 3) if hh else '',
                    _r(100.0 * (hh / t['nominal']['height'] - 1.0), 1) if hh else '',
                    t['resolved_study'] if t['resolved_study'] is not None else '',
                    ' '.join(f'{k}:{"yes" if x else "no"}' for k, x in res_c.items()),
                    ' '.join(f'{k}:{"yes" if x else "no"}' for k, x in t['agree'].items())])
    return buf.getvalue()


def _bgr(h):
    return int(h[5:7], 16), int(h[3:5], 16), int(h[1:3], 16)


def chart_png(res: CriticalCarlaResult, size=900) -> bytes:
    """CARLA's pixels against the prediction at the actor's size, every compared camera view (log scales): green
    agreeing, red not; the 1:1 line; each target's need dashed where it crosses."""
    import cv2
    pts = [(r['vantage_actor']['px'], r['carla']['px'], r['agree'], r) for r in res.rows_of(counted=True)
           if r['vantage_actor']['px'] > 0 or r['carla']['px'] > 0]
    W, H = size, size
    img = np.full((H, W, 3), (30, 29, 28), np.uint8)
    font, aa = cv2.FONT_HERSHEY_SIMPLEX, cv2.LINE_AA
    x0, y0, x1, y1 = 80, 60, W - 30, H - 70
    pos_ = [x for a, b, _, _ in pts for x in (a, b) if x > 0]
    hi = max(pos_ + [100.0]) * 1.3
    lo = 10.0 ** math.floor(math.log10(max(min(pos_) / 1.5, 1.0))) if pos_ else 1.0
    L0, L1 = math.log10(lo), math.log10(hi)

    def X(v):
        return int(x0 + (math.log10(max(v, lo)) - L0) / (L1 - L0) * (x1 - x0))

    def Y(v):
        return int(y1 - (math.log10(max(v, lo)) - L0) / (L1 - L0) * (y1 - y0))
    cv2.rectangle(img, (x0, y0), (x1, y1), (62, 61, 58), 1, aa)
    for k in range(0, int(math.ceil(L1)) + 1):
        for m in (1, 2, 5):
            v = m * 10 ** k
            if lo <= v <= hi:
                cv2.line(img, (X(v), y0), (X(v), y1), (45, 44, 42), 1)
                cv2.line(img, (x0, Y(v)), (x1, Y(v)), (45, 44, 42), 1)
                cv2.putText(img, f'{v:g}', (X(v) - 8, y1 + 18), font, 0.4, (165, 164, 160), 1, aa)
                cv2.putText(img, f'{v:g}', (x0 - 44, Y(v) + 4), font, 0.4, (165, 164, 160), 1, aa)
    cv2.line(img, (X(lo), Y(lo)), (X(hi), Y(hi)), (150, 148, 142), 1, aa)
    for a, b, ok, r in pts:
        col = _bgr(CR.GOOD if ok else CR.BAD)
        cv2.circle(img, (X(max(a, lo)), Y(max(b, lo))), 5, col, -1, aa)
    m = res.metrics
    title = f'Critical targets in CARLA  {res.variant}'
    cv2.putText(img, CR._ascii(title), (16, 26), font, 0.6, (236, 235, 230), 1, aa)
    if m['pairs']:
        cv2.putText(img, CR._ascii(f'{m["agree"]} of {m["pairs"]} camera views agree ({100 * m["agreement"]:.0f} %); '
                                   'x: the study at the actor\'s size, y: CARLA (sensor px, log)'), (16, 46), font,
                    0.42, (165, 164, 160), 1, aa)
    cv2.putText(img, 'study px (actor size)', (x1 - 170, H - 30), font, 0.42, (195, 194, 183), 1, aa)
    cv2.putText(img, 'CARLA px', (16, y0 - 8), font, 0.42, (195, 194, 183), 1, aa)
    ok, buf = cv2.imencode('.png', img)
    return buf.tobytes()


def crop(res: CriticalCarlaResult, row: dict, height: int = 220):
    """BGR image: the camera frame around the target (its rectangle and its pixels, with a margin), CARLA's car
    tinted, the actor's own pixels outlined (green), the rectangle the study measures at the actor's size (amber),
    and the figures under it; None when CARLA has no frame of it."""
    import cv2
    c = row.get('carla')
    if not c or not c.get('frame') or not c['frame'].get('image'):
        return None
    root = row['drive_dir']
    img = cv2.imread(os.path.join(root, c['frame']['image']), cv2.IMREAD_COLOR)
    if img is None:
        return None
    h, w = img.shape[:2]
    try:
        inst = cv2.imread(os.path.join(root, c['frame']['instance']), cv2.IMREAD_UNCHANGED)
        sem = cv2.imread(os.path.join(root, c['frame']['semantic']), cv2.IMREAD_UNCHANGED)
    except (TypeError, cv2.error):
        inst = sem = None
    from . import simrig as SR
    rig = with_actual_mounts(_load_json(os.path.join(root, 'rig', 'sim_rig.json')) or {}, SR.read_run(root))
    cam = next((x for x in rig.get('cameras', []) if x['tag'] == row['camera']), None)
    poly = None
    if cam is not None:
        R, pos = cam_pose(cam)
        loop = rect_outline(c['rect'], pos, 40)
        u, v, _, ok = GT.project(cam['lens'], (loop - pos) @ (BODY_TO_OPTICAL @ R.T).T)
        if ok.any():
            poly = np.stack([u, v], 1)[ok]
    boxes = []
    if poly is not None and len(poly):
        boxes.append([poly[:, 0].min(), poly[:, 1].min(), poly[:, 0].max(), poly[:, 1].max()])
    o = c.get('object') or {}
    if o.get('bbox'):
        boxes.append(o['bbox'])
    if not boxes:
        return None
    bx = np.array(boxes, float)
    xa, ya, xb, yb = bx[:, 0].min(), bx[:, 1].min(), bx[:, 2].max(), bx[:, 3].max()
    pad = max(0.3 * max(xb - xa, yb - ya), 12.0)
    xa, ya = int(max(xa - pad, 0)), int(max(ya - pad, 0))
    xb, yb = int(min(xb + pad, w)), int(min(yb + pad, h))
    if xb - xa < 4 or yb - ya < 4:
        return None
    out = img.copy()
    if sem is not None and sem.shape[:2] == (h, w):
        ego = sem == EGO
        out[ego] = (0.6 * out[ego] + 0.4 * np.array([200, 200, 0])).astype(np.uint8)
    if inst is not None and inst.shape[:2] == (h, w) and c.get('object_id') is not None:
        mask = (inst == int(c['object_id'])).astype(np.uint8)
        cs, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        cv2.drawContours(out, cs, -1, (60, 220, 60), 1, cv2.LINE_AA)
    if poly is not None and len(poly) > 1:
        cv2.polylines(out, [np.round(poly * 4).astype(np.int32)], True, (40, 180, 250), 1, cv2.LINE_AA, shift=2)
    sub = out[ya:yb, xa:xb]
    s = height / max(sub.shape[0], 1)
    sub = cv2.resize(sub, (max(int(round(sub.shape[1] * s)), 1), height),
                     interpolation=cv2.INTER_NEAREST if s > 1 else cv2.INTER_AREA)
    ww = max(sub.shape[1], 200)
    canvas = np.full((height + 34, ww, 3), (30, 29, 28), np.uint8)
    canvas[:height, :sub.shape[1]] = sub
    va = row['vantage_actor']
    col = _bgr(CR.GOOD if row['agree'] else CR.BAD) if row['agree'] is not None else (165, 164, 160)
    cv2.putText(canvas, CR._ascii(f'{row["camera"]}  CARLA {c["px"]:.0f} px {100 * c["seen"]:.0f}%'),
                (4, height + 14), cv2.FONT_HERSHEY_SIMPLEX, 0.4, col, 1, cv2.LINE_AA)
    cv2.putText(canvas, CR._ascii(f'study {va["px"]:.0f} px {100 * va["seen"]:.0f}%'), (4, height + 29),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (195, 194, 183), 1, cv2.LINE_AA)
    return canvas


def row_key(row: dict) -> tuple:
    return row['uid'], row['camera'], row['drive']


def make_crops(res: CriticalCarlaResult, height: int = 220, progress=None, cancel=None) -> bool:
    """Fill res.crops for every compared view (a few tens of ms each: the page does it in its job); False when
    cancelled."""
    rows = res.rows_of(counted=True)[:MAX_CROPS]
    for i, r in enumerate(rows):
        if cancel is not None and cancel():
            return False
        if row_key(r) not in res.crops:
            im = crop(res, r, height)
            if im is not None:
                res.crops[row_key(r)] = im
        if progress is not None:
            progress((i + 1) / max(len(rows), 1), 'Crops')
    return True


def _slug(s):
    return re.sub(r'[^A-Za-z0-9]+', '-', s).strip('-').lower() or 'x'


def files(res: CriticalCarlaResult, project=None) -> list:
    """[(name, bytes)]: critical_carla.json, critical_carla.csv (target × camera × drive), critical_carla_targets.csv,
    critical_carla.png (the pixel chart) and crops/<target>__<camera>__<drive>.jpg of every compared view."""
    import cv2
    out = [('critical_carla.json', (json.dumps(to_doc(res, project), indent=1, default=str) + '\n').encode('utf-8')),
           ('critical_carla.csv', to_csv(res).encode('utf-8')),
           ('critical_carla_targets.csv', targets_csv(res).encode('utf-8')),
           ('critical_carla.png', chart_png(res))]
    k = 0
    names = set()
    for r in res.rows_of(counted=True):
        if k >= MAX_CROPS:
            break
        im = res.crops.get(row_key(r))
        if im is None:
            im = crop(res, r)
        if im is None:
            continue
        ok, buf = cv2.imencode('.jpg', im, [int(cv2.IMWRITE_JPEG_QUALITY), 90])
        if ok:
            stem = f'crops/{_slug(r["target"])}__{_slug(r["camera"])}__{_slug(r["drive"])}'
            name, n = f'{stem}.jpg', 2
            while name in names:             # drive keys alike once slugged ('a-b/c', 'a/b-c')
                name, n = f'{stem}-{n}.jpg', n + 1
            names.add(name)
            out.append((name, buf.tobytes()))
            k += 1
    return out


def export(res: CriticalCarlaResult, project, out_dir) -> list[str]:
    """Write files() into out_dir; returns their paths."""
    paths = []
    for name, data in files(res, project):
        path = os.path.join(out_dir, *name.split('/'))
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, 'wb') as fh:
            fh.write(data)
        paths.append(path)
    return paths


def is_drive(folder: str) -> bool:
    """A drive folder: run.json and the ground truth's index."""
    return bool(folder) and os.path.isfile(os.path.join(folder, 'run.json')) and \
        os.path.isfile(os.path.join(folder, GT.INDEX))


DRIVE_SEARCH_DEPTH = 6                       # folder levels Open drives… looks down for drives
DRIVE_SEARCH_DIRS = 20000                    # folders it looks into at most (a home folder chosen by mistake)


def drives_in(folder: str, depth: int = DRIVE_SEARCH_DEPTH) -> list:
    """The drive folders at any depth under folder (each with run.json and ground truth), sorted by their path under
    it: to analyse drives again without CARLA, per-seed subfolders included. A folder holding run.json is a drive's
    own (its nuScenes folder, its rig) and is not looked into; hidden folders are skipped; a linked folder is followed
    once (a link back up the tree is not)."""
    out = []
    if not folder or not os.path.isdir(folder):
        return out
    seen = {os.path.realpath(folder)}
    todo = [(folder, 0)]
    looked = 0
    while todo and looked < DRIVE_SEARCH_DIRS:
        d, k = todo.pop()
        looked += 1
        try:
            names = sorted(os.listdir(d))
        except OSError:
            continue
        for name in names:
            sub = os.path.join(d, name)
            if name.startswith('.') or not os.path.isdir(sub):
                continue
            real = os.path.realpath(sub)
            if real in seen:
                continue
            seen.add(real)
            if is_drive(sub):
                out.append(sub)
            elif k + 1 < depth and not os.path.isfile(os.path.join(sub, 'run.json')):
                todo.append((sub, k + 1))
    return sorted(out, key=lambda x: os.path.relpath(x, folder).split(os.sep))


def drive_keys(dirs: list) -> list:
    """A key per drive folder, never the same for two of them: its path under the folders' common parent, '/'
    between the levels ('seed2/scene1-spauto'); a lone drive, or drives side by side in one folder, keep their folder
    names."""
    dirs = [os.path.normpath(os.path.abspath(str(d))) for d in dirs]
    if not dirs:
        return []
    parents = [os.path.dirname(d) for d in dirs]
    try:
        root = os.path.commonpath(parents)
    except ValueError:                       # (different drives on Windows)
        root = ''
    keys = [os.path.relpath(d, root).replace(os.sep, '/') if root else d.replace(os.sep, '/') for d in dirs]
    seen = {}
    for i, k in enumerate(keys):             # the same folder twice: numbered
        n = seen.get(k, 0)
        seen[k] = n + 1
        if n:
            keys[i] = f'{k} ({n + 1})'
    return keys


def copy_settings(st: CriticalCarlaSettings, **kw) -> CriticalCarlaSettings:
    return CriticalCarlaSettings.from_dict(dict(copy.deepcopy(st.to_dict()), **kw))
