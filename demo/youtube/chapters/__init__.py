"""The tour's chapters: the plan (order, parts, titles) and the chapter files that script them.

A chapter is a file `chapters/<nn>_<id>.py` (nn two digits) defining

    META  = {'id': ..., 'part': 1..5, 'title': ..., 'promise': ..., 'menu': ...}
    LINES = [(caption, spoken), ...]
    def scene(stage, r): ...

The intro is 00_intro.py (part 0) and the outro 99_outro.py (part 6). META and LINES are read here with `ast`
(literal values only), so the voice renderer and produce.py never import Qt or the app: keep them plain literals.

Episodes: the chapter files of every video live here side by side (ids are unique across them); an episode is a plan
over them (EPISODES: its parts, chapters, intro, outro and the texts of its deliverables). 'tour' is the full app tour
(PLAN, PARTS); 'v020' is "What's new in Vantage 0.2.0" (files 40_intro-v020 ... 49_outro-v020). The episode in use is
$VANTAGE_EPISODE (default 'tour'; produce.py --episode sets it for every child process); a chapter's own episode is
episode_of(cid), so record.py and cards.py need no variable.
"""
from __future__ import annotations

import ast
import hashlib
import importlib.util
import os
import re

DIR = os.path.dirname(os.path.abspath(__file__))

PARTS = {
    1: ('Design the rig', 'Bring in the vehicle, choose modules and lenses, and place every sensor.'),
    2: ('Prove what it sees', 'Coverage, blind spots, ranges and rules, recomputed as the rig changes.'),
    3: ('Studies', 'Tolerances, loads, timing, data, power, sky, sun and brackets.'),
    4: ('Calibrate and verify', 'From the first checkerboard to a fleet: is the rig as built the rig as planned?'),
    5: ('Data and hand-off', 'Sample data, calibration in every dataset’s format, batch mode and installs.'),
}

# (nn, id, part, title): the tour's 38 chapters in order
PLAN = [
    (1, 'projects', 1, 'Projects and the window'),
    (2, 'import', 1, 'Importing a vehicle: the wizard'),
    (3, 'viewport', 1, 'Moving around in 3D and the renderer'),
    (4, 'library', 1, 'Camera modules and lenses: the e-con library'),
    (5, 'placement', 1, 'Placing cameras'),
    (6, 'rig', 1, 'Rig parts'),
    (7, 'sensors', 1, 'Lidars, radars, GNSS and IMU'),
    (8, 'housings', 1, 'Housings block cameras'),
    (9, 'coverage', 2, 'Live coverage analysis'),
    (10, 'camviews', 2, "Camera views and the stitched bird's-eye"),
    (11, 'bowl', 2, '3D surround view'),
    (12, 'guidelines', 2, 'Parking guidelines'),
    (13, 'volume', 2, '3D blind volume'),
    (14, 'perception', 2, 'Perception range'),
    (15, 'sensorcov', 2, 'Lidar and radar coverage'),
    (16, 'issues', 2, 'Issues and rules'),
    (17, 'tolerance', 3, 'Tolerance, optimiser and variants'),
    (18, 'loads', 3, 'Load cases and suspension'),
    (19, 'timing', 3, 'Sync and timing'),
    (20, 'recording', 3, 'Recording budget'),
    (21, 'power', 3, 'Power budget'),
    (22, 'gnss', 3, 'GNSS and IMU'),
    (23, 'exposure', 3, 'Sun glare and dirt'),
    (24, 'brackets', 3, 'Mount brackets'),
    (25, 'requirements', 3, 'Requirements scorecard'),
    (26, 'assistant', 3, 'The assistant'),
    (27, 'targets', 4, 'Calibration targets'),
    (28, 'lens', 4, 'Unit lens calibration'),
    (29, 'install', 4, 'Installation check and joint solve'),
    (30, 'lidarcal', 4, 'Lidar–camera calibration'),
    (31, 'liverig', 4, 'Live rig health'),
    (32, 'drivelog', 4, 'Drive-log check'),
    (33, 'asbuilt', 4, 'As-built check'),
    (34, 'fleet', 4, 'Fleet'),
    (35, 'dataset', 5, 'Sample dataset and ROS 2 bags'),
    (36, 'exports', 5, 'Exports and dataset calibration'),
    (37, 'batch', 5, 'Batch mode and CI'),
    (38, 'install-anywhere', 5, 'Downloads, self-test and running on the Jetson'),
]
INTRO, OUTRO = 'intro', 'outro'

# "What's new in Vantage 0.2.0": the files are 41_new-critical.py ... 48_new-automation.py; the numbers on screen
# (cards, the indicator, YouTube chapters) count this video's own chapters, 1 to 8
PARTS_V020 = {
    1: ('On the road', 'What the rig must resolve, how the road moves it, and what rain, fog and night take away.'),
    2: ('In the workshop', 'What each sensor looks through, the cables behind it, and how well it can be calibrated.'),
    3: ('In the pipeline', 'A more accurate calibration, and the whole rig checked on every commit.'),
}
PLAN_V020 = [
    (1, 'new-critical', 1, 'Critical targets and mounting envelopes'),
    (2, 'new-terrain', 1, 'Road and body motion'),
    (3, 'new-weather', 1, 'Weather and night'),
    (4, 'new-materials', 2, 'Materials, glass and wipers'),
    (5, 'new-harness', 2, 'Cable harness'),
    (6, 'new-observability', 2, 'Calibration observability and fusion'),
    (7, 'new-calib', 3, 'A more accurate calibration'),
    (8, 'new-automation', 3, 'Automation and CI'),
]

# what an episode is: its plan, and what its deliverables say (assemble.py reads the texts; None = the tour's own)
EPISODES = {
    'tour': {
        'parts': PARTS, 'plan': PLAN, 'intro': INTRO, 'outro': OUTRO, 'version': '0.1.1',
        'name': 'vantage-tour',
        'title': 'Vantage: full tour of the sensor rig planner for autonomous-driving data collection',
        'video_title': 'Vantage: full app tour',
        'marks': ('Intro: what Vantage is', 'Download, website and licence'),
        'texts': ('chapters.txt', 'description.md', 'thumbnail.png'),     # in demo/out/youtube
        'description': None,
        'tags': None,
        'thumb': {'still': 'intro', 'eyebrow': 'FULL APP TOUR', 'word': 'Vantage', 'word_px': 230,
                  'lines': ('Plan a sensor rig.', 'Prove what it sees.'), 'foot': 'FREE AND OPEN SOURCE'},
    },
    'v020': {
        'parts': PARTS_V020, 'plan': PLAN_V020, 'intro': 'intro-v020', 'outro': 'outro-v020', 'version': '0.2.0',
        'name': 'vantage-whats-new-0.2.0',
        'title': "What's new in Vantage 0.2.0: targets, road, weather, materials, harness, calibration, CI",
        'video_title': "What's new in Vantage 0.2.0",
        'marks': ("Intro: what's new in 0.2.0", 'Download 0.2.0, website and licence'),
        'texts': ('{name}-chapters.txt', '{name}-description.md', '{name}-thumbnail.png'),
        'description': 'v020',
        'tags': 'v020',
        'thumb': {'still': 'intro-v020', 'shot_x': 640, 'eyebrow': "WHAT'S NEW IN", 'word': 'Vantage 0.2.0', 'word_px': 156,
                  'lines': ('Targets, road, weather,', 'harness, calibration, CI.'), 'foot': 'FREE AND OPEN SOURCE'},
    },
}
DEFAULT_EPISODE = 'tour'


def current() -> str:
    """The episode in use: $VANTAGE_EPISODE, else the tour."""
    ep = os.environ.get('VANTAGE_EPISODE') or DEFAULT_EPISODE
    if ep not in EPISODES:
        raise SystemExit(f'unknown episode {ep!r} (VANTAGE_EPISODE): one of {", ".join(EPISODES)}')
    return ep


def episode(name=None) -> dict:
    return EPISODES[name or current()]


def members(name) -> list[str]:
    """The segment ids of an episode in order (intro, chapters, outro), whether recorded or not."""
    e = EPISODES[name]
    return [e['intro']] + [c[1] for c in e['plan']] + [e['outro']]


def episode_of(cid, default=None) -> str:
    """The episode a chapter id belongs to (else `default`, else the current one)."""
    for name in EPISODES:
        if cid in members(name):
            return name
    return default or current()


def resolve(name=None, only=None) -> str:
    """The episode to produce: the one asked for, else $VANTAGE_EPISODE, else the one every id in `only` belongs to,
    else the tour."""
    if name:
        if name not in EPISODES:
            raise SystemExit(f'unknown episode {name!r}: one of {", ".join(EPISODES)}')
        return name
    if os.environ.get('VANTAGE_EPISODE'):
        return current()
    eps = {episode_of(c, DEFAULT_EPISODE) for c in (only or [])}
    return eps.pop() if len(eps) == 1 else DEFAULT_EPISODE


_NAME = re.compile(r'^(\d\d)_([a-z0-9-]+)\.py$')


def files() -> dict[str, str]:
    """chapter id -> its file, for every chapter file present."""
    out = {}
    for fn in sorted(os.listdir(DIR)):
        m = _NAME.match(fn)
        if m:
            out[m.group(2)] = os.path.join(DIR, fn)
    return out


def _literal(path, name):
    tree = ast.parse(open(path, encoding='utf-8').read(), path)
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == name for t in node.targets):
            return ast.literal_eval(node.value)
    raise ValueError(f'{path}: no literal {name}')


def meta(cid) -> dict:
    return _literal(files()[cid], 'META')


def lines(cid) -> list[tuple[str, str]]:
    return [tuple(x) for x in _literal(files()[cid], 'LINES')]


def source_hash(cid) -> str:
    return hashlib.sha1(open(files()[cid], 'rb').read()).hexdigest()[:16]


def load(cid):
    """Import a chapter file as a module (any file name, hyphens included)."""
    path = files()[cid]
    spec = importlib.util.spec_from_file_location(f'tour_chapter_{cid.replace("-", "_")}', path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def plan_entry(cid):
    return next((c for c in episode(episode_of(cid))['plan'] if c[1] == cid), None)


def sequence(only=None, name=None):
    """An episode's segments in order (default: the current episode), for the chapters whose files exist (or `only`
    of them): a list of dicts {'kind': 'segment'|'part'|'chapter-card', ...}. The intro opens and the outro closes it
    when present."""
    ep = episode(name)
    plan, parts, intro, outro = ep['plan'], ep['parts'], ep['intro'], ep['outro']
    have = files()
    want = set(only) if only else None
    seq = []
    if intro in have and (want is None or intro in want):
        seq.append({'kind': 'segment', 'id': intro})
    part = None
    for nn, cid, pt, title in plan:
        if cid not in have or (want is not None and cid not in want):
            continue
        if pt != part:
            part = pt
            items = [(n, t) for n, c, p_, t in plan if p_ == pt]
            seq.append({'kind': 'part', 'part': pt, 'title': parts[pt][0], 'blurb': parts[pt][1], 'items': items,
                        'next': cid, 'of': len(parts)})
        m = meta(cid)
        seq.append({'kind': 'chapter-card', 'id': cid, 'nn': nn, 'part': pt, 'part_title': parts[pt][0],
                    'title': m.get('title', title), 'promise': m.get('promise', ''), 'menu': m.get('menu', '')})
        seq.append({'kind': 'segment', 'id': cid})
    if outro in have and (want is None or outro in want):
        seq.append({'kind': 'segment', 'id': outro})
    return seq
