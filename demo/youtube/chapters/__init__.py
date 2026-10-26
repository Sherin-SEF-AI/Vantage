"""The tour's chapters: the plan (order, parts, titles) and the chapter files that script them.

A chapter is a file `chapters/<nn>_<id>.py` (nn two digits) defining

    META  = {'id': ..., 'part': 1..5, 'title': ..., 'promise': ..., 'menu': ...}
    LINES = [(caption, spoken), ...]
    def scene(stage, r): ...

The intro is 00_intro.py (part 0) and the outro 99_outro.py (part 6). META and LINES are read here with `ast`
(literal values only), so the voice renderer and produce.py never import Qt or the app: keep them plain literals.
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
    return next((c for c in PLAN if c[1] == cid), None)


def sequence(only=None):
    """The video's segments in order, for the chapters whose files exist (or `only` of them): a list of dicts
    {'kind': 'segment'|'part'|'chapter-card', ...}. The intro opens and the outro closes it when present."""
    have = files()
    want = set(only) if only else None
    seq = []
    if INTRO in have and (want is None or INTRO in want):
        seq.append({'kind': 'segment', 'id': INTRO})
    part = None
    for nn, cid, pt, title in PLAN:
        if cid not in have or (want is not None and cid not in want):
            continue
        if pt != part:
            part = pt
            items = [(n, t) for n, c, p_, t in PLAN if p_ == pt]
            seq.append({'kind': 'part', 'part': pt, 'title': PARTS[pt][0], 'blurb': PARTS[pt][1], 'items': items,
                        'next': cid})
        m = meta(cid)
        seq.append({'kind': 'chapter-card', 'id': cid, 'nn': nn, 'part': pt, 'part_title': PARTS[pt][0],
                    'title': m.get('title', title), 'promise': m.get('promise', ''), 'menu': m.get('menu', '')})
        seq.append({'kind': 'segment', 'id': cid})
    if OUTRO in have and (want is None or OUTRO in want):
        seq.append({'kind': 'segment', 'id': OUTRO})
    return seq
