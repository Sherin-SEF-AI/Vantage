"""Importing a vehicle model without the window: the import wizard's steps (File ▸ Import ▸ Vehicle model…) with its
own guesses, for Claude over MCP and for scripts.

    vehicle, report = import_vehicle('car.glb')                       # units, up and forward guessed
    vehicle, report = import_vehicle('car.obj', units='mm', up='+Y', forward='-Z', origin='rear_axle')

The steps, as the wizard takes them:

1. Load the file (core.loaders: .glb .gltf .obj .stl .ply .dae .fbx, .step with gmsh). Each part gets a role from its
   name and material (detect_roles): glass, wheels, ignored (interior) or mountable.
2. Units: the file's own (glTF is metres) when its size fits a car, else the one unit that makes the longest side
   2.4–25 m (guess_units). Up: the file's, else the shortest extent; forward: the file's, else the longest.
3. Scale to metres and turn into the ISO 8855 vehicle frame (x forward, y left, z up); a mesh wound inside out (its
   signed volume negative, as a mirrored export leaves it) is turned the right way out.
4. Wheels: the wheel parts if the file names them, else the lowest 12 % of the body split into four quadrants; a
   quadrant counts when its radius is 0.12–0.8 m (find_wheels).
5. Ground and origin: on the tyres' contact patch with four wheels found (else the lowest point), the origin at the
   rear-axle centre (else 20 % of the length from the rear, on the centre line), as the wizard does.
6. Glass becomes a forbidden mount zone (detect_glass).

The report says what was guessed, so a wrong guess can be given explicitly the next time.
"""
from __future__ import annotations

import os

import numpy as np

from . import loaders
from .mesh import Mesh, Part, Zone, ROLE_GLASS, ROLE_IGNORED, ROLE_MOUNTABLE, ROLE_WHEELS

UNITS = [('m', 'Metres', 1.0), ('mm', 'Millimetres', 1e-3), ('cm', 'Centimetres', 1e-2), ('in', 'Inches', 0.0254)]
AXES = ['+X', '-X', '+Y', '-Y', '+Z', '-Z']
ORIGINS = ('rear_axle', 'front_axle', 'front_bumper', 'centre', 'file')
GROUNDS = ('tyre', 'lowest', 'file')


def axis_vec(a):
    v = np.zeros(3)
    v['XYZ'.index(a[1])] = 1.0 if a[0] == '+' else -1.0
    return v


def detect_roles(parts):
    out = []
    for p in parts:
        n = (p.name + ' ' + p.material).lower()
        if any(k in n for k in ('glass', 'window', 'windshield', 'windscreen', 'windscr')) or p.alpha < 0.9:
            out.append(ROLE_GLASS)
        elif any(k in n for k in ('wheel', 'tyre', 'tire', 'rim', 'hub')):
            out.append(ROLE_WHEELS)
        elif any(k in n for k in ('interior', 'seat', 'dash', 'steering', 'cabin', 'engine')):
            out.append(ROLE_IGNORED)
        else:
            out.append(ROLE_MOUNTABLE)
    return out


def detect_glass(mesh: Mesh):
    gi = [i for i, p in enumerate(mesh.parts) if p.role == ROLE_GLASS or 'glass' in (p.name + p.material).lower() or p.alpha < 0.9]
    for i in gi:
        mesh.parts[i].role = ROLE_GLASS
    zi = next((i for i, z in enumerate(mesh.zones) if z.auto and z.kind == 'forbidden'), None)
    if zi is None:
        mesh.zones.append(Zone('Glass (auto)', 'forbidden', auto=True))
        zi = len(mesh.zones) - 1
    m = np.isin(mesh.face_part, gi)
    mesh.face_zone = np.where(m, zi, mesh.face_zone).astype(np.int16)
    mesh.zones[zi].name = f'Glass ({int(m.sum())} faces, auto)'
    return int(m.sum())


def guess_units(lo, hi, hint):
    ext = float(np.max(hi - lo))
    readings = [(u, name, ext * f) for u, name, f in UNITS]
    fits = [u for u, _, L in readings if 2.4 <= L <= 25.0]
    if hint in ('m', 'mm', 'cm', 'in') and (hint in fits or not fits):
        return hint, readings, True
    if len(fits) == 1:
        return fits[0], readings, True
    return (fits[0] if fits else 'm'), readings, False


def guess_up(lo, hi) -> str:
    return '+' + 'XYZ'[int(np.argmin(hi - lo))]          # cars: length > width >= height


def guess_forward(lo, hi) -> str:
    return '+' + 'XYZ'[int(np.argmax(hi - lo))]


def base_mesh(lm, roles, units, up, forward, flip_lr=False) -> tuple[Mesh, bool]:
    """The loaded mesh in metres and the ISO 8855 axes (before the ground and origin shift), and whether it was
    wound inside out (then turned the right way out)."""
    f = dict((u, s) for u, _, s in UNITS)[units]
    z, x = axis_vec(up), axis_vec(forward)
    y = np.cross(z, x)
    if flip_lr:
        y = -y
    R = np.stack([x, y, z])
    parts = [Part(p.name or f'Part {i + 1}', p.material, p.alpha, tuple(p.color), roles[i]) for i, p in enumerate(lm.parts)]
    m = Mesh(lm.vertices * f, lm.faces, lm.face_part, parts).transformed(R=R)
    t = m.tri
    vol = float(np.einsum('ij,ij->i', t[:, 0], np.cross(t[:, 1], t[:, 2])).sum() / 6.0)
    if vol < 0:
        m = Mesh(m.vertices, m.faces[:, ::-1].copy(), m.face_part, m.parts, m.face_zone, m.zones)
    return m, vol < 0


def find_wheels(m: Mesh, roles) -> tuple[dict, dict]:
    """Wheel centres (outer face) and radii, keyed FL FR RL RR, as the wizard's Find wheels: the wheel parts, else
    the lowest 12 % of the body by quadrant; a quadrant with a radius outside 0.12–0.8 m is left out."""
    wi = [i for i, r in enumerate(roles) if r == ROLE_WHEELS]
    cen = None
    if wi:
        pts = m.tri[np.isin(m.face_part, wi)].reshape(-1, 3)
        if len(pts) > 20:
            cen = pts
    if cen is None:
        v = m.vertices[np.unique(m.faces)]
        lo, hi = v.min(axis=0), v.max(axis=0)
        cen = v[v[:, 2] < lo[2] + 0.12 * (hi[2] - lo[2]) + 0.25]
    lo, hi = m.bbox
    cx, cy = float(np.median(cen[:, 0])), (lo[1] + hi[1]) / 2
    found, radius = {}, {}
    for k, sx, sy in (('RL', -1, 1), ('RR', -1, -1), ('FL', 1, 1), ('FR', 1, -1)):
        q = cen[(np.sign(cen[:, 0] - cx) == sx) & (np.sign(cen[:, 1] - cy) == sy)]
        if len(q) < 6:
            continue
        qlo, qhi = q.min(axis=0), q.max(axis=0)
        r = (qhi[2] - qlo[2]) / 2
        if not 0.12 < r < 0.8:
            continue
        found[k] = np.array([(qlo[0] + qhi[0]) / 2, qhi[1] if sy > 0 else qlo[1], (qlo[2] + qhi[2]) / 2])
        radius[k] = float(r)
    return found, radius


def origin_offset(m: Mesh, wheels: dict, radius: dict, origin='rear_axle', ground='tyre') -> np.ndarray:
    """The translation that puts the chosen origin at 0 (in base-mesh coordinates), as the wizard's."""
    lo, hi = m.bbox
    z0 = 0.0
    if ground == 'tyre' and len(wheels) >= 3:
        z0 = float(np.mean([c[2] - radius.get(k, 0.3) for k, c in wheels.items()]))
    elif ground in ('lowest', 'tyre'):
        z0 = float(lo[2])
    x0 = y0 = 0.0
    if origin == 'rear_axle' and 'RL' in wheels and 'RR' in wheels:
        x0, y0 = (wheels['RL'][0] + wheels['RR'][0]) / 2, (wheels['RL'][1] + wheels['RR'][1]) / 2
    elif origin == 'front_axle' and 'FL' in wheels and 'FR' in wheels:
        x0, y0 = (wheels['FL'][0] + wheels['FR'][0]) / 2, (wheels['FL'][1] + wheels['FR'][1]) / 2
    elif origin == 'front_bumper':
        x0, y0 = hi[0], (lo[1] + hi[1]) / 2
    elif origin == 'centre':
        x0, y0 = (lo[0] + hi[0]) / 2, (lo[1] + hi[1]) / 2
    elif origin in ('rear_axle', 'front_axle'):
        x0, y0 = lo[0] + (0.2 if origin == 'rear_axle' else 0.8) * (hi[0] - lo[0]), (lo[1] + hi[1]) / 2
    if origin == 'file':
        x0 = y0 = 0.0
    if ground == 'file':
        z0 = 0.0
    return np.array([x0, y0, z0])


def import_vehicle(path, units=None, up=None, forward=None, flip_lr=False, origin='rear_axle', ground='tyre',
                   name=None, occluder_target=8000, progress=None):
    """(Vehicle, report) from a model file. units / up / forward: given, or taken from the file, or guessed (the
    report says which). Raises loaders.MeshLoadError (a user-facing message) or ValueError for a bad argument."""
    from .project import Vehicle
    if units is not None and units not in dict((u, s) for u, _, s in UNITS):
        raise ValueError(f'units: one of {", ".join(u for u, *_ in UNITS)}')
    for nm, a in (('up', up), ('forward', forward)):
        if a is not None and a not in AXES:
            raise ValueError(f'{nm}: one of {", ".join(AXES)}')
    if origin not in ORIGINS:
        raise ValueError(f'origin: one of {", ".join(ORIGINS)}')
    if ground not in GROUNDS:
        raise ValueError(f'ground: one of {", ".join(GROUNDS)}')
    path = os.path.abspath(os.path.expanduser(os.fspath(path)))
    lm = loaders.load_mesh(path, progress=progress)
    roles = detect_roles(lm.parts)
    lo, hi = lm.vertices.min(axis=0), lm.vertices.max(axis=0)
    report = {'file': os.path.basename(path), 'format': lm.fmt, 'triangles': int(len(lm.faces)),
              'parts': len(lm.parts), 'notes': list(lm.notes[:4])}
    if units is None:
        units, readings, sure = guess_units(lo, hi, lm.unit_hint)
        report['units'] = f'{units} ({"from the file" if lm.unit_hint and sure else "guessed" if sure else "a guess: no unit makes it car-sized"})'
    else:
        report['units'] = f'{units} (given)'
    if up is None:
        up = lm.up_hint if lm.up_hint in AXES else guess_up(lo, hi)
        report['up'] = f'{up} ({"from the file" if lm.up_hint in AXES else "guessed: the shortest side"})'
    else:
        report['up'] = f'{up} (given)'
    if forward is None:
        forward = lm.forward_hint if lm.forward_hint in AXES else guess_forward(lo, hi)
        if forward[1] == up[1]:
            forward = '+X' if up[1] != 'X' else '+Y'
        report['forward'] = f'{forward} ({"from the file" if lm.forward_hint in AXES else "guessed: the longest side"})'
    else:
        report['forward'] = f'{forward} (given)'
    if forward[1] == up[1]:
        raise ValueError(f'forward {forward} and up {up} are the same axis')
    m, inside_out = base_mesh(lm, roles, units, up, forward, flip_lr)
    if inside_out:
        report['inside_out'] = 'the faces were wound inside out (a mirrored export): turned the right way out'
    wheels, radius = find_wheels(m, roles)
    report['wheels_found'] = sorted(wheels)
    if ground == 'tyre' and len(wheels) < 3:
        report['ground'] = 'the lowest point (fewer than three wheels found for the tyre contact)'
    axle = {'rear_axle': {'RL', 'RR'}, 'front_axle': {'FL', 'FR'}}.get(origin)
    if axle is not None and not axle <= set(wheels):
        report['origin'] = f'{origin} estimated from the length (its wheels were not found): check it'
    off = origin_offset(m, wheels, radius, origin, ground)
    m = m.transformed(t=-off)
    wc = {k: [round(float(x), 4) for x in (c - off)] for k, c in wheels.items()}
    glass = detect_glass(m)
    if glass:
        report['glass_faces'] = glass
    lo2, hi2 = m.bbox
    report['size_m'] = {'length': round(float(hi2[0] - lo2[0]), 3), 'width': round(float(hi2[1] - lo2[1]), 3),
                        'height': round(float(hi2[2] - lo2[2]), 3)}
    if wc.get('FL') and wc.get('RL'):
        report['wheelbase_m'] = round(wc['FL'][0] - wc['RL'][0], 3)
    sz = report['size_m']
    if not (2.4 <= sz['length'] <= 25.0 and 1.0 <= sz['width'] <= 3.5 and 0.8 <= sz['height'] <= 5.0):
        report['check'] = (f'{sz["length"]} × {sz["width"]} × {sz["height"]} m is not car-sized: give units, up or '
                           'forward explicitly')
    veh = Vehicle(name=name or os.path.splitext(os.path.basename(path))[0], source=path, import_units=units,
                  import_up=up, import_forward=forward, occluder_target=int(occluder_target), wheel_centres=wc,
                  mesh=m, imported_tris=m.n_tris,
                  landmarks={f'Wheel centre {k}': tuple(c) for k, c in wc.items()})
    return veh, report
