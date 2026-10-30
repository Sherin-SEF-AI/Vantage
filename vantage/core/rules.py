"""Cable routing, mount checks and the Issues list."""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from .project import Project, Camera, MAX_CAMERAS

SEVERITY_ORDER = {'critical': 0, 'serious': 1, 'warning': 2, 'good': 3}
EDGE_RULE_MM = 20.0
FLOAT_TOL_MM = 15.0


def route_length(project: Project, cam: Camera, routed: bool = True) -> float:
    """The camera's cable run, metres: routed along the body (core.harness) when Analysis ▸ Cable harness ▸ Use the
    routed lengths is on, else the Manhattan estimate × the routing slack + the service loop + the camera's extra.
    routed=False always gives the estimate (the optimiser's inner loop)."""
    if routed:
        from . import harness
        r = harness.routed_length(project, cam)
        if r is not None:
            return r
    e = project.ecu
    manhattan = abs(cam.x - e.x) + abs(cam.y - e.y) + abs(cam.z - e.z)
    return float(manhattan * e.route_factor + e.service_loop_m + cam.route_extra_m)


def route_text(project: Project, cam: Camera) -> str:
    if cam.route_note:
        return cam.route_note
    from . import harness
    routed = harness.route_text(project, cam)
    if routed is not None:
        return routed
    hops = [cam.tag]
    if abs(cam.y) > 0.5:
        hops.append('door sill' if cam.z < 0.8 else ('A-pillar' if cam.x > project.ecu.x else 'C-pillar'))
    if cam.x > project.ecu.x + 0.8:
        hops.append('engine bay grommet' if cam.z < 0.8 else 'dash')
    elif cam.x < project.ecu.x - 0.8:
        hops.append('rear parcel shelf' if cam.z > 0.7 else 'boot floor')
    hops.append(f'ECU ({project.ecu.location.lower()})' if project.ecu.location else 'ECU')
    return ' → '.join(hops)


def camera_gbps(project: Project, cam: Camera) -> float:
    m = project.module_for(cam)
    e = project.ecu
    return m.width_px * m.height_px * m.fps * e.bpp * e.overhead / 1e9


def link_budget(project: Project):
    """Per-deserializer load: list of (name, ports, cameras, used Gbps, capacity Gbps, worst link Gbps, link capacity)."""
    e = project.ecu
    rows = []
    for name, ports in (('A', range(1, 5)), ('B', range(5, 9))):
        cams = [c for c in project.variant.cameras if c.port in ports]
        used = sum(camera_gbps(project, c) for c in cams)
        worst = max((camera_gbps(project, c) for c in cams), default=0.0)
        rows.append((name, f'{min(ports)}–{max(ports)}', [c.tag for c in cams], used, e.csi_gbps_per_deser, worst, e.link_gbps))
    return rows


@dataclass
class MountInfo:
    surface: str
    zone_kind: str            # 'allowed' | 'forbidden' | ''
    gap_mm: float
    edge_mm: float
    nearest: tuple
    normal: tuple
    floating: bool
    face: int


def mount_info(project: Project, cam: Camera, vg) -> MountInfo | None:
    if vg is None:
        return None
    m = vg.mesh if vg.mesh.n_tris <= 60000 else vg.occ_mesh
    q, fi, d = m.closest_point(cam.pos(), m.occluder_faces())
    if fi < 0:
        return None
    zi = int(m.face_zone[fi]) if len(m.face_zone) else -1
    if 0 <= zi < len(m.zones):
        surface, kind = m.zones[zi].name, m.zones[zi].kind
    else:
        part = m.parts[int(m.face_part[fi])] if m.parts else None
        surface, kind = part.name if part else 'Body', ''
        if part is not None and part.material == 'rig' and 'not mountable' in part.role:
            kind = 'forbidden'                                  # a rig part that must not carry cameras
    segs = m.crease_segments(35.0, m.occluder_faces())
    edge = math.inf
    if len(segs):
        a, b = segs[:, 0], segs[:, 1]
        ab = b - a
        t = np.clip(np.einsum('ij,ij->i', q - a, ab) / np.maximum(np.einsum('ij,ij->i', ab, ab), 1e-12), 0, 1)
        edge = float(np.min(np.linalg.norm(a + ab * t[:, None] - q, axis=1)))
    gap = float(d * 1000)
    floating = bool(gap > cam.standoff_mm + FLOAT_TOL_MM)
    return MountInfo(surface, kind, gap, float(edge * 1000), tuple(float(v) for v in q),
                     tuple(float(v) for v in m.normals[fi]), floating, int(fi))


@dataclass
class Issue:
    severity: str
    message: str
    objects: str
    uids: list = field(default_factory=list)
    action: str = ''          # 'cable15', 'snap', ...
    where: tuple | None = None


def collect_issues(project: Project, result, vg, mounts: dict | None = None, engine=None) -> list[Issue]:
    out: list[Issue] = []
    v = project.variant
    cams = v.cameras
    if len(cams) > MAX_CAMERAS:
        out.append(Issue('critical', f'{len(cams)} cameras exceed the {MAX_CAMERAS} GMSL2 ports on the Jetson AGX Orin.', v.name))
    if result is not None:
        b = result.metrics['blind_m2']
        if b > 0.0005:
            g = result.grid
            blind = g.roi & (result.count == 0)
            jj, ii = np.nonzero(blind)
            where = (float(g.xs[ii].mean()), float(g.ys[jj].mean())) if len(ii) else None
            from .analysis import describe_location
            loc = describe_location(*where, project) if where else ''
            out.append(Issue('critical', f'{b:.3f} m² of ground in the ROI is blind' + (f', mostly {loc}.' if loc else '.'),
                             v.name, where=where))
        mind = result.metrics['min_density']
        if result.metrics['covered_pct'] > 0 and mind < project.analysis.min_density:
            share = 100 - result.metrics['min_ok_pct']
            out.append(Issue('warning', f'Minimum density {mind:.1f} px/cm is below {project.analysis.min_density:.1f} px/cm on '
                                        f'{share:.1f} % of the ROI.', v.name))
    # ports
    ports = {}
    for c in cams:
        if c.port:
            ports.setdefault(c.port, []).append(c)
    for port, cs in ports.items():
        if len(cs) > 1:
            out.append(Issue('serious', f'GMSL port {port} is assigned to {len(cs)} cameras.', ', '.join(c.tag for c in cs),
                             [c.uid for c in cs]))
    for c in cams:
        if not c.port:
            out.append(Issue('warning', f'{c.tag} has no GMSL port assigned.', c.tag, [c.uid]))
        rl = route_length(project, c)
        if rl > c.cable_m + 1e-6:
            alt = [x for x in project.module_for(c).cables_m if x >= rl]
            hint = f'Use the {alt[0]:g} m cable or move the deserializer.' if alt else 'Move the deserializer closer.'
            out.append(Issue('serious', f'Routed cable {rl:.2f} m is longer than the {c.cable_m:.2f} m module cable. {hint}',
                             f'{c.tag} · port {c.port}' if c.port else c.tag, [c.uid], 'cable'))
    # link budget
    for name, ports, tags, used, cap, worst, link in link_budget(project):
        if used > cap:
            out.append(Issue('serious', f'Deserializer {name} carries {used:.2f} Gbps, over its {cap:.1f} Gbps CSI-2 output. '
                                        'Move a camera to the other deserializer or lower the frame rate.', ', '.join(tags)))
        if worst > link:
            out.append(Issue('serious', f'A camera on deserializer {name} needs {worst:.2f} Gbps, over the {link:.1f} Gbps GMSL2 link.',
                             ', '.join(tags)))
    # mounts
    if mounts:
        edge_cams = []
        for c in cams:
            mi = mounts.get(c.uid)
            if mi is None:
                continue
            if mi.floating:
                out.append(Issue('critical', f'{c.tag} is {mi.gap_mm:.0f} mm off the body. Snap it to the surface or keep it as a bracket.',
                                 c.tag, [c.uid], 'snap'))
            if mi.zone_kind == 'forbidden':
                out.append(Issue('serious', f'{c.tag} sits on a forbidden zone ({mi.surface}).', c.tag, [c.uid]))
            if mi.edge_mm < EDGE_RULE_MM and not mi.floating:
                edge_cams.append((c, mi))
        if edge_cams:
            worst = min(mi.edge_mm for _, mi in edge_cams)
            out.append(Issue('warning', f'Camera sits {worst:.0f} mm from the body edge; the clearance rule is {EDGE_RULE_MM:.0f} mm.',
                             ', '.join(c.tag for c, _ in edge_cams), [c.uid for c, _ in edge_cams]))
    out += sensor_issues(project, vg)
    from . import (asbuilt, automation, clock_tree, critical, exposure, fleet, gnss,   # they import
                   harness, housings, materials, observability, power, recording,       # this module
                   requirements, suspension, terrain, thermo_vib, timing)
    out += housings.issues(project, vg)
    out += recording.issues(project)
    out += power.issues(project)
    out += gnss.issues(project, vg)
    out += exposure.issues(project, vg)
    out += fleet.issues(project)
    out += requirements.issues(project)
    out += suspension.issues(project, vg)
    out += asbuilt.issues(project)
    out += materials.issues(project, vg)          # radars through the fascia, cameras behind glass, the wipers
    out += thermo_vib.issues(project, vg)         # a mount resonating on the road, drift in full sun past the tolerance
    out += critical.issues(project, vg)           # critical targets no camera resolves, mounts tighter than the install
    out += terrain.issues(project, vg)            # on the road: a wheel lifting, the body or a sensor touching it
    out += harness.issues(project, vg)            # cables over their link's limit, GNSS RF margin, entry points
    out += timing.issues(project, vg)             # stereo pairs and trigger lines split over two deserializers
    out += clock_tree.issues(project, vg)         # grandmaster holdover, free-running FSYNC, timestamp error in pixels
    if engine is not None:                        # (its occlusion cube maps): the body filling a camera's AE metering
        from . import isp_photometrics
        out += [Issue(i['severity'], i['message'], ', '.join(c.tag for c in project.variant.cameras
                                                             if c.uid in i['uids']), list(i['uids']))
                for i in isp_photometrics.issues(project, engine)]
    out += observability.issues(project, vg, result)   # cameras the planned boards cannot pin, radars short of reflectors
    out += automation.issues(project, vg)         # dropped automation settings, the last automated run failing
    out.sort(key=lambda i: SEVERITY_ORDER.get(i.severity, 9))
    return out


def _inside(mesh, p) -> bool:
    """Whether p is inside the closed surface: rays in the six axis directions leave it through a face (the face
    normal points along the ray) for at least five of them."""
    d = np.array([[0, 0, 1], [0, 0, -1], [1, 0, 0], [-1, 0, 0], [0, 1, 0], [0, -1, 0]], float)
    _, f = mesh.raycast(np.repeat(np.asarray(p, float)[None], 6, axis=0), d)
    hit = f >= 0
    if hit.sum() < 5:
        return False
    return int((np.einsum('ij,ij->i', mesh.normals[f[hit]], d[hit]) > 0).sum()) >= 5


RADAR_FASCIA_M = 0.15             # a radar may sit this far behind the bumper fascia (it sees through the plastic)


def _body_ahead(mesh, p, d, max_hits=32) -> float:
    """How much vehicle lies ahead along the ray p + t d: the distance to the last surface it crosses (beyond it,
    free space). A radar looks through what lies along its boresight, so this, not the nearest surface (a wheel arch
    beside it), is how deep it sits."""
    p, d = np.asarray(p, float), np.asarray(d, float) / max(float(np.linalg.norm(d)), 1e-12)
    t_total, last = 0.0, 0.0
    for _ in range(max_hits):
        t, f = mesh.raycast((p + d * t_total)[None], d[None])
        if f[0] < 0 or not np.isfinite(t[0]):
            break
        t_total += float(t[0])
        last = t_total
        t_total += 1e-4                 # step past the surface just crossed
    return last


def _noun(kind):
    from .sensors import KIND_NOUN
    return KIND_NOUN.get(kind, kind)


def sensor_issues(project: Project, vg) -> list[Issue]:
    """Lidars, radars, GNSS antennas and IMUs: a tag a camera uses too, a sensor below the ground, a lidar or a GNSS
    antenna inside the body or a rig part, a radar with more of the vehicle ahead along its boresight than a bumper
    fascia (an IMU belongs inside)."""
    out = []
    v = project.variant
    cam_tags = {c.tag.lower(): c for c in v.cameras}
    for s in v.sensors:
        c = cam_tags.get(s.tag.lower())
        if c is not None:
            out.append(Issue('warning', f'Tag {s.tag} is used by camera {c.name} and by a {_noun(s.kind)}: exported '
                                        f'frames get a suffix. Rename one.', s.tag, [s.uid, c.uid]))
    occ = vg.occ_mesh if vg is not None else None
    for s in v.sensors:
        if not s.visible:
            continue
        m = project.sensor_model_for(s)
        base = s.z - m.optical_z_mm / 1000.0
        if base < -0.005:
            out.append(Issue('critical', f'{s.tag} ({m.name}) is {-base * 1000:.0f} mm below the ground.', s.tag, [s.uid]))
            continue
        if s.kind == 'imu' or occ is None or not _inside(occ, s.pos()):
            continue                            # an IMU belongs inside the cabin or an enclosure
        if s.kind == 'gnss':
            out.append(Issue('critical', f'GNSS antenna {s.tag} sits inside the body or a rig part: it has no sky. '
                                         f'Put it on the roof or the rack (Inspector ▸ Optical centre Z).',
                             s.tag, [s.uid], where=(s.x, s.y)))
        elif s.kind == 'lidar':
            out.append(Issue('critical', f'Lidar {s.tag} sits inside the body or a rig part: its beams start inside '
                                         f'the vehicle. Raise it onto the surface (Inspector ▸ Optical centre Z).',
                             s.tag, [s.uid], where=(s.x, s.y)))
        else:
            from .sensors import rotation
            d = _body_ahead(occ, s.pos(), rotation(s)[:, 0])
            if d > RADAR_FASCIA_M:
                out.append(Issue('warning', f'Radar {s.tag} sits {d * 1000:.0f} mm behind the outer surface along its '
                                            f'boresight: a radar sees through a plastic bumper fascia, not through the '
                                            f'body.', s.tag, [s.uid], where=(s.x, s.y)))
    return out


def blocks_export(issues: list[Issue]) -> bool:
    return any(i.severity == 'critical' and i.action == 'snap' for i in issues)
