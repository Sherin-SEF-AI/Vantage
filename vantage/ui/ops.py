"""Editing operations shared by menus, the viewport, the outliner and the command palette."""
from __future__ import annotations

import copy
import json
import math

import numpy as np

from ..core.mathutil import yaw_pitch_from_dir, wrap180
from ..core.project import Camera, Variant, MAX_CAMERAS, new_uid
from ..core.analysis import surface_frame

TAG_SUGGEST = ['F', 'R', 'ML', 'MR', 'QL', 'QR', 'FL', 'FR', 'RL', 'RR', 'SL', 'SR']


def _free_tag(variant, base):
    used = {c.tag for c in variant.cameras}
    if base not in used:
        return base
    for i in range(2, 20):
        t = f'{base}{i}'
        if t not in used:
            return t
    return new_uid()[:3].upper()


def _free_port(variant, ports=8):
    used = {c.port for c in variant.cameras}
    for p in range(1, ports + 1):
        if p not in used:
            return p
    return 0


def location_tag(project, pos, normal):
    """Name a mount spot from where it sits on the body."""
    meas = project.vehicle.measured()
    xf, xr = meas.get('front_x', 3.0), meas.get('rear_x', -0.8)
    x, y, z = pos
    n = np.asarray(normal)
    side = 'L' if y > 0 else 'R'
    sidew = 'left' if y > 0 else 'right'
    if n[0] > 0.6 and abs(y) < 0.35:
        return 'F', 'Front', 'Front, grille above plate' if z < 0.8 else 'Front, windscreen top'
    if n[0] < -0.6 and abs(y) < 0.35:
        return 'R', 'Rear', 'Rear, bumper top edge' if z < 0.75 else 'Rear, boot lid'
    if n[2] < -0.6:
        return 'M' + side, f'Mirror {sidew}', f'Under {sidew} mirror'
    if n[2] > 0.6:                          # an upward surface: the roof, the bonnet or the boot lid
        from ..core.rig import roof
        rf, rr, _, top = roof(project.vehicle.mesh)
        if z < top - 0.25:
            return ('FT', 'Front top', 'Bonnet') if x > (xf + xr) / 2 else ('RT', 'Rear top', 'Boot lid, top')
        where = 'front' if x >= (rf + rr) / 2 else 'rear'
        if abs(y) < 0.2:
            return 'T', 'Roof', f'Roof, {where} centre'
        return 'T' + side, f'Roof {sidew}', f'Roof, {where} {sidew} edge'
    if x > xf - 0.45:
        return 'F' + side, f'Front {sidew}', f'Front bumper, {sidew} corner'
    if x < xr + 0.45:
        return 'R' + side, f'Rear {sidew}', f'Rear bumper, {sidew} corner'
    if x < 0.6:
        return 'Q' + side, f'Quarter {sidew}', f'{sidew.capitalize()} rear quarter'
    return 'S' + side, f'Side {sidew}', f'{sidew.capitalize()} side'


def default_aim(project, pos, normal):
    n = np.asarray(normal, float)
    h = np.array([n[0], n[1], 0.0])
    if n[2] > 0.6:
        # an upward surface (roof, bonnet, boot lid): look out nearly level, to that side off the centre line, else
        # ahead from the front half of the vehicle and back from the rear half
        if abs(pos[1]) >= 0.2:
            return (90.0 if pos[1] > 0 else -90.0), 15.0
        meas = project.vehicle.measured()
        mid = (meas.get('front_x', 3.0) + meas.get('rear_x', -0.8)) / 2
        return (0.0 if pos[0] >= mid else 180.0), 15.0
    if np.linalg.norm(h) < 0.3:
        # downward-facing surface (mirror underside): look outward from the body centre line
        h = np.array([0.35, np.sign(pos[1]) or 1.0, 0.0])
    yaw, _ = yaw_pitch_from_dir(h)
    return round(yaw / 5) * 5, 45.0


def add_camera(ctx, pos, normal, module_id=None, lens_id=None, pair=False, standoff_mm=12.0):
    store = ctx.store
    p = store.project
    v = p.variant
    need = 2 if pair else 1
    if len(v.cameras) + need > MAX_CAMERAS:
        ctx.msg(f'Up to {MAX_CAMERAS} cameras: the Jetson AGX Orin has 8 GMSL2 ports.')
        return []
    module_id = module_id or ctx.place_lens[0]
    lens_id = lens_id or ctx.place_lens[1]
    pos = np.asarray(pos, float) + np.asarray(normal, float) * standoff_mm / 1000.0
    tag, name, loc = location_tag(p, pos, normal)
    yaw, pitch = default_aim(p, pos, normal)

    def fn(project):
        var = project.variant
        c = Camera(tag=_free_tag(var, tag), name=name, location=loc, slot=var.free_slot(),
                   x=round(float(pos[0]), 3), y=round(float(pos[1]), 3), z=round(float(pos[2]), 3),
                   yaw=yaw, pitch=pitch, module_id=module_id, lens_id=lens_id, port=_free_port(var),
                   standoff_mm=standoff_mm)
        var.cameras.append(c)
        made = [c]
        if pair:
            m = mirrored_copy(var, c)
            var.cameras.append(m)
            made.append(m)
        return [x.uid for x in made]

    uids = store.edit('variant', fn, f'Placed {tag}' + (' as a mirrored pair' if pair else ''))
    store.select_cams(uids)
    return uids


def mirrored_copy(var: Variant, c: Camera) -> Camera:
    m = copy.deepcopy(c)
    m.uid = new_uid()
    m.y, m.yaw, m.roll = -c.y, -c.yaw, -c.roll
    if c.tag.endswith('L'):
        m.tag = c.tag[:-1] + 'R'
    elif c.tag.endswith('R') and len(c.tag) > 1:
        m.tag = c.tag[:-1] + 'L'
    else:
        m.tag = c.tag + "'"
    m.tag = _free_tag(var, m.tag) if any(x.tag == m.tag for x in var.cameras) else m.tag
    m.name = c.name.replace('left', '§').replace('right', 'left').replace('§', 'right')
    m.location = c.location.replace('left', '§').replace('right', 'left').replace('§', 'right')
    m.slot = var.free_slot() if m.slot in {x.slot for x in var.cameras} else m.slot
    used = {x.slot for x in var.cameras} | {c.slot}
    for s in range(8):
        if s not in used:
            m.slot = s
            break
    m.port = _free_port(var)
    if m.port == c.port:
        m.port = 0
    m.link_uid = c.uid
    c.link_uid = m.uid
    return m


def delete_cameras(ctx, uids):
    if not uids:
        return
    tags = [ctx.store.cam(u).tag for u in uids if ctx.store.cam(u)]

    def fn(project):
        var = project.variant
        var.cameras = [c for c in var.cameras if c.uid not in uids]
        for c in var.cameras:
            if c.link_uid in uids:
                c.link_uid = ''
        var.seams = {k: v for k, v in var.seams.items() if not any(u in k for u in uids)}
    ctx.store.edit('variant', fn, f'Deleted {", ".join(tags)}')


def duplicate_cameras(ctx, uids):
    v = ctx.store.variant
    if len(v.cameras) + len(uids) > MAX_CAMERAS:
        ctx.msg(f'Up to {MAX_CAMERAS} cameras.')
        return

    def fn(project):
        var = project.variant
        out = []
        for u in uids:
            c = var.cam(u)
            if not c:
                continue
            d = copy.deepcopy(c)
            d.uid = new_uid()
            d.tag = _free_tag(var, c.tag)
            d.slot = var.free_slot()
            d.port = _free_port(var)
            d.link_uid = ''
            d.x += 0.05
            var.cameras.append(d)
            out.append(d.uid)
        return out
    new = ctx.store.edit('variant', fn, 'Duplicated ' + ', '.join(ctx.store.cam(u).tag for u in uids))
    ctx.store.select_cams(new)


def mirror_to_other_side(ctx, uids):
    def fn(project):
        var = project.variant
        out = []
        for u in uids:
            c = var.cam(u)
            if not c:
                continue
            if c.link_uid and var.cam(c.link_uid):
                partner = var.cam(c.link_uid)
                partner.x, partner.y, partner.z = c.x, -c.y, c.z
                partner.yaw, partner.pitch, partner.roll = -c.yaw, c.pitch, -c.roll
                partner.lens_id, partner.module_id = c.lens_id, c.module_id
                out.append(partner.uid)
            elif len(var.cameras) < MAX_CAMERAS:
                m = mirrored_copy(var, c)
                var.cameras.append(m)
                out.append(m.uid)
        return out
    made = ctx.store.edit('variant', fn, 'Mirrored ' + ', '.join(ctx.store.cam(u).tag for u in uids if ctx.store.cam(u)))
    return made


def toggle_link(ctx, uid):
    c = ctx.store.cam(uid)
    if not c:
        return
    if c.link_uid:
        def fn(project):
            a = project.variant.cam(uid)
            b = project.variant.cam(a.link_uid)
            a.link_uid = ''
            if b:
                b.link_uid = ''
        ctx.store.edit('variant', fn, f'Unlinked {c.tag}')
        return
    # find the mirror candidate
    best, bd = None, 1e9
    for o in ctx.store.variant.cameras:
        if o.uid == uid or o.link_uid:
            continue
        d = abs(o.x - c.x) + abs(o.y + c.y) + abs(o.z - c.z)
        if d < bd:
            best, bd = o, d
    if best is None or bd > 0.5:
        ctx.msg('No camera on the other side to link with. Use Mirror to other side.')
        return

    def fn(project):
        a, b = project.variant.cam(uid), project.variant.cam(best.uid)
        a.link_uid, b.link_uid = b.uid, a.uid
        b.x, b.y, b.z, b.yaw, b.pitch, b.roll = a.x, -a.y, a.z, -a.yaw, a.pitch, -a.roll
    ctx.store.edit('variant', fn, f'Linked {c.tag} and {best.tag}')


def set_lock(ctx, uids, locked=None):
    cams = [ctx.store.cam(u) for u in uids if ctx.store.cam(u)]
    if not cams:
        return
    val = (not cams[0].locked) if locked is None else locked

    def fn(project):
        for u in uids:
            c = project.variant.cam(u)
            if c:
                c.locked = val
    ctx.store.edit('variant', fn, ('Locked ' if val else 'Unlocked ') + ', '.join(c.tag for c in cams))


def set_visible(ctx, uids, vis):
    def fn(project):
        for u in uids:
            c = project.variant.cam(u)
            if c:
                c.visible = vis
    ctx.store.edit('variant', fn, ('Showed ' if vis else 'Hid ') + ', '.join(ctx.store.cam(u).tag for u in uids if ctx.store.cam(u)))


def snap_to_surface(ctx, uids, keep_as_bracket=False):
    vg = ctx.analysis.engine.vehicle_geom(ctx.store.project)
    if vg is None:
        return

    def fn(project):
        from .store import mirror_props
        for u in uids:
            c = project.variant.cam(u)
            if not c:
                continue
            q, n, _, _, fi = surface_frame(vg.occ_mesh, c.pos())
            if keep_as_bracket:
                c.standoff_mm = round(float(np.linalg.norm(c.pos() - q)) * 1000)
            else:
                p = q + n * c.standoff_mm / 1000.0
                c.x, c.y, c.z = (round(float(v), 4) for v in p)
            partner = project.variant.cam(c.link_uid) if c.link_uid else None
            if partner is not None and partner.uid not in uids:
                # a linked pair stays a mirror image (snapping the partner on its own would not be exact)
                mirror_props(c, partner, ['standoff_mm'] if keep_as_bracket else ['x', 'y', 'z'])
    ctx.store.edit('variant', fn, ('Kept as bracket ' if keep_as_bracket else 'Snapped to surface ') +
                   ', '.join(ctx.store.cam(u).tag for u in uids if ctx.store.cam(u)))


def aim_at(ctx, uid, point):
    c = ctx.store.cam(uid)
    if not c:
        return
    d = np.asarray(point, float) - c.pos()
    yaw, pitch = yaw_pitch_from_dir(d)
    ctx.store.set_cam(uid, yaw=round(yaw, 1), pitch=round(pitch, 1), text=f'Aimed {c.tag} at {point[0]:.2f}, {point[1]:.2f}')


def aim_along_normal(ctx, uid):
    vg = ctx.analysis.engine.vehicle_geom(ctx.store.project)
    c = ctx.store.cam(uid)
    if not c or vg is None:
        return
    q, n, _, _, _ = surface_frame(vg.occ_mesh, c.pos())
    yaw, pitch = yaw_pitch_from_dir(n)
    ctx.store.set_cam(uid, yaw=round(yaw, 1), pitch=round(max(pitch, 0.0), 1), text=f'Aimed {c.tag} along the surface normal')


def level_roll(ctx, uids):
    for u in uids:
        ctx.store.set_cam(u, roll=0.0, text='Levelled roll')


def set_lens(ctx, uids, module_id, lens_id):
    lens = ctx.store.project.lenses.get(lens_id)

    def fn(project):
        for u in uids:
            c = project.variant.cam(u)
            if c:
                c.module_id, c.lens_id = module_id, lens_id
                if c.link_uid and project.variant.cam(c.link_uid):
                    p = project.variant.cam(c.link_uid)
                    p.module_id, p.lens_id = module_id, lens_id
    ctx.store.edit('variant', fn, f'Changed lens of {", ".join(ctx.store.cam(u).tag for u in uids)} to {lens.name if lens else lens_id}')


def pose_json(cam: Camera, project) -> str:
    o = project.origin_offset()
    return json.dumps({'tag': cam.tag, 'frame': project.frame_label, 'position': [round(cam.x - o[0], 4), round(cam.y - o[1], 4), round(cam.z - o[2], 4)],
                       'yaw': cam.yaw, 'pitch': cam.pitch, 'roll': cam.roll, 'module': cam.module_id, 'lens': cam.lens_id}, indent=1)


def assign_port(ctx, uid, port):
    c = ctx.store.cam(uid)

    def fn(project):
        var = project.variant
        if port:                        # unassigning (0) frees the port; nothing to swap
            for o in var.cameras:
                if o.port == port and o.uid != uid:
                    o.port = c.port if c else 0
        var.cam(uid).port = port
    ctx.store.edit('variant', fn, f'Assigned {c.tag} to GMSL port {port}')
