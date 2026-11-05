"""Robust perception: the rig's tolerances (mounting, thermal warping, vibration, calibration) carried through to what
its perception sees, with error bars: "pedestrian recall at front · 40–60 m 0.91, 90 % of builds 0.87–0.94; BEV error
0.8 m (p95 1.9 m)". Measured on recordings with ground truth (CARLA drives, core.groundtruth), per class and per region
of core.regions' grid.

The error model (each source from the app's own studies, or set by hand)
------------------------------------------------------------------------
Every Monte Carlo sample is one build of the car: for every camera a TRUE pose (where it really is and looks) and the
pose the perception USES (its calibration). The recording was made at the design pose; the true pose differs from it by
  * mounting: the build against the design, each angle and each position axis independently uniform within ±tol
    (core.tolerance's 'uniform'; 'normal' draws N(0, tol/2) clipped at ±tol, its '± = 2 σ'). The tolerance is the
    installation check's (core.install: 1° / 10 mm, what an as-built car passes), the project's analysis tolerance
    (analysis.tol_deg with core.tolerance's 10 mm position default), or set ones;
  * thermal: core.thermo_vib's drift of each camera from its garage calibration, hour by hour on the study's hot day;
    a sample is one hour of the driving window (uniform), the same hour for every camera;
  * vibration: core.thermo_vib's road vibration on its ISO 8608 class at its speed: the body's pitch on the suspension
    (one angle for the whole car, N(0, σ_body) with σ_body² = the device angle RMS² − its mount-flex RMS², averaged over
    the cameras; the car pitches about the half-car's centre of gravity, so a camera also moves by the pitch's lever
    arm), the body's heave (σ from the vertical motion at the mounts not explained by the pitch), and each mount's own
    flex (RMS angle and move, spread equally over three axes, independent per camera). The perception lifts objects with
    the static calibration: it does not compensate the body's pitch (no dynamic pitch estimate);
plus, in the per-vehicle mode, the calibration's residual: each car is calibrated after it is built (core.observability's
predicted 1 σ per camera and axis for the planned boards, from the corners' Fisher information; or set ones), so the
used pose is the built pose plus N(0, σ_calibration). In the design mode every car runs on the design calibration (the
used pose is the design pose, the mounting error reaches the perception in full) and the calibration source is off.

How a pose error is carried to the images and the geometry
---------------------------------------------------------
* Image content: a rotation of the camera about its optical centre is an exact remap of the recorded image through the
  recording's own lens model (the Kannala-Brandt calibration the dataset export wrote, detection_eval.kb_project /
  kb_unproject): each pixel of the true camera is unprojected, turned into the recorded camera's frame and projected
  with the same intrinsics; pixels whose rays fall outside the recorded image or the lens's field are MISSING (black,
  never invented; their share is reported). Translations of mm–cm are applied to the geometry only: a translation δ
  moves the image of a point at distance Z by at most (max px/rad of the lens) · δ / Z pixels, below one pixel beyond
  Z_subpx = (max px/rad) · δ (stated per camera; a few metres for a 10 mm δ on these lenses).
* Ground-truth boxes in the true view: the object's visible pixels in the recorded image with their recorded ranges
  (the range image) are 3D points; each is projected through the TRUE pose (rotation and translation exactly) and the
  box is the extent of those inside the true image (the boundary pixels suffice: a smooth map takes a region's extreme
  points to its boundary). For the detector, the instance image is remapped with the image (nearest), as detection_eval
  does for its pinhole views.
* BEV localisation: each box's bottom-centre pixel is lifted to the ground (z = 0 in the vehicle frame: the ground
  truth's boxes stand on it, median bottom 0.01 m on the CARLA drives) through the USED pose, then moved along the
  horizontal viewing ray by the object's half extent along that ray (its true size: an oracle, so the figure isolates the
  pose, not size estimation). BEV error = the distance between that lift in the perturbed build and in the nominal
  build (the error the build adds); 'ate_m' (the records' and the nominal table's) = the nominal lift against the
  ground-truth centre. A lift whose ray no longer meets the ground is 'lift lost'; a box with no pixel left in the true
  image is 'out of view'.
* Detection: YOLOX (core.detection) on the remapped images, matched in 2D to the remapped ground truth with
  detection_eval's COCO matching and ignore rules (evaluate_image). Rig recall per class and region: an object (an
  actor at one instant) is found when any camera that sees it finds it; regions by its ground-truth centre.
* Cross-camera consistency: an object two or more cameras see lands in as many BEV places; a fusion gate of
  max(gate_m, gate_frac · range) associates them; an object whose lifts are farther apart in any pair is a duplicate
  (a ghost track). Ghost rate = such objects / objects with two or more valid lifts, nominal and per build.

Statistics
----------
Per class × region: the mean over builds, the 5th–95th percentile band over builds (90 % of builds within) and the
nominal build. Sensitivity: first-order and total Sobol indices of the BEV error, the ghost rate and the out-of-view rate
by source and by camera, from the pick-freeze estimators (Saltelli et al. 2010, Table 2: first order
mean(f(B)(f(A_B^i) − f(A))) / V; Jansen 1999 total effect mean((f(A) − f(A_B^i))²) / 2V), with bootstrap intervals over
the base samples; for detection (expensive) one source at a time (the variance of rig recall with only that source on;
it misses interactions, said so) unless sobol_detect is set.
"""
from __future__ import annotations

import csv
import hashlib
import io
import json
import math
import os
import time
import warnings
from dataclasses import asdict, dataclass, field

import numpy as np

from . import detection as DT
from . import detection_eval as DE
from .mathutil import euler_zyx, rot_y, rot_zyx
from .regions import RegionGrid, default_grid

SCHEMA = 'vantage.robust_perception/1'
RECORD_SCHEMA = 'vantage.perception_record/1'
FRAME = 'vehicle (ISO 8855, rear axle)'
BODY_TO_OPTICAL = np.array([[0.0, -1.0, 0.0], [0.0, 0.0, -1.0], [1.0, 0.0, 0.0]])
SOURCES = ('mounting', 'thermal', 'vibration', 'calibration')
BEV3D_METHOD = ('bev3d (setting): each build\'s 2D detections lifted to 3D through the calibration the perception '
                'uses (core.bev_backends.lift_boxes: ground contact and the class\'s size prior), fused across the '
                'cameras of a time slot (bev_backends.fuse) and matched to the slot\'s labelled objects by centre '
                'distance as core.rig_perception does (nuScenes matching, AP; recall and the centre error at 2 m). '
                'The cameras of a slot are taken as simultaneous (their capture times differ by up to one trigger '
                'period); the objects are those some camera\'s ground truth labels at the slot.')
SOURCE_LABEL = {'mounting': 'Mounting tolerance', 'thermal': 'Thermal warping', 'vibration': 'Road vibration',
                'calibration': 'Calibration residual'}
MODES = ('per_vehicle', 'design')
MODE_LABEL = {'per_vehicle': 'Each car calibrated after it is built', 'design': 'Every car on the design calibration'}
MOUNT_FROM = ('install', 'analysis', 'custom', 'fleet_twin')
STUDY_FROM = ('study', 'custom')
DISTS = ('uniform', 'normal')
CLASSES = ('pedestrian', 'cyclist', 'motorcyclist', 'car', 'truck', 'bus', 'vehicle')
ERR_CLIP_M = 100.0                     # a BEV error is clipped here in means (a ray grazing the horizon lifts to km)
MIN_BOUNDARY_PX = 3                    # an object with fewer boundary pixels left in the true image is out of view
INSTALL_DEG, INSTALL_MM = 1.0, 10.0    # core.install.InstallSettings: ang_tol_deg, pos_tol_mm
TOL_POS_MM = 10.0                      # core.tolerance.TolSettings.pos_mm
ASSUMPTIONS = (
    'Each source is the app\'s own model of it (installation tolerance or the analysis tolerance; core.thermo_vib\'s '
    'hot-day drift and ISO 8608 vibration; core.observability\'s predicted calibration σ) or a set figure; sources are '
    'independent of each other. The perception lifts 2D boxes to a flat ground through its static calibration (no '
    'dynamic pitch compensation, no ego-motion), with the objects\' true sizes. Rotations change the image exactly; '
    'translations change the geometry exactly and the image content not at all (below a pixel beyond Z_subpx). The '
    'detector is YOLOX trained on COCO run on CARLA renders: recall figures are the detector\'s on synthetic images; '
    'what carries over is how much they move between builds. Bands cover build variation on these recordings\' '
    'scenes, not other scenes, weather or traffic.')
LIMITS_TEXT = (
    'Content that a rotated camera would see beyond the recorded image is missing (black), never invented: objects '
    'entering the view there are not in the ground truth, so a turn can only lose objects at the image edge, not gain '
    'them. The body\'s heave and pitch are drawn independently (the half-car\'s correlation of the two is not kept). '
    'Mount flex is spread equally over three axes. Thermal drift is the study\'s hot day, one hour per build. Sobol '
    'indices of detection are one source at a time unless sobol_detect is set (it costs (k + 2) × n detector passes).')

LIMITS = {'mount_deg': (0.0, 20.0), 'mount_mm': (0.0, 500.0), 'thermal_deg': (0.0, 10.0), 'thermal_mm': (0.0, 100.0),
          'vib_body_deg': (0.0, 10.0), 'vib_flex_deg': (0.0, 10.0), 'vib_flex_mm': (0.0, 100.0),
          'vib_heave_mm': (0.0, 200.0), 'calib_deg': (0.0, 10.0), 'calib_mm': (0.0, 500.0), 'samples': (1, 2000),
          'sobol_n': (0, 20000), 'oat_samples': (0, 500), 'frame_step': (1, 1000), 'bev_frame_step': (1, 1000),
          'score': (0.01, 0.99), 'iou': (0.1, 0.95), 'gate_m': (0.0, 50.0), 'gate_frac': (0.0, 1.0),
          'seed': (0, 2 ** 31 - 1), 'boot': (0, 5000), 'max_slots': (1, 100000)}
INTS = ('samples', 'sobol_n', 'oat_samples', 'frame_step', 'bev_frame_step', 'seed', 'boot', 'max_slots')
BOOLS = ('mounting', 'thermal', 'vibration', 'calibration', 'detect', 'sobol_detect', 'records_per_sample',
         'bev3d')
CHOICES = {'lift': ('contact', 'box'), 'mode': MODES, 'mount_from': MOUNT_FROM, 'mount_dist': DISTS, 'thermal_from': STUDY_FROM,
           'vib_from': STUDY_FROM, 'calib_from': ('observability', 'custom')}


# =========================================================================== settings
@dataclass
class RobustSettings:
    recordings: list = field(default_factory=list)   # recording folders with ground truth (CARLA runs)
    mode: str = 'per_vehicle'          # per_vehicle | design (what calibration the perception uses)
    mounting: bool = True
    mount_from: str = 'install'        # install (1° / 10 mm) | analysis (analysis.tol_deg, 10 mm) | custom |
    #                                    fleet_twin (the fleet's measured spread and drift, core.fleet_twin)
    mount_deg: float = 1.0             # ± (custom)
    mount_mm: float = 10.0             # ± (custom)
    mount_dist: str = 'uniform'        # uniform | normal (± = 2 σ, clipped), as core.tolerance
    thermal: bool = True
    thermal_from: str = 'study'        # study (core.thermo_vib's hour-by-hour drift) | custom (1 σ below)
    thermal_deg: float = 0.05
    thermal_mm: float = 1.0
    vibration: bool = True
    vib_from: str = 'study'            # study (core.thermo_vib's RMS) | custom (below)
    vib_body_deg: float = 0.25         # 1 σ body pitch (custom)
    vib_heave_mm: float = 5.0          # 1 σ body heave (custom)
    vib_flex_deg: float = 0.01         # RMS mount flex (custom)
    vib_flex_mm: float = 0.05
    calibration: bool = True
    calib_from: str = 'observability'  # observability (predicted 1 σ per camera) | custom (1 σ below)
    calib_deg: float = 0.1
    calib_mm: float = 10.0
    samples: int = 24                  # builds evaluated with the detector
    sobol_n: int = 128                 # pick-freeze base samples for the BEV metrics ((k + 2) n builds; 0 = off)
    oat_samples: int = 8               # builds per source, one source at a time, for detection
    frame_step: int = 5                # every n-th instant for the detector
    bev_frame_step: int = 2            # every n-th instant for the BEV metrics
    max_slots: int = 100000
    detect: bool = True
    sobol_detect: bool = False
    score: float = 0.3
    iou: float = 0.5
    lift: str = 'contact'           # contact (the object's lowest visible point) | box (its 2D box's bottom-centre)
    gate_m: float = 2.0
    gate_frac: float = 0.10
    seed: int = 1729
    boot: int = 200
    records_per_sample: bool = False
    bev3d: bool = False                # also each build's 3D metric: the detector's boxes lifted through the calibration
    #                                    the perception uses, fused across cameras and matched on the ground as
    #                                    core.rig_perception does (rig_perception.evaluate_frames)

    def to_dict(self):
        return asdict(self)

    @staticmethod
    def from_dict(d) -> 'RobustSettings':
        s = RobustSettings()
        d = d if isinstance(d, dict) else {}
        recs = d.get('recordings')
        if isinstance(recs, (list, tuple)):
            s.recordings = [str(r) for r in recs if isinstance(r, str) and r.strip()][:DE.MAX_RECORDINGS]
        for k, opts in CHOICES.items():
            if d.get(k) in opts:
                setattr(s, k, d[k])
        for k in BOOLS:
            v = d.get(k)
            if isinstance(v, (bool, int)) and not isinstance(v, float):
                setattr(s, k, bool(v))
            elif isinstance(v, str):
                setattr(s, k, v.strip().lower() in ('1', 'true', 'yes', 'on'))
        for k, (lo, hi) in LIMITS.items():
            if k not in d:
                continue
            try:
                v = float(d[k])
            except (TypeError, ValueError):
                continue
            if not math.isfinite(v):
                continue
            v = min(max(v, lo), hi)
            setattr(s, k, int(round(v)) if k in INTS else v)
        return s

    def active(self) -> list:
        """The sources that act in this mode."""
        on = [k for k in SOURCES if getattr(self, k)]
        if self.mode == 'design':
            on = [k for k in on if k != 'calibration']
        return on


def settings_of(project) -> RobustSettings:
    return RobustSettings.from_dict(getattr(getattr(project, 'analysis', None), 'robust_perception', None) or {})


# =========================================================================== the sources
@dataclass
class Source:
    key: str
    on: bool
    origin: str                          # which study or setting the figures come from
    kind: str                            # how a sample draws it
    per_cam: dict = field(default_factory=dict)    # tag -> figures
    shared: dict = field(default_factory=dict)
    notes: list = field(default_factory=list)

    def table(self) -> list:
        """Rows for the page and the export: per camera its 1 σ (or ±) angles and moves."""
        out = []
        for tag, v in sorted(self.per_cam.items()):
            row = {'camera': tag} | {k: _r(x, 5) for k, x in v.items() if not isinstance(x, (list, tuple))}
            if isinstance(v.get('deg_axes'), (list, tuple)):          # measured per axis (the fleet twin)
                row.update({f'{a}_deg': _r(x, 5) for a, x in zip(('yaw', 'pitch', 'roll'), v['deg_axes'])})
            out.append(row)
        return out

    def to_doc(self):
        sh = {k: (_r(v, 6) if isinstance(v, float) else v) for k, v in self.shared.items() if k != 'series'}
        return {'source': self.key, 'label': SOURCE_LABEL[self.key], 'on': self.on, 'origin': self.origin,
                'draw': self.kind, 'per_camera': self.table(), 'shared': sh, 'notes': list(self.notes)}


def _variant_of(project, variant):
    if project is None:
        return None
    if variant is None:
        return project.variant
    if isinstance(variant, str):
        return next((v for v in project.variants if v.uid == variant or v.name == variant), project.variant)
    return variant


def resolve_sources(project, st: RobustSettings, tags, variant=None, engine=None, studies=None) -> dict:
    """Each source's figures per camera tag: {key: Source}. studies: {'thermo_vib': result, 'observability': result}
    to reuse runs (tests, the page); otherwise the studies run here (thermo_vib takes a fraction of a second,
    observability a few seconds and needs the engine)."""
    studies = dict(studies or {})
    v = _variant_of(project, variant)
    tags = list(tags)
    act = set(st.active())
    out = {}
    # ---- mounting
    if st.mount_from == 'install':
        deg, mm, origin = INSTALL_DEG, INSTALL_MM, 'core.install: the installation check\'s 1° / 10 mm (a build that ' \
                                                   'passes it)'
    elif st.mount_from == 'analysis' and project is not None:
        deg, mm = float(project.analysis.tol_deg), TOL_POS_MM
        origin = f'analysis.tol_deg ({deg:g}°) and core.tolerance\'s position default ({mm:g} mm)'
    else:
        deg, mm, origin = st.mount_deg, st.mount_mm, 'set by hand'
    dist = st.mount_dist
    fleet = None
    if st.mount_from == 'fleet_twin' and project is not None:
        from . import fleet_twin as FT
        tol = FT.cached_tolerance_inputs(project)
        if tol is None:                                          # not run yet in this session: it takes seconds
            try:
                r = FT.run(project, engine)
                tol = FT.tolerance_inputs(r, project=project) if r is not None else None
            except Exception:                                    # noqa: BLE001 — a fleet twin that cannot run is said
                tol = None
        fleet = (tol or {}).get('per_camera') or None
        if fleet:
            dist = 'normal'
            origin = (f'core.fleet_twin: the fleet\'s measured installation spread and drift after '
                      f'{tol.get("horizon_days", 180):g} days (as of {tol.get("as_of", "")}), per camera and axis')
    s = Source('mounting', 'mounting' in act, origin + (', uniform within ±' if dist == 'uniform' else
                                                        ', normal with ± = 2 σ (clipped)'), dist)
    if st.mount_from == 'fleet_twin' and not fleet:
        s.notes.append('No measured fleet drift (Calibrate ▸ Fleet twin has no camera with enough records): the '
                       'figures set by hand are used.')
    for t in tags:
        f = (fleet or {}).get(t)
        if f:
            ax = [float(f.get(k) or 0.0) for k in ('yaw', 'pitch', 'roll')]
            s.per_cam[t] = {'deg': max(ax), 'mm': float(f.get('pos_mm') or 0.0), 'deg_axes': ax}
        else:
            if fleet:
                s.notes.append(f'{t}: not followed by the fleet twin: the figures set by hand')
            s.per_cam[t] = {'deg': deg, 'mm': mm}
    out['mounting'] = s
    # ---- thermal and vibration (core.thermo_vib)
    tv = studies.get('thermo_vib')
    need_tv = (('thermal' in act and st.thermal_from == 'study') or ('vibration' in act and st.vib_from == 'study'))
    if tv is None and need_tv and project is not None:
        try:
            from . import thermo_vib as TV
            tv = TV.run(project, variant=v)
        except Exception as e:                                   # noqa: BLE001 — a study that fails is said
            tv = None
            studies['_tv_error'] = f'{type(e).__name__}: {e}'
    devs = {d['tag']: d for d in (tv.devices if tv is not None else []) if d.get('kind') == 'camera'}
    s = Source('thermal', 'thermal' in act, '', 'hour' if st.thermal_from == 'study' else 'normal')
    if st.thermal_from == 'study' and tv is not None:
        hours = sorted({float(h['hour']) for d in devs.values() for h in d.get('thermal') or ()})
        s.origin = (f'core.thermo_vib: drift from the garage calibration on {tv.day.get("date")}, hour by hour '
                    f'{_hh(hours[0]) if hours else ""}–{_hh(hours[-1]) if hours else ""} (one hour per build)')
        s.shared['hours'] = hours
        for t in tags:
            d = devs.get(t)
            ser = []
            for h in hours:
                row = next((x for x in (d or {}).get('thermal') or () if float(x['hour']) == h), None)
                ser.append([float((row or {}).get(k) or 0.0) for k in ('d_yaw_deg', 'd_pitch_deg', 'd_roll_deg',
                                                                       'dx_mm', 'dy_mm', 'dz_mm')])
            if d is None:
                s.notes.append(f'{t}: not in the thermal study (no drift)')
            ser = np.asarray(ser, float).reshape(-1, 6)
            s.per_cam[t] = {'series': ser.tolist(),
                            'worst_deg': float(np.abs(ser[:, :3]).max()) if len(ser) else 0.0,
                            'worst_mm': float(np.linalg.norm(ser[:, 3:], axis=1).max()) if len(ser) else 0.0}
    else:
        if st.thermal_from == 'study':
            s.notes.append('No thermal study for this project: the figures set by hand are used.')
        s.kind = 'normal'
        s.origin = 'set by hand (1 σ per axis)'
        for t in tags:
            s.per_cam[t] = {'deg': st.thermal_deg, 'mm': st.thermal_mm}
    out['thermal'] = s
    s = Source('vibration', 'vibration' in act, '', 'normal')
    if st.vib_from == 'study' and tv is not None:
        from . import thermo_vib as TV
        rows = [d for t, d in devs.items() if t in tags and d.get('vib')]
        b2 = [max(float(d['vib']['angle_deg_rms']) ** 2 - float(d['vib']['flex_deg_rms']) ** 2, 0.0) for d in rows]
        body = math.sqrt(float(np.mean(b2))) if b2 else 0.0
        veh = tv.vehicle or {}
        cg = (float(veh.get('cg_x_m') or 0.0), float(veh.get('cg_z_m') or 0.0))
        heave2 = []
        f = tv.psd.get('f')
        for d in rows:
            pd = tv.psd.get('devices', {}).get(d['uid']) or {}
            hb = pd.get('h_body')
            if hb is None or f is None:
                continue
            zr = math.sqrt(max(TV.integrate(f, np.asarray(hb, float) ** 2 * tv.psd['road']), 0.0))
            P = _dev_pos(project, v, d['tag'])
            lever = (P[0] - cg[0]) if P is not None else 0.0
            heave2.append(zr ** 2 - (lever * math.radians(body)) ** 2)
        heave = math.sqrt(max(min(heave2), 0.0)) * 1000.0 if heave2 else 0.0
        s.origin = (f'core.thermo_vib: ISO 8608 class {tv.settings.road_class} at {tv.settings.speed_kmh:g} km/h '
                    f'(half-car body pitch and heave, each mount\'s flex)')
        s.shared.update(body_deg=body, heave_mm=heave, cg_x=cg[0], cg_z=cg[1])
        for t in tags:
            d = devs.get(t)
            if d is None:
                s.notes.append(f'{t}: not in the vibration study (body motion only)')
            vb = (d or {}).get('vib') or {}
            s.per_cam[t] = {'flex_deg': float(vb.get('flex_deg_rms') or 0.0),
                            'flex_mm': float(vb.get('flex_mm_rms') or 0.0)}
    else:
        if st.vib_from == 'study':
            s.notes.append('No vibration study for this project: the figures set by hand are used.')
        s.origin = 'set by hand (1 σ body pitch and heave, RMS mount flex)'
        cgx = 0.0
        try:
            from . import thermo_vib as TV
            cgx = TV.half_car(project, TV.settings_of(project), v).x_cg if project is not None else 0.0
        except Exception:                                        # noqa: BLE001
            pass
        s.shared.update(body_deg=st.vib_body_deg, heave_mm=st.vib_heave_mm, cg_x=float(cgx), cg_z=0.5)
        for t in tags:
            s.per_cam[t] = {'flex_deg': st.vib_flex_deg, 'flex_mm': st.vib_flex_mm}
    out['vibration'] = s
    # ---- calibration (core.observability)
    s = Source('calibration', 'calibration' in act, '', 'normal')
    ob = studies.get('observability')
    if st.calib_from == 'observability' and ob is None and 'calibration' in act and project is not None:
        try:
            from . import observability as OB
            from .analysis import Engine
            ob = OB.run(project, engine or Engine(), variant=v)
        except Exception as e:                                   # noqa: BLE001
            ob = None
            s.notes.append(f'The observability study failed ({type(e).__name__}: {e}): the figures set by hand '
                           'are used.')
    if st.calib_from == 'observability' and ob is not None:
        by = {c['tag']: c for c in ob.cameras}
        s.origin = ('core.observability: the predicted 1 σ of the installation check\'s joint solve with the '
                    f'planned boards (pixel σ {ob.settings.pixel_sigma:g} px)')
        for t in tags:
            c = by.get(t)
            sg = (c or {}).get('sigma') or {}
            if not c or not sg or not all(math.isfinite(float(sg.get(k, math.nan))) for k in
                                          ('yaw_deg', 'pitch_deg', 'roll_deg', 'x_mm', 'y_mm', 'z_mm')):
                s.notes.append(f'{t}: no board pins it in the plan: the figures set by hand '
                               f'({st.calib_deg:g}° / {st.calib_mm:g} mm)')
                s.per_cam[t] = {'yaw_deg': st.calib_deg, 'pitch_deg': st.calib_deg, 'roll_deg': st.calib_deg,
                                'x_mm': st.calib_mm, 'y_mm': st.calib_mm, 'z_mm': st.calib_mm}
            else:
                s.per_cam[t] = {k: float(sg[k]) for k in ('yaw_deg', 'pitch_deg', 'roll_deg', 'x_mm', 'y_mm', 'z_mm')}
    else:
        s.origin = 'set by hand (1 σ per axis)'
        for t in tags:
            s.per_cam[t] = {'yaw_deg': st.calib_deg, 'pitch_deg': st.calib_deg, 'roll_deg': st.calib_deg,
                            'x_mm': st.calib_mm, 'y_mm': st.calib_mm, 'z_mm': st.calib_mm}
    out['calibration'] = s
    if studies.get('_tv_error'):
        out['thermal'].notes.append(f'The thermal and vibration study failed: {studies["_tv_error"]}')
    return out


def _dev_pos(project, v, tag):
    if v is None:
        return None
    c = next((c for c in v.cameras if c.tag == tag), None)
    return None if c is None else (float(c.x), float(c.y), float(c.z))


# =========================================================================== drawing builds
def _ppf(u):
    """Standard normal quantile (Acklam's rational approximation, |error| < 1.2e-9 after one Halley step)."""
    from scipy.special import ndtri
    return ndtri(np.clip(np.asarray(u, float), 1e-12, 1 - 1e-12))


class ErrorModel:
    """Maps a row of independent U(0, 1) numbers to every camera's pose errors. Columns: per camera 6 for mounting,
    6 for mount flex, 6 for calibration; then the shared hour (thermal), the body's pitch and heave (vibration)."""

    def __init__(self, sources: dict, tags, st: RobustSettings):
        self.src = sources
        self.tags = list(tags)
        self.st = st
        self.n_cam = len(self.tags)
        self.D = 18 * self.n_cam + 3
        self.act = set(st.active())

    def cols(self, source=None, cam=None) -> list:
        """Columns of a source (and/or a camera index); the shared columns belong to no camera."""
        out = []
        for ci in range(self.n_cam):
            if cam is not None and ci != cam:
                continue
            base = 18 * ci
            if source in (None, 'mounting'):
                out += list(range(base, base + 6))
            if source in (None, 'vibration'):
                out += list(range(base + 6, base + 12))
            if source in (None, 'calibration'):
                out += list(range(base + 12, base + 18))
        sh = 18 * self.n_cam
        if cam is None:
            if source in (None, 'thermal'):
                out.append(sh)
            if source in (None, 'vibration'):
                out += [sh + 1, sh + 2]
        return out

    def shared_cols(self) -> list:
        sh = 18 * self.n_cam
        return [sh, sh + 1, sh + 2]

    def errors(self, u, only=None) -> dict:
        """One build: {tag: {'true': (dangles deg (3,), dpos m (3,)), 'used': (...)}, 'body': (pitch deg, heave m)}.
        only: the sources that act (default: the settings' active ones)."""
        u = np.asarray(u, float)
        act = self.act if only is None else set(only) & self.act
        st = self.st
        z = _ppf(u)
        out = {}
        sh = 18 * self.n_cam
        vib = self.src['vibration']
        body = float(z[sh + 1]) * vib.shared.get('body_deg', 0.0) if 'vibration' in act else 0.0
        heave = float(z[sh + 2]) * vib.shared.get('heave_mm', 0.0) / 1000.0 if 'vibration' in act else 0.0
        th = self.src['thermal']
        hours = th.shared.get('hours') or []
        hi = min(int(u[sh] * len(hours)), len(hours) - 1) if hours else -1
        for ci, tag in enumerate(self.tags):
            b = 18 * ci
            da, dp = np.zeros(3), np.zeros(3)
            ua, up = np.zeros(3), np.zeros(3)                  # the used (calibrated) pose's errors
            if 'mounting' in act:
                m = self.src['mounting'].per_cam[tag]
                if self.src['mounting'].kind == 'normal':
                    w = np.clip(z[b:b + 6] * 0.5, -1.0, 1.0)
                else:
                    w = 2.0 * u[b:b + 6] - 1.0
                mdeg = np.asarray(m.get('deg_axes', m['deg']), float)   # per axis (yaw, pitch, roll) when measured
                da += w[:3] * mdeg
                dp += w[3:] * m['mm'] / 1000.0
                if st.mode == 'per_vehicle':                   # the calibration measures the build
                    ua += w[:3] * mdeg
                    up += w[3:] * m['mm'] / 1000.0
            if 'thermal' in act:
                c = th.per_cam[tag]
                if th.kind == 'hour' and hi >= 0:
                    r = np.asarray(c['series'][hi], float)
                    da += r[:3]
                    dp += r[3:] / 1000.0
                elif th.kind != 'hour':
                    # custom: normal per axis, from the shared hour column's neighbours (one draw per camera)
                    g = _ppf(_mix(u[b:b + 6], u[sh]))
                    da += g[:3] * c['deg']
                    dp += g[3:] * c['mm'] / 1000.0
            if 'vibration' in act:
                c = vib.per_cam[tag]
                da += z[b + 6:b + 9] * c['flex_deg'] / math.sqrt(3.0)
                dp += z[b + 9:b + 12] * c['flex_mm'] / 1000.0 / math.sqrt(3.0)
            if 'calibration' in act and st.mode == 'per_vehicle':
                c = self.src['calibration'].per_cam[tag]
                ua += z[b + 12:b + 15] * np.array([c['yaw_deg'], c['pitch_deg'], c['roll_deg']])
                up += z[b + 15:b + 18] * np.array([c['x_mm'], c['y_mm'], c['z_mm']]) / 1000.0
            out[tag] = {'true': (da, dp), 'used': (ua, up)}
        out['body'] = (body, heave)
        return out


def _mix(a, b):
    """Independent-looking uniforms from a camera's columns and the shared one (custom thermal only): the fractional
    part of a + b is U(0, 1) and independent of b when a is."""
    return np.mod(np.asarray(a, float) + float(b), 1.0)


@dataclass
class Pose:
    R: np.ndarray       # optical -> vehicle
    t: np.ndarray       # optical centre in the vehicle frame (m)


def build_poses(cam: DE.Camera, euler, err_true, err_used, body, cg) -> tuple[Pose, Pose]:
    """The true and used poses of a camera from its design Euler angles and a build's errors. body: (pitch deg nose
    down, heave m) of the whole car about its centre of gravity cg (x, z)."""
    yaw, pitch, roll = euler
    da, dp = err_true
    Rb = rot_zyx(yaw + da[0], pitch + da[1], roll + da[2])
    t = np.asarray(cam.t, float) + dp
    beta, heave = body
    if beta or heave:
        Rp = rot_y(math.radians(beta))
        c = np.array([cg[0], 0.0, cg[1]])
        Rb = Rp @ Rb
        t = Rp @ (t - c) + c + np.array([0.0, 0.0, heave])
    ua, up = err_used
    Ru = rot_zyx(yaw + ua[0], pitch + ua[1], roll + ua[2])
    tu = np.asarray(cam.t, float) + up
    return Pose(Rb @ BODY_TO_OPTICAL.T, t), Pose(Ru @ BODY_TO_OPTICAL.T, tu)


def design_euler(cam: DE.Camera) -> tuple:
    """The camera's design yaw, pitch, roll (degrees) from the recording's optical -> vehicle rotation."""
    return euler_zyx(np.asarray(cam.R, float) @ BODY_TO_OPTICAL)


# =========================================================================== image remap (rotation)
def grid_rays(cam: DE.Camera):
    """Unit optical-frame rays of every pixel centre (H × W × 3, float64) and ok (inside the lens's field)."""
    vv, uu = np.mgrid[0:cam.height, 0:cam.width]
    d, ok = DE.kb_unproject(cam, uu.ravel().astype(float), vv.ravel().astype(float))
    return d.reshape(cam.height, cam.width, 3), ok.reshape(cam.height, cam.width)


def rotation_view(cam: DE.Camera, R_rel, rays=None, name='build') -> tuple[DE.View, np.ndarray]:
    """The true camera's view as a detection_eval.View cut from the recorded image: R_rel turns the true camera's
    optical rays into the recorded camera's (R_design.T @ R_true). Returns (view, missing): missing marks the true
    image's pixels with no recorded content (outside the recorded image or the lens's field)."""
    if rays is None:
        rays = grid_rays(cam)
    d, ok_t = rays
    H, W = cam.height, cam.width
    flat = d.reshape(-1, 3)
    q = flat @ np.asarray(R_rel, float).T                       # into the recorded camera's frame
    u, v, ok = DE.kb_project(cam, q)
    # inside the recorded pixel centres: bilinear sampling then never mixes in content from beyond the recording
    ok &= ok_t.ravel() & (u >= 0.0) & (u <= W - 1.0) & (v >= 0.0) & (v <= H - 1.0)
    mx = np.where(ok, u, -10.0).reshape(H, W).astype(np.float32)
    my = np.where(ok, v, -10.0).reshape(H, W).astype(np.float32)
    back = flat @ np.asarray(R_rel, float)                       # recorded rays in the true camera's frame
    pu, pv, ok_b = DE.kb_project(cam, back)
    inside = ok_b & ok_t.ravel() & (pu >= -0.5) & (pu <= W - 0.5) & (pv >= -0.5) & (pv <= H - 0.5)
    view = DE.View(name, 0.0, W, H, np.asarray(cam.K, float), np.asarray(R_rel, float), mx, my,
                   inside.reshape(H, W))
    return view, ~ok.reshape(H, W)


def remap_image(img, view: DE.View, nearest=False):
    return DE.remap(img, view, nearest=nearest)


def max_px_per_rad(cam: DE.Camera) -> float:
    """The lens's largest image scale (px per radian of ray angle), radial or tangential, inside its field: the bound
    on how far a translation's change of a ray moves its pixel."""
    lim = DE._theta_limit(cam)
    th = np.linspace(1e-4, lim, 2000)
    f = float(cam.K[0, 0])
    if cam.D is None or cam.lens == 'pinhole':
        th = th[th < math.radians(89.0)]
        return float(max(f / np.cos(th) ** 2).max()) if len(th) else f
    k1, k2, k3, k4 = cam.D
    t2 = th * th
    rd = th * (1 + t2 * (k1 + t2 * (k2 + t2 * (k3 + t2 * k4))))
    drd = 1 + t2 * (3 * k1 + t2 * (5 * k2 + t2 * (7 * k3 + t2 * 9 * k4)))
    return float(f * max(drd.max(), (rd / np.sin(th)).max()))


# =========================================================================== the BEV lift
def lift(cam: DE.Camera, pose: Pose, uv, z_ground=0.0):
    """Pixels (n × 2) of a camera -> their rays' ground points (n × 3, vehicle frame) and ok (the ray meets the
    ground ahead of the camera)."""
    uv = np.asarray(uv, float).reshape(-1, 2)
    d, ok = DE.kb_unproject(cam, uv[:, 0], uv[:, 1])
    dv = d @ pose.R.T
    with np.errstate(divide='ignore', invalid='ignore'):
        lam = (z_ground - pose.t[2]) / dv[:, 2]
    ok &= np.isfinite(lam) & (lam > 0) & (dv[:, 2] < -1e-9)
    P = pose.t[None, :] + np.where(ok, lam, 0.0)[:, None] * dv
    return P, ok


def half_extent_along(dirs_xy, size_xy, yaw_deg):
    """Distance from a box's centre to its edge along each horizontal direction (the box's footprint)."""
    a = np.maximum(np.asarray(size_xy, float)[..., 0] / 2.0, 1e-3)
    b = np.maximum(np.asarray(size_xy, float)[..., 1] / 2.0, 1e-3)
    y = np.radians(np.asarray(yaw_deg, float))
    c, s = np.cos(y), np.sin(y)
    dx, dy = dirs_xy[..., 0], dirs_xy[..., 1]
    ex = np.abs(dx * c + dy * s)
    ey = np.abs(-dx * s + dy * c)
    return 1.0 / np.maximum(np.maximum(ex / a, ey / b), 1e-9)


def pitch_error_analytic(h, D, dpitch_deg):
    """Ground-plane lift error of a level-ish camera at height h for a ground point at horizontal distance D when the
    camera is pitched dpitch further down than its calibration says: h / tan(atan(h / D) − δ) − D (m; NaN when the
    ray no longer meets the ground)."""
    a = math.atan2(h, D) - math.radians(dpitch_deg)
    return h / math.tan(a) - D if a > 0 else math.nan


# =========================================================================== a recording prepared for the study
@dataclass
class _Inst:
    """One object seen by one camera at one instant (non-ignored in the nominal view)."""
    cam: int
    slot: int
    key: str
    cls: int            # index into DT.CLASSES
    xy: tuple           # ground-truth centre (vehicle frame)
    size: tuple         # (length, width)
    yaw: float
    rng: float          # ground-truth range from the camera (m)


class Scene:
    """A recording's cameras, instants and ground-truth objects as the study needs them (boundary pixels as 3D points
    with their recorded ranges)."""

    def __init__(self, rec: DE.Recording, st: RobustSettings, dst: DE.DetectionSettings, grid: RegionGrid,
                 cancel=None, progress=None):
        from . import groundtruth as GT
        self.rec = rec
        self.grid = grid
        self.dst = dst
        tags = sorted(t for t, c in rec.cameras.items() if c.K is not None and c.R is not None and c.lens)
        self.tags = tags
        self.cams = [rec.cameras[t] for t in tags]
        self.euler = [design_euler(c) for c in self.cams]
        self.notes = []
        skipped = sorted(set(rec.cameras) - set(tags))
        if skipped:
            self.notes.append(f'{", ".join(skipped)}: no lens calibration in the recording: left out.')
        # instants: trigger periods of the drive (the cameras fire at phases within one period: sync groups); every
        # bev_frame_step-th period for the BEV metrics, every frame_step-th for the detector
        ts_all = sorted({int(f.get('timestamp') or 0) for c in self.cams for f in c.frames})
        t0 = ts_all[0] if ts_all else 0
        per = []
        for c in self.cams:
            tt = sorted(int(f.get('timestamp') or 0) for f in c.frames)
            if len(tt) > 1:
                per.append(float(np.median(np.diff(tt))))
        self.period_us = max(min(per), 1.0) if per else 1.0
        self.frames = []
        n_slot = 0
        for c in self.cams:
            by = {}
            for f in c.frames:
                k = int(math.floor((int(f.get('timestamp') or 0) - t0) / self.period_us + 1e-6))
                by.setdefault(k, f)
                n_slot = max(n_slot, k + 1)
            self.frames.append(by)
        slots_all = list(range(0, n_slot, max(int(st.bev_frame_step), 1)))[:int(st.max_slots)]
        self.slots = slots_all
        self.slot_index = {t: i for i, t in enumerate(slots_all)}
        self.det_slots = list(range(0, n_slot, max(int(st.frame_step), 1)))[:int(st.max_slots)]
        # boundary pixels of every non-ignored object of an evaluated class, per camera
        self.inst: list[_Inst] = []
        self.X = []            # per camera: (n_px, 3) points
        self.oi = []           # per camera: (n_px,) instance index (sorted)
        cls_ok = {DT.CLASSES.index(c) for c in CLASSES if c in DT.CLASSES}
        n_work = len(self.cams) * len(slots_all) or 1
        done = 0
        for ci, c in enumerate(self.cams):
            Xs, Os = [], []
            for si, t in enumerate(slots_all):
                if cancel is not None and cancel():
                    raise _Cancelled()
                done += 1
                if progress is not None and done % 10 == 0:
                    progress(done / n_work)
                fr = self.frames[ci].get(t)
                if fr is None or not fr.get('instance') or not fr.get('range'):
                    continue
                g = DE.gt_objects(fr, dst)
                keep = [i for i in range(len(g['id'])) if g['ignore'][i] == DE.IGN_NONE and int(g['cls'][i]) in cls_ok
                        and np.all(np.isfinite(g['xy'][i]))]
                if not keep:
                    continue
                try:
                    inst = GT.load_instance(rec.gt, fr)
                    rng = GT.load_range(rec.gt, fr)
                except OSError:
                    continue
                bnd = boundary_mask(inst)
                vs, us = np.nonzero(bnd)
                ids = inst[vs, us].astype(np.int64)
                objs = {o.get('id'): o for o in fr.get('objects') or () if isinstance(o, dict)}
                d, ok = DE.kb_unproject(c, us.astype(float), vs.astype(float))
                r = rng[vs, us].astype(float)
                good = ok & (r > 0)
                for i in keep:
                    oid = int(g['id'][i])
                    sel = good & (ids == oid)
                    if sel.sum() < MIN_BOUNDARY_PX:
                        continue
                    o = objs.get(oid) or {}
                    b3 = o.get('box3d') or {}
                    size = b3.get('size') or (0.5, 0.5, 1.7)
                    k = len(self.inst)
                    self.inst.append(_Inst(ci, si, str(g['key'][i]), int(g['cls'][i]),
                                           (float(g['xy'][i][0]), float(g['xy'][i][1])),
                                           (float(size[0]), float(size[1])), float(b3.get('yaw_deg') or 0.0),
                                           float(g['range'][i]) if math.isfinite(g['range'][i]) else math.nan))
                    P = c.t[None, :] + (d[sel] * r[sel, None]) @ c.R.T
                    Xs.append(P)
                    Os.append(np.full(int(sel.sum()), k, np.int64))
            self.X.append(np.concatenate(Xs) if Xs else np.zeros((0, 3)))
            self.oi.append(np.concatenate(Os) if Os else np.zeros(0, np.int64))
        n = len(self.inst)
        self.n = n
        self.i_cam = np.array([o.cam for o in self.inst], int)
        self.i_cls = np.array([o.cls for o in self.inst], int)
        self.i_xy = np.array([o.xy for o in self.inst], float).reshape(n, 2)
        self.i_size = np.array([o.size for o in self.inst], float).reshape(n, 2)
        self.i_yaw = np.array([o.yaw for o in self.inst], float)
        self.i_rng = np.array([o.rng for o in self.inst], float)
        s, r = grid.assign(self.i_xy[:, 0], self.i_xy[:, 1]) if n else (np.zeros(0, int), np.zeros(0, int))
        self.i_sec, self.i_rb = np.asarray(s, int), np.asarray(r, int)
        # multi-view groups: the same object (key) at the same instant in two or more cameras
        groups = {}
        for k, o in enumerate(self.inst):
            groups.setdefault((o.slot, o.key), []).append(k)
        pa, pb, pg, gl = [], [], [], []
        for gk, mem in groups.items():
            if len(mem) < 2:
                continue
            gi = len(gl)
            gl.append(mem)
            for x in range(len(mem)):
                for y in range(x + 1, len(mem)):
                    pa.append(mem[x])
                    pb.append(mem[y])
                    pg.append(gi)
        self.groups = gl
        self.pa, self.pb, self.pg = np.array(pa, int), np.array(pb, int), np.array(pg, int)
        if gl:
            first = np.array([m[0] for m in gl], int)
            self.g_cls, self.g_sec, self.g_rb = self.i_cls[first], self.i_sec[first], self.i_rb[first]
            self.g_rng = np.hypot(self.i_xy[first, 0] - grid.origin_xy[0], self.i_xy[first, 1] - grid.origin_xy[1])
        else:
            self.g_cls = self.g_sec = self.g_rb = np.zeros(0, int)
            self.g_rng = np.zeros(0)
        self.rays = {}
        self.nominal = None
        self.lift_mode = st.lift

    def info(self) -> dict:
        return {'label': self.rec.label, 'cameras': list(self.tags), 'instants': len(self.slots),
                'detector_instants': len(self.det_slots), 'objects': int(self.n),
                'multi_view_objects': len(self.groups), 'scenario': self.rec.scenario, 'seed': self.rec.seed,
                'town': self.rec.town, 'weather': self.rec.weather, 'duration_s': self.rec.duration_s}

    def poses(self, errs: dict, cg) -> list:
        return [build_poses(c, self.euler[ci], errs[t]['true'], errs[t]['used'], errs['body'], cg)
                for ci, (t, c) in enumerate(zip(self.tags, self.cams))]

    def nominal_poses(self) -> list:
        z = np.zeros(3)
        return [build_poses(c, self.euler[ci], (z, z), (z, z), (0.0, 0.0), (0.0, 0.0))
                for ci, c in enumerate(self.cams)]

    # ------------------------------------------------------------------ the BEV evaluation of one build
    def bev(self, poses) -> dict:
        """Every instance's lifted centre (n × 2), ok, and out-of-view flags for a build's (true, used) poses."""
        n = self.n
        P = np.full((n, 2), np.nan)
        ok = np.zeros(n, bool)
        oov = np.zeros(n, bool)
        for ci, c in enumerate(self.cams):
            X, oi = self.X[ci], self.oi[ci]
            if not len(X):
                continue
            true, used = poses[ci]
            q = (X - true.t[None, :]) @ true.R                  # into the true optical frame
            u, v, okp = DE.kb_project(c, q)
            okp &= (u >= -0.5) & (u <= c.width - 0.5) & (v >= -0.5) & (v <= c.height - 0.5) & (q[:, 2] > -1e9)
            mine = np.flatnonzero(self.i_cam == ci)
            if not len(mine):
                continue
            # per instance (oi sorted by instance): extents of the points left in view
            cnt = np.bincount(oi, weights=okp.astype(float), minlength=n)[mine]
            big = 1e12
            umin = np.full(n, big)
            umax = np.full(n, -big)
            vmax = np.full(n, -big)
            np.minimum.at(umin, oi[okp], u[okp])
            np.maximum.at(umax, oi[okp], u[okp])
            np.maximum.at(vmax, oi[okp], v[okp])
            vis = cnt >= MIN_BOUNDARY_PX
            oov[mine] = ~vis
            sel = mine[vis]
            if not len(sel):
                continue
            if self.lift_mode == 'box':
                uv = np.column_stack([(umin[sel] + umax[sel]) / 2.0, np.minimum(vmax[sel] + 0.5, c.height - 0.5)])
                G, okl = lift(c, used, uv)
            else:
                # the object's ground contact: of its visible outline's pixels, read through the USED calibration,
                # the ray meeting the ground nearest the camera (its lowest visible point). unproject(project(q)) is
                # q's direction inside the field, so the true-view pixel's ray is q / |q| without a Newton solve.
                dq = q[okp] / np.linalg.norm(q[okp], axis=1, keepdims=True)
                dv = dq @ used.R.T
                down = dv[:, 2] < -1e-9
                lam = np.where(down, (0.0 - used.t[2]) / np.where(down, dv[:, 2], -1.0), np.inf)
                lam = np.where(lam > 0, lam, np.inf)
                hd = lam * np.hypot(dv[:, 0], dv[:, 1])
                oo = oi[okp]
                best = np.full(n, np.inf)
                np.minimum.at(best, oo, hd)
                # the point reaching that minimum (ties: any)
                hit = np.isfinite(hd) & (hd <= best[oo] + 1e-9)
                idx = np.full(n, -1, np.int64)
                idx[oo[hit]] = np.flatnonzero(hit)
                okl = idx[sel] >= 0
                G = np.zeros((len(sel), 3))
                j = idx[sel][okl]
                G[okl] = used.t[None, :] + lam[j, None] * dv[j]
            h = G[:, :2] - used.t[None, :2]
            hn = np.linalg.norm(h, axis=1)
            dirs = h / np.where(hn > 1e-9, hn, 1.0)[:, None]
            push = np.where(hn > 1e-9, half_extent_along(dirs, self.i_size[sel], self.i_yaw[sel]), 0.0)
            P[sel] = np.where(okl[:, None], G[:, :2] + dirs * push[:, None], np.nan)
            ok[sel] = okl
        return {'P': P, 'ok': ok, 'oov': oov}

    def ghosts(self, P, ok, st: RobustSettings) -> tuple[np.ndarray, np.ndarray]:
        """Per multi-view group: (has two or more valid lifts, any pair beyond the fusion gate)."""
        if not len(self.groups):
            return np.zeros(0, bool), np.zeros(0, bool)
        ng = len(self.groups)
        valid_pair = ok[self.pa] & ok[self.pb]
        # each lift against its own camera's instant's ground truth: the cameras fire up to a period apart, and a
        # fusion with ego-motion compensation and tracking compares them at one instant
        d = np.linalg.norm((P[self.pa] - self.i_xy[self.pa]) - (P[self.pb] - self.i_xy[self.pb]), axis=1)
        gate = np.maximum(st.gate_m, st.gate_frac * self.g_rng[self.pg])
        bad = valid_pair & (d > gate)
        multi = np.zeros(ng, bool)
        np.logical_or.at(multi, self.pg, valid_pair)
        ghost = np.zeros(ng, bool)
        np.logical_or.at(ghost, self.pg, bad)
        return multi, ghost & multi


class _Cancelled(Exception):
    pass


def boundary_mask(inst) -> np.ndarray:
    """Pixels of an instance image whose id differs from a 4-neighbour's (or that lie on the image's edge), id > 0."""
    a = np.asarray(inst)
    m = np.zeros(a.shape, bool)
    m[:, 1:] |= a[:, 1:] != a[:, :-1]
    m[:, :-1] |= a[:, :-1] != a[:, 1:]
    m[1:, :] |= a[1:, :] != a[:-1, :]
    m[:-1, :] |= a[:-1, :] != a[1:, :]
    m[0, :] = m[-1, :] = True
    m[:, 0] = m[:, -1] = True
    return m & (a > 0)


# =========================================================================== Sobol indices (pick-freeze)
def sobol_indices(f, D: int, groups: dict, n: int, seed: int = 0, boot: int = 200, conf: float = 0.9,
                  progress=None, cancel=None) -> dict:
    """First-order and total Sobol indices of the outputs of f over D independent U(0, 1) inputs, per group of
    columns, by pick-freeze: A, B (n × D), A_B^g = A with g's columns from B; S_g = mean(f(B)(f(A_B^g) − f(A))) / V
    (Saltelli et al. 2010, Table 2 (b)), ST_g = mean((f(A) − f(A_B^g))²) / 2V (Jansen 1999), V the variance of f over
    A and B together. f maps an (m × D) matrix to (m × k) outputs. Bootstrap percentile intervals over the n rows.
    Returns {'n', 'evaluations', 'V' (k,), 'mean' (k,), 'groups': {g: {'S' (k,), 'ST' (k,), 'S_ci', 'ST_ci'}}, 'yA',
    'yB'}."""
    rng = np.random.default_rng(seed)
    A = rng.random((n, D))
    B = rng.random((n, D))
    steps = 2 + len(groups)
    k_done = [0]

    def ev(M):
        if cancel is not None and cancel():
            raise _Cancelled()
        y = np.asarray(f(M), float)
        y = y.reshape(len(M), -1)
        k_done[0] += 1
        if progress is not None:
            progress(k_done[0] / steps)
        return y
    yA, yB = ev(A), ev(B)
    allv = np.vstack([yA, yB])
    V = allv.var(axis=0)
    mean = allv.mean(axis=0)
    out = {'n': n, 'evaluations': n * steps, 'V': V, 'mean': mean, 'groups': {}, 'yA': yA, 'yB': yB, 'A': A, 'B': B}
    br = np.random.default_rng(seed + 1)
    idx_b = br.integers(0, n, (boot, n)) if boot else None
    lo, hi = (1 - conf) / 2 * 100, (1 + conf) / 2 * 100
    for gname, cols in groups.items():
        M = A.copy()
        M[:, cols] = B[:, cols]
        yAB = ev(M)
        with np.errstate(invalid='ignore', divide='ignore'):
            S = np.mean(yB * (yAB - yA), axis=0) / V
            ST = 0.5 * np.mean((yA - yAB) ** 2, axis=0) / V
        rec = {'S': S, 'ST': ST, 'S_ci': None, 'ST_ci': None}
        if idx_b is not None:
            Sb, STb = [], []
            for ib in idx_b:
                a_, b_, ab_ = yA[ib], yB[ib], yAB[ib]
                vb = np.vstack([a_, b_]).var(axis=0)
                with np.errstate(invalid='ignore', divide='ignore'):
                    Sb.append(np.mean(b_ * (ab_ - a_), axis=0) / vb)
                    STb.append(0.5 * np.mean((a_ - ab_) ** 2, axis=0) / vb)
            Sb, STb = np.asarray(Sb), np.asarray(STb)
            with warnings.catch_warnings():
                warnings.simplefilter('ignore', RuntimeWarning)          # V = 0 (no spread): no interval
                rec['S_ci'] = np.stack([np.nanpercentile(Sb, lo, axis=0), np.nanpercentile(Sb, hi, axis=0)], -1)
                rec['ST_ci'] = np.stack([np.nanpercentile(STb, lo, axis=0), np.nanpercentile(STb, hi, axis=0)], -1)
        out['groups'][gname] = rec
    return out


def ishigami(X, a=7.0, b=0.1):
    """The Ishigami test function on X ∈ [−π, π]³ (Ishigami and Homma 1990): its first-order indices are known in
    closed form (ishigami_indices)."""
    X = np.asarray(X, float)
    return np.sin(X[:, 0]) + a * np.sin(X[:, 1]) ** 2 + b * X[:, 2] ** 4 * np.sin(X[:, 0])


def ishigami_indices(a=7.0, b=0.1) -> tuple:
    """(S1, S2, S3, ST1, ST2, ST3) of the Ishigami function (Sobol' and Levitan 1999; Saltelli et al. 2008 §1.2)."""
    pi = math.pi
    V = a * a / 8 + b * pi ** 4 / 5 + b * b * pi ** 8 / 18 + 0.5
    V1 = 0.5 * (1 + b * pi ** 4 / 5) ** 2
    V2 = a * a / 8
    V13 = b * b * pi ** 8 * (1 / 18 - 1 / 50)
    return V1 / V, V2 / V, 0.0, (V1 + V13) / V, V2 / V, V13 / V


# =========================================================================== aggregation
def _cls_members(name) -> list:
    return [DT.CLASSES.index(m) for m in DT.GROUPS.get(name, (name,)) if m in DT.CLASSES]


def _rid(grid, s, r):
    return grid.region_id(int(s), int(r))


def bev_metrics(scene: Scene, nom: dict, b: dict, st: RobustSettings) -> dict:
    """A build's scalar metrics against the nominal lifts: mean BEV error (clipped), out-of-view rate, lift-lost rate,
    ghost rate; overall and per class."""
    base = nom['ok']
    e = np.linalg.norm(b['P'] - nom['P'], axis=1)
    valid = base & b['ok']
    out = {}
    multi, ghost = scene.ghosts(b['P'], b['ok'], st)
    for cname in ('all',) + CLASSES:
        m = base if cname == 'all' else base & np.isin(scene.i_cls, _cls_members(cname))
        mv = m & valid
        out[f'err:{cname}'] = float(np.mean(np.minimum(e[mv], ERR_CLIP_M))) if mv.any() else math.nan
        out[f'oov:{cname}'] = float(b['oov'][m].mean()) if m.any() else math.nan
        out[f'lost:{cname}'] = float((~b['ok'][m] & ~b['oov'][m]).mean()) if m.any() else math.nan
        gm = multi if cname == 'all' else multi & np.isin(scene.g_cls, _cls_members(cname))
        out[f'ghost:{cname}'] = float(ghost[gm].mean()) if gm.any() else math.nan
    return out


SOBOL_METRICS = ('err:all', 'ghost:all', 'oov:all', 'err:pedestrian', 'err:vehicle')


def _region_table(grid, keys_sec, keys_rb, values, cls_idx, classes, valid, reducer):
    """{class: {region id: reducer(values in it)}} over the instances where valid."""
    out = {}
    for cname in classes:
        cm = valid & np.isin(cls_idx, _cls_members(cname))
        if not cm.any():
            continue
        row = {}
        for s, r, rid, _ in grid.regions():
            m = cm & (keys_sec == s) & (keys_rb == r)
            if m.any():
                row[rid] = reducer(values[m] if values is not None else None, m)
        if row:
            out[cname] = row
    return out


# =========================================================================== detection on one build
class _DetRunner:
    """The detector on a build's remapped images, matched to the remapped ground truth; per class × region counts of
    the rig (any camera) and of each camera."""

    def __init__(self, scene: Scene, det_fn, st: RobustSettings, classes, cache_images=True):
        self.scene = scene
        self.det_fn = det_fn
        self.st = st
        self.classes = classes
        self.cache = {} if cache_images else None
        self.rays = {}

    def _frame(self, ci, t):
        key = (ci, t)
        if self.cache is not None and key in self.cache:
            fr, img, rng, inst = self.cache[key]
            return fr, img, (rng.astype(np.float32) if rng is not None else None), inst
        fr = self.scene.frames[ci].get(t)
        if fr is None:
            return None
        try:
            img, rng, inst = DE._read_frame(self.scene.rec, self.scene.cams[ci], fr, True)
        except OSError:
            return None
        if rng is not None:
            rng = rng.astype(np.float16)         # metres to < 0.13 m at 256 m: only ranges false detections
        out = (fr, img, rng, inst)
        if self.cache is not None:
            self.cache[key] = out
        return fr, img, (rng.astype(np.float32) if rng is not None else None), inst

    def run(self, poses, keep_frames=False, cancel=None) -> dict:
        sc = self.scene
        grid = sc.grid
        dst = sc.dst
        bev = {} if self.st.bev3d else None                     # slot -> {'gt': {key: (xy, class, ignore)}, 'dets'}
        # rig objects: (slot, key, class index e) -> [region, any detected]
        rig = {}
        fp = {}                                                 # (e, region id) -> count
        missing = {}
        frames_out = {}
        for ci, c in enumerate(sc.cams):
            if ci not in self.rays:
                self.rays[ci] = grid_rays(c)
            true, used = poses[ci]
            R_rel = c.R.T @ true.R
            view, miss = rotation_view(c, R_rel, self.rays[ci], name=sc.tags[ci])
            missing[sc.tags[ci]] = float(miss.mean())
            for t in sc.det_slots:
                if cancel is not None and cancel():
                    raise _Cancelled()
                got = self._frame(ci, t)
                if got is None:
                    continue
                fr, img, rng, inst = got
                g = DE.gt_objects(fr, dst)
                vimg = remap_image(img, view)
                vrng = remap_image(rng, view, nearest=True) if rng is not None else None
                gv = DE.view_gt(g, inst, view, dst) if inst is not None else DE.gt_objects({'objects': []}, dst)
                dets = self.det_fn(vimg)
                dets = {k: dets[k] for k in ('boxes', 'scores', 'coco', 'cls') if k in dets}
                if bev is not None:
                    _bev3d_collect(bev.setdefault(t, {'gt': {}, 'dets': []}), c, used, sc.tags[ci], g, dets)
                ev = DE.evaluate_image(gv, dets, vrng, dst, self.classes)
                for e, i, s in ev['gt']:
                    if gv['ignore'][i] != DE.IGN_NONE or not np.all(np.isfinite(gv['xy'][i])):
                        continue
                    sec, rb = grid.assign(gv['xy'][i][0], gv['xy'][i][1])
                    k = (t, gv['key'][i], e)
                    cur = rig.get(k)
                    hit = s >= self.st.score
                    if cur is None:
                        rig[k] = [(int(sec), int(rb)), hit, {sc.tags[ci]: hit}]
                    else:
                        cur[1] = cur[1] or hit
                        cur[2][sc.tags[ci]] = hit
                if len(ev['det']):
                    boxes = np.asarray(dets['boxes'], float).reshape(-1, 4)
                    for e, i, s, status, r, _ in ev['det']:
                        if status != 0 or s < self.st.score:
                            continue
                        rid = 'unknown'
                        if math.isfinite(r):
                            b = boxes[i]
                            d, ok = DE.kb_unproject(c, np.array([(b[0] + b[2]) / 2]), np.array([(b[1] + b[3]) / 2]))
                            if ok[0]:
                                P = used.t + r * (d[0] @ used.R.T)
                                sec, rb = grid.assign(P[0], P[1])
                                if int(rb) >= 0:
                                    rid = _rid(grid, sec, rb)
                        fp[(e, rid)] = fp.get((e, rid), 0) + 1
                if keep_frames:
                    frames_out[(sc.tags[ci], t)] = {'boxes': np.asarray(dets.get('boxes', np.zeros((0, 4)))),
                                                    'scores': np.asarray(dets.get('scores', np.zeros(0))),
                                                    'cls': np.asarray(dets.get('cls', np.zeros(0)))}
        counts = {}
        for (t, key, e), (reg, hit, per) in rig.items():
            if reg[1] < 0:
                continue
            rid = _rid(grid, *reg)
            c = counts.setdefault((e, rid), [0, 0])
            c[0] += 1
            c[1] += int(hit)
        out = {'counts': counts, 'fp': fp, 'missing': missing, 'frames': frames_out}
        if bev is not None:
            out['bev3d'] = _bev3d_metrics(bev, grid)
        return out


# ---- the 3D metric of a build (core.rig_perception's matching and AP on the lifted, fused detections)
def _bev3d_collect(slot: dict, c, used, tag: str, g: dict, dets: dict):
    """One camera's image of a slot: its ground-truth objects (by key, in the vehicle frame) and its 2D detections
    lifted to 3D through the calibration the perception uses (bev_backends.lift_boxes)."""
    from . import bev_backends as BB
    for i, key in enumerate(g['key']):
        ci = int(g['cls'][i])
        if ci < 0 or DT.CLASSES[ci] not in BB.BEV_CLASSES or not np.all(np.isfinite(g['xy'][i])) or key in slot['gt']:
            continue
        slot['gt'][key] = (np.asarray(g['xy'][i], float), DT.CLASSES[ci], bool(g['ignore'][i] != DE.IGN_NONE))
    boxes = np.asarray(dets.get('boxes', np.zeros((0, 4))), float).reshape(-1, 4)
    if not len(boxes) or c.K is None:
        return
    sc = np.asarray(dets.get('scores', np.zeros(len(boxes))), float)
    names = [DT.CLASSES[int(k)] if 0 <= int(k) < len(DT.CLASSES) else '' for k in np.asarray(dets.get('cls', []))]
    kb = c.lens == 'kb' and c.D is not None
    cal = BB.CameraCal(tag=tag, width=int(c.width), height=int(c.height), R=np.asarray(used.R, float),
                       t=np.asarray(used.t, float), model='kb' if kb else 'pinhole', K=np.asarray(c.K, float),
                       D=np.asarray(c.D, float) if kb else None,
                       max_theta=float(c.max_theta) if c.max_theta else math.radians(89.0))
    slot['dets'] += BB.lift_boxes(boxes, names, sc, cal)


def _bev3d_metrics(bev: dict, grid) -> dict:
    """{class: {region id | 'all': {n_gt, tp, fp, fn, recall, precision, ap, ate_m}}} of the build's slots: the cameras'
    detections fused per slot and matched to the slot's objects (rig_perception.evaluate_frames; no bootstrap here,
    the band over builds is the spread)."""
    from . import bev_backends as BB
    from . import rig_perception as RPC
    items = []
    for t in sorted(bev):
        gt = bev[t]['gt']
        keys = list(gt)
        items.append(({'xy': np.array([gt[k][0] for k in keys], float).reshape(-1, 2), 'cls': [gt[k][1] for k in keys],
                       'ignore': np.array([gt[k][2] for k in keys], bool)},
                      BB.fuse(bev[t]['dets'], BB.LiftSettings())))
    return RPC.evaluate_frames(items, grid, RPC.RigPerceptionSettings(bootstrap=0))


def _bev3d_table(nominal: dict, builds: list) -> dict:
    """Per class × region: the nominal 3D recall, AP and centre error and their bands over the builds."""
    out = {}
    for cname, cells in (nominal or {}).items():
        row = {}
        for rid, nc in cells.items():
            if not nc.get('n_gt'):
                continue
            pick = lambda k: [((b.get(cname) or {}).get(rid) or {}).get(k) for b in builds]   # noqa: E731
            row[rid] = {'n_gt': nc['n_gt'], 'nominal': nc.get('recall'), 'nominal_ap': nc.get('ap'),
                        'nominal_ate_m': nc.get('ate_m'),
                        'recall': _band([x if x is not None else math.nan for x in pick('recall')]),
                        'ap': _band([x if x is not None else math.nan for x in pick('ap')]),
                        'ate_m': _band([x if x is not None else math.nan for x in pick('ate_m')])}
        if row:
            out[cname] = row
    return out


def _recall_doc(counts, fp, classes, grid) -> dict:
    """{class: {region id: {n_gt, tp, fn, fp, recall, precision}}} plus 'all' per class."""
    out = {}
    for e_name in classes:
        e = DT.EVAL_CLASSES.index(e_name)
        row = {}
        for _, _, rid, _ in grid.regions():
            n, tp = counts.get((e, rid), (0, 0))
            f = fp.get((e, rid), 0)
            if n or f:
                row[rid] = {'n_gt': n, 'tp': tp, 'fn': n - tp, 'fp': f, 'recall': tp / n if n else None,
                            'precision': tp / (tp + f) if tp + f else None}
        n = sum(v['n_gt'] for v in row.values())
        tp = sum(v['tp'] for v in row.values())
        f = sum(fp.get((e, k), 0) for k in [r for _, _, r, _ in grid.regions()] + ['unknown'])
        if n or f:
            row['all'] = {'n_gt': n, 'tp': tp, 'fn': n - tp, 'fp': f, 'recall': tp / n if n else None,
                          'precision': tp / (tp + f) if tp + f else None}
            out[e_name] = row
    return out


# =========================================================================== the study
@dataclass
class RecordingResult:
    label: str
    path: str
    info: dict
    tags: list
    z_subpx_m: dict           # per camera: the range beyond which the build's largest translation moves < 1 px
    nominal: dict             # 'bev': {class: {region: {n, ate_m}}}, 'ghost', 'det' (recall doc)
    bev: dict                 # per class × region: error stats over builds
    ghost: dict
    det: dict                 # per class × region: recall stats over builds
    sobol: dict
    oat: dict
    worst: dict
    missing: dict
    samples: list             # per build: summary
    seconds: float = 0.0
    bev3d: dict = field(default_factory=dict)   # per class × region: the 3D metric (rig_perception) over builds


@dataclass
class RobustResult:
    settings: RobustSettings
    variant: str
    grid: RegionGrid
    sources: dict
    recordings: list
    detector: dict
    records: list
    notes: list
    seconds: float = 0.0
    cancelled: bool = False
    finished_at: float = 0.0
    _scenes: dict = field(default_factory=dict, repr=False)   # recording index -> Scene (not exported)
    _builds: dict = field(default_factory=dict, repr=False)   # (recording, sample) -> poses


def _pct(a, q):
    a = np.asarray(a, float)
    a = a[np.isfinite(a)]
    return float(np.percentile(a, q)) if len(a) else None


def _band(a) -> dict:
    a = np.asarray(a, float)
    f = a[np.isfinite(a)]
    if not len(f):
        return {'mean': None, 'p05': None, 'p95': None, 'n_builds': 0}
    return {'mean': float(f.mean()), 'p05': float(np.percentile(f, 5)), 'p95': float(np.percentile(f, 95)),
            'sd': float(f.std(ddof=1)) if len(f) > 1 else 0.0, 'n_builds': int(len(f))}


def run(project=None, engine=None, settings: RobustSettings | None = None, variant=None, progress=None, cancel=None,
        detector=None, studies=None) -> RobustResult | None:
    """The study on the settings' recordings (None when cancelled). detector: a callable image -> detections (tests,
    or one already loaded); by default the project's detection model (FileNotFoundError when it is not downloaded)."""
    t0 = time.perf_counter()
    st = settings or settings_of(project)
    if not st.recordings:
        raise ValueError('Choose a recording with ground truth (a CARLA run with ground truth on).')
    stop = cancel or (lambda: False)

    def prog(f, text=''):
        if progress is not None:
            try:
                progress(min(max(float(f), 0.0), 1.0), text)
            except TypeError:
                progress(min(max(float(f), 0.0), 1.0))
    v = _variant_of(project, variant)
    grid = default_grid(project, v) if project is not None else RegionGrid()
    dst = DE.settings_of(project) if project is not None else DE.DetectionSettings()
    dst.score, dst.iou = st.score, st.iou
    recs = [DE.open_recording(p) for p in st.recordings]
    tags_all = sorted({t for r in recs for t in r.cameras})
    prog(0.01, 'The sources')
    sources = resolve_sources(project, st, tags_all, v, engine, studies)
    notes = []
    for s in sources.values():
        notes += [f'{SOURCE_LABEL[s.key]}: {n}' for n in s.notes]
    det_fn, det_info = None, {}
    if st.detect:
        if detector is None:
            det = DT.load(dst.detector())
            det_fn, det_info = det.detect, det.describe()
        else:
            det_fn = detector
            det_info = dict(getattr(detector, 'describe', lambda: {})() or {'model': 'custom', 'name': 'custom'})
    classes = [c for c in CLASSES]
    res = RobustResult(st, getattr(v, 'name', '') or (recs[0].variant if recs else ''), grid, sources, [], det_info,
                       [], notes)
    n_rec = len(recs)
    try:
        for ri, rec in enumerate(recs):
            base = ri / n_rec
            span = 1.0 / n_rec

            def sub(a, b, text, _base=base, _span=span):
                return lambda f, *_: prog(_base + _span * (a + (b - a) * f), text)
            t_rec = time.perf_counter()
            prog(base, f'{rec.label}: reading the ground truth')
            scene = Scene(rec, st, dst, grid, stop, sub(0.0, 0.08, f'{rec.label}: reading the ground truth'))
            res._scenes[ri] = scene
            notes += [f'{rec.label}: {n}' for n in scene.notes]
            missing_tags = [t for t in scene.tags if t not in sources['mounting'].per_cam]
            if missing_tags:
                raise ValueError(f'{rec.label}: cameras {missing_tags} have no source figures')
            em = ErrorModel(sources, scene.tags, st)
            cg = (sources['vibration'].shared.get('cg_x', 0.0), sources['vibration'].shared.get('cg_z', 0.0))
            nom_poses = scene.nominal_poses()
            nom = scene.bev(nom_poses)
            scene.nominal = nom
            rr = _evaluate_recording(scene, em, cg, nom, nom_poses, st, det_fn, classes, stop,
                                     sub(0.08, 1.0, rec.label), res, ri)
            rr.seconds = time.perf_counter() - t_rec
            res.recordings.append(rr)
            if st.detect and rr.nominal.get('det'):
                res.records.append(perception_record(project, v, scene, rr.nominal['det'], rr.nominal['bev'],
                                                     det_info, dst, grid))
                if st.records_per_sample:
                    for k, smp in enumerate(rr.samples):
                        if smp.get('det') is not None:
                            r = perception_record(project, v, scene, smp['det'], None, det_info, dst, grid)
                            r['build_variation'] = {'sample': k, 'mode': st.mode, 'sources': st.active(),
                                                    'flag': 'one Monte Carlo build of the tolerance study'}
                            r['id'] = _record_id(r)
                            res.records.append(r)
    except _Cancelled:
        return None
    res.seconds = time.perf_counter() - t0
    res.finished_at = time.time()
    prog(1.0, 'Done')
    return res


def _evaluate_recording(scene: Scene, em: ErrorModel, cg, nom, nom_poses, st, det_fn, classes, stop, prog, res,
                        ri) -> RecordingResult:
    grid = scene.grid
    n = scene.n
    act = st.active()
    # ---- nominal tables
    base = nom['ok']
    ate = np.linalg.norm(nom['P'] - scene.i_xy, axis=1)
    nom_bev = _region_table(grid, scene.i_sec, scene.i_rb, ate, scene.i_cls, classes, base & (scene.i_rb >= 0),
                            lambda x, m: {'n': int(m.sum()), 'ate_m': float(np.mean(np.minimum(x, ERR_CLIP_M)))})
    multi0, ghost0 = scene.ghosts(nom['P'], nom['ok'], st)
    # ---- BEV builds (Sobol base samples double as the Monte Carlo builds)
    pers_err, pers_oov, pers_ghost, smp = [], [], [], []
    by_src = {s: em.cols(s) for s in act}
    by_cam = {t: em.cols(cam=ci) for ci, t in enumerate(scene.tags)}
    shared = em.shared_cols()
    if any(c in shared for s in act for c in em.cols(s)):
        by_cam['shared (hour, body)'] = [c for c in shared if any(c in em.cols(s) for s in act)]
    n_s = max(int(st.sobol_n), 1)
    keys = list(SOBOL_METRICS)

    def build_eval(U, store=False):
        out = np.zeros((len(U), len(keys)))
        for k, u in enumerate(U):
            errs = em.errors(u)
            poses = scene.poses(errs, cg)
            b = scene.bev(poses)
            m = bev_metrics(scene, nom, b, st)
            out[k] = [m[x] for x in keys]
            if store:
                e = np.linalg.norm(b['P'] - nom['P'], axis=1)
                e = np.where(base & b['ok'], e, np.nan)
                pers_err.append(e.astype(np.float32))
                pers_oov.append(b['oov'] & base)
                mv, gh = scene.ghosts(b['P'], b['ok'], st)
                pers_ghost.append((mv, gh))
                smp.append({'errors': _errs_doc(errs, scene.tags), 'bev': {x: _r(m[x], 4) for x in keys}})
        return out
    sob = {}
    if n and act and st.sobol_n > 0:
        prog(0.0, 'BEV builds and Sobol indices (by source)')
        rows = {}

        def f_store(M):
            # the first two evaluations are A and B: keep their builds' details for the bands
            store = len(rows) < 2
            rows[len(rows)] = True
            return build_eval(M, store)
        so = sobol_indices(f_store, em.D, by_src, n_s, st.seed, st.boot, progress=lambda f: prog(0.3 * f),
                           cancel=stop)
        sob['by_source'] = _sobol_doc(so, keys)
        A_rows = so['A']
        prog(0.3, 'Sobol indices (by camera)')
        so2 = sobol_indices(lambda M: build_eval(M, False), em.D, by_cam, max(n_s // 2, 8), st.seed + 7,
                            st.boot, progress=lambda f: prog(0.3 + 0.15 * f), cancel=stop)
        sob['by_camera'] = _sobol_doc(so2, keys)
    else:
        rng = np.random.default_rng(st.seed)
        A_rows = rng.random((max(st.samples, 2), em.D))
        if n:
            build_eval(A_rows, True)
    # ---- bands over builds
    E = np.vstack(pers_err) if pers_err else np.zeros((0, n))
    O = np.vstack(pers_oov) if pers_oov else np.zeros((0, n), bool)
    bev_tab = {}
    for cname in classes:
        cm = base & np.isin(scene.i_cls, _cls_members(cname)) & (scene.i_rb >= 0)
        if not cm.any():
            continue
        row = {}
        for s, r, rid, _ in grid.regions():
            m = cm & (scene.i_sec == s) & (scene.i_rb == r)
            if not m.any():
                continue
            sub_e = np.minimum(E[:, m], ERR_CLIP_M)
            with np.errstate(invalid='ignore'), warnings.catch_warnings():
                warnings.simplefilter('ignore', RuntimeWarning)      # a region whose objects all left the view
                per_build = np.nanmean(np.where(np.isfinite(sub_e), sub_e, np.nan), axis=1) if len(E) else []
            pooled = sub_e[np.isfinite(sub_e)]
            row[rid] = {'n': int(m.sum()), 'err_m': _band(per_build),
                        'err_p50_m': _pct(pooled, 50), 'err_p95_m': _pct(pooled, 95),
                        'oov': _band(O[:, m].mean(axis=1)) if len(O) else _band([]),
                        'lost': _band(np.mean(~np.isfinite(E[:, m]) & ~O[:, m], axis=1)) if len(E) else _band([])}
        if row:
            bev_tab[cname] = row
    ghost_tab = {}
    if len(scene.groups):
        for cname in ('all',) + tuple(classes):
            gm0 = np.ones(len(scene.groups), bool) if cname == 'all' else np.isin(scene.g_cls, _cls_members(cname))
            if not gm0.any():
                continue
            rows_ = {}
            for s, r, rid, _ in [(None, None, 'all', '')] + grid.regions():
                gm = gm0 if s is None else gm0 & (scene.g_sec == s) & (scene.g_rb == r)
                if not (gm & multi0).any():
                    continue
                per = [float(gh[gm & mv].mean()) if (gm & mv).any() else math.nan for mv, gh in pers_ghost]
                rows_[rid] = {'n': int((gm & multi0).sum()), 'nominal': float(ghost0[gm & multi0].mean()),
                              'rate': _band(per)}
            if rows_:
                ghost_tab[cname] = rows_
    # ---- detection builds
    det_tab, oat, worst, missing, nominal_det, bev3d_tab = {}, {}, {}, {}, None, {}
    if det_fn is not None and scene.det_slots:
        runner = _DetRunner(scene, det_fn, st, classes)
        n_det = int(st.samples)
        total = 1 + n_det + (len(act) * int(st.oat_samples) if len(act) > 1 else 0)
        done = [0]

        def dprog(text):
            done[0] += 1
            prog(0.45 + 0.55 * done[0] / max(total, 1), text)
        nd = runner.run(nom_poses, keep_frames=True, cancel=stop)
        nominal_det = _recall_doc(nd['counts'], nd['fp'], classes, grid)
        nominal_bev3d = nd.get('bev3d')
        res._builds[(ri, 'nominal')] = {'poses': nom_poses, 'frames': nd['frames']}
        dprog('Detection: the nominal build')
        per_build = []
        miss_acc = {t: [] for t in scene.tags}
        for k in range(n_det):
            u = A_rows[k % len(A_rows)]
            errs = em.errors(u)
            poses = scene.poses(errs, cg)
            out = runner.run(poses, keep_frames=False, cancel=stop)
            doc = _recall_doc(out['counts'], out['fp'], classes, grid)
            per_build.append((doc, errs, poses, out))
            for t, m in out['missing'].items():
                miss_acc[t].append(m - nd['missing'].get(t, 0.0))     # beyond the nominal (the lens circle's)
            if k < len(smp):
                smp[k]['det'] = doc
            else:
                smp.append({'errors': _errs_doc(errs, scene.tags), 'det': doc})
            dprog(f'Detection: build {k + 1} of {n_det}')
        missing = {t: _band(v) | {'nominal': nd['missing'].get(t)} for t, v in miss_acc.items()}
        if nominal_bev3d is not None:
            bev3d_tab = _bev3d_table(nominal_bev3d, [o.get('bev3d') or {} for *_, o in per_build])
        for cname in classes:
            reg_ids = ['all'] + [rid for _, _, rid, _ in grid.regions()]
            row = {}
            for rid in reg_ids:
                nom_c = (nominal_det.get(cname) or {}).get(rid)
                if not nom_c or not nom_c['n_gt']:
                    continue
                rec_b = [((d.get(cname) or {}).get(rid) or {}).get('recall') for d, *_ in per_build]
                prec_b = [((d.get(cname) or {}).get(rid) or {}).get('precision') for d, *_ in per_build]
                row[rid] = {'n_gt': nom_c['n_gt'], 'nominal': nom_c['recall'],
                            'nominal_ci': _wilson(nom_c['tp'], nom_c['n_gt']),
                            'recall': _band([x if x is not None else math.nan for x in rec_b]),
                            'precision': _band([x if x is not None else math.nan for x in prec_b]),
                            'nominal_precision': nom_c['precision']}
            if row:
                det_tab[cname] = row
        # the worst build: the lowest rig recall over every class (the 'vehicle' group and the people)
        if per_build:
            def score_of(d):
                tot_n = sum(((d.get(c) or {}).get('all') or {}).get('n_gt', 0) for c in ('pedestrian', 'cyclist',
                                                                                        'motorcyclist', 'vehicle'))
                tot_tp = sum(((d.get(c) or {}).get('all') or {}).get('tp', 0) for c in ('pedestrian', 'cyclist',
                                                                                      'motorcyclist', 'vehicle'))
                return tot_tp / tot_n if tot_n else 1.0
            ks = [score_of(d) for d, *_ in per_build]
            kw = int(np.argmin(ks))
            d, errs, poses, out = per_build[kw]
            wout = runner.run(poses, keep_frames=True, cancel=stop)
            res._builds[(ri, 'worst')] = {'poses': poses, 'frames': wout['frames']}
            worst = {'sample': kw, 'recall': ks[kw], 'nominal_recall': score_of(nominal_det),
                     'errors': _errs_doc(errs, scene.tags), 'missing': wout['missing'],
                     'text': (f'Build {kw + 1}: rig recall {ks[kw]:.3f} (nominal {score_of(nominal_det):.3f}), '
                              f'cameras turned up to {max(_image_turn(errs, t) for t in scene.tags):.3f}° '
                              f'({max(scene.tags, key=lambda t: _image_turn(errs, t))}); calibration off by up '
                              f'to {max(_max_turn(errs, t) for t in scene.tags):.3f}° '
                              f'({max(scene.tags, key=lambda t: _max_turn(errs, t))})')}
        # one source at a time
        if len(act) > 1 and st.oat_samples > 0:
            for s in act:
                vals = {c: [] for c in ('all',) + tuple(classes)}
                for k in range(int(st.oat_samples)):
                    u = A_rows[(k + 3 * len(oat)) % len(A_rows)]
                    errs = em.errors(u, only=[s])
                    out = runner.run(scene.poses(errs, cg), cancel=stop)
                    doc = _recall_doc(out['counts'], out['fp'], classes, grid)
                    tn = sum(((doc.get(c) or {}).get('all') or {}).get('n_gt', 0) for c in ('pedestrian', 'cyclist',
                                                                                          'motorcyclist', 'vehicle'))
                    tt = sum(((doc.get(c) or {}).get('all') or {}).get('tp', 0) for c in ('pedestrian', 'cyclist',
                                                                                        'motorcyclist', 'vehicle'))
                    vals['all'].append(tt / tn if tn else math.nan)
                    for c in classes:
                        a = (doc.get(c) or {}).get('all') or {}
                        vals[c].append(a.get('recall') if a.get('recall') is not None else math.nan)
                    dprog(f'Detection: {SOURCE_LABEL[s]} alone, build {k + 1}')
                oat[s] = {c: {'mean': _nanmean(v), 'var': _nanvar(v), 'n': len(v)} for c, v in vals.items()}
            for c in ('all',) + tuple(classes):
                tot = sum((oat[s][c]['var'] or 0.0) for s in act)
                for s in act:
                    vv = oat[s][c]['var']
                    oat[s][c]['share'] = (vv / tot) if (vv is not None and tot > 0) else None
    # ---- translations: the range beyond which they move content < 1 px
    dmax = 0.0
    for e in smp:
        for t, x in (e.get('errors') or {}).items():
            if t == 'body':
                continue
            dmax = max(dmax, math.hypot(*x['true_mm']) / 1000.0)
    z_sub = {t: _r(max_px_per_rad(c) * dmax, 3) for t, c in zip(scene.tags, scene.cams)}
    return RecordingResult(scene.rec.label, scene.rec.path, scene.info(), list(scene.tags),
                           {'largest_translation_mm': _r(dmax * 1000, 3), 'per_camera_m': z_sub},
                           {'bev': nom_bev, 'ghost': (float(ghost0[multi0].mean()) if multi0.any() else None),
                            'det': nominal_det, 'lift_ok': float(base.mean()) if n else None},
                           bev_tab, ghost_tab, det_tab, sob, oat, worst, missing, smp, bev3d=bev3d_tab)


def _image_turn(errs, t):
    """The largest Euler change of a camera from its recorded pose (what turns its image)."""
    return float(np.abs(np.asarray(errs[t]['true'][0])).max())


def _max_turn(errs, t):
    """The largest Euler error the perception's calibration does not know (true − used)."""
    return float(np.abs(np.asarray(errs[t]['true'][0]) - np.asarray(errs[t]['used'][0])).max())


def _errs_doc(errs, tags) -> dict:
    out = {}
    for t in tags:
        (da, dp), (ua, up) = errs[t]['true'], errs[t]['used']
        out[t] = {'true_deg': [_r(x, 5) for x in da], 'true_mm': [_r(x * 1000, 4) for x in dp],
                  'used_deg': [_r(x, 5) for x in ua], 'used_mm': [_r(x * 1000, 4) for x in up]}
    out['body'] = {'pitch_deg': _r(errs['body'][0], 5), 'heave_mm': _r(errs['body'][1] * 1000, 4)}
    return out


def _nanmean(v):
    a = np.asarray(v, float)
    a = a[np.isfinite(a)]
    return float(a.mean()) if len(a) else None


def _nanvar(v):
    a = np.asarray(v, float)
    a = a[np.isfinite(a)]
    return float(a.var(ddof=1)) if len(a) > 1 else None


def _wilson(k, n):
    try:
        from .simcampaign import wilson
        w = wilson(int(k), int(n))
        return [round(w[0], 4), round(w[1], 4)] if w else None
    except Exception:                                            # noqa: BLE001
        return None


def _sobol_doc(so, keys) -> dict:
    out = {'n': so['n'], 'evaluations': so['evaluations'], 'metrics': list(keys),
           'variance': [_r(x, 6) for x in so['V']], 'mean': [_r(x, 6) for x in so['mean']], 'groups': {}}
    for g, r in so['groups'].items():
        out['groups'][g] = {'S': [_r(x, 4) for x in r['S']], 'ST': [_r(x, 4) for x in r['ST']],
                            'S_ci': None if r['S_ci'] is None else [[_r(a, 4), _r(b, 4)] for a, b in r['S_ci']],
                            'ST_ci': None if r['ST_ci'] is None else [[_r(a, 4), _r(b, 4)] for a, b in r['ST_ci']]}
    return out


# =========================================================================== the perception record
def _canon(o) -> bytes:
    return json.dumps(o, sort_keys=True, separators=(',', ':'), ensure_ascii=False, default=str).encode('utf-8')


def _record_id(doc) -> str:
    """perf_model.record_id's rule: core.provenance.canonical without id, created and provenance."""
    from . import provenance as PV
    return hashlib.sha256(PV.canonical({k: v for k, v in doc.items()
                                        if k not in ('id', 'created', 'provenance')})).hexdigest()


def _rig_block(project, v, scene: Scene) -> dict:
    """The rig as the record states it: the recording's cameras (pose, image, field) with the project's module and
    lens names where the project has the camera."""
    import datetime  # noqa: F401  (kept local: the record is built rarely)
    run = scene.rec.run or {}
    veh = (run.get('vehicle') or {}).get('project') or {}
    vehicle = {'name': getattr(getattr(project, 'vehicle', None), 'name', '') or run.get('project', ''),
               'length_m': veh.get('length'), 'width_m': veh.get('width'), 'height_m': veh.get('height')}
    pc = {c.tag: c for c in (v.cameras if v is not None else [])}
    cams = []
    for t, c, (yaw, pitch, roll) in zip(scene.tags, scene.cams, scene.euler):
        lim = DE._theta_limit(c)
        hf = _half_fov(c, 'h', lim)
        vf = _half_fov(c, 'v', lim)
        p = pc.get(t)
        cams.append({'tag': t, 'module': getattr(p, 'module_id', ''), 'lens': getattr(p, 'lens_id', ''),
                     'width': c.width, 'height': c.height, 'x': _r(c.t[0], 4), 'y': _r(c.t[1], 4),
                     'z': _r(c.t[2], 4), 'yaw': _r(yaw, 4), 'pitch': _r(pitch, 4), 'roll': _r(roll, 4),
                     'hfov_deg': _r(2 * hf, 3), 'vfov_deg': _r(2 * vf, 3), 'max_theta_deg': _r(math.degrees(lim), 3),
                     'projection': 'kannala_brandt' if c.lens == 'kb' else c.lens})
    sensors = []
    if v is not None:
        for s in v.sensors:
            sensors.append({'tag': s.tag, 'kind': s.kind, 'model': getattr(s, 'model_id', ''), 'x': _r(s.x, 4),
                            'y': _r(s.y, 4), 'z': _r(s.z, 4), 'yaw': _r(s.yaw, 4), 'pitch': _r(s.pitch, 4),
                            'roll': _r(s.roll, 4)})
    return {'vehicle': vehicle, 'cameras': cams, 'sensors': sensors}


def _half_fov(c: DE.Camera, axis, lim) -> float:
    """Half field along the image's horizontal or vertical centre line (deg), within the lens's field."""
    if axis == 'h':
        u = np.array([0.0, c.width - 1.0])
        vv = np.array([c.K[1, 2], c.K[1, 2]])
    else:
        u = np.array([c.K[0, 2], c.K[0, 2]])
        vv = np.array([0.0, c.height - 1.0])
    d, ok = DE.kb_unproject(c, u, vv)
    th = np.arccos(np.clip(d[:, 2], -1, 1))
    th = np.where(ok, th, lim)
    return float(math.degrees(min(float(th.max()), lim)))


def perception_record(project, v, scene: Scene, det_doc: dict, bev_doc: dict | None, det_info, dst, grid) -> dict:
    """A vantage.perception_record/1 of one build on one recording (the metrics per class × region)."""
    import datetime
    from .. import __version__
    rec = scene.rec
    sc = (rec.run or {}).get('scene') or {}
    scene_doc = {'kind': 'carla', 'town': rec.town, 'scenario': rec.scenario, 'seed': rec.seed, 'weather': rec.weather,
                 'duration_s': rec.duration_s, 'frames': len(scene.det_slots) * len(scene.tags)}
    scene_doc['scene_hash'] = hashlib.sha256(_canon({k: scene_doc[k] for k in ('town', 'scenario', 'seed', 'weather',
                                                                               'duration_s')} |
                                                    {'traffic': sc.get('traffic')})).hexdigest()
    rig = _rig_block(project, v, scene)
    dset = dst.detector().to_dict() | {'score': dst.score, 'iou': dst.iou, 'iou_small': dst.iou_small,
                                       'min_px': dst.min_px}
    metrics = {}
    for cname, row in (det_doc or {}).items():
        out = {}
        for rid, m in row.items():
            if rid == 'all':
                continue
            b = ((bev_doc or {}).get(cname) or {}).get(rid) or {}
            out[rid] = {'n_gt': m['n_gt'], 'tp': m['tp'], 'fp': m['fp'], 'fn': m['fn'],
                        'recall': _r(m['recall'], 5), 'precision': _r(m['precision'], 5), 'ap': None,
                        'ate_m': _r(b.get('ate_m'), 4) if b else None}
        if out:
            metrics[cname] = out
    prov = None
    if project is not None:
        try:
            from . import provenance as PV
            prov = PV.block(project, None, variant=v, study='robust_perception')
        except Exception:                                        # noqa: BLE001 — provenance is best effort here
            prov = None
    doc = {'schema': RECORD_SCHEMA, 'created': datetime.datetime.now(datetime.timezone.utc).isoformat(
        timespec='seconds'), 'vantage': __version__, 'source': 'carla', 'scene': scene_doc,
           'detector': {'id': det_info.get('model', 'custom'), 'backend': 'lifted',
                        'version': det_info.get('sha256', det_info.get('model', '')),
                        'settings_hash': hashlib.sha256(_canon(dset)).hexdigest()},
           'rig': rig, 'rig_hash': hashlib.sha256(_canon(rig)).hexdigest(), 'grid': grid.to_dict(),
           'metrics': metrics, 'provenance': prov}
    doc['id'] = _record_id(doc)
    return {'schema': doc.pop('schema'), 'id': doc.pop('id')} | doc


def store_records(res: RobustResult) -> dict:
    """Hand the records to the learned model's store (core.perf_model.add_record) when it exists; else they stay in
    the export (records.json). Returns {'stored', 'where'}."""
    try:
        from . import perf_model as PM
        add = getattr(PM, 'add_record')
    except Exception:                                            # noqa: BLE001 — perfmodel not installed yet
        return {'stored': 0, 'where': 'records.json in the export (core.perf_model is not available)'}
    n = 0
    for r in res.records:
        try:
            add(r)
            n += 1
        except Exception:                                        # noqa: BLE001
            pass
    return {'stored': n, 'where': 'core.perf_model\'s store'}


# =========================================================================== images of a build (the page)
def build_image(res: RobustResult, rec: int = 0, which='worst', tag=None, k: int = 0):
    """A build's image of one camera at its k-th detector instant: (BGR image with the missing area tinted, the
    ground-truth boxes in green and the detections in orange, text) or None."""
    import cv2
    sc = res._scenes.get(rec)
    b = res._builds.get((rec, which))
    if sc is None or b is None:
        return None
    tag = tag or (sc.tags[0] if sc.tags else None)
    if tag not in sc.tags or not sc.det_slots:
        return None
    ci = sc.tags.index(tag)
    t = sc.det_slots[min(max(int(k), 0), len(sc.det_slots) - 1)]
    fr = sc.frames[ci].get(t)
    if fr is None:
        return None
    c = sc.cams[ci]
    img, rng, inst = DE._read_frame(sc.rec, c, fr, True)
    true, used = b['poses'][ci]
    view, miss = rotation_view(c, c.R.T @ true.R, None, tag)
    vimg = remap_image(img, view)
    vimg[miss] = (vimg[miss] * 0.3 + np.array([60, 0, 90]) * 0.7).astype(np.uint8)
    g = DE.gt_objects(fr, sc.dst)
    gv = DE.view_gt(g, inst, view, sc.dst) if inst is not None else None
    if gv is not None:
        for i in range(len(gv['id'])):
            if gv['ignore'][i] != DE.IGN_NONE or gv['cls'][i] < 0:
                continue
            x0, y0, x1, y1 = (int(round(x)) for x in gv['box'][i])
            cv2.rectangle(vimg, (x0, y0), (x1 - 1, y1 - 1), (80, 220, 80), 1)
    d = (b.get('frames') or {}).get((tag, t))
    if d is not None:
        for bx, s in zip(np.asarray(d['boxes']).reshape(-1, 4), np.asarray(d['scores'])):
            if s >= res.settings.score:
                x0, y0, x1, y1 = (int(round(x)) for x in bx)
                cv2.rectangle(vimg, (x0, y0), (x1, y1), (0, 150, 255), 1)
    e = np.degrees(np.arccos(np.clip((np.trace(c.R.T @ true.R) - 1) / 2, -1, 1)))
    text = (f'{tag} · {which} build · turned {e:.3f}° from the recording · missing {miss.mean() * 100:.1f} % of the '
            f'image · instant {k + 1} of {len(sc.det_slots)}')
    return vimg, text


# =========================================================================== exports
def _r(v, nd=4):
    if v is None:
        return None
    try:
        f = float(v)
    except (TypeError, ValueError):
        return v
    return round(f, nd) if math.isfinite(f) else None


def _hh(h) -> str:
    h = float(h)
    return f'{int(h):02d}:{int(round((h % 1) * 60)):02d}'


def _clean(o):
    if isinstance(o, dict):
        return {str(k): _clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_clean(v) for v in o]
    if isinstance(o, np.ndarray):
        return _clean(o.tolist())
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (float, np.floating)):
        return _r(o, 5)
    return o


def headline(rr: RecordingResult, grid: RegionGrid, cls='pedestrian') -> list:
    """Plain sentences: per region with objects, recall over builds and BEV error (the strongest three)."""
    out = []
    det = rr.det.get(cls) or {}
    bev = rr.bev.get(cls) or {}
    regs = sorted((rid for rid in det if rid != 'all'), key=lambda r: -det[r]['n_gt'])[:3]
    for rid in regs:
        p = grid.parse_id(rid)
        name = grid.region_name(*p) if p else rid
        d = det[rid]
        rb = d['recall']
        txt = f'{cls} recall at {name}: {d["nominal"]:.2f} nominal'
        if rb['mean'] is not None:
            txt += f', {rb["mean"]:.2f} over builds (90 % of builds {rb["p05"]:.2f}–{rb["p95"]:.2f})'
        b = bev.get(rid)
        if b and b['err_m']['mean'] is not None:
            txt += f'; BEV error {b["err_m"]["mean"]:.2f} m (p95 {b["err_p95_m"]:.2f} m)'
        out.append(txt + f' [{d["n_gt"]} objects]')
    return out


def dominant(rr: RecordingResult, metric='err:all') -> dict | None:
    """The source and the camera with the largest first-order index of a metric."""
    out = {}
    for kind in ('by_source', 'by_camera'):
        so = rr.sobol.get(kind)
        if not so or metric not in so['metrics']:
            continue
        j = so['metrics'].index(metric)
        best = max(so['groups'].items(), key=lambda kv: (kv[1]['S'][j] if kv[1]['S'][j] is not None else -9))
        out[kind] = {'group': best[0], 'S': best[1]['S'][j], 'ST': best[1]['ST'][j]}
    return out or None


def summary(res: RobustResult) -> dict:
    st = res.settings
    out = {'schema': SCHEMA, 'variant': res.variant, 'mode': st.mode, 'sources': st.active(),
           'detector': res.detector.get('model'), 'recordings': [], 'seconds': _r(res.seconds, 1)}
    for rr in res.recordings:
        d = {'recording': rr.label, 'objects': rr.info['objects'], 'nominal_ghost_rate': _r(rr.nominal['ghost'], 4),
             'headline': headline(rr, res.grid), 'dominant': dominant(rr)}
        for c in ('pedestrian', 'vehicle'):
            a = (rr.det.get(c) or {}).get('all')
            if a:
                d[f'{c}_recall'] = {'nominal': _r(a['nominal'], 4), 'mean': _r(a['recall']['mean'], 4),
                                    'p05': _r(a['recall']['p05'], 4), 'p95': _r(a['recall']['p95'], 4)}
        for c in ('pedestrian', 'vehicle'):
            a3 = (rr.bev3d.get(c) or {}).get('all')
            if a3:
                d[f'{c}_bev3d'] = {'nominal_recall': _r(a3['nominal'], 4), 'recall_p05': _r(a3['recall']['p05'], 4),
                                   'recall_mean': _r(a3['recall']['mean'], 4), 'nominal_ap': _r(a3['nominal_ap'], 4),
                                   'nominal_ate_m': _r(a3['nominal_ate_m'], 3)}
        g = (rr.ghost.get('all') or {}).get('all')
        if g:
            d['ghost_rate'] = {'nominal': _r(g['nominal'], 4), 'mean': _r(g['rate']['mean'], 4),
                               'p95': _r(g['rate']['p95'], 4)}
        out['recordings'].append(d)
    return out


def to_doc(res: RobustResult, project=None) -> dict:
    st = res.settings
    doc = {'schema': SCHEMA, 'project': getattr(project, 'name', ''), 'variant': res.variant, 'frame': FRAME,
           'units': {'angle': 'deg (each camera\'s own yaw, pitch, roll change)', 'move': 'mm (vehicle frame)',
                     'bev_error': 'm on the ground', 'recall': 'share of objects', 'rate': 'share'},
           'settings': st.to_dict() | {'recordings': [os.path.basename(os.path.normpath(p)) for p in st.recordings]},
           'mode': {'id': st.mode, 'text': MODE_LABEL[st.mode]}, 'active_sources': st.active(),
           'sources': [s.to_doc() for s in res.sources.values()], 'grid': res.grid.to_dict(),
           'regions': {rid: name for _, _, rid, name in res.grid.regions()},
           'detector': {k: v for k, v in res.detector.items() if k != 'settings'},
           'method': __doc__.split('Statistics')[0].strip(), 'statistics': __doc__.split('Statistics')[1].strip(),
           'assumptions': ASSUMPTIONS, 'limits': LIMITS_TEXT, 'err_clip_m': ERR_CLIP_M, 'recordings': [],
           'notes': list(res.notes), 'seconds': _r(res.seconds, 2), 'cancelled': res.cancelled}
    for rr in res.recordings:
        doc['recordings'].append(_clean({
            'label': rr.label, 'folder': os.path.basename(os.path.normpath(rr.path)), 'info': rr.info,
            'translation_bound': rr.z_subpx_m, 'nominal': rr.nominal, 'bev': rr.bev, 'ghost': rr.ghost,
            'detection': rr.det, 'sobol': rr.sobol, 'one_at_a_time': rr.oat, 'worst_build': rr.worst,
            'missing_image_share': rr.missing, 'headline': headline(rr, res.grid), 'dominant': dominant(rr),
            'builds': rr.samples, 'seconds': rr.seconds,
            **({'bev3d': rr.bev3d, 'bev3d_method': BEV3D_METHOD} if rr.bev3d else {})}))
    return doc


BAND_COLUMNS = ('recording', 'class', 'region', 'region_name', 'n_gt', 'recall_nominal', 'recall_mean', 'recall_p05',
                'recall_p95', 'bev_n', 'ate_nominal_m', 'bev_err_mean_m', 'bev_err_p05_m', 'bev_err_p95_builds_m',
                'bev_err_p95_pooled_m', 'out_of_view_mean', 'ghost_nominal', 'ghost_mean', 'ghost_p95')


def to_csv(res: RobustResult) -> str:
    buf = io.StringIO()
    w = csv.writer(buf)
    w.writerow(BAND_COLUMNS)
    for rr in res.recordings:
        for cname in CLASSES:
            d, b, g = rr.det.get(cname) or {}, rr.bev.get(cname) or {}, rr.ghost.get(cname) or {}
            nb = (rr.nominal['bev'].get(cname) or {})
            rids = [r for _, _, r, _ in res.grid.regions() if r in d or r in b or r in g]
            for rid in rids:
                p = res.grid.parse_id(rid)
                dd, bb, gg = d.get(rid) or {}, b.get(rid) or {}, g.get(rid) or {}
                w.writerow([rr.label, cname, rid, res.grid.region_name(*p) if p else rid, dd.get('n_gt'),
                            _r(dd.get('nominal'), 4), _r((dd.get('recall') or {}).get('mean'), 4),
                            _r((dd.get('recall') or {}).get('p05'), 4), _r((dd.get('recall') or {}).get('p95'), 4),
                            bb.get('n'), _r((nb.get(rid) or {}).get('ate_m'), 3),
                            _r((bb.get('err_m') or {}).get('mean'), 4), _r((bb.get('err_m') or {}).get('p05'), 4),
                            _r((bb.get('err_m') or {}).get('p95'), 4), _r(bb.get('err_p95_m'), 4),
                            _r((bb.get('oov') or {}).get('mean'), 4), _r(gg.get('nominal'), 4),
                            _r((gg.get('rate') or {}).get('mean'), 4), _r((gg.get('rate') or {}).get('p95'), 4)])
    return buf.getvalue()


def sobol_csv(res: RobustResult) -> str:
    buf = io.StringIO()
    w = csv.writer(buf)
    w.writerow(('recording', 'by', 'group', 'metric', 'S', 'S_lo', 'S_hi', 'ST', 'ST_lo', 'ST_hi', 'n', 'evaluations'))
    for rr in res.recordings:
        for kind, so in rr.sobol.items():
            for g, r in so['groups'].items():
                for j, m in enumerate(so['metrics']):
                    sc, tc = (r['S_ci'] or [[None, None]] * len(so['metrics'])), (r['ST_ci'] or
                                                                                [[None, None]] * len(so['metrics']))
                    w.writerow([rr.label, kind, g, m, r['S'][j], sc[j][0], sc[j][1], r['ST'][j], tc[j][0], tc[j][1],
                                so['n'], so['evaluations']])
        for s, rows in rr.oat.items():
            for c, v in rows.items():
                w.writerow([rr.label, 'one at a time', s, f'recall:{c}', v.get('share'), None, None, None, None, None,
                            v.get('n'), v.get('n')])
    return buf.getvalue()


def png(res: RobustResult, rec=0, cls='pedestrian', size=(1100, 520)) -> bytes:
    """The recall band and the BEV error per region of one class as a plain chart (for the export)."""
    import cv2
    W, H = size
    img = np.full((H, W, 3), 255, np.uint8)
    if not res.recordings:
        return cv2.imencode('.png', img)[1].tobytes()
    rr = res.recordings[min(rec, len(res.recordings) - 1)]
    det = rr.det.get(cls) or {}
    bev = rr.bev.get(cls) or {}
    rids = [r for _, _, r, _ in res.grid.regions() if r in det or r in bev]
    ink = (40, 40, 40)
    cv2.putText(img, f'{rr.label} - {cls}: recall over builds (bars: 5-95 %), BEV error (m)', (12, 24),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, ink, 1, cv2.LINE_AA)
    if not rids:
        return cv2.imencode('.png', img)[1].tobytes()
    x0, x1, y0, y1 = 60, W - 20, 50, H - 90
    n = len(rids)
    bw = (x1 - x0) / n
    emax = max([((bev.get(r) or {}).get('err_m') or {}).get('p95') or 0.0 for r in rids] + [1e-3])
    for k, rid in enumerate(rids):
        cx = int(x0 + (k + 0.5) * bw)
        d = det.get(rid)
        if d and d['recall']['mean'] is not None:
            yy = lambda v: int(y1 - v * (y1 - y0))
            cv2.line(img, (cx - 4, yy(d['recall']['p05'])), (cx - 4, yy(d['recall']['p95'])), (200, 120, 40), 3)
            cv2.circle(img, (cx - 4, yy(d['recall']['mean'])), 4, (200, 120, 40), -1)
            cv2.circle(img, (cx - 4, yy(d['nominal'])), 4, ink, 1)
        b = bev.get(rid)
        if b and b['err_m']['mean'] is not None:
            h = int((y1 - y0) * min(b['err_m']['mean'] / emax, 1.0))
            cv2.rectangle(img, (cx + 2, y1 - h), (cx + 8, y1), (60, 160, 230), -1)
        cv2.putText(img, rid, (cx - 14, y1 + 16), cv2.FONT_HERSHEY_SIMPLEX, 0.35, ink, 1, cv2.LINE_AA)
    cv2.line(img, (x0, y1), (x1, y1), ink, 1)
    cv2.putText(img, f'recall 0-1 (orange: mean and 5-95 % of builds; ring: nominal); blue: mean BEV error, '
                     f'full height {emax:.2f} m', (12, H - 40), cv2.FONT_HERSHEY_SIMPLEX, 0.42, ink, 1, cv2.LINE_AA)
    return cv2.imencode('.png', img)[1].tobytes()


def files(res: RobustResult, project=None) -> list:
    doc = to_doc(res, project)
    out = [('robust_perception.json', json.dumps(doc, indent=1, ensure_ascii=False).encode('utf-8')),
           ('robust_perception_regions.csv', to_csv(res).encode('utf-8')),
           ('robust_perception_sensitivity.csv', sobol_csv(res).encode('utf-8'))]
    for ri, rr in enumerate(res.recordings):
        out.append((f'robust_perception_{ri + 1}.png', png(res, ri)))
    if res.records:
        out.append(('perception_records.json', json.dumps(res.records, indent=1, ensure_ascii=False).encode('utf-8')))
    return out


def export(res: RobustResult, project, out_dir) -> list[str]:
    os.makedirs(out_dir, exist_ok=True)
    paths = []
    for name, data in files(res, project):
        p = os.path.join(out_dir, name)
        with open(p, 'wb') as fh:
            fh.write(data)
        paths.append(p)
    return paths
