"""OpenUSD export of a variant's rig: the vehicle's body with its materials, the rig parts, every camera as a
UsdGeom.Camera with its exact intrinsics (fisheyes through NVIDIA's lens-distortion schema), lidars and radars as
prims with their parameters, radar-facing materials with their 77 GHz properties; and, beside it, the rig in NVIDIA's
NCore V4 format (the input format of Omniverse NuRec) and, from a log replay, the reconstructed scene's 3D Gaussians
as an OpenUSD ParticleField3DGaussianSplat. Plain text (.usda) written directly: no USD library is needed.

The stage (rig.usda)
--------------------
* Metadata: upAxis "Z", metersPerUnit 1, defaultPrim "Vehicle". Every prim is placed in the vehicle frame, ISO 8855
  (x forward, y left, z up, origin at the rear-axle centre on the ground), metres; each sensor's transform is a
  matrix4d xformOp:transform relative to /Vehicle (USD's row-vector convention: the last row is the translation).
* /Vehicle/Body: one Mesh per material (the body's faces grouped by their core.materials class, lamps and rims by
  part name, as the 3D view colours them), each bound to a UsdPreviewSurface material in /Vehicle/Looks. /Vehicle/Rig:
  the rig parts (core.rig, aluminium). /Vehicle/Sensors/Housings: every sensor's and camera's housing (core.housings'
  boxes and cylinders).
* Materials facing the radars (bumper fascia, radar covers, lamps, glass, painted plastics) carry custom attributes
  vantage:rf:frequencyGHz, vantage:rf:relativePermittivity (ε′), vantage:rf:lossTangent (tan δ),
  vantage:rf:thicknessMm
  and vantage:rf:layers (the full stack, inner to outer, with the paint's layers on a painted part), from
  core.materials' library (its cited sources) at the project's radar frequency (analysis.materials.radar_ghz); metals
  carry vantage:rf:kind "metal" (opaque at 77 GHz).
* /Vehicle/Sensors/Cameras/<tag>: a Camera. USD's camera looks down −Z with +Y up (OpenUSD UsdGeomCamera,
  openusd.org/release/api/class_usd_geom_camera.html, read 2026-11-11); Vantage's optical frame is OpenCV's (+Z
  forward, +Y down), so the prim's rotation is the optical frame's turned by diag(1, −1, −1). focalLength,
  horizontalAperture, verticalAperture and their offsets are in tenths of a scene unit (that page: "Perspective focal
  length in tenths of a scene unit"); with metersPerUnit 1 that is decimetres, so a 2.0 mm lens is focalLength 0.02.
  The apertures are the sensor's size (pixels × pixel pitch), the offsets put the principal point where the lens's
  is: horizontalApertureOffset = (W/2 − cx)·pitch, verticalApertureOffset = (cy − H/2)·pitch (USD's film y is up,
  the image's v down). A rectilinear lens is then exact as a plain USD camera. A fisheye or any other projection gets
  NVIDIA's OmniLensDistortionOpenCvFisheyeAPI (omni.usd.schema.omni_lens_distortion; the attributes as NVIDIA's
  "Camera Intrinsic Calibration" page of omni.sensors.nv.camera writes them, docs.omniverse.nvidia.com/kit/docs/
  omni.sensors.nv.camera/latest/intrinsic_calibration.html, last updated 2026-10-27, read 2026-11-11):
  omni:lensdistortion:opencvFisheye:fx, fy, cx, cy (pixels, OpenCV's convention: pixel centres on integers),
  imageSize (int2), k1…k4: OpenCV's fisheye model, θ_d = θ(1 + k1θ² + k2θ⁴ + k3θ⁶ + k4θ⁸), r = f·θ_d, which
  is Vantage's Kannala-Brandt model exactly (other projections are fitted, the fit's RMS in pixels stated in
  vantage:lens:fitRmsPx). That page says the standard camera values are then placeholders for viewers and RTX projects
  through the schema. The deprecated OmniLensDistortionKannalaBrandtK3API (its own docs recommend the newer schemas)
  is not written. Every camera also carries vantage:lens:* (projection, f in mm, pixel pitch, max θ, image size) and
  vantage:module / vantage:lens.
* /Vehicle/Sensors/Lidars/<tag> and Radars/<tag>: Xform prims at the sensor's frame (x forward, y left, z up, REP
  103, at its optical centre) with vantage:lidar:* / vantage:radar:* attributes (channels, beam elevations, fields,
  rates, ranges, zones, the ranging model and its parameters). Isaac Sim's RTX lidar and radar take a vendor profile
  of their own; these prims carry the figures such a profile needs, not the profile.

NVIDIA NuRec
------------
NuRec (docs.nvidia.com/nurec, read 2026-11-11) takes its input in NVIDIA's NCore format and produces a USDZ holding a
USD scene, a checkpoint of the trained reconstruction and a JSON of the sequence's tracks and rig trajectories; the
checkpoint's format and the USD prims NuRec writes for it (Kit apps drive it through OmniNuRecVolumeAPI) are not
publicly specified, so they are not written here. NCore is public (nvidia.github.io/ncore, V4, read 2026-11-11: Data
Formats, Conventions, Sensor Models; and its reference implementation, PyPI nvidia-ncore 19.9.0): the rig goes out as
an NCore V4 component store, <name>.ncore4.zarr, a zarr v2 directory store as nvidia-ncore's
SequenceComponentGroupsWriter writes it with store_type='directory': the root group's attributes (sequence_id,
sequence_timestamp_interval_us, version 'v4', component_group_name, generic_meta_data), an 'intrinsics' component
instance 'default' with cameras/<camera_id> (camera_model_type 'opencv-fisheye' or 'opencv-pinhole' and its
parameters: resolution, shutter_type, principal_point, focal_length, radial_coeffs, max_angle) and lidars/<lidar_id>
('row-offset-spinning': row_elevations_rad, column_azimuths_rad, row_azimuth_offsets_rad, spinning_frequency_hz,
spinning_direction, n_rows, n_columns), a 'poses' component with static_poses ('camera_<tag>', 'rig') etc.: T mapping
the sensor's points into the rig frame (NCore's rig frame is Vantage's vehicle frame: x forward, y left, z up, the
rear axle's middle on the nominal ground; its camera frame is OpenCV's), and the consolidated metadata
(.zmetadata.cbor.xz: CBOR, xz-compressed). NCore's image coordinates put the first pixel over [0, 1], as Vantage's
sensor coordinates do, so the principal point is Vantage's cx, cy. Images, lidar sweeps and ego trajectories are not
part of the rig export.

The 3D Gaussians of a log replay (core.splat_reproject's cache, scene.npz) go out as an OpenUSD
ParticleField3DGaussianSplat (UsdVol, OpenUSD 26.11: the schema in usd-core 26.11's usdVol generatedSchema.usda,
read 2026-11-11): positions, orientations (quatf, w first), scales (the Gaussians' σ, metres), opacities (0–1),
radiance:sphericalHarmonicsCoefficients of degree 0 (the colour as 3DGS's DC term, (c − 0.5) / 0.2820948) and
primvars:displayColor; under /Scene, translated by the scene's origin, in the drive's global frame (not the vehicle's).
The schema documents the attributes' meaning, not the DC convention: the 3DGS one is used and stated in the file.
"""
from __future__ import annotations

import io
import json
import lzma
import math
import os
import re
import struct
import time
from dataclasses import asdict, dataclass, field

import numpy as np

from .mathutil import BODY_TO_OPTICAL, rot_zyx

SCHEMA = 'vantage.usd_export/1'
FRAME = 'vehicle (ISO 8855, rear axle on the ground): x forward, y left, z up, metres; USD upAxis Z, metersPerUnit 1'
USD_FROM_OPTICAL = np.diag([1.0, -1.0, -1.0])         # USD camera (x right, y up, −z forward) <-> OpenCV optical
SH_C0 = 0.28209479177387814
LIDAR_COLUMNS_MAX = 4096
LIMITS = {'max_splats': (0, 20_000_000), 'decimate_tris': (0, 2_000_000)}
INTS = ('max_splats', 'decimate_tris')
BOOLS = ('body', 'rig', 'housings', 'cameras', 'lidars', 'radars', 'rf', 'ncore')
OMNI_FISHEYE_API = 'OmniLensDistortionOpenCvFisheyeAPI'
OMNI_NS = 'omni:lensdistortion:opencvFisheye'
SOURCES = {
    'usd_camera': 'OpenUSD UsdGeomCamera (openusd.org/release/api/class_usd_geom_camera.html), read 2026-11-11',
    'omni_fisheye': 'NVIDIA omni.sensors.nv.camera, Camera Intrinsic Calibration (docs.omniverse.nvidia.com/kit/docs/'
                    'omni.sensors.nv.camera/latest/intrinsic_calibration.html, last updated 2026-10-27), read '
                    '2026-11-11',
    'ncore': 'NVIDIA NCore V4 (nvidia.github.io/ncore: data/formats, data/conventions, data/sensor_models) and PyPI '
             'nvidia-ncore 19.9.0, read 2026-11-11',
    'nurec': 'NVIDIA Omniverse NuRec documentation (docs.nvidia.com/nurec), read 2026-11-11: USDZ output with a '
             'checkpoint and JSON; no public specification of the checkpoint or its USD prims',
    'particle_field': 'OpenUSD 26.11 UsdVol ParticleField3DGaussianSplat (usd-core 26.11 generatedSchema.usda), '
                      'read 2026-11-11'}
ASSUMPTIONS = (
    'Every pose is the variant\'s own (a calibrated or as-built variant carries its measured poses). A lens other '
    'than Kannala-Brandt is fitted with the OpenCV fisheye polynomial (fit RMS stated per camera); a rectilinear lens '
    'is a plain USD camera. The RF attributes are core.materials\' typical library values at the radar frequency, not '
    'a measurement of the user\'s parts. NuRec\'s own reconstruction format is not public: the rig is written in its '
    'documented input format (NCore V4) and Gaussians in OpenUSD\'s ParticleField schema.')


def _flag(v, default):
    if isinstance(v, (bool, int, np.bool_)):
        return bool(v)
    if isinstance(v, str):
        return v.strip().lower() in ('1', 'true', 'yes', 'on')
    return default


@dataclass
class UsdSettings:
    body: bool = True                   # the vehicle's body meshes
    rig: bool = True                    # the rig parts
    housings: bool = True               # sensor and camera housings as meshes
    cameras: bool = True
    lidars: bool = True
    radars: bool = True
    rf: bool = True                     # radar material attributes
    ncore: bool = True                  # the rig as an NCore V4 store beside the stage
    splats: str = ''                    # a log replay's cache folder (scene.npz): its Gaussians as a ParticleField
    max_splats: int = 2_000_000         # at most this many Gaussians (the most opaque kept)
    decimate_tris: int = 0              # body triangles (0: all)
    name: str = 'vantage_rig'           # the stage's and the NCore store's base name

    def to_dict(self):
        return asdict(self)

    @staticmethod
    def from_dict(d) -> 'UsdSettings':
        s = UsdSettings()
        d = d if isinstance(d, dict) else {}
        for k in BOOLS:
            if k in d:
                setattr(s, k, _flag(d[k], getattr(s, k)))
        for k, (lo, hi) in LIMITS.items():
            if k in d:
                try:
                    v = float(d[k])
                except (TypeError, ValueError):
                    continue
                if math.isfinite(v):
                    setattr(s, k, int(min(max(v, lo), hi)))
        if isinstance(d.get('splats'), str):
            s.splats = d['splats'][:4096]
        n = d.get('name')
        if isinstance(n, str) and re.fullmatch(r'[A-Za-z_][A-Za-z0-9_]{0,63}', n):
            s.name = n
        return s


def settings_of(project) -> UsdSettings:
    return UsdSettings.from_dict(getattr(project.analysis, 'usd_export', None))


# =========================================================================== USDA text
def ident(name: str) -> str:
    """A valid USD prim name from any text."""
    s = re.sub(r'[^A-Za-z0-9_]', '_', str(name)) or '_'
    return s if not s[0].isdigit() else '_' + s


def _f(x) -> str:
    x = float(x)
    if not math.isfinite(x):
        return '0'
    s = repr(round(x, 9))
    return s[:-2] if s.endswith('.0') else s


def _str(s) -> str:
    return json.dumps(str(s), ensure_ascii=False)


def _vec(v) -> str:
    return '(' + ', '.join(_f(x) for x in v) + ')'


def _arr(rows) -> str:
    return '[' + ', '.join(_vec(r) for r in rows) + ']'


def _iarr(a) -> str:
    return '[' + ', '.join(str(int(x)) for x in a) + ']'


def _farr(a) -> str:
    return '[' + ', '.join(_f(x) for x in a) + ']'


def _mat4(R, t) -> str:
    """USD matrix4d (row vectors: p' = p · M) of a rotation R (column convention, local -> parent) and translation."""
    M = np.eye(4)
    M[:3, :3] = np.asarray(R, float).T
    M[3, :3] = np.asarray(t, float)
    return '( ' + ', '.join(_vec(r) for r in M) + ' )'


class _W:
    def __init__(self):
        self.lines = []
        self.ind = 0

    def __call__(self, s=''):
        self.lines.append(('    ' * self.ind + s) if s else '')

    def open(self, head):
        self(head)
        self('{')
        self.ind += 1

    def close(self):
        self.ind -= 1
        self('}')

    def text(self):
        return '\n'.join(self.lines) + '\n'


# =========================================================================== cameras
@dataclass
class CamUsd:
    tag: str
    prim: str
    width: int
    height: int
    pixel_mm: float
    f_mm: float
    cx: float                           # Vantage sensor coordinates (pixel centres at i + 0.5)
    cy: float
    aspect: float
    projection: str
    kb: tuple                           # OpenCV fisheye k1..k4 (native or fitted)
    fit_rms_px: float
    max_theta_deg: float
    R: np.ndarray                       # optical -> vehicle
    pos: np.ndarray
    module: str = ''
    lens: str = ''

    @property
    def fisheye(self) -> bool:
        return self.projection != 'rectilinear'

    @property
    def f_px(self):
        return self.f_mm / self.pixel_mm

    def usd_attrs(self) -> dict:
        p = self.pixel_mm
        return {'focalLength': self.f_mm / 100.0, 'horizontalAperture': self.width * p / 100.0,
                'verticalAperture': self.height * p / self.aspect / 100.0,     # fy = aspect · fx: a taller pixel pitch
                'horizontalApertureOffset': (self.width / 2.0 - self.cx) * p / 100.0,
                'verticalApertureOffset': (self.cy - self.height / 2.0) * p / self.aspect / 100.0}


def camera_usd(project, cam) -> CamUsd:
    from .exporters import kb_coefficients
    from .lens import resolve
    intr = resolve(project.module_for(cam), project.lens_for(cam))
    k, rms = kb_coefficients(intr)
    R = rot_zyx(cam.yaw, cam.pitch, cam.roll)
    O2V = (BODY_TO_OPTICAL @ R.T).T
    return CamUsd(cam.tag, ident(cam.tag), int(intr.width), int(intr.height), float(intr.pixel_mm),
                  float(intr.f_mm), float(intr.cx), float(intr.cy), float(intr.aspect), intr.projection,
                  tuple(float(x) for x in k), float(rms), math.degrees(float(intr.max_theta)), O2V,
                  np.asarray(cam.pos(), float), getattr(cam, 'module_id', ''), getattr(cam, 'lens_id', ''))


def project_usd_camera(attrs: dict, M: np.ndarray, P_parent: np.ndarray, width: int, height: int):
    """Pixel coordinates (Vantage's sensor convention, pixel centres at i + 0.5) of parent-frame points through a USD
    camera as written here: its matrix4d (row vectors) and either the standard perspective attributes or, when it
    has them, the OpenCV fisheye attributes. The reader-side check of the export (no USD library)."""
    M = np.asarray(M, float)
    P = np.asarray(P_parent, float).reshape(-1, 3)
    Pc = (np.c_[P, np.ones(len(P))] @ np.linalg.inv(M))[:, :3]         # into the USD camera's frame
    if attrs.get(OMNI_NS + ':fx') is not None:
        x, y, z = Pc[:, 0], -Pc[:, 1], -Pc[:, 2]                          # OpenCV optical
        a, b = x / z, y / z
        r = np.hypot(a, b)
        th = np.arctan(r)
        th = np.where(z > 0, th, np.pi - np.arctan(r))
        k1, k2, k3, k4 = (float(attrs[f'{OMNI_NS}:k{i}']) for i in (1, 2, 3, 4))
        t2 = th * th
        thd = th * (1 + t2 * (k1 + t2 * (k2 + t2 * (k3 + t2 * k4))))
        rho = np.hypot(x, y)
        s = np.where(rho > 1e-15, thd / np.where(rho > 1e-15, rho, 1.0), 0.0)
        u = float(attrs[OMNI_NS + ':fx']) * x * s + float(attrs[OMNI_NS + ':cx'])
        v = float(attrs[OMNI_NS + ':fy']) * y * s + float(attrs[OMNI_NS + ':cy'])
        return u + 0.5, v + 0.5
    f = float(attrs['focalLength'])
    ha, va = float(attrs['horizontalAperture']), float(attrs['verticalAperture'])
    ho, vo = float(attrs.get('horizontalApertureOffset', 0.0)), float(attrs.get('verticalApertureOffset', 0.0))
    xf = f * Pc[:, 0] / -Pc[:, 2]
    yf = f * Pc[:, 1] / -Pc[:, 2]
    u = ((xf - ho) / (ha / 2.0) + 1.0) / 2.0 * width
    v = (1.0 - (yf - vo) / (va / 2.0)) / 2.0 * height
    return u, v


# =========================================================================== materials
def _rf_of(cls_id: str, st_mat) -> dict | None:
    """RF attributes of a core.materials class at the radar frequency."""
    from . import materials as MT
    cls = MT.CLASSES.get(cls_id)
    if cls is None:
        return None
    f = float(st_mat.radar_ghz)
    if cls.kind in ('metal', 'absorber'):
        return {'kind': cls.kind, 'frequencyGHz': f}
    ids = list(cls.layers) + (list(MT.FINISHES.get(st_mat.paint, ())) if cls.painted else [])
    rows = [(diel, MT.DIELECTRICS[diel][0], float(MT.DIELECTRICS[diel][1]), float(MT.tan_delta(diel, f)), float(mm))
            for diel, mm in ids]
    sub = rows[:1]                                   # the substrate (inner layer) is the material's own
    return {'kind': cls.kind, 'frequencyGHz': f, 'relativePermittivity': sub[0][2], 'lossTangent': sub[0][3],
            'thicknessMm': float(sum(r[4] for r in rows)), 'layers': rows}


VIS = {'paint': (None, 0.35, 0.0, 1.0), 'aluminium': ((0.55, 0.56, 0.58), 0.35, 1.0, 1.0),
       'chrome': ((0.75, 0.76, 0.78), 0.08, 1.0, 1.0), 'carbon': ((0.03, 0.03, 0.035), 0.4, 0.0, 1.0),
       'trim': ((0.035, 0.035, 0.035), 0.7, 0.0, 1.0), 'radome': ((0.2, 0.2, 0.21), 0.4, 0.0, 1.0),
       'lamp': ((0.25, 0.27, 0.30), 0.1, 0.0, 0.6), 'tail_lamp': ((0.45, 0.02, 0.02), 0.1, 0.0, 0.7),
       'glass': ((0.02, 0.025, 0.03), 0.05, 0.0, 0.35), 'tyre': ((0.025, 0.025, 0.025), 0.9, 0.0, 1.0),
       'rim': ((0.45, 0.46, 0.48), 0.3, 1.0, 1.0), 'housing': ((0.05, 0.05, 0.055), 0.6, 0.0, 1.0),
       'lens': ((0.01, 0.01, 0.012), 0.05, 0.0, 1.0)}


# =========================================================================== the export
@dataclass
class UsdResult:
    text: str
    name: str
    cameras: list = field(default_factory=list)          # [CamUsd]
    ncore: dict = field(default_factory=dict)            # relative path -> bytes
    counts: dict = field(default_factory=dict)
    notes: list = field(default_factory=list)
    settings: dict = field(default_factory=dict)
    project: str = ''
    variant: str = ''
    seconds: float = 0.0


def _mesh_block(w: _W, name: str, tri: np.ndarray, material: str):
    """A Mesh prim of triangles (vehicle frame) with welded points."""
    P = tri.reshape(-1, 3)
    key = np.round(P / 1e-6).astype(np.int64)
    uniq, idx, inv = np.unique(key, axis=0, return_index=True, return_inverse=True)
    pts = P[idx]
    w.open(f'def Mesh "{name}" (\n{"    " * (w.ind + 1)}prepend apiSchemas = ["MaterialBindingAPI"]\n{"    " * w.ind})')
    w(f'float3[] extent = [{_vec(pts.min(0))}, {_vec(pts.max(0))}]')
    w(f'int[] faceVertexCounts = {_iarr([3] * len(tri))}')
    w(f'int[] faceVertexIndices = {_iarr(inv.ravel())}')
    w(f'point3f[] points = {_arr(pts)}')
    w('uniform token subdivisionScheme = "none"')
    w('uniform token orientation = "rightHanded"')
    w('uniform bool doubleSided = 1')
    w(f'rel material:binding = </Vehicle/Looks/{material}>')
    w.close()


def run(project, engine=None, settings: UsdSettings | None = None, variant=None, progress=None,
        cancel=None) -> UsdResult | None:
    from . import ego_composite as EC
    from . import materials as MT
    from . import sensors as SN
    t0 = time.perf_counter()
    st = settings or settings_of(project)
    prog = progress or (lambda f, text='': None)
    var = variant or project.variant
    p = project
    if variant is not None and variant is not project.variant:
        p = project.snapshot()
        p.active_uid = variant.uid
        var = p.variant
    st_mat = MT.settings_of(p)
    w = _W()
    notes = []
    w('#usda 1.0')
    w('(')
    w.ind += 1
    w('defaultPrim = "Vehicle"')
    w(f'doc = {_str("Vantage rig export: " + p.name + " / " + var.name + ". " + FRAME)}')
    w('metersPerUnit = 1')
    w('upAxis = "Z"')
    w('customLayerData = {')
    w.ind += 1
    w(f'string "vantage:schema" = {_str(SCHEMA)}')
    w(f'string "vantage:frame" = {_str(FRAME)}')
    w(f'string "vantage:project" = {_str(p.name)}')
    w(f'string "vantage:variant" = {_str(var.name)}')
    try:
        from .. import __version__ as VER
    except ImportError:                         # pragma: no cover
        VER = ''
    w(f'string "vantage:version" = {_str(VER)}')
    w.ind -= 1
    w('}')
    w.ind -= 1
    w(')')
    w()
    w.open('def Xform "Vehicle" (\n    kind = "assembly"\n)')
    counts = {'body_triangles': 0, 'rig_triangles': 0, 'housing_triangles': 0, 'cameras': 0, 'lidars': 0,
              'radars': 0, 'materials': 0, 'splats': 0}
    mesh = EC.ego_mesh(p, var, EC.EgoSettings.from_dict(getattr(p.analysis, 'ego_composite', None)), engine) \
        if (st.body or st.rig or st.housings) else None
    face_cls = None
    if mesh is not None:
        mmesh, parts = MT._geometry(p, engine.vehicle_geom(p, var) if engine is not None else None)
        fm = MT.face_materials(mmesh, parts, st_mat)
        # the class of every body / rig face in ego_mesh's order (it keeps the non-ignored faces, then housings)
        from .mesh import ROLE_IGNORED
        fp = np.clip(mmesh.face_part, 0, max(len(parts) - 1, 0))
        keep = np.array([getattr(q, 'role', '') != ROLE_IGNORED for q in parts] or [True], bool)[fp]
        face_cls = np.asarray(fm.cls)[keep]
        nveh = len(p.vehicle.mesh.parts) if p.vehicle.mesh is not None else 0
        rig_face = (fp >= nveh)[keep]
    # materials
    used = []
    if mesh is not None:
        nb = int((mesh.owner < 0).sum())
        for mi in np.unique(mesh.mat):
            used.append(EC.MATERIALS[int(mi)][0])
    paint = EC.hex_linear(EC.EgoSettings.from_dict(getattr(p.analysis, 'ego_composite', None)).paint)
    w.open('def Scope "Looks"')
    rf_classes = {}
    if mesh is not None and face_cls is not None:
        nb = int((mesh.owner < 0).sum())
        for mi in np.unique(mesh.mat[:nb]):
            cls_ids = np.unique(face_cls[mesh.mat[:nb] == mi])
            rf_classes[EC.MATERIALS[int(mi)][0]] = [MT.CLASS_IDS[int(c)] if c >= 0 else 'steel' for c in cls_ids]
    for name in sorted(set(used) | {'housing'}):
        base, rough, metal, opac = VIS.get(name, ((0.5, 0.5, 0.5), 0.5, 0.0, 1.0))
        col = paint if base is None else np.asarray(base)
        w.open(f'def Material "{name}"')
        w(f'token outputs:surface.connect = </Vehicle/Looks/{name}/Surface.outputs:surface>')
        if st.rf:
            cids = rf_classes.get(name) or (['aluminium'] if name in ('aluminium', 'rim') else [])
            for k, cid in enumerate(cids):
                rf = _rf_of(cid, st_mat)
                if rf is None:
                    continue
                pre = 'vantage:rf' if k == 0 else f'vantage:rf:{cid}'
                w(f'custom string {pre}:class = {_str(cid)}')
                w(f'custom string {pre}:kind = {_str(rf["kind"])}')
                w(f'custom float {pre}:frequencyGHz = {_f(rf["frequencyGHz"])}')
                if 'relativePermittivity' in rf:
                    w(f'custom float {pre}:relativePermittivity = {_f(rf["relativePermittivity"])}')
                    w(f'custom float {pre}:lossTangent = {_f(rf["lossTangent"])}')
                    w(f'custom float {pre}:thicknessMm = {_f(rf["thicknessMm"])}')
                    lay = json.dumps([{'dielectric': r[0], 'name': r[1], 'eps_r': r[2], 'tan_delta': r[3],
                                       'thickness_mm': r[4]} for r in rf['layers']])
                    w(f'custom string {pre}:layers = {_str(lay)}')
                    counts['rf_materials'] = counts.get('rf_materials', 0) + 1
        w.open('def Shader "Surface"')
        w('uniform token info:id = "UsdPreviewSurface"')
        w(f'color3f inputs:diffuseColor = {_vec(col)}')
        w(f'float inputs:roughness = {_f(rough)}')
        w(f'float inputs:metallic = {_f(metal)}')
        w(f'float inputs:opacity = {_f(opac)}')
        w('token outputs:surface')
        w.close()
        w.close()
        counts['materials'] += 1
    w.close()
    prog(0.2, 'Body')
    if mesh is not None:
        body = mesh.owner < 0
        is_rig = np.zeros(len(mesh.tri), bool)
        if face_cls is not None:
            is_rig[:len(rig_face)] = rig_face
        for group, sel, flag in (('Body', body & ~is_rig, st.body), ('Rig', body & is_rig, st.rig)):
            if not flag or not sel.any():
                continue
            w.open(f'def Xform "{group}"')
            for mi in np.unique(mesh.mat[sel]):
                nm = EC.MATERIALS[int(mi)][0]
                T = mesh.tri[sel & (mesh.mat == mi)]
                if group == 'Body' and st.decimate_tris and len(T) > st.decimate_tris:
                    T = T[np.linspace(0, len(T) - 1, st.decimate_tris).astype(int)]
                _mesh_block(w, ident(nm), T, nm)
                counts['body_triangles' if group == 'Body' else 'rig_triangles'] += len(T)
            w.close()
    w.open('def Xform "Sensors"')
    if mesh is not None and st.housings and (mesh.owner >= 0).any():
        w.open('def Xform "Housings"')
        from . import housings as HS
        sols = HS.solids(p, var)
        for k, h in enumerate(sols):
            T = mesh.tri[mesh.owner == k]
            _mesh_block(w, ident(f'{h.what}_{h.tag}'), T, 'housing')
            counts['housing_triangles'] += len(T)
        w.close()
    cams = []
    if st.cameras:
        w.open('def Xform "Cameras"')
        for cam in [c for c in var.cameras if c.visible]:
            if cancel is not None and cancel():
                return None
            cu = camera_usd(p, cam)
            cams.append(cu)
            api = f' (\n{"    " * (w.ind + 1)}prepend apiSchemas = ["{OMNI_FISHEYE_API}"]\n{"    " * w.ind})' \
                if cu.fisheye else ''
            w.open(f'def Camera "{cu.prim}"{api}')
            w(f'matrix4d xformOp:transform = {_mat4(cu.R @ USD_FROM_OPTICAL, cu.pos)}')
            w('uniform token[] xformOpOrder = ["xformOp:transform"]')
            for k, v in cu.usd_attrs().items():
                w(f'float {k} = {_f(v)}')
            w('float2 clippingRange = (0.01, 1000)')
            w('token projection = "perspective"')
            if cu.fisheye:
                f = cu.f_px
                w(f'float {OMNI_NS}:fx = {_f(f)}')
                w(f'float {OMNI_NS}:fy = {_f(f * cu.aspect)}')
                w(f'float {OMNI_NS}:cx = {_f(cu.cx - 0.5)}')
                w(f'float {OMNI_NS}:cy = {_f(cu.cy - 0.5)}')
                w(f'int2 {OMNI_NS}:imageSize = ({cu.width}, {cu.height})')
                for i, kk in enumerate(cu.kb):
                    w(f'float {OMNI_NS}:k{i + 1} = {_f(kk)}')
            w(f'custom string vantage:tag = {_str(cam.tag)}')
            w(f'custom string vantage:module = {_str(cu.module)}')
            w(f'custom string vantage:lens = {_str(cu.lens)}')
            w(f'custom string vantage:lens:projection = {_str(cu.projection)}')
            w(f'custom float vantage:lens:focalLengthMm = {_f(cu.f_mm)}')
            w(f'custom float vantage:lens:pixelPitchMm = {_f(cu.pixel_mm)}')
            w(f'custom float vantage:lens:aspect = {_f(cu.aspect)}')
            w(f'custom float vantage:lens:maxThetaDeg = {_f(cu.max_theta_deg)}')
            w(f'custom float vantage:lens:fitRmsPx = {_f(cu.fit_rms_px)}')
            w(f'custom float2 vantage:lens:principalPointPx = ({_f(cu.cx)}, {_f(cu.cy)})')
            w(f'custom int2 vantage:imageSize = ({cu.width}, {cu.height})')
            w(f'custom double3 vantage:poseYawPitchRollDeg = ({_f(cam.yaw)}, {_f(cam.pitch)}, {_f(cam.roll)})')
            w.close()
            counts['cameras'] += 1
        w.close()
    sens = [s for s in var.sensors if s.visible]
    for kind, flag, group in (('lidar', st.lidars, 'Lidars'), ('radar', st.radars, 'Radars')):
        items = [s for s in sens if s.kind == kind]
        if not flag or not items:
            continue
        w.open(f'def Xform "{group}"')
        for s in items:
            m = p.sensor_model_for(s)
            w.open(f'def Xform "{ident(s.tag)}"')
            w(f'matrix4d xformOp:transform = {_mat4(SN.rotation(s), s.pos())}')
            w('uniform token[] xformOpOrder = ["xformOp:transform"]')
            ns = f'vantage:{kind}'
            w(f'custom string vantage:tag = {_str(s.tag)}')
            w(f'custom string {ns}:model = {_str(m.id)}')
            w(f'custom string {ns}:name = {_str(m.name)}')
            w(f'custom string {ns}:frame = {_str("REP 103 at the optical centre: x forward, y left, z up")}')
            w(f'custom float {ns}:rateHz = {_f(m.rate_hz)}')
            w(f'custom float {ns}:rangeM = {_f(m.range_m)}')
            w(f'custom float {ns}:rangeMinM = {_f(m.range_min_m)}')
            if kind == 'lidar':
                w(f'custom string {ns}:scan = {_str(m.scan)}')
                w(f'custom int {ns}:channels = {int(m.n_channels())}')
                w(f'custom float[] {ns}:beamElevationsDeg = {_farr(m.elevations())}')
                w(f'custom float {ns}:horizontalFovDeg = {_f(m.h_fov)}')
                w(f'custom float {ns}:azimuthStepDeg = {_f(m.h_res)}')
                w(f'custom float {ns}:pointsPerSecond = {_f(m.pps())}')
                w(f'custom string {ns}:rotation = {_str(m.rotation)}')
                counts['lidars'] += 1
            else:
                z = [list(map(float, zz)) for zz in (m.zones or [])]
                w(f'custom float3[] {ns}:zones = {_arr(z) if z else "[]"}')
                note = '[range m, azimuth half-angle deg, elevation half-angle deg]'
                w(f'custom string {ns}:zonesNote = {_str(note)}')
                counts['radars'] += 1
            rng = getattr(m, 'ranging', '') or ''
            if rng:
                w(f'custom string {ns}:ranging = {_str(rng)}')
            prm = getattr(m, 'params', None)
            if isinstance(prm, dict) and prm:
                w(f'custom string {ns}:params = {_str(json.dumps(prm, default=str))}')
            w.close()
        w.close()
    w.close()          # Sensors
    w.close()          # Vehicle
    # the Gaussians of a log replay
    if st.splats:
        prog(0.7, 'Gaussians')
        n = _splat_block(w, st, notes)
        counts['splats'] = n
    res = UsdResult(w.text(), st.name, cams, {}, counts, notes, st.to_dict(), p.name, var.name)
    if st.ncore:
        prog(0.85, 'NCore')
        res.ncore = ncore_files(p, var, cams, st.name)
    res.seconds = round(time.perf_counter() - t0, 3)
    prog(1.0, 'Done')
    return res


def _splat_block(w: _W, st: UsdSettings, notes: list) -> int:
    from . import splat_reproject as SP
    sc = SP.Scene.load(st.splats)
    if sc is None:
        notes.append(f'No log-replay scene in {st.splats} (scene.npz): no Gaussians written.')
        return 0
    idx = np.arange(sc.n)
    if st.max_splats and sc.n > st.max_splats:
        idx = np.argsort(-np.asarray(sc.opacity))[:st.max_splats]
        notes.append(f'{sc.n} Gaussians, the {st.max_splats} most opaque written.')
    static = np.asarray(sc.actor)[idx] < 0
    if not static.all():
        notes.append(f'{int((~static).sum())} Gaussians of moving actors left out (they live in their boxes\' frames).')
    idx = idx[static]
    P = np.asarray(sc.means, float)[idx]
    Q = np.asarray(sc.quats, float)[idx]
    S = np.asarray(sc.scales, float)[idx]
    O = np.asarray(sc.opacity, float)[idx]
    C = np.clip(np.asarray(sc.colours, float)[idx], 0, 1)
    sh = (C - 0.5) / SH_C0
    w()
    doc = 'A log replay\'s 3D Gaussians (core.splat_reproject) in the drive\'s global frame'
    w.open(f'def Xform "Scene" (\n    doc = "{doc}"\n)')
    w(f'double3 xformOp:translate = {_vec(np.asarray(sc.origin, float))}')
    w('uniform token[] xformOpOrder = ["xformOp:translate"]')
    w.open('def ParticleField3DGaussianSplat "Gaussians"')
    if len(P):
        w(f'float3[] extent = [{_vec(P.min(0))}, {_vec(P.max(0))}]')
    w(f'point3f[] positions = {_arr(P)}')
    w('quatf[] orientations = [' + ', '.join(_vec(q) for q in Q) + ']')
    w(f'float3[] scales = {_arr(S)}')
    w(f'float[] opacities = {_farr(O)}')
    w('uniform int radiance:sphericalHarmonicsDegree = 0')
    w(f'float3[] radiance:sphericalHarmonicsCoefficients = {_arr(sh)}')
    w(f'color3f[] primvars:displayColor = {_arr(C)}')
    w(f'custom string vantage:shConvention = {_str("3DGS DC term: colour = 0.5 + 0.2820948 * coefficient")}')
    w(f'custom string vantage:source = {_str(sc.source)}')
    w.close()
    w.close()
    return int(len(P))


# =========================================================================== NCore V4
def _cbor(o, out: bytearray):
    """A minimal CBOR (RFC 8949) encoder for JSON-like data."""
    def head(major, n):
        if n < 24:
            out.append((major << 5) | n)
        elif n < 256:
            out.extend(bytes([(major << 5) | 24, n]))
        elif n < 65536:
            out.append((major << 5) | 25)
            out.extend(struct.pack('>H', n))
        elif n < 2 ** 32:
            out.append((major << 5) | 26)
            out.extend(struct.pack('>I', n))
        else:
            out.append((major << 5) | 27)
            out.extend(struct.pack('>Q', n))
    if o is None:
        out.append(0xf6)
    elif o is True:
        out.append(0xf5)
    elif o is False:
        out.append(0xf4)
    elif isinstance(o, int):
        if o >= 0:
            head(0, o)
        else:
            head(1, -1 - o)
    elif isinstance(o, float):
        out.append(0xfb)
        out.extend(struct.pack('>d', o))
    elif isinstance(o, str):
        b = o.encode('utf-8')
        head(3, len(b))
        out.extend(b)
    elif isinstance(o, (list, tuple)):
        head(4, len(o))
        for x in o:
            _cbor(x, out)
    elif isinstance(o, dict):
        head(5, len(o))
        for k, v in o.items():
            _cbor(str(k), out)
            _cbor(v, out)
    else:
        raise TypeError(type(o))


def cbor_dumps(o) -> bytes:
    out = bytearray()
    _cbor(o, out)
    return bytes(out)


def cbor_loads(b: bytes):
    """Decoder for what cbor_dumps writes (tests)."""
    pos = [0]

    def n_of(info):
        if info < 24:
            return info
        k = {24: 1, 25: 2, 26: 4, 27: 8}[info]
        v = int.from_bytes(b[pos[0]:pos[0] + k], 'big')
        pos[0] += k
        return v

    def item():
        ib = b[pos[0]]
        pos[0] += 1
        major, info = ib >> 5, ib & 31
        if major == 7:
            if info == 20:
                return False
            if info == 21:
                return True
            if info == 22:
                return None
            if info == 27:
                v = struct.unpack('>d', b[pos[0]:pos[0] + 8])[0]
                pos[0] += 8
                return v
            raise ValueError(info)
        n = n_of(info)
        if major == 0:
            return n
        if major == 1:
            return -1 - n
        if major == 3:
            s = b[pos[0]:pos[0] + n].decode('utf-8')
            pos[0] += n
            return s
        if major == 4:
            return [item() for _ in range(n)]
        if major == 5:
            return {item(): item() for _ in range(n)}
        raise ValueError(major)
    return item()


def _pose_key(a, b) -> str:
    return str((a, b))


def ncore_rig(project, variant, cams: list, name: str, interval_us=(0, 1)) -> dict:
    """{zarr key: JSON-able document} of the rig's NCore V4 store (groups and attributes only)."""
    from . import sensors as SN
    from .lens import resolve
    groups = ['', 'intrinsics', 'intrinsics/default', 'intrinsics/default/cameras', 'intrinsics/default/lidars',
              'poses', 'poses/default', 'poses/default/static_poses', 'poses/default/dynamic_poses']
    attrs = {'': {'component_group_name': '', 'generic_meta_data': {'vantage': {
        'schema': SCHEMA, 'project': project.name, 'variant': variant.name, 'frame': FRAME}},
        'sequence_id': name,
        'sequence_timestamp_interval_us': {'start': int(interval_us[0]), 'stop': int(interval_us[1])},
        'version': 'v4'},
        'intrinsics/default': {'component_instance_name': 'default', 'component_name': 'intrinsics',
                               'component_version': 'v1', 'generic_meta_data': {}},
        'poses/default': {'component_instance_name': 'default', 'component_name': 'poses', 'component_version': 'v1',
                          'generic_meta_data': {}},
        'poses/default/dynamic_poses': {}}
    static = {}
    for cu in cams:
        cid = f'camera_{cu.tag}'
        gp = f'intrinsics/default/cameras/{cid}'
        groups.append(gp)
        res = [cu.width, cu.height]
        if cu.fisheye:
            attrs[gp] = {'camera_model_type': 'opencv-fisheye', 'camera_model_parameters': {
                'external_distortion_parameters': None, 'focal_length': [cu.f_px, cu.f_px * cu.aspect],
                'max_angle': math.radians(cu.max_theta_deg), 'principal_point': [cu.cx, cu.cy],
                'radial_coeffs': list(cu.kb), 'resolution': res, 'shutter_type': 'GLOBAL'}}
        else:
            attrs[gp] = {'camera_model_type': 'opencv-pinhole', 'camera_model_parameters': {
                'external_distortion_parameters': None, 'focal_length': [cu.f_px, cu.f_px * cu.aspect],
                'principal_point': [cu.cx, cu.cy], 'radial_coeffs': [0.0] * 6, 'resolution': res,
                'shutter_type': 'GLOBAL', 'tangential_coeffs': [0.0, 0.0], 'thin_prism_coeffs': [0.0] * 4}}
        T = np.eye(4)
        T[:3, :3] = cu.R
        T[:3, 3] = cu.pos
        static[_pose_key(cid, 'rig')] = {'dtype': 'float64', 'pose': T.tolist()}
    for s in [s for s in variant.sensors if s.visible and s.kind in ('lidar', 'radar', 'imu', 'gnss')]:
        m = project.sensor_model_for(s)
        sid = f'{s.kind}_{s.tag}'
        T = np.eye(4)
        T[:3, :3] = SN.rotation(s)
        T[:3, 3] = s.pos()
        static[_pose_key(sid, 'rig')] = {'dtype': 'float64', 'pose': T.tolist()}
        if s.kind == 'lidar' and m.spinning:
            gp = f'intrinsics/default/lidars/{sid}'
            groups.append(gp)
            n_cols = int(min(max(round(m.h_fov / max(m.h_res, 1e-3)), 1), LIDAR_COLUMNS_MAX))
            # NCore's order: rows by descending elevation, columns along the spin (cw: falling azimuth)
            el = [math.radians(e) for e in sorted({round(float(x), 6) for x in m.elevations()}, reverse=True)]
            az = [math.radians(-m.h_fov / 2.0 + (k + 0.5) * m.h_fov / n_cols) for k in range(n_cols)]
            if m.rotation == 'cw':
                az = az[::-1]
            attrs[gp] = {'lidar_model_parameters': {
                'column_azimuths_rad': az, 'n_columns': n_cols, 'n_rows': len(el),
                'row_azimuth_offsets_rad': [0.0] * len(el), 'row_elevations_rad': el,
                'spinning_direction': 'cw' if m.rotation == 'cw' else 'ccw', 'spinning_frequency_hz': float(m.rate_hz)},
                'lidar_model_type': 'row-offset-spinning'}
    attrs['poses/default/static_poses'] = static
    return {'groups': groups, 'attrs': attrs}


def ncore_files(project, variant, cams: list, name: str) -> dict:
    """{relative path: bytes} of <name>.ncore4.zarr (directory store, zarr v2, consolidated metadata)."""
    doc = ncore_rig(project, variant, cams, name)
    root = f'{name}.ncore4.zarr'
    files = {}
    meta = {}
    for g in doc['groups']:
        k = (g + '/' if g else '') + '.zgroup'
        meta[k] = {'zarr_format': 2}
        files[f'{root}/{k}'] = json.dumps({'zarr_format': 2}, indent=4).encode('utf-8')
        if g in doc['attrs']:
            ka = (g + '/' if g else '') + '.zattrs'
            meta[ka] = doc['attrs'][g]
            files[f'{root}/{ka}'] = json.dumps(doc['attrs'][g], indent=4, sort_keys=True).encode('utf-8')
    buf = io.BytesIO()
    with lzma.open(buf, 'wb') as fh:
        fh.write(cbor_dumps({'zarr_consolidated_format': 1, 'metadata': meta}))
    files[f'{root}/.zmetadata.cbor.xz'] = buf.getvalue()
    return files


# =========================================================================== a small reader (tests, the check below)
_TOKEN = re.compile(r'\s*(?:(#[^\n]*)|("(?:[^"\\]|\\.)*")|(@[^@]*@)|(<[^>]*>)|([()\[\]{}=,])|([^\s()\[\]{}=,"<>]+))')


def parse_usda(text: str) -> dict:
    """A small USDA reader for what this module writes: {prim path: {'type', 'attrs': {name: value}, 'meta': text}}.
    Values: numbers, strings, tuples and lists of them; a relationship's or connection's target as '<path>'."""
    if not text.startswith('#usda 1.0'):
        raise ValueError('not a USDA file (no "#usda 1.0" header)')
    toks = []
    pos = len('#usda 1.0')
    while pos < len(text):
        m = _TOKEN.match(text, pos)
        if not m or m.end() == pos:
            if text[pos:].strip() == '':
                break
            raise ValueError(f'cannot read the USDA text at {pos}: {text[pos:pos + 40]!r}')
        pos = m.end()
        if m.group(1):
            continue
        toks.append(m.group(0).strip())
    i = 0

    def value():
        nonlocal i
        t = toks[i]
        if t in ('(', '['):
            close = ')' if t == '(' else ']'
            i += 1
            out = []
            while toks[i] != close:
                out.append(value())
                if toks[i] == ',':
                    i += 1
            i += 1
            return tuple(out) if t == '(' else out
        i += 1
        if t.startswith('"'):
            return json.loads(t)
        try:
            return int(t)
        except ValueError:
            try:
                return float(t)
            except ValueError:
                return t

    def skip_meta():
        nonlocal i
        depth = 0
        start = i
        while True:
            if toks[i] in ('(', '[', '{'):
                depth += 1
            elif toks[i] in (')', ']', '}'):
                depth -= 1
            i += 1
            if depth == 0:
                return ' '.join(toks[start:i])

    prims = {}

    def block(path):
        nonlocal i
        assert toks[i] == '{', toks[i:i + 5]
        i += 1
        while toks[i] != '}':
            t = toks[i]
            if t in ('def', 'over', 'class'):
                i += 1
                typ = ''
                if not toks[i].startswith('"'):
                    typ = toks[i]
                    i += 1
                name = json.loads(toks[i])
                i += 1
                meta = skip_meta() if toks[i] == '(' else ''
                p = f'{path}/{name}'
                prims[p] = {'type': typ, 'attrs': {}, 'meta': meta}
                block(p)
                continue
            # an attribute or relationship: [custom] [uniform] type name [= value] [(meta)]
            words = []
            while toks[i] not in ('=', '(', '}') and not (toks[i] in ('def', 'over', 'class') and words):
                words.append(toks[i])
                i += 1
                if len(words) >= 2 and words[-2] not in ('custom', 'uniform', 'prepend', 'append', 'delete') and \
                        toks[i] not in ('=', '('):
                    break
            name = words[-1] if words else ''
            if toks[i] == '=':
                i += 1
                prims[path]['attrs'][name] = value()
            if i < len(toks) and toks[i] == '(':
                skip_meta()
        i += 1

    prims['/'] = {'type': '', 'attrs': {}, 'meta': ''}
    if toks and toks[0] == '(':
        prims['/']['meta'] = skip_meta()
    while i < len(toks):
        if toks[i] in ('def', 'over', 'class'):
            i += 1
            typ = ''
            if not toks[i].startswith('"'):
                typ = toks[i]
                i += 1
            name = json.loads(toks[i])
            i += 1
            meta = skip_meta() if toks[i] == '(' else ''
            prims['/' + name] = {'type': typ, 'attrs': {}, 'meta': meta}
            block('/' + name)
        else:
            raise ValueError(f'unexpected {toks[i]!r} at the top level')
    return prims


def check_cameras(res: UsdResult, project, n: int = 400, seed: int = 0) -> list:
    """Each exported camera read back from the text (parse_usda) and points projected through it, against Vantage's
    own lens model: [{'tag', 'max_px', 'points'}] (the export's validation)."""
    from .lens import resolve
    prims = parse_usda(res.text)
    out = []
    rng = np.random.default_rng(seed)
    for cu in res.cameras:
        pr = prims.get(f'/Vehicle/Sensors/Cameras/{cu.prim}')
        if pr is None:
            out.append({'tag': cu.tag, 'max_px': None, 'points': 0})
            continue
        a = pr['attrs']
        M = np.array(a['xformOp:transform'], float)
        cam = project.variant.by_tag(cu.tag)
        intr = resolve(project.module_for(cam), project.lens_for(cam))
        u = rng.uniform(0, intr.width, n)
        v = rng.uniform(0, intr.height, n)
        d, th = intr.unproject(u, v)
        ok = th <= min(intr.max_theta, math.radians(85.0) if intr.projection == 'rectilinear' else intr.max_theta)
        d, u, v = d[ok], u[ok], v[ok]
        rr = rng.uniform(1.0, 30.0, len(d))
        P = cu.pos + (d * rr[:, None]) @ cu.R.T
        uu, vv = project_usd_camera({k: (v_ if not isinstance(v_, tuple) else v_) for k, v_ in a.items()}, M, P,
                                    cu.width, cu.height)
        err = np.hypot(uu - u, vv - v)
        out.append({'tag': cu.tag, 'max_px': float(err.max()) if len(err) else None, 'points': int(len(err)),
                    'fit_rms_px': cu.fit_rms_px, 'fisheye': cu.fisheye})
    return out


# =========================================================================== contract
def summary(res: UsdResult) -> dict:
    return {'schema': SCHEMA, 'project': res.project, 'variant': res.variant, 'name': res.name, **res.counts,
            'usda_bytes': len(res.text.encode('utf-8')), 'ncore_files': len(res.ncore),
            'cameras_fisheye': [c.tag for c in res.cameras if c.fisheye], 'notes': res.notes, 'seconds': res.seconds}


def to_doc(res: UsdResult, project=None) -> dict:
    return {'schema': SCHEMA, 'project': res.project, 'variant': res.variant, 'frame': FRAME,
            'settings': res.settings, 'counts': res.counts,
            'cameras': [{'tag': c.tag, 'prim': f'/Vehicle/Sensors/Cameras/{c.prim}', 'projection': c.projection,
                         'fisheye_api': OMNI_FISHEYE_API if c.fisheye else None, 'f_px': round(c.f_px, 6),
                         'cx': c.cx, 'cy': c.cy, 'kb': list(c.kb), 'fit_rms_px': round(c.fit_rms_px, 6),
                         'image_size': [c.width, c.height], **{k: round(v, 9) for k, v in c.usd_attrs().items()}}
                        for c in res.cameras],
            'files': [f'{res.name}.usda'] + sorted(res.ncore), 'sources': SOURCES, 'assumptions': ASSUMPTIONS,
            'notes': res.notes, 'units': {'length': 'm', 'usd_focal_and_aperture': 'tenths of a metre (USD: tenths '
                                                                                 'of a scene unit)',
                                           'omni_fisheye': 'pixels, OpenCV convention'}}


def files(res: UsdResult, project=None) -> list:
    out = [(f'{res.name}.usda', res.text.encode('utf-8')),
           (f'{res.name}.json', json.dumps(to_doc(res, project), indent=1).encode('utf-8'))]
    out += [(k, v) for k, v in sorted(res.ncore.items())]
    return out


def export(res: UsdResult, project, out_dir) -> list[str]:
    paths = []
    for name, data in files(res, project):
        path = os.path.join(out_dir, name)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, 'wb') as fh:
            fh.write(data)
        paths.append(path)
    return paths


def validation_checks() -> list:
    """Spec-compatible checks for vantage/validation (the lead registers them)."""
    return [{
        'id': 'usd_camera_projection',
        'title': 'An exported USD camera projects as Vantage\'s lens does',
        'model': 'core.usd_export',
        'reference': 'Independent: the USD camera\'s standard attributes (OpenUSD UsdGeomCamera: tenths of a scene '
                     'unit, aperture offsets) and NVIDIA\'s OpenCV fisheye attributes, read back from the written '
                     'text and projected by their own published formulas, against core.lens on 400 random rays per '
                     'camera of the sample\'s data rig',
        'tolerance': '0.01 px for a Kannala-Brandt or rectilinear lens (exact up to float printing); a fitted lens: '
                     'within 3 × its stated fit RMS + 0.01 px',
        'why': 'Simulators that read the USD must see the same pixels the plan does.',
        'run': 'vantage.core.usd_export:check_cameras'}]
