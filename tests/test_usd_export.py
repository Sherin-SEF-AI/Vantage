"""core.usd_export: the rig as OpenUSD (cameras with exact intrinsics, NVIDIA's fisheye attributes, lidars, radars, RF
materials), as an NCore V4 store (NuRec's input format) and a log replay's Gaussians as a ParticleField. The text is
read back with the module's own small reader; usd-core and nvidia-ncore are used only by the test_real_* checks (from
private library folders, in subprocesses)."""
import dataclasses
import io
import json
import lzma
import math
import os
import subprocess
import sys

import numpy as np
import pytest

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
USD_PYLIB = os.environ.get('VANTAGE_USD_PYLIB', '')            # a folder with usd-core, if any
NCORE_PYLIB = os.environ.get('VANTAGE_NCORE_PYLIB', '')        # a folder with nvidia-ncore, if any


def _sample(rect=False, equisolid=False):
    from vantage.core import samples
    p = samples.tigor_sample_project()
    v = next(v for v in p.variants if v.sensors)
    p.active_uid = v.uid
    base = p.lenses['h03r2']
    if rect:
        p.lenses['t_rect'] = dataclasses.replace(base, id='t_rect', projection='rectilinear', max_half_deg=40.0,
                                                 pp_offset_px=(13.0, -7.0))
        v.by_tag('R').lens_id = 't_rect'
    if equisolid:
        p.lenses['t_eqs'] = dataclasses.replace(base, id='t_eqs', projection='equisolid', max_half_deg=80.0)
        v.by_tag('QL').lens_id = 't_eqs'
    return p


def _sensor(p, tag):
    return next(s for s in p.variant.sensors if s.tag == tag)


@pytest.fixture(scope='module')
def res():
    from vantage.core import usd_export as UX
    from vantage.core.analysis import Engine
    p = _sample(rect=True, equisolid=True)
    return p, UX.run(p, Engine())


def test_settings_tolerate_junk():
    from vantage.core import usd_export as UX
    s = UX.UsdSettings.from_dict({'body': 'no', 'max_splats': -5, 'decimate_tris': 'x', 'name': '../evil',
                                  'splats': 3, 'ncore': 0})
    assert not s.body and s.max_splats == 0 and s.decimate_tris == 0 and s.name == 'vantage_rig' and not s.ncore
    assert UX.UsdSettings.from_dict(None) == UX.UsdSettings()
    assert UX.UsdSettings.from_dict(UX.UsdSettings(name='my_rig').to_dict()).name == 'my_rig'
    assert UX.ident('3 front-left') == '_3_front_left'


def test_the_text_is_well_formed_usda(res):
    from vantage.core import usd_export as UX
    p, r = res
    assert r.text.startswith('#usda 1.0\n')
    prims = UX.parse_usda(r.text)
    meta = prims['/']['meta']
    assert 'upAxis = "Z"' in meta and 'metersPerUnit = 1' in meta and 'defaultPrim = "Vehicle"' in meta
    for path in ('/Vehicle', '/Vehicle/Looks', '/Vehicle/Body', '/Vehicle/Sensors', '/Vehicle/Sensors/Cameras',
                 '/Vehicle/Sensors/Lidars/L1', '/Vehicle/Sensors/Radars/RD1', '/Vehicle/Sensors/Housings'):
        assert path in prims, path
    meshes = {k: v for k, v in prims.items() if v['type'] == 'Mesh'}
    assert meshes
    tris = 0
    for k, m in meshes.items():
        a = m['attrs']
        n = len(a['points'])
        idx = np.array(a['faceVertexIndices'])
        assert idx.min() >= 0 and idx.max() < n and len(idx) == 3 * len(a['faceVertexCounts'])
        assert a['material:binding'].startswith('</Vehicle/Looks/')
        assert prims[a['material:binding'][1:-1]]['type'] == 'Material'
        tris += len(a['faceVertexCounts'])
    assert tris == r.counts['body_triangles'] + r.counts['rig_triangles'] + r.counts['housing_triangles']
    cams = [k for k, v in prims.items() if v['type'] == 'Camera']
    assert len(cams) == 6 == r.counts['cameras']
    # balanced braces and no NaN
    assert r.text.count('{') == r.text.count('}') and 'nan' not in r.text.lower().replace('nanoseconds', '')
    with pytest.raises(ValueError, match='not a USDA'):
        UX.parse_usda('hello')


def test_camera_parameters_round_trip_and_project_as_vantage(res):
    """Each camera read back from the text projects random rays as core.lens does: Kannala-Brandt and rectilinear
    exactly, a fitted equisolid within its stated fit."""
    from vantage.core import usd_export as UX
    p, r = res
    out = {c['tag']: c for c in UX.check_cameras(r, p)}
    for tag, c in out.items():
        assert c['points'] > 100
        if c['fit_rms_px'] == 0:
            assert c['max_px'] < 1e-3, c
        else:
            assert c['max_px'] < 3 * c['fit_rms_px'] + 0.01 + 0.5, c          # a fitted lens: its own fit error
    assert out['R']['fisheye'] is False and out['QL']['fit_rms_px'] > 0
    prims = UX.parse_usda(r.text)
    a = prims['/Vehicle/Sensors/Cameras/F']['attrs']
    cu = next(c for c in r.cameras if c.tag == 'F')
    assert a[UX.OMNI_NS + ':fx'] == pytest.approx(cu.f_px, rel=1e-9)
    assert a[UX.OMNI_NS + ':cx'] == pytest.approx(cu.cx - 0.5, abs=1e-9)
    assert tuple(a[UX.OMNI_NS + ':imageSize']) == (cu.width, cu.height)
    assert [a[UX.OMNI_NS + f':k{i}'] for i in (1, 2, 3, 4)] == pytest.approx(list(cu.kb), abs=1e-9)
    assert a['focalLength'] == pytest.approx(cu.f_mm / 100.0)                 # tenths of a metre
    assert UX.OMNI_FISHEYE_API in prims['/Vehicle/Sensors/Cameras/F']['meta']
    assert UX.OMNI_FISHEYE_API not in prims['/Vehicle/Sensors/Cameras/R']['meta']


def test_the_attributes_project_by_opencvs_own_models(res):
    """Independent: cv2.fisheye.projectPoints with the omni fisheye attributes, cv2.projectPoints with the USD
    camera's focal length and aperture offsets turned into K, both against core.lens."""
    import cv2
    from vantage.core import usd_export as UX
    from vantage.core.lens import resolve
    p, r = res
    prims = UX.parse_usda(r.text)
    rng = np.random.default_rng(5)
    for cu in r.cameras:
        a = prims[f'/Vehicle/Sensors/Cameras/{cu.prim}']['attrs']
        cam = p.variant.by_tag(cu.tag)
        intr = resolve(p.module_for(cam), p.lens_for(cam))
        u = rng.uniform(0, intr.width, 200)
        v = rng.uniform(0, intr.height, 200)
        d, th = intr.unproject(u, v)
        ok = th < min(intr.max_theta, math.radians(80))
        d, u, v = d[ok], u[ok], v[ok]
        if cu.fisheye:
            if cu.fit_rms_px > 0:
                continue
            K = np.array([[a[UX.OMNI_NS + ':fx'], 0, a[UX.OMNI_NS + ':cx']], [0, a[UX.OMNI_NS + ':fy'],
                                                                             a[UX.OMNI_NS + ':cy']], [0, 0, 1]])
            D = np.array([a[UX.OMNI_NS + f':k{i}'] for i in (1, 2, 3, 4)])
            img, _ = cv2.fisheye.projectPoints((d * 3.0).reshape(-1, 1, 3), np.zeros(3), np.zeros(3), K, D)
        else:
            f, ha, va = a['focalLength'], a['horizontalAperture'], a['verticalAperture']
            ho, vo = a['horizontalApertureOffset'], a['verticalApertureOffset']
            fx, fy = f / ha * cu.width, f / va * cu.height
            cx = cu.width / 2 - ho / ha * cu.width - 0.5
            cy = cu.height / 2 + vo / va * cu.height - 0.5
            K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])
            img, _ = cv2.projectPoints((d * 3.0).reshape(-1, 1, 3), np.zeros(3), np.zeros(3), K, np.zeros(5))
        img = img.reshape(-1, 2) + 0.5
        assert np.hypot(img[:, 0] - u, img[:, 1] - v).max() < 1e-3, cu.tag


def test_the_camera_transform_is_the_mount(res):
    from vantage.core import usd_export as UX
    p, r = res
    prims = UX.parse_usda(r.text)
    for cu in r.cameras:
        M = np.array(prims[f'/Vehicle/Sensors/Cameras/{cu.prim}']['attrs']['xformOp:transform'], float)
        cam = p.variant.by_tag(cu.tag)
        assert np.allclose(M[3, :3], cam.pos(), atol=1e-9)
        # the USD camera's −Z axis (its view) is the camera's optical axis, +Y is up in the image (−v)
        view = -M[2, :3]
        assert np.allclose(view, cu.R[:, 2], atol=1e-9)
        assert np.allclose(M[1, :3], -cu.R[:, 1], atol=1e-9)
        assert np.linalg.det(M[:3, :3]) == pytest.approx(1.0)


def test_lidars_radars_and_rf_materials(res):
    from vantage.core import materials as MT, usd_export as UX
    p, r = res
    prims = UX.parse_usda(r.text)
    L = prims['/Vehicle/Sensors/Lidars/L1']['attrs']
    m = p.sensor_model_for(_sensor(p, 'L1'))
    assert L['vantage:lidar:channels'] == m.n_channels() and len(L['vantage:lidar:beamElevationsDeg']) == m.n_channels()
    assert np.allclose(np.array(L['xformOp:transform'])[3, :3], _sensor(p, 'L1').pos())
    R = prims['/Vehicle/Sensors/Radars/RD1']['attrs']
    assert R['vantage:radar:zones'] and R['vantage:radar:rangeM'] > 0
    rf = {k: v['attrs'] for k, v in prims.items() if v['type'] == 'Material' and 'vantage:rf:kind' in v['attrs']}
    assert rf and r.counts.get('rf_materials', 0) >= 1
    ghz = MT.settings_of(p).radar_ghz
    for a in rf.values():
        assert a['vantage:rf:frequencyGHz'] == pytest.approx(ghz)
    fas = UX._rf_of('fascia', MT.settings_of(p))
    assert fas['relativePermittivity'] == pytest.approx(MT.DIELECTRICS['pp_talc'][1])
    assert fas['lossTangent'] == pytest.approx(MT.tan_delta('pp_talc', ghz))
    assert fas['layers'][0][0] == 'pp_talc' and len(fas['layers']) == 1 + len(MT.FINISHES[MT.settings_of(p).paint])
    g = UX._rf_of('windscreen', MT.settings_of(p))
    assert g['relativePermittivity'] == pytest.approx(6.31) and g['lossTangent'] > 0.01
    assert UX._rf_of('steel', MT.settings_of(p)) == {'kind': 'metal', 'frequencyGHz': ghz}


def test_cbor_encoder_against_rfc8949_vectors():
    from vantage.core import usd_export as UX
    vec = [(0, '00'), (23, '17'), (24, '1818'), (100, '1864'), (1000, '1903e8'), (1000000, '1a000f4240'),
           (-1, '20'), (-100, '3863'), (1.1, 'fb3ff199999999999a'), (False, 'f4'), (True, 'f5'), (None, 'f6'),
           ('a', '6161'), ('IETF', '6449455446'), ([1, 2, 3], '83010203'),
           ({'a': 1, 'b': [2, 3]}, 'a26161016162820203'),
           ('ü', '62c3bc')]
    for v, hx in vec:
        assert UX.cbor_dumps(v).hex() == hx, v
        assert UX.cbor_loads(bytes.fromhex(hx)) == v


def test_ncore_store_layout(res):
    """The NCore V4 directory store: zarr v2 groups, the component attributes, consolidated metadata equal to the
    files, poses camera -> rig."""
    from vantage.core import usd_export as UX
    p, r = res
    root = 'vantage_rig.ncore4.zarr'
    f = r.ncore
    for g in ('', 'intrinsics/', 'intrinsics/default/', 'poses/default/static_poses/'):
        assert json.loads(f[f'{root}/{g}.zgroup']) == {'zarr_format': 2}
    top = json.loads(f[f'{root}/.zattrs'])
    assert top['version'] == 'v4' and top['sequence_id'] == 'vantage_rig'
    cam = json.loads(f[f'{root}/intrinsics/default/cameras/camera_F/.zattrs'])
    assert cam['camera_model_type'] == 'opencv-fisheye'
    cu = next(c for c in r.cameras if c.tag == 'F')
    assert cam['camera_model_parameters']['principal_point'] == [cu.cx, cu.cy]
    assert json.loads(f[f'{root}/intrinsics/default/cameras/camera_R/.zattrs'])['camera_model_type'] == 'opencv-pinhole'
    lid = json.loads(f[f'{root}/intrinsics/default/lidars/lidar_L1/.zattrs'])['lidar_model_parameters']
    el = lid['row_elevations_rad']
    assert lid['n_rows'] == len(el) and all(a > b for a, b in zip(el, el[1:]))
    az = lid['column_azimuths_rad']
    assert all((b < a) if lid['spinning_direction'] == 'cw' else (b > a) for a, b in zip(az, az[1:]))
    poses = json.loads(f[f'{root}/poses/default/static_poses/.zattrs'])
    T = np.array(poses[str(('camera_F', 'rig'))]['pose'])
    assert np.allclose(T[:3, 3], p.variant.by_tag('F').pos()) and np.allclose(T[:3, :3], cu.R)
    meta = UX.cbor_loads(lzma.decompress(f[f'{root}/.zmetadata.cbor.xz']))
    assert meta['zarr_consolidated_format'] == 1
    for k, v in meta['metadata'].items():
        assert json.loads(f[f'{root}/{k}']) == v


def test_splats_as_a_particle_field(tmp_path):
    from vantage.core import splat_reproject as SP, usd_export as UX
    n = 50
    rng = np.random.default_rng(0)
    q = rng.normal(size=(n, 4))
    q /= np.linalg.norm(q, axis=1, keepdims=True)
    sc = SP.Scene(rng.normal(size=(n, 3)).astype(np.float32), np.full((n, 3), 0.1, np.float32), q.astype(np.float32),
                  np.linspace(0.05, 1, n).astype(np.float32), rng.uniform(0, 1, (n, 3)).astype(np.float32),
                  np.where(np.arange(n) >= n - 5, 0, -1).astype(np.int32), np.zeros(n, np.int32), [7],
                  np.zeros((SP.SKY_H, SP.SKY_W, 3), np.float32), np.array([100.0, 200.0, 0.0]))
    sc.save(str(tmp_path / 'replay'))
    p = _sample()
    r = UX.run(p, None, UX.UsdSettings(splats=str(tmp_path / 'replay'), max_splats=40, ncore=False))
    prims = UX.parse_usda(r.text)
    g = prims['/Scene/Gaussians']
    assert g['type'] == 'ParticleField3DGaussianSplat'
    a = g['attrs']
    assert len(a['positions']) == r.counts['splats'] == len(a['opacities']) == len(a['orientations'])
    assert r.counts["splats"] == 35 and min(a["opacities"]) > 0.2       # the 40 most opaque, less 5 actors
    k = 0
    i = int(np.flatnonzero(np.isclose(sc.opacity, a['opacities'][k], atol=1e-6))[0])
    assert np.allclose(a['radiance:sphericalHarmonicsCoefficients'][k], (sc.colours[i] - 0.5) / UX.SH_C0, atol=1e-6)
    assert prims['/Scene']['attrs']['xformOp:translate'] == (100.0, 200.0, 0.0)
    assert any('moving actors' in s for s in r.notes)
    r2 = UX.run(p, None, UX.UsdSettings(splats=str(tmp_path / 'nothing'), ncore=False))
    assert r2.counts['splats'] == 0 and any('No log-replay scene' in s for s in r2.notes)


def test_export_writes_the_files(res, tmp_path):
    from vantage.core import usd_export as UX
    p, r = res
    paths = UX.export(r, p, str(tmp_path))
    names = {os.path.relpath(x, tmp_path) for x in paths}
    assert 'vantage_rig.usda' in names and 'vantage_rig.json' in names
    assert 'vantage_rig.ncore4.zarr/.zmetadata.cbor.xz' in names
    doc = json.load(open(tmp_path / 'vantage_rig.json'))
    assert doc['schema'] == UX.SCHEMA and doc['sources']['ncore'] and doc['assumptions']
    s = UX.summary(r)
    assert s['cameras'] == 6 and s['usda_bytes'] > 1000 and json.dumps(s)
    # an empty variant
    from vantage.core.project import Variant
    p2 = _sample()
    v = Variant(name='Empty')
    p2.variants.append(v)
    p2.active_uid = v.uid
    r2 = UX.run(p2, None)
    assert r2.counts['cameras'] == 0 and UX.parse_usda(r2.text)


def _subproc(pylib, code, *args):
    env = dict(os.environ, PYTHONPATH=os.pathsep.join([pylib, REPO]))
    return subprocess.run([sys.executable, '-c', code] + list(args), capture_output=True, text=True, env=env,
                          timeout=600)


@pytest.mark.skipif(not os.path.isdir(os.path.join(USD_PYLIB, 'pxr')), reason='usd-core not in VANTAGE_USD_PYLIB')
def test_real_usd_core_opens_the_stage_and_its_cameras_project(tmp_path):
    """usd-core (pxr) opens the stage; its own Gf.Camera frustum projects the rectilinear camera as core.lens, and
    its ParticleField schema is the splats' prim type."""
    from vantage.core import usd_export as UX
    p = _sample(rect=True)
    r = UX.run(p, None)
    UX.export(r, p, str(tmp_path))
    cu = next(c for c in r.cameras if c.tag == 'R')
    code = r'''
import sys, json, numpy as np
from pxr import Usd, UsdGeom
st = Usd.Stage.Open(sys.argv[1])
cam = st.GetPrimAtPath('/Vehicle/Sensors/Cameras/R')
M = np.array(UsdGeom.Xformable(cam).ComputeLocalToWorldTransform(Usd.TimeCode.Default()), float)
P = np.array(json.loads(sys.argv[2]), float)
Pc = (np.c_[P, np.ones(len(P))] @ np.linalg.inv(M))[:, :3]
Pm = np.array(UsdGeom.Camera(cam).GetCamera(Usd.TimeCode.Default()).frustum.ComputeProjectionMatrix(), float)
c = np.c_[Pc, np.ones(len(Pc))] @ Pm
n = c[:, :3] / c[:, 3:4]
print(json.dumps({'ok': bool(st), 'up': UsdGeom.GetStageUpAxis(st), 'mpu': UsdGeom.GetStageMetersPerUnit(st),
                  'prims': len(list(st.Traverse())), 'ndc': n[:, :2].tolist()}))
'''
    from vantage.core.lens import resolve
    cam = p.variant.by_tag('R')
    intr = resolve(p.module_for(cam), p.lens_for(cam))
    rng = np.random.default_rng(1)
    u, v = rng.uniform(0, intr.width, 100), rng.uniform(0, intr.height, 100)
    d, th = intr.unproject(u, v)
    P = cu.pos + (d * 5.0) @ cu.R.T
    out = _subproc(USD_PYLIB, code, str(tmp_path / 'vantage_rig.usda'), json.dumps(P.tolist()))
    assert out.returncode == 0, out.stderr[-2000:]
    doc = json.loads(out.stdout.strip().splitlines()[-1])
    assert doc['ok'] and doc['up'] == 'Z' and doc['mpu'] == 1.0 and doc['prims'] > 20
    n = np.array(doc['ndc'])
    uu, vv = (n[:, 0] + 1) / 2 * intr.width, (1 - n[:, 1]) / 2 * intr.height
    assert np.hypot(uu - u, vv - v).max() < 1e-3


@pytest.mark.skipif(not os.path.isdir(os.path.join(NCORE_PYLIB, 'ncore')), reason='nvidia-ncore not in '
                                                                                   'VANTAGE_NCORE_PYLIB')
def test_real_nvidia_ncore_reads_the_rig_and_projects_as_vantage(tmp_path):
    """NVIDIA's own reader (nvidia-ncore 19.9.0) opens the store; its OpenCV fisheye camera model projects points
    through the stored intrinsics and the camera -> rig pose as core.lens does."""
    from vantage.core import usd_export as UX
    p = _sample()
    r = UX.run(p, None)
    UX.export(r, p, str(tmp_path))
    from vantage.core.lens import resolve
    rng = np.random.default_rng(2)
    pts = {}
    for cu in r.cameras:
        cam = p.variant.by_tag(cu.tag)
        intr = resolve(p.module_for(cam), p.lens_for(cam))
        u, v = rng.uniform(0, intr.width, 60), rng.uniform(0, intr.height, 60)
        d, th = intr.unproject(u, v)
        ok = th < intr.max_theta * 0.98
        pts[cu.tag] = {'P': (cu.pos + (d[ok] * 7.0) @ cu.R.T).tolist(), 'uv': np.c_[u[ok], v[ok]].tolist()}
    code = r'''
import sys, json, numpy as np, torch
import numcodecs.blosc as b
if not hasattr(b, 'cbuffer_sizes'):          # numcodecs 0.17 renamed them; zarr 2.18 still imports the old names
    b.cbuffer_sizes, b.cbuffer_metainfo = b._cbuffer_sizes, b._cbuffer_metainfo
from upath import UPath
from ncore.data.v4 import SequenceComponentGroupsReader, IntrinsicsComponent, PosesComponent
from ncore.impl.sensors.camera import CameraModel
r = SequenceComponentGroupsReader([UPath(sys.argv[1])])
I = r.open_component_readers(IntrinsicsComponent.Reader)['default']
Po = r.open_component_readers(PosesComponent.Reader)['default']
pts = json.loads(open(sys.argv[2]).read())
out = {}
for tag, d in pts.items():
    T = Po.get_static_pose('camera_' + tag, 'rig')
    P = np.array(d['P']); Pc = (P - T[:3, 3]) @ T[:3, :3]
    m = CameraModel.from_parameters(I.get_camera_model_parameters('camera_' + tag), device='cpu', dtype=torch.float64)
    q = m.camera_rays_to_image_points(torch.as_tensor(Pc)).image_points.numpy()
    out[tag] = float(np.hypot(*(q - np.array(d['uv'])).T).max())
lid = I.get_lidar_model_parameters('lidar_L1')
print(json.dumps({'id': r.sequence_id, 'err': out, 'rows': int(lid.n_rows)}))
'''
    (tmp_path / 'pts.json').write_text(json.dumps(pts))
    out = _subproc(NCORE_PYLIB, code, str(tmp_path / 'vantage_rig.ncore4.zarr'), str(tmp_path / 'pts.json'))
    assert out.returncode == 0, out.stderr[-2000:]
    doc = json.loads(out.stdout.strip().splitlines()[-1])
    assert doc['id'] == 'vantage_rig' and doc['rows'] == p.sensor_model_for(_sensor(p, 'L1')).n_channels()
    assert max(doc['err'].values()) < 1e-3, doc
