"""Tests for vantage.core.loaders.

All sample files are generated on the fly in tmp_path; nothing binary is
committed.  Run with:  python3 -m pytest tests/test_loaders.py -q
"""
from __future__ import annotations

import base64
import importlib.util
import json
import math
import os
import struct
import sys
import time
import zlib

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from vantage.core.loaders import (  # noqa: E402
    SUPPORTED_EXTS, LoadedMesh, MeshLoadError, PartInfo, load_mesh, mesh_summary,
)

# ----------------------------------------------------------------------------
# shared geometry: unit cube [0,1]^3, outward-facing quads
# ----------------------------------------------------------------------------

CUBE_V = [(0, 0, 0), (1, 0, 0), (1, 1, 0), (0, 1, 0), (0, 0, 1), (1, 0, 1), (1, 1, 1), (0, 1, 1)]
CUBE_Q = [(0, 3, 2, 1), (4, 5, 6, 7), (0, 1, 5, 4), (2, 3, 7, 6), (1, 2, 6, 5), (0, 4, 7, 3)]
CUBE_T = [t for q in CUBE_Q for t in ((q[0], q[1], q[2]), (q[0], q[2], q[3]))]


def bbox(m: LoadedMesh, part: str | None = None):
    if part is None:
        v = m.vertices[np.unique(m.faces)]
    else:
        idx = [i for i, p in enumerate(m.parts) if p.name == part]
        f = m.faces[np.isin(m.face_part, idx)]
        v = m.vertices[np.unique(f)]
    return v.min(axis=0), v.max(axis=0)


def check_basic(m: LoadedMesh, ntri: int):
    assert isinstance(m, LoadedMesh)
    assert m.vertices.dtype == np.float64 and m.vertices.ndim == 2 and m.vertices.shape[1] == 3
    assert m.faces.dtype == np.int64 and m.faces.shape == (ntri, 3)
    assert m.face_part.dtype == np.int32 and m.face_part.shape == (ntri,)
    assert m.faces.min() >= 0 and m.faces.max() < len(m.vertices)
    assert m.face_part.min() >= 0 and m.face_part.max() < len(m.parts)
    assert all(isinstance(p, PartInfo) and p.name for p in m.parts)


def outward_ok(m: LoadedMesh, center) -> bool:
    """All triangle normals point away from *center* (convex shapes only)."""
    a, b, c = (m.vertices[m.faces[:, i]] for i in range(3))
    n = np.cross(b - a, c - a)
    d = (a + b + c) / 3 - np.asarray(center)
    return bool((np.einsum('ij,ij->i', n, d) > 0).all())


# ----------------------------------------------------------------------------
# OBJ
# ----------------------------------------------------------------------------

OBJ_CUBE = """# cube
mtllib cube.mtl
o Car
v 0 0 0
v 1 0 0
v 1 1 0
v 0 1 0
v 0 0 1
v 1 0 1
v 1 1 1
v 0 1 1
vt 0 0
vn 0 0 1
g Glass
usemtl glass
f 1/1/1 4/1/1 3/1/1 2/1/1
f -4//1 -3//1 -2//1 -1//1
g Body
usemtl paint
f 1 2 6 5
f 3 4 8 7
f 2/1 3/1 7/1 6/1
f 1 5 8 4
"""

MTL = """newmtl glass
Kd 0.1 0.2 0.3
d 0.4

newmtl paint
Kd 1 0 0
Tr 0.25
"""


def test_obj_cube_groups_materials(tmp_path):
    (tmp_path / 'cube.mtl').write_text(MTL)
    p = tmp_path / 'cube.obj'
    p.write_text(OBJ_CUBE)
    m = load_mesh(str(p))
    check_basic(m, 12)
    assert m.fmt == 'obj' and m.unit_hint is None and m.up_hint is None and m.forward_hint is None
    assert len(m.vertices) == 8
    assert [(pi.name, pi.material) for pi in m.parts] == [('Glass', 'glass'), ('Body', 'paint')]
    glass, body = m.parts
    assert glass.alpha == pytest.approx(0.4)
    assert glass.color == pytest.approx((0.1, 0.2, 0.3))
    assert body.alpha == pytest.approx(0.75)  # Tr 0.25 -> 1 - Tr
    assert body.color == pytest.approx((1.0, 0.0, 0.0))
    assert any('6 quads triangulated' in n for n in m.notes)
    assert mesh_summary(m) == [('Glass', 4, 'glass'), ('Body', 8, 'paint')]
    lo, hi = bbox(m)
    assert lo == pytest.approx([0, 0, 0]) and hi == pytest.approx([1, 1, 1])
    assert outward_ok(m, (0.5, 0.5, 0.5))


def test_obj_ngon_invalid_index_and_lines(tmp_path):
    p = tmp_path / 'ngon.obj'
    p.write_text("""v 0 0 0
v 1 0 0
v 1 1 0
v 0.5 1.5 0
v 0 1 0
\tf 1 2 3 4 5
f 1 2 99
l 1 2
""")
    m = load_mesh(str(p))
    check_basic(m, 3)
    joined = ' | '.join(m.notes)
    assert 'more than 4 sides' in joined
    assert 'invalid vertex indices' in joined
    assert 'line element' in joined
    assert m.parts[0].name == 'Part 1' and m.parts[0].material == ''


def test_obj_slow_paths_and_unsafe_mtllib(tmp_path):
    # vertex colours (6 values) force the per-line vertex path; '#' comment on a
    # face line forces the forgiving face path
    p = tmp_path / 'slow.obj'
    p.write_text("""mtllib ../evil.mtl
mtllib missing.mtl
v 0 0 0 1 0 0
v 1 0 0 0 1 0
v 1 1 0 0 0 1
v 0 1 0
f 1 2 3 # comment
f 1 3 4
f a b c
""")
    m = load_mesh(str(p))
    check_basic(m, 2)
    joined = ' | '.join(m.notes)
    assert 'outside the model folder' in joined
    assert 'missing.mtl not found' in joined
    assert 'malformed face line' in joined


def test_obj_malformed(tmp_path):
    p = tmp_path / 'bin.obj'
    p.write_bytes(bytes(range(256)) * 8)
    with pytest.raises(MeshLoadError):
        load_mesh(str(p))
    p2 = tmp_path / 'nofaces.obj'
    p2.write_text('v 0 0 0\nv 1 0 0\nv 0 1 0\n')
    with pytest.raises(MeshLoadError, match='No triangles found in nofaces.obj'):
        load_mesh(str(p2))
    p3 = tmp_path / 'badidx.obj'
    p3.write_text('v 0 0 0\nv 1 0 0\nv 0 1 0\nf 4 5 6\nf -7 1 2\n')
    with pytest.raises(MeshLoadError, match='No triangles'):
        load_mesh(str(p3))


PERF = pytest.mark.skipif(os.environ.get('VANTAGE_SKIP_PERF') == '1',
                          reason='performance test disabled via VANTAGE_SKIP_PERF=1')


@PERF
def test_obj_perf_200k_triangles(tmp_path):
    n = 317  # 316*316*2 = 199,712 triangles
    xs, ys = np.meshgrid(np.arange(n, dtype=float), np.arange(n, dtype=float))
    V = np.stack([xs.ravel() * 0.01, ys.ravel() * 0.01, np.sin(xs.ravel() * 0.1) * 0.05], 1)
    lines = ['mtllib none.mtl', 'o Grid', 'usemtl paint']
    lines += [f'v {a:.6f} {b:.6f} {c:.6f}' for a, b, c in V]
    lines += ['vt 0 0', 'vn 0 0 1']
    idx = np.arange(n * n).reshape(n, n) + 1
    a, b, c, d = (idx[:-1, :-1].ravel(), idx[:-1, 1:].ravel(), idx[1:, 1:].ravel(), idx[1:, :-1].ravel())
    half = len(a) // 2
    for i in range(len(a)):
        if i < half:  # v/vt/vn triangles
            lines.append(f'f {a[i]}/1/1 {b[i]}/1/1 {c[i]}/1/1')
            lines.append(f'f {a[i]}/1/1 {c[i]}/1/1 {d[i]}/1/1')
        else:  # plain quads
            lines.append(f'f {a[i]} {b[i]} {c[i]} {d[i]}')
    p = tmp_path / 'big.obj'
    p.write_text('\n'.join(lines) + '\n')
    t0 = time.perf_counter()
    m = load_mesh(str(p))
    dt = time.perf_counter() - t0
    assert len(m.faces) == (n - 1) * (n - 1) * 2
    assert dt < 2.5, f'OBJ load took {dt:.2f}s'


# ----------------------------------------------------------------------------
# STL
# ----------------------------------------------------------------------------

def stl_binary(tris, header=b'solid exported by a CAD tool (actually binary)', count=None) -> bytes:
    out = bytearray(header.ljust(80, b' ')[:80])
    out += struct.pack('<I', len(tris) if count is None else count)
    for t in tris:
        out += struct.pack('<3f', 0, 0, 0)
        for vi in t:
            out += struct.pack('<3f', *CUBE_V[vi])
        out += b'\x00\x00'
    return bytes(out)


def stl_ascii_solid(name, tris, offset=(0, 0, 0)) -> str:
    s = [f'solid {name}']
    for t in tris:
        s.append('  facet normal 0 0 0\n    outer loop')
        for vi in t:
            x, y, z = (CUBE_V[vi][k] + offset[k] for k in range(3))
            s.append(f'      vertex {x:e} {y:e} {z:e}')
        s.append('    endloop\n  endfacet')
    s.append(f'endsolid {name}')
    return '\n'.join(s) + '\n'


def test_stl_binary_welded(tmp_path):
    p = tmp_path / 'cube.stl'
    p.write_bytes(stl_binary(CUBE_T))  # header starts with "solid" on purpose
    m = load_mesh(str(p))
    check_basic(m, 12)
    assert m.fmt == 'stl' and m.unit_hint is None
    assert len(m.vertices) == 8  # welded
    assert len(m.parts) == 1
    assert outward_ok(m, (0.5, 0.5, 0.5))


def test_stl_ascii_two_solids(tmp_path):
    p = tmp_path / 'two.stl'
    p.write_text(stl_ascii_solid('body', CUBE_T) + stl_ascii_solid('wheel', CUBE_T[:4], (3, 0, 0)))
    m = load_mesh(str(p))
    check_basic(m, 16)
    assert [pi.name for pi in m.parts] == ['body', 'wheel']
    assert mesh_summary(m) == [('body', 12, ''), ('wheel', 4, '')]
    assert len(m.vertices) == 8 + 8  # bottom+top quads of the shifted cube touch all 8 corners
    lo, hi = bbox(m, 'wheel')
    assert lo[0] == pytest.approx(3.0)


def test_stl_malformed(tmp_path):
    full = stl_binary(CUBE_T, header=b'binary')
    p = tmp_path / 'trunc.stl'
    p.write_bytes(full[:84 + 50 * 5 + 7])
    with pytest.raises(MeshLoadError, match='truncated'):
        load_mesh(str(p))
    p2 = tmp_path / 'badnum.stl'
    p2.write_text('solid x\nfacet normal 0 0 0\nouter loop\nvertex 0 0 zz\nvertex 1 0 0\nvertex 0 1 0\n'
                  'endloop\nendfacet\nendsolid x\n')
    with pytest.raises(MeshLoadError):
        load_mesh(str(p2))
    p3 = tmp_path / 'tiny.stl'
    p3.write_bytes(b'\x01\x02\x03')
    with pytest.raises(MeshLoadError):
        load_mesh(str(p3))
    p4 = tmp_path / 'huge.stl'
    p4.write_bytes(b'\x00' * 80 + struct.pack('<I', 0xFFFFFFFF) + b'\x00' * 50)
    with pytest.raises(MeshLoadError):
        load_mesh(str(p4))


# ----------------------------------------------------------------------------
# PLY
# ----------------------------------------------------------------------------

def test_ply_ascii_with_extra_props(tmp_path):
    lines = ['ply', 'format ascii 1.0', 'comment made by a test', 'element vertex 8',
             'property float x', 'property float y', 'property float z', 'property uchar red',
             'element face 6', 'property list uchar int vertex_indices', 'property uchar flags',
             'end_header']
    lines += [f'{x} {y} {z} 255' for x, y, z in CUBE_V]
    lines += [f'4 {q[0]} {q[1]} {q[2]} {q[3]} 7' for q in CUBE_Q]
    p = tmp_path / 'cube.ply'
    p.write_text('\n'.join(lines) + '\n')
    m = load_mesh(str(p))
    check_basic(m, 12)
    assert m.fmt == 'ply'
    assert any('6 quads' in n for n in m.notes)
    assert outward_ok(m, (0.5, 0.5, 0.5))


def test_ply_ascii_uniform_fast_path(tmp_path):
    lines = ['ply', 'format ascii 1.0', 'element vertex 8', 'property double x', 'property double y',
             'property double z', 'element face 12', 'property list uchar uint vertex_index', 'end_header']
    lines += [f'{x} {y} {z}' for x, y, z in CUBE_V]
    lines += [f'3 {a} {b} {c}' for a, b, c in CUBE_T]
    p = tmp_path / 'tri.ply'
    p.write_text('\n'.join(lines) + '\n')
    m = load_mesh(str(p))
    check_basic(m, 12)
    assert outward_ok(m, (0.5, 0.5, 0.5))


def ply_binary(endian: str, faces, vtype='d', count_type=('uchar', 'B'), idx_type=('int', 'i'),
               extra_vertex=True, face_count=None) -> bytes:
    vname = {'d': 'double', 'f': 'float'}[vtype]
    hdr = ['ply', f'format binary_{"little" if endian == "<" else "big"}_endian 1.0',
           'element vertex 8', f'property {vname} x', f'property {vname} y', f'property {vname} z']
    if extra_vertex:
        hdr.append('property float nx')
    hdr += [f'element face {len(faces) if face_count is None else face_count}',
            f'property list {count_type[0]} {idx_type[0]} vertex_indices', 'end_header']
    out = bytearray(('\n'.join(hdr) + '\n').encode())
    for v in CUBE_V:
        out += struct.pack(f'{endian}3{vtype}', *v)
        if extra_vertex:
            out += struct.pack(f'{endian}f', 0.5)
    for f in faces:
        out += struct.pack(f'{endian}{count_type[1]}', len(f))
        out += struct.pack(f'{endian}{len(f)}{idx_type[1]}', *f)
    return bytes(out)


def test_ply_binary_le_mixed_faces(tmp_path):
    faces = list(CUBE_Q[:5]) + [CUBE_T[10], CUBE_T[11]]  # 5 quads + 2 tris -> slow path
    p = tmp_path / 'le.ply'
    p.write_bytes(ply_binary('<', faces))
    m = load_mesh(str(p))
    check_basic(m, 12)
    assert outward_ok(m, (0.5, 0.5, 0.5))


def test_ply_binary_be_uniform(tmp_path):
    p = tmp_path / 'be.ply'
    p.write_bytes(ply_binary('>', CUBE_Q, vtype='f', count_type=('uint8', 'B'),
                             idx_type=('ushort', 'H'), extra_vertex=False))
    m = load_mesh(str(p))
    check_basic(m, 12)
    lo, hi = bbox(m)
    assert lo == pytest.approx([0, 0, 0]) and hi == pytest.approx([1, 1, 1])


def test_ply_malformed(tmp_path):
    p = tmp_path / 'absurd.ply'
    p.write_bytes(ply_binary('<', CUBE_Q, face_count=10 ** 12))
    with pytest.raises(MeshLoadError, match='plausible'):
        load_mesh(str(p))
    p2 = tmp_path / 'trunc.ply'
    p2.write_bytes(ply_binary('<', CUBE_Q)[:-10])
    with pytest.raises(MeshLoadError, match='truncated'):
        load_mesh(str(p2))
    p3 = tmp_path / 'noend.ply'
    p3.write_text('ply\nformat ascii 1.0\nelement vertex 3\nproperty float x\n')
    with pytest.raises(MeshLoadError):
        load_mesh(str(p3))
    p4 = tmp_path / 'trunc_ascii.ply'
    p4.write_text('ply\nformat ascii 1.0\nelement vertex 3\nproperty float x\nproperty float y\n'
                  'property float z\nelement face 1\nproperty list uchar int vertex_indices\nend_header\n'
                  '0 0 0\n1 0 0\n')
    with pytest.raises(MeshLoadError, match='truncated'):
        load_mesh(str(p4))
    p5 = tmp_path / 'huge_list.ply'
    p5.write_bytes(ply_binary('<', [(0, 1, 2)], count_type=('uint', 'I'))[:-12]
                   + struct.pack('<I', 0xFFFFFFF0) + b'\x00' * 8)
    with pytest.raises(MeshLoadError):
        load_mesh(str(p5))


# ----------------------------------------------------------------------------
# glTF / GLB
# ----------------------------------------------------------------------------

def cube_buffer() -> bytes:
    pos = np.array(CUBE_V, dtype='<f4').tobytes()                   # 96 bytes
    idx = np.array(CUBE_T, dtype='<u2').reshape(-1).tobytes()        # 72 bytes
    return pos + idx


def gltf_doc(uri: str | None, extra: dict | None = None) -> dict:
    buf = cube_buffer()
    q = math.sqrt(0.5)
    doc = {
        'asset': {'version': '2.0', 'generator': 'vantage-tests'},
        'scene': 0,
        'scenes': [{'nodes': [0]}],
        'nodes': [
            {'name': 'Root', 'translation': [0, 0, 5], 'children': [1, 2]},
            {'name': 'Left', 'mesh': 0, 'translation': [10, 0, 0], 'scale': [2, 2, 2]},
            {'name': 'Right', 'mesh': 0, 'translation': [-10, 0, 0], 'rotation': [0, 0, q, q]},
        ],
        'meshes': [{'name': 'CubeMesh', 'primitives': [
            {'attributes': {'POSITION': 0}, 'indices': 1, 'material': 0},
            {'attributes': {'POSITION': 0}, 'indices': 1, 'mode': 1},  # lines -> skipped
        ]}],
        'materials': [{'name': 'Glass', 'alphaMode': 'BLEND',
                       'pbrMetallicRoughness': {'baseColorFactor': [0.2, 0.4, 0.6, 0.5]}}],
        'accessors': [
            {'bufferView': 0, 'componentType': 5126, 'count': 8, 'type': 'VEC3',
             'min': [0, 0, 0], 'max': [1, 1, 1]},
            {'bufferView': 1, 'componentType': 5123, 'count': 36, 'type': 'SCALAR'},
        ],
        'bufferViews': [
            {'buffer': 0, 'byteOffset': 0, 'byteLength': 96, 'target': 34962},
            {'buffer': 0, 'byteOffset': 96, 'byteLength': 72, 'target': 34963},
        ],
        'buffers': [{'byteLength': len(buf)}],
    }
    if uri is not None:
        doc['buffers'][0]['uri'] = uri
    if extra:
        doc.update(extra)
    return doc


def data_uri(buf: bytes) -> str:
    return 'data:application/octet-stream;base64,' + base64.b64encode(buf).decode()


def make_glb(doc: dict, bin_data: bytes) -> bytes:
    j = json.dumps(doc).encode()
    j += b' ' * (-len(j) % 4)
    bd = bin_data + b'\x00' * (-len(bin_data) % 4)
    total = 12 + 8 + len(j) + 8 + len(bd)
    return (struct.pack('<4sII', b'glTF', 2, total) + struct.pack('<II', len(j), 0x4E4F534A) + j
            + struct.pack('<II', len(bd), 0x004E4942) + bd)


def check_gltf_instances(m: LoadedMesh):
    check_basic(m, 24)
    assert m.unit_hint == 'm' and m.up_hint == '+Y' and m.forward_hint == '+Z'
    assert [p.name for p in m.parts] == ['Left', 'Right']
    for p in m.parts:
        assert p.material == 'Glass'
        assert p.alpha == pytest.approx(0.5)
        assert p.color == pytest.approx((0.2, 0.4, 0.6))
    lo, hi = bbox(m, 'Left')      # scale 2, +(10,0,0), parent +(0,0,5)
    assert lo == pytest.approx([10, 0, 5]) and hi == pytest.approx([12, 2, 7])
    lo, hi = bbox(m, 'Right')     # 90 deg about Z: (x,y,z) -> (-y,x,z)
    assert lo == pytest.approx([-11, 0, 5], abs=1e-6) and hi == pytest.approx([-10, 1, 6], abs=1e-6)
    left_idx = [i for i, p in enumerate(m.parts) if p.name == 'Left'][0]
    corner = m.vertices[np.unique(m.faces[m.face_part == left_idx])]
    assert np.isclose(corner, [12, 0, 5]).all(axis=1).any()   # (1,0,0) of the Left instance
    assert any('line primitive' in n for n in m.notes)
    assert mesh_summary(m) == [('Left', 12, 'Glass'), ('Right', 12, 'Glass')]


def test_gltf_data_uri_instancing(tmp_path):
    p = tmp_path / 'car.gltf'
    p.write_text(json.dumps(gltf_doc(data_uri(cube_buffer()))))
    m = load_mesh(str(p))
    assert m.fmt == 'gltf'
    check_gltf_instances(m)


def test_glb(tmp_path):
    p = tmp_path / 'car.glb'
    p.write_bytes(make_glb(gltf_doc(None), cube_buffer()))
    m = load_mesh(str(p))
    assert m.fmt == 'glb'
    check_gltf_instances(m)


def test_gltf_external_bin(tmp_path):
    (tmp_path / 'my cube.bin').write_bytes(cube_buffer())
    p = tmp_path / 'ext.gltf'
    p.write_text(json.dumps(gltf_doc('my%20cube.bin')))
    check_gltf_instances(load_mesh(str(p)))


def test_gltf_strip_fan_matrix_stride_sparse(tmp_path):
    # interleaved position + dummy normal (stride 24) for a unit quad in z=0
    quad = [(0, 0, 0), (1, 0, 0), (0, 1, 0), (1, 1, 0)]
    inter = b''.join(struct.pack('<3f', *v) + struct.pack('<3f', 0, 0, 1) for v in quad)
    fan_pos = np.array([(0, 0, 0), (1, 0, 0), (1, 1, 0), (0, 1, 0), (-1, 0.5, 0)], '<f4').tobytes()
    sp_idx = np.array([3], '<u1').tobytes() + b'\x00\x00\x00'
    sp_val = np.array([(1, 1, 3)], '<f4').tobytes()   # sparse: lift vertex 3 to z=3
    buf = inter + fan_pos + sp_idx + sp_val
    o_fan = len(inter)
    o_spi = o_fan + len(fan_pos)
    o_spv = o_spi + len(sp_idx)
    mat = [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 100, 0, 0, 1]  # column-major, translate x+100
    doc = {
        'asset': {'version': '2.0'},
        'scenes': [{'nodes': [0, 1]}],
        'nodes': [{'name': 'Strip', 'mesh': 0, 'matrix': mat}, {'mesh': 1}],
        'meshes': [
            {'name': 'StripMesh', 'primitives': [
                {'attributes': {'POSITION': 0}, 'mode': 5},
                {'attributes': {'POSITION': 2}, 'mode': 5},
            ]},
            {'name': 'FanMesh', 'primitives': [{'attributes': {'POSITION': 1}, 'mode': 6}]},
        ],
        'accessors': [
            {'bufferView': 0, 'componentType': 5126, 'count': 4, 'type': 'VEC3'},
            {'bufferView': 1, 'componentType': 5126, 'count': 5, 'type': 'VEC3'},
            {'bufferView': 0, 'componentType': 5126, 'count': 4, 'type': 'VEC3',
             'sparse': {'count': 1, 'indices': {'bufferView': 2, 'componentType': 5121},
                        'values': {'bufferView': 3}}},
        ],
        'bufferViews': [
            {'buffer': 0, 'byteOffset': 0, 'byteLength': len(inter), 'byteStride': 24},
            {'buffer': 0, 'byteOffset': o_fan, 'byteLength': len(fan_pos)},
            {'buffer': 0, 'byteOffset': o_spi, 'byteLength': 1},
            {'buffer': 0, 'byteOffset': o_spv, 'byteLength': len(sp_val)},
        ],
        'buffers': [{'byteLength': len(buf), 'uri': data_uri(buf)}],
    }
    p = tmp_path / 'prims.gltf'
    p.write_text(json.dumps(doc))
    m = load_mesh(str(p))
    check_basic(m, 2 + 2 + 3)
    names = [pi.name for pi in m.parts]
    assert names == ['Strip', 'FanMesh']  # unnamed node falls back to mesh name
    lo, hi = bbox(m, 'Strip')
    assert lo == pytest.approx([100, 0, 0]) and hi == pytest.approx([101, 1, 3])
    # strip winding: both triangles face +z for the flat strip
    strip_idx = names.index('Strip')
    f = m.faces[m.face_part == strip_idx]
    a, b, c = (m.vertices[f[:, i]] for i in range(3))
    nz = np.cross(b - a, c - a)[:, 2]
    assert (nz[np.abs(nz) > 0] > 0).sum() >= 2
    assert m.parts[0].material == '' and m.parts[0].color == pytest.approx((0.8, 0.8, 0.8))


def test_gltf_malformed(tmp_path):
    buf = cube_buffer()
    cases = {
        'escape.gltf': json.dumps(gltf_doc('../outside.bin')),
        'abs.gltf': json.dumps(gltf_doc('/etc/passwd')),
        'nonb64.gltf': json.dumps(gltf_doc('data:application/octet-stream,AAAA')),
        'missing.gltf': json.dumps(gltf_doc('nothere.bin')),
        'badjson.gltf': '{"asset": {"version": "2.0"}, "nodes": [',
        'v1.gltf': json.dumps({'asset': {'version': '1.0'}}),
    }
    (tmp_path.parent / 'outside.bin').write_bytes(buf)
    for fname, text in cases.items():
        p = tmp_path / fname
        p.write_text(text)
        with pytest.raises(MeshLoadError):
            load_mesh(str(p))
    # Draco: both via extensionsRequired and on the primitive
    d = gltf_doc(data_uri(buf), {'extensionsUsed': ['KHR_draco_mesh_compression'],
                                 'extensionsRequired': ['KHR_draco_mesh_compression']})
    p = tmp_path / 'draco.gltf'
    p.write_text(json.dumps(d))
    with pytest.raises(MeshLoadError, match='compression'):
        load_mesh(str(p))
    d = gltf_doc(data_uri(buf))
    d['meshes'][0]['primitives'][0]['extensions'] = {'KHR_draco_mesh_compression': {'bufferView': 0}}
    p.write_text(json.dumps(d))
    with pytest.raises(MeshLoadError, match='Draco'):
        load_mesh(str(p))
    d = gltf_doc(data_uri(buf), {'extensionsRequired': ['EXT_meshopt_compression']})
    p.write_text(json.dumps(d))
    with pytest.raises(MeshLoadError, match='meshopt'):
        load_mesh(str(p))
    # absurd accessor count
    d = gltf_doc(data_uri(buf))
    d['accessors'][0]['count'] = 10 ** 12
    p = tmp_path / 'absurd.gltf'
    p.write_text(json.dumps(d))
    with pytest.raises(MeshLoadError):
        load_mesh(str(p))
    # accessor larger than its bufferView
    d = gltf_doc(data_uri(buf))
    d['accessors'][1]['count'] = 1000
    p.write_text(json.dumps(d))
    with pytest.raises(MeshLoadError):
        load_mesh(str(p))
    # truncated GLB
    glb = make_glb(gltf_doc(None), buf)
    p = tmp_path / 'trunc.glb'
    p.write_bytes(glb[:len(glb) // 2])
    with pytest.raises(MeshLoadError, match='truncated'):
        load_mesh(str(p))
    p.write_bytes(glb[:10])
    with pytest.raises(MeshLoadError):
        load_mesh(str(p))
    # out-of-range indices are dropped, not crashing
    d = gltf_doc(data_uri(buf[:96] + np.array([0, 1, 2, 0, 1, 900], '<u2').tobytes()))
    d['accessors'][1]['count'] = 6
    d['bufferViews'][1]['byteLength'] = 12
    d['buffers'][0]['byteLength'] = 108
    p = tmp_path / 'badidx.gltf'
    p.write_text(json.dumps(d))
    m = load_mesh(str(p))
    check_basic(m, 2)  # one valid triangle per instance
    assert any('invalid vertex indices' in n for n in m.notes)


# ----------------------------------------------------------------------------
# FBX
# ----------------------------------------------------------------------------

PVI = [v for q in CUBE_Q for v in (q[0], q[1], q[2], ~q[3])]
FLAT_V = [c for v in CUBE_V for c in v]


def fbx_ascii(pvi_split=True) -> str:
    pvi = ','.join(map(str, PVI))
    if pvi_split:  # long arrays wrap onto continuation lines
        cut = pvi.index(',', len(pvi) // 2)
        pvi = pvi[:cut + 1] + '\n\t\t\t' + pvi[cut + 1:]
    return f"""; FBX 7.4.0 project file
; ----------------------------------------------------
FBXHeaderExtension:  {{
\tFBXHeaderVersion: 1003
\tFBXVersion: 7400
}}
GlobalSettings:  {{
\tVersion: 1000
\tProperties70:  {{
\t\tP: "UpAxis", "int", "Integer", "",2
\t\tP: "UpAxisSign", "int", "Integer", "",1
\t\tP: "FrontAxis", "int", "Integer", "",1
\t\tP: "FrontAxisSign", "int", "Integer", "",-1
\t\tP: "UnitScaleFactor", "double", "Number", "",100
\t}}
}}
Objects:  {{
\tGeometry: 1000, "Geometry::CubeMesh", "Mesh" {{
\t\tVertices: *24 {{
\t\t\ta: {','.join(map(str, FLAT_V))}
\t\t}}
\t\tPolygonVertexIndex: *24 {{
\t\t\ta: {pvi}
\t\t}}
\t\tLayerElementMaterial: 0 {{
\t\t\tVersion: 101
\t\t\tMappingInformationType: "AllSame"
\t\t\tReferenceInformationType: "IndexToDirect"
\t\t\tMaterials: *1 {{
\t\t\t\ta: 0
\t\t\t}}
\t\t}}
\t}}
\tModel: 2000, "Model::Body", "Mesh" {{
\t\tVersion: 232
\t\tProperties70:  {{
\t\t\tP: "Lcl Translation", "Lcl Translation", "", "A",1,2,3
\t\t\tP: "Lcl Rotation", "Lcl Rotation", "", "A",0,0,90
\t\t}}
\t\tShading: T
\t}}
\tModel: 3000, "Model::Car", "Null" {{
\t\tProperties70:  {{
\t\t\tP: "Lcl Scaling", "Lcl Scaling", "", "A",2,2,2
\t\t}}
\t}}
\tMaterial: 4000, "Material::Paint", "" {{
\t\tProperties70:  {{
\t\t\tP: "DiffuseColor", "Color", "", "A",0.5,0.25,0
\t\t\tP: "Opacity", "double", "Number", "",0.3
\t\t}}
\t}}
}}
Connections:  {{
\t;Model::Car, Model::RootNode
\tC: "OO",3000,0
\tC: "OO",2000,3000
\tC: "OO",1000,2000
\tC: "OO",4000,2000
}}
"""


def check_fbx_scene(m: LoadedMesh):
    check_basic(m, 12)
    assert m.fmt == 'fbx'
    assert len(m.parts) == 1
    part = m.parts[0]
    assert part.name == 'Body' and part.material == 'Paint'
    assert part.alpha == pytest.approx(0.3)
    assert part.color == pytest.approx((0.5, 0.25, 0.0))
    # world = S(2) @ T(1,2,3) @ Rz(90): cube -> x 0..2, y 4..6, z 6..8
    lo, hi = bbox(m)
    assert lo == pytest.approx([0, 4, 6], abs=1e-9) and hi == pytest.approx([2, 6, 8], abs=1e-9)
    # vertex (1,0,0) -> Rz -> (0,1,0) -> +T -> (1,3,3) -> *2 -> (2,6,6)
    assert np.isclose(m.vertices, [2, 6, 6]).all(axis=1).any()
    assert outward_ok(m, (1, 5, 7))
    assert any('6 quads' in n for n in m.notes)


def test_fbx_ascii(tmp_path):
    p = tmp_path / 'car.fbx'
    p.write_text(fbx_ascii())
    m = load_mesh(str(p))
    check_fbx_scene(m)
    assert m.unit_hint == 'm' and m.up_hint == '+Z' and m.forward_hint == '-Y'


def test_fbx_ascii_v6_embedded_geometry(tmp_path):
    pvi = ','.join(map(str, PVI))
    text = f"""; FBX 6.1.0 project file
FBXHeaderExtension:  {{
\tFBXHeaderVersion: 1003
\tFBXVersion: 6100
}}
Objects:  {{
\tModel: "Model::Cube", "Mesh" {{
\t\tVersion: 232
\t\tProperties60:  {{
\t\t\tProperty: "Lcl Translation", "Lcl Translation", "A+",5,0,0
\t\t\tProperty: "Lcl Scaling", "Lcl Scaling", "A+",1,1,3
\t\t}}
\t\tVertices: {','.join(map(str, FLAT_V[:12]))},
\t\t{','.join(map(str, FLAT_V[12:]))}
\t\tPolygonVertexIndex: {pvi}
\t}}
\tMaterial: "Material::Red", "" {{
\t\tProperties60:  {{
\t\t\tProperty: "DiffuseColor", "ColorRGB", "",1,0,0
\t\t\tProperty: "Opacity", "double", "",0.5
\t\t}}
\t}}
}}
Connections:  {{
\tConnect: "OO", "Model::Cube", "Model::Scene"
\tConnect: "OO", "Material::Red", "Model::Cube"
}}
"""
    p = tmp_path / 'v6.fbx'
    p.write_text(text)
    m = load_mesh(str(p))
    check_basic(m, 12)
    assert [(pi.name, pi.material, pi.alpha) for pi in m.parts] == [('Cube', 'Red', 0.5)]
    assert m.parts[0].color == pytest.approx((1, 0, 0))
    lo, hi = bbox(m)
    assert lo == pytest.approx([5, 0, 0]) and hi == pytest.approx([6, 1, 3])
    assert m.unit_hint is None and m.up_hint is None


# --- minimal binary FBX writer ------------------------------------------------

def _fbx_prop(p) -> bytes:
    t, v = p[0], p[1]
    if t == 'S':
        raw = v.encode() if isinstance(v, str) else v
        return b'S' + struct.pack('<I', len(raw)) + raw
    if t in 'YCIFDL':
        fmt = {'Y': '<h', 'C': '<B', 'I': '<i', 'F': '<f', 'D': '<d', 'L': '<q'}[t]
        return t.encode() + struct.pack(fmt, v)
    # arrays: (t, values, compress?, override_length)
    compress = len(p) > 2 and p[2]
    fmt = {'d': 'd', 'f': 'f', 'i': 'i', 'l': 'q', 'b': 'B'}[t]
    payload = struct.pack(f'<{len(v)}{fmt}', *v)
    enc = 0
    if compress:
        payload, enc = zlib.compress(payload), 1
    n = p[3] if len(p) > 3 else len(v)
    return t.encode() + struct.pack('<III', n, enc, len(payload)) + payload


class FbxBinaryWriter:
    def __init__(self, version: int):
        self.version = version
        self.wide = version >= 7500

    def _write(self, out: bytearray, node) -> None:
        name, props, children = node
        hsz = 25 if self.wide else 13
        pb = b''.join(_fbx_prop(p) for p in props)
        nb = name.encode()
        start = len(out)
        out += b'\x00' * hsz + nb + pb
        if children:
            for c in children:
                self._write(out, c)
            out += b'\x00' * hsz
        fmt = '<QQQB' if self.wide else '<IIIB'
        out[start:start + hsz] = struct.pack(fmt, len(out), len(props), len(pb), len(nb))

    def build(self, nodes) -> bytes:
        out = bytearray(b'Kaydara FBX Binary  \x00\x1a\x00' + struct.pack('<I', self.version))
        for n in nodes:
            self._write(out, n)
        out += b'\x00' * (25 if self.wide else 13)
        out += b'\xfa\xbc\xab\x09\xd0\xc8\xd4\x66\xb1\x76\xfb\x83\x1c\xf7\x26\x7e' + b'\x00' * 4
        return bytes(out)


def P(name, typ, *vals):
    props = [('S', name), ('S', typ), ('S', ''), ('S', 'A')]
    for v in vals:
        props.append(('I', v) if isinstance(v, int) else ('D', float(v)))
    return ('P', props, [])


def fbx_binary_scene(version: int, compress_pvi=True, pvi_len=None) -> bytes:
    geom = ('Geometry', [('L', 1000), ('S', b'CubeMesh\x00\x01Geometry'), ('S', 'Mesh')], [
        ('Vertices', [('d', FLAT_V, False)], []),
        ('PolygonVertexIndex', [('i', PVI, compress_pvi) + ((pvi_len,) if pvi_len else ())], []),
        ('LayerElementMaterial', [('I', 0)], [
            ('MappingInformationType', [('S', 'ByPolygon')], []),
            ('Materials', [('i', [0, 0, 0, 1, 1, 1], True)], []),
        ]),
    ])
    body = ('Model', [('L', 2000), ('S', b'Body\x00\x01Model'), ('S', 'Mesh')], [
        ('Version', [('I', 232)], []),
        ('Properties70', [], [P('Lcl Translation', 'Lcl Translation', 1, 2, 3),
                              P('Lcl Rotation', 'Lcl Rotation', 0, 0, 90)]),
    ])
    car = ('Model', [('L', 3000), ('S', b'Car\x00\x01Model'), ('S', 'Null')], [
        ('Properties70', [], [P('Lcl Scaling', 'Lcl Scaling', 2, 2, 2)]),
    ])
    paint = ('Material', [('L', 4000), ('S', b'Paint\x00\x01Material'), ('S', '')], [
        ('Properties70', [], [P('DiffuseColor', 'Color', 0.5, 0.25, 0.0), P('Opacity', 'double', 0.3)]),
    ])
    glass = ('Material', [('L', 4001), ('S', b'Glass\x00\x01Material'), ('S', '')], [
        ('Properties70', [], [P('TransparencyFactor', 'double', 0.6)]),
    ])
    gs = ('GlobalSettings', [], [
        ('Version', [('I', 1000)], []),
        ('Properties70', [], [P('UpAxis', 'int', 1), P('UpAxisSign', 'int', 1),
                              P('FrontAxis', 'int', 2), P('FrontAxisSign', 'int', 1),
                              P('UnitScaleFactor', 'double', 1.0)]),
    ])
    conns = ('Connections', [], [
        ('C', [('S', 'OO'), ('L', 3000), ('L', 0)], []),
        ('C', [('S', 'OO'), ('L', 2000), ('L', 3000)], []),
        ('C', [('S', 'OO'), ('L', 1000), ('L', 2000)], []),
        ('C', [('S', 'OO'), ('L', 4000), ('L', 2000)], []),
        ('C', [('S', 'OO'), ('L', 4001), ('L', 2000)], []),
    ])
    header = ('FBXHeaderExtension', [], [('FBXHeaderVersion', [('I', 1003)], []),
                                         ('FBXVersion', [('I', version)], [])])
    objects = ('Objects', [], [geom, body, car, paint, glass])
    return FbxBinaryWriter(version).build([header, gs, objects, conns])


@pytest.mark.parametrize('version', [7400, 7500, 7700])
def test_fbx_binary(tmp_path, version):
    p = tmp_path / f'car{version}.fbx'
    p.write_bytes(fbx_binary_scene(version))
    m = load_mesh(str(p))
    check_basic(m, 12)
    assert m.unit_hint == 'cm' and m.up_hint == '+Y' and m.forward_hint == '+Z'
    assert [(pi.name, pi.material) for pi in m.parts] == [('Body', 'Paint'), ('Body', 'Glass')]
    assert m.parts[0].alpha == pytest.approx(0.3)
    assert m.parts[1].alpha == pytest.approx(0.4)   # 1 - TransparencyFactor
    assert mesh_summary(m) == [('Body', 12, 'Paint, Glass')]
    lo, hi = bbox(m)
    assert lo == pytest.approx([0, 4, 6], abs=1e-9) and hi == pytest.approx([2, 6, 8], abs=1e-9)
    assert np.isclose(m.vertices, [2, 6, 6]).all(axis=1).any()
    assert outward_ok(m, (1, 5, 7))


def test_fbx_malformed(tmp_path):
    good = fbx_binary_scene(7400)
    p = tmp_path / 'trunc.fbx'
    p.write_bytes(good[:len(good) // 2])
    with pytest.raises(MeshLoadError, match='truncated'):
        load_mesh(str(p))
    p.write_bytes(good[:20])
    with pytest.raises(MeshLoadError):
        load_mesh(str(p))
    # array header claims a billion elements
    p2 = tmp_path / 'absurd.fbx'
    p2.write_bytes(fbx_binary_scene(7500, pvi_len=1_000_000_000))
    with pytest.raises(MeshLoadError):
        load_mesh(str(p2))
    # corrupt zlib stream
    bad = bytearray(fbx_binary_scene(7400))
    i = bad.find(b'PolygonVertexIndex') + len(b'PolygonVertexIndex') + 1 + 12
    bad[i:i + 6] = b'\xff' * 6
    p3 = tmp_path / 'badzlib.fbx'
    p3.write_bytes(bytes(bad))
    with pytest.raises(MeshLoadError):
        load_mesh(str(p3))
    p4 = tmp_path / 'noobjects.fbx'
    p4.write_text('; FBX 7.4.0 project file\nFBXHeaderExtension:  {\n\tFBXVersion: 7400\n}\n')
    with pytest.raises(MeshLoadError, match='Objects'):
        load_mesh(str(p4))
    p5 = tmp_path / 'garbage.fbx'
    p5.write_bytes(bytes(range(256)) * 4)
    with pytest.raises(MeshLoadError):
        load_mesh(str(p5))
    p6 = tmp_path / 'nomesh.fbx'
    p6.write_text(fbx_ascii().replace('Vertices:', 'Vertexes:'))
    with pytest.raises(MeshLoadError, match='No triangles'):
        load_mesh(str(p6))


# ----------------------------------------------------------------------------
# STEP, dispatch, misc
# ----------------------------------------------------------------------------

STEP_MIN = """ISO-10303-21;
HEADER;
FILE_DESCRIPTION(('test'),'2;1');
FILE_NAME('cube.step','2026-01-01T00:00:00',(''),(''),'','','');
FILE_SCHEMA(('AUTOMOTIVE_DESIGN'));
ENDSEC;
DATA;
ENDSEC;
END-ISO-10303-21;
"""


@pytest.mark.skipif(importlib.util.find_spec('gmsh') is not None or importlib.util.find_spec('cascadio') is not None,
                    reason='a STEP backend is installed')
@pytest.mark.parametrize('ext', ['.step', '.stp'])
def test_step_without_backend(tmp_path, ext):
    p = tmp_path / f'part{ext}'
    p.write_text(STEP_MIN)
    with pytest.raises(MeshLoadError, match='gmsh'):
        load_mesh(str(p))


def test_step_not_step(tmp_path):
    p = tmp_path / 'fake.step'
    p.write_text('hello')
    with pytest.raises(MeshLoadError):
        load_mesh(str(p))


def test_unknown_missing_empty(tmp_path):
    p = tmp_path / 'model.3ds'
    p.write_bytes(b'xx')
    with pytest.raises(MeshLoadError) as ei:
        load_mesh(str(p))
    assert '.glb' in str(ei.value) and '.stp' in str(ei.value)
    with pytest.raises(MeshLoadError, match='not found'):
        load_mesh(str(tmp_path / 'nope.obj'))
    e = tmp_path / 'empty.stl'
    e.write_bytes(b'')
    with pytest.raises(MeshLoadError, match='empty'):
        load_mesh(str(e))
    assert SUPPORTED_EXTS == ('.glb', '.gltf', '.obj', '.stl', '.ply', '.fbx', '.step', '.stp')


def test_uppercase_extension(tmp_path):
    p = tmp_path / 'CUBE.STL'
    p.write_bytes(stl_binary(CUBE_T))
    assert len(load_mesh(str(p)).faces) == 12


def test_progress_and_cancel(tmp_path):
    p = tmp_path / 'car.glb'
    p.write_bytes(make_glb(gltf_doc(None), cube_buffer()))
    seen = []
    load_mesh(str(p), progress=seen.append)
    assert seen and seen[-1] == 1.0
    assert all(0.0 <= v <= 1.0 for v in seen)
    assert all(b > a for a, b in zip(seen, seen[1:]))

    class Cancelled(Exception):
        pass

    def cancel(v):
        if v > 0.2:
            raise Cancelled()

    with pytest.raises(Cancelled):
        load_mesh(str(p), progress=cancel)


def test_mesh_summary_aggregates_by_name():
    m = LoadedMesh(
        vertices=np.zeros((3, 3)), faces=np.zeros((4, 3), np.int64),
        face_part=np.array([0, 1, 1, 2], np.int32),
        parts=[PartInfo('Wheel', 'Rubber'), PartInfo('Body', 'Paint'), PartInfo('Wheel', 'Chrome')],
        fmt='obj', unit_hint=None, up_hint=None, forward_hint=None, notes=[])
    assert mesh_summary(m) == [('Wheel', 2, 'Rubber, Chrome'), ('Body', 2, 'Paint')]
