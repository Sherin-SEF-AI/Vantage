"""3D model importers for Vantage.

Turns model files into a single indexed triangle mesh (:class:`LoadedMesh`)
with a per-triangle part index.  Geometry is returned in the file's own units
and axes; unit / axis conversion is the caller's job (the ``*_hint`` fields
report what the file itself states).

Supported: OBJ (+MTL), STL (binary / ASCII), PLY (ascii / binary LE / BE),
glTF 2.0 (.gltf / .glb), FBX (binary 6.x-7.7 and ASCII) and STEP (through the
optional ``gmsh`` or ``cascadio`` packages).

All input is treated as untrusted: nothing from a file is ever evaluated,
element counts are checked against the file size before allocating, indices
are bounds-checked and parse failures surface as :class:`MeshLoadError` with a
short user-facing message.
"""
from __future__ import annotations

import base64
import binascii
import json
import math
import os
import re
import struct
import tempfile
import urllib.parse
import zlib
from dataclasses import dataclass, field
from typing import Callable

import numpy as np

__all__ = [
    'SUPPORTED_EXTS', 'MeshLoadError', 'PartInfo', 'LoadedMesh', 'load_mesh', 'mesh_summary',
]

SUPPORTED_EXTS = ('.glb', '.gltf', '.obj', '.stl', '.ply', '.fbx', '.step', '.stp')

# Hard limits that protect against hostile / absurd files.
MAX_VERTICES = 100_000_000
MAX_TRIANGLES = 100_000_000
MAX_FILE_BYTES = 4 << 30          # 4 GiB
MAX_ARRAY_BYTES = 2 << 30         # largest single decoded array (FBX / glTF)
_ZLIB_MAX_RATIO = 1100            # deflate cannot expand more than ~1032:1

_DEFAULT_COLOR = (0.8, 0.8, 0.8)


class MeshLoadError(Exception):
    """Raised when a model cannot be imported.  ``str(exc)`` is user-facing."""


@dataclass
class PartInfo:
    name: str                       # node/group/object name, '' -> 'Part N'
    material: str                   # material name or ''
    alpha: float = 1.0              # opacity 0..1
    color: tuple = _DEFAULT_COLOR   # rgb 0..1


@dataclass
class LoadedMesh:
    vertices: np.ndarray            # (N,3) float64, file units / file axes
    faces: np.ndarray               # (M,3) int64
    face_part: np.ndarray           # (M,) int32 index into parts
    parts: list[PartInfo]
    fmt: str                        # 'glb','gltf','obj','stl','ply','fbx','step'
    unit_hint: str | None
    up_hint: str | None
    forward_hint: str | None
    notes: list[str] = field(default_factory=list)


# ----------------------------------------------------------------------------
# small helpers
# ----------------------------------------------------------------------------

class _CallbackError(Exception):
    """Wraps an exception raised by the user's progress callback (e.g. cancel)."""

    def __init__(self, exc: BaseException):
        super().__init__(str(exc))
        self.exc = exc


class _Progress:
    """Monotonic, throttled wrapper around the optional progress callback.
    ``sub(a, b)`` returns a child reporter that maps 0..1 onto [a, b] and shares
    the monotonic state with its parent."""

    def __init__(self, cb: Callable[[float], object] | None, lo: float = 0.0, hi: float = 1.0, state=None):
        self.cb = cb
        self.lo, self.hi = lo, hi
        self.state = state if state is not None else [-1.0]

    def __call__(self, f: float) -> None:
        if self.cb is None:
            return
        try:
            f = float(f)
        except (TypeError, ValueError):
            return
        if not math.isfinite(f):
            return
        f = self.lo + (self.hi - self.lo) * min(1.0, max(0.0, f))
        last = self.state[0]
        if f <= last or (f - last < 0.004 and f < 1.0):
            return
        self.state[0] = f
        try:
            self.cb(f)
        except Exception as exc:  # let the caller cancel by raising
            raise _CallbackError(exc) from exc

    def sub(self, a: float, b: float) -> '_Progress':
        span = self.hi - self.lo
        return _Progress(self.cb, self.lo + span * a, self.lo + span * b, self.state)


def _plural(n: int, word: str, plural: str | None = None) -> str:
    return f"{n:,} {word if n == 1 else (plural or word + 's')}"


def _clamp01(x) -> float:
    try:
        x = float(x)
    except (TypeError, ValueError):
        return 1.0
    if not math.isfinite(x):
        return 1.0
    return min(1.0, max(0.0, x))


def _color3(vals, default=_DEFAULT_COLOR) -> tuple:
    try:
        c = [float(v) for v in list(vals)[:3]]
    except (TypeError, ValueError):
        return default
    if len(c) == 1:
        c = c * 3
    if len(c) != 3 or not all(math.isfinite(v) for v in c):
        return default
    return tuple(min(1.0, max(0.0, v)) for v in c)


def _decode_text(data: bytes) -> str:
    if data.startswith(b'\xef\xbb\xbf'):
        data = data[3:]
    try:
        return data.decode('utf-8')
    except UnicodeDecodeError:
        return data.decode('latin-1')


def _looks_binary(sample: bytes) -> bool:
    if b'\x00' in sample:
        return True
    ctrl = sum(1 for c in sample if c < 9 or 13 < c < 32)
    return ctrl > max(4, len(sample) // 50)


def _resolve_inside(base_dir: str, rel: str) -> str | None:
    """Resolve *rel* against *base_dir*; None if it escapes the directory."""
    if not rel or '\x00' in rel:
        return None
    rel = rel.replace('\\', '/')
    if rel.startswith('/') or re.match(r'^[A-Za-z][A-Za-z0-9+.\-]*:', rel):
        return None  # absolute path, URL scheme or Windows drive
    base = os.path.realpath(base_dir)
    cand = os.path.realpath(os.path.join(base, rel))
    try:
        if os.path.commonpath([cand, base]) != base:
            return None
    except ValueError:
        return None
    return cand


def _fan_triangulate(counts: np.ndarray, flat: np.ndarray):
    """Fan-triangulate polygons given per-polygon vertex counts and the
    concatenated index list.  Returns (tris (T,3) int64, poly_of_tri (T,))."""
    counts = np.asarray(counts, dtype=np.int64).reshape(-1)
    flat = np.asarray(flat, dtype=np.int64).reshape(-1)
    if counts.size == 0:
        return np.empty((0, 3), np.int64), np.empty(0, np.int64)
    if int(counts.sum()) != flat.size or (counts < 0).any():
        raise MeshLoadError('Polygon data is inconsistent (vertex counts do not match the index list).')
    if flat.size == 3 * counts.size and (counts == 3).all():
        return flat.reshape(-1, 3), np.arange(counts.size, dtype=np.int64)
    starts = np.cumsum(counts) - counts
    ntri = np.clip(counts - 2, 0, None)
    total = int(ntri.sum())
    if total > MAX_TRIANGLES:
        raise MeshLoadError(f'The model has more than {MAX_TRIANGLES:,} triangles, which is more than Vantage can load.')
    poly = np.repeat(np.arange(counts.size, dtype=np.int64), ntri)
    first = np.cumsum(ntri) - ntri
    j = np.arange(total, dtype=np.int64) - first[poly]
    s = starts[poly]
    tris = np.stack((flat[s], flat[s + j + 1], flat[s + j + 2]), axis=1)
    return tris, poly


def _polygon_notes(counts: np.ndarray, notes: list[str]) -> None:
    counts = np.asarray(counts)
    nq = int((counts == 4).sum())
    ng = int((counts > 4).sum())
    ns = int((counts < 3).sum())
    if nq:
        notes.append(f'{_plural(nq, "quad")} triangulated')
    if ng:
        notes.append(f'{_plural(ng, "polygon")} with more than 4 sides fan-triangulated')
    if ns:
        notes.append(f'Skipped {_plural(ns, "face")} with fewer than 3 vertices')


def _weld(points: np.ndarray):
    """Merge coincident points.  Returns (unique_points, inverse)."""
    n = len(points)
    if n == 0:
        return points.reshape(0, 3), np.empty(0, np.int64)
    ext = float(np.ptp(points, axis=0).max())
    tol = (ext if ext > 0 else 1.0) * 2.0 ** -30
    q = np.round(points / tol) + 0.0  # + 0.0 folds -0.0 into 0.0
    order = np.lexsort((q[:, 2], q[:, 1], q[:, 0]))
    qs = q[order]
    new = np.empty(n, bool)
    new[0] = True
    np.any(qs[1:] != qs[:-1], axis=1, out=new[1:])
    group = np.cumsum(new) - 1
    inverse = np.empty(n, np.int64)
    inverse[order] = group
    uniq = points[order[new]]
    return uniq, inverse


def _check_count(n: int, limit: int, what: str, name: str) -> int:
    n = int(n)
    if n < 0 or n > limit:
        raise MeshLoadError(f'{name} claims {n:,} {what}, which is not plausible; the file is probably corrupt.')
    return n


# ----------------------------------------------------------------------------
# mesh builder
# ----------------------------------------------------------------------------

class _Builder:
    def __init__(self, display_name: str):
        self.name = display_name
        self.parts: list[PartInfo] = []
        self._keys: dict = {}
        self._v: list[np.ndarray] = []
        self._f: list[np.ndarray] = []
        self._p: list[np.ndarray] = []
        self._nv = 0
        self._nt = 0
        self._bad = 0
        self.notes: list[str] = []

    def part(self, key, name: str = '', material: str = '', alpha=1.0, color=_DEFAULT_COLOR) -> int:
        idx = self._keys.get(key)
        if idx is None:
            idx = len(self.parts)
            self._keys[key] = idx
            self.parts.append(PartInfo(name=str(name or '').strip(), material=str(material or '').strip(),
                                       alpha=_clamp01(alpha), color=_color3(color)))
        return idx

    def add(self, verts, faces, part) -> None:
        verts = np.asarray(verts, dtype=np.float64).reshape(-1, 3)
        faces = np.asarray(faces, dtype=np.int64).reshape(-1, 3)
        n = len(verts)
        if self._nv + n > MAX_VERTICES:
            raise MeshLoadError(f'{self.name} has more than {MAX_VERTICES:,} vertices, which is more than Vantage can load. '
                                'Simplify the model and try again.')
        fp = np.broadcast_to(np.asarray(part, dtype=np.int32), (len(faces),))
        if len(faces):
            bad = ((faces < 0) | (faces >= n)).any(axis=1)
            if bad.any():
                self._bad += int(bad.sum())
                faces = faces[~bad]
                fp = fp[~bad]
        if self._nt + len(faces) > MAX_TRIANGLES:
            raise MeshLoadError(f'{self.name} has more than {MAX_TRIANGLES:,} triangles, which is more than Vantage can load. '
                                'Simplify the model and try again.')
        if not len(faces):
            return
        self._v.append(verts)
        self._f.append(faces + self._nv)
        self._p.append(np.array(fp, dtype=np.int32))
        self._nv += n
        self._nt += len(faces)

    def build(self, fmt: str, unit=None, up=None, forward=None) -> LoadedMesh:
        notes = list(self.notes)
        if self._bad:
            notes.append(f'Dropped {_plural(self._bad, "triangle")} with invalid vertex indices')
        if not self._f:
            raise MeshLoadError(f'No triangles found in {self.name}.')
        V = np.concatenate(self._v) if len(self._v) > 1 else self._v[0]
        F = np.concatenate(self._f) if len(self._f) > 1 else self._f[0]
        P = np.concatenate(self._p) if len(self._p) > 1 else self._p[0]
        self._v = self._f = self._p = []
        keep = np.ones(len(F), bool)
        finite = np.isfinite(V).all(axis=1)
        if not finite.all():
            badf = (~finite[F]).any(axis=1)
            if badf.any():
                notes.append(f'Dropped {_plural(int(badf.sum()), "triangle")} with invalid (NaN/infinite) coordinates')
            keep &= ~badf
            V = V.copy()
            V[~finite] = 0.0
        degen = (F[:, 0] == F[:, 1]) | (F[:, 1] == F[:, 2]) | (F[:, 0] == F[:, 2])
        degen &= keep
        if degen.any():
            notes.append(f'Dropped {_plural(int(degen.sum()), "degenerate triangle")}')
            keep &= ~degen
        if not keep.all():
            F = F[keep]
            P = P[keep]
        if len(F) == 0:
            raise MeshLoadError(f'No triangles found in {self.name}.')
        # drop vertices nothing references
        used = np.zeros(len(V), bool)
        used[F.ravel()] = True
        if not used.all():
            remap = np.cumsum(used, dtype=np.int64) - 1
            V = V[used]
            F = remap[F]
        # drop parts without triangles
        counts = np.bincount(P, minlength=len(self.parts))
        parts = self.parts
        if (counts == 0).any():
            keep_p = np.flatnonzero(counts > 0)
            remap_p = np.full(len(parts), -1, np.int32)
            remap_p[keep_p] = np.arange(len(keep_p), dtype=np.int32)
            P = remap_p[P]
            parts = [parts[i] for i in keep_p]
        for i, p in enumerate(parts):
            if not p.name:
                p.name = f'Part {i + 1}'
        return LoadedMesh(
            vertices=np.ascontiguousarray(V, dtype=np.float64),
            faces=np.ascontiguousarray(F, dtype=np.int64),
            face_part=np.ascontiguousarray(P, dtype=np.int32),
            parts=parts, fmt=fmt, unit_hint=unit, up_hint=up, forward_hint=forward, notes=notes)


# ----------------------------------------------------------------------------
# OBJ
# ----------------------------------------------------------------------------

_OBJ_SLASH = re.compile(r'/\S*')


def _parse_mtl(path: str, notes: list[str]) -> dict[str, tuple[float, tuple]]:
    try:
        if os.path.getsize(path) > 64 << 20:
            notes.append(f'Material library {os.path.basename(path)} is too large and was ignored')
            return {}
        with open(path, 'rb') as fh:
            text = _decode_text(fh.read())
    except OSError:
        notes.append(f'Could not read material library {os.path.basename(path)}')
        return {}
    mats: dict[str, dict] = {}
    cur = None
    for line in text.splitlines():
        line = line.strip()
        if not line or line[0] == '#':
            continue
        kw, _, rest = line.partition(' ')
        if kw == 'newmtl':
            cur = mats.setdefault(rest.strip(), {})
        elif cur is None:
            continue
        elif kw == 'Kd':
            cur['Kd'] = _color3(_floats(rest.split()))
        elif kw == 'd':
            vals = _floats(rest.split())
            if vals:
                cur['d'] = vals[-1]
        elif kw == 'Tr':
            vals = _floats(rest.split())
            if vals:
                cur['Tr'] = vals[-1]
    out = {}
    for name, m in mats.items():
        if 'd' in m:
            alpha = m['d']
        elif 'Tr' in m:
            alpha = 1.0 - m['Tr']
        else:
            alpha = 1.0
        out[name] = (_clamp01(alpha), m.get('Kd', _DEFAULT_COLOR))
    return out


def _floats(tokens) -> list[float]:
    out = []
    for t in tokens:
        try:
            out.append(float(t))
        except ValueError:
            pass
    return out


def _load_obj(path: str, data: bytes, prog: _Progress) -> LoadedMesh:
    name = os.path.basename(path)
    if _looks_binary(data[:4096]):
        raise MeshLoadError(f'{name} does not look like a text OBJ file.')
    b = _Builder(name)
    text = _decode_text(data)
    del data
    if '\\\n' in text or '\\\r\n' in text:
        text = re.sub(r'\\\r?\n', ' ', text)
    lines = text.splitlines()
    del text
    prog(0.1)

    v_lines: list[str] = []
    f_lines: list[str] = []
    f_nv: list[int] = []
    part_ids: dict[tuple[str, str], int] = {}
    changes: list[tuple[int, int]] = []
    mtllibs: list[str] = []
    cur_name, cur_mtl = '', ''
    cur_pid = -1
    n_line_elems = n_points = n_freeform = 0
    ws = (' ', '\t')

    def set_part():
        nonlocal cur_pid
        pid = part_ids.setdefault((cur_name, cur_mtl), len(part_ids))
        if pid != cur_pid:
            changes.append((len(f_lines), pid))
            cur_pid = pid

    set_part()
    for line in lines:
        if not line:
            continue
        c = line[0]
        if c == ' ' or c == '\t':
            line = line.lstrip()
            if not line:
                continue
            c = line[0]
        if c == 'v':
            if line[1:2] in ws:
                v_lines.append(line)
            continue
        if c == 'f':
            if line[1:2] in ws:
                f_lines.append(line)
                f_nv.append(len(v_lines))
            continue
        if c == '#':
            continue
        kw, _, rest = line.partition(' ')
        if '\t' in kw:
            kw, _, rest2 = kw.partition('\t')
            rest = rest2 + ' ' + rest
        rest = rest.strip()
        if kw == 'o' or kw == 'g':
            cur_name = rest
            set_part()
        elif kw == 'usemtl':
            cur_mtl = rest
            set_part()
        elif kw == 'mtllib':
            mtllibs.append(rest)
        elif kw == 'l':
            n_line_elems += 1
        elif kw == 'p':
            n_points += 1
        elif kw in ('curv', 'curv2', 'surf', 'cstype'):
            n_freeform += 1
    del lines
    prog(0.3)

    # --- vertices -----------------------------------------------------------
    nv = len(v_lines)
    _check_count(nv, MAX_VERTICES, 'vertices', name)
    verts = None
    if nv:
        tokens = ' '.join(v_lines).split()
        if len(tokens) == 4 * nv and all(t == 'v' for t in tokens[0::4]):
            del tokens[0::4]
            try:
                verts = np.array(list(map(float, tokens)), dtype=np.float64).reshape(nv, 3)
            except ValueError:
                verts = None
        del tokens
        if verts is None:
            verts = np.empty((nv, 3), np.float64)
            bad = 0
            for i, line in enumerate(v_lines):
                p = line.split()
                try:
                    verts[i] = (float(p[1]), float(p[2]), float(p[3]))
                except (ValueError, IndexError):
                    verts[i] = np.nan
                    bad += 1
            if bad:
                b.notes.append(f'{_plural(bad, "malformed vertex line")} ignored')
    else:
        verts = np.empty((0, 3), np.float64)
    del v_lines
    prog(0.5)

    # --- faces --------------------------------------------------------------
    nf = len(f_lines)
    counts = vals = None
    if nf:
        joined = '\n'.join(f_lines)
        if '/' in joined:
            joined = _OBJ_SLASH.sub('', joined)
        tokens = joined.split()
        del joined
        fpos = [i for i, t in enumerate(tokens) if t == 'f']
        if len(fpos) == nf:
            counts = np.diff(np.append(np.asarray(fpos, np.int64), len(tokens))) - 1
            try:
                vals = np.array(list(map(int, (t for t in tokens if t != 'f'))), dtype=np.int64)
            except (ValueError, OverflowError):
                counts = vals = None
        del tokens, fpos
        if counts is None:  # slow, forgiving path
            cl: list[int] = []
            flat: list[int] = []
            bad = 0
            for line in f_lines:
                toks = line.split('#', 1)[0].split()[1:]
                try:
                    idx = [int(t.split('/', 1)[0]) for t in toks]
                    if any(abs(i) > (1 << 62) for i in idx):
                        raise ValueError
                except ValueError:
                    bad += 1
                    idx = []
                cl.append(len(idx))
                flat.extend(idx)
            if bad:
                b.notes.append(f'{_plural(bad, "malformed face line")} ignored')
            counts = np.asarray(cl, np.int64)
            vals = np.asarray(flat, np.int64)
    del f_lines
    prog(0.65)

    # parts (materials need the mtl libraries)
    base_dir = os.path.dirname(os.path.abspath(path))
    mtl: dict[str, tuple[float, tuple]] = {}
    seen_libs = set()
    for lib in mtllibs:
        whole = _resolve_inside(base_dir, lib)
        # "mtllib a.mtl b.mtl" lists several files, but names may also contain spaces
        candidates = [lib] if (whole and os.path.isfile(whole)) else lib.split()
        for cand in candidates:
            p = _resolve_inside(base_dir, cand)
            if p is None:
                b.notes.append(f'Ignored material library outside the model folder: {cand}')
            elif p in seen_libs:
                continue
            elif os.path.isfile(p):
                seen_libs.add(p)
                mtl.update(_parse_mtl(p, b.notes))
            else:
                b.notes.append(f'Material library {cand} not found')
    for (pname, mname), pid in sorted(part_ids.items(), key=lambda kv: kv[1]):
        alpha, color = mtl.get(mname, (1.0, _DEFAULT_COLOR))
        b.part((pname, mname), pname, mname, alpha, color)

    if nf and counts is not None and counts.size:
        nv_rep = np.repeat(np.asarray(f_nv, np.int64), counts)
        idx = np.where(vals > 0, vals - 1, np.where(vals < 0, vals + nv_rep, -1))
        del nv_rep
        _polygon_notes(counts, b.notes)
        tris, poly = _fan_triangulate(counts, idx)
        starts = np.array([c[0] for c in changes] + [nf], np.int64)
        pids = np.array([c[1] for c in changes], np.int32)
        poly_part = np.repeat(pids, np.diff(starts))
        b.add(verts, tris, poly_part[poly])
    if n_line_elems:
        b.notes.append(f'Skipped {_plural(n_line_elems, "line element")}')
    if n_points:
        b.notes.append(f'Skipped {_plural(n_points, "point element")}')
    if n_freeform:
        b.notes.append('Free-form curves/surfaces are not supported and were skipped')
    prog(0.9)
    return b.build('obj')


# ----------------------------------------------------------------------------
# STL
# ----------------------------------------------------------------------------

_STL_SOLID = re.compile(rb'(?<![A-Za-z])solid\b[ \t]*([^\r\n]*)', re.IGNORECASE)
_STL_VERTEX = re.compile(rb'vertex\s+(\S+)\s+(\S+)\s+(\S+)', re.IGNORECASE)
_STL_DTYPE = np.dtype([('n', '<f4', (3,)), ('v', '<f4', (3, 3)), ('attr', '<u2')])


def _stl_is_ascii(data: bytes) -> bool:
    head = data[:1024].lstrip()
    if head[:5].lower() != b'solid':
        return False
    if _looks_binary(data[:1024]):
        return False
    return b'facet' in data[:4096].lower() or b'endsolid' in data[-1024:].lower() or len(data) < 84


def _load_stl(path: str, data: bytes, prog: _Progress) -> LoadedMesh:
    name = os.path.basename(path)
    b = _Builder(name)
    size = len(data)
    is_ascii = _stl_is_ascii(data)
    binary_cnt = None
    if size >= 84:
        cnt = struct.unpack_from('<I', data, 80)[0]
        expected = 84 + 50 * cnt
        if expected == size and not is_ascii:
            binary_cnt = cnt
        elif not is_ascii:
            if expected > size:
                raise MeshLoadError(f'{name} is truncated: the STL header lists {cnt:,} triangles '
                                    f'but the file only holds {max(0, (size - 84) // 50):,}.')
            if cnt == 0 and (size - 84) % 50 == 0:
                binary_cnt = (size - 84) // 50
                b.notes.append('STL header triangle count was 0; used the file size instead')
            else:
                binary_cnt = cnt
                b.notes.append(f'Ignored {size - expected:,} trailing bytes after the STL data')
    elif not is_ascii:
        raise MeshLoadError(f'{name} is too small to be an STL file.')
    prog(0.1)

    if binary_cnt is not None:
        _check_count(binary_cnt, MAX_TRIANGLES, 'triangles', name)
        rec = np.frombuffer(data, dtype=_STL_DTYPE, count=binary_cnt, offset=84)
        tri_pts = rec['v'].reshape(-1, 3).astype(np.float64)
        part_of_tri = np.zeros(binary_cnt, np.int32)
        b.part('stl', '', '')
    else:
        starts = [(m.start(), m.group(1).decode('utf-8', 'replace').strip()) for m in _STL_SOLID.finditer(data)]
        if not starts:
            starts = [(0, '')]
        elif starts[0][0] > 0 and _STL_VERTEX.search(data, 0, starts[0][0]):
            starts.insert(0, (0, ''))
        chunks = []
        pids = []
        incomplete = 0
        for i, (s, sname) in enumerate(starts):
            e = starts[i + 1][0] if i + 1 < len(starts) else size
            flat = [x for t in _STL_VERTEX.findall(data, s, e) for x in t]
            if not flat:
                continue
            try:
                pts = np.array(list(map(float, flat)), dtype=np.float64).reshape(-1, 3)
            except ValueError:
                raise MeshLoadError(f'{name} contains an invalid number in a vertex line; the file is corrupt.') from None
            rem = len(pts) % 3
            if rem:
                incomplete += 1
                pts = pts[:len(pts) - rem]
            if not len(pts):
                continue
            pid = b.part(('solid', i), sname, '')
            chunks.append(pts)
            pids.append(np.full(len(pts) // 3, pid, np.int32))
            prog(0.1 + 0.4 * (i + 1) / len(starts))
        if incomplete:
            b.notes.append('Dropped an incomplete facet at the end of the file (truncated STL?)')
        if not chunks:
            raise MeshLoadError(f'No triangles found in {name}.')
        tri_pts = np.concatenate(chunks)
        part_of_tri = np.concatenate(pids)
    prog(0.5)
    if not len(tri_pts):
        raise MeshLoadError(f'No triangles found in {name}.')
    finite = np.isfinite(tri_pts).all(axis=1).reshape(-1, 3).all(axis=1)
    if not finite.all():
        b.notes.append(f'Dropped {_plural(int((~finite).sum()), "triangle")} with invalid (NaN/infinite) coordinates')
        tri_pts = tri_pts.reshape(-1, 3, 3)[finite].reshape(-1, 3)
        part_of_tri = part_of_tri[finite]
    if not len(tri_pts):
        raise MeshLoadError(f'No triangles found in {name}.')
    verts, inverse = _weld(tri_pts)
    prog(0.85)
    b.add(verts, inverse.reshape(-1, 3), part_of_tri)
    return b.build('stl')


# ----------------------------------------------------------------------------
# PLY
# ----------------------------------------------------------------------------

_PLY_TYPES = {
    'char': 'i1', 'int8': 'i1', 'uchar': 'u1', 'uint8': 'u1',
    'short': 'i2', 'int16': 'i2', 'ushort': 'u2', 'uint16': 'u2',
    'int': 'i4', 'int32': 'i4', 'uint': 'u4', 'uint32': 'u4',
    'float': 'f4', 'float32': 'f4', 'double': 'f8', 'float64': 'f8',
}
_PLY_END = re.compile(rb'(?m)^end_header[ \t]*(?:\r\n|\n|\r)?')


class _PlyProp:
    __slots__ = ('name', 'is_list', 'dtype', 'count_dtype')

    def __init__(self, name, dtype, count_dtype=None):
        self.name = name
        self.dtype = dtype
        self.count_dtype = count_dtype
        self.is_list = count_dtype is not None


class _PlyElement:
    __slots__ = ('name', 'count', 'props')

    def __init__(self, name, count):
        self.name = name
        self.count = count
        self.props: list[_PlyProp] = []


def _ply_header(data: bytes, name: str):
    m = _PLY_END.search(data, 0, min(len(data), 1 << 20))
    if not m:
        raise MeshLoadError(f'{name} has an incomplete PLY header (no end_header line found).')
    header = data[:m.start()].decode('ascii', 'replace')
    lines = header.splitlines()
    if not lines or lines[0].strip() != 'ply':
        raise MeshLoadError(f'{name} is not a PLY file.')
    fmt = None
    elements: list[_PlyElement] = []
    try:
        for raw in lines[1:]:
            parts = raw.split()
            if not parts:
                continue
            kw = parts[0]
            if kw == 'format':
                fmt = parts[1]
            elif kw == 'element':
                count = int(parts[2])
                if count < 0 or count > 10 ** 10:
                    raise MeshLoadError(f'{name} claims {count:,} {parts[1]} elements, which is not plausible; '
                                        'the file is probably corrupt.')
                elements.append(_PlyElement(parts[1], count))
            elif kw == 'property':
                if not elements:
                    raise ValueError('property before element')
                if parts[1] == 'list':
                    ct, it, pname = parts[2], parts[3], parts[4]
                    if ct not in _PLY_TYPES or it not in _PLY_TYPES or _PLY_TYPES[ct][0] == 'f':
                        raise ValueError('bad list type')
                    elements[-1].props.append(_PlyProp(pname, _PLY_TYPES[it], _PLY_TYPES[ct]))
                else:
                    t, pname = parts[1], parts[2]
                    if t not in _PLY_TYPES:
                        raise ValueError('bad type')
                    elements[-1].props.append(_PlyProp(pname, _PLY_TYPES[t]))
    except (ValueError, IndexError):
        raise MeshLoadError(f'{name} has a malformed PLY header.') from None
    if fmt not in ('ascii', 'binary_little_endian', 'binary_big_endian'):
        raise MeshLoadError(f'{name} uses an unknown PLY format ({fmt}).')
    return fmt, elements, m.end()


def _ply_binary_lists(data, pos, el, endian, target: int, name: str):
    """Read an element that contains list properties.  Returns
    (new_pos, counts, flat_values) for list property index *target*
    (counts/flat are None when target < 0)."""
    n = el.count
    size = len(data)
    trunc = MeshLoadError(f'{name} is truncated (PLY {el.name} data ends early).')
    if n == 0:
        return pos, np.empty(0, np.int64), np.empty(0, np.int64)
    min_rec = sum(np.dtype(p.count_dtype if p.is_list else p.dtype).itemsize for p in el.props)
    if pos + n * min_rec > size:
        raise trunc
    # --- fast path: every list in every record has the same length as in the first
    try:
        p = pos
        first = []
        for pr in el.props:
            if pr.is_list:
                k = int(np.frombuffer(data, endian + pr.count_dtype, 1, p)[0])
                first.append(k)
                p += np.dtype(pr.count_dtype).itemsize + k * np.dtype(pr.dtype).itemsize
            else:
                p += np.dtype(pr.dtype).itemsize
            if p > size:
                raise trunc
    except ValueError:
        raise trunc from None
    fields = []
    li = 0
    for i, pr in enumerate(el.props):
        if pr.is_list:
            fields.append((f'c{i}', endian + pr.count_dtype))
            if first[li] > 0:
                fields.append((f'v{i}', endian + pr.dtype, (first[li],)))
            li += 1
        else:
            fields.append((f's{i}', endian + pr.dtype))
    dt = np.dtype(fields)
    if pos + n * dt.itemsize <= size:
        rec = np.frombuffer(data, dt, n, pos)
        li = 0
        uniform = True
        for i, pr in enumerate(el.props):
            if pr.is_list:
                if not (rec[f'c{i}'] == first[li]).all():
                    uniform = False
                    break
                li += 1
        if uniform:
            if target < 0:
                return pos + n * dt.itemsize, None, None
            k = first[[i for i, pr in enumerate(el.props) if pr.is_list].index(target)]
            if k == 0:
                return pos + n * dt.itemsize, np.zeros(n, np.int64), np.empty(0, np.int64)
            flat = rec[f'v{target}'].reshape(-1).astype(np.int64)
            return pos + n * dt.itemsize, np.full(n, k, np.int64), flat
    # --- slow path: walk record by record
    steps = []
    for i, pr in enumerate(el.props):
        if pr.is_list:
            cdt = np.dtype(pr.count_dtype)
            steps.append((True, struct.Struct(endian + cdt.char).unpack_from, cdt.itemsize,
                          np.dtype(pr.dtype).itemsize, i == target))
        else:
            steps.append((False, None, np.dtype(pr.dtype).itemsize, 0, False))
    counts = []
    starts = []
    p = pos
    try:
        if len(steps) == 1 and steps[0][0]:
            _, unpack, csz, isz, _t = steps[0]
            for _ in range(n):
                k = unpack(data, p)[0]
                p += csz
                counts.append(k)
                starts.append(p)
                p += k * isz
        else:
            for _ in range(n):
                for is_list, unpack, csz, isz, is_target in steps:
                    if is_list:
                        k = unpack(data, p)[0]
                        p += csz
                        if is_target:
                            counts.append(k)
                            starts.append(p)
                        p += k * isz
                    else:
                        p += csz
                if p > size:
                    raise trunc
    except struct.error:
        raise trunc from None
    if p > size:
        raise trunc
    if target < 0:
        return p, None, None
    pr = el.props[target]
    isz = np.dtype(pr.dtype).itemsize
    counts_a = np.asarray(counts, np.int64)
    starts_a = np.asarray(starts, np.int64)
    total = int(counts_a.sum())
    if total == 0:
        return p, counts_a, np.empty(0, np.int64)
    first_i = np.cumsum(counts_a) - counts_a
    item_pos = np.repeat(starts_a, counts_a) + (np.arange(total, dtype=np.int64) - np.repeat(first_i, counts_a)) * isz
    raw = np.frombuffer(data, np.uint8)
    gathered = raw[item_pos[:, None] + np.arange(isz)]
    flat = np.ascontiguousarray(gathered).view(endian + pr.dtype).reshape(-1).astype(np.int64)
    return p, counts_a, flat


def _ply_ascii_tokens_to_int(tokens) -> np.ndarray:
    a = np.array(list(map(float, tokens)), dtype=np.float64)
    ok = np.isfinite(a) & (np.abs(a) < 2.0 ** 62)
    return np.where(ok, a, -1).astype(np.int64)


def _load_ply(path: str, data: bytes, prog: _Progress) -> LoadedMesh:
    name = os.path.basename(path)
    b = _Builder(name)
    fmt, elements, body = _ply_header(data, name)
    v_el = next((e for e in elements if e.name == 'vertex'), None)
    f_el = next((e for e in elements if e.name == 'face'), None)
    if v_el is None:
        raise MeshLoadError(f'{name} has no vertex element.')
    pnames = [p.name for p in v_el.props]
    if not all(a in pnames for a in 'xyz') or any(v_el.props[pnames.index(a)].is_list for a in 'xyz'):
        raise MeshLoadError(f'{name} has no x/y/z vertex coordinates.')
    _check_count(v_el.count, MAX_VERTICES, 'vertices', name)
    target = -1
    if f_el is not None:
        _check_count(f_el.count, MAX_TRIANGLES, 'faces', name)
        lists = [i for i, p in enumerate(f_el.props) if p.is_list]
        named = [i for i in lists if f_el.props[i].name in ('vertex_indices', 'vertex_index')]
        target = (named or lists or [-1])[0]
        if target >= 0 and f_el.props[target].dtype[0] == 'f':
            target = -1
    if any(e.name == 'tristrips' for e in elements):
        b.notes.append('Triangle strips in PLY are not supported and were skipped')
    verts = None
    counts = flat = None
    trunc = MeshLoadError(f'{name} is truncated (PLY data ends before all elements were read).')
    prog(0.1)

    if fmt == 'ascii':
        tokens = data[body:].split()
        t = 0
        for el in elements:
            if verts is not None and (f_el is None or counts is not None):
                break
            nprops = len(el.props)
            if not any(p.is_list for p in el.props):
                need = el.count * nprops
                if t + need > len(tokens):
                    raise trunc
                if el is v_el:
                    try:
                        arr = np.array(list(map(float, tokens[t:t + need])), dtype=np.float64).reshape(el.count, nprops)
                    except ValueError:
                        raise MeshLoadError(f'{name} contains an invalid number in the vertex list.') from None
                    verts = arr[:, [pnames.index('x'), pnames.index('y'), pnames.index('z')]]
                    prog(0.4)
                t += need
                continue
            want = target if el is f_el else -1
            n = el.count
            if el is f_el and nprops == 1 and n and t < len(tokens):
                k0 = tokens[t]
                try:
                    k = int(k0)
                except ValueError:
                    raise MeshLoadError(f'{name} contains an invalid face record.') from None
                rec = k + 1
                if k >= 0 and t + n * rec <= len(tokens) and all(x == k0 for x in tokens[t:t + n * rec:rec]):
                    block = tokens[t:t + n * rec]
                    del block[0::rec]
                    try:
                        flat = _ply_ascii_tokens_to_int(block)
                    except ValueError:
                        raise MeshLoadError(f'{name} contains an invalid face record.') from None
                    counts = np.full(n, k, np.int64)
                    t += n * rec
                    continue
            cl: list[int] = []
            fl: list = []
            try:
                for _ in range(n):
                    for i, pr in enumerate(el.props):
                        if pr.is_list:
                            k = int(float(tokens[t]))
                            t += 1
                            if k < 0 or t + k > len(tokens):
                                raise trunc
                            if i == want:
                                cl.append(k)
                                fl.extend(tokens[t:t + k])
                            t += k
                        else:
                            t += 1
            except IndexError:
                raise trunc from None
            except ValueError:
                raise MeshLoadError(f'{name} contains an invalid {el.name} record.') from None
            if t > len(tokens):
                raise trunc
            if el is f_el:
                counts = np.asarray(cl, np.int64)
                try:
                    flat = _ply_ascii_tokens_to_int(fl)
                except ValueError:
                    raise MeshLoadError(f'{name} contains an invalid face record.') from None
    else:
        endian = '<' if fmt == 'binary_little_endian' else '>'
        pos = body
        for el in elements:
            if verts is not None and (f_el is None or counts is not None):
                break
            if not any(p.is_list for p in el.props):
                dt = np.dtype([(f'p{i}', endian + p.dtype) for i, p in enumerate(el.props)])
                need = el.count * dt.itemsize
                if pos + need > len(data):
                    raise trunc
                if el is v_el:
                    rec = np.frombuffer(data, dt, el.count, pos)
                    verts = np.column_stack([rec[f'p{pnames.index(a)}'].astype(np.float64) for a in 'xyz']) \
                        if el.count else np.empty((0, 3))
                    prog(0.4)
                pos += need
                continue
            want = target if el is f_el else -1
            pos, c, fl = _ply_binary_lists(data, pos, el, endian, want, name)
            if el is f_el:
                counts, flat = c, fl
    prog(0.7)
    if verts is None:
        raise trunc
    if counts is None or not len(counts):
        raise MeshLoadError(f'No triangles found in {name} (it may be a point cloud).')
    _polygon_notes(counts, b.notes)
    tris, _ = _fan_triangulate(counts, flat)
    b.part('ply', '', '')
    b.add(verts, tris, 0)
    prog(0.9)
    return b.build('ply')


# ----------------------------------------------------------------------------
# glTF 2.0
# ----------------------------------------------------------------------------

_GLTF_CT = {5120: np.dtype('<i1'), 5121: np.dtype('<u1'), 5122: np.dtype('<i2'), 5123: np.dtype('<u2'),
            5124: np.dtype('<i4'), 5125: np.dtype('<u4'), 5126: np.dtype('<f4')}
_GLTF_NORM = {5120: 127.0, 5121: 255.0, 5122: 32767.0, 5123: 65535.0, 5124: 2147483647.0, 5125: 4294967295.0}
_GLTF_NC = {'SCALAR': 1, 'VEC2': 2, 'VEC3': 3, 'VEC4': 4, 'MAT2': 4, 'MAT3': 9, 'MAT4': 16}
_MESHOPT_EXTS = ('EXT_meshopt_compression', 'KHR_meshopt_compression')
_DRACO_EXT = 'KHR_draco_mesh_compression'


def _quat_matrix(q) -> np.ndarray:
    x, y, z, w = (float(v) for v in q)
    n = math.sqrt(x * x + y * y + z * z + w * w)
    if n == 0 or not math.isfinite(n):
        return np.eye(3)
    x, y, z, w = x / n, y / n, z / n, w / n
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ])


def _gltf_node_matrix(node: dict) -> np.ndarray:
    if 'matrix' in node:
        m = np.asarray(node['matrix'], dtype=np.float64)
        if m.size != 16:
            raise MeshLoadError('A glTF node has an invalid transform matrix.')
        return m.reshape(4, 4).T.copy()  # column-major
    M = np.eye(4)
    s = np.asarray(node.get('scale', (1, 1, 1)), dtype=np.float64).reshape(3)
    M[:3, :3] = _quat_matrix(node.get('rotation', (0, 0, 0, 1))) * s[None, :]
    M[:3, 3] = np.asarray(node.get('translation', (0, 0, 0)), dtype=np.float64).reshape(3)
    return M


def _compressed_error(name: str, what: str) -> MeshLoadError:
    return MeshLoadError(f'{name} uses {what}, which Vantage cannot read. '
                         'Re-export the model without mesh compression (e.g. untick Draco/meshopt in the exporter).')


class _Gltf:
    def __init__(self, path: str, doc: dict, bin_chunk, b: _Builder, is_glb: bool):
        self.path = path
        self.name = os.path.basename(path)
        self.dir = os.path.dirname(os.path.abspath(path))
        self.doc = doc
        self.bin = bin_chunk
        self.b = b
        self.is_glb = is_glb
        self._buffers: dict[int, memoryview] = {}
        self._raw: dict[int, np.ndarray] = {}
        self._prim_cache: dict = {}
        self.skipped = {'points': 0, 'lines': 0, 'unknown': 0, 'noposition': 0}

    def _trunc(self, what='buffer data') -> MeshLoadError:
        return MeshLoadError(f'{self.name} is truncated or corrupt (glTF {what} is out of range).')

    def buffer(self, i: int) -> memoryview:
        if i in self._buffers:
            return self._buffers[i]
        buffers = self.doc.get('buffers') or []
        if not (0 <= i < len(buffers)):
            raise self._trunc('buffer index')
        bd = buffers[i]
        ext = bd.get('extensions') or {}
        if any((ext.get(e) or {}).get('fallback') for e in _MESHOPT_EXTS):
            raise _compressed_error(self.name, 'meshopt compression')
        uri = bd.get('uri')
        if uri is None:
            if self.bin is None:
                raise MeshLoadError(f'{self.name} references binary data that is missing (no GLB BIN chunk).')
            data = self.bin
        elif not isinstance(uri, str):
            raise self._trunc('buffer uri')
        elif uri.startswith('data:'):
            header, sep, payload = uri.partition(',')
            if not sep or not header.lower().endswith(';base64'):
                raise MeshLoadError(f'{self.name} contains an embedded buffer that is not base64-encoded; '
                                    'only base64 data URIs are supported.')
            try:
                data = memoryview(base64.b64decode(payload))
            except (binascii.Error, ValueError):
                raise MeshLoadError(f'{self.name} contains a corrupt base64 buffer.') from None
        else:
            rel = urllib.parse.unquote(uri)
            p = _resolve_inside(self.dir, rel)
            if p is None:
                raise MeshLoadError(f'{self.name} references a buffer file outside its folder ({uri}); '
                                    'for safety only files next to the model are loaded.')
            if not os.path.isfile(p):
                raise MeshLoadError(f'{self.name} needs the file "{rel}", which was not found next to it. '
                                    'Keep the .bin file in the same folder as the .gltf.')
            if os.path.getsize(p) > MAX_FILE_BYTES:
                raise MeshLoadError(f'Buffer file {rel} is too large to load.')
            with open(p, 'rb') as fh:
                data = memoryview(fh.read())
        blen = bd.get('byteLength')
        if blen is not None and int(blen) > len(data):
            raise MeshLoadError(f'{self.name} is truncated: a glTF buffer holds {len(data):,} bytes '
                                f'but {int(blen):,} are declared.')
        self._buffers[i] = data
        return data

    def view(self, i: int) -> tuple[memoryview, int | None]:
        views = self.doc.get('bufferViews') or []
        if not (0 <= int(i) < len(views)):
            raise self._trunc('bufferView index')
        bv = views[int(i)]
        if any(e in (bv.get('extensions') or {}) for e in _MESHOPT_EXTS):
            # meshopt views carry an (optional) uncompressed fallback in their own buffer
            bd = (self.doc.get('buffers') or [])[int(bv['buffer'])]
            if bd.get('uri') is None and not (self.bin is not None and int(bv['buffer']) == 0):
                raise _compressed_error(self.name, 'meshopt compression')
        buf = self.buffer(int(bv['buffer']))
        off = int(bv.get('byteOffset', 0))
        ln = int(bv['byteLength'])
        if off < 0 or ln < 0 or off + ln > len(buf):
            raise self._trunc('bufferView')
        stride = bv.get('byteStride')
        return buf[off:off + ln], (int(stride) if stride else None)

    @staticmethod
    def _strided(buf, base: int, count: int, ncomp: int, dt: np.dtype, stride: int, err) -> np.ndarray:
        elsize = ncomp * dt.itemsize
        if count == 0:
            return np.empty((0, ncomp), dt)
        if stride < elsize or base < 0:
            raise err
        end = base + stride * (count - 1) + elsize
        if end > len(buf):
            raise err
        if stride == elsize:
            return np.frombuffer(buf, dt, count * ncomp, base).reshape(count, ncomp).copy()
        return np.ndarray((count, ncomp), dtype=dt, buffer=buf, offset=base,
                          strides=(stride, dt.itemsize)).copy()

    def accessor_raw(self, i: int) -> tuple[np.ndarray, dict]:
        accs = self.doc.get('accessors') or []
        if not (0 <= int(i) < len(accs)):
            raise self._trunc('accessor index')
        acc = accs[int(i)]
        if int(i) in self._raw:
            return self._raw[int(i)], acc
        ct = int(acc['componentType'])
        if ct not in _GLTF_CT:
            raise MeshLoadError(f'{self.name} uses an unknown glTF component type ({ct}).')
        dt = _GLTF_CT[ct]
        ncomp = _GLTF_NC.get(acc.get('type'))
        if ncomp is None:
            raise MeshLoadError(f'{self.name} uses an unknown glTF accessor type ({acc.get("type")}).')
        count = int(acc['count'])
        if count < 0 or count * ncomp * dt.itemsize > MAX_ARRAY_BYTES:
            raise MeshLoadError(f'{self.name} declares an implausibly large accessor ({count:,} elements).')
        err = self._trunc('accessor')
        if acc.get('bufferView') is not None:
            buf, stride = self.view(acc['bufferView'])
            arr = self._strided(buf, int(acc.get('byteOffset', 0)), count, ncomp, dt,
                                stride or ncomp * dt.itemsize, err)
        else:
            sparse_n = int((acc.get('sparse') or {}).get('count', 0) or 0)
            if count > max(10_000_000, 1000 * sparse_n):
                raise MeshLoadError(f'{self.name} declares an implausibly large accessor ({count:,} elements).')
            arr = np.zeros((count, ncomp), dt)
        sp = acc.get('sparse')
        if sp:
            scount = int(sp['count'])
            if not (0 < scount <= count):
                raise self._trunc('sparse accessor')
            si, sv = sp['indices'], sp['values']
            ict = int(si['componentType'])
            if ict not in (5121, 5123, 5125):
                raise self._trunc('sparse accessor')
            idt = _GLTF_CT[ict]
            ibuf, _ = self.view(si['bufferView'])
            idx = self._strided(ibuf, int(si.get('byteOffset', 0)), scount, 1, idt, idt.itemsize, err)[:, 0]
            vbuf, _ = self.view(sv['bufferView'])
            vals = self._strided(vbuf, int(sv.get('byteOffset', 0)), scount, ncomp, dt, ncomp * dt.itemsize, err)
            idx = idx.astype(np.int64)
            if (idx >= count).any():
                raise self._trunc('sparse accessor')
            arr[idx] = vals
        self._raw[int(i)] = arr
        return arr, acc

    def accessor_float(self, i: int) -> np.ndarray:
        arr, acc = self.accessor_raw(i)
        out = arr.astype(np.float64)
        ct = int(acc['componentType'])
        if acc.get('normalized') and ct in _GLTF_NORM:
            out /= _GLTF_NORM[ct]
            if ct in (5120, 5122, 5124):
                np.maximum(out, -1.0, out=out)
        return out

    def material(self, mi) -> tuple[str, str, float, tuple]:
        if mi is None:
            return ('mat', None), '', 1.0, _DEFAULT_COLOR
        mats = self.doc.get('materials') or []
        mi = int(mi)
        if not (0 <= mi < len(mats)):
            return ('mat', None), '', 1.0, _DEFAULT_COLOR
        m = mats[mi]
        pbr = m.get('pbrMetallicRoughness') or {}
        bcf = pbr.get('baseColorFactor') or [1.0, 1.0, 1.0, 1.0]
        alpha = 1.0
        if m.get('alphaMode', 'OPAQUE') == 'BLEND' and len(bcf) >= 4:
            alpha = _clamp01(bcf[3])
        tr = (m.get('extensions') or {}).get('KHR_materials_transmission') or {}
        tf = _clamp01(tr.get('transmissionFactor', 0.0)) if tr else 0.0
        if tf > 0:  # glass is usually done with transmission, not alpha
            alpha = min(alpha, max(0.2, 1.0 - tf))
        return ('mat', mi), str(m.get('name') or ''), alpha, _color3(bcf, (1.0, 1.0, 1.0))

    def primitive(self, prim: dict, cache_key) -> tuple[np.ndarray, np.ndarray] | None:
        if cache_key in self._prim_cache:
            return self._prim_cache[cache_key]
        res = None
        ext = prim.get('extensions') or {}
        if _DRACO_EXT in ext:
            raise _compressed_error(self.name, 'Draco mesh compression')
        mode = int(prim.get('mode', 4))
        attrs = prim.get('attributes') or {}
        if mode in (0, 1, 2, 3):
            self.skipped['points' if mode == 0 else 'lines'] += 1
        elif mode not in (4, 5, 6):
            self.skipped['unknown'] += 1
        elif attrs.get('POSITION') is None:
            self.skipped['noposition'] += 1
        else:
            pos = self.accessor_float(attrs['POSITION'])
            if pos.shape[1] < 3:
                raise MeshLoadError(f'{self.name} has a POSITION accessor that is not 3D.')
            pos = pos[:, :3]
            if prim.get('indices') is not None:
                iarr, iacc = self.accessor_raw(prim['indices'])
                if iacc.get('type') != 'SCALAR' or int(iacc['componentType']) == 5126:
                    raise MeshLoadError(f'{self.name} has an invalid index accessor.')
                idx = iarr[:, 0].astype(np.int64)
            else:
                idx = np.arange(len(pos), dtype=np.int64)
            n = len(idx)
            if mode == 4:
                if n % 3:
                    self.b.notes.append('Ignored trailing indices in a triangle list that is not a multiple of 3')
                tris = idx[:n - n % 3].reshape(-1, 3)
            elif mode == 5:
                if n < 3:
                    tris = np.empty((0, 3), np.int64)
                else:
                    i = np.arange(n - 2)
                    odd = (i % 2).astype(bool)
                    tris = np.stack((idx[i], np.where(odd, idx[i + 2], idx[i + 1]),
                                     np.where(odd, idx[i + 1], idx[i + 2])), axis=1)
            else:
                if n < 3:
                    tris = np.empty((0, 3), np.int64)
                else:
                    i = np.arange(n - 2)
                    tris = np.stack((idx[i + 1], idx[i + 2], np.full(n - 2, idx[0])), axis=1)
            res = (pos, tris)
        self._prim_cache[cache_key] = res
        return res

    def run(self, prog: _Progress) -> None:
        doc = self.doc
        for e in doc.get('extensionsRequired') or []:
            if e == _DRACO_EXT:
                raise _compressed_error(self.name, 'Draco mesh compression')
            if e in _MESHOPT_EXTS:
                raise _compressed_error(self.name, 'meshopt compression')
        unknown_req = [e for e in (doc.get('extensionsRequired') or [])
                       if e not in ('KHR_mesh_quantization', 'KHR_texture_transform', 'KHR_materials_unlit',
                                    'KHR_texture_basisu', 'EXT_texture_webp', 'KHR_materials_emissive_strength')]
        if unknown_req:
            self.b.notes.append('Required glTF extensions not supported (ignored): ' + ', '.join(map(str, unknown_req)))
        nodes = doc.get('nodes') or []
        meshes = doc.get('meshes') or []

        scenes = doc.get('scenes') or []
        roots: list[int]
        if scenes:
            si = doc.get('scene', 0)
            si = int(si) if si is not None else 0
            if not (0 <= si < len(scenes)):
                si = 0
            roots = [int(r) for r in (scenes[si].get('nodes') or [])]
        else:
            child = {int(c) for n in nodes for c in (n.get('children') or [])}
            roots = [i for i in range(len(nodes)) if i not in child]
        instances = []  # (node index, world matrix)
        visited = set()
        stack = [(r, np.eye(4)) for r in reversed(roots)]
        bad_refs = 0
        while stack:
            ni, parent = stack.pop()
            if not (0 <= ni < len(nodes)) or ni in visited:
                bad_refs += 1
                continue
            visited.add(ni)
            node = nodes[ni]
            world = parent @ _gltf_node_matrix(node)
            if node.get('mesh') is not None:
                instances.append((ni, world))
            if 'EXT_mesh_gpu_instancing' in (node.get('extensions') or {}):
                self.b.notes.append('GPU instancing (EXT_mesh_gpu_instancing) is not supported; only one copy is shown')
            for c in reversed(node.get('children') or []):
                stack.append((int(c), world))
        if bad_refs:
            self.b.notes.append(f'Ignored {_plural(bad_refs, "invalid or repeated node reference")}')
        if not instances and not nodes and meshes:
            instances = [(-1 - mi, np.eye(4)) for mi in range(len(meshes))]
            self.b.notes.append('File has no node hierarchy; meshes were placed at the origin')
        prog(0.1)
        total = max(1, len(instances))
        for k, (ni, world) in enumerate(instances):
            if ni >= 0:
                node = nodes[ni]
                mi = int(node['mesh'])
            else:
                node = {}
                mi = -1 - ni
            if not (0 <= mi < len(meshes)):
                self.b.notes.append('Ignored a node that references a missing mesh')
                continue
            mesh = meshes[mi]
            pname = str(node.get('name') or mesh.get('name') or '')
            M3 = world[:3, :3]
            flip = np.linalg.det(M3) < 0
            for pi, prim in enumerate(mesh.get('primitives') or []):
                res = self.primitive(prim, (mi, pi))
                if res is None:
                    continue
                pos, tris = res
                if not len(tris):
                    continue
                mkey, mname, alpha, color = self.material(prim.get('material'))
                part = self.b.part((ni, mkey), pname, mname, alpha, color)
                v = pos @ M3.T + world[:3, 3]
                self.b.add(v, tris[:, ::-1] if flip else tris, part)
            prog(0.1 + 0.8 * (k + 1) / total)
        s = self.skipped
        if s['lines']:
            self.b.notes.append(f'Skipped {_plural(s["lines"], "line primitive")}')
        if s['points']:
            self.b.notes.append(f'Skipped {_plural(s["points"], "point primitive")}')
        if s['unknown']:
            self.b.notes.append(f'Skipped {_plural(s["unknown"], "primitive")} with an unknown mode')
        if s['noposition']:
            self.b.notes.append(f'Skipped {_plural(s["noposition"], "primitive")} without positions')


def _load_gltf(path: str, data: bytes, prog: _Progress) -> LoadedMesh:
    name = os.path.basename(path)
    b = _Builder(name)
    mv = memoryview(data)
    bin_chunk = None
    is_glb = data[:4] == b'glTF'
    if is_glb:
        if len(data) < 20:
            raise MeshLoadError(f'{name} is truncated (incomplete GLB header).')
        _, version, length = struct.unpack_from('<4sII', data, 0)
        if version != 2:
            raise MeshLoadError(f'{name} is glTF version {version}; only glTF 2.0 is supported. '
                                'Re-export it as glTF 2.0.')
        if length > len(data):
            raise MeshLoadError(f'{name} is truncated: the GLB header says {length:,} bytes '
                                f'but the file has {len(data):,}.')
        off = 12
        json_bytes = None
        while off + 8 <= length:
            clen, ctype = struct.unpack_from('<II', data, off)
            off += 8
            if off + clen > length:
                raise MeshLoadError(f'{name} is truncated (a GLB chunk runs past the end of the file).')
            if ctype == 0x4E4F534A and json_bytes is None:
                json_bytes = bytes(mv[off:off + clen])
            elif ctype == 0x004E4942 and bin_chunk is None:
                bin_chunk = mv[off:off + clen]
            off += clen
        if json_bytes is None:
            raise MeshLoadError(f'{name} has no JSON chunk; it is not a valid GLB file.')
    else:
        json_bytes = data
    try:
        doc = json.loads(_decode_text(json_bytes).strip('\x00 \t\r\n'))
    except (ValueError, RecursionError):
        raise MeshLoadError(f'{name} does not contain valid glTF JSON.') from None
    if not isinstance(doc, dict):
        raise MeshLoadError(f'{name} does not contain valid glTF JSON.')
    ver = str((doc.get('asset') or {}).get('version', '2.0'))
    if ver.startswith('1'):
        raise MeshLoadError(f'{name} is glTF 1.0; only glTF 2.0 is supported. Re-export it as glTF 2.0.')
    prog(0.05)
    g = _Gltf(path, doc, bin_chunk, b, is_glb)
    g.run(prog.sub(0.05, 0.9))
    return b.build('glb' if is_glb else 'gltf', unit='m', up='+Y', forward='+Z')


# ----------------------------------------------------------------------------
# FBX (binary + ASCII)
# ----------------------------------------------------------------------------

class _FbxNode:
    __slots__ = ('name', 'props', 'children')

    def __init__(self, name: str, props: list, children: list):
        self.name = name
        self.props = props
        self.children = children

    def child(self, name: str) -> '_FbxNode | None':
        for c in self.children:
            if c.name == name:
                return c
        return None


class _FbxArray:
    """Lazily decoded binary FBX array property."""
    __slots__ = ('data', 'tcode', 'length', 'encoding', 'offset', 'clen', 'fname')
    _DT = {'f': np.dtype('<f4'), 'd': np.dtype('<f8'), 'l': np.dtype('<i8'), 'i': np.dtype('<i4'), 'b': np.dtype('u1')}

    def __init__(self, data, tcode, length, encoding, offset, clen, fname):
        self.data, self.tcode, self.length = data, tcode, length
        self.encoding, self.offset, self.clen, self.fname = encoding, offset, clen, fname

    def decode(self) -> np.ndarray:
        dt = self._DT[self.tcode]
        nbytes = self.length * dt.itemsize
        bad = MeshLoadError(f'{self.fname} contains a corrupt FBX array.')
        if nbytes > MAX_ARRAY_BYTES:
            raise MeshLoadError(f'{self.fname} declares an implausibly large FBX array ({self.length:,} elements).')
        raw = memoryview(self.data)[self.offset:self.offset + self.clen]
        if self.encoding == 0:
            if self.clen < nbytes:
                raise bad
            return np.frombuffer(raw, dt, self.length).copy()
        if self.encoding == 1:
            if nbytes > self.clen * _ZLIB_MAX_RATIO + 4096:
                raise bad
            try:
                d = zlib.decompressobj()
                out = d.decompress(raw, nbytes) if nbytes else b''
            except zlib.error:
                raise bad from None
            if len(out) != nbytes:
                raise bad
            return np.frombuffer(out, dt, self.length).copy()
        raise bad


def _fbx_parse_binary(data: bytes, name: str) -> tuple[int, list[_FbxNode]]:
    n = len(data)
    if n < 27:
        raise MeshLoadError(f'{name} is truncated (incomplete FBX header).')
    version = struct.unpack_from('<I', data, 23)[0]
    wide = version >= 7500
    hdr = struct.Struct('<QQQB' if wide else '<IIIB')
    null_len = 25 if wide else 13
    trunc = MeshLoadError(f'{name} is truncated or corrupt (FBX node data runs past the end of the file).')
    unpack_scalar = {
        'Y': struct.Struct('<h'), 'C': struct.Struct('<B'), 'I': struct.Struct('<i'),
        'F': struct.Struct('<f'), 'D': struct.Struct('<d'), 'L': struct.Struct('<q'),
    }
    arr_hdr = struct.Struct('<III')
    u32 = struct.Struct('<I')

    def read_node(pos: int, limit: int, depth: int):
        if depth > 64:
            raise MeshLoadError(f'{name} has an implausibly deep FBX node hierarchy.')
        if pos + hdr.size > limit:
            raise trunc
        end_off, nprops, plen, nlen = hdr.unpack_from(data, pos)
        if end_off == 0:
            return None, pos + null_len
        if end_off > limit or end_off <= pos:
            raise trunc
        pos += hdr.size
        nm = data[pos:pos + nlen].decode('utf-8', 'replace')
        pos += nlen
        pend = pos + plen
        if pend > end_off or nprops > plen:
            raise trunc
        props = []
        for _ in range(nprops):
            if pos >= pend:
                raise trunc
            t = chr(data[pos])
            pos += 1
            st = unpack_scalar.get(t)
            if st is not None:
                if pos + st.size > pend:
                    raise trunc
                v = st.unpack_from(data, pos)[0]
                props.append(bool(v) if t == 'C' else v)
                pos += st.size
            elif t in 'fdlib':
                if pos + 12 > pend:
                    raise trunc
                alen, enc, clen = arr_hdr.unpack_from(data, pos)
                pos += 12
                if pos + clen > pend:
                    raise trunc
                props.append(_FbxArray(data, t, alen, enc, pos, clen, name))
                pos += clen
            elif t in 'SR':
                if pos + 4 > pend:
                    raise trunc
                ln = u32.unpack_from(data, pos)[0]
                pos += 4
                if pos + ln > pend:
                    raise trunc
                raw = data[pos:pos + ln]
                props.append(raw.decode('utf-8', 'replace') if t == 'S' else raw)
                pos += ln
            else:
                raise MeshLoadError(f'{name} is corrupt (unknown FBX property type).')
        pos = pend
        children = []
        while pos < end_off:
            child, pos = read_node(pos, end_off, depth + 1)
            if child is None:
                break
            children.append(child)
        return _FbxNode(nm, props, children), end_off

    pos = 27
    top = []
    while pos + hdr.size <= n:
        node, pos = read_node(pos, n, 0)
        if node is None:
            break
        top.append(node)
    return version, top


_FBX_KEY = re.compile(r'([A-Za-z_][A-Za-z0-9_]*)\s*:\s*(.*)$')
_FBX_TOKEN = re.compile(r'"[^"]*"|[^,\s][^,]*')
_FBX_INT = re.compile(r'[-+]?\d+$')


def _fbx_ascii_value(tok: str):
    tok = tok.strip()
    if tok.startswith('"'):
        return tok[1:-1] if tok.endswith('"') and len(tok) >= 2 else tok[1:]
    if _FBX_INT.match(tok):
        return int(tok)
    try:
        return float(tok)
    except ValueError:
        return tok


def _fbx_ascii_props(rest: str) -> list:
    rest = rest.strip()
    if not rest:
        return []
    if '"' not in rest and ',' in rest:
        body = rest.rstrip(',')
        try:
            return [np.array(list(map(float, body.split(','))), dtype=np.float64)]
        except ValueError:
            pass
    return [_fbx_ascii_value(t) for t in _FBX_TOKEN.findall(rest)]


def _fbx_parse_ascii(text: str, name: str) -> list[_FbxNode]:
    root = _FbxNode('', [], [])
    stack = [root]
    lines = text.splitlines()
    i, n = 0, len(lines)
    while i < n:
        line = lines[i].strip()
        i += 1
        if not line or line[0] == ';':
            continue
        if line[0] == '}':
            if len(stack) > 1:
                node = stack.pop()
                if node.props and isinstance(node.props[0], str) and node.props[0].startswith('*'):
                    a = node.child('a')
                    node.props = a.props if a is not None else []
                    node.children = []
            continue
        m = _FBX_KEY.match(line)
        if not m:
            continue
        key, rest = m.group(1), m.group(2)
        if rest.endswith(','):
            parts = [rest]
            while i < n and parts[-1].rstrip().endswith(','):
                nxt = lines[i].strip()
                i += 1
                if nxt and nxt[0] != ';':
                    parts.append(nxt)
            rest = ''.join(parts)
        opens = rest.endswith('{')
        if opens:
            rest = rest[:-1].rstrip()
        node = _FbxNode(key, _fbx_ascii_props(rest), [])
        stack[-1].children.append(node)
        if opens:
            if len(stack) > 128:
                raise MeshLoadError(f'{name} has an implausibly deep FBX node hierarchy.')
            stack.append(node)
    return root.children


def _fbx_array(node: _FbxNode | None) -> np.ndarray | None:
    if node is None or not node.props:
        return None
    p0 = node.props[0]
    if isinstance(p0, _FbxArray):
        return p0.decode()
    if isinstance(p0, np.ndarray):
        return p0
    try:
        return np.array([float(v) for v in node.props], dtype=np.float64)
    except (TypeError, ValueError):
        return None


def _fbx_name(node: _FbxNode) -> str:
    for p in node.props[:3]:
        if isinstance(p, str):
            if '\x00\x01' in p:
                return p.split('\x00\x01', 1)[0]
            if '::' in p:
                return p.split('::', 1)[1]
    if len(node.props) > 1 and isinstance(node.props[1], str):
        return node.props[1]
    return ''


def _fbx_id(node: _FbxNode):
    if not node.props:
        return None
    p0 = node.props[0]
    if isinstance(p0, (int, str)):
        return p0
    if isinstance(p0, float) and p0.is_integer():
        return int(p0)
    return None


def _fbx_props(node: _FbxNode | None) -> dict:
    out: dict = {}
    if node is None:
        return out
    for pn in node.children:
        if pn.name == 'Properties70':
            for p in pn.children:
                if p.name == 'P' and p.props and isinstance(p.props[0], str):
                    out[p.props[0]] = p.props[4:]
        elif pn.name == 'Properties60':
            for p in pn.children:
                if p.name == 'Property' and p.props and isinstance(p.props[0], str):
                    out[p.props[0]] = p.props[3:]
    return out


def _fbx_vec(props: dict, key: str, default) -> np.ndarray:
    vals = props.get(key)
    if vals:
        try:
            v = [float(x) for x in vals[:3]]
            if len(v) == 3 and all(math.isfinite(x) for x in v):
                return np.array(v)
        except (TypeError, ValueError):
            pass
    return np.array(default, dtype=np.float64)


def _fbx_num(props: dict, key: str, default=None):
    vals = props.get(key)
    if vals:
        try:
            v = float(vals[0])
            if math.isfinite(v):
                return v
        except (TypeError, ValueError):
            pass
    return default


def _rot_euler(deg, order: int = 0) -> np.ndarray:
    rx, ry, rz = (math.radians(float(a)) for a in deg)
    cx, sx, cy, sy, cz, sz = math.cos(rx), math.sin(rx), math.cos(ry), math.sin(ry), math.cos(rz), math.sin(rz)
    R = {
        'X': np.array([[1, 0, 0], [0, cx, -sx], [0, sx, cx]]),
        'Y': np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]]),
        'Z': np.array([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]]),
    }
    seq = {0: 'XYZ', 1: 'XZY', 2: 'YZX', 3: 'YXZ', 4: 'ZXY', 5: 'ZYX'}.get(int(order), 'XYZ')
    M = np.eye(3)
    for ax in seq:  # first axis is applied first
        M = R[ax] @ M
    return M


def _m4(R=None, t=None, s=None) -> np.ndarray:
    M = np.eye(4)
    if R is not None:
        M[:3, :3] = R
    if s is not None:
        M[:3, :3] = M[:3, :3] * np.asarray(s)[None, :]
    if t is not None:
        M[:3, 3] = t
    return M


def _fbx_local_matrix(props: dict) -> np.ndarray:
    z = (0.0, 0.0, 0.0)
    T = _fbx_vec(props, 'Lcl Translation', z)
    R = _fbx_vec(props, 'Lcl Rotation', z)
    S = _fbx_vec(props, 'Lcl Scaling', (1.0, 1.0, 1.0))
    order = int(_fbx_num(props, 'RotationOrder', 0) or 0)
    Roff = _fbx_vec(props, 'RotationOffset', z)
    Rp = _fbx_vec(props, 'RotationPivot', z)
    Soff = _fbx_vec(props, 'ScalingOffset', z)
    Sp = _fbx_vec(props, 'ScalingPivot', z)
    active = _fbx_num(props, 'RotationActive', 1.0)
    pre = _fbx_vec(props, 'PreRotation', z) if active else np.zeros(3)
    post = _fbx_vec(props, 'PostRotation', z) if active else np.zeros(3)
    return (_m4(t=T) @ _m4(t=Roff) @ _m4(t=Rp) @ _m4(R=_rot_euler(pre)) @ _m4(R=_rot_euler(R, order))
            @ _m4(R=_rot_euler(post).T) @ _m4(t=-Rp) @ _m4(t=Soff) @ _m4(t=Sp) @ _m4(s=S) @ _m4(t=-Sp))


def _fbx_geometric_matrix(props: dict) -> np.ndarray:
    z = (0.0, 0.0, 0.0)
    return (_m4(t=_fbx_vec(props, 'GeometricTranslation', z))
            @ _m4(R=_rot_euler(_fbx_vec(props, 'GeometricRotation', z)))
            @ _m4(s=_fbx_vec(props, 'GeometricScaling', (1.0, 1.0, 1.0))))


def _fbx_axis(props: dict, axis_key: str, sign_key: str) -> str | None:
    a = _fbx_num(props, axis_key)
    if a is None or int(a) not in (0, 1, 2):
        return None
    s = _fbx_num(props, sign_key, 1.0)
    return ('-' if s is not None and s < 0 else '+') + 'XYZ'[int(a)]


def _fbx_material_info(node: _FbxNode) -> tuple[str, float, tuple]:
    p = _fbx_props(node)
    color = _DEFAULT_COLOR
    for key in ('DiffuseColor', 'Diffuse'):
        if p.get(key):
            color = _color3(p[key])
            break
    alpha = 1.0
    op = _fbx_num(p, 'Opacity')
    if op is not None:
        alpha = op
    else:
        tf = _fbx_num(p, 'TransparencyFactor')
        if tf is not None:
            tc = p.get('TransparentColor')
            tcv = float(np.mean(_color3(tc, (1.0, 1.0, 1.0)))) if tc else 1.0
            alpha = 1.0 - tf * tcv
    return _fbx_name(node), _clamp01(alpha), color


def _fbx_extract(top: list[_FbxNode], name: str, prog: _Progress) -> LoadedMesh:
    b = _Builder(name)
    sections = {}
    for nd in top:
        sections.setdefault(nd.name, nd)
    objects = sections.get('Objects')
    if objects is None:
        raise MeshLoadError(f'{name} has no Objects section; it is not a valid FBX scene.')
    geoms: dict = {}
    models: dict = {}
    materials: dict = {}
    for o in objects.children:
        oid = _fbx_id(o)
        if oid is None:
            continue
        if o.name == 'Geometry':
            geoms[oid] = o
        elif o.name == 'Model':
            models[oid] = o
            if o.child('Vertices') is not None:  # FBX 6: geometry lives inside the model
                geoms[('embedded', oid)] = o
        elif o.name == 'Material':
            materials[oid] = o
    parent_of: dict = {}
    geom_models: dict = {}
    model_mats: dict = {}
    conns = sections.get('Connections')
    for c in (conns.children if conns is not None else []):
        if c.name not in ('C', 'Connect') or len(c.props) < 3 or not isinstance(c.props[0], str) \
                or c.props[0] != 'OO':
            continue
        child, parent = c.props[1], c.props[2]
        if isinstance(child, float) and child.is_integer():
            child = int(child)
        if isinstance(parent, float) and parent.is_integer():
            parent = int(parent)
        if not isinstance(child, (int, str)) or not isinstance(parent, (int, str)):
            continue
        if child in geoms and parent in models:
            geom_models.setdefault(child, []).append(parent)
        elif child in models and (parent in models or parent == 0):
            # a model is also connected to display layers, selection sets...: only a model (or the root) is its parent
            parent_of.setdefault(child, parent)
        elif child in materials and parent in models:
            model_mats.setdefault(parent, []).append(child)
    for key in list(geoms):
        if isinstance(key, tuple):
            geom_models.setdefault(key, []).append(key[1])

    world_cache: dict = {}

    def world(mid, depth=0):
        if mid in world_cache:
            return world_cache[mid]
        if depth > 256:
            return np.eye(4)
        world_cache[mid] = np.eye(4)  # cycle guard
        M = _fbx_local_matrix(_fbx_props(models[mid]))
        par = parent_of.get(mid)
        if par is not None and par in models and par != mid:
            M = world(par, depth + 1) @ M
        world_cache[mid] = M
        return M

    prog(0.1)
    gp = _fbx_props(sections.get('GlobalSettings'))
    total = max(1, len(geoms))
    orphans = 0
    for gi, (gid, g) in enumerate(geoms.items()):
        prog(0.1 + 0.8 * gi / total)
        if len(g.props) > 2 and isinstance(g.props[2], str) and g.props[2] not in ('Mesh', '') \
                and not isinstance(gid, tuple):
            continue  # Shape (blend shape), NurbsCurve, Line ...
        va = _fbx_array(g.child('Vertices'))
        pa = _fbx_array(g.child('PolygonVertexIndex'))
        if va is None or pa is None or not len(va) or not len(pa):
            continue
        if len(va) % 3:
            b.notes.append('A geometry had a vertex array that is not a multiple of 3; extra values ignored')
        verts = np.asarray(va[:len(va) - len(va) % 3], dtype=np.float64).reshape(-1, 3)
        pvi = np.asarray(pa)
        if pvi.dtype.kind == 'f':
            ok = np.isfinite(pvi) & (np.abs(pvi) < 2.0 ** 62)
            pvi = np.where(ok, pvi, 0)
        pvi = pvi.astype(np.int64)
        ends = pvi < 0
        real = np.where(ends, ~pvi, pvi)
        end_pos = np.flatnonzero(ends)
        if not len(end_pos):
            continue
        last = int(end_pos[-1])
        if last != len(pvi) - 1:
            b.notes.append('Ignored an unterminated polygon at the end of a geometry')
        counts = np.diff(np.concatenate(([-1], end_pos)))
        _polygon_notes(counts, b.notes)
        tris, poly = _fan_triangulate(counts, real[:last + 1])
        if not len(tris):
            continue
        # per-polygon material slots
        mat_poly = None
        lem = g.child('LayerElementMaterial')
        if lem is not None:
            mit = lem.child('MappingInformationType')
            mapping = mit.props[0] if mit is not None and mit.props else ''
            ma = _fbx_array(lem.child('Materials'))
            if ma is not None and len(ma):
                ma = np.where(np.isfinite(ma.astype(np.float64)), ma, 0).astype(np.int64)
                if mapping == 'ByPolygon' and len(ma) >= len(counts):
                    mat_poly = ma[:len(counts)]
                else:
                    mat_poly = np.full(len(counts), ma[0], np.int64)
        owners = geom_models.get(gid, [])
        if not owners:
            orphans += 1
            owners = [None]
        for mid in owners:
            if mid is None:
                W = np.eye(4)
                pname = _fbx_name(g)
                mats = []
            else:
                mprops = _fbx_props(models[mid])
                W = world(mid) @ _fbx_geometric_matrix(mprops)
                pname = _fbx_name(models[mid])
                mats = model_mats.get(mid, [])
            if mat_poly is not None and mats:
                slot = mat_poly[poly]
                uniq = np.unique(slot)
                fpart = np.empty(len(tris), np.int32)
                for s in uniq:
                    if 0 <= s < len(mats):
                        mname, alpha, color = _fbx_material_info(materials[mats[s]])
                    else:
                        mname, alpha, color = '', 1.0, _DEFAULT_COLOR
                    fpart[slot == s] = b.part((mid, gid, int(s)), pname, mname, alpha, color)
            elif mats:
                mname, alpha, color = _fbx_material_info(materials[mats[0]])
                fpart = b.part((mid, gid, 0), pname, mname, alpha, color)
            else:
                fpart = b.part((mid, gid, -1), pname, '')
            M3 = W[:3, :3]
            v = verts @ M3.T + W[:3, 3]
            b.add(v, tris[:, ::-1] if np.linalg.det(M3) < 0 else tris, fpart)
    if orphans:
        b.notes.append(f'{_plural(orphans, "geometry", "geometries")} not attached to a model placed at the origin')
    usf = _fbx_num(gp, 'UnitScaleFactor')
    unit = None
    if usf is not None:
        for val, u in ((1.0, 'cm'), (100.0, 'm'), (0.1, 'mm'), (2.54, 'in')):
            if abs(usf - val) <= 1e-6 * val:
                unit = u
                break
        else:
            b.notes.append(f'FBX unit scale factor is {usf:g} cm per unit')
    up = _fbx_axis(gp, 'UpAxis', 'UpAxisSign')
    fwd = _fbx_axis(gp, 'FrontAxis', 'FrontAxisSign')
    prog(0.9)
    return b.build('fbx', unit=unit, up=up, forward=fwd)


def _load_fbx(path: str, data: bytes, prog: _Progress) -> LoadedMesh:
    name = os.path.basename(path)
    if data.startswith(b'Kaydara FBX Binary'):
        version, top = _fbx_parse_binary(data, name)
        prog(0.4)
        mesh = _fbx_extract(top, name, prog.sub(0.4, 1.0))
        if version < 7100 or version > 7700:
            mesh.notes.append(f'FBX version {version} is outside the tested range (7100-7700)')
        return mesh
    if _looks_binary(data[:4096]):
        raise MeshLoadError(f'{name} is not a recognised FBX file (neither binary nor ASCII FBX).')
    text = _decode_text(data)
    head = text[:8192]
    if 'FBX' not in head and 'Objects' not in text and 'FBXHeaderExtension' not in head:
        raise MeshLoadError(f'{name} is not a recognised FBX file.')
    top = _fbx_parse_ascii(text, name)
    prog(0.4)
    return _fbx_extract(top, name, prog.sub(0.4, 1.0))


# ----------------------------------------------------------------------------
# STEP
# ----------------------------------------------------------------------------

_STEP_MISSING = ('STEP needs the optional gmsh package (pip install gmsh). '
                 'Or export the model from CAD as .glb or .obj.')


def _load_step(path: str, prog: _Progress) -> LoadedMesh:
    name = os.path.basename(path)
    if not os.path.isfile(path):
        raise MeshLoadError(f'File not found: {path}')
    try:
        with open(path, 'rb') as fh:
            head = fh.read(64)
    except OSError as exc:
        raise MeshLoadError(f'Could not open {name}: {exc.strerror or exc}.') from None
    if not head.lstrip().startswith(b'ISO-10303-21'):
        raise MeshLoadError(f'{name} is not a STEP (ISO 10303-21) file.')
    try:
        import gmsh  # type: ignore
    except Exception:
        gmsh = None
    if gmsh is not None:
        return _load_step_gmsh(gmsh, path, prog)
    try:
        import cascadio  # type: ignore
    except Exception:
        cascadio = None
    if cascadio is not None:
        return _load_step_cascadio(cascadio, path, prog)
    raise MeshLoadError(_STEP_MISSING)


def _load_step_gmsh(gmsh, path: str, prog: _Progress) -> LoadedMesh:
    name = os.path.basename(path)
    b = _Builder(name)
    own = not gmsh.isInitialized()
    if own:
        try:
            gmsh.initialize(readConfigFiles=False, interruptible=False)
        except TypeError:
            gmsh.initialize()
    model_name = f'vantage_step_{os.getpid()}_{id(b)}'
    try:
        gmsh.option.setNumber('General.Terminal', 0)
        try:
            gmsh.option.setString('Geometry.OCCTargetUnit', 'MM')
        except Exception:
            pass
        gmsh.model.add(model_name)
        try:
            gmsh.model.occ.importShapes(path)
            gmsh.model.occ.synchronize()
        except Exception as exc:
            raise MeshLoadError(f'gmsh could not read {name}; the STEP file may be damaged '
                                'or use unsupported entities.') from exc
        prog(0.25)
        surfaces = gmsh.model.getEntities(2)
        if not surfaces:
            raise MeshLoadError(f'No surfaces found in {name}.')
        bb = gmsh.model.getBoundingBox(-1, -1)
        diag = math.dist(bb[:3], bb[3:])
        if math.isfinite(diag) and diag > 0:
            gmsh.option.setNumber('Mesh.MeshSizeMax', diag / 40.0)
            gmsh.option.setNumber('Mesh.MeshSizeMin', diag / 4000.0)
        gmsh.option.setNumber('Mesh.MeshSizeFactor', 1.0)
        gmsh.option.setNumber('Mesh.MeshSizeFromCurvature', 12)
        gmsh.option.setNumber('Mesh.Algorithm', 6)
        try:
            gmsh.model.mesh.generate(2)
        except Exception as exc:
            raise MeshLoadError(f'gmsh could not tessellate {name}.') from exc
        prog(0.8)
        tags, coords, _ = gmsh.model.mesh.getNodes()
        tags = np.asarray(tags, dtype=np.int64)
        verts = np.asarray(coords, dtype=np.float64).reshape(-1, 3)
        if not len(tags):
            raise MeshLoadError(f'No triangles found in {name}.')
        order = np.argsort(tags)
        stags = tags[order]

        def lookup(t):
            t = np.asarray(t, dtype=np.int64)
            pos = np.clip(np.searchsorted(stags, t), 0, len(stags) - 1)
            ok = stags[pos] == t
            return np.where(ok, order[pos], -1)

        surf_part: dict[int, tuple] = {}
        for _, vtag in gmsh.model.getEntities(3):
            try:
                vname = gmsh.model.getEntityName(3, vtag) or ''
            except Exception:
                vname = ''
            _, down = gmsh.model.getAdjacencies(3, vtag)
            for s in down:
                surf_part.setdefault(int(s), vname)
        tri_chunks = []
        part_chunks = []
        for _, stag in surfaces:
            if int(stag) in surf_part:
                pname = surf_part[int(stag)]
            else:
                try:
                    pname = gmsh.model.getEntityName(2, stag) or ''
                except Exception:
                    pname = ''
            pname = pname.rstrip('/').rsplit('/', 1)[-1]
            etypes, _, enodes = gmsh.model.mesh.getElements(2, stag)
            for et, en in zip(etypes, enodes):
                en = np.asarray(en, dtype=np.int64)
                if et == 2:
                    t = lookup(en).reshape(-1, 3)
                elif et == 3:
                    q = lookup(en).reshape(-1, 4)
                    t = np.concatenate((q[:, [0, 1, 2]], q[:, [0, 2, 3]]))
                else:
                    continue
                pid = b.part(('step', pname), pname, '')
                tri_chunks.append(t)
                part_chunks.append(np.full(len(t), pid, np.int32))
        if not tri_chunks:
            raise MeshLoadError(f'No triangles found in {name}.')
        b.add(verts, np.concatenate(tri_chunks), np.concatenate(part_chunks))
        b.notes.append('STEP model tessellated with gmsh')
    finally:
        try:
            gmsh.model.setCurrent(model_name)
            gmsh.model.remove()
        except Exception:
            pass
        if own:
            try:
                gmsh.finalize()
            except Exception:
                pass
    prog(0.95)
    return b.build('step', unit='mm')


def _load_step_cascadio(cascadio, path: str, prog: _Progress) -> LoadedMesh:
    name = os.path.basename(path)
    fd, tmp = tempfile.mkstemp(suffix='.glb', prefix='vantage_step_')
    os.close(fd)
    try:
        try:
            cascadio.step_to_glb(path, tmp)
        except Exception as exc:
            raise MeshLoadError(f'cascadio could not convert {name}; the STEP file may be damaged.') from exc
        prog(0.6)
        with open(tmp, 'rb') as fh:
            data = fh.read()
        if not data:
            raise MeshLoadError(f'No triangles found in {name}.')
        mesh = _load_gltf(tmp, data, prog.sub(0.6, 1.0))
    finally:
        try:
            os.remove(tmp)
        except OSError:
            pass
    mesh.fmt = 'step'
    # cascadio writes metres but keeps the CAD axes (often Z up, X forward or rearward), not glTF's Y up / Z forward;
    # STEP has no axis convention, so leave the orientation to the import wizard's guess, as for gmsh
    mesh.up_hint = mesh.forward_hint = None
    mesh.notes.append('STEP model converted with cascadio')
    return mesh


# ----------------------------------------------------------------------------
# public API
# ----------------------------------------------------------------------------

_LOADERS = {
    '.obj': _load_obj,
    '.stl': _load_stl,
    '.ply': _load_ply,
    '.gltf': _load_gltf,
    '.glb': _load_gltf,
    '.fbx': _load_fbx,
}

_FMT_LABEL = {'.obj': 'OBJ', '.stl': 'STL', '.ply': 'PLY', '.gltf': 'glTF', '.glb': 'GLB', '.fbx': 'FBX',
              '.step': 'STEP', '.stp': 'STEP'}


def load_mesh(path: str, progress=None) -> LoadedMesh:
    """Load *path* into a :class:`LoadedMesh`.

    ``progress`` is an optional ``callable(float)`` receiving monotonically
    increasing values in 0..1.  An exception raised by the callback (e.g. to
    cancel) propagates unchanged.  All file problems raise
    :class:`MeshLoadError` with a short, user-facing message.
    """
    path = os.fspath(path)
    name = os.path.basename(path) or path
    ext = os.path.splitext(path)[1].lower()
    if ext not in SUPPORTED_EXTS:
        raise MeshLoadError(f'Unsupported file type "{ext or name}". '
                            f'Supported formats: {", ".join(SUPPORTED_EXTS)}.')
    prog = _Progress(progress)
    try:
        with np.errstate(all='ignore'):  # NaN/inf are handled explicitly by the builder
            return _load_dispatch(path, name, ext, prog)
    except _CallbackError as exc:
        raise exc.exc from None
    except MeshLoadError:
        raise
    except MemoryError:
        raise MeshLoadError(f'{name} is too large to load into memory.') from None
    except Exception as exc:  # anything else means the file is not what it claims
        raise MeshLoadError(f'Could not read {name}: the {_FMT_LABEL[ext]} file appears to be '
                            f'corrupt or truncated.') from exc


def _load_dispatch(path: str, name: str, ext: str, prog: _Progress) -> LoadedMesh:
    prog(0.0)
    if ext in ('.step', '.stp'):
        mesh = _load_step(path, prog)
    else:
        try:
            if not os.path.isfile(path):
                raise MeshLoadError(f'File not found: {path}')
            if os.path.getsize(path) > MAX_FILE_BYTES:
                raise MeshLoadError(f'{name} is larger than 4 GB, which is more than Vantage can load.')
            with open(path, 'rb') as fh:
                data = fh.read()
        except OSError as exc:
            raise MeshLoadError(f'Could not open {name}: {exc.strerror or exc}.') from None
        if not data:
            raise MeshLoadError(f'{name} is empty.')
        mesh = _LOADERS[ext](path, data, prog)
    prog(1.0)
    return mesh


def mesh_summary(m: LoadedMesh) -> list[tuple[str, int, str]]:
    """Rows of (part name, triangle count, material) aggregated by part name,
    in order of first appearance.  Several materials are joined with ', '."""
    counts = np.bincount(np.asarray(m.face_part, dtype=np.int64), minlength=len(m.parts))
    rows: dict[str, list] = {}
    for i, p in enumerate(m.parts):
        row = rows.setdefault(p.name, [0, []])
        row[0] += int(counts[i]) if i < len(counts) else 0
        if p.material and p.material not in row[1]:
            row[1].append(p.material)
    return [(n, c, ', '.join(mats)) for n, (c, mats) in rows.items()]
