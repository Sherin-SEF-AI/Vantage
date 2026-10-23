"""ROS 2 bags without ROS: a CDR codec for ROS 2 messages, a writer and a reader for rosbag2 bags in MCAP storage, and a
reader for rosbag2 SQLite (.db3) bags.

Why: the rig's synthetic drives go out as ROS 2 bags that `ros2 bag play`, Foxglove and rosbags open, and recorded
drives come back in for checks, with no ROS installation and no new dependency (numpy and the standard library; the
zstandard / lz4 modules only when a bag's chunks use that compression, OpenCV only to decode compressed images).

Messages are plain dicts keyed by the field names of the .msg definitions: nested messages are dicts, arrays lists,
uint8 / byte / char arrays (image and point-cloud data) bytes. encode() fills missing fields with their defaults
(0, '', empty, a Quaternion's w = 1) and rejects keys the type does not have.

CDR. XCDR1 ("plain CDR", what every ROS 2 middleware writes): a 4-byte encapsulation header 00 01 00 00 (little-
endian; 00 00 big-endian is decoded too), then the fields in order, each primitive aligned to its own size relative
to the start of the payload after the header (8-byte types to 8); a string is a uint32 length that counts the
terminating NUL, then the bytes and the NUL; a sequence a uint32 count, then the elements (an empty sequence's
elements are not aligned, as Fast-CDR does). byte is ROS 2's octet (unsigned, as C++ uint8_t; rosbags reads it as
int8, the same byte). XCDR2 encapsulations and wstring are not supported.

Definitions. MSG_DEFS holds the types below as ROS 2 Humble and Jazzy define them; every field layout here is the
same in both distributions. Two types differ in constants only: Jazzy adds Marker.ARROW_STRIP and
NavSatStatus.STATUS_UNKNOWN / SERVICE_UNKNOWN, which MSG_DEFS carries (constants do not change the wire format).
A bag's own definitions (an MCAP schema, a .db3 message_definitions table) take precedence when they differ, and
types this module does not know are decoded from them, so messages from other distributions and from vendor packages
decode too. register() adds more definitions for writing.

MCAP (https://mcap.dev/spec) as rosbag2's mcap storage writes it: profile 'ros2', schema encoding 'ros2msg' (the
.msg text followed by each dependency after a line of 80 '=' and 'MSG: pkg/Type'), message encoding 'cdr',
channel metadata offered_qos_profiles; messages in chunks (none, zstd or lz4) each followed by its message indexes;
a summary (schemas, channels, statistics, chunk indexes, summary offsets) and real CRC-32s for the chunks, the data
section and the summary. The bag directory holds <name>_0.mcap and a metadata.yaml in rosbag2's version 5 layout,
which Humble and every later distribution read.

Reading streams: a chunk at a time, merged into log-time order whatever order the writer used (a heap over chunks
whose time ranges overlap). An MCAP file without a summary (a recorder that was killed, a truncated copy) is read by
a linear scan that stops at the first incomplete record. A chunk whose CRC or size is wrong raises ValueError, or is
skipped and noted with messages(skip_corrupt=True). Limits: rosbag2 'file' compression (a whole .mcap.zstd) is
not read (decompress it with `ros2 bag decompress` first); MCAP attachments and metadata records are skipped; the
QoS profile written for /tf_static (transient local, so late subscribers get the static transforms on playback) is
the typical rosbag2 Humble form, every other topic gets the empty profile ros2 bag play replaces with its defaults.
"""
from __future__ import annotations

import ast
import glob
import heapq
import os
import re
import sqlite3
import struct
import urllib.parse
import zlib
from typing import NamedTuple

import numpy as np

from .mathutil import quat_from_matrix

# ---------------------------------------------------------------------------------------------------------------------
# message definitions (ROS 2 Humble / Jazzy; dependencies written 'pkg/Type' as in the .msg files)

_MARKER = """int32 ARROW=0
int32 CUBE=1
int32 SPHERE=2
int32 CYLINDER=3
int32 LINE_STRIP=4
int32 LINE_LIST=5
int32 CUBE_LIST=6
int32 SPHERE_LIST=7
int32 POINTS=8
int32 TEXT_VIEW_FACING=9
int32 MESH_RESOURCE=10
int32 TRIANGLE_LIST=11
int32 ARROW_STRIP=12
int32 ADD=0
int32 MODIFY=0
int32 DELETE=2
int32 DELETEALL=3
std_msgs/Header header
string ns
int32 id
int32 type
int32 action
geometry_msgs/Pose pose
geometry_msgs/Vector3 scale
std_msgs/ColorRGBA color
builtin_interfaces/Duration lifetime
bool frame_locked
geometry_msgs/Point[] points
std_msgs/ColorRGBA[] colors
string texture_resource
sensor_msgs/CompressedImage texture
visualization_msgs/UVCoordinate[] uv_coordinates
string text
string mesh_resource
visualization_msgs/MeshFile mesh_file
bool mesh_use_embedded_materials
"""

MSG_DEFS: dict[str, str] = {
    'builtin_interfaces/msg/Time': 'int32 sec\nuint32 nanosec\n',
    'builtin_interfaces/msg/Duration': 'int32 sec\nuint32 nanosec\n',
    'std_msgs/msg/Header': 'builtin_interfaces/Time stamp\nstring frame_id\n',
    'std_msgs/msg/String': 'string data\n',
    'std_msgs/msg/Bool': 'bool data\n',
    'std_msgs/msg/Int32': 'int32 data\n',
    'std_msgs/msg/UInt32': 'uint32 data\n',
    'std_msgs/msg/Float32': 'float32 data\n',
    'std_msgs/msg/Float64': 'float64 data\n',
    'std_msgs/msg/ColorRGBA': 'float32 r\nfloat32 g\nfloat32 b\nfloat32 a\n',
    'geometry_msgs/msg/Vector3': 'float64 x\nfloat64 y\nfloat64 z\n',
    'geometry_msgs/msg/Vector3Stamped': 'std_msgs/Header header\ngeometry_msgs/Vector3 vector\n',
    'geometry_msgs/msg/Point': 'float64 x\nfloat64 y\nfloat64 z\n',
    'geometry_msgs/msg/Quaternion': 'float64 x 0\nfloat64 y 0\nfloat64 z 0\nfloat64 w 1\n',
    'geometry_msgs/msg/Pose': 'geometry_msgs/Point position\ngeometry_msgs/Quaternion orientation\n',
    'geometry_msgs/msg/PoseStamped': 'std_msgs/Header header\ngeometry_msgs/Pose pose\n',
    'geometry_msgs/msg/Transform': 'geometry_msgs/Vector3 translation\ngeometry_msgs/Quaternion rotation\n',
    'geometry_msgs/msg/TransformStamped': ('std_msgs/Header header\nstring child_frame_id\n'
                                           'geometry_msgs/Transform transform\n'),
    'geometry_msgs/msg/Twist': 'geometry_msgs/Vector3 linear\ngeometry_msgs/Vector3 angular\n',
    'geometry_msgs/msg/TwistStamped': 'std_msgs/Header header\ngeometry_msgs/Twist twist\n',
    'geometry_msgs/msg/Accel': 'geometry_msgs/Vector3 linear\ngeometry_msgs/Vector3 angular\n',
    'geometry_msgs/msg/AccelStamped': 'std_msgs/Header header\ngeometry_msgs/Accel accel\n',
    'geometry_msgs/msg/PoseWithCovariance': 'geometry_msgs/Pose pose\nfloat64[36] covariance\n',
    'geometry_msgs/msg/PoseWithCovarianceStamped': 'std_msgs/Header header\ngeometry_msgs/PoseWithCovariance pose\n',
    'geometry_msgs/msg/TwistWithCovariance': 'geometry_msgs/Twist twist\nfloat64[36] covariance\n',
    'geometry_msgs/msg/TwistWithCovarianceStamped': ('std_msgs/Header header\n'
                                                     'geometry_msgs/TwistWithCovariance twist\n'),
    'tf2_msgs/msg/TFMessage': 'geometry_msgs/TransformStamped[] transforms\n',
    'nav_msgs/msg/Odometry': ('std_msgs/Header header\nstring child_frame_id\n'
                              'geometry_msgs/PoseWithCovariance pose\ngeometry_msgs/TwistWithCovariance twist\n'),
    'sensor_msgs/msg/Image': ('std_msgs/Header header\nuint32 height\nuint32 width\nstring encoding\n'
                              'uint8 is_bigendian\nuint32 step\nuint8[] data\n'),
    'sensor_msgs/msg/CompressedImage': 'std_msgs/Header header\nstring format\nuint8[] data\n',
    'sensor_msgs/msg/RegionOfInterest': ('uint32 x_offset\nuint32 y_offset\nuint32 height\nuint32 width\n'
                                         'bool do_rectify\n'),
    'sensor_msgs/msg/CameraInfo': ('std_msgs/Header header\nuint32 height\nuint32 width\nstring distortion_model\n'
                                   'float64[] d\nfloat64[9] k\nfloat64[9] r\nfloat64[12] p\nuint32 binning_x\n'
                                   'uint32 binning_y\nsensor_msgs/RegionOfInterest roi\n'),
    'sensor_msgs/msg/PointField': ('uint8 INT8=1\nuint8 UINT8=2\nuint8 INT16=3\nuint8 UINT16=4\nuint8 INT32=5\n'
                                   'uint8 UINT32=6\nuint8 FLOAT32=7\nuint8 FLOAT64=8\n'
                                   'string name\nuint32 offset\nuint8 datatype\nuint32 count\n'),
    'sensor_msgs/msg/PointCloud2': ('std_msgs/Header header\nuint32 height\nuint32 width\n'
                                    'sensor_msgs/PointField[] fields\nbool is_bigendian\nuint32 point_step\n'
                                    'uint32 row_step\nuint8[] data\nbool is_dense\n'),
    'sensor_msgs/msg/Imu': ('std_msgs/Header header\ngeometry_msgs/Quaternion orientation\n'
                            'float64[9] orientation_covariance\ngeometry_msgs/Vector3 angular_velocity\n'
                            'float64[9] angular_velocity_covariance\ngeometry_msgs/Vector3 linear_acceleration\n'
                            'float64[9] linear_acceleration_covariance\n'),
    'sensor_msgs/msg/NavSatStatus': ('int8 STATUS_UNKNOWN=-2\nint8 STATUS_NO_FIX=-1\nint8 STATUS_FIX=0\n'
                                     'int8 STATUS_SBAS_FIX=1\nint8 STATUS_GBAS_FIX=2\nint8 status\n'
                                     'uint16 SERVICE_UNKNOWN=0\nuint16 SERVICE_GPS=1\nuint16 SERVICE_GLONASS=2\n'
                                     'uint16 SERVICE_COMPASS=4\nuint16 SERVICE_GALILEO=8\nuint16 service\n'),
    'sensor_msgs/msg/NavSatFix': ('uint8 COVARIANCE_TYPE_UNKNOWN=0\nuint8 COVARIANCE_TYPE_APPROXIMATED=1\n'
                                  'uint8 COVARIANCE_TYPE_DIAGONAL_KNOWN=2\nuint8 COVARIANCE_TYPE_KNOWN=3\n'
                                  'std_msgs/Header header\nsensor_msgs/NavSatStatus status\nfloat64 latitude\n'
                                  'float64 longitude\nfloat64 altitude\nfloat64[9] position_covariance\n'
                                  'uint8 position_covariance_type\n'),
    'sensor_msgs/msg/TimeReference': 'std_msgs/Header header\nbuiltin_interfaces/Time time_ref\nstring source\n',
    'sensor_msgs/msg/MagneticField': ('std_msgs/Header header\ngeometry_msgs/Vector3 magnetic_field\n'
                                      'float64[9] magnetic_field_covariance\n'),
    'visualization_msgs/msg/UVCoordinate': 'float32 u\nfloat32 v\n',
    'visualization_msgs/msg/MeshFile': 'string filename\nuint8[] data\n',
    'visualization_msgs/msg/Marker': _MARKER,
    'visualization_msgs/msg/MarkerArray': 'visualization_msgs/Marker[] markers\n',
}

# QoS of a latched topic (/tf_static) in rosbag2's version-5 metadata form: reliable, transient local, durations
# infinite. Every other topic is written with the empty profile, which ros2 bag play replaces with its defaults.
QOS_TRANSIENT_LOCAL = ('- history: 3\n  depth: 0\n  reliability: 1\n  durability: 1\n  deadline:\n    sec: 9223372036\n'
                       '    nsec: 854775807\n  lifespan:\n    sec: 9223372036\n    nsec: 854775807\n  liveliness: 1\n'
                       '  liveliness_lease_duration:\n    sec: 9223372036\n    nsec: 854775807\n'
                       '  avoid_ros_namespace_conventions: false')
LATCHED_TOPICS = ('/tf_static',)

# ---------------------------------------------------------------------------------------------------------------------
# .msg parsing

_PRIM = {'bool': ('?', 1), 'byte': ('B', 1), 'char': ('B', 1), 'int8': ('b', 1), 'uint8': ('B', 1),
         'int16': ('h', 2), 'uint16': ('H', 2), 'int32': ('i', 4), 'uint32': ('I', 4),
         'int64': ('q', 8), 'uint64': ('Q', 8), 'float32': ('f', 4), 'float64': ('d', 8)}
_BYTES = frozenset(('uint8', 'byte', 'char'))     # arrays of these are bytes
_STRINGS = ('string', 'wstring')
_SEP = '=' * 80
_TYPE_RE = re.compile(r'([A-Za-z_][\w/]*)(?:<=\d+)?(?:\[(<=)?(\d*)\])?$')
_IDENT = re.compile(r'[A-Za-z_]\w*$')


class _Field(NamedTuple):
    name: str
    type: str          # a primitive, 'string', 'wstring' or a full type 'pkg/msg/Type'
    kind: int          # 0 a single value, 1 a fixed array [n], 2 a sequence [] / [<=n]
    n: int
    default: object    # the definition's default value, None when it gives none


class _Spec(NamedTuple):
    fields: tuple
    constants: dict


def _norm(name: str, pkg: str = '') -> str:
    """'pkg/Type', 'pkg/msg/Type' or a bare 'Type' of package pkg -> 'pkg/msg/Type'."""
    parts = name.strip().split('/')
    if len(parts) == 1:
        if parts[0] == 'Header':                    # ROS 1 style, as some vendor definitions still write it
            return 'std_msgs/msg/Header'
        return f'{pkg}/msg/{parts[0]}' if pkg else parts[0]
    if len(parts) == 2:
        return f'{parts[0]}/msg/{parts[1]}'
    return '/'.join(parts)


def _short(t: str) -> str:
    """'pkg/msg/Type' -> 'pkg/Type' (as rosbag2 names the dependencies of a schema)."""
    p = t.split('/')
    return f'{p[0]}/{p[-1]}' if len(p) == 3 and p[1] == 'msg' else t


def _literal(text: str, base: str, array: bool):
    """A default value or constant as Python, None when it cannot be read (it never changes the layout)."""
    s = text.strip()
    try:
        if base in _STRINGS and not array:
            return ast.literal_eval(s) if s[:1] in '"\'' else s
        if base == 'bool' and not array:
            return s.lower() in ('true', '1')
        v = ast.literal_eval(s)
        if array:
            return list(v)
        return float(v) if base.startswith('float') else int(v)
    except (ValueError, SyntaxError, TypeError):
        return None


def _parse(text: str, msg_type: str) -> _Spec:
    pkg = msg_type.split('/')[0]
    fields, consts = [], {}
    for raw in text.splitlines():
        line = raw.split('#', 1)[0].strip()
        if not line:
            continue
        parts = line.split(None, 1)
        mt = _TYPE_RE.match(parts[0])
        if len(parts) < 2 or not mt:
            raise ValueError(f'{msg_type}: cannot read the definition line {raw.strip()!r}')
        base, bounded, num = mt.groups()
        if base not in _PRIM and base not in _STRINGS:
            base = _norm(base, pkg)
        rest = parts[1]
        name, eq, value = rest.partition('=')
        if eq and _IDENT.match(name.strip()):
            consts[name.strip()] = _literal(value, base, False)
            continue
        bits = rest.split(None, 1)
        if not _IDENT.match(bits[0]):
            raise ValueError(f'{msg_type}: bad field name in {raw.strip()!r}')
        if num is None:
            kind, n = 0, 0
        elif bounded or num == '':
            kind, n = 2, int(num or 0)
        else:
            kind, n = 1, int(num)
        default = _literal(bits[1], base, kind != 0) if len(bits) > 1 else None
        fields.append(_Field(bits[0], base, kind, n, default))
    return _Spec(tuple(fields), consts)


def _schema_sections(root: str, text: str) -> dict:
    """{type: .msg text} of a ros2msg schema: the root type's text, then each dependency after a line of '=' and a
    line 'MSG: pkg/Type'. The first text of a type wins."""
    out, cur, buf = {}, root, []
    lines = text.splitlines()
    i = 0
    while i < len(lines):
        s = lines[i].strip()
        if len(s) >= 3 and set(s) == {'='}:
            if cur is not None:
                out.setdefault(cur, '\n'.join(buf) + '\n')
            buf = []
            nxt = lines[i + 1].strip() if i + 1 < len(lines) else ''
            cur = _norm(nxt[4:].strip()) if nxt.startswith('MSG:') else None
            i += 2 if cur else 1
            continue
        buf.append(lines[i])
        i += 1
    if cur is not None:
        out.setdefault(cur, '\n'.join(buf) + '\n')
    return out


# ---------------------------------------------------------------------------------------------------------------------
# CDR code generation: one Python function per type (and byte order), with every run of fixed-size fields whose
# padding is known statically read or written by a single struct call

_Z = tuple(bytes(i) for i in range(9))
_U32LE = struct.Struct('<I').pack


def _str(b, p, n):
    if n > len(b) - p:
        raise ValueError('a string runs past the end of the data')
    e = p + n
    if n and b[e - 1] == 0:
        e -= 1
    return str(b[p:e], 'utf-8', 'replace')


def _bytes(b, p, n):
    if n > len(b) - p:
        raise ValueError('a byte array runs past the end of the data')
    return bytes(b[p:p + n])


def _seq(b, p, n, dt, size):
    if not n:
        return [], p
    p += -p % size
    if n * size > len(b) - p:
        raise ValueError('an array runs past the end of the data')
    return np.frombuffer(b, dt, n, p).tolist(), p + n * size


def _read_str_fn(endian):
    u = struct.Struct(endian + 'I').unpack_from

    def rs(b, p):
        p += -p % 4
        n, = u(b, p)
        return _str(b, p + 4, n), p + 4 + n
    return rs


def _ws(o, v):
    bs = v.encode() if isinstance(v, str) else bytes(v)
    o += _Z[(4 - len(o)) % 4]
    o += _U32LE(len(bs) + 1)
    o += bs
    o += b'\0'


def _wb(o, v):
    if v is None:
        v = b''
    elif isinstance(v, np.ndarray):
        v = np.ascontiguousarray(v).reshape(-1).view(np.uint8)
    elif not isinstance(v, (bytes, bytearray, memoryview)):
        v = bytes(v)
    n = memoryview(v).nbytes
    o += _Z[(4 - len(o)) % 4]
    o += _U32LE(n)
    o += v


def _wq(o, v, code, size, dt):
    if v is None:
        v = ()
    if isinstance(v, np.ndarray):
        a = np.asarray(v, dt).reshape(-1)
        n, raw = a.size, a.tobytes()
    else:
        n = len(v)
        raw = struct.pack(f'<{n}{code}', *v)
    o += _Z[(4 - len(o)) % 4]
    o += _U32LE(n)
    if n:
        o += _Z[(4 - len(o)) % size]
        o += raw


def _fl(v, n):
    if isinstance(v, np.ndarray):
        v = v.reshape(-1).tolist()
    if len(v) != n:
        raise ValueError(f'a fixed-size array needs {n} values, got {len(v)}')
    return v


def _fb(v, n):
    if isinstance(v, np.ndarray):
        v = np.ascontiguousarray(v).reshape(-1).view(np.uint8).tobytes()
    v = bytes(v)
    if len(v) != n:
        raise ValueError(f'a fixed-size byte array needs {n} bytes, got {len(v)}')
    return v


def _sq(v):
    return () if v is None else v


def _badkeys(m, keys, t):
    raise ValueError(f'{t} has no field {", ".join(sorted(str(k) for k in set(m) - keys))}')


class _Gen:
    """Source of one generated function. The position modulo k is r (k = 8 at the start of a message, 1 when
    unknown: after a string or a sequence); the pending block is a struct format still to be read or written."""

    def __init__(self, types, endian, k):
        self.types, self.e = types, endian
        self.k, self.r = k, 0
        self.lines, self.nid, self.stack = [], 0, []
        self.fmt, self.size, self.vals = [], 0, []
        self.ns = {'_Z': _Z, '_U32': _U32LE, '_str': _str, '_bytes': _bytes, '_seq': _seq, '_rs': _read_str_fn(endian),
                   '_ws': _ws, '_wb': _wb, '_wq': _wq, '_fl': _fl, '_fb': _fb, '_sq': _sq, '_badkeys': _badkeys,
                   '_E': {}}

    def new(self, prefix):
        self.nid += 1
        return f'{prefix}{self.nid}'

    def const(self, value, prefix='_c'):
        name = self.new(prefix)
        self.ns[name] = value
        return name

    def emit(self, line):
        self.lines.append('    ' + line)

    def advance(self, size):
        self.size += size
        self.r = (self.r + size) % self.k

    def align(self, a):
        if a <= 1:
            return
        if self.k >= a:
            pad = -self.r % a
            if pad:
                self.add(f'{pad}x', pad, ())
        else:
            self.flush()
            self.dynamic_align(a)
            self.k, self.r = a, 0

    def unknown(self):
        self.k, self.r = 1, 0

    def enter(self, t):
        if t in self.stack:
            raise ValueError(f'{t} contains itself')
        self.stack.append(t)
        return self.types.spec(t)

    def build(self, name, args, ret):
        src = f'def {name}({args}):\n' + '\n'.join(self.lines) + f'\n    return {ret}\n'
        exec(compile(src, f'<cdr {self.stack_root}>', 'exec'), self.ns)
        return self.ns[name]


class _DecGen(_Gen):
    def dynamic_align(self, a):
        self.emit(f'p += -p % {a}')

    def add(self, fmt, size, nvals):
        if not self.fmt:
            self.blk, self.nv = self.new('t'), 0
        self.fmt.append(fmt)
        self.advance(size)
        i = self.nv
        self.nv += len(nvals) if isinstance(nvals, tuple) else nvals
        return i

    def flush(self):
        if not self.fmt:
            return
        if self.nv:
            fn = self.const(struct.Struct(self.e + ''.join(self.fmt)).unpack_from, '_u')
            self.emit(f'{self.blk} = {fn}(b, p)')
        self.emit(f'p += {self.size}')
        self.fmt, self.size = [], 0

    def count(self):
        self.align(4)
        i = self.add('I', 4, 1)
        expr = f'{self.blk}[{i}]'
        self.flush()
        return expr

    def struct(self, t):
        spec = self.enter(t)
        if not spec.fields:                       # an empty message is one byte on the wire
            self.add('x', 1, ())
            items = []
        else:
            items = [(f.name, self.value(f)) for f in spec.fields]
        self.stack.pop()
        return '{' + ', '.join(f'{n!r}: {e}' for n, e in items) + '}'

    def sub(self, t):
        fn = self.const(self.types.decoder(t, self.e, top=False), '_d')
        return f'{fn}(b, p)'

    def loop(self, var, count, body, guard):
        self.emit(f'{var} = []')
        if guard:
            self.emit(f'if {count} > len(b) - p: raise ValueError("an array length runs past the end of the data")')
        self.emit(f'for _ in range({count}):')
        self.emit(f'    x, p = {body}')
        self.emit(f'    {var}.append(x)')

    def value(self, f):
        t, kind, n = f.type, f.kind, f.n
        if t == 'wstring':
            raise ValueError('wstring fields are not supported')
        if kind == 0:
            if t in _PRIM:
                code, size = _PRIM[t]
                self.align(size)
                i = self.add(code, size, 1)
                return f'{self.blk}[{i}]'
            if t == 'string':
                cnt = self.count()
                v = self.new('v')
                self.emit(f'{v} = _str(b, p, {cnt}); p += {cnt}')
                self.unknown()
                return v
            return self.struct(t)
        if kind == 1:
            if n == 0:
                return '[]'
            if t in _BYTES:
                i = self.add(f'{n}s', n, 1)
                return f'{self.blk}[{i}]'
            if t in _PRIM:
                code, size = _PRIM[t]
                self.align(size)
                i = self.add(f'{n}{code}', size * n, n)
                return f'list({self.blk}[{i}:{i + n}])'
            self.flush()
            v = self.new('v')
            self.loop(v, n, '_rs(b, p)' if t == 'string' else self.sub(t), False)
            self.unknown()
            return v
        cnt = self.count()
        v = self.new('v')
        if t in _BYTES:
            self.emit(f'{v} = _bytes(b, p, {cnt}); p += {cnt}')
        elif t in _PRIM:
            code, size = _PRIM[t]
            self.emit(f'{v}, p = _seq(b, p, {cnt}, {self.const(np.dtype(self.e + code))}, {size})')
        else:
            self.loop(v, cnt, '_rs(b, p)' if t == 'string' else self.sub(t), True)
        self.unknown()
        return v


class _EncGen(_Gen):
    def dynamic_align(self, a):
        self.emit(f'o += _Z[(4 - len(o)) % {a}]')

    def add(self, fmt, size, vals):
        self.fmt.append(fmt)
        self.advance(size)
        self.vals.extend(vals)

    def flush(self):
        if not self.fmt:
            return
        if self.vals:
            fn = self.const(struct.Struct('<' + ''.join(self.fmt)).pack, '_p')
            self.emit(f'o += {fn}({", ".join(self.vals)})')
        else:
            self.emit(f'o += {self.const(bytes(self.size))}')
        self.fmt, self.size, self.vals = [], 0, []

    def struct(self, t, src):
        spec = self.enter(t)
        m = self.new('m')
        self.emit(f'{m} = {src}')
        keys = self.const(frozenset(f.name for f in spec.fields), '_k')
        self.emit(f'if not {m}.keys() <= {keys}: _badkeys({m}, {keys}, {t!r})')
        if not spec.fields:
            self.add('x', 1, ())
        for f in spec.fields:
            self.put(f, m)
        self.stack.pop()

    def default(self, f):
        if f.kind == 0:
            if f.type in _PRIM:
                d = f.default if f.default is not None else (False if f.type == 'bool' else
                                                             0.0 if f.type.startswith('float') else 0)
                return repr(d)
            if f.type == 'string':
                return repr(f.default if isinstance(f.default, str) else '')
            return 'None'
        if f.kind == 1:
            if isinstance(f.default, list) and len(f.default) == f.n:
                return self.const(tuple(f.default))
            if f.type in _BYTES:
                return self.const(bytes(f.n))
            if f.type in _PRIM:
                return self.const((False if f.type == 'bool' else 0,) * f.n)
            return self.const(('',) * f.n if f.type == 'string' else (None,) * f.n)
        return self.const(tuple(f.default)) if isinstance(f.default, list) else '()'

    def sub(self, t):
        return self.const(self.types.encoder(t, top=False), '_e')

    def put(self, f, m):
        t, kind, n = f.type, f.kind, f.n
        src = f'{m}.get({f.name!r}, {self.default(f)})'
        if t == 'wstring':
            raise ValueError('wstring fields are not supported')
        if kind == 0:
            if t in _PRIM:
                code, size = _PRIM[t]
                self.align(size)
                self.add(code, size, (src,))
            elif t == 'string':
                self.flush()
                self.emit(f'_ws(o, {src})')
                self.unknown()
            else:
                self.struct(t, f'{m}.get({f.name!r}) or _E')
            return
        if kind == 1:
            if n == 0:
                return
            if t in _BYTES:
                self.add(f'{n}s', n, (f'_fb({src}, {n})',))
            elif t in _PRIM:
                code, size = _PRIM[t]
                self.align(size)
                self.add(f'{n}{code}', size * n, (f'*_fl({src}, {n})',))
            else:
                self.flush()
                x = self.new('x')
                self.emit(f'for {x} in _fl({src}, {n}):')
                self.emit(f'    _ws(o, {x})' if t == 'string' else f'    {self.sub(t)}(o, {x} or _E)')
                self.unknown()
            return
        self.flush()
        if t in _BYTES:
            self.emit(f'_wb(o, {src})')
        elif t in _PRIM:
            code, size = _PRIM[t]
            self.emit(f'_wq(o, {src}, {code!r}, {size}, {self.const(np.dtype("<" + code))})')
        else:
            s, x = self.new('s'), self.new('x')
            self.emit(f'{s} = _sq({src})')
            self.emit(f'o += _Z[(4 - len(o)) % 4]; o += _U32(len({s}))')
            self.emit(f'for {x} in {s}:')
            self.emit(f'    _ws(o, {x})' if t == 'string' else f'    {self.sub(t)}(o, {x} or _E)')
        self.unknown()


class _Types:
    """Message definitions by type name, and the codecs compiled from them (cached). A bag's own definitions live
    in a child whose parent is the shared registry."""

    def __init__(self, texts: dict, parent: '_Types | None' = None):
        self.texts, self.parent = texts, parent
        self._specs, self._enc, self._dec = {}, {}, {}

    def has(self, t: str) -> bool:
        return t in self.texts or (self.parent is not None and self.parent.has(t))

    def text(self, t: str) -> str:
        if t in self.texts:
            return self.texts[t]
        if self.parent is not None:
            return self.parent.text(t)
        raise ValueError(f'unknown message type {t!r}: not in MSG_DEFS (register() adds definitions)')

    def spec(self, t: str) -> _Spec:
        s = self._specs.get(t)
        if s is None:
            if t not in self.texts and self.parent is not None:
                return self.parent.spec(t)
            s = self._specs[t] = _parse(self.text(t), t)
        return s

    def deps(self, t: str) -> set:
        return {f.type for f in self.spec(t).fields if f.type not in _PRIM and f.type not in _STRINGS}

    def closure(self, t: str) -> list:
        """t and every type it uses, in the order rosbag2 writes them into a schema."""
        out, seen = [t], {t}

        def rec(x):
            for d in sorted(self.deps(x), key=_short):
                if d not in seen:
                    seen.add(d)
                    out.append(d)
                    rec(d)
        rec(t)
        return out

    def layout(self, t: str) -> tuple:
        return tuple((f.name, f.type, f.kind, f.n) for f in self.spec(t).fields)

    def decoder(self, t, endian='<', top=True):
        key = (t, endian, top)
        fn = self._dec.get(key)
        if fn is None:
            g = _DecGen(self, endian, 8 if top else 1)
            g.stack_root = t
            expr = g.struct(t)
            g.flush()
            fn = self._dec[key] = g.build('decode', 'b, p', f'{expr}, p')
        return fn

    def encoder(self, t, top=True):
        key = (t, top)
        fn = self._enc.get(key)
        if fn is None:
            g = _EncGen(self, '<', 8 if top else 1)
            g.stack_root = t
            g.struct(t, 'm')
            g.flush()
            fn = self._enc[key] = g.build('encode', 'o, m', 'o')
        return fn

    def encode(self, t: str, msg) -> bytearray:
        fn = self.encoder(t)
        o = bytearray(b'\x00\x01\x00\x00')
        try:
            fn(o, msg)
        except (struct.error, TypeError, AttributeError, OverflowError, ValueError) as e:
            raise ValueError(f'cannot encode {t}: {e}') from None
        return o

    def decode(self, t: str, data) -> dict:
        mv = memoryview(data)
        if mv.ndim != 1 or mv.itemsize != 1:
            mv = mv.cast('B')
        if len(mv) < 4 or mv[0] != 0 or mv[1] > 1:
            head = bytes(mv[:4]).hex(' ')
            raise ValueError(f'{t}: not plain CDR data (encapsulation header {head}; only XCDR1 is read)')
        fn = self.decoder(t, '<' if mv[1] == 1 else '>')
        try:
            msg, _ = fn(mv[4:], 0)
        except (struct.error, IndexError, ValueError, TypeError) as e:
            raise ValueError(f'{t}: truncated or corrupt CDR data ({e})') from None
        return msg


_TYPES = _Types(MSG_DEFS)


def register(msg_type: str, definition: str) -> None:
    """Add a message definition for encode/decode/BagWriter: one .msg text, or a full ros2msg schema (the type's text
    followed by its dependencies as schema_text writes them). A type already known must keep its field layout."""
    t = _norm(msg_type)
    secs = _schema_sections(t, definition)
    probe = _Types(secs, parent=_TYPES)
    for name in secs:
        probe.layout(name)                                      # parse errors surface here
        if name in MSG_DEFS and _TYPES.layout(name) != probe.layout(name):
            raise ValueError(f'{name} is already defined with a different layout')
    for name in probe.closure(t):
        probe.layout(name)                                      # every dependency must be known
    for name, text in secs.items():
        MSG_DEFS.setdefault(name, text)


def schema_text(msg_type: str) -> str:
    """The full ros2msg schema of a type, as rosbag2 writes it into an MCAP Schema record: the type's .msg text, then
    for each dependency (depth first, sorted by name at each level, each once) a blank line, 80 '=', 'MSG: pkg/Type'
    and its text."""
    t = _norm(msg_type)
    parts = [_TYPES.text(t)]
    for d in _TYPES.closure(t)[1:]:
        parts.append(f'\n{_SEP}\nMSG: {_short(d)}\n{_TYPES.text(d)}')
    return ''.join(parts)


def constants(msg_type: str) -> dict:
    """The constants a type defines, e.g. constants('visualization_msgs/msg/Marker')['CUBE'] == 1."""
    return dict(_TYPES.spec(_norm(msg_type)).constants)


def encode(msg_type: str, msg: dict) -> bytes:
    """A message dict -> CDR bytes (little-endian, with the 4-byte encapsulation header 00 01 00 00)."""
    return bytes(_TYPES.encode(_norm(msg_type), msg))


def decode(msg_type: str, data) -> dict:
    """CDR bytes (with the encapsulation header) -> a message dict. Raises ValueError on unknown types and on data
    that is truncated or corrupt."""
    return _TYPES.decode(_norm(msg_type), data)


# ---------------------------------------------------------------------------------------------------------------------
# MCAP records

MCAP_MAGIC = b'\x89MCAP0\r\n'
OP_HEADER, OP_FOOTER, OP_SCHEMA, OP_CHANNEL, OP_MESSAGE, OP_CHUNK, OP_MESSAGE_INDEX, OP_CHUNK_INDEX = range(1, 9)
OP_ATTACHMENT, OP_ATTACHMENT_INDEX, OP_STATISTICS, OP_METADATA, OP_METADATA_INDEX = range(9, 14)
OP_SUMMARY_OFFSET, OP_DATA_END = 0x0E, 0x0F
_REC = struct.Struct('<BQ')
_MSG = struct.Struct('<BQHIQQ')         # opcode, length, channel, sequence, log time, publish time
_MSG_FIELDS = struct.Struct('<HIQQ')
_SQLITE_MAGIC = b'SQLite format 3\x00'


def _s(text: str) -> bytes:
    b = text.encode()
    return struct.pack('<I', len(b)) + b


def _map(d: dict) -> bytes:
    body = b''.join(_s(k) + _s(v) for k, v in d.items())
    return struct.pack('<I', len(body)) + body


def _rec(op: int, body: bytes) -> bytes:
    return _REC.pack(op, len(body)) + body


def _get_s(buf, p):
    n, = struct.unpack_from('<I', buf, p)
    return bytes(buf[p + 4:p + 4 + n]).decode('utf-8', 'replace'), p + 4 + n


def _get_map(buf, p):
    n, = struct.unpack_from('<I', buf, p)
    p += 4
    end, out = p + n, {}
    while p < end:
        k, p = _get_s(buf, p)
        v, p = _get_s(buf, p)
        out[k] = v
    return out, end


def _records(buf):
    """(opcode, body start, body length) of the records in a buffer; stops at a record that runs past its end."""
    pos, n = 0, len(buf)
    while pos + 9 <= n:
        op, ln = _REC.unpack_from(buf, pos)
        if pos + 9 + ln > n:
            raise ValueError('a record runs past the end of its chunk')
        yield op, pos + 9, ln
        pos += 9 + ln


def _compressor(name: str):
    """(MCAP compression string, compress function) for 'none' | 'zstd' | 'lz4' | 'auto'."""
    if name == 'auto':
        try:
            import zstandard  # noqa: F401
            name = 'zstd'
        except ImportError:
            name = 'none'
    if name == 'none':
        return '', None
    if name == 'zstd':
        try:
            import zstandard
        except ImportError:
            raise ImportError("zstd compression needs the Python module 'zstandard' (pip install zstandard)") from None
        return 'zstd', zstandard.ZstdCompressor(level=1).compress
    if name == 'lz4':
        try:
            import lz4.frame
        except ImportError:
            raise ImportError("lz4 compression needs the Python module 'lz4' (pip install lz4)") from None
        return 'lz4', lambda data: lz4.frame.compress(bytes(data))
    raise ValueError(f"compression must be 'none', 'zstd', 'lz4' or 'auto', not {name!r}")


def _decompress(comp: str, data, size: int) -> bytes:
    if not comp:
        return bytes(data)
    if comp == 'zstd':
        try:
            import zstandard
        except ImportError:
            raise ImportError("this bag's chunks are zstd-compressed: reading them needs the Python module "
                              "'zstandard' (pip install zstandard)") from None
        fn = lambda: zstandard.ZstdDecompressor().decompress(data, max_output_size=max(size, 1))  # noqa: E731
    elif comp == 'lz4':
        try:
            import lz4.frame
        except ImportError:
            raise ImportError("this bag's chunks are lz4-compressed: reading them needs the Python module 'lz4' "
                              "(pip install lz4)") from None
        fn = lambda: lz4.frame.decompress(bytes(data))  # noqa: E731
    else:
        raise ValueError(f'unknown MCAP chunk compression {comp!r}')
    try:
        return fn()
    except Exception as e:                          # zstandard.ZstdError, lz4's RuntimeError: corrupt data
        raise ValueError(f'a {comp} chunk does not decompress ({e})') from None


# ---------------------------------------------------------------------------------------------------------------------
# writer

class BagWriter:
    """A rosbag2 bag: the directory `path` with <name>_0.mcap (name = the directory's name) and metadata.yaml.

        with BagWriter('out/drive_01') as bag:
            bag.add_topic('/cam/F/image', 'sensor_msgs/msg/Image')
            bag.write('/cam/F/image', t_ns, image_msg(img, t_ns, 'cam_F'))

    compression: 'none' | 'zstd' | 'lz4' | 'auto' (zstd when the zstandard module imports, else none), per chunk;
    chunk_size: uncompressed bytes per chunk (a larger message makes a chunk of its own). Messages may be written in
    any time order; readers return them in time order. An existing bag in `path` is replaced only with
    overwrite=True; other files in the directory are left alone."""

    def __init__(self, path: str, compression: str = 'auto', chunk_size: int = 4 << 20, overwrite: bool = False):
        self.path = os.fspath(path)
        self.compression, self._compress = _compressor(compression)
        self.chunk_size = max(1024, int(chunk_size))
        name = os.path.basename(os.path.normpath(self.path)) or 'bag'
        self.file_name = f'{name}_0.mcap'
        self.mcap_path = os.path.join(self.path, self.file_name)
        meta = os.path.join(self.path, 'metadata.yaml')
        if os.path.isfile(self.path):
            raise FileExistsError(f'{self.path} is a file; a bag is a directory')
        if not overwrite and (os.path.exists(meta) or os.path.exists(self.mcap_path)):
            raise FileExistsError(f'{self.path} already holds a bag (pass overwrite=True to replace it)')
        os.makedirs(self.path, exist_ok=True)
        if overwrite and os.path.exists(meta):
            os.remove(meta)
        self._f = open(self.mcap_path, 'wb')
        self._pos, self._crc = 0, 0
        self._topics = {}        # topic -> [channel id, type, count, encoder, qos]
        self._schemas = {}       # type -> (schema id, record body)
        self._channel_recs = []
        self._chunk_indexes, self._chunks = [], 0
        self._count, self._start, self._end = 0, None, None
        self._new_chunk()
        self.closed = False
        self._out(MCAP_MAGIC)
        self._out(_rec(OP_HEADER, _s('ros2') + _s('vantage')))

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()

    def _out(self, b):
        self._f.write(b)
        self._crc = zlib.crc32(b, self._crc)
        self._pos += len(b)

    def _new_chunk(self):
        self._buf = bytearray()
        self._index = {}        # channel id -> [(log time, offset)]
        self._c_start = self._c_end = None

    def add_topic(self, topic: str, msg_type: str, qos: str | None = None) -> None:
        """Declare a topic and its message type (one of MSG_DEFS, or one register() added). qos: the offered QoS
        profiles text for metadata.yaml and the channel (None: QOS_TRANSIENT_LOCAL for /tf_static, else '')."""
        if self.closed:
            raise ValueError('the bag is closed')
        t = _norm(msg_type)
        if topic in self._topics:
            if self._topics[topic][1] != t:
                raise ValueError(f'{topic} is already a {self._topics[topic][1]} topic, not {t}')
            return
        text = schema_text(t)                         # raises for an unknown type
        enc = _TYPES.encoder(t)
        if t not in self._schemas:
            sid = len(self._schemas) + 1
            data = text.encode()
            body = struct.pack('<H', sid) + _s(t) + _s('ros2msg') + struct.pack('<I', len(data)) + data
            self._schemas[t] = (sid, body)
            self._out(_rec(OP_SCHEMA, body))
        if qos is None:
            qos = QOS_TRANSIENT_LOCAL if topic in LATCHED_TOPICS else ''
        cid = len(self._topics) + 1
        body = (struct.pack('<HH', cid, self._schemas[t][0]) + _s(topic) + _s('cdr')
                + _map({'offered_qos_profiles': qos}))
        self._channel_recs.append(body)
        self._out(_rec(OP_CHANNEL, body))
        self._topics[topic] = [cid, t, 0, enc, qos]

    def write(self, topic: str, t_ns: int, msg) -> None:
        """One message at log time t_ns (integer nanoseconds): a dict, encoded as the topic's type, or bytes that are
        already CDR (with the encapsulation header)."""
        if self.closed:
            raise ValueError('the bag is closed')
        info = self._topics.get(topic)
        if info is None:
            raise ValueError(f'unknown topic {topic!r}: add_topic() it first')
        t = int(t_ns)
        if not 0 <= t < 1 << 64:
            raise ValueError(f'log time {t_ns} is not a uint64 of nanoseconds')
        if isinstance(msg, dict):
            o = bytearray(b'\x00\x01\x00\x00')
            try:
                info[3](o, msg)
            except (struct.error, TypeError, AttributeError, OverflowError, ValueError) as e:
                raise ValueError(f'cannot encode {info[1]} for {topic}: {e}') from None
            data = o
        else:
            data = msg if isinstance(msg, (bytes, bytearray)) else bytes(msg)
            if len(data) < 4 or data[0] != 0 or data[1] > 1:
                raise ValueError(f'{topic}: raw message data must be CDR with its 4-byte encapsulation header')
        buf = self._buf
        self._index.setdefault(info[0], []).append((t, len(buf)))
        buf += _MSG.pack(OP_MESSAGE, 22 + len(data), info[0], info[2] & 0xffffffff, t, t)
        buf += data
        info[2] += 1
        self._count += 1
        if self._c_start is None or t < self._c_start:
            self._c_start = t
        if self._c_end is None or t > self._c_end:
            self._c_end = t
        if len(buf) >= self.chunk_size:
            self._flush_chunk()

    def _flush_chunk(self):
        raw = self._buf
        if not raw:
            return
        ucrc = zlib.crc32(raw)
        data = self._compress(raw) if self._compress else raw
        start = self._pos
        head = struct.pack('<QQQI', self._c_start, self._c_end, len(raw), ucrc) + _s(self.compression)
        head += struct.pack('<Q', len(data))
        self._out(_REC.pack(OP_CHUNK, len(head) + len(data)))
        self._out(head)
        self._out(data)
        length = self._pos - start
        offsets = {}
        for cid in sorted(self._index):
            entries = sorted(self._index[cid], key=lambda e: e[0])
            offsets[cid] = self._pos
            body = struct.pack('<HI', cid, 16 * len(entries)) + np.asarray(entries, '<u8').tobytes()
            self._out(_rec(OP_MESSAGE_INDEX, body))
        mi_len = self._pos - start - length
        mio = b''.join(struct.pack('<HQ', c, off) for c, off in offsets.items())
        self._chunk_indexes.append(
            struct.pack('<QQQQ', self._c_start, self._c_end, start, length) + struct.pack('<I', len(mio)) + mio
            + struct.pack('<Q', mi_len) + _s(self.compression) + struct.pack('<QQ', len(data), len(raw)))
        self._chunks += 1
        self._start = self._c_start if self._start is None else min(self._start, self._c_start)
        self._end = self._c_end if self._end is None else max(self._end, self._c_end)
        self._new_chunk()

    def close(self) -> dict:
        """Finish the MCAP file (last chunk, summary, footer) and write metadata.yaml. Returns
        {'path', 'file', 'messages', 'topics': {topic: count}, 'bytes'}; closing again returns the same."""
        if self.closed:
            return self._result
        self._flush_chunk()
        self._out(_rec(OP_DATA_END, struct.pack('<I', self._crc)))
        summary_start = self._pos
        groups, summ = [], bytearray()

        def group(op, bodies):
            if bodies:
                g0 = summary_start + len(summ)
                for b in bodies:
                    summ.extend(_rec(op, b))
                groups.append((op, g0, summary_start + len(summ) - g0))
        group(OP_SCHEMA, [b for _, b in self._schemas.values()])
        group(OP_CHANNEL, self._channel_recs)
        counts = b''.join(struct.pack('<HQ', v[0], v[2]) for v in self._topics.values())
        group(OP_STATISTICS, [struct.pack('<QHIIIIQQ', self._count, len(self._schemas), len(self._topics), 0, 0,
                                          self._chunks, self._start or 0, self._end or 0)
                              + struct.pack('<I', len(counts)) + counts])
        group(OP_CHUNK_INDEX, self._chunk_indexes)
        offsets_start = summary_start + len(summ)
        for op, g0, gl in groups:
            summ.extend(_rec(OP_SUMMARY_OFFSET, struct.pack('<BQQ', op, g0, gl)))
        footer = _REC.pack(OP_FOOTER, 20) + struct.pack('<QQ', summary_start, offsets_start)
        crc = zlib.crc32(footer, zlib.crc32(summ))
        self._out(bytes(summ))
        self._out(footer + struct.pack('<I', crc))
        self._out(MCAP_MAGIC)
        self._f.close()
        self.closed = True
        self._write_metadata()
        self._result = {'path': self.path, 'file': self.mcap_path, 'messages': self._count,
                        'topics': {tp: v[2] for tp, v in self._topics.items()}, 'bytes': self._pos}
        return self._result

    def _write_metadata(self):
        import yaml
        start = self._start if self._count else 0
        dur = (self._end - self._start) if self._count else 0
        info = {
            'version': 5,
            'storage_identifier': 'mcap',
            'duration': {'nanoseconds': dur},
            'starting_time': {'nanoseconds_since_epoch': start},
            'message_count': self._count,
            'topics_with_message_count': [
                {'topic_metadata': {'name': tp, 'type': v[1], 'serialization_format': 'cdr',
                                    'offered_qos_profiles': v[4]},
                 'message_count': v[2]} for tp, v in self._topics.items()],
            'compression_format': '',
            'compression_mode': '',
            'relative_file_paths': [self.file_name],
            'files': [{'path': self.file_name, 'starting_time': {'nanoseconds_since_epoch': start},
                       'duration': {'nanoseconds': dur}, 'message_count': self._count}],
        }
        with open(os.path.join(self.path, 'metadata.yaml'), 'w', encoding='utf-8') as f:
            yaml.safe_dump({'rosbag2_bagfile_information': info}, f, sort_keys=False, default_flow_style=False,
                           width=1 << 16)


# ---------------------------------------------------------------------------------------------------------------------
# readers

class _Block(NamedTuple):
    start: int
    end: int
    offset: int
    length: int
    chunk: bool             # a Chunk record, or a run of plain Message records
    chans: frozenset | None  # channel ids in it (None: unknown)


_RUN_MAX = 8 << 20          # plain messages are read in runs of up to this many bytes


class _McapFile:
    """One .mcap file: its schemas, channels, per-channel counts and time range, and the blocks (chunks, or runs of
    messages outside chunks) to stream from. Uses the summary when the file has a valid one, a linear scan else."""

    def __init__(self, path):
        self.path = path
        self.f = open(path, 'rb')
        try:
            self._open()
        except BaseException:
            self.f.close()
            raise

    def close(self):
        self.f.close()

    def _open(self):
        f = self.f
        self.size = os.fstat(f.fileno()).st_size
        if f.read(8) != MCAP_MAGIC:
            raise ValueError(f'{self.path}: not an MCAP file')
        self.schemas, self.channels, self.counts = {}, {}, {}
        self.blocks, self.warnings = [], []
        self.start = self.end = None
        self.indexed = self._read_summary()
        if not self.indexed:
            self.schemas, self.channels, self.counts, self.blocks = {}, {}, {}, []
            self._scan()

    def _schema(self, buf, p=0):
        sid, = struct.unpack_from('<H', buf, p)
        name, q = _get_s(buf, p + 2)
        enc, q = _get_s(buf, q)
        n, = struct.unpack_from('<I', buf, q)
        self.schemas.setdefault(sid, (name, enc, bytes(buf[q + 4:q + 4 + n])))

    def _channel(self, buf, p=0):
        cid, sid = struct.unpack_from('<HH', buf, p)
        topic, q = _get_s(buf, p + 4)
        enc, q = _get_s(buf, q)
        meta, q = _get_map(buf, q)
        self.channels.setdefault(cid, (topic, sid, enc, meta))

    def _read_summary(self) -> bool:
        f = self.f
        if self.size < 8 + 37:
            self.warnings.append('the file is truncated: read by a linear scan')
            return False
        f.seek(self.size - 37)
        tail = f.read(37)
        if tail[29:] != MCAP_MAGIC or tail[0] != OP_FOOTER:
            self.warnings.append('no footer (the recording was not closed, or the file is truncated): '
                                 'read by a linear scan')
            return False
        _, ln, s_start, _, s_crc = struct.unpack('<BQQQI', tail[:29])
        footer_pos = self.size - 37
        if ln != 20 or not s_start or not 8 <= s_start <= footer_pos:
            if s_start:
                self.warnings.append('the summary offset is invalid: read by a linear scan')
            return False
        f.seek(s_start)
        summ = f.read(footer_pos - s_start)
        if s_crc and zlib.crc32(tail[:25], zlib.crc32(summ)) != s_crc:
            self.warnings.append('the summary CRC does not match: read by a linear scan')
            return False
        chunks, stats = [], None
        try:
            for op, p, n in _records(summ):
                if op == OP_SCHEMA:
                    self._schema(summ, p)
                elif op == OP_CHANNEL:
                    self._channel(summ, p)
                elif op == OP_STATISTICS:
                    stats = summ[p:p + n]
                elif op == OP_CHUNK_INDEX:
                    s, e, off, length, mlen = struct.unpack_from('<QQQQI', summ, p)
                    q = p + 36
                    mio = {struct.unpack_from('<H', summ, q + i)[0]: struct.unpack_from('<Q', summ, q + i + 2)[0]
                           for i in range(0, mlen, 10)}
                    chunks.append((s, e, off, length, mio))
        except (ValueError, struct.error):
            self.warnings.append('the summary is corrupt: read by a linear scan')
            return False
        total, per_channel = None, {}
        if stats is not None and len(stats) >= 46:
            total = struct.unpack_from('<Q', stats, 0)[0]
            s0, e0 = struct.unpack_from('<QQ', stats, 26)
            mlen, = struct.unpack_from('<I', stats, 42)
            for i in range(46, 46 + mlen, 10):
                cid, c = struct.unpack_from('<HQ', stats, i)
                per_channel[cid] = c
        if not chunks:
            if total == 0:
                return True                         # an empty, indexed file
            self.warnings.append('the summary has no chunk index: read by a linear scan')
            return False
        if total is not None and sum(per_channel.values()) == total:
            self.counts = {c: n for c, n in per_channel.items() if n}
            for s, e, off, length, mio in chunks:
                self.blocks.append(_Block(s, e, off, length, True, frozenset(mio) or None))
            if total:
                self.start, self.end = s0, e0
            return True
        # no usable statistics: count from the message indexes
        try:
            for s, e, off, length, mio in chunks:
                if not mio:
                    self._count_block(_Block(s, e, off, length, True, None))
                    continue
                seen = False
                for cid, moff in mio.items():
                    f.seek(moff)
                    op, _, c, rl = struct.unpack('<BQHI', f.read(15))
                    if op == OP_MESSAGE_INDEX and rl:
                        self.counts[c] = self.counts.get(c, 0) + rl // 16
                        seen = True
                if seen:
                    self.blocks.append(_Block(s, e, off, length, True, frozenset(mio)))
                    self._extend(s, e)
        except (ValueError, struct.error):
            self.warnings.append('the chunk index is inconsistent: read by a linear scan')
            return False
        return True

    def _extend(self, s, e):
        self.start = s if self.start is None else min(self.start, s)
        self.end = e if self.end is None else max(self.end, e)

    def _count_block(self, blk):
        """Read a block to learn its channels, counts and exact time range (a chunk without message indexes). A
        corrupt chunk is left out, with a warning."""
        try:
            msgs = self._read_block(blk)
        except (ValueError, struct.error) as e:
            self.warnings.append(f'{e}: left out')
            return
        if not msgs:
            return
        ts = [m[0] for m in msgs]
        chans = frozenset(m[1] for m in msgs)
        for m in msgs:
            self.counts[m[1]] = self.counts.get(m[1], 0) + 1
        self.blocks.append(blk._replace(start=min(ts), end=max(ts), chans=chans))
        self._extend(min(ts), max(ts))

    def _scan(self):
        f, size = self.f, self.size
        pos, run = 8, None
        chunks = []             # [block, {channel: count} from its message indexes]
        owner = None            # the chunk whose message indexes follow it

        def close_run():
            if run is not None:
                self.blocks.append(_Block(run[0], run[1], run[2], run[3] - run[2], False, frozenset(run[4])))
                self._extend(run[0], run[1])

        while True:
            if pos + 9 > size:
                if pos < size:
                    self.warnings.append(f'the file ends inside a record at byte {pos}: the rest was skipped')
                break
            f.seek(pos)
            op, ln = _REC.unpack(f.read(9))
            end = pos + 9 + ln
            if end > size:
                self.warnings.append(f'the file is truncated at byte {size}: the record at byte {pos} is incomplete '
                                     f'and was skipped')
                break
            if op != OP_MESSAGE or (run is not None and end - run[2] > _RUN_MAX):
                close_run()
                run = None
            if ln < {OP_MESSAGE: 22, OP_CHUNK: 40, OP_MESSAGE_INDEX: 6, OP_SCHEMA: 14, OP_CHANNEL: 16}.get(op, 0):
                self.warnings.append(f'a corrupt record at byte {pos}: the rest was skipped')
                break
            if op in (OP_SCHEMA, OP_CHANNEL):
                try:
                    (self._schema if op == OP_SCHEMA else self._channel)(f.read(ln))
                except struct.error:
                    self.warnings.append(f'a corrupt record at byte {pos}: the rest was skipped')
                    break
            elif op == OP_MESSAGE:
                cid, _, t = struct.unpack('<HIQ', f.read(14))
                self.counts[cid] = self.counts.get(cid, 0) + 1
                if run is None:
                    run = [t, t, pos, end, {cid}]
                else:
                    run[0], run[1], run[3] = min(run[0], t), max(run[1], t), end
                    run[4].add(cid)
            elif op == OP_CHUNK:
                s, e = struct.unpack('<QQ', f.read(16))
                chunks.append([_Block(s, e, pos, 9 + ln, True, None), {}])
            elif op == OP_MESSAGE_INDEX:
                if owner is not None:
                    cid, rl = struct.unpack('<HI', f.read(6))
                    if rl:
                        owner[1][cid] = owner[1].get(cid, 0) + rl // 16
            elif op in (OP_DATA_END, OP_FOOTER):
                break
            owner = chunks[-1] if op == OP_CHUNK else owner if op == OP_MESSAGE_INDEX else None
            pos = end
        close_run()
        for blk, idx in chunks:                       # in file order, so channels defined in chunks come first
            if not idx or not set(idx) <= set(self.channels):
                self._count_block(blk)
            else:
                for cid, c in idx.items():
                    self.counts[cid] = self.counts.get(cid, 0) + c
                self.blocks.append(blk._replace(chans=frozenset(idx)))
                self._extend(blk.start, blk.end)

    def _read_block(self, blk) -> list:
        """[(log time, channel id, data)] of a block in file order; registers the schemas and channels in it."""
        self.f.seek(blk.offset)
        buf = self.f.read(blk.length)
        if len(buf) < blk.length:
            raise ValueError(f'{self.path}: the file ends inside the record at byte {blk.offset}')
        if blk.chunk:
            size, crc, clen = struct.unpack_from('<QII', buf, 25)    # after the start and end times
            comp = buf[41:41 + clen].decode()
            q = 41 + clen
            rlen, = struct.unpack_from('<Q', buf, q)
            recs = _decompress(comp, memoryview(buf)[q + 8:q + 8 + rlen], size)
            if len(recs) != size or (crc and zlib.crc32(recs) != crc):
                raise ValueError(f'{self.path}: the chunk at byte {blk.offset} is corrupt (size or CRC mismatch)')
        else:
            recs = buf
        out = []
        for op, p, n in _records(recs):
            if op == OP_MESSAGE:
                cid, _, t, _ = _MSG_FIELDS.unpack_from(recs, p)
                out.append((t, cid, recs[p + 22:p + n]))
            elif op == OP_SCHEMA:
                self._schema(recs, p)
            elif op == OP_CHANNEL:
                self._channel(recs, p)
        return out

    def iter(self, chans, start, end, corrupt=None):
        """(log time, channel id, data) in log-time order: blocks by start time, a heap across the ones that overlap.
        A corrupt chunk raises ValueError, or is passed to corrupt(message) and skipped when corrupt is given."""
        blocks = sorted((b for b in self.blocks if b.end >= start and b.start < end
                         and (chans is None or b.chans is None or not b.chans.isdisjoint(chans))),
                        key=lambda b: (b.start, b.offset))
        heap, order = [], 0
        for i, blk in enumerate(blocks):
            nxt = blocks[i + 1].start if i + 1 < len(blocks) else None
            try:
                msgs = self._read_block(blk)
            except (ValueError, struct.error) as e:
                if corrupt is None:
                    raise ValueError(str(e)) from None
                corrupt(f'{os.path.basename(self.path)}: {e}: skipped')
                msgs = ()
            for t, cid, data in msgs:
                if start <= t < end and (chans is None or cid in chans):
                    heapq.heappush(heap, (t, order, cid, data))
                    order += 1
            while heap and (nxt is None or heap[0][0] <= nxt):
                t, _, cid, data = heapq.heappop(heap)
                yield t, cid, data


class _Db3File:
    """One rosbag2 SQLite file: tables topics(id, name, type, serialization_format, ...) and
    messages(id, topic_id, timestamp, data); message_definitions (Iron and later) when present."""

    def __init__(self, path):
        self.path = path
        self.warnings = []
        uri = 'file:' + urllib.parse.quote(os.path.abspath(path)) + '?mode=ro'
        try:
            self.con = sqlite3.connect(uri, uri=True, check_same_thread=False)
        except sqlite3.Error as e:
            raise ValueError(f'{path}: cannot open the SQLite bag ({e})') from None
        try:
            tables = {r[0] for r in self.con.execute("SELECT name FROM sqlite_master WHERE type='table'")}
            if not {'topics', 'messages'} <= tables:
                raise ValueError(f'{path}: not a rosbag2 SQLite bag (no topics and messages tables)')
            self.channels = {tid: (name, typ, fmt) for tid, name, typ, fmt in
                             self.con.execute('SELECT id, name, type, serialization_format FROM topics')}
            self.counts, self.start, self.end = {}, None, None
            for tid, c, s, e in self.con.execute('SELECT topic_id, COUNT(*), MIN(timestamp), MAX(timestamp) '
                                                 'FROM messages GROUP BY topic_id'):
                self.counts[tid] = c
                self.start = s if self.start is None else min(self.start, s)
                self.end = e if self.end is None else max(self.end, e)
            self.definitions = {}
            if 'message_definitions' in tables:
                for typ, enc, text in self.con.execute('SELECT topic_type, encoding, encoded_message_definition '
                                                       'FROM message_definitions'):
                    self.definitions[typ] = (enc, text.encode() if isinstance(text, str) else bytes(text or b''))
        except sqlite3.Error as e:
            self.con.close()
            raise ValueError(f'{path}: not a readable SQLite bag ({e})') from None
        except BaseException:
            self.con.close()
            raise

    def close(self):
        self.con.close()

    def iter(self, chans, start, end, corrupt=None):
        conds, args = [], []
        if start > _I64[0]:
            conds.append('timestamp >= ?')
            args.append(min(start, _I64[1]))
        if end <= _I64[1]:
            conds.append('timestamp < ?')
            args.append(max(end, _I64[0]))
        if chans is not None:
            if not chans:
                return
            conds.append(f'topic_id IN ({",".join("?" * len(chans))})')
            args += sorted(chans)
        q = 'SELECT timestamp, topic_id, data FROM messages' + (' WHERE ' + ' AND '.join(conds) if conds else '')
        for t, tid, data in self.con.execute(q + ' ORDER BY timestamp', args):
            yield t, tid, data


_T_MIN, _T_END = -(1 << 63), 1 << 64
_I64 = (-(1 << 63), (1 << 63) - 1)


def _channel_types(name: str, encoding: str, data: bytes, cache: dict):
    """The type registry that decodes a channel whose schema is (name, encoding, data), None when there is none: the
    shared one when the bag's definitions match MSG_DEFS (or it has none), one built from the bag's own otherwise."""
    t = _norm(name) if name else ''
    key = (t, encoding, data)
    if key in cache:
        return cache[key]
    types = _TYPES if t and _TYPES.has(t) else None
    if encoding == 'ros2msg' and data and t:
        try:
            local = _Types(_schema_sections(t, data.decode('utf-8', 'replace')), parent=_TYPES)
            same = all(s in MSG_DEFS and _TYPES.layout(s) == local.layout(s) for s in local.closure(t))
            if not same:
                local.decoder(t)
            types = _TYPES if same else local
        except ValueError:
            pass
    cache[key] = types
    return types


def _natural(path):
    return [int(s) if s.isdigit() else s for s in re.split(r'(\d+)', os.path.basename(path))]


class BagReader:
    """A ROS 2 bag for reading: a rosbag2 directory (metadata.yaml with .mcap or .db3 files; without metadata.yaml
    every .mcap / .db3 in it), a single .mcap file or a single .db3 file.

        with BagReader('drive_01') as bag:
            for topic, t_ns, msg in bag.messages(['/imu']):
                ...

    topics: {topic: type}; counts: {topic: messages}; start_ns / end_ns: the first and last log times (0 when
    empty); message_count; files; warnings (a scanned or truncated file); decode_errors: {topic: messages that could
    not be decoded and were yielded as raw bytes}."""

    def __init__(self, path: str):
        self.path = os.fspath(path)
        self.metadata, self.warnings = None, []
        self.decode_errors = {}
        self._msg_zstd = False
        files = self._files()
        self.sources = []
        try:
            for fp in files:
                with open(fp, 'rb') as fh:
                    magic = fh.read(16)
                if magic.startswith(MCAP_MAGIC):
                    src = _McapFile(fp)
                elif magic.startswith(_SQLITE_MAGIC):
                    src = _Db3File(fp)
                else:
                    raise ValueError(f'{fp}: neither an MCAP nor a SQLite bag file')
                self.sources.append(src)
                self.warnings += [f'{os.path.basename(fp)}: {w}' for w in src.warnings]
        except BaseException:
            self.close()
            raise
        self.files = files
        self.storage = ('mcap' if all(isinstance(s, _McapFile) for s in self.sources) else
                        'sqlite3' if all(isinstance(s, _Db3File) for s in self.sources) else 'mixed')
        self._chan = []          # per source: {channel id: (topic, type, encoding, schema)}
        self.topics, self.counts = {}, {}
        for src in self.sources:
            chans = {}
            if isinstance(src, _McapFile):
                for cid, (topic, sid, enc, _) in src.channels.items():
                    sname, senc, sdata = src.schemas.get(sid, ('', '', b''))
                    chans[cid] = (topic, _norm(sname) if sname else '', enc, (senc, sdata))
            else:
                for tid, (topic, typ, fmt) in src.channels.items():
                    chans[tid] = (topic, _norm(typ) if typ else '', fmt, src.definitions.get(typ, ('', b'')))
            self._chan.append(chans)
            for cid, info in chans.items():
                self.topics.setdefault(info[0], info[1])
                self.counts[info[0]] = self.counts.get(info[0], 0) + src.counts.get(cid, 0)
        for cid_counts, src in zip(self._chan, self.sources):
            for cid, c in src.counts.items():           # messages on channels the file never defined
                if cid not in cid_counts and c:
                    self.warnings.append(f'{os.path.basename(src.path)}: {c} messages on undefined channel {cid}')
        live = [s for s in self.sources if s.start is not None and any(s.counts.values())]
        self.start_ns = min((s.start for s in live), default=0)
        self.end_ns = max((s.end for s in live), default=0)
        self.message_count = sum(self.counts.values())
        self._decoders, self._cache, self._by_topic = {}, {}, {}

    def _files(self) -> list:
        p = self.path
        if os.path.isfile(p):
            return [p]
        if not os.path.isdir(p):
            raise FileNotFoundError(f'{p}: no such bag')
        meta = os.path.join(p, 'metadata.yaml')
        if os.path.isfile(meta):
            import yaml
            try:
                with open(meta, encoding='utf-8') as f:
                    info = (yaml.safe_load(f) or {}).get('rosbag2_bagfile_information') or {}
            except (yaml.YAMLError, AttributeError) as e:
                raise ValueError(f'{meta}: not a rosbag2 metadata file ({e})') from None
            self.metadata = info
            mode = str(info.get('compression_mode') or '').lower()
            fmt = str(info.get('compression_format') or '').lower()
            if mode == 'file':
                raise ValueError(f'{p}: the bag is compressed as whole files ({fmt}); decompress it first '
                                 f'(ros2 bag decompress)')
            if mode == 'message':
                if fmt != 'zstd':
                    raise ValueError(f'{p}: message compression {fmt!r} is not supported (zstd is)')
                self._msg_zstd = True
            rel = info.get('relative_file_paths') or []
            files = [os.path.join(p, os.path.basename(str(r))) for r in rel]
            missing = [f for f in files if not os.path.isfile(f)]
            if missing:
                raise ValueError(f'{p}: files listed in metadata.yaml are missing: '
                                 f'{", ".join(os.path.basename(m) for m in missing)}')
            if files:
                return files
        files = sorted(glob.glob(os.path.join(glob.escape(p), '*.mcap')) +
                       glob.glob(os.path.join(glob.escape(p), '*.db3')), key=_natural)
        if not files:
            raise ValueError(f'{p}: no .mcap or .db3 files (not a bag)')
        return files

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()

    def close(self) -> None:
        for s in getattr(self, 'sources', []):
            s.close()

    def _decoder(self, i, cid):
        key = (i, cid)
        if key in self._decoders:
            return self._decoders[key]
        info = self._chan[i].get(cid)
        fn = None
        if info is not None and info[2] == 'cdr' and info[1]:
            types = _channel_types(info[1], info[3][0], info[3][1], self._cache)
            if types is not None:
                try:
                    types.decoder(info[1])
                    t = info[1]
                    fn = lambda data, types=types, t=t: types.decode(t, data)    # noqa: E731
                except ValueError:
                    fn = None
        self._decoders[key] = (info[0] if info else f'#{cid}', fn)
        return self._decoders[key]

    def decode(self, topic: str, data) -> dict:
        """Decode a raw message of `topic` (from messages(decode=False)) with the bag's own definition of its type,
        so vendor types and other distributions' layouts decode as the recorder wrote them. ValueError when the
        type cannot be decoded or the data is corrupt."""
        hit = self._by_topic.get(topic)
        if hit is None:
            hit = next(((i, cid) for i, chans in enumerate(self._chan) for cid, info in chans.items()
                        if info[0] == topic), None)
            if hit is None:
                raise ValueError(f'the bag has no topic {topic!r}')
            self._by_topic[topic] = hit
        _, fn = self._decoder(*hit)
        if fn is None:
            raise ValueError(f'{topic}: messages of type {self.topics.get(topic) or "?"} cannot be decoded '
                             f'(no ros2msg definition, or not CDR)')
        return fn(data)

    def messages(self, topics=None, start_ns=None, end_ns=None, decode=True, skip_corrupt=False):
        """Yield (topic, t_ns, msg) in log-time order for the topics given (a name or a list; None: all) with
        start_ns <= t_ns < end_ns. msg is a dict, or the raw CDR bytes with decode=False and for a type that cannot
        be decoded (counted in decode_errors when decoding failed). A chunk whose CRC or size is wrong raises
        ValueError, or with skip_corrupt=True is skipped and noted in warnings."""
        if isinstance(topics, str):
            topics = [topics]
        want = None if topics is None else set(topics)
        start = _T_MIN if start_ns is None else int(start_ns)
        end = _T_END if end_ns is None else int(end_ns)
        its = []
        for i, src in enumerate(self.sources):
            chans = None if want is None else {c for c, info in self._chan[i].items() if info[0] in want}
            its.append(((t, i, cid, data) for t, cid, data in
                        src.iter(chans, start, end, self.warnings.append if skip_corrupt else None)))
        stream = its[0] if len(its) == 1 else heapq.merge(*its, key=lambda m: m[0])
        unz = None
        if self._msg_zstd:
            try:
                import zstandard
            except ImportError:
                raise ImportError("this bag's messages are zstd-compressed: reading them needs the Python module "
                                  "'zstandard' (pip install zstandard)") from None
            unz = zstandard.ZstdDecompressor()
        for t, i, cid, data in stream:
            topic, fn = self._decoder(i, cid)
            if unz is not None:
                data = unz.decompressobj().decompress(data)
            msg = None
            if decode and fn is not None:
                try:
                    msg = fn(data)
                except ValueError:
                    self.decode_errors[topic] = self.decode_errors.get(topic, 0) + 1
            yield topic, t, bytes(data) if msg is None else msg


# ---------------------------------------------------------------------------------------------------------------------
# message builders and readers

def stamp(t_ns: int) -> dict:
    """builtin_interfaces/Time of integer nanoseconds (a negative time gives a negative sec, nanosec >= 0)."""
    sec, ns = divmod(int(t_ns), 1_000_000_000)
    return {'sec': sec, 'nanosec': ns}


def stamp_ns(st: dict) -> int:
    return int(st.get('sec', 0)) * 1_000_000_000 + int(st.get('nanosec', 0))


def header(t_ns: int, frame_id: str) -> dict:
    return {'stamp': stamp(t_ns), 'frame_id': str(frame_id)}


# PointField datatypes <-> numpy (kind, size)
_PF_CODE = {1: 'i1', 2: 'u1', 3: 'i2', 4: 'u2', 5: 'i4', 6: 'u4', 7: 'f4', 8: 'f8'}
_PF_TYPE = {('i', 1): 1, ('u', 1): 2, ('i', 2): 3, ('u', 2): 4, ('i', 4): 5, ('u', 4): 6, ('f', 4): 7, ('f', 8): 8}
_BIG = not np.little_endian


def pointcloud2(points: np.ndarray, t_ns: int, frame_id: str) -> dict:
    """sensor_msgs/PointCloud2 of a numpy structured array (fields int8/uint8/int16/uint16/int32/uint32/float32/
    float64, a field may be a fixed sub-array: count > 1). A 1-D array is one row (height 1); a 2-D array an organised
    cloud. The data is the array's own memory layout (its offsets and itemsize: padding kept) in little-endian;
    is_dense is False when a float x, y or z is not finite."""
    a = np.asarray(points)
    if a.dtype.names is None:
        raise ValueError('pointcloud2 needs a structured array (fields x, y, z, ...)')
    if a.ndim not in (1, 2):
        raise ValueError('a point cloud is a 1-D (unorganised) or 2-D (organised) array')
    fields = []
    for name in a.dtype.names:
        fd, off = a.dtype.fields[name][:2]
        base, shape = fd.base, fd.shape
        code = _PF_TYPE.get((base.kind, base.itemsize))
        if code is None or base.names is not None:
            raise ValueError(f'point field {name!r} has dtype {fd}: PointField takes int8/16/32, uint8/16/32, '
                             f'float32 and float64 only')
        fields.append({'name': name, 'offset': int(off), 'datatype': code,
                       'count': int(np.prod(shape)) if shape else 1})
    fields.sort(key=lambda f: f['offset'])
    data = np.ascontiguousarray(a.astype(a.dtype.newbyteorder('<'), copy=False))
    h, w = (1, a.shape[0]) if a.ndim == 1 else a.shape
    dense = all(np.isfinite(a[n]).all() for n in ('x', 'y', 'z')
                if n in a.dtype.names and a.dtype.fields[n][0].base.kind == 'f')
    return {'header': header(t_ns, frame_id), 'height': int(h), 'width': int(w), 'fields': fields,
            'is_bigendian': False, 'point_step': int(a.dtype.itemsize), 'row_step': int(w * a.dtype.itemsize),
            'data': data.tobytes(), 'is_dense': bool(dense)}



def pointcloud2_array(msg: dict) -> np.ndarray:
    """A PointCloud2 dict -> a structured array (native byte order) with the message's fields at their offsets and
    itemsize point_step; row_step padding and is_bigendian honoured, fields of unknown datatypes skipped. Shape (width,)
    for one row, (height, width) for an organised cloud."""
    order = '>' if msg.get('is_bigendian') else '<'
    ps, w, h = int(msg['point_step']), int(msg['width']), int(msg['height'])
    rs = int(msg.get('row_step') or w * ps)
    names, formats, offsets = [], [], []
    for f in msg.get('fields', ()):
        code = _PF_CODE.get(int(f.get('datatype', 0)))
        name, off, cnt = f.get('name', ''), int(f.get('offset', 0)), max(1, int(f.get('count', 1)))
        if code is None or not name or name in names:
            continue
        dt = np.dtype(order + code) if cnt == 1 else np.dtype((np.dtype(order + code), (cnt,)))
        if off + dt.itemsize > ps:
            continue
        names.append(name)
        formats.append(dt)
        offsets.append(off)
    dtype = np.dtype({'names': names, 'formats': formats, 'offsets': offsets, 'itemsize': max(ps, 1)})
    raw = np.frombuffer(msg['data'], np.uint8)
    if w * h == 0:
        a = np.zeros(0, dtype)
    elif h == 1 or rs == w * ps:
        if raw.size < w * h * ps:
            raise ValueError('the point cloud data is shorter than width × height × point_step')
        a = raw[:w * h * ps].view(dtype)
    else:
        if rs < w * ps or raw.size < (h - 1) * rs + w * ps:
            raise ValueError('the point cloud data does not match row_step × height')
        rows = np.zeros((h, rs), np.uint8)
        rows.reshape(-1)[:min(raw.size, h * rs)] = raw[:h * rs]
        a = np.ascontiguousarray(rows[:, :w * ps]).reshape(-1).view(dtype)
    if order == '>' or _BIG:
        a = a.astype(dtype.newbyteorder('='))
    a = a.copy()
    return a.reshape(h, w) if h > 1 else a.reshape(w * h)


# sensor_msgs/image_encodings: encoding -> (channels, numpy dtype)
_IMG = {'rgb8': (3, 'u1'), 'rgba8': (4, 'u1'), 'rgb16': (3, 'u2'), 'rgba16': (4, 'u2'),
        'bgr8': (3, 'u1'), 'bgra8': (4, 'u1'), 'bgr16': (3, 'u2'), 'bgra16': (4, 'u2'),
        'mono8': (1, 'u1'), 'mono16': (1, 'u2'),
        'yuv422': (2, 'u1'), 'uyvy': (2, 'u1'), 'yuyv': (2, 'u1'), 'yuv422_yuy2': (2, 'u1')}
for _p in ('rggb', 'bggr', 'gbrg', 'grbg'):
    _IMG[f'bayer_{_p}8'], _IMG[f'bayer_{_p}16'] = (1, 'u1'), (1, 'u2')
_CV_DEPTH = {'8U': 'u1', '8S': 'i1', '16U': 'u2', '16S': 'i2', '32S': 'i4', '32F': 'f4', '64F': 'f8'}
_CV_RE = re.compile(r'(8U|8S|16U|16S|32S|32F|64F)C(\d+)$')


def _image_layout(encoding: str):
    if encoding in _IMG:
        c, d = _IMG[encoding]
        return c, np.dtype(d)
    m = _CV_RE.match(encoding.upper())
    if m and int(m.group(2)) >= 1:
        return int(m.group(2)), np.dtype(_CV_DEPTH[m.group(1)])
    raise ValueError(f'image encoding {encoding!r} is not supported (rgb8, bgr8, mono8, mono16, bayer_*, '
                     f'yuv422, 8UC3, 32FC1, ...)')


def image_msg(img: np.ndarray, t_ns: int, frame_id: str, encoding: str | None = None) -> dict:
    """sensor_msgs/Image of an H×W or H×W×C array. Default encoding by dtype and channels: uint8 mono8 / rgb8 /
    rgba8, uint16 mono16 / rgb16 / rgba16, else the OpenCV form (32FC1, 16SC1 ...). An explicit encoding must match
    the array's channels and dtype (bgr8, bayer_rggb8, yuv422 as H×W×2 ...)."""
    a = np.asarray(img)
    if a.ndim == 3 and a.shape[2] == 1:
        a = a[:, :, 0]
    if a.ndim not in (2, 3):
        raise ValueError('an image is an H×W or H×W×C array')
    c = 1 if a.ndim == 2 else a.shape[2]
    if encoding is None:
        named = {('u', 1): {1: 'mono8', 3: 'rgb8', 4: 'rgba8'}, ('u', 2): {1: 'mono16', 3: 'rgb16', 4: 'rgba16'}}
        encoding = named.get((a.dtype.kind, a.dtype.itemsize), {}).get(c)
        if encoding is None:
            depth = {v: k for k, v in _CV_DEPTH.items()}.get(a.dtype.kind + str(a.dtype.itemsize))
            if depth is None:
                raise ValueError(f'no ROS image encoding for dtype {a.dtype}')
            encoding = f'{depth}C{c}'
    ch, dt = _image_layout(encoding)
    if ch != c or dt.kind != a.dtype.kind or dt.itemsize != a.dtype.itemsize:
        raise ValueError(f'encoding {encoding} needs {ch} channel(s) of {dt}, the image has {c} of {a.dtype}')
    a = np.ascontiguousarray(a, dtype=dt.newbyteorder('<'))
    h, w = a.shape[:2]
    return {'header': header(t_ns, frame_id), 'height': int(h), 'width': int(w), 'encoding': encoding,
            'is_bigendian': 0, 'step': int(w * c * dt.itemsize), 'data': a.tobytes()}


def image_array(msg: dict, to_rgb: bool = False) -> np.ndarray:
    """An Image dict -> H×W or H×W×C array (a copy, native byte order) in the encoding's own channel order (to_rgb:
    bgr8/bgra8/bgr16/bgra16 reordered to RGB/RGBA); step padding and is_bigendian honoured; bayer_* and yuv422 come
    back raw (H×W, H×W×2). A CompressedImage dict (it has 'format') -> decoded with OpenCV, colour in RGB / RGBA
    order, 16-bit PNG kept; compressedDepth (image_transport) decoded to uint16 or float32 metres."""
    if 'format' in msg and 'encoding' not in msg:
        return _decode_compressed(msg)
    enc = msg['encoding']
    ch, dt = _image_layout(enc)
    dt = dt.newbyteorder('>' if msg.get('is_bigendian') else '<')
    h, w = int(msg['height']), int(msg['width'])
    row = w * ch * dt.itemsize
    step = int(msg.get('step') or row)
    raw = np.frombuffer(msg['data'], np.uint8)
    if step < row or (h and raw.size < (h - 1) * step + row):
        raise ValueError(f'the image data does not match {h} rows of step {step}')
    if h == 0 or w == 0:
        rows = np.zeros((h, row), np.uint8)
    elif raw.size >= h * step:
        rows = raw[:h * step].reshape(h, step)[:, :row]
    else:
        rows = np.zeros((h, step), np.uint8)
        rows.reshape(-1)[:raw.size] = raw
        rows = rows[:, :row]
    a = np.ascontiguousarray(rows).view(dt).astype(dt.newbyteorder('='))
    a = a.reshape(h, w) if ch == 1 else a.reshape(h, w, ch)
    if to_rgb and enc.startswith('bgr'):
        a = a[..., [2, 1, 0, 3][:ch]]
    return np.ascontiguousarray(a)


def _decode_compressed(msg: dict) -> np.ndarray:
    import cv2
    fmt = str(msg.get('format', '')).lower()
    data = bytes(msg['data'])
    depth = 'compresseddepth' in fmt
    quant = None
    if depth and len(data) > 12 and not data.startswith(b'\x89PNG'):
        _, qa, qb = struct.unpack_from('<iff', data, 0)
        quant, data = (qa, qb), data[12:]
    img = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_UNCHANGED)
    if img is None:
        raise ValueError(f'cannot decode the compressed image (format {msg.get("format", "")!r})')
    if depth:
        if '32fc1' in fmt and quant is not None:
            inv = img.astype(np.float32)
            with np.errstate(divide='ignore'):
                out = np.where(inv > 0, quant[0] / (inv - quant[1]), np.nan).astype(np.float32)
            return out
        return img
    if img.ndim == 3 and img.shape[2] == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    elif img.ndim == 3 and img.shape[2] == 4:
        img = cv2.cvtColor(img, cv2.COLOR_BGRA2RGBA)
    return img


def compressed_image(jpeg_or_png: bytes, t_ns: int, frame_id: str, fmt: str = 'jpeg') -> dict:
    """sensor_msgs/CompressedImage of already-encoded JPEG or PNG bytes (format 'jpeg' or 'png')."""
    return {'header': header(t_ns, frame_id), 'format': fmt, 'data': bytes(jpeg_or_png)}


def camera_info(width: int, height: int, distortion_model: str = 'plumb_bob', d=(), k=None, r=None, p=None,
                t_ns: int = 0, frame_id: str = '', binning_x: int = 0, binning_y: int = 0, roi=None) -> dict:
    """sensor_msgs/CameraInfo. k: 3×3 (or 9 values) camera matrix, zeros when not given (uncalibrated); r the
    rectification rotation (identity when not given); p the 3×4 projection ([k | 0] when not given). d: the
    distortion coefficients of distortion_model ('plumb_bob' k1 k2 p1 p2 k3, 'rational_polynomial' 8 values,
    'equidistant' k1..k4 for fisheye)."""
    kk = np.zeros(9) if k is None else np.asarray(k, float).reshape(9)
    rr = np.eye(3).reshape(9) if r is None else np.asarray(r, float).reshape(9)
    if p is None:
        pp = np.zeros((3, 4))
        pp[:, :3] = kk.reshape(3, 3)
    else:
        pp = np.asarray(p, float).reshape(3, 4)
    return {'header': header(t_ns, frame_id), 'height': int(height), 'width': int(width),
            'distortion_model': distortion_model, 'd': [float(v) for v in np.asarray(d, float).reshape(-1)],
            'k': kk.tolist(), 'r': rr.tolist(), 'p': pp.reshape(12).tolist(),
            'binning_x': int(binning_x), 'binning_y': int(binning_y),
            'roi': dict(roi) if roi else {'x_offset': 0, 'y_offset': 0, 'height': 0, 'width': 0,
                                          'do_rectify': False}}


def quat_msg(R: np.ndarray) -> dict:
    """geometry_msgs/Quaternion of a rotation matrix."""
    x, y, z, w = quat_from_matrix(np.asarray(R, float))
    return {'x': float(x), 'y': float(y), 'z': float(z), 'w': float(w)}


def quat_matrix(q) -> np.ndarray:
    """Rotation matrix of a Quaternion dict or (x, y, z, w) (normalised first; a zero quaternion is the identity)."""
    x, y, z, w = (q['x'], q['y'], q['z'], q['w']) if isinstance(q, dict) else q
    n = float(np.sqrt(x * x + y * y + z * z + w * w))
    if n == 0:
        return np.eye(3)
    x, y, z, w = x / n, y / n, z / n, w / n
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                     [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                     [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])


def transform_stamped(T_parent_child: np.ndarray, t_ns: int, parent: str, child: str) -> dict:
    """geometry_msgs/TransformStamped of a 4×4 pose of `child` in `parent` (it maps child coordinates into parent
    coordinates), as tf2 expects: header.frame_id = parent, child_frame_id = child."""
    T = np.asarray(T_parent_child, float)
    x, y, z = T[:3, 3]
    return {'header': header(t_ns, parent), 'child_frame_id': str(child),
            'transform': {'translation': {'x': float(x), 'y': float(y), 'z': float(z)},
                          'rotation': quat_msg(T[:3, :3])}}


def transform_matrix(msg: dict) -> np.ndarray:
    """4×4 matrix of a TransformStamped or Transform dict (the inverse of transform_stamped)."""
    tr = msg.get('transform', msg)
    T = np.eye(4)
    T[:3, :3] = quat_matrix(tr['rotation'])
    T[:3, 3] = [tr['translation']['x'], tr['translation']['y'], tr['translation']['z']]
    return T


def box_marker(center, R, size, t_ns: int, frame_id: str, ns: str = '', id: int = 0,
               color=(1.0, 0.6, 0.0, 0.5), lifetime_ns: int = 0) -> dict:
    """visualization_msgs/Marker CUBE: a box of size (length, width, height) centred at `center` with rotation R in
    frame_id (a ground-truth box for RViz / Foxglove). lifetime_ns 0 keeps it until it is replaced."""
    cx, cy, cz = (float(v) for v in center)
    sx, sy, sz = (float(v) for v in size)
    r, g, b, a = (float(v) for v in color)
    return {'header': header(t_ns, frame_id), 'ns': ns, 'id': int(id), 'type': 1, 'action': 0,
            'pose': {'position': {'x': cx, 'y': cy, 'z': cz}, 'orientation': quat_msg(R)},
            'scale': {'x': sx, 'y': sy, 'z': sz}, 'color': {'r': r, 'g': g, 'b': b, 'a': a},
            'lifetime': stamp(lifetime_ns), 'frame_locked': False}
