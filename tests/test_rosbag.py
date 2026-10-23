"""ROS 2 bag I/O (vantage.core.rosbag): the CDR codec against hand-computed bytes and round trips of every type, the
point-cloud and image helpers, MCAP bags cross-checked with the mcap package both ways, rosbag2 SQLite bags, time
ordering, and damaged files."""
import os
import random
import shutil
import sqlite3
import string
import struct
import time
import zlib

import numpy as np
import pytest

from vantage.core import rosbag as rb
from vantage.core.mathutil import rot_zyx

BRIEF_TYPES = [
    'builtin_interfaces/msg/Time', 'builtin_interfaces/msg/Duration', 'std_msgs/msg/Header', 'std_msgs/msg/String',
    'std_msgs/msg/ColorRGBA', 'geometry_msgs/msg/Vector3', 'geometry_msgs/msg/Point', 'geometry_msgs/msg/Quaternion',
    'geometry_msgs/msg/Pose', 'geometry_msgs/msg/Transform', 'geometry_msgs/msg/TransformStamped',
    'geometry_msgs/msg/Twist', 'geometry_msgs/msg/PoseWithCovariance', 'geometry_msgs/msg/TwistWithCovariance',
    'tf2_msgs/msg/TFMessage', 'nav_msgs/msg/Odometry', 'sensor_msgs/msg/Image', 'sensor_msgs/msg/CompressedImage',
    'sensor_msgs/msg/CameraInfo', 'sensor_msgs/msg/RegionOfInterest', 'sensor_msgs/msg/PointCloud2',
    'sensor_msgs/msg/PointField', 'sensor_msgs/msg/Imu', 'sensor_msgs/msg/NavSatFix', 'sensor_msgs/msg/NavSatStatus',
    'sensor_msgs/msg/TimeReference', 'visualization_msgs/msg/Marker', 'visualization_msgs/msg/MarkerArray']

_INT = {'int8': (-128, 127), 'uint8': (0, 255), 'byte': (0, 255), 'char': (0, 255), 'int16': (-32768, 32767),
        'uint16': (0, 65535), 'int32': (-2**31, 2**31 - 1), 'uint32': (0, 2**32 - 1),
        'int64': (-2**63, 2**63 - 1), 'uint64': (0, 2**64 - 1)}


def _prim(t, rng):
    if t == 'bool':
        return rng.random() < 0.5
    if t in _INT:
        return rng.randint(*_INT[t])
    if t == 'float32':
        return float(np.float32(rng.uniform(-1e3, 1e3)))       # exactly representable
    if t == 'float64':
        return rng.uniform(-1e6, 1e6)
    return ''.join(rng.choice(string.ascii_letters + 'é→') for _ in range(rng.randint(0, 9)))


def _random_msg(t, rng, depth=0):
    """A message of type t with every field set to a random value (sequences 0..3 long)."""
    d = {}
    for f in rb._TYPES.spec(t).fields:
        simple = f.type in rb._PRIM or f.type == 'string'
        if f.kind == 0:
            d[f.name] = _prim(f.type, rng) if simple else _random_msg(f.type, rng, depth + 1)
            continue
        n = f.n if f.kind == 1 else rng.randint(0, 3 if depth < 2 else 1)
        if f.type in rb._BYTES:
            d[f.name] = bytes(rng.randint(0, 255) for _ in range(n if f.kind == 1 else rng.randint(0, 13)))
        elif simple:
            d[f.name] = [_prim(f.type, rng) for _ in range(n)]
        else:
            d[f.name] = [_random_msg(f.type, rng, depth + 1) for _ in range(n)]
    return d


@pytest.fixture(autouse=True)
def _restore_definitions():
    """Types a test register()s are removed afterwards, so other test modules see the stock MSG_DEFS."""
    before = set(rb.MSG_DEFS)
    yield
    for t in set(rb.MSG_DEFS) - before:
        del rb.MSG_DEFS[t]
        rb._TYPES._specs.pop(t, None)
        for cache in (rb._TYPES._enc, rb._TYPES._dec):
            for key in [k for k in cache if k[0] == t]:
                del cache[key]


@pytest.fixture
def tmp_bags(tmp_path):
    """A folder for bags, deleted afterwards (/tmp is often RAM)."""
    yield tmp_path
    shutil.rmtree(tmp_path, ignore_errors=True)


def _compressions():
    out = ['none']
    for name, mod in (('zstd', 'zstandard'), ('lz4', 'lz4.frame')):
        try:
            __import__(mod)
            out.append(name)
        except ImportError:
            pass
    return out


# ---------------------------------------------------------------------------------------------------------------------
# CDR

def test_header_bytes_known():
    data = rb.encode('std_msgs/msg/Header', rb.header(1_500_000_000, 'abc'))
    assert data.hex(' ') == '00 01 00 00 01 00 00 00 00 65 cd 1d 04 00 00 00 61 62 63 00'
    assert rb.decode('std_msgs/msg/Header', data) == {'stamp': {'sec': 1, 'nanosec': 500_000_000},
                                                      'frame_id': 'abc'}


def test_alignment_hand_computed():
    """A uint8 before a float64, strings of every length mod 4 before 8- and 2-byte fields, against a layout
    computed by hand with struct (offsets relative to the payload after the 4-byte header)."""
    rb.register('vantage_test_msgs/msg/Align', 'uint8 a\nfloat64 b\nstring s\nfloat64 c\nstring u\nint16 d\n')

    def by_hand(a, b, s, c, u, d):
        out = bytearray(struct.pack('<B', a))
        out += bytes(7) + struct.pack('<d', b)                   # b at 8
        sb = s.encode() + b'\0'
        out += struct.pack('<I', len(sb)) + sb                   # 16: length, then the bytes
        out += bytes(-len(out) % 8) + struct.pack('<d', c)
        ub = u.encode() + b'\0'
        out += bytes(-len(out) % 4) + struct.pack('<I', len(ub)) + ub
        out += bytes(-len(out) % 2) + struct.pack('<h', d)
        return b'\x00\x01\x00\x00' + bytes(out)

    for n in range(9):
        for m in range(5):
            msg = {'a': 200, 'b': -1.25, 's': 'x' * n, 'c': 3.5e10, 'u': 'y' * m, 'd': -7}
            data = rb.encode('vantage_test_msgs/msg/Align', msg)
            assert data == by_hand(**msg), (n, m)
            assert rb.decode('vantage_test_msgs/msg/Align', data) == msg


def test_empty_sequence_elements_not_aligned():
    rb.register('vantage_test_msgs/msg/Seq', 'uint8 a\nfloat64[] d\nuint8 b\n')
    empty = rb.encode('vantage_test_msgs/msg/Seq', {'a': 1, 'd': [], 'b': 2})
    assert empty[4:].hex(' ') == '01 00 00 00 00 00 00 00 02'               # no padding to 8 for no elements
    one = rb.encode('vantage_test_msgs/msg/Seq', {'a': 1, 'd': [2.0], 'b': 3})
    assert one[4:] == bytes([1, 0, 0, 0, 1, 0, 0, 0]) + struct.pack('<d', 2.0) + b'\x03'
    assert rb.decode('vantage_test_msgs/msg/Seq', one) == {'a': 1, 'd': [2.0], 'b': 3}


@pytest.mark.parametrize('t', BRIEF_TYPES)
def test_round_trip_every_type(t):
    rng = random.Random(hash(t) & 0xffff)
    for _ in range(20):
        msg = _random_msg(t, rng)
        data = rb.encode(t, msg)
        assert data[:4] == b'\x00\x01\x00\x00'
        back = rb.decode(t, data)
        assert back == msg
        assert rb.encode(t, back) == data
    # defaults: an empty dict encodes every field at its default, and that decodes and re-encodes identically
    zero = rb.encode(t, {})
    assert rb.encode(t, rb.decode(t, zero)) == zero


def test_defaults_and_errors():
    q = rb.decode('geometry_msgs/msg/Pose', rb.encode('geometry_msgs/msg/Pose', {}))
    assert q['orientation'] == {'x': 0.0, 'y': 0.0, 'z': 0.0, 'w': 1.0}
    assert rb.constants('visualization_msgs/msg/Marker')['CUBE'] == 1
    assert rb.constants('sensor_msgs/msg/NavSatStatus')['STATUS_NO_FIX'] == -1
    with pytest.raises(ValueError, match='no field'):
        rb.encode('geometry_msgs/msg/Point', {'x': 1.0, 'w': 2.0})
    with pytest.raises(ValueError):
        rb.encode('sensor_msgs/msg/Imu', {'orientation_covariance': [0.0] * 8})
    with pytest.raises(ValueError):
        rb.encode('std_msgs/msg/Header', {'stamp': {'sec': 2**31}})              # int32 overflow
    with pytest.raises(ValueError, match='unknown message type'):
        rb.encode('nope_msgs/msg/Nothing', {})
    good = rb.encode('sensor_msgs/msg/NavSatFix', {'latitude': 12.5, 'header': rb.header(3, 'gnss')})
    for cut in (3, 10, 20, len(good) - 1):
        with pytest.raises(ValueError):
            rb.decode('sensor_msgs/msg/NavSatFix', good[:cut])
    with pytest.raises(ValueError, match='XCDR1'):
        rb.decode('std_msgs/msg/String', b'\x00\x07\x00\x00' + good[4:])
    # short type names and numpy inputs
    k = np.array([[800.0, 0, 640], [0, 800, 360], [0, 0, 1]])
    info = rb.camera_info(1280, 720, 'plumb_bob', np.array([0.1, -0.05, 0, 0, 0.01]), k, t_ns=5, frame_id='cam')
    back = rb.decode('sensor_msgs/CameraInfo', rb.encode('sensor_msgs/CameraInfo', dict(info, k=k)))
    assert back['k'] == k.reshape(9).tolist() and back['p'][:3] == [800.0, 0.0, 640.0] and back['r'][4] == 1.0
    assert back['d'] == [0.1, -0.05, 0.0, 0.0, 0.01]


def test_big_endian_decoded():
    msg = {'header': rb.header(7_000_000_123, 'gps'), 'status': {'status': -1, 'service': 9},
           'latitude': 48.1, 'longitude': 11.5, 'altitude': 520.0, 'position_covariance': [float(i) for i in range(9)],
           'position_covariance_type': 2}
    le = rb.encode('sensor_msgs/msg/NavSatFix', msg)[4:]
    # the same message written big-endian by hand: every primitive byte-swapped at the same offsets
    hb = struct.pack('>iI', 7, 123) + struct.pack('>I', 4) + b'gps\0'
    hb += struct.pack('>bxH', -1, 9)
    hb += bytes(-len(hb) % 8)                                   # latitude aligns to 8
    hb += struct.pack('>3d', 48.1, 11.5, 520.0) + struct.pack('>9d', *range(9)) + b'\x02'
    assert len(hb) == len(le)
    assert rb.decode('sensor_msgs/msg/NavSatFix', b'\x00\x00\x00\x00' + hb) == msg


def test_schema_text_like_rosbag2():
    text = rb.schema_text('nav_msgs/msg/Odometry')
    assert text.startswith(rb.MSG_DEFS['nav_msgs/msg/Odometry'])
    names = [ln[5:] for ln in text.splitlines() if ln.startswith('MSG: ')]
    assert names == ['geometry_msgs/PoseWithCovariance', 'geometry_msgs/Pose', 'geometry_msgs/Point',
                     'geometry_msgs/Quaternion', 'geometry_msgs/TwistWithCovariance', 'geometry_msgs/Twist',
                     'geometry_msgs/Vector3', 'std_msgs/Header', 'builtin_interfaces/Time']
    assert all(ln == '=' * 80 for ln in text.splitlines() if ln.startswith('='))
    assert '\n\n' + '=' * 80 + '\nMSG: geometry_msgs/PoseWithCovariance\n' in text


# ---------------------------------------------------------------------------------------------------------------------
# point clouds and images

def test_pointcloud2_mixed_fields_padding():
    dt = np.dtype({'names': ['x', 'y', 'z', 'intensity', 'ring', 't', 'flag', 'rgb'],
                   'formats': ['<f4', '<f4', '<f4', '<u2', '<u1', '<f8', '<i1', ('<u1', (3,))],
                   'offsets': [0, 4, 8, 12, 14, 16, 24, 25], 'itemsize': 32})
    rng = np.random.default_rng(3)
    pts = np.zeros(257, dt)
    for n in ('x', 'y', 'z'):
        pts[n] = rng.normal(0, 30, 257)
    pts['intensity'] = rng.integers(0, 65535, 257)
    pts['ring'] = rng.integers(0, 128, 257)
    pts['t'] = rng.uniform(0, 0.1, 257)
    pts['flag'] = rng.integers(-128, 127, 257)
    pts['rgb'] = rng.integers(0, 255, (257, 3))
    msg = rb.pointcloud2(pts, 42, 'lidar')
    assert msg['point_step'] == 32 and msg['row_step'] == 32 * 257 and msg['height'] == 1 and msg['is_dense']
    assert [(f['name'], f['offset'], f['datatype'], f['count']) for f in msg['fields']] == [
        ('x', 0, 7, 1), ('y', 4, 7, 1), ('z', 8, 7, 1), ('intensity', 12, 4, 1), ('ring', 14, 2, 1),
        ('t', 16, 8, 1), ('flag', 24, 1, 1), ('rgb', 25, 2, 3)]
    # independent check of the data layout: point 5's z and t read straight from the bytes
    assert struct.unpack_from('<f', msg['data'], 5 * 32 + 8)[0] == pts['z'][5]
    assert struct.unpack_from('<d', msg['data'], 5 * 32 + 16)[0] == pts['t'][5]
    back = rb.pointcloud2_array(rb.decode('sensor_msgs/msg/PointCloud2', rb.encode('sensor_msgs/msg/PointCloud2', msg)))
    assert back.shape == (257,) and back.dtype.itemsize == 32
    for n in dt.names:
        assert np.array_equal(back[n], pts[n]), n
    pts['x'][3] = np.nan
    assert not rb.pointcloud2(pts, 0, 'lidar')['is_dense']


def test_pointcloud2_organised_bigendian_rowpad_unknown_field():
    h, w = 3, 4
    be = np.dtype({'names': ['x', 'y', 'z', 'i'], 'formats': ['>f4', '>f4', '>f4', '>u2'],
                   'offsets': [0, 4, 8, 12], 'itemsize': 16})
    pts = np.zeros((h, w), be)
    pts['x'] = np.arange(12).reshape(h, w) * 1.5
    pts['y'] = -pts['x']
    pts['z'] = 0.25
    pts['i'] = np.arange(12).reshape(h, w) * 1000
    row_step = w * 16 + 8                                            # 8 padding bytes after every row
    data = b''.join(pts[r].tobytes() + b'\xee' * 8 for r in range(h))
    fields = [{'name': n, 'offset': o, 'datatype': d, 'count': 1}
              for n, o, d in (('x', 0, 7), ('y', 4, 7), ('z', 8, 7), ('i', 12, 4),
                              ('odd', 14, 42))]                     # an unknown datatype: skipped
    msg = {'header': rb.header(0, 'l'), 'height': h, 'width': w, 'fields': fields, 'is_bigendian': True,
           'point_step': 16, 'row_step': row_step, 'data': data, 'is_dense': True}
    back = rb.pointcloud2_array(rb.decode('sensor_msgs/msg/PointCloud2', rb.encode('sensor_msgs/msg/PointCloud2', msg)))
    assert back.shape == (h, w) and back.dtype.names == ('x', 'y', 'z', 'i')
    for n in back.dtype.names:
        assert np.array_equal(back[n], pts[n].astype(pts[n].dtype.newbyteorder('='))), n
    # written back it is little-endian and dense rows, same values
    again = rb.pointcloud2_array(rb.pointcloud2(back, 0, 'l'))
    assert again.shape == (h, w) and np.array_equal(again['i'], back['i'])
    with pytest.raises(ValueError):
        rb.pointcloud2(np.zeros(3, [('x', '<u8')]), 0, 'l')


def test_images_round_trip():
    rng = np.random.default_rng(5)
    rgb = rng.integers(0, 256, (37, 53, 3), dtype=np.uint8)
    for img, enc in ((rgb, 'rgb8'), (rgb[:, :, 0], 'mono8'),
                     (rng.integers(0, 65536, (9, 11), dtype=np.uint16), 'mono16'),
                     (rng.normal(size=(5, 7)).astype(np.float32), '32FC1'),
                     (rng.integers(0, 256, (6, 8, 4), dtype=np.uint8), 'rgba8')):
        msg = rb.image_msg(img, 10, 'cam')
        channels = img.shape[2] if img.ndim == 3 else 1
        assert msg['encoding'] == enc and msg['step'] == img.shape[1] * img.itemsize * channels
        back = rb.image_array(rb.decode('sensor_msgs/msg/Image', rb.encode('sensor_msgs/msg/Image', msg)))
        assert back.dtype == img.dtype and np.array_equal(back, img), enc
    bgr = rb.image_msg(rgb[:, :, ::-1], 0, 'cam', 'bgr8')
    assert np.array_equal(rb.image_array(bgr)[:, :, ::-1], rgb)
    assert np.array_equal(rb.image_array(bgr, to_rgb=True), rgb)
    bayer = rb.image_msg(rgb[:, :, 1], 0, 'cam', 'bayer_rggb8')
    assert np.array_equal(rb.image_array(bayer), rgb[:, :, 1])
    # step padding and big-endian 16-bit
    m16 = rng.integers(0, 65536, (4, 5), dtype=np.uint16)
    rows = b''.join(m16[r].astype('>u2').tobytes() + b'\0\0\0\0\0\0' for r in range(4))
    msg = {'header': rb.header(0, 'c'), 'height': 4, 'width': 5, 'encoding': 'mono16', 'is_bigendian': 1,
           'step': 16, 'data': rows}
    assert np.array_equal(rb.image_array(msg), m16)
    with pytest.raises(ValueError):
        rb.image_msg(rgb, 0, 'cam', 'mono16')
    with pytest.raises(ValueError):
        rb.image_array(dict(msg, encoding='nv12_odd'))


def test_compressed_images():
    cv2 = pytest.importorskip('cv2')
    rng = np.random.default_rng(6)
    rgb = rng.integers(0, 256, (24, 32, 3), dtype=np.uint8)
    ok, png = cv2.imencode('.png', cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
    msg = rb.compressed_image(png.tobytes(), 5, 'cam', 'png')
    back = rb.decode('sensor_msgs/msg/CompressedImage', rb.encode('sensor_msgs/msg/CompressedImage', msg))
    assert back['format'] == 'png' and np.array_equal(rb.image_array(back), rgb)
    smooth = np.repeat(np.linspace(0, 255, 64).astype(np.uint8)[None, :, None], 48, 0).repeat(3, 2)
    smooth[..., 0] = 200
    ok, jpg = cv2.imencode('.jpg', cv2.cvtColor(smooth, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 95])
    dec = rb.image_array(rb.compressed_image(jpg.tobytes(), 0, 'cam'))
    assert dec.shape == smooth.shape and np.abs(dec.astype(int) - smooth).mean() < 3
    assert abs(int(dec[..., 0].mean()) - 200) < 3                       # red stays red: RGB order
    d16 = rng.integers(0, 65535, (8, 9), dtype=np.uint16)
    ok, p16 = cv2.imencode('.png', d16)
    assert np.array_equal(rb.image_array(rb.compressed_image(p16.tobytes(), 0, 'depth', 'png')), d16)
    with pytest.raises(ValueError):
        rb.image_array(rb.compressed_image(b'not an image', 0, 'cam'))


def test_helpers():
    for t in (0, 1, 999_999_999, 1_000_000_000, 1_700_000_000_123_456_789, -1, -1_500_000_000):
        st = rb.stamp(t)
        assert 0 <= st['nanosec'] < 1_000_000_000 and rb.stamp_ns(st) == t
    R = rot_zyx(30.0, -10.0, 5.0)
    T = np.eye(4)
    T[:3, :3], T[:3, 3] = R, [1.2, -0.4, 1.9]
    ts = rb.transform_stamped(T, 77, 'base_link', 'cam_F')
    assert ts['header']['frame_id'] == 'base_link' and ts['child_frame_id'] == 'cam_F'
    back = rb.decode('geometry_msgs/msg/TransformStamped', rb.encode('geometry_msgs/msg/TransformStamped', ts))
    assert np.allclose(rb.transform_matrix(back), T, atol=1e-12)
    q = rb.quat_msg(R)
    assert abs(q['x'] ** 2 + q['y'] ** 2 + q['z'] ** 2 + q['w'] ** 2 - 1) < 1e-12
    # independent: rotating x by the quaternion (q v q*) equals R @ x
    w, v = q['w'], np.array([q['x'], q['y'], q['z']])
    x = np.array([1.0, 0, 0])
    assert np.allclose(x + 2 * w * np.cross(v, x) + 2 * np.cross(v, np.cross(v, x)), R @ x)
    mk = rb.box_marker([5, 1, 0.8], R, [4.5, 1.8, 1.5], 9, 'map', 'gt', 3)
    m2 = rb.decode('visualization_msgs/msg/Marker', rb.encode('visualization_msgs/msg/Marker', mk))
    assert m2['type'] == rb.constants('visualization_msgs/msg/Marker')['CUBE'] and m2['scale']['x'] == 4.5
    arr = rb.decode('visualization_msgs/msg/MarkerArray', rb.encode('visualization_msgs/msg/MarkerArray',
                                                                    {'markers': [mk, mk]}))
    assert len(arr['markers']) == 2 and arr['markers'][1]['id'] == 3


# ---------------------------------------------------------------------------------------------------------------------
# bags

def _check_with_mcap(path, sent_counts, sent_data):
    """Read a .mcap we wrote with the mcap package (CRCs validated): schemas, channels, counts and bytes."""
    from mcap.reader import make_reader
    from mcap.stream_reader import StreamReader
    with open(path, 'rb') as f:
        rd = make_reader(f, validate_crcs=True)
        summary = rd.get_summary()
        assert summary.statistics.message_count == sum(sent_counts.values())
        assert rd.get_header().profile == 'ros2' and rd.get_header().library == 'vantage'
        got_counts, got = {}, {}
        for schema, channel, message in rd.iter_messages(log_time_order=True):
            assert schema.encoding == 'ros2msg' and channel.message_encoding == 'cdr'
            assert schema.data.decode() == rb.schema_text(schema.name)
            assert 'offered_qos_profiles' in channel.metadata
            got_counts[channel.topic] = got_counts.get(channel.topic, 0) + 1
            got.setdefault(channel.topic, []).append((message.log_time, message.data))
        assert got_counts == sent_counts
        for topic, lst in sent_data.items():
            assert sorted(got[topic]) == sorted(lst), topic
        chunks = summary.chunk_indexes
    with open(path, 'rb') as f:                     # a linear read of every record checks the data-section CRC
        recs = list(StreamReader(f, validate_crcs=True).records)
    assert type(recs[-1]).__name__ == 'Footer'
    return chunks


@pytest.mark.parametrize('comp', _compressions())
def test_bag_three_topics_chunked(tmp_bags, comp):
    pytest.importorskip('mcap')
    rng = np.random.default_rng(11)
    big = rng.integers(0, 256, (512, 683, 3), dtype=np.uint8)            # ~1 MB rgb8
    small = rng.integers(0, 256, (16, 16, 3), dtype=np.uint8)
    path = str(tmp_bags / f'drive_{comp}')
    sent = {'/imu': [], '/tf': [], '/cam': []}
    types = {'/imu': 'sensor_msgs/msg/Imu', '/tf': 'tf2_msgs/msg/TFMessage', '/cam': 'sensor_msgs/msg/Image'}
    with rb.BagWriter(path, compression=comp, chunk_size=1 << 20) as bag:
        bag.add_topic('/imu', 'sensor_msgs/msg/Imu')
        bag.add_topic('/tf', 'tf2_msgs/msg/TFMessage')
        bag.add_topic('/cam', 'sensor_msgs/msg/Image')
        for i in range(1000):
            t = 1_700_000_000_000_000_000 + i * 10_000_000
            imu = {'header': rb.header(t, 'imu'), 'angular_velocity': {'z': i * 1e-3},
                   'linear_acceleration': {'x': 0.1, 'z': 9.81}}
            tf = {'transforms': [rb.transform_stamped(np.eye(4), t, 'odom', 'base_link')]}
            cam = rb.image_msg(big if i % 50 == 0 else small, t + 3, 'cam_F')
            for topic, m, tt in (('/imu', imu, t + 1), ('/tf', tf, t + 2), ('/cam', cam, t + 3)):
                data = rb.encode(types[topic], m)
                bag.write(topic, tt, m if topic != '/tf' else data)          # dicts and pre-encoded bytes
                sent[topic].append((tt, data))
        res = bag.close()
    assert res['messages'] == 3000 and res['topics'] == {'/imu': 1000, '/tf': 1000, '/cam': 1000}
    assert sorted(os.listdir(path)) == sorted(['metadata.yaml', f'drive_{comp}_0.mcap'])
    chunks = _check_with_mcap(res['file'], res['topics'], sent)
    assert len(chunks) >= 20 and {c.compression for c in chunks} == {'' if comp == 'none' else comp}
    with rb.BagReader(path) as r:
        assert r.topics == {'/imu': 'sensor_msgs/msg/Imu', '/tf': 'tf2_msgs/msg/TFMessage',
                            '/cam': 'sensor_msgs/msg/Image'}
        assert r.counts == res['topics'] and r.message_count == 3000 and not r.warnings
        assert r.start_ns == 1_700_000_000_000_000_001 and r.end_ns == 1_700_000_000_000_000_000 + 999 * 10_000_000 + 3
        prev, n = -1, 0
        for topic, t, msg in r.messages():
            assert t >= prev
            prev = t
            if topic == '/cam' and n % 150 == 2:
                img = rb.image_array(msg)
                assert np.array_equal(img, big if img.shape[0] == 512 else small)
            n += 1
        assert n == 3000
        raw = [(t, m) for _, t, m in r.messages('/tf', decode=False)]
        assert raw == sent['/tf']
        window = list(r.messages(['/imu', '/cam'], start_ns=r.start_ns + 100_000_000, end_ns=r.start_ns + 200_000_000))
        assert len(window) == 20 and {w[0] for w in window} == {'/imu', '/cam'}
        assert all(r.start_ns + 100_000_000 <= w[1] < r.start_ns + 200_000_000 for w in window)
    import yaml
    meta = yaml.safe_load(open(os.path.join(path, 'metadata.yaml')))['rosbag2_bagfile_information']
    assert meta['version'] == 5 and meta['storage_identifier'] == 'mcap' and meta['message_count'] == 3000
    assert meta['relative_file_paths'] == [f'drive_{comp}_0.mcap']
    assert meta['starting_time']['nanoseconds_since_epoch'] == r.start_ns
    assert meta['duration']['nanoseconds'] == r.end_ns - r.start_ns
    assert {x['topic_metadata']['name']: x['message_count'] for x in meta['topics_with_message_count']} == res['topics']


def test_out_of_order_writes_come_back_in_time_order(tmp_bags):
    rng = random.Random(4)
    path = str(tmp_bags / 'shuffled')
    times = [rng.randint(0, 10**12) for _ in range(3000)]
    with rb.BagWriter(path, compression='none', chunk_size=4096) as bag:   # many chunks with overlapping times
        bag.add_topic('/a', 'std_msgs/msg/String')
        bag.add_topic('/b', 'geometry_msgs/msg/Point')
        for i, t in enumerate(times):
            if i % 2:
                bag.write('/a', t, {'data': str(t)})
            else:
                bag.write('/b', t, {'x': float(t)})
    with rb.BagReader(path) as r:
        got = list(r.messages())
        assert [g[1] for g in got] == sorted(times)
        assert all((m['data'] == str(t)) if tp == '/a' else (m['x'] == float(t)) for tp, t, m in got)
        assert [g[1] for g in r.messages('/b')] == sorted(t for i, t in enumerate(times) if i % 2 == 0)
        lo, hi = sorted(times)[1000], sorted(times)[2000]
        assert [g[1] for g in r.messages(start_ns=lo, end_ns=hi)] == [t for t in sorted(times) if lo <= t < hi]
        assert r.start_ns == min(times) and r.end_ns == max(times)


@pytest.mark.parametrize('variant', ['zstd', 'lz4', 'none', 'unchunked', 'no_stats', 'no_index', 'chunk_index_only'])
def test_files_written_by_mcap_package(tmp_bags, variant):
    """Files from mcap.writer.Writer (schemas and channels inside chunks, other index layouts) read by BagReader."""
    pytest.importorskip('mcap')
    from mcap.writer import CompressionType, IndexType, Writer
    if variant == 'zstd':
        pytest.importorskip('zstandard')
    if variant == 'lz4':
        pytest.importorskip('lz4')
    kw = {'compression': {'zstd': CompressionType.ZSTD, 'lz4': CompressionType.LZ4}.get(variant, CompressionType.NONE),
          'chunk_size': 2048}
    if variant == 'unchunked':
        kw['use_chunking'] = False
    if variant == 'no_stats':
        kw['use_statistics'] = False
    if variant == 'no_index':
        kw['index_types'] = IndexType.NONE
    if variant == 'chunk_index_only':
        kw.update(index_types=IndexType.CHUNK, use_statistics=False)
    path = str(tmp_bags / 'theirs.mcap')
    rng = random.Random(9)
    sent = []
    with open(path, 'wb') as f:
        w = Writer(f, **kw)
        w.start(profile='ros2', library='test')
        chans = {}
        for t, topic in (('sensor_msgs/msg/NavSatFix', '/gnss'), ('nav_msgs/msg/Odometry', '/odom'),
                         ('visualization_msgs/msg/MarkerArray', '/boxes')):
            sid = w.register_schema(name=t, encoding='ros2msg', data=rb.schema_text(t).encode())
            chans[topic] = (w.register_channel(topic=topic, message_encoding='cdr', schema_id=sid), t)
        for i in range(400):
            topic = rng.choice(list(chans))
            msg = _random_msg(chans[topic][1], rng)
            t = rng.randint(0, 10**10)
            w.add_message(chans[topic][0], log_time=t, data=rb.encode(chans[topic][1], msg), publish_time=t)
            sent.append((t, topic, msg))
        w.finish()
    with rb.BagReader(path) as r:
        assert r.topics == {'/gnss': 'sensor_msgs/msg/NavSatFix', '/odom': 'nav_msgs/msg/Odometry',
                            '/boxes': 'visualization_msgs/msg/MarkerArray'}
        assert r.message_count == 400 and sum(r.counts.values()) == 400
        assert r.start_ns == min(s[0] for s in sent) and r.end_ns == max(s[0] for s in sent)
        got = list(r.messages())
        assert [g[1] for g in got] == sorted(s[0] for s in sent)
        assert sorted((g[1], g[0], repr(g[2])) for g in got) == sorted((s[0], s[1], repr(s[2])) for s in sent)
        assert not r.decode_errors
        assert bool(r.warnings) == (variant in ('unchunked', 'no_index'))       # those are read by a linear scan


def test_bag_own_definitions_and_undecodable(tmp_bags):
    """A type MSG_DEFS does not know is decoded from the bag's schema; a bag's own definition of a known name wins;
    a non-CDR channel and a corrupt message come back as bytes without stopping the iteration."""
    pytest.importorskip('mcap')
    from mcap.writer import CompressionType, Writer
    path = str(tmp_bags / 'vendor.mcap')
    vendor = ('std_msgs/Header header\nuint8 MODE_A=1\nuint8 mode\nfloat64[] ranges\nstring[] names\n'
              '\n' + '=' * 80 + '\nMSG: std_msgs/Header\nbuiltin_interfaces/Time stamp\nstring frame_id\n'
              '\n' + '=' * 80 + '\nMSG: builtin_interfaces/Time\nint32 sec\nuint32 nanosec\n')
    with open(path, 'wb') as f:
        w = Writer(f, compression=CompressionType.NONE)
        w.start(profile='ros2', library='test')
        s1 = w.register_schema('acme_msgs/msg/Scan', 'ros2msg', vendor.encode())
        s2 = w.register_schema('std_msgs/msg/String', 'ros2msg', b'int32 data\n')        # an odd String
        s3 = w.register_schema('foo', 'jsonschema', b'{}')
        c1 = w.register_channel('/scan', 'cdr', s1)
        c2 = w.register_channel('/odd', 'cdr', s2)
        c3 = w.register_channel('/json', 'json', s3)
        c4 = w.register_channel('/broken', 'cdr', w.register_schema('sensor_msgs/msg/Imu', 'ros2msg',
                                                                    rb.schema_text('sensor_msgs/msg/Imu').encode()))
        body = (struct.pack('<iI', 3, 4) + struct.pack('<I', 2) + b'l\0' + b'\x01' + bytes(1)   # mode at 14, pad
                + struct.pack('<I', 2) + bytes(4) + struct.pack('<2d', 1.5, 2.5)            # ranges at 24
                + struct.pack('<I', 1) + struct.pack('<I', 3) + b'ab\0')
        w.add_message(c1, log_time=10, data=b'\x00\x01\x00\x00' + body, publish_time=10)
        w.add_message(c2, log_time=20, data=b'\x00\x01\x00\x00' + struct.pack('<i', -5), publish_time=20)
        w.add_message(c3, log_time=30, data=b'{"a": 1}', publish_time=30)
        w.add_message(c4, log_time=40, data=b'\x00\x01\x00\x00\x01\x02', publish_time=40)     # truncated Imu
        w.add_message(c2, log_time=50, data=b'\x00\x01\x00\x00' + struct.pack('<i', 6), publish_time=50)
        w.finish()
    with rb.BagReader(path) as r:
        got = list(r.messages())
    assert [g[0] for g in got] == ['/scan', '/odd', '/json', '/broken', '/odd']
    assert got[0][2] == {'header': {'stamp': {'sec': 3, 'nanosec': 4}, 'frame_id': 'l'}, 'mode': 1,
                         'ranges': [1.5, 2.5], 'names': ['ab']}
    assert got[1][2] == {'data': -5} and got[4][2] == {'data': 6}
    assert got[2][2] == b'{"a": 1}' and isinstance(got[3][2], bytes)
    assert r.decode_errors == {'/broken': 1}
    with rb.BagReader(path) as r:                     # lazy decoding of raw messages with the bag's definitions
        raw = {tp: m for tp, _, m in r.messages(decode=False)}
        assert r.decode('/scan', raw['/scan']) == got[0][2] and r.decode('/odd', raw['/odd']) == {'data': 6}
        for tp in ('/json', '/broken', '/nothing'):
            with pytest.raises(ValueError):
                r.decode(tp, raw.get(tp, b''))
    assert rb.decode('std_msgs/msg/String', rb.encode('std_msgs/msg/String', {'data': 'ok'})) == {'data': 'ok'}


def _make_db3(path, rows, topics, definitions=None):
    con = sqlite3.connect(path)
    con.execute('CREATE TABLE topics(id INTEGER PRIMARY KEY, name TEXT NOT NULL, type TEXT NOT NULL, '
                'serialization_format TEXT NOT NULL, offered_qos_profiles TEXT NOT NULL)')
    con.execute('CREATE TABLE messages(id INTEGER PRIMARY KEY, topic_id INTEGER NOT NULL, timestamp INTEGER NOT NULL, '
                'data BLOB NOT NULL)')
    con.execute('CREATE INDEX timestamp_idx ON messages (timestamp ASC)')
    for tid, (name, typ) in topics.items():
        con.execute('INSERT INTO topics VALUES (?, ?, ?, ?, ?)', (tid, name, typ, 'cdr', ''))
    if definitions:
        con.execute('CREATE TABLE message_definitions(id INTEGER PRIMARY KEY, topic_type TEXT NOT NULL, '
                    'encoding TEXT NOT NULL, encoded_message_definition TEXT NOT NULL, type_description_hash TEXT)')
        for typ, text in definitions.items():
            con.execute('INSERT INTO message_definitions(topic_type, encoding, encoded_message_definition, '
                        'type_description_hash) VALUES (?, ?, ?, ?)', (typ, 'ros2msg', text, ''))
    con.executemany('INSERT INTO messages(topic_id, timestamp, data) VALUES (?, ?, ?)', rows)
    con.commit()
    con.close()


def test_db3_bags(tmp_bags):
    rng = random.Random(2)
    d = tmp_bags / 'sqlite_bag'
    d.mkdir()
    topics = {1: ('/imu', 'sensor_msgs/msg/Imu'), 2: ('/fix', 'sensor_msgs/msg/NavSatFix'),
              3: ('/custom', 'acme_msgs/msg/Speed'), 4: ('/unknown', 'acme_msgs/msg/Mystery')}
    rows, sent = [], []
    for i in range(600):
        tid = rng.choice([1, 2, 3, 4])
        t = rng.randint(0, 10**9)
        if tid in (1, 2):
            msg = _random_msg(topics[tid][1], rng)
            data = rb.encode(topics[tid][1], msg)
        elif tid == 3:
            msg = {'kmh': float(i), 'valid': True}
            data = b'\x00\x01\x00\x00' + struct.pack('<d?', float(i), True)
        else:
            data = msg = b'\x00\x01\x00\x00\x09'
        rows.append((tid, t, data))
        sent.append((t, topics[tid][0], msg))
    _make_db3(str(d / 'sqlite_bag_0.db3'), rows, topics, {'acme_msgs/msg/Speed': 'float64 kmh\nbool valid\n'})
    for target in (str(d / 'sqlite_bag_0.db3'), str(d)):
        with rb.BagReader(target) as r:
            assert r.storage == 'sqlite3'
            assert r.topics == {v[0]: v[1] for v in topics.values()}
            assert r.message_count == 600 and r.start_ns == min(s[0] for s in sent)
            got = list(r.messages())
            assert [g[1] for g in got] == sorted(s[0] for s in sent)
            assert sorted((g[1], g[0], repr(g[2])) for g in got) == sorted((s[0], s[1], repr(s[2])) for s in sent)
            only = list(r.messages(['/fix'], start_ns=200_000_000, end_ns=600_000_000))
            assert [g[1] for g in only] == sorted(s[0] for s in sent if s[1] == '/fix' and 2e8 <= s[0] < 6e8)
    # with a rosbag2 metadata.yaml naming the file
    import yaml
    meta = {'rosbag2_bagfile_information': {'version': 5, 'storage_identifier': 'sqlite3',
                                            'relative_file_paths': ['sqlite_bag_0.db3'], 'compression_mode': '',
                                            'compression_format': ''}}
    (d / 'metadata.yaml').write_text(yaml.safe_dump(meta))
    with rb.BagReader(str(d)) as r:
        assert r.metadata['storage_identifier'] == 'sqlite3' and r.message_count == 600
    con = sqlite3.connect(str(tmp_bags / 'other.db3'))
    con.execute('CREATE TABLE x(a)')
    con.close()
    with pytest.raises(ValueError, match='not a rosbag2 SQLite bag'):
        rb.BagReader(str(tmp_bags / 'other.db3'))


def test_split_bag_merged(tmp_bags):
    """A bag of two .mcap files (as rosbag2 splits a long recording) reads as one time-ordered stream."""
    import yaml
    parts = []
    for k, offset in enumerate((0, 5)):
        p = str(tmp_bags / f'part{k}')
        with rb.BagWriter(p, compression='none') as bag:
            bag.add_topic('/n', 'std_msgs/msg/Int32')
            for i in range(10):
                bag.write('/n', offset + 10 * i, {'data': offset + 10 * i})
        parts.append(os.path.join(p, f'part{k}_0.mcap'))
    d = tmp_bags / 'split'
    d.mkdir()
    for k, p in enumerate(parts):
        shutil.move(p, d / f'split_{k}.mcap')
    meta = {'rosbag2_bagfile_information': {'version': 5, 'storage_identifier': 'mcap',
                                            'relative_file_paths': ['split_0.mcap', 'split_1.mcap']}}
    (d / 'metadata.yaml').write_text(yaml.safe_dump(meta))
    with rb.BagReader(str(d)) as r:
        assert r.counts == {'/n': 20} and len(r.files) == 2
        expect = sorted([10 * i for i in range(10)] + [5 + 10 * i for i in range(10)])
        assert [m['data'] for _, _, m in r.messages()] == expect


def test_truncated_and_corrupt_files(tmp_bags):
    path = str(tmp_bags / 'cut')
    with rb.BagWriter(path, compression='none', chunk_size=2048) as bag:
        bag.add_topic('/s', 'std_msgs/msg/String')
        for i in range(500):
            bag.write('/s', 1000 + i, {'data': f'message {i:04d}'})
        res = bag.close()
    full = open(res['file'], 'rb').read()
    for frac in (0.02, 0.3, 0.77, 0.999):
        cut = str(tmp_bags / f'cut_{frac}.mcap')
        with open(cut, 'wb') as f:
            f.write(full[:int(len(full) * frac)])
        t0 = time.time()
        with rb.BagReader(cut) as r:
            got = [m['data'] for _, _, m in r.messages()]
            assert r.warnings                                          # read by a linear scan, said so
        assert time.time() - t0 < 10
        assert got == [f'message {i:04d}' for i in range(len(got))]   # a clean prefix
        if frac > 0.5:
            assert len(got) > 300
    # a corrupt byte inside a chunk's messages is caught by the chunk's CRC
    bad = bytearray(full)
    bad[full.index(b'message 0100') + 3] ^= 0xff
    with open(str(tmp_bags / 'bad.mcap'), 'wb') as f:
        f.write(bad)
    with pytest.raises(ValueError):
        with rb.BagReader(str(tmp_bags / 'bad.mcap')) as r:
            list(r.messages())
    with open(str(tmp_bags / 'junk.mcap'), 'wb') as f:
        f.write(b'not a bag at all' * 10)
    with pytest.raises(ValueError):
        rb.BagReader(str(tmp_bags / 'junk.mcap'))
    with pytest.raises(FileNotFoundError):
        rb.BagReader(str(tmp_bags / 'missing'))


def test_writer_rules(tmp_bags):
    path = str(tmp_bags / 'rules')
    bag = rb.BagWriter(path, compression='none')
    bag.add_topic('/x', 'std_msgs/msg/Float64')
    bag.add_topic('/x', 'std_msgs/Float64')                            # the same again: fine
    with pytest.raises(ValueError):
        bag.add_topic('/x', 'std_msgs/msg/String')
    with pytest.raises(ValueError):
        bag.add_topic('/y', 'acme_msgs/msg/Nothing')
    with pytest.raises(ValueError):
        bag.write('/z', 0, {'data': 1.0})
    with pytest.raises(ValueError):
        bag.write('/x', -5, {'data': 1.0})
    with pytest.raises(ValueError):
        bag.write('/x', 5, b'\x07\x07')
    bag.write('/x', 5, {'data': 2.5})
    first = bag.close()
    assert bag.close() == first and first['messages'] == 1
    with pytest.raises(ValueError):
        bag.write('/x', 6, {'data': 1.0})
    with pytest.raises(FileExistsError):
        rb.BagWriter(path)
    with rb.BagWriter(path, overwrite=True) as again:
        again.add_topic('/tf_static', 'tf2_msgs/msg/TFMessage')
        again.write('/tf_static', 0, {'transforms': []})
    with rb.BagReader(path) as r:
        assert r.counts == {'/tf_static': 1}
    import yaml
    meta = yaml.safe_load(open(os.path.join(path, 'metadata.yaml')))['rosbag2_bagfile_information']
    qos = meta['topics_with_message_count'][0]['topic_metadata']['offered_qos_profiles']
    assert yaml.safe_load(qos)[0]['durability'] == 1                    # transient local, so playback latches it
    empty = str(tmp_bags / 'empty')
    with rb.BagWriter(empty) as bag:
        pass
    with rb.BagReader(empty) as r:
        assert r.message_count == 0 and r.start_ns == 0 and list(r.messages()) == []
    with pytest.raises(ValueError):
        rb.BagWriter(str(tmp_bags / 'c'), compression='brotli')


def test_write_speed(tmp_bags):
    """About 100 MB of camera frames are written in seconds, not minutes, and read back streaming."""
    img = np.random.default_rng(0).integers(0, 256, (1080, 1920, 3), dtype=np.uint8)
    path = str(tmp_bags / 'speed')
    t0 = time.perf_counter()
    with rb.BagWriter(path) as bag:
        bag.add_topic('/cam', 'sensor_msgs/msg/Image')
        for i in range(16):
            bag.write('/cam', i * 33_000_000, rb.image_msg(img, i * 33_000_000, 'cam'))
        res = bag.close()
    assert res['bytes'] > 99e6 and time.perf_counter() - t0 < 15
    t0 = time.perf_counter()
    with rb.BagReader(path) as r:
        n = sum(1 for _ in r.messages(decode=False))
    assert n == 16 and time.perf_counter() - t0 < 15


def test_skip_corrupt_chunk_and_fuzz(tmp_bags):
    path = str(tmp_bags / 'fz')
    with rb.BagWriter(path, compression='none', chunk_size=2048) as bag:
        bag.add_topic('/s', 'std_msgs/msg/String')
        for i in range(300):
            bag.write('/s', i, {'data': f'message {i:04d}'})
        res = bag.close()
    full = open(res['file'], 'rb').read()
    bad = bytearray(full)
    bad[full.index(b'message 0150') + 2] ^= 0x20
    fp = str(tmp_bags / 'one_bad.mcap')
    open(fp, 'wb').write(bad)
    with rb.BagReader(fp) as r:
        got = [m['data'] for _, _, m in r.messages(skip_corrupt=True)]
        assert 'message 0150' not in got and 250 < len(got) < 300            # one chunk left out
        assert got == sorted(got) and any('skipped' in w for w in r.warnings)
    # random damage: a reader either reads or raises ValueError, never anything else, never hangs
    rng = random.Random(5)
    for trial in range(60):
        b = bytearray(full)
        for _ in range(rng.randint(1, 8)):
            b[rng.randrange(len(b))] = rng.randrange(256)
        b = b[:rng.randrange(len(b) // 2, len(b) + 1)]
        open(fp, 'wb').write(b)
        try:
            with rb.BagReader(fp) as r:
                list(r.messages(skip_corrupt=True))
        except ValueError:
            pass


def test_rosbag2_message_compression_and_file_compression(tmp_bags):
    zstd = pytest.importorskip('zstandard')
    import yaml
    d = tmp_bags / 'msgz'
    d.mkdir()
    rows = [(1, 100 + i, zstd.ZstdCompressor().compress(rb.encode('std_msgs/msg/Float64', {'data': i * 0.5})))
            for i in range(50)]
    _make_db3(str(d / 'msgz_0.db3'), rows, {1: ('/v', 'std_msgs/msg/Float64')})
    meta = {'rosbag2_bagfile_information': {'version': 5, 'storage_identifier': 'sqlite3',
                                            'relative_file_paths': ['msgz_0.db3'], 'compression_mode': 'MESSAGE',
                                            'compression_format': 'zstd'}}
    (d / 'metadata.yaml').write_text(yaml.safe_dump(meta))
    with rb.BagReader(str(d)) as r:
        assert [m['data'] for _, _, m in r.messages()] == [i * 0.5 for i in range(50)]
    meta['rosbag2_bagfile_information']['compression_mode'] = 'FILE'
    (d / 'metadata.yaml').write_text(yaml.safe_dump(meta))
    with pytest.raises(ValueError, match='decompress it first'):
        rb.BagReader(str(d))


def test_register_rules():
    with pytest.raises(ValueError, match='different layout'):
        rb.register('geometry_msgs/msg/Point', 'float32 x\nfloat32 y\nfloat32 z\n')
    rb.register('geometry_msgs/msg/Point', '# same layout, other comments\nfloat64 x\nfloat64 y\nfloat64 z\n')
    with pytest.raises(ValueError, match='unknown message type'):
        rb.register('vantage_test_msgs/msg/Orphan', 'vantage_test_msgs/Missing m\n')
    with pytest.raises(ValueError):
        rb.register('vantage_test_msgs/msg/Bad', 'float64[x] nope\n')
    rb.register('vantage_test_msgs/msg/Wide', 'wstring w\n')
    with pytest.raises(ValueError, match='wstring'):
        rb.encode('vantage_test_msgs/msg/Wide', {})
