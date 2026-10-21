"""Small geometry helpers shared by the engine and the UI.

Conventions (ISO 8855): x forward, y left, z up. Camera body frame uses the same
axes with the optical axis along +x. Orientation is intrinsic Z-Y-X
(yaw, pitch, roll) in degrees, pitch positive *down* (right-hand rotation about +y).
"""
from __future__ import annotations

import math
import numpy as np

DEG = math.pi / 180.0

# body (x fwd, y left, z up) -> OpenCV optical (x right, y down, z forward)
BODY_TO_OPTICAL = np.array([[0.0, -1.0, 0.0],
                            [0.0, 0.0, -1.0],
                            [1.0, 0.0, 0.0]])


def wrap180(a: float) -> float:
    a = (a + 180.0) % 360.0 - 180.0
    return 180.0 if a == -180.0 else a


def rot_x(a):
    c, s = math.cos(a), math.sin(a)
    return np.array([[1, 0, 0], [0, c, -s], [0, s, c]], float)


def rot_y(a):
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]], float)


def rot_z(a):
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]], float)


def rot_zyx(yaw: float, pitch: float, roll: float) -> np.ndarray:
    """Body->world rotation for yaw/pitch/roll in degrees (pitch positive down)."""
    return rot_z(yaw * DEG) @ rot_y(pitch * DEG) @ rot_x(roll * DEG)


def euler_zyx(R: np.ndarray) -> tuple[float, float, float]:
    """Inverse of rot_zyx -> (yaw, pitch, roll) degrees."""
    sp = -R[2, 0]
    sp = max(-1.0, min(1.0, sp))
    pitch = math.asin(sp)
    if abs(sp) < 0.999999:
        yaw = math.atan2(R[1, 0], R[0, 0])
        roll = math.atan2(R[2, 1], R[2, 2])
    else:  # gimbal lock: put everything into yaw
        yaw = math.atan2(-R[0, 1], R[1, 1])
        roll = 0.0
    return wrap180(yaw / DEG), pitch / DEG, wrap180(roll / DEG)


def quat_from_matrix(R: np.ndarray) -> tuple[float, float, float, float]:
    """(x, y, z, w) quaternion of a rotation matrix."""
    m = R
    t = m[0, 0] + m[1, 1] + m[2, 2]
    if t > 0:
        s = math.sqrt(t + 1.0) * 2
        w = 0.25 * s
        x = (m[2, 1] - m[1, 2]) / s
        y = (m[0, 2] - m[2, 0]) / s
        z = (m[1, 0] - m[0, 1]) / s
    elif m[0, 0] > m[1, 1] and m[0, 0] > m[2, 2]:
        s = math.sqrt(1.0 + m[0, 0] - m[1, 1] - m[2, 2]) * 2
        w = (m[2, 1] - m[1, 2]) / s
        x = 0.25 * s
        y = (m[0, 1] + m[1, 0]) / s
        z = (m[0, 2] + m[2, 0]) / s
    elif m[1, 1] > m[2, 2]:
        s = math.sqrt(1.0 + m[1, 1] - m[0, 0] - m[2, 2]) * 2
        w = (m[0, 2] - m[2, 0]) / s
        x = (m[0, 1] + m[1, 0]) / s
        y = 0.25 * s
        z = (m[1, 2] + m[2, 1]) / s
    else:
        s = math.sqrt(1.0 + m[2, 2] - m[0, 0] - m[1, 1]) * 2
        w = (m[1, 0] - m[0, 1]) / s
        x = (m[0, 2] + m[2, 0]) / s
        y = (m[1, 2] + m[2, 1]) / s
        z = 0.25 * s
    n = math.sqrt(x * x + y * y + z * z + w * w) or 1.0
    return x / n, y / n, z / n, w / n


def rpy_from_matrix(R: np.ndarray) -> tuple[float, float, float]:
    """URDF fixed-axis roll/pitch/yaw (radians), R = Rz(y) Ry(p) Rx(r)."""
    yaw, pitch, roll = euler_zyx(R)
    # euler_zyx reports pitch positive down which is the same right-hand rotation as URDF pitch.
    return roll * DEG, pitch * DEG, yaw * DEG


def rodrigues(R: np.ndarray) -> np.ndarray:
    """Rotation matrix -> axis*angle vector."""
    c = (np.trace(R) - 1.0) / 2.0
    c = max(-1.0, min(1.0, c))
    ang = math.acos(c)
    if ang < 1e-9:
        return np.zeros(3)
    if abs(ang - math.pi) < 1e-6:
        # 180 degrees: axis from the diagonal
        ax = np.sqrt(np.maximum((np.diag(R) + 1.0) / 2.0, 0))
        ax[1] = math.copysign(ax[1], R[0, 1] if ax[0] > 0 else 1)
        ax[2] = math.copysign(ax[2], R[0, 2] if ax[0] > 0 else R[1, 2])
        return ax / (np.linalg.norm(ax) or 1) * ang
    v = np.array([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]]) / (2 * math.sin(ang))
    return v * ang


def normalize(v: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(v, axis=-1, keepdims=True)
    n = np.where(n < 1e-15, 1.0, n)
    return v / n


def look_rotation(forward: np.ndarray, up_hint=(0, 0, 1)) -> np.ndarray:
    """Body->world rotation whose +x points along forward, keeping roll level."""
    f = normalize(np.asarray(forward, float))
    up = np.asarray(up_hint, float)
    left = np.cross(up, f)
    if np.linalg.norm(left) < 1e-9:
        left = np.cross(np.array([1.0, 0, 0]), f)
    left = normalize(left)
    z = np.cross(f, left)
    return np.stack([f, left, z], axis=1)


def yaw_pitch_from_dir(d) -> tuple[float, float]:
    d = np.asarray(d, float)
    yaw = math.degrees(math.atan2(d[1], d[0]))
    pitch = math.degrees(math.atan2(-d[2], math.hypot(d[0], d[1])))
    return wrap180(yaw), pitch


def fmt_num(v: float, dp: int = 3) -> str:
    """Fixed precision with a typographic minus and no '-0.000'."""
    if v is None or (isinstance(v, float) and math.isnan(v)):
        return '—'
    t = 0.5 * 10 ** (-dp)
    if abs(v) < t:
        v = 0.0
    s = f'{v:.{dp}f}'
    return s.replace('-', '−')


def parse_number(text: str, unit_kind: str = 'length', default_unit: str = 'm') -> float:
    """Parse '1.88+0.05', '950 mm', '65deg', '−0.8' into a float in base units
    (metres for lengths, degrees for angles). Raises ValueError."""
    import ast
    import operator as op
    import re

    s = text.strip().replace('−', '-').replace('×', '*').replace(',', '.')
    if not s:
        raise ValueError('empty')
    scale = {
        'length': {'m': 1.0, 'mm': 1e-3, 'cm': 1e-2, 'in': 0.0254, 'km': 1e3},
        'angle': {'deg': 1.0, '°': 1.0, 'rad': 180.0 / math.pi},
        'plain': {},
    }[unit_kind]
    base = {'length': default_unit, 'angle': 'deg', 'plain': None}[unit_kind]

    # attach units to each number token: "950 mm" -> "(950*0.001)"
    def repl(mo):
        num, unit = mo.group(1), mo.group(2)
        unit = unit.strip().lower() if unit else ''
        if unit in ('', None):
            f = scale.get(base, 1.0) if base else 1.0
        elif unit in scale:
            f = scale[unit]
        else:
            raise ValueError(f'unknown unit {unit}')
        return f'({num}*{f!r})'

    expr = re.sub(r'(\d+(?:\.\d*)?(?:[eE][-+]?\d+)?|\.\d+)\s*(mm|cm|km|m|in|deg|rad|°)?', repl, s)
    ops = {ast.Add: op.add, ast.Sub: op.sub, ast.Mult: op.mul, ast.Div: op.truediv,
           ast.USub: op.neg, ast.UAdd: op.pos}

    def ev(node):
        if isinstance(node, ast.Expression):
            return ev(node.body)
        if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
            return float(node.value)
        if isinstance(node, ast.BinOp) and type(node.op) in ops:
            return ops[type(node.op)](ev(node.left), ev(node.right))
        if isinstance(node, ast.UnaryOp) and type(node.op) in ops:
            return ops[type(node.op)](ev(node.operand))
        raise ValueError('bad expression')

    try:
        tree = ast.parse(expr, mode='eval')
    except (SyntaxError, RecursionError, MemoryError) as e:
        raise ValueError(f'cannot read {text!r}') from e
    if len(expr) > 200:
        raise ValueError('expression too long')
    val = ev(tree)
    # result is in metres (lengths) or degrees (angles); convert back to the display unit
    if unit_kind == 'length' and base != 'm':
        val = val / scale[base]
    if not math.isfinite(val):
        raise ValueError('not finite')
    return val
