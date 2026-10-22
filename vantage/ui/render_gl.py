"""Shader renderer for the 3D viewport (OpenGL 2.1 / GLSL 1.20 and up, compatibility profile).

Pure PyOpenGL against whatever context is current, so the same code draws inside the Qt widget, into an off-screen
framebuffer for high-resolution renders, and in a headless EGL context for the tests.

What it adds over the fixed-function path: geometry kept on the GPU (uploaded once per mesh), smooth vertex normals
with crease angles, hemisphere + key + rim lighting with a little specular, a shadow map for the vehicle on the ground
and on itself, a contact-shadow (ambient occlusion) term under the body, a ground plane that fades into the background,
the textured 3D surround-view bowl, and "camera light": every camera's real lens model and body occlusion evaluated per
pixel, so the body and the ground show exactly what each camera sees while it is being moved.

If anything here fails to compile or link the viewport falls back to the fixed-function path.
"""
from __future__ import annotations

import ctypes
import math
from dataclasses import dataclass, field

import numpy as np

PROJ_ID = {'rectilinear': 0, 'equidistant': 1, 'equisolid': 2, 'stereographic': 3, 'orthographic': 4,
           'kannala_brandt': 5}
MAX_CAMS = 8
SHADOW_N = 2048
AO_N = 512
LIGHT_DIR = np.array([0.35, 0.55, 0.76]) / np.linalg.norm([0.35, 0.55, 0.76])


# ---------------------------------------------------------------------------- inputs
@dataclass
class BowlDraw:
    vertices: np.ndarray         # (V,3) float32
    uv: np.ndarray               # (V,2) float32: u azimuth 0..1 (wraps), v texture row 0..1
    faces: np.ndarray            # (F,3) int32
    texture: np.ndarray          # (H,W,4) uint8 RGBA, row 0 = v 0
    version: int = 0


@dataclass
class Style:
    shadows: bool = True
    ao: bool = True
    ground_plane: bool = True
    ground_rgb: tuple = (0.22, 0.22, 0.23)
    sky_rgb: tuple = (0.62, 0.66, 0.72)
    bounce_rgb: tuple = (0.24, 0.23, 0.22)
    fade: tuple = (18.0, 45.0)   # ground plane fades out between these distances from the view target, m
    camlight_mix: float = 0.6


@dataclass
class Frame:
    V: np.ndarray
    P: np.ndarray
    width: int
    height: int
    target: np.ndarray = field(default_factory=lambda: np.zeros(3))
    mesh: object = None          # viewport3d.MeshDraw
    ground: object = None        # viewport3d.GroundDraw
    lines: list = field(default_factory=list)
    tris: list = field(default_factory=list)
    bowl: BowlDraw | None = None
    cams: list = field(default_factory=list)       # [core.visibility.CamView] for the camera light
    style: Style = field(default_factory=Style)
    clear: tuple | None = None   # RGBA to clear to (off-screen); None keeps what is there (the widget's background)
    ortho: bool = False


# ---------------------------------------------------------------------------- shaders
_COMMON = '''
uniform vec3 uLight;
uniform vec3 uSky;
uniform vec3 uBounce;
uniform sampler2D uShadow;
uniform mat4 uShadowVP;
uniform float uShadowOn;
uniform float uShadowTexel;

float shadow_at(vec3 p, float bias) {
    if (uShadowOn < 0.5) return 1.0;
    vec4 q = uShadowVP * vec4(p, 1.0);
    vec3 c = q.xyz / q.w * 0.5 + 0.5;
    if (c.x < 0.0 || c.x > 1.0 || c.y < 0.0 || c.y > 1.0 || c.z > 1.0) return 1.0;
    float lit = 0.0;
    for (int i = -1; i <= 1; i++) {
        for (int j = -1; j <= 1; j++) {
            float d = texture2D(uShadow, c.xy + vec2(float(i), float(j)) * uShadowTexel).r;
            lit += (c.z - bias > d) ? 0.0 : 1.0;
        }
    }
    return lit / 9.0;
}

uniform int uNCam;
uniform mat3 uCamR[8];
uniform vec3 uCamP[8];
uniform vec4 uCamK[8];
uniform vec4 uCamD[8];
uniform vec4 uCamS[8];
uniform vec3 uCamC[8];
uniform sampler2D uCube;
uniform vec4 uCubeInfo;      // N, atlas width, atlas height, on
uniform float uCamMix;

float r_of(float th, float proj, vec4 k) {
    if (proj < 0.5) return tan(min(th, 1.5690));
    if (proj < 1.5) return th;
    if (proj < 2.5) return 2.0 * sin(th * 0.5);
    if (proj < 3.5) return 2.0 * tan(min(th, 3.1241) * 0.5);
    if (proj < 4.5) return sin(min(th, 1.5707963));
    float t2 = th * th;
    return th * (1.0 + t2 * (k.x + t2 * (k.y + t2 * (k.z + t2 * k.w))));
}

float cube_depth(int cam, vec3 d) {
    vec3 a = abs(d);
    float face; float u; float v;
    if (a.x >= a.y && a.x >= a.z) {
        if (d.x >= 0.0) { face = 0.0; u = -d.y / d.x; v = d.z / d.x; }
        else            { face = 1.0; u = d.y / -d.x; v = d.z / -d.x; }
    } else if (a.y >= a.z) {
        if (d.y >= 0.0) { face = 2.0; u = d.x / d.y; v = d.z / d.y; }
        else            { face = 3.0; u = -d.x / -d.y; v = d.z / -d.y; }
    } else {
        if (d.z >= 0.0) { face = 4.0; u = d.x / d.z; v = -d.y / d.z; }
        else            { face = 5.0; u = d.x / -d.z; v = d.y / -d.z; }
    }
    float N = uCubeInfo.x;
    float px = clamp(floor((u + 1.0) * 0.5 * N), 0.0, N - 1.0);
    float py = clamp(floor((1.0 - v) * 0.5 * N), 0.0, N - 1.0);
    vec2 tc = vec2((face * N + px + 0.5) / uCubeInfo.y, (float(cam) * N + py + 0.5) / uCubeInfo.z);
    vec4 t = texture2D(uCube, tc);
    if (t.b < 0.5) return -1.0;
    return (t.r * 255.0 * 256.0 + t.g * 255.0) / 1000.0;
}

bool cam_sees(int i, vec3 p) {
    vec3 dw = p - uCamP[i];
    vec3 d = uCamR[i] * dw;
    float rho = length(d.xy);
    float th = atan(rho, d.z);
    if (th > uCamS[i].z) return false;
    if (uCamS[i].w < 0.5 && d.z <= 1e-6) return false;
    float r = uCamK[i].x * r_of(th, uCamS[i].w, uCamD[i]);
    vec2 dir = rho > 1e-9 ? d.xy / rho : vec2(0.0);
    float u = uCamK[i].y + dir.x * r;
    float v = uCamK[i].z + dir.y * r * uCamK[i].w;
    if (u < 0.0 || u > uCamS[i].x || v < 0.0 || v > uCamS[i].y) return false;
    if (uCubeInfo.w > 0.5) {
        float dist = length(dw);
        float occ = cube_depth(i, dw);
        if (occ > 0.0 && occ < dist - (0.02 + 0.01 * dist)) return false;
    }
    return true;
}

vec3 camlight(vec3 base, vec3 p) {
    if (uNCam <= 0) return base;
    vec3 sum = vec3(0.0);
    float n = 0.0;
    for (int i = 0; i < 8; i++) {
        if (i >= uNCam) break;
        if (cam_sees(i, p)) { sum += uCamC[i]; n += 1.0; }
    }
    float lum = dot(base, vec3(0.299, 0.587, 0.114));
    if (n < 0.5) return mix(base, vec3(lum) * 0.55, 0.6);
    vec3 c = mix(base, (sum / n) * (0.30 + 1.05 * lum), uCamMix);
    if (n > 1.5) {
        float h = step(0.5, fract((p.x + p.y + p.z) * 6.0));
        c *= 0.90 + 0.10 * h;
    }
    return c;
}
'''

MESH_VS = '''
#version 120
attribute vec3 aPos;
attribute vec3 aNrm;
attribute vec4 aCol;
uniform mat4 uVP;
varying vec3 vPos;
varying vec3 vNrm;
varying vec4 vCol;
void main() {
    vPos = aPos; vNrm = aNrm; vCol = aCol;
    gl_Position = uVP * vec4(aPos, 1.0);
}
'''

MESH_FS = '''
#version 120
varying vec3 vPos;
varying vec3 vNrm;
varying vec4 vCol;
uniform vec3 uEye;
uniform float uSpec;
''' + _COMMON + '''
void main() {
    vec3 N = normalize(vNrm);
    vec3 V = normalize(uEye - vPos);
    if (dot(N, V) < 0.0) N = -N;                       // two-sided: open meshes and inward faces
    float ndl = max(dot(N, uLight), 0.0);
    float sh = shadow_at(vPos + N * 0.004, 0.0015);
    vec3 hemi = mix(uBounce, uSky, N.z * 0.5 + 0.5);
    vec3 H = normalize(uLight + V);
    float spec = pow(max(dot(N, H), 0.0), 56.0) * uSpec;
    float fres = pow(1.0 - max(dot(N, V), 0.0), 4.0);
    vec3 c = vCol.rgb * (hemi * 0.62 + ndl * sh * 0.70) + vec3(spec * sh * 0.32) + uSky * fres * 0.12;
    c = camlight(c, vPos + N * 0.03);                  // off the surface: a face does not occlude itself
    gl_FragColor = vec4(c, vCol.a);
}
'''

GROUND_VS = '''
#version 120
attribute vec3 aPos;
uniform mat4 uVP;
varying vec3 vPos;
void main() { vPos = aPos; gl_Position = uVP * vec4(aPos, 1.0); }
'''

GROUND_FS = '''
#version 120
varying vec3 vPos;
uniform vec3 uBase;
uniform sampler2D uTex;
uniform float uTexOn;
uniform vec4 uExt;           // x0, y0, x1, y1
uniform sampler2D uAO;
uniform mat4 uAOVP;
uniform vec4 uAOInfo;        // on, texel, near, far
uniform vec3 uTarget;
uniform vec2 uFade;
uniform float uPlane;        // 1 = draw the plane colour, 0 = only the texture
''' + _COMMON + '''
float under(vec2 xy) {
    vec4 q = uAOVP * vec4(xy, 0.0, 1.0);
    vec2 c = q.xy / q.w * 0.5 + 0.5;
    if (c.x < 0.0 || c.x > 1.0 || c.y < 0.0 || c.y > 1.0) return 9.0;
    float d = texture2D(uAO, c).r;
    return uAOInfo.z + d * (uAOInfo.w - uAOInfo.z) - 1.0;    // height of the body's underside above the ground
}
float ao_at(vec2 xy) {
    if (uAOInfo.x < 0.5) return 1.0;
    float occ = 0.0;
    float wsum = 0.0;
    for (int i = 0; i < 16; i++) {
        float a = float(i) * 2.39996;
        float r = 0.07 + 0.55 * sqrt((float(i) + 0.5) / 16.0);
        vec2 o = vec2(cos(a), sin(a)) * r;
        float h = under(xy + o);
        float w = 1.0 - r / 0.75;
        occ += w * (1.0 - smoothstep(0.02, 0.85, h));
        wsum += w;
    }
    return 1.0 - 0.72 * occ / wsum;
}
void main() {
    vec4 col = vec4(uBase, uPlane);
    if (uTexOn > 0.5) {
        vec2 tc = vec2((vPos.x - uExt.x) / (uExt.z - uExt.x), (uExt.w - vPos.y) / (uExt.w - uExt.y));
        if (tc.x >= 0.0 && tc.x <= 1.0 && tc.y >= 0.0 && tc.y <= 1.0) {
            vec4 t = texture2D(uTex, tc);
            col.rgb = mix(col.rgb, t.rgb, t.a);
            col.a = max(col.a, t.a);
        }
    }
    float sh = shadow_at(vPos, 0.0008);
    float ao = ao_at(vPos.xy);
    col.rgb *= (0.55 + 0.45 * sh) * ao;
    col.a = max(col.a, (1.0 - min(sh, ao)) * 0.85);
    col.rgb = camlight(col.rgb, vPos);
    float dist = length(vPos.xy - uTarget.xy);
    col.a *= 1.0 - smoothstep(uFade.x, uFade.y, dist);
    gl_FragColor = col;
}
'''

FLAT_VS = '''
#version 120
attribute vec3 aPos;
attribute vec4 aCol;
uniform mat4 uVP;
varying vec3 vPos;
varying vec4 vCol;
void main() { vPos = aPos; vCol = aCol; gl_Position = uVP * vec4(aPos, 1.0); }
'''

FLAT_FS = '''
#version 120
varying vec3 vPos;
varying vec4 vCol;
uniform vec3 uEye;
uniform vec3 uLight;
uniform vec3 uSky;
uniform vec3 uBounce;
void main() {
    vec3 N = normalize(cross(dFdx(vPos), dFdy(vPos)));
    if (dot(N, uEye - vPos) < 0.0) N = -N;
    float ndl = max(dot(N, uLight), 0.0);
    vec3 hemi = mix(uBounce, uSky, N.z * 0.5 + 0.5);
    gl_FragColor = vec4(vCol.rgb * (hemi * 0.75 + ndl * 0.55), vCol.a);
}
'''

BOWL_VS = '''
#version 120
attribute vec3 aPos;
attribute vec2 aUV;
uniform mat4 uVP;
varying vec2 vUV;
varying vec3 vPos;
void main() { vUV = aUV; vPos = aPos; gl_Position = uVP * vec4(aPos, 1.0); }
'''

BOWL_FS = '''
#version 120
varying vec2 vUV;
varying vec3 vPos;
uniform sampler2D uTex;
void main() {
    vec4 t = texture2D(uTex, vUV);
    gl_FragColor = vec4(t.rgb, t.a);
}
'''

DEPTH_VS = '''
#version 120
attribute vec3 aPos;
uniform mat4 uVP;
void main() { gl_Position = uVP * vec4(aPos, 1.0); }
'''

DEPTH_FS = '''
#version 120
void main() { gl_FragColor = vec4(1.0); }
'''


# ---------------------------------------------------------------------------- helpers
def vertex_normals(tri: np.ndarray, face_n: np.ndarray, crease_deg=40.0) -> np.ndarray:
    """Per-corner normals (M,3,3): area-weighted average of the faces sharing a vertex position, except across
    edges sharper than the crease angle, where the face normal is kept (so panel edges stay crisp)."""
    M = len(tri)
    if M == 0:
        return np.zeros((0, 3, 3), np.float32)
    P = tri.reshape(-1, 3).astype(np.float64)
    fn = np.asarray(face_n, np.float64)
    fn = fn / np.maximum(np.linalg.norm(fn, axis=1, keepdims=True), 1e-12)
    area = 0.5 * np.linalg.norm(np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0]), axis=1)
    key = np.round(P / 1e-5).astype(np.int64)
    _, inv = np.unique(key, axis=0, return_inverse=True)
    inv = inv.ravel()
    acc = np.zeros((inv.max() + 1, 3))
    np.add.at(acc, inv, np.repeat(fn * area[:, None], 3, axis=0))
    vn = acc[inv]
    vn /= np.maximum(np.linalg.norm(vn, axis=1, keepdims=True), 1e-12)
    corner_fn = np.repeat(fn, 3, axis=0)
    keep = np.sum(vn * corner_fn, axis=1) >= math.cos(math.radians(crease_deg))
    out = np.where(keep[:, None], vn, corner_fn)
    return out.reshape(M, 3, 3).astype(np.float32)


def _look_at(eye, target, up):
    f = target - eye
    f = f / np.linalg.norm(f)
    s = np.cross(f, up)
    if np.linalg.norm(s) < 1e-6:
        s = np.cross(f, np.array([1.0, 0, 0]))
    s = s / np.linalg.norm(s)
    u = np.cross(s, f)
    M = np.eye(4)
    M[0, :3], M[1, :3], M[2, :3] = s, u, -f
    M[:3, 3] = -M[:3, :3] @ eye
    return M


def _ortho(l, r, b, t, n, f):
    P = np.eye(4)
    P[0, 0], P[1, 1], P[2, 2] = 2 / (r - l), 2 / (t - b), -2 / (f - n)
    P[0, 3], P[1, 3], P[2, 3] = -(r + l) / (r - l), -(t + b) / (t - b), -(f + n) / (f - n)
    return P


def shadow_matrix(lo, hi, light=LIGHT_DIR):
    """Orthographic light view-projection covering a box (and the ground behind it along the light)."""
    c = (lo + hi) / 2
    rad = float(np.linalg.norm(hi - lo)) / 2 + 0.2
    eye = c + light * (rad + 2.0)
    V = _look_at(eye, c, np.array([0, 0, 1.0]))
    corners = np.array([[x, y, z] for x in (lo[0], hi[0]) for y in (lo[1], hi[1]) for z in (lo[2], hi[2])])
    q = corners @ V[:3, :3].T + V[:3, 3]
    pad = 0.15
    P = _ortho(q[:, 0].min() - pad, q[:, 0].max() + pad, q[:, 1].min() - pad, q[:, 1].max() + pad,
               max(0.01, -q[:, 2].max() - 0.5), -q[:, 2].min() + 25.0)
    return P @ V


def ao_matrix(lo, hi, margin=0.9, near=0.5, far=4.0):
    """Orthographic camera below the ground looking up: its depth is the height of the body's underside."""
    c = (lo + hi) / 2
    eye = np.array([c[0], c[1], -1.0])
    V = _look_at(eye, np.array([c[0], c[1], 0.0]), np.array([1.0, 0, 0]))
    hx, hy = (hi[0] - lo[0]) / 2 + margin, (hi[1] - lo[1]) / 2 + margin
    q = np.array([[lo[0] - margin, lo[1] - margin, 0], [hi[0] + margin, hi[1] + margin, 0]]) @ V[:3, :3].T + V[:3, 3]
    P = _ortho(q[:, 0].min(), q[:, 0].max(), q[:, 1].min(), q[:, 1].max(), near, far)
    return P @ V, (near, far)


def encode_cubes(cubes: list) -> tuple[np.ndarray, int]:
    """Occlusion cube maps of several cameras as one RGBA8 atlas (faces across, cameras down), depth in mm in R/G
    (16 bit), B = occluder present."""
    N = cubes[0].shape[1]
    H = N * len(cubes)
    out = np.zeros((H, 6 * N, 4), np.uint8)
    for i, cb in enumerate(cubes):
        for f in range(6):
            d = cb[f]
            ok = np.isfinite(d)
            mm = np.where(ok, np.clip(np.round(d * 1000), 0, 65535), 0).astype(np.uint32)
            blk = out[i * N:(i + 1) * N, f * N:(f + 1) * N]
            blk[..., 0] = mm >> 8
            blk[..., 1] = mm & 255
            blk[..., 2] = np.where(ok, 255, 0)
            blk[..., 3] = 255
    return out, N


# ---------------------------------------------------------------------------- renderer
class GLRenderer:
    def __init__(self):
        self.ok = False
        self.error = ''
        self.GL = None
        self.prog = {}
        self._mesh = None            # (MeshDraw ref, vbo, count)
        self._ground_tex = None      # (key, tex)
        self._bowl = None            # (BowlDraw ref, vbo, ibo, tex, count)
        self._shadow = None          # dict(fbo, tex, key, VP)
        self._ao = None
        self._cube = None            # (key, tex, N, W, H)
        self._plane = None
        self.fbo_ok = False
        self.stats = {}

    # ---- setup
    def init(self):
        from OpenGL import GL
        self.GL = GL
        try:
            self.prog['mesh'] = self._program(MESH_VS, MESH_FS)
            self.prog['ground'] = self._program(GROUND_VS, GROUND_FS)
            self.prog['flat'] = self._program(FLAT_VS, FLAT_FS)
            self.prog['bowl'] = self._program(BOWL_VS, BOWL_FS)
            self.prog['depth'] = self._program(DEPTH_VS, DEPTH_FS)
            self.fbo_ok = bool(GL.glGenFramebuffers) and bool(GL.glBindFramebuffer)
            self.max_tex = int(GL.glGetIntegerv(GL.GL_MAX_TEXTURE_SIZE))
            self.ok = True
        except Exception as e:      # no shaders on this driver: the caller keeps the fixed-function path
            self.ok = False
            self.error = str(e)
        return self.ok

    def _program(self, vs, fs):
        GL = self.GL

        def sh(kind, src):
            s = GL.glCreateShader(kind)
            GL.glShaderSource(s, src)
            GL.glCompileShader(s)
            if not GL.glGetShaderiv(s, GL.GL_COMPILE_STATUS):
                raise RuntimeError(GL.glGetShaderInfoLog(s).decode(errors='replace'))
            return s
        p = GL.glCreateProgram()
        GL.glAttachShader(p, sh(GL.GL_VERTEX_SHADER, vs))
        GL.glAttachShader(p, sh(GL.GL_FRAGMENT_SHADER, fs))
        for i, name in enumerate(('aPos', 'aNrm', 'aCol', 'aUV')):
            GL.glBindAttribLocation(p, i, name)
        GL.glLinkProgram(p)
        if not GL.glGetProgramiv(p, GL.GL_LINK_STATUS):
            raise RuntimeError(GL.glGetProgramInfoLog(p).decode(errors='replace'))
        locs = {}
        n = GL.glGetProgramiv(p, GL.GL_ACTIVE_UNIFORMS)
        for i in range(n):
            name, size, typ = GL.glGetActiveUniform(p, i)
            name = name.decode() if isinstance(name, bytes) else name
            base = name.split('[')[0]
            locs[base] = GL.glGetUniformLocation(p, base if size == 1 else base + '[0]')
        return {'id': p, 'u': locs}

    # ---- uniforms
    def _u(self, prog, name):
        return prog['u'].get(name, -1)

    def _set(self, prog, name, val):
        GL = self.GL
        loc = self._u(prog, name)
        if loc is None or loc < 0:
            return
        v = np.asarray(val, np.float32)
        if v.shape == (4, 4):
            GL.glUniformMatrix4fv(loc, 1, GL.GL_TRUE, v)
        elif v.ndim == 0:
            GL.glUniform1f(loc, float(v))
        elif v.shape == (2,):
            GL.glUniform2f(loc, *v)
        elif v.shape == (3,):
            GL.glUniform3f(loc, *v)
        elif v.shape == (4,):
            GL.glUniform4f(loc, *v)

    def _common(self, prog, f: Frame):
        GL = self.GL
        st = f.style
        self._set(prog, 'uLight', LIGHT_DIR)
        self._set(prog, 'uSky', st.sky_rgb)
        self._set(prog, 'uBounce', st.bounce_rgb)
        sh = self._shadow if (st.shadows and self._shadow) else None
        self._set(prog, 'uShadowOn', 1.0 if sh else 0.0)
        if sh:
            GL.glActiveTexture(GL.GL_TEXTURE1)
            GL.glBindTexture(GL.GL_TEXTURE_2D, sh['tex'])
            GL.glUniform1i(self._u(prog, 'uShadow'), 1)
            self._set(prog, 'uShadowVP', sh['VP'])
            self._set(prog, 'uShadowTexel', 1.0 / SHADOW_N)
        # camera light
        n = min(len(f.cams), MAX_CAMS)
        loc = self._u(prog, 'uNCam')
        if loc is not None and loc >= 0:
            GL.glUniform1i(loc, n)
        if n:
            R = np.zeros((MAX_CAMS, 3, 3), np.float32)
            Pp = np.zeros((MAX_CAMS, 3), np.float32)
            K = np.zeros((MAX_CAMS, 4), np.float32)
            D = np.zeros((MAX_CAMS, 4), np.float32)
            S = np.zeros((MAX_CAMS, 4), np.float32)
            C = np.zeros((MAX_CAMS, 3), np.float32)
            for i, c in enumerate(f.cams[:n]):
                it = c.intr
                R[i] = c.W2O
                Pp[i] = c.pos
                K[i] = (it.f_px, it.cx, it.cy, it.aspect)
                D[i] = tuple(it.kb)[:4]
                S[i] = (it.width, it.height, it.max_theta, PROJ_ID.get(it.projection, 5))
                C[i] = c.color
            GL.glUniformMatrix3fv(self._u(prog, 'uCamR'), MAX_CAMS, GL.GL_TRUE, R)
            GL.glUniform3fv(self._u(prog, 'uCamP'), MAX_CAMS, Pp)
            GL.glUniform4fv(self._u(prog, 'uCamK'), MAX_CAMS, K)
            GL.glUniform4fv(self._u(prog, 'uCamD'), MAX_CAMS, D)
            GL.glUniform4fv(self._u(prog, 'uCamS'), MAX_CAMS, S)
            GL.glUniform3fv(self._u(prog, 'uCamC'), MAX_CAMS, C)
            self._set(prog, 'uCamMix', f.style.camlight_mix)
            cube = self._cube_tex(f.cams[:n])
            if cube is not None:
                GL.glActiveTexture(GL.GL_TEXTURE3)
                GL.glBindTexture(GL.GL_TEXTURE_2D, cube[1])
                GL.glUniform1i(self._u(prog, 'uCube'), 3)
                self._set(prog, 'uCubeInfo', (cube[2], cube[3], cube[4], 1.0))
            else:
                self._set(prog, 'uCubeInfo', (1, 1, 1, 0.0))
        GL.glActiveTexture(GL.GL_TEXTURE0)

    # ---- resources
    def _texture(self, data: np.ndarray, nearest=False, wrap_s=None, tex=None):
        GL = self.GL
        tex = tex if tex is not None else GL.glGenTextures(1)
        GL.glBindTexture(GL.GL_TEXTURE_2D, tex)
        GL.glPixelStorei(GL.GL_UNPACK_ALIGNMENT, 1)
        h, w = data.shape[:2]
        GL.glTexImage2D(GL.GL_TEXTURE_2D, 0, GL.GL_RGBA, w, h, 0, GL.GL_RGBA, GL.GL_UNSIGNED_BYTE,
                        np.ascontiguousarray(data, np.uint8))
        flt = GL.GL_NEAREST if nearest else GL.GL_LINEAR
        GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_MIN_FILTER, flt)
        GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_MAG_FILTER, flt)
        GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_WRAP_S, wrap_s or GL.GL_CLAMP_TO_EDGE)
        GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_WRAP_T, GL.GL_CLAMP_TO_EDGE)
        GL.glBindTexture(GL.GL_TEXTURE_2D, 0)
        return tex

    def _mesh_vbo(self, md):
        GL = self.GL
        if self._mesh is not None and self._mesh[0] is md:
            return self._mesh
        if self._mesh is not None:
            GL.glDeleteBuffers(1, [self._mesh[1]])
        n = len(md.tri)
        nrm = vertex_normals(md.tri, md.normals)
        col = np.c_[md.base, md.alpha].astype(np.float32)
        buf = np.empty((n, 3, 10), np.float32)
        buf[..., 0:3] = md.tri
        buf[..., 3:6] = nrm
        buf[..., 6:10] = col[:, None, :]
        opaque = md.alpha > 0.99
        order = np.r_[np.nonzero(opaque)[0], np.nonzero(~opaque)[0]]      # opaque first, then translucent
        buf = np.ascontiguousarray(buf[order])
        vbo = GL.glGenBuffers(1)
        GL.glBindBuffer(GL.GL_ARRAY_BUFFER, vbo)
        GL.glBufferData(GL.GL_ARRAY_BUFFER, buf.nbytes, buf, GL.GL_STATIC_DRAW)
        GL.glBindBuffer(GL.GL_ARRAY_BUFFER, 0)
        lo = md.tri.reshape(-1, 3).min(axis=0)
        hi = md.tri.reshape(-1, 3).max(axis=0)
        self._mesh = (md, vbo, int(opaque.sum()) * 3, n * 3, lo, hi)
        self._shadow_key = None
        return self._mesh

    def _cube_tex(self, cams):
        GL = self.GL
        if any(c.cube is None for c in cams):
            return None
        key = tuple(id(c.cube) for c in cams)
        if self._cube is not None and self._cube[0] == key:
            return self._cube
        atlas, N = encode_cubes([c.cube.depth for c in cams])
        while (atlas.shape[0] > self.max_tex or atlas.shape[1] > self.max_tex) and N > 32:
            atlas = atlas[::2, ::2]
            N //= 2
        tex = self._texture(atlas, nearest=True, tex=self._cube[1] if self._cube else None)
        self._cube = (key, tex, float(N), float(atlas.shape[1]), float(atlas.shape[0]), [c.cube for c in cams])   # keeps ids valid
        return self._cube

    def _depth_fbo(self, n):
        GL = self.GL
        tex = GL.glGenTextures(1)
        GL.glBindTexture(GL.GL_TEXTURE_2D, tex)
        GL.glTexImage2D(GL.GL_TEXTURE_2D, 0, GL.GL_DEPTH_COMPONENT24, n, n, 0, GL.GL_DEPTH_COMPONENT, GL.GL_FLOAT, None)
        for p, v in ((GL.GL_TEXTURE_MIN_FILTER, GL.GL_NEAREST), (GL.GL_TEXTURE_MAG_FILTER, GL.GL_NEAREST),
                     (GL.GL_TEXTURE_WRAP_S, GL.GL_CLAMP_TO_EDGE), (GL.GL_TEXTURE_WRAP_T, GL.GL_CLAMP_TO_EDGE)):
            GL.glTexParameteri(GL.GL_TEXTURE_2D, p, v)
        GL.glBindTexture(GL.GL_TEXTURE_2D, 0)
        fbo = GL.glGenFramebuffers(1)
        GL.glBindFramebuffer(GL.GL_FRAMEBUFFER, fbo)
        GL.glFramebufferTexture2D(GL.GL_FRAMEBUFFER, GL.GL_DEPTH_ATTACHMENT, GL.GL_TEXTURE_2D, tex, 0)
        GL.glDrawBuffer(GL.GL_NONE)
        GL.glReadBuffer(GL.GL_NONE)
        ok = GL.glCheckFramebufferStatus(GL.GL_FRAMEBUFFER) == GL.GL_FRAMEBUFFER_COMPLETE
        return {'fbo': fbo, 'tex': tex, 'ok': ok}

    def _depth_pass(self, slot, n, VP, mesh, target_fbo):
        GL = self.GL
        d = getattr(self, slot)
        if d is None:
            d = self._depth_fbo(n)
            setattr(self, slot, d)
        if not d['ok']:
            return False
        GL.glBindFramebuffer(GL.GL_FRAMEBUFFER, d['fbo'])
        GL.glViewport(0, 0, n, n)
        GL.glClearDepth(1.0)
        GL.glClear(GL.GL_DEPTH_BUFFER_BIT)
        GL.glEnable(GL.GL_DEPTH_TEST)
        GL.glDepthMask(GL.GL_TRUE)
        GL.glDisable(GL.GL_BLEND)
        GL.glColorMask(GL.GL_FALSE, GL.GL_FALSE, GL.GL_FALSE, GL.GL_FALSE)
        p = self.prog['depth']
        GL.glUseProgram(p['id'])
        self._set(p, 'uVP', VP)
        md, vbo, n_opaque, n_all = mesh[:4]
        GL.glBindBuffer(GL.GL_ARRAY_BUFFER, vbo)
        GL.glEnableVertexAttribArray(0)
        GL.glVertexAttribPointer(0, 3, GL.GL_FLOAT, GL.GL_FALSE, 40, None)
        GL.glDrawArrays(GL.GL_TRIANGLES, 0, n_opaque)
        GL.glDisableVertexAttribArray(0)
        GL.glBindBuffer(GL.GL_ARRAY_BUFFER, 0)
        GL.glColorMask(GL.GL_TRUE, GL.GL_TRUE, GL.GL_TRUE, GL.GL_TRUE)
        GL.glBindFramebuffer(GL.GL_FRAMEBUFFER, target_fbo)
        d['VP'] = VP
        return True

    def _update_shadows(self, f: Frame, mesh, target_fbo):
        if not self.fbo_ok or mesh is None:
            return
        key = id(mesh[0])
        if getattr(self, '_shadow_key', None) == key:
            return
        lo, hi = mesh[4], mesh[5]
        lo = np.minimum(lo, [lo[0], lo[1], 0.0])
        if not self._depth_pass('_shadow', SHADOW_N, shadow_matrix(lo, hi), mesh, target_fbo):
            self._shadow = None
        VP, nf = ao_matrix(lo, hi)
        if not self._depth_pass('_ao', AO_N, VP, mesh, target_fbo):
            self._ao = None
        else:
            self._ao['nf'] = nf
        self._shadow_key = key

    # ---- draw
    def _no_vao(self):
        # QPainter's GL engine may leave its vertex array object bound; client-side arrays need the default one
        if getattr(self, '_vao_ok', None) is None:
            self._vao_ok = bool(getattr(self.GL, 'glBindVertexArray', None))
        if self._vao_ok:
            try:
                self.GL.glBindVertexArray(0)
            except Exception:
                self._vao_ok = False

    def render(self, f: Frame, target_fbo=0):
        GL = self.GL
        self._no_vao()
        mesh = self._mesh_vbo(f.mesh) if f.mesh is not None and len(f.mesh.tri) else None
        if mesh is not None and (f.style.shadows or f.style.ao):
            self._update_shadows(f, mesh, target_fbo)
        GL.glBindFramebuffer(GL.GL_FRAMEBUFFER, target_fbo)
        GL.glViewport(0, 0, int(f.width), int(f.height))
        if f.clear is not None:
            GL.glClearColor(*f.clear)
            GL.glClear(GL.GL_COLOR_BUFFER_BIT | GL.GL_DEPTH_BUFFER_BIT)
        else:
            GL.glClear(GL.GL_DEPTH_BUFFER_BIT)
        VP = (f.P @ f.V).astype(np.float32)
        eye = np.linalg.inv(f.V)[:3, 3]
        GL.glEnable(GL.GL_BLEND)
        GL.glBlendFuncSeparate(GL.GL_SRC_ALPHA, GL.GL_ONE_MINUS_SRC_ALPHA, GL.GL_ONE, GL.GL_ONE_MINUS_SRC_ALPHA)
        GL.glEnable(GL.GL_DEPTH_TEST)
        GL.glDepthFunc(GL.GL_LEQUAL)
        GL.glDisable(GL.GL_CULL_FACE)
        GL.glEnable(GL.GL_MULTISAMPLE)
        # ground plane (no depth write, so lines and boards on the ground stay visible)
        if f.style.ground_plane or f.ground is not None:
            self._draw_ground(f, VP)
        self._fixed(f)
        for l in f.lines:
            if l.depth and _on_ground(l):
                self._draw_line(l)
        if f.bowl is not None:
            self._draw_bowl(f, VP)
        if mesh is not None:
            self._draw_mesh(f, VP, eye, mesh)
        for td in f.tris:
            self._draw_flat(f, VP, eye, td)
        self._fixed(f)
        for l in f.lines:
            if not (l.depth and _on_ground(l)):
                self._draw_line(l)
        # leave a clean state for QPainter
        GL.glUseProgram(0)
        GL.glBindBuffer(GL.GL_ARRAY_BUFFER, 0)
        GL.glBindBuffer(GL.GL_ELEMENT_ARRAY_BUFFER, 0)
        for unit in (3, 2, 1, 0):
            GL.glActiveTexture(GL.GL_TEXTURE0 + unit)
            GL.glBindTexture(GL.GL_TEXTURE_2D, 0)
        GL.glDisable(GL.GL_DEPTH_TEST)
        GL.glDepthMask(GL.GL_TRUE)
        GL.glBlendFunc(GL.GL_SRC_ALPHA, GL.GL_ONE_MINUS_SRC_ALPHA)
        GL.glMatrixMode(GL.GL_PROJECTION)
        GL.glLoadIdentity()
        GL.glMatrixMode(GL.GL_MODELVIEW)
        GL.glLoadIdentity()

    def _fixed(self, f):
        """Matrices for the fixed-function lines."""
        GL = self.GL
        GL.glUseProgram(0)
        GL.glMatrixMode(GL.GL_PROJECTION)
        GL.glLoadMatrixd(f.P.T.astype(np.float64).ravel())
        GL.glMatrixMode(GL.GL_MODELVIEW)
        GL.glLoadMatrixd(f.V.T.astype(np.float64).ravel())

    def _draw_ground(self, f: Frame, VP):
        GL = self.GL
        p = self.prog['ground']
        GL.glUseProgram(p['id'])
        self._set(p, 'uVP', VP)
        self._common(p, f)
        st = f.style
        self._set(p, 'uBase', st.ground_rgb)
        self._set(p, 'uPlane', 1.0 if st.ground_plane else 0.0)
        self._set(p, 'uTarget', f.target)
        self._set(p, 'uFade', st.fade)
        g = f.ground
        if g is not None:
            from PyQt6.QtGui import QImage
            key = (g.version, g.image.width(), g.image.height(), g.nearest, id(g))
            if self._ground_tex is None or self._ground_tex[0] != key:
                img = g.image.convertToFormat(QImage.Format.Format_RGBA8888)
                ptr = img.constBits()
                ptr.setsize(img.sizeInBytes())
                data = np.frombuffer(ptr, np.uint8).reshape(img.height(), img.bytesPerLine())[:, :img.width() * 4]
                data = data.reshape(img.height(), img.width(), 4).copy()
                tex = self._texture(data, nearest=g.nearest, tex=self._ground_tex[1] if self._ground_tex else None)
                self._ground_tex = (key, tex, g)
            GL.glActiveTexture(GL.GL_TEXTURE0)
            GL.glBindTexture(GL.GL_TEXTURE_2D, self._ground_tex[1])
            GL.glUniform1i(self._u(p, 'uTex'), 0)
            self._set(p, 'uTexOn', 1.0)
            self._set(p, 'uExt', g.extent)
            z = g.z
        else:
            self._set(p, 'uTexOn', 0.0)
            z = 0.0
        ao = self._ao if (st.ao and self._ao is not None and self._mesh is not None) else None
        if ao is not None:
            GL.glActiveTexture(GL.GL_TEXTURE2)
            GL.glBindTexture(GL.GL_TEXTURE_2D, ao['tex'])
            GL.glUniform1i(self._u(p, 'uAO'), 2)
            self._set(p, 'uAOVP', ao['VP'])
            self._set(p, 'uAOInfo', (1.0, 1.0 / AO_N, ao['nf'][0], ao['nf'][1]))
        else:
            self._set(p, 'uAOInfo', (0.0, 0.0, 0.0, 1.0))
        GL.glActiveTexture(GL.GL_TEXTURE0)
        # a large quad around the view target (and the coverage texture)
        R = st.fade[1] + 5.0
        cx, cy = float(f.target[0]), float(f.target[1])
        x0, y0, x1, y1 = cx - R, cy - R, cx + R, cy + R
        if not st.ground_plane and g is not None:
            x0, y0, x1, y1 = g.extent
        q = np.array([[x0, y0, z], [x1, y0, z], [x1, y1, z], [x0, y0, z], [x1, y1, z], [x0, y1, z]], np.float32)
        GL.glDepthMask(GL.GL_FALSE)
        GL.glEnableVertexAttribArray(0)
        GL.glVertexAttribPointer(0, 3, GL.GL_FLOAT, GL.GL_FALSE, 0, q)
        GL.glDrawArrays(GL.GL_TRIANGLES, 0, 6)
        GL.glDisableVertexAttribArray(0)
        GL.glDepthMask(GL.GL_TRUE)

    def _draw_mesh(self, f, VP, eye, mesh):
        GL = self.GL
        md, vbo, n_opaque, n_all = mesh[:4]
        p = self.prog['mesh']
        GL.glUseProgram(p['id'])
        self._set(p, 'uVP', VP)
        self._set(p, 'uEye', eye)
        self._set(p, 'uSpec', 0.55)
        self._common(p, f)
        GL.glBindBuffer(GL.GL_ARRAY_BUFFER, vbo)
        for i, (n, off) in enumerate(((3, 0), (3, 12), (4, 24))):
            GL.glEnableVertexAttribArray(i)
            GL.glVertexAttribPointer(i, n, GL.GL_FLOAT, GL.GL_FALSE, 40, ctypes.c_void_p(off) if off else None)
        GL.glEnable(GL.GL_POLYGON_OFFSET_FILL)
        GL.glPolygonOffset(1.0, 1.0)
        GL.glPolygonMode(GL.GL_FRONT_AND_BACK, GL.GL_LINE if md.wire else GL.GL_FILL)
        if n_opaque:
            GL.glDepthMask(GL.GL_TRUE)
            GL.glDrawArrays(GL.GL_TRIANGLES, 0, n_opaque)
        if n_all > n_opaque:
            GL.glDepthMask(GL.GL_FALSE)
            GL.glDrawArrays(GL.GL_TRIANGLES, n_opaque, n_all - n_opaque)
            GL.glDepthMask(GL.GL_TRUE)
        GL.glPolygonMode(GL.GL_FRONT_AND_BACK, GL.GL_FILL)
        GL.glDisable(GL.GL_POLYGON_OFFSET_FILL)
        for i in range(3):
            GL.glDisableVertexAttribArray(i)
        GL.glBindBuffer(GL.GL_ARRAY_BUFFER, 0)

    def _draw_flat(self, f, VP, eye, td):
        GL = self.GL
        if not len(td.tri):
            return
        p = self.prog['flat']
        GL.glUseProgram(p['id'])
        self._set(p, 'uVP', VP)
        self._set(p, 'uEye', eye)
        self._set(p, 'uLight', LIGHT_DIR)
        self._set(p, 'uSky', f.style.sky_rgb)
        self._set(p, 'uBounce', f.style.bounce_rgb)
        v = np.ascontiguousarray(td.tri.reshape(-1, 3), np.float32)
        c = np.ascontiguousarray(np.repeat(np.asarray(td.colors, np.float32), 3, axis=0))
        if td.depth:
            GL.glEnable(GL.GL_DEPTH_TEST)
        else:
            GL.glDisable(GL.GL_DEPTH_TEST)
        translucent = (np.asarray(td.colors)[:, 3] < 0.99).any()
        GL.glDepthMask(GL.GL_FALSE if translucent else GL.GL_TRUE)
        GL.glEnableVertexAttribArray(0)
        GL.glEnableVertexAttribArray(2)
        GL.glVertexAttribPointer(0, 3, GL.GL_FLOAT, GL.GL_FALSE, 0, v)
        GL.glVertexAttribPointer(2, 4, GL.GL_FLOAT, GL.GL_FALSE, 0, c)
        GL.glDrawArrays(GL.GL_TRIANGLES, 0, len(v))
        GL.glDisableVertexAttribArray(0)
        GL.glDisableVertexAttribArray(2)
        GL.glDepthMask(GL.GL_TRUE)
        GL.glEnable(GL.GL_DEPTH_TEST)

    def _draw_bowl(self, f, VP):
        GL = self.GL
        b = f.bowl
        if self._bowl is None or self._bowl[0] is not b:
            if self._bowl is not None:
                GL.glDeleteBuffers(2, [self._bowl[1], self._bowl[2]])
            data = np.ascontiguousarray(np.c_[b.vertices, b.uv], np.float32)
            vbo, ibo = GL.glGenBuffers(1), GL.glGenBuffers(1)
            GL.glBindBuffer(GL.GL_ARRAY_BUFFER, vbo)
            GL.glBufferData(GL.GL_ARRAY_BUFFER, data.nbytes, data, GL.GL_STATIC_DRAW)
            idx = np.ascontiguousarray(b.faces, np.uint32)
            GL.glBindBuffer(GL.GL_ELEMENT_ARRAY_BUFFER, ibo)
            GL.glBufferData(GL.GL_ELEMENT_ARRAY_BUFFER, idx.nbytes, idx, GL.GL_STATIC_DRAW)
            GL.glBindBuffer(GL.GL_ARRAY_BUFFER, 0)
            GL.glBindBuffer(GL.GL_ELEMENT_ARRAY_BUFFER, 0)
            tex = self._texture(b.texture, wrap_s=GL.GL_REPEAT, tex=self._bowl[3] if self._bowl else None)
            self._bowl = (b, vbo, ibo, tex, idx.size)
        _, vbo, ibo, tex, count = self._bowl
        p = self.prog['bowl']
        GL.glUseProgram(p['id'])
        self._set(p, 'uVP', VP)
        GL.glActiveTexture(GL.GL_TEXTURE0)
        GL.glBindTexture(GL.GL_TEXTURE_2D, tex)
        GL.glUniform1i(self._u(p, 'uTex'), 0)
        GL.glBindBuffer(GL.GL_ARRAY_BUFFER, vbo)
        GL.glBindBuffer(GL.GL_ELEMENT_ARRAY_BUFFER, ibo)
        GL.glEnableVertexAttribArray(0)
        GL.glEnableVertexAttribArray(3)
        GL.glVertexAttribPointer(0, 3, GL.GL_FLOAT, GL.GL_FALSE, 20, None)
        GL.glVertexAttribPointer(3, 2, GL.GL_FLOAT, GL.GL_FALSE, 20, ctypes.c_void_p(12))
        GL.glDepthMask(GL.GL_TRUE)
        GL.glDrawElements(GL.GL_TRIANGLES, count, GL.GL_UNSIGNED_INT, None)
        GL.glDisableVertexAttribArray(0)
        GL.glDisableVertexAttribArray(3)
        GL.glBindBuffer(GL.GL_ARRAY_BUFFER, 0)
        GL.glBindBuffer(GL.GL_ELEMENT_ARRAY_BUFFER, 0)

    def _draw_line(self, l):
        from .viewport3d import dash_segments
        GL = self.GL
        segs = l.pts if l.segments else (np.stack([l.pts[:-1], l.pts[1:]], 1) if len(l.pts) > 1 else np.zeros((0, 2, 3)))
        if l.dash and len(segs):
            segs = np.concatenate([dash_segments(s, l.dash_len) for s in segs]) if l.segments else dash_segments(l.pts, l.dash_len)
        if not len(segs):
            return
        v = np.ascontiguousarray(segs.reshape(-1, 3), np.float32)
        if l.depth:
            GL.glEnable(GL.GL_DEPTH_TEST)
        else:
            GL.glDisable(GL.GL_DEPTH_TEST)
        GL.glLineWidth(float(l.width) * getattr(self, 'dpr', 1.0))
        c = l.color
        GL.glColor4f(c.redF(), c.greenF(), c.blueF(), c.alphaF())
        GL.glEnableClientState(GL.GL_VERTEX_ARRAY)
        GL.glVertexPointer(3, GL.GL_FLOAT, 0, v)
        GL.glDrawArrays(GL.GL_LINES, 0, len(v))
        GL.glDisableClientState(GL.GL_VERTEX_ARRAY)
        GL.glEnable(GL.GL_DEPTH_TEST)

    # ---- off-screen
    def render_offscreen(self, f: Frame, samples=4) -> np.ndarray:
        """Render into a temporary framebuffer and return (H, W, 4) uint8 RGBA, premultiplied alpha, row 0 = top."""
        GL = self.GL
        w, h = int(f.width), int(f.height)
        fbos, rbs = [], []

        def make(ms):
            fbo = GL.glGenFramebuffers(1)
            GL.glBindFramebuffer(GL.GL_FRAMEBUFFER, fbo)
            c, d = GL.glGenRenderbuffers(1), GL.glGenRenderbuffers(1)
            for rb, fmt, att in ((c, GL.GL_RGBA8, GL.GL_COLOR_ATTACHMENT0), (d, GL.GL_DEPTH_COMPONENT24, GL.GL_DEPTH_ATTACHMENT)):
                GL.glBindRenderbuffer(GL.GL_RENDERBUFFER, rb)
                if ms:
                    GL.glRenderbufferStorageMultisample(GL.GL_RENDERBUFFER, ms, fmt, w, h)
                else:
                    GL.glRenderbufferStorage(GL.GL_RENDERBUFFER, fmt, w, h)
                GL.glFramebufferRenderbuffer(GL.GL_FRAMEBUFFER, att, GL.GL_RENDERBUFFER, rb)
            fbos.append(fbo)
            rbs.extend([c, d])
            return fbo, GL.glCheckFramebufferStatus(GL.GL_FRAMEBUFFER) == GL.GL_FRAMEBUFFER_COMPLETE
        try:
            ms_fbo, ok = make(samples) if samples > 1 else (None, False)
            if not ok:
                samples = 0
            res_fbo, ok2 = make(0)
            if not ok2:
                raise RuntimeError('framebuffer incomplete')
            target = ms_fbo if samples else res_fbo
            if f.clear is None:
                f.clear = (0, 0, 0, 0)
            self.render(f, target)
            if samples:
                GL.glBindFramebuffer(GL.GL_READ_FRAMEBUFFER, ms_fbo)
                GL.glBindFramebuffer(GL.GL_DRAW_FRAMEBUFFER, res_fbo)
                GL.glBlitFramebuffer(0, 0, w, h, 0, 0, w, h, GL.GL_COLOR_BUFFER_BIT, GL.GL_NEAREST)
            GL.glBindFramebuffer(GL.GL_FRAMEBUFFER, res_fbo)
            GL.glPixelStorei(GL.GL_PACK_ALIGNMENT, 1)
            data = GL.glReadPixels(0, 0, w, h, GL.GL_RGBA, GL.GL_UNSIGNED_BYTE)
            img = np.frombuffer(data, np.uint8).reshape(h, w, 4)[::-1].copy()
            return img
        finally:
            GL.glBindFramebuffer(GL.GL_FRAMEBUFFER, 0)
            if rbs:
                GL.glDeleteRenderbuffers(len(rbs), rbs)
            if fbos:
                GL.glDeleteFramebuffers(len(fbos), fbos)

    def release(self):
        GL = self.GL
        if GL is None:
            return
        try:
            for name in ('_shadow', '_ao'):
                d = getattr(self, name)
                if d:
                    GL.glDeleteFramebuffers(1, [d['fbo']])
                    GL.glDeleteTextures([d['tex']])
            if self._mesh:
                GL.glDeleteBuffers(1, [self._mesh[1]])
            if self._bowl:
                GL.glDeleteBuffers(2, [self._bowl[1], self._bowl[2]])
                GL.glDeleteTextures([self._bowl[3]])
            for t in (self._ground_tex, self._cube):
                if t:
                    GL.glDeleteTextures([t[1]])
            for p in self.prog.values():
                GL.glDeleteProgram(p['id'])
        except Exception:
            pass
        self.prog = {}
        self._mesh = self._bowl = self._ground_tex = self._cube = self._shadow = self._ao = None
        self.ok = False


def _on_ground(l):
    pts = l.pts.reshape(-1, 3)
    return len(pts) and float(np.abs(pts[:, 2]).max()) < 0.03
