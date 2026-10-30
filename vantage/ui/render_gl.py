"""Shader renderer for the 3D viewport (OpenGL 2.1 / GLSL 1.20 and up, compatibility profile).

Pure PyOpenGL against whatever context is current, so the same code draws inside the Qt widget, into an off-screen
framebuffer for high-resolution renders, and in a headless EGL context for the tests.

What it adds over the fixed-function path: geometry kept on the GPU (uploaded once per mesh), smooth vertex normals
with crease angles, lighting in linear space with a gentle filmic shoulder and an sRGB encode at the end, materials
(clear-coated paint reflecting a procedural studio environment, dark see-through glass, matte tyres, black trim, light
lenses, rig metal), sky occlusion (ambient occlusion) from depth maps of the body and the rig seen from many directions
over the sky, soft shadow maps (Poisson-disc PCF with a normal-offset bias) on the body, the rig and the ground, a
contact shadow under the body, a ground plane that fades into the background (with an anti-aliased floor grid on the
display frame's 0.5 m and 1 m marks and its x and y axes), the textured 3D surround-view bowl,
round depth-tested point sprites, crisp screen-space outlines around selected triangles, and "camera light": every
camera's real lens model and occlusion (the body, the rig and the sensor and camera housings: the cube maps the
analysis uses, housings composited in) evaluated per pixel, so the body and the ground show exactly what each camera
sees while it is being moved, housing shadows included.

Data colours stay true: the ground textures (coverage, density, redundancy, footprints), the bowl texture and the
camera-light colours are not tone-mapped; they are shaded exactly as before (shadow and contact darkening only).

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
AO_N = 512                   # the contact-shadow height map under the body
CONTACT_SIGMA = 0.20         # m: how far the contact shadow spreads from where the body is low
SKY_DIRS = 24                # sky-occlusion depth maps (directions over the upper hemisphere)
SKY_TILE = 384               # pixels per sky-occlusion map
SKY_COLS = 6
SKY_ROWS = (SKY_DIRS + SKY_COLS - 1) // SKY_COLS
SKY_BUDGET = 1_500_000       # triangles per frame spent re-rendering sky maps (larger meshes refresh over frames)
SKY_DRAG_TILES = 4           # sky maps per frame while only rig parts / housings move (a drag stays smooth)
LIGHT_DIR = np.array([0.35, 0.55, 0.76]) / np.linalg.norm([0.35, 0.55, 0.76])
TONE_KNEE = 0.6              # linear luminance above which highlights roll off (sRGB ~0.80); identity below
POINT_VBO_MIN = 20000        # point clouds at least this large are kept on the GPU between frames


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
    exposure: float = 1.0        # linear scale before the tone curve (1 keeps a mid grey a mid grey)
    tone: bool = True            # filmic highlight shoulder; False = plain sRGB encode (clips)
    materials: bool = True       # MeshDraw.materials honoured; False = everything is paint
    outline_px: float = 2.0      # TriDraw.outline width, logical pixels
    backdrop: bool = True        # studio backdrop behind perspective views (sky gradient, horizon haze, the floor)
    grid: bool = False           # the floor grid drawn by the ground shader (anti-aliased, fades with distance)
    grid_minor: tuple = (0.5, 0.5, 0.5, 0.10)     # RGBA of the 0.5 m lines
    grid_major: tuple = (0.6, 0.6, 0.6, 0.22)     # RGBA of the 1 m lines
    grid_step: tuple = (0.5, 1.0)                 # minor and major spacing, m
    axes: bool = False           # the vehicle frame's x (red) and y (green) axes on the floor
    origin: tuple = (0.0, 0.0)   # world x, y of the display frame's origin (the grid and the axes start there)


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
    points: list = field(default_factory=list)     # [viewport3d.PointDraw]


# ---------------------------------------------------------------------------- shaders
_COLOR = '''
const vec3 LUMA = vec3(0.2126, 0.7152, 0.0722);
uniform vec4 uTone;          // exposure, knee, dither (in 1/255), on
vec3 to_lin(vec3 c) {
    c = clamp(c, 0.0, 1.0);
    return mix(c / 12.92, pow((c + 0.055) / 1.055, vec3(2.4)), step(0.04045, c));
}
vec3 to_srgb(vec3 c) {
    c = clamp(c, 0.0, 1.0);
    return mix(c * 12.92, 1.055 * pow(c, vec3(1.0 / 2.4)) - 0.055, step(0.0031308, c));
}
vec3 tone(vec3 x) {
    x = max(x * uTone.x, 0.0);
    if (uTone.w < 0.5) return x;
    float k = uTone.y;
    float L = dot(x, LUMA);
    if (L > k) x *= (k + (1.0 - k) * (1.0 - exp(-(L - k) / (1.0 - k)))) / L;     // luminance shoulder: no hue shift
    float m = max(x.r, max(x.g, x.b));
    if (m > 1.0) {                                                                   // still out of gamut: toward white
        float l = dot(x, LUMA);
        x = mix(x, vec3(l), clamp((m - 1.0) / max(m - l, 1e-4), 0.0, 1.0));
    }
    return x;
}
vec3 finish(vec3 lin) {
    vec3 c = to_srgb(tone(lin));
    float n = fract(52.9829189 * fract(dot(gl_FragCoord.xy, vec2(0.06711056, 0.00583715))));
    return c + (n - 0.5) * uTone.z / 255.0;                                         // dither: no banding on panels
}
'''

_COMMON = '''
uniform vec3 uLight;
uniform vec3 uSky;
uniform vec3 uBounce;
uniform sampler2DShadow uShadow;
uniform mat4 uShadowVP;
uniform float uShadowOn;
uniform vec4 uShadowInfo;    // PCF radius (uv), constant bias (depth), normal offset (m), max plane bias (depth)

const vec2 POISSON[12] = vec2[12](vec2(0.962, -0.195), vec2(0.519, 0.767), vec2(-0.696, 0.457),
    vec2(-0.322, -0.933), vec2(-0.326, -0.406), vec2(-0.840, -0.074), vec2(-0.203, 0.621), vec2(0.473, -0.480),
    vec2(0.185, -0.893), vec2(0.507, 0.064), vec2(0.896, 0.412), vec2(-0.792, -0.598));

vec3 shadow_uvz(vec3 p) {     // shadow map u, v, depth
    vec4 q = uShadowVP * vec4(p, 1.0);
    return q.xyz / q.w * 0.5 + 0.5;
}

float shadow_at(vec3 p, vec3 n) {
    if (uShadowOn < 0.5) return 1.0;
    vec3 c = shadow_uvz(p + n * uShadowInfo.z);       // normal offset
    // receiver-plane depth bias from the triangle itself (screen derivatives of the shadow coordinates): each tap
    // compares at the depth the surface has there, so a wide kernel does not shadow its own surface (no acne)
    vec3 cx = dFdx(c), cy = dFdy(c);
    float det = cx.x * cy.y - cx.y * cy.x;
    vec2 g = abs(det) > 1e-14 ? vec2(cx.z * cy.y - cy.z * cx.y, cy.z * cx.x - cx.z * cy.x) / det : vec2(0.0);
    if (c.x < 0.0 || c.x > 1.0 || c.y < 0.0 || c.y > 1.0 || c.z > 1.0) return 1.0;
    float zmax = uShadowInfo.w;                        // the plane may move a tap at most this much (grazing)
    float z = c.z - uShadowInfo.y;
    float a = 6.2831853 * fract(52.9829189 * fract(dot(gl_FragCoord.xy, vec2(0.06711056, 0.00583715))));
    float ca = cos(a), sa = sin(a);
    mat2 rot = mat2(ca, sa, -sa, ca);
    float lit = 0.0;
    for (int i = 0; i < 12; i++) {
        if (i == 4 && (lit < 0.004 || lit > 3.996)) return lit * 0.25;   // the outer ring agreed: done
        vec2 o = rot * POISSON[i] * uShadowInfo.x;
        lit += shadow2D(uShadow, vec3(c.xy + o, z + clamp(dot(g, o), -zmax, zmax))).r;
    }
    return lit / 12.0;
}

uniform sampler2DShadow uSkyMap;
uniform mat4 uSkyM[''' + str(SKY_DIRS) + '''];
uniform vec3 uSkyD[''' + str(SKY_DIRS) + '''];
uniform vec4 uSkyInfo;       // on, normal offset (m), one texel (depth units), strength

const float AO_KEY = 0.35;    // how much sky occlusion also dims the key light (crevices read; open panels keep it)

float sky_vis(vec3 p, vec3 n) {
    // the part of the sky this point sees past the body and the rig, cosine-weighted around the normal
    if (uSkyInfo.x < 0.5) return 1.0;
    vec3 q0 = p + n * uSkyInfo.y;
    float down = n.z < -0.2 ? 0.4 : 0.0;              // facing down: the sky low around the horizon still counts
    float s = 0.0, ws = 0.0;
    for (int i = 0; i < ''' + str(SKY_DIRS) + '''; i++) {
        float cn = dot(n, uSkyD[i]);
        float w = cn + down;
        if (w <= 0.0) continue;
        float tn = sqrt(max(1.0 - cn * cn, 0.0)) / max(abs(cn), 0.02);     // slope: grazing maps need more bias
        vec3 t = (uSkyM[i] * vec4(q0, 1.0)).xyz;
        float v = 1.0;
        if (t.x > 0.0 && t.x < 1.0 && t.y > 0.0 && t.y < 1.0 && t.z < 1.0) {
            float fi = float(i);
            vec2 tile = vec2(mod(fi, ''' + f'{SKY_COLS}.0' + '''), floor(fi / ''' + f'{SKY_COLS}.0' + '''));
            vec2 uv = (tile + clamp(t.xy, 0.004, 0.996)) / vec2(''' + f'{SKY_COLS}.0, {SKY_ROWS}.0' + ''');
            v = shadow2D(uSkyMap, vec3(uv, t.z - uSkyInfo.z * (1.0 + 1.5 * min(tn, 8.0)))).r;
        }
        s += w * v;
        ws += w;
    }
    float vis = ws > 0.0 ? s / ws : 1.0;
    return mix(1.0, vis, uSkyInfo.w);
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
        if (occ >= 0.0 && occ < dist - (0.02 + 0.01 * dist)) return false;   // 0: the camera is inside a housing
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

// the light rig, linear: a warm key along uLight, a sky dome, the ground's bounce, and a studio environment for
// reflections (sky gradient, a bright horizon band, a dark floor, an overhead strip light and a softbox at the key)
uniform vec3 uKeyL;
uniform vec3 uSkyL;
uniform vec3 uBounceL;
uniform vec3 uEnvTop;
uniform vec3 uEnvHor;
uniform vec3 uEnvLow;
uniform vec3 uEye;
uniform vec3 uViewDir;       // towards the viewer (orthographic views)
uniform float uOrtho;

vec3 env(vec3 r) {
    float z = r.z;
    vec3 c = z >= 0.0 ? mix(uEnvHor, uEnvTop, smoothstep(0.0, 0.55, z))
                      : mix(uEnvHor * 0.6, uEnvLow, smoothstep(0.0, 0.2, -z));
    c += uEnvHor * 0.9 * exp(-abs(z) * 22.0);                                       // horizon glow
    float strip = smoothstep(0.62, 0.74, z) * (1.0 - smoothstep(0.12, 0.2, abs(r.y)));
    float box = smoothstep(0.88, 0.95, dot(r, uLight));
    return c + uEnvHor * (strip * 1.3 + box * 2.2);
}

vec3 view_dir(vec3 p) {
    return uOrtho > 0.5 ? uViewDir : normalize(uEye - p);
}
'''

MESH_VS = '''
#version 120
invariant gl_Position;
attribute vec3 aPos;
attribute vec3 aNrm;
attribute vec4 aCol;
attribute float aMat;
uniform mat4 uVP;
varying vec3 vPos;
varying vec3 vNrm;
varying vec4 vCol;
varying float vMat;
void main() {
    vPos = aPos; vNrm = aNrm; vCol = aCol; vMat = aMat;
    gl_Position = uVP * vec4(aPos, 1.0);
}
'''

MESH_FS = '''
#version 120
varying vec3 vPos;
varying vec3 vNrm;
varying vec4 vCol;
varying float vMat;
uniform float uMatOn;
''' + _COLOR + _COMMON + '''
void main() {
    vec3 N = normalize(vNrm);
    vec3 V = view_dir(vPos);
    if (dot(N, V) < 0.0) N = -N;                       // two-sided: open meshes and inward faces
    float m = uMatOn > 0.5 ? floor(vMat + 0.5) : 0.0;
    vec3 alb = to_lin(vCol.rgb);
    float a = vCol.a;
    // material: diffuse weight, clear coat (env reflection) strength, base reflectance, key highlight gloss/strength
    float kd = 1.0, coat = 1.0, f0 = 0.04, gloss = 160.0, ks = 0.55;
    vec3 tint = vec3(1.0);
    if (m == 1.0) {                                   // glass: dark, sharp reflections, a little see-through
        alb *= 0.30; coat = 1.0; gloss = 900.0; ks = 1.4;
    } else if (m == 2.0) {                            // tyre rubber: matte black
        alb = min(alb, vec3(0.022)); coat = 0.15; f0 = 0.02; gloss = 10.0; ks = 0.06;
    } else if (m == 3.0) {                            // black trim / plastic
        alb = min(alb, vec3(0.03)); coat = 0.45; gloss = 40.0; ks = 0.18;
    } else if (m == 4.0) {                            // light lens
        alb *= 0.55; coat = 1.4; gloss = 600.0; ks = 1.2;
    } else if (m == 5.0) {                            // rig metal: anodised, tinted reflections
        kd = 0.55; f0 = 0.35; tint = alb / max(dot(alb, LUMA), 0.02) * 0.8; gloss = 70.0; ks = 0.5; coat = 0.8;
    }
    float ndl = max(dot(N, uLight), 0.0);
    float sh = shadow_at(vPos, N);
    float ao = sky_vis(vPos, N);
    float NdV = max(dot(N, V), 0.0);
    float F = f0 + (1.0 - f0) * pow(1.0 - NdV, 5.0);
    vec3 amb = mix(uBounceL, uSkyL, N.z * 0.5 + 0.5) * ao;
    vec3 c = alb * kd * (amb + uKeyL * ndl * sh * mix(1.0, ao, AO_KEY)) * (1.0 - F * min(coat, 1.0));
    vec3 R = reflect(-V, N);
    c += env(R) * tint * F * coat * mix(ao, 1.0, 0.25);
    vec3 H = normalize(uLight + V);
    c += uKeyL * tint * pow(max(dot(N, H), 0.0), gloss) * (gloss + 8.0) / 25.13 * ks * F * 6.0 * sh * ndl;
    if (m == 1.0) a = min(a, mix(0.88, 1.0, F));       // glass: see-through head-on, a mirror at grazing angles
    c = camlight(finish(c), vPos + N * 0.03);          // off the surface: a face does not occlude itself
    gl_FragColor = vec4(c, a);
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
uniform sampler2D uContact;  // contact occlusion under the body, baked from the underside height map
uniform mat4 uAOVP;
uniform vec4 uAOInfo;        // on, -, -, -
uniform vec3 uTarget;
uniform vec2 uFade;
uniform float uPlane;        // 1 = draw the plane colour, 0 = only the texture
uniform vec4 uGrid;          // on, minor step (m), major step (m), axes on
uniform vec4 uGridC1;        // minor line RGBA
uniform vec4 uGridC2;        // major line RGBA
uniform vec2 uOrigin;        // the display frame's origin
''' + _COMMON + '''
float grid_lines(vec2 p, float step) {
    // anti-aliased lines about one pixel wide; they fade out where they would crowd closer than a few pixels
    vec2 q = p / step;
    vec2 w = max(fwidth(q), vec2(1e-6));
    vec2 d = abs(fract(q - 0.5) - 0.5) / w;
    float l = 1.0 - min(min(d.x, d.y), 1.0);
    return l * (1.0 - smoothstep(0.12, 0.35, max(w.x, w.y)));
}
float axis_line(float v, float px) {
    float w = max(fwidth(v), 1e-6);
    return 1.0 - smoothstep(px * 0.5, px * 0.5 + 1.0, abs(v) / w);
}
float ao_at(vec2 xy) {
    if (uAOInfo.x < 0.5) return 1.0;
    vec4 q = uAOVP * vec4(xy, 0.0, 1.0);
    vec2 c = q.xy / q.w * 0.5 + 0.5;
    if (c.x < 0.0 || c.x > 1.0 || c.y < 0.0 || c.y > 1.0) return 1.0;
    return texture2D(uContact, c).r;
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
    float sh = shadow_at(vPos, vec3(0.0, 0.0, 1.0));
    float ao = ao_at(vPos.xy);
    col.rgb *= (0.55 + 0.45 * sh) * ao;
    col.a = max(col.a, (1.0 - min(sh, ao)) * 0.85);
    float dist = length(vPos.xy - uTarget.xy);
    // the camera light at full strength near the rig, toned down far out on the floor (it reaches the horizon)
    col.rgb = mix(col.rgb, camlight(col.rgb, vPos), 1.0 - 0.65 * smoothstep(9.0, 26.0, dist));
    if (uGrid.x > 0.5) {
        float g1 = grid_lines(vPos.xy - uOrigin, uGrid.y) * uGridC1.a;
        float g2 = grid_lines(vPos.xy - uOrigin, uGrid.z) * uGridC2.a;
        col.rgb = mix(col.rgb, uGridC1.rgb, g1);
        col.rgb = mix(col.rgb, uGridC2.rgb, g2);
        col.a = max(col.a, max(g1, g2));
    }
    if (uGrid.w > 0.5) {         // the vehicle frame's axes through the origin: x red, y green
        vec2 o = vPos.xy - uOrigin;
        float ax = axis_line(o.y, 1.6) * step(0.0, o.x) * 0.85;
        float ay = axis_line(o.x, 1.6) * step(0.0, o.y) * 0.85;
        col.rgb = mix(col.rgb, vec3(0.88, 0.33, 0.25), ax);
        col.rgb = mix(col.rgb, vec3(0.25, 0.68, 0.29), ay);
        col.a = max(col.a, max(ax, ay));
    }
    col.a *= 1.0 - smoothstep(uFade.x, uFade.y, dist);
    gl_FragColor = col;
}
'''

FLAT_VS = '''
#version 120
attribute vec3 aPos;
attribute vec3 aNrm;
attribute vec4 aCol;
uniform mat4 uVP;
varying vec3 vPos;
varying vec3 vNrm;
varying vec4 vCol;
void main() { vPos = aPos; vNrm = aNrm; vCol = aCol; gl_Position = uVP * vec4(aPos, 1.0); }
'''

FLAT_FS = '''
#version 120
varying vec3 vPos;
varying vec3 vNrm;
varying vec4 vCol;
uniform float uSmooth;
uniform float uLit;          // 1: lit like the body (linear, shadows, sky occlusion); 0: the plain see-through shading
''' + _COLOR + _COMMON + '''
void main() {
    vec3 N = uSmooth > 0.5 ? normalize(vNrm) : normalize(cross(dFdx(vPos), dFdy(vPos)));
    vec3 V = view_dir(vPos);
    if (dot(N, V) < 0.0) N = -N;
    float ndl = max(dot(N, uLight), 0.0);
    if (uLit < 0.5) {
        vec3 hemi = mix(uBounce, uSky, N.z * 0.5 + 0.5);
        gl_FragColor = vec4(vCol.rgb * (hemi * 0.75 + ndl * 0.55), vCol.a);
        return;
    }
    vec3 alb = to_lin(vCol.rgb);
    float sh = shadow_at(vPos, N);
    float ao = sky_vis(vPos, N);
    vec3 amb = mix(uBounceL, uSkyL, N.z * 0.5 + 0.5) * ao;
    float NdV = max(dot(N, V), 0.0);
    float F = 0.04 + 0.96 * pow(1.0 - NdV, 5.0);
    vec3 c = alb * (amb + uKeyL * ndl * sh * mix(1.0, ao, AO_KEY)) * (1.0 - 0.5 * F);
    c += env(reflect(-V, N)) * F * 0.5 * mix(ao, 1.0, 0.25);                         // a satin finish
    vec3 H = normalize(uLight + V);
    c += uKeyL * pow(max(dot(N, H), 0.0), 48.0) * 0.12 * sh * ndl;
    gl_FragColor = vec4(finish(c), vCol.a);
}
'''

CONTACT_VS = '''
#version 120
attribute vec3 aPos;
void main() { gl_Position = vec4(aPos, 1.0); }
'''

CONTACT_FS = '''
#version 120
uniform sampler2D uAO;       // pass 0: depth of the body's underside seen from below; passes 1-2: the previous pass
uniform vec4 uAOInfo;        // pass, size (px), near, far
uniform vec2 uStep;          // blur: texel step (uv) between taps; 33 taps cover +-3 sigma
void main() {
    vec2 c = gl_FragCoord.xy / uAOInfo.y;
    if (uAOInfo.x < 0.5) {
        // how much the body's underside above this point shades the ground: low parts fully, 0.85 m up not at all
        float h = uAOInfo.z + texture2D(uAO, c).r * (uAOInfo.w - uAOInfo.z) - 1.0;
        gl_FragColor = vec4(vec3(1.0 - smoothstep(0.02, 0.85, h)), 1.0);
        return;
    }
    float s = 0.0, ws = 0.0;
    for (int i = -16; i <= 16; i++) {
        float t = float(i) / 16.0 * 3.0;
        float w = exp(-0.5 * t * t);
        s += w * texture2D(uAO, c + uStep * float(i)).r;
        ws += w;
    }
    float v = s / ws;
    if (uAOInfo.x > 1.5) v = 1.0 - 0.80 * v;           // last pass: the ground's brightness factor
    gl_FragColor = vec4(vec3(v), 1.0);
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
invariant gl_Position;
attribute vec3 aPos;
uniform mat4 uVP;
void main() { gl_Position = uVP * vec4(aPos, 1.0); }
'''

DEPTH_FS = '''
#version 120
void main() { gl_FragColor = vec4(1.0); }
'''

POINT_VS = '''
#version 120
attribute vec3 aPos;
attribute vec4 aCol;
uniform mat4 uVP;
uniform float uSize;
varying vec4 vCol;
void main() { vCol = aCol; gl_Position = uVP * vec4(aPos, 1.0); gl_PointSize = uSize; }
'''

POINT_FS = '''
#version 120
varying vec4 vCol;
uniform float uSize;
void main() {
    float r = length(gl_PointCoord * 2.0 - 1.0);
    float px = 2.0 / max(uSize, 1.0);                  // one pixel in disc radii
    float a = 1.0 - smoothstep(1.0 - px, 1.0 + px * 0.5, r);
    if (a <= 0.003) discard;
    gl_FragColor = vec4(vCol.rgb, vCol.a * a);
}
'''

OUTLINE_VS = '''
#version 120
attribute vec3 aPos;
uniform mat4 uVP;
uniform vec2 uOff;           // screen offset, NDC
void main() {
    vec4 p = uVP * vec4(aPos, 1.0);
    p.xy += uOff * p.w;
    p.z -= 0.0004 * p.w;                               // just in front of what the object stands on
    gl_Position = p;
}
'''

OUTLINE_FS = '''
#version 120
uniform vec4 uColor;
void main() { gl_FragColor = uColor; }
'''

PROBE_VS = '''
#version 120
attribute vec3 aPos;
attribute vec4 aCol;
varying vec4 vCol;
void main() { vCol = aCol; gl_Position = vec4(aPos, 1.0); gl_PointSize = 1.0; }
'''

PROBE_FS = '''
#version 120
varying vec4 vCol;
''' + _COLOR + '''
void main() { gl_FragColor = vec4(finish(vCol.rgb), 1.0); }
'''

# texture units: every sampler has its own, so a shadow sampler and a plain one never share a unit
UNITS = {'uTex': 0, 'uShadow': 1, 'uAO': 2, 'uContact': 2, 'uCube': 3, 'uSkyMap': 4}


# ---------------------------------------------------------------------------- colour (reference for the tests)
def to_linear(c):
    c = np.clip(np.asarray(c, float), 0, 1)
    return np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)


def to_srgb(c):
    c = np.clip(np.asarray(c, float), 0, 1)
    return np.where(c <= 0.0031308, c * 12.92, 1.055 * c ** (1 / 2.4) - 0.055)


def tone_reference(lin, exposure=1.0, knee=TONE_KNEE):
    """What the shaders' tone(): exposure, a luminance shoulder above the knee, out-of-gamut colours toward white."""
    x = np.maximum(np.atleast_2d(np.asarray(lin, float)) * exposure, 0)
    L = x @ np.array([0.2126, 0.7152, 0.0722])
    over = L > knee
    s = np.ones_like(L)
    s[over] = (knee + (1 - knee) * (1 - np.exp(-(L[over] - knee) / (1 - knee)))) / L[over]
    x = x * s[:, None]
    m = x.max(axis=1)
    l2 = x @ np.array([0.2126, 0.7152, 0.0722])
    t = np.where(m > 1, np.clip((m - 1) / np.maximum(m - l2, 1e-4), 0, 1), 0)
    return x + (l2[:, None] - x) * t[:, None]


def light_rig(st: Style) -> dict:
    """Linear light intensities and the reflection environment from the theme's style colours."""
    lum = np.array([0.2126, 0.7152, 0.0722])
    sky = to_linear(st.sky_rgb)
    bounce = to_linear(st.bounce_rgb)
    ground = to_linear(st.ground_rgb)
    light = float(np.dot(to_linear(st.sky_rgb), lum)) > 0.6          # the light theme's brighter sky
    sky_n = sky / max(float(sky @ lum), 1e-3)
    return dict(
        key=np.array([1.0, 0.97, 0.92]) * 0.95,
        sky=sky_n * (0.38 if light else 0.28),
        bounce=bounce / max(float(bounce @ lum), 1e-3) * (0.12 if light else 0.07),
        env_top=sky_n * (0.45 if light else 0.26),
        env_hor=np.array([1.0, 1.0, 1.0]) * (0.75 if light else 0.55),
        env_low=np.maximum(ground * 0.8, 0.012))


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
    VP, _ = shadow_frustum(lo, hi, light)
    return VP


def shadow_frustum(lo, hi, light=LIGHT_DIR):
    """shadow_matrix plus (world size of a texel side in metres at SHADOW_N, depth range in metres)."""
    c = (lo + hi) / 2
    rad = float(np.linalg.norm(hi - lo)) / 2 + 0.2
    eye = c + light * (rad + 2.0)
    V = _look_at(eye, c, np.array([0, 0, 1.0]))
    corners = np.array([[x, y, z] for x in (lo[0], hi[0]) for y in (lo[1], hi[1]) for z in (lo[2], hi[2])])
    q = corners @ V[:3, :3].T + V[:3, 3]
    pad = 0.15
    l, r, b, t = q[:, 0].min() - pad, q[:, 0].max() + pad, q[:, 1].min() - pad, q[:, 1].max() + pad
    n, f = max(0.01, -q[:, 2].max() - 0.5), -q[:, 2].min() + 25.0
    return _ortho(l, r, b, t, n, f) @ V, (max(r - l, t - b) / SHADOW_N, f - n)


def ao_matrix(lo, hi, margin=0.9, near=0.5, far=4.0):
    """Orthographic camera below the ground looking up: its depth is the height of the body's underside."""
    c = (lo + hi) / 2
    eye = np.array([c[0], c[1], -1.0])
    V = _look_at(eye, np.array([c[0], c[1], 0.0]), np.array([1.0, 0, 0]))
    q = np.array([[lo[0] - margin, lo[1] - margin, 0], [hi[0] + margin, hi[1] + margin, 0]]) @ V[:3, :3].T + V[:3, 3]
    P = _ortho(q[:, 0].min(), q[:, 0].max(), q[:, 1].min(), q[:, 1].max(), near, far)
    return P @ V, (near, far)


def ao_extent(lo, hi, margin=0.9):
    """World size (metres) the under-body height map spans along its texture u and v: its camera looks up with +x
    as up, so u runs along world y and v along world x."""
    return float(hi[1] - lo[1] + 2 * margin), float(hi[0] - lo[0] + 2 * margin)


def sky_dirs(n=SKY_DIRS) -> np.ndarray:
    """(n,3) unit directions spread evenly (equal solid angle, golden-angle spiral) over the sky above the horizon."""
    i = np.arange(n) + 0.5
    z = 0.03 + 0.95 * i / n
    phi = i * math.pi * (3 - math.sqrt(5))
    r = np.sqrt(1 - z * z)
    return np.c_[r * np.cos(phi), r * np.sin(phi), z]


def sky_matrices(lo, hi, dirs) -> tuple[np.ndarray, float, float]:
    """For each direction an orthographic view of the box's bounding sphere seen from the sky along it, mapped to
    [0,1]^3 (tile u, v, depth). Returns (n,4,4), the texel size in metres and the depth range in metres."""
    c = (np.asarray(lo, float) + np.asarray(hi, float)) / 2
    R = float(np.linalg.norm(np.asarray(hi, float) - lo)) / 2 + 0.15
    S = np.array([[0.5, 0, 0, 0.5], [0, 0.5, 0, 0.5], [0, 0, 0.5, 0.5], [0, 0, 0, 1.0]])
    out = np.zeros((len(dirs), 4, 4))
    for k, d in enumerate(dirs):
        up = np.array([0, 0, 1.0]) if abs(d[2]) < 0.95 else np.array([1.0, 0, 0])
        V = _look_at(c + d * (R + 1.0), c, up)
        out[k] = S @ _ortho(-R, R, -R, R, 0.5, 2 * R + 1.5) @ V
    return out, 2 * R / SKY_TILE, 2 * R + 1.0


def encode_cubes(cubes: list) -> tuple[np.ndarray, int]:
    """Occlusion cube maps (depth arrays (6, N, N) of several cameras: the body, the rig and the housings composited
    in) as one RGBA8 atlas (faces across, cameras down), depth in mm in R/G (16 bit), B = occluder present."""
    N = cubes[0].shape[1]
    D = np.stack([np.asarray(c, np.float32) for c in cubes])           # (cams, faces, rows, cols)
    ok = np.isfinite(D)
    mm = np.rint(np.minimum(D, np.float32(65.535)) * np.float32(1000)).astype(np.uint16)
    mm[~ok] = 0
    out = np.empty((len(cubes), N, 6, N, 4), np.uint8)                # (cams, rows, faces, cols, rgba)
    out[..., 0] = (mm >> 8).transpose(0, 2, 1, 3)
    out[..., 1] = (mm & 255).transpose(0, 2, 1, 3)
    out[..., 2] = (ok.view(np.uint8) * 255).transpose(0, 2, 1, 3)
    out[..., 3] = 255
    return out.reshape(len(cubes) * N, 6 * N, 4), N


def _fingerprint(a: np.ndarray, refs: list):
    """A cheap identity for a triangle array: its bytes when small (scenes rebuild small arrays every frame), else
    its id and shape (kept alive in refs so the id stays unique)."""
    if a.nbytes <= 262144:
        return hash(np.ascontiguousarray(a).tobytes())
    refs.append(a)
    return (id(a), a.shape)


def outline_offsets(width_px: float) -> list:
    """Screen offsets (pixels) the outline pass draws the object at: eight around a circle of the outline width
    (and a ring at half the width for wide outlines, so thin parts leave no gaps)."""
    out = []
    rings = (width_px,) if width_px <= 2.5 else (width_px * 0.5, width_px)
    for r in rings:
        for k in range(8):
            a = k * math.pi / 4
            out.append((r * math.cos(a), r * math.sin(a)))
    return out


# ---------------------------------------------------------------------------- renderer
class GLRenderer:
    def __init__(self):
        self.ok = False
        self.error = ''
        self.GL = None
        self.prog = {}
        self._mesh = None            # (MeshDraw ref, vbo, n_opaque, n_occluding, n_all, lo, hi)
        self._ground_tex = None      # (key, tex)
        self._bowl = None            # (BowlDraw ref, vbo, ibo, tex, count)
        self._shadow = None          # dict(fbo, tex, VP, ...)
        self._ao = None
        self._sky = None             # dict(fbo, tex, M (n,4,4) per tile, next, key, ...)
        self._occ = None             # dict(key, vbo, count, refs, lo, hi): the occluding triangles of the TriDraws
        self._maps_key = None
        self._cube = None            # (key, tex, N, W, H)
        self._points = {}            # (id(points), id(colors)) -> (vbo, n, refs)
        self.fbo_ok = False
        self.pending = False         # sky maps still being refreshed: draw again soon
        self.no_stencil = False      # draw outlines without the stencil buffer (as on framebuffers without one)
        self.dpr = 1.0
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
            self.prog['point'] = self._program(POINT_VS, POINT_FS)
            self.prog['outline'] = self._program(OUTLINE_VS, OUTLINE_FS)
            self.prog['probe'] = self._program(PROBE_VS, PROBE_FS)
            self.prog['contact'] = self._program(CONTACT_VS, CONTACT_FS)
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
        for i, name in enumerate(('aPos', 'aNrm', 'aCol', 'aUV', 'aMat')):
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
        GL.glUseProgram(p)
        for name, unit in UNITS.items():           # fixed texture units, set once
            if locs.get(name, -1) >= 0:
                GL.glUniform1i(locs[name], unit)
        GL.glUseProgram(0)
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
        rig = self._rig
        self._set(prog, 'uKeyL', rig['key'])
        self._set(prog, 'uSkyL', rig['sky'])
        self._set(prog, 'uBounceL', rig['bounce'])
        self._set(prog, 'uEnvTop', rig['env_top'])
        self._set(prog, 'uEnvHor', rig['env_hor'])
        self._set(prog, 'uEnvLow', rig['env_low'])
        self._set(prog, 'uEye', self._eye)
        self._set(prog, 'uViewDir', self._view_dir)
        self._set(prog, 'uOrtho', 1.0 if f.ortho else 0.0)
        self._set(prog, 'uTone', (st.exposure, TONE_KNEE, 0.9, 1.0 if st.tone else 0.0))
        sh = self._shadow if (st.shadows and self._shadow) else None
        self._set(prog, 'uShadowOn', 1.0 if sh else 0.0)
        if sh:
            GL.glActiveTexture(GL.GL_TEXTURE1)
            GL.glBindTexture(GL.GL_TEXTURE_2D, sh['tex'])
            self._set(prog, 'uShadowVP', sh['VP'])
            texel, depth_m = sh['texel'], sh['depth_m']
            self._set(prog, 'uShadowInfo', (0.022 / (texel * SHADOW_N), 0.002 / depth_m, 2.0 * texel, 0.08 / depth_m))
        sky = self._sky if (st.ao and self._sky is not None and self._sky.get('ok')) else None
        loc = self._u(prog, 'uSkyM')
        if loc is not None and loc >= 0:
            if sky is not None:
                GL.glActiveTexture(GL.GL_TEXTURE4)
                GL.glBindTexture(GL.GL_TEXTURE_2D, sky['tex'])
                GL.glUniformMatrix4fv(loc, SKY_DIRS, GL.GL_TRUE, sky['M'].astype(np.float32))
                dl = self._u(prog, 'uSkyD')
                if dl is not None and dl >= 0:
                    GL.glUniform3fv(dl, SKY_DIRS, sky['dirs'].astype(np.float32))
                self._set(prog, 'uSkyInfo', (1.0, 1.0 * sky['texel'], sky['texel'] / sky['depth_m'], 0.92))
            else:
                self._set(prog, 'uSkyInfo', (0.0, 0.0, 0.0, 0.0))
        # camera light
        n = min(len(f.cams), MAX_CAMS)
        loc = self._u(prog, 'uNCam')
        if loc is not None and loc >= 0:
            GL.glUniform1i(loc, n)
        if n and loc is not None and loc >= 0:
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
                self._set(prog, 'uCubeInfo', (cube[2], cube[3], cube[4], 1.0))
            else:
                self._set(prog, 'uCubeInfo', (1, 1, 1, 0.0))
        GL.glActiveTexture(GL.GL_TEXTURE0)

    # ---- resources
    def _texture(self, data: np.ndarray, nearest=False, wrap_s=None, tex=None):
        GL = self.GL
        tex = tex if tex is not None else GL.glGenTextures(1)
        GL.glBindTexture(GL.GL_TEXTURE_2D, tex)
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
        mat = getattr(md, 'materials', None)
        mat = np.asarray(mat, np.float32) if mat is not None and len(mat) == n else np.zeros(n, np.float32)
        buf = np.empty((n, 3, 11), np.float32)
        buf[..., 0:3] = md.tri
        buf[..., 3:6] = nrm
        buf[..., 6:10] = col[:, None, :]
        buf[..., 10] = mat[:, None]
        glass = mat == 1
        solid = md.alpha > 0.99
        # opaque first, then glass (occludes, drawn see-through), then translucent
        groups = (solid & ~glass, solid & glass, ~solid)
        order = np.concatenate([np.nonzero(g)[0] for g in groups])
        buf = np.ascontiguousarray(buf[order])
        vbo = GL.glGenBuffers(1)
        GL.glBindBuffer(GL.GL_ARRAY_BUFFER, vbo)
        GL.glBufferData(GL.GL_ARRAY_BUFFER, buf.nbytes, buf, GL.GL_STATIC_DRAW)
        GL.glBindBuffer(GL.GL_ARRAY_BUFFER, 0)
        lo = md.tri.reshape(-1, 3).min(axis=0)
        hi = md.tri.reshape(-1, 3).max(axis=0)
        n_op, n_gl = int(groups[0].sum()), int(groups[1].sum())
        self._mesh = (md, vbo, n_op * 3, (n_op + n_gl) * 3, n * 3, lo, hi)
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

    def _depth_fbo(self, w, h, compare=False, linear=False):
        GL = self.GL
        tex = GL.glGenTextures(1)
        GL.glBindTexture(GL.GL_TEXTURE_2D, tex)
        GL.glTexImage2D(GL.GL_TEXTURE_2D, 0, GL.GL_DEPTH_COMPONENT24, w, h, 0, GL.GL_DEPTH_COMPONENT, GL.GL_FLOAT, None)
        flt = GL.GL_LINEAR if (compare or linear) else GL.GL_NEAREST    # linear + compare = 2x2 hardware PCF per tap
        for p, v in ((GL.GL_TEXTURE_MIN_FILTER, flt), (GL.GL_TEXTURE_MAG_FILTER, flt),
                     (GL.GL_TEXTURE_WRAP_S, GL.GL_CLAMP_TO_EDGE), (GL.GL_TEXTURE_WRAP_T, GL.GL_CLAMP_TO_EDGE)):
            GL.glTexParameteri(GL.GL_TEXTURE_2D, p, v)
        if compare:
            GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_COMPARE_MODE, GL.GL_COMPARE_R_TO_TEXTURE)
            GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_COMPARE_FUNC, GL.GL_LEQUAL)
        GL.glBindTexture(GL.GL_TEXTURE_2D, 0)
        fbo = GL.glGenFramebuffers(1)
        GL.glBindFramebuffer(GL.GL_FRAMEBUFFER, fbo)
        GL.glFramebufferTexture2D(GL.GL_FRAMEBUFFER, GL.GL_DEPTH_ATTACHMENT, GL.GL_TEXTURE_2D, tex, 0)
        GL.glDrawBuffer(GL.GL_NONE)
        GL.glReadBuffer(GL.GL_NONE)
        ok = GL.glCheckFramebufferStatus(GL.GL_FRAMEBUFFER) == GL.GL_FRAMEBUFFER_COMPLETE
        return {'fbo': fbo, 'tex': tex, 'ok': ok}

    def _begin_depth(self):
        GL = self.GL
        GL.glEnable(GL.GL_DEPTH_TEST)
        GL.glDepthFunc(GL.GL_LEQUAL)
        GL.glDepthMask(GL.GL_TRUE)
        GL.glDisable(GL.GL_BLEND)
        GL.glColorMask(GL.GL_FALSE, GL.GL_FALSE, GL.GL_FALSE, GL.GL_FALSE)
        p = self.prog['depth']
        GL.glUseProgram(p['id'])
        GL.glEnableVertexAttribArray(0)
        return p

    def _end_depth(self, target_fbo):
        GL = self.GL
        GL.glDisableVertexAttribArray(0)
        GL.glBindBuffer(GL.GL_ARRAY_BUFFER, 0)
        GL.glColorMask(GL.GL_TRUE, GL.GL_TRUE, GL.GL_TRUE, GL.GL_TRUE)
        GL.glDisable(GL.GL_SCISSOR_TEST)
        GL.glBindFramebuffer(GL.GL_FRAMEBUFFER, target_fbo)

    def _draw_occluders(self, p, VP, mesh, occ):
        """The occluding triangles (the mesh's opaque and glass parts, the TriDraws' opaque parts) into the bound
        depth buffer."""
        GL = self.GL
        self._set(p, 'uVP', VP)
        if mesh is not None and mesh[3]:
            GL.glBindBuffer(GL.GL_ARRAY_BUFFER, mesh[1])
            GL.glVertexAttribPointer(0, 3, GL.GL_FLOAT, GL.GL_FALSE, 44, None)
            GL.glDrawArrays(GL.GL_TRIANGLES, 0, mesh[3])
        if occ is not None and occ['count']:
            GL.glBindBuffer(GL.GL_ARRAY_BUFFER, occ['vbo'])
            GL.glVertexAttribPointer(0, 3, GL.GL_FLOAT, GL.GL_FALSE, 12, None)
            GL.glDrawArrays(GL.GL_TRIANGLES, 0, occ['count'])

    def _depth_pass(self, slot, n, VP, mesh, occ, target_fbo):
        GL = self.GL
        d = getattr(self, slot)
        if d is None:
            d = self._depth_fbo(n, n, compare=(slot == '_shadow'), linear=(slot == '_ao'))
            setattr(self, slot, d)
        if not d['ok']:
            return False
        GL.glBindFramebuffer(GL.GL_FRAMEBUFFER, d['fbo'])
        GL.glViewport(0, 0, n, n)
        GL.glDisable(GL.GL_SCISSOR_TEST)
        GL.glClearDepth(1.0)
        GL.glClear(GL.GL_DEPTH_BUFFER_BIT)
        p = self._begin_depth()
        self._draw_occluders(p, VP, mesh, occ)
        self._end_depth(target_fbo)
        d['VP'] = VP
        return True

    def _occluders(self, f: Frame):
        """The opaque, depth-tested triangles of the TriDraws that carry normals (rig parts, sensor housings) as one
        position buffer, refreshed when their geometry changes; they cast shadows and occlude the sky."""
        GL = self.GL
        refs, keys, parts = [], [], []
        for td in f.tris:
            if not td.depth or getattr(td, 'normals', None) is None or not len(td.tri):
                continue
            alpha = np.asarray(td.colors)[:, 3]
            solid = alpha > 0.99
            if not solid.any():
                continue
            keys.append((_fingerprint(np.asarray(td.tri), refs), _fingerprint(solid, refs)))
            parts.append((td, solid))
        key = tuple(keys)
        if self._occ is not None and self._occ['key'] == key:
            return self._occ
        if not parts:
            if self._occ is not None and self._occ['vbo']:
                GL.glDeleteBuffers(1, [self._occ['vbo']])
            self._occ = {'key': key, 'vbo': 0, 'count': 0, 'refs': refs, 'lo': None, 'hi': None}
            return self._occ
        pos = np.ascontiguousarray(np.concatenate([np.asarray(td.tri, np.float32)[s] for td, s in parts]).reshape(-1, 3))
        vbo = self._occ['vbo'] if self._occ is not None and self._occ['vbo'] else GL.glGenBuffers(1)
        GL.glBindBuffer(GL.GL_ARRAY_BUFFER, vbo)
        GL.glBufferData(GL.GL_ARRAY_BUFFER, pos.nbytes, pos, GL.GL_DYNAMIC_DRAW)
        GL.glBindBuffer(GL.GL_ARRAY_BUFFER, 0)
        self._occ = {'key': key, 'vbo': vbo, 'count': len(pos), 'refs': refs, 'lo': pos.min(axis=0),
                     'hi': pos.max(axis=0)}
        return self._occ

    def _update_maps(self, f: Frame, mesh, occ, target_fbo, complete=False):
        """Shadow map, the under-body height map and the sky-occlusion maps, re-rendered when the occluders change.
        The sky maps are spread over frames for very large meshes (self.pending says more frames are needed)."""
        if not self.fbo_ok or (mesh is None and (occ is None or not occ['count'])):
            self.pending = False
            return
        key = (id(mesh[0]) if mesh is not None else None, occ['key'] if occ is not None else None)
        boxes = []
        if mesh is not None:
            boxes.append((mesh[5], mesh[6]))
        if occ is not None and occ['count']:
            boxes.append((occ['lo'], occ['hi']))
        # bounds snapped outward to 10 cm: small moves keep the same map frusta (cached; no shimmering while dragging)
        lo = np.floor(np.min([b[0] for b in boxes], axis=0).astype(float) * 10) / 10
        hi = np.ceil(np.max([b[1] for b in boxes], axis=0).astype(float) * 10) / 10
        lo_g = np.minimum(lo, [lo[0], lo[1], 0.0])
        drag = False
        if self._maps_key != key:
            box = (tuple(lo), tuple(hi))
            if getattr(self, '_frusta', (None,))[0] != box:
                self._frusta = (box, shadow_frustum(lo_g, hi), sky_matrices(lo, hi, sky_dirs()))
            VP, (texel, depth_m) = self._frusta[1]
            if not self._depth_pass('_shadow', SHADOW_N, VP, mesh, occ, target_fbo):
                self._shadow = None
            else:
                self._shadow.update(texel=texel, depth_m=depth_m)
            # the contact shadow is the body's own (rig parts and housings sit high): only a new mesh redoes it
            if mesh is not None and (self._maps_key is None or self._maps_key[0] != key[0] or self._ao is None):
                mlo = np.minimum(mesh[5], [mesh[5][0], mesh[5][1], 0.0])
                VPa, nf = ao_matrix(mlo, mesh[6])
                if not self._depth_pass('_ao', AO_N, VPa, mesh, None, target_fbo):
                    self._ao = None
                else:
                    self._ao['nf'] = nf
                    if not self._bake_contact(target_fbo, ao_extent(mlo, mesh[6])):
                        self._ao = None
            drag = self._maps_key is not None and self._maps_key[0] == key[0]     # only the TriDraw occluders moved
            self._maps_key = key
            self._sky_target = self._frusta[2]
            if self._sky is not None and 'dirty' in self._sky:
                self._sky['dirty'][:] = True
        self._sky_pass(mesh, occ, target_fbo, complete, SKY_DRAG_TILES if drag else SKY_DIRS)

    def _bake_contact(self, target_fbo, extent) -> bool:
        """The ground's contact occlusion under the body into a texture, once per geometry change: where the body's
        underside is low (from the height map), blurred with a Gaussian of CONTACT_SIGMA metres in two separable
        passes, so it is smooth (no tap patterns) and the ground reads it with one filtered lookup."""
        GL = self.GL
        d = self._ao
        if d.get('ctex') is None:
            d['ctex'], d['cfbo'], d['cok'] = [], [], True
            for _ in range(2):
                t = self._texture(np.full((AO_N, AO_N, 4), 255, np.uint8))
                fb = GL.glGenFramebuffers(1)
                GL.glBindFramebuffer(GL.GL_FRAMEBUFFER, fb)
                GL.glFramebufferTexture2D(GL.GL_FRAMEBUFFER, GL.GL_COLOR_ATTACHMENT0, GL.GL_TEXTURE_2D, t, 0)
                d['cok'] &= GL.glCheckFramebufferStatus(GL.GL_FRAMEBUFFER) == GL.GL_FRAMEBUFFER_COMPLETE
                d['ctex'].append(t)
                d['cfbo'].append(fb)
        if not d['cok']:
            GL.glBindFramebuffer(GL.GL_FRAMEBUFFER, target_fbo)
            return False
        GL.glViewport(0, 0, AO_N, AO_N)
        GL.glDisable(GL.GL_DEPTH_TEST)
        GL.glDisable(GL.GL_BLEND)
        GL.glDisable(GL.GL_SCISSOR_TEST)
        GL.glDisable(GL.GL_MULTISAMPLE)
        p = self.prog['contact']
        GL.glUseProgram(p['id'])
        q = np.array([[-1, -1, 0], [1, -1, 0], [1, 1, 0], [-1, -1, 0], [1, 1, 0], [-1, 1, 0]], np.float32)
        GL.glEnableVertexAttribArray(0)
        GL.glVertexAttribPointer(0, 3, GL.GL_FLOAT, GL.GL_FALSE, 0, q)
        GL.glActiveTexture(GL.GL_TEXTURE2)
        sig = np.array([CONTACT_SIGMA / max(extent[0], 1e-3), CONTACT_SIGMA / max(extent[1], 1e-3)]) * 3.0 / 16.0
        for k, (src, dst, step) in enumerate(((d['tex'], 0, (0.0, 0.0)), (d['ctex'][0], 1, (sig[0], 0.0)),
                                             (d['ctex'][1], 0, (0.0, sig[1])))):
            GL.glBindFramebuffer(GL.GL_FRAMEBUFFER, d['cfbo'][dst])
            GL.glBindTexture(GL.GL_TEXTURE_2D, src)
            self._set(p, 'uAOInfo', (float(k), float(AO_N), d['nf'][0], d['nf'][1]))
            self._set(p, 'uStep', step)
            GL.glDrawArrays(GL.GL_TRIANGLES, 0, 6)
        GL.glDisableVertexAttribArray(0)
        GL.glUseProgram(0)
        GL.glBindTexture(GL.GL_TEXTURE_2D, 0)
        GL.glActiveTexture(GL.GL_TEXTURE0)
        GL.glEnable(GL.GL_DEPTH_TEST)
        GL.glEnable(GL.GL_BLEND)
        GL.glEnable(GL.GL_MULTISAMPLE)
        GL.glBindFramebuffer(GL.GL_FRAMEBUFFER, target_fbo)
        return True

    def _sky_pass(self, mesh, occ, target_fbo, complete, cap=SKY_DIRS):
        GL = self.GL
        W, H = SKY_COLS * SKY_TILE, SKY_ROWS * SKY_TILE
        if self._sky is None:
            if max(W, H) > getattr(self, 'max_tex', 4096):
                self._sky = {'ok': False}
                return
            d = self._depth_fbo(W, H, compare=True)
            d.update(next=0, dirty=np.ones(SKY_DIRS, bool), M=np.zeros((SKY_DIRS, 4, 4)), dirs=sky_dirs(), texel=1.0,
                     depth_m=1.0)
            if d['ok']:
                GL.glBindFramebuffer(GL.GL_FRAMEBUFFER, d['fbo'])
                GL.glViewport(0, 0, W, H)
                GL.glClearDepth(1.0)
                GL.glClear(GL.GL_DEPTH_BUFFER_BIT)     # nothing occludes until a tile is drawn
                GL.glBindFramebuffer(GL.GL_FRAMEBUFFER, target_fbo)
            self._sky = d
        d = self._sky
        if not d.get('ok') or not d['dirty'].any():
            self.pending = False
            return
        Mt, texel, depth_m = self._sky_target
        tris = (mesh[3] if mesh is not None else 0) // 3 + (occ['count'] if occ is not None else 0) // 3
        n = SKY_DIRS if complete else max(1, min(cap, SKY_BUDGET // max(tris, 1)))
        GL.glBindFramebuffer(GL.GL_FRAMEBUFFER, d['fbo'])
        GL.glEnable(GL.GL_SCISSOR_TEST)
        GL.glClearDepth(1.0)
        p = self._begin_depth()
        ring = [(d['next'] + k) % SKY_DIRS for k in range(SKY_DIRS)]
        todo = [i for i in ring if d['dirty'][i]][:n]           # round robin: a long drag refreshes every tile
        for i in todo:
            x, y = (i % SKY_COLS) * SKY_TILE, (i // SKY_COLS) * SKY_TILE
            GL.glViewport(x, y, SKY_TILE, SKY_TILE)
            GL.glScissor(x, y, SKY_TILE, SKY_TILE)
            GL.glClear(GL.GL_DEPTH_BUFFER_BIT)
            # Mt maps to [0,1]^3; the depth pass wants clip space of the tile: undo the bias/scale
            clip = np.array([[2, 0, 0, -1], [0, 2, 0, -1], [0, 0, 2, -1], [0, 0, 0, 1.0]]) @ Mt[i]
            self._draw_occluders(p, clip, mesh, occ)
            d['M'][i] = Mt[i]
            d['dirty'][i] = False
        d['next'] = (todo[-1] + 1) % SKY_DIRS
        d['texel'], d['depth_m'] = texel, depth_m
        self._end_depth(target_fbo)
        self.pending = bool(d['dirty'].any())

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

    def render(self, f: Frame, target_fbo=0, complete=False):
        """Draw a frame into target_fbo. complete: finish every cached map now (stills) instead of over frames."""
        GL = self.GL
        self._no_vao()
        GL.glDisable(GL.GL_SCISSOR_TEST)
        mesh = self._mesh_vbo(f.mesh) if f.mesh is not None and len(f.mesh.tri) else None
        st = f.style
        if st.shadows or st.ao:
            self._update_maps(f, mesh, self._occluders(f), target_fbo, complete)
        else:
            self.pending = False
        GL.glBindFramebuffer(GL.GL_FRAMEBUFFER, target_fbo)
        GL.glViewport(0, 0, int(f.width), int(f.height))
        if f.clear is not None:
            GL.glClearColor(*f.clear)
            GL.glClear(GL.GL_COLOR_BUFFER_BIT | GL.GL_DEPTH_BUFFER_BIT)
        else:
            GL.glClear(GL.GL_DEPTH_BUFFER_BIT)
        VP = (f.P @ f.V).astype(np.float32)
        Vi = np.linalg.inv(f.V)
        self._eye = Vi[:3, 3]
        self._view_dir = Vi[:3, 2] / max(np.linalg.norm(Vi[:3, 2]), 1e-9)
        self._rig = light_rig(st)
        self._frame_size = (int(f.width), int(f.height))
        GL.glEnable(GL.GL_BLEND)
        GL.glBlendFuncSeparate(GL.GL_SRC_ALPHA, GL.GL_ONE_MINUS_SRC_ALPHA, GL.GL_ONE, GL.GL_ONE_MINUS_SRC_ALPHA)
        GL.glEnable(GL.GL_DEPTH_TEST)
        GL.glDepthFunc(GL.GL_LEQUAL)
        GL.glDisable(GL.GL_CULL_FACE)
        GL.glEnable(GL.GL_MULTISAMPLE)
        try:
            self._msaa = int(GL.glGetIntegerv(GL.GL_SAMPLE_BUFFERS)) > 0
            self._stencil = int(GL.glGetIntegerv(GL.GL_STENCIL_BITS)) > 0 and not self.no_stencil
        except Exception:
            self._msaa, self._stencil = False, False
        # ground plane (no depth write, so lines and boards on the ground stay visible)
        if st.ground_plane or st.grid or f.ground is not None:
            self._draw_ground(f, VP)
        self._fixed(f)
        for l in f.lines:
            if l.depth and _on_ground(l):
                self._draw_line(l)
        if f.bowl is not None:
            self._draw_bowl(f, VP)
        solid_tris = [td for td in f.tris if not _translucent(td)]
        clear_tris = [td for td in f.tris if _translucent(td)]
        if mesh is not None:
            self._draw_mesh(f, VP, mesh, 'opaque')
        for td in solid_tris:
            self._draw_flat(f, VP, td)
        for pd in f.points:
            if pd.depth:
                self._draw_points(VP, pd)
        if mesh is not None:
            self._draw_mesh(f, VP, mesh, 'translucent')
        for td in clear_tris:
            self._draw_flat(f, VP, td)
        self._fixed(f)
        for l in f.lines:
            if not (l.depth and _on_ground(l)):
                self._draw_line(l)
        for pd in f.points:
            if not pd.depth:
                self._draw_points(VP, pd)
        self._gc_points()
        # leave a clean state for QPainter
        GL.glUseProgram(0)
        GL.glBindBuffer(GL.GL_ARRAY_BUFFER, 0)
        GL.glBindBuffer(GL.GL_ELEMENT_ARRAY_BUFFER, 0)
        for unit in (4, 3, 2, 1, 0):
            GL.glActiveTexture(GL.GL_TEXTURE0 + unit)
            GL.glBindTexture(GL.GL_TEXTURE_2D, 0)
        GL.glDisable(GL.GL_DEPTH_TEST)
        GL.glDisable(GL.GL_STENCIL_TEST)
        GL.glDepthMask(GL.GL_TRUE)
        GL.glBlendFunc(GL.GL_SRC_ALPHA, GL.GL_ONE_MINUS_SRC_ALPHA)
        _default_pixel_store(GL)
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
        self._set(p, 'uGrid', (1.0 if st.grid else 0.0, st.grid_step[0], st.grid_step[1], 1.0 if st.axes else 0.0))
        self._set(p, 'uGridC1', st.grid_minor)
        self._set(p, 'uGridC2', st.grid_major)
        self._set(p, 'uOrigin', st.origin)
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
            self._set(p, 'uTexOn', 1.0)
            self._set(p, 'uExt', g.extent)
            z = g.z
        else:
            self._set(p, 'uTexOn', 0.0)
            z = 0.0
        ao = self._ao if (st.ao and self._ao is not None and self._maps_key is not None) else None
        if ao is not None:
            GL.glActiveTexture(GL.GL_TEXTURE2)
            GL.glBindTexture(GL.GL_TEXTURE_2D, ao['ctex'][0])
            self._set(p, 'uAOVP', ao['VP'])
            self._set(p, 'uAOInfo', (1.0, 0.0, 0.0, 0.0))
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

    def _draw_mesh(self, f, VP, mesh, phase):
        GL = self.GL
        md, vbo, n_opaque, n_occ, n_all = mesh[:5]
        first, count = (0, n_opaque) if phase == 'opaque' else (n_opaque, n_all - n_opaque)
        if count <= 0:
            return
        p = self.prog['mesh']
        GL.glUseProgram(p['id'])
        self._set(p, 'uVP', VP)
        self._set(p, 'uMatOn', 1.0 if f.style.materials else 0.0)
        self._common(p, f)
        GL.glBindBuffer(GL.GL_ARRAY_BUFFER, vbo)
        for i, (n, off) in enumerate(((3, 0), (3, 12), (4, 24))):
            GL.glEnableVertexAttribArray(i)
            GL.glVertexAttribPointer(i, n, GL.GL_FLOAT, GL.GL_FALSE, 44, ctypes.c_void_p(off) if off else None)
        GL.glEnableVertexAttribArray(4)
        GL.glVertexAttribPointer(4, 1, GL.GL_FLOAT, GL.GL_FALSE, 44, ctypes.c_void_p(40))
        GL.glEnable(GL.GL_POLYGON_OFFSET_FILL)
        GL.glPolygonOffset(1.0, 1.0)
        GL.glPolygonMode(GL.GL_FRONT_AND_BACK, GL.GL_LINE if md.wire else GL.GL_FILL)
        if phase == 'opaque' and not md.wire:
            # depth first, then shade only the nearest surface: the lighting runs once per pixel, not per layer
            d = self.prog['depth']
            GL.glUseProgram(d['id'])
            self._set(d, 'uVP', VP)
            GL.glColorMask(GL.GL_FALSE, GL.GL_FALSE, GL.GL_FALSE, GL.GL_FALSE)
            GL.glDepthMask(GL.GL_TRUE)
            GL.glDrawArrays(GL.GL_TRIANGLES, first, count)
            GL.glColorMask(GL.GL_TRUE, GL.GL_TRUE, GL.GL_TRUE, GL.GL_TRUE)
            GL.glUseProgram(p['id'])
            GL.glDepthMask(GL.GL_FALSE)
        else:
            GL.glDepthMask(GL.GL_TRUE if phase == 'opaque' else GL.GL_FALSE)
        GL.glDrawArrays(GL.GL_TRIANGLES, first, count)
        GL.glDepthMask(GL.GL_TRUE)
        GL.glPolygonMode(GL.GL_FRONT_AND_BACK, GL.GL_FILL)
        GL.glDisable(GL.GL_POLYGON_OFFSET_FILL)
        for i in (0, 1, 2, 4):
            GL.glDisableVertexAttribArray(i)
        GL.glBindBuffer(GL.GL_ARRAY_BUFFER, 0)

    def _draw_flat(self, f, VP, td):
        GL = self.GL
        if not len(td.tri):
            return
        v = np.ascontiguousarray(np.asarray(td.tri).reshape(-1, 3), np.float32)
        outline = getattr(td, 'outline', None)
        if outline is not None and not self._stencil:
            self._draw_outline(f, VP, td, v, outline, stencil=False)      # under the object: it covers the inside
        p = self.prog['flat']
        GL.glUseProgram(p['id'])
        self._set(p, 'uVP', VP)
        translucent = _translucent(td)
        self._set(p, 'uLit', 0.0 if translucent else 1.0)
        self._common(p, f)
        c = np.ascontiguousarray(np.repeat(np.asarray(td.colors, np.float32), 3, axis=0))
        nrm = getattr(td, 'normals', None)
        smooth = nrm is not None and len(nrm) == len(td.tri)
        self._set(p, 'uSmooth', 1.0 if smooth else 0.0)
        if smooth:
            n = np.ascontiguousarray(np.asarray(nrm, np.float32).reshape(-1, 3))
            GL.glEnableVertexAttribArray(1)
            GL.glVertexAttribPointer(1, 3, GL.GL_FLOAT, GL.GL_FALSE, 0, n)
        if td.depth:
            GL.glEnable(GL.GL_DEPTH_TEST)
        else:
            GL.glDisable(GL.GL_DEPTH_TEST)
        GL.glDepthMask(GL.GL_FALSE if translucent else GL.GL_TRUE)
        if outline is not None and self._stencil:
            GL.glStencilMask(0xFF)                                    # clears honour the write mask
            GL.glClearStencil(0)
            GL.glClear(GL.GL_STENCIL_BUFFER_BIT)
            GL.glEnable(GL.GL_STENCIL_TEST)
            GL.glStencilFunc(GL.GL_ALWAYS, 1, 0xFF)
            GL.glStencilOp(GL.GL_KEEP, GL.GL_KEEP, GL.GL_REPLACE)     # the object's visible pixels
        GL.glEnableVertexAttribArray(0)
        GL.glEnableVertexAttribArray(2)
        GL.glVertexAttribPointer(0, 3, GL.GL_FLOAT, GL.GL_FALSE, 0, v)
        GL.glVertexAttribPointer(2, 4, GL.GL_FLOAT, GL.GL_FALSE, 0, c)
        GL.glDrawArrays(GL.GL_TRIANGLES, 0, len(v))
        GL.glDisableVertexAttribArray(0)
        GL.glDisableVertexAttribArray(2)
        if smooth:
            GL.glDisableVertexAttribArray(1)
        GL.glDepthMask(GL.GL_TRUE)
        GL.glEnable(GL.GL_DEPTH_TEST)
        if outline is not None and self._stencil:
            self._draw_outline(f, VP, td, v, outline, stencil=True)

    def _draw_outline(self, f, VP, td, v, color, stencil):
        """A crisp outline of constant screen width around the object's visible silhouette: the object drawn in the
        outline colour at small screen offsets, only where the object itself is not (stencil), each pixel once."""
        GL = self.GL
        p = self.prog['outline']
        GL.glUseProgram(p['id'])
        self._set(p, 'uVP', VP)
        rgba = tuple(float(x) for x in color) + ((1.0,) if len(color) == 3 else ())
        self._set(p, 'uColor', rgba[:4])
        w, h = self._frame_size
        width = max(1.0, float(getattr(f.style, 'outline_px', 2.0)) * float(self.dpr or 1.0))
        if td.depth:
            GL.glEnable(GL.GL_DEPTH_TEST)
        else:
            GL.glDisable(GL.GL_DEPTH_TEST)
        GL.glDepthMask(GL.GL_FALSE)
        if stencil:
            GL.glEnable(GL.GL_STENCIL_TEST)
            GL.glStencilFunc(GL.GL_EQUAL, 0, 0xFF)
            GL.glStencilOp(GL.GL_KEEP, GL.GL_KEEP, GL.GL_INCR)       # drawn once per pixel: no double blending
        GL.glEnableVertexAttribArray(0)
        GL.glVertexAttribPointer(0, 3, GL.GL_FLOAT, GL.GL_FALSE, 0, v)
        for dx, dy in outline_offsets(width):
            self._set(p, 'uOff', (2.0 * dx / max(w, 1), 2.0 * dy / max(h, 1)))
            GL.glDrawArrays(GL.GL_TRIANGLES, 0, len(v))
        GL.glDisableVertexAttribArray(0)
        GL.glDepthMask(GL.GL_TRUE)
        GL.glEnable(GL.GL_DEPTH_TEST)
        if stencil:
            GL.glDisable(GL.GL_STENCIL_TEST)
            GL.glClear(GL.GL_STENCIL_BUFFER_BIT)      # QPainter keeps its own idea of the stencil: leave it clear

    def _point_buffer(self, pd):
        """GPU buffer for a large point cloud, kept while the same arrays are passed (scenes cache their sweeps)."""
        GL = self.GL
        key = (id(pd.points), id(pd.colors), len(pd.points))
        hit = self._points.get(key)
        if hit is not None:
            hit[3] = True
            return hit
        data = np.ascontiguousarray(np.c_[np.asarray(pd.points, np.float32).reshape(-1, 3),
                                          np.asarray(pd.colors, np.float32).reshape(-1, 4)], np.float32)
        vbo = GL.glGenBuffers(1)
        GL.glBindBuffer(GL.GL_ARRAY_BUFFER, vbo)
        GL.glBufferData(GL.GL_ARRAY_BUFFER, data.nbytes, data, GL.GL_STATIC_DRAW)
        GL.glBindBuffer(GL.GL_ARRAY_BUFFER, 0)
        hit = [vbo, len(data), (pd.points, pd.colors), True]
        self._points[key] = hit
        return hit

    def _gc_points(self):
        """Drop the point buffers not drawn in the last frame."""
        GL = self.GL
        for k in [k for k, v in self._points.items() if not v[3]]:
            GL.glDeleteBuffers(1, [self._points.pop(k)[0]])
        for v in self._points.values():
            v[3] = False

    def _draw_points(self, VP, pd):
        GL = self.GL
        n = len(pd.points)
        if not n:
            return
        p = self.prog['point']
        GL.glUseProgram(p['id'])
        self._set(p, 'uVP', VP)
        self._set(p, 'uSize', max(1.0, float(pd.size) * float(self.dpr or 1.0)))
        GL.glEnable(0x8642)                    # GL_VERTEX_PROGRAM_POINT_SIZE
        GL.glEnable(0x8861)                    # GL_POINT_SPRITE (compatibility profile: gl_PointCoord)
        if pd.depth:
            GL.glEnable(GL.GL_DEPTH_TEST)
        else:
            GL.glDisable(GL.GL_DEPTH_TEST)
        a2c = self._msaa and pd.depth
        if a2c:                                 # order-independent soft edges with depth writes
            GL.glDisable(GL.GL_BLEND)
            GL.glEnable(GL.GL_SAMPLE_ALPHA_TO_COVERAGE)
            GL.glEnable(0x809F)                 # GL_SAMPLE_ALPHA_TO_ONE: covered samples are opaque (premultiplied)
        GL.glDepthMask(GL.GL_TRUE if pd.depth else GL.GL_FALSE)
        GL.glEnableVertexAttribArray(0)
        GL.glEnableVertexAttribArray(2)
        if n >= POINT_VBO_MIN:
            vbo = self._point_buffer(pd)[0]
            GL.glBindBuffer(GL.GL_ARRAY_BUFFER, vbo)
            GL.glVertexAttribPointer(0, 3, GL.GL_FLOAT, GL.GL_FALSE, 28, None)
            GL.glVertexAttribPointer(2, 4, GL.GL_FLOAT, GL.GL_FALSE, 28, ctypes.c_void_p(12))
        else:
            v = np.ascontiguousarray(np.asarray(pd.points, np.float32).reshape(-1, 3))
            c = np.ascontiguousarray(np.asarray(pd.colors, np.float32).reshape(-1, 4))
            GL.glVertexAttribPointer(0, 3, GL.GL_FLOAT, GL.GL_FALSE, 0, v)
            GL.glVertexAttribPointer(2, 4, GL.GL_FLOAT, GL.GL_FALSE, 0, c)
        GL.glDrawArrays(GL.GL_POINTS, 0, n)
        GL.glDisableVertexAttribArray(0)
        GL.glDisableVertexAttribArray(2)
        GL.glBindBuffer(GL.GL_ARRAY_BUFFER, 0)
        if a2c:
            GL.glDisable(GL.GL_SAMPLE_ALPHA_TO_COVERAGE)
            GL.glDisable(0x809F)
            GL.glEnable(GL.GL_BLEND)
        GL.glDisable(0x8861)
        GL.glDisable(0x8642)
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
    def _offscreen_fbos(self, w, h, samples):
        """(multisampled fbo or None, resolve fbo, fbos, renderbuffers); depth with stencil (outlines)."""
        GL = self.GL
        fbos, rbs = [], []

        def make(ms):
            fbo = GL.glGenFramebuffers(1)
            GL.glBindFramebuffer(GL.GL_FRAMEBUFFER, fbo)
            c, d = GL.glGenRenderbuffers(1), GL.glGenRenderbuffers(1)
            for rb, fmt, att in ((c, GL.GL_RGBA8, GL.GL_COLOR_ATTACHMENT0),
                                 (d, GL.GL_DEPTH24_STENCIL8, GL.GL_DEPTH_STENCIL_ATTACHMENT)):
                GL.glBindRenderbuffer(GL.GL_RENDERBUFFER, rb)
                if ms:
                    GL.glRenderbufferStorageMultisample(GL.GL_RENDERBUFFER, ms, fmt, w, h)
                else:
                    GL.glRenderbufferStorage(GL.GL_RENDERBUFFER, fmt, w, h)
                GL.glFramebufferRenderbuffer(GL.GL_FRAMEBUFFER, att, GL.GL_RENDERBUFFER, rb)
            fbos.append(fbo)
            rbs.extend([c, d])
            return fbo, GL.glCheckFramebufferStatus(GL.GL_FRAMEBUFFER) == GL.GL_FRAMEBUFFER_COMPLETE
        ms_fbo, ok = make(samples) if samples > 1 else (None, False)
        if not ok:
            ms_fbo = None
        res_fbo, ok2 = make(0)
        if not ok2:
            raise RuntimeError('framebuffer incomplete')
        return ms_fbo, res_fbo, fbos, rbs

    def render_offscreen(self, f: Frame, samples=4) -> np.ndarray:
        """Render into a temporary framebuffer and return (H, W, 4) uint8 RGBA, premultiplied alpha, row 0 = top."""
        GL = self.GL
        w, h = int(f.width), int(f.height)
        fbos, rbs = [], []
        try:
            ms_fbo, res_fbo, fbos, rbs = self._offscreen_fbos(w, h, samples)
            target = ms_fbo if ms_fbo is not None else res_fbo
            if f.clear is None:
                f.clear = (0, 0, 0, 0)
            self.render(f, target, complete=True)
            if ms_fbo is not None:
                GL.glBindFramebuffer(GL.GL_READ_FRAMEBUFFER, ms_fbo)
                GL.glBindFramebuffer(GL.GL_DRAW_FRAMEBUFFER, res_fbo)
                GL.glBlitFramebuffer(0, 0, w, h, 0, 0, w, h, GL.GL_COLOR_BUFFER_BIT, GL.GL_NEAREST)
            GL.glBindFramebuffer(GL.GL_FRAMEBUFFER, res_fbo)
            data = GL.glReadPixels(0, 0, w, h, GL.GL_RGBA, GL.GL_UNSIGNED_BYTE)
            img = np.frombuffer(data, np.uint8).reshape(h, w, 4)[::-1].copy()
            return img
        finally:
            GL.glBindFramebuffer(GL.GL_FRAMEBUFFER, 0)
            _default_pixel_store(GL)
            if rbs:
                GL.glDeleteRenderbuffers(len(rbs), rbs)
            if fbos:
                GL.glDeleteFramebuffers(len(fbos), fbos)

    def tone_probe(self, lin: np.ndarray, exposure=1.0) -> np.ndarray:
        """The shaders' finish() (exposure, tone curve, sRGB encode; no dither) applied to (N,3) linear colours on the
        GPU, as (N,3) uint8: what a lit surface of that linear radiance comes out as."""
        GL = self.GL
        lin = np.asarray(lin, np.float32).reshape(-1, 3)
        n = len(lin)
        fbos, rbs = [], []
        try:
            _, res_fbo, fbos, rbs = self._offscreen_fbos(n, 1, 0)
            GL.glBindFramebuffer(GL.GL_FRAMEBUFFER, res_fbo)
            GL.glViewport(0, 0, n, 1)
            GL.glDisable(GL.GL_BLEND)
            GL.glDisable(GL.GL_DEPTH_TEST)
            GL.glDisable(GL.GL_MULTISAMPLE)
            p = self.prog['probe']
            GL.glUseProgram(p['id'])
            self._set(p, 'uTone', (exposure, TONE_KNEE, 0.0, 1.0))
            pos = np.ascontiguousarray(np.c_[(np.arange(n) + 0.5) / n * 2 - 1, np.zeros(n), np.zeros(n)], np.float32)
            col = np.ascontiguousarray(np.c_[lin, np.ones(n)], np.float32)
            GL.glEnableVertexAttribArray(0)
            GL.glEnableVertexAttribArray(2)
            GL.glVertexAttribPointer(0, 3, GL.GL_FLOAT, GL.GL_FALSE, 0, pos)
            GL.glVertexAttribPointer(2, 4, GL.GL_FLOAT, GL.GL_FALSE, 0, col)
            GL.glEnable(0x8642)
            GL.glDrawArrays(GL.GL_POINTS, 0, n)
            GL.glDisable(0x8642)
            GL.glDisableVertexAttribArray(0)
            GL.glDisableVertexAttribArray(2)
            GL.glUseProgram(0)
            data = GL.glReadPixels(0, 0, n, 1, GL.GL_RGBA, GL.GL_UNSIGNED_BYTE)
            GL.glEnable(GL.GL_BLEND)
            return np.frombuffer(data, np.uint8).reshape(n, 4)[:, :3].copy()
        finally:
            GL.glBindFramebuffer(GL.GL_FRAMEBUFFER, 0)
            _default_pixel_store(GL)
            if rbs:
                GL.glDeleteRenderbuffers(len(rbs), rbs)
            if fbos:
                GL.glDeleteFramebuffers(len(fbos), fbos)

    def release(self):
        GL = self.GL
        if GL is None:
            return
        try:
            for name in ('_shadow', '_ao', '_sky'):
                d = getattr(self, name)
                if d and d.get('fbo'):
                    GL.glDeleteFramebuffers(1, [d['fbo']])
                    GL.glDeleteTextures([d['tex']])
                if d and d.get('cfbo'):
                    GL.glDeleteFramebuffers(len(d['cfbo']), d['cfbo'])
                    GL.glDeleteTextures(d['ctex'])
            if self._mesh:
                GL.glDeleteBuffers(1, [self._mesh[1]])
            if self._occ and self._occ['vbo']:
                GL.glDeleteBuffers(1, [self._occ['vbo']])
            for v in self._points.values():
                GL.glDeleteBuffers(1, [v[0]])
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
        self._mesh = self._bowl = self._ground_tex = self._cube = self._shadow = self._ao = self._sky = None
        self._occ = None
        self._points = {}
        self._maps_key = None
        self.ok = False


def _default_pixel_store(GL):
    """GL's default row alignment of 4: PyOpenGL sets 1 on its image calls, and QPainter's glyph cache uploads rows
    assuming 4 (with 1 every label drawn after the 3D view comes out sheared)."""
    GL.glPixelStorei(GL.GL_PACK_ALIGNMENT, 4)
    GL.glPixelStorei(GL.GL_UNPACK_ALIGNMENT, 4)


def _on_ground(l):
    pts = l.pts.reshape(-1, 3)
    return len(pts) and float(np.abs(pts[:, 2]).max()) < 0.03


def _translucent(td) -> bool:
    c = np.asarray(td.colors)
    return bool(len(c) and (c[:, 3] < 0.99).any())
