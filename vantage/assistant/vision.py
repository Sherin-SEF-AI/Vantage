"""Pictures of the design for the assistant's model (the look tool): drawn headless from core results, never from the
main window, so the model sees exactly what the analysis computed.

Three views:
- top: the bird's-eye coverage map of the ROI ground grid at the analysis height: blind cells, cells below the
  minimum density, and the rest shaded by the best camera's pixel density (or, as the redundancy layer, by how many
  cameras see them), with the vehicle outline, every camera's tag and heading, the largest blind patches numbered,
  a scale bar and a legend. The front of the vehicle is at the top and its left on the left of the picture.
- closeup: the same map around one ground point (x, y), larger, with the cell grid and the point marked.
- camera: one camera's synthetic view of the parking scene (core.synth.render_world: the same lens model, pose and
  body occlusion as the analysis). A synthetic scene, never a capture.

Each picture comes with a caption: what is drawn and the key figures, read from the analysis result (the same numbers
the read tools give), so the model can tie the picture to the figures and quote the figures rather than read pixels.

Size: every picture fits in MAX_EDGE px on its long edge and MAX_PIXELS in all (about 1.2 megapixels), PNG for the
maps (flat colours compress well) and JPEG for camera views; image_tokens() is the provider's estimate
(width × height / 750, Anthropic's documented rule of thumb for image tokens), a little over 1,600 tokens at the cap.

Privacy: a picture is project data like any tool result. It is drawn from the design (no photo, recording or vehicle
mesh file is sent; the map is the analysis grid and the outline its body footprint) and leaves the machine only inside
a question the user sends to a provider that takes images (Claude), after the consent that provider asks for."""
from __future__ import annotations

import base64
import math
from dataclasses import dataclass, field

import numpy as np

VIEWS = ('top', 'closeup', 'camera')
LAYERS = ('density', 'redundancy')
MAX_EDGE = 1280                 # px, the long edge
MAX_PIXELS = 1_250_000          # about 1.2 megapixels
TOKEN_PIXELS = 750.0            # Anthropic: image tokens ≈ width × height / 750
JPEG_QUALITY = 85

# colours (RGB); the picture uses a light background so thin lines and text stay legible after compression
BG = (250, 250, 248)
OUTSIDE = (236, 236, 232)       # ground outside the ROI
BODY = (92, 94, 100)            # under the body
OUTLINE = (20, 20, 22)
BLIND = (214, 39, 40)
BELOW = (245, 158, 11)
INK = (24, 24, 27)
MUTED = (110, 110, 118)
RAMP = ((198, 232, 214), (122, 197, 170), (54, 152, 148), (24, 99, 128), (16, 52, 94))   # density, low to high
REDUNDANCY = ((186, 220, 245), (110, 170, 225), (45, 110, 190), (20, 55, 125))           # seen by 1, 2, 3, 4+


@dataclass
class Picture:
    """One rendered picture: encoded bytes for the model, its size, a caption and the figures it shows."""
    view: str
    media_type: str
    data: bytes
    width: int
    height: int
    caption: str
    info: dict = field(default_factory=dict)
    transform: tuple | None = None          # top/closeup: (x_max, y_max, px_per_m, left, top) for world_to_px

    @property
    def tokens(self) -> int:
        return image_tokens(self.width, self.height)

    def b64(self) -> str:
        return base64.standard_b64encode(self.data).decode('ascii')

    def world_to_px(self, x, y):
        """(column, row) of a ground point in the picture (top and closeup views)."""
        if self.transform is None:
            raise ValueError('only the map views have a ground mapping')
        xm, ym, s, left, top = self.transform
        return left + (ym - float(y)) * s, top + (xm - float(x)) * s

    def summary(self) -> dict:
        """What the tool result says about the picture itself."""
        return {'view': self.view, 'format': self.media_type, 'width_px': self.width, 'height_px': self.height,
                'bytes': len(self.data), 'approx_image_tokens': self.tokens}


def image_tokens(w, h) -> int:
    return int(math.ceil(float(w) * float(h) / TOKEN_PIXELS))


def fit(w, h, max_edge=MAX_EDGE, max_pixels=MAX_PIXELS):
    """The largest (w, h) with the same aspect within the limits."""
    s = min(1.0, max_edge / max(w, h, 1), math.sqrt(max_pixels / max(w * h, 1)))
    return max(1, int(w * s)), max(1, int(h * s))


def encode(rgb: np.ndarray, fmt='png') -> tuple[bytes, str]:
    import cv2
    bgr = np.ascontiguousarray(rgb[..., ::-1])
    if fmt == 'jpeg':
        ok, buf = cv2.imencode('.jpg', bgr, [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])
        return buf.tobytes(), 'image/jpeg'
    ok, buf = cv2.imencode('.png', bgr, [cv2.IMWRITE_PNG_COMPRESSION, 6])
    return buf.tobytes(), 'image/png'


def ascii_text(s) -> str:
    """OpenCV's fonts draw ASCII only."""
    s = str(s).replace('²', '2').replace('°', ' deg').replace('–', '-').replace('—', '-').replace('−', '-')
    s = s.replace('×', 'x').replace('·', '-').replace('≥', '>=').replace('≤', '<=').replace('…', '...')
    return s.encode('ascii', 'replace').decode('ascii')


class _Canvas:
    def __init__(self, img):
        import cv2
        self.cv2, self.img = cv2, img

    def text(self, t, x, y, col=INK, s=0.45, th=1, centre=False, right=False, box=None):
        cv2 = self.cv2
        t = ascii_text(t)
        (tw, tht), base = cv2.getTextSize(t, cv2.FONT_HERSHEY_SIMPLEX, s, th)
        if right:
            x -= tw
        elif centre:
            x -= tw / 2
        if box is not None:
            cv2.rectangle(self.img, (int(x) - 3, int(y) - tht - 3), (int(x) + tw + 3, int(y) + base + 1), box, -1)
        cv2.putText(self.img, t, (int(x), int(y)), cv2.FONT_HERSHEY_SIMPLEX, s, col, th, cv2.LINE_AA)
        return tw

    def line(self, x0, y0, x1, y1, col=INK, th=1):
        self.cv2.line(self.img, (int(round(x0)), int(round(y0))), (int(round(x1)), int(round(y1))), col, th,
                      self.cv2.LINE_AA)

    def rect(self, x0, y0, x1, y1, col, fill=True, th=1):
        self.cv2.rectangle(self.img, (int(round(x0)), int(round(y0))), (int(round(x1)), int(round(y1))), col,
                           -1 if fill else th)

    def dot(self, x, y, col, r=3, fill=True, th=1):
        self.cv2.circle(self.img, (int(round(x)), int(round(y))), int(r), col, -1 if fill else th, self.cv2.LINE_AA)

    def poly(self, pts, col, th=1, closed=False):
        P = np.round(np.asarray(pts, float)).astype(np.int32).reshape(-1, 1, 2)
        if len(P) >= 2:
            self.cv2.polylines(self.img, [P], closed, col, th, self.cv2.LINE_AA)


def _ramp(t):
    """t in 0..1 -> RGB along RAMP (vectorised)."""
    t = np.clip(np.asarray(t, float), 0.0, 1.0) * (len(RAMP) - 1)
    i = np.minimum(t.astype(int), len(RAMP) - 2)
    f = (t - i)[..., None]
    R = np.asarray(RAMP, float)
    return R[i] * (1 - f) + R[i + 1] * f


def cell_colours(result, project, layer='density') -> np.ndarray:
    """(ny, nx, 3) uint8: the colour of every grid cell, as the map draws it."""
    g = result.grid
    mind = float(project.analysis.min_density)
    tgt = float(project.analysis.target_density)
    out = np.empty((g.ny, g.nx, 3), float)
    out[:] = OUTSIDE
    out[g.body] = BODY
    roi = g.roi
    count = np.asarray(result.count)
    dens = np.asarray(result.best_dens, float)
    seen = roi & (count > 0)
    if layer == 'redundancy':
        k = np.clip(count, 1, 4) - 1
        out[seen] = np.asarray(REDUNDANCY, float)[k[seen]]
    else:
        hi = max(2.0 * tgt, mind + 1e-6)
        t = (dens - mind) / (hi - mind)
        good = seen & (dens >= mind)
        out[good] = _ramp(t[good])
        out[seen & (dens < mind)] = BELOW
    out[roi & (count == 0)] = BLIND
    return out.astype(np.uint8)


def blind_patches(result, project, limit=6) -> list:
    """The largest connected blind patches of the ROI (8-connected), largest first: area, centre, where."""
    from scipy import ndimage
    g = result.grid
    mask = g.roi & (np.asarray(result.count) == 0)
    lab, n = ndimage.label(mask, structure=np.ones((3, 3)))
    if not n:
        return []
    sizes = np.bincount(lab.ravel())[1:]
    out = []
    try:
        from ..core.analysis import describe_location
    except ImportError:                                 # pragma: no cover
        describe_location = None
    for k in np.argsort(-sizes)[:limit]:
        jj, ii = np.nonzero(lab == k + 1)
        x, y = float(g.xs[ii].mean()), float(g.ys[jj].mean())
        where = ''
        if describe_location is not None:
            try:
                where = describe_location(x, y, project)
            except Exception:                           # noqa: BLE001 — a vehicle without its measures
                where = ''
        out.append({'n': len(out) + 1, 'area_m2': round(float(sizes[k]) * g.cell ** 2, 4),
                    'centre': [round(x, 2), round(y, 2)], 'where': where})
    return out


def _nice_bar(m_per_px, target_px=140):
    """A scale-bar length in metres (1, 2 or 5 × 10^n) near target_px long."""
    raw = target_px * m_per_px
    mag = 10 ** math.floor(math.log10(raw))
    return next(v * mag for v in (5, 2, 1) if v * mag <= raw) if raw >= mag else mag


# ====================================================================================================== maps
def render_map(project, result, layer='density', centre=None, radius_m=None) -> Picture:
    """The coverage map (top view), or with centre=(x, y) and radius_m a close-up around that ground point."""
    if result is None:
        raise ValueError('no analysis result yet')
    layer = layer if layer in LAYERS else 'density'
    g = result.grid
    cols = cell_colours(result, project, layer)
    # the window in grid cells
    if centre is not None:
        cx, cy = float(centre[0]), float(centre[1])
        r = float(min(max(radius_m or 1.0, 0.25), 5.0))
        i0 = int(math.floor((cx - r - g.x0) / g.cell))
        i1 = int(math.ceil((cx + r - g.x0) / g.cell))
        j0 = int(math.floor((cy - r - g.y0) / g.cell))
        j1 = int(math.ceil((cy + r - g.y0) / g.cell))
    else:
        i0, i1, j0, j1 = 0, g.nx, 0, g.ny
    i0, i1 = max(0, i0), min(g.nx, i1)
    j0, j1 = max(0, j0), min(g.ny, j1)
    if i1 <= i0 or j1 <= j0:
        raise ValueError('the point is outside the analysis grid')
    nxw, nyw = i1 - i0, j1 - j0
    # picture layout: a title band, the map, a legend band; front up (rows from x), left on the left (cols from y)
    pad, top_band, legend_band = 16, 46, 92
    budget_w, budget_h = MAX_EDGE - 2 * pad, MAX_EDGE - top_band - legend_band - pad
    k = max(1, int(min(budget_w / nyw, budget_h / nxw, 24)))
    while k > 1 and (nyw * k + 2 * pad) * (nxw * k + top_band + legend_band + pad) > MAX_PIXELS:
        k -= 1
    mw, mh = nyw * k, nxw * k
    W = max(mw + 2 * pad, 560)
    H = mh + top_band + legend_band + pad
    img = np.empty((H, W, 3), np.uint8)
    img[:] = BG
    left, top = (W - mw) // 2, top_band
    block = cols[j0:j1, i0:i1]                              # (nyw, nxw, 3) indexed [j, i]
    view = block.transpose(1, 0, 2)[::-1, ::-1]             # rows: x descending, cols: y descending
    img[top:top + mh, left:left + mw] = np.repeat(np.repeat(view, k, 0), k, 1)
    x_max, y_max = g.x0 + i1 * g.cell, g.y0 + j1 * g.cell
    s = k / g.cell                                          # px per metre
    pic = Picture('closeup' if centre is not None else 'top', 'image/png', b'', W, H, '', transform=(
        x_max, y_max, s, left, top))
    P = lambda x, y: pic.world_to_px(x, y)                  # noqa: E731
    cv = _Canvas(img)
    import cv2
    clip = (left, top, left + mw, top + mh)

    def inside(px, py, m=0):
        return clip[0] - m <= px <= clip[2] + m and clip[1] - m <= py <= clip[3] + m
    # the cell grid in a close-up
    if centre is not None and k >= 10:
        for c in range(0, mw + 1, k):
            cv.line(left + c, top, left + c, top + mh, (255, 255, 255), 1)
        for rr in range(0, mh + 1, k):
            cv.line(left, top + rr, left + mw, top + rr, (255, 255, 255), 1)
    # the vehicle outline (the body footprint the analysis uses) and the ROI boundary
    sub = img[top:top + mh, left:left + mw]
    off = np.array([left, top], float)
    for poly in getattr(g, 'outline', []) or []:
        pts = np.array([P(x, y) for x, y in np.asarray(poly, float)[:, :2]]) - off
        cv2.polylines(sub, [np.round(pts).astype(np.int32).reshape(-1, 1, 2)], False, OUTLINE, 2, cv2.LINE_AA)
    for poly in getattr(g, 'boundary', []) or []:
        pts = np.array([P(x, y) for x, y in np.asarray(poly, float)[:, :2]]) - off
        cv2.polylines(sub, [np.round(pts).astype(np.int32).reshape(-1, 1, 2)], False, MUTED, 1, cv2.LINE_AA)
    # blind patches, numbered beside them with a leader line (the cells themselves stay visible)
    patches = blind_patches(result, project)
    mx, my = P(*result.centre) if getattr(result, 'centre', None) is not None else (left + mw / 2, top + mh / 2)
    for b in patches:
        px, py = P(*b['centre'])
        if not inside(px, py):
            continue
        dx, dy = px - mx, py - my
        n = math.hypot(dx, dy) or 1.0
        lx, ly = px + dx / n * 30, py + dy / n * 30
        cv.line(px + dx / n * 6, py + dy / n * 6, lx, ly, BLIND, 1)
        cv.dot(lx, ly, (255, 255, 255), 10)
        cv.dot(lx, ly, BLIND, 10, fill=False, th=2)
        cv.text(str(b['n']), lx, ly + 5, BLIND, 0.45, 1, centre=True)
    # cameras: a ring, the heading, the tag
    for c in project.variant.cameras:
        if not c.visible:
            continue
        px, py = P(c.x, c.y)
        if not inside(px, py, 4):
            continue
        a = math.radians(c.yaw)
        ex, ey = P(c.x + math.cos(a) * 28 / s, c.y + math.sin(a) * 28 / s)
        dx, dy = ex - px, ey - py
        cv.line(px + dx * 0.25, py + dy * 0.25, ex, ey, INK, 2)
        cv.dot(px, py, INK, 4, fill=False, th=2)
        tx = px + (10 if ex <= px else -10)
        cv.text(c.tag, tx, py - 10, INK, 0.5, 1, right=ex > px, box=(255, 255, 255))
    # the marked point of a close-up: four ticks round it, its cell left visible
    if centre is not None:
        px, py = P(cx, cy)
        for ux, uy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            cv.line(px + ux * 14, py + uy * 14, px + ux * 30, py + uy * 30, INK, 2)
    # title band
    m = result.metrics
    var = getattr(project.variant, 'name', '')
    if centre is not None:
        title = f'Close-up around ({cx:.2f}, {cy:.2f}) m, {2 * r:.1f} m wide | {var}'
    else:
        title = f'Coverage map of the ground at z = {float(m.get("z", 0.0)):.2f} m | {var}'
    cv.text(title, pad, 22, INK, 0.55, 1)
    cv.text('FRONT ^  (x forward up, y left to the left)', pad, 40, MUTED, 0.4, 1)
    # scale bar
    bar = _nice_bar(1.0 / s)
    bx, by = W - pad - bar * s, 36
    cv.line(bx, by, bx + bar * s, by, INK, 3)
    cv.line(bx, by - 5, bx, by + 5, INK, 2)
    cv.line(bx + bar * s, by - 5, bx + bar * s, by + 5, INK, 2)
    cv.text(f'{bar:g} m', bx + bar * s / 2, by - 9, INK, 0.42, 1, centre=True)
    # legend band
    ly = top + mh + 26
    x = pad
    mind, tgt = float(project.analysis.min_density), float(project.analysis.target_density)
    if layer == 'redundancy':
        items = [(BLIND, 'blind (no camera)')] + [(REDUNDANCY[i], f'seen by {i + 1}' + ('+' if i == 3 else ''))
                                                   for i in range(4)]
    else:
        items = [(BLIND, 'blind (no camera)'), (BELOW, f'below min density {mind:g} px/cm')]
    items += [(BODY, 'under the body'), (OUTSIDE, 'outside the ROI')]
    for col, text in items:
        cv.rect(x, ly - 11, x + 16, ly + 3, col)
        cv.rect(x, ly - 11, x + 16, ly + 3, MUTED, fill=False)
        x += 22 + cv.text(text, x + 22, ly, INK, 0.4, 1) + 18
        if x > W - 200:
            x, ly = pad, ly + 22
    if layer == 'density':
        ly2 = top + mh + 70
        rw = 220
        t = np.linspace(0, 1, rw)
        strip = _ramp(t).astype(np.uint8)
        img[ly2 - 10:ly2 + 2, pad:pad + rw] = strip[None]
        cv.rect(pad, ly2 - 10, pad + rw, ly2 + 2, MUTED, fill=False)
        cv.text(f'best camera density: {mind:g} px/cm', pad, ly2 + 16, MUTED, 0.38, 1)
        cv.text(f'{2 * tgt:g}+', pad + rw, ly2 + 16, MUTED, 0.38, 1, right=True)
        cv.text('circled numbers: blind patches (largest first)  |  line from a camera: its heading', pad + rw + 24,
                ly2, MUTED, 0.38, 1)
    else:
        cv.text('circled numbers: blind patches (largest first)  |  line from a camera: its heading', pad,
                top + mh + 70, MUTED, 0.38, 1)
    pic.data, pic.media_type = encode(img, 'png')
    # caption and figures (all read from the result)
    win_cells = (slice(j0, j1), slice(i0, i1))
    roi_w = g.roi[win_cells]
    cnt_w = np.asarray(result.count)[win_cells]
    a = g.cell ** 2
    info = {'view': pic.view, 'layer': layer, 'variant': var, 'height_m': round(float(m.get('z', 0.0)), 3),
            'cell_m': round(float(g.cell), 4), 'px_per_m': round(s, 2),
            'blind_m2': round(float(m.get('blind_m2', 0.0)), 4),
            'covered_pct': round(float(m.get('covered_pct', 0.0)), 2),
            'multi_pct': round(float(m.get('multi_pct', 0.0)), 2),
            'min_density_px_cm': round(float(m.get('min_density', 0.0)), 3),
            'min_ok_pct': round(float(m.get('min_ok_pct', 0.0)), 2),
            'min_density_threshold': mind, 'target_density': tgt, 'roi_area_m2': round(float(g.area), 3),
            'blind_patches': patches, 'cameras': [c.tag for c in project.variant.cameras if c.visible]}
    if centre is not None:
        info.update({'centre': [round(cx, 3), round(cy, 3)], 'radius_m': r,
                     'window_blind_m2': round(float((roi_w & (cnt_w == 0)).sum() * a), 4),
                     'window_roi_m2': round(float(roi_w.sum() * a), 4)})
        dens_w = np.asarray(result.best_dens)[win_cells]
        low = roi_w & (cnt_w > 0) & (dens_w < mind)
        info['window_below_min_m2'] = round(float(low.sum() * a), 4)
    pic.info = info
    pic.caption = map_caption(info)
    return pic


def map_caption(info) -> str:
    lay = 'shaded by the best camera\'s pixel density' if info['layer'] == 'density' else \
        'shaded by how many cameras see each cell'
    if info['view'] == 'closeup':
        head = (f'Close-up of the coverage map around ({info["centre"][0]:.2f}, {info["centre"][1]:.2f}) m '
                f'({2 * info["radius_m"]:.1f} m across, the point marked by four ticks, white lines are the '
                f'{info["cell_m"] * 100:g} cm analysis cells): in this window {info["window_blind_m2"]:.3f} m² blind, '
                f'{info["window_below_min_m2"]:.3f} m² below the minimum density, of {info["window_roi_m2"]:.3f} m² ROI.')
    else:
        head = (f'Bird\'s-eye coverage map of the ROI ground at z = {info["height_m"]:.2f} m, variant '
                f'"{info["variant"]}": the front of the vehicle is at the top, its left on the left of the picture; '
                f'red is blind ground, amber below the minimum density ({info["min_density_threshold"]:g} px/cm), '
                f'the rest {lay}; dark grey is under the body, the black line the body outline the analysis uses, '
                f'each camera a dot with a line along its heading and its tag.')
    figs = (f'Figures (from the analysis): blind {info["blind_m2"]:.3f} m², covered {info["covered_pct"]:.1f} %, seen '
            f'by 2 or more cameras {info["multi_pct"]:.1f} %, minimum density {info["min_density_px_cm"]:.2f} px/cm, '
            f'{info["min_ok_pct"]:.1f} % at or above the minimum density, ROI {info["roi_area_m2"]:.2f} m², '
            f'{info["px_per_m"]:.0f} px per metre.')
    pats = info.get('blind_patches') or []
    if pats:
        figs += ' Blind patches (numbered on the map): ' + '; '.join(
            f'{b["n"]}: {b["area_m2"]:.3f} m² at ({b["centre"][0]:.2f}, {b["centre"][1]:.2f}) m'
            + (f', {b["where"]}' if b.get('where') else '') for b in pats) + '.'
    else:
        figs += ' No blind ground in the ROI.'
    return head + ' ' + figs


# ====================================================================================================== camera view
def render_camera(project, engine, cam, result=None, fov=None) -> Picture:
    """A camera's synthetic view of the parking scene (core.synth.render_world), scaled to fit the limits."""
    from ..core import synth
    from ..core.lens import resolve
    intr = resolve(project.module_for(cam), project.lens_for(cam))
    w, h = fit(int(intr.width), int(intr.height), 1024, 800_000)
    world = synth.parking_world(project)
    img = synth.render_world(project, engine, cam, world, w, h, seed=0, noise=0.0)
    img = np.ascontiguousarray(np.asarray(img, np.uint8)[..., :3])
    cv = _Canvas(img)
    cv.text(f'{cam.tag} | synthetic parking scene | {w}x{h} of {intr.width}x{intr.height}', 10, 22, (255, 255, 255),
            0.5, 1, box=(20, 20, 22))
    data, mt = encode(img, 'jpeg')
    lens = project.lens_for(cam)
    mod = project.module_for(cam)
    d = (result.per_cam.get(cam.uid) if result is not None else None) or {}
    info = {'view': 'camera', 'camera': cam.tag, 'pose': {k: round(float(getattr(cam, k)), 3) for k in
                                                         ('x', 'y', 'z', 'yaw', 'pitch', 'roll')},
            'lens': getattr(lens, 'name', ''), 'module': getattr(mod, 'name', ''),
            'image_px': [int(intr.width), int(intr.height)], 'rendered_px': [w, h]}
    if fov is not None:
        info['fov_deg'] = {'horizontal': round(float(fov[0]), 1), 'vertical': round(float(fov[1]), 1)}
    if d:
        info.update({'seen_m2': round(float(d.get('seen_m2', 0.0)), 3),
                     'only_this_camera_m2': round(float(d.get('unique_m2', 0.0)), 3),
                     'min_density_px_cm': round(float(d.get('min_dens', 0.0)), 3),
                     'blind_if_removed_m2': round(float(d.get('if_removed_blind', 0.0)), 3)})
    pic = Picture('camera', mt, data, w, h, '', info)
    pic.caption = camera_caption(info)
    return pic


def camera_caption(info) -> str:
    p = info['pose']
    out = (f'Synthetic view through camera {info["camera"]} ({info["module"]}, {info["lens"]}) at x {p["x"]:.3f}, '
           f'y {p["y"]:.3f}, z {p["z"]:.3f} m, yaw {p["yaw"]:.1f}°, pitch {p["pitch"]:.1f}° (down), roll '
           f'{p["roll"]:.1f}°: a generated parking scene (bay lines, parked cars, a pillar, a bollard, a wall) drawn '
           f'with the same lens model, pose and body occlusion as the analysis; dark grey outside the image circle; '
           f'the vehicle\'s own body appears where it blocks the view. Not a real capture.')
    if 'fov_deg' in info:
        out += f' Field of view {info["fov_deg"]["horizontal"]:.1f}° × {info["fov_deg"]["vertical"]:.1f}°.'
    if 'seen_m2' in info:
        out += (f' Figures (from the analysis): sees {info["seen_m2"]:.2f} m² of the ROI ground, '
                f'{info["only_this_camera_m2"]:.2f} m² that no other camera sees, minimum density where it is the '
                f'best camera {info["min_density_px_cm"]:.2f} px/cm; without it {info["blind_if_removed_m2"]:.2f} m² '
                f'would be blind.')
    return out


def thumbnail(pic: Picture, edge=240) -> bytes:
    """A small PNG of the picture for the chat transcript."""
    import cv2
    arr = cv2.imdecode(np.frombuffer(pic.data, np.uint8), cv2.IMREAD_COLOR)
    if arr is None:
        return b''
    h, w = arr.shape[:2]
    s = edge / max(w, h)
    arr = cv2.resize(arr, (max(1, int(w * s)), max(1, int(h * s))), interpolation=cv2.INTER_AREA)
    ok, buf = cv2.imencode('.png', arr)
    return buf.tobytes() if ok else b''
