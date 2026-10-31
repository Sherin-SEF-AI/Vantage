"""Recommended actions for the assistant: what to look at or do next, worked out locally from the app's own results.

recommend(state, context) reads what the app already computed (the analysis, the Issues list, the worst single-camera
error, the objects in the stitched view, the requirements scorecard, the perception range, the pages' results: the
drive-log check, detection quality, the simulation campaign, the CARLA page, the live rig, the Copilot's plans) and
returns Recommendations, most useful first. Every figure a recommendation quotes is read off those results, never
estimated here. No Qt, no network, deterministic: the same state and context give the same list.

The context is what the user is doing now (the open page, the selection, the last edits with the analysis before and
after them, a job that just finished): recommendations about the current thing rank first, and context_items() picks
the ones the panel's live strip shows.

A recommendation never does anything by itself. Its kind says what a click does:
- ask: sends its prompt to the assistant exactly as typing it would (the consent and the daily cap apply);
- goal: the same in agent mode;
- plan: sends a planning request, so the Copilot's plan card appears and still needs the user's approval;
- open: opens a page or a panel, or points at a button; it runs nothing and changes nothing.

Dismissed recommendations stay hidden while their evidence is the same; a material change brings them back."""
from __future__ import annotations

import math
import re
import time
from dataclasses import dataclass, field

import numpy as np

KINDS = ('ask', 'goal', 'plan', 'open')
BUTTON = {'ask': 'Ask', 'goal': 'Start goal', 'plan': 'Plan it', 'open': 'Open'}
SEVERITY = {'critical': 90.0, 'serious': 70.0, 'warning': 50.0, 'info': 30.0, 'start': 8.0}
FOCUS_BOOST = 200.0          # about the selection, the last edit or a job that just finished
PAGE_BOOST = 100.0           # about the open page
JOB_RECENT_S = 300.0         # a finished job counts as current this long
SHOWN = 5                    # the list shows this many; the rest are one click away
STRIP = 3                    # the live strip shows at most this many

BLIND_MIN_M2 = 0.0005        # the Issues list's own threshold for blind ground in the ROI
EDIT_BLIND_M2 = 0.005        # an edit that adds this much blind ground is worth a word
TOL_EXTRA_M2 = 0.01          # a single camera's mounting error adding this much blind ground
UNIQUE_M2 = 0.05             # ground only the selected camera sees
WEAK_RECALL = 0.5            # a detection cell below this recall (with enough objects) is weak
VANISH_M2 = 0.005            # ground where a standing object vanishes at a seam


@dataclass
class Recommendation:
    id: str                      # stable per rule and target
    title: str                   # short and action-first
    why: str                     # one line of evidence, the figures read from the app's results
    kind: str                    # ask | goal | plan | open
    prompt: str = ''             # ask, goal, plan: what is sent when clicked
    target: dict | None = None   # open: {'page', ...} (see the panel's open_target)
    priority: float = 0.0        # severity × value, plus the context's boost (set by recommend)
    source: str = ''             # which result it came from
    about: tuple = ()            # what it concerns: ('camera', 'QR'), ('page', 'sim'), ('edit', 7), ('job', 'drivelog')…
    evidence: dict = field(default_factory=dict)    # the figures: a material change brings a dismissed one back
    topic: str = ''              # recommendations on one topic are deduplicated (the most useful kept)
    base: float = 0.0            # the priority before the context's boost

    @property
    def button(self) -> str:
        return BUTTON.get(self.kind, 'Open')

    def to_dict(self) -> dict:
        return {'id': self.id, 'title': self.title, 'why': self.why, 'kind': self.kind, 'prompt': self.prompt,
                'target': dict(self.target) if self.target else None, 'priority': round(self.priority, 2),
                'source': self.source, 'about': [list(a) for a in self.about], 'evidence': dict(self.evidence)}


@dataclass
class State:
    """What the app knows now (plain objects; the panel gathers them, the tests build them)."""
    project: object
    result: object = None            # core.analysis.AnalysisResult
    issues: list = field(default_factory=list)          # [rules.Issue]
    worst: tuple | None = None       # (blind m², camera uid, perturbation) from tolerance.worst_single_camera
    stale: bool = False              # the analysis is out of date (manual mode, or an older run)
    stitched: object = None          # core.stitched_objects.StitchedObjects
    advice: object = None            # its seam advice
    requirements: object = None      # core.requirements.RequirementsResult (the scorecard page's)
    perception: object = None        # core.perception.PerceptionResult
    drivelog: object = None          # core.drivelog.DriveLogResult
    detection: object = None         # core.detection_eval.DetectionEvalResult
    campaign: object = None          # core.simcampaign.CampaignResult
    carla: dict | None = None        # see carla_rules
    rig_health: dict | None = None   # the Live rig page's snapshot while it streams
    lidar_calib_done: bool = False   # the Lidar–camera calibration page has a plan or a result
    plans: list = field(default_factory=list)           # copilot.Plan shown in the conversation (latest versions)
    records: list = field(default_factory=list)         # copilot approval records in the project
    brief: str = ''
    hints: dict = field(default_factory=dict)           # worked out off the UI thread: pitch_hint(), scale_hint()


@dataclass
class Context:
    """What the user is doing now."""
    page: str = 'viewports'
    page_label: str = ''
    selection: tuple = ()            # (('camera', 'QR'),) (('sensor', 'L1'),) (('rig', uid),) (('target', uid),)…
    edits: list = field(default_factory=list)   # recent edits, oldest first: {'index', 'text', 'before', 'after'}
    undo_index: int = -1             # the undo stack's index now (an edit can be undone while it is the last one)
    job: dict | None = None          # {'kind', 'ok', 't'}: a job that finished
    now: float = 0.0

    def focus(self) -> set:
        """The keys of the current thing: the selection, the last edit, a job that just finished."""
        out = {tuple(s[:2]) for s in self.selection if len(s) >= 2}
        if self.edits:
            out.add(('edit', self.edits[-1].get('index')))
        if self.job and (self.now or time.time()) - float(self.job.get('t', 0)) <= JOB_RECENT_S:
            out.add(('job', self.job.get('kind')))
        return out


# ====================================================================================================== helpers
def _num(v, default=None):
    try:
        v = float(v)
    except (TypeError, ValueError):
        return default
    return v if math.isfinite(v) else default


def _plural(n, word, many=None):
    return f'{n:g} {word if n == 1 else (many or word + "s")}'


def _tags(project) -> dict:
    try:
        return {c.uid: c.tag for c in project.variant.cameras}
    except AttributeError:
        return {}


def _cam(project, tag):
    return next((c for c in project.variant.cameras if c.tag == tag), None)


def _short(text, n=72):
    """At most n characters, cut at a word boundary with an ellipsis."""
    text = ' '.join(str(text).split())
    if len(text) <= n:
        return text
    cut = text[:n - 1]
    if ' ' in cut[n // 2:]:
        cut = cut[:cut.rindex(' ')]
    return cut.rstrip(' ,;:-–') + '…'


_URL = re.compile(r'[a-z][a-z0-9+.-]*://\S+', re.IGNORECASE)


def _rec(rid, title, why, kind, sev, value=1.0, **kw) -> Recommendation:
    base = SEVERITY.get(sev, 30.0) * min(max(float(value), 0.6), 1.4)
    return Recommendation(rid, title, why, kind, priority=base, base=base, **kw)


def _location(project, x, y) -> str:
    try:
        from ..core.analysis import describe_location
        return describe_location(x, y, project)
    except Exception:                                   # noqa: BLE001 — a vehicle without its measures
        return ''


# ====================================================================================================== the rules
ISSUE_PAGES = (('requirement', 'requirements'), ('recording', 'recording'), ('disk', 'recording'),
               ('power', 'power'), ('gnss', 'gnss'), ('imu', 'gnss'), ('trigger', 'timing'), ('sync', 'timing'),
               ('harness', 'harness'), ('cable run', 'harness'), ('wiper', 'materials'), ('fascia', 'materials'),
               ('glass', 'materials'), ('critical target', 'critical'), ('road', 'terrain'), ('fleet', 'fleet'),
               ('as-built', 'asbuilt'), ('observab', 'observability'), ('lidar', 'coverage'), ('radar', 'coverage'),
               ('housing', 'viewports'), ('glare', 'exposure'), ('dirt', 'exposure'), ('load case', 'terrain'))


def issue_rules(st: State, cx: Context):
    """The Issues list: the most severe first, by name (at most four; the blind-ground issue is the ground rule's)."""
    order = {'critical': 0, 'serious': 1, 'warning': 2}
    tags = _tags(st.project)
    out = []
    for i, iss in enumerate(sorted(st.issues, key=lambda x: order.get(x.severity, 3))[:4]):
        msg = str(iss.message)
        if 'of ground in the ROI is blind' in msg:
            continue                                    # the ground rule says it with the patch and the camera
        first = re.split(r'(?<=[.;])\s', msg, maxsplit=1)[0].rstrip('.')
        head, _, rest = first.partition(': ')         # 'X falls short: the detail' -> title X, the detail below
        rest = (rest + msg[len(first):]).strip(' .') if rest else msg[len(first):].strip(' .')
        key = re.sub(r'[\d.,]+', '#', f'{iss.objects}|{first}')[:90]
        cams = [tags.get(u, u) for u in getattr(iss, 'uids', [])]
        low = f'{iss.objects} {msg}'.lower()
        pages = tuple(('page', pg) for word, pg in ISSUE_PAGES if word in low)[:1]
        topic = f'requirement:{iss.objects.split("·")[-1].strip()}' if iss.action == 'requirement' else \
            'vanish' if 'vanish' in msg else ''
        out.append(_rec(f'issue:{key}', f'Fix: {_short(head, 80)}',
                        f'{iss.severity.capitalize()} on the Issues list' + (f': {rest}.' if rest else '.'),
                        'ask', iss.severity, 1.0 - 0.05 * i, source='Issues list',
                        prompt=f'Explain this issue and suggest the smallest fix, checked with simulate_changes '
                               f'where it is a camera change: "{msg}"',
                        about=tuple(('camera', t) for t in cams) + pages,
                        evidence={'message': msg}, topic=topic))
    return out


def ground_rules(st: State, cx: Context):
    """Blind ground in the ROI (the largest patch, where, the nearest camera), a camera whose minimum density is below
    the target, and a seam where a standing object vanishes."""
    res, p = st.result, st.project
    if res is None:
        return []
    out = []
    tags = _tags(p)
    m = res.metrics
    blind = _num(m.get('blind_m2'), 0.0)
    g = res.grid
    if blind > BLIND_MIN_M2:
        mask = g.roi & (res.count == 0)
        from scipy import ndimage
        lab, n = ndimage.label(mask, structure=np.ones((3, 3)))
        if n:
            sizes = np.bincount(lab.ravel())[1:]
            k = int(np.argmax(sizes))
            jj, ii = np.nonzero(lab == k + 1)
            area = float(sizes[k] * g.cell ** 2)
            x, y = float(g.xs[ii].mean()), float(g.ys[jj].mean())
            loc = _location(p, x, y)
            cams = [c for c in p.variant.cameras if c.visible]
            near = min(cams, key=lambda c: math.hypot(c.x - x, c.y - y)).tag if cams else ''
            more = f' of {blind:.3f} m² in {n} patches' if n > 1 else ''
            where = (f' {loc}' if loc.split(' ')[0] in ('below', 'behind', 'ahead') else f' at the {loc}') if loc \
                else ''
            out.append(_rec('ground:blind', f'Close the {area:.3f} m² blind patch{where}' + (f' near {near}' if near
                                                                                             else ''),
                            f'The analysis finds {area:.3f} m² of ROI ground no camera sees{more}, centred at '
                            f'({x:.2f}, {y:.2f}) m; nearest camera {near or "none"}.', 'goal', 'critical',
                            1.0 + min(blind / 0.2, 0.4), source='Analysis · coverage',
                            prompt=f'Close the {area:.3f} m² blind patch{where} (centred at x {x:.2f}, y {y:.2f} m) '
                                   f'without lowering the minimum density anywhere; keep left/right pairs symmetric.',
                            about=(('camera', near), ('page', 'viewports')) if near else (('page', 'viewports'),),
                            evidence={'blind_m2': round(blind, 3), 'patch_m2': round(area, 3)}, topic='blind'))
    # the camera with the lowest minimum density where it is the best (and each selected camera)
    mind, tgt = float(p.analysis.min_density), float(p.analysis.target_density)
    rows = [(tags.get(u, u), d) for u, d in res.per_cam.items() if d.get('best_m2', 0) > 0]
    picked = {sel[1] for sel in cx.selection if sel[0] == 'camera'}
    if rows:
        worst = min(rows, key=lambda r: r[1].get('min_dens', 0.0))[0]
        picked.add(worst)
    for tag, d in rows:
        if tag not in picked:
            continue
        md = float(d.get('min_dens', 0.0))
        if md >= tgt:
            continue
        sev = 'serious' if md < mind else 'warning'
        best = float(d.get('best_m2', 0.0))
        h = st.hints.get(f'pitch:{tag}')
        if isinstance(h, dict) and abs(h.get('min_now', -1) - md) < 1e-6:
            dp = h['delta']
            out.append(_rec(f'density:{tag}', f'{tag}\'s minimum density is {md:.2f} px/cm (target {tgt:g}): try pitch '
                                              f'{dp:+g}°',
                            f'Simulated on a copy: {tag} pitched {dp:+g}° (mirrored partner too) gives {h["min_dens"]:.2f} '
                            f'px/cm where it is the best camera, blind ground {h["blind_now"]:.3f} → '
                            f'{h["blind_m2"]:.3f} m².', 'ask', sev, 0.8 + min((tgt - md) / max(tgt, 1e-6), 0.6),
                            source='Analysis · per camera, simulated pitch',
                            prompt=f'{tag}\'s minimum density is {md:.2f} px/cm (target {tgt:g} px/cm). A copy with '
                                   f'{tag} pitched {dp:+g}° gives {h["min_dens"]:.2f} px/cm and blind ground '
                                   f'{h["blind_m2"]:.3f} m². Check it with simulate_changes, look for anything it makes '
                                   f'worse, and propose it if it holds up.',
                            about=(('camera', tag),), evidence={'min_dens': round(md, 2), 'target': tgt,
                                                                'delta': dp}))
            continue
        out.append(_rec(f'density:{tag}', f'Raise {tag}\'s minimum density: {md:.2f} px/cm, target {tgt:g}',
                        f'{tag} is the best camera on {best:.2f} m² of the ROI; its lowest density there is '
                        f'{md:.2f} px/cm (minimum {mind:g}, target {tgt:g} px/cm).', 'ask', sev,
                        0.8 + min((tgt - md) / max(tgt, 1e-6), 0.6), source='Analysis · per camera',
                        prompt=f'How can I raise {tag}\'s minimum density ({md:.2f} px/cm where it is the best camera) '
                               f'towards the {tgt:g} px/cm target without creating blind ground? Check the options '
                               f'with simulate_changes and propose the best one.',
                        about=(('camera', tag),), evidence={'min_dens': round(md, 2), 'target': tgt}))
    # a seam where a standing object vanishes (the objects in the stitched view)
    so = st.stitched
    if so is not None and getattr(so, 'seams', None):
        seams = sorted(so.seams, key=lambda s: -float(s.get('vanish_m2', 0.0)))
        s = seams[0]
        v = float(s.get('vanish_m2', 0.0))
        if v > VANISH_M2:
            a, b = s.get('tags', ('?', '?'))
            h = _num(getattr(so.settings, 'height', None), 0.0)
            adv = st.advice
            after = ''
            if adv is not None and getattr(adv, 'changes', None):
                after = f'; the seam advice brings it to {float(adv.vanish_after):.3f} m²'
            out.append(_rec(f'seam:{a}|{b}', f'Move the {a}–{b} seam: a {h:g} m object vanishes on {v:.3f} m²',
                            f'In the stitched top view a {h:g} m standing object vanishes on {v:.3f} m² at the '
                            f'{a}–{b} seam{after}.', 'ask', 'serious', 0.9 + min(v / 0.1, 0.5),
                            source='Objects in the stitched view',
                            prompt=f'A {h:g} m object vanishes on {v:.3f} m² at the {a}–{b} stitch seam. Where should '
                                   f'the seam go (or what should change) so it stays visible? Check with '
                                   f'simulate_changes.',
                            about=(('camera', a), ('camera', b), ('page', 'viewports')),
                            evidence={'vanish_m2': round(v, 3)}, topic='vanish'))
    return out


def selection_rules(st: State, cx: Context):
    """The selected camera, sensor, rig part or target: what is worth asking about it."""
    p, res = st.project, st.result
    out = []
    tags = _tags(p)
    for kind, key in [s[:2] for s in cx.selection][:3]:
        if kind == 'camera' and res is not None:
            c = _cam(p, key)
            d = res.per_cam.get(c.uid) if c is not None else None
            if d is None:
                continue
            u, gone = float(d.get('unique_m2', 0.0)), float(d.get('if_removed_blind', 0.0))
            if u > UNIQUE_M2:
                out.append(_rec(f'unique:{key}', f'{key} alone sees {u:.2f} m²: check what its loss or a mounting '
                                                 f'error costs',
                                f'Only {key} sees {u:.2f} m² of the ROI; without it {gone:.2f} m² would be blind.',
                                'ask', 'info', 1.0 + min(u / 5.0, 0.4), source='Analysis · per camera',
                                prompt=f'{key} alone sees {u:.2f} m² of the ROI. What happens to coverage if {key} '
                                       f'fails or is mounted off by the analysis tolerance? Use run_study '
                                       f'mounting_tolerance and say which mounting direction hurts most.',
                                about=(('camera', key),), evidence={'unique_m2': round(u, 2)}))
        elif kind == 'sensor':
            s = next((x for x in p.variant.sensors if x.tag == key or x.uid == key), None)
            if s is None:
                continue
            m = p.sensor_model_for(s)
            name = f'{m.vendor} {m.name}'.strip()
            out.append(_rec(f'sensor:{s.tag}', f'Ask what {s.tag} adds and what limits it',
                            f'{s.tag} is a {s.kind} ({name}) at ({s.x:.2f}, {s.y:.2f}, {s.z:.2f}) m.', 'ask', 'info',
                            0.9, source='Sensors', about=(('sensor', s.tag),),
                            prompt=f'What does {s.tag} ({name}) cover around the car, what blocks it, and what would '
                                   f'make its placement better?'))
        elif kind == 'rig':
            q = next((x for x in getattr(p, 'rig', ()) if x.uid == key), None)
            if q is None:
                continue
            out.append(_rec(f'rig:{q.uid}', f'Ask what the {q.name} hides', f'The {q.name} is a {q.kind} on the '
                                                                            f'vehicle' + (', blocking sensors'
                                                                                          if q.occludes else '') + '.',
                            'ask', 'info', 0.9, source='Rig parts', about=(('rig', q.uid),),
                            prompt=f'Which cameras and sensors does the rig part "{q.name}" hide ground or field from, '
                                   f'and how much?'))
        elif kind == 'target':
            t = next((x for x in getattr(p, 'targets', ()) if x.uid == key), None)
            if t is None:
                continue
            out.append(_rec(f'target:{t.uid}', f'Ask which cameras resolve {t.name}',
                            f'Target {t.name} at ({t.x:.2f}, {t.y:.2f}) m.', 'ask', 'info', 0.85, source='Targets',
                            about=(('target', t.uid),),
                            prompt=f'Which cameras see the target "{t.name}" at ({t.x:.2f}, {t.y:.2f}) m, at what '
                                   f'pixel density, and how robust is that to the mounting tolerance?'))
    return out


def tolerance_rules(st: State, cx: Context):
    """A camera whose mounting error (the analysis tolerance) moves the blind ground a lot."""
    if st.result is None or not st.worst or not st.worst[1]:
        return []
    b, uid, pert = st.worst[0], st.worst[1], st.worst[2]
    now = _num(st.result.metrics.get('blind_m2'), 0.0)
    extra = float(b) - now
    if extra < TOL_EXTRA_M2:
        return []
    p = st.project
    tag = _tags(p).get(uid, uid)
    a = p.analysis
    d = tuple(pert or ())
    if len(d) >= 2 and (d[0] or d[1]):
        what = ' and '.join(f'{v:+g}° {n}' for v, n in ((d[0], 'yaw'), (d[1], 'pitch')) if v)
    elif len(d) >= 6:
        what = f'a {math.hypot(*d[3:6]) * 1000:.0f} mm slide along the body'
    else:
        what = 'its mounting error'
    return [_rec(f'tolerance:{tag}', f'Make {tag} robust to mounting errors: {what} adds {extra:.3f} m² blind',
                 f'With {tag} off by {what} (the analysis tolerance {a.tol_deg:g}° / {a.tol_mm:g} mm) the blind ground '
                 f'goes from {now:.3f} to {float(b):.3f} m², the worst single-camera error.', 'goal', 'serious',
                 0.8 + min(extra / 0.1, 0.6), source='Analysis · worst single-camera error',
                 prompt=f'Make the rig robust to the mounting tolerance: {tag} off by {what} adds {extra:.3f} m² blind '
                        f'ground. Reduce that worst case without lowering the minimum density.',
                 about=(('camera', tag), ('page', 'tolerance')), evidence={'worst_m2': round(float(b), 3)})]


def requirement_rules(st: State, cx: Context):
    """Failing requirements by name; none written yet → write them from the brief."""
    p = st.project
    items = ((getattr(p.analysis, 'requirements', None) or {}).get('items') or [])
    if not items:
        brief = (st.brief or '').strip()
        return [_rec('requirements:write', 'Write requirements from your brief' if brief else
                     'Write requirements for this rig',
                     ('Your brief has no measurable requirements in the scorecard yet.' if brief else
                      'No requirements are written yet: the scorecard and the CI check have nothing to check.'),
                     'ask', 'info', 1.0, source='Requirements scorecard',
                     prompt=('Turn my project brief into measurable requirements with propose_requirements.' if brief
                             else 'Propose measurable requirements for this data-collection rig with '
                                  'propose_requirements (coverage, density, perception range, sync), from the design '
                                  'as it is.'),
                     about=(('page', 'requirements'),), evidence={'items': 0})]
    res = st.requirements
    if res is None:
        return []
    out = []
    sel = {k for kind, k in [s[:2] for s in cx.selection] if kind == 'requirement'}
    fails = [r for r in res.items if r.status == 'fail']
    fails.sort(key=lambda r: (r.item.get('priority') != 'must', r.item.get('id') not in sel))
    from ..core import requirements as RQ
    for i, r in enumerate(fails[:3]):
        it = r.item
        m = r.metric
        val, tgt = RQ.fmt_value(r.value, m), RQ.fmt_value(it.get('target'), m)
        sev = 'serious' if it.get('priority') == 'must' else 'warning'
        out.append(_rec(f'requirement:{it["id"]}', f'Fix requirement {it["id"]}: {_short(it.get("text", ""), 48)}',
                        f'{it["id"]} ({it.get("priority")}) fails: {val} against {RQ.OP_TEXT.get(it.get("op"), "")} '
                        f'{tgt}' + (' (from an earlier run)' if r.stale else '') + '.', 'ask', sev, 1.0 - 0.05 * i,
                        source='Requirements scorecard',
                        prompt=f'Requirement {it["id"]} "{it.get("text", "")}" fails: {val} against '
                               f'{RQ.OP_TEXT.get(it.get("op"), "")} {tgt}. Why, and what is the smallest change that '
                               f'meets it? Check it with simulate_changes.',
                        about=(('requirement', it['id']), ('page', 'requirements')),
                        evidence={'value': r.value, 'status': r.status}, topic=f'requirement:{it["id"]}'))
    return out


def perception_rules(st: State, cx: Context):
    """A sector whose shortest range for a class is below its required range."""
    pr = st.perception
    if pr is None:
        return []
    short = []
    for s in pr.sectors:
        for cls, v in s['classes'].items():
            if not v.get('pass', True) and s['required_m'] > 0:
                short.append((v['shortfall_m'] / s['required_m'], s, cls, v))
    short.sort(key=lambda t: -t[0])
    out = []
    for frac, s, cls, v in short[:2]:
        out.append(_rec(f'perception:{s["name"]}:{cls}',
                        f'Extend the {s["name"]} range for {cls.lower()}: {v["min_m"]:.1f} of {s["required_m"]:g} m',
                        f'Perception range: in the {s["name"]} sector a {cls.lower()} is detected from {v["min_m"]:.1f} m '
                        f'at its worst azimuth, {v["shortfall_m"]:.1f} m short of the {s["required_m"]:g} m required.',
                        'ask', 'serious', 0.8 + min(frac, 0.6), source='Perception range',
                        prompt=f'In the {s["name"]} sector a {cls.lower()} is detected from {v["min_m"]:.1f} m, short of '
                               f'the {s["required_m"]:g} m required. Which camera limits it, and what change would '
                               f'reach the required range?',
                        about=(('page', 'perception'),), evidence={'min_m': round(v['min_m'], 1)}))
    return out


def lidar_rules(st: State, cx: Context):
    """A lidar beside cameras with no calibration plan."""
    p = st.project
    lids = [s for s in p.variant.sensors if s.kind == 'lidar' and s.visible]
    if not lids or not p.variant.cameras or st.lidar_calib_done or (getattr(p.analysis, 'lidar_calib', None) or {}):
        return []
    tag = lids[0].tag
    return [_rec('lidar_calib:plan', 'Plan the lidar–camera calibration',
                 f'{tag} and {len(p.variant.cameras)} cameras have no calibration plan yet (board poses, the '
                 f'uncertainty it reaches).', 'open', 'warning', 0.9, source='Lidar–camera calibration',
                 target={'page': 'lidar_calib'},
                 about=(('sensor', tag), ('page', 'lidar_calib'), ('page', 'coverage')),
                 evidence={'lidars': len(lids)})]


def stale_rules(st: State, cx: Context):
    if not st.stale:
        return []
    return [_rec('analysis:stale', 'Update the analysis', 'The figures are older than the design (the analysis is '
                                                          'stale): press F5 or Analysis ▸ Run now.', 'open', 'serious',
                 1.2, source='Analysis', target={'page': 'viewports', 'point': 'analysis'},
                 about=(('page', 'viewports'),), evidence={'stale': True})]


def carla_rules(st: State, cx: Context):
    """CARLA: installed and never run on this project; the server; the render budget; the last drive unchecked or not
    scored. st.carla = {'installed', 'port', 'server_up' (None: unknown), 'runs', 'last_run': {'path', 'name',
    'status', 'ground_truth', 'checked', 'scored'}, 'render': {'mpx', 'gt_mpx', 'budget', 'scale'}}."""
    c = st.carla
    if not c:
        return []
    out = []
    on_page = cx.page == 'sim'
    if c.get('installed') and not c.get('runs'):
        out.append(_rec('carla:first', 'Plan a CARLA drive with ground truth',
                        'CARLA is set up on this machine and this project has no drive yet.', 'plan', 'info', 1.2,
                        source='CARLA', about=(('page', 'sim'),),
                        prompt='Plan this for my approval with propose_plan: a 10 s CARLA drive in Town10HD with the '
                               'rig\'s cameras and ground truth on (semantic, instance and depth), clear noon.',
                        evidence={'runs': 0}))
    if on_page and c.get('installed') and c.get('server_up') is False:
        out.append(_rec('carla:server', f'No server on port {c.get("port")}: start it',
                        f'No CARLA server answers on port {c.get("port")}; a drive needs one (the page starts it with '
                        f'its memory cap).', 'open', 'warning', 1.0, source='CARLA page',
                        target={'page': 'sim', 'focus': 'b_start'}, about=(('page', 'sim'),),
                        evidence={'port': c.get('port'), 'up': False}))
    r = c.get('render') or {}
    total, budget = _num(r.get('mpx'), 0.0) + _num(r.get('gt_mpx'), 0.0), _num(r.get('budget'), 0.0)
    if on_page and budget and total > budget:
        sc = _num(r.get('scale'), 0.0)
        gt = ' with ground truth' if _num(r.get('gt_mpx'), 0.0) else ''
        h = st.hints.get('scale')
        fit = h.get('scale') if isinstance(h, dict) and h.get('for') == sc else None
        out.append(_rec('carla:budget', f'Image scale {sc:g}{gt} exceeds the render budget: use {fit:g}' if fit else
                        f'Image scale {sc:g}{gt} is over the render budget: plan one that fits',
                        f'The cameras render {total:.0f} Mpx a frame{gt}; the bridge refuses more than {budget:g}.'
                        + (f' At {fit:g} the rig, built again, renders {h["mpx"]:.0f} Mpx.' if fit else ''),
                        'plan', 'serious', 1.0 + min((total - budget) / budget, 0.4), source='CARLA page',
                        prompt=f'Plan this for my approval with propose_plan: the CARLA page\'s drive as it is set '
                               f'now, with the largest image scale that fits the render budget (now {total:.0f} Mpx a '
                               f'frame at scale {sc:g}, the budget is {budget:g})'
                               + (f': the rig fits at {fit:g}.' if fit else '.'),
                        about=(('page', 'sim'),), evidence={'mpx': round(total), 'scale': sc}))
    last = c.get('last_run') or {}
    if last.get('status') == 'ok':
        job = (('job', 'carla_run'),)
        if not last.get('checked'):
            out.append(_rec(f'carla:check:{last.get("name")}', 'Check the last CARLA drive with the drive-log check',
                            f'The drive {last.get("name")} has not been through the drive-log check (streams, drops, '
                            f'sync, rotations).', 'open', 'warning', 1.1, source='CARLA runs',
                            target={'page': 'sim', 'run': last.get('path'), 'focus': 'b_drivelog'},
                            about=(('page', 'sim'),) + job, evidence={'run': last.get('name')}))
        if last.get('ground_truth') and not last.get('scored'):
            out.append(_rec(f'carla:score:{last.get("name")}', 'Score detection on the last drive',
                            f'The drive {last.get("name")} has ground truth and no detection evaluation yet.', 'open',
                            'warning', 1.0, source='CARLA runs',
                            target={'page': 'sim', 'run': last.get('path'), 'focus': 'b_detect'},
                            about=(('page', 'sim'), ('page', 'detection')) + job, evidence={'run': last.get('name')}))
    elif last.get('status') not in (None, 'ok'):
        out.append(_rec(f'carla:failed:{last.get("name")}', 'Ask why the last CARLA drive failed',
                        f'The drive {last.get("name")} ended {last.get("status")}: '
                        f'{_short(last.get("error") or "see its report", 90)}', 'ask', 'serious', 1.0,
                        source='CARLA runs', about=(('page', 'sim'), ('job', 'carla_run')),
                        prompt=f'My last CARLA drive ended "{last.get("status")}": '
                               f'{_short(last.get("error") or "", 300)}. What went wrong and what should I change?',
                        evidence={'status': last.get('status')}))
    return out


def drivelog_rules(st: State, cx: Context):
    """The drive-log check: a camera found rotated; the other serious findings."""
    res = st.drivelog
    if res is None:
        return []
    out = []
    rows = [r for r in getattr(res, 'extrinsics', []) if r.get('verdict') == 'rotated' and r.get('delta_deg')]
    rows.sort(key=lambda r: -float(np.linalg.norm(r['delta_deg'])))
    for r in rows[:2]:
        ang = float(np.linalg.norm(r['delta_deg']))
        y, pt, rl = (float(v) for v in r['delta_deg'])
        out.append(_rec(f'drivelog:rotated:{r["tag"]}', f'Ask why {r["tag"]} is off by {ang:.2f}°',
                        f'The drive-log check finds {r["tag"]} rotated by {ang:.2f}° (yaw {y:+.2f}°, pitch {pt:+.2f}°, '
                        f'roll {rl:+.2f}°; confidence {r.get("confidence") or "—"}).', 'ask', 'serious',
                        0.9 + min(ang / 2.0, 0.5), source='Drive-log check',
                        prompt=f'The drive-log check finds {r["tag"]} rotated by {ang:.2f}° (yaw {y:+.2f}°, pitch '
                               f'{pt:+.2f}°, roll {rl:+.2f}°). What could cause it and what should I do about it?',
                        about=(('camera', r['tag']), ('page', 'drivelog'), ('job', 'drivelog')),
                        evidence={'deg': round(ang, 2)}))
    for f in [f for f in getattr(res, 'findings', []) if f.get('severity') in ('critical', 'serious')][:2]:
        msg = str(f.get('message', ''))
        key = re.sub(r'[\d.,]+', '#', msg)[:80]
        out.append(_rec(f'drivelog:finding:{key}', f'Ask about: {_short(msg, 60)}', f'Drive-log check: {msg}', 'ask',
                        f['severity'], 0.9, source='Drive-log check',
                        prompt=f'The drive-log check reports: "{msg}". What does it mean for the dataset, and what '
                               f'should I change?',
                        about=(('page', 'drivelog'), ('job', 'drivelog')), evidence={'message': msg}))
    return out


def detection_rules(st: State, cx: Context):
    """Detection quality: the weakest cell (recall below 0.5 with enough objects) → plan data there."""
    res = st.detection
    if res is None:
        return []
    min_gt = int(getattr(res.settings, 'min_gt', 5) or 5)
    weak = [r for r in res.rows if r.get('camera') == 'all' and r.get('range_bin') not in ('all', None)
            and (r.get('n_gt') or 0) >= min_gt and r.get('recall') is not None and r['recall'] < WEAK_RECALL]
    if not weak:
        return []
    weak.sort(key=lambda r: (r['recall'], -r['n_gt'], str(r['class']), str(r['range_bin'])))
    r = weak[0]
    rec = res.recordings[r['rec']] if 0 <= r['rec'] < len(res.recordings) else {}
    weather = rec.get('weather') or ''
    town = rec.get('town') or ''
    at = f' in {weather}' if weather else ''
    cls = str(r['class'])
    return [_rec(f'detection:weak:{cls}:{r["range_bin"]}',
                 f'Generate data where recall is weak: {cls} {r["range_bin"]}{at}',
                 f'Detection quality: recall {r["recall"]:.2f} on {r["n_gt"]} {cls} objects at {r["range_bin"]} '
                 f'({r.get("recording") or rec.get("label", "")}, {r["mode"]}).', 'plan', 'serious',
                 0.8 + min((WEAK_RECALL - r['recall']) / WEAK_RECALL, 0.6), source='Detection quality',
                 prompt=f'Plan this for my approval with propose_plan: CARLA drives (a drive or a small campaign) that '
                        f'record more {cls} objects at {r["range_bin"]}{at}' + (f' in {town}' if town else '') +
                        f', with ground truth on, where the detector\'s recall is {r["recall"]:.2f} on '
                        f'{r["n_gt"]} objects.',
                 about=(('page', 'detection'), ('job', 'detection')),
                 evidence={'recall': round(r['recall'], 2), 'n': r['n_gt']})]


def campaign_rules(st: State, cx: Context):
    """A simulation campaign that misses the detection probability the user set; false alarms on clean drives."""
    res = st.campaign
    if res is None:
        return []
    from ..core import simcampaign as SC
    det = res.detectability or {}
    out = []
    for t in det.get('thresholds', []):
        if t.get('axis') not in ('', 'any', None) or t.get('empirical') is not None:
            continue
        lab = SC.CHECKS.get(t['check'], (t['check'],))[0]
        what = t['kind'].replace('_', ' ')
        tested = max(t.get('tested') or [0])
        out.append(_rec(f'campaign:miss:{t["check"]}:{t["tag"]}:{t["kind"]}',
                        f'{lab} misses a {what} of {t["tag"]} up to {tested:g} {t["unit"]}',
                        f'Simulation campaign: it does not reach your {float(t["target"]) * 100:.0f} % detection target '
                        f'for a {what} of {t["tag"]} up to {tested:g} {t["unit"]}.', 'ask', 'serious', 1.0,
                        source='Simulation campaign',
                        prompt=f'The campaign\'s {lab} does not reach {float(t["target"]) * 100:.0f} % for a {what} of '
                               f'{t["tag"]} up to {tested:g} {t["unit"]}. Why, and what would make it detectable?',
                        about=(('page', 'campaign'), ('job', 'campaign')), evidence={'tested': tested}))
    for f in det.get('false_alarms', []):
        if f.get('k') and f.get('applicable', True):
            lab = SC.CHECKS.get(f['check'], (f['check'],))[0]
            out.append(_rec(f'campaign:false:{f["check"]}:{f["tag"]}',
                            f'{lab} flags {f["tag"]} on {f["k"]} of {f["n"]} clean drives',
                            f'Simulation campaign: false alarms of {lab} on {f["tag"]} on {f["k"]} of {f["n"]} drives '
                            f'with no fault.', 'ask', 'warning', 0.8 + min(f['k'] / max(f['n'], 1), 0.6),
                            source='Simulation campaign',
                            prompt=f'The campaign\'s {lab} flags {f["tag"]} on {f["k"]} of {f["n"]} clean drives. Why, '
                                   f'and how do I tell a real fault from this?',
                            about=(('page', 'campaign'), ('job', 'campaign')), evidence={'k': f['k'], 'n': f['n']}))
    return out[:3]


def rig_rules(st: State, cx: Context):
    """The live rig: a stream that is unhealthy."""
    h = st.rig_health
    if not h:
        return []
    out = []
    order = {'fault': 0, 'warning': 1}
    cams = sorted(((t, c) for t, c in (h.get('cameras') or {}).items() if c.get('status') in order),
                  key=lambda tc: (order[tc[1]['status']], tc[0]))
    for tag, c in cams[:2]:
        why = _URL.sub('<stream URL>', '; '.join(c.get('reasons') or []) or c['status'])
        out.append(_rec(f'rig:{tag}', f'Ask why {tag}\'s stream is unhealthy', f'Live rig: {tag} {c["status"]}: '
                                                                               f'{_short(why, 100)}', 'ask',
                        'serious' if c['status'] == 'fault' else 'warning', 1.0, source='Live rig',
                        prompt=f'The live rig marks {tag} as {c["status"]}: {why}. What is wrong with the stream and '
                               f'what should I check?',
                        about=(('camera', tag), ('page', 'rig')), evidence={'status': c['status']}))
    return out


# the revisions a plan's own warnings suggest (on a draft): (test on the warnings' text, label, the change)
def plan_rules(st: State, cx: Context):
    """The Copilot: a draft waiting for approval and the revisions its warnings suggest; a plan whose run failed;
    after a plan ran, the checks that fit it (the CARLA rules give those for a drive)."""
    out = []
    for pl in st.plans:
        if getattr(pl, 'status', '') != 'draft':
            continue
        about = (('plan', pl.id),)
        out.append(_rec(f'plan:review:{pl.id}', f'Review the plan waiting for approval: {_short(pl.title, 40)}',
                        f'Plan {pl.id} v{pl.version} ({_plural(len(pl.steps), "step")}) '
                        + ('needs your approval; nothing runs until you approve it.' if pl.ok else
                           'is not valid yet: its errors must be fixed first.'), 'open', 'warning', 1.1,
                        source='Copilot', target={'plan': pl.id, 'version': pl.version}, about=about,
                        evidence={'version': pl.version}))
        out += plan_revisions(pl)
    for rec in st.records:
        runs = rec.get('runs') or []
        if not runs or runs[-1].get('status') != 'failed':
            continue
        run = runs[-1]
        bad = next(((i, s) for i, s in enumerate(run.get('steps') or []) if s.get('status') == 'failed'), None)
        msg = _short(bad[1].get('message') or '', 200) if bad else ''
        step = f'step {bad[0] + 1}' if bad else 'a step'
        out.append(_rec(f'plan:failed:{rec["id"]}:{rec["version"]}', f'See why plan {rec["id"]} failed',
                        f'Plan {rec["id"]} v{rec["version"]} "{_short(rec.get("title", ""), 40)}" failed at {step}'
                        + (f': {msg}' if msg else '.'), 'ask', 'serious', 1.0, source='Copilot records',
                        prompt=f'Plan {rec["id"]} v{rec["version"]} ("{rec.get("title", "")}") failed at {step}'
                               + (f': {msg}' if msg else '') + '. Why, and what should change before running it again?',
                        about=(('plan', rec['id']), ('job', 'plan')),
                        evidence={'finished': run.get('finished_utc')}))
    return out


def plan_revisions(pl) -> list:
    """Follow-up revisions of a draft plan, taken from its own warnings and estimates ('Halve the duration', 'Use
    half-size images'); each is a prompt that revises the plan on screen (a new version, approval again)."""
    out = []
    for i, s in enumerate(getattr(pl, 'steps', [])):
        w = ' '.join(s.warnings).lower()
        est = s.estimate or {}
        set_ = s.settings or {}
        heavy = 'memavailable' in w or 'render budget' in w or 'mpx a frame' in w
        dur = _num(set_.get('duration_s'))
        long_ = _num(est.get('carla_s'), 0.0) > 600 or _num(est.get('disk_mb'), 0.0) > 4000
        step = f' (step {i + 1})' if len(pl.steps) > 1 else ''
        if s.kind == 'carla_run' and dur and dur >= 2 and (heavy or long_):
            why = (f'The plan\'s warnings: {_short(s.warnings[0], 90)}' if heavy else
                   f'Estimate: {est.get("text") or "a long drive"}.')
            out.append(_rec(f'plan:rev:{pl.id}:{i}:duration', f'Halve the duration{step}: {dur:g} → {dur / 2:g} s', why,
                            'plan', 'warning', 1.0, source='Copilot plan', about=(('plan', pl.id),),
                            prompt=f'Revise the plan on screen{step}: halve the duration to {dur / 2:g} s.',
                            evidence={'duration_s': dur}))
        sc = _num(set_.get('image_scale'))
        if s.kind in ('carla_run', 'sample_dataset') and sc and sc > 0.2 and (heavy or long_):
            new = max(0.05, round(sc / 2, 2))
            out.append(_rec(f'plan:rev:{pl.id}:{i}:scale', f'Use half-size images{step}: scale {sc:g} → {new:g}',
                            (f'The plan\'s warnings: {_short(s.warnings[0], 90)}' if heavy else
                             f'Estimate: {est.get("text") or ""}'), 'plan', 'warning', 0.95, source='Copilot plan',
                            about=(('plan', pl.id),),
                            prompt=f'Revise the plan on screen{step}: use half-size images, image scale {new:g}.',
                            evidence={'image_scale': sc}))
        if s.kind == 'carla_run' and ('lidar and radar ignore rain and fog' in w):
            out.append(_rec(f'plan:rev:{pl.id}:{i}:clear', f'Add a clear-weather drive to compare{step}',
                            'The plan\'s warning: CARLA\'s lidar and radar ignore rain and fog, so only the cameras see '
                            'the weather.', 'plan', 'info', 1.0, source='Copilot plan', about=(('plan', pl.id),),
                            prompt=f'Revise the plan on screen: add a second step, the same drive{step} in clear '
                                   f'weather, so the weather\'s effect on the cameras can be compared.'))
        seeds = set_.get('seeds')
        if s.kind == 'campaign' and isinstance(seeds, list) and len(seeds) > 1 and long_:
            out.append(_rec(f'plan:rev:{pl.id}:{i}:seeds', f'Use {len(seeds) - 1} seeds instead of {len(seeds)}{step}',
                            f'Estimate: {est.get("text") or ""}', 'plan', 'warning', 0.9, source='Copilot plan',
                            about=(('plan', pl.id),),
                            prompt=f'Revise the plan on screen{step}: drop the last seed ({len(seeds) - 1} seeds).'))
    return out


def edit_rules(st: State, cx: Context):
    """The last edit, with the analysis before and after it: blind ground it added, a minimum density it lowered."""
    if not cx.edits or st.result is None:
        return []
    e = cx.edits[-1]
    b, a = e.get('before'), e.get('after')
    if not b or not a or e.get('index') != cx.undo_index:
        return []
    out = []
    text = str(e.get('text') or 'your last change')
    about = (('edit', e.get('index')),)
    db = float(a.get('blind_m2', 0.0)) - float(b.get('blind_m2', 0.0))
    mind = float(st.project.analysis.min_density)
    lowered = float(a.get('min_density', 0.0)) < mind <= float(b.get('min_density', 0.0)) + 1e-9
    if db > EDIT_BLIND_M2 or lowered:
        if db > EDIT_BLIND_M2:
            what = f'added {db:.3f} m² blind ground'
            ev = (f'"{text}": blind ground {float(b["blind_m2"]):.3f} → {float(a["blind_m2"]):.3f} m² (analysis '
                  f'before and after).')
        else:
            what = f'lowered the minimum density to {float(a["min_density"]):.2f} px/cm'
            ev = (f'"{text}": minimum density {float(b["min_density"]):.2f} → {float(a["min_density"]):.2f} px/cm, '
                  f'below the {mind:g} px/cm minimum.')
        out.append(_rec(f'edit:{e.get("index")}:undo', f'Your last change {what}: undo it', ev, 'open', 'critical',
                        1.0 + min(max(db, 0.0) / 0.1, 0.4), source='Undo history · analysis before and after',
                        target={'page': None, 'undo': e.get('index'), 'text': text}, about=about,
                        evidence={'delta_m2': round(db, 3)}))
        out.append(_rec(f'edit:{e.get("index")}:why', f'Ask why your last change {what}', ev, 'ask', 'critical', 0.98,
                        source='Undo history · analysis before and after', about=about,
                        prompt=f'My last change ("{text}") {what} (blind ground {float(b["blind_m2"]):.3f} → '
                               f'{float(a["blind_m2"]):.3f} m², minimum density {float(b["min_density"]):.2f} → '
                               f'{float(a["min_density"]):.2f} px/cm). Why, and is there a change that keeps what I '
                               f'wanted without the loss?',
                        evidence={'delta_m2': round(db, 3)}))
    return out


STARTERS = [
    ('start:summary', 'Summarise this design', 'Coverage, weak spots and what to look at first.',
     'Summarise this design: coverage, weak spots and what to look at first.'),
    ('start:child', 'Where can a child go unseen?', 'A 0.9 m child next to the car, checked camera by camera.',
     'Where could a 0.9 m child stand next to the car without any camera seeing them, and how could the rig be '
     'improved?'),
    ('start:review', 'Write a design review', 'Strengths, risks and recommended changes with their measured effect.',
     'Write a short design review of this camera rig for a colleague: strengths, risks, and recommended changes with '
     'their measured effect.'),
]


def default_rules(st: State, cx: Context):
    return [_rec(i, t, w, 'ask', 'start', 1.0 - 0.05 * k, prompt=p, source='Starter') for k, (i, t, w, p)
            in enumerate(STARTERS)]


RULES = (edit_rules, selection_rules, issue_rules, ground_rules, tolerance_rules, requirement_rules,
         perception_rules, lidar_rules, stale_rules, carla_rules, drivelog_rules, detection_rules, campaign_rules,
         rig_rules, plan_rules)


# ====================================================================================================== hints
# Worked out off the UI thread by the panel (they analyse copies: too slow for recommend's 50 ms) and passed in as
# State.hints; recommend() only reads them. Each figure is the app's own engine on a modified copy, never a guess.
PITCH_STEPS = (-4.0, -2.0, 2.0, 4.0)


def pitch_hint(project, engine, tag, steps=PITCH_STEPS, cancel=None) -> dict | None:
    """The pitch change (one of steps, the mirrored partner moved with it as an edit would) that raises the camera's
    minimum density where it is the best camera most (by at least 0.02 px/cm or 0.5 %) without adding blind ground;
    None when none helps. The project
    is never changed: each step is analysed on a snapshot."""
    from . import tools as T
    base = engine.analyse(project.snapshot())
    c = _cam(project, tag)
    if base is None or c is None or c.locked or c.uid not in base.per_cam:
        return None
    md0, b0 = float(base.per_cam[c.uid]['min_dens']), float(base.metrics['blind_m2'])
    best = None
    for d in steps:
        if cancel is not None and cancel():
            return None
        q = project.snapshot()
        _, _, _, errors = T.apply_changes(q, [{'camera': tag, 'dpitch': float(d)}])
        if errors:
            continue
        r = engine.analyse(q)
        if r is None or c.uid not in r.per_cam:
            continue
        md, b = float(r.per_cam[c.uid]['min_dens']), float(r.metrics['blind_m2'])
        if b <= b0 + 1e-9 and md > md0 + max(0.02, 0.005 * md0) and (best is None or md > best['min_dens']):
            best = {'delta': float(d), 'min_dens': md, 'blind_m2': b, 'min_now': md0, 'blind_now': b0}
    return best


def scale_hint(project, settings: dict, budget: float) -> dict | None:
    """The largest image scale (5 % steps) whose rig, built again at it, renders within the bridge's budget (the
    Copilot's own check); None when none fits."""
    from . import copilot as CP
    fit, built = CP._fit_scale(project, settings, budget)
    if fit is None:
        return None
    return {'scale': float(fit), 'mpx': float(built[0]), 'for': float(settings.get('image_scale', 0.5))}


# ====================================================================================================== ranking
def material(old: dict, new: dict) -> bool:
    """Has the evidence changed enough to show a dismissed recommendation again? Another key, another text or status,
    or a figure moved by more than 15 % (and more than a rounding step)."""
    old, new = old or {}, new or {}
    if set(old) != set(new):
        return True
    for k, a in old.items():
        b = new[k]
        fa, fb = _num(a), _num(b)
        if isinstance(a, bool) or isinstance(b, bool) or fa is None or fb is None:
            if a != b:
                return True
            continue
        if abs(fa - fb) > max(0.15 * max(abs(fa), abs(fb)), 1e-3):
            return True
    return False


class Dismissed:
    """Dismissed recommendations: id → the evidence when dismissed (kept per project file by the panel, in QSettings,
    never in the project)."""

    def __init__(self, data=None):
        self.data = {str(k): dict(v) if isinstance(v, dict) else {} for k, v in (data or {}).items()}

    def dismiss(self, rec: Recommendation):
        self.data[rec.id] = dict(rec.evidence)

    def hides(self, rec: Recommendation) -> bool:
        return rec.id in self.data and not material(self.data[rec.id], rec.evidence)

    def prune(self, recs):
        """Forget dismissals whose evidence changed materially (they show again, and stay shown)."""
        by = {r.id: r for r in recs}
        for rid in list(self.data):
            if rid in by and material(self.data[rid], by[rid].evidence):
                del self.data[rid]


def recommend(state: State, context: Context | None = None, dismissed: Dismissed | None = None,
              previous: list | None = None) -> list:
    """Every recommendation for the state, the ones about the current thing first, deduplicated, dismissed ones left
    out. previous: the ids in the order shown last time, so an item whose priority barely moved keeps its place."""
    cx = context or Context()
    recs = []
    for rule in RULES:
        try:
            recs += rule(state, cx)
        except Exception as e:                          # noqa: BLE001 — one rule's trouble never empties the list
            recs.append(_rec(f'error:{rule.__name__}', 'A recommendation rule failed', f'{type(e).__name__}: {e}',
                             'open', 'start', 0.6, source=rule.__name__, target={'page': None}))
    if sum(1 for r in recs if not r.id.startswith('error:')) < 3:
        recs += default_rules(state, cx)
    focus, page = cx.focus(), ('page', cx.page)
    for r in recs:
        r.priority = r.base
        keys = {tuple(a) for a in r.about}
        if keys & focus:
            r.priority += FOCUS_BOOST
        elif page in keys:
            r.priority += PAGE_BOOST
    out, seen, topics = [], set(), {}
    for r in sorted(recs, key=lambda r: (-r.priority, r.id)):
        if r.id in seen:
            continue
        if r.topic and r.topic in topics:
            continue
        seen.add(r.id)
        if r.topic:
            topics[r.topic] = r
        if dismissed is not None and dismissed.hides(r):
            continue
        out.append(r)
    return stable(out, previous)


def stable(recs: list, previous: list | None) -> list:
    """Sort by priority in steps of 20 (a severity level), then the order shown last time, then the id: an item whose
    priority changes a little does not jump around."""
    prev = {rid: i for i, rid in enumerate(previous or [])}
    return sorted(recs, key=lambda r: (-int(r.priority // 20), prev.get(r.id, len(prev)), -r.priority, r.id))


def context_items(recs: list, context: Context | None, n: int = STRIP) -> list:
    """The recommendations about the current thing (the selection, the last edit, a job that just finished, the open
    page), for the live strip."""
    if context is None:
        return []
    focus = context.focus() | {('page', context.page)}
    return [r for r in recs if {tuple(a) for a in r.about} & focus][:n]


def context_label(context: Context | None, items=None) -> str:
    """'For QR · Coverage page': what the strip is about. With items (the strip's), only the parts they are about."""
    if context is None:
        return ''
    about = None if items is None else {tuple(a) for r in items for a in r.about}
    has = lambda keys: about is None or bool(about & set(keys))
    bits = []
    sel = list(context.selection)
    if sel and has(tuple(s[:2]) for s in sel):
        names = [str(s[2] if len(s) >= 3 else s[1]) for s in sel[:3]]
        bits.append('For ' + ', '.join(names) + (' …' if len(sel) > 3 else ''))
    elif context.edits and context.edits[-1].get('index') == context.undo_index and \
            has([('edit', context.edits[-1].get('index'))]):
        bits.append('After: ' + _short(context.edits[-1].get('text', ''), 40))
    if context.job and context.focus() & {('job', context.job.get('kind'))} and \
            has([('job', context.job.get('kind'))]):
        bits.append(f'{JOB_LABEL.get(context.job.get("kind"), context.job.get("kind"))} '
                    + ('finished' if context.job.get('ok', True) else 'failed'))
    if context.page_label:
        bits.append(context.page_label)
    return ' · '.join(bits)


JOB_LABEL = {'drivelog': 'Drive-log check', 'detection': 'Detection quality', 'campaign': 'Campaign',
             'carla_run': 'CARLA drive', 'plan': 'Plan', 'perception': 'Perception range',
             'requirements': 'Requirements', 'asbuilt': 'As-built check'}


# ====================================================================================================== suggest_next
NEXT_KINDS = ('ask', 'goal', 'plan')
NEXT_MAX = 3


def clean_next(items, local=()) -> list:
    """The model's suggest_next items, validated: a label (≤ 60 characters) and a prompt (≤ 400), a kind of ask, goal
    or plan; at most three, duplicates (with each other and with the local recommendations' titles or prompts) left
    out. They never run anything: each becomes a chip the user may click."""
    def norm(s):
        return re.sub(r'\W+', ' ', str(s).lower()).strip()
    taken = {norm(r.title) for r in local} | {norm(r.prompt) for r in local if r.prompt}
    out = []
    for it in items if isinstance(items, list) else []:
        if not isinstance(it, dict):
            continue
        label = ' '.join(str(it.get('label') or '').split())[:60]
        prompt = ' '.join(str(it.get('prompt') or '').split())[:400]
        kind = it.get('kind') if it.get('kind') in NEXT_KINDS else 'ask'
        if not label or not prompt or norm(label) in taken or norm(prompt) in taken:
            continue
        taken |= {norm(label), norm(prompt)}
        if kind == 'plan' and not re.search(r'propose_plan|\bplan\b', prompt, re.IGNORECASE):
            prompt = f'Plan this for my approval with propose_plan: {prompt}'
        out.append(Recommendation(f'next:{norm(label)[:40]}', label, 'Suggested by the model after its answer.', kind,
                                  prompt=prompt, source='Model'))
        if len(out) >= NEXT_MAX:
            break
    return out
