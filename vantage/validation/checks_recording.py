"""Checks of the recording checks: the drive-log check's frozen-camera detector and rig health's capture-time skew
against the sync plan."""
from __future__ import annotations

import numpy as np

from .suite import Outcome, check


def _noisy_frames(rng, n, size=(48, 64), sigma=2.0):
    """n frames of one still scene through a camera with sensor noise (every frame differs, as a real camera's do)."""
    base = rng.uniform(40, 200, size)
    return [np.clip(base + rng.normal(0.0, sigma, size), 0, 255).astype(np.uint8) for _ in range(n)]


@check('recording.frozen_runs', 'recording', 'A frozen camera is found with its repeated frames',
       'frames wrongly counted (missed or extra) by the drive-log check\'s freeze detector (content fingerprints, '
       'runs of images identical to the one before) on 300 frames of a still scene with sensor noise into which a '
       'stalled camera\'s repeats are written at known places (runs of 1, 3 and 10 repeats, one after a dropped '
       'frame; the same frames also as 0.8 MB raw payloads, fingerprinted by samples), checked against rig health\'s '
       'own '
       'frozen counter on the same frames',
       'the runs as written into the sequence; rig health\'s independent counter (pixel equality of the thumbnails)',
       0.0, 'frames', refs=('rfc7693',))
def _frozen(ctx):
    from ..core import drivelog as DL
    from ..core import rig_health as RH
    rng = ctx.rng('recording.frozen_runs')
    frames = _noisy_frames(rng, 300)
    truth = {40: 1, 100: 3, 200: 10}                      # held frame index -> repeats after it
    seq, stamps, t = [], [], 0
    for k, f in enumerate(frames):
        if k == 150:
            t += 1                                        # a dropped frame: a gap in the stamps, no repeat
        seq.append(f)
        stamps.append(t)
        t += 1
        for _ in range(truth.get(k, 0)):
            seq.append(f)
            stamps.append(t)
            t += 1
    worst, detail = 0, {}
    for name, digest in (('encoded', lambda f: DL.content_digest(f.tobytes())),
                         ('raw 0.8 MB', lambda f: DL.content_digest(np.repeat(np.repeat(f, 16, 0), 16, 1).tobytes()))):
        dig = [digest(f) for f in seq]
        runs = DL.frozen_runs(np.array(stamps) * 100_000_000, dig)
        found = {h: n for h, a, b, n in runs}
        # held frame positions in the sequence
        held_at, pos = {}, 0
        for k in range(len(frames)):
            if k in truth:
                held_at[pos] = truth[k]
            pos += 1 + truth.get(k, 0)
        miss = sum(abs(found.get(h, 0) - n) for h, n in held_at.items())
        extra = sum(n for h, n in found.items() if h not in held_at)
        worst = max(worst, miss + extra)
        detail[f'{name}: runs found'] = len(runs)
        detail[f'{name}: frames found'] = int(sum(found.values()))
    st = RH.StreamStats(10.0)
    rh_runs, cur = [], 0
    for k, f in enumerate(seq):
        st.add(f, k * 0.1)
        if st.frozen_run == 0 and cur:
            rh_runs.append(cur)
        cur = st.frozen_run
    if cur:
        rh_runs.append(cur)
    detail['frames repeated (written)'] = int(sum(truth.values()))
    detail['rig health runs'] = sorted(rh_runs)
    worst = max(worst, abs(sum(rh_runs) - sum(truth.values())))
    # a still scene with noise never repeats
    still = _noisy_frames(rng, 200, sigma=1.0)
    false = DL.frozen_runs(np.arange(200), [DL.content_digest(f.tobytes()) for f in still])
    detail['runs in a still scene with noise'] = len(false)
    return Outcome(float(worst + len(false)), detail, ok=sorted(rh_runs) == sorted(truth.values()))


def _rig(fps=10.0):
    """Three level pinhole cameras ahead, left and right on their own trigger lines, and a 10 Hz lidar (the timing
    checks' bare rig)."""
    from .checks_rig import SIDES, _timing_rig
    p = _timing_rig(SIDES, fps=fps)
    for i, cam in enumerate(p.variant.cameras):
        cam.sync = 'ABC'[i]
    return p


@check('recording.capture_skew', 'recording', 'Rig health judges the capture skew against the sync plan',
       'largest difference, in milliseconds, of rig health\'s capture-time skew from the skew written into the '
       'capture times of three cameras of a lidar-synchronised rig (fired 25 and 75 ms after the front camera, as '
       'the sweep crosses them, with 0.4 ms of jitter): every camera on time, then the left one 7 ms late, then '
       'every camera fired together; and without the plan the stagger itself',
       'the trigger times of a clockwise 10 Hz lidar sweep written out in the check (the front camera at 0 ms, the '
       'right one at 25 ms, the left one at 75 ms)', 0.5, 'ms', refs=('ieee1588',))
def _skew(ctx):
    from ..core import rig_health as RH
    rng = ctx.rng('recording.capture_skew')
    planned = RH.planned_offsets(_rig())
    sweep = {'F': 0.0, 'R': 25.0, 'L': 75.0}
    t = np.arange(40) * 0.1 + 1000.0
    detail, worst = {}, 0.0

    def times(late=None, together=False):
        out = {}
        for tag, ms in sweep.items():
            off = 0.0 if together else ms + (late or {}).get(tag, 0.0)
            out[tag] = t + off / 1000.0 + rng.normal(0.0, 0.0004, len(t))
        return out
    for name, tt, want in (('on time', times(), {'R': 0.0, 'L': 0.0}),
                           ('L 7 ms late', times({'L': 7.0}), {'R': 0.0, 'L': 7.0}),
                           ('fired together', times(together=True), {'R': -25.0, 'L': 25.0})):
        offs, ref = RH.capture_offsets(tt, planned)
        for tag, w in want.items():
            e = abs(offs[tag]['skew_ms'] - w)
            worst = max(worst, e)
            detail[f'{name}: {tag} skew ms'] = round(offs[tag]['skew_ms'], 3)
        detail[f'{name}: reference'] = ref
    skew, who = RH.sync_skew(times())
    worst = max(worst, abs(skew - 25.0))                   # without a plan the nearest frame is 25 ms away
    detail['without the plan: worst skew ms'] = round(skew, 3)
    detail['planned offsets ms'] = {k: round(v[0], 3) for k, v in planned.items()}
    rel = {k: (v[0] - planned['F'][0]) % 100.0 for k, v in planned.items()}
    worst = max(worst, max(abs(rel[k] - sweep[k]) for k in sweep))
    return Outcome(worst, detail)
