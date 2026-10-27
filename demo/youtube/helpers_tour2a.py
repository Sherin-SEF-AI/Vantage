"""Helpers for tour2-a's chapters (41 critical targets, 42 road and body motion). Imported inside scene()."""
from __future__ import annotations

import time


def light_wait_for(r, cond, timeout=180.0, compose_hz=2.0):
    """Record in real time while the app works until cond() is true, composing a new frame at most compose_hz times
    a second and repeating it in between (as helpers_tourc.light_wait does for a fixed time). The recorder, which
    composes 2x frames on the main thread, then leaves the computer to the app's own background job: the job runs
    about as fast as it does without a recording, so the run time the page shows is the app's, and a page that re-runs
    by itself only after quick runs keeps doing so. The picture rate drops meanwhile; keep zooms out of it.
    Raises TimeoutError after `timeout` s."""
    t0, f0 = time.time(), r.f
    img = None
    while True:
        r.app.processEvents()
        if img is not None and cond():
            break
        if time.time() - t0 > timeout:
            raise TimeoutError(f'light_wait_for: still waiting after {timeout} s')
        r._advance()
        img = r.compose()
        want = int((time.time() - t0) * r.fps)
        for _ in range(max(1, want - (r.f - f0))):
            r._write(img)
            r.f += 1
        nxt = time.time() + 1.0 / compose_hz
        while time.time() < nxt:
            r.app.processEvents()
            if cond():
                break
            time.sleep(0.01)
    behind = int((time.time() - t0) * r.fps) - (r.f - f0)       # the job ended between two frames: catch up
    for _ in range(max(0, behind)):
        r._write(img)
        r.f += 1
    r.events.append({'kind': 'wait', 'start': round(f0 / r.fps, 3), 'video_s': round((r.f - f0) / r.fps, 3),
                     'wall_s': round(time.time() - t0, 3), 'speed': 1.0})
    r.tick()
