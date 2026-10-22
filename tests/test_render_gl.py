"""The shader renderer, run headless (EGL) in a subprocess; skipped where there is no EGL OpenGL."""
import json
import os
import subprocess
import sys

import numpy as np
import pytest

HERE = os.path.dirname(os.path.abspath(__file__))


@pytest.fixture(scope='module')
def run(tmp_path_factory):
    out = tmp_path_factory.mktemp('gl')
    r = subprocess.run([sys.executable, os.path.join(HERE, 'gl_harness.py'), str(out)], capture_output=True, text=True,
                       timeout=600)
    lines = [ln for ln in r.stdout.splitlines() if ln.startswith('{')]
    if not lines:
        pytest.fail(f'harness failed: {r.stderr[-2000:]}')
    doc = json.loads(lines[-1])
    if 'skip' in doc:
        pytest.skip(f'no headless OpenGL: {doc["skip"]}')
    return doc, out


def test_shaders_compile_and_draw(run):
    doc, out = run
    assert doc['init'], doc['error']
    assert doc['gl_error'] == 0 and doc['gl_error_end'] == 0
    assert doc['car_pixels'] > 0.15                        # the car is there
    lit = np.load(out / 'lit.npy')
    assert lit.shape == (400, 640, 4) and lit[..., 3].min() == 255


def test_shadows_darken_the_ground(run):
    doc, _ = run
    assert doc['shadow_darker'] > 0.5


def test_camera_light_matches_the_cpu_visibility(run):
    doc, _ = run
    assert doc['camlight_seen'] > 200
    assert doc['camlight_agree'] > 0.97


def test_bowl_texture_is_drawn(run):
    doc, _ = run
    assert doc['bowl_green'] > 0.2


def test_leaves_the_pixel_store_as_qpainter_expects(run):
    """QPainter's GL text cache uploads glyph rows assuming the default alignment of 4; 1 shears every label."""
    doc, _ = run
    assert doc['pixel_store'] == [4, 4]
