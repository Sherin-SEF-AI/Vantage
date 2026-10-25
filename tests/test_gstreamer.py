"""GStreamer pipelines without OpenCV's GStreamer (the PyPI wheels and the bundles have none): the pipeline runs in the
system's gst-launch-1.0 and the frames come back raw over a pipe; and the environment system programs get when Vantage
runs from the bundle. The pipeline tests run where gst-launch-1.0 is installed (gstreamer1.0-tools) and are skipped
elsewhere."""
import os
import shutil
import sys
import time

import numpy as np
import pytest

from vantage import sysenv
from vantage.core import capture as C

needs_gst = pytest.mark.skipif(not shutil.which('gst-launch-1.0'), reason='gst-launch-1.0 is not installed')


def test_pipeline_arguments_replace_the_appsink():
    tail = ['!', 'videoconvert', '!', 'video/x-raw,format=BGR', '!', 'fdsink', 'name=vantagesink', 'fd=9', 'sync=false']
    jetson = ('nvarguscamerasrc sensor-id=0 ! "video/x-raw(memory:NVMM), width=1920, height=1536" ! nvvidconv ! '
              'video/x-raw,format=BGRx ! videoconvert ! video/x-raw,format=BGR ! appsink drop=1 max-buffers=2')
    args = C.gst_pipeline(jetson, 9)
    assert args[:3] == ['nvarguscamerasrc', 'sensor-id=0', '!'] and 'appsink' not in ' '.join(args)
    assert 'video/x-raw(memory:NVMM), width=1920, height=1536' in args and args[-len(tail):] == tail
    assert C.gst_pipeline('v4l2src device=/dev/video0 !', 9) == ['v4l2src', 'device=/dev/video0'] + tail
    assert C.gst_pipeline('videotestsrc', 9) == ['videotestsrc'] + tail


def test_caps_line_gives_the_frame_size():
    line = ('/GstPipeline:pipeline0/GstFdSink:vantagesink.GstPad:sink: caps = video/x-raw, format=(string)BGR, '
            'width=(int)1920, height=(int)1536, framerate=(fraction)30/1, multiview-mode=(string)mono')
    assert C.parse_caps(line) == {'width': 1920, 'height': 1536, 'fps': 30.0}
    other = line.replace('vantagesink', 'videoconvert0')                       # another element's pad
    assert C.parse_caps(other) is None and C.parse_caps('Setting pipeline to PLAYING ...') is None


@needs_gst
def test_frames_come_back_exact_with_padded_rows():
    # 333 px wide: 999 bytes of BGR per row, padded to 1000 by GStreamer
    g = C.GstPipe('videotestsrc num-buffers=5 pattern=solid-color foreground-color=0xff3366cc ! '
                  'video/x-raw,width=333,height=50,format=RGB')
    assert (g.width, g.height, g.stride) == (333, 50, 1000) and g.isOpened()
    import cv2
    assert (g.get(cv2.CAP_PROP_FRAME_WIDTH), g.get(cv2.CAP_PROP_FPS)) == (333.0, 30.0) and not g.set(0, 1)
    frames = []
    while True:
        ok, f = g.read()
        if not ok:
            break
        frames.append(f)
    g.release()
    assert len(frames) == 5 and frames[0].shape == (50, 333, 3) and frames[0].flags['C_CONTIGUOUS']
    assert all((f == np.array([204, 102, 51], np.uint8)).all() for f in frames)         # 0x3366cc as BGR
    assert not g.isOpened() and g.read() == (False, None)


@needs_gst
def test_a_broken_pipeline_says_why_at_once():
    t = time.monotonic()
    with pytest.raises(RuntimeError, match='no element "nosuchelement"'):
        C.GstPipe('nosuchelement ! fakesink', timeout=10)
    assert time.monotonic() - t < 5


@needs_gst
def test_live_source_reads_a_pipeline(monkeypatch):
    monkeypatch.setattr(C, 'opencv_has_gstreamer', lambda: False)      # the bundles' OpenCV
    ls = C.LiveSource('videotestsrc is-live=true pattern=ball ! video/x-raw,width=320,height=240,framerate=30/1 ! '
                      'videoconvert ! appsink')
    ls.start()
    deadline = time.monotonic() + 10
    while ls.frames < 10 and time.monotonic() < deadline:
        time.sleep(0.05)
    f = ls.latest()
    ls.stop()
    assert ls.frames >= 10 and ls.size == (320, 240) and ls.error == '' and f.shape == (240, 320, 3)
    bad = C.LiveSource('nosuchelement ! appsink')
    bad.start()
    bad._thread.join(10)
    assert 'Cannot open' in bad.error and 'nosuchelement' in bad.error


def test_system_env_outside_the_bundle_is_the_session(monkeypatch):
    monkeypatch.delattr(sys, 'frozen', raising=False)
    env = {'LD_LIBRARY_PATH': '/opt/x', 'PATH': '/usr/bin'}
    assert sysenv.system_env(env) == env and not sysenv.frozen()


def test_system_env_inside_the_bundle_drops_its_libraries(monkeypatch, tmp_path):
    bundle = tmp_path / 'Vantage' / '_internal'
    monkeypatch.setattr(sys, 'frozen', True, raising=False)
    monkeypatch.setattr(sys, '_MEIPASS', str(bundle), raising=False)
    env = {'LD_LIBRARY_PATH': f'{bundle}', 'LD_LIBRARY_PATH_ORIG': '/usr/local/cuda/lib64',
           'QT_PLUGIN_PATH': str(bundle / 'PyQt6' / 'Qt6' / 'plugins'), 'GST_PLUGIN_PATH': '/opt/gst',
           'PATH': '/usr/bin:/bin', 'HOME': '/home/u'}
    out = sysenv.system_env(env)
    assert out == {'LD_LIBRARY_PATH': '/usr/local/cuda/lib64', 'GST_PLUGIN_PATH': '/opt/gst', 'PATH': '/usr/bin:/bin',
                   'HOME': '/home/u'}
    # no session value: the variable goes
    assert 'LD_LIBRARY_PATH' not in sysenv.system_env({'LD_LIBRARY_PATH': str(bundle), 'PATH': '/usr/bin'})
    assert env['LD_LIBRARY_PATH'] == str(bundle)                       # the app's own environment is untouched
    assert os.environ is not out
