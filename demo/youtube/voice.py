"""Render the tour's narration with Kokoro-82M (kokoro-onnx, voice af_heart), one WAV per line, cached by the line's
spoken text: only new or changed lines are rendered. Runs in the Kokoro virtualenv; nothing is sent anywhere.

    KOKORO_ENV/bin/python demo/youtube/voice.py --models DIR --espeak-data SHORT_PATH [--only intro,projects]

--espeak-data: espeak-ng keeps its data path in a 160-byte buffer, so the virtualenv's long path fails; pass a short
symlink to espeakng_loader's espeak-ng-data directory (inside it, a link espeak-ng-data -> . is also needed). See
demo/voice.py.
"""
from __future__ import annotations

import argparse
import fcntl
import json
import os
import sys
import time

import numpy as np
import soundfile as sf

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import chapters as C  # noqa: E402
from common import LANG, SPEED, VOICE, VOICE_DIR, line_hash, voice_index, voice_wav  # noqa: E402


def trim(a, sr, thr=0.004, keep=0.04):
    idx = np.nonzero(np.abs(a) > thr)[0]
    if not len(idx):
        return a
    k = int(keep * sr)
    return a[max(0, idx[0] - k):min(len(a), idx[-1] + k)]


def pending(only=None):
    """[(cid, i, caption, spoken, hash)] of lines not rendered yet."""
    idx = voice_index()
    out = []
    for cid in C.files():
        if only and cid not in only:
            continue
        for i, (cap, spk) in enumerate(C.lines(cid)):
            h = line_hash(spk)
            if h not in idx or not os.path.exists(voice_wav(h)):
                out.append((cid, i, cap, spk, h))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--models', required=True)
    ap.add_argument('--espeak-data', required=True)
    ap.add_argument('--only', default='')
    a = ap.parse_args()
    only = [x for x in a.only.split(',') if x] or None
    todo = pending(only)
    if not todo:
        print('voice: every line is rendered')
        return
    import phonemizer.backend.espeak.api as api
    orig = api.EspeakAPI.__init__

    def init(self, library, data_path):
        orig(self, library, a.espeak_data)
    api.EspeakAPI.__init__ = init
    from kokoro_onnx import Kokoro
    from kokoro_onnx.config import EspeakConfig
    k = Kokoro(os.path.join(a.models, 'kokoro-v1.0.onnx'), os.path.join(a.models, 'voices-v1.0.bin'),
               espeak_config=EspeakConfig(data_path=a.espeak_data))
    os.makedirs(VOICE_DIR, exist_ok=True)
    t0 = time.time()
    for cid, i, cap, spk, h in todo:
        audio, sr = k.create(spk, voice=VOICE, speed=SPEED, lang=LANG)
        audio = trim(np.asarray(audio, np.float32), sr)
        sf.write(voice_wav(h), audio, sr)
        with open(os.path.join(VOICE_DIR, 'index.lock'), 'w') as lk:    # several agents may render at once
            fcntl.flock(lk, fcntl.LOCK_EX)
            idx = voice_index()
            idx[h] = {'seconds': round(len(audio) / sr, 3), 'sr': sr, 'spoken': spk, 'first_use': f'{cid}_{i}'}
            tmp = os.path.join(VOICE_DIR, f'index.json.{os.getpid()}')
            with open(tmp, 'w', encoding='utf-8') as fh:
                json.dump(idx, fh, indent=1, ensure_ascii=False)
            os.replace(tmp, os.path.join(VOICE_DIR, 'index.json'))
        print(f'{cid}_{i}: {len(audio) / sr:5.1f} s  {cap[:70]}', flush=True)
    print(f'voice: {len(todo)} lines in {time.time() - t0:.0f} s')


if __name__ == '__main__':
    main()
