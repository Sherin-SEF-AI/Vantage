"""Render the narration with Kokoro (kokoro-onnx), one WAV per line, plus durations.json.

    python voice.py MODEL_DIR OUT_DIR [--voice af_heart] [--speed 1.0] [--espeak-data SHORT_PATH]

MODEL_DIR holds kokoro-v1.0.onnx and voices-v1.0.bin (github.com/thewh1teagle/kokoro-onnx releases).
--espeak-data: espeak-ng keeps its data path in a 160-byte buffer, so a long virtualenv path fails; pass a short
symlink to espeakng_loader's espeak-ng-data directory (inside it, a link espeak-ng-data -> . is also needed).
"""
import argparse
import json
import os
import sys

import numpy as np
import soundfile as sf

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from narration import lines  # noqa: E402


def trim(a, sr, thr=0.004, keep=0.04):
    idx = np.nonzero(np.abs(a) > thr)[0]
    if not len(idx):
        return a
    k = int(keep * sr)
    return a[max(0, idx[0] - k):min(len(a), idx[-1] + k)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('models')
    ap.add_argument('out')
    ap.add_argument('--voice', default='af_heart')
    ap.add_argument('--speed', type=float, default=1.0)
    ap.add_argument('--espeak-data')
    a = ap.parse_args()
    if a.espeak_data:
        import phonemizer.backend.espeak.api as api
        orig = api.EspeakAPI.__init__

        def init(self, library, data_path):
            orig(self, library, a.espeak_data)
        api.EspeakAPI.__init__ = init
    from kokoro_onnx import Kokoro
    k = Kokoro(os.path.join(a.models, 'kokoro-v1.0.onnx'), os.path.join(a.models, 'voices-v1.0.bin'))
    os.makedirs(a.out, exist_ok=True)
    info = {}
    for sid, i, cap, spk in lines():
        audio, sr = k.create(spk, voice=a.voice, speed=a.speed, lang='en-us')
        audio = trim(np.asarray(audio, np.float32), sr)
        name = f'{sid}_{i}.wav'
        sf.write(os.path.join(a.out, name), audio, sr)
        info[f'{sid}_{i}'] = {'file': name, 'seconds': round(len(audio) / sr, 3), 'caption': cap, 'spoken': spk}
        print(f'{sid}_{i}: {len(audio) / sr:5.1f} s  {cap[:70]}', flush=True)
    with open(os.path.join(a.out, 'durations.json'), 'w') as fh:
        json.dump({'sample_rate': sr, 'voice': a.voice, 'lines': info}, fh, indent=1)
    print(f'total {sum(v["seconds"] for v in info.values()):.1f} s of speech')


if __name__ == '__main__':
    main()
