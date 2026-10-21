"""Mix the narration onto the recorded tour and add subtitles.

    python mix.py VOICE_DIR REC_DIR OUT_DIR [--name vantage-demo]

Writes OUT_DIR/NAME.mp4 (H.264 + AAC + a soft English subtitle track), NAME-captioned.mp4 (captions burned in)
and NAME.srt.
"""
import argparse
import json
import os
import re
import subprocess
import wave

import numpy as np
from scipy.signal import resample_poly

SR = 48000


def read_wav(path):
    with wave.open(path) as w:
        sr, n, ch, sw = w.getframerate(), w.getnframes(), w.getnchannels(), w.getsampwidth()
        a = np.frombuffer(w.readframes(n), np.int16 if sw == 2 else np.int32).astype(np.float32)
    a /= 32768.0 if sw == 2 else 2147483648.0
    if ch > 1:
        a = a.reshape(-1, ch).mean(1)
    if sr != SR:
        g = np.gcd(sr, SR)
        a = resample_poly(a, SR // g, sr // g).astype(np.float32)
    return a


def ts(t):
    ms = int(round(t * 1000))
    h, ms = divmod(ms, 3600000)
    m, ms = divmod(ms, 60000)
    s, ms = divmod(ms, 1000)
    return f'{h:02d}:{m:02d}:{s:02d},{ms:03d}'


def _balance(c, limit):
    """Split text into the fewest pieces of at most `limit` characters, of similar length, at spaces."""
    if len(c) <= limit:
        return [c]
    n = -(-len(c) // limit)
    target = len(c) / n
    pieces, cur = [], ''
    for w in c.split():
        if cur and len(cur) + 1 + len(w) > target and len(pieces) < n - 1:
            pieces.append(cur)
            cur = w
        else:
            cur = f'{cur} {w}'.strip()
    pieces.append(cur)
    return pieces


def chunks(text, limit=84):
    """Split a caption into cues of at most two ~42-character lines, at sentence or clause breaks."""
    parts = re.split(r'(?<=[.:;?!])\s+|(?<=,)\s+(?=\S{4,})', text)
    out, cur = [], ''
    for p in parts:
        if cur and len(cur) + 1 + len(p) > limit:
            out.append(cur)
            cur = p
        else:
            cur = f'{cur} {p}'.strip()
    if cur:
        out.append(cur)
    return [piece for c in out for piece in _balance(c, limit)]


def wrap2(c, width=44):
    """Two lines, broken at the space nearest the middle."""
    if len(c) <= width:
        return c
    mid = len(c) / 2
    spaces = [i for i, ch in enumerate(c) if ch == ' ']
    k = min(spaces, key=lambda i: abs(i - mid))
    return c[:k] + '\n' + c[k + 1:]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('voice')
    ap.add_argument('rec')
    ap.add_argument('out')
    ap.add_argument('--name', default='vantage-demo')
    a = ap.parse_args()
    tl = json.load(open(os.path.join(a.rec, 'timeline.json')))
    info = json.load(open(os.path.join(a.voice, 'durations.json')))['lines']
    total = tl['seconds']
    track = np.zeros(int(total * SR) + SR, np.float32)
    cues = []
    for ln in tl['lines']:
        clip = read_wav(os.path.join(a.voice, info[ln['key']]['file']))
        f = int(0.006 * SR)
        clip[:f] *= np.linspace(0, 1, f)
        clip[-f:] *= np.linspace(1, 0, f)
        s0 = int(ln['start'] * SR)
        e0 = min(len(track), s0 + len(clip))
        track[s0:e0] += clip[:e0 - s0]
        # subtitles: split long lines, time each piece by its share of the characters
        parts = chunks(ln['caption'])
        dur = len(clip) / SR
        n = sum(len(p) for p in parts)
        t = ln['start']
        for p in parts:
            d = dur * len(p) / n
            cues.append((t, t + d, wrap2(p)))
            t += d
    track = track[:int(total * SR)]
    peak = float(np.abs(track).max()) or 1.0
    track *= 0.89 / peak
    os.makedirs(a.out, exist_ok=True)
    wav = os.path.join(a.rec, 'narration.wav')
    with wave.open(wav, 'w') as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(SR)
        w.writeframes((np.clip(track, -1, 1) * 32767).astype(np.int16).tobytes())
    srt = os.path.join(a.out, f'{a.name}.srt')
    with open(srt, 'w', encoding='utf-8') as fh:
        for i, (s, e, text) in enumerate(cues, 1):
            nxt = cues[i][0] - 0.02 if i < len(cues) else total
            fh.write(f'{i}\n{ts(s)} --> {ts(min(e + 0.15, nxt))}\n{text}\n\n')
    video = os.path.join(a.rec, 'video.mp4')
    audio_filter = 'loudnorm=I=-16:TP=-1.5:LRA=11,aresample=48000,apad'
    out = os.path.join(a.out, f'{a.name}.mp4')
    subprocess.run(['ffmpeg', '-y', '-loglevel', 'error', '-i', video, '-i', wav, '-i', srt,
                    '-map', '0:v', '-map', '1:a', '-map', '2:s', '-c:v', 'copy', '-af', audio_filter, '-c:a', 'aac',
                    '-b:a', '192k', '-ac', '2', '-c:s', 'mov_text', '-metadata:s:a:0', 'language=eng',
                    '-metadata:s:s:0', 'language=eng', '-metadata', 'title=Vantage · demo tour', '-movflags', '+faststart',
                    '-t', f'{total:.3f}', out], check=True)
    style = ('FontName=IBM Plex Sans,FontSize=17,PrimaryColour=&H00ECEFF0,OutlineColour=&H00111111,BackColour=&H30111111,'
             'BorderStyle=4,Outline=0,Shadow=0,MarginV=34,Alignment=2')
    capd = os.path.join(a.out, f'{a.name}-captioned.mp4')
    esc = srt.replace('\\', '\\\\').replace(':', '\\:').replace("'", "\\'")
    subprocess.run(['ffmpeg', '-y', '-loglevel', 'error', '-i', out, '-vf', f"subtitles='{esc}':force_style='{style}'",
                    '-map', '0:v', '-map', '0:a', '-c:v', 'libx264', '-preset', 'medium', '-crf', '18', '-pix_fmt', 'yuv420p',
                    '-c:a', 'copy', '-movflags', '+faststart', capd], check=True)
    print(f'{out}\n{capd}\n{srt}\n{len(cues)} subtitle cues, {total:.1f} s')


if __name__ == '__main__':
    main()
