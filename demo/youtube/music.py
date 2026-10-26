"""The tour's music, synthesised here (numpy): licence-free by construction.

    python3 demo/youtube/music.py [--out DIR]       writes bed.wav (a seamless loop), whoosh.wav, logo.wav, part.wav,
                                                    card.wav and click.wav (48 kHz stereo float) into DIR
                                                    (default demo/build/youtube/music); cached by VERSION

The bed: soft pads in D major (detuned band-limited waves through a gentle low-pass), a sub bass, and in some
sections a quiet pluck arpeggio and a soft heartbeat at 90 BPM; chords change every two bars; reverb by circular
convolution, so the 64-bar loop (about 171 s) joins without a seam. The mixer ducks it under the voice.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import common  # noqa: E402

VERSION = 4
SR = 48000
BPM = 90.0
BEAT = 60.0 / BPM
BAR = 4 * BEAT
RNG = np.random.default_rng(7)

# two bars per chord; midi notes of each voicing (D major: calm, open voicings)
CHORDS = {
    'Dmaj9': [50, 57, 61, 64, 66], 'Bm9': [47, 54, 57, 61, 62], 'Gmaj9': [43, 50, 54, 57, 59],
    'Asus2': [45, 52, 57, 59, 64], 'F#m7': [42, 49, 52, 57, 61], 'Em9': [40, 47, 50, 54, 55],
    'Gmaj7': [43, 50, 54, 59, 62], 'A6': [45, 52, 54, 57, 61],
}
PROG = ['Dmaj9', 'Bm9', 'Gmaj9', 'Asus2', 'Dmaj9', 'F#m7', 'Gmaj7', 'Em9',
        'Bm9', 'Gmaj9', 'Dmaj9', 'A6', 'Gmaj9', 'F#m7', 'Em9', 'Asus2']   # 16 chords x 2 bars = 32 bars
# sections over the 64-bar loop (the progression twice): which layers play
SECTIONS = [{'pulse': False, 'arp': False}, {'pulse': True, 'arp': False},
            {'pulse': True, 'arp': True}, {'pulse': False, 'arp': True}]


def hz(m):
    return 440.0 * 2 ** ((m - 69) / 12)


def _table(n=4096, harmonics=28, tilt=1.35):
    ph = np.arange(n) / n
    t = np.zeros(n)
    for k in range(1, harmonics + 1):
        t += np.sin(2 * np.pi * k * ph) / k ** tilt
    return (t / np.abs(t).max()).astype(np.float32)


TABLE = _table()


def osc(freq, n, phase0=0.0):
    """Band-limited soft wave from the table (harmonics above ~9 kHz faded by the tilt)."""
    ph = (phase0 + np.cumsum(np.full(n, freq / SR))) % 1.0
    return np.interp(ph * len(TABLE), np.arange(len(TABLE) + 1), np.append(TABLE, TABLE[0])).astype(np.float32)


def env_adsr(n, a, r_, total):
    """Attack a s, hold, release r_ s; n samples of note body, total samples with the release."""
    e = np.ones(total, np.float32)
    na = min(int(a * SR), n)
    e[:na] = np.linspace(0, 1, na) ** 1.6
    nr = total - n
    if nr > 0:
        e[n:] = np.linspace(1, 0, nr) ** 2
    return e


def add_wrap(buf, start, sig):
    """Add sig into buf at start, wrapping around the end (the loop is circular)."""
    L = len(buf)
    i = start % L
    while len(sig):
        k = min(len(sig), L - i)
        buf[i:i + k] += sig[:k]
        sig = sig[k:]
        i = 0


def lowpass_fft(x, fc, order=2):
    """Gentle circular low-pass (Butterworth magnitude) of a periodic signal."""
    X = np.fft.rfft(x, axis=0)
    f = np.fft.rfftfreq(x.shape[0], 1 / SR)
    g = 1 / np.sqrt(1 + (f / fc) ** (2 * order))
    return np.fft.irfft(X * (g[:, None] if x.ndim == 2 else g), n=x.shape[0], axis=0).astype(np.float32)


def reverb_circ(x, rt60=2.8, wet=0.32, seed=3):
    """Stereo reverb by circular convolution with an exponentially decaying, darkened noise tail."""
    L = x.shape[0]
    n = int(rt60 * SR * 1.2)
    rng = np.random.default_rng(seed)
    t = np.arange(n) / SR
    decay = np.exp(-6.91 * t / rt60)
    out = np.zeros_like(x)
    for ch in range(2):
        ir = rng.standard_normal(n).astype(np.float32) * decay
        ir[:int(0.012 * SR)] *= np.linspace(0, 1, int(0.012 * SR))          # a short pre-delay
        ir = lowpass_fft(ir, 3200.0)
        ir /= np.sqrt(np.sum(ir ** 2))
        h = np.zeros(L, np.float32)
        h[:min(n, L)] = ir[:min(n, L)]
        src = x[:, ch] if x.ndim == 2 else x
        out[:, ch] = np.fft.irfft(np.fft.rfft(src) * np.fft.rfft(h), n=L).astype(np.float32)
    dry = x if x.ndim == 2 else np.stack([x, x], 1)
    return (dry * (1 - wet) + out * wet).astype(np.float32)


def pan(sig, p):
    """Equal-power pan, p in -1..1."""
    a = (p + 1) * np.pi / 4
    return np.stack([sig * np.cos(a), sig * np.sin(a)], 1).astype(np.float32)


def bed():
    bars_total = 2 * 2 * len(PROG)                       # the progression twice: 64 bars
    L = int(round(bars_total * BAR * SR))
    pads = np.zeros((L, 2), np.float32)
    bass = np.zeros(L, np.float32)
    pulse = np.zeros((L, 2), np.float32)
    chord_len = 2 * BAR
    for k in range(2 * len(PROG)):
        name = PROG[k % len(PROG)]
        notes = CHORDS[name]
        t0 = k * chord_len
        sec = SECTIONS[min(3, int(k / (2 * len(PROG)) * 4))]
        n_body = int(chord_len * SR)
        total = n_body + int(2.2 * SR)
        e = env_adsr(n_body, 1.4, 2.2, total)
        for j, m in enumerate(notes):
            f = hz(m)
            for det, p in ((-6, -0.45), (0, 0.0), (6, 0.45)):
                s = osc(f * 2 ** (det / 1200), total, RNG.random()) * e * (0.16 / (1 + 0.15 * j))
                add_wrap(pads, int(t0 * SR), pan(s, p * 0.8))
        # sub bass: the root an octave down, sine with a touch of the second harmonic
        f = hz(notes[0] - 12)
        tt = np.arange(total) / SR
        s = (np.sin(2 * np.pi * f * tt) + 0.18 * np.sin(4 * np.pi * f * tt)).astype(np.float32) * e * 0.13
        add_wrap(bass, int(t0 * SR), s)
        # the pulse: a soft heartbeat on beats 1 and 3, and an eighth-note pluck arpeggio
        for b in range(8):                               # 8 beats per chord
            tb = t0 + b * BEAT
            if sec['pulse'] and b % 2 == 0:
                n = int(0.45 * SR)
                tt = np.arange(n) / SR
                fr = 46 + 20 * np.exp(-tt * 30)
                th = (np.sin(2 * np.pi * np.cumsum(fr) / SR) * np.exp(-tt * 9)).astype(np.float32) * 0.30
                add_wrap(pulse, int(tb * SR), pan(th, 0.0))
            if sec['arp']:
                up = [notes[i] + 24 for i in (1, 2, 3, 4, 3, 2)]
                for h in range(2):
                    m = up[(b * 2 + h) % len(up)]
                    n = int(0.9 * SR)
                    tt = np.arange(n) / SR
                    pl = (np.sin(2 * np.pi * hz(m) * tt) + 0.25 * np.sin(4 * np.pi * hz(m) * tt)) \
                        * np.exp(-tt * 5.5) * np.minimum(1, tt / 0.004)
                    vel = 0.075 * (0.85 + 0.3 * RNG.random())
                    add_wrap(pulse, int((tb + h * BEAT / 2) * SR), pan(pl.astype(np.float32) * vel, 0.35 if h else -0.35))
    pads = lowpass_fft(pads, 1700.0)
    # a very slow filter-like movement: amplitude breathing over 8 bars
    tt = np.arange(L) / SR
    breathe = (0.88 + 0.12 * np.sin(2 * np.pi * tt / (8 * BAR))).astype(np.float32)
    mix = pads * breathe[:, None] + np.stack([bass, bass], 1) + lowpass_fft(pulse, 5000.0)
    mix = reverb_circ(mix, 3.0, 0.34)
    mix /= np.sqrt(np.mean(mix ** 2)) / 10 ** (-20 / 20)                 # RMS -20 dBFS
    return mix.astype(np.float32)


def whoosh(dur=1.6, hit=1.2, lo=250.0, hi=4200.0):
    n = int(dur * SR)
    t = np.arange(n) / SR
    noise = RNG.standard_normal((n, 2)).astype(np.float32)
    # a band-pass sweeping up to the hit and down after it, done in short frames
    out = np.zeros_like(noise)
    fr = 1024
    for i in range(0, n, fr // 2):
        seg = noise[i:i + fr]
        if len(seg) < 8:
            break
        tc = (i + len(seg) / 2) / SR
        k = min(1.0, tc / hit) if tc < hit else max(0.0, 1 - (tc - hit) / (dur - hit))
        fc = lo * (hi / lo) ** k
        X = np.fft.rfft(seg * np.hanning(len(seg))[:, None], axis=0)
        f = np.fft.rfftfreq(len(seg), 1 / SR)
        g = np.exp(-0.5 * (np.log(np.maximum(f, 1) / fc) / 0.55) ** 2)
        out[i:i + len(seg)] += np.fft.irfft(X * g[:, None], n=len(seg), axis=0)
    e = np.where(t < hit, (t / hit) ** 2.2, np.exp(-(t - hit) * 7)).astype(np.float32)
    out = out * e[:, None]
    out = reverb_circ(np.concatenate([out, np.zeros((int(1.5 * SR), 2), np.float32)]), 1.6, 0.35)
    return (out / np.abs(out).max() * 0.5).astype(np.float32)


def bell(notes, dur=4.5, boom=True, level=0.5):
    n = int(dur * SR)
    t = np.arange(n) / SR
    out = np.zeros((n, 2), np.float32)
    for i, m in enumerate(notes):
        f = hz(m)
        s = np.zeros(n)
        for ratio, amp, dec in ((1.0, 1.0, 1.2), (2.76, 0.32, 3.0), (5.4, 0.12, 6.0), (2.0, 0.2, 2.0)):
            s += amp * np.sin(2 * np.pi * f * ratio * t) * np.exp(-t * dec)
        s *= np.minimum(1, t / 0.003)
        out += pan(s.astype(np.float32) * (0.5 / len(notes)), -0.4 + 0.8 * i / max(1, len(notes) - 1))
    if boom:
        fr = 42 + 30 * np.exp(-t * 18)
        b = np.sin(2 * np.pi * np.cumsum(fr) / SR) * np.exp(-t * 2.2) * 0.55
        out += pan(b.astype(np.float32), 0.0)
    out = reverb_circ(np.concatenate([out, np.zeros((int(2 * SR), 2), np.float32)]), 3.2, 0.42)
    return (out / np.abs(out).max() * level).astype(np.float32)


def click():
    n = int(0.03 * SR)
    t = np.arange(n) / SR
    s = (RNG.standard_normal(n) * np.exp(-t * 900) + 0.6 * np.sin(2 * np.pi * 2300 * t) * np.exp(-t * 400))
    s = lowpass_fft(s.astype(np.float32), 6000.0)
    return pan((s / np.abs(s).max() * 0.25).astype(np.float32), 0.0)


def write(path, a):
    from scipy.io import wavfile
    wavfile.write(path, SR, np.ascontiguousarray(a, np.float32))


def build(out):
    os.makedirs(out, exist_ok=True)
    stamp = os.path.join(out, 'version.json')
    if os.path.exists(stamp) and json.load(open(stamp)).get('version') == VERSION and \
            all(os.path.exists(os.path.join(out, f)) for f in ('bed.wav', 'whoosh.wav', 'logo.wav', 'part.wav',
                                                                 'card.wav', 'click.wav')):
        print('music: up to date')
        return
    write(os.path.join(out, 'bed.wav'), bed())
    write(os.path.join(out, 'whoosh.wav'), whoosh())
    logo = bell([74, 78, 81, 88], 5.0, True, 0.55)
    write(os.path.join(out, 'logo.wav'), logo)
    w = whoosh(1.3, 0.95)
    part = bell([81, 86], 4.0, False, 0.32)
    m = np.zeros((max(len(w), int(0.95 * SR) + len(part)), 2), np.float32)
    m[:len(w)] += w * 0.8
    m[int(0.95 * SR):int(0.95 * SR) + len(part)] += part
    write(os.path.join(out, 'part.wav'), m)
    write(os.path.join(out, 'card.wav'), whoosh(1.0, 0.7, 400.0, 3000.0) * 0.55)
    write(os.path.join(out, 'click.wav'), click())
    json.dump({'version': VERSION, 'sr': SR, 'bpm': BPM, 'loop_s': round(64 * BAR, 3),
               'hits': {'whoosh': 1.2, 'part': 0.95, 'card': 0.7, 'logo': 0.0}}, open(stamp, 'w'))
    print(f'music: wrote {out}')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', default=os.path.join(common.BUILD, 'music'))
    a = ap.parse_args()
    build(a.out)


if __name__ == '__main__':
    main()
