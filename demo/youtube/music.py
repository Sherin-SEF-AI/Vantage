"""The tour's music, synthesised here (numpy): licence-free by construction.

    python3 demo/youtube/music.py [--out DIR]       writes bed.wav (a seamless loop), whoosh.wav, logo.wav, part.wav,
                                                    card.wav and click.wav (48 kHz stereo float) into DIR
                                                    (default demo/build/youtube/music); cached by VERSION
    python3 demo/youtube/music.py --variant as070   also the assistant episode's own bed, bed-as070.wav (VERSION_AS)

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


# ====================================================================== the assistant episode's bed (as070)
# A richer bed for "An AI assistant for sensor-rig design", selected per episode (chapters.EPISODES[...]['music']) and
# written as bed-as070.wav beside bed.wav: the other episodes' bed and stings are untouched. Same tempo (90 BPM, so the
# intro's cuts stay on the beat), A minor: warm detuned pads that breathe with a soft side-chain on every beat, a
# moving sub bass, a soft kick, a swung shaker, a ping-pong pluck arpeggio and a bell motif, in three 32-bar sections
# (full, sparse, building) over a 96-bar circular loop (256 s) that joins without a seam. The video starts at bar 0,
# the full section, under the cold open.
VERSION_AS = 2          # 2: no bell motif, a softer, narrower arpeggio an octave lower (v1's layers sounded like two pieces)
CHORDS_AS = {
    'Am9': [45, 52, 55, 59, 60], 'Fmaj9': [41, 48, 52, 55, 57], 'C69': [48, 52, 55, 57, 62],
    'Gsus2': [43, 50, 55, 57, 62], 'Em7': [40, 47, 50, 55, 59], 'Dm9': [38, 45, 48, 52, 53],
}
PROG_AS = ['Am9', 'Fmaj9', 'C69', 'Gsus2', 'Am9', 'Fmaj9', 'Dm9', 'Em7',
           'Fmaj9', 'C69', 'Am9', 'Gsus2', 'Dm9', 'Fmaj9', 'Em7', 'Gsus2']      # 16 chords x 2 bars = 32 bars
SECTIONS_AS = [{'kick': True, 'arp': True, 'bell': False, 'shaker': True},      # bars 0-31: full (the cold open)
               {'kick': False, 'arp': False, 'bell': False, 'shaker': True},    # 32-63: sparse, under the talk
               {'kick': True, 'arp': True, 'bell': False, 'shaker': True}]      # 64-95: building back
BELL_AS = [(0, 76), (1.5, 74), (2, 72), (3, 69), (4.5, 72), (6, 71)]           # (beat in a 4-bar phrase, midi)


def _pluck(m, n, decay=5.0, bright=0.3):
    tt = np.arange(n) / SR
    f = hz(m)
    return ((np.sin(2 * np.pi * f * tt) + bright * np.sin(4 * np.pi * f * tt) + 0.08 * np.sin(6 * np.pi * f * tt))
            * np.exp(-tt * decay) * np.minimum(1, tt / 0.003)).astype(np.float32)


def bed_as():
    rng = np.random.default_rng(70)
    n_chords = 3 * len(PROG_AS)
    chord_len = 2 * BAR
    L = int(round(n_chords * chord_len * SR))
    pads = np.zeros((L, 2), np.float32)
    bass = np.zeros(L, np.float32)
    kick = np.zeros(L, np.float32)
    perc = np.zeros((L, 2), np.float32)
    tops = np.zeros((L, 2), np.float32)
    for k in range(n_chords):
        name = PROG_AS[k % len(PROG_AS)]
        notes = CHORDS_AS[name]
        t0 = k * chord_len
        sec = SECTIONS_AS[k // len(PROG_AS)]
        n_body = int(chord_len * SR)
        total = n_body + int(2.4 * SR)
        e = env_adsr(n_body, 1.1, 2.4, total)
        for j, m in enumerate(notes):
            f = hz(m)
            for det, pn in ((-7, -0.55), (0, 0.0), (7, 0.55)):
                s = osc(f * 2 ** (det / 1200), total, rng.random()) * e * (0.15 / (1 + 0.15 * j))
                add_wrap(pads, int(t0 * SR), pan(s, pn * 0.85))
        # an airy octave on top of the voicing, quieter, for width
        s = osc(hz(notes[-1] + 12), total, rng.random()) * e * 0.06
        add_wrap(pads, int(t0 * SR), pan(s, 0.3))
        # the sub bass: root, then the fifth on beat 7, an octave below the voicing
        tt_b = np.arange(int(1.5 * BAR * SR) + int(0.4 * SR)) / SR
        for b0, interval, dur_b in ((0, 0, 6), (6, 7, 2)):
            f = hz(notes[0] - 12 + interval)
            n = int(dur_b * BEAT * SR)
            tt = np.arange(n + int(0.25 * SR)) / SR
            env = np.minimum(1, tt / 0.02) * np.where(tt < n / SR, 1.0, np.exp(-(tt - n / SR) * 18))
            sig = (np.sin(2 * np.pi * f * tt) + 0.22 * np.sin(4 * np.pi * f * tt)) * env * 0.12
            add_wrap(bass, int((t0 + b0 * BEAT) * SR), sig.astype(np.float32))
        del tt_b
        for b in range(8):                               # 8 beats per chord
            tb = t0 + b * BEAT
            if sec['kick'] and b % 2 == 0:               # a soft kick on beats 1 and 3
                n = int(0.42 * SR)
                tt = np.arange(n) / SR
                fr = 48 + 70 * np.exp(-tt * 38)
                th = (np.sin(2 * np.pi * np.cumsum(fr) / SR) * np.exp(-tt * 10)).astype(np.float32) * 0.28
                add_wrap(kick, int(tb * SR), th)
            if sec['shaker']:                            # a swung 16th shaker, accents on the off-beats
                for q in range(4):
                    swing = 0.055 * BEAT if q % 2 else 0.0
                    n = int(0.06 * SR)
                    tt = np.arange(n) / SR
                    nz = rng.standard_normal(n).astype(np.float32) * np.exp(-tt * (70 if q % 2 else 55))
                    vel = (0.060, 0.030, 0.085, 0.036)[q] * (0.8 + 0.4 * rng.random())
                    add_wrap(perc, int((tb + q * BEAT / 4 + swing) * SR), pan(nz * vel, 0.25 if q % 2 else -0.15))
            if sec['arp']:                               # a ping-pong pluck arpeggio in 8ths
                up = [notes[i] + 12 for i in (0, 2, 4, 3, 1, 3, 4, 2)]       # inside the pads' range: one texture
                for h in range(2):
                    m = up[(b * 2 + h) % len(up)]
                    pl = _pluck(m, int(0.8 * SR), 6.0) * 0.07 * (0.9 + 0.2 * rng.random())
                    add_wrap(tops, int((tb + h * BEAT / 2) * SR), pan(pl, 0.15 if (b * 2 + h) % 2 else -0.15))
        if sec['bell'] and k % 2 == 0:                   # the motif over each 4-bar phrase
            for beat, m in BELL_AS:
                n = int(2.6 * SR)
                tt = np.arange(n) / SR
                f = hz(m)
                s = sum(a * np.sin(2 * np.pi * f * r_ * tt) * np.exp(-tt * d) for r_, a, d in
                        ((1.0, 1.0, 1.6), (2.0, 0.25, 2.6), (2.76, 0.18, 3.6), (5.4, 0.05, 7.0)))
                s = (s * np.minimum(1, tt / 0.003) * 0.09).astype(np.float32)
                add_wrap(tops, int((t0 + beat * BEAT) * SR), pan(s, 0.15))
    # the side-chain: every beat the pads and the bass duck a little and swell back (a soft pump)
    tt = np.arange(L) / SR
    ph = (tt % BEAT) / BEAT
    pump = (1 - 0.28 * np.exp(-ph * BEAT * 7.0)).astype(np.float32)
    pads = lowpass_fft(pads, 2600.0) * pump[:, None]
    breathe = (0.9 + 0.1 * np.sin(2 * np.pi * tt / (8 * BAR))).astype(np.float32)
    del tt, ph
    bass = bass * pump
    perc = lowpass_fft(perc, 9000.0)
    hp = perc - lowpass_fft(perc, 2500.0)                # the shaker is a high, airy tick
    mix = pads * breathe[:, None] + np.stack([bass, bass], 1) + np.stack([kick, kick], 1) + hp * 1.3 + \
        lowpass_fft(tops, 6500.0)
    del pads, perc, hp, tops
    wet = reverb_circ(mix, 2.6, 0.30, seed=11)
    wet /= np.sqrt(np.mean(wet ** 2)) / 10 ** (-20 / 20)                 # RMS -20 dBFS, as the tour's bed
    return wet.astype(np.float32)


def build_variant(out, name):
    """The bed of an episode that has its own (chapters.EPISODES[...]['music']): bed-<name>.wav in `out`, cached by
    its version. The stings and the click stay the shared ones (build(out))."""
    build(out)
    if name != 'as070':
        raise SystemExit(f'no music variant {name!r}')
    path = os.path.join(out, f'bed-{name}.wav')
    stamp = os.path.join(out, f'version-{name}.json')
    if os.path.exists(stamp) and os.path.exists(path) and json.load(open(stamp)).get('version') == VERSION_AS:
        print(f'music {name}: up to date')
        return path
    write(path, bed_as())
    json.dump({'version': VERSION_AS, 'sr': SR, 'bpm': BPM, 'loop_s': round(96 * BAR, 3)}, open(stamp, 'w'))
    print(f'music {name}: wrote {path}')
    return path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', default=os.path.join(common.BUILD, 'music'))
    ap.add_argument('--variant', help='also an episode\'s own bed (as070)')
    a = ap.parse_args()
    build(a.out)
    if a.variant:
        build_variant(a.out, a.variant)


if __name__ == '__main__':
    main()
