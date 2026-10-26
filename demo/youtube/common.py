"""Paths and the voice cache shared by every tour script (no Qt, no app imports: the Kokoro virtualenv imports this
too)."""
from __future__ import annotations

import hashlib
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
DEMO = os.path.dirname(HERE)
ROOT = os.path.dirname(DEMO)
BUILD = os.path.join(DEMO, 'build', 'youtube')        # intermediate: voice WAVs, segments, music (gitignored)
OUT = os.path.join(DEMO, 'out', 'youtube')            # deliverables (gitignored)
VOICE_DIR = os.path.join(BUILD, 'voice')
SEG_DIR = os.path.join(BUILD, 'seg')
DRAFT_DIR = os.path.join(BUILD, 'draft')

VOICE = 'af_heart'
SPEED = 1.0
LANG = 'en-us'
FPS = 30
W, H = 1920, 1080
GAP = 0.4               # seconds of silence at least between two narration lines


def line_hash(spoken: str, voice=VOICE, speed=SPEED) -> str:
    return hashlib.sha1(f'{voice}|{speed}|{LANG}|{spoken}'.encode()).hexdigest()[:16]


def voice_index() -> dict:
    path = os.path.join(VOICE_DIR, 'index.json')
    if os.path.exists(path):
        with open(path, encoding='utf-8') as fh:
            return json.load(fh)
    return {}


def voice_wav(h: str) -> str:
    return os.path.join(VOICE_DIR, f'{h}.wav')


def chapter_voice(cid, lines) -> list[dict]:
    """[{key, file, seconds, caption, spoken}] for a chapter's lines; raises if a line has not been rendered."""
    idx = voice_index()
    out = []
    for i, (cap, spk) in enumerate(lines):
        h = line_hash(spk)
        if h not in idx or not os.path.exists(voice_wav(h)):
            raise SystemExit(f'voice line {cid}_{i} is not rendered yet: run produce.py (or voice.py) first')
        out.append({'key': f'{cid}_{i}', 'hash': h, 'file': voice_wav(h), 'seconds': idx[h]['seconds'],
                    'caption': cap, 'spoken': spk})
    return out
