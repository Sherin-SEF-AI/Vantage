"""Produce the YouTube tour in one command:

    python3 demo/youtube/produce.py                         everything that is missing or out of date
    python3 demo/youtube/produce.py --only intro,coverage   just these segments (and a video of just them)
    python3 demo/youtube/produce.py --rerecord              record the segments again even if up to date
    python3 demo/youtube/produce.py --draft                 960 x 540, 15 fps, 1x pixels: a quick check
    python3 demo/youtube/produce.py --record-only           voice and segments, no video
    python3 demo/youtube/produce.py --name NAME             the deliverables' base name
    python3 demo/youtube/produce.py --episode v020          another episode (default: the full tour; see
                                                            chapters/__init__.py EPISODES); --only ids of one episode
                                                            choose it without --episode

Steps: (a) render the narration lines not rendered yet (Kokoro, in its own virtualenv); (b) record each segment whose
script, voice lines or the recorder's style changed, or that is missing (one capped, offscreen process per segment,
holding the heavy lock only while it records); (c, d) paint the cards and assemble the segments with their
transitions; (e) mix the voice, the music (ducked) and the effects, loudness -14 LUFS / -1 dBTP; (f) write the
subtitles, the YouTube chapters, the description, the thumbnail and contact sheets.

Every process runs under a memory cap with a throw-away settings folder and Qt offscreen, as the repository's safety
rules ask (systemd-run --user --scope -p MemoryMax=... env XDG_CONFIG_HOME=... QT_QPA_PLATFORM=offscreen ...). The
Kokoro virtualenv and models are found from $VANTAGE_TTS (a folder holding env/ and models/) and the short espeak-ng
data link from $VANTAGE_ESPEAK (see AUTHORING.md).
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import chapters as C  # noqa: E402
import common  # noqa: E402

SCRATCH = os.environ.get('VANTAGE_SCRATCH', os.path.join(common.ROOT, 'demo', 'build', 'youtube', 'work'))  # TTS, the lock
TTS = os.environ.get('VANTAGE_TTS', os.path.join(SCRATCH, 'tts'))
ESPEAK = os.environ.get('VANTAGE_ESPEAK', os.path.join(SCRATCH, 'e'))
LOCK = os.environ.get('VANTAGE_LOCK', os.path.join(SCRATCH, 'heavy.lock'))
CFG = os.environ.get('VANTAGE_CFG', os.path.join(SCRATCH, 'cfg-tour'))
RECORD_TIMEOUT = 3600


def capped(cmd, mem='8G', lock=True, timeout=RECORD_TIMEOUT, log=None):
    """Run cmd offscreen under a memory cap with a throw-away settings folder (and the heavy lock), in the episode
    being produced (VANTAGE_EPISODE)."""
    os.makedirs(CFG, exist_ok=True)
    full = (['flock', LOCK] if lock else []) + [
        'systemd-run', '--user', '--scope', '-q', '-p', f'MemoryMax={mem}', '-p', 'MemorySwapMax=0',
        'env', f'XDG_CONFIG_HOME={CFG}', 'QT_QPA_PLATFORM=offscreen', 'PYTHONUNBUFFERED=1',
        f'VANTAGE_EPISODE={C.current()}',
        'timeout', '-s', 'KILL', str(timeout)] + cmd
    if log:
        with open(log, 'w') as fh:
            return subprocess.run(full, stdout=fh, stderr=subprocess.STDOUT, cwd=common.ROOT).returncode
    return subprocess.run(full, cwd=common.ROOT).returncode


def segment_ids(only):
    ids = [e['id'] for e in C.sequence(only) if e['kind'] == 'segment']
    return ids


def up_to_date(cid, draft):
    import record
    d = record.seg_dir(cid, draft)
    meta = os.path.join(d, 'meta.json')
    if not (os.path.exists(meta) and os.path.exists(os.path.join(d, 'video.mp4'))):
        return False
    m = json.load(open(meta))
    key = record.segment_key(cid, draft)
    return all(m.get(k) == v for k, v in key.items())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--only', default='')
    ap.add_argument('--rerecord', action='store_true')
    ap.add_argument('--draft', action='store_true')
    ap.add_argument('--record-only', action='store_true')
    ap.add_argument('--name')
    ap.add_argument('--episode', choices=sorted(C.EPISODES))
    a = ap.parse_args()
    only = [x for x in a.only.split(',') if x] or None
    unknown = [x for x in (only or []) if x not in C.files()]
    if unknown:
        raise SystemExit(f'no chapter file for: {", ".join(unknown)}')
    ep = C.resolve(a.episode, only)
    os.environ['VANTAGE_EPISODE'] = ep                     # this process and every child: voice, record, assemble
    other = [x for x in (only or []) if x not in C.members(ep)]
    if other:
        raise SystemExit(f'not in episode {ep}: {", ".join(other)} (use --episode {C.episode_of(other[0])})')
    t0 = time.time()
    logs = os.path.join(common.DRAFT_DIR if a.draft else common.BUILD, 'logs')
    os.makedirs(logs, exist_ok=True)
    ids = segment_ids(only)
    tag = ('' if ep == C.DEFAULT_EPISODE else f'-{ep}') + (f'-{"-".join(only)}'[:80] if only else '')
    print(f'episode {ep}; segments: {", ".join(ids)}', flush=True)

    # (a) voice
    py = os.path.join(TTS, 'env', 'bin', 'python')
    rc = capped([py, os.path.join(HERE, 'voice.py'), '--models', os.path.join(TTS, 'models'), '--espeak-data', ESPEAK,
                 '--only', ','.join(ids)], mem='3G', lock=False, timeout=3600, log=os.path.join(logs, f'voice{tag}.log'))
    print(open(os.path.join(logs, f'voice{tag}.log')).read().strip().splitlines()[-1:], flush=True)
    if rc:
        raise SystemExit(f'voice failed ({rc}): see {logs}/voice{tag}.log')

    # (b) the segments, one capped process each
    failed = []
    for cid in ids:
        if not a.rerecord and up_to_date(cid, a.draft):
            print(f'  {cid}: up to date', flush=True)
            continue
        t1 = time.time()
        log = os.path.join(logs, f'record-{cid}.log')
        cmd = ['python3', os.path.join(HERE, 'record.py'), cid] + (['--draft'] if a.draft else [])
        rc = capped(cmd, mem='8G', lock=True, log=log)
        tail = [ln for ln in open(log).read().splitlines() if ln.startswith(f'{cid}:')]
        print(f'  {cid}: {"recorded" if rc == 0 else f"FAILED ({rc})"} in {time.time() - t1:.0f} s  {tail[-1] if tail else ""}',
              flush=True)
        if rc:
            failed.append(cid)
    if failed:
        print(f'failed: {", ".join(failed)} (logs in {logs}); the video leaves them out', flush=True)
    if a.record_only:
        return

    # (c-f) cards, assembly, mix, loudness, texts
    cmd = ['python3', os.path.join(HERE, 'assemble.py')] + (['--draft'] if a.draft else []) + \
          (['--only', ','.join(only)] if only else []) + (['--name', a.name] if a.name else []) + ['--episode', ep]
    alog = os.path.join(logs, f'assemble{tag}.log')
    rc = capped(cmd, mem='6G', lock=True, timeout=6 * 3600, log=alog)
    print(open(alog).read().strip(), flush=True)
    if rc:
        raise SystemExit(f'assembly failed ({rc}): see {alog}')
    print(f'produced in {(time.time() - t0) / 60:.1f} min', flush=True)


if __name__ == '__main__':
    main()
