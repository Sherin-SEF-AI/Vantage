# Demo tour

A narrated walkthrough of Vantage, recorded from the real application (offscreen, software 3D renderer) at
1920 × 1080, 30 fps. The voice is Kokoro-82M (`af_heart`), rendered locally; nothing is sent to an online service.

| File | What it does |
|---|---|
| `narration.py` | The script: scenes and lines, each with a caption and the text as spoken |
| `voice.py` | Renders every line to a WAV with kokoro-onnx (run in its own virtualenv) |
| `tour.py` | Drives the app scene by scene, starting each line as its action happens; writes `video.mp4` and `timeline.json` |
| `mix.py` | Places the lines on the timeline, normalises loudness (−16 LUFS), muxes video, voice and subtitles |

## Regenerate

```bash
# voice (once): python3 -m venv tts && tts/bin/pip install kokoro-onnx==0.4.7 soundfile
# model files: kokoro-v1.0.onnx and voices-v1.0.bin from github.com/thewh1teagle/kokoro-onnx (release model-files-v1.0)
tts/bin/python demo/voice.py MODELS build/voice --espeak-data SHORT_LINK   # see voice.py for the espeak path note
python3 demo/tour.py build/voice build/rec --batch-script build/script.json  # --preview saves stills instead
python3 demo/mix.py build/voice build/rec demo/out
```

`--batch-script` is the terminal shown in the last scene: a JSON list of `["cmd", text]` and `["out", text, tone]`
taken from a real run of `run_vantage.py --install-check`.

Time-lapses (the tolerance study, the optimiser, the installation check) are marked on screen with the measured
speed-up.

## Website clips

`clips.py` records the short silent clips on the website (`site/media`): the real application driven offscreen, one
window per clip, its 3D views drawn by the shader renderer through a headless EGL context. The sample's reference
photos are never shown. The assistant clip talks to `tests/fake_groq.py`, so its model replies are scripted (the
website says so); the lookups, the optimisation and the proposal are computed live.

```bash
python3 demo/clips.py build/clips                      # every clip: <name>.mp4 (web, H.264) and <name>.jpg (poster)
python3 demo/clips.py build/clips --clips hero,bowl    # some of them
cp build/clips/*.mp4 site/media/                       # then pick the posters (see the site's media folder)
```

