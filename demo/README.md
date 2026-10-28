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

## Website clips and stills

`clips.py` records the short silent clips on the website (`site/media`): the real application driven offscreen, one
window per clip at 1600 × 900, dark theme, its 3D views drawn by the shader renderer through a headless EGL context.
Most clips are methods of `Stage` in `clips.py`; the others are `clip(stage, r)` in `clip_<name>.py`. The sample's
reference photos are never shown. The assistant clip talks to `tests/fake_groq.py`, so its model replies are scripted
(the website says so); the lookups, the optimisation and the proposal are computed live. The clips that need a
recording (`drivelog`, `asbuilt`) generate a short synthetic drive with known faults or mounting errors first, as
the tests do.

Each run uses a throw-away settings folder and a throw-away home folder (removed at the end): the recordings and the
sample dataset the clips write go there, and paths the app shows in full are shown from `~` (`tilde()`), so no folder
of the machine that records appears on screen.

`stills.py` takes the website's screenshots with the same stage and recorder, one 1600 × 900 frame each: `window`,
`compare`, `export`, `suspension` and `prefs`.

```bash
python3 demo/clips.py build/clips                      # every clip: <name>.mp4 (web, H.264) and <name>.jpg (poster)
python3 demo/clips.py build/clips --clips hero,bowl    # some of them
python3 demo/stills.py build/stills                    # the stills: <name>.jpg
cp build/clips/*.mp4 build/clips/*.jpg build/stills/*.jpg site/media/    # after looking at every one
```

The poster is the frame at 72 % of a clip unless `POSTER_AT` in `clips.py` names the second. Recording runs at about
real time, plus the web encode; the clips that generate a drive first take 15–30 s longer.

The `automation` clip types `make test` as the command, approves it in the app's own dialog and runs a harmless
standard-library script from the throw-away home (a Makefile written there in the set-up); the `materials` clip adds a
windscreen camera (FW) for the glass and wiper part, as the materials tests do: the sample itself has none. The
`carla` clip shows a real CARLA drive made beforehand, so that CARLA and the recording never share the machine: run
`vantage PROJECT --carla-run $VANTAGE_CLIP_RUNS/<variant>-<YYYYmmdd-HHMMSS>` in a CARLA server on the port the clip
sets (2300 by default, never CARLA's default ports), with `clip_carla.setup`'s settings, and stop the server. The clip
opens the drive with Simulate ▸ CARLA ▸ Open run…, shows its report and its Ground truth tab (image and boxes, a
picked object, instances, range, the lidar from above, semantic). While it records, the app can neither start a server
nor probe CARLA's default ports. The `detection`, `campaign` and `critcarla` clips open working copies of real CARLA
drives (`VANTAGE_CLIP_DETECT`, `VANTAGE_CLIP_CAMPAIGN`, `VANTAGE_CLIP_CRITICAL`) and start nothing; the campaign
reuses all 49 drives and their checks. `groundtruth.jpg`, `actors.jpg` and `carlalive.jpg` are drawn from real drives
by `demo/groundtruth_figure.py`, `demo/actors_figure.py` and `demo/live_figure.py`. The `experiments` clip runs a
6-point factorial and a 512-point Sobol design live (about 10 s, shown as a time-lapse). The validation figure
(`site/media/validation.jpg`) is drawn from the published report by `demo/validation_figure.py`.

## YouTube tour

`demo/youtube/` produces the full narrated tour (about 70 minutes, 1920 × 1080, 30 fps, every feature in 38
chapters): `python3 demo/youtube/produce.py` renders the voice lines (Kokoro, as above), records each chapter that is
missing or changed in its own capped, offscreen process (the app captured at 2× through the EGL shader renderer on a
full-size offscreen screen, `offscreen-screen.json`), then assembles the chapter and part cards, transitions,
synthesised music, the voice mix at −14 LUFS, subtitles, YouTube chapter timestamps, a description and a thumbnail
into `demo/out/youtube/`. `demo/youtube/AUTHORING.md` documents the chapter API (`demo/youtube/chapters/`), the
stage set-ups and the review tools (`review.py`). Every narrated figure is checked against the app while recording.
