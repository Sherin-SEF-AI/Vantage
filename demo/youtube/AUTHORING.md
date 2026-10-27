# Writing a chapter of the YouTube tour

The tour is recorded from the real application, one segment per chapter, each in its own process, then assembled with
cards, transitions, narration, music and subtitles by `produce.py`. This page is everything a chapter needs: the file
format, the recorder and stage API, the look, timing, the commands, and the pitfalls already hit.

Read the repository's rules first (they apply to every command below): every Python process under a memory cap,
offscreen, with a throw-away settings folder; never touch `~/.config/Vantage` or the user's running app; never show
the sample's reference photos or a machine path; never call the Groq API or any network service; never commit.

## Files

| Path | What it is |
|---|---|
| `demo/youtube/chapters/<nn>_<id>.py` | one chapter: `META`, `LINES`, `scene(stage, r)` (see below) |
| `demo/youtube/chapters/__init__.py` | the plans: the tour's 38 chapters (`PLAN`, `PARTS`), the other episodes' (`PLAN_V020`, `PARTS_V020`) and what each episode's deliverables say (`EPISODES`; see Episodes below) |
| `demo/youtube/studio.py` | the recorder `Rec` (pointer, menus, zooms, narration, overlays) and the `Stage` (windows, set-ups) |
| `demo/youtube/gfx.py` | the motion graphics: overlay layers, cards, the logo, the end card, the terminal, colours, easing |
| `demo/youtube/record.py` | records one segment in this process |
| `demo/youtube/voice.py` | renders narration lines with Kokoro (cached per line) |
| `demo/youtube/cards.py` | chapter and part cards (`python3 demo/youtube/cards.py preview <id>`) |
| `demo/youtube/music.py` | the synthesised music bed and stings |
| `demo/youtube/assemble.py` | video + mix + loudness + SRT + chapters + description + thumbnail + contact sheets |
| `demo/youtube/produce.py` | the one command that runs all of the above |
| `demo/youtube/review.py` | frames, a contact sheet and the narration timing of a segment or a video |
| `demo/build/youtube/` | voice WAVs (`voice/`), segments (`seg/<id>/`), music, logs; `draft/` for `--draft` runs |
| `demo/out/youtube/` | the deliverables |

Existing chapters to copy from: `00_intro.py` (full-frame 3D, cards painted in a scene, statements, an agenda),
`01_projects.py` (the start window, the palette, menus, combo boxes, dialogs shown without blocking), `08_housings.py`
(split views, a sensor moved in the Inspector, figures read from the app), `09_coverage.py` (top view, hover read-outs,
bar hover, variants, typing into a field, the finer grid), `99_outro.py` (a card-only segment).

## A chapter file

```python
"""17 Tolerance, optimiser and variants: what this chapter shows, and every figure it narrates with where it comes
from (and that it is checked with r.expect while recording)."""

META = {
    'id': 'tolerance',                       # the id in chapters.PLAN (file 17_tolerance.py)
    'part': 3,                               # 1-5 (0 intro, 6 outro)
    'title': 'Tolerance, optimiser and variants',
    'promise': 'How sure is the layout, and is there a better one?',     # one line on the chapter card
    'menu': 'Analysis ▸ Tolerance study  ·  Layout ▸ Optimise',  # the card and the lower third show it
}

LINES = [   # (caption: written English, British spelling; spoken: what the voice reads)
    ("Brackets are never perfect. The tolerance study perturbs every camera 500 times within the tolerances you set.",
     "Brackets are never perfect. The tolerance study perturbs every camera five hundred times, within the "
     "tolerances you set."),
    ...
]


def scene(stage, r):
    from PyQt6.QtCore import QPointF, QRectF        # import Qt and the app INSIDE scene(): see "Rules for the file"
    w = stage.window()                              # the sample, dark theme, 1920 x 1080
    r.win = w
    r.lower_third()                                 # the title and META['menu'], bottom left
    r.say(0)                                        # line 0 starts now
    r.menu('Analysis', 'Tolerance study')           # ... and the menu opens with the words
    pg = w.page('tolerance')
    r.click(pg.b_run)
    r.wait_for(lambda: 'tolerance' not in w._jobs, timeout=300, speed=4)   # a time-lapse, marked x4 (measured)
    r.zoom(pg.tornado, 1.0)
    r.figure('Worst camera', 1.234, 'm²', 3, src=some_label)          # a real value, its source outlined
    r.finish_speech(0.4)
    r.zoom(None, 0.8)
```

Rules for the file:

* `META` and `LINES` are plain literals (no expressions, no names): the voice renderer and `produce.py` read them with
  `ast`, without importing the file or Qt. Unicode escapes in strings are fine.
* Import PyQt, the app and `gfx` inside `scene()` (the Kokoro environment has no Qt).
* Every line must be said exactly once, in any order (`r.say` raises on a repeat, the recorder raises at the end if
  a line was never said). Change a line's spoken text and only that line is re-rendered; the segment is re-recorded
  because its timing changed.
* A segment is re-recorded when its file, one of its spoken lines, or `studio.STYLE_VERSION` changes.

## Narration

* Kokoro `af_heart` reads the `spoken` text as written. It speaks about **2.7 words a second** (measured on these
  chapters: 2.6 to 2.9); plan 2.5 words a second to be safe. A line of 20 to 30 words (8 to 11 s) suits one action
  and its result. `r.line(i)` is the rendered line's length in seconds.
* Spell for speech the way the existing chapters do: "Tee-gor", "E V", "Sturdy-Cam thirty-one", "Jetson A G X Orin",
  "G M S L two", "G N S S", "I M U", "O S one", "E C U", "R O S two", "mac O S", "G P L", "dot vantage". Write numbers
  in words in `spoken` ("thirteen point seven per cent", "zero point two four square metres", "fifty-three per cent")
  and in figures in the caption ("13.7 per cent", "0.24 square metres", "53 per cent"). Say "the inspector" in lower
  case in `spoken` (capitals are for the caption).
* Plain, specific, in the README's voice; British spelling in captions (colour, centre, metres, analyse, licence).
  Say what is on screen and what it means; no hype words.
* Every figure narrated is the app's own output on the sample: read it from the app in the scene and check it with
  `r.expect(shown, narrated, what)`, which raises (and the recording fails) if the app ever shows something else.
  Library figures from datasheets are "nominal" or "typical", as the app labels them.

## The recorder `r` (studio.Rec)

Coordinates: everything is in the window's logical pixels (1920 x 1080), before any zoom. A **target** is a widget,
a `QRect`/`QRectF`, an `(x, y, w, h)` tuple, or a callable returning one of those (evaluated every frame, so it
follows widgets that move and zooms). Helpers that make targets: `r.wrect(widget)`, `r.local_rect(widget, rect)`
(a rectangle in the widget's own coordinates), `r.tree_item(tree, 'ROI')` (an Outliner row by its text).

Time and narration

| Call | Does |
|---|---|
| `r.hold(s)`, `r.tick(n)`, `r.until(t)`, `r.now()` | record frames for s seconds / n frames / until t |
| `r.say(i)` (or `r.say(chapter_id, i)`) | start line i now; if the previous line is still playing it first waits for it plus 0.4 s, so the action you write next starts with the words |
| `r.finish_speech(pad=0.3)` | wait until the current line has finished (plus pad) |
| `r.speech_left()`, `r.line(i)` | seconds left of the current line; length of line i |
| `r.until_word(i, 'fragment')`, `r.at_word(i, 'fragment')` | wait until (the time when) a fragment of line i's spoken text is said, estimated from its position in the text |
| `r.wait_for(cond, timeout=180, speed=1)` | record while the app works until `cond()`; the video keeps pace with the wall clock divided by speed: 1 = real time, >1 = a time-lapse with a "TIME-LAPSE ×n" badge showing the factor measured from the clock; raises TimeoutError |
| `r.timelapse(factor)` / `r.timelapse(None)` | show / hide the badge yourself (when you compress time another way and measured the factor) |

Pointer and input (from demo/tour.py, unchanged)

| Call | Does |
|---|---|
| `r.move(QPointF, dur)`, `r.move_to(widget, local=None, dur)` | glide the pointer; `r.cursor_on = False` hides it |
| `r.click(widget, local=None, dur=0.7, real=True)` | move, ripple, and a real click (`real=False`: only the look) |
| `r.drag(widget, a, b, dur)` | press, drag, release with real mouse events |
| `r.menu('View', 'Overlays', 'Heatmap', 'Density')` | open the menu bar's menu, walk the path (items match by the start of their text), trigger the last |
| `r.peek_menu('File', 'Save as', hold=1.4)` | open a menu path to show it, close it without triggering |
| `r.toolbar('view_top')` | click a toolbar action by its id (`w.actions_by_id`) |
| `r.choose(combo, index_or_text)` | open a combo box, move to the item in its list, pick it (emits `activated`) |
| `r.enter(field, '3')` | click a text or number field, select it, type, press Enter (the app's spin boxes commit on Enter) |
| `r.type(widget, text, cps=16)` | type at a natural pace |
| `r.hover(widget, local, dur)`, `r.glide(widget, a, b, dur)` | move with mouse-move events sent to the widget (hover read-outs, highlights) |
| `r.send_move(widget, pt)`, `r.leave(widget)` | one mouse-move / a leave event, straight to the widget |
| `r.ripple()` | a click ripple at the pointer (also a click sound in the mix) |

Camera

| Call | Does |
|---|---|
| `r.zoom(target, dur=1.0, wait=True, margin=24)` | ease the frame to a target, overshoot-free (quintic), zooming geometrically so it feels even; `None` = the whole frame; `dur=0` = a cut. Deepest zoom 2.6x; text is crisp to 2x (frames are grabbed at 2x) |
| `r.crossfade(dur)` | dissolve from the last frame into what follows (a page switch, a theme change, a new window) |
| `stage.orbit(r, vp, d_az, dur, d_el=0, d_dist=0, linear=False)` | turn a 3D view's camera, eased |
| `stage.fly(r, vp, az=, el=, dist=, target=, dur=1.5)` | ease a 3D view's camera to a pose |

Overlays (the look; one accent colour, the app's blue)

| Call | Shows |
|---|---|
| `r.lower_third(title=None, sub=None, dur=4.5)` | bottom-left title + menu path (defaults from META). **Every chapter starts with one** |
| `r.callout(target, text='', dur=3, style='box'/'circle', dim=0.38, pad=6, side='auto')` | outline drawn on with a glow, the rest dimmed, an optional label on a leader line |
| `r.figure(label, value, unit, decimals, dur=3.5, src=target, start=0, note='')` | a stat tile counting up (from `start`) to a value **read from the app**; `src` outlines where the app shows it and the tile sits beside it |
| `r.arrow(a, b, text='', dur=2.5)` | an accent arrow between two points or targets |
| `r.key_combo('Ctrl+K', dur=1.8)` | keycaps, pressed as they appear |
| `r.note(text, dur=None)` / `r.note(None)` | a chip at the top centre ("Assistant replies scripted for this video...") |
| `r.statement(text, sub='', dur=3.5)` | a big line over a dark scrim on the left (openers, summaries over a full-frame view) |
| `r.agenda(items, title)` + `layer.reveal(r, i)` | a numbered list revealed item by item |
| `r.terminal(script, title='bash')` | a terminal card typing `[('cmd', text), ('out', text, 'ok'/'warn'/'bad'/'')]`; returns its length in s; `r.card = None` ends it |
| `r.card = fn(rec) -> QImage` | any full-frame painting instead of the app (see the intro's logo, the outro) |
| `r.close_layer(layer, after=0)` | end any overlay early; every call above returns its layer |

Also: `r.expect(shown, narrated, what)` (honesty guard), `r.still(name)` (save the frame to `stills/`),
`r.mark(name, **data)` (a timeline event; `r.mark('music', db=-6, dur=8)` raises the music for a stretch without voice,
`r.mark('sfx', sound='part'|'card'|'whoosh'|'logo')` plays a sting with its hit at that moment).

A persistent indicator is drawn automatically: a hairline map of the 38 chapters along the bottom edge (this one in the
accent colour) and, on a full un-zoomed frame without a lower third, the part and chapter at the top centre.

## The stage (studio.Stage, built on demo/clips.py's Stage)

| Call | Gives |
|---|---|
| `stage.window(edit=None)` | the sample project (6-cam · v3) in a main window, 1920 x 1080, dark theme, photos removed; `edit(p)` changes the project first |
| `stage._rig_window(edit=None)` | the sample with its **Data rig** variant active (6 cameras, roof lidar L1 OS1-128, five radars RD1-RD5) |
| `stage._nav_window(edit=None)` | the data rig plus two GNSS antennas and a dual-antenna INS where Vehicle ▸ Add puts them |
| `stage.vp` | the window's first 3D view |
| `stage.settle(timeout)` | let the window and the analysis catch up **without recording** (time stands still in the video) |
| `stage.wait_analysis(r)` | record in real time until the live analysis has a result newer than the last one seen; `stage.mark_result()` first if you started the window yourself and then edit |
| `stage.start_window(recent=[paths])`, `stage.save_project(name, edit, thumb_from)` | the start window with real recent projects saved in the throw-away home (paths shown from ~) |
| `stage.film_centred(r, widget)`, `stage.film_window(r, w)` | film a dialog on its own, centred on the stage background / film a main window |
| `stage.full_view(provider, cam_from=None)` | a chromeless 3D view filling the frame (cold opens, backgrounds): `r.win = view` |
| `stage.car_box(w)` | the vehicle's box with a margin, for `view.frame_box(lo, hi, 0.92)` |
| `stage.home`, `stage.tilde(widgets...)`, `stage.tilde_path_label(page)` | the throw-away home folder; show paths in widgets from `~` |
| `stage.extra.append(widget)` | widgets you create that must be hidden and deleted when the segment ends |

Prepared data for the calibration chapters (demo/clip_*.py, importable from a scene):

* `clip_asbuilt.make_drive(dest)`: a 2.5 s synthetic ROS 2 bag of the data rig as built (ML pitched 2° and rolled 1°,
  R yawed 1.5°), about 15 s to make; put it under `os.path.join(stage.home, 'drives', ...)`.
* `clip_drivelog.make_bag(work, out)`: a 1.5 s bag with stream faults (MR drops every tenth frame, R's stamps 7 ms
  late, QR missing) and F 0.6° off in yaw.
* Each `clip_<name>.py` and each `clips.Stage` method shows one feature in about 15 s: read the matching one before
  scripting a chapter (`clips.Stage.assistant` for the assistant, `clip_lidarcalib`, `clip_fleet`, `clip_dataset`,
  `clip_library`, `clip_loadcases`, ...). They are the tested way to reach every page from its real menu.
* The assistant: `tests/fake_groq.py` with scripted replies, exactly as `clips.Stage.assistant` does, and
  `r.note('Assistant replies scripted for this video; the tools and figures are computed live')` on screen all the
  time the assistant is visible. Never a real key, never the network.

## Style rules

1. A lower third in every chapter, in its first seconds.
2. Zoom on what is being talked about, and back out when the talk moves on. Hold a zoom at least 1.5 s.
3. Callouts for the controls and the key figures as they are named (`r.until_word` lines them up with the words).
4. No figure without its source on screen: `r.figure(..., src=the widget that shows it)`, values read from the app.
5. No dead air over 1.5 s between lines (the cold open and cards aside): start the line, then do the action it
   describes (`r.say(i)` before `r.menu(...)`, not after). `review.py` flags the gaps.
6. Background work: `r.wait_for` / `stage.wait_analysis` show it in real time; long jobs as a marked time-lapse
   (`speed=3..10`). Never `r.hold` while a job runs if the job's duration matters: the recorder renders slower than real
   time (about 10 frames a second at 2x), so a job in the background of `hold()` looks up to 3x faster than it is.
7. End a chapter on a calm frame: zoom out, pointer off (`r.cursor_on = False`), `r.finish_speech(0.8..1.2)`. The
   assembler dips to the background colour over the last 0.5 s.
8. Restraint: one callout at a time, two at most; no overlay over the thing being shown.
9. A chapter is typically 1.5 to 3 minutes; split a feature over more lines rather than speeding up.

## Commands

Set `VANTAGE_SCRATCH` to a working folder outside the repository (the voice's environment, the heavy-job lock and a
throw-away config folder live there; default `demo/build/youtube/work`) and run from the repository root.

```bash
# one chapter, quick draft (960 x 540, 15 fps, 1x pixels; about 3x faster than real time): voice + record
systemd-run --user --scope -q -p MemoryMax=2G -p MemorySwapMax=0 python3 demo/youtube/produce.py --draft --only tolerance --record-only
# review it: frames every 3 s, a contact sheet with the line being spoken, the timing report (gaps over 1.5 s)
systemd-run --user --scope -q -p MemoryMax=2G -p MemorySwapMax=0 python3 demo/youtube/review.py tolerance --draft --every 3
#   -> demo/build/youtube/draft/seg/tolerance/review/contact.jpg (and frame_*.jpg)
# the final recording of it (1920 x 1080, 30 fps, grabbed at 2x: about 3x slower than real time)
systemd-run --user --scope -q -p MemoryMax=2G -p MemorySwapMax=0 python3 demo/youtube/produce.py --only tolerance --record-only
systemd-run --user --scope -q -p MemoryMax=2G -p MemorySwapMax=0 python3 demo/youtube/review.py tolerance
# its card, and a short video of just some chapters with every deliverable (names it vantage-tour-draft)
python3 demo/youtube/cards.py preview tolerance        # under the 2G cap as above
systemd-run --user --scope -q -p MemoryMax=2G -p MemorySwapMax=0 python3 demo/youtube/produce.py --only intro,tolerance,outro
# everything (only what is missing or changed is redone)
systemd-run --user --scope -q -p MemoryMax=2G -p MemorySwapMax=0 python3 demo/youtube/produce.py
```

`produce.py` runs every child under its own cap (`record.py` 8G, the assembly 6G, Kokoro 3G), offscreen, with
`XDG_CONFIG_HOME=$S/cfg-tour`, and takes `$S/heavy.lock` only around one recording or the assembly at a time.
To record by hand (what produce does):

```bash
flock $S/heavy.lock systemd-run --user --scope -q -p MemoryMax=8G -p MemorySwapMax=0 \
  env XDG_CONFIG_HOME=$S/cfg-tour QT_QPA_PLATFORM=offscreen PYTHONUNBUFFERED=1 timeout -s KILL 3600 \
  python3 demo/youtube/record.py tolerance [--draft]
```

Kokoro: the virtualenv and models are `$S/tts/env` and `$S/tts/models`; espeak-ng's data path must be short, so the
link `$S/e` points at `espeakng_loader/espeak-ng-data` (with a link `espeak-ng-data -> .` inside it). Elsewhere, set
`VANTAGE_TTS` (a folder with `env/` and `models/`), `VANTAGE_ESPEAK`, `VANTAGE_LOCK` and `VANTAGE_CFG`.

Time: a final recording runs about 3.2x slower than real time (a 2-minute chapter takes 6 to 7 minutes); a draft
runs faster than real time. Assembling costs about 10 s per minute of video plus the x264 encode (preset slow).

The recorder scans the text of every visible widget once a second of video for a machine path (the throw-away home,
`/home/`, `/tmp/`); a hit is printed as `WARNING path on screen` in the recording's log and listed by `review.py`
(`PATH ON SCREEN`). It cannot see text a delegate paints itself (the Outliner, the start window's list), so still look.

Look at the contact sheet and a few full frames of every recording (`frame_*.jpg` are 640 px; extract full frames with
`ffmpeg -ss T -i demo/build/youtube/seg/<id>/video.mp4 -frames:v 1 f.png`) for: a machine path or a name on screen,
the reference photos, a dialog left open, a callout on the wrong thing, a zoom that cuts what is talked about, a
figure that is not on screen.

## Episodes

The same pipeline makes more than one video. An **episode** is a plan over the chapter files (which all live in
`chapters/`, ids unique across episodes): its parts, its chapters in order, its intro and outro, and what its
deliverables are called and say. `chapters.EPISODES` holds them:

| Episode | Video | Files | Parts |
|---|---|---|---|
| `tour` (the default) | `vantage-tour.mp4`: the full app tour, v0.1.1 | `00_intro` ... `38_install-anywhere`, `99_outro` | 5 |
| `v020` | `vantage-whats-new-0.2.0.mp4`: what is new in 0.2.0 | `40_intro-v020`, `41_new-critical` ... `48_new-automation`, `49_outro-v020` | 3 |

* `produce.py --episode v020` chooses it, and every child (voice, record, cards, assemble) gets it as
  `VANTAGE_EPISODE`. `--only` with ids of one episode chooses that episode without `--episode`; ids of another episode
  are refused. With neither, it is the tour, exactly as before.
* `record.py <id>` and `cards.py preview <id>` find the chapter's own episode by its id (`chapters.episode_of`): no
  variable needed.
* What changes with the episode: the part cards ("PART 1 OF 3"), the chapter cards (the episode's part titles), the
  indicator (the hairline map of the episode's chapters and "PART 3 · 07 ..." at the top), the numbers on screen (an
  episode's plan numbers its chapters from 1: the v020 files are 41-48 but show 01-08), the YouTube chapters, the
  description, the thumbnail (the still `thumb` its intro saves), the video's name and title. The v020 texts are
  written next to the video with its name in front (`vantage-whats-new-0.2.0-chapters.txt`, `...-description.md`,
  `...-thumbnail.png`) so they never overwrite the tour's `chapters.txt`, `description.md`, `thumbnail.png`.
* A chapter's `META['part']` is its part in its own episode (v020: 1 On the road, 2 In the workshop, 3 In the
  pipeline; the intro 0, the outro 4). `gfx.logo_reveal(..., version=)` and `gfx.end_card(..., version=, rows=)` take
  the episode's version and links (the tour's cards keep v0.1.1).
* The recordings of both live in `demo/build/youtube/seg/<id>/` (ids do not collide); the voice cache is shared.
* v020's intro quotes its chapters: its cold open plays excerpts of their final recordings and its overview drifts over
  stills of them (`COLD`, `BACKDROPS` in `40_intro-v020.py`), so record it after them (a draft falls back to drafts).
  Its segment key does not include them: when a quoted chapter is re-recorded, check the seconds quoted still show the
  same moment and re-record the intro (`produce.py --episode v020 --only intro-v020 --record-only --rerecord`).
* Chapter 47's before/after cards read `demo/build/youtube/data/calib_measure.json`, written by
  `demo/youtube/calib_measure.py` (0.1.1's joint and lidar–camera solves, read from the tag v0.1.1 with `git show`,
  against 0.2.0's on the same simulated captures; about 8 minutes, under the 2G cap): run it before recording 47.
* The sound alone: `assemble.py --episode v020 --remux` (under the heavy lock and the 6G cap, as produce runs it) keeps
  the video of the deliverable already assembled from the same pieces and redoes the mix, loudness, subtitles, texts
  and sheets (about 4 minutes instead of 18). When the AAC encode overshoots -1 dBTP, the assembler now limits the mix
  to a lower true-peak ceiling (-2.4, then -3.2 dBTP) before it gives up loudness, so the result stays at -14 LUFS (the
  tour's own run passed on its first try and is unchanged).

```bash
# a v020 chapter: draft, review, final (as above, with the episode)
systemd-run --user --scope -q -p MemoryMax=2G -p MemorySwapMax=0 python3 demo/youtube/produce.py --episode v020 --draft --only new-critical --record-only
systemd-run --user --scope -q -p MemoryMax=2G -p MemorySwapMax=0 python3 demo/youtube/review.py new-critical --draft
systemd-run --user --scope -q -p MemoryMax=2G -p MemorySwapMax=0 python3 demo/youtube/produce.py --episode v020 --only new-critical --record-only
systemd-run --user --scope -q -p MemoryMax=2G -p MemorySwapMax=0 python3 demo/youtube/review.py new-critical
python3 demo/youtube/cards.py preview new-critical          # its card and part card ("PART 1 OF 3"), under the cap
# the whole episode (only what is missing or changed is re-recorded)
systemd-run --user --scope -q -p MemoryMax=2G -p MemorySwapMax=0 python3 demo/youtube/produce.py --episode v020
```

## Pitfalls already hit

* **Modal dialogs block the recorder.** Anything the app opens with `exec()` (the command palette, Preferences, the
  shortcuts sheet, message boxes, file dialogs) stops the scene. Construct the dialog yourself and `show()` it (see the
  palette and the shortcuts in `01_projects.py`); hide it when done. File dialogs cannot be filmed: click the button
  with `real=False` and call the page's `open_path(...)` (as `clip_asbuilt` does).
* **Keyboard shortcuts do not fire offscreen** (no focus): show `r.key_combo('Ctrl+K')` and trigger the action
  directly (`w.actions_by_id[...]`, or the dialog's own method). Key presses into a widget (`QTest.keyClick(field, ...)`)
  do work.
* **QTest.mouseMove is not delivered to widgets without mouse tracking** (the Coverage panel's bars): use
  `r.hover` / `r.glide` / `r.send_move`, which send the event to the widget directly.
* **`r.menu` matches items by the start of their text**: "4-cam" also matches "4-cam, mirror cameras 5° off". For
  combo boxes pass an index (`combo.findData(uid)`).
* **`stage.settle()` freezes time in the video** (no frames while it waits): fine for set-up and instant UI changes,
  wrong for anything the viewer should see being computed (use `stage.wait_analysis(r)` / `r.wait_for`).
* **`ScrubSpinBox` fields commit on Enter** (`committed`), not on every keystroke: use `r.enter(field, text)`; if a
  commit did not happen, emit `field.committed.emit(value)`.
* **Popups**: combo lists, menus, tooltips and dialogs are composited from their own top-level windows; anything
  still visible at the end of a step shows in the next frames. Hide what you open.
* **Analysis results**: `stage.wait_analysis(r)` waits for a result *newer* than the last one it saw; after opening a
  window yourself and editing straight away, call `stage.mark_result()` before the edit.
* **Draft vs final**: drafts grab at 1x, finals at 2x (`QT_SCALE_FACTOR=2`); text metrics differ by a pixel or two, so
  check the final frames of anything tight (a zoom framed on text).
* **The intro's cold open is timed to the music** (90 BPM, cuts on beats): if you change its shots keep the cuts on
  multiples of 2/3 s and the logo at 10.67 s.
* **Paths**: anything the app shows from the home folder must go through `stage.tilde(...)` /
  `stage.tilde_path_label(page)`; the throw-away home is `/tmp/h...` and must never appear. Check stills.
* **Never edit `vantage/`, `tests/`, `site/` or the website clip code** for a chapter; if the app misbehaves on
  camera, work around it in the scene and report it to the lead with the steps and the traceback.
