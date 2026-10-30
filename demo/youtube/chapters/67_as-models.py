"""67 Automatic model choice and pictures (episode as070, chapter 7): the sample's data rig. Preferences ▸ Assistant
(shown without blocking: the dialog is built and shown, its key field's placeholder replaced so no part of the key,
not even the masked tail, is on screen) -> "Automatic: choose the model for each request" ticked -> Apply. A picture
question to the REAL model: this machine's throw-away settings hold no Claude key, so the route falls back to GPT-OSS
120B on Groq and the answer says so (its label, the fallback note, "Pictures need Claude"); the model's reply is its
own, unedited. Then the coverage picture the look tool draws (vision.render_map on the live analysis), drawn here on
this machine and shown full frame: nothing of it is sent anywhere (Groq's models do not take images, and look is not
offered to them).

Checked with r.expect while recording: the route of the picture question is Groq · GPT-OSS 120B with a fallback note
naming Claude Opus 5.5; the model was not offered look (no look call); the picture's caption figures are the
analysis's (blind 0.24 m²). Nothing goes to Anthropic: no Claude key exists in these settings.
"""

META = {
    'id': 'as-models',
    'part': 3,
    'title': 'Automatic model choice and pictures',
    'promise': 'The model picked per request, named on every answer.',
    'menu': 'Preferences ▸ Assistant ▸ Automatic',
}

LINES = [
    # 0
    ("Preferences ▸ Assistant can choose the model for each request: quick questions go to a fast model; goals, "
     "plans and picture questions to Claude Opus.",
     "Preferences, assistant, can choose the model for each request: quick questions go to a fast model; goals, "
     "plans, and picture questions, to Claude Opus."),
    # 1
    ("Each answer names its model and why. With no Claude key on this machine, a picture question falls back to "
     "GPT-OSS 120B, and the panel says so.",
     "Each answer names its model, and why. With no Claude key on this machine, a picture question falls back to "
     "G P T O S S one twenty B, and the panel says so."),
    # 2
    ("This is what the look tool draws from the analysis, on this machine: blind ground in red, the rest shaded by "
     "pixel density.",
     "This is what the look tool draws from the analysis, on this machine: blind ground in red, the rest shaded by "
     "pixel density."),
    # 3
    ("With a Claude key the assistant can look at it. Groq's models do not take images: they answer from the figures, "
     "and nothing here was sent.",
     "With a Claude key, the assistant can look at it. Grok's models do not take images: they answer from the "
     "figures, and nothing here was sent."),
]

Q = 'Look at the coverage map: is any ground blind ahead of the car? One sentence.'


def scene(stage, r):
    import helpers_as as H
    H.groq_on()
    try:
        _scene(stage, r, H)
    finally:
        H.groq_off()


def _scene(stage, r, H):
    from PyQt6.QtCore import QRectF, Qt
    from PyQt6.QtGui import QImage, QPainter
    from PyQt6.QtWidgets import QPushButton
    import gfx
    from vantage.assistant import vision as V
    from vantage.ui.dialogs.misc import PreferencesDialog

    w = H.window(stage, auto=False)
    r.win = w
    chat = H.Chat(stage, r, w, 'as-models')
    panel = w.assistant_panel
    ctl = w.ctx.assistant
    vp = stage.vp
    vp.cam.az, vp.cam.el = -30.0, 30.0
    w.viewarea.frame_all()
    stage.settle(2)
    r.cursor_on = False
    try:
        r.lower_third()
        r.note(H.NOTE)
        # ---- 0: Preferences ▸ Assistant ▸ Automatic
        d = PreferencesDialog(w.ctx, w, tab='assistant')
        a = d.assistant
        a.key_edit.setPlaceholderText('Set by GROQ_API_KEY (not shown in this video)')
        d.resize(900, 900)          # its 660 x 520 does not grow with the 130 % UI scale: the notes would be clipped
        stage.extra.append(d)
        r.say(0)
        r.cursor_on = True
        r.peek_menu('File', 'Preferences', hold=0.4)      # File ▸ Preferences… (Ctrl+,): its dialog is modal,
        d.show()                                           # so the same dialog is built and shown here instead
        stage.settle(1)
        r.crossfade(0.3)
        r.zoom(lambda: r.wrect(a.auto).adjusted(-140, -60, 520, 150) if a.auto.isVisible() else None, 0.9)
        r.until_word(0, 'quick questions')
        from PyQt6.QtCore import QPointF
        r.click(a.auto, QPointF(9, a.auto.height() / 2), 0.6)     # on the box itself
        print('Automatic ticked:', a.auto.isChecked(), flush=True)
        if not a.auto.isChecked():                 # the offscreen click landed beside the box: tick it as a click does
            a.auto.click()
            r.tick(1)
        r.callout(a.auto_note, '', dur=max(1.8, r.speech_left() - 1.2), pad=4, dim=0.2)
        r.until(r.speech_end - 0.9)
        ok = next(b for b in d.findChildren(QPushButton) if b.text() == 'Apply')
        r.zoom(None, 0.5, wait=False)
        r.click(ok, None, 0.5, real=False)
        d.apply()
        d.hide()
        stage.settle(1)
        r.crossfade(0.25)
        r.expect(ctl.auto, True, 'Automatic on after Apply')
        r.finish_speech(0.1)

        # ---- 1: a picture question: falls back, and says so
        r.say(1)
        r.zoom(H.chat_rect(r, w, 620), 0.8)
        chat.type_in(Q, cps=32)
        chat.send()
        r.cursor_on = False
        chat.wait(timeout=150)
        turn = chat.turn()
        route = turn.route or {}
        print('route', route, '| label', turn.model_lbl.text(), '| calls', [c[0] for c in chat.calls()], flush=True)
        r.expect((route.get('provider'), route.get('model')), ('groq', H.MODEL), 'the fallback model')
        r.expect('Claude Opus 5.5 would answer this' in (route.get('fallback') or ''), True, 'the fallback note')
        r.expect(chat.last('look'), None, 'look is not offered on Groq')
        H.show_widget(r, panel, turn, margin=6)
        r.callout(turn.model_lbl, '', dur=1.8, pad=3, dim=0.22)
        r.hold(1.6)
        H.read_through(r, panel, turn, max(2.5, r.speech_left() + 0.6), lead=6)
        r.finish_speech(0.1)

        # ---- 2/3: the picture look draws, made here and shown full frame
        p = w.store.project
        pic = V.render_map(p, w.ctx.analysis.result)
        blind = pic.info.get('blind_m2') if isinstance(pic.info, dict) else None
        print('picture', pic.width, pic.height, len(pic.data), 'info', str(pic.info)[:300], flush=True)
        print('caption', pic.caption[:300], flush=True)
        if blind is not None:
            r.expect(round(float(blind), 2), 0.24, 'the picture\'s blind ground')
        img = QImage.fromData(pic.data)
        f0 = r.f

        def card(rec):
            out = QImage(gfx.W, gfx.H, QImage.Format.Format_RGB32)
            out.fill(gfx.col(gfx.BASE))
            q = gfx.painter(out)
            gfx.vignette(q, 0.45)
            gfx.glow(q, 760, 540, 700, gfx.ACCENT, 0.06)
            q.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform)
            t = (rec.f - f0) / rec.fps
            s = 1.0 + 0.03 * gfx.smoother(t / 12.0)
            hh = 860 * s
            ww = hh * img.width() / max(1, img.height())
            box = QRectF(660 - ww / 2, 560 - hh / 2, ww, hh)
            gfx.panel(q, box.adjusted(-14, -14, 14, 14), radius=14, fill=gfx.col('#0f0f0e', 0.95))
            q.drawImage(box, img)
            x = 1180
            gfx.draw_text(q, x, 300, 'THE LOOK TOOL', gfx.font(18, 'semibold', mono=True, spacing=3), gfx.ACCENT)
            gfx.draw_text(q, x, 370, 'The coverage map,', gfx.font(46, 'semibold'), gfx.INK)
            gfx.draw_text(q, x, 428, 'drawn on this machine', gfx.font(46, 'semibold'), gfx.INK)
            yy = 500
            for line in (f'{pic.width} × {pic.height} px PNG, {len(pic.data) / 1024:.0f} kB',
                         f'≈ {pic.tokens:,} image tokens for a model that takes images',
                         'With Claude: sent inside your question, like a tool result',
                         'On Groq: not offered, nothing sent'):
                gfx.draw_text(q, x, yy, line, gfx.font(26), gfx.INK2)
                yy += 46
            q.end()
            return out
        r.say(2)
        r.note(None)                           # the picture is not a reply
        r.crossfade(0.6)
        r.card = card
        r.zoom(None, 0)
        r.finish_speech(0.1)
        r.say(3)
        H.kinetic(r, 'Groq answers\nfrom the figures.', accent=('figures.',), dur=3.2, at='bottom', px=46)
        r.finish_speech(0.8)
        r.crossfade(0.5)
        r.card = None
        r.hold(0.6)
        chat.dump('ok')
        r.hold(0.3)
    except Exception:
        chat.dump('failed')
        raise
