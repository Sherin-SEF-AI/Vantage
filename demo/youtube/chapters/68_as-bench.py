"""68 The benchmark (episode as070, chapter 8): a terminal card replaying the REAL run of the assistant benchmark that
was made for this video before the recording, on 2026-10-29/30 with the user's Groq key:

    python3 -m vantage.assistant.evals --provider groq --model openai/gpt-oss-120b --yes --out ~/vantage-evals/groq-120b

(51 questions, the full set, no subset: Groq did not rate-limit it). Its printed output is shown as it printed it
(each line in order; the per-question lines in full, the long error lines cut at the terminal's edge with "…"), the
4 min 18 s it took replayed in seconds and marked so; machine paths shown from ~. Then a scorecard drawn from its
results.json: the overall score, each category's score and question count, the requests and tokens, the errors.

Figures narrated, read from the run's own results.json while recording and checked with r.expect: 51 questions, 50
scored, 89 % overall, 3 errors (Groq refused tool calls whose optional arguments the model filled with null: the
lines say so). The run's files are read from $VANTAGE_AS_BENCH (default $VANTAGE_SCRATCH/as070-bench: run.log,
results.json, report.md).
"""

META = {
    'id': 'as-bench',
    'part': 3,
    'title': 'The benchmark',
    'promise': 'How good are the answers? Measured, without a judge model.',
    'menu': 'python -m vantage.assistant.evals',
}

LINES = [
    # 0
    ("How good are the answers? The benchmark asks the assistant 51 design questions about the built-in samples, "
     "through the panel's own conversation loop.",
     "How good are the answers? The benchmark asks the assistant fifty-one design questions about the built-in "
     "samples, through the panel's own conversation loop."),
    # 1
    ("It grades without a language model: the right tools, each figure with its unit against the app's own numbers, "
     "plans left as drafts, the project unchanged.",
     "It grades without a language model: the right tools, each figure with its unit against the app's own numbers, "
     "plans left as drafts, the project unchanged."),
    # 2
    ("This real run on GPT-OSS 120B, made before the recording, scored 89 per cent over 50 questions. Three failed: "
     "Groq refused tool calls the model wrote with empty arguments.",
     "This real run on G P T O S S one twenty B, made before the recording, scored eighty-nine per cent over fifty "
     "questions. Three failed: Grok refused tool calls the model wrote with empty arguments."),
]

CMD1 = 'python3 -m vantage.assistant.evals --provider groq \\'
CMD2 = '    --model openai/gpt-oss-120b --yes --out ~/vantage-evals/groq-120b'
WIDTH = 86           # characters that fit the terminal card's width at its size


def scene(stage, r):
    import json
    import os
    import gfx
    import gfx_as
    from PyQt6.QtGui import QImage
    from vantage.assistant import evals as E

    root = os.environ.get('VANTAGE_AS_BENCH') or os.path.join(os.environ.get('VANTAGE_SCRATCH', ''), 'as070-bench')
    log = open(os.path.join(root, 'run.log'), encoding='utf-8').read().splitlines()
    res = json.load(open(os.path.join(root, 'results.json'), encoding='utf-8'))
    sm = res['summary']
    n_q = len(sm['questions'])
    print('summary', {k: v for k, v in sm.items() if k not in ('questions', 'categories')}, res['totals'], flush=True)
    r.expect((n_q, sm['scored'], round(sm['overall'] * 100), sm['errors']), (51, 50, 89, 3), 'the run\'s summary')
    errs = [q for q in sm['questions'] if q['status'] == 'error']
    r.expect(all('expected' in q['note'] and 'got null' in q['note'] for q in errs), True,
             'the three errors are tool calls Groq refused for null arguments')

    def home(s):                                   # the run's throw-away home folder, shown from ~
        j = s.find('/vantage-evals/')
        if j < 0:
            return s
        i = j
        while i > 0 and s[i - 1] != ' ':
            i -= 1
        return s[:i] + '~' + s[j:]

    def cut(s):
        return s if len(s) <= WIDTH else s[:WIDTH - 1] + '…'
    i0 = next(i for i, ln in enumerate(log) if ln.startswith('Key: '))
    out = []
    for ln in log[i0:]:
        if not ln.strip() or ln.startswith(('real', 'user', 'sys')):
            continue
        ln = home(ln)
        for leak in ('/tmp/', '/home/', 'claude-1000'):
            r.expect(leak in ln, False, 'no machine path on the terminal')
        tone = 'dim'
        if ln.startswith('  ') and ' % ' in ln:
            tone = 'ok' if ': 100 %' in ln else 'bad' if ': 0 %' in ln else 'warn'
        elif ln.startswith('Overall'):
            tone = 'ok'
        out.append(('out', cut(ln.replace(' · report: ', '  ·  report: ')), tone, 0.11))
    script = [('cmd', CMD1), ('out', CMD2, '', 0.5)] + out
    cats = [(E.CATEGORIES[k].split(' (')[0].replace(', detection and campaign read-outs', ' read-outs'), v['questions'],
             v['score']) for k, v in sm['categories'].items()]
    tot = res['totals']

    r.cursor_on = False
    f0 = r.f
    span = 0.3 + sum(len(it[1]) / 44.0 + 0.4 if it[0] == 'cmd' else it[3] for it in script)

    def term(rec):
        img = QImage(gfx.W, gfx.H, QImage.Format.Format_RGB32)
        img.fill(gfx.col(gfx.BASE))
        q = gfx.painter(img)
        gfx.vignette(q, 0.5)
        q.end()
        gfx_as.terminal(img, (rec.f - f0) / rec.fps, script, title='~  ·  bash')
        return img
    r.card = term
    r.lower_third()
    r.say(0)
    r.note('A real run made before this recording: 4 min 18 s, its output replayed in seconds')
    r.until(f0 / r.fps + span + 0.8)
    r.finish_speech(0.1)
    f1 = r.f

    def card(rec):
        img = QImage(gfx.W, gfx.H, QImage.Format.Format_RGB32)
        img.fill(gfx.col(gfx.BASE))
        gfx_as.scorecard(img, (rec.f - f1) / rec.fps, 'GPT-OSS 120B ON GROQ  ·  51 QUESTIONS', cats,
                         sm['overall'] * 100, f'overall, over {sm["scored"]} scored questions',
                         foot=(f'{tot["requests"]} requests · {tot["input_tokens"] / 1e6:.2f} M tokens in · '
                               f'{tot["output_tokens"] / 1000:.1f} k out',
                               f'{sm["errors"]} errors: tool calls Groq refused',
                               'for null arguments · 1 n/a (needs pictures)'))
        return img
    r.say(1)
    r.crossfade(0.5)
    r.card = card
    r.finish_speech(0.1)
    r.say(2)
    r.finish_speech(0.8)
    r.note(None)
    r.hold(0.6)
