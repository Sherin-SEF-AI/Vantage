"""The design agent: documents read into sentences (PDF, DOCX, HTML, Markdown), requirements extracted with their
sources (the brief parser, and the LLM through fake Groq and Anthropic servers, invented figures caught), the search's
options non-dominated and re-checked independently, marginal contributions that add up, determinism, cancel, the
live project untouched until an option is accepted, accepting as one undo step, and the page driven like a user."""
from __future__ import annotations

import copy
import io
import json
import math
import os
import sys
import time
import types
import zipfile
import zlib

import pytest

sys.path.insert(0, os.path.dirname(__file__))

from vantage.core import docs_ingest as DI                        # noqa: E402
from vantage.core import requirements as RQ                       # noqa: E402
from vantage.core import rig_designer as RD                       # noqa: E402
from vantage.core import samples                                  # noqa: E402
from vantage.assistant import designer_llm as DL                  # noqa: E402

BRIEF = ('Robotaxi, pedestrians at 80 m forward, 360° up to 30 m, budget under $6k, under 300 W, survives a 2 mm '
         'mounting tolerance')

# A TEST FIXTURE, written for these tests: not a real programme's document.
FIXTURE_ODD = """TEST FIXTURE - NOT A REAL PROGRAMME DOCUMENT. Operational design domain and sensor requirements of the Vantage test robotaxi, fixture version 0.1.

1. Scope. This fixture describes the operational design domain (ODD) of an urban robotaxi data-collection vehicle, for test purposes only.

2. Operational design domain. The vehicle operates in urban and suburban areas at speeds up to 50 km/h. Operation is limited to daylight and dusk, in dry conditions and in light rain up to 4 mm/h. The ambient operating temperature range is -10 to +45 degrees Celsius. Road types are two-lane urban streets, roundabouts and signalised junctions.

3. Perception. The rig shall resolve pedestrians at 80 m in the forward direction. Pedestrians shall be resolved to at least 30 m in every sector around the vehicle. Cyclists should be resolved to 40 m in the front-left and front-right sectors. Cars shall be resolved to 60 m ahead.

4. Ground coverage. There shall be no blind ground within the region of interest around the vehicle.

5. Power and installation. The rig shall draw under 300 W in steady operation. The rig must survive a 2 mm mounting tolerance at every camera. At most 8 cameras may be fitted.

6. Data. The recorder should hold at least 8 hours of driving per disk set. Motion blur should not exceed 2 px at the planned speed.

7. Budget. The sensor budget is under $6k.

8. Compliance. The installation shall comply with UN Regulation No. 10 on electromagnetic compatibility. Every sensor bracket shall be inspected after 1000 km of driving.
"""


# =========================================================================== fixture documents
def make_pdf(pages, compress=True, a85=False, image_page=False, tounicode=False):
    """A small PDF: each page a list of lines (an empty line is a paragraph gap), Helvetica, Td line steps; optional
    ASCII85 + Flate filter chain, a last page that is only an image, or a 2-byte font with a ToUnicode CMap."""
    import base64
    objs = []

    def add(b):
        objs.append(b)
        return len(objs)
    if tounicode:
        cmap = (b'/CIDInit /ProcSet findresource begin 12 dict begin begincmap 1 begincodespacerange <0000> <FFFF> '
                b'endcodespacerange 1 beginbfrange <0020> <007E> <0020> endbfrange endcmap end end')
        cm = add(b'<< /Length %d >>\nstream\n' % len(cmap) + cmap + b'\nendstream')
        font = add(b'<< /Type /Font /Subtype /Type0 /BaseFont /Fixture /Encoding /Identity-H /ToUnicode %d 0 R >>'
                   % cm)
    else:
        font = add(b'<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>')
    contents = []
    for lines in pages:
        ops = [b'BT /F1 11 Tf 72 760 Td 14 TL']
        first = True
        for ln in lines:
            if ln == '':
                ops.append(b'0 -28 Td')
                continue
            if tounicode:
                s = b'<' + ''.join(f'{ord(ch):04X}' for ch in ln).encode() + b'>'
            else:
                s = b'(' + ln.replace('\\', '\\\\').replace('(', '\\(').replace(')', '\\)').encode('cp1252') + b')'
            ops.append((b'' if first else b'0 -14 Td ') + s + b' Tj')
            first = False
        ops.append(b'ET')
        data = b'\n'.join(ops)
        if a85:
            z = base64.a85encode(zlib.compress(data), adobe=True)
            contents.append(add(b'<< /Length %d /Filter [/ASCII85Decode /FlateDecode] >>\nstream\n' % len(z) + z +
                                b'\nendstream'))
        elif compress:
            z = zlib.compress(data)
            contents.append(add(b'<< /Length %d /Filter /FlateDecode >>\nstream\n' % len(z) + z + b'\nendstream'))
        else:
            contents.append(add(b'<< /Length %d >>\nstream\n' % len(data) + data + b'\nendstream'))
    img = None
    if image_page:
        px = zlib.compress(bytes(300))
        img = add(b'<< /Type /XObject /Subtype /Image /Width 10 /Height 10 /ColorSpace /DeviceRGB /BitsPerComponent 8 '
                  b'/Length %d /Filter /FlateDecode >>\nstream\n' % len(px) + px + b'\nendstream')
        d = b'q 100 0 0 100 0 0 cm /Im1 Do Q'
        contents.append(add(b'<< /Length %d >>\nstream\n' % len(d) + d + b'\nendstream'))
    pages_obj = len(objs) + len(contents) + 1
    kids = []
    for k, c in enumerate(contents):
        xo = b' /XObject << /Im1 %d 0 R >>' % img if (image_page and k == len(contents) - 1) else b''
        kids.append(add(b'<< /Type /Page /Parent %d 0 R /MediaBox [0 0 612 792] /Resources << /Font << /F1 %d 0 R >>'
                        b'%s >> /Contents %d 0 R >>' % (pages_obj, font, xo, c)))
    assert add(b'<< /Type /Pages /Kids [' + b' '.join(b'%d 0 R' % k for k in kids) + b'] /Count %d >>'
               % len(kids)) == pages_obj
    cat = add(b'<< /Type /Catalog /Pages %d 0 R >>' % pages_obj)
    out = io.BytesIO()
    out.write(b'%PDF-1.4\n%\xe2\xe3\xcf\xd3\n')
    offs = []
    for i, b in enumerate(objs, 1):
        offs.append(out.tell())
        out.write(b'%d 0 obj\n' % i + b + b'\nendobj\n')
    x = out.tell()
    out.write(b'xref\n0 %d\n0000000000 65535 f \n' % (len(objs) + 1))
    for o in offs:
        out.write(b'%010d 00000 n \n' % o)
    out.write(b'trailer\n<< /Size %d /Root %d 0 R >>\nstartxref\n%d\n%%%%EOF\n' % (len(objs) + 1, cat, x))
    return out.getvalue()


def make_docx(paragraphs, page_break_after=None):
    W = 'http://schemas.openxmlformats.org/wordprocessingml/2006/main'
    body = []
    for i, p in enumerate(paragraphs):
        esc = p.replace('&', '&amp;').replace('<', '&lt;').replace('>', '&gt;')
        br = '<w:r><w:br w:type="page"/></w:r>' if page_break_after is not None and i == page_break_after else ''
        body.append(f'<w:p><w:r><w:t xml:space="preserve">{esc}</w:t></w:r>{br}</w:p>')
    xml = (f'<?xml version="1.0" encoding="UTF-8" standalone="yes"?><w:document xmlns:w="{W}"><w:body>'
           f'{"".join(body)}<w:tbl><w:tr><w:tc><w:p><w:r><w:t>Cell text in a table.</w:t></w:r></w:p></w:tc></w:tr>'
           f'</w:tbl></w:body></w:document>')
    b = io.BytesIO()
    with zipfile.ZipFile(b, 'w') as z:
        z.writestr('[Content_Types].xml', '<?xml version="1.0"?><Types xmlns="http://schemas.openxmlformats.org/'
                                          'package/2006/content-types"/>')
        z.writestr('word/document.xml', xml)
    return b.getvalue()


def odd_pages():
    """The fixture laid out on two PDF pages (lines of about 90 characters, a gap between paragraphs)."""
    import textwrap
    paras = [p.strip() for p in FIXTURE_ODD.strip().split('\n\n')]
    lines = []
    for p in paras:
        lines += textwrap.wrap(p, 90) + ['']
    half = len(lines) // 2
    while lines[half - 1] != '':
        half += 1
    return [lines[:half], lines[half:]]


# =========================================================================== documents
def test_docs_ingest_pdf_builtin_reader():
    pdf = make_pdf(odd_pages(), image_page=True)
    d = DI.read('odd-fixture.pdf', 'D1', pdf)
    assert d.method == 'pdf-builtin' or d.method == 'pypdf'
    assert d.pages == 3
    texts = [s.text for s in d.sentences]
    assert 'The rig shall resolve pedestrians at 80 m in the forward direction.' in texts
    assert 'The sensor budget is under $6k.' in texts
    s = next(s for s in d.sentences if 'under 300 W' in s.text)
    assert s.page in (1, 2) and s.id.startswith(f'D1-p{s.page}-') and s.location().startswith(f'p. {s.page}')
    # the image-only page is named as unread, with why
    assert any('Page 3' in p and 'OCR' in p for p in d.problems)
    # a word hyphenated at a line end is joined, abbreviations do not split sentences
    assert any('UN Regulation No. 10 on electromagnetic' in t for t in texts)
    # stable: the same bytes give the same ids
    assert [s.id for s in DI.read('odd-fixture.pdf', 'D1', pdf).sentences] == [s.id for s in d.sentences]


def test_docs_ingest_pdf_filters_tounicode_and_encrypted():
    pdf = make_pdf([['Cars shall be resolved to 60 m ahead.', '', 'Second paragraph here.']], a85=True)
    paras, problems, n = DI.pdf_paragraphs_builtin(pdf)
    assert n == 1 and not problems
    assert [t for _, t in paras] == ['Cars shall be resolved to 60 m ahead.', 'Second paragraph here.']
    pdf = make_pdf([['Pedestrians at 40 m.']], tounicode=True)
    paras, problems, n = DI.pdf_paragraphs_builtin(pdf)
    assert [t for _, t in paras] == ['Pedestrians at 40 m.']
    enc = pdf.replace(b'trailer\n<<', b'trailer\n<< /Encrypt 99 0 R')
    paras, problems, n = DI.pdf_paragraphs_builtin(enc)
    assert not paras and 'encrypted' in problems[0]
    assert 'Not a PDF' in DI.pdf_paragraphs_builtin(b'hello')[1][0]


def test_docs_ingest_docx_html_markdown_text():
    dx = make_docx(['Requirements fixture.', 'Cyclists shall be resolved to 40 m ahead. No critical issues.',
                    'Second page text.'], page_break_after=1)
    d = DI.read('req.docx', 'D2', dx)
    assert d.method == 'docx'
    got = {s.text: s for s in d.sentences}
    assert got['No critical issues.'].page == 1 and got['Second page text.'].page == 2
    assert got['No critical issues.'].id == 'D2-p1-2.2'
    assert 'Cell text in a table.' in got
    h = DI.read('x.html', 'D3', b'<html><head><title>T</title><style>p{}</style><script>var a=1;</script></head>'
                                b'<body><h1>ODD</h1><p>Rain up to 10 mm/h. Night &amp; day.</p><ul><li>Item one.</li>'
                                b'</ul></body></html>')
    assert [s.text for s in h.sentences] == ['ODD', 'Rain up to 10 mm/h.', 'Night & day.', 'Item one.']
    m = DI.read('x.md', 'D4', b'# Title\n\nSome **bold** text with [a link](http://x). Second one.\n\n- bullet one\n'
                              b'- bullet two\n\n```\ncode is skipped.\n```\n')
    assert [s.text for s in m.sentences] == ['Title', 'Some bold text with a link.', 'Second one.', 'bullet one',
                                             'bullet two']
    t = DI.read('x.txt', 'D5', b'First page.\fSecond page, e.g. this one. Approx. 5 m here.')
    assert [(s.page, s.text) for s in t.sentences] == [(1, 'First page.'), (2, 'Second page, e.g. this one.'),
                                                       (2, 'Approx. 5 m here.')]
    bad = DI.read('x.xlsx', 'D6', b'\x00\x01')
    assert not bad.sentences and 'is not one of them' in bad.problems[0]
    assert DI.split_sentences('Speeds up to 50 km/h. The rig shall draw 3.5 W at most. Fig. 2 shows it.') == \
        ['Speeds up to 50 km/h.', 'The rig shall draw 3.5 W at most.', 'Fig. 2 shows it.']


# =========================================================================== extraction
@pytest.fixture(scope='module')
def project():
    p = samples.tigor_sample_project()
    return p


def test_deterministic_extraction_of_the_robotaxi_brief(project):
    ex = DL.deterministic([DI.from_text(BRIEF)], project)
    got = {(r['item']['metric'], r['item']['params'].get('cls', ''), r['item']['params'].get('sector', ''),
            r['item']['op'], r['item']['target']) for r in ex.requirements}
    assert got == {('perception.range', 'Pedestrian', 'front', '>=', 80.0),
                   ('perception.range', 'Pedestrian', '', '>=', 30.0), ('power.steady_w', '', '', '<', 300.0)}
    cons = {(c['kind'], c['op'], c['value'], c['unit']) for c in ex.constraints}
    assert cons == {('budget', '<', 6000.0, 'USD'), ('tolerance_mm', '==', 2.0, 'mm')}
    for r in ex.requirements + ex.constraints:      # every row cites the brief's own words
        assert r['source']['id'] == 'B-1.1' and DL.quote_in(r['source']['quote'], BRIEF)
        assert RQ.validate_item(r['item'], project)[0] if 'item' in r else True
    assert not ex.rejected


def test_deterministic_extraction_of_the_odd_fixture(project, tmp_path):
    pdf = tmp_path / 'odd-fixture.pdf'
    pdf.write_bytes(make_pdf(odd_pages()))
    d = DI.read(str(pdf), 'D1')
    ex = DL.deterministic([d], project)
    by = {(r['item']['metric'], r['item']['params'].get('cls', ''), r['item']['params'].get('sector', '')):
          r for r in ex.requirements}
    assert by[('perception.range', 'Pedestrian', 'front')]['item']['target'] == 80.0
    assert by[('perception.range', 'Pedestrian', '')]['item']['target'] == 30.0
    assert by[('perception.range', 'Cyclist', 'front-left')]['item']['priority'] == 'should'
    assert by[('perception.range', 'Cyclist', 'front-right')]['item']['target'] == 40.0
    assert by[('perception.range', 'Car', 'front')]['item']['target'] == 60.0
    assert by[('coverage.blind_m2', '', '')]['item']['target'] == 0.0005
    assert by[('power.steady_w', '', '')]['item']['op'] == '<'
    assert by[('recording.hours_per_set', '', '')]['item']['priority'] == 'should'
    assert by[('timing.blur_px', '', '')]['item']['target'] == 2.0
    kinds = {c['kind']: c['value'] for c in ex.constraints}
    assert kinds == {'tolerance_mm': 2.0, 'max_cameras': 8.0, 'budget': 6000.0}
    # the statements no metric checks are listed with their quotes, never dropped
    nc = ' | '.join(n['source']['quote'] for n in ex.not_checkable)
    for words in ('50 km/h', 'UN Regulation No. 10', '1000 km', '4 mm/h'):
        assert words in nc
    for r in ex.requirements:                        # the page and paragraph of each source
        assert r['source']['doc'] == 'odd-fixture.pdf' and r['source']['location'].startswith('p. ')


def test_figure_checks():
    assert DL.target_ok('perception.range', 80, 'pedestrians at 80 m forward')[0]
    assert DL.target_ok('perception.range', 200, 'cars to 0.2 km ahead')[0]
    ok, why = DL.target_ok('perception.range', 120, 'pedestrians at 80 m forward')
    assert not ok and '120' in why and 'states 80' in why
    assert DL.target_ok('power.runtime_min', 90, '1.5 h with the engine off')[0]
    assert DL.target_ok('issues.count', 0, 'no critical issues')[0]
    assert not DL.target_ok('issues.count', 0, 'critical issues are fine')[0]
    assert DL.target_ok('coverage.blind_m2', 0.0005, 'no blind ground')[0]
    assert not DL.target_ok('coverage.blind_m2', 0.5, 'no blind ground')[0]
    assert DL.constraint_ok('budget', 6000, 'USD', 'budget under $6k')[0]
    assert DL.constraint_ok('budget', 12500, 'EUR', 'below 12,500 EUR')[0]
    assert not DL.constraint_ok('budget', 9000, 'USD', 'budget under $6k')[0]
    assert DL.constraint_ok('max_cameras', 8, '', 'at most eight cameras')[0]
    assert DL.constraint_ok('tolerance_mm', 2, 'mm', 'a 2 mm tolerance')[0]
    assert not DL.constraint_ok('tolerance_mm', 5, 'mm', 'a 2 mm tolerance')[0]


def _report(good=True):
    rows = [{'source_id': 'B-1.1', 'quote': 'pedestrians at 80 m forward', 'metric': 'perception.range',
             'params': [{'key': 'cls', 'value': 'Pedestrian'}, {'key': 'sector', 'value': 'front'}], 'op': '>=',
             'target': 80, 'priority': 'must', 'text': 'Pedestrians at 80 m forward'},
            # invented: 120 m is not in the quote
            {'source_id': 'B-1.1', 'quote': 'pedestrians at 80 m forward', 'metric': 'perception.range',
             'params': [{'key': 'cls', 'value': 'Car'}, {'key': 'sector', 'value': 'front'}], 'op': '>=',
             'target': 120, 'priority': 'must', 'text': 'Cars at 120 m'},
            # a quote that is not the sentence's words
            {'source_id': 'B-1.1', 'quote': 'cyclists at 50 m', 'metric': 'perception.range',
             'params': [{'key': 'cls', 'value': 'Cyclist'}, {'key': 'sector', 'value': ''}], 'op': '>=',
             'target': 50, 'priority': 'must', 'text': 'Cyclists'},
            # a metric Vantage does not have: not checkable, kept with its quote
            {'source_id': 'B-1.2', 'quote': 'operate from -10 to 45 degrees', 'metric': 'environment.temperature',
             'params': [], 'op': '>=', 'target': 45, 'priority': 'must', 'text': 'Temperature'},
            {'source_id': 'B-1.1', 'quote': 'under 300 W', 'metric': 'power.steady_w', 'params': [], 'op': '<',
             'target': 300, 'priority': 'must', 'text': 'Under 300 W'}]
    cons = [{'source_id': 'B-1.1', 'quote': 'budget under $6k', 'kind': 'budget', 'op': '<', 'value': 6000,
             'unit': 'USD', 'priority': 'must'},
            {'source_id': 'B-1.1', 'quote': 'budget under $6k', 'kind': 'budget', 'op': '<', 'value': 9000,
             'unit': 'USD', 'priority': 'must'},
            {'source_id': 'B-1.1', 'quote': 'survives a 2 mm mounting tolerance', 'kind': 'tolerance_mm', 'op': '==',
             'value': 2, 'unit': 'mm', 'priority': 'must'}]
    return {'requirements': rows if good else rows[1:2], 'constraints': cons, 'not_checkable': []}


BRIEF2 = BRIEF + '. The rig must operate from -10 to 45 degrees. A sentence the model never cites shall be listed.'


def _check_llm_extraction(ex, project):
    keys = {(r['item']['metric'], r['item']['params'].get('cls', ''), r['item']['target']) for r in ex.requirements}
    assert keys == {('perception.range', 'Pedestrian', 80.0), ('power.steady_w', '', 300.0)}
    why = {(r['row'].get('target', r['row'].get('value')) if isinstance(r['row'], dict) else None): r['why']
           for r in ex.rejected}
    assert 'not in the quoted source' in why[120] and 'states 80' in why[120]
    assert 'quote is not the words' in why[50]
    assert 'amount 9000 is not in the quoted source' in why[9000]
    assert {(c['kind'], c['value']) for c in ex.constraints} == {('budget', 6000.0), ('tolerance_mm', 2.0)}
    nc = {n['source']['id']: n for n in ex.not_checkable}
    assert 'environment.temperature' in nc['B-1.2']['why'] and nc['B-1.2']['source']['quote'].startswith('operate')
    assert nc['B-1.3']['origin'] == 'code'            # never cited by the model: listed by the code
    for r in ex.requirements:
        assert RQ.validate_item(r['item'], project)[0]


def test_llm_extraction_through_fake_groq(project):
    from fake_groq import FakeGroq
    srv = FakeGroq(key='test-key').start()
    try:
        srv.script.append(srv.tool_call(DL.TOOL, _report()))
        docs = [DI.from_text(BRIEF2)]
        ex = DL.extract(docs, project, 'groq', 'test-key', 'openai/gpt-oss-120b', url=srv.url)
        assert ex.method == 'groq' and ex.model == 'openai/gpt-oss-120b'
        _check_llm_extraction(ex, project)
        req = srv.requests[-1]
        assert req['tools'][0]['function']['name'] == DL.TOOL
        assert '[B-1.1] Robotaxi' in req['messages'][1]['content'] and 'perception.range' in req['messages'][1]['content']
        # a report written as text (no tool call) is read too
        srv.script.append(srv.answer('```json\n' + json.dumps(_report(False)) + '\n```'))
        ex = DL.extract(docs, project, 'groq', 'test-key', 'openai/gpt-oss-120b', url=srv.url)
        assert not ex.requirements and ex.rejected
        # no JSON at all: an error the page reports (and falls back on)
        srv.script.append(srv.answer('I cannot help with that.'))
        with pytest.raises(DL.ExtractionError):
            DL.extract(docs, project, 'groq', 'test-key', 'openai/gpt-oss-120b', url=srv.url)
        # a wrong key: the provider's error, not a crash
        from vantage.assistant.client import ApiError
        with pytest.raises(ApiError):
            DL.extract(docs, project, 'groq', 'wrong', 'openai/gpt-oss-120b', url=srv.url)
    finally:
        srv.stop()


def test_llm_extraction_through_fake_anthropic(project):
    pytest.importorskip('anthropic')
    from fake_anthropic import FakeAnthropic
    srv = FakeAnthropic(key='sk-ant-test').start()
    try:
        srv.script.append(srv.reply(tools=[(DL.TOOL, _report())], thinking='Reading the brief.'))
        ex = DL.extract([DI.from_text(BRIEF2)], project, 'claude', 'sk-ant-test', 'claude-opus-5-5', url=srv.url)
        assert ex.method == 'claude'
        _check_llm_extraction(ex, project)
        body = srv.requests[-1]
        assert body['tools'][0]['strict'] is True and body['tools'][0]['input_schema']['additionalProperties'] is False
        assert body['tool_choice'] == {'type': 'auto'}
        srv.script.append(srv.refusal())
        with pytest.raises(DL.ExtractionError):
            DL.extract([DI.from_text(BRIEF2)], project, 'claude', 'sk-ant-test', 'claude-opus-5-5', url=srv.url)
    finally:
        srv.stop()


def test_no_provider_means_deterministic(project):
    ex = DL.extract([DI.from_text(BRIEF)], project)
    assert ex.method == 'deterministic' and len(ex.requirements) == 3


# =========================================================================== the designer
def _design_project():
    """The sample project on its 6-camera surround variant, no sensors: small enough for quick searches."""
    p = samples.tigor_sample_project()
    v = next(v for v in p.variants if v.name == '6-cam · v3')
    p.active_uid = v.uid
    return p


def _settings(p, **kw):
    items = [{'id': 'R1', 'metric': 'perception.range', 'params': {'cls': 'Pedestrian', 'sector': 'front'},
              'op': '>=', 'target': 80.0, 'priority': 'must', 'text': 'Pedestrians at 80 m forward'},
             {'id': 'R2', 'metric': 'perception.range', 'params': {'cls': 'Pedestrian', 'sector': ''},
              'op': '>=', 'target': 20.0, 'priority': 'must', 'text': 'Pedestrians to 20 m all round'},
             {'id': 'R3', 'metric': 'power.steady_w', 'params': {}, 'op': '<', 'target': 300.0, 'priority': 'must',
              'text': 'Under 300 W'},
             {'id': 'R4', 'metric': 'coverage.min_density', 'params': {}, 'op': '>=', 'target': 1.0,
              'priority': 'should', 'text': 'Ground at 1 px/cm'}]
    d = {'requirements': [{'item': RQ.clean_item(i, p), 'source': {'id': 'B-1.1', 'quote': i['text']},
                           'accepted': True, 'origin': 'test'} for i in items],
         'constraints': [{'kind': 'tolerance_mm', 'op': '==', 'value': 2.0, 'unit': 'mm', 'priority': 'must',
                          'text': 'Survives a 2 mm mounting tolerance', 'accepted': True}],
         'options': 3, 'evaluations': 22, 'budget_s': 900, 'seed': 5, 'reference_seeds': False,
         'vary_lidar': False, 'vary_radar': False, 'prices': {}}
    d.update(kw)
    return RD.DesignerSettings.from_dict(d)


@pytest.fixture(scope='module')
def design():
    from vantage.core.analysis import Engine
    p = _design_project()
    before = json.dumps([v.to_dict() for v in p.variants] + [p.analysis.to_dict()], default=str)
    eng = Engine()
    st = _settings(p)
    res = RD.run(p, eng, st)
    return p, eng, st, res, before


def _objs(o):
    return (o['cost'], o['power_w'], -(o['margin'] if o['margin'] is not None else 10.0), -o['detection'])


def _dominates(a, b):
    if a['feasible'] != b['feasible']:
        return a['feasible']
    oa, ob = _objs(a), _objs(b)
    pairs = [(x, y) for x, y in zip(oa, ob) if x is not None and y is not None]
    return all(x <= y + 1e-9 for x, y in pairs) and any(x < y - 1e-9 for x, y in pairs)


def test_designer_options_nondominated_and_rechecked(design):
    p, eng, st, res, before = design
    assert 3 <= len(res.options) <= 7
    assert res.detection_kind == 'geometric' and 'geometric only' in res.detection_text
    feas = [o for o in res.options if o['feasible']]
    assert feas, 'the search should find a rig that meets the must requirements'
    for a in res.options:                             # non-dominated among the options on the front
        if a['on_front']:
            assert not any(_dominates(b, a) for b in res.options if b is not a and b['on_front'])
    # the current variant (6 wide cameras) cannot see pedestrians at 80 m ahead; the options can
    cur = {r['id']: r for r in res.current['items']}
    assert cur['R1']['status'] == 'fail'
    for o in feas:
        # independently: the option as a variant in a copy of the project, scored by requirements.run itself
        q = p.snapshot()
        v = RD.option_variant(q, o)
        q.variants.append(v)
        q.active_uid = v.uid
        r = RQ.run(q, eng, RQ.RequirementsSettings.from_dict({'items': st.items()}), issues_list=False)
        got = {x.item['id']: x for x in r.items}
        for row in o['items']:
            if row['priority'] == 'must':
                assert got[row['id']].status == 'pass', (o['label'], row)
            assert math.isclose(got[row['id']].value, row['value'], rel_tol=1e-6, abs_tol=1e-6)
        assert o['robust'] is not None and o['robust']['ok'] and o['robust']['samples'] == RD.ROBUST_SAMPLES
        assert o['cost'] is None and o['cost_missing']          # no prices entered: cost incomplete, said so
        from vantage.core import power as PW
        assert math.isclose(PW.run(q, variant=v).totals['steady_w'], o['power_w'], abs_tol=0.01)


def test_marginal_contributions_add_up(design):
    p, eng, st, res, before = design
    o = next(o for o in res.options if o['feasible'])
    q = p.snapshot()
    v = RD.option_variant(q, o)
    q.variants.append(v)
    q.active_uid = v.uid
    cov = eng.analyse(q)
    a = cov.grid.cell ** 2
    roi = cov.grid.roi
    multi = float((roi & (cov.count >= 2)).sum() * a)
    uniq = sum(s['unique_ground_m2'] for s in o['sensors'] if s['kind'] == 'camera')
    # the ground each camera alone sees, the ground two or more see and the blind ground make up the ROI
    assert math.isclose(uniq + multi + cov.metrics['blind_m2'], cov.metrics['roi_area'], abs_tol=0.01)
    items = st.items()
    for s in o['sensors']:
        if s['kind'] == 'camera':
            # removing the camera blinds exactly the ground only it sees
            assert math.isclose(s['blind_without_m2'] - cov.metrics['blind_m2'], s['unique_ground_m2'], abs_tol=1e-3)
        else:
            assert s['unique_ground_m2'] is None
        # and the requirements that fail without it are those an independent run finds failing
        w = copy.deepcopy(v)
        w.uid = 'without'
        w.cameras = [c for c in w.cameras if c.tag != s['tag']]
        w.sensors = [x for x in w.sensors if x.tag != s['tag']]
        for c in w.cameras:
            if c.link_uid and not any(x.uid == c.link_uid for x in w.cameras):
                c.link_uid = ''
        q2 = p.snapshot()
        q2.variants.append(w)
        q2.active_uid = w.uid
        r = RQ.run(q2, eng, RQ.RequirementsSettings.from_dict({'items': items}), issues_list=False)
        fails = sorted(x.item['id'] for x in r.items if x.status != 'pass'
                       and next(i for i in o['items'] if i['id'] == x.item['id'])['status'] == 'pass')
        assert sorted(s['fails_without']) == fails, s
        assert s['why']
    assert any(s['serves'] == 'R1' for s in o['sensors'])   # some camera is there for the 80 m requirement


def test_critique_is_computed(design):
    p, eng, st, res, before = design
    for o in res.options:
        sev = [c['severity'] for c in o['critique']]
        assert sev == sorted(sev, key=lambda s: RD.SEVERITIES.index(s))
        if not o['feasible']:
            assert sev and sev[0] == 'critical'
    cur = res.current
    assert any(c['kind'] == 'requirement' and 'Pedestrians at 80 m forward' in c['message'] for c in cur['critique'])


def test_sensors_in_view_and_power_headroom():
    """The data rig's front radar sits in the front camera's image: the critique says so."""
    from vantage.core.analysis import Engine
    p = samples.tigor_sample_project()
    v = next(v for v in p.variants if v.sensors)
    p.active_uid = v.uid
    st = RD.DesignerSettings.from_dict({'requirements': [{'item': RQ.clean_item(
        {'metric': 'power.steady_w', 'params': {}, 'op': '<', 'target': 175.0, 'priority': 'must', 'text': 'U'}, p)}]})
    d = RD.Designer(p, Engine(), st)
    d.seeds()
    ev = d.evaluate(RD.Rig.from_variant(v), 'full')
    found = d._in_view(ev)
    assert any(f['tags'] == ['F', 'RD1'] for f in found)
    crit = d.critique(ev, [])
    assert any(c['kind'] == 'power' and 'headroom' in c['message'] for c in crit)      # 167 W of 175 W


def test_deterministic_with_a_seed():
    from vantage.core.analysis import Engine
    p = _design_project()
    st = _settings(p, evaluations=12, options=3, placement=False, seed=11)
    a = RD.run(p, Engine(), st)
    b = RD.run(p, Engine(), st)
    assert [o['rig_key'] for o in a.options] == [o['rig_key'] for o in b.options]
    assert [(o['power_w'], o['margin'], o['detection']) for o in a.options] == \
        [(o['power_w'], o['margin'], o['detection']) for o in b.options]
    assert a.search['rigs_scored'] == b.search['rigs_scored']


def test_cancel_and_no_requirements():
    from vantage.core.analysis import Engine
    p = _design_project()
    st = _settings(p, evaluations=200)
    calls = []

    def cancel():
        calls.append(1)
        return len(calls) > 25
    t0 = time.perf_counter()
    assert RD.run(p, Engine(), st, cancel=cancel) is None
    assert time.perf_counter() - t0 < 60
    with pytest.raises(ValueError):
        RD.run(p, Engine(), RD.DesignerSettings())


def test_prices_cost_and_budget(design):
    p, eng, st, res, before = design
    o = res.options[0]
    rig = RD.Rig(o['rig']['cams'], o['rig']['sensors'])
    prices = {ln['key']: 100.0 + k for k, ln in enumerate(RD.bill(p, rig, {})['lines'])}
    b = RD.bill(p, rig, prices)
    assert b['cost'] == sum(ln['qty'] * prices[ln['key']] for ln in b['lines']) and not b['missing']
    one = dict(prices)
    one.pop(next(iter(one)))
    assert RD.bill(p, rig, one)['cost'] is None
    rows, viol, ok = RD._check_constraints({'budget': {'kind': 'budget', 'op': '<', 'value': b['cost'] - 1,
                                                       'priority': 'must', 'text': 'Budget'}}, b['cost'], rig.counts())
    assert rows[0]['status'] == 'fail' and not ok and viol > 0
    rows, viol, ok = RD._check_constraints({'budget': {'kind': 'budget', 'op': '<', 'value': 1e6, 'priority': 'must',
                                                       'text': 'Budget'}}, None, rig.counts())
    assert rows[0]['status'] == 'not_evaluated' and ok


def test_learned_model_is_used_when_trained(monkeypatch):
    """With a trained perception model (perf_model.load_model), detection is its prediction with its uncertainty."""
    from vantage.core.analysis import Engine
    calls = []

    class Model:
        def predict(self, feats):
            calls.append(feats)
            n = feats['cameras']
            mean = {'Pedestrian': {f'S{s}R{r}': min(1.0, 0.1 * n + 0.01 * s) for s in range(8) for r in range(6)}}
            std = {'Pedestrian': {k: 0.05 for k in mean['Pedestrian']}}
            return mean, std
    fake = types.ModuleType('vantage.core.perf_model')
    fake.load_model = lambda: Model()
    fake.describe_rig = lambda project, variant, grid: {'cameras': len(variant.cameras)}
    monkeypatch.setitem(sys.modules, 'vantage.core.perf_model', fake)
    import vantage.core as core
    monkeypatch.setattr(core, 'perf_model', fake, raising=False)
    p = _design_project()
    res = RD.run(p, Engine(), _settings(p, evaluations=8, options=3, placement=False))
    assert res.detection_kind == 'learned' and calls
    o = res.options[0]
    assert o['detection_kind'] == 'learned' and o['detection_std'] is not None and o['detection_std'] > 0
    assert math.isclose(o['detection'], min(1.0, 0.1 * o['counts']['cameras'] + 0.035), abs_tol=0.02)


def test_settings_tolerant_of_junk():
    st = RD.DesignerSettings.from_dict({'options': 99, 'evaluations': -5, 'budget_s': 'x', 'seed': 3.7,
                                        'prices': {'module:a': '12', 'lens:b': -1, 'bad:c': 5, 'sensor:d': 'nan'},
                                        'requirements': [None, {'item': {'metric': 'power.steady_w', 'target': 'x'}}],
                                        'constraints': [{'kind': 'budget', 'value': 'lots'}, {'kind': 'budget',
                                                                                              'value': 500}],
                                        'axes': ['cost', 'cost'], 'currency': 7})
    assert st.options == 7 and st.evaluations == 10 and st.budget_s == 0.0 and st.seed == 4
    assert st.prices == {'module:a': 12.0}
    assert len(st.requirements) == 1 and st.requirements[0]['item']['target'] is None
    assert [c['value'] for c in st.constraints] == [500.0] and st.axes == ['cost', 'margin']
    assert RD.DesignerSettings.from_dict(st.to_dict()).to_dict() == st.to_dict()


def test_accept_writes_variant_and_log(design):
    from vantage.assistant import memory as M
    p, eng, st, res, before = design
    assert json.dumps([v.to_dict() for v in p.variants] + [p.analysis.to_dict()],
                      default=str) == before                 # running the designer changed nothing
    q = p.snapshot()
    q.analysis.designer = st.to_dict()
    n_var, n_log = len(q.variants), len(M.entries(q))
    o = next(o for o in res.options if o['feasible'])
    if RD.needs_approval(o):
        with pytest.raises(ValueError):                      # a large change set needs its approval record first
            RD.accept(q, res, o['label'])
        RD.approve(q, o)
        n_log = len(M.entries(q))
    v = RD.accept(q, res, o['label'], 'cheapest that meets everything')
    assert len(q.variants) == n_var + 1 and q.active_uid == v.uid and v.name.endswith(f'design {o["label"]}')
    log = M.entries(q)[n_log:]
    kinds = [e['kind'] for e in log]
    assert kinds[0] == 'note' and 'decision' in kinds and kinds.count('rejected') == len(res.options) - 1
    assert sum(k == 'constraint' for k in kinds) == len(st.items()) + len(st.active_constraints())
    assert any('cheapest that meets everything' in e['text'] for e in log if e['kind'] == 'decision')
    assert any('B-1.1' in e['text'] for e in log if e['kind'] == 'constraint')
    assert RD.settings_of(q).accepted[-1]['variant_uid'] == v.uid
    big = dict(o, changes=[f'change {k}' for k in range(RD.APPROVE_OVER + 2)])
    res2 = copy.copy(res)
    res2.options = [big]
    with pytest.raises(ValueError):
        RD.accept(q, res2, big['label'])
    RD.approve(q, big)
    RD.accept(q, res2, big['label'])


def test_exports(design, tmp_path):
    p, eng, st, res, before = design
    paths = RD.export(res, p, str(tmp_path))
    names = sorted(os.path.basename(x) for x in paths)
    assert names == sorted(RD.file_names(p))
    doc = json.load(open(next(x for x in paths if x.endswith('.json'))))
    assert doc['schema'] == RD.SCHEMA and len(doc['options']) == len(res.options)
    assert doc['frame'].startswith('vehicle') and doc['limits'] == RD.LIMITS_TEXT
    assert 'path' not in json.dumps(doc['documents'])
    s = RD.summary(res)
    assert json.dumps(s) and len(s['options']) == len(res.options)
    assert open(next(x for x in paths if x.endswith('.png')), 'rb').read(8) == b'\x89PNG\r\n\x1a\n'


def test_reference_seeds_on_this_vehicle():
    p = samples.tigor_sample_project()
    seeds = RD.reference_seeds(p)
    rigs = [r for r, _ in seeds if r is not None]
    assert rigs, 'the nuScenes and KITTI layouts are published and scale onto the body'
    lo, hi = p.vehicle.mesh.bbox
    for r in rigs:
        assert 1 <= len(r.cams) <= RD.MAX_CAMERAS
        for c in r.cams:                                 # snapped onto (or just off) the body
            assert lo[0] - 0.1 <= c['x'] <= hi[0] + 0.1 and lo[1] - 0.1 <= c['y'] <= hi[1] + 0.1
            assert c['module_id'] in p.modules and c['lens_id'] in p.lenses


# =========================================================================== the page
from test_gui import app, win, pump, settle, _wait  # noqa: E402,F401 (fixtures)


def _open(win):
    from vantage.ui.workspaces.designer_ws import DesignerPage
    if 'designer' not in win._page_widgets:
        pg = DesignerPage(win.ctx)
        win.pages.addWidget(pg)
        win._page_widgets['designer'] = pg
    win.set_page('designer')
    return win._page_widgets['designer']


def test_page_brief_read_run_accept_undo(app, win, tmp_path):
    from PyQt6.QtCore import Qt
    from PyQt6.QtTest import QTest
    from vantage.assistant import memory as M
    p = win.ctx.store.project
    v6 = next(v for v in p.variants if v.name == '6-cam · v3')
    if p.active_uid != v6.uid:
        win.switch_variant(v6.uid)
    settle(app, win)
    pg = _open(win)
    pump(app, 0.2)
    pg.brief.setFocus()
    QTest.keyClicks(pg.brief, 'Pedestrians at 80 m forward, 360° up to 20 m, under 300 W, survives a 2 mm mounting '
                              'tolerance')
    pump(app, 0.1)
    assert RD.settings_of(win.ctx.store.project).brief.startswith('Pedestrians at 80 m')
    # a document too: the ODD fixture as a DOCX
    doc = tmp_path / 'odd-fixture.docx'
    doc.write_bytes(make_docx([x.strip() for x in FIXTURE_ODD.split('\n\n') if x.strip()]))
    pg.add_documents([str(doc)])
    pump(app, 0.1)
    assert pg.docs.count() == 1
    pg.method.setCurrentIndex(0)
    QTest.mouseClick(pg.b_read, Qt.MouseButton.LeftButton)
    _wait(app, lambda: not pg.reading)
    pump(app, 0.2)
    st = RD.settings_of(win.ctx.store.project)
    metrics = [(r['item']['metric'], r['item']['params'].get('sector', '')) for r in st.requirements]
    assert ('perception.range', 'front') in metrics and ('power.steady_w', '') in metrics
    assert pg.req.rowCount() == len(st.requirements) + len(st.constraints)
    assert pg.unchecked.rowCount() == len(st.not_checkable) > 0
    src = pg.req.item(0, 6).text()
    assert '"' in src and ('B-' in pg.req.item(0, 6).toolTip() or 'Brief' in src or '¶' in src)
    # keep the brief's rows for a quick search: untick the document's (as a user rejects rows)
    for k in range(pg.req.rowCount()):
        rows = pg._rows()
        kind, i, r = rows[k]
        if (r.get('source') or {}).get('id', '').startswith('D'):
            it = pg.req.item(k, 0)
            if it.checkState() == Qt.CheckState.Checked:
                it.setCheckState(Qt.CheckState.Unchecked)
                pump(app, 0.02)
    st = RD.settings_of(win.ctx.store.project)
    assert all(not r['accepted'] for r in st.requirements + st.constraints if r['source'].get('id', '').startswith('D'))
    # edit a target in the table: one undo step
    k = next(k for k, (kind, i, r) in enumerate(pg._rows()) if kind == 'req' and r['accepted'] and
             r['item']['params'].get('sector') == '')
    n_undo = win.store.undo.index()
    pg.req.item(k, 4).setText('≥ 25')
    pump(app, 0.05)
    assert win.store.undo.index() == n_undo + 1
    st = RD.settings_of(win.ctx.store.project)
    assert any(r['item']['target'] == 25.0 and r['source'].get('edited') for r in st.requirements)
    # a price for one item (the rest stay unpriced: the cost is incomplete)
    pg.c_all.setChecked(False)
    pump(app, 0.05)
    pg.prices.item(0, 2).setText('480')
    pump(app, 0.05)
    assert list(RD.settings_of(win.ctx.store.project).prices.values()) == [480.0]
    pg.s_evals.setValue(18)
    pg.s_options.setValue(3)
    pg.checks['reference_seeds'].setChecked(False)
    pg.checks['vary_lidar'].setChecked(False)
    pg.checks['vary_radar'].setChecked(False)
    pump(app, 0.05)
    before = [v.to_dict() for v in win.ctx.store.project.variants]
    log_before = list(M.entries(win.ctx.store.project))
    QTest.mouseClick(pg.b_run, Qt.MouseButton.LeftButton)
    _wait(app, lambda: not pg.busy, timeout=600)
    pump(app, 0.3)
    assert pg.res is not None, pg.status_text
    assert 3 <= len(pg.card_widgets) == len(pg.res.options)
    # the live project untouched by the run
    assert [v.to_dict() for v in win.ctx.store.project.variants] == before
    assert M.entries(win.ctx.store.project) == log_before
    # the chart: click an option's point
    pg.chart.repaint()
    lab, pt = next((l, q) for l, q in pg.chart._pts if l != 'current') if pg.chart._pts else (None, None)
    if lab is not None:
        QTest.mouseClick(pg.chart, Qt.MouseButton.LeftButton, pos=pt.toPoint())
        pump(app, 0.05)
        assert pg.sel == lab
    for cb, a in zip(pg.ax, ('power', 'detection')):
        cb.setCurrentIndex(cb.findData(a))
    pg._axes_picked()
    pump(app, 0.05)
    assert RD.settings_of(win.ctx.store.project).axes == ['power', 'detection']
    card = pg.card_widgets[0]
    QTest.mouseClick(card.b_show, Qt.MouseButton.LeftButton)
    pump(app, 0.05)
    assert pg.sel == card.label and pg.cmp.rowCount() > 0 and pg.sens.rowCount() == \
        len(pg.res.option(card.label)['sensors']) and pg.crit.count() > 0
    # create the variant: one undo step (with the approval step when the change set is large)
    n_var = len(win.ctx.store.project.variants)
    n_undo = win.store.undo.index()
    pg.why.setText('meets the 80 m requirement')
    QTest.mouseClick(card.b_create, Qt.MouseButton.LeftButton)
    pump(app, 0.1)
    if pg.approve_box.isVisible():
        assert pg.approve_list.count() == len(pg.res.option(card.label)['changes'])
        assert len(win.ctx.store.project.variants) == n_var      # nothing yet
        QTest.mouseClick(pg.b_approve, Qt.MouseButton.LeftButton)
        pump(app, 0.1)
    settle(app, win)
    proj = win.ctx.store.project
    assert len(proj.variants) == n_var + 1 and proj.variant.name.endswith(f'design {card.label}')
    assert win.store.undo.index() == n_undo + 1
    log = M.entries(proj)
    assert any(e['kind'] == 'decision' and 'meets the 80 m requirement' in e['text'] for e in log)
    assert pg.b_compare.isEnabled()
    paths = pg.export(str(tmp_path / 'out'))
    assert len(paths) >= 4
    win.store.undo.undo()
    settle(app, win)
    proj = win.ctx.store.project
    assert len(proj.variants) == n_var and M.entries(proj) == log_before
    win.store.undo.redo()
    settle(app, win)
    assert len(win.ctx.store.project.variants) == n_var + 1


def test_page_cancel_and_project_replaced(app, win):
    from PyQt6.QtCore import Qt
    from PyQt6.QtTest import QTest
    pg = _open(win)
    p = win.ctx.store.project
    st = _settings(p, evaluations=400)
    win.ctx.store.edit('analysis', lambda q: setattr(q.analysis, 'designer', st.to_dict()), 'designer settings')
    pump(app, 0.1)
    QTest.mouseClick(pg.b_run, Qt.MouseButton.LeftButton)
    pump(app, 0.5)
    assert pg.busy
    QTest.mouseClick(pg.b_cancel, Qt.MouseButton.LeftButton)
    _wait(app, lambda: not pg.busy, timeout=120)
    assert pg.res is None and 'Cancelled' in pg.status_text
    pg.res = RD.DesignerResult(RD.DesignerSettings(), [], {}, [], {}, 'geometric', '', [], '', '', '', '', '', False,
                               {})
    win.store.set_project(samples.tigor_sample_project())
    pump(app, 0.2)
    assert pg.res is None and pg.req.rowCount() == 0
