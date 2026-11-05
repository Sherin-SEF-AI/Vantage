"""Documents in, sentences out: the text of a requirements brief, an ODD specification, a regulation excerpt or a test
catalogue, read from .txt, .md, .pdf, .docx or .html, split into sentences that each carry a stable id and where they
were found (page and paragraph), so every requirement the design agent extracts can cite its exact source.

Readers (standard library only, so they work the same in the bundles):

- Text and Markdown: paragraphs are separated by blank lines; a list item or a heading is a paragraph of its own; a
  form feed starts a new page. Markdown emphasis, code marks, link targets and heading marks are dropped.
- HTML: the text of block elements (p, li, h1–h6, td, pre, div …) as paragraphs; script, style and head are skipped.
- DOCX: word/document.xml from the zip: each w:p a paragraph (tables cell by cell), w:tab a space; explicit page
  breaks and the page breaks Word last rendered give the page number. Headers, footers, comments and text boxes are
  not read.
- PDF: pypdf when it is installed (page by page); else a pure-Python reader of the PDF's own content streams: the
  objects (object streams of PDF 1.5 too), the page tree, each page's content streams decoded (FlateDecode,
  ASCIIHexDecode, ASCII85Decode, chained), and the text operators Tj, TJ, ', ", Td, TD, Tm, T* and BT/ET, through the
  fonts' ToUnicode maps where a font has one (else the bytes as Latin-1, which is right for the standard fonts).
  It says which pages it cannot read: a page with images and no text (a scan: needs OCR, which Vantage does not do),
  an encrypted file, a filter it does not know (LZW, JBIG2, …), text in a font without a ToUnicode map and a
  multi-byte encoding (the glyphs cannot be turned back into letters). Columns, footnotes and tables come out in the
  order the PDF draws them.

Sentence ids: '<doc tag>-p<page>-<paragraph>.<sentence>' (the brief: 'B-<paragraph>.<sentence>'), counted from 1 in
reading order, so the same file read again gives the same ids. Each sentence also carries the SHA-1 of its text, so a
citation can be checked against the document it names.
"""
from __future__ import annotations

import base64
import hashlib
import html as _html
import html.parser
import io
import os
import re
import zipfile
import zlib
from dataclasses import dataclass, field

SCHEMA = 'vantage.docs_ingest/1'
FORMATS = ('.txt', '.md', '.markdown', '.pdf', '.docx', '.html', '.htm')
FORMAT_LABEL = {'.txt': 'text', '.md': 'Markdown', '.markdown': 'Markdown', '.pdf': 'PDF', '.docx': 'Word (DOCX)',
                '.html': 'HTML', '.htm': 'HTML'}
MAX_BYTES = 60 * 1024 * 1024
MAX_SENTENCES = 5000
LIMITS = ('Reads the text layer only: a scanned PDF page (an image) needs OCR, which Vantage does not do, and is '
          'named as unread. PDF text comes out in the order the file draws it (two columns may interleave); DOCX '
          'headers, footers, comments and text boxes are not read. Page numbers of a DOCX are those Word last '
          'rendered (or its explicit page breaks).')
_ABBREV = {'e.g', 'i.e', 'etc', 'fig', 'figs', 'no', 'nos', 'approx', 'vs', 'cf', 'incl', 'sec', 'ch', 'eq', 'ref',
           'mr', 'mrs', 'ms', 'dr', 'prof', 'st', 'resp', 'max', 'min', 'typ', 'tab', 'vol', 'pp', 'p', 'al', 'ca',
           'dept', 'est', 'jan', 'feb', 'mar', 'apr', 'jun', 'jul', 'aug', 'sep', 'sept', 'oct', 'nov', 'dec', 'u.s',
           'u.k', 'para', 'art', 'annex'}


@dataclass
class Sentence:
    id: str
    text: str
    doc: str                    # the document's tag ('B' for the brief, 'D1', 'D2' … for documents)
    page: int | None            # 1-based; None where the format has no pages
    para: int                   # paragraph, 1-based, counted over the whole document
    index: int                  # sentence in its paragraph, 1-based
    sha1: str = ''

    def location(self) -> str:
        return (f'p. {self.page}, ' if self.page else '') + f'¶ {self.para}'

    def to_dict(self):
        return {'id': self.id, 'text': self.text, 'doc': self.doc, 'page': self.page, 'para': self.para,
                'index': self.index, 'sha1': self.sha1}


@dataclass
class Document:
    tag: str
    name: str                   # file name (never the full path in exports)
    kind: str                   # FORMAT_LABEL value, or 'brief'
    sentences: list = field(default_factory=list)
    pages: int = 0              # pages read (0: a format without pages)
    problems: list = field(default_factory=list)      # what could not be read, in words
    method: str = ''            # 'text' | 'markdown' | 'html' | 'docx' | 'pypdf' | 'pdf-builtin'
    sha256: str = ''
    path: str = ''              # where it was read from (kept in the project, left out of exports)

    @property
    def readable(self) -> bool:
        return bool(self.sentences)

    def sentence(self, sid):
        return next((s for s in self.sentences if s.id == sid), None)

    def to_dict(self, with_path=False):
        d = {'tag': self.tag, 'name': self.name, 'kind': self.kind, 'pages': self.pages, 'method': self.method,
             'sha256': self.sha256, 'problems': list(self.problems), 'sentences': [s.to_dict() for s in self.sentences]}
        if with_path:
            d['path'] = self.path
        return d


# =========================================================================== sentences
_SPLIT = re.compile(r'(?<=[.!?])["\')\]]?\s+(?=["\'(\[]?[A-Z0-9§•–\-])')


def _clean_ws(s: str) -> str:
    s = s.replace('­', '').replace(' ', ' ').replace(' ', ' ').replace(' ', ' ')
    return re.sub(r'\s+', ' ', s).strip()


def split_sentences(text: str) -> list:
    """The sentences of one paragraph: split after . ! ? followed by a capital or a digit, never after an
    abbreviation (e.g., Fig., No., approx. …), an initial, or a number's decimal point."""
    text = _clean_ws(text)
    if not text:
        return []
    out, start = [], 0
    for mt in _SPLIT.finditer(text):
        cut = mt.start()
        head = text[start:cut]
        last = re.search(r'(?:^|[\s(])([A-Za-z][A-Za-z.]*)\.["\')\]]?$', head)
        if last:
            w = last.group(1).lower().rstrip('.')
            if w in _ABBREV or len(w) == 1:
                continue
        if re.search(r'(?:^|\s)\d{1,2}\.$', head) and len(head.strip()) <= 3:   # a list number "3."
            continue
        piece = head.strip()
        if piece:
            out.append(piece)
        start = mt.end()
    tail = text[start:].strip()
    if tail:
        out.append(tail)
    return out


def _slug(tag: str) -> str:
    return re.sub(r'[^A-Za-z0-9]', '', str(tag))[:8] or 'D'


def build(tag, name, kind, paragraphs, method='', paged=False, problems=(), sha256='', path='') -> Document:
    """A Document from [(page, paragraph text)]: sentence ids in reading order."""
    tag = _slug(tag)
    doc = Document(tag, name, kind, [], 0, list(problems), method, sha256, path)
    pages = set()
    n_para = 0
    for page, text in paragraphs:
        sents = split_sentences(text)
        if not sents:
            continue
        n_para += 1
        if paged and page:
            pages.add(page)
        for k, s in enumerate(sents, 1):
            if len(doc.sentences) >= MAX_SENTENCES:
                doc.problems.append(f'Only the first {MAX_SENTENCES} sentences were read.')
                doc.pages = len(pages)
                return doc
            sid = f'{tag}-p{page}-{n_para}.{k}' if paged and page else f'{tag}-{n_para}.{k}'
            doc.sentences.append(Sentence(sid, s, tag, page if paged else None, n_para, k,
                                          hashlib.sha1(s.encode('utf-8')).hexdigest()[:12]))
    doc.pages = len(pages)
    return doc


def from_text(text: str, name='Brief', tag='B', kind='brief') -> Document:
    """The brief typed into the page (or any plain text) as a Document."""
    text = text if isinstance(text, str) else ''
    return build(tag, name, kind, [(None, p) for _, p in _text_paragraphs(text)], 'text',
                 sha256=hashlib.sha256(text.encode('utf-8')).hexdigest())


# =========================================================================== plain text and Markdown
_BULLET = re.compile(r'^\s*(?:[-*+•▪◦–]|\(?\d{1,3}[.)]|\(?[a-z][.)])\s+')


def _text_paragraphs(text: str, markdown=False) -> list:
    """[(page, paragraph)]: blank lines end a paragraph, a list item or heading is one of its own, \\f a page."""
    out = []
    for pno, page in enumerate(text.replace('\r\n', '\n').replace('\r', '\n').split('\f'), 1):
        buf = []

        def flush():
            if buf:
                out.append((pno, ' '.join(buf)))
                buf.clear()
        in_code = False
        for line in page.split('\n'):
            s = line.strip()
            if markdown and s.startswith('```'):
                in_code = not in_code
                flush()
                continue
            if markdown and in_code:
                continue
            if not s:
                flush()
                continue
            if markdown:
                if re.match(r'^(?:-{3,}|\*{3,}|_{3,})$', s):
                    flush()
                    continue
                if re.match(r'^\|?\s*:?-{2,}', s):              # a table's rule
                    continue
                head = re.match(r'^#{1,6}\s+(.*)$', s)
                if head:
                    flush()
                    out.append((pno, _md_inline(head.group(1))))
                    continue
                s = re.sub(r'^>\s?', '', s)
                if s.startswith('|') and s.endswith('|'):
                    flush()
                    for cell in s.strip('|').split('|'):
                        if cell.strip():
                            out.append((pno, _md_inline(cell)))
                    continue
            if _BULLET.match(line):
                flush()
                buf.append(_md_inline(_BULLET.sub('', line).strip()) if markdown else _BULLET.sub('', line).strip())
                continue
            buf.append(_md_inline(s) if markdown else s)
        flush()
    return out


def _md_inline(s: str) -> str:
    s = re.sub(r'!\[([^\]]*)\]\([^)]*\)', r'\1', s)
    s = re.sub(r'\[([^\]]+)\]\([^)]*\)', r'\1', s)
    s = re.sub(r'`([^`]*)`', r'\1', s)
    s = re.sub(r'(\*\*|__)(.+?)\1', r'\2', s)
    s = re.sub(r'(?<![\w*])\*(?!\s)(.+?)(?<!\s)\*(?![\w*])', r'\1', s)
    s = re.sub(r'<[^>]+>', '', s)
    return s.strip()


# =========================================================================== HTML
_BLOCKS = {'p', 'li', 'h1', 'h2', 'h3', 'h4', 'h5', 'h6', 'td', 'th', 'pre', 'div', 'section', 'article', 'dd', 'dt',
           'blockquote', 'caption', 'figcaption', 'tr', 'ul', 'ol', 'table', 'body', 'main', 'header', 'footer'}
_SKIP = {'script', 'style', 'head', 'noscript', 'template', 'svg'}


class _HtmlText(html.parser.HTMLParser):
    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.paras, self.buf, self.skip = [], [], 0

    def _flush(self):
        t = _clean_ws(''.join(self.buf))
        if t:
            self.paras.append(t)
        self.buf = []

    def handle_starttag(self, tag, attrs):
        if tag in _SKIP:
            self.skip += 1
        elif tag in _BLOCKS:
            self._flush()
        elif tag == 'br':
            self.buf.append(' ')

    def handle_endtag(self, tag):
        if tag in _SKIP:
            self.skip = max(0, self.skip - 1)
        elif tag in _BLOCKS:
            self._flush()

    def handle_data(self, data):
        if not self.skip:
            self.buf.append(data)


def html_paragraphs(data: bytes) -> list:
    text = _decode(data)
    p = _HtmlText()
    p.feed(text)
    p.close()
    p._flush()
    return [(None, t) for t in p.paras]


def _decode(data: bytes) -> str:
    for enc in ('utf-8-sig', 'utf-16') if data[:2] in (b'\xff\xfe', b'\xfe\xff') else ('utf-8-sig',):
        try:
            return data.decode(enc)
        except UnicodeDecodeError:
            pass
    return data.decode('cp1252', 'replace')


# =========================================================================== DOCX
_W = '{http://schemas.openxmlformats.org/wordprocessingml/2006/main}'


def docx_paragraphs(data: bytes) -> tuple[list, list]:
    """([(page, paragraph)], problems) from a .docx: word/document.xml, paragraphs in order (table cells too)."""
    import xml.etree.ElementTree as ET
    try:
        z = zipfile.ZipFile(io.BytesIO(data))
        xml = z.read('word/document.xml')
    except (zipfile.BadZipFile, KeyError) as e:
        return [], [f'Not a Word document (no word/document.xml): {e}']
    try:
        root = ET.fromstring(xml)
    except ET.ParseError as e:
        return [], [f'The document XML could not be parsed: {e}']
    out, page = [], 1
    rendered = xml.count(b'lastRenderedPageBreak') > 0 or b'w:type="page"' in xml

    def para_text(p):
        nonlocal page
        bits = []
        brk = False
        for el in p.iter():
            if el.tag == _W + 't':
                bits.append(el.text or '')
            elif el.tag == _W + 'tab':
                bits.append(' ')
            elif el.tag == _W + 'br':
                if el.get(_W + 'type') == 'page':
                    brk = True
                else:
                    bits.append(' ')
            elif el.tag == _W + 'lastRenderedPageBreak' and bits:
                brk = True
            elif el.tag == _W + 'lastRenderedPageBreak':
                page += 1
            elif el.tag == _W + 'noBreakHyphen':
                bits.append('-')
        return ''.join(bits), brk
    body = root.find(_W + 'body')
    for p in (body.iter(_W + 'p') if body is not None else ()):
        text, brk = para_text(p)
        if text.strip():
            out.append((page, text))
        if brk:
            page += 1
    problems = []
    names = set(z.namelist())
    if any(n.startswith('word/media/') for n in names) and not out:
        problems.append('The document holds images and no text: scanned pages need OCR, which Vantage does not do.')
    return [(pg if rendered else None, t) for pg, t in out], problems


# =========================================================================== PDF
def _pdf_unescape(s: bytes) -> bytes:
    out = bytearray()
    i, n = 0, len(s)
    while i < n:
        c = s[i]
        if c == 0x5C and i + 1 < n:                    # backslash
            d = s[i + 1]
            m = {ord('n'): b'\n', ord('r'): b'\r', ord('t'): b'\t', ord('b'): b'\b', ord('f'): b'\f',
                 ord('('): b'(', ord(')'): b')', ord('\\'): b'\\'}.get(d)
            if m is not None:
                out += m
                i += 2
                continue
            if 0x30 <= d <= 0x37:
                j = i + 1
                while j < n and j < i + 4 and 0x30 <= s[j] <= 0x37:
                    j += 1
                out.append(int(s[i + 1:j], 8) & 0xFF)
                i = j
                continue
            if d in (0x0A, 0x0D):                       # a line continuation
                i += 2
                if d == 0x0D and i < n and s[i] == 0x0A:
                    i += 1
                continue
            i += 1
            continue
        out.append(c)
        i += 1
    return bytes(out)


_TOKEN = re.compile(rb'\s*(?:(%[^\r\n]*)|(\()|(<<|>>)|(<[0-9A-Fa-f\s]*>)|(\[|\])|(/[^\s/\[\]()<>{}%]*)|'
                    rb'([+-]?(?:\d+\.?\d*|\.\d+))|([A-Za-z\'"*][A-Za-z0-9\'"*]*)|(\{|\}))')


def _content_tokens(data: bytes):
    """(kind, value) tokens of a content stream: 'str' (bytes), 'hex' (bytes), 'num', 'name', 'op', '[' , ']'."""
    i, n = 0, len(data)
    while i < n:
        mt = _TOKEN.match(data, i)
        if not mt:
            i += 1
            continue
        if mt.group(1) is not None:
            i = mt.end()
            continue
        if mt.group(2) is not None:                       # a literal string, nested parentheses
            j, depth = mt.end(), 1
            while j < n and depth:
                c = data[j]
                if c == 0x5C:
                    j += 2
                    continue
                if c == 0x28:
                    depth += 1
                elif c == 0x29:
                    depth -= 1
                j += 1
            yield 'str', _pdf_unescape(data[mt.end():j - 1])
            i = j
            continue
        i = mt.end()
        if mt.group(3) is not None:
            yield 'dict', mt.group(3)
        elif mt.group(4) is not None:
            h = re.sub(rb'\s', b'', mt.group(4)[1:-1])
            if len(h) % 2:
                h += b'0'
            yield 'hex', bytes.fromhex(h.decode('ascii'))
        elif mt.group(5) is not None:
            yield mt.group(5).decode(), None
        elif mt.group(6) is not None:
            yield 'name', mt.group(6)[1:].decode('latin-1')
        elif mt.group(7) is not None:
            yield 'num', float(mt.group(7))
        elif mt.group(8) is not None:
            op = mt.group(8).decode('latin-1')
            if op == 'BI':                                # an inline image: skip to EI
                k = data.find(b'EI', i)
                i = n if k < 0 else k + 2
                continue
            yield 'op', op


class _Pdf:
    """The objects of a PDF file (by number), streams decoded on request."""

    def __init__(self, data: bytes):
        self.data = data
        self.objs = {}               # num -> raw bytes of the object body (dict and stream)
        self.problems = []
        for mt in re.finditer(rb'(?<![\d])(\d+)\s+(\d+)\s+obj\b', data):
            num = int(mt.group(1))
            end = data.find(b'endobj', mt.end())
            if end < 0:
                end = len(data)
            body = data[mt.end():end]
            st = body.find(b'stream')
            if st >= 0:                                    # endobj may sit inside binary stream data: use /Length
                length = self._length(body[:st])
                if length is not None:
                    s0 = mt.end() + st + 6
                    if data[s0:s0 + 2] == b'\r\n':
                        s0 += 2
                    elif data[s0:s0 + 1] in (b'\n', b'\r'):
                        s0 += 1
                    e2 = data.find(b'endobj', s0 + length)
                    if e2 > 0:
                        body = data[mt.end():e2]
            self.objs[num] = body
        for num, body in list(self.objs.items()):         # objects inside object streams (PDF 1.5)
            d = self.dict_of(body)
            if re.search(rb'/Type\s*/ObjStm', d):
                raw = self.stream(num)
                if raw is None:
                    continue
                nn = self._int(d, b'N') or 0
                first = self._int(d, b'First') or 0
                head = raw[:first].split()
                for k in range(min(nn, len(head) // 2)):
                    onum, off = int(head[2 * k]), int(head[2 * k + 1])
                    nxt = int(head[2 * k + 3]) if 2 * k + 3 < len(head) else len(raw) - first
                    self.objs.setdefault(onum, raw[first + off:first + nxt])

    @staticmethod
    def _int(d, key):
        mt = re.search(rb'/' + key + rb'\s+(\d+)(?!\s+\d+\s+R)', d)
        return int(mt.group(1)) if mt else None

    def _length(self, d):
        mt = re.search(rb'/Length\s+(\d+)(\s+(\d+)\s+R)?', d)
        if not mt:
            return None
        if mt.group(2):
            ref = self.objs.get(int(mt.group(1)))
            if ref is None:
                return None
            v = re.match(rb'\s*(\d+)', ref)
            return int(v.group(1)) if v else None
        return int(mt.group(1))

    @staticmethod
    def dict_of(body: bytes) -> bytes:
        st = body.find(b'stream')
        return body[:st] if st >= 0 else body

    def resolve(self, d: bytes, key: bytes) -> bytes:
        """The value of /key in dictionary text d (an indirect reference followed): raw bytes."""
        mt = re.search(rb'/' + key + rb'\s*(\d+)\s+\d+\s+R', d)
        if mt:
            return self.dict_of(self.objs.get(int(mt.group(1)), b''))
        mt = re.search(rb'/' + key + rb'\s*(<<)', d)
        if mt:
            return _balanced(d, mt.start(1))
        mt = re.search(rb'/' + key + rb'\s*(\[[^\]]*\]|/[A-Za-z0-9#+-]+)', d)
        return mt.group(1) if mt else b''

    def stream(self, num):
        body = self.objs.get(num)
        if body is None:
            return None
        st = body.find(b'stream')
        if st < 0:
            return None
        d = body[:st]
        s0 = st + 6
        if body[s0:s0 + 2] == b'\r\n':
            s0 += 2
        elif body[s0:s0 + 1] in (b'\n', b'\r'):
            s0 += 1
        length = self._length(d)
        e = body.rfind(b'endstream')
        raw = body[s0:s0 + length] if length is not None and s0 + length <= len(body) else body[s0:e if e > 0 else None]
        filters = re.findall(rb'/(FlateDecode|Fl|ASCIIHexDecode|AHx|ASCII85Decode|A85|LZWDecode|LZW|RunLengthDecode|'
                             rb'RL|CCITTFaxDecode|CCF|DCTDecode|DCT|JBIG2Decode|JPXDecode|Crypt)\b',
                             (re.search(rb'/Filter\s*(\[[^\]]*\]|/\w+)', d) or [b'', b''])[1] or b'')
        for f in filters:
            f = f.decode()
            if f in ('FlateDecode', 'Fl'):
                try:
                    raw = zlib.decompress(raw)
                except zlib.error:
                    try:
                        raw = zlib.decompressobj().decompress(raw)
                    except zlib.error:
                        return None
                pred = re.search(rb'/Predictor\s+(\d+)', d)
                if pred and int(pred.group(1)) >= 10:
                    cols = re.search(rb'/Columns\s+(\d+)', d)
                    raw = _png_unpredict(raw, int(cols.group(1)) if cols else 1)
            elif f in ('ASCIIHexDecode', 'AHx'):
                h = re.sub(rb'[^0-9A-Fa-f]', b'', raw.split(b'>')[0])
                raw = bytes.fromhex((h + b'0' * (len(h) % 2)).decode())
            elif f in ('ASCII85Decode', 'A85'):
                t = raw.strip()
                if t.startswith(b'<~'):
                    t = t[2:]
                t = t.split(b'~>')[0]
                try:
                    raw = base64.a85decode(re.sub(rb'\s', b'', t))
                except ValueError:
                    return None
            else:
                return ('unsupported', f)
        return raw


def _png_unpredict(raw: bytes, cols: int) -> bytes:
    """The PNG 'Up' predictor of xref streams (other predictors as the raw rows)."""
    out, prev = bytearray(), bytearray(cols)
    step = cols + 1
    for i in range(0, len(raw) - step + 1, step):
        ft, row = raw[i], bytearray(raw[i + 1:i + step])
        if ft == 2:
            row = bytearray((a + b) & 0xFF for a, b in zip(row, prev))
        out += row
        prev = row
    return bytes(out)


def _balanced(d: bytes, i: int) -> bytes:
    """The << … >> dictionary starting at d[i]."""
    depth, j = 0, i
    while j < len(d) - 1:
        if d[j:j + 2] == b'<<':
            depth += 1
            j += 2
            continue
        if d[j:j + 2] == b'>>':
            depth -= 1
            j += 2
            if depth == 0:
                return d[i:j]
            continue
        j += 1
    return d[i:]


def _cmap(data: bytes) -> tuple[dict, int]:
    """A ToUnicode CMap: ({code bytes: text}, code length in bytes)."""
    m, width = {}, 1
    for mt in re.finditer(rb'begincodespacerange(.*?)endcodespacerange', data, re.S):
        hx = re.findall(rb'<([0-9A-Fa-f]+)>', mt.group(1))
        if hx:
            width = max(width, len(hx[0]) // 2)

    def uni(h):
        try:
            return bytes.fromhex(h.decode()).decode('utf-16-be', 'replace')
        except ValueError:
            return ''
    for mt in re.finditer(rb'beginbfchar(.*?)endbfchar', data, re.S):
        for a, b in re.findall(rb'<([0-9A-Fa-f]+)>\s*<([0-9A-Fa-f]*)>', mt.group(1)):
            m[bytes.fromhex(a.decode())] = uni(b)
    for mt in re.finditer(rb'beginbfrange(.*?)endbfrange', data, re.S):
        body = mt.group(1)
        for a, b, rest in re.findall(rb'<([0-9A-Fa-f]+)>\s*<([0-9A-Fa-f]+)>\s*(\[[^\]]*\]|<[0-9A-Fa-f]*>)', body):
            lo, hi, w = int(a, 16), int(b, 16), len(a) // 2
            if hi - lo > 65535:
                continue
            if rest.startswith(b'['):
                vals = re.findall(rb'<([0-9A-Fa-f]*)>', rest)
                for k, v in enumerate(vals[:hi - lo + 1]):
                    m[(lo + k).to_bytes(w, 'big')] = uni(v)
            else:
                base = bytes.fromhex(rest[1:-1].decode())
                for k in range(hi - lo + 1):
                    if len(base) >= 2:
                        cp = int.from_bytes(base[-2:], 'big') + k
                        m[(lo + k).to_bytes(w, 'big')] = (base[:-2] + cp.to_bytes(2, 'big')).decode('utf-16-be',
                                                                                                   'replace')
                    else:
                        m[(lo + k).to_bytes(w, 'big')] = chr(base[0] + k) if base else ''
    return m, width


class _Font:
    def __init__(self, cmap=None, width=1, multibyte=False):
        self.cmap, self.width, self.multibyte = cmap, width, multibyte

    def text(self, b: bytes) -> str:
        if self.cmap:
            w = self.width
            out = []
            for i in range(0, len(b) - w + 1, w):
                out.append(self.cmap.get(b[i:i + w], ''))
            return ''.join(out)
        if self.multibyte:
            return ''
        return b.decode('cp1252', 'replace')


def _page_fonts(pdf: _Pdf, page_dict: bytes) -> dict:
    res = pdf.resolve(page_dict, b'Resources')
    fd = pdf.resolve(res, b'Font') if res else b''
    out = {}
    for name, num in re.findall(rb'/([^\s/<>\[\]]+)\s+(\d+)\s+\d+\s+R', fd):
        body = pdf.dict_of(pdf.objs.get(int(num), b''))
        tu = re.search(rb'/ToUnicode\s+(\d+)\s+\d+\s+R', body)
        multibyte = bool(re.search(rb'/Subtype\s*/Type0', body)) or b'Identity-H' in body
        cmap, width = (None, 2 if multibyte else 1)
        if tu:
            raw = pdf.stream(int(tu.group(1)))
            if isinstance(raw, bytes):
                cmap, width = _cmap(raw)
        out[name.decode('latin-1')] = _Font(cmap, width, multibyte)
    return out


def _page_text(pdf: _Pdf, streams: list, fonts: dict) -> tuple[list, bool]:
    """([paragraph], undecodable): the text drawn by the page's content streams, lines joined into paragraphs at a
    vertical gap larger than 1.6 × the typical line step."""
    lines, cur, font = [], [], _Font()
    y, line_y, steps = 0.0, 0.0, []
    leading = 0.0                      # the text leading (TL), else 1.2 × the font size
    undecodable = False
    stack = []

    def newline():
        nonlocal cur, line_y
        t = ''.join(cur).strip()
        if t:
            lines.append((line_y, t))
        cur = []
        line_y = y

    def show(b):
        nonlocal undecodable
        t = font.text(b)
        if b and not t and font.multibyte and not font.cmap:
            undecodable = True
        cur.append(t)
    for data in streams:
        for kind, val in _content_tokens(data):
            if kind == 'op':
                if val == 'Tf' and len(stack) >= 2 and stack[-2][0] == 'name':
                    font = fonts.get(stack[-2][1], _Font())
                    if stack[-1][0] == 'num' and not leading:
                        leading = 1.2 * abs(stack[-1][1])
                elif val == 'TL' and stack and stack[-1][0] == 'num':
                    leading = abs(stack[-1][1])
                elif val in ('Tj', "'", '"'):
                    if val != 'Tj':
                        newline()
                    s = next((v for k, v in reversed(stack) if k in ('str', 'hex')), None)
                    if s is not None:
                        show(s)
                elif val == 'TJ':
                    arr = []
                    depth = None
                    for k in range(len(stack) - 1, -1, -1):
                        if stack[k][0] == '[':
                            depth = k
                            break
                    for k, v in stack[(depth + 1) if depth is not None else 0:]:
                        if k in ('str', 'hex'):
                            arr.append(font.text(v))
                            if v and not arr[-1] and font.multibyte and not font.cmap:
                                undecodable = True
                        elif k == 'num' and v < -180:
                            arr.append(' ')
                    cur.append(''.join(arr))
                elif val in ('Td', 'TD'):
                    nums = [v for k, v in stack[-2:] if k == 'num']
                    if len(nums) == 2:
                        tx, ty = nums
                        if abs(ty) > 0.01:
                            y += ty
                            steps.append(abs(ty))
                            newline()
                        elif tx > 1.0 and cur and not ''.join(cur).endswith(' '):
                            cur.append(' ')
                elif val == 'Tm':
                    nums = [v for k, v in stack[-6:] if k == 'num']
                    if len(nums) == 6:
                        if abs(nums[5] - y) > 0.01:
                            steps.append(abs(nums[5] - y))
                            y = nums[5]
                            newline()
                        elif cur and not ''.join(cur).endswith(' '):
                            cur.append(' ')
                elif val == 'T*':
                    y -= steps[-1] if steps else 12.0
                    newline()
                elif val in ('ET', 'BT'):
                    if val == 'ET' and cur:
                        newline()
                stack = []
            else:
                stack.append((kind, val))
                if len(stack) > 4000:
                    stack = stack[-2000:]
    if cur:
        newline()
    if not lines:
        return [], undecodable
    # the typical line step: the lower quartile of the steps between consecutive lines (the first move of a text
    # block is an absolute position, and paragraph gaps are the larger steps)
    gaps = sorted(abs(a[0] - b[0]) for a, b in zip(lines, lines[1:]) if abs(a[0] - b[0]) > 0.5)
    typical = gaps[len(gaps) // 4] if gaps else (leading or 12.0)
    if leading:
        typical = min(typical, leading)
    paras, buf, prev_y = [], [], None
    for ly, t in lines:
        if prev_y is not None and abs(prev_y - ly) > 1.6 * typical and buf:
            paras.append(_join_lines(buf))
            buf = []
        if _BULLET.match(t):
            if buf:
                paras.append(_join_lines(buf))
                buf = []
            t = _BULLET.sub('', t)
        buf.append(t)
        prev_y = ly
    if buf:
        paras.append(_join_lines(buf))
    return paras, undecodable


def _join_lines(lines) -> str:
    out = ''
    for t in lines:
        t = t.strip()
        if out.endswith('-') and t[:1].islower() and not out.endswith(' -'):
            out = out[:-1] + t                 # a word broken at the line end
        else:
            out = (out + ' ' + t) if out else t
    return out


def _page_order(pdf: _Pdf) -> list:
    """The page objects' numbers in reading order: the page tree from the catalogue, else by object number."""
    pages = []
    cat = next((n for n, b in pdf.objs.items() if re.search(rb'/Type\s*/Catalog', pdf.dict_of(b))), None)
    if cat is not None:
        mt = re.search(rb'/Pages\s+(\d+)\s+\d+\s+R', pdf.dict_of(pdf.objs[cat]))
        seen = set()

        def walk(num, depth=0):
            if num in seen or depth > 40 or num not in pdf.objs:
                return
            seen.add(num)
            d = pdf.dict_of(pdf.objs[num])
            if re.search(rb'/Type\s*/Pages\b', d):
                kids = re.search(rb'/Kids\s*\[([^\]]*)\]', d)
                for k in re.findall(rb'(\d+)\s+\d+\s+R', kids.group(1) if kids else b''):
                    walk(int(k), depth + 1)
            elif re.search(rb'/Type\s*/Page\b', d):
                pages.append(num)
        if mt:
            walk(int(mt.group(1)))
    if not pages:
        pages = sorted(n for n, b in pdf.objs.items() if re.search(rb'/Type\s*/Page\b', pdf.dict_of(b)))
    return pages


def _has_images(pdf: _Pdf, page_dict: bytes) -> bool:
    res = pdf.resolve(page_dict, b'Resources')
    xo = pdf.resolve(res, b'XObject') if res else b''
    for num in re.findall(rb'(\d+)\s+\d+\s+R', xo):
        if re.search(rb'/Subtype\s*/Image', pdf.dict_of(pdf.objs.get(int(num), b''))):
            return True
    return False


def pdf_paragraphs_builtin(data: bytes) -> tuple[list, list, int]:
    """([(page, paragraph)], problems, pages) by the pure-Python reader (see the module doc)."""
    if not data.startswith(b'%PDF'):
        return [], ['Not a PDF file (no %PDF header).'], 0
    pdf = _Pdf(data)
    if re.search(rb'/Encrypt\s+\d+\s+\d+\s+R|/Encrypt\s*<<', data):
        return [], ['The PDF is encrypted: Vantage cannot read it (save an unencrypted copy).'], 0
    order = _page_order(pdf)
    out, problems, unread = [], [], []
    for pno, num in enumerate(order, 1):
        d = pdf.dict_of(pdf.objs[num])
        refs = []
        mt = re.search(rb'/Contents\s*(\[([^\]]*)\]|(\d+)\s+\d+\s+R)', d)
        if mt:
            refs = [int(x) for x in re.findall(rb'(\d+)\s+\d+\s+R', mt.group(1))]
        streams, bad = [], ''
        for r in refs:
            s = pdf.stream(r)
            if isinstance(s, tuple):
                bad = s[1]
            elif s is not None:
                streams.append(s)
        paras, undecodable = _page_text(pdf, streams, _page_fonts(pdf, d))
        if paras:
            out += [(pno, t) for t in paras]
        if bad:
            problems.append(f'Page {pno}: a content stream uses the {bad} filter, which this reader does not decode.')
        elif undecodable and not paras:
            problems.append(f'Page {pno}: its text uses a font without a ToUnicode map: the glyphs cannot be read '
                            'back as letters.')
        elif not paras:
            unread.append(pno)
            if _has_images(pdf, d):
                problems.append(f'Page {pno}: an image and no text (a scanned page?): it needs OCR, which Vantage '
                                'does not do.')
            else:
                problems.append(f'Page {pno}: no text found.')
    if not order:
        problems.append('No pages found in the PDF (damaged, or a structure this reader does not know).')
    return out, problems, len(order)


def pdf_paragraphs(data: bytes) -> tuple[list, list, int, str]:
    """([(page, paragraph)], problems, pages, method): pypdf when installed, else the built-in reader."""
    try:
        import pypdf                                   # optional: a better reader when the user has it
    except ImportError:
        pypdf = None
    if pypdf is not None:
        try:
            r = pypdf.PdfReader(io.BytesIO(data))
            if r.is_encrypted:
                return [], ['The PDF is encrypted: Vantage cannot read it (save an unencrypted copy).'], 0, 'pypdf'
            out, problems = [], []
            for pno, pg in enumerate(r.pages, 1):
                t = pg.extract_text() or ''
                paras = [p for _, p in _text_paragraphs(t)]
                if not paras:
                    problems.append(f'Page {pno}: no text (a scanned page needs OCR, which Vantage does not do).')
                out += [(pno, p) for p in paras]
            return out, problems, len(r.pages), 'pypdf'
        except Exception as e:                            # fall back to the built-in reader
            pass
    out, problems, n = pdf_paragraphs_builtin(data)
    return out, problems, n, 'pdf-builtin'


# =========================================================================== reading a file
def read(path: str, tag='D1', data: bytes | None = None) -> Document:
    """A document from a file (or its bytes): its sentences with ids and locations, and what could not be read."""
    name = os.path.basename(path or '') or 'document'
    ext = os.path.splitext(name)[1].lower()
    if data is None:
        try:
            size = os.path.getsize(path)
            if size > MAX_BYTES:
                return Document(_slug(tag), name, FORMAT_LABEL.get(ext, ext), [], 0,
                                [f'The file is {size / 1e6:.0f} MB, over the {MAX_BYTES / 1e6:.0f} MB limit.'], '',
                                '', path)
            with open(path, 'rb') as f:
                data = f.read()
        except OSError as e:
            return Document(_slug(tag), name, FORMAT_LABEL.get(ext, ext), [], 0, [f'Could not open it: {e}'], '', '',
                            path)
    sha = hashlib.sha256(data).hexdigest()
    kind = FORMAT_LABEL.get(ext)
    if kind is None:
        if data.startswith(b'%PDF'):
            ext, kind = '.pdf', 'PDF'
        elif data.startswith(b'PK'):
            ext, kind = '.docx', 'Word (DOCX)'
        else:
            return Document(_slug(tag), name, ext or '?', [], 0,
                            [f'Vantage reads {", ".join(FORMATS)}; {ext or "this file"} is not one of them.'], '', sha,
                            path)
    if ext == '.pdf':
        paras, problems, pages, method = pdf_paragraphs(data)
        doc = build(tag, name, kind, paras, method, True, problems, sha, path)
        doc.pages = pages
    elif ext == '.docx':
        paras, problems = docx_paragraphs(data)
        doc = build(tag, name, kind, paras, 'docx', any(p for p, _ in paras), problems, sha, path)
    elif ext in ('.html', '.htm'):
        doc = build(tag, name, kind, html_paragraphs(data), 'html', False, (), sha, path)
    else:
        md = ext in ('.md', '.markdown')
        text = _decode(data)
        paras = _text_paragraphs(text, markdown=md)
        paged = '\f' in text
        doc = build(tag, name, kind, paras if paged else [(None, p) for _, p in paras], 'markdown' if md else 'text',
                    paged, (), sha, path)
    if not doc.sentences and not doc.problems:
        doc.problems.append('No text found in it.')
    return doc


def find(docs, sid):
    """The sentence with id sid in any of the documents (None when none has it)."""
    for d in docs:
        s = d.sentence(sid)
        if s is not None:
            return d, s
    return None, None


def summary(docs) -> dict:
    return {'documents': [{'tag': d.tag, 'name': d.name, 'kind': d.kind, 'pages': d.pages,
                           'sentences': len(d.sentences), 'method': d.method, 'problems': list(d.problems)}
                          for d in docs]}


def to_doc(docs) -> dict:
    return {'schema': SCHEMA, 'limits': LIMITS, 'documents': [d.to_dict() for d in docs]}
