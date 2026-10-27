"""Text on the PNG exports drawn with OpenCV: its Hershey fonts are ASCII only, so typographic characters are spelled
out (· → -, × → x, ° → deg, → → ->, …) instead of coming out as '??'."""
from __future__ import annotations

TABLE = str.maketrans({'·': '-', '–': '-', '—': '-', '°': ' deg', '×': 'x', '≥': '>=', '≤': '<=', '²': '2', '³': '3',
                       '→': '->', '←': '<-', 'µ': 'u', 'λ': 'lambda', '±': '+/-', 'Δ': 'd', 'σ': 'sigma', '…': '...',
                       '‘': "'", '’': "'", '“': '"', '”': '"', '▸': '>'})


def ascii_text(text) -> str:
    return str(text).translate(TABLE).encode('ascii', 'replace').decode('ascii')


def put_text(img, text, *args, **kw):
    """cv2.putText with the text spelled in ASCII."""
    import cv2
    return cv2.putText(img, ascii_text(text), *args, **kw)
