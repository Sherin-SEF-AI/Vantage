"""Checks a model's own module gives: `validation_checks()` in each module of suite.MODULE_MODELS returns check
specifications as dicts, registered here under that model.

    def validation_checks() -> list[dict]:
        return [{'id': 'thermal.planck', 'title': '…', 'quantity': '…', 'reference': '…', 'tolerance': 1e-6,
                 'unit': '%', 'refs': ('planck1901',), 'fn': _check_planck, 'cost': 'cheap'}]

`fn(ctx) -> suite.Outcome` (or a dict of Outcome's fields). A module may also give `validation_references()`:
{key: citation} merged into references.REFERENCES (a key already there is kept). A module that is not installed, or
whose checks cannot be read, adds nothing (the suite runs without it)."""
from __future__ import annotations

import importlib

from . import references as RF
from .suite import CHECKS, MODULE_MODELS, Outcome, Spec


def _wrap(fn):
    import inspect
    try:
        takes_ctx = bool(inspect.signature(fn).parameters)
    except (TypeError, ValueError):
        takes_ctx = True

    def run(ctx):
        out = fn(ctx) if takes_ctx else fn()          # a check of a closed form may need no context
        if isinstance(out, dict):
            return Outcome(**out)
        return Outcome(*out) if isinstance(out, tuple) else out        # (error, detail, …) as Outcome's fields
    return run


# checks given as {'run': 'module:function'} with a text tolerance (core.ego_composite, core.usd_export): each read
# into an error figure against a numeric tolerance here
def _ego_gl(ctx):
    from ..core import ego_composite as EC
    out = EC.check_layer_against_gl()
    rows = out.get('rows') or []
    if not rows:
        return Outcome(0.0, {'skipped': 'no EGL OpenGL here'}, note='skipped: no EGL OpenGL on this computer')
    err = max(100.0 * r['coverage_disagree_share'] for r in rows)
    return Outcome(err, {'views': rows}, ok=bool(out.get('ok')) and all(r['range_median_mm'] < 1.0 for r in rows))


def _ego_cube(ctx):
    from ..core import ego_composite as EC
    out = EC.check_layer_against_cube()
    rows = out.get('rows') or []
    return Outcome(max((1.0 - r['iou'] for r in rows), default=1.0), {'cameras': rows}, ok=bool(out.get('ok')))


def _usd_cameras(ctx):
    from ..core import usd_export as UE
    p = ctx.sample()
    res = UE.run(p, ctx.engine)
    rows = [r for r in UE.check_cameras(res, p) if r.get('max_px') is not None]
    # each camera against its own allowance: 0.01 px, plus 3 × the stated fit RMS for a fitted lens
    ratio = [r['max_px'] / (0.01 + 3.0 * float(r.get('fit_rms_px') or 0.0)) for r in rows]
    return Outcome(max(ratio, default=float('inf')), {'cameras': rows}, ok=bool(rows))


RUN_ADAPTERS = {'ego_layer_vs_gl_depth': (_ego_gl, 1.0, '% of the body\'s pixels',
                                          'share of the composited body\'s pixels whose coverage differs from OpenGL\'s '
                                          'depth buffer (two virtual pinhole views of the sample), the range median '
                                          'below 1 mm besides'),
                'ego_layer_vs_occlusion_cube': (_ego_cube, 0.05, '1 − IoU',
                                                '1 − the IoU of each camera\'s exact body layer with the analysis\'s '
                                                'occlusion-cube mask on the sample\'s data rig (worst camera)'),
                'usd_camera_projection': (_usd_cameras, 1.0, '× the camera\'s allowance',
                                          'largest reprojection error of an exported USD camera against Vantage\'s '
                                          'lens, over its allowance (0.01 px, plus 3 × a fitted lens\'s RMS)')}


def _model_and_id(mid, cid):
    """A check's model and its id as model.name: an id already under a known model keeps it ('formats.…'); one under
    a short form of its model ('neural.…', 'event_…', 'usd_…', 'ego_…') is put under the model's own id."""
    from .suite import MODEL_IDS
    head, dot, rest = cid.partition('.')
    if dot and head in MODEL_IDS:
        return head, cid
    if not dot:
        head, _, rest = cid.partition('_')
    if rest and mid.startswith(head):
        return mid, f'{mid}.{rest}'
    return mid, f'{mid}.{cid}'


def register():
    have = {c.id for c in CHECKS}
    for mid, _title, mod, _anchor in MODULE_MODELS:
        try:
            m = importlib.import_module(mod)
        except ImportError:
            continue
        for name in ('validation_references', 'VALIDATION_REFERENCES', 'REFERENCES'):
            refs = getattr(m, name, None)
            if not refs:
                continue
            try:
                for k, v in dict(refs() if callable(refs) else refs).items():
                    RF.REFERENCES.setdefault(str(k), str(v))
            except Exception:                         # noqa: BLE001 — a module's citations never break the suite
                pass
        fn = getattr(m, 'validation_checks', None)
        if not callable(fn):
            continue
        try:
            items = list(fn())
        except Exception:                             # noqa: BLE001
            continue
        for d in items:
            try:
                cid = str(d['id'])
                if cid in have:
                    continue
                if 'fn' not in d and cid in RUN_ADAPTERS:     # {'run': 'module:function'}: its adapter here
                    fn_, tol, unit, qty = RUN_ADAPTERS[cid]
                    d = dict(d, fn=fn_, tolerance=tol, unit=unit, quantity=qty, cost=d.get('cost', 'study'))
                model, cid = _model_and_id(mid, cid)
                if cid in have:
                    continue
                refs = tuple(str(r) for r in d.get('refs', ()) or ())
                if not refs:                            # no citation key: its reference text is its citation
                    refs = (cid,)
                    RF.REFERENCES.setdefault(cid, str(d.get('reference', '')) or str(d['title']))
                CHECKS.append(Spec(cid, model, str(d['title']), str(d.get('quantity', '')), str(d.get('reference', '')),
                                   float(d['tolerance']), str(d.get('unit', '')), refs,
                                   _wrap(d['fn']), str(d.get('cost', 'cheap'))))
                have.add(cid)
            except (KeyError, TypeError, ValueError):
                continue


register()
