"""Validation suite: quantitative checks of Vantage's models against independent references, as a report.

Each check (suite.check) takes one quantity a model computes and compares it with a reference that does not share
its code: a closed form written out again in the check, another library's implementation of the same model (OpenCV's
pinhole and fisheye projections, scipy's root finders and rotations), a published table or worked example (ITU-R
P.838-3 Table 5, ITU-R P.1814, UN R112's test points, Meeus's and the SPA paper's examples), a measurement (e-con's
stated fields of view, Emilsson's painted bumper plate), brute-force ray casting or Monte Carlo simulation with the
machinery that solves real captures. It reports the error figure, the tolerance and pass or fail; a check that raises
fails with the exception named.

    from vantage.validation import run, export
    res = run()                         # the sample project; run(project) for a project's vehicle and exports
    export(res, None, 'validation/')    # validation.json, validation.csv, validation.html, figures/*.svg

The project-dependent checks (occlusion against exact ray casting, the dataset calibration files) look at the
project given (a copy; it is never changed), else the built-in sample; the others build their own references. Runs
are reproducible: every random draw comes from the seed (ValidationSettings.seed) and the check's id.

Limits: the suite checks the models against references for the cases it builds; it does not show that a model
describes a real vehicle, lens or sensor (that needs measurements of them: the as-built and drive-log checks). Where a
model is a documented approximation, its check's tolerance and note say how far it is from the exact answer.
"""
from __future__ import annotations

import json
import os

from .references import REFERENCES, cite
from .suite import (SCHEMA, MODELS, MODEL_IDS, ValidationSettings, ValidationResult, Check, check, specs, run,
                    environment, available_models)
from . import report as _report

FRAME = 'vehicle (ISO 8855, rear axle): x forward, y left, z up, metres; each check states its error\'s unit'
LIMITS_TEXT = ('The suite checks the models against references for the cases it builds; it does not show that a '
               'model describes a real vehicle, lens or sensor. Monte Carlo checks are statistical: their tolerance '
               'is 4.5 standard errors of the estimated spread, at a fixed seed.')

__all__ = ['SCHEMA', 'MODELS', 'MODEL_IDS', 'available_models', 'ValidationSettings', 'ValidationResult', 'Check',
           'check', 'specs',
           'run', 'summary', 'to_doc', 'files', 'export', 'catalogue', 'REFERENCES', 'cite', 'exit_code']


def catalogue(settings: ValidationSettings | None = None) -> list:
    """Every check the settings select, without running them: [{'id', 'model', 'title', 'quantity', 'reference',
    'tolerance', 'unit', 'references', 'cost'}] (for the documentation and the assistant)."""
    return [{'id': s.id, 'model': s.model, 'title': s.title, 'quantity': s.quantity, 'reference': s.reference,
             'tolerance': s.tolerance, 'unit': s.unit, 'references': list(s.refs), 'cost': s.cost}
            for s in specs(settings)]


def summary(res: ValidationResult) -> dict:
    """Compact figures (the assistant, the batch log): counts, the failures and the run's identity."""
    models = {}
    for mid, title, _, _, cs in res.by_model():
        models[mid] = f'{sum(c.passed for c in cs)}/{len(cs)}'
    return {'schema': SCHEMA, 'vantage': res.environment.get('vantage'), 'checks': len(res.checks),
            'passed': res.passed, 'failed': [{'id': c.id, 'error': _report.clean(c.error), 'tolerance': c.tolerance,
                                              'unit': c.unit, 'exception': c.exception} for c in res.failed],
            'ok': res.ok, 'models': models, 'seconds': round(res.seconds, 1), 'seed': res.settings.seed,
            'quick': res.settings.quick, 'cancelled': res.cancelled}


def to_doc(res: ValidationResult, project=None) -> dict:
    """The JSON export: every check with its quantity, reference, error, tolerance, verdict, the numbers behind it and
    its figure's data, grouped by model, with the references cited and the environment."""
    used = []
    models = []
    for mid, title, mods, anchor, cs in res.by_model():
        for c in cs:
            used += [r for r in c.refs if r not in used]
        models.append({'id': mid, 'title': title, 'modules': mods, 'methods_anchor': anchor,
                       'passed': sum(c.passed for c in cs), 'checks': [_report.check_doc(c) for c in cs]})
    return _report.clean({
        'schema': SCHEMA, 'project': res.project, 'variant': res.variant, 'created_utc': res.started_utc,
        'seconds': round(res.seconds, 2), 'settings': res.settings.to_dict(), 'environment': res.environment,
        'frame': FRAME, 'limits': LIMITS_TEXT, 'cancelled': res.cancelled,
        'summary': {'checks': len(res.checks), 'passed': res.passed, 'failed': [c.id for c in res.failed],
                    'ok': res.ok},
        'models': models, 'references': {k: t for k, t in cite(used)}})


def files(res: ValidationResult, project=None) -> list:
    """[(name, bytes)]: validation.json, validation.csv, validation.html and figures/<check id>.svg."""
    out = [('validation.json', (json.dumps(to_doc(res, project), indent=1, ensure_ascii=False, allow_nan=False)
                                + '\n').encode('utf-8')),
           ('validation.csv', _report.to_csv(res).encode('utf-8')),
           ('validation.html', _report.to_html(res, project).encode('utf-8'))]
    for c in res.checks:
        if c.figure:
            out.append((f'figures/{c.id}.svg', _report.svg_figure(c.figure, standalone=True).encode('utf-8')))
    return out


def export(res: ValidationResult, project, out_dir) -> list[str]:
    """Write the report files into out_dir; returns their paths."""
    paths = []
    for name, data in files(res, project):
        path = os.path.join(out_dir, *name.split('/'))
        os.makedirs(os.path.dirname(path), exist_ok=True)
        tmp = path + '.part'
        with open(tmp, 'wb') as fh:
            fh.write(data)
        os.replace(tmp, path)
        paths.append(path)
    return paths


def exit_code(res: ValidationResult) -> int:
    """0 when every check passed, 6 when one failed (or the run was cancelled or ran no check)."""
    return 0 if res.ok else 6
