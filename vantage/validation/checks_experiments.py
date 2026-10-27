"""Checks of the experiments' estimators (core.experiments): Sobol indices on the Ishigami function's analytic indices,
Morris elementary effects on a linear function, Wilson's interval against scipy, and Monte Carlo uncertainty on a
linear function of normal inputs."""
from __future__ import annotations

import math

import numpy as np

from .suite import Outcome, check, figure, series

ISHIGAMI_A, ISHIGAMI_B = 7.0, 0.1


def ishigami(X, a=ISHIGAMI_A, b=ISHIGAMI_B):
    return np.sin(X[:, 0]) + a * np.sin(X[:, 1]) ** 2 + b * X[:, 2] ** 4 * np.sin(X[:, 0])


def ishigami_indices(a=ISHIGAMI_A, b=ISHIGAMI_B):
    """The analytic first-order and total Sobol indices of the Ishigami function on [−π, π]³ (Ishigami and Homma;
    Sobol' and Levitan): V = a²/8 + bπ⁴/5 + b²π⁸/18 + 1/2, V1 = (1 + bπ⁴/5)²/2, V2 = a²/8, V13 = 8b²π⁸/225."""
    pi = math.pi
    V = a * a / 8 + b * pi ** 4 / 5 + b * b * pi ** 8 / 18 + 0.5
    V1 = 0.5 * (1 + b * pi ** 4 / 5) ** 2
    V2 = a * a / 8
    V13 = 8 * b * b * pi ** 8 / 225
    return np.array([V1, V2, 0.0]) / V, np.array([V1 + V13, V2, V13]) / V


@check('experiments.sobol_ishigami', 'experiments', 'Sobol indices of the Ishigami function',
       'largest difference of the first-order (Saltelli et al. 2010) and total (Jansen 1999) indices '
       'experiments.sobol_indices estimates from a scrambled Sobol sequence (8192 base samples, A, B and AB_i '
       'matrices) from the Ishigami function\'s analytic indices (a = 7, b = 0.1); every analytic index must lie in '
       'its 95 % bootstrap interval (a margin of 0.01 allowed for the bootstrap\'s own spread)',
       'the analytic Sobol indices of the Ishigami function', 0.02, 'index',
       refs=('sobol2001', 'saltelli2010', 'jansen1999', 'ishigami1990', 'efron1993'), cost='study')
def _sobol(ctx):
    from scipy.stats import qmc
    from ..core import experiments as EX
    k = 3
    N = 2048 if ctx.quick else 8192
    U = qmc.Sobol(2 * k, scramble=True, seed=int(ctx.settings.seed) % (2 ** 31)).random(N)
    lo, hi = -math.pi, math.pi
    A, B = lo + (hi - lo) * U[:, :k], lo + (hi - lo) * U[:, k:]
    fA, fB = ishigami(A), ishigami(B)
    fAB = []
    for i in range(k):
        AB = A.copy()
        AB[:, i] = B[:, i]
        fAB.append(ishigami(AB))
    out = EX.sobol_indices(fA, fB, np.array(fAB), bootstrap=200, seed=int(ctx.settings.seed))
    S1, ST = ishigami_indices()
    e1 = np.abs(np.array(out['S1']) - S1)
    eT = np.abs(np.array(out['ST']) - ST)
    inside = all(lo_ - 0.01 <= s <= hi_ + 0.01 for (lo_, hi_), s in zip(out['S1_ci'] + out['ST_ci'],
                                                                        list(S1) + list(ST)))
    fig = figure('Sobol indices of the Ishigami function', 'factor x1, x2, x3', 'index',
                 [series('analytic S1', [1, 2, 3], S1), series('estimated S1', [1, 2, 3], out['S1'], 'points'),
                  series('analytic ST', [1, 2, 3], ST), series('estimated ST', [1, 2, 3], out['ST'], 'points')])
    return Outcome(float(max(e1.max(), eT.max())), {'N': N, 'S1': out['S1'], 'ST': out['ST'], 'S1_ci': out['S1_ci'],
                                                    'ST_ci': out['ST_ci'], 'analytic_S1': S1.round(5).tolist(),
                                                    'analytic_ST': ST.round(5).tolist()}, fig, ok=inside)


@check('experiments.morris_linear', 'experiments', 'Morris elementary effects of a linear function',
       'largest difference of the μ, μ* and σ experiments.morris_indices gives for y = Σ c_i x_i on the unit cube '
       '(20 random trajectories on a 4-level grid) from c_i, |c_i| and 0',
       'the elementary effects of a linear function, which are its coefficients exactly', 1e-9,
       'response per range', refs=('morris1991', 'campolongo2007'))
def _morris(ctx):
    from ..core import experiments as EX
    rng = ctx.rng('experiments.morris_linear')
    c = np.array([3.0, -2.0, 0.5, 0.0, 7.5])
    k, p, r = len(c), 4, 20
    delta = p / (2 * (p - 1))
    U = []
    for _ in range(r):
        x = rng.integers(0, p // 2, k) / (p - 1)
        rows = [x.copy()]
        for j in rng.permutation(k):
            x = x.copy()
            x[j] = x[j] + delta if x[j] + delta <= 1 + 1e-12 else x[j] - delta
            rows.append(x)
        U += rows
    U = np.array(U)
    out = EX.morris_indices(U, U @ c, k, bootstrap=0)
    e = max(np.abs(np.array(out['mu']) - c).max(), np.abs(np.array(out['mu_star']) - np.abs(c)).max(),
            np.abs(np.array(out['sigma'])).max())
    return Outcome(float(e), {'coefficients': c.tolist(), 'mu': out['mu'], 'mu_star': out['mu_star'],
                              'sigma': out['sigma'], 'trajectories': out['r']})


@check('experiments.wilson', 'experiments', 'Wilson\'s interval for a share',
       'largest difference of experiments.wilson\'s 95 % and 90 % intervals for 0/10, 3/10, 17/40, 99/100 and '
       '500/1000 from scipy.stats.binomtest\'s Wilson interval',
       'scipy.stats.binomtest(k, n).proportion_ci(method=\'wilson\')', 1e-9, 'share', refs=('wilson1927',))
def _wilson(ctx):
    from scipy.stats import binomtest
    from ..core import experiments as EX
    e = 0.0
    for k, n in ((0, 10), (3, 10), (17, 40), (99, 100), (500, 1000)):
        for conf in (0.95, 0.90):
            lo, hi = EX.wilson(k, n, conf)
            ci = binomtest(k, n).proportion_ci(confidence_level=conf, method='wilson')
            e = max(e, abs(lo - ci.low), abs(hi - ci.high))
    return Outcome(e, {'cases': 10})


@check('experiments.uncertainty_linear', 'experiments', 'Uncertainty propagated through a linear function',
       'relative difference of the standard deviation experiments.uncertainty gives for y = 2·x1 + 3·x2 − 1 with '
       'x1, x2 standard normal (20 000 draws) from the analytic √13, and of its mean from −1 in units of that σ; '
       'the analytic σ must lie in its bootstrap interval; tolerance 4.5 standard errors of a σ estimated from the '
       'draws',
       'the variance of a linear combination of independent normals, written out in the check', 0.0225, 'relative',
       refs=('efron1993', 'saltelli2010'))
def _uncert(ctx):
    from ..core import experiments as EX
    n = 5000 if ctx.quick else 20000
    x = ctx.rng('experiments.uncertainty_linear').normal(size=(n, 2))
    y = 2 * x[:, 0] + 3 * x[:, 1] - 1
    out = EX.uncertainty(y, bootstrap=300, seed=int(ctx.settings.seed))
    sd = math.sqrt(13.0)
    e = abs(out['sd'] - sd) / sd
    em = abs(out['mean'] + 1.0) / sd
    lo, hi = out['sd_ci']
    conv = out['convergence']
    fig = figure('Running mean and its interval against the draws', 'draws', 'mean of y',
                 [series('running mean', [c['n'] for c in conv], [c['mean'] for c in conv]),
                  series('mean + half interval', [c['n'] for c in conv],
                         [c['mean'] + (c['half'] or 0.0) for c in conv]),
                  series('mean − half interval', [c['n'] for c in conv],
                         [c['mean'] - (c['half'] or 0.0) for c in conv])], hline=-1.0, hline_label='−1 (analytic)')
    return Outcome(max(e, em), {'n': n, 'sd': out['sd'], 'sd_ci': out['sd_ci'], 'mean': out['mean'],
                                'analytic_sd': sd}, fig, ok=lo <= sd <= hi,
                   tolerance=4.5 / math.sqrt(2 * (n - 1)))
