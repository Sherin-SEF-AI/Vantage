"""The downloads leave PyTorch out (packaging/vantage.spec): gradient refinement and log replay then say so and how to
get it, in the core, the batch and on their pages, instead of failing with an ImportError."""
import builtins
import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(__file__))


@pytest.fixture
def no_torch(monkeypatch):
    real = builtins.__import__

    def fake(name, *a, **k):
        if name == 'torch' or name.startswith('torch.'):
            raise ImportError('No module named torch')
        return real(name, *a, **k)
    monkeypatch.setattr(builtins, '__import__', fake)
    for m in [m for m in sys.modules if m == 'torch' or m.startswith('torch.')]:
        monkeypatch.delitem(sys.modules, m)


def test_messages_name_the_feature_and_the_fix(no_torch):
    from vantage.core import diff_optim as DO, splat_reproject as SP
    for why, what in ((DO.torch_missing(), 'Gradient refinement'), (SP.torch_missing(), 'Log replay')):
        assert why.startswith(what + ' needs PyTorch') and 'pip install torch' in why


def test_with_torch_nothing_is_missing():
    pytest.importorskip('torch')
    from vantage.core import diff_optim as DO, splat_reproject as SP
    assert DO.torch_missing() is None and SP.torch_missing() is None


def test_export_formats_have_nothing_to_write(no_torch):
    """The bundles' self-test exports every format: without torch these two write nothing rather than fail."""
    from vantage.core import exporters as ex, samples
    p = samples.tigor_sample_project()
    assert ex._diff_optim_names(p, p.variant) == [] and ex._log_replay_names(p, p.variant) == []


def test_core_and_batch_refuse_cleanly(no_torch):
    from vantage.core import diff_optim as DO, samples, splat_reproject as SP
    p = samples.tigor_sample_project()
    with pytest.raises(ValueError, match='needs PyTorch'):
        DO.run(p)
    doc, rc = DO.batch(p)
    assert rc == 4 and 'needs PyTorch' in doc['error']
    with pytest.raises(ValueError, match='needs PyTorch'):
        SP.run(p)


from test_gui import app, win, pump  # noqa: E402,F401  (fixtures)


def test_pages_refuse_with_the_message(win, no_torch):
    from vantage.ui.workspaces.optimiser_ws import OptimiserPage
    from vantage.ui.workspaces.splat_ws import SplatPage
    msgs = []
    win.ctx.msg, real = (lambda m, *a: msgs.append(m)), win.ctx.msg
    try:
        pg = SplatPage(win.ctx)
        assert pg._start('build', 'x', lambda progress, cancel: None) is False
        assert 'needs PyTorch' in pg.status_text
        assert OptimiserPage(win.ctx).refiner.start_run() is False
    finally:
        win.ctx.msg = real
    assert [m.split(' needs')[0] for m in msgs] == ['Log replay', 'Gradient refinement']
