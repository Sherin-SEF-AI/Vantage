"""The calibration coach (core.calib_coach, ui.live_calib_coach) uses only public helpers of the observability and
lidar–camera calibration modules: no `module._private` calls into them, and the helpers it shares have public names
with docstrings (the two old private names kept for an outside script are the same functions)."""
import re
from pathlib import Path

from vantage.core import lidar_calib as LC, observability as OB

ROOT = Path(__file__).resolve().parents[1] / 'vantage'
SHARED = {OB: ('project_for', 'finite_pose', 'lidar_model'),
          LC: ('world_to_optical', 'pose_sigmas', 'board_pnp', 'paint_board')}


def test_the_coach_calls_no_private_helpers_of_other_modules():
    for rel in ('core/calib_coach.py', 'ui/live_calib_coach.py'):
        src = (ROOT / rel).read_text(encoding='utf-8')
        hits = re.findall(r'\b(?:OB|LC|observability|lidar_calib|CC|rh)\._[a-z]\w*', src)
        assert not hits, (rel, hits)


def test_shared_helpers_are_public_and_documented():
    for mod, names in SHARED.items():
        for n in names:
            f = getattr(mod, n)
            assert callable(f) and f.__name__ == n and len((f.__doc__ or '').strip()) > 40, (mod.__name__, n)
            if (mod, n) != (LC, 'board_pnp'):
                assert not hasattr(mod, '_' + n), (mod.__name__, n)                  # no private twin left behind
    assert LC._w2o is LC.world_to_optical and LC._board_pnp is LC.board_pnp
