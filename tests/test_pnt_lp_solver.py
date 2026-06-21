"""Solver-equivalence tests: the warm-started HiGHS path matches the scipy reference.

Both backends delegate to HiGHS, so the optimum S must agree and both must be grid-feasible.
We assert on S and feasibility, not on f itself: the LP can be degenerate (multiple optimal f
with the same objective), so f need not be identical between backends.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from arena.pnt_lp import RHS, solve_support  # noqa: E402


@pytest.mark.unit
@pytest.mark.parametrize("m", [2, 60, 120], ids=["m2", "m60", "m120"])
def test_highs_matches_scipy(m: int):
    """Same optimum S and feasibility from solver='highs' and solver='scipy' on small supports."""
    rh = solve_support(range(1, m + 1), solver="highs")
    rs = solve_support(range(1, m + 1), solver="scipy")
    assert rh.success and rs.success
    assert rh.S == pytest.approx(rs.S, abs=1e-7)
    assert rh.grid_max <= RHS + 1e-9
    assert rs.grid_max <= RHS + 1e-9


@pytest.mark.unit
def test_highs_is_default():
    """The default solver path is the warm-started HiGHS one and returns a feasible optimum."""
    res = solve_support(range(1, 60 + 1))
    assert res.success
    assert 0.0 < res.S < 1.0
    assert res.grid_max <= RHS + 1e-9
