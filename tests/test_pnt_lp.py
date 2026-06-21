"""PNT LP layer tests: monotone feasible truncations, f(1) substitution, objective, hand optimum."""

from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from arena.pnt_lp import RHS, objective_vector, reconstruct_f1, solve_support  # noqa: E402


@pytest.mark.unit
@pytest.mark.parametrize("m", [10, 30, 60, 120], ids=["m10", "m30", "m60", "m120"])
def test_truncated_support_is_grid_feasible(m: int):
    """The LP optimum on {1..m} is exactly feasible on the honest integer grid (g(x) <= 1)."""
    res = solve_support(range(1, m + 1))
    assert res.success
    assert res.grid_max <= RHS + 1e-9
    assert 0.0 < res.S < 1.0  # below the theoretical Mobius maximum of 1


@pytest.mark.unit
def test_score_increases_with_support():
    """Larger truncations cannot do worse: S is non-decreasing in m (superset of variables)."""
    scores = [solve_support(range(1, m + 1)).S for m in (30, 60, 120)]
    assert scores[0] < scores[1] < scores[2]


@pytest.mark.unit
def test_f1_substitution_enforces_equality():
    """The reconstructed f(1) makes sum_k f(k)/k == 0, exactly as the verifier patch does."""
    res = solve_support(range(1, 60 + 1))
    total = sum(v / k for k, v in res.f.items())
    assert total == pytest.approx(0.0, abs=1e-9)
    fplus = {k: v for k, v in res.f.items() if k != 1}
    assert res.f[1] == pytest.approx(reconstruct_f1(fplus), abs=1e-12)


@pytest.mark.unit
def test_objective_vector_excludes_k1():
    """log(1)=0, so the cost vector is defined only over k>=2 with c[j]=log(k)/k."""
    c = objective_vector([2, 3, 5])
    assert len(c) == 3
    assert c == pytest.approx([math.log(2) / 2, math.log(3) / 3, math.log(5) / 5])


@pytest.mark.unit
def test_tiny_support_matches_hand_optimum():
    """On K={1,2}: constraint forces f(2) >= -2, so S* = -log(2)/2 * (-2) = log(2).

    With f(2)=-2, f(1)=1 and g(x) = x - 2*floor(x/2) = x mod 2 in {0,1} -> exactly feasible,
    and the period (2) is covered by the grid.
    """
    res = solve_support([1, 2])
    assert res.S == pytest.approx(math.log(2), abs=1e-9)
    assert res.f[2] == pytest.approx(-2.0, abs=1e-9)
    assert res.f[1] == pytest.approx(1.0, abs=1e-9)
    assert res.grid_max <= RHS + 1e-9


@pytest.mark.unit
def test_support_always_includes_key_one():
    """Key 1 is materialized even if omitted from the input support."""
    res = solve_support([2, 3, 5])
    assert 1 in res.f
    assert set(res.f.keys()) == {1, 2, 3, 5}


@pytest.mark.unit
def test_objective_matches_manual_score():
    """S equals -sum_{k>=2} f(k) log(k)/k computed directly from the returned f."""
    res = solve_support(range(1, 30 + 1))
    manual = -sum(v * math.log(k) / k for k, v in res.f.items() if k != 1)
    assert res.S == pytest.approx(manual, abs=1e-12)
    assert not np.isnan(res.S)
