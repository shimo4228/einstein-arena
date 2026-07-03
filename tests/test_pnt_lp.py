"""PNT LP layer tests: monotone feasible truncations, f(1) substitution, objective, hand optimum."""

from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from arena.pnt_lp import (  # noqa: E402
    RHS,
    grid_g,
    grid_g_fast,
    objective_vector,
    reconstruct_f1,
    solve_support,
)


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
def test_grid_g_fast_matches_grid_g():
    """The divisor-sum + cumsum scan reproduces the exact dense scan (well under LP tolerances)."""
    rng = np.random.default_rng(42)
    for _ in range(5):
        keys = np.unique(rng.integers(1, 400, size=60).astype(np.int64))
        if keys[0] != 1:
            keys = np.concatenate(([1], keys))
        vals = rng.uniform(-10, 10, size=len(keys))
        x_max = int(10 * keys.max())
        fast = grid_g_fast(keys, vals, x_max)
        exact = grid_g(keys, vals, x_max)
        assert fast.shape == exact.shape
        # Unconstrained random f lets |g| grow to ~1e4, so cumsum rounding scales with the
        # running-sum magnitude -> compare with a relative term. In the LP's regime (g <= ~1)
        # the absolute error is ~1e-12; the design never certifies with the fast scan anyway.
        assert np.allclose(fast, exact, rtol=1e-12, atol=1e-9)
        # Realistic magnitude: scale values so g stays O(1); absolute agreement is then tight.
        fast_s = grid_g_fast(keys, vals / 1000.0, x_max)
        exact_s = grid_g(keys, vals / 1000.0, x_max)
        assert float(np.max(np.abs(fast_s - exact_s))) < 1e-10


@pytest.mark.unit
def test_grid_g_fast_ignores_keys_beyond_domain():
    """Keys with k >= x_max contribute floor(x/k)=0 everywhere; both scans agree."""
    keys = np.asarray([1, 2, 3, 500], dtype=np.int64)
    vals = np.asarray([0.5, -1.0, 0.25, 3.0], dtype=np.float64)
    fast = grid_g_fast(keys, vals, 20)
    exact = grid_g(keys, vals, 20)
    assert np.allclose(fast, exact, atol=1e-12)


@pytest.mark.unit
def test_objective_matches_manual_score():
    """S equals -sum_{k>=2} f(k) log(k)/k computed directly from the returned f."""
    res = solve_support(range(1, 30 + 1))
    manual = -sum(v * math.log(k) / k for k, v in res.f.items() if k != 1)
    assert res.S == pytest.approx(manual, abs=1e-12)
    assert not np.isnan(res.S)
