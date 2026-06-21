"""WarmLP tests: a persistent HiGHS model must reproduce solve_support and support live edits.

Correctness oracle = the trusted `solve_support` (cold cutting-plane). WarmLP solves the SAME LP
on a persistent model so it can read duals and add/drop columns between solves (for column gen).
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from arena.pnt_lp import RHS, solve_support  # noqa: E402
from arena.pnt_warm import WarmLP  # noqa: E402


def _is_squarefree(n: int) -> bool:
    d = 2
    while d * d <= n:
        if n % (d * d) == 0:
            return False
        d += 1
    return True


@pytest.mark.unit
@pytest.mark.parametrize("m", [10, 30, 60], ids=["m10", "m30", "m60"])
def test_warmlp_reproduces_solve_support(m: int):
    """WarmLP([2..m]).row_generate_to_feasible() == solve_support({1..m}) in S, grid_max, success."""
    ref = solve_support(range(1, m + 1))
    lp = WarmLP(list(range(2, m + 1)))
    res = lp.row_generate_to_feasible()
    assert res.success == ref.success
    assert res.S == pytest.approx(ref.S, abs=1e-9)
    assert res.grid_max == pytest.approx(ref.grid_max, abs=1e-9)


@pytest.mark.unit
def test_warmlp_reproduces_on_sparse_squarefree_support():
    """Reproduction also holds on a non-contiguous (squarefree) support."""
    K = [k for k in range(2, 50) if _is_squarefree(k)]
    ref = solve_support(K)
    res = WarmLP(K).row_generate_to_feasible()
    assert res.S == pytest.approx(ref.S, abs=1e-9)
    assert res.grid_max == pytest.approx(ref.grid_max, abs=1e-9)


@pytest.mark.unit
def test_duals_aligned_and_signed():
    """duals() returns aligned (row_x, row_dual); <= rows in a minimize have dual <= 0, some bind."""
    lp = WarmLP(list(range(2, 60 + 1)))
    lp.row_generate_to_feasible()
    row_x, row_dual = lp.duals()
    assert row_x.shape == row_dual.shape
    assert row_x.ndim == 1 and len(row_x) > 0
    assert len(set(row_x.tolist())) == len(row_x)  # no duplicate x rows
    assert np.all(row_dual <= 1e-9)
    assert np.any(row_dual < -1e-9)


@pytest.mark.unit
def test_objective_S_matches_colvalue():
    lp = WarmLP(list(range(2, 30 + 1)))
    res = lp.row_generate_to_feasible()
    assert lp.objective_S() == pytest.approx(res.S, abs=1e-12)


@pytest.mark.unit
def test_add_key_does_not_decrease_S():
    """Adding a column can only relax the optimum: S non-decreasing (LP monotonicity)."""
    lp = WarmLP([k for k in range(2, 41) if _is_squarefree(k)])
    s0 = lp.row_generate_to_feasible().S
    lp.add_key(4)  # non-squarefree, within current reach (no x_max extension)
    s1 = lp.row_generate_to_feasible().S
    assert s1 >= s0 - 1e-9
    assert 4 in lp.kplus


@pytest.mark.unit
def test_add_then_drop_within_reach_roundtrips_S():
    """Add a within-reach key then drop it -> back to the original optimum (no stale far rows)."""
    base = [k for k in range(2, 41) if _is_squarefree(k)]
    lp = WarmLP(base)
    s0 = lp.row_generate_to_feasible().S
    lp.add_key(4)
    lp.row_generate_to_feasible()
    lp.drop_keys([4])
    s2 = lp.row_generate_to_feasible().S
    assert 4 not in lp.kplus
    assert sorted(lp.kplus) == base
    assert s2 == pytest.approx(s0, abs=1e-7)


@pytest.mark.unit
def test_add_large_key_extends_reach_and_stays_feasible():
    """Adding a key beyond current reach grows x_max; re-solve stays grid-feasible at RHS 1.0."""
    lp = WarmLP([2, 3, 5, 7])
    lp.row_generate_to_feasible()
    lp.add_key(200)
    res = lp.row_generate_to_feasible()
    assert max(lp.kplus) == 200
    assert res.success
    assert res.grid_max <= RHS + 1e-9


@pytest.mark.unit
def test_add_large_key_raises_S_vs_small_support():
    """Extending reach with a far key should help (more reach -> higher S) on a tiny seed."""
    lp = WarmLP([2, 3, 5, 7])
    s_small = lp.row_generate_to_feasible().S
    for k in (11, 13, 30, 60, 120):
        lp.add_key(k)
    s_big = lp.row_generate_to_feasible().S
    assert s_big > s_small
