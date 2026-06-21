"""Column-generation engine tests: grow a support by dual-priced keys, accept real LP gains only."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from arena.pnt_lp import RHS, solve_support  # noqa: E402
from arena.pnt_colgen import ColgenConfig, colgen_support  # noqa: E402
from arena.pnt_seeds import candidate_pool, squarefree_first_n  # noqa: E402


@pytest.mark.unit
def test_colgen_grows_and_beats_seed():
    """Growing a small seed with dual-priced keys raises honest S above the seed's optimum."""
    seed = squarefree_first_n(40)
    seed_S = solve_support(seed).S
    res = colgen_support(seed, candidate_pool(600), ColgenConfig(cap=120, cols_in_per_iter=20))
    assert res.S > seed_S
    assert len([k for k in res.support if k != 1]) <= 120
    assert res.success


@pytest.mark.unit
def test_colgen_final_support_is_a_real_lp_optimum():
    """The returned S/grid_max equal an independent solve_support on the final support (no drift)."""
    res = colgen_support(
        [2, 3, 5, 7], candidate_pool(400), ColgenConfig(cap=40, cols_in_per_iter=10)
    )
    ref = solve_support(res.support)
    assert res.S == pytest.approx(ref.S, abs=1e-7)
    assert res.grid_max <= RHS + 1e-9


@pytest.mark.unit
def test_colgen_history_is_monotone_nondecreasing():
    res = colgen_support(
        squarefree_first_n(30), candidate_pool(400), ColgenConfig(cap=80, cols_in_per_iter=15)
    )
    h = res.history
    assert all(h[i] <= h[i + 1] + 1e-12 for i in range(len(h) - 1))


@pytest.mark.unit
def test_colgen_is_deterministic():
    seed = squarefree_first_n(30)
    pool = candidate_pool(500)
    cfg = ColgenConfig(cap=90, cols_in_per_iter=12)
    r1 = colgen_support(seed, pool, cfg)
    r2 = colgen_support(seed, pool, cfg)
    assert r1.support == r2.support
    assert r1.S == r2.S
    assert r1.history == r2.history


@pytest.mark.unit
def test_colgen_respects_cap_and_reports_stop_reason():
    res = colgen_support(
        squarefree_first_n(20), candidate_pool(800), ColgenConfig(cap=50, cols_in_per_iter=10)
    )
    assert len([k for k in res.support if k != 1]) <= 50
    assert res.stop_reason in {
        "budget_full",
        "no_improvement",
        "pool_exhausted",
        "dual_optimal",
        "max_outer_iters",
    }


@pytest.mark.unit
def test_colgen_budget_resolves_cap_triggers_stop():
    """A tight resolve budget stops the loop early with the matching stop_reason."""
    res = colgen_support(
        squarefree_first_n(20),
        candidate_pool(800),
        ColgenConfig(cap=400, cols_in_per_iter=5, budget_resolves=3),
    )
    assert res.n_resolves <= 3
    assert res.stop_reason == "budget_resolves"
