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


# ---- cut-drop / chunked rows / domain shrink (reach-ladder engine upgrades) ----------------


@pytest.mark.unit
@pytest.mark.parametrize("m", [30, 60], ids=["m30", "m60"])
def test_drop_slack_matches_solve_support(m: int):
    """drop_slack=True returns the SAME optimum as the trusted cold oracle (thin seed forces
    many cut rounds so rows actually churn)."""
    ref = solve_support(range(1, m + 1))
    lp = WarmLP(list(range(2, m + 1)))
    res = lp.row_generate_to_feasible(seed_rows=5, cuts_per_iter=10, drop_slack=True)
    assert res.success == ref.success
    assert res.S == pytest.approx(ref.S, abs=1e-9)
    assert res.grid_max == pytest.approx(ref.grid_max, abs=1e-9)


@pytest.mark.unit
def test_drop_slack_on_sparse_squarefree_support():
    """Same equivalence on a non-contiguous support (the multiscale-family shape)."""
    K = [k for k in range(2, 80) if _is_squarefree(k)]
    ref = solve_support(K)
    lp = WarmLP(K)
    res = lp.row_generate_to_feasible(seed_rows=8, cuts_per_iter=12, drop_slack=True)
    assert res.S == pytest.approx(ref.S, abs=1e-9)
    assert res.grid_max == pytest.approx(ref.grid_max, abs=1e-9)


@pytest.mark.unit
def test_drop_slack_reduces_final_rows():
    """With slack rows deleted the working LP ends smaller than the keep-everything run."""
    keep = WarmLP(list(range(2, 121)))
    keep.row_generate_to_feasible(seed_rows=5, cuts_per_iter=15)
    drop = WarmLP(list(range(2, 121)))
    drop.row_generate_to_feasible(seed_rows=5, cuts_per_iter=15, drop_slack=True)
    assert len(drop.row_x) < len(keep.row_x)


@pytest.mark.unit
def test_dropped_row_reenters_pinned():
    """Bookkeeping: a slack-dropped x that re-enters is pinned and never dropped again."""
    lp = WarmLP(list(range(2, 31)))
    lp.row_generate_to_feasible()
    # force-drop every currently-slack row regardless of age
    n_dropped = lp.drop_slack_rows(slack_tol=1e-9, min_age=0)
    assert n_dropped > 0
    assert len(lp.row_x) == lp.h.getNumRow()
    x_back = next(iter(lp._dropped_once))
    lp.add_x_rows([x_back])
    assert x_back in lp._pinned
    assert x_back in lp.row_x
    lp.h.run()
    lp.drop_slack_rows(slack_tol=1e-9, min_age=0)
    assert x_back in lp.row_x  # pinned -> survives even though it is slack


@pytest.mark.unit
def test_alignment_invariants_after_drop_and_add_key():
    """After a drop_slack run the ledgers stay aligned and warm column edits still behave."""
    lp = WarmLP([k for k in range(2, 41) if _is_squarefree(k)])
    s0 = lp.row_generate_to_feasible(seed_rows=6, cuts_per_iter=8, drop_slack=True).S
    assert len(lp.row_x) == lp.h.getNumRow()
    assert len(lp.row_round) == len(lp.row_x)
    assert len(set(lp.row_x)) == len(lp.row_x)
    assert len(lp.kplus) == lp.h.getNumCol()
    row_x, row_dual = lp.duals()
    assert row_x.shape == row_dual.shape
    lp.add_key(4)
    s1 = lp.row_generate_to_feasible(drop_slack=True).S
    assert s1 >= s0 - 1e-9


@pytest.mark.unit
@pytest.mark.parametrize("window", [2, 7], ids=["w2", "w7"])
def test_windowed_cuts_match_solve_support(window: int):
    """Stratified cut selection changes wall time only — the certified S is identical."""
    ref = solve_support(range(1, 121))
    lp = WarmLP(list(range(2, 121)))
    # windowed cuts add fewer rows per round -> needs more (cheap) iterations to converge
    res = lp.row_generate_to_feasible(
        seed_rows=5, cuts_per_iter=10, drop_slack=True, cut_window=window, max_iters=500
    )
    assert res.success == ref.success
    assert res.S == pytest.approx(ref.S, abs=1e-9)
    assert res.grid_max == pytest.approx(ref.grid_max, abs=1e-9)


@pytest.mark.unit
def test_select_cuts_keeps_worst_per_window():
    """Bucketing keeps the max-violation x per window, in descending-violation order."""
    lp = WarmLP([2, 3])
    # violations sorted by descending violation: 100 (worst), 102, 5, 7, 200
    viol = np.asarray([100, 102, 5, 7, 200], dtype=np.int64)
    picked = lp._select_cuts(viol, cuts_per_iter=10, cut_window=10)
    # buckets of width 10: {10: first hit 100}, {0: first hit 5}, {20: 200} — 102 and 7 shadowed
    assert picked == [100, 5, 200]
    # window off -> global order preserved
    assert lp._select_cuts(viol, 10, None) == [100, 102, 5, 7, 200]
    # already-added x's are excluded before bucketing
    lp._added.update({100})
    assert lp._select_cuts(viol, 10, 10)[0] == 102


@pytest.mark.unit
def test_ledger_drift_raises_typed_error():
    """A ledger/model row mismatch must raise LedgerDriftError (survives python -O),
    which the sweep driver catches for a fresh-model retry."""
    from arena.pnt_warm import LedgerDriftError

    lp = WarmLP(list(range(2, 31)))
    lp.row_generate_to_feasible()
    lp.row_x.append(999_999)  # corrupt the ledger on purpose
    lp.row_round.append(0)
    with pytest.raises(LedgerDriftError):
        lp.drop_slack_rows(slack_tol=1e-9, min_age=0)


@pytest.mark.unit
def test_chunked_add_rows_equivalence(monkeypatch):
    """Chunk size must not affect the model: force tiny blocks and reproduce the oracle."""
    import arena.pnt_warm as warm_mod

    monkeypatch.setattr(warm_mod, "_ADD_ROW_CHUNK", 3)
    ref = solve_support(range(1, 31))
    res = WarmLP(list(range(2, 31))).row_generate_to_feasible()
    assert res.S == pytest.approx(ref.S, abs=1e-12)
    assert res.grid_max == pytest.approx(ref.grid_max, abs=1e-12)


@pytest.mark.unit
def test_drop_rows_above_matches_cold_after_domain_shrink():
    """Dropping the far key + its out-of-domain rows returns to the small-support optimum."""
    base = [2, 3, 5, 7]
    lp = WarmLP(base + [200])
    lp.row_generate_to_feasible()
    lp.drop_keys([200])
    n = lp.drop_rows_above(lp.current_x_max())
    assert n > 0
    res = lp.row_generate_to_feasible(lp.current_x_max())
    ref = solve_support(base)
    assert res.S == pytest.approx(ref.S, abs=1e-7)
    assert len(lp.row_x) == lp.h.getNumRow()
