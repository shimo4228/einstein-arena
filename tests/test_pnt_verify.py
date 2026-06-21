"""PNT verification tests: server fidelity, exact grid check, classical identity, breakpoints."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from arena.pnt_lp import grid_g, solve_support  # noqa: E402
from arena.pnt_verify import dual_verify, server_score  # noqa: E402
from scripts.solve_prime_number_theorem import mobius_upto  # noqa: E402


@pytest.mark.unit
def test_grid_rejects_infeasible_candidate():
    """f(2)=-10 on K={1,2} gives g(x)=5 on odd x>1: the grid must flag it, no server call needed."""
    rep = dual_verify({1: 5.0, 2: -10.0}, [1, 2], s_lp=0.0)
    assert not rep.grid_feasible
    assert rep.n_violations > 0
    assert not rep.submittable


@pytest.mark.unit
def test_grid_accepts_lp_optimum():
    """The LP optimum is grid-feasible (excludes the server call; that is the integration test)."""
    res = solve_support([1, 2])
    rep = dual_verify(res.f, [1, 2], res.S)
    assert rep.grid_feasible
    assert rep.grid_max <= 1.0 + 1e-9


@pytest.mark.unit
def test_mobius_identity_on_grid():
    """Classical identity sum_{k<=x} mu(k) floor(x/k) = 1 holds for pure Mobius up to x=M."""
    m = 200
    mu = mobius_upto(m)
    keys = np.arange(1, m + 1, dtype=np.int64)
    values = mu[1:].astype(np.float64)
    g = grid_g(keys, values, m + 1)  # x = 1..m
    assert np.allclose(g, 1.0, atol=1e-9)


@pytest.mark.unit
def test_noninteger_x_never_exceeds_integer_grid_max():
    """Breakpoint argument: g is constant on [n, n+1), so the integer grid captures the sup."""
    res = solve_support(range(1, 60 + 1))
    keys = np.fromiter(res.f.keys(), dtype=np.int64)
    values = np.fromiter(res.f.values(), dtype=np.float64)
    m = int(keys.max())
    g_int = grid_g(keys, values, 10 * m)
    grid_max = float(g_int.max())

    rng = np.random.default_rng(0)
    xs = rng.uniform(1.0, 10 * m, size=5000)
    floors = np.floor(xs[:, None] / keys[None, :])
    g_cont = floors @ values
    assert g_cont.max() <= grid_max + 1e-9


@pytest.mark.integration
def test_local_score_equals_server_score():
    """CLAUDE.md fidelity invariant: the verbatim server verifier reproduces the LP's S."""
    res = solve_support(range(1, 60 + 1))
    rep = dual_verify(res.f, range(1, 60 + 1), res.S)
    assert rep.server_finite
    assert rep.score_matches
    assert rep.s_server == pytest.approx(res.S, abs=1e-7)
    assert rep.submittable


@pytest.mark.integration
def test_server_rejects_infeasible_candidate():
    """The server verifier returns -inf for an honestly-infeasible candidate (g exceeds 1)."""
    assert server_score({1: 5.0, 2: -10.0}) == float("-inf")
