"""Phase B tests: verifier fidelity, feasibility projection, and jax-gradient correctness."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from arena.verifier import load_evaluate  # noqa: E402

evaluate = load_evaluate("erdos-min-overlap")
ARENA_TOP = 0.3808703105862199


@pytest.mark.unit
def test_seed_reproduces_arena_top():
    """The saved arena rank-1 seed scored by the server verifier == its reported score."""
    seed = json.loads((REPO_ROOT / "results" / "erdos-min-overlap" / "seed_arena_rank1.json").read_text())
    assert evaluate({"values": seed["values"]}) == pytest.approx(ARENA_TOP, abs=1e-12)


@pytest.mark.unit
def test_box_violation_raises():
    with pytest.raises(AssertionError):
        evaluate({"values": [1.5] * 10})  # > 1


@pytest.mark.unit
def test_projection_is_feasible():
    from scripts.solve_erdos_min_overlap import project

    rng = np.random.default_rng(3)
    h = rng.normal(0.5, 0.4, size=200)
    p = project(h)
    assert p.min() >= -1e-12 and p.max() <= 1 + 1e-12
    assert p.sum() == pytest.approx(len(p) / 2.0, abs=1e-6)


@pytest.mark.unit
def test_jax_gradient_matches_finite_difference():
    """The jax smooth-max gradient must match central finite differences (guards the solver)."""
    from scripts.solve_erdos_min_overlap import _make_grad

    rng = np.random.default_rng(7)
    n, beta = 40, 50.0
    h = rng.uniform(0.2, 0.8, size=n)
    vg = _make_grad(n, beta)
    _, g = vg(h)
    g = np.asarray(g)

    import jax.numpy as jnp
    from jax.scipy.special import logsumexp

    def f(x):
        c = jnp.correlate(x, 1.0 - x, mode="full")
        return float((2.0 / n) * (logsumexp(beta * c) / beta))

    eps = 1e-6
    fd = np.zeros(n)
    for i in range(n):
        hp, hm = h.copy(), h.copy()
        hp[i] += eps
        hm[i] -= eps
        fd[i] = (f(hp) - f(hm)) / (2 * eps)
    assert np.allclose(g, fd, atol=1e-5)
