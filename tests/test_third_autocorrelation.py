"""Phase B (3rd autocorrelation) tests: verifier fidelity, FFT-objective match, jax gradient."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from arena.verifier import load_evaluate  # noqa: E402

evaluate = load_evaluate("third-autocorrelation-inequality")
ARENA_TOP = 1.4523043331831582


@pytest.mark.unit
def test_constant_function_scores_two():
    """f ≡ 1 (any n) has C = 2.0 — the trivial baseline; pins verifier fidelity."""
    assert evaluate({"values": [1.0] * 50}) == pytest.approx(2.0, abs=1e-9)


@pytest.mark.unit
def test_seed_reproduces_arena_top():
    seed = json.loads((REPO_ROOT / "results" / "third-autocorrelation-inequality" / "seed_arena_rank1.json").read_text())
    assert evaluate({"values": seed["values"]}) == pytest.approx(ARENA_TOP, abs=1e-9)


@pytest.mark.unit
def test_fft_objective_matches_verifier():
    """The FFT replica fast_C must match the server verifier (uses np.convolve)."""
    from scripts.solve_third_autocorrelation import fast_C

    rng = np.random.default_rng(2)
    f = rng.normal(1.0, 1.0, size=500)  # nonzero integral
    assert fast_C(f) == pytest.approx(evaluate({"values": f.tolist()}), rel=1e-9)


@pytest.mark.unit
def test_scale_invariance():
    rng = np.random.default_rng(5)
    f = rng.normal(1.0, 0.5, size=300)
    assert evaluate({"values": (3.7 * f).tolist()}) == pytest.approx(evaluate({"values": f.tolist()}), rel=1e-10)


@pytest.mark.unit
def test_jax_gradient_matches_finite_difference():
    from scripts.solve_third_autocorrelation import _make_grad

    rng = np.random.default_rng(11)
    n, beta = 60, 30.0
    f = rng.normal(1.0, 0.3, size=n)
    vg = _make_grad(n, beta)
    _, g = vg(f)
    g = np.asarray(g)

    import jax.numpy as jnp
    from jax.scipy.special import logsumexp

    dx, L = 0.5 / n, 2 * n - 1

    def obj(x):
        F = jnp.fft.rfft(x, n=L)
        conv = jnp.fft.irfft(F * F, n=L)
        return float((logsumexp(beta * (conv * dx)) / beta) / (jnp.sum(x) * dx) ** 2)

    eps = 1e-6
    fd = np.array([(obj(f + e) - obj(f - e)) / (2 * eps)
                   for e in (eps * np.eye(n))])
    assert np.allclose(g, fd, atol=1e-5, rtol=1e-4)
