"""Local solver for `erdos-min-overlap` (id=1): minimize the min-overlap upper bound.

Objective (exact, from the server verifier):
    C(h) = (2/n) * max( np.correlate(h, 1-h, 'full') )
over the feasible set h in [0,1]^n with sum(h) = n/2 (h is a step function on [0,2],
dx = 2/n, integral 1). Lower C is a stronger upper bound on the Erdős minimum-overlap
constant. Arena top (4 agents tied): 0.3808703105862199. Literature lower bound: 0.379005.

Strategy: smooth-max (logsumexp) surrogate with EXACT jax gradient, projected-gradient
(Adam) onto the box+sum feasible set, beta annealing, multistart from the arena rank-1 seed
plus symmetric / perturbed variants. Every candidate is scored with the SERVER verifier so
the local score equals the server score. NO submission happens here.

Run:  uv run python scripts/solve_erdos_min_overlap.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
from jax.scipy.special import logsumexp

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from arena.verifier import load_evaluate  # noqa: E402

jax.config.update("jax_enable_x64", True)  # match the verifier's float64 scoring

SLUG = "erdos-min-overlap"
ARENA_TOP = 0.3808703105862199
RESULTS = REPO_ROOT / "results" / SLUG
_evaluate = load_evaluate(SLUG)


def exact_C(h: np.ndarray) -> float:
    """Score with the server's verbatim verifier; +inf if it rejects (infeasible)."""
    try:
        return _evaluate({"values": np.asarray(h, dtype=float).tolist()})
    except AssertionError:
        return float("inf")


def project(h: np.ndarray) -> np.ndarray:
    """Euclidean-ish projection onto {h in [0,1]^n, sum(h)=n/2} via clip(h-lambda,0,1) bisection."""
    n = len(h)
    target = n / 2.0
    lo, hi = float(h.min()) - 1.0, float(h.max())
    for _ in range(60):
        lam = 0.5 * (lo + hi)
        s = np.clip(h - lam, 0.0, 1.0).sum()
        if s > target:  # too much mass -> raise lambda
            lo = lam
        else:
            hi = lam
    return np.clip(h - 0.5 * (lo + hi), 0.0, 1.0)


def _make_grad(n: int, beta: float):
    @jax.jit
    def obj(h):
        c = jnp.correlate(h, 1.0 - h, mode="full")
        return (2.0 / n) * (logsumexp(beta * c) / beta)
    return jax.jit(jax.value_and_grad(obj))


def optimize(h0: np.ndarray, betas: list[float], steps: int, lr: float = 0.01) -> np.ndarray:
    """Projected Adam on the annealed smooth-max surrogate."""
    h = project(h0.astype(np.float64).copy())
    m = np.zeros_like(h)
    v = np.zeros_like(h)
    t = 0
    for beta in betas:
        vg = _make_grad(len(h), beta)
        for _ in range(steps):
            t += 1
            _, g = vg(jnp.asarray(h))
            g = np.asarray(g)
            m = 0.9 * m + 0.1 * g
            v = 0.999 * v + 0.001 * (g * g)
            mhat = m / (1 - 0.9**t)
            vhat = v / (1 - 0.999**t)
            h = project(h - lr * mhat / (np.sqrt(vhat) + 1e-12))
    return h


def load_seed() -> np.ndarray:
    seed = json.loads((RESULTS / "seed_arena_rank1.json").read_text())
    return np.array(seed["values"], dtype=np.float64)


def make_inits(seed: np.ndarray, n_perturb: int, rng: np.random.Generator) -> list[np.ndarray]:
    sym = project(0.5 * (seed + seed[::-1]))  # symmetric average of the seed
    inits = [seed.copy(), sym]
    for _ in range(n_perturb):
        inits.append(project(np.clip(seed + rng.normal(0, 0.05, size=seed.shape), 0, 1)))
    return inits


def solve(seed_arr: np.ndarray | None = None, *, n_perturb: int = 6, steps: int = 250,
          betas: tuple[float, ...] = (200, 1000, 5000, 20000, 80000), seed: int = 0):
    rng = np.random.default_rng(seed)
    base = seed_arr if seed_arr is not None else load_seed()
    best_C, best_h = exact_C(base), base.copy()  # never do worse than the seed
    for init in make_inits(base, n_perturb, rng):
        h = optimize(init, list(betas), steps)
        c = exact_C(h)
        if c < best_C:
            best_C, best_h = c, h.copy()
    return best_C, best_h


def main() -> None:
    best_C, best_h = solve()
    gap = best_C - ARENA_TOP
    print(f"problem        : {SLUG}")
    print(f"best local C   : {best_C:.16f}")
    print(f"arena top      : {ARENA_TOP:.16f}")
    print(f"gap (local-top): {gap:+.3e}   ({'BEATS top' if gap < -1e-9 else 'tie' if abs(gap) <= 1e-6 else 'above top'})")
    print(f"literature LB  : 0.379005  (room below: {best_C - 0.379005:+.5f})")
    RESULTS.mkdir(parents=True, exist_ok=True)
    (RESULTS / "best.json").write_text(json.dumps({
        "problem": SLUG, "C": best_C, "arena_top": ARENA_TOP, "gap": gap,
        "n": len(best_h), "values": best_h.tolist(),
    }))
    print(f"saved          : {RESULTS / 'best.json'}")


if __name__ == "__main__":
    main()
