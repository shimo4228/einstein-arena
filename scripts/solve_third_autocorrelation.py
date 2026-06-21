"""Local solver for `third-autocorrelation-inequality` (id=4): minimize the sign-unrestricted
autoconvolution ratio.

Objective (exact, from the server verifier), with dx = 0.5/n and f free (may be negative):
    C(f) = |max( np.convolve(f, f, 'full') * dx )| / (sum(f) * dx)**2
Lower is better. The "max-then-abs" measures only the single most-positive autoconvolution
peak (negative troughs are ignored). C is scale-invariant. Only feasibility: (sum(f)*dx)**2 >= 1e-9.
Arena top: 1.4523043331831582 (OrganonAgent, n=100000). minImprovement = 1e-4, so taking #1
needs C <= ~1.4522043. This is a sign-unrestricted variant with little prior human literature.

Strategy: FFT autoconvolution (O(n log n)) so high n is tractable; smooth-max (logsumexp)
surrogate with EXACT jax gradient; Adam with a fixed scale gauge (sum(f)=2n); beta annealing;
two paths — (a) refine the arena rank-1 seed + perturbations, (b) from-scratch even-symmetric
structured inits at moderate n, then upsample to high n and refine. Final candidates scored with
the SERVER verifier. NO submission happens here.

Run:  uv run python scripts/solve_third_autocorrelation.py
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

jax.config.update("jax_enable_x64", True)

SLUG = "third-autocorrelation-inequality"
ARENA_TOP = 1.4523043331831582
RESULTS = REPO_ROOT / "results" / SLUG
_evaluate = load_evaluate(SLUG)


def exact_C(f: np.ndarray) -> float:
    """Authoritative score via the server verifier; +inf if it rejects."""
    try:
        return _evaluate({"values": np.asarray(f, dtype=float).tolist()})
    except (ValueError, AssertionError):
        return float("inf")


def _conv_full_np(f: np.ndarray) -> np.ndarray:
    n = len(f)
    L = 2 * n - 1
    F = np.fft.rfft(f, L)
    return np.fft.irfft(F * F, L)[:L]


def fast_C(f: np.ndarray) -> float:
    """FFT replica of the verifier objective (matches to ~1e-10); for cheap ranking."""
    n = len(f)
    dx = 0.5 / n
    scaled = _conv_full_np(f) * dx
    den = (f.sum() * dx) ** 2
    if den < 1e-9:
        return float("inf")
    return abs(scaled.max()) / den


def _gauge(f: np.ndarray) -> np.ndarray:
    """Fix the scale degree of freedom: rescale so sum(f) = 2n (i.e. sum(f)*dx = 1)."""
    s = f.sum()
    if abs(s) < 1e-9:
        f = f + 1.0  # break away from zero-integral
        s = f.sum()
    return f * (2.0 * len(f) / s)


def _make_grad(n: int, beta: float):
    dx = 0.5 / n
    L = 2 * n - 1

    @jax.jit
    def obj(f):
        F = jnp.fft.rfft(f, n=L)
        conv = jnp.fft.irfft(F * F, n=L)            # linear autoconvolution, length L
        num = logsumexp(beta * (conv * dx)) / beta  # smooth max(conv*dx) (peak is positive)
        den = (jnp.sum(f) * dx) ** 2
        return num / den

    return jax.jit(jax.value_and_grad(obj))


def optimize(f0: np.ndarray, betas: list[float], steps: int, lr: float = 0.02) -> np.ndarray:
    f = _gauge(f0.astype(np.float64).copy())
    m = np.zeros_like(f)
    v = np.zeros_like(f)
    t = 0
    for beta in betas:
        vg = _make_grad(len(f), beta)
        for _ in range(steps):
            t += 1
            _, g = vg(jnp.asarray(f))
            g = np.asarray(g)
            m = 0.9 * m + 0.1 * g
            v = 0.999 * v + 0.001 * (g * g)
            mhat = m / (1 - 0.9**t)
            vhat = v / (1 - 0.999**t)
            f = _gauge(f - lr * mhat / (np.sqrt(vhat) + 1e-12))
    return f


def _upsample(f: np.ndarray, n_hi: int) -> np.ndarray:
    xs_lo = np.linspace(0, 1, len(f))
    xs_hi = np.linspace(0, 1, n_hi)
    return np.interp(xs_hi, xs_lo, f)


def _structured_inits(n: int, rng: np.random.Generator) -> list[np.ndarray]:
    x = np.linspace(-0.25, 0.25, n)
    bump = np.exp(-(x**2) / (2 * 0.08**2))                          # central positive bump
    side = np.exp(-((np.abs(x) - 0.18) ** 2) / (2 * 0.04**2))       # flanking lobes
    return [
        _gauge(bump),
        _gauge(bump - 0.6 * side),                                  # bump minus negative sidelobes
        _gauge(bump - 1.0 * side),
        _gauge(np.cos(np.pi * x / 0.5) ** 2 - 0.3 * side),
        _gauge(bump + 0.2 * rng.normal(size=n)),
    ]


def load_seed() -> np.ndarray:
    return np.array(json.loads((RESULTS / "seed_arena_rank1.json").read_text())["values"], dtype=np.float64)


def solve(*, refine_seed: bool = True, n_perturb: int = 3, from_scratch: bool = True,
          n_lo: int = 4000, seed: int = 0):
    rng = np.random.default_rng(seed)
    seed_arr = load_seed()
    n_hi = len(seed_arr)
    best_C, best_f, best_src = exact_C(seed_arr), seed_arr.copy(), "arena_seed"

    if refine_seed:
        inits = [seed_arr] + [seed_arr + rng.normal(0, 0.02 * seed_arr.std(), size=seed_arr.shape)
                              for _ in range(n_perturb)]
        for k, init in enumerate(inits):
            f = optimize(init, [200.0, 1000.0, 5000.0], steps=120)
            c = exact_C(f)
            if c < best_C:
                best_C, best_f, best_src = c, f.copy(), f"refine_{k}"

    if from_scratch:
        for j, init in enumerate(_structured_inits(n_lo, rng)):
            f_lo = optimize(init, [50.0, 200.0, 1000.0, 5000.0], steps=400)
            f_hi = optimize(_upsample(f_lo, n_hi), [2000.0, 20000.0], steps=120)
            c = exact_C(f_hi)
            if c < best_C:
                best_C, best_f, best_src = c, f_hi.copy(), f"scratch_{j}"

    return best_C, best_f, best_src


def main() -> None:
    best_C, best_f, src = solve()
    gap = best_C - ARENA_TOP
    verdict = "BEATS top" if gap < -1e-4 else "ties/near top" if gap <= 1e-4 else "above top"
    print(f"problem        : {SLUG}")
    print(f"best local C   : {best_C:.16f}   (source: {src})")
    print(f"arena top      : {ARENA_TOP:.16f}")
    print(f"gap (local-top): {gap:+.3e}   ({verdict})")
    print(f"to TAKE #1 need: C <= {ARENA_TOP - 1e-4:.6f} (minImprovement 1e-4)")
    RESULTS.mkdir(parents=True, exist_ok=True)
    (RESULTS / "best.json").write_text(json.dumps({
        "problem": SLUG, "C": best_C, "arena_top": ARENA_TOP, "gap": gap,
        "source": src, "n": len(best_f), "values": best_f.tolist(),
    }))
    print(f"saved          : {RESULTS / 'best.json'}")


if __name__ == "__main__":
    main()
