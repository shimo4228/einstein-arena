"""Local solver for `min-distance-ratio-2d` (id=5): minimize R = (d_max/d_min)^2 over 16 pts.

Validation goal (Phase A): reach R <= arena top (12.889229907717521) using our OWN
multistart search, scoring every candidate with the server's verbatim verifier so the
local score equals the server score. NO submission happens here.

Run:  uv run python scripts/solve_min_distance_ratio.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from scipy.optimize import minimize
from scipy.special import logsumexp

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from arena.verifier import load_evaluate  # noqa: E402

SLUG = "min-distance-ratio-2d"
N = 16
ARENA_TOP = 12.889229907717521
_evaluate = load_evaluate(SLUG)


def exact_R(points: np.ndarray) -> float:
    """Score with the server's verbatim verifier; +inf on invalid (e.g. collapsed points)."""
    try:
        return _evaluate({"vectors": points.tolist()})
    except ValueError:
        return float("inf")


def _pairwise(points: np.ndarray) -> np.ndarray:
    iu = np.triu_indices(N, k=1)
    diff = points[:, None, :] - points[None, :, :]
    return np.sqrt(np.sum(diff**2, axis=-1))[iu]


def _surrogate_and_grad(flat: np.ndarray, beta: float) -> tuple[float, np.ndarray]:
    """Smooth approx of R via soft-max/soft-min over the 120 pair distances, with analytic grad.

    Analytic gradient avoids L-BFGS finite-difference (which cost 64 evals/grad and made the
    finite-diff version ~30-60x slower). This is the harness pattern to carry to Phase B.
    """
    pts = flat.reshape(N, 2)
    rows, cols = np.triu_indices(N, k=1)
    diff = pts[rows] - pts[cols]                       # (120, 2)
    d = np.sqrt(np.sum(diff**2, axis=1)) + 1e-15       # (120,)
    s_max = logsumexp(beta * d) / beta
    s_min = -logsumexp(-beta * d) / beta
    R = (s_max / s_min) ** 2
    w_max = np.exp(beta * d - logsumexp(beta * d))     # softmax weights = dS_max/dd_k
    w_min = np.exp(-beta * d - logsumexp(-beta * d))   # softmin weights = dS_min/dd_k
    dR_dd = 2.0 * (s_max / s_min) * (w_max / s_min - s_max * w_min / s_min**2)
    contrib = (dR_dd / d)[:, None] * diff              # chain through d_k = ||p_rows - p_cols||
    grad = np.zeros((N, 2))
    np.add.at(grad, rows, contrib)
    np.add.at(grad, cols, -contrib)
    return float(R), grad.ravel()


def _normalize(points: np.ndarray) -> np.ndarray:
    """Center and scale so the median pairwise distance is ~1 (keeps beta*d well-conditioned)."""
    pts = points - points.mean(axis=0)
    med = np.median(_pairwise(pts))
    return pts / med if med > 0 else pts


# ---- structured & random initial configurations -----------------------------
def _rings(counts: list[int], radii: list[float]) -> np.ndarray:
    pts = []
    for k, r in zip(counts, radii):
        if k == 1 and r == 0.0:
            pts.append([0.0, 0.0])
            continue
        for j in range(k):
            ang = 2 * np.pi * j / k
            pts.append([r * np.cos(ang), r * np.sin(ang)])
    return np.array(pts, dtype=np.float64)


def _hex_cluster() -> np.ndarray:
    a = np.array([1.0, 0.0])
    b = np.array([0.5, np.sqrt(3) / 2])
    lattice = np.array([i * a + j * b for i in range(-4, 5) for j in range(-4, 5)])
    order = np.argsort(np.sum(lattice**2, axis=1))
    return lattice[order[:N]]


def _grid_4x4() -> np.ndarray:
    return np.array([[i, j] for i in range(4) for j in range(4)], dtype=np.float64)


def structured_inits() -> list[np.ndarray]:
    cand = [
        _hex_cluster(),
        _grid_4x4(),
        _rings([1, 5, 10], [0.0, 1.0, 2.0]),
        _rings([1, 6, 9], [0.0, 1.0, 2.0]),
        _rings([4, 12], [1.0, 2.2]),
        _rings([1, 7, 8], [0.0, 1.0, 1.9]),
    ]
    return [_normalize(c) for c in cand if len(c) == N]


def _polish(flat: np.ndarray, betas: list[float]) -> np.ndarray:
    x = flat.copy()
    for beta in betas:
        res = minimize(_surrogate_and_grad, x, args=(beta,), method="L-BFGS-B", jac=True,
                       options={"maxiter": 300, "ftol": 1e-14})
        x = _normalize(res.x.reshape(N, 2)).ravel()
    return x


def solve(seed: int = 0, n_random: int = 200,
          betas: tuple[float, ...] = (4, 8, 16, 32, 64, 128, 256, 512)) -> tuple[float, np.ndarray]:
    rng = np.random.default_rng(seed)
    betas_l = list(betas)
    inits = structured_inits()
    inits += [_normalize(rng.normal(size=(N, 2))) for _ in range(n_random)]

    best_R, best_pts = float("inf"), None
    for init in inits:
        flat = _polish(init.ravel(), betas_l)
        pts = flat.reshape(N, 2)
        r = exact_R(pts)
        if r < best_R:
            best_R, best_pts = r, pts.copy()

    # final exact-objective polish (Nelder-Mead) on the best candidate
    if best_pts is not None:
        res = minimize(lambda f: exact_R(f.reshape(N, 2)), best_pts.ravel(),
                       method="Nelder-Mead", options={"maxiter": 2000, "xatol": 1e-10, "fatol": 1e-12})
        if res.fun < best_R:
            best_R, best_pts = float(res.fun), res.x.reshape(N, 2)
    return best_R, best_pts


def main() -> None:
    best_R, best_pts = solve()
    gap = best_R - ARENA_TOP
    print(f"problem      : {SLUG}")
    print(f"best local R : {best_R:.15f}")
    print(f"arena top    : {ARENA_TOP:.15f}")
    print(f"gap (local-top): {gap:+.3e}   ({'<= top (tie/beat)' if gap <= 1e-9 else 'above top'})")
    print(f"sqrt(R) ratio: {np.sqrt(best_R):.10f}  (Friedman compendium ~3.5901)")

    out_dir = REPO_ROOT / "results" / SLUG
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "best.json").write_text(json.dumps({
        "problem": SLUG, "R": best_R, "arena_top": ARENA_TOP, "gap": gap,
        "vectors": best_pts.tolist() if best_pts is not None else None,
    }, indent=2))
    print(f"saved        : {out_dir / 'best.json'}")


if __name__ == "__main__":
    main()
