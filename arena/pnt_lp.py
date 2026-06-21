"""LP layer for `prime-number-theorem` (id=7): optimal f on a FIXED support K.

Built against the *verbatim* server verifier (see verifiers/prime_number_theorem.py):

    pf = {k: clip(f(k), -10, 10)}            # box applies to submitted values
    pf[1] = pf.get(1, 0) - sum(v/k)          # f(1) is OVERWRITTEN, NOT re-clipped
    constraint: sum_k pf[k]*floor(x/k) <= 1.0001   for x ~ Uniform(1, 10*max(K))
    score:      S = -sum_k pf[k]*log(k)/k          (log(1)=0 -> f(1) absent from S)

Three load-bearing consequences encoded here:

1. The equality sum_k f(k)/k = 0 is auto-enforced by the verifier patching f(1).
   So f(1) is a *dependent* variable: f(1) = -sum_{k>=2} f(k)/k. We optimize over
   k>=2 only and substitute f(1) into the constraints (no A_eq row). f(1) is NOT
   box-constrained (the verifier does not re-clip it) and not in the objective.

2. The constraint is sampled on CONTINUOUS x in [1, 10*max(K)). Since
   g(x) = sum_k f(k)*floor(x/k) is piecewise-constant in x and only changes at
   breakpoints x = j*k (all integers), sup over [1, 10M) is attained on the integer
   grid {1, ..., 10M-1}. So an exact, finite check replaces the Monte-Carlo sampling.

3. The server tolerates <= 1.0001. We solve and verify against the HONEST bound 1.0
   and never spend the 0.0001 slack to inflate the score (that would be gaming).

This module is pure (no network, no disk). It mirrors the verifier; it never edits it.
"""

from __future__ import annotations

import math
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Literal

import numpy as np
from scipy.optimize import linprog

VALUE_BOUND = 10.0
"""Box bound |f(k)| <= 10 on the submitted (k>=2) values, matching the verifier clip."""

RHS = 1.0
"""Honest constraint bound. The server tolerates 1.0001; we deliberately do not use it."""

MAX_SUPPORT = 2000
"""Verifier rejects partial_function with more than 2000 keys."""

X_CAP_FACTOR = 10
"""Verifier samples x in [1, 10*max(K)); the integer grid [1, 10*max(K)) is exhaustive."""

_GRID_BATCH_BYTES = 40 * 1024 * 1024
"""Batch the (x, k) floor matrix to bound peak memory (same idea as the verifier)."""


@dataclass(frozen=True)
class LPResult:
    """Outcome of solving the LP on one fixed support.

    f          full partial function incl. the reconstructed dependent f(1).
    S          objective value -sum_{k>=2} f(k)*log(k)/k (equals the server score).
    grid_max   max of g(x) over the integer grid [1, 10*max(K)) for the returned f.
    success    True iff linprog converged AND the returned f is grid-feasible at RHS.
    status     human-readable status ("optimal" / linprog message / "max_iters").
    n_rows     number of cutting-plane constraint rows in the final LP.
    n_iters    number of constraint-generation iterations performed.
    """

    f: dict[int, float]
    S: float
    grid_max: float
    success: bool
    status: str
    n_rows: int
    n_iters: int


def normalize_support(K: Iterable[int]) -> list[int]:
    """Return sorted unique integer support with 1 included (verifier always materializes key 1)."""
    ks = {int(k) for k in K}
    ks.add(1)
    if any(k < 1 for k in ks):
        raise ValueError("support keys must be >= 1")
    out = sorted(ks)
    if len(out) > MAX_SUPPORT:
        raise ValueError(f"support has {len(out)} keys, exceeds MAX_SUPPORT={MAX_SUPPORT}")
    return out


def objective_vector(kplus: list[int]) -> np.ndarray:
    """Cost c[j] = log(k)/k for k>=2; linprog minimizes c.f = -S, so S = -res.fun."""
    return np.array([math.log(k) / k for k in kplus], dtype=np.float64)


def reconstruct_f1(fplus: dict[int, float]) -> float:
    """f(1) = -sum_{k>=2} f(k)/k, exactly as the verifier patches pf[1]."""
    return -sum(v / k for k, v in fplus.items())


def _constraint_rows(xs: np.ndarray, kplus: np.ndarray) -> np.ndarray:
    """Rows a(x,k) = floor(x/k) - x/k for k>=2 (integer x, so the k=1 term floor(x/1)=x).

    Derivation: with f(1) = -sum f(k)/k substituted,
        g(x) = f(1)*x + sum_{k>=2} f(k)*floor(x/k)
             = sum_{k>=2} f(k)*(floor(x/k) - x/k).
    """
    floors = xs[:, None] // kplus[None, :]  # exact integer division
    return floors.astype(np.float64) - xs[:, None].astype(np.float64) / kplus[None, :]


def grid_g(keys: np.ndarray, values: np.ndarray, x_max: int) -> np.ndarray:
    """g(x) = sum_k values*floor(x/k) over the FULL key set, for integer x in [1, x_max).

    Batched over x to bound memory. Returns an array g of length x_max-1 (index i -> x=i+1).
    """
    keys = np.asarray(keys, dtype=np.int64)
    values = np.asarray(values, dtype=np.float64)
    nk = max(1, len(keys))
    batch = max(1, _GRID_BATCH_BYTES // (nk * 8))
    out = np.empty(max(0, x_max - 1), dtype=np.float64)
    x = 1
    while x < x_max:
        hi = min(x + batch, x_max)
        xs = np.arange(x, hi, dtype=np.int64)
        floors = (xs[:, None] // keys[None, :]).astype(np.float64)
        out[x - 1 : hi - 1] = floors @ values
        x = hi
    return out


def _violations(g: np.ndarray, rhs: float) -> np.ndarray:
    """Integer x (1-based) where g(x) > rhs, sorted by descending violation."""
    mask = g > rhs
    if not mask.any():
        return np.empty(0, dtype=np.int64)
    xs = np.nonzero(mask)[0] + 1
    return xs[np.argsort(-g[mask])]


def _result_from_x(
    xvec: np.ndarray,
    kplus: list[int],
    c: np.ndarray,
    x_max: int,
    rhs: float,
    feas_tol: float,
    n_rows: int,
    it: int,
) -> tuple[LPResult, np.ndarray]:
    """Build the LPResult and the grid g(x) from a solver's f(k>=2) solution vector.

    S = -c.x (objective is minimize c.f = -S); identical whichever backend produced x.
    """
    fplus = {k: float(v) for k, v in zip(kplus, xvec)}
    f_full = dict(fplus)
    f_full[1] = reconstruct_f1(fplus)
    full_keys = np.fromiter(f_full.keys(), dtype=np.int64)
    full_vals = np.fromiter(f_full.values(), dtype=np.float64)
    g = grid_g(full_keys, full_vals, x_max)
    grid_max = float(g.max()) if g.size else 0.0
    S = -float(np.dot(c, xvec))
    res = LPResult(f_full, S, grid_max, grid_max <= rhs + feas_tol, "optimal", n_rows, it)
    return res, g


def _cutting_plane_scipy(
    kplus: list[int],
    kplus_arr: np.ndarray,
    c: np.ndarray,
    M: int,
    x_max: int,
    rhs: float,
    max_iters: int,
    cuts_per_iter: int,
    seed_rows: int,
    feas_tol: float,
) -> LPResult:
    """Cold-resolve cutting plane via scipy.optimize.linprog (kept as a fallback / reference)."""
    bounds = [(-VALUE_BOUND, VALUE_BOUND)] * len(kplus)
    X = sorted(range(1, min(M, seed_rows) + 1))
    last: LPResult | None = None
    for it in range(1, max_iters + 1):
        xs = np.asarray(X, dtype=np.int64)
        a_ub = _constraint_rows(xs, kplus_arr)
        sol = linprog(c, A_ub=a_ub, b_ub=np.full(len(X), rhs), bounds=bounds, method="highs")
        if not sol.success:
            return LPResult({1: 0.0}, 0.0, math.inf, False, f"linprog: {sol.message}", len(X), it)
        last, g = _result_from_x(sol.x, kplus, c, x_max, rhs, feas_tol, len(X), it)
        viol = _violations(g, rhs + feas_tol)
        if viol.size == 0:
            return last
        X = sorted(set(X) | {int(x) for x in viol[:cuts_per_iter]})
    assert last is not None
    return LPResult(last.f, last.S, last.grid_max, False, "max_iters", last.n_rows, max_iters)


def _cutting_plane_highs(
    kplus: list[int],
    kplus_arr: np.ndarray,
    c: np.ndarray,
    M: int,
    x_max: int,
    rhs: float,
    max_iters: int,
    cuts_per_iter: int,
    seed_rows: int,
    feas_tol: float,
) -> LPResult:
    """Warm-started cutting plane via the persistent `WarmLP` model (single source of the loop).

    `kplus_arr`/`c`/`M` stay in the signature only because `solve_support` dispatches this and
    `_cutting_plane_scipy` through one call (the scipy sibling uses them); `WarmLP` recomputes them.
    Numerically identical to the previously inlined loop (pinned by tests/test_pnt_warm.py).
    """
    from arena.pnt_warm import WarmLP  # noqa: PLC0415 -- lazy: WarmLP imports from this module

    return WarmLP(kplus, rhs=rhs, feas_tol=feas_tol).row_generate_to_feasible(
        x_max, max_iters=max_iters, cuts_per_iter=cuts_per_iter, seed_rows=seed_rows
    )


def solve_support(
    K: Iterable[int],
    *,
    rhs: float = RHS,
    max_iters: int = 60,
    cuts_per_iter: int = 1_000_000,
    seed_rows: int = 1_000_000,
    feas_tol: float = 1e-9,
    solver: Literal["highs", "scipy"] = "highs",
) -> LPResult:
    """Maximize S on a fixed support via cutting-plane LP, honest bound `rhs`.

    Seeds the integer rows [1, min(max(K), seed_rows)], solves, finds the rows where the
    continuous constraint is violated on the full grid [1, 10*max(K)), adds them as cuts, and
    re-solves until grid-feasible or `max_iters` is hit. f=0 (S=0) is always feasible, so the
    LP is never infeasible/unbounded.

    solver="highs" (default) keeps one persistent warm-started HiGHS model (adds only new cuts
    each round); solver="scipy" cold-resolves via scipy.optimize.linprog (reference/fallback).
    Both delegate to HiGHS, so the optimum is identical; "highs" is markedly faster at large |K|.
    """
    if solver not in ("highs", "scipy"):
        raise ValueError(f"unknown solver {solver!r}; use 'highs' or 'scipy'")
    if max_iters < 1:
        raise ValueError("max_iters must be >= 1")
    keys = normalize_support(K)
    kplus = [k for k in keys if k != 1]
    if not kplus:
        return LPResult({1: 0.0}, 0.0, 0.0, True, "trivial-support", 0, 0)

    M = keys[-1]
    x_max = X_CAP_FACTOR * M
    kplus_arr = np.asarray(kplus, dtype=np.int64)
    c = objective_vector(kplus)
    impl = _cutting_plane_highs if solver == "highs" else _cutting_plane_scipy
    return impl(kplus, kplus_arr, c, M, x_max, rhs, max_iters, cuts_per_iter, seed_rows, feas_tol)
