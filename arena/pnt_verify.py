"""Dual verification for `prime-number-theorem` candidates.

Every candidate that the LP claims feasible is checked TWO ways before it can be trusted:

(a) Server replay  - run the verbatim server verifier locally on the submission dict and
    confirm it returns a finite score equal to the LP's S (the CLAUDE.md fidelity invariant
    local score == server score).

(b) Exact grid check - the server only samples the constraint on 1e7 Monte-Carlo points;
    we instead evaluate g(x) = sum_k f(k)*floor(x/k) on the FULL integer grid [1, 10*max(K)).
    Because g is piecewise-constant in continuous x and only jumps at integer breakpoints
    x = j*k, this grid is the exact supremum over the whole sampled domain, not a sample.
    We require g(x) <= 1.0 (honest bound), so a candidate that survives MC sampling but
    violates the exact grid is discarded as gaming.

A separate tail note covers x >= 10*max(K) (outside the domain the server ever evaluates):
since sum_k f(k)/k = 0, g(x) = -sum_k f(k)*{x/k} is periodic with period lcm(K); we certify
the tail when the grid already covers a full period, else by an explicit period check or a
residue bound, and report honestly when only the competition range is certified.

This module mirrors the verifier and reuses arena.pnt_lp; it performs no network writes.
"""

from __future__ import annotations

import math
from collections.abc import Iterable
from dataclasses import dataclass

import numpy as np

from arena.pnt_lp import RHS, X_CAP_FACTOR, grid_g, normalize_support, reconstruct_f1
from arena.verifier import load_evaluate

SLUG = "prime-number-theorem"
TAIL_PERIOD_CAP = 5_000_000
"""Max integers to scan for an explicit tail period check before falling back to the bound."""

_LCM_CAP = 1 << 62
"""Stop accumulating lcm once it exceeds this; the period is then 'huge' for our purposes."""


@dataclass(frozen=True)
class VerifyReport:
    """Verdict for one candidate partial function.

    s_lp           the LP's reported S (objective), passed in for cross-check.
    s_server       the verbatim server verifier's score (-inf if it rejects the MC check).
    server_finite  True iff s_server is finite (server accepts the candidate).
    score_matches  True iff |s_server - s_lp| <= score_tol (local == server fidelity).
    grid_max       max g(x) over the exact integer grid [1, 10*max(K)).
    grid_feasible  True iff grid_max <= RHS + feas_tol (honest, exhaustive over the domain).
    n_violations   number of integer x in the domain with g(x) > RHS + feas_tol.
    tail_method    how x >= 10*max(K) was handled: covered_by_grid / period_check /
                   residue_bound / residue_bound_uncertified.
    tail_certified True iff the infinite tail is provably feasible (requires grid_feasible:
                   tail feasibility is meaningless when the domain itself is violated).
    submittable    grid_feasible AND server_finite AND score_matches (competition-valid &
                   not gamed). Tail certification is reported but not a submission gate, since
                   the server defines feasibility on [1, 10*max(K)) only.
    """

    s_lp: float
    s_server: float
    server_finite: bool
    score_matches: bool
    grid_max: float
    grid_feasible: bool
    n_violations: int
    tail_method: str
    tail_certified: bool
    submittable: bool


def to_solution(f: dict[int, float]) -> dict:
    """Build the submission dict, emitting only k>=2.

    f(1) is intentionally omitted: the verifier clips submitted values BEFORE patching pf[1],
    so submitting an out-of-box f(1) would be clipped and corrupt the reconstruction. Omitting
    key 1 makes the verifier set pf[1] = -sum_{k>=2} f(k)/k, exactly our reconstruct_f1.
    """
    return {"partial_function": {str(k): float(v) for k, v in f.items() if k != 1}}


def server_score(f: dict[int, float]) -> float:
    """Score with the verbatim server verifier (runs the 1e7-point MC constraint check)."""
    return float(load_evaluate(SLUG)(to_solution(f)))


def _full_arrays(f: dict[int, float], keys: list[int]) -> tuple[np.ndarray, np.ndarray]:
    """Full (keys, values) incl. the reconstructed f(1), matching what the verifier scores."""
    fplus = {k: f[k] for k in keys if k != 1 and k in f}
    full = dict(fplus)
    full[1] = reconstruct_f1(fplus)
    ks = np.fromiter(full.keys(), dtype=np.int64)
    vs = np.fromiter((full[int(k)] for k in ks), dtype=np.float64)
    return ks, vs


def _lcm_capped(keys: list[int]) -> int | None:
    """lcm of the support, or None if it exceeds _LCM_CAP (treated as 'huge')."""
    cur = 1
    for k in keys:
        cur = math.lcm(cur, k)
        if cur > _LCM_CAP:
            return None
    return cur


def tail_certificate(
    f: dict[int, float], keys: list[int], *, rhs: float = RHS, feas_tol: float = 1e-9
) -> tuple[str, bool]:
    """Certify g(x) <= rhs for x >= 10*max(K). Returns (method, certified).

    g(x) = -sum_k f(k)*{x/k} is periodic with period L = lcm(K) (sum f(k)/k = 0). So the
    supremum over all x is attained within one period.
    """
    ks, vs = _full_arrays(f, keys)
    m = int(ks.max())
    x_max = X_CAP_FACTOR * m
    period = _lcm_capped([int(k) for k in keys])

    if period is not None and period <= x_max:
        return "covered_by_grid", True  # grid [1, 10M) already spans a full period
    if period is not None and period <= TAIL_PERIOD_CAP:
        g = grid_g(ks, vs, period + 1)
        return "period_check", bool(g.max() <= rhs + feas_tol)
    # Residue bound: |g(x)| = |sum f(k){x/k}| <= sum |f(k)|*(k-1)/k.
    bound = float(np.sum(np.abs(vs) * (ks - 1) / ks))
    if bound <= rhs + feas_tol:
        return "residue_bound", True
    return "residue_bound_uncertified", False


def dual_verify(
    f: dict[int, float],
    K: Iterable[int],
    s_lp: float,
    *,
    rhs: float = RHS,
    feas_tol: float = 1e-9,
    score_tol: float = 1e-7,
) -> VerifyReport:
    """Run both verifications and the tail note; return a single verdict."""
    keys = normalize_support(K)
    ks, vs = _full_arrays(f, keys)
    m = int(ks.max())
    x_max = X_CAP_FACTOR * m

    g = grid_g(ks, vs, x_max)
    grid_max = float(g.max()) if g.size else 0.0
    n_violations = int(np.count_nonzero(g > rhs + feas_tol))
    grid_feasible = grid_max <= rhs + feas_tol

    s_server = server_score(f)
    server_finite = math.isfinite(s_server)
    score_matches = server_finite and abs(s_server - s_lp) <= score_tol

    tail_method, tail_raw = tail_certificate(f, keys, rhs=rhs, feas_tol=feas_tol)
    # A clean tail is only meaningful if the domain itself is feasible: a covered-by-grid /
    # period-check verdict reuses values that include any in-domain violation.
    tail_certified = tail_raw and grid_feasible

    submittable = grid_feasible and server_finite and score_matches
    return VerifyReport(
        s_lp=s_lp,
        s_server=s_server,
        server_finite=server_finite,
        score_matches=score_matches,
        grid_max=grid_max,
        grid_feasible=grid_feasible,
        n_violations=n_violations,
        tail_method=tail_method,
        tail_certified=tail_certified,
        submittable=submittable,
    )
