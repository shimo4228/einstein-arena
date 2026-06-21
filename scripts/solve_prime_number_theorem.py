"""Local LP baseline for `prime-number-theorem` (id=7): maximize S on chosen supports.

Strategy (step 2 of the PNT plan, see docs/SATURATION-SCAN.md):
    pick a SUPPORT set K (math-derived only -- never a competitor's solution), then let the
    cutting-plane LP find the optimal f on it (arena.pnt_lp.solve_support). Each candidate is
    dual-verified: the verbatim server verifier AND the exact integer-grid full-constraint
    check at the honest bound 1.0 (arena.pnt_verify.dual_verify). The best submittable S is
    saved to results/prime-number-theorem/best.json. NO submission happens here.

Supports tried are all derived from the mathematics of the problem (truncations of N, the
squarefree filter on which Mobius mu is supported). We never ingest the arena top / SOTA
solution as optimizer input -- see the originality note printed at the end.

Run:  uv run python scripts/solve_prime_number_theorem.py            # default M ladder
      uv run python scripts/solve_prime_number_theorem.py 30 60 120  # explicit M values
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from arena.pnt_lp import MAX_SUPPORT, LPResult, solve_support  # noqa: E402
from arena.pnt_verify import VerifyReport, dual_verify  # noqa: E402

SLUG = "prime-number-theorem"
ARENA_TOP = 0.994901  # leaderboard rank-1 at recon time (docs/SATURATION-SCAN.md)
THEORETICAL_MAX = 1.0  # attained by the Mobius function mu
RESULTS = REPO_ROOT / "results" / SLUG
DEFAULT_LADDER = (30, 60, 120, 300, 600)


def mobius_upto(n: int) -> np.ndarray:
    """mu(1..n) via a linear sieve. mu[i] is the Mobius value of i (index 0 unused)."""
    mu = np.ones(n + 1, dtype=np.int64)
    mu[0] = 0
    primes: list[int] = []
    is_comp = np.zeros(n + 1, dtype=bool)
    for i in range(2, n + 1):
        if not is_comp[i]:
            primes.append(i)
            mu[i] = -1
        for p in primes:
            if i * p > n:
                break
            is_comp[i * p] = True
            if i % p == 0:
                mu[i * p] = 0
                break
            mu[i * p] = -mu[i]
    return mu


def truncated_support(m: int) -> list[int]:
    """K = {1, ..., m} (full truncation)."""
    return list(range(1, m + 1))


def squarefree_support(m: int) -> list[int]:
    """K = {1} + {squarefree k <= m} -- the set where mu is nonzero."""
    mu = mobius_upto(m)
    return [1] + [int(k) for k in range(2, m + 1) if mu[k] != 0]


def mobius_distance(f: dict[int, float], keys: list[int]) -> dict[str, float]:
    """Originality signal: how far the LP's f is from plain truncated Mobius on the same K."""
    mu = mobius_upto(max(keys))
    diffs = [abs(f.get(k, 0.0) - float(mu[k])) for k in keys if k != 1]
    if not diffs:
        return {"linf": 0.0, "mean_abs": 0.0, "frac_changed": 0.0}
    arr = np.array(diffs)
    return {
        "linf": float(arr.max()),
        "mean_abs": float(arr.mean()),
        "frac_changed": float(np.mean(arr > 1e-6)),
    }


def evaluate_support(K: list[int]) -> tuple[LPResult, VerifyReport]:
    """Solve the LP on K and dual-verify the result."""
    res = solve_support(K)
    rep = dual_verify(res.f, K, res.S)
    return res, rep


def main(argv: list[str]) -> None:
    ms = [int(a) for a in argv] if argv else list(DEFAULT_LADDER)
    candidates: list[tuple[str, list[int]]] = []
    for m in ms:
        candidates.append((f"truncated-{m}", truncated_support(m)))
    # One squarefree variant sized to stay within the 2000-key cap.
    sf_m = ms[-1]
    sf = squarefree_support(sf_m)
    if len(sf) <= MAX_SUPPORT:
        candidates.append((f"squarefree-{sf_m}", sf))

    best: dict | None = None
    print(f"problem        : {SLUG}")
    print(f"arena top      : {ARENA_TOP:.6f}   theoretical max: {THEORETICAL_MAX:.1f} (Mobius)")
    print(
        f"{'support':<16}{'|K|':>6}{'S_lp':>12}{'S_server':>12}{'grid_ok':>9}"
        f"{'match':>7}{'submit':>8}"
    )
    for name, K in candidates:
        res, rep = evaluate_support(K)
        flag = (
            "BEATS"
            if res.S > ARENA_TOP + 1e-9
            else ("tie" if abs(res.S - ARENA_TOP) <= 1e-6 else "")
        )
        print(
            f"{name:<16}{len(K):>6}{res.S:>12.6f}{rep.s_server:>12.6f}"
            f"{str(rep.grid_feasible):>9}{str(rep.score_matches):>7}{str(rep.submittable):>8}  {flag}"
        )
        if rep.submittable and (best is None or res.S > best["S"]):
            best = {
                "problem": SLUG,
                "support": name,
                "n": len(K),
                "S": res.S,
                "S_server": rep.s_server,
                "arena_top": ARENA_TOP,
                "gap_to_top": res.S - ARENA_TOP,
                "gap_to_max": THEORETICAL_MAX - res.S,
                "grid_max": rep.grid_max,
                "tail_method": rep.tail_method,
                "tail_certified": rep.tail_certified,
                "mobius_distance": mobius_distance(res.f, K),
                "partial_function": {str(k): v for k, v in res.f.items() if k != 1},
            }

    if best is None:
        print("\nno submittable candidate found.")
        return

    md = best["mobius_distance"]
    print(f"\nbest submittable: {best['support']} (|K|={best['n']})")
    print(
        f"  S            : {best['S']:.6f}   gap to top: {best['gap_to_top']:+.6f}   "
        f"gap to max: {best['gap_to_max']:.6f}"
    )
    print(
        f"  originality  : Linf vs Mobius={md['linf']:.3f}, "
        f"frac changed={md['frac_changed']:.2%} (math-derived support; no SOTA ingested)"
    )
    RESULTS.mkdir(parents=True, exist_ok=True)
    (RESULTS / "best.json").write_text(json.dumps(best, ensure_ascii=False))
    print(f"  saved        : {RESULTS / 'best.json'}")
    print("\nNOTE: baseline only. Submission requires explicit approval (see CLAUDE.md).")


if __name__ == "__main__":
    main(sys.argv[1:])
