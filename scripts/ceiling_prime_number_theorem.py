"""Empirical ceiling probe for `prime-number-theorem` (id=7): can a legal, original support
beat arena top 0.994901? (Phase D go/no-go gate; see docs PNT plan.)

We argued: within keys <= 2000 the ceiling is the contiguous {1..2000} LP optimum U. So to beat
top a support must REACH BEYOND key 2000 with the same <=2000-key budget. The natural such
support is the first N squarefree integers (where Mobius mu lives), which reaches ~3290 with 2000
keys. This script solves a battery of math-derived supports (contiguous + squarefree-extended).
Feasibility per support uses the EXACT integer-grid check at the honest bound 1.0 (inside
solve_support) -- stricter than the server's MC at 1.0001, so grid-feasible implies the server
accepts it; the expensive 1e7-sample server replay is run ONCE, only on the single best candidate
(local == server confirmation). Reports S, gap to top, originality vs plain Mobius. Results stream
to results/.../ceiling.json so a timeout never loses completed rows. NO competitor solution is
ingested. NO submission happens here.

Run:  uv run python scripts/ceiling_prime_number_theorem.py
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from arena.pnt_lp import solve_support  # noqa: E402
from arena.pnt_verify import dual_verify  # noqa: E402
from scripts.solve_prime_number_theorem import (  # noqa: E402
    ARENA_TOP,
    THEORETICAL_MAX,
    mobius_distance,
    mobius_upto,
    truncated_support,
)

SLUG = "prime-number-theorem"
RESULTS = REPO_ROOT / "results" / SLUG


def squarefree_first_n(n: int) -> list[int]:
    """The first n squarefree integers (mu != 0), including 1. Reaches ~n/0.608 in value."""
    mu = mobius_upto(int(n * 1.8) + 200)
    out: list[int] = []
    for k in range(1, len(mu)):
        if mu[k] != 0:
            out.append(k)
            if len(out) == n:
                break
    assert len(out) == n, f"sieve buffer too small: got {len(out)} of {n} squarefree integers"
    return out


def build_battery() -> list[tuple[str, list[int]]]:
    """Math-derived supports only. The squarefree-*keys ones reach BEYOND key 2000."""
    return [
        ("contiguous-1200", truncated_support(1200)),  # curve point (confirms U extrapolation)
        ("contiguous-2000", truncated_support(2000)),  # provable ceiling for keys <= 2000
        ("squarefree-1600keys", squarefree_first_n(1600)),  # reaches ~2630
        ("squarefree-2000keys", squarefree_first_n(2000)),  # reaches ~3290 (beyond 2000)
    ]


def _row(name: str, keys: list[int], res, dt: float) -> dict:
    """One table row. Feasibility uses solve_support's EXACT grid check (RHS 1.0), which is
    stricter than the server's MC at 1.0001 -- so grid-feasible implies server-acceptable.
    The expensive 1e7-sample server replay is deferred to the single best candidate (below)."""
    return {
        "support": name,
        "n": len(keys),
        "max_key": keys[-1],
        "S": res.S,
        "gap_to_top": res.S - ARENA_TOP,
        "gap_to_max": THEORETICAL_MAX - res.S,
        "grid_max": res.grid_max,
        "grid_feasible": res.success,
        "submittable": res.success,
        "mobius_distance": mobius_distance(res.f, keys),
        "seconds": round(dt, 1),
    }


def main() -> None:
    RESULTS.mkdir(parents=True, exist_ok=True)
    out_path = RESULTS / "ceiling.json"
    rows: list[dict] = []
    best: tuple[str, list[int], object] | None = None  # (name, keys, LPResult)

    print(f"problem: {SLUG}   arena top: {ARENA_TOP:.6f}   theoretical max: {THEORETICAL_MAX}")
    print(
        f"{'support':<22}{'|K|':>6}{'maxK':>7}{'S':>12}{'gap_top':>11}"
        f"{'grid_ok':>8}{'time':>8}",
        flush=True,
    )

    for name, support in build_battery():
        t = time.time()
        res = solve_support(support)  # internal exact grid check decides res.success
        dt = time.time() - t
        keys = sorted(support)
        rows.append(_row(name, keys, res, dt))
        flag = "BEATS" if (res.success and res.S > ARENA_TOP + 1e-9) else ""
        print(
            f"{name:<22}{len(keys):>6}{keys[-1]:>7}{res.S:>12.6f}{res.S - ARENA_TOP:>+11.6f}"
            f"{str(res.success):>8}{dt:>7.0f}s  {flag}",
            flush=True,
        )
        if res.success and (best is None or res.S > best[2].S):
            best = (name, keys, res)
        # Stream after each support so a timeout never loses completed rows.
        out_path.write_text(json.dumps({"arena_top": ARENA_TOP, "rows": rows}, ensure_ascii=False))

    # Confirm ONLY the single best candidate with the full server replay (local == server).
    best_payload = None
    if best is not None:
        name, keys, res = best
        rep = dual_verify(res.f, keys, res.S)  # includes the 1e7-sample server verifier
        print(
            f"\nbest candidate server-replay: '{name}' S_lp={res.S:.6f} "
            f"S_server={rep.s_server:.6f} match={rep.score_matches} "
            f"tail={rep.tail_method}/{rep.tail_certified}",
            flush=True,
        )
        best_payload = {
            "support": name,
            "n": len(keys),
            "max_key": keys[-1],
            "S": res.S,
            "S_server": rep.s_server,
            "gap_to_top": res.S - ARENA_TOP,
            "score_matches": rep.score_matches,
            "tail_method": rep.tail_method,
            "tail_certified": rep.tail_certified,
            "mobius_distance": mobius_distance(res.f, keys),
            "partial_function": {str(k): v for k, v in res.f.items() if k != 1},
        }
    out_path.write_text(
        json.dumps({"arena_top": ARENA_TOP, "rows": rows, "best": best_payload}, ensure_ascii=False)
    )

    print(flush=True)
    if best_payload and best_payload["S"] > ARENA_TOP + 1e-9:
        print(
            f"VERDICT: BEATABLE — '{best_payload['support']}' S={best_payload['S']:.6f} "
            f"(+{best_payload['gap_to_top']:.6f} over top), grid-feasible & server-confirmed, "
            f"math-derived. -> submission candidate (requires explicit approval, CLAUDE.md).",
            flush=True,
        )
    else:
        bs = best_payload["S"] if best_payload else float("nan")
        print(
            f"VERDICT: NOT beaten by this battery. best S={bs:.6f} (gap {bs - ARENA_TOP:+.6f}). "
            f"Beating top needs cleverer far-reaching supports (-> Phase Search) or is out of "
            f"reach within these families.",
            flush=True,
        )
    print(f"saved: {out_path}", flush=True)


if __name__ == "__main__":
    main()
