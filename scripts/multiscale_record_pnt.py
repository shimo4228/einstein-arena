"""Solve, VERIFY, and record a multiscale PNT support candidate. NO submission (approval-gated).

The probe found multiscale-2000 (dense_upto=2300, target_reach=4500, reach 4501) honest @1.0 with
S=0.9955806 -- above arena top 0.994901 (which itself spends the 1.0001 slack). The probe did not
save f, so this driver deterministically rebuilds the support, re-solves (saving the full f this
time), then dual-verifies (server replay + exact grid @1.0) and measures originality vs OrganonAgent.

Run:  uv run python -m scripts.multiscale_record_pnt [dense_upto] [target_reach] [cap]
Writes results/prime-number-theorem/multiscale_record.json. Submits nothing.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from arena.pnt_lp import solve_support  # noqa: E402
from arena.pnt_seeds import arena_distance, multiscale_support  # noqa: E402
from arena.pnt_verify import dual_verify  # noqa: E402
from scripts.solve_prime_number_theorem import mobius_distance  # noqa: E402

ARENA_TOP = 0.994901
OUT = Path("results/prime-number-theorem/multiscale_record.json")
ARENA_COMPARE = Path("results/prime-number-theorem/arena_compare.json")


def _load_arena_top_f() -> dict[int, float] | None:
    if not ARENA_COMPARE.exists():
        return None
    sols = json.loads(ARENA_COMPARE.read_text()).get("solutions", [])
    if not sols:
        return None
    return {int(k): float(v) for k, v in sols[0]["partial_function"].items()}


def main() -> None:
    dense_upto = int(sys.argv[1]) if len(sys.argv) > 1 else 2300
    target_reach = int(sys.argv[2]) if len(sys.argv) > 2 else 4500
    cap = int(sys.argv[3]) if len(sys.argv) > 3 else 2000

    K = multiscale_support(cap, dense_upto=dense_upto, target_reach=target_reach)
    print(f"[support] multiscale cap={cap} dense_upto={dense_upto} target_reach={target_reach} "
          f"-> |K|={len(K)} reach={max(K)}", flush=True)

    t0 = time.time()
    res = solve_support(K)
    print(f"[solve] honest S={res.S:.7f} grid_max={res.grid_max:.8f} success={res.success} "
          f"status={res.status} ({time.time()-t0:.0f}s)", flush=True)

    t1 = time.time()
    rep = dual_verify(res.f, K, res.S)
    print(f"[verify] s_server={rep.s_server:.7f} score_matches={rep.score_matches} "
          f"grid_feasible={rep.grid_feasible} n_violations={rep.n_violations} "
          f"submittable={rep.submittable} tail={rep.tail_method}/{rep.tail_certified} "
          f"({time.time()-t1:.0f}s)", flush=True)

    arena_f = _load_arena_top_f()
    dist = arena_distance(res.f, arena_f) if arena_f else None
    md = mobius_distance(res.f, K)

    record = {
        "support": "multiscale-record",
        "params": {"dense_upto": dense_upto, "target_reach": target_reach, "cap": cap},
        "keys": len(K),
        "reach": int(max(K)),
        "S_lp": res.S,
        "S_server": rep.s_server,
        "score_matches": rep.score_matches,
        "grid_max": rep.grid_max,
        "grid_feasible": rep.grid_feasible,
        "n_violations": rep.n_violations,
        "tail_method": rep.tail_method,
        "tail_certified": rep.tail_certified,
        "submittable": rep.submittable,
        "honest_rhs": 1.0,
        "vs_arena_top": res.S - ARENA_TOP,
        "beats_arena_top_honestly": bool(rep.submittable and res.S > ARENA_TOP),
        "originality": {
            "vs_mobius": md,
            "vs_arena_organon": dist,  # support Jaccard + L-inf vs the actual competitor
        },
        "provenance": {
            "method": "multiscale support (dense squarefree prefix + geometric sparse tail), LP-optimal f",
            "seed_origin": "math-derived; OrganonAgent NOT an optimizer input (comparison only)",
            "honest_note": "RHS=1.0 grid-exact; arena top uses RHS 1.0001 slack",
        },
        "partial_function": {str(k): float(v) for k, v in res.f.items() if k != 1},
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(record, indent=2))

    print(f"\n[record] saved -> {OUT}", flush=True)
    if record["beats_arena_top_honestly"]:
        print(f">>> VERIFIED HONEST RECORD: S={res.S:.7f} > arena top {ARENA_TOP} "
              f"(+{res.S-ARENA_TOP:.6f}), server-confirmed, grid-feasible @1.0.", flush=True)
        if dist:
            print(f"    originality vs OrganonAgent: support_jaccard={dist['support_jaccard']:.3f} "
                  f"linf_shared={dist['linf_shared']:.3f} n_shared={int(dist['n_shared'])}", flush=True)
    else:
        print(">>> not a verified honest record (check submittable / score_matches / grid).", flush=True)
    print("\n[safety] NO submission performed. Submit only via the approval gate (CLAUDE.md).", flush=True)


if __name__ == "__main__":
    main()
