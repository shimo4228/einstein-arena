"""Solve/load, VERIFY, and record a multiscale PNT candidate. NO submission (approval-gated).

v2 of the record driver. Two ways to obtain the candidate f:

  --from-sweep <jsonl> [--cell D,R]   load f from a pnt_reach_sweep ledger cell and go straight
                                      to dual_verify (the honest claim rests on the independent
                                      exact-grid + server-replay check, not on how f was produced;
                                      warm==cold is separately pinned by the sweep's --cold-check)
  positional [dense_upto] [target_reach] [cap]   deterministic rebuild + cold LP re-solve
                                      (also forced by --resolve on top of --from-sweep)

Verification and record: dual_verify (verbatim server verifier replay + exact integer grid at the
HONEST bound 1.0), originality vs the top arena competitors (arena_compare.json snapshot; refresh
it first via scripts/fetch_pnt_arena_compare.py), Mobius distance, deficit coefficient, and the
full partial_function. Writes --out (default multiscale_record_v2.json; v1 multiscale_record.json
is the frozen 2026-06-21 #1 artifact — do not overwrite it).

Run:  uv run python -m scripts.multiscale_record_pnt --from-sweep results/prime-number-theorem/reach_sweep.jsonl --cell 2300,48000
"""

from __future__ import annotations

import argparse
import json
import math
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from arena.pnt_lp import MAX_SUPPORT, VALUE_BOUND, solve_support  # noqa: E402
from arena.pnt_seeds import arena_distance, multiscale_support  # noqa: E402
from arena.pnt_verify import dual_verify  # noqa: E402
from scripts.pnt_reach_sweep import cell_key, load_done  # noqa: E402
from scripts.solve_prime_number_theorem import mobius_distance  # noqa: E402

ARENA_TOP_FALLBACK = 0.9965177307
"""CHRONOS, 2026-07-03. Used only if the arena_compare.json snapshot is missing."""

OUT_DEFAULT = Path("results/prime-number-theorem/multiscale_record_v2.json")
ARENA_COMPARE = Path("results/prime-number-theorem/arena_compare.json")
N_ORIGINALITY = 3


def _load_top_solutions(n: int = N_ORIGINALITY) -> list[dict[str, Any]]:
    """Best solution per distinct agent from the snapshot, ordered by score (top n agents).

    The snapshot is an UNTRUSTED cache of API data: drop entries with non-finite scores
    (json.loads parses bare NaN/Infinity into real floats, which would poison max()/sorting
    and the beat-the-top gate downstream) or a malformed partial_function.
    """
    if not ARENA_COMPARE.exists():
        return []
    sols = json.loads(ARENA_COMPARE.read_text()).get("solutions", [])
    best_by_agent: dict[str, dict[str, Any]] = {}
    for sol in sols:
        if not isinstance(sol, dict):
            continue
        name = str(sol.get("agentName"))
        score = sol.get("score")
        if not isinstance(score, (int, float)) or not math.isfinite(score):
            continue
        if not isinstance(sol.get("partial_function"), dict):
            continue
        if name not in best_by_agent or score > best_by_agent[name]["score"]:
            best_by_agent[name] = sol
    ranked = sorted(best_by_agent.values(), key=lambda s: -float(s["score"]))
    return ranked[:n]


def _pf_to_f(pf: dict[str, Any]) -> dict[int, float]:
    """Tolerant coercion of an untrusted partial_function; skips malformed entries."""
    out: dict[int, float] = {}
    for k, v in pf.items():
        try:
            ki, vf = int(k), float(v)
        except (TypeError, ValueError):
            continue
        if math.isfinite(vf):
            out[ki] = vf
    return out


def _git_commit() -> str | None:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            timeout=10,
            check=True,
        )
        return out.stdout.strip()
    except Exception:  # noqa: BLE001 -- provenance nicety, never fatal
        return None


def _pick_sweep_cell(
    path: Path, cell: str | None, default_cap: int
) -> tuple[dict[str, Any], tuple[int, int, int]]:
    done = load_done(path)
    if not done:
        sys.exit(f"[abort] no successful cells in {path}")
    if cell:
        parts = [int(t) for t in cell.split(",")]
        if len(parts) == 2:
            parts.append(default_cap)
        if len(parts) != 3:
            sys.exit("[abort] --cell expects D,R or D,R,CAP")
        key = (parts[0], parts[1], parts[2])
        if key not in done:
            have = ", ".join(f"{k[0]},{k[1]},{k[2]}" for k in sorted(done))
            sys.exit(f"[abort] cell {key} not in ledger; have: {have}")
        return done[key], key
    best = max(done.values(), key=lambda rec: float(rec["S"]))
    return best, cell_key(best)


def _cold_solve(
    dense_upto: int, target_reach: int, cap: int
) -> tuple[dict[int, float], list[int], float]:
    """Deterministic support rebuild + trusted cold LP solve; returns (f_{k>=2}, K, S)."""
    K = multiscale_support(cap, dense_upto=dense_upto, target_reach=target_reach)
    print(
        f"[support] multiscale cap={cap} dense_upto={dense_upto} "
        f"target_reach={target_reach} -> |K|={len(K)} reach={max(K)}",
        flush=True,
    )
    t0 = time.time()
    res = solve_support(K)
    print(
        f"[solve] honest S={res.S:.7f} grid_max={res.grid_max:.10f} "
        f"success={res.success} status={res.status} ({time.time() - t0:.0f}s)",
        flush=True,
    )
    if not res.success:
        sys.exit("[abort] cold solve did not converge grid-feasible")
    return {k: v for k, v in res.f.items() if k != 1}, K, res.S


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("dense_upto", nargs="?", type=int, default=2300)
    ap.add_argument("target_reach", nargs="?", type=int, default=4500)
    ap.add_argument("cap", nargs="?", type=int, default=2000)
    ap.add_argument("--from-sweep", type=Path, default=None, metavar="JSONL")
    ap.add_argument(
        "--cell",
        default=None,
        metavar="D,R[,CAP]",
        help="which sweep cell (default: highest-S successful cell)",
    )
    ap.add_argument(
        "--resolve",
        action="store_true",
        help="cold re-solve the support even when loading from a sweep ledger",
    )
    ap.add_argument("--out", type=Path, default=OUT_DEFAULT)
    args = ap.parse_args(argv)

    if (
        args.out.resolve()
        == Path("results/prime-number-theorem/multiscale_record.json").resolve()
    ):
        sys.exit(
            "[abort] refusing to overwrite the frozen v1 record; choose another --out"
        )

    engine: dict[str, Any] = {}
    if args.from_sweep:
        rec_cell, (dense_upto, target_reach, cap) = _pick_sweep_cell(
            args.from_sweep, args.cell, args.cap
        )
        f = {int(k): float(v) for k, v in rec_cell["partial_function"].items()}
        K = sorted({1, *f})
        S_lp = float(rec_cell["S"])
        engine = {
            k: rec_cell.get(k)
            for k in (
                "seed_rows",
                "cuts_per_iter",
                "drop_slack",
                "n_rows_final",
                "n_iters",
                "wall_s",
                "warm",
                "cold_check",
            )
        }
        print(
            f"[load] sweep cell D={dense_upto} R={target_reach} cap={cap} "
            f"S={S_lp:.7f} ({len(f)} keys)",
            flush=True,
        )
        if args.resolve:
            f2, K2, S2 = _cold_solve(dense_upto, target_reach, cap)
            if abs(S2 - S_lp) > 1e-7:
                sys.exit(
                    f"[abort] cold re-solve S={S2!r} disagrees with sweep S={S_lp!r}"
                )
            f, K, S_lp = f2, K2, S2
    else:
        dense_upto, target_reach, cap = args.dense_upto, args.target_reach, args.cap
        f, K, S_lp = _cold_solve(dense_upto, target_reach, cap)

    # ---- submission-shape guards (desync detectors, cheap) ----
    if not f:
        sys.exit("[abort] empty partial_function (degenerate support, e.g. cap<=1)")
    if len(f) + 1 > MAX_SUPPORT:
        sys.exit(
            f"[abort] {len(f)} submitted keys + key 1 exceeds MAX_SUPPORT={MAX_SUPPORT}"
        )
    worst = max(abs(v) for v in f.values())
    if worst > VALUE_BOUND + 1e-12:
        sys.exit(f"[abort] |f| = {worst} exceeds the verifier box {VALUE_BOUND}")

    t1 = time.time()
    rep = dual_verify(f, K, S_lp)
    print(
        f"[verify] s_server={rep.s_server:.7f} score_matches={rep.score_matches} "
        f"grid_feasible={rep.grid_feasible} n_violations={rep.n_violations} "
        f"submittable={rep.submittable} tail={rep.tail_method}/{rep.tail_certified} "
        f"({time.time() - t1:.0f}s)",
        flush=True,
    )

    top_sols = _load_top_solutions()
    arena_top = float(top_sols[0]["score"]) if top_sols else ARENA_TOP_FALLBACK
    originality_arena = {
        str(sol["agentName"]): arena_distance(f, _pf_to_f(sol["partial_function"]))
        for sol in top_sols
    }
    reach = int(max(K))

    record = {
        "schema_version": 2,
        "support": "multiscale-record",
        "params": {"dense_upto": dense_upto, "target_reach": target_reach, "cap": cap},
        "keys": len(K),
        "reach": reach,
        "S_lp": S_lp,
        "S_server": rep.s_server,
        "score_matches": rep.score_matches,
        "grid_max": rep.grid_max,
        "grid_margin_honest": 1.0 - rep.grid_max,
        "grid_margin_server_slack": 1.0001 - rep.grid_max,
        "grid_feasible": rep.grid_feasible,
        "n_violations": rep.n_violations,
        "tail_method": rep.tail_method,
        "tail_certified": rep.tail_certified,
        "submittable": rep.submittable,
        "honest_rhs": 1.0,
        "coeff_c": (1.0 - S_lp) * math.log(10 * reach),
        "arena_top": arena_top,
        "vs_arena_top": S_lp - arena_top,
        "beats_arena_top_honestly": bool(rep.submittable and S_lp > arena_top),
        "leaderboard_snapshot": [
            {"agentName": s["agentName"], "score": s["score"]} for s in top_sols
        ],
        "engine": engine,
        "git_commit": _git_commit(),
        "originality": {
            "vs_mobius": mobius_distance(f, K),
            "vs_arena_top_agents": originality_arena,
        },
        "provenance": {
            "method": "multiscale support (dense squarefree prefix + geometric sparse tail), "
            "LP-optimal f, cut-drop warm engine",
            "seed_origin": "math-derived; competitor solutions are NEVER optimizer inputs "
            "(distance comparison only)",
            "honest_note": "RHS=1.0 grid-exact; current arena top spends the 1.0001 slack",
            "tail_note": "x >= 10*max(K) is outside the server-scored domain; geometric tails "
            "generally leave it residue-bounded but uncertified",
        },
        "partial_function": {str(k): float(v) for k, v in sorted(f.items())},
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    if args.out.exists():  # one-generation soft-delete instead of silent clobber
        backup = args.out.with_suffix(args.out.suffix + ".bak")
        args.out.replace(backup)
        print(f"[record] previous {args.out.name} moved to {backup.name}", flush=True)
    args.out.write_text(json.dumps(record, indent=2))

    print(f"\n[record] saved -> {args.out}", flush=True)
    if record["beats_arena_top_honestly"]:
        print(
            f">>> VERIFIED HONEST RECORD: S={S_lp:.7f} > arena top {arena_top:.7f} "
            f"(+{S_lp - arena_top:.6f}), server-confirmed, grid-feasible @1.0.",
            flush=True,
        )
        for name, dist in originality_arena.items():
            print(
                f"    vs {name}: support_jaccard={dist['support_jaccard']:.3f} "
                f"linf_shared={dist['linf_shared']:.3f} n_shared={int(dist['n_shared'])}",
                flush=True,
            )
    else:
        print(
            ">>> not a verified honest record vs the snapshot top "
            "(check submittable / score_matches / grid / arena_top).",
            flush=True,
        )
    print(
        "\n[safety] NO submission performed. Submit only via the approval gate (CLAUDE.md).",
        flush=True,
    )


if __name__ == "__main__":
    main()
