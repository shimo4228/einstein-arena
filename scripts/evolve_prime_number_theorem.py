"""Evolve the SUPPORT set for `prime-number-theorem` (id=7), scoring each by LP (step 3).

FunSearch-lite over integer supports (arena.pnt_evolve). Seeds are math-derived only
(truncations and the squarefree filter); a competitor's solution is NEVER read into the
optimizer. The best evolved genome is dual-verified (server replay + exact integer-grid check)
before it is reported as submittable, and an originality note records its distance from plain
Mobius. Outputs go to results/prime-number-theorem/: evolve_best.json, provenance.json,
evolution_log.jsonl. NO submission happens here (see CLAUDE.md approval gate).

Run:  uv run python scripts/evolve_prime_number_theorem.py
      uv run python scripts/evolve_prime_number_theorem.py --generations 30 --key-max 1200
      uv run python scripts/evolve_prime_number_theorem.py --use-llm   # optional LLM proposer
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from arena.pnt_evolve import EvolveConfig, Proposer, run_evolution  # noqa: E402
from arena.pnt_verify import dual_verify  # noqa: E402
from scripts.solve_prime_number_theorem import (  # noqa: E402
    ARENA_TOP,
    THEORETICAL_MAX,
    mobius_distance,
    squarefree_support,
    truncated_support,
)

SLUG = "prime-number-theorem"
RESULTS = REPO_ROOT / "results" / SLUG
DEFAULT_SEED_MS = (30, 60, 120, 240)


def build_seeds(seed_ms: list[int], key_max: int) -> list[tuple[str, frozenset[int]]]:
    """Math-derived seed supports: contiguous truncations plus a squarefree variant."""
    seeds = [(f"truncated-{m}", frozenset(truncated_support(m))) for m in seed_ms]
    sf_m = min(max(seed_ms), key_max)
    seeds.append((f"squarefree-{sf_m}", frozenset(squarefree_support(sf_m))))
    return seeds


def load_llm_proposer(model: str) -> Proposer:
    """Lazily import the optional LLM proposer (keeps anthropic an optional dependency)."""
    from scripts.pnt_llm_propose import make_llm_proposer

    return make_llm_proposer(model=model)


def originality_report(f: dict[int, float], keys: list[int]) -> dict:
    """Distance from plain Mobius, plus a comparison to a SEPARATELY-stored arena copy if any.

    The arena reference (if present) is used for comparison ONLY and is never an optimizer input.
    """
    report = {"vs_mobius": mobius_distance(f, keys), "vs_arena": None}
    arena_file = RESULTS / "arena_compare.json"
    if arena_file.exists():
        arena = json.loads(arena_file.read_text()).get("partial_function", {})
        shared = set(int(k) for k in arena) & set(keys)
        union = set(int(k) for k in arena) | set(keys)
        jaccard = len(shared) / len(union) if union else 0.0
        report["vs_arena"] = {"support_jaccard": jaccard, "note": "comparison-only; not ingested"}
    return report


def main(argv: list[str]) -> None:
    p = argparse.ArgumentParser(description="Evolve PNT supports, score by LP.")
    p.add_argument("--generations", type=int, default=12)
    p.add_argument("--islands", type=int, default=3)
    p.add_argument("--pop", type=int, default=6)
    p.add_argument("--children", type=int, default=6)
    p.add_argument("--key-max", type=int, default=800)
    p.add_argument(
        "--patience", type=int, default=5, help="stop after N generations w/o improvement"
    )
    p.add_argument("--migrate-every", type=int, default=4)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--time-budget", type=float, default=None, help="seconds")
    p.add_argument("--target", type=float, default=None, help="early-exit S threshold")
    p.add_argument("--seed-ms", type=int, nargs="*", default=list(DEFAULT_SEED_MS))
    p.add_argument("--use-llm", action="store_true", help="enable the optional LLM proposer")
    p.add_argument("--llm-model", default="claude-opus-4-8")
    args = p.parse_args(argv)

    if not 2 <= args.key_max <= 50_000:  # x_max = 10*key_max bounds memory/compute
        p.error("--key-max must be in [2, 50000]")

    cfg = EvolveConfig(
        islands=args.islands,
        pop_per_island=args.pop,
        children_per_island=args.children,
        max_generations=args.generations,
        no_improve_patience=args.patience,
        migrate_every=args.migrate_every,
        key_max=args.key_max,
        target_S=args.target,
        time_budget_s=args.time_budget,
        seed=args.seed,
    )
    named_seeds = build_seeds(args.seed_ms, args.key_max)
    proposer = load_llm_proposer(args.llm_model) if args.use_llm else None

    RESULTS.mkdir(parents=True, exist_ok=True)
    log_path = RESULTS / "evolution_log.jsonl"

    print(f"problem: {SLUG}   arena top: {ARENA_TOP:.6f}   seeds: {[n for n, _ in named_seeds]}")
    print(
        f"config : islands={cfg.islands} pop={cfg.pop_per_island} children={cfg.children_per_island} "
        f"gens={cfg.max_generations} key_max={cfg.key_max} seed={cfg.seed} llm={args.use_llm}"
    )

    with log_path.open("w") as log_file:

        def on_generation(record: dict) -> None:
            log_file.write(json.dumps(record) + "\n")
            log_file.flush()
            print(
                f"gen {record['gen']:>3}  best_S={record['best_S']:.6f}  "
                f"|K|={record['best_n']:>4}  evals={record['evaluations']:>4}  "
                f"t={record['elapsed_s']:.0f}s"
            )

        result = run_evolution(
            [g for _, g in named_seeds], cfg, proposer=proposer, on_generation=on_generation
        )

    best = result.best
    keys = sorted(best.genome)
    rep = dual_verify(best.f, keys, best.S)
    flag = (
        "BEATS top"
        if best.S > ARENA_TOP + 1e-9
        else "tie" if abs(best.S - ARENA_TOP) <= 1e-6 else "below top"
    )
    orig = originality_report(best.f, keys)

    print(f"\nstop reason   : {result.stop_reason}   evaluations: {result.n_evaluations}")
    print(
        f"best evolved S: {best.S:.6f}   gap to top: {best.S - ARENA_TOP:+.6f}   "
        f"gap to max: {THEORETICAL_MAX - best.S:.6f}   ({flag})"
    )
    print(
        f"verify        : server={rep.s_server:.6f} match={rep.score_matches} "
        f"grid_ok={rep.grid_feasible} submittable={rep.submittable}"
    )
    print(
        f"originality   : Linf vs Mobius={orig['vs_mobius']['linf']:.3f}, "
        f"frac changed={orig['vs_mobius']['frac_changed']:.2%} (math-derived; no SOTA ingested)"
    )

    payload = {
        "problem": SLUG,
        "method": "evolution",
        "n": len(keys),
        "S": best.S,
        "S_server": rep.s_server,
        "arena_top": ARENA_TOP,
        "gap_to_top": best.S - ARENA_TOP,
        "gap_to_max": THEORETICAL_MAX - best.S,
        "grid_max": rep.grid_max,
        "tail_method": rep.tail_method,
        "tail_certified": rep.tail_certified,
        "submittable": rep.submittable,
        "originality": orig,
        "partial_function": {str(k): v for k, v in best.f.items() if k != 1},
    }
    (RESULTS / "evolve_best.json").write_text(json.dumps(payload, ensure_ascii=False))
    provenance = {
        "problem": SLUG,
        "method": "FunSearch-lite over integer supports (algorithm-first)",
        "config": vars(args),
        "seeds": [{"name": n, "size": len(g)} for n, g in named_seeds],
        "seed_origin": "math-derived only (truncations + squarefree filter); NO competitor "
        "solution was used as optimizer input",
        "stop_reason": result.stop_reason,
        "evaluations": result.n_evaluations,
        "best_S": best.S,
        "originality": orig,
    }
    (RESULTS / "provenance.json").write_text(json.dumps(provenance, ensure_ascii=False, indent=2))

    if rep.submittable:
        baseline = RESULTS / "best.json"
        prev = json.loads(baseline.read_text())["S"] if baseline.exists() else -1.0
        if best.S > prev:
            baseline.write_text(json.dumps(payload, ensure_ascii=False))
            print(f"  updated best.json (evolution {best.S:.6f} > previous {prev:.6f})")
    print(f"saved         : {RESULTS / 'evolve_best.json'}, provenance.json, evolution_log.jsonl")
    print("\nNOTE: baseline only. Submission requires explicit approval (see CLAUDE.md).")


if __name__ == "__main__":
    main(sys.argv[1:])
