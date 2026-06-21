"""Evolution engine for `prime-number-theorem`: evolve the SUPPORT set, score by LP.

FunSearch-lite, algorithm-first (the LLM proposer is an optional, pluggable extra -- see
scripts/pnt_llm_propose.py). The genome is an integer support K (a frozenset, always with 1
and at most 2000 keys). Fitness is the LP-optimal S on that support (arena.pnt_lp.solve_support),
which already includes the exact integer-grid feasibility check -- an infeasible genome scores
-inf and dies. The expensive server replay (dual_verify) is run by the caller only on the final
best, never per genome.

Design choices:
- Island model with periodic migration to preserve support-structure diversity (avoid every
  island collapsing onto the same truncation).
- Pure mathematics drives every operator (add/remove keys, extend the range, fill squarefree
  windows, crossover). A competitor's solution is NEVER read into the optimizer -- originality
  is a hard invariant, checked separately by the caller.
- Fully deterministic given `EvolveConfig.seed`: a single numpy Generator is threaded through
  all draws in a fixed control-flow order, so the same seed reproduces the same run.

This module is engine-only: no disk, no network. I/O (seeds, logging, provenance, the final
dual_verify) lives in scripts/evolve_prime_number_theorem.py.
"""

from __future__ import annotations

import math
import time
from collections.abc import Callable
from dataclasses import dataclass, field

import numpy as np

from arena.pnt_lp import MAX_SUPPORT, solve_support

Genome = frozenset[int]
Proposer = Callable[[list["Evaluated"], np.random.Generator], list[Genome]]


@dataclass(frozen=True)
class EvolveConfig:
    """Knobs for one evolution run. Defaults are deliberately small (prototype scale)."""

    islands: int = 3
    pop_per_island: int = 6
    children_per_island: int = 6
    max_generations: int = 12
    no_improve_patience: int = 5
    migrate_every: int = 4
    key_max: int = 2000  # never add a key above this (bounds x_max = 10*max(K) cost)
    support_cap: int = MAX_SUPPORT
    target_S: float | None = None  # early-exit once best S reaches this (caller verifies)
    time_budget_s: float | None = None
    seed: int = 0


@dataclass(frozen=True)
class Evaluated:
    """A scored genome. `f` is the LP solution (used by operators and the final report)."""

    genome: Genome
    S: float
    success: bool
    f: dict[int, float] = field(default_factory=dict)


# --------------------------------------------------------------------------- evaluation


def evaluate_genome(genome: Genome, cache: dict[Genome, Evaluated]) -> Evaluated:
    """Solve the LP on `genome` (cached). Infeasible / non-converged genomes score -inf."""
    if genome in cache:
        return cache[genome]
    res = solve_support(genome)
    S = res.S if res.success else -math.inf
    ev = Evaluated(genome=genome, S=S, success=res.success, f=res.f)
    cache[genome] = ev
    return ev


# --------------------------------------------------------------------------- operators


def _trim(genome: Genome, cap: int) -> Genome:
    """Keep key 1 plus the smallest cap-1 keys (small k dominate the binding constraints)."""
    keys = sorted(genome | {1})
    if len(keys) <= cap:
        return frozenset(keys)
    rest = [k for k in keys if k != 1][: cap - 1]
    return frozenset([1, *rest])


def add_keys(
    genome: Genome, rng: np.random.Generator, cfg: EvolveConfig, n_max: int = 24
) -> Genome:
    """Insert up to n_max fresh keys sampled uniformly from {2..key_max} \\ genome."""
    pool = np.array([k for k in range(2, cfg.key_max + 1) if k not in genome], dtype=np.int64)
    if pool.size == 0:
        return genome
    n = int(rng.integers(1, min(n_max, pool.size) + 1))
    chosen = rng.choice(pool, size=n, replace=False)
    return _trim(frozenset(genome | {int(k) for k in chosen}), cfg.support_cap)


def remove_inactive(
    genome: Genome,
    f: dict[int, float],
    rng: np.random.Generator,
    eps: float = 1e-6,
    n_max: int = 24,
) -> Genome:
    """Drop keys the LP left inactive (|f(k)| ~ 0); frees budget for more useful keys."""
    inactive = sorted(k for k in genome if k != 1 and abs(f.get(k, 0.0)) < eps)
    pool = inactive or sorted(k for k in genome if k != 1)
    if not pool:
        return genome
    n = int(rng.integers(1, min(n_max, len(pool)) + 1))
    drop = {int(k) for k in rng.choice(np.array(pool, dtype=np.int64), size=n, replace=False)}
    return frozenset(genome - drop)


def extend_max(genome: Genome, rng: np.random.Generator, cfg: EvolveConfig) -> Genome:
    """Append a contiguous block above the current max key (push toward the budget)."""
    m = max(genome)
    block = int(rng.integers(10, 120))
    hi = min(cfg.key_max, m + block)
    if hi <= m:
        return genome
    return _trim(frozenset(genome | set(range(m + 1, hi + 1))), cfg.support_cap)


def fill_squarefree(
    genome: Genome, mu: np.ndarray, rng: np.random.Generator, cfg: EvolveConfig
) -> Genome:
    """Add the squarefree keys (where Mobius mu != 0) inside a random window."""
    m = max(genome)
    if m <= 1:  # singleton {1}: no window to fill (rng.integers(2, 2) would raise)
        return genome
    a = int(rng.integers(2, m + 1))
    b = min(cfg.key_max, a + int(rng.integers(10, 100)))
    add = {k for k in range(a, b + 1) if mu[k] != 0}
    return _trim(frozenset(genome | add), cfg.support_cap)


def crossover(a: Genome, b: Genome, rng: np.random.Generator, cfg: EvolveConfig) -> Genome:
    """Child = (a & b) + a random subset of the symmetric difference, plus key 1."""
    sym = np.array(sorted((a ^ b) - {1}), dtype=np.int64)
    pick: set[int] = set()
    if sym.size:
        n = int(rng.integers(0, sym.size + 1))
        if n:
            pick = {int(k) for k in rng.choice(sym, size=n, replace=False)}
    return _trim(frozenset((a & b) | pick | {1}), cfg.support_cap)


def mutate(ev: Evaluated, mu: np.ndarray, rng: np.random.Generator, cfg: EvolveConfig) -> Genome:
    """Apply one randomly-chosen mutation operator to a scored genome."""
    op = int(rng.integers(0, 4))
    if op == 0:
        return add_keys(ev.genome, rng, cfg)
    if op == 1:
        return remove_inactive(ev.genome, ev.f, rng)
    if op == 2:
        return extend_max(ev.genome, rng, cfg)
    return fill_squarefree(ev.genome, mu, rng, cfg)


# --------------------------------------------------------------------------- selection / loop


def _tournament(pop: list[Evaluated], rng: np.random.Generator, k: int = 3) -> Evaluated:
    idx = rng.integers(0, len(pop), size=min(k, len(pop)))
    return max((pop[int(i)] for i in idx), key=lambda e: e.S)


def _seed_islands(seeds: list[Evaluated], cfg: EvolveConfig) -> list[list[Evaluated]]:
    """Round-robin the evaluated seeds across islands (every island gets >=1)."""
    islands: list[list[Evaluated]] = [[] for _ in range(cfg.islands)]
    for i, ev in enumerate(seeds):
        islands[i % cfg.islands].append(ev)
    for isl in islands:
        if not isl:  # ensure no empty island
            isl.append(seeds[0])
    return islands


@dataclass(frozen=True)
class EvolveResult:
    best: Evaluated
    history: list[dict]
    n_evaluations: int
    stop_reason: str


def run_evolution(
    seed_genomes: list[Genome],
    cfg: EvolveConfig,
    *,
    proposer: Proposer | None = None,
    on_generation: Callable[[dict], None] | None = None,
    clock: Callable[[], float] = time.monotonic,
) -> EvolveResult:
    """Evolve `seed_genomes` under `cfg`. Returns the best Evaluated plus a per-generation log.

    `proposer` (optional) injects extra candidate genomes each generation (the LLM hook); it
    NEVER scores. `on_generation` receives a log dict per generation (kept here so I/O stays in
    the caller). `clock` is injectable for deterministic testing of the time budget.
    """
    rng = np.random.default_rng(cfg.seed)
    mu = _mobius_upto(cfg.key_max)
    cache: dict[Genome, Evaluated] = {}

    seeds = [evaluate_genome(_trim(g, cfg.support_cap), cache) for g in seed_genomes]
    islands = _seed_islands(seeds, cfg)
    best = max(seeds, key=lambda e: e.S)
    start = clock()
    history: list[dict] = []
    no_improve = 0
    stop_reason = "max_generations"

    for gen in range(1, cfg.max_generations + 1):
        archive_view: list[Evaluated] = []
        if proposer is not None:  # the LLM sees the GLOBAL best, not one island's local pop
            archive_view = sorted(
                (e for isl in islands for e in isl), key=lambda e: e.S, reverse=True
            )[: cfg.pop_per_island]
        for isl_idx, pop in enumerate(islands):
            children: list[Genome] = []
            for _ in range(cfg.children_per_island):
                if len(pop) >= 2 and rng.random() < 0.5:
                    a, b = _tournament(pop, rng), _tournament(pop, rng)
                    children.append(crossover(a.genome, b.genome, rng, cfg))
                else:
                    children.append(mutate(_tournament(pop, rng), mu, rng, cfg))
            if proposer is not None:
                children.extend(_trim(g, cfg.support_cap) for g in proposer(archive_view, rng))
            evaluated = [evaluate_genome(c, cache) for c in children]
            merged = pop + evaluated
            merged.sort(key=lambda e: e.S, reverse=True)
            islands[isl_idx] = merged[: cfg.pop_per_island]

        gen_best = max((isl[0] for isl in islands), key=lambda e: e.S)
        improved = gen_best.S > best.S + 1e-12
        best = gen_best if improved else best
        no_improve = 0 if improved else no_improve + 1
        elapsed = clock() - start

        record = {
            "gen": gen,
            "best_S": best.S,
            "best_n": len(best.genome),
            "gen_best_S": gen_best.S,
            "evaluations": len(cache),
            "elapsed_s": elapsed,
        }
        history.append(record)
        if on_generation is not None:
            on_generation(record)

        if cfg.target_S is not None and best.S >= cfg.target_S:
            stop_reason = "target_reached"
            break
        if no_improve >= cfg.no_improve_patience:
            stop_reason = "no_improvement"
            break
        if cfg.time_budget_s is not None and elapsed >= cfg.time_budget_s:
            stop_reason = "time_budget"
            break
        if cfg.migrate_every and gen % cfg.migrate_every == 0:
            _migrate(islands)

    return EvolveResult(
        best=best, history=history, n_evaluations=len(cache), stop_reason=stop_reason
    )


def _migrate(islands: list[list[Evaluated]]) -> None:
    """Copy each island's champion into the next island (ring migration)."""
    champions = [isl[0] for isl in islands]
    for i, isl in enumerate(islands):
        donor = champions[(i - 1) % len(islands)]
        if donor.genome not in {e.genome for e in isl}:
            isl.append(donor)
            isl.sort(key=lambda e: e.S, reverse=True)


def _mobius_upto(n: int) -> np.ndarray:
    """mu(0..n) via a linear sieve (duplicated from the solver to keep the engine self-contained)."""
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
