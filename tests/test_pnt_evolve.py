"""PNT evolution tests: determinism, operator invariants, monotone best, originality, LLM parse."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from arena.pnt_evolve import (  # noqa: E402
    EvolveConfig,
    Evaluated,
    _mobius_upto,
    _trim,
    add_keys,
    crossover,
    extend_max,
    fill_squarefree,
    remove_inactive,
    run_evolution,
)
from scripts.pnt_llm_propose import build_prompt, parse_supports  # noqa: E402
from scripts.solve_prime_number_theorem import mobius_distance  # noqa: E402

_TINY = EvolveConfig(
    islands=2, pop_per_island=3, children_per_island=3, max_generations=3, key_max=120, seed=0
)
_SEEDS = [frozenset(range(1, 31)), frozenset(range(1, 61))]


@pytest.mark.unit
def test_run_is_deterministic():
    """Same seed -> identical best genome and per-generation best-S trace."""
    r1 = run_evolution(_SEEDS, _TINY)
    r2 = run_evolution(_SEEDS, _TINY)
    assert r1.best.genome == r2.best.genome
    assert r1.best.S == r2.best.S
    assert [h["best_S"] for h in r1.history] == [h["best_S"] for h in r2.history]


@pytest.mark.unit
def test_best_is_monotone_non_decreasing():
    """Elitism guarantees the running best never regresses across generations."""
    hist = run_evolution(_SEEDS, _TINY).history
    series = [h["best_S"] for h in hist]
    assert all(b >= a - 1e-12 for a, b in zip(series, series[1:]))


@pytest.mark.unit
def test_trim_keeps_key_one_and_cap():
    trimmed = _trim(frozenset(range(2, 50)), cap=10)
    assert 1 in trimmed
    assert len(trimmed) == 10


@pytest.mark.unit
def test_operators_preserve_invariants():
    """Every operator returns a valid genome: 1 present, size <= cap, keys within [1, key_max]."""
    rng = np.random.default_rng(1)
    cfg = EvolveConfig(key_max=200, support_cap=50)
    mu = _mobius_upto(cfg.key_max)
    ev = Evaluated(
        genome=frozenset(range(1, 40)),
        S=0.5,
        success=True,
        f={k: 0.0 if k % 2 else 1.0 for k in range(1, 40)},
    )
    for child in (
        add_keys(ev.genome, rng, cfg),
        remove_inactive(ev.genome, ev.f, rng),
        extend_max(ev.genome, rng, cfg),
        fill_squarefree(ev.genome, mu, rng, cfg),
        crossover(_SEEDS[0], _SEEDS[1], rng, cfg),
    ):
        assert 1 in child
        assert len(child) <= cfg.support_cap
        assert all(1 <= k <= cfg.key_max for k in child)


@pytest.mark.unit
def test_remove_inactive_is_a_subset():
    rng = np.random.default_rng(2)
    g = frozenset(range(1, 30))
    f = {k: (0.0 if k > 15 else 1.0) for k in g}
    child = remove_inactive(g, f, rng)
    assert child <= g
    assert 1 in child


@pytest.mark.unit
def test_originality_flags_mobius_copy():
    """A pure-Mobius f has ~zero distance (a copy); a perturbed f is clearly different."""
    mu = _mobius_upto(60)
    keys = list(range(1, 61))
    copy_f = {k: float(mu[k]) for k in keys}
    md_copy = mobius_distance(copy_f, keys)
    assert md_copy["linf"] < 1e-9
    assert md_copy["frac_changed"] == pytest.approx(0.0)

    perturbed = {k: float(mu[k]) + (2.0 if k == 7 else 0.0) for k in keys}
    md_pert = mobius_distance(perturbed, keys)
    assert md_pert["linf"] >= 2.0 - 1e-9
    assert md_pert["frac_changed"] > 0.0


@pytest.mark.unit
def test_parse_supports_extracts_clamps_and_adds_one():
    """LLM output parsing: JSON array-of-arrays -> genomes, key 1 added, out-of-range dropped."""
    text = 'sure: [[2, 3, 5, 7], [4, 9, 9999, -1, "x"]]'
    genomes = parse_supports(text, key_max=1000)
    assert genomes[0] == frozenset({1, 2, 3, 5, 7})
    assert genomes[1] == frozenset({1, 4, 9})  # 9999 (>key_max) and -1/"x" dropped


@pytest.mark.unit
def test_parse_supports_handles_garbage():
    assert parse_supports("no json here") == []
    assert parse_supports("[[1, 2,]] broken") == []  # trailing comma -> JSONDecodeError -> []


@pytest.mark.unit
def test_build_prompt_mentions_count_and_best():
    archive = [Evaluated(frozenset({1, 2, 3}), 0.5, True), Evaluated(frozenset({1, 2}), 0.9, True)]
    prompt = build_prompt(archive, n_proposals=4, key_max=500)
    assert "Propose 4" in prompt
    assert "0.900000" in prompt  # the best archived S is summarized
    assert "500" in prompt
