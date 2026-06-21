"""Seed / candidate-pool / originality-distance helpers for the column-generation engine."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from arena.pnt_seeds import (  # noqa: E402
    arena_distance,
    candidate_pool,
    multiscale_support,
    squarefree_first_n,
)


def _is_squarefree(n: int) -> bool:
    d = 2
    while d * d <= n:
        if n % (d * d) == 0:
            return False
        d += 1
    return True


@pytest.mark.unit
def test_squarefree_first_n_count_and_content():
    out = squarefree_first_n(10)
    assert len(out) == 10
    assert out[0] == 1
    assert out == sorted(out)
    assert all(_is_squarefree(k) for k in out)
    # first squarefree: 1,2,3,5,6,7,10,11,13,14 (4,8,9,12 skipped)
    assert out == [1, 2, 3, 5, 6, 7, 10, 11, 13, 14]


@pytest.mark.unit
def test_multiscale_dense_prefix_then_sparse_tail():
    K = multiscale_support(120, dense_upto=80, target_reach=2000)
    assert len(K) <= 120
    assert K == sorted(set(K))
    assert all(_is_squarefree(k) for k in K)
    # dense prefix present in full
    dense = [k for k in K if k <= 80]
    assert dense == [k for k in squarefree_first_n(200) if k <= 80]
    # tail strictly increasing and reaches far beyond the dense region
    tail = [k for k in K if k > 80]
    assert tail == sorted(tail)
    assert max(K) > 1000  # sparse tail extended reach


@pytest.mark.unit
def test_multiscale_reach_exceeds_squarefree_first_at_equal_budget():
    """The whole point: at the same key count, multiscale reaches much further than squarefree-first."""
    n = 200
    ms = multiscale_support(n, dense_upto=160, target_reach=3000)
    sf = squarefree_first_n(n)
    assert len(ms) <= n
    assert max(ms) > max(sf)


@pytest.mark.unit
def test_candidate_pool_squarefree_only_vs_with_nonsquarefree():
    sf_only = candidate_pool(100, include_nonsquarefree=False)
    withns = candidate_pool(100, include_nonsquarefree=True)
    assert all(_is_squarefree(k) for k in sf_only)
    assert all(k >= 2 for k in sf_only)  # key 1 is never a candidate column
    assert set(sf_only).issubset(set(withns))
    assert len(withns) > len(sf_only)
    # the extra members are exactly the non-squarefree integers in [2,100]
    extra = set(withns) - set(sf_only)
    assert extra == {k for k in range(2, 101) if not _is_squarefree(k)}


@pytest.mark.unit
def test_arena_distance_identical_is_zero():
    f = {1: 0.5, 2: -1.0, 3: 0.3}
    d = arena_distance(f, f)
    assert d["support_jaccard"] == pytest.approx(1.0)
    assert d["linf_shared"] == pytest.approx(0.0)


@pytest.mark.unit
def test_arena_distance_disjoint_and_perturbed():
    f = {2: -1.0, 3: 0.5, 5: 0.2}
    arena = {2: -0.4, 3: 0.5, 7: 1.0}  # shares {2,3}, differs on 2 by 0.6
    d = arena_distance(f, arena)
    # union {2,3,5,7}=4, intersection {2,3}=2 -> Jaccard 0.5
    assert d["support_jaccard"] == pytest.approx(0.5)
    assert d["linf_shared"] == pytest.approx(0.6, abs=1e-9)
