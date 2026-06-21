"""Seeds, candidate pools, and originality distance for the PNT column-generation engine.

Pure helpers (no network, no disk). A local squarefree sieve keeps arena/ independent of scripts/.

The "multiscale" seed is the originality lever vs the arena top (OrganonAgent): a dense squarefree
prefix (small k carry the binding low-x constraints) plus a geometric sparse tail (far k cheaply
extend reach). At a FIXED key budget this reaches much further than first-N-squarefree, which the
probe showed lifts honest S (see docs plan).
"""

from __future__ import annotations

import math
from collections.abc import Mapping

import numpy as np

_SQUAREFREE_DENSITY = 6.0 / math.pi**2  # ~0.6079


def squarefree_flags(bound: int) -> np.ndarray:
    """Boolean array is_sf[0..bound]; is_sf[k] True iff k is squarefree (is_sf[1]=True, is_sf[0]=False)."""
    is_sf = np.ones(bound + 1, dtype=bool)
    if bound >= 0:
        is_sf[0] = False
    d = 2
    while d * d <= bound:
        if is_sf[d]:  # only prime d adds new marks; composite d^2 multiples are already covered
            is_sf[d * d :: d * d] = False
        d += 1
    return is_sf


def squarefree_upto(bound: int) -> list[int]:
    """Sorted squarefree integers in [1, bound] (includes 1)."""
    flags = squarefree_flags(bound)
    return [int(k) for k in np.nonzero(flags)[0]]


def squarefree_first_n(n: int) -> list[int]:
    """The first n squarefree integers (includes 1). Reaches ~n / 0.608 in value."""
    if n <= 0:
        return []
    bound = int(n / _SQUAREFREE_DENSITY * 1.15) + 100
    out = squarefree_upto(bound)
    while len(out) < n:  # extremely unlikely; widen the sieve if the density estimate undershot
        bound *= 2
        out = squarefree_upto(bound)
    return out[:n]


def multiscale_support(cap: int, *, dense_upto: int, target_reach: int) -> list[int]:
    """Dense squarefree prefix (<= dense_upto) + geometric sparse squarefree tail up to target_reach.

    Tail keys satisfy k_{i+1} ~= ratio * k_i with ratio chosen to hit target_reach using the
    remaining budget. Returns <= cap sorted unique squarefree keys (includes 1).
    """
    if dense_upto < 1 or target_reach < dense_upto:
        raise ValueError("require 1 <= dense_upto <= target_reach")
    # Sieve a margin BEYOND target_reach: the geometric tail's last step needs a squarefree key
    # at/above target_reach, which may sit just past it (else the tail stops one key short).
    sf = squarefree_upto(int(target_reach * 1.1) + 200)
    dense = [k for k in sf if k <= dense_upto]
    if len(dense) >= cap:
        return dense[:cap]
    after = [k for k in sf if k > dense_upto]
    n_tail = cap - len(dense)
    if not after or n_tail <= 0:
        return dense
    ratio = (target_reach / dense_upto) ** (1.0 / n_tail)
    out = list(dense)
    target = float(dense_upto)
    idx = 0
    for _ in range(n_tail):
        target *= ratio
        while idx < len(after) and (after[idx] <= out[-1] or after[idx] < target):
            idx += 1
        if idx >= len(after):
            break
        out.append(after[idx])
        idx += 1
    return out


def candidate_pool(pool_max: int, *, include_nonsquarefree: bool = True) -> list[int]:
    """Pool of candidate key-columns in [2, pool_max] (key 1 is never a column).

    Squarefree keys always; non-squarefree (4,8,9,12,...) optionally. The LP, not a heuristic,
    decides whether non-squarefree keys price in (the team's "mu=0 -> wasted" assumption is untested).
    """
    flags = squarefree_flags(pool_max)
    if include_nonsquarefree:
        return list(range(2, pool_max + 1))
    return [int(k) for k in range(2, pool_max + 1) if flags[k]]


def arena_distance(f: Mapping[int, float], arena_f: Mapping[int, float]) -> dict[str, float]:
    """Originality signal vs a competitor solution (e.g. OrganonAgent), over k>=2 (key 1 is dependent).

    support_jaccard: |K_f ∩ K_a| / |K_f ∪ K_a|  (1.0 == identical support, lower == more original).
    linf_shared / mean_abs_shared: max / mean |f(k) - arena_f(k)| over shared keys (0 if none shared).
    """
    kf = {int(k) for k in f if int(k) != 1}
    ka = {int(k) for k in arena_f if int(k) != 1}
    union = kf | ka
    shared = kf & ka
    jacc = len(kf & ka) / len(union) if union else 1.0
    if shared:
        diffs = [abs(float(f[k]) - float(arena_f[k])) for k in shared]
        linf = max(diffs)
        mean_abs = sum(diffs) / len(diffs)
    else:
        linf = 0.0
        mean_abs = 0.0
    return {
        "support_jaccard": jacc,
        "linf_shared": linf,
        "mean_abs_shared": mean_abs,
        "n_shared": float(len(shared)),
    }
