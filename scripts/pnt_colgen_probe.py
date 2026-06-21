"""PROTOTYPE / SPIKE (not production): go/no-go for the "smarter support" hypothesis.

Plan: `replicated-singing-sprout.md`, section "go/no-go 実験".

The LP (`solve_support`) is OPTIMAL given a support, so the entire game is support selection.
Both the team and OrganonAgent (arena top) used squarefree-first-N supports. Central question:

    At a FIXED key budget N, can a different support STRUCTURE beat squarefree-first-N honestly?

We test "multiscale": a dense squarefree prefix (small k, where the g(x)<=1 constraint is tightest)
plus a geometrically-spaced sparse tail (far k, to extend reach cheaply). If multiscale beats
squarefree-first at small N, the "smarter support" frame has legs -> build the column-generation
engine. If squarefree-first is unbeatable, the only lever is budget/reach + LP speed (re-plan).

Uses ONLY the trusted `solve_support` (RHS=1.0, exact-grid-feasible). No dual/pricing machinery
(that lives in the production engine under TDD; this probe stays bug-proof and decisive).
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from arena.pnt_lp import solve_support  # noqa: E402
from scripts.solve_prime_number_theorem import mobius_upto  # noqa: E402

ARENA_TOP = 0.994901  # OrganonAgent, reach 3498, RHS 1.0001 (reference)


def squarefree_list(bound: int) -> list[int]:
    """Sorted squarefree integers in [1, bound] (includes 1, since mu(1)=1)."""
    mu = mobius_upto(bound)
    return [k for k in range(1, bound + 1) if mu[k] != 0]


def multiscale_support(
    cap: int, dense_upto: int, target_reach: int, sf: list[int]
) -> list[int]:
    """Dense squarefree prefix (<= dense_upto) + geometric sparse squarefree tail up to target_reach.

    The tail keys are spaced so key_{i+1} ~= ratio * key_i, ratio chosen to hit target_reach with
    the remaining budget. All keys squarefree (mu != 0). Returns <= cap sorted unique keys.
    """
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


def report(name: str, K: list[int]) -> tuple[float, bool]:
    t = time.time()
    res = solve_support(K)
    print(
        f"  {name:20s} |K|={len(K):4d} reach={max(K):6d}  honest S={res.S:.6f}  "
        f"feasible@1.0={res.success}  status={res.status}  ({time.time()-t:.1f}s)",
        flush=True,
    )
    return res.S, res.success


def main() -> None:
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 600
    reach_b = int(sys.argv[2]) if len(sys.argv) > 2 else 2500
    dense_upto = round(1.15 * n)  # ~0.7*N dense squarefree keys, rest is the sparse tail

    print(f"Budget N={n} keys; compare honest (RHS=1.0) S across support structures.")
    print(f"(reference: arena top {ARENA_TOP} uses 2000 keys reach 3498 @ RHS 1.0001)\n", flush=True)

    sf = squarefree_list(max(40000, reach_b * 4))
    a = sf[:n]
    b = multiscale_support(n, dense_upto=dense_upto, target_reach=reach_b, sf=sf)
    print(f"[A] squarefree-first-{n}: reach {a[-1]}")
    print(
        f"[B] multiscale: dense sf<= {dense_upto} ({sum(1 for k in b if k <= dense_upto)} keys) "
        f"+ sparse tail to ~{reach_b}: |K|={len(b)} reach {b[-1]}\n",
        flush=True,
    )

    sa, _ = report("squarefree-first", a)
    sb, fb = report("multiscale", b)
    print()
    if fb and sb > sa + 1e-6:
        print(
            f"  >>> GO: multiscale beats squarefree-first by {sb-sa:+.6f} at N={n} "
            f"(reach {a[-1]}->{b[-1]}). Smarter support helps -> build the colgen engine.",
            flush=True,
        )
    elif fb:
        print(
            f"  >>> squarefree-first wins/ties ({sb-sa:+.6f}) at N={n}. Sparse-tail reach does NOT "
            "beat density here. Re-plan: lever is budget/reach + LP speed, not structure "
            "(try other reach_b before concluding).",
            flush=True,
        )
    else:
        print("  >>> multiscale infeasible at this reach; lower target_reach and retry.", flush=True)


if __name__ == "__main__":
    main()
