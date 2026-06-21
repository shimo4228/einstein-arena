"""Column generation over key-columns for the PNT LP: let the dual price which keys to add.

The LP is optimal given a support, so the only freedom is support selection. This grows a support
by repeatedly (1) converging the rows for trustworthy duals, (2) pricing candidate keys by reduced
cost, (3) adding the best-priced batch, (4) re-solving to ground truth. Because adding a column can
only relax the optimum, the grow phase accepts every re-solve (monotone) and stops on diminishing
returns / budget / cap / pool exhaustion / time.

Reduced cost note (load-bearing): with c_k=log k/k>0, row duals <=0, and a(x,k)=floor(x/k)-x/k<=0,
rc(k')=c_{k'}+sum_x dual[x]*a(x,k') is structurally > 0 for EVERY candidate, so "rc<0" is not the
improvement test here -- below budget every key helps a little (LP monotonicity). rc is used only as
a RANKING (largest first = steepest first-order improvement); the LP+grid re-solve is ground truth.
See memory pnt-colgen-lp-pitfalls.
"""

from __future__ import annotations

import time
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field

import numpy as np

from arena.pnt_lp import RHS, _constraint_rows
from arena.pnt_warm import WarmLP

MAX_SUPPORT = 2000


@dataclass(frozen=True)
class ColgenConfig:
    """Bounds and tolerances for column generation. Every loop has an explicit stop."""

    rhs: float = RHS
    cap: int = MAX_SUPPORT
    cols_in_per_iter: int = 50
    max_outer_iters: int = 400
    budget_resolves: int = 600
    min_gain: float = 1e-7
    patience: int = 6
    rc_tol: float = 1e-12
    row_max_iters: int = 60
    time_budget_s: float | None = None
    pool_batch: int = 4000


@dataclass(frozen=True)
class ColgenResult:
    f: dict[int, float]
    S: float
    grid_max: float
    support: list[int]  # k>=2, sorted
    success: bool
    stop_reason: str
    n_resolves: int
    history: list[float] = field(default_factory=list)


def _price(
    cand: np.ndarray, row_x: np.ndarray, row_dual: np.ndarray, *, batch: int = 4000
) -> np.ndarray:
    """rc(k') = c_{k'} + sum_x row_dual[x]*a(x,k'),  c=log k'/k', a=floor(x/k')-x/k' (dual signed)."""
    c = np.log(cand) / cand
    out = np.empty(cand.shape[0], dtype=np.float64)
    for i in range(0, cand.shape[0], batch):
        chunk = cand[i : i + batch]
        a = _constraint_rows(row_x, chunk)  # (n_rows, n_chunk)
        out[i : i + chunk.shape[0]] = c[i : i + chunk.shape[0]] + row_dual @ a
    return out


def colgen_support(
    seed: Iterable[int],
    pool: Iterable[int],
    cfg: ColgenConfig = ColgenConfig(),
    *,
    clock: Callable[[], float] = time.time,
) -> ColgenResult:
    """Grow `seed` (<= cfg.cap keys) by dual-priced keys drawn from `pool`, honest at cfg.rhs.

    Deterministic: candidates are priced and ranked by reduced cost (stable sort); the same
    seed/pool/cfg yields the same support. Submits nothing; pure compute.
    """
    lp = WarmLP([k for k in seed if int(k) != 1], rhs=cfg.rhs)
    best = lp.row_generate_to_feasible(max_iters=cfg.row_max_iters)
    n_resolves = 1
    history = [best.S]
    pool_set = sorted({int(k) for k in pool if int(k) != 1})
    no_improve = 0
    stop = "max_outer_iters"
    t0 = clock()

    for _ in range(1, cfg.max_outer_iters + 1):
        if len(lp.kplus) >= cfg.cap:
            stop = "budget_full"
            break
        if not best.success:
            # f=0 is always LP-feasible, so a failure here is a solver error or a row-gen timeout,
            # never true infeasibility -- distinguish the two for the caller.
            stop = "infeasible" if best.status.startswith("highs:") else "row_max_iters"
            break
        if n_resolves >= cfg.budget_resolves:
            stop = "budget_resolves"
            break
        if cfg.time_budget_s is not None and clock() - t0 >= cfg.time_budget_s:
            stop = "time"
            break

        row_x, row_dual = lp.duals()
        held = set(lp.kplus)
        cand = np.asarray([k for k in pool_set if k not in held], dtype=np.int64)
        if cand.size == 0:
            stop = "pool_exhausted"
            break

        rc = _price(cand, row_x, row_dual, batch=cfg.pool_batch)
        room = cfg.cap - len(lp.kplus)
        k_add = min(cfg.cols_in_per_iter, room)
        order = np.argsort(-rc, kind="stable")[:k_add]
        entering = [int(cand[i]) for i in order if rc[i] > cfg.rc_tol]
        if not entering:
            stop = "dual_optimal"
            break

        for k in entering:
            lp.add_key(k)
        cand_res = lp.row_generate_to_feasible(max_iters=cfg.row_max_iters)
        n_resolves += 1
        gain = cand_res.S - best.S
        best = cand_res  # monotone: more columns -> weakly higher optimum, always accept
        history.append(best.S)
        no_improve = no_improve + 1 if gain < cfg.min_gain else 0
        if no_improve >= cfg.patience:
            stop = "no_improvement"
            break

    return ColgenResult(
        f=best.f,
        S=best.S,
        grid_max=best.grid_max,
        support=sorted(lp.kplus),
        success=best.success,
        stop_reason=stop,
        n_resolves=n_resolves,
        history=history,
    )
