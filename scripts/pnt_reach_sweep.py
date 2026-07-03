"""Reach-ladder sweep for `prime-number-theorem`: warm-chained LP over dense_upto x reach cells.

Exploration-only driver (no network, no submission): walks multiscale_support cells in reach-
ascending order on ONE persistent WarmLP, so each cell reuses previous cuts (rows a(x,k) depend only
on (x,k) — cuts stay valid when reach grows) and previous columns (support diffs via add/drop).
Results append to a JSONL (fsync per line) that doubles as the resume ledger and the single
source of the winning f for scripts/multiscale_record_pnt.py.

The optimum per cell is EXACT for its support (certified by the exact grid scan inside
row_generate_to_feasible), so a cell's S depends only on (dense_upto, target_reach, cap) —
engine knobs (seed_rows / cuts_per_iter / drop_slack) change wall time, never the score.

Run (campaign):
  uv run python -m scripts.pnt_reach_sweep --dense-upto 2300 --reach 16000 --cold-check
  uv run python -m scripts.pnt_reach_sweep --dense-upto 1800,2300,2800 --reach 16000
  uv run python -m scripts.pnt_reach_sweep --dense-upto 2300 --reach 24000,32000,48000

Stop-rule hints are printed (plateau / deficit-law break) — the human decides the pivot.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np  # noqa: E402

from arena.pnt_lp import RHS, grid_g_fast, reconstruct_f1, solve_support  # noqa: E402
from arena.pnt_seeds import multiscale_support  # noqa: E402
from arena.pnt_warm import LedgerDriftError, WarmLP  # noqa: E402

OUT_DEFAULT = Path("results/prime-number-theorem/reach_sweep.jsonl")
GAIN_TOL = 5e-5
"""Plateau hint: marginal honest gain per ~1.5x reach step below this suggests pivoting."""


def _parse_int_list(text: str) -> list[int]:
    vals = [int(t) for t in text.split(",") if t.strip()]
    if not vals:
        raise argparse.ArgumentTypeError("expected a comma-separated list of ints")
    return vals


def cell_key(rec: dict[str, Any]) -> tuple[int, int, int]:
    return (int(rec["dense_upto"]), int(rec["target_reach"]), int(rec["cap"]))


def load_done(path: Path) -> dict[tuple[int, int, int], dict[str, Any]]:
    """Completed cells from the JSONL ledger; tolerates a truncated final line."""
    done: dict[tuple[int, int, int], dict[str, Any]] = {}
    if not path.exists():
        return done
    for i, line in enumerate(path.read_text().splitlines(), 1):
        if not line.strip():
            continue
        try:
            rec = json.loads(line)
            if rec.get("success") is True:  # strict: non-bool truthy must NOT count
                done[cell_key(rec)] = rec  # cell_key inside the try: foreign-schema
        except (json.JSONDecodeError, KeyError, TypeError, ValueError, AttributeError):
            print(
                f"[resume] warning: skipping unparseable line {i} in {path}", flush=True
            )
    return done


def plateau_hints(
    cells: list[dict[str, Any]], *, gain_tol: float = GAIN_TOL
) -> list[str]:
    """Stop-rule hints from successful cells: reach-ladder plateau and coeff_c drift."""
    hints: list[str] = []
    by_d: dict[int, list[dict[str, Any]]] = {}
    for rec in cells:
        if rec.get("success"):
            by_d.setdefault(int(rec["dense_upto"]), []).append(rec)
    for d, group in sorted(by_d.items()):
        group = sorted(group, key=lambda r: int(r["target_reach"]))
        for prev, cur in zip(group, group[1:]):
            gain = float(cur["S"]) - float(prev["S"])
            if gain < gain_tol:
                hints.append(
                    f"plateau? D={d}: reach {prev['target_reach']}->{cur['target_reach']} "
                    f"gained only {gain:+.2e} (< {gain_tol:.0e}) — consider structure pivot"
                )
        cs = [float(r["coeff_c"]) for r in group]
        if len(cs) >= 3 and cs[-1] > cs[-2] > cs[-3]:
            hints.append(
                f"deficit-law break? D={d}: coeff_c rising over last 3 reaches "
                f"({cs[-3]:.5f} -> {cs[-2]:.5f} -> {cs[-1]:.5f})"
            )
    return hints


def _append(path: Path, rec: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as fh:
        fh.write(json.dumps(rec) + "\n")
        fh.flush()
        os.fsync(fh.fileno())


def _ledgers_ok(model: WarmLP) -> bool:
    return (
        len(model.kplus) == model.h.getNumCol()
        and len(model.row_x) == model.h.getNumRow()
        and len(model.row_round) == len(model.row_x)
    )


def near_binding_rows(
    pf: dict[str, float], *, eps: float = 1e-6, cap_rows: int = 8000
) -> list[int]:
    """Reconstruct the near-binding x's (g(x) > 1-eps) of a solved cell from its ledger f.

    Cross-process transfer of cut knowledge: which x's bind is nearly support-independent
    within the multiscale family, so seeding a fresh model with them replaces most of the
    cheap-but-heavy range seed after a resume. Uses the fast scan — precision is irrelevant
    here (seed quality only; feasibility is re-certified exactly during the solve).
    """
    f = {int(k): float(v) for k, v in pf.items() if int(k) != 1}
    if not f:
        return []
    keys = np.asarray([1, *sorted(f)], dtype=np.int64)
    vals = np.asarray([reconstruct_f1(f), *(f[k] for k in sorted(f))], dtype=np.float64)
    x_max = 10 * int(keys.max())
    g = grid_g_fast(keys, vals, x_max)
    xs = np.nonzero(g > 1.0 - eps)[0] + 1
    if xs.size > cap_rows:
        order = np.argsort(-g[xs - 1])
        xs = xs[order[:cap_rows]]
    return sorted(int(x) for x in xs)


_REBUILD_DIFF_FRAC = 0.15
"""Rebuild fresh (transferring only the cut set) when more than this fraction of columns would
swap: multiscale cells exchange 25-40% of keys, and editing that many columns leaves HiGHS with
a mangled basis whose 'warm' first solve is slower than a clean cold start on the same rows."""


def _transition(model: WarmLP | None, kplus: list[int]) -> tuple[WarmLP, bool]:
    """Move the persistent model to the new support; returns (model, warm).

    Small support diffs edit the live model (columns + basis carry over). Large diffs keep the
    valuable, support-independent knowledge — WHICH x's are binding (row_x) — but discard the
    basis by rebuilding a fresh model seeded with those rows (filtered to the new domain).
    """
    if model is None:
        return WarmLP(kplus), False
    new = set(kplus)
    old = set(model.kplus)
    if len(old ^ new) > _REBUILD_DIFF_FRAC * (len(old) + len(new)):
        fresh = WarmLP(kplus)
        carried = [x for x in model.row_x if x < fresh.current_x_max()]
        fresh.add_x_rows(carried)
        print(
            f"  [transition] {len(old ^ new)} keys swap -> fresh model, "
            f"{len(carried)} cut rows carried over",
            flush=True,
        )
        return fresh, True
    model.drop_keys(old - new)
    for k in sorted(new - old):
        model.add_key(k)
    # Reach shrank (fine-tune direction): purge rows outside the new sampled domain.
    model.drop_rows_above(model.current_x_max())
    return model, True


def run_cell(
    model: WarmLP | None,
    *,
    dense_upto: int,
    target_reach: int,
    cap: int,
    seed_rows: int,
    cuts_per_iter: int,
    max_iters: int,
    drop_slack: bool,
    cut_window: int = -1,
    carry_rows: list[int] | None = None,
) -> tuple[WarmLP, dict[str, Any]]:
    """Solve one cell on the (possibly reused) model; returns (model, JSONL record).

    cut_window: -1 = auto (x_max // 4000, stratified cuts spread over the domain),
    0 = off (global top-N), >0 = explicit window width.
    carry_rows: near-binding x's from a prior ledger cell; used only when starting a
    fresh model (cross-process warm knowledge — see near_binding_rows).
    """
    K = multiscale_support(cap, dense_upto=dense_upto, target_reach=target_reach)
    kplus = [k for k in K if k != 1]
    model, warm = _transition(model, kplus)
    if not warm and carry_rows:
        carried = [x for x in carry_rows if x < model.current_x_max()]
        model.add_x_rows(carried)
        seed_rows = min(seed_rows, 1000)  # carried rows replace most of the range seed
        print(
            f"  [seed] {len(carried)} near-binding rows carried from ledger", flush=True
        )
    if cut_window < 0:
        cut_window = max(1, model.current_x_max() // 4000)
    t0 = time.time()

    def _progress(it: int, n_rows: int, s_now: float) -> None:
        print(
            f"  [iter {it:3d}] rows={n_rows} S~{s_now:.7f} ({time.time() - t0:.0f}s)",
            flush=True,
        )

    res = model.row_generate_to_feasible(
        seed_rows=seed_rows,
        cuts_per_iter=cuts_per_iter,
        max_iters=max_iters,
        drop_slack=drop_slack,
        cut_window=cut_window or None,
        progress=_progress,
    )
    wall = time.time() - t0
    reach_actual = max(K)
    rec: dict[str, Any] = {
        "ts": time.time(),
        "dense_upto": dense_upto,
        "target_reach": target_reach,
        "cap": cap,
        "n_keys": len(K),
        "reach_actual": reach_actual,
        "S": res.S,
        "grid_max": res.grid_max,
        "success": bool(res.success),
        "status": res.status,
        "n_rows_final": len(model.row_x),
        "n_iters": res.n_iters,
        "wall_s": round(wall, 1),
        "warm": warm,
        "seed_rows": seed_rows,
        "cuts_per_iter": cuts_per_iter,
        "drop_slack": drop_slack,
        "cut_window": cut_window,
        "rhs": RHS,
        "coeff_c": (1.0 - res.S) * math.log(10 * reach_actual),
        "partial_function": {str(k): float(v) for k, v in res.f.items() if k != 1},
    }
    return model, rec


def run_sweep(argv: list[str] | None = None) -> list[dict[str, Any]]:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dense-upto", type=_parse_int_list, required=False)
    ap.add_argument("--reach", type=_parse_int_list, required=False)
    ap.add_argument("--cap", type=int, default=2000)
    ap.add_argument("--seed-rows", type=int, default=6000)
    ap.add_argument("--cuts-per-iter", type=int, default=3000)
    ap.add_argument("--max-iters", type=int, default=120)
    ap.add_argument(
        "--cut-window",
        type=int,
        default=-1,
        help="stratified-cut window width: -1 auto (x_max//4000), 0 off, >0 explicit",
    )
    ap.add_argument("--no-drop-slack", action="store_true")
    ap.add_argument(
        "--cold-check",
        action="store_true",
        help="re-solve the FIRST cell on a fresh no-drop model and assert S agreement",
    )
    ap.add_argument("--budget-hours", type=float, default=None)
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--smoke", action="store_true", help="tiny 2-cell run for tests")
    args = ap.parse_args(argv)

    if args.smoke:
        dense_list = args.dense_upto or [40]
        reach_list = args.reach or [120, 240]
        cap, seed_rows, cuts = min(args.cap, 60), min(args.seed_rows, 100), 50
        out = args.out or Path("results/prime-number-theorem/reach_sweep_smoke.jsonl")
    else:
        if not args.dense_upto or not args.reach:
            ap.error("--dense-upto and --reach are required (unless --smoke)")
        dense_list, reach_list = args.dense_upto, args.reach
        cap, seed_rows, cuts = args.cap, args.seed_rows, args.cuts_per_iter
        out = args.out or OUT_DEFAULT

    drop_slack = not args.no_drop_slack
    done = load_done(out)
    # reach-ascending order is what keeps old cuts valid as the domain grows
    cells = sorted(
        ((d, r) for r in reach_list for d in dense_list), key=lambda dr: (dr[1], dr[0])
    )
    print(
        f"[sweep] {len(cells)} cells -> {out}  (done already: {len(done)})", flush=True
    )

    model: WarmLP | None = None
    carry: list[int] | None = None
    if done:
        best_prev = max(done.values(), key=lambda c: float(c["S"]))
        carry = near_binding_rows(best_prev["partial_function"])
        if carry:
            print(
                f"[resume] cut seeds from D={best_prev['dense_upto']} "
                f"R={best_prev['target_reach']} ({len(carry)} near-binding rows)",
                flush=True,
            )
    t_start = time.monotonic()
    session: list[dict[str, Any]] = list(done.values())
    for d, r in cells:
        key = (d, r, cap)
        if key in done:
            print(
                f"[skip] D={d} R={r} cap={cap} already done (S={done[key]['S']:.7f})",
                flush=True,
            )
            continue
        if args.budget_hours is not None and (
            args.budget_hours <= 0
            or (time.monotonic() - t_start) > args.budget_hours * 3600
        ):
            _append(
                out,
                {
                    "ts": time.time(),
                    "dense_upto": d,
                    "target_reach": r,
                    "cap": cap,
                    "success": False,
                    "status": "skipped_budget",
                },
            )
            print(
                f"[budget] exceeded {args.budget_hours}h -> skipping D={d} R={r}",
                flush=True,
            )
            continue
        print(
            f"[cell] D={d} R={r} cap={cap} (warm={model is not None}) ...", flush=True
        )
        try:
            model, rec = run_cell(
                model,
                dense_upto=d,
                target_reach=r,
                cap=cap,
                seed_rows=seed_rows,
                cuts_per_iter=cuts,
                max_iters=args.max_iters,
                drop_slack=drop_slack,
                cut_window=args.cut_window,
                carry_rows=carry,
            )
        except LedgerDriftError as exc:  # bookkeeping drift -> one fresh-model retry
            print(
                f"[warn] cell D={d} R={r} failed on warm model ({exc}); retrying fresh",
                flush=True,
            )
            model, rec = run_cell(
                None,
                dense_upto=d,
                target_reach=r,
                cap=cap,
                seed_rows=seed_rows,
                cuts_per_iter=cuts,
                max_iters=args.max_iters,
                drop_slack=drop_slack,
                cut_window=args.cut_window,
                carry_rows=carry,
            )
        carry = None  # ledger seeds are for the first fresh build only
        if not _ledgers_ok(model):
            print(
                "[warn] ledger drift detected after cell; next cell will rebuild fresh",
                flush=True,
            )
            model = None

        if args.cold_check and not any(c.get("cold_check") for c in session):
            K = multiscale_support(cap, dense_upto=d, target_reach=r)
            cold = solve_support(
                K, seed_rows=seed_rows, cuts_per_iter=cuts, max_iters=args.max_iters
            )
            agree = abs(cold.S - rec["S"]) <= 1e-7
            rec["cold_check"] = {"S_cold": cold.S, "agree": bool(agree)}
            print(
                f"[cold-check] S_warm={rec['S']:.9f} S_cold={cold.S:.9f} agree={agree}",
                flush=True,
            )
            if not agree:
                rec["success"] = False
                rec["status"] = "cold_check_mismatch"

        _append(out, rec)
        session.append(rec)
        print(
            f"[done] D={d} R={r} S={rec['S']:.7f} grid_max={rec['grid_max']:.10f} "
            f"success={rec['success']} rows={rec['n_rows_final']} iters={rec['n_iters']} "
            f"wall={rec['wall_s']}s coeff_c={rec['coeff_c']:.5f}",
            flush=True,
        )
        for hint in plateau_hints(session):
            print(f"[hint] {hint}", flush=True)

    ok = [c for c in session if c.get("success")]
    if ok:
        best = max(ok, key=lambda c: float(c["S"]))
        print(
            f"\n[best] D={best['dense_upto']} R={best['target_reach']} S={best['S']:.7f} "
            f"(coeff_c={best['coeff_c']:.5f}, {len(ok)} successful cells)",
            flush=True,
        )
    return session


if __name__ == "__main__":
    run_sweep()
