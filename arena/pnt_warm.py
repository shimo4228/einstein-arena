"""Persistent warm-started HiGHS model for the PNT LP, enabling column generation.

`solve_support` (arena/pnt_lp.py) cold-builds a model per call. Column generation needs to keep ONE
model alive across many support edits so it can (a) read row duals, (b) add/drop key-columns warm.
`WarmLP` is that stateful wrapper. Its `row_generate_to_feasible` is the SAME cutting-plane inner
loop as `_cutting_plane_highs`, so on a fresh model it reproduces `solve_support` exactly (see
tests/test_pnt_warm.py). Numerics live in arena/pnt_lp.py; this module only adds statefulness.

Engine-only: no network, no disk.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Iterable

import numpy as np

from arena.pnt_lp import (
    RHS,
    VALUE_BOUND,
    X_CAP_FACTOR,
    LPResult,
    _constraint_rows,
    _result_from_x,
    _violations,
    grid_g_fast,
    normalize_support,
    objective_vector,
    reconstruct_f1,
)

_ADD_ROW_CHUNK = 4000
"""Build/add constraint rows in blocks of this many x's: bounds the dense (block, n) coefficient
matrix to ~64MB at |K|=2000 instead of materializing all seed rows at once (~1.5GB at reach 48k)."""


class LedgerDriftError(RuntimeError):
    """WarmLP bookkeeping (row_x/row_round/kplus) diverged from the HiGHS model.

    Raised (never assert, so it survives `python -O`) when a solve-time readback disagrees
    with the ledgers; callers may recover by rebuilding a fresh model."""


class WarmLP:
    """One persistent HiGHS model over the current support `kplus` (k>=2; key 1 is dependent).

    Column order in the model is kept aligned with `self.kplus` at all times (append on add_key,
    compact on drop_keys), so col_value()[j] is f(kplus[j]) and pricing/duals stay consistent.
    """

    def __init__(
        self, kplus: Iterable[int], *, rhs: float = RHS, feas_tol: float = 1e-9
    ) -> None:
        import highspy  # lazy: keep the scipy path import-free

        keys = normalize_support(kplus)  # validates >=1, dedups, adds key 1
        self.kplus: list[int] = [k for k in keys if k != 1]
        self._kset: set[int] = set(self.kplus)
        self.rhs = rhs
        self.feas_tol = feas_tol
        self._inf = highspy.kHighsInf
        self._opt = highspy.HighsModelStatus.kOptimal
        self.h = highspy.Highs()
        self.h.setOptionValue("output_flag", False)
        self.h.setOptionValue("presolve", "off")
        self.row_x: list[int] = []  # ordered, aligned with model row insertion order
        self.row_round: list[int] = (
            []
        )  # per-row: value of _round when the row was added
        self._round = (
            0  # monotone cutting-plane iteration counter (persists across calls)
        )
        self._added: set[int] = set()
        self._dropped_once: set[int] = (
            set()
        )  # x's removed by drop_slack_rows at least once
        self._pinned: set[int] = set()  # re-entered after a drop -> never dropped again
        self._sync_kplus()
        ei, ev = np.empty(0, dtype=np.int32), np.empty(0, dtype=np.float64)
        for cost in self.c:
            self.h.addCol(float(cost), -VALUE_BOUND, VALUE_BOUND, 0, ei, ev)

    # ---- internal state sync ----------------------------------------------------
    def _sync_kplus(self) -> None:
        self.n = len(self.kplus)
        self.kplus_arr = np.asarray(self.kplus, dtype=np.int64)
        self.c = (
            objective_vector(self.kplus)
            if self.kplus
            else np.empty(0, dtype=np.float64)
        )

    def max_key(self) -> int:
        return max(self.kplus) if self.kplus else 1

    def current_x_max(self) -> int:
        return X_CAP_FACTOR * self.max_key()

    # ---- rows (cutting planes over integer x) -----------------------------------
    def add_x_rows(self, xs: Iterable[int]) -> None:
        """Append <= rows g(x) <= rhs for the integer xs not already present; track ordered row_x.

        Chunked in blocks of `_ADD_ROW_CHUNK` to bound the dense coefficient build. An x that
        re-enters after having been slack-dropped is pinned (never dropped again), which keeps
        the total number of row additions finite (<= 2 per x) and the cutting plane terminating.
        """
        new = sorted({int(x) for x in xs} - self._added)
        if not new or self.n == 0:
            return
        self._pinned |= self._dropped_once.intersection(new)
        for lo in range(0, len(new), _ADD_ROW_CHUNK):
            block = new[lo : lo + _ADD_ROW_CHUNK]
            xs_arr = np.asarray(block, dtype=np.int64)
            a = _constraint_rows(xs_arr, self.kplus_arr)  # (m, n): floor(x/k)-x/k
            m = a.shape[0]
            starts = np.arange(m, dtype=np.int32) * self.n
            indices = np.tile(np.arange(self.n, dtype=np.int32), m)
            self.h.addRows(
                m,
                np.full(m, -self._inf),
                np.full(m, self.rhs),
                m * self.n,
                starts,
                indices,
                a.ravel(),
            )
            self.row_x.extend(block)
            self.row_round.extend([self._round] * m)
            self._added.update(block)

    def _delete_row_indices(self, idxs: list[int]) -> list[int]:
        """Delete model rows by position; keep row_x/row_round/_added aligned. Returns dropped x's."""
        if not idxs:
            return []
        self.h.deleteRows(len(idxs), np.asarray(sorted(idxs), dtype=np.int32))
        kill = set(idxs)
        dropped_x = [x for i, x in enumerate(self.row_x) if i in kill]
        self.row_x = [x for i, x in enumerate(self.row_x) if i not in kill]
        self.row_round = [r for i, r in enumerate(self.row_round) if i not in kill]
        self._added.difference_update(dropped_x)
        return dropped_x

    def drop_slack_rows(self, *, slack_tol: float = 1e-7, min_age: int = 2) -> int:
        """Delete rows slack in the CURRENT solution (call right after an optimal run()).

        A row is dropped iff slack = rhs - g(x) > slack_tol, it has survived >= min_age
        cutting-plane rounds since being added, and it is not pinned. Dropping rows only
        RELAXES the working LP, and convergence is certified by the exact full-grid scan in
        `row_generate_to_feasible`, so the final S is still the exact optimum (sandwich:
        relaxation optimum >= full optimum >= any feasible value). Termination stays finite
        because each x is dropped at most once (re-entrants are pinned by add_x_rows).
        """
        if not self.row_x:
            return 0
        row_value = np.asarray(self.h.getSolution().row_value, dtype=np.float64)
        if row_value.shape[0] != len(self.row_x):
            raise LedgerDriftError(
                f"solution has {row_value.shape[0]} rows but ledger tracks {len(self.row_x)}"
            )
        idxs = [
            i
            for i, (x, rnd) in enumerate(zip(self.row_x, self.row_round))
            if self.rhs - row_value[i] > slack_tol
            and self._round - rnd >= min_age
            and x not in self._pinned
        ]
        self._dropped_once.update(self._delete_row_indices(idxs))
        return len(idxs)

    def drop_rows_above(self, x_limit: int) -> int:
        """Delete rows with x >= x_limit (domain shrink when the max key decreases).

        Rows beyond the new [1, 10*max(K)) domain are constraints the verifier never samples;
        keeping them would understate S. Removal is relaxation-only, so correctness is preserved
        (the next row_generate_to_feasible re-certifies on the current domain). Not recorded in
        _dropped_once: this is a domain change, not the slack heuristic.
        """
        idxs = [i for i, x in enumerate(self.row_x) if x >= x_limit]
        self._delete_row_indices(idxs)
        return len(idxs)

    # ---- columns (keys) ---------------------------------------------------------
    def add_key(self, k: int) -> None:
        """Warm-add a key column (cost log k/k, box) with its coefficients over EXISTING rows."""
        k = int(k)
        if k == 1:
            raise ValueError(
                "key 1 is the dependent variable; it is never an LP column"
            )
        if k in self._kset:
            return
        nrows = len(self.row_x)
        if nrows:
            coeffs = _constraint_rows(
                np.asarray(self.row_x, dtype=np.int64), np.asarray([k], dtype=np.int64)
            ).ravel()
            idx = np.arange(nrows, dtype=np.int32)
        else:
            coeffs = np.empty(0, dtype=np.float64)
            idx = np.empty(0, dtype=np.int32)
        self.h.addCol(
            float(math.log(k) / k), -VALUE_BOUND, VALUE_BOUND, nrows, idx, coeffs
        )
        self.kplus.append(k)  # column appended at the end -> keep kplus order aligned
        self._kset.add(k)
        self._sync_kplus()

    def drop_keys(self, ks: Iterable[int]) -> None:
        """Warm-remove key columns by value; remaining columns compact, kplus stays aligned."""
        drop = {int(k) for k in ks} & self._kset
        if not drop:
            return
        idxs = [i for i, k in enumerate(self.kplus) if k in drop]
        self.h.deleteCols(len(idxs), np.asarray(sorted(idxs), dtype=np.int32))
        self.kplus = [k for i, k in enumerate(self.kplus) if i not in set(idxs)]
        self._kset -= drop
        self._sync_kplus()

    # ---- solve / read -----------------------------------------------------------
    def _select_cuts(
        self, viol: np.ndarray, cuts_per_iter: int, cut_window: int | None
    ) -> list[int]:
        """Pick cut rows from violations (sorted by descending violation).

        Plain mode takes the global top-N, which clusters cuts in one x-region and bloats
        the working LP with near-duplicate rows. Windowed (stratified) mode keeps only the
        WORST violation per window of width `cut_window` on the x-axis, spreading cuts across
        the domain — fewer, more informative rows per round. Selection strategy never affects
        the final S (exact-grid certification), only wall time.
        """
        fresh = viol[~np.isin(viol, list(self._added))] if len(self._added) else viol
        if cut_window and cut_window > 1 and fresh.size:
            # viol is sorted by descending violation, so the FIRST hit per bucket is its max
            buckets = fresh // cut_window
            _, first_idx = np.unique(buckets, return_index=True)
            fresh = fresh[np.sort(first_idx)]  # keep descending-violation order
        return [int(v) for v in fresh[:cuts_per_iter]]

    def _fast_violation_cuts(
        self, x: np.ndarray, x_max: int, cuts_per_iter: int, cut_window: int | None
    ) -> list[int]:
        """Cut candidates from the O(x_max * sum 1/k) scan (find-only; never certifies)."""
        f1 = reconstruct_f1(dict(zip(self.kplus, x)))
        full_keys = np.concatenate((np.asarray([1], dtype=np.int64), self.kplus_arr))
        full_vals = np.concatenate((np.asarray([f1], dtype=np.float64), x))
        g_fast = grid_g_fast(full_keys, full_vals, x_max)
        viol = _violations(g_fast, self.rhs + self.feas_tol)
        return self._select_cuts(viol, cuts_per_iter, cut_window)

    def row_generate_to_feasible(
        self,
        x_max: int | None = None,
        *,
        max_iters: int = 60,
        cuts_per_iter: int = 1_000_000,
        seed_rows: int = 1_000_000,
        drop_slack: bool = False,
        slack_tol: float = 1e-7,
        min_age: int = 2,
        cut_window: int | None = None,
        progress: Callable[[int, int, float], None] | None = None,
    ) -> LPResult:
        """Cutting-plane loop on the persistent model until grid-feasible (or max_iters).

        Idempotent/incremental: only adds rows not already present, so it can be re-called after
        add_key/drop_keys. Mirrors `_cutting_plane_highs` numerics on a fresh model.

        Cut candidates come from the fast divisor-sum scan; convergence is ALWAYS certified by
        the exact `grid_g` scan (the returned LPResult's grid_max is exact). With drop_slack,
        rows slack for >= min_age rounds are deleted between solves to keep the working LP small
        (see drop_slack_rows for why the final optimum is unaffected). The terminating iteration
        never edits the model, so duals()/basis stay valid for the returned solution.

        `progress(it, n_rows, S)` is called after each optimal solve (observability only;
        exceptions are NOT swallowed — pass a callback that cannot raise).
        """
        if self.n == 0:
            return LPResult({1: 0.0}, 0.0, 0.0, True, "trivial-support", 0, 0)
        if x_max is None:
            x_max = self.current_x_max()
        m = self.max_key()
        self.add_x_rows(range(1, min(m, seed_rows) + 1))
        x = np.empty(0, dtype=np.float64)
        for it in range(1, max_iters + 1):
            self._round += 1
            self.h.run()
            if self.h.getModelStatus() != self._opt:
                return LPResult(
                    {1: 0.0},
                    0.0,
                    math.inf,
                    False,
                    f"highs: {self.h.getModelStatus()}",
                    len(self.row_x),
                    it,
                )
            x = np.asarray(self.h.getSolution().col_value, dtype=np.float64)
            if progress is not None:
                progress(it, len(self.row_x), -float(np.dot(self.c, x)))
            new_x = self._fast_violation_cuts(x, x_max, cuts_per_iter, cut_window)
            if not new_x:
                # Fast scan sees nothing new -> certify (or refute) with the exact grid.
                last, g = _result_from_x(
                    x,
                    self.kplus,
                    self.c,
                    x_max,
                    self.rhs,
                    self.feas_tol,
                    len(self.row_x),
                    it,
                )
                viol = _violations(g, self.rhs + self.feas_tol)
                new_x = self._select_cuts(viol, cuts_per_iter, cut_window)
                if not new_x:
                    return last
            if drop_slack:
                self.drop_slack_rows(slack_tol=slack_tol, min_age=min_age)
            self.add_x_rows(new_x)
        last, _ = _result_from_x(
            x,
            self.kplus,
            self.c,
            x_max,
            self.rhs,
            self.feas_tol,
            len(self.row_x),
            max_iters,
        )
        return LPResult(
            last.f, last.S, last.grid_max, False, "max_iters", last.n_rows, max_iters
        )

    def col_value(self) -> np.ndarray:
        """Current f over kplus (aligned with self.kplus); valid after a run()."""
        return np.asarray(self.h.getSolution().col_value, dtype=np.float64)

    def duals(self) -> tuple[np.ndarray, np.ndarray]:
        """(row_x, row_dual) aligned, valid after a converged run(). <= rows -> dual <= 0."""
        sol = self.h.getSolution()
        if not sol.dual_valid:
            raise RuntimeError("duals not valid; call row_generate_to_feasible first")
        row_dual = np.asarray(sol.row_dual, dtype=np.float64)
        row_x = np.asarray(self.row_x, dtype=np.int64)
        assert row_dual.shape == row_x.shape, (row_dual.shape, row_x.shape)
        return row_x, row_dual

    def objective_S(self) -> float:
        """S = -c . f over the current columns (= LPResult.S after convergence)."""
        if self.n == 0:
            return 0.0
        return -float(np.dot(self.c, self.col_value()))
