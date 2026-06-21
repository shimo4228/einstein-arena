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
from collections.abc import Iterable

import numpy as np

from arena.pnt_lp import (
    RHS,
    VALUE_BOUND,
    X_CAP_FACTOR,
    LPResult,
    _constraint_rows,
    _result_from_x,
    _violations,
    normalize_support,
    objective_vector,
)


class WarmLP:
    """One persistent HiGHS model over the current support `kplus` (k>=2; key 1 is dependent).

    Column order in the model is kept aligned with `self.kplus` at all times (append on add_key,
    compact on drop_keys), so col_value()[j] is f(kplus[j]) and pricing/duals stay consistent.
    """

    def __init__(self, kplus: Iterable[int], *, rhs: float = RHS, feas_tol: float = 1e-9) -> None:
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
        self._added: set[int] = set()
        self._sync_kplus()
        ei, ev = np.empty(0, dtype=np.int32), np.empty(0, dtype=np.float64)
        for cost in self.c:
            self.h.addCol(float(cost), -VALUE_BOUND, VALUE_BOUND, 0, ei, ev)

    # ---- internal state sync ----------------------------------------------------
    def _sync_kplus(self) -> None:
        self.n = len(self.kplus)
        self.kplus_arr = np.asarray(self.kplus, dtype=np.int64)
        self.c = objective_vector(self.kplus) if self.kplus else np.empty(0, dtype=np.float64)

    def max_key(self) -> int:
        return max(self.kplus) if self.kplus else 1

    def current_x_max(self) -> int:
        return X_CAP_FACTOR * self.max_key()

    # ---- rows (cutting planes over integer x) -----------------------------------
    def add_x_rows(self, xs: Iterable[int]) -> None:
        """Append <= rows g(x) <= rhs for the integer xs not already present; track ordered row_x."""
        new = sorted({int(x) for x in xs} - self._added)
        if not new or self.n == 0:
            return
        xs_arr = np.asarray(new, dtype=np.int64)
        a = _constraint_rows(xs_arr, self.kplus_arr)  # (m, n): floor(x/k)-x/k
        m = a.shape[0]
        starts = np.arange(m, dtype=np.int32) * self.n
        indices = np.tile(np.arange(self.n, dtype=np.int32), m)
        self.h.addRows(
            m, np.full(m, -self._inf), np.full(m, self.rhs), m * self.n, starts, indices, a.ravel()
        )
        self.row_x.extend(new)
        self._added.update(new)

    # ---- columns (keys) ---------------------------------------------------------
    def add_key(self, k: int) -> None:
        """Warm-add a key column (cost log k/k, box) with its coefficients over EXISTING rows."""
        k = int(k)
        if k == 1:
            raise ValueError("key 1 is the dependent variable; it is never an LP column")
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
        self.h.addCol(float(math.log(k) / k), -VALUE_BOUND, VALUE_BOUND, nrows, idx, coeffs)
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
    def row_generate_to_feasible(
        self,
        x_max: int | None = None,
        *,
        max_iters: int = 60,
        cuts_per_iter: int = 1_000_000,
        seed_rows: int = 1_000_000,
    ) -> LPResult:
        """Cutting-plane loop on the persistent model until grid-feasible (or max_iters).

        Idempotent/incremental: only adds rows not already present, so it can be re-called after
        add_key/drop_keys. Mirrors `_cutting_plane_highs` numerics on a fresh model.
        """
        if self.n == 0:
            return LPResult({1: 0.0}, 0.0, 0.0, True, "trivial-support", 0, 0)
        if x_max is None:
            x_max = self.current_x_max()
        m = self.max_key()
        self.add_x_rows(range(1, min(m, seed_rows) + 1))
        last: LPResult | None = None
        for it in range(1, max_iters + 1):
            self.h.run()
            if self.h.getModelStatus() != self._opt:
                return LPResult(
                    {1: 0.0}, 0.0, math.inf, False,
                    f"highs: {self.h.getModelStatus()}", len(self.row_x), it,
                )
            x = np.asarray(self.h.getSolution().col_value, dtype=np.float64)
            last, g = _result_from_x(
                x, self.kplus, self.c, x_max, self.rhs, self.feas_tol, len(self.row_x), it
            )
            viol = _violations(g, self.rhs + self.feas_tol)
            new_x = [int(v) for v in viol if int(v) not in self._added][:cuts_per_iter]
            if not new_x:
                return last
            self.add_x_rows(new_x)
        assert last is not None
        return LPResult(last.f, last.S, last.grid_max, False, "max_iters", last.n_rows, max_iters)

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
