"""Sweep-driver tests: smoke cells match the cold oracle, resume skips, ledger is tolerant."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from arena.pnt_lp import solve_support  # noqa: E402
from arena.pnt_seeds import multiscale_support  # noqa: E402
from scripts.pnt_reach_sweep import load_done, plateau_hints, run_sweep  # noqa: E402


def _smoke_args(out: Path) -> list[str]:
    return ["--smoke", "--out", str(out)]


@pytest.mark.unit
def test_smoke_two_cells_match_cold_oracle(tmp_path: Path):
    out = tmp_path / "sweep.jsonl"
    session = run_sweep(_smoke_args(out) + ["--cold-check"])
    ok = [c for c in session if c.get("success")]
    assert len(ok) == 2
    by_reach = sorted(ok, key=lambda c: c["target_reach"])
    assert by_reach[0]["target_reach"] == 120 and by_reach[1]["target_reach"] == 240
    # more reach cannot hurt (superset domain but far key extends the score frontier)
    assert by_reach[1]["S"] >= by_reach[0]["S"] - 1e-9
    # driver S == trusted cold solve on the same support, for BOTH cells
    for cell in by_reach:
        K = multiscale_support(
            cell["cap"],
            dense_upto=cell["dense_upto"],
            target_reach=cell["target_reach"],
        )
        ref = solve_support(K)
        assert cell["S"] == pytest.approx(ref.S, abs=1e-9)
        assert cell["coeff_c"] > 0
        assert cell["rhs"] == 1.0
        # payload is submission-shaped: string keys, no key 1, within budget
        pf = cell["partial_function"]
        assert "1" not in pf
        assert len(pf) <= cell["cap"]
    # first cell carries the cold-check stamp
    assert by_reach[0]["cold_check"]["agree"] is True


@pytest.mark.unit
def test_resume_skips_completed_cells(tmp_path: Path):
    out = tmp_path / "sweep.jsonl"
    run_sweep(_smoke_args(out))
    n_lines = len(out.read_text().splitlines())
    run_sweep(_smoke_args(out))  # second run: everything already done
    assert len(out.read_text().splitlines()) == n_lines


@pytest.mark.unit
def test_load_done_tolerates_truncated_line(tmp_path: Path):
    out = tmp_path / "sweep.jsonl"
    good = {"dense_upto": 40, "target_reach": 120, "cap": 60, "success": True, "S": 0.5}
    out.write_text(json.dumps(good) + "\n" + '{"dense_upto": 40, "target_re')
    done = load_done(out)
    assert list(done.keys()) == [(40, 120, 60)]


@pytest.mark.unit
def test_plateau_hints_fire_on_flat_ladder_and_rising_coeff():
    def cell(d: int, r: int, s: float, c: float) -> dict:
        return {
            "dense_upto": d,
            "target_reach": r,
            "S": s,
            "coeff_c": c,
            "success": True,
        }

    flat = [cell(100, 1000, 0.99, 0.046), cell(100, 1500, 0.990001, 0.047)]
    hints = plateau_hints(flat)
    assert any("plateau" in h for h in hints)

    rising = [
        cell(100, 1000, 0.99, 0.044),
        cell(100, 1500, 0.991, 0.045),
        cell(100, 2250, 0.9915, 0.046),
    ]
    hints = plateau_hints(rising)
    assert any("deficit-law" in h for h in hints)

    healthy = [
        cell(100, 1000, 0.99, 0.046),
        cell(100, 1500, 0.992, 0.0455),
        cell(100, 2250, 0.9935, 0.045),
    ]
    assert plateau_hints(healthy) == []


@pytest.mark.unit
def test_transition_rebuilds_on_big_diff_and_edits_on_small():
    """>15% key swap -> fresh model carrying the cut rows; tiny diff -> live edit."""
    from arena.pnt_lp import solve_support
    from arena.pnt_warm import WarmLP
    from scripts.pnt_reach_sweep import _transition

    base = list(range(2, 61))
    lp = WarmLP(base)
    lp.row_generate_to_feasible()
    n_rows_before = len(lp.row_x)

    # small diff: one key added -> same object, column edited in place
    same, warm = _transition(lp, base + [61])
    assert warm and same is lp
    assert 61 in same.kplus

    # big diff: sparse squarefree support swaps most columns -> fresh model, rows carried
    big = [k for k in range(2, 200) if all(k % (d * d) for d in range(2, 15))]
    fresh, warm = _transition(same, big)
    assert warm and fresh is not same
    assert sorted(fresh.kplus) == sorted(big)
    assert 0 < len(fresh.row_x) <= n_rows_before + 1  # carried cuts (within old domain)
    res = fresh.row_generate_to_feasible()
    ref = solve_support(big)
    assert res.S == pytest.approx(ref.S, abs=1e-9)


@pytest.mark.unit
def test_near_binding_rows_seed_fresh_model_to_oracle(tmp_path: Path):
    """Ledger f -> near-binding x's -> seeding a fresh model on a NEIGHBOR support still
    converges to that support's exact optimum (seeds affect wall time, never S)."""
    from arena.pnt_warm import WarmLP
    from scripts.pnt_reach_sweep import near_binding_rows

    out = tmp_path / "sweep.jsonl"
    session = run_sweep(_smoke_args(out))
    cell = max((c for c in session if c.get("success")), key=lambda c: c["S"])
    rows = near_binding_rows(cell["partial_function"])
    assert rows == sorted(rows) and len(rows) > 0
    x_dom = 10 * max(int(k) for k in cell["partial_function"])
    assert all(1 <= x < x_dom for x in rows)

    neighbor = multiscale_support(60, dense_upto=50, target_reach=200)
    lp = WarmLP([k for k in neighbor if k != 1])
    lp.add_x_rows([x for x in rows if x < lp.current_x_max()])
    res = lp.row_generate_to_feasible(seed_rows=20, cuts_per_iter=50, drop_slack=True)
    ref = solve_support(neighbor)
    assert res.S == pytest.approx(ref.S, abs=1e-9)


@pytest.mark.unit
def test_budget_skip_records_and_does_not_mark_done(tmp_path: Path):
    out = tmp_path / "sweep.jsonl"
    run_sweep(_smoke_args(out) + ["--budget-hours", "0"])
    lines = [json.loads(ln) for ln in out.read_text().splitlines()]
    assert lines and all(ln["status"] == "skipped_budget" for ln in lines)
    assert load_done(out) == {}
