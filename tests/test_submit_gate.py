"""Submit-gate tests: pure integrity checks + arena-top resolution (no network in tests)."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from scripts.pnt_reach_sweep import run_sweep  # noqa: E402
from scripts.submit_pnt_record import check_gate, resolve_arena_top  # noqa: E402


def _rec(**over) -> dict:
    rec = {
        "S_server": 0.9966000000,
        "submittable": True,
        "grid_feasible": True,
        "leaderboard_snapshot": [
            {"agentName": "CHRONOS", "score": 0.9965177307},
            {"agentName": "MAOJIASONG", "score": 0.9963196933},
        ],
    }
    rec.update(over)
    return rec


@pytest.mark.unit
def test_gate_passes_on_clean_record():
    assert check_gate(0.9966000000, _rec(), 0.9965177307) == []


@pytest.mark.unit
def test_gate_rejects_score_mismatch():
    reasons = check_gate(0.9966000000 + 1e-8, _rec(), 0.9965177307)
    assert any("S_server" in r for r in reasons)


@pytest.mark.unit
def test_gate_rejects_unsubmittable_record():
    reasons = check_gate(0.9966, _rec(submittable=False), 0.9965177307)
    assert any("submittable" in r for r in reasons)


@pytest.mark.unit
def test_gate_rejects_when_top_moved_above_us():
    reasons = check_gate(0.9966, _rec(), 0.9967)
    assert any("does not beat" in r for r in reasons)


@pytest.mark.unit
def test_resolve_arena_top_precedence_without_network():
    rec = _rec()
    top, src = resolve_arena_top(rec, 0.42, live=False)
    assert (top, src) == (0.42, "override")
    top, src = resolve_arena_top(rec, None, live=False)
    assert src == "record-snapshot"
    assert top == pytest.approx(0.9965177307)
    top, src = resolve_arena_top({"leaderboard_snapshot": []}, None, live=False)
    assert src == "pinned-fallback"


@pytest.mark.unit
def test_nan_scores_cannot_poison_the_gate():
    """A NaN score anywhere in an untrusted feed must not defeat the beat-the-top check
    (max([nan, ...]) == nan and `s <= nan` is always False)."""
    poisoned = _rec(
        leaderboard_snapshot=[
            {"agentName": "evil", "score": float("nan")},
            {"agentName": "CHRONOS", "score": 0.9965177307},
            {"agentName": "weird", "score": float("inf")},
            {"agentName": "junk", "score": "0.99"},
        ]
    )
    top, src = resolve_arena_top(poisoned, None, live=False)
    assert src == "record-snapshot"
    assert top == pytest.approx(0.9965177307)  # NaN/inf/str all dropped
    # all-poisoned snapshot falls through to the pinned constant
    all_bad = _rec(leaderboard_snapshot=[{"agentName": "evil", "score": float("nan")}])
    top, src = resolve_arena_top(all_bad, None, live=False)
    assert src == "pinned-fallback"
    # defense in depth: even a NaN anchor reaching check_gate must block, not pass
    reasons = check_gate(0.5, _rec(S_server=0.5), float("nan"))
    assert any("not finite" in r for r in reasons)


@pytest.mark.unit
def test_nonfinite_expected_top_override_is_rejected():
    with pytest.raises(ValueError):
        resolve_arena_top(_rec(), float("nan"), live=False)


@pytest.mark.unit
def test_load_done_requires_boolean_true_success(tmp_path: Path):
    """Truthy non-boolean success values (\"false\", 1) must not mark a cell as done."""
    from scripts.pnt_reach_sweep import load_done

    out = tmp_path / "sweep.jsonl"
    rows = [
        {
            "dense_upto": 40,
            "target_reach": 120,
            "cap": 60,
            "success": "false",
            "S": 0.1,
        },
        {"dense_upto": 40, "target_reach": 240, "cap": 60, "success": 1, "S": 0.2},
        {"dense_upto": 40, "target_reach": 360, "cap": 60, "success": True, "S": 0.3},
    ]
    out.write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    assert list(load_done(out).keys()) == [(40, 360, 60)]


@pytest.mark.unit
def test_missing_record_fails_with_clear_message(tmp_path: Path):
    from scripts.submit_pnt_record import load_payload

    with pytest.raises(SystemExit) as exc:
        load_payload(tmp_path / "nope.json")
    assert "record not found" in str(exc.value)


@pytest.mark.unit
def test_record_v2_from_smoke_sweep(tmp_path: Path):
    """End-to-end record generation off a sweep ledger: dual_verify must certify the smoke cell
    (server replay + exact grid), and the v2 record must carry the fidelity fields."""
    from scripts.multiscale_record_pnt import main as record_main

    ledger = tmp_path / "sweep.jsonl"
    run_sweep(["--smoke", "--out", str(ledger)])
    out = tmp_path / "record_v2.json"
    record_main(["--from-sweep", str(ledger), "--out", str(out)])

    rec = json.loads(out.read_text())
    assert rec["schema_version"] == 2
    assert rec["score_matches"] is True
    assert rec["grid_feasible"] is True
    assert rec["submittable"] is True
    assert rec["honest_rhs"] == 1.0
    assert rec["S_server"] == pytest.approx(rec["S_lp"], abs=1e-7)
    assert rec["grid_max"] <= 1.0 + 1e-9
    assert "1" not in rec["partial_function"]
    assert rec["keys"] == len(rec["partial_function"]) + 1
    # smoke cell is far below the real arena top -> must NOT claim a record
    assert rec["beats_arena_top_honestly"] is False
    # regenerating over an existing record soft-deletes the old one to .bak
    record_main(["--from-sweep", str(ledger), "--out", str(out)])
    assert out.with_suffix(".json.bak").exists()


@pytest.mark.unit
def test_record_v2_refuses_to_overwrite_frozen_v1(tmp_path: Path):
    from scripts.multiscale_record_pnt import main as record_main

    ledger = tmp_path / "sweep.jsonl"
    run_sweep(["--smoke", "--out", str(ledger)])
    with pytest.raises(SystemExit):
        record_main(
            [
                "--from-sweep",
                str(ledger),
                "--out",
                "results/prime-number-theorem/multiscale_record.json",
            ]
        )
