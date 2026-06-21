"""Gating safety for registration/submission: nothing fires without explicit approval."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from arena.client import (  # noqa: E402
    ArenaClient,
    RegistrationPlan,
    SubmissionPlan,
    ThreadPlan,
    load_api_key,
    validate_agent_name,
)


@pytest.mark.unit
def test_register_default_is_dry_run_plan():
    plan = ArenaClient().register("MyAgent_1")
    assert isinstance(plan, RegistrationPlan)
    assert plan.name == "MyAgent_1"
    assert plan.difficulty == 25
    assert "DRY RUN" in plan.note


@pytest.mark.unit
def test_register_not_fired_without_approval():
    """dry_run=False alone is insufficient; approved=False keeps it a plan (no network)."""
    plan = ArenaClient().register("MyAgent_1", dry_run=False, approved=False)
    assert isinstance(plan, RegistrationPlan)


@pytest.mark.unit
@pytest.mark.parametrize(
    "name,ok",
    [
        ("ab", True),
        ("a" * 30, True),
        ("good-name_1", True),
        ("a", False),
        ("x" * 31, False),
        ("bad name", False),
        ("dot.dot", False),
    ],
    ids=["min2", "max30", "dash-underscore", "too-short", "too-long", "space", "dot"],
)
def test_validate_agent_name(name: str, ok: bool):
    if ok:
        validate_agent_name(name)
    else:
        with pytest.raises(ValueError):
            validate_agent_name(name)


@pytest.mark.unit
def test_submit_solution_default_is_dry_run_plan():
    plan = ArenaClient().submit_solution(7, {"partial_function": {"2": -1.0}})
    assert isinstance(plan, SubmissionPlan)
    assert plan.problem_id == 7
    assert plan.solution["partial_function"]["2"] == -1.0
    assert "DRY RUN" in plan.note


@pytest.mark.unit
def test_create_thread_default_is_dry_run_plan():
    plan = ArenaClient().create_thread("prime-number-theorem", "Title", "Body")
    assert isinstance(plan, ThreadPlan)
    assert plan.slug == "prime-number-theorem"
    assert plan.title == "Title"
    assert "DRY RUN" in plan.note


@pytest.mark.unit
def test_load_api_key_prefers_env(monkeypatch):
    monkeypatch.setenv("EINSTEIN_ARENA_API_KEY", "ea_envkey")
    assert load_api_key() == "ea_envkey"
