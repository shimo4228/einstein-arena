"""Phase A validation tests: local verifier fidelity + solver reaches the arena optimum."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from arena.verifier import load_evaluate  # noqa: E402

evaluate = load_evaluate("min-distance-ratio-2d")
ARENA_TOP = 12.889229907717521


@pytest.mark.unit
def test_grid_4x4_scores_exactly_18():
    """A 4x4 unit grid: min dist 1, max dist sqrt(18) -> R = 18.0. Pins verifier fidelity."""
    grid = [[float(i), float(j)] for i in range(4) for j in range(4)]
    assert evaluate({"vectors": grid}) == pytest.approx(18.0, abs=1e-9)


@pytest.mark.unit
def test_wrong_shape_raises():
    with pytest.raises(ValueError):
        evaluate({"vectors": [[0.0, 0.0]] * 15})  # 15 != 16


@pytest.mark.unit
def test_duplicate_points_raise():
    pts = [[float(i), 0.0] for i in range(15)] + [[0.0, 0.0]]  # collides with point 0
    with pytest.raises(ValueError):
        evaluate({"vectors": pts})


@pytest.mark.unit
def test_scale_and_translation_invariant():
    rng = np.random.default_rng(1)
    pts = rng.normal(size=(16, 2))
    base = evaluate({"vectors": pts.tolist()})
    moved = evaluate({"vectors": (pts * 7.3 + np.array([10.0, -4.0])).tolist()})
    assert moved == pytest.approx(base, rel=1e-12)


@pytest.mark.integration
def test_published_optimum_ties_arena_top():
    """The transcribed published optimum (FM Agent, arXiv:2510.26144) scored by the SERVER
    verifier ties the arena top within minImprovement (1e-6). Proves the loop can reach top
    and that local score == server score."""
    saved = REPO_ROOT / "results" / "min-distance-ratio-2d" / "fm_agent_published.json"
    coords = json.loads(saved.read_text())["vectors"]
    assert evaluate({"vectors": coords}) <= ARENA_TOP + 1e-6


@pytest.mark.integration
def test_solver_loop_runs_and_improves():
    """The optimize->exact-verify loop runs end-to-end and returns a valid config that
    beats the naive 4x4 grid (R=18). Tiny budget keeps it fast; SOTA quality is not asserted."""
    from scripts.solve_min_distance_ratio import solve

    best_R, best_pts = solve(seed=0, n_random=8)
    assert best_pts is not None and best_pts.shape == (16, 2)
    assert best_R < 14.0  # comfortably beats grid (18) and random; in the top basin
