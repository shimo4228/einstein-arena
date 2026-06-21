"""Thin EinsteinArena REST client.

SAFETY (see CLAUDE.md): GET endpoints are public and free to call. All MUTATING
actions (register / submit / threads / votes) are OUTWARD-FACING and require explicit
human approval. To enforce this in code:
  - submit_solution / register default to dry_run=True and only BUILD the payload.
  - A live call requires dry_run=False AND approved=True AND an API key from the
    EINSTEIN_ARENA_API_KEY env var. Authorization-bearing requests use
    allow_redirects=False to avoid leaking the Bearer token on cross-host redirects.
"""

from __future__ import annotations

import hashlib
import os
import time
from dataclasses import dataclass
from typing import Any

import requests

BASE_URL = "https://einsteinarena.com"


class ApprovalRequired(RuntimeError):
    """Raised when a mutating call is attempted without explicit human approval."""


@dataclass(frozen=True)
class SubmissionPlan:
    """The exact payload a submit WOULD send (dry-run result)."""

    endpoint: str
    problem_id: int
    solution: dict
    note: str = "DRY RUN — not sent. Re-run with approved=True after human approval."


class ArenaClient:
    def __init__(self, base_url: str = BASE_URL, *, timeout: float = 30.0) -> None:
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout

    # ---- public, read-only GET ----------------------------------------------
    def _get(self, path: str, params: dict | None = None) -> Any:
        url = f"{self.base_url}{path}"
        for _ in range(5):
            resp = requests.get(url, params=params, timeout=self.timeout, allow_redirects=False)
            if resp.status_code == 429:
                wait = float(resp.headers.get("Retry-After", resp.json().get("retry_after_seconds", 5)))
                time.sleep(min(wait, 60))
                continue
            resp.raise_for_status()
            return resp.json()
        raise RuntimeError(f"GET {path} rate-limited after retries")

    def get_problems(self) -> list[dict]:
        return self._get("/api/problems")

    def get_problem(self, slug: str) -> dict:
        return self._get(f"/api/problems/{slug}")

    def get_leaderboard(self, problem_id: int, limit: int = 100) -> Any:
        return self._get("/api/leaderboard", {"problem_id": problem_id, "limit": limit})

    def get_best_solutions(self, problem_id: int, limit: int = 10) -> Any:
        return self._get("/api/solutions/best", {"problem_id": problem_id, "limit": limit})

    # ---- mutating, gated -----------------------------------------------------
    def submit_solution(
        self,
        problem_id: int,
        solution: dict,
        *,
        dry_run: bool = True,
        approved: bool = False,
    ) -> SubmissionPlan | dict:
        """Build (and only if explicitly approved, send) a solution submission.

        Default is a DRY RUN that returns the payload without contacting the server.
        """
        plan = SubmissionPlan(endpoint="/api/solutions", problem_id=problem_id, solution=solution)
        if dry_run or not approved:
            return plan
        api_key = os.environ.get("EINSTEIN_ARENA_API_KEY")
        if not api_key:
            raise ApprovalRequired("EINSTEIN_ARENA_API_KEY not set; cannot submit.")
        resp = requests.post(
            f"{self.base_url}/api/solutions",
            json={"problem_id": problem_id, "solution": solution},
            headers={"Authorization": f"Bearer {api_key}"},
            timeout=self.timeout,
            allow_redirects=False,  # never follow a redirect while carrying the Bearer token
        )
        resp.raise_for_status()
        return resp.json()


def solve_pow(challenge: str, difficulty: int, *, max_iters: int = 1 << 28) -> str:
    """Find a nonce s.t. SHA256(challenge + nonce) has `difficulty` leading zero bits.

    Pure-CPU; bounded by max_iters. Used only during (human-approved) registration.
    """
    nbytes = challenge.encode()
    for nonce in range(max_iters):
        h = hashlib.sha256(nbytes + str(nonce).encode()).digest()
        bits = 0
        for byte in h:
            if byte == 0:
                bits += 8
            else:
                bits += 8 - byte.bit_length()
                break
        if bits >= difficulty:
            return str(nonce)
    raise RuntimeError(f"no nonce found within {max_iters} iters for difficulty {difficulty}")
