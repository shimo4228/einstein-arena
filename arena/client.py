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
import json
import os
import re
import stat
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import requests

BASE_URL = "https://einsteinarena.com"
CREDENTIALS_PATH = Path.home() / ".config" / "einsteinarena" / "credentials.json"
_NAME_RE = re.compile(r"^[A-Za-z0-9_-]+$")  # server rule: 2-30 chars, alnum + dash + underscore


class ApprovalRequired(RuntimeError):
    """Raised when a mutating call is attempted without explicit human approval."""


@dataclass(frozen=True)
class SubmissionPlan:
    """The exact payload a submit WOULD send (dry-run result)."""

    endpoint: str
    problem_id: int
    solution: dict
    note: str = "DRY RUN — not sent. Re-run with approved=True after human approval."


@dataclass(frozen=True)
class RegistrationPlan:
    """What a registration WOULD do (dry-run result). Registration mints a PERMANENT public identity."""

    endpoint: str
    name: str
    difficulty: int = 25
    note: str = (
        "DRY RUN — not sent. Registration creates a PERMANENT, public agent identity and api_key. "
        "Re-run with approved=True only after explicit human approval (CLAUDE.md)."
    )


@dataclass(frozen=True)
class ThreadPlan:
    """What a thread post WOULD send (dry-run result). Posting is outward-facing + public."""

    endpoint: str
    slug: str
    title: str
    body: str
    note: str = "DRY RUN — not sent. Posting is public (moderation queue); re-run with approved=True after human approval."


def validate_agent_name(name: str) -> None:
    """Enforce the server's name rule locally (fail fast before any network call)."""
    if not (2 <= len(name) <= 30) or not _NAME_RE.match(name):
        raise ValueError("agent name must be 2-30 chars, [A-Za-z0-9_-] only")


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
        api_key = load_api_key()
        if not api_key:
            raise ApprovalRequired("no API key (env or credentials.json); register first.")
        resp = requests.post(
            f"{self.base_url}/api/solutions",
            json={"problem_id": problem_id, "solution": solution},
            headers={"Authorization": f"Bearer {api_key}"},
            timeout=self.timeout,
            allow_redirects=False,  # never follow a redirect while carrying the Bearer token
        )
        resp.raise_for_status()
        return resp.json()

    def request_challenge(self, name: str) -> dict:
        """POST /api/agents/challenge -> {challenge, difficulty}. Fired only inside approved register()."""
        resp = requests.post(
            f"{self.base_url}/api/agents/challenge",
            json={"name": name},
            timeout=self.timeout,
            allow_redirects=False,
        )
        resp.raise_for_status()
        return resp.json()

    def register(
        self,
        name: str,
        *,
        dry_run: bool = True,
        approved: bool = False,
        save: bool = True,
    ) -> RegistrationPlan | dict:
        """Mint a PERMANENT public agent identity (challenge -> PoW -> register). OUTWARD-FACING.

        Default is a DRY RUN that returns a RegistrationPlan without contacting the server. A live
        call requires dry_run=False AND approved=True. On success the api_key is shown ONCE, so it
        is saved to CREDENTIALS_PATH (mode 0600) immediately unless save=False.
        """
        validate_agent_name(name)
        if dry_run or not approved:
            return RegistrationPlan(endpoint="/api/agents/register", name=name)
        ch = self.request_challenge(name)
        nonce = int(solve_pow(ch["challenge"], int(ch["difficulty"])))
        resp = requests.post(
            f"{self.base_url}/api/agents/register",
            json={"name": name, "challenge": ch["challenge"], "nonce": nonce},
            timeout=self.timeout,
            allow_redirects=False,
        )
        resp.raise_for_status()
        data = resp.json()
        api_key = data.get("agent", {}).get("api_key")
        if save and api_key:
            save_credentials(name, api_key)
        return data

    def create_thread(
        self,
        slug: str,
        title: str,
        body: str,
        *,
        dry_run: bool = True,
        approved: bool = False,
    ) -> ThreadPlan | dict:
        """Post a discussion thread. OUTWARD-FACING + PUBLIC (enters a moderation queue).

        Default is a DRY RUN returning a ThreadPlan without contacting the server.
        """
        plan = ThreadPlan(endpoint=f"/api/problems/{slug}/threads", slug=slug, title=title, body=body)
        if dry_run or not approved:
            return plan
        api_key = load_api_key()
        if not api_key:
            raise ApprovalRequired("no API key (env or credentials.json); register first.")
        resp = requests.post(
            f"{self.base_url}/api/problems/{slug}/threads",
            json={"title": title, "body": body},
            headers={"Authorization": f"Bearer {api_key}"},
            timeout=self.timeout,
            allow_redirects=False,
        )
        resp.raise_for_status()
        return resp.json()


def load_api_key() -> str | None:
    """API key from EINSTEIN_ARENA_API_KEY, else CREDENTIALS_PATH. None if unregistered."""
    env = os.environ.get("EINSTEIN_ARENA_API_KEY")
    if env:
        return env
    if CREDENTIALS_PATH.exists():
        try:
            return json.loads(CREDENTIALS_PATH.read_text()).get("api_key")
        except (json.JSONDecodeError, OSError):
            return None
    return None


def save_credentials(name: str, api_key: str) -> None:
    """Persist the one-time api_key to CREDENTIALS_PATH with 0600 perms (never to the repo)."""
    CREDENTIALS_PATH.parent.mkdir(parents=True, exist_ok=True)
    CREDENTIALS_PATH.write_text(json.dumps({"agent_name": name, "api_key": api_key}, indent=2))
    CREDENTIALS_PATH.chmod(stat.S_IRUSR | stat.S_IWUSR)


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
