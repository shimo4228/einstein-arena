"""Gated orchestration: register (if needed) + submit a VERIFIED PNT honest record. DRY-RUN default.

Fires NOTHING to the network without --i-have-approval. Registration mints a PERMANENT public agent
identity; submission posts the solution. Both are outward-facing and require explicit human approval
(CLAUDE.md). Before any send, the payload is re-scored by the verbatim server verifier and compared
against the LIVE arena top (read-only GET; falls back to the record's snapshot, then a pinned
constant) -- we never submit a payload that fails to reproduce its recorded score or that no longer
beats the leaderboard.

Usage:
  uv run python -m scripts.submit_pnt_record                          # dry-run vs default record
  uv run python -m scripts.submit_pnt_record --record PATH            # dry-run, explicit record
  uv run python -m scripts.submit_pnt_record --agent-name NAME --i-have-approval   # FIRE

Operational note (server variance): evaluation runs in a sandbox with a 120s timeout and
~2000-key payloads sit near that edge (2026-06-21: id 2359 error/timeout, id 2360 evaluated).
An approval to submit covers up to 2 re-sends of the SAME payload on timeout-class errors;
switching to a different record (e.g. a 1900-key fallback) needs a fresh dry-run + approval.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from arena.client import (  # noqa: E402
    ArenaClient,
    RegistrationPlan,
    SubmissionPlan,
    load_api_key,
    validate_agent_name,
)
from arena.pnt_verify import server_score  # noqa: E402

PROBLEM_ID = 7
ARENA_TOP_FALLBACK = 0.9965177307
"""CHRONOS, 2026-07-03 (was 0.994901/OrganonAgent before the 2026-06/07 copy-and-extend wave)."""

RECORD_DEFAULT = Path("results/prime-number-theorem/multiscale_record_v2.json")


def load_payload(record_path: Path) -> tuple[dict[int, float], dict, dict]:
    if not record_path.exists():
        sys.exit(
            f"[abort] record not found: {record_path}\n"
            "  Generate it first (scripts/multiscale_record_pnt.py --from-sweep ...) "
            "or point --record at an existing verified record."
        )
    rec = json.loads(record_path.read_text())
    f = {
        int(k): float(v) for k, v in rec["partial_function"].items()
    }  # k>=2 (key 1 reconstructed)
    solution = {"partial_function": {str(k): v for k, v in f.items()}}
    return f, solution, rec


def _finite_scores(rows: list) -> list[float]:
    """Scores from an UNTRUSTED source, finite-only: json.loads parses bare NaN/Infinity into
    real floats, and a single NaN poisons max() (max([nan, x]) == nan) which would then make
    `s <= arena_top` vacuously False and silently defeat the beat-the-top gate."""
    out: list[float] = []
    for r in rows:
        if not isinstance(r, dict):
            continue
        score = r.get("score")
        if isinstance(score, (int, float)) and math.isfinite(score):
            out.append(float(score))
    return out


def resolve_arena_top(
    rec: dict, override: float | None, *, live: bool = True
) -> tuple[float, str]:
    """Freshest available top score: override > live GET > record snapshot > pinned constant."""
    if override is not None:
        if not math.isfinite(float(override)):
            raise ValueError("--expected-top must be a finite number")
        return float(override), "override"
    if live:
        try:
            rows = ArenaClient().get_best_solutions(PROBLEM_ID, limit=10)
            scores = _finite_scores(rows if isinstance(rows, list) else [])
            if scores:
                return max(scores), "live"
        except (
            Exception
        ) as exc:  # noqa: BLE001 -- network is best-effort; gate still has fallbacks
            print(f"[warn] live top fetch failed ({exc}); falling back", flush=True)
    snap_scores = _finite_scores(rec.get("leaderboard_snapshot") or [])
    if snap_scores:
        return max(snap_scores), "record-snapshot"
    return ARENA_TOP_FALLBACK, "pinned-fallback"


def check_gate(s: float, rec: dict, arena_top: float) -> list[str]:
    """Integrity gate, pure: reasons this payload must NOT be sent (empty == pass)."""
    reasons: list[str] = []
    if not math.isfinite(arena_top):  # defense in depth vs a poisoned comparison anchor
        reasons.append(f"arena top {arena_top!r} is not finite; refusing to compare")
    if abs(s - rec["S_server"]) > 1e-9:
        reasons.append(
            f"replayed server score {s!r} != recorded S_server {rec['S_server']!r} (>1e-9)"
        )
    if not rec.get("submittable"):
        reasons.append("record is not marked submittable (dual_verify failed)")
    if s <= arena_top:
        reasons.append(f"score {s:.7f} does not beat the arena top {arena_top:.7f}")
    return reasons


def integrity_gate(
    f: dict[int, float], rec: dict, arena_top: float, top_src: str
) -> float:
    """Re-score the payload with the verbatim server verifier; abort on any mismatch/regression."""
    s = server_score(f)
    print(f"[record] payload server-score = {s:.7f}  (recorded {rec['S_server']:.7f})")
    print(
        f"[record] submittable={rec.get('submittable')} grid_feasible={rec.get('grid_feasible')} "
        f"grid_max={rec.get('grid_max')}"
    )
    print(
        f"[gate] arena top = {arena_top:.10f} ({top_src}); margin = {s - arena_top:+.7f}"
    )
    reasons = check_gate(s, rec, arena_top)
    if reasons:
        for r in reasons:
            print(f"[abort] {r}")
        print("[abort] payload failed the integrity/record gate; refusing to proceed.")
        sys.exit(1)
    return s


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--record",
        type=Path,
        default=RECORD_DEFAULT,
        help=f"verified record JSON (default {RECORD_DEFAULT})",
    )
    ap.add_argument(
        "--agent-name", help="permanent public identity (2-30 chars, [A-Za-z0-9_-])"
    )
    ap.add_argument(
        "--expected-top",
        type=float,
        default=None,
        help="manual arena-top override for the gate (skips the live GET)",
    )
    ap.add_argument(
        "--i-have-approval",
        action="store_true",
        help="FIRE network calls (else dry-run)",
    )
    args = ap.parse_args()

    f, solution, rec = load_payload(args.record)
    arena_top, top_src = resolve_arena_top(rec, args.expected_top)
    integrity_gate(f, rec, arena_top, top_src)

    client = ArenaClient()
    key = load_api_key()
    need_register = key is None
    n_keys = len(solution["partial_function"])

    if not args.i_have_approval:
        print("\n=== DRY RUN — nothing sent ===")
        print(
            f"  RECORD  {args.record} (S_server={rec['S_server']:.10f}, {n_keys} keys, "
            f"reach={rec.get('reach')}, honest RHS={rec.get('honest_rhs')})"
        )
        if need_register:
            shown = args.agent_name or "<choose with --agent-name>"
            if args.agent_name:
                validate_agent_name(args.agent_name)
            plan = client.register(args.agent_name or "placeholder")
            assert isinstance(plan, RegistrationPlan)
            print(
                f"  STEP 1  REGISTER (permanent identity): name='{shown}' POST {plan.endpoint} "
                f"(PoW difficulty {plan.difficulty})"
            )
        else:
            print("  STEP 1  REGISTER: SKIP — api_key already present")
        sub = client.submit_solution(PROBLEM_ID, solution)
        assert isinstance(sub, SubmissionPlan)
        print(
            f"  STEP 2  SUBMIT: POST {sub.endpoint} problem_id={sub.problem_id} "
            f"(partial_function, {n_keys} keys, direct/inline)"
        )
        print(
            "\n  ~2000-key payloads sit near the server's 120s eval timeout; an approval covers"
        )
        print(
            "  up to 2 re-sends of this SAME payload on timeout-class errors (id 2359/2360 precedent)."
        )
        print(
            "  Copy warning: past submissions were re-scaled x1.0001 by copiers (Hyra) — the"
        )
        print(
            "  durable claim is the HONEST record + originality metrics, not the raw #1 slot."
        )
        print("\n  Both steps are OUTWARD-FACING and require approval. To FIRE:")
        print(
            "    uv run python -m scripts.submit_pnt_record --agent-name NAME --i-have-approval"
        )
        return

    # ---- FIRE (explicit approval) ----
    print("\n=== LIVE — approval flag set ===")
    if need_register:
        if not args.agent_name:
            print("[abort] --agent-name is required to register a new identity.")
            sys.exit(1)
        validate_agent_name(args.agent_name)
        print(
            f"[register] minting permanent identity '{args.agent_name}' (solving PoW)...",
            flush=True,
        )
        data = client.register(args.agent_name, dry_run=False, approved=True)
        assert isinstance(data, dict)
        print(
            f"[register] OK -> {data.get('agent', {}).get('name')}; api_key saved to "
            f"credentials.json"
        )
    print("[submit] sending solution to /api/solutions ...", flush=True)
    res = client.submit_solution(PROBLEM_ID, solution, dry_run=False, approved=True)
    print(f"[submit] server response: {json.dumps(res)[:400]}")
    print(
        "\n[done] Check status via GET /api/solutions/{id} or /api/agents/me/activity."
    )


if __name__ == "__main__":
    main()
