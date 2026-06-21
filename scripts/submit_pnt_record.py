"""Gated orchestration: register (if needed) + submit the VERIFIED PNT honest record. DRY-RUN default.

Fires NOTHING to the network without --i-have-approval. Registration mints a PERMANENT public agent
identity; submission posts the solution. Both are outward-facing and require explicit human approval
(CLAUDE.md). Before any send, the payload is re-scored by the verbatim server verifier as an
integrity gate -- we never submit a payload that doesn't reproduce the recorded honest score.

Usage:
  uv run python -m scripts.submit_pnt_record                                   # dry-run: full plan
  uv run python -m scripts.submit_pnt_record --agent-name NAME                 # dry-run, named
  uv run python -m scripts.submit_pnt_record --agent-name NAME --i-have-approval  # FIRE register+submit
"""

from __future__ import annotations

import argparse
import json
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
ARENA_TOP = 0.994901
RECORD = Path("results/prime-number-theorem/multiscale_record.json")


def _load_payload() -> tuple[dict[int, float], dict, dict]:
    rec = json.loads(RECORD.read_text())
    f = {int(k): float(v) for k, v in rec["partial_function"].items()}  # k>=2 (key 1 reconstructed)
    solution = {"partial_function": {str(k): v for k, v in f.items()}}
    return f, solution, rec


def _integrity_gate(f: dict[int, float], rec: dict) -> float:
    """Re-score the payload with the verbatim server verifier; abort on any mismatch/regression."""
    s = server_score(f)
    print(f"[record] payload server-score = {s:.7f}  (recorded {rec['S_server']:.7f})")
    print(f"[record] submittable={rec['submittable']} grid_feasible={rec['grid_feasible']} "
          f"beats_top_honestly={rec['beats_arena_top_honestly']} vs_top={s - ARENA_TOP:+.6f}")
    if abs(s - rec["S_server"]) > 1e-9 or not rec["submittable"] or s <= ARENA_TOP:
        print("[abort] payload failed the integrity/record gate; refusing to proceed.")
        sys.exit(1)
    return s


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--agent-name", help="permanent public identity (2-30 chars, [A-Za-z0-9_-])")
    ap.add_argument("--i-have-approval", action="store_true", help="FIRE network calls (else dry-run)")
    args = ap.parse_args()

    f, solution, rec = _load_payload()
    _integrity_gate(f, rec)

    client = ArenaClient()
    key = load_api_key()
    need_register = key is None
    n_keys = len(solution["partial_function"])

    if not args.i_have_approval:
        print("\n=== DRY RUN — nothing sent ===")
        if need_register:
            shown = args.agent_name or "<choose with --agent-name>"
            if args.agent_name:
                validate_agent_name(args.agent_name)
            plan = client.register(args.agent_name or "placeholder")
            assert isinstance(plan, RegistrationPlan)
            print(f"  STEP 1  REGISTER (permanent identity): name='{shown}' POST {plan.endpoint} "
                  f"(PoW difficulty {plan.difficulty})")
        else:
            print("  STEP 1  REGISTER: SKIP — api_key already present")
        sub = client.submit_solution(PROBLEM_ID, solution)
        assert isinstance(sub, SubmissionPlan)
        print(f"  STEP 2  SUBMIT: POST {sub.endpoint} problem_id={sub.problem_id} "
              f"(partial_function, {n_keys} keys, direct/inline)")
        print("\n  Both steps are OUTWARD-FACING and require approval. To FIRE:")
        print("    uv run python -m scripts.submit_pnt_record --agent-name NAME --i-have-approval")
        return

    # ---- FIRE (explicit approval) ----
    print("\n=== LIVE — approval flag set ===")
    if need_register:
        if not args.agent_name:
            print("[abort] --agent-name is required to register a new identity.")
            sys.exit(1)
        validate_agent_name(args.agent_name)
        print(f"[register] minting permanent identity '{args.agent_name}' (solving PoW)...", flush=True)
        data = client.register(args.agent_name, dry_run=False, approved=True)
        assert isinstance(data, dict)
        print(f"[register] OK -> {data.get('agent', {}).get('name')}; api_key saved to credentials.json")
    print("[submit] sending solution to /api/solutions ...", flush=True)
    res = client.submit_solution(PROBLEM_ID, solution, dry_run=False, approved=True)
    print(f"[submit] server response: {json.dumps(res)[:400]}")
    print("\n[done] Check status via GET /api/solutions/{id} or /api/agents/me/activity.")


if __name__ == "__main__":
    main()
