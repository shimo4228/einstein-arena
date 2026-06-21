"""Fetch the arena's best PNT solutions for ORIGINALITY COMPARISON ONLY (read-only GET).

CLAUDE.md: GET endpoints are public and free. The fetched competitor solution (OrganonAgent) is
used solely to measure distance (arena_distance) -- it is NEVER fed into the optimizer as a seed.
Writes results/prime-number-theorem/arena_compare.json.
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from arena.client import ArenaClient  # noqa: E402

PROBLEM_ID = 7
OUT = Path("results/prime-number-theorem/arena_compare.json")
MAX_PF_KEYS = 50_000  # untrusted-input guard: the verifier caps real solutions at 2000 keys


def _clean_partial_function(raw: object) -> dict[str, float]:
    """Sanitize an untrusted partial_function from the API: dict-only, finite floats, size-capped."""
    if not isinstance(raw, dict):
        return {}
    out: dict[str, float] = {}
    for k, v in list(raw.items())[:MAX_PF_KEYS]:
        try:
            fv = float(v)  # type: ignore[arg-type]
        except (TypeError, ValueError):
            continue
        if math.isfinite(fv):
            out[str(k)] = fv
    return out


def main() -> None:
    client = ArenaClient()
    try:
        rows = client.get_best_solutions(PROBLEM_ID, limit=10)
    except Exception as exc:  # noqa: BLE001 -- script boundary: report and exit, never crash silently
        print(f"[error] failed to fetch arena solutions: {exc}", file=sys.stderr)
        sys.exit(1)
    best = []
    for r in rows if isinstance(rows, list) else []:
        data = r.get("data") if isinstance(r, dict) else None
        raw_pf = data.get("partial_function") if isinstance(data, dict) else None
        pf = _clean_partial_function(raw_pf)
        best.append(
            {
                "agentName": r.get("agentName"),
                "score": r.get("score"),
                "createdAt": r.get("createdAt"),
                "n_keys": len(pf),
                "partial_function": pf,
            }
        )
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps({"problem_id": PROBLEM_ID, "solutions": best}, indent=2))
    top = best[0] if best else None
    if top:
        print(f"saved {len(best)} solutions; top = {top['agentName']} {top['score']} "
              f"({top['n_keys']} keys) -> {OUT}")
    else:
        print(f"no solutions returned -> {OUT}")


if __name__ == "__main__":
    main()
