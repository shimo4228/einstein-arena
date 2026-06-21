"""Scan every arena problem's leaderboard for SATURATION, to pick under-attacked targets.

Edge-via-problem-selection: a problem whose top scores are bit-identical across several
agents is a converged optimum (no headroom). A problem with few competitors, a non-tied top,
or recently added is where a fresh search can plausibly land a new best. Read-only GETs only.

Run:  uv run python scripts/saturation_scan.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from arena.client import ArenaClient  # noqa: E402

# the 17 problems known at first recon (2026-06-21); anything else is newly added
ORIGINAL_IDS = {19, 18, 4, 22, 7, 9, 11, 5, 10, 3, 14, 1, 6, 2, 12, 13, 15}


def main() -> None:
    client = ArenaClient()
    problems = client.get_problems()
    rows = []
    for p in problems:
        lb = client.get_leaderboard(p["id"], limit=20)
        scores = [e.get("bestScore") for e in lb if e.get("bestScore") is not None]
        n = len(scores)
        r1 = scores[0] if n >= 1 else None
        r2 = scores[1] if n >= 2 else None
        r3 = scores[2] if n >= 3 else None
        bit_tie = (n >= 2 and r1 == r2)            # rank1 == rank2 to full precision
        tie3 = (n >= 3 and r1 == r2 == r3)
        gap12 = abs(r1 - r2) if (r1 is not None and r2 is not None) else None
        mi = p.get("minImprovement") or 0.0
        rows.append({
            "id": p["id"], "slug": p["slug"], "scoring": p["scoring"],
            "n": n, "r1": r1, "gap12": gap12, "mi": mi,
            "bit_tie": bit_tie, "tie3": tie3, "new": p["id"] not in ORIGINAL_IDS,
        })

    # attackability: prefer NOT-tied top, fewer competitors, newly added
    def key(r):
        return (r["tie3"], r["bit_tie"], r["n"])

    rows.sort(key=key)
    print(f"{'id':>3} {'slug':32} {'dir':4} {'#':>3} {'tie?':5} {'new':3} {'gap1-2':>11} {'minImpr':>9}  rank1")
    print("-" * 100)
    for r in rows:
        tie = "TIE3" if r["tie3"] else "tie2" if r["bit_tie"] else "-"
        gap = f"{r['gap12']:.2e}" if r["gap12"] is not None else "n/a"
        new = "NEW" if r["new"] else ""
        r1 = f"{r['r1']:.6g}" if r["r1"] is not None else "n/a"
        print(f"{r['id']:>3} {r['slug'][:32]:32} {r['scoring'][:4]:4} {r['n']:>3} {tie:5} {new:3} {gap:>11} {r['mi']:.0e}  {r1}")

    print("\nMost attackable (top, NOT bit-tied / few competitors / newly added):")
    for r in rows[:6]:
        flag = "NEW " if r["new"] else ""
        tie = "bit-tied top" if r["bit_tie"] else "top NOT tied"
        print(f"  - {flag}{r['slug']} (id {r['id']}, {r['n']} agents, {tie})")


if __name__ == "__main__":
    main()
