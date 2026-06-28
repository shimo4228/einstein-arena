# einstein-arena — an honest, server-verified record on the Prime Number Theorem problem

[![Ask DeepWiki](https://deepwiki.com/badge.svg)](https://deepwiki.com/shimo4228/einstein-arena)

A local-search lab for the "construction" math problems on [EinsteinArena](https://einsteinarena.com).
Its headline result: a server-verified score of **0.9955806** on the *Prime Number Theorem* problem
(id 7) that beats the prior published top (0.994901) — an honest **#1 at submission** (2026-06-21).

> **Agent `Agent-Knowledge-Cycle` — honest S = 0.9955806** (RHS = 1.0, grid-checked), beating the prior
> top 0.994901 by +0.00068. The server score equals the local score to all digits, and we hold the
> *clean* bound g(x) ≤ 1 rather than the 1.0001 tolerance the prior top spent. (A leaderboard rank is
> time-relative; the durable, anchored claim is the verified score beating the published top.)

## The problem (one paragraph)

Submit a partial function f on integer keys. The server sets f(1) so Σ f(k)/k = 0, checks
g(x) = Σ f(k)⌊x/k⌋ ≤ 1 for all x, and scores S = −Σ f(k)·log k / k (max S = 1, at f = Möbius μ).
**Fix the support K and the optimal f is a single LP** — so the entire game is *support selection*.

## The insight: structure, not key count

We started from a hunch that "just use more keys" was the wrong frame. Dissecting the actual top
solution (read-only) showed it is the **same** squarefree+LP frame; the gap was ~92% reach/budget and
~1e-4 from spending the tolerance — not a different paradigm. The real lever turned out to be support
**structure**: a *multiscale* support (dense squarefree prefix + a geometrically-spaced sparse tail
that buys reach cheaply) beats first-N-squarefree at a **fixed key budget**, honestly:

| keys | first-N-squarefree | multiscale |
|---|---|---|
| 600 | 0.988901 | **0.991717** |
| 1200 | 0.992884 | **0.994153** |
| 2000 | (LP didn't converge in budget) | **0.995581** ✓ server-verified |

Dual-guided column generation *underperformed* the structural prior (its reduced cost is myopic to
reach), and is shelved. The strategy pattern we extracted from this — *dissect the incumbent before
chasing it* — lives in [`.claude/skills/learned/dissect-the-incumbent.md`](.claude/skills/learned/dissect-the-incumbent.md).

## What's here

```
arena/pnt_lp.py       LP on a fixed support: warm-started cutting-plane + exact-grid feasibility
arena/pnt_warm.py     WarmLP — persistent HiGHS model (dual extraction, key add/drop)
arena/pnt_seeds.py    multiscale support, candidate pool, originality distance
arena/pnt_colgen.py   dual-guided column generation (shelved; underperforms multiscale)
arena/pnt_verify.py   dual_verify — server replay + exact grid @1.0 (the acceptance gate)
arena/client.py       read-only GET + APPROVAL-GATED register/submit/thread (dry-run by default)
arena/verifier.py     fetches the server verifier and runs it verbatim (local score == server score)
scripts/              drivers: solve / ceiling / record / fetch / submit (all dry-run by default)
results/prime-number-theorem/multiscale_record.json   the verified #1 (support + f + verify report)
```

## Reproduce

```bash
uv venv && uv sync
uv run pytest -q                      # 77 tests (the verifier auto-fetches on first use)
uv run python -m scripts.multiscale_record_pnt 2300 4500 2000   # solve + dual_verify + record (~50 min)
```

The verifier and problem detail are fetched from the public API at runtime (see
[verifiers/README.md](verifiers/README.md)); they are not redistributed here.

## Honesty & safety

- **Honest by construction.** We check g(x) ≤ 1.0 on the full integer grid [1, 10·maxK) to
  floating-point tolerance (grid max ≈ 1 + 6e-10) — far inside the prior top's 1.0001 slack and stricter
  than the server's Monte-Carlo sampling. The geometric tail is residue-bounded but not separately
  certified; the server score still equals the local score to all digits.
- **Approval-gated.** GET is free; every outward action (register / submit / thread) defaults to a
  dry-run and only fires with an explicit `approved=True` / `--i-have-approval`. API keys live outside
  the repo (`~/.config/einsteinarena/credentials.json`), never committed.

## Agent environment

This result was produced by an autonomous agent running in **[Claude Code](https://claude.com/claude-code)**
on **Claude Opus 4.8**, driven by the author's personal harness — included as a reference for *what kind
of agent environment* reached this result:

- **Harness:** [shimo4228/claude-harness](https://github.com/shimo4228/claude-harness) — the daily-use
  Claude Code skills / agents / rules. Every outward action (register / submit / post) passed through a
  human approval gate, and every claim through the local↔server verifier check.
- **Identity:** the agent name `Agent-Knowledge-Cycle` is named after the author's
  [agent-knowledge-cycle](https://github.com/shimo4228/agent-knowledge-cycle) research line — a
  six-phase knowledge cycle for AI agents. The strategy pattern used here (*dissect the incumbent before
  chasing it*) was extracted back into that cycle as a learned skill.
- **Hub:** the author's profile [github.com/shimo4228](https://github.com/shimo4228) links the broader
  research program (Agent Knowledge Cycle, Contemplative Agent, Agent Attribution Practice, Authorship
  Strategy).

## As an existence proof

This result is also recorded as a worked instance of the **Existence Proof Format** from
[existence-proof](https://github.com/shimo4228/existence-proof) — a document format for *"I held no
credential in this domain, and a verifiable artifact exists anyway,"* where every claim terminates in a
third-party-verifiable anchor (a resolving DOI, a live URL, a public repository).

The author holds no degree or research background in mathematics, analytic number theory, or numerical
optimization. The construction here was reached by an autonomous agent running under the author's
approval-gated harness; the AI was the *path*, and the result is the author's to account for (the same
operator-accountability framing as [Agent Attribution Practice](https://github.com/shimo4228/agent-attribution-practice)).
The further the path carries the cognitive load, the sharper — not the weaker — the credential-absence
point becomes. The anchored claim is deliberately narrow and checkable: a construction whose **server
score equals its local score to all digits** (0.9955806), exceeding the prior published top (0.994901)
by +0.00068 while holding the clean g(x) ≤ 1 bound — reproducible from this public repository against
the platform's own verifier.

It is filed as Existence Proof Format instance #1
([instance-1.md](https://github.com/shimo4228/existence-proof/blob/main/docs/format/instance-1.md)),
anchored to this repository and the [EinsteinArena](https://einsteinarena.com) platform. For the broader
research program this sits beside, see the hub [shimo4228/shimo4228](https://github.com/shimo4228/shimo4228).

## Attribution

[EinsteinArena](https://einsteinarena.com) and its verifiers are by the platform authors
([vinid/einstein-arena](https://github.com/vinid/einstein-arena)); this project is independent and not
affiliated. Code and original results here are MIT-licensed (see [LICENSE](LICENSE)).
