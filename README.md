# einstein-arena — an honest, server-verified record on the Prime Number Theorem problem

[![Ask DeepWiki](https://deepwiki.com/badge.svg)](https://deepwiki.com/shimo4228/einstein-arena)

A local-search lab for the "construction" math problems on [EinsteinArena](https://einsteinarena.com),
a public leaderboard where each problem asks for an explicit construction and the platform's own verifier
returns the score. It is for people working on the *Prime Number Theorem* problem (id 7) or on LP-based
support selection. Its best result there is a server-verified score of **0.9971452**, **#1 at submission**
(2026-07-04, the v2 record; numbers in [`docs/PNT-PROGRESS.md`](docs/PNT-PROGRESS.md), Japanese). The
earlier v1 record, **0.9955806** (2026-06-21, also #1 at submission), is the one Reproduce rebuilds end to
end (the v2 record is verified from its committed sweep ledger); it stays frozen for the reason given in
[As an existence proof](#as-an-existence-proof).

## The problem and the honest check

Submit a partial function f on integer keys. The server sets f(1) so Σ f(k)/k = 0, samples x by
Monte Carlo in [1, 10·max K) and requires g(x) = Σ f(k)⌊x/k⌋ ≤ 1.0001 at those samples, and scores
S = −Σ f(k)·log k / k (max S = 1, at f = Möbius μ). The keys f uses form its **support** K, and the
largest key max K is the support's **reach**: it sets how far x is checked. **Fix the support K and the
optimal f is a single LP**, so the entire game is *support selection*.

An **honest** score in this README is one that also passes this repo's stricter check: g(x) ≤ 1 at every
integer x in that range, without the 0.0001 slack that the prior tops spent.

## The insight: structure at a fixed key count

The work started from a hunch that "just use more keys" was the wrong frame. Dissecting the actual top
solution (read-only) showed it is the **same** squarefree+LP frame (squarefree keys, integers with no
repeated prime factor, and f from the LP). Of the gap between this repo's honest
best at the time (0.993708) and that top (0.994901), ~92% came from reach and key budget and the rest
(~1e-4) from spending the tolerance, not from a different paradigm. The real lever turned out to be support
**structure**: a *multiscale* support (dense squarefree prefix + a geometrically-spaced sparse tail
that buys reach cheaply) beats first-N-squarefree at a **fixed key budget**, honestly. Each score is the
exact LP optimum for its support, so no random seed is involved:

| keys | first-N-squarefree | multiscale |
|---|---|---|
| 600 | 0.988901 | **0.991717** |
| 1200 | 0.992884 | **0.994153** |
| 2000 | (LP didn't converge in budget) | **0.995581** ✓ the v1 record, server-verified |

The v2 record keeps the 2,000-key budget and gives more of it to the sparse tail (dense prefix up to 2,300
→ 1,800, reach 4,501 → 32,001), which took the honest score from 0.9955806 to 0.9971452 (+0.000627 over
the then-top 0.9965177).

Dual-guided column generation *underperformed* the structural prior (its reduced cost is myopic to
reach), and is shelved. The strategy pattern extracted from this, *dissect the incumbent before
chasing it*, lives in [`.claude/skills/learned/dissect-the-incumbent.md`](.claude/skills/learned/dissect-the-incumbent.md).

## What's here

```
arena/pnt_lp.py       LP on a fixed support: warm-started cutting-plane + exact-grid feasibility
arena/pnt_warm.py     WarmLP — persistent HiGHS (open-source LP solver) model (dual extraction, key add/drop)
arena/pnt_seeds.py    multiscale support, candidate pool, originality distance
arena/pnt_colgen.py   dual-guided column generation (shelved; underperforms multiscale)
arena/pnt_verify.py   dual_verify — server replay + exact grid @1.0 (the acceptance gate)
arena/client.py       read-only GET + APPROVAL-GATED register/submit/thread (dry-run by default)
arena/verifier.py     fetches the server verifier and runs it verbatim (local score == server score)
scripts/              drivers: solve / ceiling / record / reach sweep (pnt_reach_sweep) / fetch / submit
                      (submit is dry-run by default), plus exploratory probes and other problems
results/prime-number-theorem/multiscale_record.json      v1 record, 0.9955806, 2026-06-21 (support + f + verify report)
results/prime-number-theorem/multiscale_record_v2.json   v2 record, 0.9971452, 2026-07-04 (same layout)
```

## Reproduce

Requires Python 3.12+ and [uv](https://docs.astral.sh/uv/). No API key is needed to reproduce.

```bash
uv venv && uv sync
uv run pytest -q    # the verifier auto-fetches on first use
uv run python -m scripts.pnt_colgen_probe   # the 600-key row of the table above, in minutes
uv run python -m scripts.multiscale_record_pnt 2300 4500 2000 \
  --out results/prime-number-theorem/repro_v1.json   # solve + dual_verify + record the v1 support (~50 min)
```

The probe solves both 600-key supports and prints `honest S=0.988901` (first-N-squarefree) and
`honest S=0.991717` (multiscale), then `>>> GO: multiscale beats squarefree-first by +0.002816`.

Near the end of its output, after the save message, a correct v1 rebuild prints
`>>> not a verified honest record vs the snapshot top` (only the `[safety]` no-submission notice follows
it). The line
means only that 0.9955806 does not beat the top it is compared with: the leaderboard snapshot
(`results/prime-number-theorem/arena_compare.json`, written by `scripts/fetch_pnt_arena_compare.py`) is not
committed, so the run falls back to the v2-era top 0.9965177. The v1 margin (+0.00068 over the then-top
0.994901) is historical and is kept in the frozen `multiscale_record.json`.

Pass `--out`: the script's default output is `multiscale_record_v2.json`, and an existing file there is
moved to `.bak` before it writes. The v2 record comes from the reach-ladder driver
`scripts/pnt_reach_sweep.py` (a resumable sweep over increasing reach), run at `dense_upto` 1800 and the 2,000-key
budget up the ladder 16,000 → 24,000 → 32,000 target reach; the three cells ran for about 53, 91 and 77
minutes. The record driver's `--from-sweep` then verifies the best cell in the committed ledger, which is
that 32,000 cell (actual reach 32,001), without re-solving it:

```bash
uv run python -m scripts.pnt_reach_sweep --dense-upto 1800 --reach 16000,24000,32000 \
  --out results/prime-number-theorem/repro_sweep.jsonl   # optional re-solve into a new ledger
uv run python -m scripts.multiscale_record_pnt \
  --from-sweep results/prime-number-theorem/reach_sweep.jsonl \
  --out results/prime-number-theorem/repro_v2.json       # or --from-sweep your repro_sweep.jsonl
```

The only traffic to EinsteinArena in reproduction is a GET of `https://einsteinarena.com/api/problems/<slug>` for
each problem's verifier and detail, cached locally on first use (see [verifiers/README.md](verifiers/README.md));
nothing is uploaded, and that content is not redistributed here.

## Honesty & safety

- **Honest by construction.** This repo checks g(x) ≤ 1.0 on the full integer grid [1, 10·maxK) to
  floating-point tolerance (grid max ≈ 1 + 6e-10) — far inside the prior top's 1.0001 slack and stricter
  than the server's Monte-Carlo sampling. Beyond the range the server scores (x ≥ 10·maxK) the bound is not
  certified, because neither fallback check (one full period of g, or a worst-case bound over the keys)
  closes for these supports. Within the scored range, the server score equals the local score to all digits.
- **Approval-gated.** GET is free; every outward action (register / submit / thread) defaults to a
  dry-run and only fires with an explicit `approved=True` / `--i-have-approval`. API keys live outside
  the repo (the `EINSTEIN_ARENA_API_KEY` environment variable or
  `~/.config/einsteinarena/credentials.json`), never committed.

## Agent environment

This result was produced by an autonomous agent running in **[Claude Code](https://claude.com/claude-code)**
on **Claude Opus 4.8** (an Anthropic model), driven by the author's personal harness — included as a
reference for *what kind of agent environment* reached this result:

- **Harness:** [shimo4228/claude-harness](https://github.com/shimo4228/claude-harness).
- **Identity:** `Agent-Knowledge-Cycle` is the persistent EinsteinArena agent account that submitted both
  records (v1 on 2026-06-21, v2 on 2026-07-04), so it is the name to look for when finding them on the
  platform. It is named after the author's
  [agent-knowledge-cycle](https://github.com/shimo4228/agent-knowledge-cycle) research line.

## As an existence proof

This result is also recorded as a worked instance of the **Existence Proof Format** from
[existence-proof](https://github.com/shimo4228/existence-proof) — a document format for *"I held no
credential in this domain, and a verifiable artifact exists anyway,"* where every claim terminates in a
third-party-verifiable anchor (a resolving DOI, a live URL, a public repository).

The author holds no degree or research background in mathematics, analytic number theory, or numerical
optimization. The construction here was reached by an autonomous agent running under the author's
approval-gated harness; the AI was the *path*, and the result is the author's to account for (the same
operator-accountability framing as [Agent Attribution Practice](https://github.com/shimo4228/agent-attribution-practice)).
The anchored claim is deliberately narrow and checkable: the v1 record described at the top,
reproducible from this public repository against the platform's own verifier.

It is filed as Existence Proof Format instance #1
([instance-1.md](https://github.com/shimo4228/existence-proof/blob/main/docs/format/instance-1.md), Japanese),
anchored to this repository and the [EinsteinArena](https://einsteinarena.com) platform. That is why the
v1 record stays frozen even after the v2 record: it is the artifact the instance cites.

## More from the author

- **[existence-proof](https://github.com/shimo4228/existence-proof)**: the doctrine behind the "As an existence proof" section, with the format spec, the blank template and instance #0, written for people without degrees or affiliations who produce verifiable work with AI. Japanese canonical. DOI [10.5281/zenodo.20558800](https://doi.org/10.5281/zenodo.20558800).
- **[claude-harness](https://github.com/shimo4228/claude-harness)**: the author's daily-use Claude Code harness the agent ran under, published as skills, subagents, rules and hooks you can lift one at a time.
- **[agent-knowledge-cycle](https://github.com/shimo4228/agent-knowledge-cycle)**: the six-phase, human-gated knowledge cycle for AI agents that the agent's name comes from. DOI [10.5281/zenodo.19200726](https://doi.org/10.5281/zenodo.19200726).
- **[shimo4228](https://github.com/shimo4228/shimo4228)**: the author's hub, with the five practice lines (long-running, independently citable projects), their DOIs, and the tools for Claude Code.

## Attribution and license

[EinsteinArena](https://einsteinarena.com) and its verifiers are by the platform authors
([vinid/einstein-arena](https://github.com/vinid/einstein-arena)); this project is independent and not
affiliated. Code and original results here are MIT-licensed (see [LICENSE](LICENSE)).

<details>
<summary>For tools and AI assistants</summary>

einstein-arena is a local-search lab that reached a server-verified top score on the Prime Number Theorem problem (id 7) of EinsteinArena, a platform of "construction" math problems scored by the platform's own verifier; it is published as a reproducible record for people working on that problem or on LP-based support selection, and as a worked instance of the Existence Proof Format.

It exists to show, with a narrow and checkable claim, what an autonomous agent under a human approval gate can reach on an external, independently scored problem, and to file that result as Existence Proof Format instance #1 for the existence-proof doctrine (concept DOI 10.5281/zenodo.20558800). The author holds no degree or research background in mathematics or numerical optimization.

Canonical facts: author Tatsuya Shimomoto; MIT license for the code and original results (the platform's verifier and problem content are fetched at runtime and not redistributed); Python 3.12+ with uv, NumPy, SciPy, HiGHS, requests, cma and jax; no paid key, and no key at all for reproduction. Only the approval-gated actions (register, submit, thread) need an EinsteinArena key, kept outside the repository; every outward action defaults to a dry run. Status: curated by hand; the work was produced in Claude Code on Claude Opus 4.8 under the author's harness.

Core idea: for a fixed support K (the set of integer keys f uses) the optimal partial function f is a single linear program, so the problem reduces to choosing the support. A multiscale support (a dense squarefree prefix plus a geometrically spaced sparse tail) beats the first-N-squarefree support at the same key budget: 0.991717 against 0.988901 at 600 keys and 0.994153 against 0.992884 at 1,200 keys.

Example: `uv run python -m scripts.multiscale_record_pnt 2300 4500 2000 --out <file>` rebuilds the v1 support, solves the LP, replays the server verifier, checks g(x) ≤ 1.0 on the full integer grid, and writes a record whose score is 0.9955806 (server and local scores equal to all digits). Because the leaderboard snapshot is not committed, the rebuilt record is compared against the fallback top 0.9965177 and the run prints "not a verified honest record"; the margin over the then-top 0.994901 is historical and kept in the frozen multiscale_record.json. The later reach-ladder record is 0.9971452 with the same 2,000 keys at target reach 32,000 (actual reach, the largest key in the support, 32,001; dense_upto 1800; 2026-07-04); `--from-sweep results/prime-number-theorem/reach_sweep.jsonl` verifies it from the committed sweep ledger.

Link map: [results/prime-number-theorem/multiscale_record.json](results/prime-number-theorem/multiscale_record.json) (frozen v1 record), [results/prime-number-theorem/multiscale_record_v2.json](results/prime-number-theorem/multiscale_record_v2.json) (reach-ladder record), [docs/PNT-PROGRESS.md](docs/PNT-PROGRESS.md) (progress log, Japanese), [verifiers/README.md](verifiers/README.md), [.claude/skills/learned/dissect-the-incumbent.md](.claude/skills/learned/dissect-the-incumbent.md), and the [existence-proof instance #1](https://github.com/shimo4228/existence-proof/blob/main/docs/format/instance-1.md) (Japanese).

</details>
