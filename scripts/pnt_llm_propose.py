"""Optional LLM proposer for the PNT evolution harness (the FunSearch idea engine).

The LLM proposes NEW candidate support sets given the current archive; it NEVER scores them
(scoring is always the LP in arena.pnt_evolve). This keeps the search grounded: every LLM idea
is validated by the exact LP + grid pipeline like any other genome.

Pluggable and OFF by default. `anthropic` is imported lazily inside make_llm_proposer, so the
rest of the harness runs without the dependency. Enabling this makes a run non-reproducible
(LLM sampling), which is why the default harness is algorithm-only.

build_prompt() and parse_supports() are pure and deterministic so they can be unit-tested
without any API call.
"""

from __future__ import annotations

import json

import numpy as np

from arena.pnt_evolve import MAX_SUPPORT, Evaluated, Genome, Proposer

_PROBLEM = (
    "We are maximizing S = -sum_k f(k)*log(k)/k over a partial function f on an integer support "
    "K (1 in K, |K| <= 2000), subject to sum_k f(k)*floor(x/k) <= 1 for all x>=1 and "
    "sum_k f(k)/k = 0. The optimum on a FIXED support is found by an LP; your job is only to "
    "propose promising SUPPORTS (which integers k to include). Theoretical max S=1 (Mobius mu). "
    "Diversify support STRUCTURE: e.g. squarefree-only, prime-power ladders, products p*q, "
    "smooth/highly-composite numbers, or windows that deviate from a plain {1..M} truncation."
)


def build_prompt(archive: list[Evaluated], n_proposals: int, *, key_max: int = 2000) -> str:
    """Deterministic prompt summarizing the best current supports and asking for new ones."""
    top = sorted(archive, key=lambda e: e.S, reverse=True)[:5]
    lines = [_PROBLEM, "", "Current best supports (score S, size, sample of keys):"]
    for ev in top:
        keys = sorted(ev.genome)
        sample = keys[:12] + (["..."] if len(keys) > 12 else [])
        lines.append(f"  S={ev.S:.6f}  |K|={len(keys)}  keys={sample}")
    lines += [
        "",
        f"Propose {n_proposals} NEW candidate supports, each a JSON array of distinct integers "
        f"in [2, {key_max}] (key 1 is added automatically), size <= {MAX_SUPPORT}. "
        "Return ONLY a JSON array of arrays, no prose.",
    ]
    return "\n".join(lines)


def _first_json_array(text: str) -> list | None:
    """Return the first complete JSON array in `text` (raw_decode stops at its real end).

    Robust to trailing prose or a second array block -- unlike a greedy regex, which would
    span both and fail to parse.
    """
    decoder = json.JSONDecoder()
    start = text.find("[")
    while start != -1:
        try:
            obj, _ = decoder.raw_decode(text[start:])
        except json.JSONDecodeError:
            start = text.find("[", start + 1)
            continue
        if isinstance(obj, list):
            return obj
        start = text.find("[", start + 1)
    return None


def parse_supports(
    text: str, *, key_min: int = 2, key_max: int = 2000, cap: int = MAX_SUPPORT
) -> list[Genome]:
    """Extract a JSON array-of-arrays from the model text into valid genomes (1 added, clamped)."""
    raw = _first_json_array(text)
    if raw is None:
        return []
    out: list[Genome] = []
    for item in raw:
        if not isinstance(item, list):
            continue
        keys = {1}
        for v in item:
            try:
                k = int(v)
            except (TypeError, ValueError):
                continue
            if key_min <= k <= key_max:
                keys.add(k)
        trimmed = sorted(keys)[:cap]
        if len(trimmed) >= 2:
            out.append(frozenset(trimmed))
    return out


def make_llm_proposer(
    *,
    model: str = "claude-opus-4-8",
    n_proposals: int = 4,
    key_max: int = 2000,
    max_tokens: int = 2048,
) -> Proposer:
    """Return a Proposer that asks the model for new supports each generation (lazy anthropic)."""
    import anthropic  # noqa: PLC0415 -- lazy: optional dependency

    client = anthropic.Anthropic()

    def propose(archive: list[Evaluated], _rng: np.random.Generator) -> list[Genome]:
        if not archive:
            return []
        prompt = build_prompt(archive, n_proposals, key_max=key_max)
        msg = client.messages.create(
            model=model,
            max_tokens=max_tokens,
            messages=[{"role": "user", "content": prompt}],
        )
        text = "".join(block.text for block in msg.content if block.type == "text")
        return parse_supports(text, key_max=key_max)

    return propose
