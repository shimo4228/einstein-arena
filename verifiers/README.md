# verifiers/ (and problems/) — auto-fetched, not redistributed

These directories are intentionally empty in this repository.

The per-problem **verifier source** and **problem detail** are the intellectual property of the
[EinsteinArena](https://einsteinarena.com) platform ([vinid/einstein-arena](https://github.com/vinid/einstein-arena)),
which publishes them through its public API. The upstream repository carries **no license**, so this
project does **not** redistribute that content.

Instead, the code fetches it on demand:

```python
from arena.verifier import load_evaluate
evaluate = load_evaluate("prime-number-theorem")  # GETs the verifier, caches to verifiers/, returns evaluate()
```

`arena/verifier.py::load_evaluate()` writes `verifiers/<slug>.py` (and `arena.verifier.fetch_problem`
caches `problems/<slug>.json`) on first use. We run the **server's own** verifier verbatim so that the
local score is byte-for-byte the server score — we never re-implement scoring.
