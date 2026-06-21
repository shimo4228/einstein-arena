"""Fetch a problem's server-side verifier and load it locally.

Design rule (see CLAUDE.md): we run the *server's own* verifier source locally so that
the local score is byte-for-byte the server score. We never re-implement the scoring.
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from types import ModuleType
from typing import Callable

import requests

REPO_ROOT = Path(__file__).resolve().parent.parent
PROBLEMS_DIR = REPO_ROOT / "problems"
VERIFIERS_DIR = REPO_ROOT / "verifiers"
BASE_URL = "https://einsteinarena.com"

_HEADER = '''"""EinsteinArena verifier for problem `{slug}` (id={pid}).

VERBATIM copy of the server-side verifier, fetched from
    GET {url}  (field "verifier")
Do NOT edit: byte-identical to the server guarantees local score == server score.
"""

'''


def _module_name(slug: str) -> str:
    return slug.replace("-", "_")


def fetch_problem(slug: str, *, save: bool = True) -> dict:
    """GET a problem detail (public, read-only). Optionally cache to problems/<slug>.json."""
    url = f"{BASE_URL}/api/problems/{slug}"
    resp = requests.get(url, timeout=30, allow_redirects=False)
    resp.raise_for_status()
    data = resp.json()
    if save:
        PROBLEMS_DIR.mkdir(exist_ok=True)
        (PROBLEMS_DIR / f"{slug}.json").write_text(json.dumps(data, ensure_ascii=False))
    return data


def extract_verifier(slug: str, *, problem: dict | None = None) -> Path:
    """Write the server verifier source to verifiers/<module>.py and return its path."""
    if problem is None:
        cached = PROBLEMS_DIR / f"{slug}.json"
        problem = json.loads(cached.read_text()) if cached.exists() else fetch_problem(slug)
    assert problem is not None
    source = problem["verifier"]
    VERIFIERS_DIR.mkdir(exist_ok=True)
    out = VERIFIERS_DIR / f"{_module_name(slug)}.py"
    header = _HEADER.format(slug=slug, pid=problem.get("id", "?"), url=f"{BASE_URL}/api/problems/{slug}")
    out.write_text(header + source.rstrip() + "\n")
    return out


def load_evaluate(slug: str) -> Callable[[dict], float]:
    """Import verifiers/<module>.py and return its evaluate(data) callable.

    Extracts the verifier first if the local file is missing.
    """
    path = VERIFIERS_DIR / f"{_module_name(slug)}.py"
    if not path.exists():
        path = extract_verifier(slug)
    module: ModuleType = _load_module_from_path(_module_name(slug), path)
    return module.evaluate


def _load_module_from_path(name: str, path: Path) -> ModuleType:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load verifier module from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module
