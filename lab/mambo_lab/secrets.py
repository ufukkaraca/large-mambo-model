"""Load local secrets from the git-ignored repo ``.env`` into the environment.

Secrets (ElevenLabs key, any future text-LLM key) live ONLY in ``<repo>/.env``
(git-ignored) or the ambient environment — never in a tracked file, commit, or
log. ``load_env`` is idempotent and uses ``setdefault`` so a real environment
variable always wins over the file.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def load_env(path: Optional[Path] = None) -> None:
    path = path or (_repo_root() / ".env")
    if not path.exists():
        return
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        os.environ.setdefault(k.strip(), v.strip())


def get(key: str, *, required: bool = False) -> Optional[str]:
    load_env()
    val = os.environ.get(key)
    if required and not val:
        raise RuntimeError(
            f"{key} not set — expected in the environment or the git-ignored repo .env"
        )
    return val
