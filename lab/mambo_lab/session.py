"""Session state — the `mambo.session.v1` operational store (F1 in
`the session contract`).

This is the third, NON-contract state file. The two contract schemas
(`mambo.utterance.v1`, `mambo.action.v1`) are non-negotiable; session/take/
section state is *operational* (what we're doing), not *what was meant* or
*what to do*, so it lives here and never touches a fixture.

A session is "the Notebook + the live tracks + tempo/key + recent takes". On
disk (git-ignored, mirrors how `out/reaper_inbox` already works):

    out/sessions/<slug>/session.json      # the live, mutable session
    out/sessions/<slug>/notebook.txt      # lyrics (migrated out of localStorage)
    out/sessions/current.txt              # the slug the server reads on boot

`session.json` is a SUPERSET of the existing `session_context` block — the
contract `session_context` keys stay byte-identical (so nothing downstream
breaks). `session_context()` projects exactly those keys back out, which is
what `studio._process` feeds the UIR.

Pure stdlib, file-based, DAW-free — testable without a server (see
`tests/test_session.py`).
"""

from __future__ import annotations

import json
import re
import time
from pathlib import Path
from typing import Any, Optional

from . import oracle

REPO = Path(__file__).resolve().parents[2]
SESSIONS = REPO / "out" / "sessions"

SCHEMA = "mambo.session.v1"

# The exact contract `session_context` keys (ir.py:228). A session is a superset
# of these; `session_context()` projects precisely this set so the UIR carries a
# byte-identical block whether sourced from a session or the cold-start template.
_CONTEXT_KEYS = ("daw", "selected_track", "tracks", "project_tempo_bpm",
                 "project_key", "transport")

# Canonical build order (rhythm -> harmony -> melody -> vocals). Phase nouns map
# here so "we're tracking drums now" / "move on to vocals" set a known phase.
PHASES = ("idle", "tracking_drums", "tracking_harmony", "tracking_melody",
          "tracking_vocals", "mixing")


def slugify(name: str) -> str:
    """A filesystem-safe slug. Empty/odd names fall back to a timestamp."""
    slug = re.sub(r"[^a-z0-9]+", "-", (name or "").lower()).strip("-")
    return slug or f"session-{int(time.time())}"


def _root(root: Optional[Path]) -> Path:
    return Path(root) if root is not None else SESSIONS


def _dir(slug: str, root: Optional[Path] = None) -> Path:
    return _root(root) / slug


def default_session(name: str = "untitled") -> dict[str, Any]:
    """A cold-start session built on the existing context template — so a fresh
    session is byte-identical (in its contract keys) to today's behaviour."""
    ctx = oracle.default_session_context()
    sess = {"schema": SCHEMA, "name": name, "slug": slugify(name)}
    sess.update({k: ctx[k] for k in _CONTEXT_KEYS})  # contract keys, unchanged
    sess.update({"phase": "idle", "sections": [], "takes": [],
                 "created_ts": time.time(), "updated_ts": time.time()})
    return sess


def session_context(sess: dict[str, Any]) -> dict[str, Any]:
    """Project just the contract `session_context` keys out of a session, in the
    same shape/order `oracle.default_session_context()` returns. This is what
    `studio._process` hands to the UIR — keeping the contract block unchanged."""
    return {k: sess[k] for k in _CONTEXT_KEYS if k in sess}


# ── persistence ────────────────────────────────────────────────────────────

def create(name: str, *, root: Optional[Path] = None,
           seed: Optional[dict] = None, make_current: bool = True) -> dict[str, Any]:
    """Create a new named session on disk, seeding contract keys from `seed`
    (e.g. a live REAPER snapshot) or the cold-start template. Returns it."""
    sess = default_session(name)
    if seed:
        sess.update({k: seed[k] for k in _CONTEXT_KEYS if k in seed})
    save(sess, root=root)
    if make_current:
        set_current(sess["slug"], root=root)
    return sess


def save(sess: dict[str, Any], *, root: Optional[Path] = None,
         notebook: Optional[str] = None) -> Path:
    """Write `session.json` (and `notebook.txt` if given) for this session.
    Returns the session dir. Stamps `updated_ts` and re-derives the slug."""
    sess.setdefault("schema", SCHEMA)
    sess["slug"] = sess.get("slug") or slugify(sess.get("name", "untitled"))
    sess["updated_ts"] = time.time()
    d = _dir(sess["slug"], root)
    d.mkdir(parents=True, exist_ok=True)
    (d / "session.json").write_text(json.dumps(sess, indent=2, sort_keys=False))
    if notebook is not None:
        (d / "notebook.txt").write_text(notebook)
    return d


def load(slug: str, *, root: Optional[Path] = None) -> dict[str, Any]:
    """Load a session by slug. Raises FileNotFoundError if it doesn't exist."""
    f = _dir(slug, root) / "session.json"
    return json.loads(f.read_text())


def exists(slug: str, *, root: Optional[Path] = None) -> bool:
    return (_dir(slug, root) / "session.json").exists()


def list_sessions(*, root: Optional[Path] = None) -> list[dict[str, Any]]:
    """A lightweight manifest of every saved session (newest-updated first):
    name, slug, phase, take count, tempo/key, updated_ts. Skips corrupt dirs."""
    base = _root(root)
    out: list[dict[str, Any]] = []
    if not base.exists():
        return out
    for d in sorted(p for p in base.iterdir() if p.is_dir()):
        f = d / "session.json"
        if not f.exists():
            continue
        try:
            s = json.loads(f.read_text())
        except (json.JSONDecodeError, OSError):
            continue
        nb = d / "notebook.txt"
        lines = len([x for x in nb.read_text().splitlines() if x.strip()]) if nb.exists() else 0
        out.append({"name": s.get("name", d.name), "slug": s.get("slug", d.name),
                    "phase": s.get("phase", "idle"),
                    "takes": len(s.get("takes", [])),
                    "notebook_lines": lines,
                    "has_reaper": (d / f"{s.get('slug', d.name)}.rpp").exists(),
                    "project_tempo_bpm": s.get("project_tempo_bpm"),
                    "project_key": s.get("project_key"),
                    "updated_ts": s.get("updated_ts", 0)})
    out.sort(key=lambda s: s["updated_ts"], reverse=True)
    return out


def reaper_project_path(slug: str, *, root: Optional[Path] = None, create: bool = False) -> Path:
    """The per-project REAPER document (`out/sessions/<slug>/<slug>.rpp`). With
    create=True, writes a minimal valid empty project (seeded with the project's
    tempo) if none exists, so "open in REAPER" opens *this* project's document."""
    d = _dir(slug, root)
    d.mkdir(parents=True, exist_ok=True)
    rpp = d / f"{slug}.rpp"
    if create and not rpp.exists():
        sess = load(slug, root=root) if exists(slug, root=root) else {}
        tempo = sess.get("project_tempo_bpm") or 120
        rpp.write_text(f'<REAPER_PROJECT 0.1 "7.0" 0\n  RIPPLE 0\n  TEMPO {tempo} 4 4\n>\n')
    return rpp


def read_notebook(slug: str, *, root: Optional[Path] = None) -> str:
    f = _dir(slug, root) / "notebook.txt"
    return f.read_text() if f.exists() else ""


def write_notebook(slug: str, text: str, *, root: Optional[Path] = None) -> Path:
    d = _dir(slug, root)
    d.mkdir(parents=True, exist_ok=True)
    f = d / "notebook.txt"
    f.write_text(text)
    return f


# ── "current session" pointer (the server + bridge both read this) ───────────

def set_current(slug: str, *, root: Optional[Path] = None) -> None:
    base = _root(root)
    base.mkdir(parents=True, exist_ok=True)
    (base / "current.txt").write_text(slug)


def current_slug(*, root: Optional[Path] = None) -> Optional[str]:
    f = _root(root) / "current.txt"
    if not f.exists():
        return None
    slug = f.read_text().strip()
    return slug if slug and exists(slug, root=root) else None


def load_current(*, root: Optional[Path] = None) -> Optional[dict[str, Any]]:
    """The live session, or None if none is active (cold start / offline)."""
    slug = current_slug(root=root)
    return load(slug, root=root) if slug else None


def load_session_context(*, root: Optional[Path] = None) -> dict[str, Any]:
    """The contract `session_context` for the current turn: the active session's
    projected contract keys, or the cold-start template when no session is active.
    This is the one-line swap `studio._process` makes (see §2 of the spec)."""
    sess = load_current(root=root)
    return session_context(sess) if sess else oracle.default_session_context()


# ── mutations ────────────────────────────────────────────────────────────────

def switch(slug: str, *, root: Optional[Path] = None) -> dict[str, Any]:
    """Make `slug` the current session and return it (raises if missing)."""
    sess = load(slug, root=root)
    set_current(slug, root=root)
    return sess


def rename(slug: str, new_name: str, *, root: Optional[Path] = None) -> dict[str, Any]:
    """Rename a session's display name in place (slug/dir are stable)."""
    sess = load(slug, root=root)
    sess["name"] = new_name
    save(sess, root=root)
    return sess


def set_phase(slug: str, phase: str, *, root: Optional[Path] = None) -> dict[str, Any]:
    sess = load(slug, root=root)
    sess["phase"] = phase
    save(sess, root=root)
    return sess


def append_take(slug: str, take: dict[str, Any], *,
                root: Optional[Path] = None) -> dict[str, Any]:
    """Append a take record (newest last) to the session take log and persist.
    Assigns a 1-based ``id`` (the take number) if the caller didn't set one."""
    sess = load(slug, root=root)
    takes = sess.setdefault("takes", [])
    take.setdefault("id", len(takes) + 1)
    take.setdefault("kept", False)
    takes.append(take)
    save(sess, root=root)
    return sess


def mark_take_kept(slug: str, *, label: Optional[str] = None,
                   root: Optional[Path] = None) -> Optional[dict[str, Any]]:
    """The keeper: mark the latest take kept (optionally labelled). Returns it/None."""
    sess = load(slug, root=root)
    takes = sess.get("takes", [])
    if not takes:
        return None
    takes[-1]["kept"] = True
    if label:
        takes[-1]["label"] = label
    save(sess, root=root)
    return takes[-1]


def scratch_take(slug: str, *, root: Optional[Path] = None) -> Optional[dict[str, Any]]:
    """Take-log-level 'scratch that take': drop the latest take. Returns it/None."""
    sess = load(slug, root=root)
    takes = sess.get("takes", [])
    if not takes:
        return None
    dropped = takes.pop()
    save(sess, root=root)
    return dropped


def find_take(slug: str, ref, *, root: Optional[Path] = None) -> Optional[dict[str, Any]]:
    """Resolve a take reference — an int N (1-based 'take 3') or 'keeper' — to a
    take record, else None."""
    takes = load(slug, root=root).get("takes", [])
    if not takes:
        return None
    if isinstance(ref, int) and 1 <= ref <= len(takes):
        return takes[ref - 1]
    if ref == "keeper":
        kept = [t for t in takes if t.get("kept")]
        return kept[-1] if kept else None
    return None


def set_take(slug: str, take_id: int, *, kept: Optional[bool] = None,
             label: Optional[str] = None, root: Optional[Path] = None) -> Optional[dict[str, Any]]:
    """Update a *specific* take by its 1-based ``id`` (the UI's per-row keep ★ /
    relabel — distinct from `mark_take_kept`, which only touches the latest).
    Returns the updated take, or None if no take has that id."""
    sess = load(slug, root=root)
    t = next((t for t in sess.get("takes", []) if t.get("id") == take_id), None)
    if t is None:
        return None
    if kept is not None:
        t["kept"] = bool(kept)
    if label is not None:
        t["label"] = label
    save(sess, root=root)
    return t


def remove_take(slug: str, take_id: int, *, root: Optional[Path] = None) -> Optional[dict[str, Any]]:
    """Scratch a *specific* take by ``id`` (the UI's per-row ✕). Keeps remaining
    ids stable (no renumber — 'take 3' stays take 3). Returns the dropped take/None."""
    sess = load(slug, root=root)
    takes = sess.get("takes", [])
    t = next((t for t in takes if t.get("id") == take_id), None)
    if t is None:
        return None
    takes.remove(t)
    save(sess, root=root)
    return t


def read_takes(slug: str, *, root: Optional[Path] = None) -> list[dict[str, Any]]:
    """The take log for a session (newest last), for the UI's Takes panel."""
    return load(slug, root=root).get("takes", []) if slug else []
