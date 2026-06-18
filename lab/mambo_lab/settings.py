"""Studio settings — the planner provider/model/API key chosen in the UI.

Persisted to a git-ignored local file (`out/studio_settings.json`). **API keys live
here ONLY** — never in a tracked file, a commit, or a log. The planner reads the
active provider via `planner.backend_from_settings(get_active())`; the Studio
serves a masked view (`public()`) to the UI so a key is never sent back to the
browser in full.

Providers are OpenAI-compatible chat-completions endpoints (one code path), so
"local vs a number of hosted ones" is just a base URL + key + model. OpenRouter
fans out to Claude/GPT/Llama/… so a direct Anthropic/OpenAI key is optional.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Optional

REPO = Path(__file__).resolve().parents[2]
PATH = REPO / "out" / "studio_settings.json"

# provider registry — label, endpoint, whether it needs a key, a sane default model,
# the .env var consulted as a key fallback, and a one-line hint for the UI.
PROVIDERS: dict[str, dict[str, Any]] = {
    "local": {
        "label": "Local (Ollama)", "base_url": "http://localhost:11434/v1",
        "needs_key": False, "default_model": "qwen2.5:7b", "env_key": None,
        "hint": "free · offline · private · ~1–3 min/plan at 7B", "local": True,
    },
    "openrouter": {
        "label": "OpenRouter", "base_url": "https://openrouter.ai/api/v1",
        "needs_key": True, "default_model": "openai/gpt-oss-120b:free", "env_key": "OPENROUTER_API_KEY",
        "hint": "one key, every model — Claude / GPT / Llama / Qwen; free tier available", "local": False,
    },
    "openai": {
        "label": "OpenAI", "base_url": "https://api.openai.com/v1",
        "needs_key": True, "default_model": "gpt-4o-mini", "env_key": "OPENAI_API_KEY",
        "hint": "GPT-4o / 4o-mini, native tool calling", "local": False,
    },
    "groq": {
        "label": "Groq", "base_url": "https://api.groq.com/openai/v1",
        "needs_key": True, "default_model": "llama-3.3-70b-versatile", "env_key": "GROQ_API_KEY",
        "hint": "very fast hosted Llama / Qwen; generous free tier", "local": False,
    },
    "custom": {
        "label": "Custom (OpenAI-compatible)", "base_url": "",
        "needs_key": True, "default_model": "", "env_key": None,
        "hint": "any OpenAI-compatible server — Together, Fireworks, an MLX server, …", "local": False,
    },
}
DEFAULT_PROVIDER = "openrouter"  # matches the shipped default; UI can switch to local/etc.


def _read(path: Optional[Path] = None) -> dict:
    p = path or PATH
    if not p.exists():
        return {}
    try:
        return json.loads(p.read_text())
    except (json.JSONDecodeError, OSError):
        return {}


def load(path: Optional[Path] = None) -> dict:
    """Raw settings: {provider, model, base_url?, keys:{provider:key}}."""
    d = _read(path)
    d.setdefault("provider", DEFAULT_PROVIDER)
    d.setdefault("keys", {})
    return d


def save(d: dict, *, path: Optional[Path] = None) -> None:
    p = path or PATH
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(d, indent=2))


def _key_for(provider: str, d: dict) -> Optional[str]:
    """Resolve a provider's key: saved-in-UI first, then the .env fallback."""
    k = (d.get("keys") or {}).get(provider)
    if k:
        return k
    env_key = PROVIDERS.get(provider, {}).get("env_key")
    if env_key:
        from . import secrets
        return secrets.get(env_key)
    return None


def get_active(path: Optional[Path] = None) -> dict:
    """The resolved active provider for the planner: provider, model, base_url, key,
    plus `ready` (a hosted provider with no key isn't usable)."""
    d = load(path)
    pid = d.get("provider", DEFAULT_PROVIDER)
    meta = PROVIDERS.get(pid, PROVIDERS[DEFAULT_PROVIDER])
    model = d.get("model") or meta["default_model"]
    base_url = (d.get("base_url") if pid == "custom" else meta["base_url"]) or meta["base_url"]
    key = _key_for(pid, d)
    ready = bool(model) and bool(base_url) and (key is not None or not meta["needs_key"])
    return {"provider": pid, "label": meta["label"], "model": model, "base_url": base_url,
            "key": key, "local": meta.get("local", False), "needs_key": meta["needs_key"], "ready": ready}


def set_active(provider: str, *, model: Optional[str] = None, key: Optional[str] = None,
               base_url: Optional[str] = None, path: Optional[Path] = None) -> dict:
    """Persist a provider choice (+ optional model / key / custom base_url). An empty
    key string is ignored (keeps any existing key); pass key=None to leave untouched."""
    if provider not in PROVIDERS:
        raise ValueError(f"unknown provider: {provider}")
    d = load(path)
    d["provider"] = provider
    if model is not None:
        d["model"] = model.strip() or PROVIDERS[provider]["default_model"]
    elif not d.get("model"):
        d["model"] = PROVIDERS[provider]["default_model"]
    if base_url is not None and provider == "custom":
        d["base_url"] = base_url.strip()
    if key:  # non-empty → store; empty/None → leave existing key intact
        d.setdefault("keys", {})[provider] = key.strip()
    save(d, path=path)
    return get_active(path)


def get_voiceprint(path: Optional[Path] = None) -> Optional[dict]:
    """The user's saved voiceprint (per-machine calibration), or None if uncalibrated."""
    return load(path).get("voiceprint")


def set_voiceprint(vp: dict, *, path: Optional[Path] = None) -> dict:
    """Persist the calibration voiceprint (applies to every project)."""
    d = load(path)
    d["voiceprint"] = vp
    save(d, path=path)
    return vp


def _mask(key: Optional[str]) -> Optional[str]:
    if not key:
        return None
    return f"…{key[-4:]}" if len(key) > 4 else "set"


def public(path: Optional[Path] = None) -> dict:
    """UI-facing view: the registry, the active choice, and per-provider key status
    (masked — never the full key)."""
    d = load(path)
    active = get_active(path)
    providers = []
    for pid, m in PROVIDERS.items():
        providers.append({
            "id": pid, "label": m["label"], "hint": m["hint"], "needs_key": m["needs_key"],
            "local": m.get("local", False), "default_model": m["default_model"],
            "base_url": m["base_url"],
            "key_set": _mask(_key_for(pid, d)),  # masked; None if no key anywhere
        })
    return {"active": {k: active[k] for k in ("provider", "model", "label", "ready", "needs_key", "local")},
            "custom_base_url": d.get("base_url", ""), "providers": providers}
