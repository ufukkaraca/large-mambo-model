"""Planner backend selection (no network) — the D12 swappable backends, incl. the
local Ollama path for the budget/offline planner."""

import os

import pytest

from mambo_lab import settings
from mambo_lab.planner import (OllamaBackend, OpenRouterBackend, _default_backend,
                               backend_from_settings)


@pytest.fixture(autouse=True)
def _isolate(tmp_path, monkeypatch):
    # no env override, and no real out/studio_settings.json bleeding into the default
    for k in ("MAMBO_PLANNER_BACKEND", "MAMBO_OLLAMA_MODEL", "MAMBO_OLLAMA_URL"):
        monkeypatch.delenv(k, raising=False)
    monkeypatch.setattr(settings, "PATH", tmp_path / "no_settings.json")
    yield


def test_default_is_openrouter():
    assert isinstance(_default_backend(), OpenRouterBackend)
    assert not isinstance(_default_backend(), OllamaBackend)


def test_env_selects_ollama():
    os.environ["MAMBO_PLANNER_BACKEND"] = "ollama"
    b = _default_backend()
    assert isinstance(b, OllamaBackend)
    assert b.model == "qwen2.5:7b"  # default local model
    assert b.URL == "http://localhost:11434/v1/chat/completions"
    assert "Authorization" not in b._headers()  # local server needs no key
    assert b.max_tokens == 512  # output cap so small models commit to tool calls (not ramble→timeout)


def test_env_overrides_model_and_url():
    os.environ.update({"MAMBO_PLANNER_BACKEND": "local",
                       "MAMBO_OLLAMA_MODEL": "qwen2.5:3b",
                       "MAMBO_OLLAMA_URL": "http://127.0.0.1:8080/v1"})
    b = _default_backend()
    assert isinstance(b, OllamaBackend) and b.model == "qwen2.5:3b"
    assert b.URL == "http://127.0.0.1:8080/v1/chat/completions"  # MLX/other OpenAI server


def test_ollama_inherits_openrouter_parsing():
    # the local backend reuses OpenRouter's chat()/response parsing unchanged
    assert OllamaBackend.chat is OpenRouterBackend.chat


def test_backend_from_settings_local():
    b = backend_from_settings({"local": True, "model": "qwen2.5:3b",
                               "base_url": "http://localhost:11434/v1", "key": None})
    assert isinstance(b, OllamaBackend) and b.model == "qwen2.5:3b"


def test_backend_from_settings_hosted_sets_url_and_key():
    b = backend_from_settings({"local": False, "model": "gpt-4o-mini",
                               "base_url": "https://api.openai.com/v1", "key": "sk-abc"})
    assert isinstance(b, OpenRouterBackend) and not isinstance(b, OllamaBackend)
    assert b.URL == "https://api.openai.com/v1/chat/completions"
    assert b._headers()["Authorization"] == "Bearer sk-abc"  # uses the saved key, not .env


def test_settings_file_drives_default(tmp_path, monkeypatch):
    # a saved provider in the settings file overrides the shipped default
    monkeypatch.setattr(settings, "PATH", tmp_path / "s.json")
    settings.set_active("local", path=settings.PATH)
    assert isinstance(_default_backend(), OllamaBackend)
