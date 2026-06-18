"""Studio settings store + provider registry (no network). Keys must persist
locally and never appear unmasked in the UI-facing view."""

import json

import pytest

from mambo_lab import settings


@pytest.fixture
def store(tmp_path):
    return tmp_path / "studio_settings.json"


def test_default_active_is_openrouter(store):
    a = settings.get_active(store)  # no file yet
    assert a["provider"] == "openrouter" and a["model"]


def test_set_local_is_ready_without_key(store):
    a = settings.set_active("local", path=store)
    assert a["provider"] == "local" and a["local"] is True
    assert a["ready"] is True and a["base_url"].startswith("http://localhost")


def test_hosted_needs_key_for_ready(store, monkeypatch):
    # hermetic: ignore any ambient .env key fallback so the no-key path is deterministic
    monkeypatch.setattr("mambo_lab.secrets.get", lambda key, **kw: None)
    a = settings.set_active("openai", model="gpt-4o-mini", key="", path=store)
    assert a["needs_key"] and a["ready"] is False  # no key yet
    a = settings.set_active("openai", key="sk-test-1234", path=store)
    assert a["ready"] is True and a["key"] == "sk-test-1234"


def test_key_persists_and_public_view_is_masked(store):
    settings.set_active("openai", key="sk-secret-ABCD", path=store)
    pub = settings.public(store)
    oai = next(p for p in pub["providers"] if p["id"] == "openai")
    assert oai["key_set"] == "…ABCD"                  # masked to last 4
    assert "sk-secret-ABCD" not in json.dumps(pub)    # full key never leaves the box


def test_empty_key_keeps_existing(store):
    settings.set_active("openai", key="sk-keep-9999", path=store)
    settings.set_active("openai", key="", path=store)  # blank must not wipe it
    assert settings.get_active(store)["key"] == "sk-keep-9999"


def test_custom_base_url_round_trip(store):
    a = settings.set_active("custom", model="my-model", key="k", base_url="https://x.y/v1", path=store)
    assert a["base_url"] == "https://x.y/v1" and a["model"] == "my-model" and a["ready"] is True


def test_unknown_provider_raises(store):
    with pytest.raises(ValueError):
        settings.set_active("nope", path=store)


def test_voiceprint_round_trip(store):
    assert settings.get_voiceprint(store) is None  # uncalibrated by default
    settings.save({"voiceprint": {"vibrato_semitones": 0.8, "f0_max": 600.0}}, path=store)
    assert settings.get_voiceprint(store)["vibrato_semitones"] == 0.8
