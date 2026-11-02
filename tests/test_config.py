"""Provider defaults: an Anthropic key on its own is enough to run the server."""

import os

from server import config

_PROVIDER_KEYS = ("OPENROUTER_API_KEY", "OPENROUTER_BASE_URL", "OPENPOKE_LLM_API",
                  "OPENPOKE_STRONG_MODEL", "OPENPOKE_CHEAP_MODEL")


def _clear(monkeypatch):
    for key in _PROVIDER_KEYS + ("ANTHROPIC_API_KEY",):
        monkeypatch.delenv(key, raising=False)


def test_anthropic_key_alone_selects_the_native_api(monkeypatch):
    _clear(monkeypatch)
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-test")
    config.apply_provider_defaults()
    assert os.environ["OPENROUTER_API_KEY"] == "sk-ant-test"
    assert os.environ["OPENROUTER_BASE_URL"] == "https://api.anthropic.com/v1"
    assert os.environ["OPENPOKE_LLM_API"] == "anthropic"
    assert os.environ["OPENPOKE_STRONG_MODEL"] == "claude-sonnet-4-5"
    assert os.environ["OPENPOKE_CHEAP_MODEL"] == "claude-haiku-4-5"


def test_explicit_models_survive_the_defaults(monkeypatch):
    _clear(monkeypatch)
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-test")
    monkeypatch.setenv("OPENPOKE_STRONG_MODEL", "claude-opus-4-1")
    config.apply_provider_defaults()
    assert os.environ["OPENPOKE_STRONG_MODEL"] == "claude-opus-4-1"
    assert os.environ["OPENPOKE_CHEAP_MODEL"] == "claude-haiku-4-5"


def test_openrouter_key_is_left_alone(monkeypatch):
    _clear(monkeypatch)
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-test")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-test")
    config.apply_provider_defaults()
    assert os.environ["OPENROUTER_API_KEY"] == "sk-or-test"
    assert "OPENPOKE_LLM_API" not in os.environ
    assert "OPENROUTER_BASE_URL" not in os.environ


def test_no_keys_changes_nothing(monkeypatch):
    _clear(monkeypatch)
    config.apply_provider_defaults()
    assert not any(key in os.environ for key in _PROVIDER_KEYS)
