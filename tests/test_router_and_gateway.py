"""Tests for role-based model routing and the usage ledger."""

from __future__ import annotations

import pytest

from server.openrouter_client import client as openrouter_client
from server.services.llm import (
    ROLE_EMAIL_CLASSIFIER,
    ROLE_EXECUTION,
    ROLE_INTERACTION,
    ROLE_MEMORY_COMPACTOR,
    ModelRouter,
    UsageLedger,
    complete,
    set_model_router,
    set_usage_ledger,
)


@pytest.fixture
def router():
    r = ModelRouter(strong_model="strong/model", cheap_model="cheap/model",
                    prices={"strong/model": (10.0, 20.0), "cheap/model": (1.0, 2.0)})
    set_model_router(r)
    yield r
    set_model_router(None)


@pytest.fixture
def ledger():
    l = UsageLedger()
    set_usage_ledger(l)
    yield l
    set_usage_ledger(None)


def test_default_tiers_keep_user_facing_roles_strong(router):
    assert router.model_for(ROLE_INTERACTION) == "strong/model"
    assert router.model_for(ROLE_EXECUTION) == "strong/model"
    assert router.model_for(ROLE_EMAIL_CLASSIFIER) == "cheap/model"
    assert router.model_for(ROLE_MEMORY_COMPACTOR) == "cheap/model"


def test_strong_only_mode_reproduces_upstream():
    r = ModelRouter(strong_model="s", cheap_model="c", mode="strong_only")
    assert all(r.model_for(role) == "s" for role in (ROLE_INTERACTION, ROLE_EMAIL_CLASSIFIER))


def test_tier_override_per_role():
    r = ModelRouter(strong_model="s", cheap_model="c", tiers={ROLE_EXECUTION: "cheap"})
    assert r.model_for(ROLE_EXECUTION) == "c"
    assert r.model_for(ROLE_INTERACTION) == "s"


def test_env_parsing(monkeypatch):
    monkeypatch.setenv("OPENPOKE_STRONG_MODEL", "x/strong")
    monkeypatch.setenv("OPENPOKE_CHEAP_MODEL", "x/cheap")
    monkeypatch.setenv("OPENPOKE_TIER_INTERACTION", "cheap")
    monkeypatch.setenv("OPENPOKE_MODEL_PRICES", '{"x/cheap": [0.1, 0.2]}')
    r = ModelRouter.from_env()
    assert r.model_for(ROLE_INTERACTION) == "x/cheap"
    assert r.price_for("x/cheap").input_per_million == 0.1


async def test_gateway_routes_and_records_reported_usage(router, ledger, monkeypatch):
    seen = {}

    async def fake(*, model, messages, system=None, api_key=None, tools=None, base_url=None):
        seen["model"] = model
        return {
            "choices": [{"message": {"role": "assistant", "content": "ok"}}],
            "usage": {"prompt_tokens": 1000, "completion_tokens": 100},
        }

    monkeypatch.setattr(openrouter_client, "request_chat_completion", fake)
    await complete(role=ROLE_EMAIL_CLASSIFIER, messages=[{"role": "user", "content": "hi"}])

    assert seen["model"] == "cheap/model"
    [rec] = ledger.records()
    assert rec.role == ROLE_EMAIL_CLASSIFIER
    assert rec.estimated is False
    assert rec.cost_usd == pytest.approx((1000 * 1.0 + 100 * 2.0) / 1_000_000)


async def test_gateway_estimates_when_usage_missing(router, ledger, monkeypatch):
    async def fake(*, model, messages, system=None, api_key=None, tools=None, base_url=None):
        return {"choices": [{"message": {"role": "assistant", "content": "x" * 400}}]}

    monkeypatch.setattr(openrouter_client, "request_chat_completion", fake)
    await complete(role=ROLE_INTERACTION, messages=[{"role": "user", "content": "y" * 4000}],
                   system="z" * 400)
    [rec] = ledger.records()
    assert rec.estimated is True
    assert rec.model == "strong/model"
    assert 1000 <= rec.prompt_tokens <= 1150
    assert rec.completion_tokens == 100
    totals = ledger.totals()
    assert totals["total"]["calls"] == 1
    assert ROLE_INTERACTION in totals["by_role"]


async def test_explicit_model_overrides_routing(router, ledger, monkeypatch):
    seen = {}

    async def fake(*, model, **kwargs):
        seen["model"] = model
        return {"choices": [{"message": {"content": ""}}]}

    monkeypatch.setattr(openrouter_client, "request_chat_completion", fake)
    await complete(role=ROLE_INTERACTION, messages=[], model="pinned/model")
    assert seen["model"] == "pinned/model"
