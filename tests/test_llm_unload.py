"""Tests for LLMClient.unload_model — the LM Studio native API call."""

from __future__ import annotations

import httpx

from voice.llm import LLMClient


def test_unload_returns_true_on_success(monkeypatch):
    captured: dict[str, object] = {}

    def fake_post(url, json=None, timeout=None):
        captured["url"] = url
        captured["json"] = json
        return httpx.Response(200, json={"status": "ok"})

    monkeypatch.setattr(httpx, "post", fake_post)
    with LLMClient(model="my-model") as c:
        assert c.unload_model() is True
    assert captured["url"].endswith("/api/v1/models/unload")
    assert captured["json"] == {"instance_id": "my-model"}


def test_unload_returns_false_on_http_error(monkeypatch):
    def fake_post(*_a, **_k):
        return httpx.Response(500, json={"error": "boom"})

    monkeypatch.setattr(httpx, "post", fake_post)
    with LLMClient() as c:
        assert c.unload_model() is False


def test_unload_returns_false_on_transport_error(monkeypatch):
    def fake_post(*_a, **_k):
        raise httpx.ConnectError("refused")

    monkeypatch.setattr(httpx, "post", fake_post)
    with LLMClient() as c:
        assert c.unload_model() is False


def test_unload_uses_explicit_instance_id(monkeypatch):
    captured: dict[str, object] = {}

    def fake_post(url, json=None, timeout=None):
        captured["json"] = json
        return httpx.Response(200, json={})

    monkeypatch.setattr(httpx, "post", fake_post)
    with LLMClient() as c:
        c.unload_model("other-instance")
    assert captured["json"] == {"instance_id": "other-instance"}


def test_unload_target_url_strips_v1(monkeypatch):
    captured: dict[str, object] = {}

    def fake_post(url, json=None, timeout=None):
        captured["url"] = url
        return httpx.Response(200, json={})

    monkeypatch.setattr(httpx, "post", fake_post)
    with LLMClient(base_url="http://desktop.local:1234/v1") as c:
        c.unload_model()
    assert captured["url"] == "http://desktop.local:1234/api/v1/models/unload"
