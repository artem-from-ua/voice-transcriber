"""Offline unit tests for LLMClient — no live server required."""

import httpx
import pytest

from voice.llm import LLMClient, LLMError


def _client_with_transport(handler):
    transport = httpx.MockTransport(handler)
    c = LLMClient()
    c._client.close()
    c._client = httpx.Client(base_url=c.base_url, transport=transport)
    return c


def test_health_check_reports_server_down():
    def handler(_req):
        raise httpx.ConnectError("refused")
    with _client_with_transport(handler) as c:
        with pytest.raises(LLMError, match="not reachable"):
            c.health_check()


def test_health_check_returns_model_ids():
    def handler(req):
        assert req.url.path.endswith("/models")
        return httpx.Response(200, json={"data": [{"id": "foo"}, {"id": "bar"}]})
    with _client_with_transport(handler) as c:
        assert c.health_check() == ["foo", "bar"]


def test_chat_returns_assistant_text():
    def handler(_req):
        return httpx.Response(200, json={
            "choices": [{"message": {"content": "hi there"}}],
        })
    with _client_with_transport(handler) as c:
        assert c.chat([{"role": "user", "content": "hi"}]) == "hi there"


def test_chat_json_uses_json_schema_response_format():
    """LM Studio's MLX runtime rejects 'json_object'; we must send 'json_schema'."""
    captured = {}
    def handler(req):
        captured["body"] = req.read().decode()
        return httpx.Response(200, json={"choices": [{"message": {"content": '{"x":1}'}}]})
    with _client_with_transport(handler) as c:
        c.chat_json([{"role": "user", "content": "x"}])
    assert '"type":"json_schema"' in captured["body"]
    assert "json_object" not in captured["body"]


def test_chat_json_retries_on_invalid_json():
    calls = []
    def handler(req):
        calls.append(req)
        if len(calls) == 1:
            return httpx.Response(200, json={"choices": [{"message": {"content": "not json"}}]})
        return httpx.Response(200, json={"choices": [{"message": {"content": '{"ok": true}'}}]})
    with _client_with_transport(handler) as c:
        assert c.chat_json([{"role": "user", "content": "x"}]) == {"ok": True}
    assert len(calls) == 2


def test_chat_json_fails_after_retries_exhausted():
    def handler(_req):
        return httpx.Response(200, json={"choices": [{"message": {"content": "bad"}}]})
    with _client_with_transport(handler) as c:
        with pytest.raises(LLMError, match="invalid JSON"):
            c.chat_json([{"role": "user", "content": "x"}], retries=1)
