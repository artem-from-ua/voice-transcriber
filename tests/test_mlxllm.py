"""Offline tests for the mlx-lm-backed MlxLLM.

`mlx_lm.load` and `mlx_lm.generate` are monkeypatched so the suite stays
fast and does not need ~8 GB of model weights on disk.
"""

from __future__ import annotations

import json

import pytest

import voice.llm as llm_module
from voice.llm import MlxLLM, LLMError


def test_health_check_missing_dir_raises(tmp_path):
    llm = MlxLLM(model_path=str(tmp_path / "does-not-exist"))
    with pytest.raises(LLMError, match="not found"):
        llm.health_check()


def test_health_check_missing_config_raises(tmp_path):
    (tmp_path / "weights.safetensors").write_text("dummy")
    llm = MlxLLM(model_path=str(tmp_path))
    with pytest.raises(LLMError, match="config.json"):
        llm.health_check()


def test_health_check_accepts_valid_dir(tmp_path):
    (tmp_path / "config.json").write_text("{}")
    llm = MlxLLM(model_path=str(tmp_path))
    assert llm.health_check() == str(tmp_path)


class _FakeTokenizer:
    """Just enough of mlx-lm's TokenizerWrapper for our chat path."""

    eos_token_id = 0
    all_special_ids: list[int] = []

    def __len__(self) -> int:
        return 256

    def encode(self, s, *_, **__):
        return [ord(c) % 256 for c in s] or [1]

    def decode(self, ids, *_, **__):
        return "".join(chr(i) for i in ids if i < 256)

    def apply_chat_template(self, messages, *, add_generation_prompt=False, tokenize=False):
        joined = "\n".join(f"{m['role']}: {m['content']}" for m in messages)
        return joined + ("\nassistant:" if add_generation_prompt else "")

    @property
    def _tokenizer(self):
        return self


def _install_fake_runtime(monkeypatch, *, generate_reply: str):
    """Replace mlx_lm.load/generate so tests don't touch real weights."""
    captured: dict = {}

    def fake_load(_path):
        return ("FAKE_MODEL", _FakeTokenizer())

    def fake_generate(model, tokenizer, *, prompt, **kwargs):
        captured["prompt"] = prompt
        captured["kwargs"] = kwargs
        return generate_reply

    # Skip the heavy tokenizer-data build (it iterates the vocab) and the
    # JSON-constrained logits processor (it needs that data).
    monkeypatch.setattr(llm_module, "_build_tokenizer_data", lambda _t: object())
    monkeypatch.setattr(
        llm_module, "_make_json_logits_processor",
        lambda _td, _schema: (lambda _tokens, logits: logits),
    )
    monkeypatch.setattr(llm_module.mlx_lm, "load", fake_load)
    monkeypatch.setattr(llm_module.mlx_lm, "generate", fake_generate)
    return captured


def _ready_llm(tmp_path):
    (tmp_path / "config.json").write_text("{}")
    return MlxLLM(model_path=str(tmp_path))


def test_chat_passes_through_text(tmp_path, monkeypatch):
    captured = _install_fake_runtime(monkeypatch, generate_reply="hi there")
    llm = _ready_llm(tmp_path)
    out = llm.chat([{"role": "user", "content": "hello"}], max_tokens=10)
    assert out == "hi there"
    assert "user: hello" in captured["prompt"]
    assert captured["kwargs"]["max_tokens"] == 10


def test_chat_json_parses_valid_json(tmp_path, monkeypatch):
    _install_fake_runtime(monkeypatch, generate_reply='{"name": "Артем", "confidence": "high"}')
    llm = _ready_llm(tmp_path)
    out = llm.chat_json([{"role": "user", "content": "intro"}])
    assert out == {"name": "Артем", "confidence": "high"}


def test_chat_json_raises_on_bad_json(tmp_path, monkeypatch):
    _install_fake_runtime(monkeypatch, generate_reply="not json")
    llm = _ready_llm(tmp_path)
    with pytest.raises(LLMError, match="unparseable JSON"):
        llm.chat_json([{"role": "user", "content": "x"}])


def test_close_is_idempotent(tmp_path, monkeypatch):
    _install_fake_runtime(monkeypatch, generate_reply="x")
    llm = _ready_llm(tmp_path)
    llm.load()
    llm.close()
    llm.close()  # must not raise


def test_load_is_idempotent(tmp_path, monkeypatch):
    calls = {"n": 0}

    def counting_load(_path):
        calls["n"] += 1
        return ("M", _FakeTokenizer())

    monkeypatch.setattr(llm_module, "_build_tokenizer_data", lambda _t: object())
    monkeypatch.setattr(llm_module.mlx_lm, "load", counting_load)
    llm = _ready_llm(tmp_path)
    llm.load()
    llm.load()
    assert calls["n"] == 1


def test_context_manager_closes(tmp_path, monkeypatch):
    _install_fake_runtime(monkeypatch, generate_reply="x")
    llm = _ready_llm(tmp_path)
    with llm:
        llm.load()
        assert llm._model is not None
    assert llm._model is None
