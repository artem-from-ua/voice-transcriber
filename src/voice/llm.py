"""In-process LLM runtime backed by mlx-lm.

The pipeline used to talk to LM Studio over HTTP, which crashed on long
prompts after dozens of short ones (Metal allocator fragmentation). This
module loads the same MLX-quantised model file in-process, gives us full
control over the KV cache, and uses lm-format-enforcer as a logits
processor to guarantee valid JSON output for the structure / identify
stages.

Public surface mirrors the old LM Studio client:

    with MlxLLM() as llm:
        llm.load()
        text = llm.chat(messages, temperature=0.2, max_tokens=512)
        payload = llm.chat_json(messages, temperature=0.1, max_tokens=64)
"""

from __future__ import annotations

import functools
import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Sequence

import mlx.core as mx
import mlx_lm
from lmformatenforcer import JsonSchemaParser, TokenEnforcer
from lmformatenforcer.tokenenforcer import TokenEnforcerTokenizerData

from ._memory import free_mlx


DEFAULT_MODEL = os.path.expanduser(
    "~/.cache/lm-studio/models/mlx-community/gemma-3-12b-it-qat-4bit"
)


class LLMError(RuntimeError):
    """Raised when the model directory is missing or output is unusable."""


def _build_tokenizer_data(tokenizer) -> TokenEnforcerTokenizerData:
    """Mirror of lmformatenforcer.integrations.transformers.build_token_enforcer_tokenizer_data,
    written here so we don't have to import the transformers extra."""
    underlying = getattr(tokenizer, "_tokenizer", tokenizer)
    vocab_size = len(underlying)
    token_0 = underlying.encode("0")[-1]
    regular: list[tuple[int, str, bool]] = []
    for idx in range(vocab_size):
        if idx in underlying.all_special_ids:
            continue
        decoded_after_0 = underlying.decode([token_0, idx])[1:]
        decoded_regular = underlying.decode([idx])
        is_word_start = len(decoded_after_0) > len(decoded_regular)
        regular.append((idx, decoded_after_0, is_word_start))

    def _decode(tokens: list[int]) -> str:
        return underlying.decode(tokens).rstrip("�")

    return TokenEnforcerTokenizerData(
        regular, _decode, underlying.eos_token_id, False, vocab_size
    )


def _make_json_logits_processor(
    tokenizer_data: TokenEnforcerTokenizerData,
    schema: dict[str, Any],
) -> Callable[[mx.array, mx.array], mx.array]:
    """Build a logits_processor that masks tokens disallowed by the schema."""
    parser = JsonSchemaParser(schema)
    enforcer = TokenEnforcer(tokenizer_data, parser)

    def processor(input_tokens: mx.array, logits: mx.array) -> mx.array:
        # mlx-lm passes the full prompt + generated tokens; the enforcer
        # tracks its own state via repeated calls with growing prefixes.
        seq = input_tokens.tolist()
        token_list = enforcer.get_allowed_tokens(seq)
        # Built with use_bitmask=False, so .allowed_tokens is a plain list[int].
        allowed = token_list.allowed_tokens
        if not allowed:
            return logits
        mask = mx.full(logits.shape, -mx.inf, dtype=logits.dtype)
        idx = mx.array(allowed)
        mask[..., idx] = 0
        return logits + mask

    return processor


@dataclass
class MlxLLM:
    """In-process Gemma (or any MLX-supported instruct model)."""

    model_path: str = DEFAULT_MODEL
    log: Callable[[str], None] = field(default=lambda _s: None)
    _model: Any = None
    _tokenizer: Any = None
    _tokenizer_data: TokenEnforcerTokenizerData | None = None

    # ------------------------------------------------------------------ lifecycle

    def health_check(self) -> str:
        """Verify the model directory looks like an MLX checkpoint.

        Returns the resolved path on success. Raises `LLMError` otherwise.
        """
        path = Path(os.path.expanduser(self.model_path))
        if not path.is_dir():
            raise LLMError(
                f"LLM model not found at {path}. "
                "Download it in LM Studio (Models → Search → mlx-community/...)."
            )
        if not (path / "config.json").is_file():
            raise LLMError(f"Missing config.json in {path}; is this an MLX model?")
        return str(path)

    def load(self) -> None:
        """Load weights + tokenizer. Idempotent."""
        if self._model is not None:
            return
        path = self.health_check()
        self.log(f"Loading LLM: {path}")
        self._model, self._tokenizer = mlx_lm.load(path)
        self._tokenizer_data = _build_tokenizer_data(self._tokenizer)

    def close(self) -> None:
        """Drop the model and clear MLX cache. Idempotent."""
        if self._model is None and self._tokenizer is None:
            return
        self._model = None
        self._tokenizer = None
        self._tokenizer_data = None
        free_mlx(self.log)

    def __enter__(self) -> "MlxLLM":
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    # ------------------------------------------------------------------ helpers

    def _ensure_loaded(self) -> None:
        if self._model is None:
            self.load()

    def _build_prompt(self, messages: Sequence[dict[str, str]]) -> str:
        assert self._tokenizer is not None
        return self._tokenizer.apply_chat_template(
            list(messages), add_generation_prompt=True, tokenize=False
        )

    def _stream_generate(
        self,
        *,
        prompt: str,
        max_tokens: int,
        sampler: Callable,
        logits_processors: list,
        on_token: Callable[[int], None] | None,
    ) -> str:
        """Run mlx_lm.stream_generate, accumulate text, ping on_token."""
        parts: list[str] = []
        for response in mlx_lm.stream_generate(
            self._model,
            self._tokenizer,
            prompt=prompt,
            max_tokens=max_tokens,
            sampler=sampler,
            logits_processors=logits_processors,
        ):
            parts.append(response.text)
            if on_token is not None:
                on_token(1)
        return "".join(parts)

    # ------------------------------------------------------------------ chat

    def chat(
        self,
        messages: Sequence[dict[str, str]],
        *,
        temperature: float = 0.2,
        max_tokens: int = 512,
        top_p: float = 1.0,
        repetition_penalty: float | None = None,
        on_token: Callable[[int], None] | None = None,
    ) -> str:
        """Plain-text generation. `on_token(1)` is called for each emitted token."""
        self._ensure_loaded()
        prompt = self._build_prompt(messages)
        sampler = mlx_lm.sample_utils.make_sampler(
            temp=temperature, top_p=top_p,
        )
        logits_processors = mlx_lm.sample_utils.make_logits_processors(
            repetition_penalty=repetition_penalty,
        )
        text = self._stream_generate(
            prompt=prompt,
            max_tokens=max_tokens,
            sampler=sampler,
            logits_processors=logits_processors,
            on_token=on_token,
        )
        # Free KV cache between calls so postprocess's 60+ short prompts
        # don't poison the long structure prompt that follows.
        mx.clear_cache()
        return text

    def chat_json(
        self,
        messages: Sequence[dict[str, str]],
        *,
        schema: dict[str, Any] | None = None,
        temperature: float = 0.2,
        max_tokens: int = 1024,
        top_p: float = 1.0,
        on_token: Callable[[int], None] | None = None,
    ) -> Any:
        """Schema-constrained generation. Returns parsed JSON.

        With `lm-format-enforcer` driving the logits, the output is forced
        into a valid JSON object — no retry loop necessary. `LLMError` is
        only raised on a programmatic bug (parser collapse, empty output).
        """
        self._ensure_loaded()
        assert self._tokenizer_data is not None
        prompt = self._build_prompt(messages)
        sampler = mlx_lm.sample_utils.make_sampler(
            temp=temperature, top_p=top_p,
        )
        json_processor = _make_json_logits_processor(
            self._tokenizer_data,
            schema or {"type": "object"},
        )
        text = self._stream_generate(
            prompt=prompt,
            max_tokens=max_tokens,
            sampler=sampler,
            logits_processors=[json_processor],
            on_token=on_token,
        )
        mx.clear_cache()

        try:
            return json.loads(text)
        except json.JSONDecodeError as exc:
            raise LLMError(
                f"Constrained generation produced unparseable JSON: {text!r}"
            ) from exc
