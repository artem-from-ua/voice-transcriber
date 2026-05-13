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
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterator, Sequence

import mlx.core as mx
import mlx_lm
from mlx_lm.models.cache import make_prompt_cache
from lmformatenforcer import JsonSchemaParser, TokenEnforcer
from lmformatenforcer.tokenenforcer import TokenEnforcerTokenizerData

from ._memory import free_mlx
from ._memory_stats import read_memory_snapshot


_GB = 1024 ** 3


def _reset_peak_memory() -> None:
    """Best-effort reset of MLX peak counter so per-call peaks are meaningful.

    `mx.get_peak_memory()` is a high-water mark over the process lifetime
    unless explicitly reset. Available as `mx.reset_peak_memory` in mlx
    ≥ 0.21 and under `mx.metal` on older builds. Silent no-op otherwise.
    """
    fn = getattr(mx, "reset_peak_memory", None)
    if fn is None:
        metal = getattr(mx, "metal", None)
        fn = getattr(metal, "reset_peak_memory", None) if metal is not None else None
    if callable(fn):
        try:
            fn()
        except Exception:  # noqa: BLE001 — telemetry must not crash the pipeline
            pass


DEFAULT_MODEL = "mlx-community/Qwen2.5-7B-Instruct-4bit"


class LLMError(RuntimeError):
    """Raised when the model directory is missing or output is unusable."""


def _resolve_model_path(model_spec: str) -> str:
    """Accept either a filesystem path or a HuggingFace `org/repo` id.

    - Filesystem paths (anything that exists on disk after `expanduser`) are
      returned as-is — this keeps the legacy LM Studio cache layout working
      for users with `--llm-model ~/.cache/lm-studio/...`.
    - HF repo ids are resolved via `try_to_load_from_cache` (no network).
      The returned snapshot directory is what `mlx_lm.load` accepts. If the
      repo is not cached, `LLMError` is raised with an actionable message —
      we never trigger an implicit multi-GB download from the pipeline.
    """
    expanded = os.path.expanduser(model_spec)
    if os.path.isdir(expanded):
        return expanded
    if "/" not in model_spec or model_spec.startswith((".", "/", "~")):
        # Looks like a filesystem path but the directory does not exist.
        # Let the caller see the path as-is so health_check can produce a
        # filesystem-flavoured error message.
        return expanded

    try:
        from huggingface_hub import try_to_load_from_cache
        from huggingface_hub.errors import CacheNotFound
    except ImportError as exc:
        raise LLMError(
            f"huggingface_hub not installed; cannot resolve repo id {model_spec!r}"
        ) from exc

    try:
        hit = try_to_load_from_cache(repo_id=model_spec, filename="config.json")
    except CacheNotFound:
        hit = None
    if hit is None:
        raise LLMError(
            f"LLM model {model_spec!r} is not in the HuggingFace cache. "
            f"Fetch it once with `huggingface-cli download {model_spec}` "
            f"(or `uv run python -c \"from huggingface_hub import snapshot_download; "
            f"snapshot_download('{model_spec}')\"`)."
        )
    return os.path.dirname(str(hit))


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
    """In-process Gemma (or any MLX-supported instruct model).

    `sampling_overrides` lets a caller force sampling params (temperature,
    top_p, top_k, repetition_penalty) across every chat/chat_json call,
    overriding what the prompt frontmatter requests. Keys missing from the
    dict leave the per-call defaults in place. Used by the pipeline to
    apply each model's officially-recommended params globally; the prompt
    frontmatter still drives max_tokens (which is task-, not model-,
    specific).
    """

    model_path: str = DEFAULT_MODEL
    log: Callable[[str], None] = field(default=lambda _s: None)
    log_memory: bool = False
    sampling_overrides: dict[str, Any] = field(default_factory=dict)
    _model: Any = None
    _tokenizer: Any = None
    _tokenizer_data: TokenEnforcerTokenizerData | None = None
    _resolved_path: str | None = None
    _last_load_s: float = 0.0
    _prompt_cache: list[Any] | None = None

    # ------------------------------------------------------------------ lifecycle

    def health_check(self) -> str:
        """Verify the model directory looks like an MLX checkpoint.

        Returns the resolved path on success. Raises `LLMError` otherwise.
        `self.model_path` may be a filesystem path (legacy LM Studio layout)
        or a HuggingFace `org/repo` id; both shapes resolve to a concrete
        directory here.
        """
        resolved = _resolve_model_path(self.model_path)
        path = Path(resolved)
        if not path.is_dir():
            raise LLMError(
                f"LLM model not found at {path} (from {self.model_path!r}). "
                f"For HF repos, run `huggingface-cli download {self.model_path}`. "
                f"For local paths, point --llm-model at an extracted MLX checkpoint."
            )
        if not (path / "config.json").is_file():
            raise LLMError(f"Missing config.json in {path}; is this an MLX model?")
        return str(path)

    def load(self) -> None:
        """Load weights + tokenizer. Idempotent."""
        if self._model is not None:
            return
        import time
        path = self.health_check()
        self.log(f"Loading LLM: {path}")
        t0 = time.perf_counter()
        self._model, self._tokenizer = mlx_lm.load(path)
        self._last_load_s = time.perf_counter() - t0
        self._tokenizer_data = _build_tokenizer_data(self._tokenizer)
        # Remember the resolved directory so the pipeline can compare two
        # MlxLLM instances by what they actually loaded — not by what the
        # caller typed (a repo id vs an absolute snapshot path that resolve
        # to the same directory should be treated as "same model").
        self._resolved_path = path

    def close(self) -> None:
        """Drop the model and clear MLX cache. Idempotent."""
        if self._model is None and self._tokenizer is None:
            return
        self._model = None
        self._tokenizer = None
        self._tokenizer_data = None
        self._resolved_path = None
        self._prompt_cache = None
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
        prompt_cache: list[Any] | None = None,
    ) -> tuple[str, int]:
        """Run mlx_lm.stream_generate, accumulate text, ping on_token.

        Returns (text, output_token_count). When `prompt_cache` is set, the
        kwarg is forwarded to mlx_lm; otherwise the call site stays
        byte-identical to the no-cache path (no kwarg passed at all).
        """
        parts: list[str] = []
        tokens = 0
        extra: dict[str, Any] = {}
        if prompt_cache is not None:
            extra["prompt_cache"] = prompt_cache
        for response in mlx_lm.stream_generate(
            self._model,
            self._tokenizer,
            prompt=prompt,
            max_tokens=max_tokens,
            sampler=sampler,
            logits_processors=logits_processors,
            **extra,
        ):
            parts.append(response.text)
            tokens += 1
            if on_token is not None:
                on_token(1)
        return "".join(parts), tokens

    def _prompt_token_count(self, prompt: str) -> int:
        """Best-effort token count of the rendered prompt for telemetry."""
        if self._tokenizer is None:
            return 0
        try:
            ids = self._tokenizer.encode(prompt)
            return len(ids) if ids is not None else 0
        except Exception:  # noqa: BLE001
            return 0

    def _log_memory_line(
        self,
        *,
        op: str,
        pre_active_gb: float | None,
        peak_gb: float | None,
        post_active_gb: float | None,
        prompt_tokens: int,
        output_tokens: int,
    ) -> None:
        def _fmt(v: float | None) -> str:
            return f"{v:.2f} GB" if v is not None else "n/a"

        self.log(
            f"llm.{op}: pre {_fmt(pre_active_gb)}, peak {_fmt(peak_gb)}, "
            f"post-clear {_fmt(post_active_gb)}, "
            f"prompt {prompt_tokens} tok, out {output_tokens} tok"
        )

    # ------------------------------------------------------------------ chat

    def chat(
        self,
        messages: Sequence[dict[str, str]],
        *,
        temperature: float = 0.2,
        max_tokens: int = 512,
        top_p: float = 1.0,
        top_k: int = 0,
        repetition_penalty: float | None = None,
        on_token: Callable[[int], None] | None = None,
    ) -> str:
        """Plain-text generation. `on_token(1)` is called for each emitted token."""
        self._ensure_loaded()
        prompt = self._build_prompt(messages)
        temperature = self.sampling_overrides.get("temperature", temperature)
        top_p = self.sampling_overrides.get("top_p", top_p)
        top_k = self.sampling_overrides.get("top_k", top_k)
        repetition_penalty = self.sampling_overrides.get(
            "repetition_penalty", repetition_penalty
        )
        sampler = mlx_lm.sample_utils.make_sampler(
            temp=temperature, top_p=top_p, top_k=top_k,
        )
        logits_processors = mlx_lm.sample_utils.make_logits_processors(
            repetition_penalty=repetition_penalty,
        )
        if self.log_memory:
            _reset_peak_memory()
            pre = read_memory_snapshot()
            prompt_tok = self._prompt_token_count(prompt)
        text, out_tokens = self._stream_generate(
            prompt=prompt,
            max_tokens=max_tokens,
            sampler=sampler,
            logits_processors=logits_processors,
            on_token=on_token,
            prompt_cache=self._prompt_cache,
        )
        # Free KV cache between calls so proofread's 60+ short prompts
        # don't poison the long structure prompt that follows. Inside a
        # prompt_cache_session(), we deliberately keep the cache alive
        # across calls (shared system prompt); the session's __exit__
        # clears it exactly once on the way out.
        if self.log_memory:
            peak = read_memory_snapshot()
        if self._prompt_cache is None:
            mx.clear_cache()
        if self.log_memory:
            post = read_memory_snapshot()
            self._log_memory_line(
                op="chat",
                pre_active_gb=pre.mlx_active_gb,
                peak_gb=peak.mlx_peak_gb,
                post_active_gb=post.mlx_active_gb,
                prompt_tokens=prompt_tok,
                output_tokens=out_tokens,
            )
        return text

    def chat_json(
        self,
        messages: Sequence[dict[str, str]],
        *,
        schema: dict[str, Any] | None = None,
        temperature: float = 0.2,
        max_tokens: int = 1024,
        top_p: float = 1.0,
        top_k: int = 0,
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
        temperature = self.sampling_overrides.get("temperature", temperature)
        top_p = self.sampling_overrides.get("top_p", top_p)
        top_k = self.sampling_overrides.get("top_k", top_k)
        sampler = mlx_lm.sample_utils.make_sampler(
            temp=temperature, top_p=top_p, top_k=top_k,
        )
        json_processor = _make_json_logits_processor(
            self._tokenizer_data,
            schema or {"type": "object"},
        )
        if self.log_memory:
            _reset_peak_memory()
            pre = read_memory_snapshot()
            prompt_tok = self._prompt_token_count(prompt)
        text, out_tokens = self._stream_generate(
            prompt=prompt,
            max_tokens=max_tokens,
            sampler=sampler,
            logits_processors=[json_processor],
            on_token=on_token,
        )
        if self.log_memory:
            peak = read_memory_snapshot()
        mx.clear_cache()
        if self.log_memory:
            post = read_memory_snapshot()
            self._log_memory_line(
                op="chat_json",
                pre_active_gb=pre.mlx_active_gb,
                peak_gb=peak.mlx_peak_gb,
                post_active_gb=post.mlx_active_gb,
                prompt_tokens=prompt_tok,
                output_tokens=out_tokens,
            )

        try:
            return json.loads(text)
        except json.JSONDecodeError as exc:
            raise LLMError(
                f"Constrained generation produced unparseable JSON: {text!r}"
            ) from exc

    # ------------------------------------------------------------------ prompt-cache session

    @contextmanager
    def prompt_cache_session(self) -> Iterator[None]:
        """Share an mlx-lm prompt cache across `chat()` calls inside the block.

        For stages like proofread that send 60+ short user prompts under the
        same system prompt: the system-prompt KV state is computed once on
        the first call and reused, cutting wall-clock by ~30-40%.

        Scope: only `chat()` participates. `chat_json()` is excluded —
        lm-format-enforcer × prompt_cache is not verified.

        Re-entrancy: nested sessions are forbidden (asserted). On exit the
        cache is dropped and `mx.clear_cache()` runs exactly once, matching
        the per-call clear behaviour callers outside the session see.
        """
        self._ensure_loaded()
        assert self._prompt_cache is None, "prompt_cache_session is not re-entrant"
        self._prompt_cache = make_prompt_cache(self._model)
        try:
            yield
        finally:
            self._prompt_cache = None
            mx.clear_cache()
