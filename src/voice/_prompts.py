"""Prompt loader.

Reads `src/voice/prompts/<name>.md`, parses an optional YAML frontmatter,
and renders simple `<<placeholder>>` substitutions in the body.

The frontmatter doubles as both documentation (`name`, `used_by`, `role`,
`placeholders`) and runtime parameters for the LLM call (`temperature`,
`max_tokens`, `top_p`, `response_format`). Callers can read those via
`load_prompt(name).params`.

`<<...>>` delimiters were chosen so JSON braces in the prompt body
(e.g. example schemas) survive verbatim.
"""

from __future__ import annotations

import ast
import re
from dataclasses import dataclass, field
from functools import lru_cache
from importlib.resources import files
from typing import Any


_FRONTMATTER_RE = re.compile(r"\A---\s*\n(.*?)\n---\s*\n", re.DOTALL)
_PLACEHOLDER_RE = re.compile(r"<<\s*([A-Za-z_][A-Za-z0-9_]*)\s*>>")

# Keys reserved for documentation/metadata. Everything else in the frontmatter
# is exposed as LLM call parameters.
_METADATA_KEYS = frozenset({"name", "used_by", "role", "language"})


@dataclass(frozen=True)
class PromptFile:
    name: str
    body: str
    placeholders: list[str] = field(default_factory=list)
    params: dict[str, Any] = field(default_factory=dict)

    def render(self, **values: Any) -> str:
        """Substitute `<<key>>` with `str(values[key])` in the body."""
        def repl(match: re.Match[str]) -> str:
            key = match.group(1)
            if key not in values:
                raise KeyError(f"Prompt {self.name!r} expects placeholder <<{key}>>")
            return str(values[key])
        return _PLACEHOLDER_RE.sub(repl, self.body)


def _parse_scalar(raw: str) -> Any:
    """Parse a YAML scalar value the loose, dependency-free way.

    Supports ints, floats, booleans, null, single-line lists, and strings.
    Wraps `ast.literal_eval` for the structural cases.
    """
    s = raw.strip()
    if not s:
        return ""
    lower = s.lower()
    if lower == "null" or lower == "none" or lower == "~":
        return None
    if lower == "true":
        return True
    if lower == "false":
        return False
    if s.startswith(("'", '"')):
        try:
            return ast.literal_eval(s)
        except (ValueError, SyntaxError):
            return s
    if s.startswith("[") and s.endswith("]"):
        try:
            return ast.literal_eval(s)
        except (ValueError, SyntaxError):
            inner = s[1:-1].strip()
            if not inner:
                return []
            return [_parse_scalar(part) for part in inner.split(",")]
    try:
        return int(s)
    except ValueError:
        pass
    try:
        return float(s)
    except ValueError:
        pass
    return s


def _parse_frontmatter(text: str) -> tuple[dict[str, Any], str]:
    """Return (parsed_keys, body_without_frontmatter)."""
    match = _FRONTMATTER_RE.match(text)
    if not match:
        return {}, text
    body = text[match.end():]
    keys: dict[str, Any] = {}
    for line in match.group(1).splitlines():
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        if ":" not in line:
            continue
        key, _, value = line.partition(":")
        keys[key.strip()] = _parse_scalar(value)
    return keys, body


@lru_cache(maxsize=None)
def load_prompt(name: str) -> PromptFile:
    """Read, parse, and cache a prompt file."""
    res = files("voice.prompts").joinpath(f"{name}.md")
    raw = res.read_text(encoding="utf-8")
    frontmatter, body = _parse_frontmatter(raw)
    body = body.strip()
    declared = frontmatter.get("placeholders")
    if isinstance(declared, list):
        placeholders = [str(p) for p in declared]
    else:
        placeholders = []
    params = {
        k: v for k, v in frontmatter.items()
        if k not in _METADATA_KEYS and k != "placeholders"
    }
    return PromptFile(name=name, body=body, placeholders=placeholders, params=params)


def render(name: str, **values: Any) -> str:
    """Convenience: load and render in one call."""
    return load_prompt(name).render(**values)


_USER_CONTEXT_PREFIX_UK = "Ця розмова описана користувачем як:"
_USER_CONTEXT_PREFIX_EN = "The user described this conversation as:"


def with_user_context(
    system_text: str,
    user_context: str | None,
    *,
    language: str,
) -> str:
    """Prepend a per-run user-supplied context block to a rendered system prompt.

    When `user_context` is `None` or blank, returns `system_text` unchanged so
    the call site stays a no-op for the default case. Otherwise prepends a
    short header (Ukrainian for `language="uk"`, English for everything else)
    followed by the trimmed context and a blank line.

    See ADR 0033 for the rationale (prefix-outside vs `<<placeholder>>`-inside).
    """
    if user_context is None:
        return system_text
    text = user_context.strip()
    if not text:
        return system_text
    prefix = _USER_CONTEXT_PREFIX_UK if language == "uk" else _USER_CONTEXT_PREFIX_EN
    return f"{prefix}\n{text}\n\n{system_text}"


_LLM_PARAM_KEYS = frozenset({
    "temperature", "max_tokens", "top_p", "repetition_penalty",
})


def call_kwargs(name: str) -> dict[str, Any]:
    """Extract LLM call kwargs (temperature, max_tokens, ...) from a prompt's
    frontmatter. `response_format` is intentionally omitted — JSON mode is
    selected at the call site by picking `chat_json()` over `chat()`."""
    params = load_prompt(name).params
    return {k: v for k, v in params.items() if k in _LLM_PARAM_KEYS}


def list_placeholders(name: str) -> list[str]:
    """Return placeholders actually referenced in the prompt body."""
    body = load_prompt(name).body
    seen: list[str] = []
    for match in _PLACEHOLDER_RE.finditer(body):
        key = match.group(1)
        if key not in seen:
            seen.append(key)
    return seen
