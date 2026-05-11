"""Prompt loader.

Reads `src/voice/prompts/<name>.md` as package data, strips an optional
YAML frontmatter block, and renders simple `<<placeholder>>` substitutions.

The custom `<<...>>` delimiters were chosen to avoid clashing with JSON
braces (used inline in many prompts as example schemas) and shell-style
`$`-substitutions.
"""

from __future__ import annotations

import re
from functools import lru_cache
from importlib.resources import files
from typing import Any


_FRONTMATTER_RE = re.compile(r"\A---\s*\n.*?\n---\s*\n", re.DOTALL)
_PLACEHOLDER_RE = re.compile(r"<<\s*([A-Za-z_][A-Za-z0-9_]*)\s*>>")


@lru_cache(maxsize=None)
def _read_raw(name: str) -> str:
    """Read a prompt file as text. Cached because prompts are immutable."""
    res = files("voice.prompts").joinpath(f"{name}.md")
    return res.read_text(encoding="utf-8")


def _strip_frontmatter(text: str) -> str:
    return _FRONTMATTER_RE.sub("", text, count=1)


def render(name: str, **values: Any) -> str:
    """Load prompt `name` and substitute `<<key>>` with `str(values[key])`.

    Missing placeholders raise `KeyError`; extra placeholders are ignored.
    """
    body = _strip_frontmatter(_read_raw(name)).strip()

    def repl(match: re.Match[str]) -> str:
        key = match.group(1)
        if key not in values:
            raise KeyError(f"Prompt {name!r} expects placeholder <<{key}>>")
        return str(values[key])

    return _PLACEHOLDER_RE.sub(repl, body)


def list_placeholders(name: str) -> list[str]:
    """Return the placeholder names referenced in a prompt body (order-preserving, deduped)."""
    body = _strip_frontmatter(_read_raw(name))
    seen: list[str] = []
    for match in _PLACEHOLDER_RE.finditer(body):
        key = match.group(1)
        if key not in seen:
            seen.append(key)
    return seen
