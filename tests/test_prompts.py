"""Tests for the prompt loader and the shipped prompt files."""

from __future__ import annotations

import pytest

from voice._prompts import list_placeholders, render


def test_render_strips_frontmatter():
    out = render("identify_system", language="uk")
    assert not out.startswith("---")
    assert "name: identify_system" not in out


def test_render_substitutes_placeholders():
    out = render("identify_system", language="Ukrainian")
    assert "expected language is Ukrainian" in out


def test_render_preserves_json_braces():
    """Custom <<>> delimiter must not eat JSON braces in the prompt body."""
    out = render("identify_system", language="uk")
    assert '"name"' in out
    assert "{" in out and "}" in out


def test_render_user_prompt_with_snippet():
    out = render("identify_user", snippet="hello world")
    assert "hello world" in out


def test_missing_placeholder_raises_key_error():
    with pytest.raises(KeyError, match="language"):
        render("identify_system")  # no kwargs


def test_extra_kwargs_are_ignored():
    out = render("identify_system", language="uk", extra="ignored")
    assert "ignored" not in out


def test_list_placeholders_for_structure_user():
    placeholders = list_placeholders("structure_user")
    assert placeholders == ["total_start_ms", "total_end_ms", "script"]


def test_list_placeholders_dedupes_and_preserves_order():
    placeholders = list_placeholders("identify_system")
    assert placeholders == ["language"]


def test_tldr_prompts_have_no_placeholders():
    assert list_placeholders("tldr_system_uk") == []
    assert list_placeholders("tldr_system_en") == []
    # And render without args.
    render("tldr_system_uk")
    render("tldr_system_en")


def test_all_pipeline_prompts_load():
    for name in [
        "identify_system",
        "identify_user",
        "postprocess_system",
        "postprocess_user",
        "structure_system",
        "structure_user",
        "tldr_system_uk",
        "tldr_system_en",
    ]:
        # Use empty values for every placeholder the file declares.
        kwargs = {p: "_" for p in list_placeholders(name)}
        text = render(name, **kwargs)
        assert text.strip(), f"{name} rendered to empty"
