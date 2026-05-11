"""Tests for the prompt loader and the shipped prompt files."""

from __future__ import annotations

import pytest

from voice._prompts import call_kwargs, list_placeholders, load_prompt, render


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


# ---------------------------------------------------------------------------
# Frontmatter params

@pytest.mark.parametrize("name", [
    "identify_system",
    "postprocess_system",
    "structure_system",
    "tldr_system_uk",
    "tldr_system_en",
])
def test_system_prompts_declare_temperature_and_max_tokens(name):
    params = load_prompt(name).params
    assert "temperature" in params, f"{name} missing temperature"
    assert "max_tokens" in params, f"{name} missing max_tokens"
    assert 0.0 <= params["temperature"] <= 2.0
    assert params["max_tokens"] > 0


@pytest.mark.parametrize("name,expected", [
    ("identify_system", "json_object"),
    ("structure_system", "json_object"),
    ("postprocess_system", "text"),
    ("tldr_system_uk", "text"),
    ("tldr_system_en", "text"),
])
def test_response_format_is_known_value(name, expected):
    params = load_prompt(name).params
    assert params.get("response_format") == expected


def test_call_kwargs_strips_response_format():
    """call_kwargs() must not return response_format — it's not an LLM param."""
    kwargs = call_kwargs("identify_system")
    assert "response_format" not in kwargs
    assert kwargs["temperature"] == 0.1
    assert kwargs["max_tokens"] == 64


def test_call_kwargs_omits_unknown_params():
    """Anything in frontmatter that's not a recognized LLM kwarg is dropped."""
    kwargs = call_kwargs("tldr_system_uk")
    # `language` lives in frontmatter as metadata but must not leak as a kwarg.
    assert "language" not in kwargs


def test_load_prompt_lists_placeholders_from_frontmatter():
    p = load_prompt("identify_system")
    assert p.placeholders == ["language"]


def test_load_prompt_parses_yaml_style_list():
    """The bracketed [language] list in frontmatter must parse cleanly,
    even though `language` is not quoted (which fails ast.literal_eval)."""
    p = load_prompt("structure_system")
    assert "language" in p.placeholders
