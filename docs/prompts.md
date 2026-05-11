# LLM prompts

Prompts are kept as plain Markdown files in `src/voice/prompts/` and loaded by `voice._prompts.render()`. The Python modules only describe *how* a prompt is used (parameters, response parsing, safety net); the prompt text itself lives in one canonical place.

## Files

| File | Role | Used by | Placeholders |
|------|------|---------|--------------|
| [`identify_system.md`](../src/voice/prompts/identify_system.md) | system | `voice.identify` | `language` |
| [`identify_user.md`](../src/voice/prompts/identify_user.md) | user | `voice.identify` | `snippet` |
| [`postprocess_system.md`](../src/voice/prompts/postprocess_system.md) | system | `voice.postprocess` | `language` |
| [`postprocess_user.md`](../src/voice/prompts/postprocess_user.md) | user | `voice.postprocess` | `text` |
| [`structure_system.md`](../src/voice/prompts/structure_system.md) | system | `voice.structure` | `language` |
| [`structure_user.md`](../src/voice/prompts/structure_user.md) | user | `voice.structure` | `total_start_ms`, `total_end_ms`, `script` |
| [`tldr_system_uk.md`](../src/voice/prompts/tldr_system_uk.md) | system | `voice.tldr` (uk) | — |
| [`tldr_system_en.md`](../src/voice/prompts/tldr_system_en.md) | system | `voice.tldr` (en) | — |

## Template syntax

Each file starts with a YAML frontmatter (`name`, `used_by`, `role`, `placeholders`, sometimes `response_format` or `language`). The loader strips the frontmatter; only the body reaches the model.

Placeholder substitution uses `<<key>>` delimiters — not `{}` or `${}` — to keep JSON examples in the prompt body intact. `render("identify_system", language="uk")` returns the prompt with `<<language>>` replaced; a missing argument raises `KeyError`.

`list_placeholders(name)` returns the order-preserving, de-duplicated set of placeholders a file uses. The `test_all_pipeline_prompts_load` test exercises every shipped file with empty values to guarantee everything resolves at startup.

## LLM call contracts

The wire-level contract (temperature, max_tokens, response_format) stays in the Python module that calls each prompt. The combinations are:

| Stage | Function | Temperature | max_tokens | response_format | Retries |
|-------|----------|-------------|------------|-----------------|---------|
| identify | `LLMClient.chat_json` | 0.1 | 64 | json_schema (object) | 1 |
| postprocess | `LLMClient.chat` | 0.1 | 512 | text (default) | 0 |
| structure | `LLMClient.chat_json` | 0.2 | 2048 | json_schema (object) | 1 |
| tldr | `LLMClient.chat` | 0.3 | 1024 | text (default) | 0 |

For LM Studio's MLX runtime the response format value is sent as `{"type": "json_schema", "json_schema": {"name": "response", "strict": false, "schema": {"type": "object"}}}` — the OpenAI-classic `{"type": "json_object"}` is rejected with HTTP 400. See [ADR 0001](adr/0001-local-llm-via-lm-studio.md) for the LM Studio choice and `CHANGELOG.md` 0.6.1 for the fix.

## Safety nets

| Stage | Validation |
|-------|------------|
| identify | name must start with an uppercase letter, ≥ 2 chars, confidence ≠ low; duplicate-name conflicts resolved by confidence then first appearance |
| postprocess | reply rejected if length ratio > 2× either way or Levenshtein / max length > 0.5; surrounding quotes stripped |
| structure | JSON must be `{"sections":[…]}` with 2–7 entries, contiguous, covering the dialogue exactly; otherwise fallback to single "Розмова" / "Conversation" section |
| tldr | LLM error → empty string; renderer omits the whole "## TL;DR" block silently |

See each module under `src/voice/` for the implementation of these checks.

## Editing prompts

Prompt edits do not require a code change: edit the `.md` file and rerun. The loader is `lru_cache`-d, so a fresh Python process is needed to pick up the change (the CLI is single-shot anyway). The frontmatter is informational; keep `placeholders:` honest for the test suite to remain green.
