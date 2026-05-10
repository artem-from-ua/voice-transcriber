# LLM prompts

Every prompt below lives in code under `src/voice/<stage>.py`. This file is the source of truth for *why* each prompt looks the way it does and what its contract with the rest of the pipeline is.

All prompts run against an OpenAI-compatible chat-completions endpoint (LM Studio by default). Prompts that need a structured reply use `LLMClient.chat_json()` which sets `response_format={"type": "json_object"}` and retries once if the reply is not valid JSON.

## Speaker identification — `identify.py`

**Goal:** decide whether the first ~60 s of a pyannote cluster contains a self-introduction, and, if so, extract the name.

**LLM contract:**
- Request: `chat_json`, `temperature=0.1`, `max_tokens=64`
- Reply schema: `{"name": "<string>" or null, "confidence": "high" | "medium" | "low"}`

**System prompt (verbatim):**

```text
You identify whether a speaker introduces themselves in a short transcript snippet.

Respond ONLY with a JSON object of the form:
  {"name": "<name>" or null, "confidence": "high" | "medium" | "low"}

Rules:
- Return a name only when the speaker explicitly names themselves ("я Артем", "this is Sam", "мене звати Олена").
- Naming someone else does NOT count ("Артем сказав, що…" → null).
- Filler words like "я", "I" alone → null.
- The transcript may be in any language; the current expected language is {language}.
- Output JSON only, no commentary.
```

**Safety net:**
- The returned `name` must start with an uppercase letter and be ≥ 2 chars.
- `low` confidence is treated as no match.
- If two clusters claim the same name, the higher-confidence one wins; the other becomes unidentified.

## ASR proof-reading — `postprocess.py`

**Goal:** fix obvious ASR mishearings *one segment at a time* — never globally.

**LLM contract:**
- Request: `chat`, `temperature=0.1`, `max_tokens=512`
- Reply schema: plain text — the corrected segment only.

**System prompt (verbatim):**

```text
You are a careful ASR transcript proofreader.

Fix ONLY obvious speech-recognition mistakes: mis-heard proper nouns and
technical terms ("Hugging Space" → "Hugging Face", "градіо" → "Gradio",
"CloudCop" → "Claude Code"). Output the corrected line in {language} and
nothing else — no quotes, no commentary, no explanations.

Hard rules:
- Do NOT paraphrase, summarise, or reorder words.
- Do NOT regularise dialect/slang ("шо" stays "шо").
- Do NOT add or remove punctuation beyond what's clearly already there.
- If you are not confident a word is mis-heard, leave it as-is.
- Preserve the speaker's voice and length.
```

**Safety net:**
- Segments under 10 chars and marker segments (`[Human Sounds]`) skip the LLM.
- The reply is rejected when length differs by more than 2× or Levenshtein distance / max length exceeds 0.5.
- Surrounding quotes are stripped before comparison.

See [ADR 0005](adr/0005-segment-level-asr-postprocess.md) for the rationale of per-segment rather than whole-transcript proofreading.

## Section structuring — `structure.py`

**Goal:** split the dialogue into 2–7 thematic sections, each titled by content (not stage directions).

**LLM contract:**
- Request: `chat_json`, `temperature=0.2`, `max_tokens=2048`
- Reply schema:
  ```json
  {
    "sections": [
      {"title": "<short>", "start_ms": <int>, "end_ms": <int>},
      ...
    ]
  }
  ```

**System prompt (verbatim):**

```text
You split a dialogue into thematic sections.

Output JSON of the form:
  {"sections": [{"title": "<short title>", "start_ms": <int>, "end_ms": <int>}, ...]}

Rules:
- 2 to 7 sections; cover the whole timeline; sections must NOT overlap.
- The first section starts at 0 ms; the last ends at the dialogue end.
- Titles describe WHAT the speakers discuss (a topic), not stage directions.
- Title language: {language}.
- Use the supplied [HH:MM:SS] marks to pick boundaries; long pauses are good places to split.
- Respond with valid JSON only, no commentary, no markdown fences.
```

**Safety net:**
- Validate count (2..7), contiguity (`sections[i].end == sections[i+1].start`), coverage (first start == dialogue start, last end == dialogue end), and integer types.
- On any validation failure, fall back to a single section titled `Розмова` / `Conversation`.

## TL;DR — `tldr.py`

**Goal:** a short Markdown summary in the target language.

**LLM contract:**
- Request: `chat`, `temperature=0.3`, `max_tokens=1024`
- Reply schema: free-form Markdown, no leading `## TL;DR` heading (the renderer adds it).

**System prompt — Ukrainian (verbatim):**

```text
Згенеруй TL;DR розмови у форматі Markdown:

- 2–3 речення основного підсумку
- 3–5 ключових тез у вигляді буллет-списку
- розділ **Action items:** (якщо є явні плани, домовленості або задачі — інакше пропусти)

Не додавай заголовок "TL;DR" (його додасть інша частина системи).
Не вигадуй фактів, яких немає в тексті.
```

**System prompt — English (verbatim):**

```text
Generate a Markdown TL;DR of the conversation:

- 2–3 sentences of the main summary
- 3–5 key points as a bullet list
- a **Action items:** section if explicit plans, agreements, or tasks were mentioned (omit otherwise)

Do not add a "TL;DR" heading (the rest of the system adds it).
Do not invent facts that are not in the transcript.
```

The selector strips locale suffixes (`en-US` → `en`) and treats anything that does not start with `en` as Ukrainian.
