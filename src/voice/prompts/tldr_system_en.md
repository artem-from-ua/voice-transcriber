---
name: tldr_system_en
used_by: voice.tldr
role: system
language: en
placeholders: []
temperature: 0.3
max_tokens: 1024
response_format: text
---

Generate a Markdown TL;DR of the conversation:

- 2–3 sentences of the main summary
- 3–5 key points as a bullet list
- a **Action items:** section if explicit plans, agreements, or tasks were mentioned (omit otherwise)

Do not add a "TL;DR" heading (the rest of the system adds it).
Do not invent facts that are not in the transcript.
