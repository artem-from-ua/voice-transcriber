# Output format

The pipeline writes a single Markdown file. Its structure is fixed; what varies is whether the TL;DR and section H2 headings are present.

## Anatomy

```markdown
# Транскрипт: <audio basename>

> 📅 **Початок:** <ISO datetime, UTC>
> ⏱️ **Тривалість:** <H:MM:SS or MM:SS>
> 🌐 **Мова:** <language code>

## TL;DR                              # optional — omitted with --no-tldr

<llm-generated markdown>

---

## <section title 1>

<emoji> **<name>:** <utterance 1> <utterance 2 if same speaker, short pause> …

<emoji> **<other name>:** …

## <section title 2>
…
```

## Emoji palette

The palette in `speaker_emojis.py` is fixed and cycles past nine speakers:

```
🔵 🟢 🔴 🟡 🟣 🟠 🟤 ⚪️ ⚫️
```

Speakers are assigned by **first appearance in time**, not by pyannote label index. A given speaker keeps the same emoji throughout the document.

## Paragraph rules

A new paragraph in a section starts when:

- the speaker changes, **or**
- the gap from the previous utterance of the same speaker exceeds **2 seconds**.

A gap of **3 seconds or more** is rendered explicitly as a quoted pause marker between paragraphs:

```markdown
🔵 **Олена:** перше речення.

> _[пауза 8с]_

🔵 **Олена:** друге речення.
```

## Speakerless segments

ASR markers like `[Human Sounds]` and `[Silence]` have no pyannote speaker and are **not rendered**. They still influence pause detection because they consume timeline.

## Unidentified speakers

If a cluster has no name (because `--unknown-speaker keep` was passed, or `ask` was declined), it appears under its pyannote label:

```markdown
⚫️ **SPEAKER_02:** …
```

## TL;DR section

The LLM is asked to omit the `## TL;DR` heading; the renderer adds the heading and a `---` separator after the section body. If the LLM call fails or returns an empty string, the whole section (heading included) is dropped — there's no "TL;DR unavailable" placeholder.

## Example — minimal

```markdown
# Транскрипт: hello.m4a

> 📅 **Початок:** 2026-01-15 10:00 UTC
> ⏱️ **Тривалість:** 0:04
> 🌐 **Мова:** uk

## Привітання

🔵 **Олена:** Привіт!

🟢 **Тарас:** Привіт-привіт.
```

## Example — full (excerpt)

```markdown
# Транскрипт: standup.m4a

> 📅 **Початок:** 2026-01-20 07:01 UTC
> ⏱️ **Тривалість:** 14:32
> 🌐 **Мова:** uk

## TL;DR

Олена і Тарас розбирають, чому впав нічний білд, і домовляються про план на спринт.

- Падіння спричинив таймаут у інтеграційних тестах, не сам код.
- Кеш залежностей вирішили піднімати окремим кроком у CI.
- Gradio-демо переносять на наступний тиждень.

**Action items:** Тарас — полагодити таймаут; Олена — винести кеш у окремий крок.

---

## Розбір нічного білду

🔵 **Олена:** Ну що, дивився, чого воно вночі впало?

🟢 **Тарас:** Ага, подивився. Там, короче, таймаут в інтеграційних, а не сам код.

> _[пауза 4с]_

🟢 **Тарас:** Я думаю, це через те, що кеш не піднявся…

…
```
