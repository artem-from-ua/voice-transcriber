# Output format

The pipeline writes a single Markdown file. Its structure is fixed; what varies is whether the TL;DR and section H2 headings are present.

## Anatomy

```markdown
# Транскрипт: <audio basename>

> 📅 **Початок:** <ISO datetime>
> ⏱️ **Тривалість:** <H:MM:SS or MM:SS>
> 🏁 **Кінець:** <ISO datetime>
> 🎙️ **Транскрипція:** VibeVoice-ASR-<Nbit> + pyannote 3.1
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
🔵 **Артем:** перше речення.

> _[пауза 8с]_

🔵 **Артем:** друге речення.
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

> 📅 **Початок:** 2026-05-10T15:44:02+00:00
> ⏱️ **Тривалість:** 0:04
> 🏁 **Кінець:** 2026-05-10T15:44:06+00:00
> 🎙️ **Транскрипція:** VibeVoice-ASR-6bit + pyannote 3.1
> 🌐 **Мова:** uk

## Привітання

🔵 **Артем:** Привіт!

🟢 **Остап:** Привіт-привіт.
```

## Example — full (excerpt)

```markdown
# Транскрипт: standup.m4a

> 📅 **Початок:** 2026-05-11T09:01:00+02:00
> ⏱️ **Тривалість:** 14:32
> 🏁 **Кінець:** 2026-05-11T09:15:32+02:00
> 🎙️ **Транскрипція:** VibeVoice-ASR-6bit + pyannote 3.1
> 🌐 **Мова:** uk

## TL;DR

Артем і Остап налаштовують мікрофони перед тестом діаризації, потім обговорюють Hugging Face як майданчик для AI-агентів.

- Voice Isolation відсікає все, крім голосу мовця; Wide Spectrum записує ширший спектр.
- HF можна використати як "вітрину" для агентів і плагінів.
- Gradio — фреймворк для таких демо.

**Action items:** "розвідати" Hugging Face.

---

## Налаштування мікрофона

🔵 **Артем:** Ну, я все. Зайшло?

🟢 **Остап:** Ну, я запустив, короче. Так, і куди, бляха, шторку відкривати?

> _[пауза 4с]_

🟢 **Остап:** Automatic, voice isolation, wide spectrum…

…
```
