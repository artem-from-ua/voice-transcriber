# Judge input — 9 segment(s)

For each segment below, return a mark in {better, worse, neutral}:

- **better** — proofread fixed a real ASR error: ASR produced a non-word
  (`корище`, `контролуси`, `твими`), a clear mishearing of a technical
  term (`HugginsFace` → `Hugging Face`), or a grammatical impossibility,
  and proofread restored what the speaker most plausibly said.
- **worse** — proofread changed text that did not need fixing, or moved
  it further from what the speaker said. This includes:
    - swapping a word for a synonym (e.g. `чуваки` → `хлопці`) when the
      original was correctly transcribed;
    - rewriting a colloquial form into a literary one or vice versa when
      the ASR-rendered form already matched the speech;
    - introducing a regional / non-standard variant when the ASR form
      was the conventional one for the context (e.g. tech slang).
- **neutral** — difference is immaterial: punctuation, capitalisation,
  near-equivalent demonstratives, or a guess on a garbled segment that
  is plausible but cannot be verified from text alone.

Guiding principle: the goal of this tool is to capture **real, lively
Ukrainian conversation as it was actually spoken** (CLAUDE.md "Project
goal"). We do **not** want proofread to enforce literary norms,
substitute synonyms, or 'clean up' how people talk. Proofread's only
job is to fix ASR mishearings — anything beyond that is a regression.

You do NOT have access to the audio. Judge from text and context only.
When the direction of a change (literary↔colloquial, dialect↔standard)
cannot be verified without listening, prefer `neutral` over guessing.
The human evaluator has audio access — your role is text-only judgment
for calibration: an agreement metric will tell us whether text-only LLM
judgment can be trusted for future automated measurements.

Reply format (single JSON block, between markers — save as judge-marks.json):

    ---JUDGE-MARKS-START---
    {
      "<segment-index>": {"mark": "better", "rationale": "..."},
      ...
    }
    ---JUDGE-MARKS-END---

## Segment 4 — timestamp 00:19.00-00:21.00 (speaker: SPEAKER_01)

**ASR (merge):** Ну там де контролуси у тебе.
**Proofread:**  Ну там де контролери у тебе.

## Segment 9 — timestamp 00:41.48-00:43.48 (speaker: SPEAKER_00)

**ASR (merge):** Ну окей, добре, що мені взяти зараз?
**Proofread:**  Ну окей, добре, шо мені взяти зараз?

## Segment 11 — timestamp 00:45.48-00:49.48 (speaker: SPEAKER_01)

**ASR (merge):** то він перестане писати голоси, от транспорт їде.
**Proofread:**  то він перестане писати голоси, а транспорт їде.

## Segment 51 — timestamp 03:11.48-03:15.48 (speaker: SPEAKER_01)

**ASR (merge):** що пов'язане зі тими твими скелами чи ще щось.
**Proofread:**  що пов'язане зі тими твоїми скелами чи ще щось.

## Segment 69 — timestamp 05:07.08-05:09.08 (speaker: SPEAKER_00)

**ASR (merge):** Так, треба корище цей.
**Proofread:**  Так, треба користуватися цим.

## Segment 70 — timestamp 05:09.08-05:14.08 (speaker: SPEAKER_01)

**ASR (merge):** Якщо вони там тусуться, ну я не знаю, просто хто тусується на хайміні...ну там, ну, навірно, чуваки, от, подивляться.
**Proofread:**  Якщо вони там тусуются, ну я не знаю, просто хто тусується на хайміні...ну там, ну, навірно, хлопці, от, подивляться.

## Segment 75 — timestamp 05:26.08-05:30.08 (speaker: SPEAKER_01)

**ASR (merge):** Мені цікаво, чи є там оця прослойка, яка потенційно може найняти для чогось.
**Proofread:**  Мені цікаво, чи є там ця прослойка, яка потенційно може найняти для чогось.

## Segment 77 — timestamp 05:34.08-05:38.08 (speaker: SPEAKER_01)

**ASR (merge):** Ти зробиш, наприклад, агента для продуктового інтерв'ю.
**Proofread:**  Ти зробиш, наприклад, агента для продуктівого інтерв'ю.

## Segment 81 — timestamp 05:44.08-05:46.08 (speaker: SPEAKER_01)

**ASR (merge):** Ось, поспілкуйтесь з продуктовим агентом.
**Proofread:**  Ось, поспілкуйтесь з продуктівим агентом.

