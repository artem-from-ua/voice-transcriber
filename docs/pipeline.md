# Pipeline — step by step

`pipeline.run(PipelineOptions)` is the single entry point used by both the CLI and tests. It runs ten sequential stages; each stage gets the previous one's output and adds a layer of information.

## Sequence

```plantuml
@startuml
title Pipeline — full run from audio to Markdown
hide footbox
skinparam sequenceArrowThickness 1.5
skinparam LifeLineBorderColor #C0C0C0

actor User
participant cli
participant pipeline
participant transcode
participant audio_meta
participant whisper_asr
participant diarize_speakers
participant merge
participant llm as "mlx-lm\n(in-process)"
participant identify_speakers
participant proofread
participant speech_structure
participant speech_summary
participant render

User -> cli : voice transcribe foo.m4a
cli -> pipeline : run(options)
pipeline -> transcode : foo.m4a → 16 kHz mono WAV
pipeline -> audio_meta : extract_metadata(foo.m4a)
audio_meta --> pipeline : AudioMeta
pipeline -> whisper_asr : transcribe(wav, language)
whisper_asr --> pipeline : AsrSegment[]
pipeline -> diarize_speakers : diarize(wav)
diarize_speakers --> pipeline : DiarTurn[]
pipeline -> merge : merge(asr, turns)
merge --> pipeline : Segment[]
pipeline -> llm : MlxLLM.load() + health_check()
llm --> pipeline : ok
pipeline -> proofread : fix_asr_errors(segments)
proofread --> pipeline : Segment[] (proof-read)
pipeline -> identify_speakers : identify_speakers(segments)
identify_speakers --> pipeline : {label: name}
pipeline -> speech_structure : structure_dialog(segments)
speech_structure --> pipeline : StructuredDialog
pipeline -> speech_summary : generate_tldr(segments)
speech_summary --> pipeline : markdown
pipeline -> render : render_markdown(...)
render --> pipeline : markdown
pipeline --> cli : path/to/foo.md
cli --> User : foo.md

legend right
  ACK responses shown — each stage's output drives the next call.
end legend
@enduml
```

![Pipeline sequence](https://www.plantuml.com/plantuml/svg/XLNDRjD04BxlKupA0LiqAPM0Gm-eeJqWaKX853W0ihRs95vblMjcrqtQ2eaJ3u1umZu9iuwJUfqW5CdnplVjDxFpipvtNf9lAYsyyXhXlQfHAuFmyFCFp1kjWHe3Sx8LoAPG5ho5cQHbOLT6bAf0c5lhh-rQkAKojIHPWSFl3PeS9qHsTLMgV6dGEJWTl-oHfcgEKtRqnbA1T66r9NXoyJpyX92vv-L7XoIOxLMkQcayv5f5wxeBDZ9waiRbjeYjRV1PXLv6vbMfN8sKIHTxAfGaTOSPWtA9v2AmGbh4wbfpUXmSLNezqjLNaoWpgidcVEpqA69onenNyzl3ohp7pWbb4LcPYdcPEKzDxXl2Ws1JLP9k8uZGS6g527c4qUkGFZY36wjoxB94whejuBXw8KM0cRLDAbEvz8cjlRB6fM9dPyuknqpgjiF3hzzm-WgMR--WiiR2vycdQCzZ1NWJhbaYz-soa5ucdKegUhHH7Cqa8BEsWZtPNWMPytYiP2LlZa5BisZa0bFHvms57Nt0HSM5-V8jqXvs0NCxKr1FnHu-4Bva_AeXCv1j-uVXzZ_XW8w1onfol446AeTZ2ort1ZEzdavdOsrbaQJm34gKsfTPNcA-J58HI0CrkunaTXqNAgdM8JqPyjoIIzp6RoZzZlIlq21fEQD0Ybjbh-5vsvwjvskVF_1vh-KrwZCmii8VaQlXd31vzvvnhRHTz1pjiOTdsmB5PRlreAlDr35xWGP9UioyBkY0buuuy55jRzA-zcPim_YrBzcMb8p7urHqwF-5TWDVIr-UU7lIZbYn6NC6srkX6zz228ryX098BKel02ONxzY_gtdqqO4hMRlz9A3CIyugZzHJ1xRnTSEN9QaR9laImV1WGowr7ekWjX4Lv_pWJynV)

## Stages

Approximate wall-clock figures are for a 6-minute Ukrainian conversation on an M-series Mac with the default Whisper ASR backend and the v0.21 default LLM `mlx-community/gemma-3-12b-it-qat-4bit` already loaded. The v0.22 default `mlx-community/Qwen2.5-7B-Instruct-4bit` (see [ADR 0020](adr/0020-default-llm-qwen25-7b.md)) is ~2× smaller and typically faster on LLM stages; treat the numbers below as an upper bound for the new default. Re-measurement on Qwen is tracked in [#82](https://github.com/artem-from-ua/voice-transcriber/issues/82).

| # | Stage | Library | Input | Output | Typical time |
|---|-------|---------|-------|--------|--------------|
| 1 | WAV conversion | `transcode` (`ffmpeg` subprocess) | original audio | 16 kHz mono PCM WAV | < 1 s |
| 2 | Metadata | `audio_meta` (`ffprobe` subprocess) | original audio | `AudioMeta` (start/end/duration) | < 1 s |
| 3 | Diarization | `diarize_speakers` (`pyannote.audio` 3.1) | WAV | `DiarTurn[]` (exclusive) | ~30 s |
| 4 | lang_detect (conditional) | `mlx-whisper` (`model.detect_language` over the longest pyannote turn) | WAV + turns | ISO language code | ~3–5 s; skipped when `--language` is set. See [ADR 0022](adr/0022-asr-language-autodetect.md). |
| 5 | clear_speech | `clear_speech` (`soundfile` + `scipy` + `ffmpeg`) | WAV + turns | cleaned WAV | < 5 s (autogain only; longer chains add per-effect overhead) |
| 6 | ASR | `whisper_asr` (`mlx-whisper`) | cleaned WAV | `AsrSegment[]` | ~1 min |
| 7 | Merge | pure Python | ASR + diar | `Segment[]` | < 1 s |
| 8 | ASR proof-read | LLM via in-process `mlx-lm` | `Segment[]` | `Segment[]` | ~30–60 s |
| 9 | identify_speakers | LLM via in-process `mlx-lm` | `Segment[]` | `{label: name}` | ~5–10 s |
| 10 | speech_structure | LLM via in-process `mlx-lm` | `Segment[]` | `StructuredDialog` | ~10–20 s |
| 11 | speech_summary | LLM via in-process `mlx-lm` | `Segment[]` | Markdown string | ~10–20 s |

## Errors and recovery

| Failure | Stage | Behaviour |
|---------|-------|-----------|
| LLM model directory missing or invalid | health check | `MlxLLM.health_check()` raises `LLMError` with the actionable message *"… is not in the HuggingFace cache. Fetch it once with `huggingface-cli download <repo>`"* (or the equivalent for a filesystem path); pipeline exits before any LLM stage runs. See [`troubleshooting.md`](troubleshooting.md) and [ADR 0020](adr/0020-default-llm-qwen25-7b.md) |
| Hugging Face token missing | diarize_speakers | `DiarizationError` with path/`chmod` instructions |
| Gated repo not accepted on HF | diarize_speakers | `GatedRepoError` from pyannote; see [`troubleshooting.md`](troubleshooting.md) |
| ASR repetition loop | asr | `mlx-whisper` runs an internal temperature schedule that escapes most loops; `condition_on_previous_text=False` removes the dominant trigger. See troubleshooting if it persists. |
| LLM returns non-JSON for identify_speakers/speech_structure | identify_speakers / speech_structure | `lm-format-enforcer` guarantees valid JSON on the first try; if generation itself fails (e.g. model crash) → cluster stays unidentified / fallback to a single "Розмова" section |
| LLM rewrites text too aggressively | proofread | Levenshtein + length ratio check rejects the reply; original kept |
| LLM error in speech_summary | speech_summary | empty string returned; render simply omits the `## TL;DR` section |

The pipeline never aborts in the middle: if a non-critical LLM stage fails, that artefact is dropped and the rest still produces output.

## See also

- [`prompts.md`](prompts.md) — the LLM call contracts (temperature, max_tokens, response_format, safety nets) used by stages 6–9.
- [`adr/README.md`](adr/README.md) — index of the architectural decisions behind the stage layout.
