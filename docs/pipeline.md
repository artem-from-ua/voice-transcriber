# Pipeline — step by step

`pipeline.run(PipelineOptions)` is the single entry point used by both the CLI and tests. It runs eleven sequential stages; each stage gets the previous one's output and adds a layer of information.

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
participant safe_speech
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
pipeline -> safe_speech : redact_dialog(dialog)
safe_speech --> pipeline : StructuredDialog (redacted)
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

![Pipeline sequence](https://www.plantuml.com/plantuml/svg/XLNDRjD04BxlKupA0LiqAPM0Gm-eeJqWaKX853W0ihRUYRtAUjVihfkq28aJ3u1umZu9ixQJUfsWAb9iptpppUpyxRdptFgcKinBhn3UongLr0Ztl_x2ib4ARADXQKq5l17IW3Umu7Obp5gpKWg4fJ7-scoOMqbTSyihSFYzGPtZp5gplYfbljBe79nENmv0Sxd4EJbwOwn0Us6KiV3audduCSPpJvyV7Lf6Q2zpMNFj8LSo-gxxO2EXjroxt8XOsWQVLUXv95wNqjLeC-vYJq9oA-ym8oLVeNMHia9RnEnAKNeS75LgCr7LLvr8FQcjoUdOwL64f8nfBvUtXvd9nYmjSX592OfvcJblczmtDdRj-B8Dan07ZPggulOsKbdKb7B6GdvXyZga5SxWniWS--nPUTqMTbgzu2oe2RLDDa6f9H9JUscqIzbEJfXTxWdKcyFzxpzm-WfMR--WCjh0vzcdoEQXCcI46wB8VVifkET9pvAo0MmIHpCBcaLRsG7jeBA4UJXMikOtnw2uBXfUOCg6k36nin-mgAXWNxv5tEFk86ml2kmfszEFY2z9VzLOFQ9j-uhKxJEXW8w1oXrot6b6B8TZ2wru1WkrcSyNKsMuI59u1YLovSiiBp5V9IaBe16RMKKqkquCbPIRa9uCQPwjIrpdDvH-1_fVQ92qc4a0nQsoDmXajYSR-Dh7Ztp-KFmQrHbeNk7FoDLuVWYyUy-eLieK0qTxwF7PjWfnsPh6hXwcCSm9Yj3ClOlkGGu6c4Uu8UaeS9IyUBZ9i42DbdlCl1Bsm65wuCXTjLtaG-vkExJHXvTi2qgcqsdAUktZHBkzKdDVddXpqawowBO9ATlbqsy9mPX2Ee80AulICu3PnJloxshQCEZ0bSJTtaZ8AMbqPHNur85fVDtGhhRoXa2-HD2qFo3dIarPOEj8sJdzqGtt3m00)

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
| 11 | safe_speech | LLM via in-process `mlx-lm` | `StructuredDialog` | `StructuredDialog` (redacted) | ~5–15 s; skipped when `--no-safe-speech` or `--safe-speech-topics ""`. See [ADR 0023](adr/0023-safe-speech-stage.md). |
| 12 | speech_summary | LLM via in-process `mlx-lm` | `Segment[]` (redacted) | Markdown string | ~10–20 s |

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
| LLM error in safe_speech (one section) | safe_speech | that section is left unredacted; a warning is logged; other sections are processed normally |

The pipeline never aborts in the middle: if a non-critical LLM stage fails, that artefact is dropped and the rest still produces output.

## See also

- [`prompts.md`](prompts.md) — the LLM call contracts (temperature, max_tokens, response_format, safety nets) used by stages 8–12.
- [`adr/README.md`](adr/README.md) — index of the architectural decisions behind the stage layout.
