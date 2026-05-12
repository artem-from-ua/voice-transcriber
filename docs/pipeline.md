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
participant audiometa
participant whisper_asr
participant diarize
participant merge
participant llm as "mlx-lm\n(in-process)"
participant identify
participant proofread
participant structure
participant tldr
participant render

User -> cli : voice transcribe foo.m4a
cli -> pipeline : run(options)
pipeline -> transcode : foo.m4a → 16 kHz mono WAV
pipeline -> audiometa : extract_metadata(foo.m4a)
audiometa --> pipeline : AudioMeta
pipeline -> whisper_asr : transcribe(wav, language)
whisper_asr --> pipeline : AsrSegment[]
pipeline -> diarize : diarize(wav)
diarize --> pipeline : DiarTurn[]
pipeline -> merge : merge(asr, turns)
merge --> pipeline : Segment[]
pipeline -> llm : MlxLLM.load() + health_check()
llm --> pipeline : ok
pipeline -> proofread : fix_asr_errors(segments)
proofread --> pipeline : Segment[] (proof-read)
pipeline -> identify : identify_speakers(segments)
identify --> pipeline : {label: name}
pipeline -> structure : structure_dialog(segments)
structure --> pipeline : StructuredDialog
pipeline -> tldr : generate_tldr(segments)
tldr --> pipeline : markdown
pipeline -> render : render_markdown(...)
render --> pipeline : markdown
pipeline --> cli : path/to/foo.md
cli --> User : foo.md

legend right
  ACK responses shown — each stage's output drives the next call.
end legend
@enduml
```

![Pipeline sequence](https://www.plantuml.com/plantuml/svg/XLJ1RjGm4BtxAqRbG2AwAPM0Gm-eItj0ObT2ei01KEJ6iucrZXrifxkqGkB41o2-i5_2E9kaSGH2AsNjcUStupSpFdUUMz_MYddf5S9RsQ2I6k7-ns_Oj4g1RJLihAc1jq8Qy0RMt6w5sMbMIO6mCSPVcprpMwaRRdaD3h-sg0jSMcjsLvKijXgTWzFi-GIqaXjSKQ1NnWgq5qOP2uykdeOVOxpmj7tlq392Urd8XciFXPBHlkcJZOpUSkqA8s9hbtoDdaVMNILTWpRdBWua9BVoDcQeqPQnHIaInS5HhVOBLN_MYTIBnfg2BfiUHKZIINkv-HPdRutPMEGYiZflsyAtTdOf9U8CBMgIZR6W4InU1cdW36wCBB1NmChhhZXP_Oopu2RK81X1gQo9QRmqsgLij1Dcr8z0_N6u__aBJb_0zlKjr4OR-BZy49qPrQKpk2U6mkTXAxZdIK-IiWVK8ivb6HphhZWJqabn2FDmgMJ7Ruv1SLsslCIKJN5pOcVVOLcJ-9--HDnzWGdIhm9foWRpZEQIp5UjrJEIhYF8tVqd5FuOg711puDdnlBtJ48JdS5QxLUhTQOC5qaAJw12hdoL5nKMsoHb0JHZCzk8PcocK3Mv3sBaIFDdNU8ESKEPHz2_KeEamom2A6wBeOK9FInpafrlCGenmcOHxXI_HdK6cjVuFI8U6vvGupgdIYXJJeWVOFFS1uUuxCx4pKnpGvWIDLhkCG_x2Mdddl7Lm-isvJcCMnYRRf4Fe2JBifJrtlyJZOFQS5-TU7FIpOOuZ2Svksdknqumff0I5s1bMNa6iBnuG_5TGoEB3bn5tDqpZRoeI3eQXiSEJEkRbbumAsy8v2i4JHC91LSgOu7jGChEwKFF_Xy0)

## Stages

Approximate wall-clock figures are for a 6-minute Ukrainian conversation on an M-series Mac with the default Whisper ASR backend and the v0.21 default LLM `mlx-community/gemma-3-12b-it-qat-4bit` already loaded. The v0.22 default `mlx-community/Qwen2.5-7B-Instruct-4bit` (see [ADR 0020](adr/0020-default-llm-qwen25-7b.md)) is ~2× smaller and typically faster on LLM stages; treat the numbers below as an upper bound for the new default. Re-measurement on Qwen is tracked in [#82](https://github.com/artem-from-ua/voice-transcriber/issues/82).

| # | Stage | Library | Input | Output | Typical time |
|---|-------|---------|-------|--------|--------------|
| 1 | WAV conversion | `transcode` (`ffmpeg` subprocess) | original audio | 16 kHz mono PCM WAV | < 1 s |
| 2 | Metadata | `audiometa` (`ffprobe` subprocess) | original audio | `AudioMeta` (start/end/duration) | < 1 s |
| 3 | Diarization | `pyannote.audio` 3.1 | WAV | `DiarTurn[]` (exclusive) | ~30 s |
| 4 | Clearspeech | `clearspeech` (`soundfile` + `scipy` + `ffmpeg`) | WAV + turns | cleaned WAV | < 5 s (autogain only; longer chains add per-effect overhead) |
| 5 | ASR | `whisper_asr` (`mlx-whisper`) | cleaned WAV | `AsrSegment[]` | ~1 min |
| 6 | Merge | pure Python | ASR + diar | `Segment[]` | < 1 s |
| 7 | ASR proof-read | LLM via in-process `mlx-lm` | `Segment[]` | `Segment[]` | ~30–60 s |
| 8 | Identify | LLM via in-process `mlx-lm` | `Segment[]` | `{label: name}` | ~5–10 s |
| 9 | Structure | LLM via in-process `mlx-lm` | `Segment[]` | `StructuredDialog` | ~10–20 s |
| 10 | TL;DR | LLM via in-process `mlx-lm` | `Segment[]` | Markdown string | ~10–20 s |

## Errors and recovery

| Failure | Stage | Behaviour |
|---------|-------|-----------|
| LLM model directory missing or invalid | health check | `MlxLLM.health_check()` raises `LLMError` with the actionable message *"… is not in the HuggingFace cache. Fetch it once with `huggingface-cli download <repo>`"* (or the equivalent for a filesystem path); pipeline exits before any LLM stage runs. See [`troubleshooting.md`](troubleshooting.md) and [ADR 0020](adr/0020-default-llm-qwen25-7b.md) |
| Hugging Face token missing | diarize | `DiarizationError` with path/`chmod` instructions |
| Gated repo not accepted on HF | diarize | `GatedRepoError` from pyannote; see [`troubleshooting.md`](troubleshooting.md) |
| ASR repetition loop | asr | `mlx-whisper` runs an internal temperature schedule that escapes most loops; `condition_on_previous_text=False` removes the dominant trigger. See troubleshooting if it persists. |
| LLM returns non-JSON for identify/structure | identify / structure | `lm-format-enforcer` guarantees valid JSON on the first try; if generation itself fails (e.g. model crash) → cluster stays unidentified / fallback to a single "Розмова" section |
| LLM rewrites text too aggressively | proofread | Levenshtein + length ratio check rejects the reply; original kept |
| LLM error in TL;DR | tldr | empty string returned; render simply omits the `## TL;DR` section |

The pipeline never aborts in the middle: if a non-critical LLM stage fails, that artefact is dropped and the rest still produces output.

## See also

- [`prompts.md`](prompts.md) — the LLM call contracts (temperature, max_tokens, response_format, safety nets) used by stages 6–9.
- [`adr/README.md`](adr/README.md) — index of the architectural decisions behind the stage layout.
