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
participant ffmpeg as "ffmpeg"
participant ffprobe
participant asr
participant diarize
participant merge
participant llm as "mlx-lm\n(in-process)"
participant identify
participant postprocess
participant structure
participant tldr
participant render

User -> cli : voice transcribe foo.m4a
cli -> pipeline : run(options)
pipeline -> ffmpeg : foo.m4a → 16 kHz mono WAV
pipeline -> ffprobe : extract_metadata(foo.m4a)
ffprobe --> pipeline : AudioMeta
pipeline -> asr : transcribe(wav, bits, language)
asr --> pipeline : AsrSegment[]
pipeline -> diarize : diarize(wav)
diarize --> pipeline : DiarTurn[]
pipeline -> merge : merge(asr, turns)
merge --> pipeline : Segment[]
pipeline -> llm : MlxLLM.load() + health_check()
llm --> pipeline : ok
pipeline -> postprocess : fix_asr_errors(segments)
postprocess --> pipeline : Segment[] (proof-read)
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

![Pipeline sequence](https://www.plantuml.com/plantuml/svg/XLJDRjGm4BxxAKRbG2AwAPM0mnvGb_Q0nAw4H843eCYRJ18h_WcsqouW94uy0E89-oICiybs7O5GfCIU-VpDP6O-NpZFhU-LP5vuYV1QT2Y5HhZxyHkgNagmlORA6WMyBuK1Rs33RLkQMyqQKI9KnlYjsJ7N2jrnonKu_DoZBd1bhRczRaJHQdGEphEdHw2rg71DWLuOMwAzDD9OU73vE3oCySBJzgr3omZjHI4whZqKKaJxRamsCbQLwh06xk1alpoPkJjhjl4Hxk8ufU1MV8qn2cqTMwHKGnGbTmkfFkf4w0Ln5_IlQHoJogIzgBx4oHldHtXaTzxsXUzj7CpBCixHegQwCHQA18ldeJQmX1iZ2WHlkNQ55TkXEvbwmbbm4sgg646fhudfl33QfUnW9ynOmELq5kv-_eBpPz2-_0hAQ0FlL-zc1uQIqWdSKUZ2vmezBxddoKYHiWcpY9DOXPdQ43eYf7wGx_ulabj-SmfRuTqfIAxhdjUOie2Qqpdx1cj5rVxmAM8SEqgGSHKeKpQPPpHNPBxkhPwH32D0xk6RKFXJe3w5ykqzCvQ_PnAcPWaRkLklDvaql4nIU0GDSkcRl6YmQ9EK1T2CpRGHpT7qX4w9NKxvv4YAiovn-yYXhqUmVwK72I5CjR38otWIfhabzBJCNOUynIZ80JQBy4toBSebQAxmUqHyc7527TOvTKEQ-eZu7ZRFVNAKLyEPY3n8XJ0rQhJSOnxsHwI3UyQdfXljc6UliA2KOP5Fe2JBifIDtlyJ7RJPST-SUNCsQABSAvASWu17iPMCIQJ4Ix2YRZm3M5s-elYk8vMY0zSGzt0r8oyQAXq9uQ43q_kkfslBYXi2-GP1anAXu59cBB3jITa5lUYg_mC0)

## Stages

Approximate wall-clock figures are for a 6-minute Ukrainian conversation on an M-series Mac with the default 6-bit ASR and `mlx-community/gemma-3-12b-it-qat-4bit` already loaded.

| # | Stage | Library | Input | Output | Typical time |
|---|-------|---------|-------|--------|--------------|
| 1 | WAV conversion | `ffmpeg` (subprocess) | original audio | 16 kHz mono PCM WAV | < 1 s |
| 2 | Metadata | `ffprobe` (subprocess) | original audio | `AudioMeta` (start/end/duration) | < 1 s |
| 3 | ASR | `mlx-audio` (`mlx_audio.stt.generate_transcription`) | WAV | `AsrSegment[]` | ~2–4 min |
| 4 | Diarization | `pyannote.audio` 3.1 | WAV | `DiarTurn[]` (exclusive) | ~30 s |
| 5 | Merge | pure Python | ASR + diar | `Segment[]` | < 1 s |
| 6 | ASR proof-read | LLM via in-process `mlx-lm` | `Segment[]` | `Segment[]` | ~30–60 s |
| 7 | Identify | LLM via in-process `mlx-lm` | `Segment[]` | `{label: name}` | ~5–10 s |
| 8 | Structure | LLM via in-process `mlx-lm` | `Segment[]` | `StructuredDialog` | ~10–20 s |
| 9 | TL;DR | LLM via in-process `mlx-lm` | `Segment[]` | Markdown string | ~10–20 s |
| 10 | Render | pure Python | everything | Markdown file | < 1 s |

## Errors and recovery

| Failure | Stage | Behaviour |
|---------|-------|-----------|
| LLM model directory missing or invalid | health check | `MlxLLM.health_check()` raises with the actionable message *"LLM model not found at …"* pointing at LM Studio's Models tab; pipeline exits before any LLM stage runs. See [`troubleshooting.md`](troubleshooting.md) |
| Hugging Face token missing | diarize | `DiarizationError` with path/`chmod` instructions |
| Gated repo not accepted on HF | diarize | `GatedRepoError` from pyannote; see [`troubleshooting.md`](troubleshooting.md) |
| ASR repetition loop | asr | model usually escapes within seconds thanks to `repetition_penalty=1.3`; if it persists, retry with `--asr-bits 8` |
| LLM returns non-JSON for identify/structure | identify / structure | `lm-format-enforcer` guarantees valid JSON on the first try; if generation itself fails (e.g. model crash) → cluster stays unidentified / fallback to a single "Розмова" section |
| LLM rewrites text too aggressively | postprocess | Levenshtein + length ratio check rejects the reply; original kept |
| LLM error in TL;DR | tldr | empty string returned; render simply omits the `## TL;DR` section |

The pipeline never aborts in the middle: if a non-critical LLM stage fails, that artefact is dropped and the rest still produces output.
