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
participant proofread
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

![Pipeline sequence](https://www.plantuml.com/plantuml/svg/XLJDRjGm4BxxAKRbG2AwAPM0mnvGb_Q0nAw4H843eCYRJ19h_HDifrr0I9nu0CGJzaaOPvDq7O5GfCIU-VpDP6O-NpZFhU-KP5vuYV1Qj2Y5HhZxyHkgJagmdORA6WMyAuK1Rs33xQuqjvereaIeZF5RisTk9tJBBLVWy7E7ki2LjURskX75JgDpS9uzFGAjHOLh2lJ2s1BjfP76meFBn-5XZ1UUjcyTMaPeBmhHSksXa2BQjqEoaR6gL8irS0SdX-N9pDrQiusFS1V7AGMtuckCKMZhs2AbwgCekLz8zL4dGY-8kw1_JUEOL2RjHVKbJjuQKrdaPMHrtdQ5xsmSoiiopj2Yfgen5ae4Y-UXCh246oCA16-vTeKLsxutcNh2MN0JQgmNGQchYMczCDgbRB8JPYZVSZmBTpz_mVapsBty2ife0-zNxsO7-eBI2TnJwCBd2ZqlkUV9G96o4RE8arY5YTeGEYAaRf3l_Y-IMtvp2blXtIb8hkkErvYoW9hJEVi6QqMr_l0fOXpwIP1X5IXJDffdD5Talkwidf7q0q3k_fjG-5EWFeNo7Jmpbhzd4cPc2HkvNwytcJIyJ59u10rowPkyQB3O9IaBe1cRsKKqq-o4FebzJjdaI6gpBd67kA6h4-XVgK7IOnO154_1EB647fUvQv7lC0enmMOHlac-HRa4pHL-ZuYd-IRKjCwf4zBKHyJti7dkewEywiz4v44cXAbHe-KUyx0_8ktTCpurtcN7F0Tr1PNqYtm49LcMfMpm_fzeqcNBVNFcpLclX_AWHdBsuXs4LZ8caH8lmOgwyGnWTVcAuhkM58eEN4FS_QMCl6YeT2I2XmvCvzkEBYmhRWZa6mHDAeI2Ivcnm7OWPHVqeal-3m00)

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
| LLM rewrites text too aggressively | proofread | Levenshtein + length ratio check rejects the reply; original kept |
| LLM error in TL;DR | tldr | empty string returned; render simply omits the `## TL;DR` section |

The pipeline never aborts in the middle: if a non-critical LLM stage fails, that artefact is dropped and the rest still produces output.

## See also

- [`prompts.md`](prompts.md) — the LLM call contracts (temperature, max_tokens, response_format, safety nets) used by stages 6–9.
- [`adr/README.md`](adr/README.md) — index of the architectural decisions behind the stage layout.
