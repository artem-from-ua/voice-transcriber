# Pipeline — step by step

`pipeline.run(PipelineOptions)` is the single entry point used by both the CLI and tests. It runs nine sequential stages; each stage gets the previous one's output and adds a layer of information.

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
participant llm as "LM Studio"
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
pipeline -> llm : GET /v1/models (health)
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

![Pipeline sequence](https://www.plantuml.com/plantuml/svg/XLJ1ZjCm4BtxAqRBWLJgjbG23ZsWBRi89BeIqYvmG2XoaqbYrR63FUakYv0uyG68BzmlOPmctJe2eKgjFVFyPlh6psUUfAEsre8KQOHtga6j3CBzZzzGj5g3QmqKpjOWsrnP80kNqcrpUsD4fNA4mbhQs5lXjyeqqiaQF7vfqMIuSixUN5Sgsnhq7XQpPqUWjIfmpOLUMfUZExVQEdXq_YHyX90PyVQzHoSOJIfJZJG4cLRHlkcRZO95KJTOWlHmib-UZDADivluYFHndLn9f-vYJ8skZ2DQrrsLzILSKT0cBiJQ65B5rxXZwudhPwn75FVasenQ5rSWdSUDEJGibX116JXz4GI19UoioX38IUCpfpRTI6RrKob2cb63J0pbOIQs8MMDduX3d36zQilXBDp__0MBvx1zSmUrDHO-hZwC3dGwyWcyvT8PfJMIp2N9fAUOY05p6ZUn2c9TCZeYv25mxk5V93ToDuMD8Zy5BKtPoX8d8g36TDvTOLcpsfy-Hupz61dIhmBbH0pX4Sq5XwzRPqOatTmvtVqcN7uAFASWtpupOlbx9-6gBE7rgskOxnRpskQeFIGLIatLH8JiYCPke_D7roQCIDsct4YAx2_d4xyl6GPw1FjNNv0mmXQd3cKUNu7XmZ9wMAQ-GRd5gCW1DghmJSiDwYKOMUFtYFXmkHbrMASy1ctB8-87sBZt8P5VT6SYyk0HnfHeq4d2DEoFIBlqYAyUtgzZdhsrWaMwHJg0ajbiDX5zzlz41rCsagevsNbdXdnlHKvspkrTbWkXaHlFmQco8W6mEd_BzNt3zaGFlcBkxY56cLKi7JlWiGVRKjFo8-NKZa5K8HYs86HIwva8R7jISSPV_B3_0G00)

## Stages

Approximate wall-clock figures are for a 6-minute Ukrainian conversation on an M-series Mac with the default 6-bit ASR and `mlx-community/gemma-3-12b-it-qat-4bit` already loaded.

| # | Stage | Library | Input | Output | Typical time |
|---|-------|---------|-------|--------|--------------|
| 1 | WAV conversion | `ffmpeg` (subprocess) | original audio | 16 kHz mono PCM WAV | < 1 s |
| 2 | Metadata | `ffprobe` (subprocess) | original audio | `AudioMeta` (start/end/duration) | < 1 s |
| 3 | ASR | `mlx-audio` (`mlx_audio.stt.generate_transcription`) | WAV | `AsrSegment[]` | ~2–4 min |
| 4 | Diarization | `pyannote.audio` 3.1 | WAV | `DiarTurn[]` (exclusive) | ~30 s |
| 5 | Merge | pure Python | ASR + diar | `Segment[]` | < 1 s |
| 6 | ASR proofread | LLM via LM Studio | `Segment[]` | `Segment[]` | ~30–60 s |
| 7 | Identify | LLM via LM Studio | `Segment[]` | `{label: name}` | ~5–10 s |
| 8 | Structure | LLM via LM Studio | `Segment[]` | `StructuredDialog` | ~10–20 s |
| 9 | TL;DR | LLM via LM Studio | `Segment[]` | Markdown string | ~10–20 s |
| 10 | Render | pure Python | everything | Markdown file | < 1 s |

## Errors and recovery

| Failure | Stage | Behaviour |
|---------|-------|-----------|
| LM Studio not running | health check | `RuntimeError` from CLI with the message *"LM Studio server is not reachable…"*; pipeline exits before any stage runs |
| Hugging Face token missing | diarize | `DiarizationError` with path/`chmod` instructions |
| Gated repo not accepted on HF | diarize | `GatedRepoError` from pyannote; see [`troubleshooting.md`](troubleshooting.md) |
| ASR repetition loop | asr | model usually escapes within seconds thanks to `repetition_penalty=1.2`; if it persists, retry with `--asr-bits 8` |
| LLM returns non-JSON for identify/structure | identify / structure | one automatic retry inside `llm.chat_json()`; on failure → cluster stays unidentified / fallback to a single "Розмова" section |
| LLM rewrites text too aggressively | postprocess | Levenshtein + length ratio check rejects the reply; original kept |
| LLM error in TL;DR | tldr | empty string returned; render simply omits the `## TL;DR` section |

The pipeline never aborts in the middle: if a non-critical LLM stage fails, that artefact is dropped and the rest still produces output.
