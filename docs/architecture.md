# Architecture

voice-transcriber is a thin CLI on top of a linear in-process pipeline. The CLI parses arguments, builds a `PipelineOptions`, and hands it to `pipeline.run()`. Each stage is a small module with a single public function; intermediate results live in memory, and the only on-disk artefact (besides the input audio) is the final Markdown file.

External work — ASR inference, diarization, LLM prompting — is delegated to local libraries so the pipeline contains no model code itself: mlx-audio drives the VibeVoice model directly, pyannote.audio runs the diarization graph on the local GPU/CPU, and `llm.py` loads the LLM in-process via `mlx-lm` (no HTTP).

## Component diagram

```plantuml
@startuml
title voice-transcriber — component overview
skinparam componentStyle rectangle

actor User

package "voice CLI" {
  component "cli" as CLI
  component "pipeline" as Pipeline
}

package "Stages" {
  component "ffprobe" as FF
  component "asr" as ASR
  component "diarize" as Diar
  component "merge" as Merge
  component "postprocess" as Post
  component "identify" as Ident
  component "structure" as Struct
  component "tldr" as TLDR
  component "render" as Render
  component "llm (mlx-lm wrapper)" as LLM
}

cloud "External" {
  [ffmpeg / ffprobe] as Ffmpeg
  [VibeVoice-ASR via mlx-audio] as MLX
  [pyannote 3.1] as Pyannote
  [mlx-lm (in-process)] as MlxLm
}

database "Local" {
  folder "~/.cache/lm-studio/models" as LMSCache
  file "~/.cache/huggingface/token" as HFToken
}

User --> CLI
CLI --> Pipeline
Pipeline --> FF
Pipeline --> ASR
Pipeline --> Diar
Pipeline --> Merge
Pipeline --> Post
Pipeline --> Ident
Pipeline --> Struct
Pipeline --> TLDR
Pipeline --> Render

FF --> Ffmpeg
ASR --> MLX
MLX --> LMSCache
Diar --> Pyannote
Diar ..> HFToken
Post --> LLM
Ident --> LLM
Struct --> LLM
TLDR --> LLM
LLM --> MlxLm
MlxLm --> LMSCache
Render --> User : transcript.md
@enduml
```

![Architecture overview](https://www.plantuml.com/plantuml/svg/PLJBRjim4BphAmYTaeDi5Btr4AH8OZI030Hn6XGeUbXJAuKmNo0fnsxHeX_HB-oNb5pAIfWUPEtEBBdZtO4kVG0NHYMh8894jZU2OnCSQC-TsA9ZVt__OTmeQpJgmCmUtLxWS-LtGbjme5x8JJZ66npo07gGM5N0Wt7iiqTNLHRu3WPaDNLWL-rjpNvKxDNLDPUYPk0JLn9MM9H28x5tKrBzV7Nf9iIN_-_6lhVERFEvrQham3l2FsxkIw8JuCJtVEWwnYMhq0sPMwVeZL3ZG-p8qVkiDUPbXUZYI_H7eczJKl8-k967qUKM6yhAYY2xBFoXlNwZtA7kC9Ft59Qqb8gTANbeullPWRNepgcuRTTfcboQiMFrpI6Wqo3pDB_slR8ui2MRXlcDXabWeX-ZIHx9D76GR2-0fGumTi9GvRhzaihi4RGs0TdxnJl2xoOWaPEcCw6RQNhd-Qmyj2efwo305dnST6luILblPFoBhFwrN73WJxYKgl4XDLugqw7CAsZNcwl4fWCYslEb_6aS1g677ZWkWzcXflfFguSKfwx9kAnfBiYGyQ5ujjyf83IQgyYJgGg0Z5GWsJu5H7OfaEoG0feyKM1aXGAPzb-jLbbVtbTQ4VMEIuaFnVE0aiGiuxGQGXQBYtaeIM0-51r3skOiovhJf6XEufFRqZxfjsoTrH96G__0JbyW6nQggouZ7xzH_m00)

## Module layout

```
src/voice/
├── cli.py           # argparse, --help, dispatch to pipeline.run()
├── pipeline.py      # PipelineOptions + run(): ffmpeg → ffprobe → ASR → diarize → merge → post → identify → structure → tldr → render
├── types.py         # shared dataclasses (AsrSegment, DiarTurn, Segment, Section, StructuredDialog, AudioMeta)
├── ffprobe.py       # extract_metadata(): start/end/duration from ffprobe → birthtime → mtime
├── asr.py           # transcribe(): mlx-audio wrapper, bitness 4/5/6/8, JSON timeline parse
├── diarize.py       # diarize(): pyannote 3.1, MPS+CPU fallback, exclusive turns
├── merge.py         # merge(): per-segment max-overlap mapping ASR↔pyannote
├── postprocess.py   # fix_asr_errors(): per-segment LLM proofreader with safety net
├── identify.py      # identify_speakers(): LLM self-intro detection with override / ask / keep
├── structure.py     # structure_dialog(): LLM-driven section layout with validation
├── tldr.py          # generate_tldr(): LLM markdown summary
├── render.py        # render_markdown(): final output
├── speaker_emojis.py # fixed palette assignment
└── llm.py           # MlxLLM: in-process mlx-lm wrapper with chat / chat_json (lm-format-enforcer)
```

## Data flow

The pipeline passes increasingly enriched `Segment` lists from stage to stage:

1. `asr.transcribe()` produces `AsrSegment[]` (text + ASR-side speaker hint)
2. `diarize.diarize()` produces `DiarTurn[]` (pyannote timeline)
3. `merge.merge()` joins them into `Segment[]` (text + pyannote speaker label)
4. `postprocess.fix_asr_errors()` updates `content` per segment
5. `identify.identify_speakers()` returns a `{label → name}` map; pipeline assigns `.name`
6. `structure.structure_dialog()` wraps `Segment[]` into `StructuredDialog` with sections
7. `tldr.generate_tldr()` produces a Markdown string
8. `render.render_markdown()` combines `AudioMeta` + `StructuredDialog` + `tldr` into one file

See [`pipeline.md`](pipeline.md) for the step-by-step sequence with timing details, and [`adr/`](adr/) for the decisions behind these boundaries.
