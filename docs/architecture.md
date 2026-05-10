# Architecture

voice-transcriber is a thin CLI on top of a linear in-process pipeline. The CLI parses arguments, builds a `PipelineOptions`, and hands it to `pipeline.run()`. Each stage is a small module with a single public function; intermediate results live in memory, and the only on-disk artefact (besides the input audio) is the final Markdown file.

External work — ASR inference, diarization, LLM prompting — is delegated to local tools so the pipeline contains no model code itself: mlx-audio drives the VibeVoice model directly, pyannote.audio runs the diarization graph on the local GPU/CPU, and `llm.py` is a `httpx` client against LM Studio's OpenAI-compatible API.

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
  component "llm (LM Studio client)" as LLM
}

cloud "External" {
  [ffmpeg / ffprobe] as Ffmpeg
  [VibeVoice-ASR via mlx-audio] as MLX
  [pyannote 3.1] as Pyannote
  [LM Studio :1234] as LMS
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
LLM --> LMS
Render --> User : transcript.md
@enduml
```

![Architecture overview](https://www.plantuml.com/plantuml/svg/PLJ1Rjim3BthAmYVjaCITDsfXw93ks057A1ekcB0i8CYqwuGMJ8aEMitRDX7x0lt9QMenAcwWs7oFOfyfXvy4QBqSUoDY3eQX9tJ2cVHInkKrnlqyE_FNr2k7vn56y7jqEyqVXTXg-qWlUoFv3e-KGUFAahR6HH2gkWyV0xeXHYasieEeU89S5bVL_1Jm8lMbJAw0XaILnA37j1eYypUx1FnwqNJTQHN-AzZsmxURVAvvRBaPF2CVrZVbaIZfTS_yg4hYaksHzzbRfMYLq9TY3HHOGXPA-LbXMxefTidfgzJKl8X-b75qUSHQyxAYcYQBFokldgbtADjC9Et79QqCJsygLVKTMoq0retmM-vlAvNQP_AkB61wkDZH6-boVlyshRzW1qiOB_ERxnE1XDzJpQvPzlGAc6d9VJcSIRJ3AvSrLzIsV0ahNKHuNH-mlZD7aZaKTNPoRlJzypNgtNIrCWeDpBG9TTE7JIrpj3dGVLxCLTIFU32zBF03HQzQz3azLE7oyIc0vgiUIn_6BjEswwL2XVHRT5o_QVbNOhJr6HQcCtEsOpqS3nPxn0mIiOgycIe0c0J5KYsJW6nLGeakwE0zdueCBP1WUplNYoNML--gNGvF9kkWnwEfmKbYLd6uL8OcS_FfwKaWVaGUOMrJLcMDQL9qPJGSnWbiZ1EURzdSFZH377UD-A2UFeLFGC0)

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
└── llm.py           # LM Studio client (httpx, chat, chat_json with retry)
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
