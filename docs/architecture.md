# Architecture

voice-transcriber is a thin CLI on top of a linear in-process pipeline. The CLI parses arguments, builds a `PipelineOptions`, and hands it to `pipeline.run()`. Each stage is a small module with a single public function; intermediate results live in memory, and the only on-disk artefact (besides the input audio) is the final Markdown file.

External work — ASR inference, diarization, LLM prompting — is delegated to local libraries so the pipeline contains no model code itself: mlx-audio drives the VibeVoice model directly, pyannote.audio runs the diarization graph on the local GPU/CPU, and `llm.py` loads the LLM in-process via `mlx-lm` (no HTTP).

## High-level architecture

The CLI hands a `PipelineOptions` (whose load-bearing field is the audio path) to `pipeline.run()`. The orchestrator delegates the heavy work to four external libraries — `ffmpeg`/`ffprobe`, `mlx-audio` (VibeVoice-ASR), `pyannote.audio`, `mlx-lm` — and writes a single Markdown file back to the user.

```plantuml
@startuml
title voice-transcriber — high-level architecture
skinparam componentStyle rectangle
skinparam ArrowThickness 2

actor User

package "voice CLI" {
  component "cli" as CLI
  component "pipeline" as Pipeline
}

cloud "External libraries" {
  [ffmpeg / ffprobe] as Ffmpeg
  [VibeVoice-ASR via mlx-audio] as MLX
  [pyannote 3.1] as Pyannote
  [mlx-lm (in-process)] as MlxLm
}

database "Local files" {
  folder "~/.cache/lm-studio/models" as LMSCache
  file "~/.cache/huggingface/token" as HFToken
}

User -[#red]-> CLI : audio path + opts
CLI -[#red]-> Pipeline : PipelineOptions (audio path)
Pipeline -[#7CCD7C]-> User : transcript.md

Pipeline ..> Ffmpeg : ffmpeg / ffprobe subprocess
Pipeline ..> MLX : ASR inference
Pipeline ..> Pyannote : diarization
Pipeline ..> MlxLm : LLM stages (postprocess / identify / structure / tldr)

MLX ..> LMSCache
MlxLm ..> LMSCache
Pyannote ..> HFToken

legend right
  <color:red>Red</color>: audio in
  <color:#7CCD7C>Green</color>: Markdown out
  Dotted arrows: library / file dependencies
end legend
@enduml
```

![High-level architecture](https://www.plantuml.com/plantuml/svg/NLJDRjim3BxhAGZlqc9mWcs70a4Hj4ZNRO235KrN32XwK2JQ5Y9BWYJtPyE6FSIUS4yoIh6JfXT38Z-I7_c9FWkH-zXtcaKLDSA3LGBBwBa9mgirUlZtvo-qgcbBZG-eWNlHgeWYzXvPs2ZZkEST2DivQz34LNocD9u0t3Jw9UJSU_juqognCHW2l6UCYsWzV0le6NDSR7Y3K6G6iAY-5F2JmJun54Ah0dX8laE7KmwrCfYzLyE5_M9CQDjBA3u-HVI6Qz1gxRbN6BQvx-gwSzZ05EhQURl6-vJWCXkJ-vO6S9i7ShwwXWV5eTDF9U-biXcvhBudc7lcnjY8y67oBjkl1aDofWZTmP4o9PKGrFdnDbO_LLtYA7daQnweyyeAubWhFVAhhPQaGF5xEX5Sj3ZLNHbYAZ_jh4GTSiFLShL8tXH0iI_WRTyqoZGr5pYDTeCcupzVtgHpgfr63-NT6u_olfodmS8CSd_WU6pXBLWN0qlsFMeSC477urSNbJK1ZlQnnso7ez2JnUBYP96YSyaPZ2_CnKadsHuxcSm70GZqMXu8_NeOuc442K7m998oDNeq0Wy1eoA4aefUm0-U2BzAaXGXG5KjWRQYGUh7sH27YaH3INfkgdwcOuY-ppj0vwYbFsOiaKXvTdfi4nwOTZoITHI2QMpGIF2qPP5KF1LMMp-ZSS-lKPvEytK-2gFC7ZBCTVx98vezSCdzHjf70xPF-IviZ2XfSsd_mcnOWjH4VYuI7HMdaT5Qi4HZouQTqOV-2_y1)

## Pipeline stages

`pipeline.run()` runs ten steps in order. ffprobe and the ffmpeg-WAV conversion both start from the user's input file; ASR and diarize then read the same temp WAV in parallel. Everything from merge onwards passes structured Python dataclasses around. Only the final two edges (TL;DR string → render → file) are Markdown.

```plantuml
@startuml
title voice-transcriber — pipeline stages
skinparam componentStyle rectangle
skinparam ArrowThickness 2

[input audio] as Input
[transcript.md] as Output

component "[1] ffprobe" as FF
component "[2] (ffmpeg via subprocess)" as WAV
component "[3] asr" as ASR
component "[4] diarize" as Diar
component "[5] merge" as Merge
component "[6] postprocess" as Post
component "[7] identify" as Ident
component "[8] structure" as Struct
component "[9] tldr" as TLDR
component "[10] render" as Render

Input -[#red]-> FF
Input -[#red]-> WAV

WAV -[#red]-> ASR : 16 kHz mono WAV
WAV -[#red]-> Diar : 16 kHz mono WAV

ASR -[#blue]-> Merge : AsrSegment[]\n(text + ASR speaker hint)
Diar -[#blue]-> Merge : DiarTurn[]\n(pyannote timeline)

Merge -[#blue]-> Post : Segment[]\n(text + pyannote label)
Post -[#blue]-> Ident : Segment[]\n(content proof-read)
Ident -[#blue]-> Struct : Segment[] + {label: name}
Struct -[#blue]-> TLDR : StructuredDialog\n(sections + segments)

FF -[#blue]-> Render : AudioMeta
TLDR -[#7CCD7C]-> Render : Markdown TL;DR string
Render -[#7CCD7C]-> Output

legend right
  <color:red>Red</color>: audio bytes
  <color:blue>Blue</color>: structured data (dataclass / JSON)
  <color:#7CCD7C>Green</color>: Markdown
end legend
@enduml
```

![Pipeline stages](https://www.plantuml.com/plantuml/svg/RPJFRXCn4CRlVefHkIH2eQH00w6A6XA55X6AaWWEOG_UzMHZPTTUsTwM527n43mXJyBOcp-xLIwMxVbz_kmPszVMSUFAF6DEkWpXNii4EyvmPHCZOpJmxyzVA6I1cLG8HATecTr8LN33SqXqNcY5oitTbkG64yTLcc4D6HgZ7nPhcMmKMWiNZ2qfL3hWfP0w0cxXre_PSczRk1Uv286xqla8EzZ0sR8RmMfL61tZcKScaqRq8eBMQfKNcCAzv63BcD24ZDk1_Zxyri1VUHiJGiFvh15w7O6GtCZ7ocTC_KRyJGGvchIAJdsl4RwCeD3MxTm3z9N63QONWHJKbQjj06xze46yZIZdfanSQIgZUHWrc7SHk4nKXrXy7ZTFqNqPKLMKm2e-2rt6GiQXitncK4ITWS_YqocVAaPDROfI17teNiBcvX5ohNI0cepFqmc8UIrHSLQYnqB2Y1jQCySqfyqken-gvV2dW-V1o1R8DtG1rrAvDWlBdj3x-KLfL50NMwwKTWXEvk72jXchm9hILu649rmFyep7cBLP86lAK9udqaGhvsUfpdhLCSX5crLSJLWLhQuajr_1fG-Av_YRxP2Qo9VII_Rb5tJKJAVaQUcLIQqiBMAh1IqTT3Afgwj2-mJxLpP5nrbOa93UQ3JkZHhGb9kDq0-AJDyJH5TEgfJjmWt9_aLcc58KZ4pNZW5S9JhJfa-x7CnGN9s7uQ1VlN68juv-ZGFbzpZuGCsHErno0O8x3YtV9Xcd3-CSFitllhIFyjrk1XyDeZekiJy3y_kgjiUkgQ7FxJy0)

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
├── postprocess.py   # fix_asr_errors(): per-segment LLM proof-reader with safety net
├── identify.py      # identify_speakers(): LLM self-intro detection with override / ask / keep
├── structure.py     # structure_dialog(): LLM-driven section layout with validation
├── tldr.py          # generate_tldr(): LLM markdown summary
├── render.py        # render_markdown(): final output
├── speaker_emojis.py # fixed palette assignment
└── llm.py           # MlxLLM: in-process mlx-lm wrapper with chat / chat_json (lm-format-enforcer)
```

## Data flow

The pipeline passes increasingly enriched `Segment` lists from stage to stage. Numbers match the labels in the diagram and the log lines.

1. **ffprobe** — `extract_metadata(audio)` → `AudioMeta` (start/end/duration). Goes straight to render.
2. **WAV conversion** — `ffmpeg` subprocess writes a 16 kHz mono WAV to a temp dir.
3. **ASR** — `asr.transcribe(wav)` → `AsrSegment[]` (text + ASR-side speaker hint).
4. **Diarize** — `diarize.diarize(wav)` → `DiarTurn[]` (pyannote timeline).
5. **Merge** — joins (3) and (4) into `Segment[]` (text + pyannote speaker label).
6. **Postprocess** — LLM proof-reads `content` per segment in-place.
7. **Identify** — LLM returns `{label → name}`; pipeline assigns `.name` on each segment.
8. **Structure** — LLM produces `StructuredDialog` (segments + section titles).
9. **TL;DR** — LLM emits a Markdown summary string.
10. **Render** — combines `AudioMeta`, `StructuredDialog`, and the TL;DR into the final Markdown file.

See [`pipeline.md`](pipeline.md) for the step-by-step sequence with timing details, and [`adr/`](adr/) for the decisions behind these boundaries.
