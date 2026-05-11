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

`pipeline.run()` runs eleven steps in order. ffprobe and the ffmpeg-WAV conversion both start from the user's input file; diarize then reads the raw temp WAV first to produce per-turn boundaries, the loudness-normalize stage uses those boundaries as guard-rails for per-segment AGC, and ASR finally consumes the normalised WAV. Everything from merge onwards passes structured Python dataclasses around. Only the final two edges (TL;DR string → render → file) are Markdown.

```plantuml
@startuml
title voice-transcriber — pipeline stages
skinparam componentStyle rectangle
skinparam ArrowThickness 2

[input audio] as Input
[transcript.md] as Output

component "[1] ffprobe" as FF
component "[2] (ffmpeg via subprocess)" as WAV
component "[3] diarize" as Diar
component "[4] normalize" as Norm
component "[5] asr" as ASR
component "[6] merge" as Merge
component "[7] postprocess" as Post
component "[8] identify" as Ident
component "[9] structure" as Struct
component "[10] tldr" as TLDR
component "[11] render" as Render

Input -[#red]-> FF
Input -[#red]-> WAV

WAV -[#red]-> Diar : 16 kHz mono WAV
WAV -[#red]-> Norm : 16 kHz mono WAV
Diar -[#blue]-> Norm : DiarTurn[]\n(gain guard-rails)
Norm -[#red]-> ASR : 16 kHz mono WAV\n(per-turn AGC)

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

![Pipeline stages](https://www.plantuml.com/plantuml/svg/RPJDRXCn4CVlVeeHNBf2eRH02w6A6bAb5D4MDHKSmXwSzMHhnMklR6zBYf1u29wG9s7ipSUwxSMoFR__p7Wyzh7ptFgcKinBhn1kZIomvotNhh1oXXR-_VaBjQnHIOr0T8cEkONKDRUyWi9KjT6e_SGlIMwny5oN2Zl8q5fpTtqZYuL6vs2ViIcP6W-y4TBam1sSXJMRhiFM_cKbek6oyS72Dc7WsNGlX_cyjcQ6pm8o7YVM_HnsvlEgnX9k9GVNp0WjA70MwU_3Rmd-EWSXkPNthRCJcYVsDpbeOokkriG5hHBYRKZKHjjmSfMO3dAeq9Qjy3pC4lDX3hLnVfLVXBxIEc7UvI05JULy6O6pi4Y8zpdTYcqAtzWsq2IkqfAzoi4hqMPv_UKaJNEFAcfH2spjLt7AMBmKw4sVMnHvRn0A_N0hL9FHqDaA5OG-x1t0ujCzL4QRIALGAE8JKDGIDLCDTh2mVTrOFSr_w9sIImrbmwteMIwLorYajhxf4Xwx9c6DjaSrqZ0y7MMC1QmJAjuEwOREJh2ig30nciUV7bv4dwv6lg3tS2ErpnwbkjOdkTPBhhNn25vMyVrGt1RiACELa_29g1kvuZDK6OjaHnbxuO6qCDg7NMegC-zPv29ZBTSHjXtILLAqNp5A7pIly3TR8HrHQ9iWMVUQeACgKr9CHuzU6kt8YMiTqgsmyRYhRfig53Wy-dFqd4M7X1oEHYU7emGwvtOXp9sce1y8elwMkcGhSo9PVn0AIxA2bUMDPm17XL76zYdBmHMAezsu7FJRBmTcIq-Vs8OAEGu-qh3bDYzAWE2UmquO2yNf_zg5pvFBYsmhNsKpEBM8UkjYVGOMychJOySqq5_x7m00)

## Module layout

```
src/voice/
├── cli.py           # argparse, --help, dispatch to pipeline.run()
├── pipeline.py      # PipelineOptions + run(): ffmpeg → ffprobe → diarize → normalize → ASR → merge → post → identify → structure → tldr → render
├── types.py         # shared dataclasses (AsrSegment, DiarTurn, Segment, Section, StructuredDialog, AudioMeta)
├── ffprobe.py       # extract_metadata(): start/end/duration from ffprobe → birthtime → mtime
├── diarize.py       # diarize(): pyannote 3.1, MPS+CPU fallback, exclusive turns
├── audio_preprocess.py # loudness_normalize(): per-pyannote-turn RMS AGC with linear crossfades
├── asr.py           # transcribe(): mlx-audio wrapper, bitness 4/5/6/8, JSON timeline parse
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
3. **Diarize** — `diarize.diarize(wav)` → `DiarTurn[]` (pyannote timeline). Runs on the raw WAV so its boundaries are not influenced by AGC.
4. **Normalize** — `audio_preprocess.loudness_normalize(wav, turns)` → sibling `<stem>.normalized.wav` with per-turn AGC (RMS → target dBFS, clamped by max gain, 100 ms linear crossfades on boundaries). Skipped when `--no-loudness-normalize` is passed; ASR then receives the raw WAV instead.
5. **ASR** — `asr.transcribe(normalized_wav)` → `AsrSegment[]` (text + ASR-side speaker hint).
6. **Merge** — joins (3) and (5) into `Segment[]` (text + pyannote speaker label).
7. **Postprocess** — LLM proof-reads `content` per segment in-place.
8. **Identify** — LLM returns `{label → name}`; pipeline assigns `.name` on each segment.
9. **Structure** — LLM produces `StructuredDialog` (segments + section titles).
10. **TL;DR** — LLM emits a Markdown summary string.
11. **Render** — combines `AudioMeta`, `StructuredDialog`, and the TL;DR into the final Markdown file.

See [`pipeline.md`](pipeline.md) for the step-by-step sequence with timing details, [`prompts.md`](prompts.md) for the LLM call contracts powering stages 7–10, and [`adr/README.md`](adr/README.md) for the index of architectural decisions behind these boundaries.
