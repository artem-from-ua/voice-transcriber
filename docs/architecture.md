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

package "Pipeline stages" {
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

User --> CLI : audio path + opts
CLI --> Pipeline : PipelineOptions

Pipeline --> WAV : input audio
WAV --> ASR : 16 kHz mono WAV
Pipeline --> FF : input audio
FF --> Render : AudioMeta

ASR --> Merge : AsrSegment[]\n(text + ASR speaker hint)
Pipeline --> Diar : WAV
Diar --> Merge : DiarTurn[]\n(pyannote timeline)
Merge --> Post : Segment[]\n(text + pyannote label)
Post --> Ident : Segment[]\n(content proof-read)
Ident --> Struct : Segment[] + {label: name}
Struct --> TLDR : StructuredDialog\n(sections + segments)
TLDR --> Render : Markdown TL;DR string
Render --> User : transcript.md

WAV ..> Ffmpeg
FF ..> Ffmpeg
ASR ..> MLX
MLX ..> LMSCache
Diar ..> Pyannote
Diar ..> HFToken
Post ..> LLM
Ident ..> LLM
Struct ..> LLM
TLDR ..> LLM
LLM ..> MlxLm
MlxLm ..> LMSCache

legend right
  Solid arrows: data flow between stages
  Dotted arrows: library / file dependencies
end legend
@enduml
```

![Architecture overview](https://www.plantuml.com/plantuml/svg/RLNRZXCv47ttLvJoD15AihFiSibAY153158Y41a6fDb-SBghErRSTijsJsP083w2B-HBgBAxCqdlFYHnrJbLvRevBtsGBhIr5a45ZN1hLOwJuAJnkLCRTF3hnq_8RTrOWoQ0lKLtgt0l_4wPHZfPFu3hS4yU7EP1cagZ438Fri57Zqw8HkOxMI6COWHujNmxWgy2ZboFSgr683rZfq2Z6jJAO4JVTuBuTkIqLmAbKw7_Z-kRympAid5savmi5WFypmpEoh9ki89R9S6t6oBdwFquyZzTN0yC7cUaTn6yM7yOW7zbK2Zfr9SKxPBE0yRV6TJegeIl-3GWFCcWiJvqjqYPapnWFSr05NHKvNsal6LXm7cMKLLScuVMfN3hA0rOppC8kaW9NIqlXncT_v5HPqs1YV4X7ayvMjTmLkkx2VtkdMmQTAbuo-MAcvLhsnOmUdqNq1cfKuzkkgBF-lPaiJrHoV0rpU1rd4cgScmDHv1jeMnahfQVcTRSIsDiG7WyFO_wzvs2mUvEPyfC-gOcMtsthFbcXGno8ptDqTBc_SrAgob567sVJNEPRt6cwuaF77XMsm9rwibojNx5A1ieclu7-hQjAcMgKkOu2tQ79lBVBAxup55vBs0oUS7p3dE8EK4ZmnOUWMs25wnd_33SyyFnNHEKDLw80yP4cbBYq5ws8RaJh664YpU7yoUmU_C5QcjidEWJuyLYO4iArgTM4tR1sXK6AGIxOomEBaFUhR6gQGnkid_DMS2xG4amopSeTsI-LIQCJmFoLf0jNoGUZnsougfr9hexD3UeEfgFHUB5uj1IaC7_X3-OQRb1JS6PoIPnIGOskJM1jJGWjfmubCLO91vRf8qvDY7_NwFVEHXPupVHKPZDwyFSVkaAoaRRYg9uUXkvRsJjaoS_5f5-KkcLTBl2xWrv-eSmsbuQ9D71p8oZCuV-cMx2j2v4RFPq-gBV7khVaSJDO94NXZxnV1ZUM7_M7DRceEa7DbOl6j4sfyhqKfTwByPyUe4-AMpSjVXz6bfeh2WnSAhQyfkqjbeL89spUpy7tamejTt31iCUqNGFF14lRGZumDHguwIxvuU4zx30XijbSaLazf_2Y9VqHVzslm40)

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
