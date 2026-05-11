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
Pipeline ..> MlxLm : LLM stages (proofread / identify / structure / tldr)

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

![High-level architecture](https://www.plantuml.com/plantuml/svg/NLJ1QXin4BthAmRt4bFMfkqXO899S9gsiAOXJaCXv21Bixl2Mab8MiTkQUb7z0lx9Ht9NZlEPP5ctioycQVPon2vZrshMLHH8woj4bX4pqqGNYtHmxy_Vw5HTLDeNAC6xaMZ8ehOUMHXfOpZdhSWREkiGHCNSKjfF06ugVLBo9Ntzlc-KM9bC0HupnWNqNhu5j0pvhXOyHfXa1d0hFmoW9yCZebX8BGQ00_9T-fmog5M1hFtTdzXlnWJsdOI1Xyt4RtX6hHQUkuLXbtknwfg7TOmXgfotYxnAIMuoSRaVg01FEI1N2tkOAquj7fJy4ugcv7pydk2kIqtnaQ43wDtsNwxDoHdYj0jd2bJK0b1dGztiNfJjecZv94lUQ3EIok8OwLqpw-oMf82Wz_ZaU2YmR5kYn1J-N5h9UgG6oxdYrdofW2AVG5lkhfMfgwum76qApGP__dcFfrJxJHwA1xVU9HFnJGD5YQG-mF7OmDlmReOMB8VKVs02TeVlxgeh0bmTemTiWEEGizdi-lpMOhE9IVGlp0NHwraH-neDDqBGA3Nya3ebliXdaQG40HF8YbJeKSZy1JGAq8egUW5_E29yAiaIH42bEKSQ2jgf7wecgqySaaabAJ7fgej7KFqNTu0EaSj_P2nH25bEQYnItTYEh18rbu6fh568y7JZaNIy49ORVs4XZszGtanpjTfBuioHyX-gjDF7j4SWNFkLz8-6x1TodTjOqH9UqlR5oRx5KXDvCSYqL5n6XWj1KiqTcpO9Nteh_0V)

## Pipeline stages

`pipeline.run()` runs eleven steps in order. audiotranscode (ffmpeg → 16 kHz mono WAV) and audiometa (ffprobe) both start from the user's input file; diarize then reads the temp WAV first to produce per-turn boundaries, the loudness-normalize stage uses those boundaries as guard-rails for per-segment AGC, and ASR finally consumes the normalised WAV. Everything from merge onwards passes structured Python dataclasses around. Only the final two edges (TL;DR string → render → file) are Markdown.

```plantuml
@startuml
title voice-transcriber — pipeline stages
skinparam componentStyle rectangle
skinparam ArrowThickness 2

[<b>input audio</b>] as Input #A9A9A9
[<b>transcript.md</b>] as Output #A9A9A9

component "<b>[1] audiotranscode</b>\n<ffmpeg>" as WAV #FFA07A
component "<b>[2] audiometa</b>\n<ffprobe>" as FF #FFA07A
component "<b>[3] diarize</b>\n<pyannote-audio>\nspeaker-diarization-3.1" as Diar #87CEFA
component "<b>[4] normalize</b>\n<soundfile> + <numpy>" as Norm #E8E8E8
component "<b>[5] asr</b>\n<mlx-audio>\nVibeVoice-ASR" as ASR #90EE90
component "<b>[6] merge</b>" as Merge #E8E8E8
component "<b>[7] proofread</b>\n<mlx-lm>\ngemma-3-12b" as Post #DDA0DD
component "<b>[8] identify</b>\n<mlx-lm>\ngemma-3-12b" as Ident #DDA0DD
component "<b>[9] structure</b>\n<mlx-lm>\ngemma-3-12b" as Struct #DDA0DD
component "<b>[10] tldr</b>\n<mlx-lm>\ngemma-3-12b" as TLDR #DDA0DD
component "<b>[11] render</b>" as Render #E8E8E8

Input -[#red]-> FF
Input -[#red]-> WAV

WAV -[#red]-> Diar : <color:#404040>16 kHz mono WAV</color>
WAV -[#red]-> Norm : <color:#404040>16 kHz mono WAV</color>
Diar -[#blue]-> Norm : <color:#404040>DiarTurn[]</color>\n<color:#404040>(gain guard-rails)</color>
Norm -[#red]-> ASR : <color:#404040>16 kHz mono WAV</color>\n<color:#404040>(per-turn AGC)</color>

ASR -[#blue]-> Merge : <color:#404040>AsrSegment[]</color>\n<color:#404040>(text + ASR speaker hint)</color>
Diar -[#blue]-> Merge : <color:#404040>DiarTurn[]</color>\n<color:#404040>(pyannote timeline)</color>

Merge -[#blue]-> Post : <color:#404040>Segment[]</color>\n<color:#404040>(text + pyannote label)</color>
Post -[#blue]-> Ident : <color:#404040>Segment[]</color>\n<color:#404040>(content proof-read)</color>
Ident -[#blue]-> Struct : <color:#404040>Segment[] + {label: name}</color>
Struct -[#blue]-> TLDR : <color:#404040>StructuredDialog</color>\n<color:#404040>(sections + segments)</color>

FF -[#blue]-> Render : <color:#404040>AudioMeta</color>
TLDR -[#7CCD7C]-> Render : <color:#404040>Markdown TL;DR string</color>
Render -[#7CCD7C]-> Output

legend top right
  <color:#404040>**Box fill** (stage category):</color>
  <back:#FFA07A>   </back> <color:#404040>subprocess (external binary)</color>
  <back:#87CEFA>   </back> <color:#404040>AI model — "pyannote/speaker-diarization-3.1"</color>
  <back:#90EE90>   </back> <color:#404040>AI model — "mlx-community/VibeVoice-ASR-6bit"</color>
  <back:#DDA0DD>   </back> <color:#404040>AI model — "mlx-community/gemma-3-12b-it-qat-4bit"</color>
  <back:#E8E8E8>   </back> <color:#404040>pure-Python / DSP</color>
  <back:#A9A9A9>   </back> <color:#404040>I/O payload</color>
  <color:#404040>**Arrows:**</color>
  <color:red>Red</color>: <color:#404040>audio bytes</color>
  <color:blue>Blue</color>: <color:#404040>structured data (dataclass / JSON)</color>
  <color:#7CCD7C>Green</color>: <color:#404040>Markdown</color>
end legend
@enduml
```

![Pipeline stages](https://www.plantuml.com/plantuml/svg/dLRRRkCs47tNLmpYbyI5rdQInebh2AkDussAPXF4GVgGwe6IQ8KmHQeKrOstAD2Fw1VsIpgaR5cn8iTO2A0jQiuvGt8k_9XhedIHSaSppIdyALb4NQs8o2F5GghWlt_-XOnbb3D10QqJcZlvWecCA9929DDC2YhqJ2yHhcYaYKWuhPduIicl3yyiMWYQvt3iE4_ZqCElXGPIn4oEUw4N0CdXsivr_0lpMAkr8vd-aCQLtMsXwuPEvGGS8EPf49Iy9LZ6r01_5-FvFCreuXqOYj_yH-XCftv_w6_3ZrVmb6fI8JCbGrf2fzCsv4a0CIEAVLihPaiYXDJKjOGuasUKBAXoIpEYcHJkoOU1vPtW77JEXvTNqmRpQG12gfJm3NSk2n7F6QSU_01ZKQJPilJl2nf2v-hSFDiyFvhzKok6bBzKZZtYMJ_Qi_TdzvO6Vw5pqR-wkkXlivm5a5ALM4-ixOrvQzCS1e2x9-UAahYcp5Fybz0q9Uw9EpWEBT6TpF5S9nE_Fvbiyvm7m69yO_FbUpJNngwDvoB0C5P5f0j5toEQMSCsfa4_0CrZzHxBmw-J-rOEZ5P5HKnLjPltzhNQJgTC2_UfesWSk1u6O6CAmzbnJ4nlfcmuZM0SIIxLg7FQDuytE8F5pzyWbK8QqBXdFtjRM1j2UsEj4e93Nj1sjB5wA9Hu2jP0tBBN9eS9OGAIWgZOLOJn_AZIi9mR1qrmxkjVKoR33CJJ5-1_ljn8E8Qqjemog1igVgvcD4dn07UjHDCNZNbfA5Sv3yzCwAFMJMjHssVNrbK6D4jjcQwjgQIjwTWCQyZilw9AZ9EGyesIvQtfb2duFKAH5DfWRT5mJTNOg9IiDPbLWhRheCj_MLT784XA_wweLiWQbqtJ9jEwLiHu5bmcxOxdsFomeESecPVojG1si6tKj5OvtemkKuzlRETP0QrNY1nUNaw6bxkmDqGjOlbLu49-GWmMEIOgTvqLwXLJsKWTXzC4luAM6IYMF6i7jhcxtK_o1R3Ty6uN3kqr02AYQIBLycXKYI0k9D5Yj6gG7k14pynusuHv4U81H-P6S8YHHPKW748c2F8rwSgkk8FElyQyZoctrvM3TP3sslfjKw7iT7ihc5gFbJmj1DFBtglkwPw5JB-XKDR_xrIejHINQVSFejtJjsNAhh539iDOTk-M-baAwC5aTjVaA2zMEpYkUxUGaIMNfgLlqDinOwz--QZRRHfXFddtj48tWjbUJ21SQfetmIQBl4yujCAh3Xz3J3I1GpD6d62uzU2NsUsNepVyBXF3-wme5QtCwoohy2PpoWHoFkA0T-d_0G00)

## Module layout

```
src/voice/
├── cli.py           # argparse, --help, dispatch to pipeline.run()
├── pipeline.py      # PipelineOptions + run(): audiotranscode → audiometa → diarize → normalize → ASR → merge → proofread → identify → structure → tldr → render
├── types.py         # shared dataclasses (AsrSegment, DiarTurn, Segment, Section, StructuredDialog, AudioMeta)
├── ffprobe.py       # extract_metadata(): start/end/duration from ffprobe → birthtime → mtime
├── diarize.py       # diarize(): pyannote 3.1, MPS+CPU fallback, exclusive turns
├── audio_preprocess.py # loudness_normalize(): per-pyannote-turn RMS AGC with linear crossfades
├── asr.py           # transcribe(): mlx-audio wrapper, bitness 4/5/6/8, JSON timeline parse
├── merge.py         # merge(): per-segment max-overlap mapping ASR↔pyannote
├── proofread.py     # fix_asr_errors(): per-segment LLM proof-reader with safety net
├── identify.py      # identify_speakers(): LLM self-intro detection with override / ask / keep
├── structure.py     # structure_dialog(): LLM-driven section layout with validation
├── tldr.py          # generate_tldr(): LLM markdown summary
├── render.py        # render_markdown(): final output
├── speaker_emojis.py # fixed palette assignment
└── llm.py           # MlxLLM: in-process mlx-lm wrapper with chat / chat_json (lm-format-enforcer)
```

## Data flow

The pipeline passes increasingly enriched `Segment` lists from stage to stage. Numbers match the labels in the diagram and the log lines.

1. **audiotranscode** — `ffmpeg` subprocess writes a 16 kHz mono PCM WAV to a temp dir.
2. **audiometa** — `ffprobe.extract_metadata(audio)` → `AudioMeta` (start/end/duration). Goes straight to render.
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
