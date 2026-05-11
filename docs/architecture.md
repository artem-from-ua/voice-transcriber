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

`pipeline.run()` runs eleven steps in order. audiotranscode (ffmpeg → 16 kHz mono WAV) and audiometa (ffprobe) both start from the user's input file; diarize then reads the temp WAV first to produce per-turn boundaries, the **clearspeech** stage uses those boundaries as guard-rails for a chain of DSP effects (AGC, optional bandpass), and ASR finally consumes the output of the last applied effect. Everything from merge onwards passes structured Python dataclasses around. Only the final two edges (TL;DR string → render → file) are Markdown. The clearspeech chain absorbs new audio-cleanup effects (denoise, presence boost, de-esser planned in PR-2..4) without growing the stage count.

```plantuml
@startuml
title voice-transcriber — pipeline stages
skinparam componentStyle rectangle
skinparam ArrowThickness 2
skinparam legendBorderColor transparent
skinparam legendBackgroundColor #EEEEEE

[<b>input audio</b>\n<i>mp3, wav, m4a, mp4 ...</i>] as Input #A9A9A9
[<b>transcript.md</b>] as Output #A9A9A9

component "<b>[1] audiotranscode</b>\n<i><ffmpeg></i>" as WAV #FFA07A
component "<b>[2] audiometa</b>\n<i><ffprobe></i>" as FF #FFA07A
component "<b>[3] diarize</b>\n<i><pyannote-audio> speaker-diarization-3.1</i>" as Diar #87CEFA
component "<b>[4] clearspeech</b>\n<i><soundfile> agc</i>\n<i><scipy> <s>bandpass</s></i>\n<i><scipy> <s>presence</s></i>\n<i><ffmpeg> <s>denoise</s></i>" as CS #E8E8E8
component "<b>[5] asr</b>\n<i><mlx-audio> VibeVoice-ASR</i>" as ASR #90EE90
component "<b>[6] merge</b>" as Merge #E8E8E8
component "<b>[7] proofread</b>\n<i><mlx-lm> gemma-3-12b</i>" as Post #DDA0DD
component "<b>[8] identify</b>\n<i><mlx-lm> gemma-3-12b</i>" as Ident #DDA0DD
component "<b>[9] structure</b>\n<i><mlx-lm> gemma-3-12b</i>" as Struct #DDA0DD
component "<b>[10] tldr</b>\n<i><mlx-lm> gemma-3-12b</i>" as TLDR #DDA0DD
component "<b>[11] render</b>" as Render #E8E8E8

Input -[#red]-> FF
Input -[#red]-> WAV

WAV -[#red]-> Diar : <color:#404040>16 kHz mono WAV</color>
WAV -[#red]-> CS : <color:#404040>16 kHz mono WAV</color>
Diar -[#blue]-> CS : <color:#404040>speaker turn boundaries</color>
CS -[#red]-> ASR : <color:#404040>cleaned audio</color>

ASR -[#blue]-> Merge : <color:#404040>recognized text</color>
Diar -[#blue]-> Merge : <color:#404040>speaker turn boundaries</color>

Merge -[#blue]-> Post : <color:#404040>text with speaker labels</color>
Post -[#blue]-> Ident : <color:#404040>proof-read text</color>
Ident -[#blue]-> Struct : <color:#404040>text + speaker names</color>
Struct -[#blue]-> TLDR : <color:#404040>text split into sections</color>

FF -[#blue]-> Render : <color:#404040>recording date + duration</color>
TLDR -[#7CCD7C]-> Render : <color:#404040>summary</color>
Render -[#7CCD7C]-> Output

legend top right
  <color:#404040>**Box fill** (stage category):</color>
  <back:#FFA07A>   </back> <color:#404040>subprocess (external binary)</color>
  <back:#87CEFA>   </back> <color:#404040>AI model — "pyannote/speaker-diarization-3.1"</color>
  <back:#90EE90>   </back> <color:#404040>AI model — "mlx-community/VibeVoice-ASR-6bit"</color>
  <back:#DDA0DD>   </back> <color:#404040>AI model — "mlx-community/gemma-3-12b-it-qat-4bit"</color>
  <back:#E8E8E8>   </back> <color:#404040>pure-Python</color>
  <back:#A9A9A9>   </back> <color:#404040>I/O payload</color>
  <color:#404040>**Arrows:**</color>
  <color:red>Red</color>: <color:#404040>audio bytes</color>
  <color:blue>Blue</color>: <color:#404040>structured data (dataclass / JSON)</color>
  <color:#3A9D3A>Green</color>: <color:#404040>Markdown</color>
end legend
@enduml
```

![Pipeline stages](https://www.plantuml.com/plantuml/svg/dLRRRkCs47tNLmpYbyHTnKwSvc8OmZfst6RHRHR98dr8_K194uMmHAeajOdtQJ-YNzWlwP1oPCMoKgCM85Zad7E6v5puKHkcJ96bdk4cHVWcUOI-KKpeIF4G5Vppus_8UOuf5mXadQ3sz8ABd2cMGIIpN0eKviui2QumCamaATPCnah9vwzFF5e8r1gEQrCf9YZYIwbYL1EPIWLEcIQ9iMd7eaMYP27YqhPpvNwUzp0A0p8j3B0YvdBK2uC_nOW7MJxu0C_ismV8JXYzyXCuF3mSzNWm1wRXsa4wumlxE9BNTUVcC8ijZRExAKpTqAlM37k4UJYQbx8bMCPOwOyU7xCSay0gxbccFyRtq9dDnlspyIRByOebGyFg1BcI8QuPPhCsWi4SOiuK_rxJpvTC26dGTzG1w1pP0fLV6ZB3fV07XqSL-vJ6eND-DhcQDVXFvX2bo1HnOFIqrj3sG1vvYW6m9B9Kg_68vyi0HZe8cOXpflMefuDjyxb2ZIB2j_EhlRC6CGh9TJNlN9tSqVcVssVJqP_jeQcrWrdwyhh-UuheUnVXuxlRYeh-G-UYVtLrqT_aEfr3XYfnM-fiFzklDkMpET1noKU5B7whdsO199Xbp1_uHyTX9Vr5QWgiwNJSdqutsSxdm6dbXZykToExjjPjR1TpIbvLHAPGk1lTdJDluplgpy6aiTgDw-jlqzjM9aeWodSg0TKktxhFQfkzCbFzXux2UEu7b0MD8KejpxFvjHvoqJo4KMIhnR1pqhTFS7GAYr-_GoQ5jA1Hpqq76rWAhvsHJeUWOLfW6tQLUK3RBo2qAKCPYBhY8CnQt0PaWy4cdy3ujSIjS9wrhMcNyTa0KrsMYQ3Q48F15zFgUWlylxptIboDo0LsWyTgmpCtJw-L258MOhecSh0QJHdI3HwNPRvDixVBASrh-5KCRtVafyeBmRBQQbQW6eqBt-qaEa-v0IwC14tDZofgRLkeMjTOLb6zzN1Kp4K2CJD8RiM5ShMv8dBwn7GscKpF9kznwOAIJoqhvChk3RRiPvvNTbKmCWV5aoVZmIPRjtifNu2AUzhjmhvh_X2HZubKouDX9KAua7hpSDMP0g21dXq9ckw5T7QHlGViq-wX4Yo5a0jo-Q19LpQYT-Z6rvIDCQRkahBtskrwBMrkhwbGLlsT5Mo9e-gL5OARPUzDD_5FGswsA9Grxtygr4gfput_5pF-oNQPibA-8vDJtVU_BCrJBQmgT7cpUGTztRk1d2rJQHlR6hqPBUwgfuVTRjE8IbjmYnMw4RYkh46uDBL4hC0sWu9BUhN2gzuMsoHYi6_VKKgt3kZ1fxkRtm-s-3qONqm7u-0NXIXQcJypjOZbythNRCwKgUDzf1VTdVy5)

## Module layout

```
src/voice/
├── cli.py           # argparse, --help, dispatch to pipeline.run()
├── pipeline.py      # PipelineOptions + run(): audiotranscode → audiometa → diarize → clearspeech → ASR → merge → proofread → identify → structure → tldr → render
├── types.py         # shared dataclasses (AsrSegment, DiarTurn, Segment, Section, StructuredDialog, AudioMeta)
├── ffprobe.py       # extract_metadata(): start/end/duration from ffprobe → birthtime → mtime
├── diarize.py       # diarize(): pyannote 3.1, MPS+CPU fallback, exclusive turns
├── clearspeech.py   # clearspeech(): chain-of-DSP-effects (AGC + bandpass + presence; de-ess / denoise planned)
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
4. **Clearspeech** — `clearspeech.clearspeech(wav, chain, agc_turns=turns, ...)` runs an ordered chain of DSP effects. Available effects: `agc`, `bandpass`, `presence`. Chain is configured by the single `--clearspeech-chain` CLI flag (default `"agc"`). Each enabled effect writes a sibling WAV (`<stem>.agc.wav`, `<stem>.agc.bandpass.wav`, `<stem>.agc.bandpass.presence.wav`, …) and the next effect reads it; ASR consumes the last applied effect's output. With `--clearspeech-chain ""` the chain is empty and ASR receives the raw WAV. See [ADR 0010](adr/0010-clearspeech-chain.md) for the chain-of-effects design and [ADR 0012](adr/0012-clearspeech-chain-string-cli.md) for the single-string CLI.
5. **ASR** — `asr.transcribe(cleaned_wav)` → `AsrSegment[]` (text + ASR-side speaker hint).
6. **Merge** — joins (3) and (5) into `Segment[]` (text + pyannote speaker label).
7. **Postprocess** — LLM proof-reads `content` per segment in-place.
8. **Identify** — LLM returns `{label → name}`; pipeline assigns `.name` on each segment.
9. **Structure** — LLM produces `StructuredDialog` (segments + section titles).
10. **TL;DR** — LLM emits a Markdown summary string.
11. **Render** — combines `AudioMeta`, `StructuredDialog`, and the TL;DR into the final Markdown file.

See [`pipeline.md`](pipeline.md) for the step-by-step sequence with timing details, [`prompts.md`](prompts.md) for the LLM call contracts powering stages 7–10, and [`adr/README.md`](adr/README.md) for the index of architectural decisions behind these boundaries.
