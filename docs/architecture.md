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

User -[#FF6B35]-> CLI : audio path + opts
CLI -[#FF6B35]-> Pipeline : PipelineOptions (audio path)
Pipeline -[#A8D957]-> User : transcript.md

Pipeline ..> Ffmpeg : ffmpeg / ffprobe subprocess
Pipeline ..> MLX : ASR inference
Pipeline ..> Pyannote : diarization
Pipeline ..> MlxLm : LLM stages (proofread / identify / structure / tldr)

MLX ..> LMSCache
MlxLm ..> LMSCache
Pyannote ..> HFToken

legend right
  <color:red>Red</color>: audio in
  <color:#A8D957>Green</color>: Markdown out
  Dotted arrows: library / file dependencies
end legend
@enduml
```

![High-level architecture](https://www.plantuml.com/plantuml/svg/NLJRRjim37tNLn3PIul12RQYkmH5qAnTjW4ELZHTCQ3eWoBHjX1P4YIvJJTis4Vi2_SbevIuQVfYIEGXUSWZ-ZH4xcFNQXPLr0XtLWaiekSc2A-MwE7Vdx_GgBefDDwX1kv5eoAAs7baOQMCuvwt86phh44J5_61qdW2S5Fhfv2fz_R-gb5YPJ04UCqO5z5w-1RGC-QuMF4QOP0Pm5dvPG0_6UmImq1eDG0Uac_VuPH3hGncxyNsmdun9hJj90m-hYDwmpLejVJSAmoRt3TLrJgiOGHLvRnTucrACCl6vBwc0LpdWKmNbt2dEBHwNV1EAfkHy_9xWha7ReoD24V3Lzb-iJKaPuhGBHmeKr09GPqVRcBrkcmJHyaZN_90dPTM4CTAwPvVPRKa1GQ_HqF1HOCZtHOXflAZrahK8JTSpXTdoPi2AFO9lEdgMfcwuW97qQxGPFpdsLKwfzffz53Sl9ZDtdmuEhuj9ccsC8RS8ZWU6tW9riN0adqFs8-PqFtngulAcW07k_13zeYZwEcxy_V7RrDqBZo6_fsvE6mbss67mybM1W8z5mb2jzoESZ-2v21uaagP2ZqQWVk0NXT2IKNlu0TFX9ybIReGe2pdGBjH8_L3rMpbaKiYeIGzELKzq354t-Kze7FKqXyobYYaB8-QRDBjcHvP96il1jDOev7WQTCYANaYhBP-x55EBb6UZF9rqYkZp0wodUhaaqSqE-2S-vMqzmPib_ATsnXHqlRI3eRnTX5I4_d9I7HKd0P6gy4IZGqRTaeV-ZVy1m00)

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

component "<b>[1] audiotranscode</b>\n<i><ffmpeg></i>" as WAV #E8E8E8
component "<b>[2] audiometa</b>\n<i><ffprobe></i>" as FF #E8E8E8
component "<b>[3] diarize</b>\n<i><pyannote-audio> speaker-diarization-3.1</i>" as Diar #87CEFA
component "<b>[4] clearspeech</b>\n<i><soundfile> agc</i>\n<i><scipy> <s>bandpass</s></i>\n<i><scipy> <s>presence</s></i>\n<i><ffmpeg> <s>denoise</s></i>\n<i><scipy> <s>dereverb</s></i>" as CS #E8E8E8
component "<b>[5] asr</b>\n<i><mlx-audio> VibeVoice-ASR</i>" as ASR #90EE90
component "<b>[6] merge</b>" as Merge #E8E8E8
component "<b>[7] proofread</b>\n<i><mlx-lm> gemma-3-12b</i>" as Post #FFCC66
component "<b>[8] identify</b>\n<i><mlx-lm> gemma-3-12b</i>" as Ident #FFCC66
component "<b>[9] structure</b>\n<i><mlx-lm> gemma-3-12b</i>" as Struct #FFCC66
component "<b>[10] tldr</b>\n<i><mlx-lm> gemma-3-12b</i>" as TLDR #FFCC66
component "<b>[11] render</b>" as Render #E8E8E8

Input -[#FF6B35]-> FF
Input -[#FF6B35]-> WAV

WAV -[#FF6B35]-> Diar : <color:#404040>16 kHz mono WAV</color>
WAV -[#FF6B35]-> CS : <color:#404040>16 kHz mono WAV</color>
Diar -[#3B82F6]-> CS : <color:#404040>speaker turn boundaries</color>
CS -[#FF6B35]-> ASR : <color:#404040>cleaned audio</color>

ASR -[#A8D957]-> Merge : <color:#404040>recognized text</color>
Diar -[#3B82F6]-> Merge : <color:#404040>speaker turn boundaries</color>

Merge -[#A8D957]-> Post : <color:#404040>text with speaker labels</color>
Post -[#A8D957]-> Ident : <color:#404040>proof-read text</color>
Ident -[#A8D957]-> Struct : <color:#404040>text + speaker names</color>
Struct -[#A8D957]-> TLDR : <color:#404040>text split into sections</color>

FF -[#3B82F6]-> Render : <color:#404040>recording date + duration</color>
TLDR -[#A8D957]-> Render : <color:#404040>summary</color>
Render -[#A8D957]-> Output

legend top right
  <back:#A9A9A9>   </back> <i><color:#404040>In/Out artifacts</color></i>
  <back:#87CEFA>   </back> <i><color:#404040>AI model: "pyannote/speaker-diarization-3.1"</color></i>
  <back:#90EE90>   </back> <i><color:#404040>AI model: "mlx-community/VibeVoice-ASR-6bit"</color></i>
  <back:#FFCC66>   </back> <i><color:#404040>AI model: "mlx-community/gemma-3-12b-it-qat-4bit"</color></i>
  <color:#FF6B35>**▬▶**</color> <i><color:#404040>audio bytes</color></i>
  <color:#3B82F6>**▬▶**</color> <i><color:#404040>metadata (turns, recording date)</color></i>
  <color:#6E9E1F>**▬▶**</color> <i><color:#404040>transcript text (recognized → enriched → Markdown)</color></i>
end legend
@enduml
```

![Pipeline stages](https://www.plantuml.com/plantuml/svg/dLRDRkCs4BxhAGR9PPj6TXmxJXmOmZg9ZGtGnHR98ZsaFb3IM24iaIf99-izzTG7A8fzYfxxG7cI3YbPjc9hQpG2Z8YSx_luC_EDtcl3b5bagMUuIH6U9O_GDue97IaUeeBNt_-2dEUOSe50qGbgJy-vo9bY6KGoowL0OUxDak0A8yD4ak96o4Wf-VBvYKTpWLh3wSPKYWcA-4gg6DMrJAK2fqoJnBWTnw9vekH2n4NiuTZzUTxZC0meT660BM8kX-qm-5KCUP3btMDuOS_7aFKO_UGzQBLQmpOFfi0qt3h8uMXW7qUosdTkMbbiQLpSfuNP3FIgFSC1OHuxqqAs0CiOA_tXR9Rbc0HMyS0o_J9we5LVsESjosd9ag5XcmIvaY6k6IQJ9eBk56BE5F-wePylcH3IeE-e0z0viZaglmXaXalXTrkTYls6nk7mulnwF1czvUzD8KgHAUB0w6cjeUs5p7YA0R0aibJbUCJpPG137OHCn3dJUjZMmQxvNA564M5zlZmx6n2Za5nZ4vwI1vzHXQjvjvNh-wQ3EhENgjORoD8lg_Dve8n_S1Kmkh-hgEX_E1oSZCU3axTS_Ibag19tv2xsetrhKZwV0bsdd2baSLq_pG98CCkOt_Kxfs4b_RFKb7YJoVLrl_-MxM8Ad4x6yDboFx9R6zt4DfXISQj5P1OAzwExT-5DV9sJAPWqLljnVVxfvgwHYGgC_81kkJhbE_TQ7RDNLBB_I0pzg-xPr0-eK7QDKW5wdgt2sgXB-qiOHjPMBWzx9_O9EdsOV_WAcHJIuePjDnrimodLzWOxAK9thovE9_qcT5cdG9SX8BG5HlMAkc8XJ6q5DaEtI6or2enNdbX2FHjBuD75pU3it8ABbDs2avNBH92Tn63mY_dE3Xe8_ciJNe6hhSNb-nQJrOSNRfvM1WOf2p5TKpbOZQZ8zIqcLtw-hR_wfehm6aEPthiNysEr4i6oZJsLe1gHo-tTD3fFkG4kZ0HDdPFiUEDuoEfhnrmc_SwBKZ4N2SJC82qjNYXdxHMLMq5jJKrSUa6rgPOLieohOOjsw7b5KmOZSr0yUJ8Um32a5drPDie0QA1jHyYbgVxhIhUY_SasR4N-nI9JxTjQmfgfQ4NVPnhTKfd5c5x2mQhVjHiQtS5kbSBRzrQnHaOUbIq4DyjshMVu_P2R1fN2tVwdoePf-jpulp7ZztP9bNI5DGH7Hw_V_dxzzi_HqIfkXwRp1mYN1hSkeGmiSd0_DllzGdd8u9sjUtqCzVJyeK6YFnwCEvFz9DPVQQw8uTs6LRt-yIUWK3nwAby-CZMFvOke2zl4BVBNUqy_zFtxBm00)

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
