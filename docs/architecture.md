# Architecture

voice-transcriber is a thin CLI on top of a linear in-process pipeline. The CLI parses arguments, builds a `PipelineOptions`, and hands it to `pipeline.run()`. Each stage is a small module with a single public function; intermediate results live in memory, and the only on-disk artefact (besides the input audio) is the final Markdown file.

External work — ASR inference, diarization, LLM prompting — is delegated to local libraries so the pipeline contains no model code itself: mlx-audio drives the VibeVoice model directly, pyannote.audio runs the diarization graph on the local GPU/CPU, and `llm.py` loads the LLM in-process via `mlx-lm` (no HTTP).

## High-level architecture

The CLI hands a `PipelineOptions` (whose load-bearing field is the audio path) to `pipeline.run()`. The orchestrator delegates the heavy work to four external libraries — `ffmpeg`/`ffprobe`, `mlx-audio` (VibeVoice-ASR), `pyannote.audio`, `mlx-lm` — and writes a single Markdown file back to the user.

The split between the CLI and the pipeline is intentional: `cli.py` only resolves user input (argument parsing, defaults, `--help`) and `pipeline.run()` is the single entry point both the CLI and the test suite call. That makes ordering tests possible without a subprocess — `tests/test_pipeline_ordering.py` monkey-patches each module-level reference and asserts the call sequence directly.

Models are never loaded twice. ASR (mlx-audio), diarization (pyannote 3.1) and the LLM (mlx-lm) each load their weights inside their stage and free them on return; the LLM is then loaded once for the four LLM-driven stages (`proofread`, `identify`, `structure`, `tldr`) and unloaded before `render`. Combined with the in-process design — no HTTP, no daemon — this keeps peak GPU memory bounded by the largest single model, not by their sum.

The only on-disk artefacts the pipeline writes itself are the final Markdown file and an optional per-stage dump tree (`--dump-stages DIR`) used for regression triage. Everything else — temp WAV files, model caches — lives outside the pipeline contract: `~/.cache/lm-studio/models` for MLX weights, `~/.cache/huggingface/token` for the pyannote gated-repo token, a `tempfile.mkdtemp()` directory for the ffmpeg WAV that is cleaned up on return.

## Pipeline stages

`pipeline.run()` runs eleven steps in order. transcode (ffmpeg → 16 kHz mono WAV) and audiometa (ffprobe) both start from the user's input file; diarize then reads the temp WAV first to produce per-turn boundaries, the **clearspeech** stage uses those boundaries as guard-rails for a chain of DSP effects (autogain, optional bandpass), and speech2text finally consumes the output of the last applied effect. Everything from merge onwards passes structured Python dataclasses around. Only the final two edges (TL;DR string → render → file) are Markdown. The clearspeech chain absorbs new audio-cleanup effects (denoise, presence boost, de-esser planned in PR-2..4) without growing the stage count.

```plantuml
@startuml
title voice-transcriber — pipeline stages
skinparam componentStyle rectangle
skinparam ArrowThickness 2
skinparam legendBorderColor transparent
skinparam legendBackgroundColor #EEEEEE

[<b>input audio</b>\n<i>mp3, wav, m4a, mp4 ...</i>] as Input #A9A9A9
[<b>transcript.md</b>] as Output #A9A9A9

component "<b>[1] transcode</b>\n<i><ffmpeg></i>" as WAV #E8E8E8
component "<b>[2] audiometa</b>\n<i><ffprobe></i>" as FF #E8E8E8
component "<b>[3] diarize</b>\n<i><pyannote-audio> speaker-diarization-3.1</i>" as Diar #87CEFA
component "<b>[4] clearspeech</b>\n<i><soundfile> autogain</i>\n<i><scipy> <s>bandpass</s></i>\n<i><scipy> <s>presence</s></i>\n<i><ffmpeg> <s>denoise</s></i>\n<i><scipy> <s>dereverb</s></i>" as CS #E8E8E8
component "<b>[5] speech2text</b>\n<i><mlx-audio> VibeVoice-ASR</i>" as ASR #90EE90
component "<b>[6] merge</b>" as Merge #E8E8E8
component "<b>[7] proofread</b>\n<i><mlx-lm> gemma-3-12b</i>" as Post #FFCC66
component "<b>[8] identify</b>\n<i><mlx-lm> gemma-3-12b</i>" as Ident #FFCC66
component "<b>[9] structure</b>\n<i><mlx-lm> gemma-3-12b</i>" as Struct #FFCC66
component "<b>[10] tldr</b>\n<i><mlx-lm> gemma-3-12b</i>" as TLDR #FFCC66
component "<b>[11] render</b>" as Render #E8E8E8

Input -[#FF6B35]-> FF
Input -[#FF6B35]-> WAV

WAV -[#FF6B35]-> Diar : <color:#404040>  16 kHz mono WAV</color>
WAV -[#FF6B35]-> CS : <color:#404040>  16 kHz mono WAV</color>
Diar -[#3B82F6]-> CS : <color:#404040>  speaker timecodes</color>
CS -[#FF6B35]-> ASR : <color:#404040>  cleaned audio</color>

ASR -[#6E9E1F]-> Merge : <color:#404040>  recognized text</color>
Diar -[#3B82F6]-> Merge : <color:#404040>  speaker timecodes</color>

Merge -[#6E9E1F]-> Post : <color:#404040>  text with speaker labels</color>
Post -[#6E9E1F]-> Ident : <color:#404040>  proof-read text</color>
Ident -[#6E9E1F]-> Struct : <color:#404040>  text + speaker names</color>
Struct -[#6E9E1F]-> TLDR : <color:#404040>  split by topic sections</color>

FF -[#3B82F6]-> Render : <color:#404040>  recording date + duration</color>
TLDR -[#6E9E1F]-> Render : <color:#404040>  summary</color>
Render -[#6E9E1F]-> Output

legend top center
  <back:#A9A9A9>   </back> <i><color:#404040>In/Out artifacts</color></i>
  <back:#87CEFA>   </back> <i><color:#404040>AI model: "pyannote/speaker-diarization-3.1"</color></i>
  <back:#90EE90>   </back> <i><color:#404040>AI model: "mlx-community/VibeVoice-ASR-6bit"</color></i>
  <back:#FFCC66>   </back> <i><color:#404040>AI model: "mlx-community/gemma-3-12b-it-qat-4bit"</color></i>
  <color:#FF6B35>━━►</color> <i><color:#404040>audio bytes</color></i>
  <color:#3B82F6>━━►</color> <i><color:#404040>metadata (turns, recording date)</color></i>
  <color:#6E9E1F>━━►</color> <i><color:#404040>transcript text (recognized → enriched → Markdown)</color></i>
end legend
@enduml
```

![Pipeline stages](https://www.plantuml.com/plantuml/svg/dPRDRkCs483lUef1VDcYaHt_h3SE364Thuq6w6ABP94UKXyeQUmGbaYLefBrdjfB7w1Oz67wE7cI3abPjXohS6iPXYNEVCCPpe_UvvefNQI9fxbE49uaZz3NYeayKZn41I-_VOECPvXmWK3IAyozVCr5nXHB8P9f9WKAVQStfAumqaoi4jmJcIWbdpy_ycWjCC-XjxUKu0f5V2rLZ6eg4wd0MgP58hwMOz5wfMGXOYVRcjcFvpsCmu142msiYBaSTyBW5p7cGPhrp-6PFPr3Ec3qamsWtMwFEpnO0ClXnggq9YDpMSZMxqotqzXWhDodGkyBUfNFS4Ow3zs5stKaOwnCZvVBDCDLO8oT6SZFatlQyAMv3W6zXTjtYfhj0p8bGzmHvlCcG7y1CMUAVzspdssO45AZRz41v1coDIhV2JBDfV3xxMv5_q3FeNNvRZgRJmxvWmL42J9538mUTpPoSnPBdc103cYvObmONhaOyMmJm3WFGYRYZENvk9C7nzOpXJcA2ElhPG2DG8n2yXoRz2bvy0bLk5st_aplcgBrTW7EaPx6BthdJPfysGRhdZB_tbR2vEwsGj9_Q8qkPhFHnI5pk80KrSh6tyf-D7TDEtYt03fRkLJ8uhhz90rWXMdA_BxVxOMLwPzaJWaudq-dm-4XxN81d2Aa-N9p6kp6I3VHHXGThOf85mfFmzrPyIPUzuBA8udLQQpFFtwuRIHHeL5VeDEkednhRwimUwwY_GSY3A_xRnT-G5Lpx2bLe-UPagmzjJLm1UF8j9Uhrk32N050TmZh7xv2AeKqckEE5GXU0oZf_eEwDKVw_UlBtdpOh5-MBcYUecaoUKKW-Pfzaw570APu1SRRxbWgUqQQr8UpqQmxD-ekQOy0gAtBbQ3-4eEhcKOF6X7DJdXEfxOJc-z7ACOwF7FzMF4I5cAoWrd56ifb-n6MBK7Vr63TAQTGOvGfthIXxwlT29RkULQgrL0smu_69-4QmWrecV48SXgarABteaJjlnRfClSRJal5NAmWPXffTt6XRCElO7OJjMqrq_A2obHjAjrIigRj9gJdkJbjN828WeZA0nY7DBQlok59FA3gfIVKkQaNr8tTYCudCyGLzJ8MwSfvqnvs93UZ_fqqkQ5oYp6vWhFj8EmqJC2purPSdp_PYcbgrA_IGd2zwTJcXpyCkMwmuZhT_xIorq1zhlrVcVO7nqoLEDScWfTllvllNtzlnOwOj8s2ab7ZgpCe1LqUdWGphpMKY0pUq4mH-Vb1VdxNOC5brqaMTgzjhXxVxFMhbp_-11IAHu_bpKUcrh5y5dMx9crTzdhlwOTUYFy1)

## Module layout

```
src/voice/
├── cli.py           # argparse, --help, dispatch to pipeline.run()
├── pipeline.py      # PipelineOptions + run(): transcode → audiometa → diarize → clearspeech → speech2text → merge → proofread → identify → structure → tldr → render
├── types.py         # shared dataclasses (AsrSegment, DiarTurn, Segment, Section, StructuredDialog, AudioMeta)
├── transcode.py     # transcode(): ffmpeg → 16 kHz mono PCM WAV
├── audiometa.py     # extract_metadata(): start/end/duration from ffprobe → birthtime → mtime
├── diarize.py       # diarize(): pyannote 3.1, MPS+CPU fallback, exclusive turns
├── clearspeech.py   # clearspeech(): chain-of-DSP-effects (autogain + bandpass + presence; de-ess / denoise planned)
├── speech2text.py   # transcribe(): mlx-audio wrapper, bitness 4/5/6/8, JSON timeline parse
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

1. **transcode** — `transcode.transcode(audio, wav)` runs `ffmpeg` and writes a 16 kHz mono PCM WAV to a temp dir.
2. **audiometa** — `audiometa.extract_metadata(audio)` → `AudioMeta` (start/end/duration). Goes straight to render.
3. **Diarize** — `diarize.diarize(wav)` → `DiarTurn[]` (pyannote timeline). Runs on the raw WAV so its boundaries are not influenced by autogain.
4. **Clearspeech** — `clearspeech.clearspeech(wav, chain, autogain_turns=turns, ...)` runs an ordered chain of DSP effects. Available effects: `autogain`, `bandpass`, `presence`. Chain is configured by the single `--clearspeech-chain` CLI flag (default `"autogain"`). Each enabled effect writes a sibling WAV (`<stem>.autogain.wav`, `<stem>.autogain.bandpass.wav`, `<stem>.autogain.bandpass.presence.wav`, …) and the next effect reads it; speech2text consumes the last applied effect's output. With `--clearspeech-chain ""` the chain is empty and speech2text receives the raw WAV. See [ADR 0010](adr/0010-clearspeech-chain.md) for the chain-of-effects design, [ADR 0012](adr/0012-clearspeech-chain-string-cli.md) for the single-string CLI, and [ADR 0016](adr/0016-rename-agc-to-autogain.md) for the `agc` → `autogain` rename.
5. **speech2text** — `speech2text.transcribe(cleaned_wav)` → `AsrSegment[]` (text + ASR-side speaker hint).
6. **Merge** — joins (3) and (5) into `Segment[]` (text + pyannote speaker label).
7. **Postprocess** — LLM proof-reads `content` per segment in-place.
8. **Identify** — LLM returns `{label → name}`; pipeline assigns `.name` on each segment.
9. **Structure** — LLM produces `StructuredDialog` (segments + section titles).
10. **TL;DR** — LLM emits a Markdown summary string.
11. **Render** — combines `AudioMeta`, `StructuredDialog`, and the TL;DR into the final Markdown file.

See [`pipeline.md`](pipeline.md) for the step-by-step sequence with timing details, [`prompts.md`](prompts.md) for the LLM call contracts powering stages 7–10, and [`adr/README.md`](adr/README.md) for the index of architectural decisions behind these boundaries.
