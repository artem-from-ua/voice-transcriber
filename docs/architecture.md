# Architecture

voice-transcriber is a thin CLI on top of a linear in-process pipeline. The CLI parses arguments, builds a `PipelineOptions`, and hands it to `pipeline.run()`. Each stage is a small module with a single public function; intermediate results live in memory, and the only on-disk artefact (besides the input audio) is the final Markdown file.

External work — ASR inference, diarization, LLM prompting — is delegated to local libraries so the pipeline contains no model code itself: mlx-audio drives the VibeVoice model directly, pyannote.audio runs the diarization graph on the local GPU/CPU, and `llm.py` loads the LLM in-process via `mlx-lm` (no HTTP).

## High-level architecture

The CLI hands a `PipelineOptions` (whose load-bearing field is the audio path) to `pipeline.run()`. The orchestrator delegates the heavy work to four external libraries — `ffmpeg`/`ffprobe`, `mlx-audio` (VibeVoice-ASR), `pyannote.audio`, `mlx-lm` — and writes a single Markdown file back to the user.

The split between the CLI and the pipeline is intentional: `cli.py` only resolves user input (argument parsing, defaults, `--help`) and `pipeline.run()` is the single entry point both the CLI and the test suite call. That makes ordering tests possible without a subprocess — `tests/test_pipeline_ordering.py` monkey-patches each module-level reference and asserts the call sequence directly.

Models are never loaded twice. ASR (mlx-audio), diarization (pyannote 3.1) and the LLM (mlx-lm) each load their weights inside their stage and free them on return; the LLM is then loaded for the four LLM-driven stages (`proofread`, `identify`, `structure`, `tldr`) and unloaded before `render`. By default a single LLM serves all four stages; the per-stage flags `--llm-{proofread,identify,structure,tldr}-model` opt into a different model per stage and the pipeline cold-swaps the resident model between stages whose paths differ (see [ADR 0019](adr/0019-per-stage-llm-models.md)). Combined with the in-process design — no HTTP, no daemon — this keeps peak GPU memory bounded by the largest single model, not by their sum.

The only on-disk artefacts the pipeline writes itself are the final Markdown file and an optional per-stage dump tree (`--dump-stages DIR`) used for regression triage. Everything else — temp WAV files, model caches — lives outside the pipeline contract: `~/.cache/huggingface/hub/` for the default LLM (`Qwen2.5-7B-Instruct-4bit`, see [ADR 0020](adr/0020-default-llm-qwen25-7b.md)) and Whisper ASR; `~/.cache/lm-studio/models/` for legacy VibeVoice and any LLM checkpoints populated through LM Studio's GUI; `~/.cache/huggingface/token` for the pyannote gated-repo token; a `tempfile.mkdtemp()` directory for the ffmpeg WAV that is cleaned up on return.

## Pipeline stages

> Numbering convention: this document and the component diagram below count **eleven** boxes (render = `[11]`). The runtime log lines in `pipeline.py` and the stage table in [`pipeline.md`](pipeline.md) count **ten** stages because `render` is pure Python without a `_timed` context — its wall-clock rolls into the total. Both views describe the same pipeline; see [#83](https://github.com/artem-from-ua/voice-transcriber/issues/83) for the tracking issue on unifying the count.

`pipeline.run()` runs eleven steps in order. transcode (ffmpeg → 16 kHz mono WAV) and audiometa (ffprobe) both start from the user's input file; diarize then reads the temp WAV first to produce per-turn boundaries, the **clearspeech** stage uses those boundaries as guard-rails for a chain of DSP effects (autogain, optional bandpass), and speech2text finally consumes the output of the last applied effect. Everything from merge onwards passes structured Python dataclasses around. Only the final two edges (TL;DR string → render → file) are Markdown. The clearspeech chain currently absorbs autogain, bandpass, presence, denoise, and dereverb effects (added incrementally in v0.10–v0.17); the chain is open-ended and new effects can be appended without growing the stage count.

```plantuml
@startuml
title voice-transcriber — pipeline stages
skinparam componentStyle rectangle
skinparam ArrowThickness 2
skinparam legendBorderColor transparent
skinparam legendBackgroundColor #EEEEEE

<style>
  document {
    Margin 25
  }
  actor {
    LineThickness 4
  }
</style>

actor " " as User

component "<b>[1] transcode</b>\n<i><ffmpeg></i>" as WAV #E8E8E8
component "<b>[2] audiometa</b>\n<i><ffprobe></i>" as FF #E8E8E8
component "<b>[3] diarize</b>\n<i><pyannote-audio> speaker-diarization-3.1</i>" as Diar #87CEFA
component "<b>[4] clearspeech</b>\n<i><soundfile> autogain</i>\n<i><scipy> <s>bandpass</s></i>\n<i><scipy> <s>presence</s></i>\n<i><ffmpeg> <s>denoise</s></i>\n<i><scipy> <s>dereverb</s></i>" as CS #E8E8E8
component "<b>[5] speech2text</b>\n<i><mlx-audio> VibeVoice-ASR</i>\n<i><mlx-whisper> Whisper-large-v3</i>" as ASR #90EE90
component "<b>[6] merge</b>" as Merge #E8E8E8
component "<b>[7] proofread</b>\n<i><mlx-lm> Qwen2.5-7B</i>" as Post #FFCC66
component "<b>[8] identify</b>\n<i><mlx-lm> Qwen2.5-7B</i>" as Ident #FFCC66
component "<b>[9] structure</b>\n<i><mlx-lm> Qwen2.5-7B</i>" as Struct #FFCC66
component "<b>[10] tldr</b>\n<i><mlx-lm> Qwen2.5-7B</i>" as TLDR #FFCC66
component "<b>[11] render</b>" as Render #E8E8E8

User -[#FF6B35]-> FF : <color:#404040>  audio file</color>\n<color:#404040>  (wav, m4a, mp3 ...)</color>
User -[#FF6B35]-> WAV : <color:#404040>  audio file</color>\n<color:#404040>  (wav, m4a, mp3 ...)</color>
User -[#3B82F6,dashed]-> Ident : <color:#404040>  unknown speaker ids</color>\n<color:#404040>  (optional)</color>

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
Render -[#6E9E1F]-> User : <color:#404040>  transcript.md</color>

legend top center
  <back:#87CEFA>   </back> <i><color:#404040>AI model: diarization</color></i>
  <back:#90EE90>   </back> <i><color:#404040>AI model: speech2text</color></i>
  <back:#FFCC66>   </back> <i><color:#404040>AI model: LLM</color></i>
  <color:#FF6B35>━━►</color> <i><color:#404040>audio bytes</color></i>
  <color:#3B82F6>━━►</color> <i><color:#404040>metadata</color></i>
  <color:#6E9E1F>━━►</color> <i><color:#404040>transcript text</color></i>
end legend
@enduml
```

![Pipeline stages](https://www.plantuml.com/plantuml/svg/jPRDRjj6483lV8f1T6dHKBOaM_u1GKHs93I03QHswXnI7LRaI5g8t2LsLrQKea3p3a4Vfe-J9-aibo95cIpKGqrRCBKpt_pipEo-qOOfiquJpt2J83nB7g5l51CwKdo62hx_-Gqodc721G99Br1xUiL5nXHB8P9f9WKAysYsfAumCami4jmJ6IabDn-MF5e9r1hwUqi9Bb34DrB5g6vb8XNabccHY2_bMBHQABaMiPFjZFCVpmkqDHvw0B6CrYdfmk_q0d3Fr88Bw9_Jsn_qno93UcxjZkAfl3hB9OAJWkGvoLVqC0s_QbIULuOAhu9P-AatTSv6CiRWP1R-9W8U1lDvck4Y34vucAj-73sHdvVsEGJqfy3MCPSf6hOFo9ISOKMOJDe0Wod4d2d-PSz-jcL2I8D-ZWv1PyXMg7mdo0oNmXzqUoNzBNqFdSkBs_5aTCW_cqAK85F4m6XPsT1s2-QSyaG16BbWN5XUiHZnR1j2eCCP4t76jAQaXathcKAD8iBwUf50An2Za5nZcpxL33wZcktMytXk7zkoTJu55qZVu6TJHPCcdtV9UgA2VyeROFJuK9cq4fib9sqLmaVtZvzGQQ7_F2Wjamfqhax7uwlJGzF3AQH8ujPeBdjltzeSlPW2bO2SAsHntSqa3U6N3Of-zzo_k2ajlvVQG6SokRqT3WzXbrFWb4V3vzkZME-iS1liYb9er3eoQuL7qHvpwJPSxvHQA8dLKQWFTsyVMa7KYpGng23A33_ahsMAFTl2u7yYmF1cS3xrGzjNrn14Tf9STyvExHE2wqYmnHsSv6lMiqEX7pRi-JMaPum-iW5qkzqVTz8DXkm8-3yj3Muk-vFXwvZf9SRMeDl31fDhiH9o8tP3WIf3_vjjcTbnmPBAf6TZgGMNpuy6KxqXh7x-0gaKqYQW9Bm0KCF-1_NSN1bpk_ukGiDJj0EwZDCZ-PfzsxOD03lu1CPkbwhuhJIf3yTNuzx4ghjERW3GIIWNWcPp36xUj4RGYcWFmdCwDK_oAT10iTPXmysov2LiXaa5onLhgDO2okUIRmTJFIYdK6CKhT_cq4-bDuAbUv4LQZLKtlgD-KcuWTaMZCnu19hk7bIlUrcY5gzbkXWABRkbOYuM43E3v5syLlbXMS9o9sfkjTFqEasPsfQwXMHDE-_UfkmKLw_CTDEu2iNTW6oa456kwJu24CpeGdHT7Dwa2ZJzw1iw76cKrhcZTzHBCIRNi7SHsD7jZArmxWWx5bSxKfjmRc8VYxkxkp_45AAkOyFltxxQtx_-tearuDnadMrDLLS7C5SIHy7ixOngWhMGt8OUHQfs5bwao-wjssBl3NtGVVmV)

## Module layout

```
src/voice/
├── cli.py           # argparse, --help, dispatch to pipeline.run()
├── pipeline.py      # PipelineOptions + run(): transcode → audiometa → diarize → clearspeech → speech2text → merge → proofread → identify → structure → tldr → render
├── types.py         # shared dataclasses (AsrSegment, DiarTurn, Segment, Section, StructuredDialog, AudioMeta)
├── transcode.py     # transcode(): ffmpeg → 16 kHz mono PCM WAV
├── audiometa.py     # extract_metadata(): start/end/duration from ffprobe → birthtime → mtime
├── diarize.py       # diarize(): pyannote 3.1, MPS+CPU fallback, exclusive turns
├── clearspeech.py   # clearspeech(): chain-of-DSP-effects (autogain + bandpass + presence + denoise + dereverb)
├── speech2text.py   # transcribe(): mlx-audio wrapper, bitness 4/5/6/8, JSON timeline parse (default ASR backend)
├── whisper_asr.py   # transcribe(): mlx-whisper wrapper for `--asr-engine whisper` (opt-in, see ADR 0017)
├── download_whisper.py # `voice download-whisper` subcommand: prefetch Whisper model into HF cache
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
5. **speech2text** — `speech2text.transcribe(cleaned_wav)` or `whisper_asr.transcribe(cleaned_wav)` → `AsrSegment[]`. The default backend is **Whisper-large-v3-MLX** (mlx-community/whisper-large-v3-mlx via `mlx-whisper`); pass `--asr-engine vibevoice` to switch to the legacy VibeVoice-ASR backend (which emits per-segment `speaker_asr` hints and in-band `[Silence]` / `[Music]` markers). Whisper returns segments with `speaker_asr=None` — speaker labels come entirely from pyannote turns at stage 6. See [ADR 0017](adr/0017-whisper-asr-backend.md) for the empirical reason Whisper is the default.
6. **Merge** — joins (3) and (5) into `Segment[]` (text + pyannote speaker label).
7. **Proofread** — LLM proof-reads `content` per segment in-place.
8. **Identify** — LLM returns `{label → name}`; pipeline assigns `.name` on each segment.
9. **Structure** — LLM produces `StructuredDialog` (segments + section titles). Short dialogues (≤60 segments) go through one LLM call; longer dialogues are split into overlapping ~35-segment chunks, the LLM is called per chunk, and `_reconcile_chunks()` stitches the per-chunk section lists into a single contiguous layout (see [ADR 0018](adr/0018-chunked-structure-dialog.md)).
10. **TL;DR** — LLM emits a Markdown summary string.
11. **Render** — combines `AudioMeta`, `StructuredDialog`, and the TL;DR into the final Markdown file.

See [`pipeline.md`](pipeline.md) for the step-by-step sequence with timing details, [`prompts.md`](prompts.md) for the LLM call contracts powering stages 7–10, and [`adr/README.md`](adr/README.md) for the index of architectural decisions behind these boundaries.
