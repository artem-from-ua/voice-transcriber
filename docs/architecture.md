# Architecture

voice-transcriber is a thin CLI on top of a linear in-process pipeline. The CLI parses arguments, builds a `PipelineOptions`, and hands it to `pipeline.run()`. Each stage is a small module with a single public function; intermediate results live in memory, and the only on-disk artefact (besides the input audio) is the final Markdown file.

External work — ASR inference, diarization, LLM prompting — is delegated to local libraries so the pipeline contains no model code itself: `mlx-whisper` drives the Whisper model directly, pyannote.audio runs the diarization graph on the local GPU/CPU, and `llm.py` loads the LLM in-process via `mlx-lm` (no HTTP).

## High-level architecture

The CLI hands a `PipelineOptions` (whose load-bearing field is the audio path) to `pipeline.run()`. The orchestrator delegates the heavy work to four external libraries — `ffmpeg`/`ffprobe`, `mlx-whisper` (Whisper-large-v3-MLX), `pyannote.audio`, `mlx-lm` — and writes a single Markdown file back to the user.

The split between the CLI and the pipeline is intentional: `cli.py` only resolves user input (argument parsing, defaults, `--help`) and `pipeline.run()` is the single entry point both the CLI and the test suite call. That makes ordering tests possible without a subprocess — `tests/test_pipeline_ordering.py` monkey-patches each module-level reference and asserts the call sequence directly.

Models are never loaded twice. ASR (mlx-whisper), diarization (pyannote 3.1) and the LLM (mlx-lm) each load their weights inside their stage and free them on return; the LLM is then loaded for the four LLM-driven stages (`proofread`, `identify_speakers`, `speech_structure`, `speech_summary`) and unloaded before `render`. By default a single LLM serves all four stages; the per-stage flags `--llm-{proofread,identify,structure,tldr}-model` opt into a different model per stage and the pipeline cold-swaps the resident model between stages whose paths differ (see [ADR 0019](adr/0019-per-stage-llm-models.md)). Combined with the in-process design — no HTTP, no daemon — this keeps peak GPU memory bounded by the largest single model, not by their sum.

The only on-disk artefacts the pipeline writes itself are the final Markdown file and an optional per-stage dump tree (`--dump-stages DIR`) used for regression triage. Everything else — temp WAV files, model caches — lives outside the pipeline contract: `~/.cache/huggingface/hub/` for the default LLM (`Qwen2.5-7B-Instruct-4bit`, see [ADR 0020](adr/0020-default-llm-qwen25-7b.md)) and Whisper ASR; `~/.cache/lm-studio/models/` for any LLM checkpoints populated through LM Studio's GUI (deprecated — LM Studio runtime was removed in [ADR 0006](adr/0006-mlx-lm-over-lm-studio.md); managed-LLM downloads CLI tracked in [#116](https://github.com/artem-from-ua/voice-transcriber/issues/116)); `~/.cache/huggingface/token` for the pyannote gated-repo token; a `tempfile.mkdtemp()` directory for the ffmpeg WAV that is cleaned up on return.

## Pipeline stages

> All three sources — component diagram, runtime log labels, and the stage table in [`pipeline.md`](pipeline.md) — count **thirteen** stages (render = `[13/13]`). `render` has no `_timed` context, so its wall-clock rolls into the pipeline total rather than appearing as a named timing entry. Stage `[4] lang_detect` is conditional: it runs only when `--language` is omitted; the progress label `[4/13]` is emitted whether the stage runs or is skipped.

`pipeline.run()` runs thirteen steps in order. transcode (ffmpeg → 16 kHz mono WAV) and audiometa (ffprobe) both start from the user's input file; diarize then reads the temp WAV first to produce per-turn boundaries, the **clearspeech** stage uses those boundaries as guard-rails for a chain of DSP effects (autogain, optional bandpass), and speech2text finally consumes the output of the last applied effect. Everything from merge onwards passes structured Python dataclasses around. Only the final two edges (TL;DR string → render → file) are Markdown. The clearspeech chain currently absorbs autogain, bandpass, presence, denoise, and dereverb effects (added incrementally in v0.10–v0.17); the chain is open-ended and new effects can be appended without growing the stage count.

The component-level diagram of these thirteen stages — with colour-coded data-flow edges (audio / metadata / transcript text) and library annotations on each box — lives in [`pipeline.md` → `## Stage details`](pipeline.md#stage-details), next to the per-stage purpose/input/action/output table it captions. See that section for the visual reference.

## Module layout

```
src/voice/
├── cli.py           # argparse, --help, dispatch to pipeline.run()
├── pipeline.py      # PipelineOptions + run(): transcode → audio_meta → diarize_speakers → clear_speech → speech2text → merge → proofread → identify_speakers → speech_structure → safe_speech → speech_summary → render
├── types.py         # shared dataclasses (AsrSegment, DiarTurn, Segment, Section, StructuredDialog, AudioMeta)
├── transcode.py     # transcode(): ffmpeg → 16 kHz mono PCM WAV
├── audio_meta.py    # extract_metadata(): start/end/duration from ffprobe → birthtime → mtime
├── diarize_speakers.py  # diarize(): pyannote 3.1, MPS+CPU fallback, exclusive turns
├── clear_speech.py  # clearspeech(): chain-of-DSP-effects (autogain + bandpass + presence + denoise + dereverb)
├── whisper_asr.py   # transcribe(): mlx-whisper wrapper, the ASR backend (see ADR 0017, 0021)
├── lang_detect.py   # detect_language_on_longest_turn(): Whisper-based pre-ASR LID (see ADR 0022)
├── download_whisper.py # `voice download-whisper` subcommand: prefetch Whisper model into HF cache
├── merge.py         # merge(): per-segment max-overlap mapping ASR↔pyannote
├── proofread.py     # fix_asr_errors(): per-segment LLM proof-reader with safety net
├── identify_speakers.py  # identify_speakers(): LLM self-intro detection with override / ask / keep
├── speech_structure.py   # structure_dialog(): LLM-driven section layout with validation
├── safe_speech.py       # redact_dialog(): per-section sensitive-content redaction (see ADR 0023)
├── speech_summary.py   # generate_tldr(): LLM markdown summary
├── render.py        # render_markdown(): final output
├── speaker_emojis.py # fixed palette assignment
└── llm.py           # MlxLLM: in-process mlx-lm wrapper with chat / chat_json (lm-format-enforcer)
```

## Data flow

The pipeline passes increasingly enriched `Segment` lists from stage to stage. Numbers match the labels in the diagram and the log lines.

1. **transcode** — `transcode.transcode(audio, wav)` runs `ffmpeg` and writes a 16 kHz mono PCM WAV to a temp dir.
2. **audio_meta** — `audio_meta.extract_metadata(audio)` → `AudioMeta` (start/end/duration). Goes straight to render.
3. **diarize_speakers** — `diarize_speakers.diarize(wav)` → `DiarTurn[]` (pyannote timeline). Runs on the raw WAV so its boundaries are not influenced by autogain.
4. **lang_detect** (conditional — runs only when `--language` was not given) — `lang_detect.detect_language_on_longest_turn(wav, turns)` picks the longest pyannote turn from (3), slices it from the raw WAV, computes its log-mel spectrogram, and calls `model.detect_language()` on Whisper-large-v3-MLX. The top-1 ISO code becomes `effective_language` and flows through every downstream stage that reads `language=`. When `--language` is given explicitly, this stage is skipped entirely. See [ADR 0022](adr/0022-asr-language-autodetect.md).
5. **clear_speech** — `clear_speech.clearspeech(wav, chain, autogain_turns=turns, ...)` runs an ordered chain of DSP effects. Available effects: `autogain`, `bandpass`, `presence`. Chain is configured by the single `--clearspeech-chain` CLI flag (default `"autogain"`). Each enabled effect writes a sibling WAV (`<stem>.autogain.wav`, `<stem>.autogain.bandpass.wav`, `<stem>.autogain.bandpass.presence.wav`, …) and the next effect reads it; speech2text consumes the last applied effect's output. With `--clearspeech-chain ""` the chain is empty and speech2text receives the raw WAV. See [ADR 0010](adr/0010-clearspeech-chain.md) for the chain-of-effects design, [ADR 0012](adr/0012-clearspeech-chain-string-cli.md) for the single-string CLI, and [ADR 0016](adr/0016-rename-agc-to-autogain.md) for the `agc` → `autogain` rename.
6. **speech2text** — `whisper_asr.transcribe(cleaned_wav, language=effective_language)` → `AsrSegment[]` via `mlx-whisper` against `mlx-community/whisper-large-v3-mlx`. The `effective_language` value is either the CLI `--language` override or the ISO code computed by `[4] lang_detect`. Whisper produces no speaker hint; speaker labels come entirely from pyannote turns at stage 7 (merge). See [ADR 0017](adr/0017-whisper-asr-backend.md) for how this backend was chosen and [ADR 0021](adr/0021-remove-vibevoice-backend.md) for why it is now the only backend.
7. **Merge** — joins (3) and (6) into `Segment[]` (text + pyannote speaker label).
8. **Proofread** — LLM proof-reads `content` per segment in-place.
9. **identify_speakers** — LLM returns `{label → name}`; pipeline assigns `.name` on each segment.
10. **speech_structure** — LLM produces `StructuredDialog` (segments + section titles). Short dialogues (≤60 segments) go through one LLM call; longer dialogues are split into overlapping ~35-segment chunks, the LLM is called per chunk, and `_reconcile_chunks()` stitches the per-chunk section lists into a single contiguous layout (see [ADR 0018](adr/0018-chunked-structure-dialog.md)).
11. **safe_speech** — LLM scans each section for sensitive content (health, drugs, alcohol by default) and replaces flagged utterance ranges with `[muted, X.Xs]` placeholders or drops them, depending on `--safe-speech-policy`. Runs per section so the working set is bounded by the largest section (see [ADR 0023](adr/0023-safe-speech-stage.md)). Stage is skipped when `--no-safe-speech` or `--safe-speech-topics ""`.
12. **speech_summary** — LLM emits a Markdown TL;DR string from the already-redacted `dialog`. Built recursively, one LLM call per section, so no single prompt ever contains the full transcript: per-section TL;DRs at level 0, optional aggregate passes when the section count exceeds `TLDR_FANOUT=7`, then a final canonical-format pass. Skips with a logged warning when `--no-structure` leaves only the synthetic single-section fallback. See [ADR 0030](adr/0030-section-based-tldr.md).
13. **Render** — combines `AudioMeta`, `StructuredDialog`, and the TL;DR into the final Markdown file.

See [`pipeline.md`](pipeline.md) for the step-by-step sequence with timing details, [`prompts.md`](prompts.md) for the LLM call contracts powering stages 8–12, and [`adr/README.md`](adr/README.md) for the index of architectural decisions behind these boundaries.
