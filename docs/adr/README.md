# Architecture Decision Records

Short, dated records of choices that shape how the pipeline works. Format: Context → Decision → Consequences → Alternatives considered.

| #   | Title                                                                                  | Status                       |
| --- | -------------------------------------------------------------------------------------- | ---------------------------- |
| ~~1~~   | ~~[Use LM Studio as the local LLM runtime](0001-local-llm-via-lm-studio.md)~~          | superseded by 0006 and 0020  |
| 2   | [Prefer MLX format over GGUF for local models](0002-mlx-format-preference.md)          | accepted                     |
| 3   | [Detect self-introductions with an LLM, not regexes](0003-self-intro-via-llm-not-regex.md) | accepted                 |
| 4   | [Use pyannote 3.1 for diarization](0004-pyannote-for-diarization.md)                   | accepted                     |
| 5   | [Proof-read ASR per segment, not over the whole transcript](0005-segment-level-asr-proofread.md) | accepted           |
| 6   | [Run the LLM in-process via mlx-lm](0006-mlx-lm-over-lm-studio.md)                     | accepted (supersedes 0001)   |
| 7   | [Defaults for the loudness-normalize stage](0007-loudness-normalize-defaults.md)       | accepted                     |
| 8   | [Keep proofread before identify; do not reorder](0008-keep-proofread-before-identify.md) | accepted                   |
| 9   | [Single diarize pass on the raw WAV](0009-single-diarize-pass.md)                      | accepted                     |
| 10  | [Audio cleanup as a chain-of-effects stage](0010-clearspeech-chain.md)                 | accepted                     |
| 11  | [Defaults for the clearspeech bandpass effect](0011-clearspeech-bandpass-defaults.md)  | accepted                     |
| 12  | [Single `--clearspeech-chain` string replaces per-effect toggles](0012-clearspeech-chain-string-cli.md) | accepted     |
| 13  | [Defaults for the clearspeech presence effect](0013-clearspeech-presence-defaults.md)  | accepted                     |
| 14  | [Defaults for the clearspeech denoise effect](0014-clearspeech-denoise.md)             | accepted                     |
| 15  | [Defaults for the clearspeech dereverb effect](0015-clearspeech-dereverb.md)           | accepted                     |
| 16  | [Rename clearspeech effect `agc` to `autogain`](0016-rename-agc-to-autogain.md)        | accepted                     |
| ~~17~~  | ~~[Whisper-large-v3-MLX as a second ASR backend behind `--asr-engine`](0017-whisper-asr-backend.md)~~ | superseded by 0021     |
| 18  | [Chunk the structure_dialog prompt for hour-long recordings](0018-chunked-structure-dialog.md) | accepted                     |
| 19  | [Allow a different LLM model per pipeline stage](0019-per-stage-llm-models.md)         | accepted                     |
| 20  | [Default LLM is now `Qwen2.5-7B-Instruct-4bit` with model-recommended sampling](0020-default-llm-qwen25-7b.md) | accepted (supersedes 0001) |
| 21  | [Remove the VibeVoice ASR backend](0021-remove-vibevoice-backend.md)                  | accepted                     |
| 22  | [`--language` defaults to detection on the longest pyannote turn](0022-asr-language-autodetect.md) | accepted                     |
| 23  | [`safe_speech` stage for sensitive-content redaction](0023-safe-speech-stage.md)       | accepted                     |
| ~~24~~  | ~~[Defaults for the `safe_speech` stage](0024-safe-speech-defaults.md)~~               | accepted (render display superseded by 0025) |
| 25  | [Silence events: unified pause/muted rendering with timestamp ranges](0025-render-silence-events.md) | accepted (supersedes 0024 render display) |
| ~~26~~  | ~~[Proofread stage: default off, opt-in via `--proofread`](0026-proofread-default-off.md)~~ | accepted (superseded by 0028) |
| 27  | [`prompt_cache_session()`: amortise the system-prompt KV across LLM-stage loops](0027-prompt-cache-llm-stages.md) | accepted                     |
| 28  | [Proofread stage: default on again after the iteration-2.1 rework](0028-proofread-default-on-after-rework.md) | accepted (supersedes 0026)   |
| 29  | [Issue label taxonomy: 4 axes (type, priority, stage, area)](0029-issue-label-taxonomy.md) | accepted                     |
| 30  | [Section-based TL;DR with recursive aggregation](0030-section-based-tldr.md) | accepted                     |
| 31  | [Chunk Whisper ASR in Python for long recordings](0031-chunked-asr.md) | accepted                     |
| 32  | [Dedicated chunked-mode prompt + snap-to-speaker-boundary for `speech_structure`](0032-structure-chunk-prompt-and-snap.md) | accepted (extends 0018) |
| 33  | [`--user-context` injected as a prefix to every LLM stage's system prompt](0033-user-context-system-prompt-prefix.md) | accepted                     |

New decisions land here as `NNNN-kebab-case-title.md` with the same frontmatter (`status`, `date`, optional `supersedes` / `superseded_by`).
