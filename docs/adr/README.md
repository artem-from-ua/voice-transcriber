# Architecture Decision Records

Short, dated records of choices that shape how the pipeline works. Format: Context → Decision → Consequences → Alternatives considered.

| #   | Title                                                                                  | Status                       |
| --- | -------------------------------------------------------------------------------------- | ---------------------------- |
| 1   | [Use LM Studio as the local LLM runtime](0001-local-llm-via-lm-studio.md)              | superseded by 0006 and 0020  |
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
| 17  | [Whisper-large-v3-MLX as a second ASR backend behind `--asr-engine`](0017-whisper-asr-backend.md) | accepted             |
| 18  | [Chunk the structure_dialog prompt for hour-long recordings](0018-chunked-structure-dialog.md) | accepted                     |
| 19  | [Allow a different LLM model per pipeline stage](0019-per-stage-llm-models.md)         | accepted                     |
| 20  | [Default LLM is now `Qwen2.5-7B-Instruct-4bit` with model-recommended sampling](0020-default-llm-qwen25-7b.md) | accepted (supersedes 0001) |

New decisions land here as `NNNN-kebab-case-title.md` with the same frontmatter (`status`, `date`, optional `supersedes` / `superseded_by`).
