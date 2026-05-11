# Architecture Decision Records

Short, dated records of choices that shape how the pipeline works. Format: Context → Decision → Consequences → Alternatives considered.

| #   | Title                                                                                  | Status                       |
| --- | -------------------------------------------------------------------------------------- | ---------------------------- |
| 1   | [Use LM Studio as the local LLM runtime](0001-local-llm-via-lm-studio.md)              | superseded by 0006           |
| 2   | [Prefer MLX format over GGUF for local models](0002-mlx-format-preference.md)          | accepted                     |
| 3   | [Detect self-introductions with an LLM, not regexes](0003-self-intro-via-llm-not-regex.md) | accepted                 |
| 4   | [Use pyannote 3.1 for diarization](0004-pyannote-for-diarization.md)                   | accepted                     |
| 5   | [Proof-read ASR per segment, not over the whole transcript](0005-segment-level-asr-proofread.md) | accepted           |
| 6   | [Run the LLM in-process via mlx-lm](0006-mlx-lm-over-lm-studio.md)                     | accepted (supersedes 0001)   |
| 7   | [Defaults for the loudness-normalize stage](0007-loudness-normalize-defaults.md)       | accepted                     |
| 8   | [Keep proofread before identify; do not reorder](0008-keep-proofread-before-identify.md) | accepted                   |
| 9   | [Single diarize pass on the raw WAV](0009-single-diarize-pass.md)                      | accepted                     |
| 10  | [Audio cleanup as a chain-of-effects stage](0010-clearspeech-chain.md)                 | accepted                     |

New decisions land here as `NNNN-kebab-case-title.md` with the same frontmatter (`status`, `date`, optional `supersedes` / `superseded_by`).
