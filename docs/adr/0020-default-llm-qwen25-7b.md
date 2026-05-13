---
status: accepted
date: 2026-05-12
supersedes: 0001
---

> **Note (v0.25.0):** modules formerly called `identify`, `structure`, `tldr` are now `identify_speakers`, `speech_structure`, `speech_summary`. Body of this ADR preserves the historical names.

# ADR 0020 — Default LLM is now `Qwen2.5-7B-Instruct-4bit` with model-recommended sampling

## Context

[ADR 0001](0001-local-llm-via-lm-studio.md) (superseded) and [ADR 0006](0006-mlx-lm-over-lm-studio.md) established the in-process MLX runtime. From v0.1 through v0.21 the default LLM was `mlx-community/gemma-3-12b-it-qat-4bit` (~8 GB resident), resolved by a hardcoded absolute path inside the LM Studio cache layout (`~/.cache/lm-studio/models/...`).

Two findings from the v0.21 benchmark (one-off dev-time runs comparing six MLX models; raw artefacts not committed) made that default untenable:

1. **gemma-3-12b does not fit on a 16 GB Mac end-to-end.** Even with chunked `structure_dialog` (ADR 0018), per-stage `free_mlx()` (v0.21), and a hybrid swap to a smaller proofread model, the third chunked structure call hits a Metal OOM. The 16 GB target is the primary developer machine, and shipping a default that crashes on it is a regression no flag can hide.
2. **Sampling defaults from prompt frontmatter (temp 0.1–0.3, no top_p/top_k) were chosen for gemma-3-12b's behaviour on structured output.** They produce different, often worse, results on other models — most starkly on Llama-3.2-3B, whose JSON parser fails entirely at temp 0.2 (trailing commas) but produces valid 4-section layouts at the model's own recommended temp 0.6. Defaults that hard-code one model's tuning are a footgun for users who swap models via `--llm-model`.

The bench identified `mlx-community/Qwen2.5-7B-Instruct-4bit` (~4 GB) as the best all-stages single-model option on 16 GB Macs **when paired with the model's own recommended sampling parameters** (`temperature=0.7`, `top_p=0.8`, `top_k=20`, `repetition_penalty=1.05`, sourced from the official `generation_config.json` on the Qwen Hugging Face repo). With those values it produces:

- 6 thematic sections from a 6-minute Ukrainian recording (vs 3 with our previous default sampling, vs 1 OOM with gemma-12b);
- 274-token TL;DR (vs 1024-token blow-up on gemma-3-1b);
- 34/90 proofread fixes (most aggressive, with `repetition_penalty=1.05` curbing the model's tendency to repeat itself);
- ~3 min total LLM wall-clock on a 6-min recording;
- pipeline reaches the end without OOM.

We also need a way to express "the default model" that does not encode an LM Studio path. The HF cache layout (`~/.cache/huggingface/hub/models--<org>--<repo>/snapshots/<sha>/`) is the natural choice now that we already use it for Whisper.

## Decision

1. **`MlxLLM.DEFAULT_MODEL` is now the Hugging Face repo id `mlx-community/Qwen2.5-7B-Instruct-4bit`** (not a filesystem path).
2. **`MlxLLM` resolves `model_path` flexibly.** A new `_resolve_model_path(spec)` accepts either:
   - A filesystem path (legacy LM Studio cache or any explicit MLX checkpoint directory) — returned as-is after `expanduser`.
   - A Hugging Face `org/repo` id — resolved via `huggingface_hub.try_to_load_from_cache(repo_id, "config.json")`. The snapshot directory hosting that file becomes the path passed to `mlx_lm.load`. If the repo is not in the cache, a `LLMError` is raised with the exact `huggingface-cli download <repo>` command the user can run to fix it. **No implicit network fetches** — same policy as the Whisper backend (ADR 0017).
3. **`MlxLLM.load()` records the resolved snapshot directory in `_resolved_path`.** The pipeline's `_ensure_llm` uses it to decide whether to reuse the resident model across stages: two specs that resolve to the same on-disk directory count as the same model (no spurious cold reload between, say, `--llm-proofread-model mlx-community/Qwen2.5-7B-Instruct-4bit` and `--llm-structure-model /Users/me/.cache/huggingface/.../snapshots/<sha>`).
4. **Default sampling now matches Qwen2.5's recommendation.** `PipelineOptions` defaults:
   - `llm_temperature = 0.7`
   - `llm_top_p = 0.8`
   - `llm_top_k = 20`
   - `llm_repetition_penalty = 1.05`
   These are global overrides applied by `MlxLLM.sampling_overrides`, which take precedence over the per-prompt `temperature`/`max_tokens`/etc. in the prompt frontmatter. The frontmatter values still set `max_tokens` (a task-specific budget) and remain the documented baseline for a "no overrides" run.
5. **CLI flags `--llm-temperature`, `--llm-top-p`, `--llm-top-k`, `--llm-repetition-penalty`** accept `None` (or simply omitting the flag), which keeps the dataclass default. Passing an explicit `0` works as one would expect. Users on a different model should pass that model's recommended values explicitly — there is no per-model auto-tuning.
6. **`pyproject.toml` does not gain a `huggingface-cli` install step.** The package is already a transitive dep via `huggingface_hub`. Users run `huggingface-cli download mlx-community/Qwen2.5-7B-Instruct-4bit` once before their first transcribe; the failure mode is a clear actionable error, not a silent multi-GB fetch.

## Consequences

**Default behaviour changes.** Existing users who relied on the v0.21 default see a different model from v0.22.0 onwards. The CHANGELOG flags this under **Changed (BREAKING for behaviour, not for API)** and gives the one-line migration: pass `--llm-model ~/.cache/lm-studio/models/mlx-community/gemma-3-12b-it-qat-4bit` to keep the v0.21 model. Existing scripts that pass `--llm-model` to a custom path are unaffected.

**One-time download** of ~4 GB into `~/.cache/huggingface/hub/`. The first `voice transcribe` run on a fresh machine fails fast with the exact `huggingface-cli download` command to run. Same pattern as Whisper (ADR 0017). No `voice download-llm` subcommand for now — the `huggingface-cli` invocation is short, and a project-specific wrapper would add another command surface for marginal benefit. If onboarding feedback says otherwise we can add it later without churning the default.

**Sampling parameters now apply globally across stages.** That means `proofread` (which used `temperature=0.1`) is now run with `0.7`. Empirically this **increases** the fix rate (36 fixes / 90 segments vs 25 at temp 0.1 with the same model). Some of those extra fixes are noise — the Levenshtein safety net in `proofread.py` catches the worst false positives, so we accept the trade-off in return for the structure/tldr quality gains. If a future user reports regression on proofread quality, the per-stage temperature override (not yet implemented; documented in [ADR 0019](0019-per-stage-llm-models.md) as a possible follow-up) would be the correct response.

**Other models still work but at default Qwen sampling.** Switching to gemma-3-4b via `--llm-model mlx-community/gemma-3-4b-it-qat-4bit` without also passing `--llm-temperature 1.0 --llm-top-p 0.95 --llm-top-k 64` runs the model at Qwen's recommended params. Empirically this works (the chunked structure validator still produces valid output some of the time), but it is not the model's preferred regime. The `docs/troubleshooting.md` note about per-model sampling explicitly shows the override commands.

**Header rendering exposes the chosen toolchain.** The rendered Markdown now lists `Діаризація: pyannote/...`, `ASR: Whisper-large-v3-MLX`, and `LLM: Qwen2.5-7B-Instruct-4bit` (or `proofread=… · structure=…` when the user split stages). This makes archived transcripts self-documenting and makes it easy for the user to spot which run produced which markdown when comparing options. Together with the new processing-time line (`Обробка: 5m43s · AI-стадії: diarize=25s · asr=53s · proofread=131s · ...`) this turns each transcript into its own benchmark record.

**Resolved-path equality on swap.** `_ensure_llm` now treats `current.model_path == want_path`, `current._resolved_path == want_path`, and `current._resolved_path == resolve(want_path)` as equivalent. This handles all three combos (repo id ↔ repo id, repo id ↔ resolved path, resolved path ↔ resolved path). The cost is one `try_to_load_from_cache` call per `_ensure_llm` invocation when no spec matches directly — negligible.

**Tests gained one regression case** (`test_per_stage_override_does_not_leak_to_unspecified_stages` and `test_per_stage_models_swap_when_different`) and `_StubLLM` now mirrors `_resolved_path`. These keep the swap logic honest under refactoring.

## Alternatives considered

- **Keep gemma-3-12b default, ship `--llm-{stage}-model` workarounds.** The plan we shipped in v0.21. Even with the workarounds documented in troubleshooting, a fresh user on a 16 GB Mac will hit OOM on their first run. Defaults must work out of the box.

- **Default to gemma-3-4b instead of Qwen2.5-7B.** Smaller (3.5 GB vs 4 GB), almost as fast. But empirically gemma-3-4b's chunked structure validator fails on every chunk regardless of sampling params — all three chunks invalid, fallback to a single section. That regresses on the structure feature for the median run. Qwen2.5-7B is the smallest model in the bench that produces multi-section structure reliably.

- **Default to Qwen3-4B-Instruct-2507.** Faster, smaller, and the most generous structure output at our default sampling (7 sections vs Qwen2.5's 6). But our default sampling (chosen here for Qwen2.5) is wrong for Qwen3 — at Qwen3's own recommended temp the section count drops from 7 → 4. Picking Qwen3 as the default would require either accepting that drop or making the defaults model-specific (rejected — too much per-model magic). Qwen2.5-7B is the model whose recommended sampling **also** happens to be a good general-purpose baseline for the others.

- **Auto-detect model family and pick sampling per-family.** Tempting but premature. We have data on 5 models; that is not enough to write a reliable auto-picker, and getting it subtly wrong is worse than asking users to pass three numbers when they swap models. Re-evaluate once the bench covers 15+ models.

- **Make `--llm-model` accept only repo ids (no filesystem paths).** Cleaner API but breaks every existing script that points at an LM Studio extracted checkpoint. The flexible resolver is a one-time complexity tax that respects existing usage.

- **Trigger `snapshot_download` on cache miss instead of failing.** Same argument as ADR 0017 for Whisper: a 4 GB silent download in the middle of `voice transcribe` is a startling failure mode. The explicit `huggingface-cli download` step is one command and surfaces the cost up front.

- **Ship a `voice download-llm <repo>` subcommand mirroring `download-whisper`.** Considered, deferred. Whisper has one canonical repo so the subcommand encodes useful project knowledge (which repo, why this revision). LLMs are more user-choice and the cli would just wrap `huggingface-cli download`. Adding it would be ~30 lines whenever real onboarding feedback asks for it.
