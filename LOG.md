# Build Log

Sequential record of implementation decisions, experiments, commands, and results.

## 2026-09-17 — Repository and hardware inspection

1. Confirmed the repository contained only the planning document.
2. Confirmed the host supports MLX with Metal-accelerated inference.
3. Selected MLX as the native runtime and Python 3.12 through `uv` because the system Python is 3.14.

## 2026-09-17 — Reference implementation review

1. Inspected the complete public file list for [`harshatheg/Qwen-2.5-1B-RLCD`](https://huggingface.co/harshatheg/Qwen-2.5-1B-RLCD).
2. Cloned the Hugging Face repository to a temporary directory and read:
   - `core/engine_mlx.py`
   - `core/schema.py`
   - `core/benchmark.py`
   - `core/prompt_builder.py`
   - `requirements-mlx.txt`
3. Confirmed the public MLX method uses:
   - `mlx-community/Qwen2.5-1.5B-Instruct-4bit`.
   - One compact shared-context prefill.
   - KV-cache broadcasting across schema fields.
   - One batched suffix evaluation for all fields.
   - Allowed-token logit slicing and programmatic JSON assembly.
4. Noted important limitations in the reference:
   - Choice scores are described as calibrated, but normalization over allowed tokens alone does not establish empirical calibration.
   - Multi-token collision handling uses heuristic generation and assigns a minimum reported probability of `0.75`.
   - The benchmark presets do not provide labeled accuracy or calibration evaluation.

## 2026-09-17 — Initial Levels 0–2 implementation

1. Added `pyproject.toml` with only the MLX runtime dependencies.
2. Added `data/benchmark.jsonl` with six labeled support-routing cases and three decisions per case.
3. Added Level 0 metrics:
   - Field accuracy.
   - Schema validity.
   - Mean and p95 latency.
   - Expected calibration error.
4. Added Level 1 autoregressive JSON generation using the same model as Level 2.
5. Added Level 2 parallel constrained decisions using:
   - One shared prefill.
   - Broadcast MLX prompt caches.
   - One batched suffix pass.
   - Single-token internal labels mapped back to arbitrary external choices.
6. Deliberately named outputs `score` rather than `confidence` because calibration has not been demonstrated.
7. Added minimal standard-library unit tests.

## Experiment backlog

Experiments are added only after the faithful baseline runs successfully.

1. **Prior-corrected label scoring:** subtract each label's neutral-prompt logit to reduce systematic preference for particular label tokens.
2. **Permutation-consistency scoring:** rotate the mapping between labels and choices; reduce confidence or abstain when the semantic answer changes with label order.
3. **Entropy-triggered verification:** keep one-pass inference for clear decisions and run a second permuted read only for uncertain fields.
4. **Shared-prefix multi-token trie:** support human-readable multi-token choices without the reference implementation's probability floor heuristic.
5. **Held-out temperature scaling:** convert constrained-choice scores into empirically calibrated probabilities.

Each experiment must record its accuracy, p95 latency, calibration error, and comparison with the unchanged Level 2 baseline before it is retained.

## 2026-09-17 — First Level 2 benchmark

1. Installed the Python 3.12 environment and 34 packages with `uv`.
2. Unit tests passed: 2/2.
3. Downloaded and warmed `mlx-community/Qwen2.5-1.5B-Instruct-4bit`.
4. Initial parallel results across 18 labeled field decisions:
   - Accuracy: `55.56%`.
   - Schema validity: `100%`.
   - Mean latency: `149.68 ms`, including a `533.18 ms` first compilation case.
   - Steady-state latency: `61.7–91.2 ms` for three fields.
   - ECE: `0.4422`.
5. Prediction inspection showed a strong preference for internal label `B`, especially for department and sentiment. The boolean field remained accurate because its two-label distinction was stronger.
6. Conclusion: the parallel inference path and target latency work, but raw fixed-label scores are not yet an acceptable decision method.

## 2026-09-17 — Experiment 1: permutation-consistent readout

1. Added cyclic label-to-choice permutations for every field.
2. All permutations share the same context prefill and remain rows in one batched suffix pass, so the method adds no sequential decoding steps.
3. Semantic choice probabilities are averaged after undoing each permutation.
4. Added cross-permutation winner agreement to each field result.
5. Kept the original single-mapping mode unchanged for an honest comparison.

### Result

- Accuracy: `22.22%`, down from `55.56%`.
- ECE: `0.3576`, down from `0.4422`, but not useful given the accuracy loss.
- Steady-state latency: `67.4–83.7 ms`.
- Decision: rejected and removed from the runtime. The experiment remains documented here.

## 2026-09-17 — Faithful direct-choice token correction

1. Inspected tokenization for every benchmark choice.
2. Confirmed all choices become exactly one Qwen token when prefixed with a space, including `infrastructure`, `calm`, `concerned`, and `angry`.
3. Removed the A/B/C indirection and now score the actual allowed choice tokens, matching the fast path in Harsha's published implementation.

### Result

- Accuracy: `72.22%`, up from `55.56%`.
- Schema validity: `100%`.
- ECE: `0.2246`, improved from `0.4422`.
- Steady-state latency: `67.0–112.3 ms` for three fields.
- Decision: retained as the Level 2 baseline.

## 2026-09-17 — Level 1 versus Level 2 comparison

### Level 1: autoregressive JSON

- Accuracy: `83.33%`.
- Schema validity: `100%` on the six initial cases.
- Mean latency: `203.38 ms`.
- Steady-state latency after first compilation: `175.8–220.2 ms`.

### Level 2: direct constrained-choice readout

- Accuracy: `72.22%`.
- Schema validity: `100%` by programmatic assembly.
- Mean latency: `127.02 ms`.
- Steady-state latency after first compilation: `67.0–112.3 ms`.

### Interpretation

1. Level 2 is approximately `1.6×` faster including each mode's first compiled case.
2. Excluding the first compiled case, mean Level 2 latency is approximately `87.9 ms` versus `186.3 ms` for Level 1, or about `2.1×` faster.
3. The current three-field output is too short to reproduce the reference's advertised `5×` latency gain; its advantage should grow with more fields because autoregressive output length grows while the Level 2 suffixes remain batched.
4. The six synthetic cases are a smoke test, not evidence of production accuracy. The next data task is expanding to at least 200 real or carefully reviewed examples.

## 2026-09-17 — Cross-model benchmark preparation

1. Verified the requested Hugging Face models and their MLX conversions:
   - `mlx-community/gemma-3-270m-it-bf16`.
   - `mlx-community/LFM2.5-VL-1.6B-bf16`.
2. Found that the original implementation still embedded Qwen-specific chat control tokens in both inference modes.
3. Replaced those control tokens with each tokenizer's native `apply_chat_template` output so comparisons do not penalize non-Qwen models with an invalid prompt format.
4. Added `--model` to the benchmark CLI.

## 2026-09-17 — Expanded 4-bit model matrix

1. Switched the requested 1–2B comparison models to 4-bit MLX checkpoints.
2. Selected:
   - `mlx-community/LFM2.5-VL-1.6B-4bit`.
   - `Qwen/Qwen3-1.7B-MLX-4bit`.
   - `SirSahOl/K2-Horizon-0.9B-chat-mlx-4bit` (third-party conversion).
3. No MLX conversion exists for `XHToken/Spark-X2.5-1.7B`; its custom `spark2_5` architecture cannot be loaded by the current MLX engine.
4. The LFM checkpoint downloaded successfully but `mlx-lm` rejected its `lfm2_vl` architecture. Its model card specifies `mlx-vlm`, so that runtime is being evaluated separately.
5. Added `mlx-vlm` support by using the model's text backbone and broadcasting both attention KV caches and recurrent array caches.
6. LFM tokenization makes several semantic choices multi-token. Added a hybrid readout: direct choice-token scoring where possible and A/B/C indirection only for fields whose choices cannot be represented by one token.
7. Qwen3 initially produced no valid Level 1 JSON because its template enabled reasoning and exhausted the generation limit. Disabled thinking through the tokenizer's native `enable_thinking=False` template argument for a fair structured-output comparison.
8. The template flag alone did not disable reasoning for this Qwen3 checkpoint; added its documented `/no_think` prompt switch.
9. The K2 third-party MLX checkpoint still requires unsupported custom `k2_horizon` model code and interactive remote-code trust. Rejected it from the reproducible benchmark instead of executing unreviewed repository code.

## 2026-09-17 — Cross-model results

The dataset contains only six synthetic cases and 18 field decisions, so these are engineering smoke-test results.

| Model | Precision | Mode | Accuracy | Validity | Mean latency | p95 latency | ECE |
|---|---|---|---:|---:|---:|---:|---:|
| Qwen2.5 1.5B Instruct | 4-bit | Parallel | 66.67% | 100% | 58.53 ms | 61.85 ms | 0.1911 |
| Gemma 3 270M IT | BF16 | Autoregressive | 44.44% | 100% | 476.90 ms | 1825.38 ms | 0.5556 |
| Gemma 3 270M IT | BF16 | Parallel | 72.22% | 100% | 29.62 ms | 111.82 ms | 0.2195 |
| LFM2.5-VL 1.6B | 4-bit | Autoregressive | 72.22% | 100% | 178.13 ms | 234.12 ms | 0.2778 |
| LFM2.5-VL 1.6B | 4-bit | Parallel | 55.56% | 100% | 86.27 ms | 104.42 ms | 0.2525 |
| Qwen3 1.7B | 4-bit | Autoregressive | 77.78% | 100% | 287.26 ms | 334.57 ms | 0.2222 |
| Qwen3 1.7B | 4-bit | Parallel | 55.56% | 100% | 121.50 ms | 276.62 ms | 0.3761 |

Notes:

1. Gemma's first calls include compilation outliers; its steady-state parallel latency was `10.4–17.2 ms`.
2. LFM required `mlx-vlm` and hybrid direct/label readouts because some choices are multi-token.
3. Qwen3 requires `/no_think`; results before that correction were discarded.
4. K2 and Spark were not benchmarked because neither has a reproducible, supported MLX 4-bit path in the current runtime.
5. Final unit-test regression passed: 2/2.

## 2026-09-17 — Qwen3 0.6B 8-bit benchmark

Tested `Qwen/Qwen3-0.6B-MLX-8bit` with `/no_think` enabled.

| Mode | Accuracy | Validity | Mean latency | p95 latency | Steady-state latency | ECE |
|---|---:|---:|---:|---:|---:|---:|
| Autoregressive | 72.22% | 100% | 152.66 ms | 241.27 ms | 133.0–137.8 ms | 0.2778 |
| Parallel | 61.11% | 100% | 30.28 ms | 42.75 ms | 26.9–30.5 ms | 0.3446 |

The parallel path is approximately `5×` faster by mean latency and `4.8×` faster after warmup, at an `11.11` percentage-point accuracy cost on the six-case smoke dataset.

## 2026-09-17 — LFM2.5-VL 450M BF16 benchmark

Tested the official `LiquidAI/LFM2.5-VL-450M-MLX-bf16` checkpoint through its MLX-VLM language backbone.

| Mode | Accuracy | Validity | Mean latency | p95 latency | Steady-state latency | ECE |
|---|---:|---:|---:|---:|---:|---:|
| Autoregressive | 66.67% | 100% | 157.58 ms | 352.56 ms | 113.9–124.9 ms | 0.3333 |
| Parallel | 61.11% | 100% | 16.98 ms | 32.78 ms | 13.6–14.3 ms | 0.3711 |

The parallel path is approximately `9.3×` faster by mean latency and about `8.5×` faster after warmup, at a `5.56` percentage-point accuracy cost on the smoke dataset.

## 2026-09-17 — Austin Huang method review and source attribution

1. Reviewed Austin Huang's post reporting a trained tiny Jev-like model.
2. The author explicitly distinguishes it from both a classifier and parallel sampling from a prefix.
3. The post does not disclose the architecture, objective, dataset, source code, weights, or evaluation, so technical novelty cannot currently be verified or reproduced.
4. Classified it as a potentially distinct trained-model direction rather than a new implementation method available to this project.
5. Added every user-shared post and article to the README, plus the Hugging Face implementation and OpenJev sources used directly during development.

## 2026-09-17 — Side-by-side demo video

1. Created a 12-second, 1280×720 comparison of autoregressive JSON generation and parallel constrained readout.
2. Used measured LFM2.5-VL 450M BF16 mean latency: `157.6 ms` versus `17.0 ms`, or `9.3×`.
3. Slowed the visual animation deliberately because the real inference timings are too short to perceive.
4. Added a dependency-free project renderer using the already-installed Pillow and FFmpeg packages.
5. Visually inspected the final frame and verified the MP4 duration and encoding.

## 2026-09-17 — 20-case, six-model comparison

1. Expanded the benchmark from 6 to 20 hand-labeled cases and from 18 to 60 field decisions.
2. Added security, email, commerce-fraud, and content-moderation tasks alongside support routing.
3. Added a dataset regression check that validates the case count, schema fields, and every expected label.
4. Added one unmeasured warm-up inference per mode so compilation is excluded from latency comparisons.
5. Added `benchmark_all.py`, which runs the six supported cached checkpoints sequentially and saves JSON and Markdown results under `results/`.
6. Ran the full matrix successfully; all parallel outputs were schema-valid.

| Model | AR accuracy | Parallel accuracy | AR validity | Parallel validity | AR mean | Parallel mean | Speedup |
|---|---:|---:|---:|---:|---:|---:|---:|
| Qwen2.5 1.5B 4-bit | 70.0% | 60.0% | 100% | 100% | 192.5 ms | 62.2 ms | 3.09× |
| Gemma 3 270M BF16 | 20.0% | 48.3% | 30% | 100% | 98.7 ms | 15.3 ms | 6.46× |
| LFM2.5-VL 1.6B 4-bit | 66.7% | 58.3% | 100% | 100% | 126.7 ms | 46.5 ms | 2.72× |
| Qwen3 1.7B 4-bit | 75.0% | 56.7% | 100% | 100% | 199.8 ms | 64.3 ms | 3.11× |
| Qwen3 0.6B 8-bit | 55.0% | 53.3% | 100% | 100% | 131.7 ms | 31.3 ms | 4.20× |
| LFM2.5-VL 450M BF16 | 53.3% | 51.7% | 100% | 100% | 116.9 ms | 13.7 ms | 8.51× |

The best parallel accuracy was Qwen2.5 1.5B at 60.0%. LFM2.5-VL 450M was the fastest at 13.7 ms and delivered the largest mean speedup, 8.51×, with a 1.66 percentage-point accuracy loss. Gemma exposed a major benefit of constrained assembly: its autoregressive output was schema-valid on only 30% of cases, while the parallel path remained 100% valid.

## 2026-09-17 — Llama 3.2 1B Instruct BF16 benchmark

1. Added the official MLX BF16 conversion of `meta-llama/Llama-3.2-1B-Instruct` to the reproducible model matrix.
2. Ran the same warmed 20-case, 60-field benchmark.

| Mode | Accuracy | Validity | Mean latency | p95 latency | ECE |
|---|---:|---:|---:|---:|---:|
| Autoregressive | 60.0% | 95% | 477.95 ms | 540.01 ms | 0.4000 |
| Parallel | 43.33% | 100% | 34.02 ms | 37.36 ms | 0.2715 |

The parallel path was `14.05×` faster by mean and `14.45×` by p95, but lost `16.67` accuracy points. This is the largest measured speedup in the matrix, but the weakest parallel accuracy; Llama is therefore not the best current quality/speed tradeoff without task-specific tuning.

## 2026-09-17 — Trained OpenJev NLI method review

1. Reviewed `AlexWortega/openjev`, a Qwen3.5-4B cross-encoder trained with three-way NLI classification.
2. Classified it as a separate trained semantic-decision method: candidate choices become hypotheses and are ranked by entailment probability.
3. Added the model card to the README references with attribution to Alex Wortega.
4. Added the method to `JEV_REPLICATION_PLAN.md` as Level 3, including its premise–hypothesis formulation, implementation steps, tradeoffs, exit condition, and hybrid shared-context experiment.

## 2026-09-17 — Progressive experimental-ideas roadmap

1. Created `exp_ideas.md` with thirteen implementation levels ordered from evaluation and training-free inference changes through custom System-One research.
2. Included calibration, verbalizer ensembles, semantic suffix scoring, adaptive cascades, retrieval, joint constraints, OpenJev, hidden-state prototypes, early exits, dynamic heads, multi-slot outputs, RLCD, and diffusion.
3. Added a typed decision compiler as the unifying runtime design.
4. Credited the original synthesis and proposed combinations to `gpt-5.6-sol` while preserving attribution for all referenced external methods.

## 2026-09-17 — Benchmark-scope clarification

1. Documented that all current benchmark cases are System-One-shaped Boolean or enum decision tasks rather than generative tasks.
2. Clarified that the 20-case benchmark is not a reproduction of TypeSafe's workflow evals because it lacks calibrated reference distributions, high-cardinality and numeric outputs, larger schemas, structured program state, and probability-dependent multi-step workflows.

## 2026-09-17 — Direct Level 2 decision training plan

1. Confirmed the replication plan mentioned generic LoRA and distillation but did not explicitly describe training the existing parallel allowed-choice readout.
2. Added direct Level 2 training as the preferred first Level 6 experiment: allowed-choice cross-entropy, soft teacher distributions, consistency loss, explicit unknown cases, and post-training calibration.
3. Preserved the shared-prefill inference architecture so training does not add autoregressive JSON generation.
4. Structured the RLCD-style extension as an optional post-supervised stage and clarified that it still requires held-out calibration evaluation.

## 2026-09-17 — Decode throughput measurement

1. Added `prefill_ms`, `decode_ms`, and `tokens_per_s` to the autoregressive result. The decode rate is measured over the token loop only, so prefill and time-to-first-token are excluded.
2. Added `equivalent_tokens` to the parallel result. The parallel path emits no JSON tokens, so its throughput is reported as the encoded size of the assembled object divided by the suffix pass.
3. Extended `benchmark.py` to aggregate `mean_tokens` and `tokens_per_s` per mode.
4. Measured `mlx-community/Qwen3-1.7B-bf16` over the 20-case benchmark: 24.3 generated tokens per case at 39.9–47.5 tok/s autoregressive, versus 20.8 equivalent tokens at 498.9 tokens/s parallel.
5. Ran under heavy external CPU load (load average 11.5, an unrelated 680% CPU Python job). Absolute latencies were roughly double the earlier `run-4.json` numbers for identical accuracy and ECE, so the decode rates above are a lower bound. Re-measure on an idle host before quoting them.

## 2026-09-18 — Qwen3.5-0.8B BF16 benchmark

1. Benchmarked `Qwen/Qwen3.5-0.8B`, the native BF16 transformers repository, without conversion or quantization.
2. mlx-lm 0.31.3 ships a native `qwen3_5` implementation, so the model loads through mlx-lm as `mlx_lm.models.qwen3_5.Model`. The mlx-vlm fallback was not used even though the repository is published as image-text-to-text with a `Qwen3_5ForConditionalGeneration` architecture.
3. The text tower uses hybrid attention: three linear-attention layers per full-attention layer, 24 layers, 1024 hidden size, tied embeddings.
4. Results: autoregressive 66.67% at 245.19 ms and 140.0 tok/s; parallel 60.00% at 29.85 ms and 2132.6 equivalent tok/s; 8.21x mean speedup.
5. Parallel ECE is 0.1206, the lowest measured so far and roughly a third of the Qwen3-1.7B value, so the parallel readout is better ordered on this model despite lower accuracy.
6. Added the model to `benchmark_all.py` and added decode-rate columns to the generated comparison table. The committed comparison table is stale until the full suite is re-run.

## 2026-09-18 — Qwen3.5-2B BF16 benchmark

1. Benchmarked `Qwen/Qwen3.5-2B`, also native BF16 with no quantization, loading through the same `mlx_lm.models.qwen3_5.Model` path. 24 layers, 2048 hidden size, 4.3 GB on disk.
2. Results: autoregressive 81.67% at 321.65 ms, 68.3 tok/s, ECE 0.1833; parallel 63.33% at 56.03 ms, 1128.6 equivalent tok/s, ECE 0.1354; 5.74x mean speedup.
3. The autoregressive result is the strongest measured so far, above Qwen3-1.7B BF16 at 71.67% and Qwen3.5-0.8B at 66.67%.
4. The parallel readout does not track the autoregressive gain: 63.33% against 81.67%, an 18.3 point gap, compared with a 6.7 point gap on Qwen3.5-0.8B. As this family scales, allowed-choice argmax accuracy improves far more slowly than generation accuracy.
5. Decode rate falls from 140.0 tok/s at 0.8B to 68.3 tok/s at 2B, roughly the 2.5x parameter scaling, so the autoregressive path stays memory-bandwidth bound.
6. Added the model to `benchmark_all.py`.

## 2026-09-18 — Live single-case demo

1. Added `demo/live.py`, which runs one benchmark case through both real paths, prints the transcript, and can replay it into an MP4. Unlike `comparison.mp4`, this is not a hand-drawn animation: every number and the JSON itself come from an actual run.
2. Defaulted to case 4, where the parallel readout scores 3/3 and the autoregressive path scores 2/3 on Qwen3.5-2B BF16, at roughly 6x lower latency. Case selection was made after measuring all 20 cases for both paths.
3. Case 4 is the clearest contrast of the 20. Across the whole set the autoregressive path was ahead more often than not, so the demo should not be read as an aggregate accuracy claim.
4. Sourced the framing from Harsha Gundala's original post announcing Qwen-2.5-1B-RLCD with an on-device video demo, and kept it hardware-neutral.
5. Added a unit test covering the transcript match count and speedup arithmetic, which needs no model.

## 2026-09-18 — Repository restructure

1. Moved the flat top-level modules into packages. `jev_like.py` split into `jev_like/runtime.py` for inference and `jev_like/benchmark.py` for the dataset, metrics, evaluation, and CLI; the evaluation metric no longer lives in the model module.
2. Folded `benchmark_all.py` into `jev_like.benchmark --all`, so the suite list, the per-model run loop, and the comparison table live with the harness instead of in a second script. Each model still runs in its own process so checkpoints do not stack in memory.
3. Removed the mutable `MODEL_ID` global. `get_engine` now caches one engine per model id and every call takes the model explicitly, so a process can switch models without reassigning module state.
4. Added `demo/video.py` holding the palette, font helpers, and the FFmpeg writer. `demo/render.py` and `demo/live.py` previously duplicated all of that, including the ffmpeg command line.
5. Removed the `sys.path` hack in `demo/live.py`. Both demos now run as modules from the repository root, which puts the root on the path naturally.
6. Moved tests to `tests/` and dropped the importlib file-loading workaround for the demo script. Dataset paths now resolve from the module location rather than the working directory.
7. Verified the restructure is behavior-preserving: Qwen3.5-0.8B reproduces `run-9.json` exactly on accuracy, schema validity, ECE, and mean token counts, and `comparison.mp4` re-renders to the same byte size.
8. Left in place: `results/run-N.json` is still numbered by position in the suite list, so it is not a stable identifier. Each file carries a `model` field, and `model_comparison.json` is the canonical record.
