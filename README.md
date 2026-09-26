# Jev-Like Levels 0–2

A minimal experiment reproducing the public core of Harsha Gundala's approach:

- **Level 0:** labeled JSONL dataset plus accuracy, validity, latency, and calibration metrics.
- **Level 1:** ordinary autoregressive JSON generation.
- **Level 2:** one shared context prefill followed by one batched suffix pass that scores every field's allowed labels.

The Level 2 `score` is conditional on the allowed choices. It is not calibrated confidence.

## Layout

```text
jev_like/
  runtime.py     model loading plus the autoregressive and parallel inference paths
  benchmark.py   dataset, metrics, per-mode evaluation, and the CLI
demo/
  video.py       shared palette, fonts, and the FFmpeg writer
  render.py      the comparison animation
  live.py        a real single-case run, printed and optionally replayed
tests/           model-free unit tests
data/            the labeled decision cases
results/         measured runs, one JSON file per checkpoint
```

Inference and evaluation are deliberately separate: `runtime.py` returns measurements but takes no view on how they should be scored, and every metric lives in `benchmark.py`.

## Demo

[Live single-case demo](demo/live.mp4) — one real ticket run through both paths on Qwen3.5-2B BF16, the strongest checkpoint measured here. The parallel readout scores 3/3 against the autoregressive path's 2/3, at roughly 6x lower latency. The numbers and the JSON come from an actual run; only the replay pacing is artificial.

```bash
uv run --python 3.12 python -m demo.live          # real run, prints the transcript
uv run --python 3.12 python -m demo.live --video  # also writes demo/live.mp4
```

An earlier [side-by-side comparison animation](demo/comparison.mp4) illustrates the same idea against Qwen3-1.7B BF16 with deliberately slowed pacing. Re-render it with `sh demo/render.sh`.

## Run

`uv sync` picks the right MLX wheel for the machine: Metal on macOS, the CUDA 12 backend on Linux with an NVIDIA GPU (plain `mlx` ships no Linux backend). On a CPU-only Linux box install the CPU backend instead: `uv pip install "mlx[cpu]"` — it runs the same code but slowly, fine for smoke tests, not for the benchmark. Startup reports the detected backend — `cuda`, `metal`, or `cpu` — and fails with install instructions when none is usable. macOS and Linux only; MLX has no Windows build.

```bash
uv sync --python 3.12
uv run --python 3.12 python -m unittest discover -s tests -t .
uv run --python 3.12 python -m jev_like.benchmark --mode parallel
uv run --python 3.12 python -m jev_like.benchmark --mode both --model Qwen/Qwen3.5-2B
uv run --python 3.12 python -m jev_like.benchmark --all
```

`--all` runs every checkpoint in `jev_like.benchmark.MODELS` in its own process, so models never stack in memory, and writes [the full JSON results](results/model_comparison.json) plus [a Markdown comparison](results/model_comparison.md). Each mode gets an unmeasured warm-up inference before its 20 measured cases.

## Current 20-case result

Every checkpoint above is BF16 and measured in a single `--all` pass, so all nine saw the same machine conditions. The generated table is [here](results/model_comparison.md); this is a copy of it.

| Model | AR acc. | Parallel acc. | AR tok/s | Parallel tok/s | AR mean | Parallel mean | Speedup |
|---|---:|---:|---:|---:|---:|---:|---:|
| mlx-community/Qwen2.5-1.5B-Instruct-bf16 | 70.0% | 56.7% | 78 | 1367 | 396.1 ms | 44.4 ms | 8.93x |
| mlx-community/gemma-3-270m-it-bf16 | 20.0% | 48.3% | 295 | 4977 | 96.8 ms | 12.8 ms | 7.57x |
| mlx-community/LFM2.5-VL-1.6B-bf16 | 71.7% | 56.7% | 114 | 2280 | 273.5 ms | 31.8 ms | 8.59x |
| mlx-community/Qwen3-1.7B-bf16 | 71.7% | 60.0% | 75 | 1238 | 352.9 ms | 46.1 ms | 7.65x |
| mlx-community/Qwen3-0.6B-bf16 | 53.3% | 50.0% | 164 | 2476 | 187.8 ms | 22.4 ms | 8.36x |
| LiquidAI/LFM2.5-VL-450M-MLX-bf16 | 53.3% | 51.7% | 284 | 5600 | 115.7 ms | 13.8 ms | 8.37x |
| mlx-community/Llama-3.2-1B-Instruct-bf16 | 58.3% | 46.7% | 103 | 1866 | 458.4 ms | 33.6 ms | 13.64x |
| Qwen/Qwen3.5-0.8B | 66.7% | 60.0% | 136 | 2192 | 235.1 ms | 29.4 ms | 8.01x |
| Qwen/Qwen3.5-2B | 81.7% | 63.3% | 67 | 1142 | 325.9 ms | 55.3 ms | 5.89x |

Qwen3.5-2B leads both paths at 81.7% autoregressive and 63.3% parallel. Llama 3.2 1B shows the widest latency gap at 13.6x, and LFM2.5-VL 450M has the fastest parallel readout at 13.8 ms. Note that the parallel readout does not track autoregressive gains as models scale: the 0.8B to 2B jump adds 15 points of autoregressive accuracy but only 3.3 points parallel.

These 20 hand-labeled synthetic cases (60 field decisions) are a broader engineering benchmark, not a production-quality eval. Replace `data/benchmark.jsonl` with real domain data before drawing deployment conclusions.

## Benchmark scope

The benchmark intentionally contains only **System-One-shaped decision tasks**: unstructured text goes in and predefined Boolean or enum fields come out. Its support-routing, security, email, fraud, and moderation cases test classification, routing, and branching decisions rather than chat, essays, code, or free-form reasoning. This matches TypeSafe AI's description of Jev as “unstructured state in, typed probabilistic decisions out.”

It does **not** yet reproduce TypeSafe's full workflow evaluations. The current benchmark uses hard labels and three fields per case; its choice scores are conditional but uncalibrated. A closer evaluation should add held-out reference probability distributions, numeric scoring and extraction, explicit abstention, 10–30 simultaneous decisions, high-cardinality choices, structured program state, probability-dependent multi-step workflows, and end-to-end workflow latency and quality.

## Attribution and references

This repository is an independent experimental implementation. “Jev” and “System One Models” refer to TypeSafe AI's work; no affiliation or equivalence is implied.

- [TypeSafe AI — Introducing System One Models & Jev](https://typesafe.ai/blog/introducing-system-one-models-and-jev): original System One, parallel sampling, typed probabilistic output, and RLCD claims.
- [Harsha Gundala — Qwen2.5 parallel constrained readout](https://x.com/harshagundal/status/2100044305536889015): open-source demonstration that motivated the shared-prefill, broadcast-KV, and allowed-token scoring path implemented here.
- [Harsha Gundala — explanation of next-token choice probabilities](https://x.com/harshagundal/status/2100097423783989297): clarification that the public experiment reads an existing LLM's probabilities without post-training.
- [Bruno Galvão — Astra optimization experiment](https://x.com/brunoqgalvao/status/2100192162260439198): independently reported optimization of the parallel readout approach, reaching a claimed 8× improvement.
- [Matt Mastracci — DiffusionGemma Jev-like mode](https://x.com/mmastrac/status/2100358223643718023) and [vLLM PR #57250](https://github.com/vllm-project/vllm/pull/57250): fixed diffusion canvas, parallel answer positions, entropy, and adaptive re-sampling approach.
- [Austin Huang — trained tiny Jev-like model](https://x.com/austinvhuang/status/2100246355318817175): reports a trained model that is neither a classifier nor prefix-parallel sampling. No architecture, training recipe, code, or weights were disclosed in the referenced post, so this is tracked as a distinct claim rather than a reproducible method.
- [Harsha Gundala — Qwen-2.5-1B-RLCD repository](https://huggingface.co/harshatheg/Qwen-2.5-1B-RLCD): public MLX implementation reviewed while building this repository.
- [OpenJev](https://openjev.com/): browser-only direct-logprob versus autoregressive-generation comparison that clearly distinguishes constrained-choice scores from calibrated probabilities.
- [Alex Wortega — trained OpenJev NLI cross-encoder](https://huggingface.co/AlexWortega/openjev): Qwen3.5-4B sequence-classification checkpoint trained to score premise–hypothesis pairs as contradiction, entailment, or neutral; a distinct trained semantic-decision method supporting reranking and grading.
