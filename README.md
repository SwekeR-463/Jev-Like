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

[Side-by-side comparison video](demo/comparison.mp4) — Qwen3-1.7B BF16, using measured local mean latency. The animation is intentionally slowed so the inference paths are visible.

Re-render it with:

```bash
sh demo/render.sh
```

[Live single-case demo](demo/live.mp4) — one real ticket run through both paths on Qwen3.5-2B BF16: the parallel readout scores 3/3 against the autoregressive path's 2/3, at roughly 6x lower latency. The numbers and the JSON come from an actual run; only the replay pacing is artificial.

```bash
uv run --python 3.12 python -m demo.live          # real run, prints the transcript
uv run --python 3.12 python -m demo.live --video  # also writes demo/live.mp4
```

## Run

```bash
uv run --python 3.12 python -m unittest discover -s tests -t .
uv run --python 3.12 python -m jev_like.benchmark --mode parallel
uv run --python 3.12 python -m jev_like.benchmark --mode both --model Qwen/Qwen3.5-2B
uv run --python 3.12 python -m jev_like.benchmark --all
```

`--all` runs every checkpoint in `jev_like.benchmark.MODELS` in its own process, so models never stack in memory, and writes [the full JSON results](results/model_comparison.json) plus [a Markdown comparison](results/model_comparison.md). Each mode gets an unmeasured warm-up inference before its 20 measured cases.

## Current 20-case result

| Model | AR accuracy | Parallel accuracy | AR mean | Parallel mean | Speedup |
|---|---:|---:|---:|---:|---:|
| Qwen2.5 1.5B 4-bit | 70.0% | **60.0%** | 192.5 ms | 62.2 ms | 3.09× |
| Gemma 3 270M BF16 | 20.0% | 48.3% | 98.7 ms | 15.3 ms | 6.46× |
| LFM2.5-VL 1.6B 4-bit | 66.7% | 58.3% | 126.7 ms | 46.5 ms | 2.72× |
| Qwen3 1.7B 4-bit | **75.0%** | 56.7% | 199.8 ms | 64.3 ms | 3.11× |
| Qwen3 0.6B 8-bit | 55.0% | 53.3% | 131.7 ms | 31.3 ms | 4.20× |
| LFM2.5-VL 450M BF16 | 53.3% | 51.7% | 116.9 ms | **13.7 ms** | **8.51×** |
| Llama 3.2 1B Instruct BF16 | 60.0% | 43.3% | 478.0 ms | 34.0 ms | **14.05×** |

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
