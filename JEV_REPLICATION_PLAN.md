# Jev-Like Model Replication Plan

Build incrementally and stop at the first level that meets the required accuracy, calibration, and latency targets.

## Overview

| Level | Build | Estimated time | Hardware | Expected result |
|---|---|---:|---|---|
| 0 | Benchmark harness | 0.5 day | Any | Honest baseline |
| 1 | Autoregressive structured output | 0.5–1 day | Local or CUDA | Functional typed API |
| 2 | Qwen logit slicing | 1–2 days | Local GPU or CUDA | Fast, simple prototype |
| 3 | Trained NLI cross-encoder | 1–3 days | Apple Silicon or CUDA | General semantic decision model |
| 4 | DiffusionGemma fixed canvas | 2–4 days | NVIDIA GPU | Closest open Jev architecture |
| 5 | Calibration and adaptive sampling | 3–7 days | Same GPU as inference | Trustworthy uncertainty |
| 6 | Fine-tuning or distillation | 2–6 weeks | Strong CUDA GPU | Task-specific production model |
| 7 | Custom System-One model | 2–6+ months | GPU cluster | Research project |

## Level 0 — Define the Benchmark

Create a representative dataset containing:

- Input state or text.
- Typed questions.
- Allowed answers.
- Correct answers.
- Optional human confidence or ambiguity labels.

Measure:

- Accuracy and F1.
- Schema validity.
- p50 and p95 latency.
- Throughput.
- Brier score and expected calibration error.
- Abstention accuracy at different confidence thresholds.

**Exit condition:** At least 200–1,000 representative examples and one repeatable benchmark command.

## Level 1 — Reliable Baseline

Use an existing instruction-tuned model with constrained JSON generation:

```text
state + schema → constrained generation → validated JSON
```

Initially support only:

- Boolean.
- Enum.
- Numeric scale.
- Optional `unknown`.

This verifies that the model understands the task before inference is optimized.

**Exit condition:** Acceptable task accuracy and 100% schema validity.

## Level 2 — Qwen Parallel Logit Scoring

Implement the Qwen/OpenJev approach:

```text
one shared context prefill
    ├── question 1 → allowed-token logits
    ├── question 2 → allowed-token logits
    └── question N → allowed-token logits
```

Steps:

1. Use Qwen 1.5B–3B through MLX or PyTorch.
2. Require single-token answer labels initially.
3. Mask logits to the allowed choices.
4. Normalize across those choices.
5. Assemble JSON programmatically.
6. Reuse the context KV cache across questions.

Targets:

- 5–10× faster than JSON generation.
- 100% valid typed output.
- Accuracy close to the Level 1 baseline.

**Limitation:** Outputs are parallel at the batch level, but each question still has its own suffix computation.

## Level 3 — Trained NLI Cross-Encoder

Evaluate [Alex Wortega's OpenJev checkpoint](https://huggingface.co/AlexWortega/openjev), a Qwen3.5-4B cross-encoder trained for three-way natural-language inference:

```text
context + candidate statement
        ↓
contradiction / entailment / neutral
```

For each schema field, convert every allowed choice into a hypothesis:

```text
context:    "The card was charged twice."
hypotheses: "The department is billing."
            "The department is infrastructure."
            "The department is support."
```

Batch all context–hypothesis pairs, choose the option with the highest entailment probability, and assemble typed JSON programmatically.

Steps:

1. Add a separate PyTorch/MPS runner for `AlexWortega/openjev`; do not force it through the current MLX causal-LM path.
2. Generate explicit hypotheses from every field description and allowed value.
3. Score all candidates in one batch.
4. Normalize entailment scores within each field.
5. Benchmark accuracy, validity, latency, throughput, ECE, and Brier score against Levels 1 and 2.
6. Test a hybrid optimization that reuses shared context computation across candidate hypotheses if the architecture permits it.

Advantages:

- Choices may contain arbitrary multi-token concepts.
- The model is trained specifically for semantic decisions rather than next-token generation.
- The same primitive supports classification, reranking, grading, and guards without task-specific retraining.

Limitations:

- It is a trained sequence classifier, not a training-free readout.
- Each context–hypothesis pair may repeat context computation.
- The 4B BF16 checkpoint is larger than the current 270M–1.7B comparison models.
- Entailment scores still require held-out calibration before being treated as confidence.

**Exit condition:** It materially improves accuracy or calibration over Level 2 at an acceptable latency and memory cost.

## Level 4 — DiffusionGemma Fixed Canvas

This is the closest public Jev-like implementation:

```text
{
  "urgent":     [MASK],
  "department": [MASK],
  "sentiment":  [MASK]
}
        ↓ one diffusion pass
all masked positions predicted together
```

Steps:

1. Use the [vLLM DiffusionGemma PR](https://github.com/vllm-project/vllm/pull/57250).
2. Pre-seed a fixed JSON or token canvas.
3. Put noise tokens only in answer slots.
4. Run one read-only diffusion step.
5. Retrieve logits and entropy for every answer position.
6. Convert results into the external API schema.
7. Benchmark concurrency and batching.

Targets:

- Approximately 200 ms for confident single-pass requests on suitable NVIDIA hardware.
- Multiple independent fields predicted in one model pass.
- Accuracy equal to or better than Level 2.

**Initial constraint:** Use single-token internal labels such as `A`, `B`, and `C`, then map them to human-readable values after inference.

## Level 5 — Real Uncertainty Handling

Raw softmax and entropy are not reliable confidence measures.

Add:

1. Temperature scaling using held-out data.
2. Reliability diagrams and Brier-score tracking.
3. An explicit `unknown` choice.
4. Confidence thresholds for automatic action.
5. Adaptive re-sampling only when entropy is high.
6. Agreement scoring across uncertain samples.

Example policy:

```text
confidence ≥ 0.90 → automate
0.65–0.90        → retry or sample
< 0.65           → abstain or escalate
```

**Exit condition:** Predictions reporting roughly 90% confidence are correct approximately 90% of the time on unseen data.

## Level 6 — Fine-Tune for the Workload

Only begin training if Levels 2–5 prove useful but accuracy remains insufficient.

Training data should include:

- Real labeled workflow decisions.
- Synthetic examples from stronger teacher models.
- Difficult counterexamples.
- Ambiguous examples labeled `unknown`.
- Perturbations for consistency testing.

Training objectives:

- Correct categorical decisions.
- Stable answers under equivalent wording.
- Lower confidence on ambiguous inputs.
- Agreement with teacher distributions instead of only hard labels.

Start with LoRA or distillation rather than modifying the model architecture.

### Preferred first experiment: train Level 2 directly

Keep the Level 2 shared-prefill and batched-suffix inference path unchanged. Train only the decision behavior: for every field, mask to its allowed choices and minimize cross-entropy on the correct choice.

```text
context + field description + allowed choices
    → allowed-choice logits
    → cross-entropy or teacher-distribution loss
```

For a correct choice `y` from allowed set `C`:

```text
loss = -log(exp(logit[y]) / sum(exp(logit[c]) for c in C))
```

Implement in increasing order:

1. LoRA with hard labeled choices.
2. Soft-label distillation from strong teacher probability distributions.
3. Consistency loss across paraphrases, reordered choices, and equivalent field descriptions.
4. Explicit `unknown` examples and penalties for confident errors.
5. Refit temperature scaling on a held-out calibration split after training.

### Optional next step: RLCD-style training

After supervised LoRA or distillation, optimize for:

- Correct choices.
- Honest uncertainty.
- Stable probabilities.
- Proper abstention.
- Consistency across equivalent inputs.

The resulting probabilities remain conditional on the supplied choices. Neither LoRA nor an RLCD-style objective automatically guarantees calibration. Evaluate on a held-out split and apply temperature scaling or another post-hoc calibrator when needed.

This approach avoids spending training capacity on braces, field names, explanations, or other JSON tokens while preserving the Level 2 shared-prefill inference path.

**Exit condition:** A meaningful accuracy or calibration improvement over the untrained model on a locked test set.

## Level 7 — Custom System-One Model

This level requires:

- A non-autoregressive decision architecture.
- Native multi-field output heads.
- Custom probabilistic training.
- Large-scale decision datasets.
- Calibration-aware reinforcement learning.
- A specialized inference runtime.

This is startup or research-lab scope, not an initial implementation step.

## Recommended Route

Build **Levels 0 → 2 → 3 → 5** first. Levels 0–2 already run locally; Level 3 adds a trained semantic-decision baseline, and Level 5 calibrates whichever method performs best.

Add **Level 4** when NVIDIA hardware is available and architectural similarity to Jev matters. Attempt Level 6 only when benchmark results demonstrate a specific accuracy or calibration gap. Skip Level 7 unless inference economics justify maintaining a new model architecture.

## Decision Gates

| After level | Continue only when |
|---|---|
| 0 | The dataset represents the real workload. |
| 1 | Model accuracy is useful enough to optimize. |
| 2 | Latency or cost still prevents deployment. |
| 3 | Trained entailment materially improves the quality/speed tradeoff. |
| 4 | Parallel diffusion materially beats Levels 2 and 3. |
| 5 | Calibrated automation provides measurable value. |
| 6 | A custom architecture has a demonstrated economic advantage. |
