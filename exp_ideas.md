# Experimental Paths Toward an Open Jev-Like System

**Original synthesis and experiment roadmap:** `gpt-5.6-sol`  
**Date:** 2026-09-17

This document orders the ideas from easiest and least invasive to genuine model research. Each level should be attempted only when the preceding level has been measured. “Jev-like” here means runtime-defined typed decisions, parallel outputs, useful probabilities, low latency, and safe abstention—not merely faster JSON generation.

## Progression at a Glance

| Level | Experiment | Training | Implementation effort | Main goal |
|---|---|---|---|---|
| 0 | Strong evaluation protocol | None | Very low | Make improvements measurable |
| 1 | Calibration and abstention | One scalar fit | Low | Make confidence actionable |
| 2 | Verbalizer and consistency ensemble | None | Low | Reduce prompt/token sensitivity |
| 3 | Semantic candidate scoring | None | Low–medium | Support arbitrary multi-token choices |
| 4 | Adaptive model cascade | Small router optional | Medium | Optimize average quality and latency |
| 5 | Retrieval and symbolic consistency | None | Medium | Add precedent and joint constraints |
| 6 | OpenJev NLI cross-encoder | Existing trained model | Medium | Add a trained semantic baseline |
| 7 | Hidden-state prototypes | Optional projection only | Medium–high | Avoid vocabulary-token scoring |
| 8 | Early-exit decision heads | Tiny heads | High | Skip unnecessary transformer layers |
| 9 | Dynamic open-vocabulary heads | Lightweight generic training | High | Score runtime labels in one matrix operation |
| 10 | Multi-slot and joint decision model | LoRA or supervised training | Very high | Predict all typed fields in one pass |
| 11 | RLCD and specialist adapters | Decision training | Very high | Improve accuracy, stability, and calibration |
| 12 | Diffusion/System-One architecture | Full research training | Research | Native non-autoregressive decisions |

## Level 0 — Make the Evaluation Hard to Fool

Before optimizing inference, expand the benchmark to include:

- Realistic domain examples and locked train/calibration/test splits.
- Accuracy, macro-F1, schema validity, p50/p95 latency, and throughput.
- ECE, Brier score, negative log-likelihood, and risk–coverage curves.
- Choice-order permutation, paraphrase, negation, and irrelevant-detail tests.
- Out-of-domain and deliberately ambiguous examples.
- Per-domain and per-field reporting instead of only a global average.

**Exit condition:** repeated runs produce stable conclusions and no prompt variant reverses the model ranking.

## Level 1 — Calibrate, Abstain, and Spend Compute Selectively

Keep every model frozen. Fit a single temperature on held-out labels, then define policies such as:

```text
calibrated confidence ≥ 0.90 → accept
0.65–0.90                  → retry or stronger model
< 0.65                     → abstain or escalate
```

Add conformal prediction sets so uncertain cases may return more than one plausible value instead of a falsely precise answer. Measure selective accuracy as coverage falls.

**Why first:** it requires almost no machinery and becomes the control system for every later level.

**Exit condition:** confidence bins are trustworthy enough to route decisions without increasing high-confidence error.

## Level 2 — Verbalizer and Consistency Ensemble

Score several semantic descriptions for each choice rather than one fragile token:

```text
billing
"This belongs to billing."
"The billing team should handle this."
"This is a payment-related issue."
```

Aggregate paraphrase scores with a mean or log-sum-exp. Also repeat decisions under:

- Choice-order permutation.
- Equivalent schema descriptions.
- Context paraphrases.
- Removal of irrelevant names or emotional language.
- Counterfactual changes to facts that should not affect the answer.

Use disagreement as an uncertainty feature, not a majority vote that hides instability.

**Exit condition:** robustness improves enough to justify the additional batched suffixes.

## Level 3 — Full Semantic Candidate Scoring

Remove the single-token restriction. Convert each allowed value into a complete candidate statement and score its normalized sequence likelihood:

```text
context cache
    ├── "The billing team should own this issue."
    ├── "The infrastructure team should own this issue."
    └── "The support team should own this issue."
```

Reuse one context KV cache, batch all suffixes, and normalize scores within each field. Add length normalization and test both direct semantic values and neutral A/B/C verbalizers.

**Exit condition:** multi-token choices improve accuracy without erasing the parallel latency advantage.

## Level 4 — Adaptive Jev Cascade

Use the fastest model first and spend more compute only on uncertainty:

```text
LFM2.5-VL 450M parallel
        │ uncertain
        ▼
Qwen2.5 1.5B parallel
        │ uncertain or disagreement
        ▼
OpenJev NLI
        │ unresolved
        ▼
autoregressive fallback or human review
```

Start with hand-written thresholds based on calibrated confidence, entropy, top-two margin, paraphrase agreement, and out-of-domain distance. Train a tiny logistic-regression router only if the rules plateau.

Optional weighted consensus:

```text
final score = w₁·LFM + w₂·Qwen + w₃·OpenJev
```

Fit only the few ensemble weights on calibration data.

**Exit condition:** accuracy approaches the strongest path while average latency stays close to the cheapest path.

## Level 5 — Retrieval and Joint Consistency

### Retrieval-backed decisions

Retrieve similar reviewed cases and use them either as shared-prefix demonstrations or as a prior over choices. Preserve provenance, timestamps, and human outcome so stale or incorrect precedent can be rejected.

### Symbolic joint constraints

Reject impossible combinations after parallel field scoring:

```text
severity=critical → action cannot be monitor
remove=false      → escalation should normally be none
fraud=high        → fulfillment cannot be ship
```

### Energy-based joint reranking

For dependent fields, retain the top two values per field, construct a small number of joint assignments, and semantically score each complete assignment. This avoids enumerating the full Cartesian product.

**Exit condition:** cross-field contradictions fall materially without unacceptable latency or brittle rule growth.

## Level 6 — OpenJev NLI and Shared-Context NLI

Benchmark [Alex Wortega's OpenJev](https://huggingface.co/AlexWortega/openjev) as a separate method. It uses Qwen3.5-4B as a three-way entailment/contradiction/neutral cross-encoder.

```text
(context, candidate hypothesis) → entailment probability
```

Then test a more original shared-context variant:

```text
context → cached representation
              ├── hypothesis A
              ├── hypothesis B
              └── hypothesis C
```

Possible implementations are KV-cache branching or a small cross-attention candidate head. Verify that either preserves the semantics of the original full premise–hypothesis model.

**Exit condition:** the trained semantic scorer materially improves the quality/calibration frontier over causal-LM readout.

## Level 7 — Hidden-State Semantic Prototypes

Skip vocabulary logits. Extract a context representation and compare it directly with embeddings of runtime choice descriptions:

```text
score(context, choice) = context_vector · choice_vector
```

Try two variants in order:

1. Fully frozen embeddings and cosine similarity.
2. A tiny learned projection while the language model remains frozen.

This could score thousands of choices as a matrix multiplication and enables embedding-based shortlisting before a stronger semantic reranker.

**Exit condition:** prototype scoring retains useful accuracy and beats full-vocabulary projection or candidate sequence scoring in latency.

## Level 8 — Early-Exit Decision Heads

Attach small probes to intermediate transformer layers:

```text
layer 8  → confidence 0.54 → continue
layer 12 → confidence 0.78 → continue
layer 18 → confidence 0.94 → exit
```

Freeze the backbone and train only the probes. Calibrate every exit separately and penalize incorrect early exits more heavily than continued computation.

Combine with Level 4: easy cases exit both the model cascade and the transformer depth early.

**Exit condition:** average executed layers decrease with no regression at the target selective-accuracy threshold.

## Level 9 — Runtime-Generated Open-Vocabulary Heads

Generate classifier vectors from natural-language field and choice descriptions:

```text
choice description → label encoder → dynamic weight vector
context            → context encoder
score               = context_vector · dynamic_weight
```

Unlike fixed classifiers, this supports schemas never seen during training. Train across randomized field names, descriptions, choice wording, order, domain, and number of options.

This is a strong candidate for a distinctive open-source Jev-like method because all runtime choices can be scored in one matrix operation without vocabulary-token restrictions.

**Exit condition:** unseen schemas and labels generalize better than frozen prototypes and run faster than NLI cross-encoding.

## Level 10 — Multi-Slot Joint Decision Model

Insert one special slot per requested field:

```text
<context>
<DECISION:urgent>
<DECISION:department>
<DECISION:sentiment>
```

Read all slot hidden states in one forward pass and score their runtime choices with Level 9's dynamic label vectors. Add a small set decoder or factor graph only when fields are dependent.

Training examples should randomize schema shape so the model learns the general operation “make typed decisions,” not one task.

**Exit condition:** the model predicts variable multi-field schemas in one pass and exceeds the best inference-only accuracy/latency frontier.

## Level 11 — RLCD, Soft Targets, and Specialist Adapters

Train against teacher distributions rather than only hard answers:

```text
billing=0.82, support=0.15, infrastructure=0.03
```

Reward:

- Correct typed decisions.
- Stability under paraphrase and label permutation.
- Low confidence on ambiguity and out-of-domain inputs.
- Correct abstention.
- Consistent multi-field assignments.
- Early exits on easy cases.

Keep a shared decision adapter and optional domain adapters:

```text
frozen base + generic decision LoRA + security/fraud/support LoRA
```

**Exit condition:** training improves a locked test set and calibration—not merely training or synthetic-teacher agreement.

## Level 12 — Diffusion and Native System-One Research

### Fixed masked canvas

```text
{
  "urgent": <MASK>,
  "department": <MASK>,
  "sentiment": <MASK>
}
```

Predict all slots in parallel, commit confident positions, and resample only uncertain ones. Start from the public [vLLM DiffusionGemma work](https://github.com/vllm-project/vllm/pull/57250).

### Native System-One model

If every earlier method proves economically insufficient, train a purpose-built non-autoregressive model with dynamic typed heads, joint consistency, calibrated uncertainty, and an inference runtime designed around decision slots.

**Exit condition:** the architecture materially beats Levels 4, 9, and 10 on quality per millisecond—not merely architectural novelty.

## Typed Decision Compiler — Runtime End State

The production system should compile each schema into the cheapest suitable inference graph:

```text
boolean or tiny enum     → direct token score
multi-token small enum   → semantic candidate score
large enum               → prototype shortlist + rerank
dependent fields         → joint constraint/energy pass
free-form constrained    → grammar decoding
high-risk decision       → calibrated ensemble + abstention
```

This is the unifying product idea: the schema is compiled into an execution strategy instead of merely inserted into a prompt.

## Recommended Build Order

1. Levels 0–1: trustworthy benchmark, calibration, and abstention.
2. Levels 2–3: robust training-free semantic scoring.
3. Level 4: adaptive LFM/Qwen/OpenJev/autoregressive cascade.
4. Level 5: retrieval and joint consistency where the data proves they are needed.
5. Level 6: trained NLI comparison and shared-context experiment.
6. Levels 7–8: hidden-state prototypes and early exit.
7. Levels 9–10: the most promising original architecture direction.
8. Level 11 only after a locked benchmark identifies the training gap.
9. Level 12 only when suitable NVIDIA hardware and a compelling economic case exist.

## Attribution and Related Work

The organization, system synthesis, experiment ordering, adaptive cascade, typed decision compiler, and proposed combinations in this document were authored by **`gpt-5.6-sol`** for this project. Referenced external mechanisms remain attributed to their respective authors:

- [TypeSafe AI — System One Models and Jev](https://typesafe.ai/blog/introducing-system-one-models-and-jev)
- [Harsha Gundala — public Qwen/RLCD work](https://huggingface.co/harshatheg/Qwen-2.5-1B-RLCD)
- [Alex Wortega — trained OpenJev NLI cross-encoder](https://huggingface.co/AlexWortega/openjev)
- [Matt Mastracci — vLLM DiffusionGemma implementation](https://github.com/vllm-project/vllm/pull/57250)
- [vLLM — automatic prefix caching](https://github.com/vllm-project/vllm/blob/main/docs/features/automatic_prefix_caching.md)
- [vLLM — structured outputs](https://github.com/vllm-project/vllm/blob/main/docs/features/structured_outputs.md)
- [MLX-LM — prompt caching](https://github.com/ml-explore/mlx-lm/blob/main/mlx_lm/examples/chat.py)
- [Medusa — parallel multi-head decoding](https://arxiv.org/abs/2401.10774)
- [Conformal zero-shot text classification](https://arxiv.org/abs/2210.12619)

Attribution here records provenance; it does not claim that the external mechanisms themselves were invented by `gpt-5.6-sol`.
