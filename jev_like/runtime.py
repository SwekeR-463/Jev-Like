"""Autoregressive and parallel constrained decision engines for MLX."""

from __future__ import annotations

import copy
import json
import os
import re
import time
from dataclasses import dataclass
from typing import Any

import mlx.core as mx
from mlx_lm import load
from mlx_lm.models.cache import make_prompt_cache

DEFAULT_MODEL_ID = os.getenv("MODEL_ID", "mlx-community/Qwen2.5-1.5B-Instruct-4bit")

_engines: dict[str, tuple[Any, Any]] = {}


@dataclass(frozen=True)
class Field:
    description: str
    choices: tuple[str, ...]


def parse_schema(raw: dict[str, Any]) -> dict[str, Field]:
    schema = {}
    for name, spec in raw.items():
        choices = ("true", "false") if spec["type"] == "boolean" else tuple(spec["choices"])
        if not choices or len(choices) > 255:
            raise ValueError(f"{name}: expected 1–255 choices")
        schema[name] = Field(spec.get("description", ""), choices)
    return schema


def get_engine(model_id: str = DEFAULT_MODEL_ID):
    """Load and cache one engine per model, so a process can switch models safely."""
    if model_id not in _engines:
        started = time.perf_counter()
        print(f"Loading {model_id}…")
        try:
            model, tokenizer = load(model_id)
        except ValueError as error:
            if "not supported" not in str(error):
                raise
            from mlx_vlm import load as load_vlm

            vlm, processor = load_vlm(model_id)
            if not hasattr(vlm, "language_model"):
                raise ValueError(f"{model_id} is unsupported by mlx-lm and is not an MLX-VLM model") from error
            model = vlm.language_model
            tokenizer = getattr(processor, "tokenizer", processor)
        logits = _forward(model, mx.array([tokenizer.encode("warmup")]), _make_cache(model))
        mx.eval(logits)
        _engines[model_id] = (model, tokenizer)
        print(f"Loaded and warmed in {time.perf_counter() - started:.1f}s")
    return _engines[model_id]


def _make_cache(model):
    return model.make_cache() if hasattr(model, "make_cache") else make_prompt_cache(model)


def _forward(model, tokens, cache):
    output = model(tokens, cache=cache)
    return output.logits if hasattr(output, "logits") else output


def _chat_prompt(tokenizer, model_id: str, instruction: str) -> str:
    """Use each model's native chat template; Gemma does not accept a system role."""
    if "qwen3" in model_id.lower():
        instruction += "\n/no_think"
    if hasattr(tokenizer, "apply_chat_template"):
        return tokenizer.apply_chat_template(
            [{"role": "user", "content": instruction}],
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
    return instruction + "\nAnswer:\n"


def _prompt(tokenizer, model_id: str, context: str, schema: dict[str, Field]) -> str:
    lines = [f'  "{name}": one of {list(field.choices)} // {field.description}' for name, field in schema.items()]
    return _chat_prompt(
        tokenizer,
        model_id,
        "Return only a valid JSON object matching this schema:\n"
        + "{\n" + "\n".join(lines) + "\n}\n"
        + f"Context:\n{context}",
    )


def _extract_json(text: str) -> dict[str, Any] | None:
    match = re.search(r"\{.*\}", text, re.S)
    if not match:
        return None
    try:
        value = json.loads(match.group())
        return value if isinstance(value, dict) else None
    except json.JSONDecodeError:
        return None


def _valid(value: dict[str, Any] | None, schema: dict[str, Field]) -> bool:
    if value is None or set(value) != set(schema):
        return False
    return all(str(value[name]).lower() in field.choices for name, field in schema.items())


def generate_json(
    context: str,
    raw_schema: dict[str, Any],
    max_tokens: int = 160,
    model_id: str = DEFAULT_MODEL_ID,
) -> dict[str, Any]:
    """Level 1: ordinary token-by-token JSON generation baseline."""
    model, tokenizer = get_engine(model_id)
    schema = parse_schema(raw_schema)
    cache = _make_cache(model)
    started = time.perf_counter()
    logits = _forward(model, mx.array([tokenizer.encode(_prompt(tokenizer, model_id, context, schema))]), cache)
    mx.eval(logits)
    prefill_ms = (time.perf_counter() - started) * 1000
    decode_started = time.perf_counter()
    tokens: list[int] = []
    stop = {tokenizer.eos_token_id}

    for _ in range(max_tokens):
        token = int(mx.argmax(logits[0, -1]))
        if token in stop:
            break
        tokens.append(token)
        text = tokenizer.decode(tokens)
        if text.rstrip().endswith("}"):
            break
        logits = _forward(model, mx.array([[token]]), cache)
        mx.eval(logits)

    elapsed = (time.perf_counter() - started) * 1000
    decode_ms = (time.perf_counter() - decode_started) * 1000
    text = tokenizer.decode(tokens)
    value = _extract_json(text)
    return {
        "mode": "autoregressive",
        "elapsed_ms": round(elapsed, 2),
        "prefill_ms": round(prefill_ms, 2),
        "decode_ms": round(decode_ms, 2),
        "forward_passes": len(tokens),
        "tokens_per_s": round(len(tokens) / (decode_ms / 1000), 1) if decode_ms else 0.0,
        "valid": _valid(value, schema),
        "value": value,
        "raw": text,
    }


def _single_token_id(tokenizer, text: str) -> int:
    ids = tokenizer.encode(text, add_special_tokens=False)
    if len(ids) != 1:
        raise ValueError(f"Internal label {text!r} is not one token")
    return ids[0]


def _readout(tokenizer, field: Field) -> tuple[list[int], str]:
    direct = [tokenizer.encode(" " + choice, add_special_tokens=False) for choice in field.choices]
    if all(len(ids) == 1 for ids in direct):
        return [ids[0] for ids in direct], ", ".join(field.choices)
    labels = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    if len(field.choices) > len(labels):
        raise ValueError("Multi-token choices support at most 26 options")
    token_ids = [_single_token_id(tokenizer, " " + label) for label in labels[: len(field.choices)]]
    return token_ids, ", ".join(f"{labels[i]}={choice}" for i, choice in enumerate(field.choices))


def decide_parallel(
    context: str,
    raw_schema: dict[str, Any],
    model_id: str = DEFAULT_MODEL_ID,
) -> dict[str, Any]:
    """Level 2: one prefill plus one batched suffix pass for every field."""
    model, tokenizer = get_engine(model_id)
    schema = parse_schema(raw_schema)
    readouts = {name: _readout(tokenizer, field) for name, field in schema.items()}
    catalog = "\n".join(
        f"{name}: {field.description}; choices: {readouts[name][1]}" for name, field in schema.items()
    )
    prefix = _chat_prompt(
        tokenizer,
        model_id,
        "Choose the best exact choice for each requested field.\n"
        f"{catalog}\nContext:\n{context}",
    )
    suffixes = [f"{name}:" for name in schema]
    suffix_ids = [tokenizer.encode(text, add_special_tokens=False) for text in suffixes]
    lengths = [len(ids) for ids in suffix_ids]
    width = max(lengths)
    pad = tokenizer.pad_token_id or 0
    suffix_batch = mx.array([ids + [pad] * (width - len(ids)) for ids in suffix_ids])

    started = time.perf_counter()
    cache = _make_cache(model)
    prefill = _forward(model, mx.array([tokenizer.encode(prefix)]), cache)
    mx.eval(prefill)
    prefill_ms = (time.perf_counter() - started) * 1000

    batch_cache, cache_arrays = _broadcast_cache(cache, len(schema))
    mx.eval(*cache_arrays)

    suffix_started = time.perf_counter()
    logits = _forward(model, suffix_batch, batch_cache)
    mx.eval(logits)
    suffix_ms = (time.perf_counter() - suffix_started) * 1000

    result = {}
    for row, ((name, field), length) in enumerate(zip(schema.items(), lengths)):
        token_ids = readouts[name][0]
        scores = mx.array([logits[row, length - 1, token_id] for token_id in token_ids])
        probabilities = mx.softmax(scores)
        mx.eval(probabilities)
        winner = int(mx.argmax(probabilities))
        result[name] = {
            "value": field.choices[winner],
            # Conditional score among allowed choices; not calibrated confidence.
            "score": round(float(probabilities[winner]), 4),
            "distribution": {
                choice: round(float(probability), 4)
                for choice, probability in zip(field.choices, probabilities.tolist())
            },
        }

    assembled = {name: entry["value"] for name, entry in result.items()}
    equivalent_tokens = len(tokenizer.encode(json.dumps(assembled)))

    return {
        "mode": "parallel",
        "elapsed_ms": round((time.perf_counter() - started) * 1000, 2),
        "prefill_ms": round(prefill_ms, 2),
        "suffix_ms": round(suffix_ms, 2),
        "decode_ms": round(suffix_ms, 2),
        "forward_passes": 2,
        # No JSON tokens are generated; this is the equivalent size of the assembled object,
        # timed against the suffix pass so it is comparable to the autoregressive decode rate.
        "equivalent_tokens": equivalent_tokens,
        "tokens_per_s": round(equivalent_tokens / (suffix_ms / 1000), 1) if suffix_ms else 0.0,
        "valid": True,
        "value": result,
    }


def _broadcast_cache(cache, batch_size: int) -> tuple[list[Any], list[Any]]:
    """Replicate a single-sequence prompt cache across every schema field."""
    batch_cache, arrays = [], []
    for part in cache:
        clone = copy.copy(part)
        if getattr(part, "keys", None) is not None:
            clone.keys = mx.repeat(part.keys, batch_size, axis=0)
            clone.values = mx.repeat(part.values, batch_size, axis=0)
            arrays.extend((clone.keys, clone.values))
        if getattr(part, "cache", None):
            clone.cache = [
                mx.repeat(value, batch_size, axis=0) if value is not None else None for value in part.cache
            ]
            arrays.extend(value for value in clone.cache if value is not None)
        batch_cache.append(clone)
    return batch_cache, arrays
