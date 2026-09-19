"""In-process JevBench adapter for the local Jev-Like shared-prefill readout.

Loads `jev_like.runtime` (MLX) and reads the model's own choice probabilities at
the answer slot: one prefill over state + question + options, then one suffix
pass that scores the allowed option tokens. Native, not verbalized - not
generated text. The option ids we score are always listed in the prompt.

Install: copy this file into a fresh clone of fstandhartinger/jevbench as
`jevbench/adapters/jev_like_direct.py`, add the three export/dispatch lines from
the README, and `--adapter jev_like_direct` becomes available.
"""

from __future__ import annotations

import json
import os
import sys
import time

import mlx.core as mx

# Absolute rather than relative so the file imports from either location.
from jevbench.adapters.base import DecisionResult

# When installed into a jevbench clone that sits next to this repo, put our
# sibling Jev-Like root on the path so `import jev_like` resolves.
_JEZ_LIKE_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__)))))
if os.path.isdir(os.path.join(_JEZ_LIKE_ROOT, "jev_like")) and _JEZ_LIKE_ROOT not in sys.path:
    sys.path.insert(0, _JEZ_LIKE_ROOT)

_LETTERS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"


def option_display(tokenizer, labels):
    """Return the display ids the prompt must list and the tokens we score.

    Single-token labels (no/yes, 0/1/2) are shown and scored as themselves;
    longer labels fall back to letter options, exactly like runtime._readout.
    The scored ids MUST appear in the prompt - scoring letters the model was
    never shown reads out at chance, silently.
    """
    direct = [tokenizer.encode(" " + label, add_special_tokens=False)
              for label in labels]
    if all(len(ids) == 1 for ids in direct):
        return list(labels), [ids[0] for ids in direct]
    if len(labels) > len(_LETTERS):
        raise ValueError(f"too many options for a letter readout: {len(labels)}")
    letters = list(_LETTERS[: len(labels)])
    return letters, [tokenizer.encode(" " + letter, add_special_tokens=False)[0]
                     for letter in letters]


def _labels_and_rubric(task):
    qtype = task.question["type"]
    crit = task.question.get("criteria")
    if qtype == "noul":
        crit = crit or {}
        # yes first: this family is order-sensitive, and yes/no-first is the
        # canonical order the bench's own yes_no() helper uses.
        return ["yes", "no"], {"yes": crit.get("true", "Yes"),
                               "no": crit.get("false", "No")}
    if qtype == "score":
        labels = [str(i) for i in range(len(crit))]
        return labels, dict(zip(labels, crit))
    labels = list(crit)
    return labels, {k: (v or k) for k, v in crit.items()}


class JevLikeDirectAdapter:
    name = "jev_like_direct"
    cost_basis = "local_cpu_no_provider_tariff"

    def __init__(self, endpoint=None, model=None, key_env="", timeout_s=None,
                 price_input_per_m=None, price_output_per_m=None,
                 revision=None, **kwargs):
        # `endpoint` carries the Hugging Face model id for jev_like; keep the
        # same --endpoint convention as LocalOpenJevAdapter (local snapshot
        # directory there, HF model id here).
        self.model = model or endpoint or ""
        self.key_env = key_env
        self.timeout_s = timeout_s
        self.price_input_per_m = price_input_per_m
        self.price_output_per_m = price_output_per_m
        self.revision = revision
        self._runtime = None

    def load(self):
        if self._runtime is None:
            from jev_like import runtime
            self._runtime = runtime.get_engine(self.model)
        return self._runtime

    def build_request(self, task) -> dict:
        labels, rubric = _labels_and_rubric(task)
        state = task.state if isinstance(task.state, str) else json.dumps(
            task.state, ensure_ascii=False)
        return {"task_id": task.id, "state": state,
                "instructions": task.question["instructions"],
                "labels": labels, "rubric": rubric}

    def run(self, task) -> DecisionResult:
        res = DecisionResult(adapter=self.name, ok=False,
                             probs_source="native", model=self.model)
        body = self.build_request(task)
        try:
            model, tok = self.load()
        except Exception as e:  # noqa: BLE001
            res.error = f"load failed: {type(e).__name__}: {str(e)[:250]}"
            return res

        from jev_like import runtime
        mx.eval("warm")

        labels, rubric = body["labels"], body["rubric"]
        try:
            display, idx = option_display(tok, labels)
        except ValueError as e:
            res.error = str(e)
            return res

        options = "\n".join(f"{d}: {rubric[lab]}"
                            for d, lab in zip(display, labels))
        context = (f"State:\n{body['state']}\n\n"
                   f"Question: {body['instructions']}\n\n"
                   f"Options:\n{options}\n\n"
                   f"Answer with exactly one of: {', '.join(display)}.")
        res.request_body = {"context": context,
                            "options": dict(zip(display, labels))}

        cache = runtime._make_cache(model)
        # The chat template is load-bearing: raw text puts the answer slot out of
        # distribution and flattens the softmax to 0.5/0.5.
        prefix = runtime._chat_prompt(tok, self.model, context)
        prefix_ids = tok.encode(prefix)
        suffix = "\nAnswer:"
        suffix_ids = tok.encode(suffix, add_special_tokens=False)

        t0 = time.perf_counter()
        try:
            prefill = runtime._forward(model, mx.array([prefix_ids]), cache)
            mx.eval(prefill)
            batch_cache, arrays = runtime._broadcast_cache(cache, 1)
            mx.eval(*arrays)
            sfx = mx.array(suffix_ids)[None, :]
            logits = runtime._forward(model, sfx, batch_cache)
            mx.eval(logits)
            spot = logits[0, len(suffix_ids) - 1]
            scores = mx.array([float(spot[i]) for i in idx])
            probs = mx.softmax(scores)
            mx.eval(probs)
            res.latency_s = time.perf_counter() - t0
        except Exception as e:  # noqa: BLE001
            res.latency_s = time.perf_counter() - t0
            res.error = f"{type(e).__name__}: {str(e)[:300]}"
            return res

        # Space must be trimmed: JevBench labels are bare, our readout adds a
        # leading space for tokenizer stability.
        res.probs = {label: float(p)
                     for label, p in zip(labels, probs.tolist())}
        res.usage = {"input_tokens": len(prefix_ids) + len(suffix_ids),
                     "output_tokens": 0}
        res.raw = {"prompt_chars": len(context),
                   "label_readout": dict(zip(display, labels)),
                   "runtime": {
                       "path": "jev_like.runtime.decide_parallel-style readout",
                       "probability_origin": "native-suffix-softmax",
                       "revision": self.revision,
                       "forward_passes": 2}}
        res.ok = True
        return res

    def reserve_estimate(self, task) -> float:
        return 0.0
