"""Run Levels 0–2 against the same labeled examples."""

import argparse
import json
import statistics
from pathlib import Path

import jev_like
from jev_like import decide_parallel, expected_calibration_error, generate_json


def load_cases(path: str):
    with open(path) as handle:
        return [json.loads(line) for line in handle if line.strip()]


def normalized(value):
    return str(value).lower()


def evaluate(mode: str, cases: list[dict], warmup: bool = True):
    if mode == "autoregressive":
        runner = generate_json
    else:
        runner = decide_parallel
    if warmup:
        print(f"{mode:14} warming up...")
        runner(cases[0]["context"], cases[0]["schema"])
    latencies, correct, confidences = [], [], []
    tokens, token_rates = [], []
    valid = 0
    for index, case in enumerate(cases, 1):
        output = runner(case["context"], case["schema"])
        latencies.append(output["elapsed_ms"])
        tokens.append(output.get("equivalent_tokens") or output["forward_passes"])
        if output.get("tokens_per_s"):
            token_rates.append(output["tokens_per_s"])
        valid += output["valid"]
        values = output["value"] or {}
        for field, expected in case["expected"].items():
            prediction = values.get(field)
            score = 1.0
            if mode != "autoregressive" and prediction:
                score, prediction = prediction["score"], prediction["value"]
            correct.append(normalized(prediction) == normalized(expected))
            confidences.append(score)
        print(f"{mode:14} case {index}: {output['elapsed_ms']:8.1f} ms  valid={output['valid']}")

    ordered = sorted(latencies)
    p95 = ordered[min(len(ordered) - 1, round(0.95 * len(ordered)) - 1)]
    return {
        "mode": mode,
        "cases": len(cases),
        "fields": len(correct),
        "accuracy": round(sum(correct) / len(correct), 4),
        "schema_validity": round(valid / len(cases), 4),
        "mean_ms": round(statistics.mean(latencies), 2),
        "p95_ms": round(p95, 2),
        "ece": round(expected_calibration_error(confidences, correct), 4),
        "mean_tokens": round(statistics.mean(tokens), 1),
        "tokens_per_s": round(statistics.mean(token_rates), 1) if token_rates else 0.0,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data/benchmark.jsonl")
    parser.add_argument("--mode", choices=("autoregressive", "parallel", "both"), default="both")
    parser.add_argument("--model", default=jev_like.MODEL_ID)
    parser.add_argument("--no-warmup", action="store_true")
    parser.add_argument("--json-output", type=Path)
    args = parser.parse_args()
    jev_like.MODEL_ID = args.model
    cases = load_cases(args.data)
    modes = ("autoregressive", "parallel") if args.mode == "both" else (args.mode,)
    results = [evaluate(mode, cases, not args.no_warmup) for mode in modes]
    for result in results:
        result["model"] = args.model
    if len(results) == 2:
        results.append({
            "mode": "comparison",
            "model": args.model,
            "mean_speedup": round(results[0]["mean_ms"] / results[1]["mean_ms"], 2),
            "p95_speedup": round(results[0]["p95_ms"] / results[1]["p95_ms"], 2),
        })
    output = json.dumps(results, indent=2)
    if args.json_output:
        args.json_output.parent.mkdir(parents=True, exist_ok=True)
        args.json_output.write_text(output + "\n")
    print("\n" + output)


if __name__ == "__main__":
    main()
