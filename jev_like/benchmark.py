"""Benchmark harness for Levels 0–2, plus the command line entry point."""

from __future__ import annotations

import argparse
import json
import statistics
import subprocess
import sys
from pathlib import Path

from .runtime import DEFAULT_MODEL_ID, decide_parallel, generate_json

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DATA = ROOT / "data/benchmark.jsonl"
DEFAULT_RESULTS = ROOT / "results"

# Every checkpoint known to load and run here. `--all` walks this list.
MODELS = [
    "mlx-community/Qwen2.5-1.5B-Instruct-bf16",
    "mlx-community/gemma-3-270m-it-bf16",
    "mlx-community/LFM2.5-VL-1.6B-bf16",
    "mlx-community/Qwen3-1.7B-bf16",
    "mlx-community/Qwen3-0.6B-bf16",
    "LiquidAI/LFM2.5-VL-450M-MLX-bf16",
    "mlx-community/Llama-3.2-1B-Instruct-bf16",
    "Qwen/Qwen3.5-0.8B",
    "Qwen/Qwen3.5-2B",
]


def load_cases(path: Path | str) -> list[dict]:
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def normalized(value) -> str:
    return str(value).lower()


def expected_calibration_error(confidences: list[float], correct: list[bool], bins: int = 10) -> float:
    """Level 0 metric; corrected with held-out calibration at Level 4."""
    if not confidences:
        return 0.0
    total = 0.0
    for index in range(bins):
        low, high = index / bins, (index + 1) / bins
        members = [i for i, score in enumerate(confidences) if low <= score <= high and (score < high or high == 1)]
        if members:
            accuracy = sum(correct[i] for i in members) / len(members)
            confidence = sum(confidences[i] for i in members) / len(members)
            total += len(members) / len(confidences) * abs(accuracy - confidence)
    return total


def evaluate(mode: str, cases: list[dict], model_id: str, warmup: bool = True) -> dict:
    runner = generate_json if mode == "autoregressive" else decide_parallel
    call = lambda case: runner(case["context"], case["schema"], model_id=model_id)
    if warmup:
        print(f"{mode:14} warming up...")
        call(cases[0])

    latencies, correct, confidences, tokens, token_rates = [], [], [], [], []
    valid = 0
    for index, case in enumerate(cases, 1):
        output = call(case)
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


def run_model(model_id: str, cases: list[dict], modes: tuple[str, ...], warmup: bool = True) -> list[dict]:
    results = [evaluate(mode, cases, model_id, warmup) for mode in modes]
    for result in results:
        result["model"] = model_id
    if len(results) == 2:
        results.append({
            "mode": "comparison",
            "model": model_id,
            "mean_speedup": round(results[0]["mean_ms"] / results[1]["mean_ms"], 2),
            "p95_speedup": round(results[0]["p95_ms"] / results[1]["p95_ms"], 2),
        })
    return results


def run_suite(models: list[str], output_dir: Path) -> None:
    """Benchmark every checkpoint in its own process so models do not stack in memory."""
    output_dir.mkdir(exist_ok=True)
    rows = []
    for index, model in enumerate(models, 1):
        path = output_dir / f"run-{index}.json"
        print(f"\n[{index}/{len(models)}] {model}", flush=True)
        subprocess.run(
            [sys.executable, "-m", "jev_like.benchmark", "--model", model, "--json-output", str(path)],
            check=True,
            cwd=ROOT,
        )
        autoregressive, parallel, comparison = json.loads(path.read_text())
        rows.append({
            "model": model,
            "autoregressive_accuracy": autoregressive["accuracy"],
            "parallel_accuracy": parallel["accuracy"],
            "autoregressive_mean_ms": autoregressive["mean_ms"],
            "parallel_mean_ms": parallel["mean_ms"],
            "mean_speedup": comparison["mean_speedup"],
            "autoregressive_p95_ms": autoregressive["p95_ms"],
            "parallel_p95_ms": parallel["p95_ms"],
            "p95_speedup": comparison["p95_speedup"],
            "autoregressive_schema_validity": autoregressive["schema_validity"],
            "parallel_schema_validity": parallel["schema_validity"],
            "parallel_ece": parallel["ece"],
            "autoregressive_tokens_per_s": autoregressive["tokens_per_s"],
            "parallel_tokens_per_s": parallel["tokens_per_s"],
        })
    (output_dir / "model_comparison.json").write_text(json.dumps(rows, indent=2) + "\n")
    header = (
        "| Model | AR acc. | Parallel acc. | AR tok/s | Parallel tok/s | AR mean | Parallel mean | Speedup |\n"
        "|---|---:|---:|---:|---:|---:|---:|---:|\n"
    )
    body = "".join(
        f"| {row['model']} | {row['autoregressive_accuracy']:.1%} | {row['parallel_accuracy']:.1%} | "
        f"{row['autoregressive_tokens_per_s']:.0f} | {row['parallel_tokens_per_s']:.0f} | "
        f"{row['autoregressive_mean_ms']:.1f} ms | {row['parallel_mean_ms']:.1f} ms | {row['mean_speedup']:.2f}x |\n"
        for row in rows
    )
    (output_dir / "model_comparison.md").write_text(header + body)
    print("\n" + header + body)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--mode", choices=("autoregressive", "parallel", "both"), default="both")
    parser.add_argument("--model", default=DEFAULT_MODEL_ID)
    parser.add_argument("--all", action="store_true", help="run every checkpoint in MODELS")
    parser.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS)
    parser.add_argument("--no-warmup", action="store_true")
    parser.add_argument("--json-output", type=Path)
    args = parser.parse_args()

    if args.all:
        run_suite(MODELS, args.results_dir)
        return

    modes = ("autoregressive", "parallel") if args.mode == "both" else (args.mode,)
    results = run_model(args.model, load_cases(args.data), modes, not args.no_warmup)
    output = json.dumps(results, indent=2)
    if args.json_output:
        args.json_output.parent.mkdir(parents=True, exist_ok=True)
        args.json_output.write_text(output + "\n")
    print("\n" + output)


if __name__ == "__main__":
    main()
