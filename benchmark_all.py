"""Benchmark every locally supported model and save one comparison table."""

import json
import subprocess
import sys
from pathlib import Path


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


def main():
    output_dir = Path("results")
    output_dir.mkdir(exist_ok=True)
    rows = []
    for index, model in enumerate(MODELS, 1):
        path = output_dir / f"run-{index}.json"
        print(f"\n[{index}/{len(MODELS)}] {model}", flush=True)
        subprocess.run(
            [sys.executable, "benchmark.py", "--model", model, "--json-output", str(path)],
            check=True,
        )
        results = json.loads(path.read_text())
        autoregressive, parallel, comparison = results
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


if __name__ == "__main__":
    main()
