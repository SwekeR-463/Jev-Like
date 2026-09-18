"""Live single-case demo of the parallel constrained readout.

Runs the real model on one benchmark case, prints a terminal transcript, and
optionally replays that transcript into a video. The numbers are measured; only
the pacing of the replay is artificial.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import textwrap
import time
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import jev_like  # noqa: E402  (the repo root must be importable first)
from jev_like import decide_parallel, generate_json, get_engine  # noqa: E402

BG, TEXT, DIM, ACCENT, WARM = "#0a0a0b", "#f5f5f7", "#5f5f68", "#ff7a18", "#a4836a"
MONO = "/System/Library/Fonts/SFNSMono.ttf"
W, H, FPS, HOLD_SECONDS = 1280, 720, 30, 3.5
LINE_HEIGHT = 30


def load_case(path: Path, index: int) -> dict:
    cases = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    return cases[index - 1]


def transcript(model: str, case: dict, auto: dict, par: dict) -> list[tuple[str, str]]:
    produced = json.dumps(auto["value"] if auto["value"] is not None else {})
    choices = {name: entry["value"] for name, entry in par["value"].items()}
    matches = sum(
        1
        for name, expected in case["expected"].items()
        if str(choices.get(name)).lower() == str(expected).lower()
    )

    lines: list[tuple[str, str]] = [("$ jev-like --model " + model, DIM), ("", TEXT), ("INPUT", DIM)]
    for wrapped in textwrap.wrap(case["context"], 76):
        lines.append(("  " + wrapped, TEXT))
    lines += [
        ("", TEXT),
        ("SCHEMA", DIM),
        ("  " + ", ".join(f"{name}: {spec['type']}" for name, spec in case["schema"].items()), TEXT),
        ("", TEXT),
        (f"AUTOREGRESSIVE JSON · {auto['forward_passes']} tokens", DIM),
    ]
    for wrapped in textwrap.wrap("  " + produced, 76):
        lines.append((wrapped, WARM))
    lines += [
        (f"  {auto['elapsed_ms']:.1f} ms total · {auto['tokens_per_s']:.1f} tok/s decode", WARM),
        ("", TEXT),
        ("PARALLEL CONSTRAINED READOUT · 1 batched pass", DIM),
    ]
    for name, entry in par["value"].items():
        label = f'  "{name}": "{entry["value"]}"'
        lines.append((f"{label:<48} {entry['score']:.2f}", ACCENT))
    lines += [
        (f"  {par['elapsed_ms']:.1f} ms total · {par['tokens_per_s']:.0f} equivalent tok/s", ACCENT),
        ("", TEXT),
        (
            f"{auto['elapsed_ms'] / par['elapsed_ms']:.1f}x faster · "
            f"{matches}/{len(case['expected'])} fields match expected",
            ACCENT,
        ),
        (f"Real run · measured locally · {model}", DIM),
    ]
    return lines


def frame(lines: list[tuple[str, str]], reveal_at: float, t: float) -> Image.Image:
    image = Image.new("RGB", (W, H), BG)
    draw = ImageDraw.Draw(image)
    draw.rounded_rectangle((40, 30, W - 40, H - 30), 10, outline="#26262b", width=1)
    # Fit the whole transcript inside the terminal box, whatever its length.
    step = min(LINE_HEIGHT, (H - 100) // max(len(lines), 1))
    font = ImageFont.truetype(MONO, min(19, step - 6))
    for index, (text, color) in enumerate(lines):
        if reveal_at * (index + 1) > t:
            break
        if text:
            draw.text((72, 70 + index * step), text, font=font, fill=color)
    return image


def render(lines: list[tuple[str, str]], output: Path, reveal_at: float) -> None:
    total = reveal_at * len(lines) + HOLD_SECONDS
    command = [
        "ffmpeg", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W}x{H}",
        "-r", str(FPS), "-i", "-", "-an", "-c:v", "libx264", "-preset", "medium",
        "-crf", "18", "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(output),
    ]
    with subprocess.Popen(command, stdin=subprocess.PIPE) as process:
        for index in range(int(total * FPS)):
            process.stdin.write(frame(lines, reveal_at, index / FPS).tobytes())
        process.stdin.close()
        if process.wait():
            raise SystemExit("ffmpeg failed")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Qwen/Qwen3.5-2B")
    # Case 4 is the clearest single-case contrast: the parallel readout scores 3/3
    # against the autoregressive path's 2/3 on this model.
    parser.add_argument("--case", type=int, default=4)
    parser.add_argument("--data", type=Path, default=ROOT / "data/benchmark.jsonl")
    parser.add_argument("--video", action="store_true", help="also replay the run into an MP4")
    parser.add_argument("--reveal-ms", type=float, default=190.0, help="replay pacing, not a measurement")
    parser.add_argument("--out", type=Path, default=Path(__file__).with_name("live.mp4"))
    args = parser.parse_args()

    case = load_case(args.data, args.case)
    jev_like.MODEL_ID = args.model
    get_engine()
    decide_parallel(case["context"], case["schema"])  # unmeasured warm-up
    started = time.perf_counter()
    auto = generate_json(case["context"], case["schema"])
    par = decide_parallel(case["context"], case["schema"])
    print(f"both paths ran in {time.perf_counter() - started:.1f}s wall clock\n")

    lines = transcript(args.model, case, auto, par)
    for text, _ in lines:
        print(text)

    if args.video:
        render(lines, args.out, args.reveal_ms / 1000)
        print(f"\n{args.out}")


if __name__ == "__main__":
    main()
