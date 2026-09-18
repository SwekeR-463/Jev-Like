"""Live single-case demo of the parallel constrained readout.

Runs the real model on one benchmark case, prints a terminal transcript, and can
replay that transcript into a video. The numbers and the JSON are measured; only
the replay pacing is artificial.
"""

from __future__ import annotations

import argparse
import json
import textwrap
import time
from pathlib import Path

from demo.video import ACCENT, BORDER, FPS, H, MUTED, TEXT, WARM, font, new_frame, write_video
from jev_like import decide_parallel, generate_json, get_engine
from jev_like.benchmark import DEFAULT_DATA, load_cases

OUTPUT = Path(__file__).with_name("live.mp4")
DEFAULT_MODEL = "Qwen/Qwen3.5-2B"
TERMINAL_BOX = (40, 30, 1240, 690)
HOLD_SECONDS, LINE_HEIGHT = 3.5, 30


def transcript(model_id: str, case: dict, auto: dict, par: dict) -> list[tuple[str, str]]:
    produced = json.dumps(auto["value"] if auto["value"] is not None else {})
    choices = {name: entry["value"] for name, entry in par["value"].items()}
    matches = sum(
        1
        for name, expected in case["expected"].items()
        if str(choices.get(name)).lower() == str(expected).lower()
    )

    lines: list[tuple[str, str]] = [("$ jev-like --model " + model_id, MUTED), ("", TEXT), ("INPUT", MUTED)]
    for wrapped in textwrap.wrap(case["context"], 76):
        lines.append(("  " + wrapped, TEXT))
    lines += [
        ("", TEXT),
        ("SCHEMA", MUTED),
        ("  " + ", ".join(f"{name}: {spec['type']}" for name, spec in case["schema"].items()), TEXT),
        ("", TEXT),
        (f"AUTOREGRESSIVE JSON · {auto['forward_passes']} tokens", MUTED),
    ]
    for wrapped in textwrap.wrap("  " + produced, 76):
        lines.append((wrapped, WARM))
    lines += [
        (f"  {auto['elapsed_ms']:.1f} ms total · {auto['tokens_per_s']:.1f} tok/s decode", WARM),
        ("", TEXT),
        ("PARALLEL CONSTRAINED READOUT · 1 batched pass", MUTED),
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
        (f"Real run · measured locally · {model_id}", MUTED),
    ]
    return lines


def frame(lines: list[tuple[str, str]], reveal_at: float, t: float):
    image, draw = new_frame()
    draw.rounded_rectangle(TERMINAL_BOX, 10, outline=BORDER, width=1)
    # Fit the whole transcript inside the terminal box, whatever its length.
    step = min(LINE_HEIGHT, (H - 100) // max(len(lines), 1))
    text_font = font(min(19, step - 6), mono=True)
    for index, (text, color) in enumerate(lines):
        if reveal_at * (index + 1) > t:
            break
        if text:
            draw.text((72, 70 + index * step), text, font=text_font, fill=color)
    return image


def render(lines: list[tuple[str, str]], output: Path, reveal_at: float) -> Path:
    total = reveal_at * len(lines) + HOLD_SECONDS
    frames = (frame(lines, reveal_at, index / FPS) for index in range(int(total * FPS)))
    return write_video(frames, output)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=DEFAULT_MODEL)
    # Case 4 is the clearest single-case contrast: the parallel readout scores 3/3
    # against the autoregressive path's 2/3 on this model.
    parser.add_argument("--case", type=int, default=4)
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--video", action="store_true", help="also replay the run into an MP4")
    parser.add_argument("--reveal-ms", type=float, default=190.0, help="replay pacing, not a measurement")
    parser.add_argument("--out", type=Path, default=OUTPUT)
    args = parser.parse_args()

    case = load_cases(args.data)[args.case - 1]
    get_engine(args.model)
    decide_parallel(case["context"], case["schema"], model_id=args.model)  # unmeasured warm-up
    started = time.perf_counter()
    auto = generate_json(case["context"], case["schema"], model_id=args.model)
    par = decide_parallel(case["context"], case["schema"], model_id=args.model)
    print(f"both paths ran in {time.perf_counter() - started:.1f}s wall clock\n")

    lines = transcript(args.model, case, auto, par)
    for text, _ in lines:
        print(text)

    if args.video:
        print(f"\n{render(lines, args.out, args.reveal_ms / 1000)}")


if __name__ == "__main__":
    main()
