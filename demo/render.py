"""Render the benchmark comparison animation.

The pacing is deliberate: real inference is too fast to perceive, so the reveal
timings are slowed. The numbers come from a measured run.
"""

from __future__ import annotations

from pathlib import Path

from demo.video import ACCENT, BORDER, FPS, MUTED, PANEL, TINT, WARM, font, new_frame, put, write_video

SECONDS = 12
OUTPUT = Path(__file__).with_name("comparison.mp4")

LEFT = [
    (2.7, "{"),
    (3.35, '  "urgent": true,'),
    (4.1, '  "department": "infrastructure",'),
    (4.85, '  "sentiment": "concerned"'),
    (5.45, "}"),
]
RIGHT = [
    (2.65, "{"),
    (2.72, '  "urgent": true,              0.98'),
    (2.79, '  "department": "infrastructure", 0.73'),
    (2.86, '  "sentiment": "concerned"    0.66'),
    (2.93, "}"),
]


def frame(t: float):
    image, draw = new_frame()
    put(draw, (640, 28), "Jev-like structured decisions", 40, anchor="ma")
    put(draw, (640, 75), "Same model · same ticket · two inference paths", 19, MUTED, anchor="ma")

    draw.rounded_rectangle((70, 105, 1210, 180), 14, fill=PANEL, outline=BORDER, width=1)
    put(draw, (94, 122), "INPUT", 16, MUTED)
    put(draw, (94, 148), "Checkout fails for 42% of customers. Acme reports lost sales and requests an immediate response.", 18)

    draw.rounded_rectangle((70, 205, 620, 555), 18, fill=PANEL, outline=BORDER, width=1)
    draw.rounded_rectangle((660, 205, 1210, 555), 18, fill=PANEL, outline=ACCENT, width=2)
    draw.rectangle((660, 205, 665, 555), fill=ACCENT)
    put(draw, (95, 232), "Autoregressive JSON", 25, MUTED)
    put(draw, (95, 268), "Sequential token generation", 16, MUTED)
    put(draw, (690, 232), "Parallel constrained readout", 25, ACCENT)
    put(draw, (690, 268), "Shared prefix + one batched suffix pass", 16, MUTED)

    if 2.1 <= t < 2.7:
        put(draw, (95, 310), "generating…", 21, WARM)
        put(draw, (690, 310), "scoring allowed choices…", 21, ACCENT)
    for index, (start, line) in enumerate(LEFT):
        if t >= start:
            put(draw, (95, 315 + index * 43), line, 19, mono=True)
    for index, (start, line) in enumerate(RIGHT):
        if t >= start:
            put(draw, (690, 315 + index * 43), line, 17, mono=True)
    if t >= 3.02:
        put(draw, (690, 520), "45.9 ms mean", 22, ACCENT)
    if t >= 5.6:
        put(draw, (95, 520), "369.0 ms mean", 22, WARM)
    if t >= 6.4:
        bbox = draw.textbbox((640, 590), "8.0x faster", font=font(36), anchor="ma")
        draw.rounded_rectangle((bbox[0] - 26, bbox[1] - 12, bbox[2] + 26, bbox[3] + 14), 24,
                               fill=TINT, outline=ACCENT, width=1)
        put(draw, (640, 590), "8.0x faster", 36, ACCENT, anchor="ma")
    if t >= 7.2:
        put(draw, (640, 650), "No JSON tokens generated · output assembled in code", 20, MUTED, anchor="ma")
    put(draw, (640, 696), "Local smoke test · Qwen3-1.7B BF16 · animation slowed for visibility", 14, MUTED, anchor="ms")
    return image


def main() -> None:
    frames = (frame(index / FPS) for index in range(FPS * SECONDS))
    print(write_video(frames, OUTPUT))


if __name__ == "__main__":
    main()
