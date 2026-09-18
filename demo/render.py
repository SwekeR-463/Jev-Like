"""Render the benchmark comparison video with Pillow and FFmpeg."""

from pathlib import Path
import subprocess

from PIL import Image, ImageDraw, ImageFont

W, H, FPS, SECONDS = 1280, 720, 30, 12
FONT = "/System/Library/Fonts/SFNS.ttf"
MONO = "/System/Library/Fonts/SFNSMono.ttf"
BG, PANEL, TEXT, MUTED = "#0a0a0b", "#17171a", "#f5f5f7", "#8f8f98"
BORDER, ACCENT, WARM = "#26262b", "#ff7a18", "#a4836a"
TINT = "#2a1a0e"


def font(size, mono=False):
    return ImageFont.truetype(MONO if mono else FONT, size)


def put(draw, xy, text, size, color=TEXT, mono=False, anchor="la"):
    draw.text(xy, text, font=font(size, mono), fill=color, anchor=anchor)


def frame(t):
    image = Image.new("RGB", (W, H), BG)
    draw = ImageDraw.Draw(image)
    put(draw, (W // 2, 28), "Jev-like structured decisions", 40, anchor="ma")
    put(draw, (W // 2, 75), "Same model · same ticket · two inference paths", 19, MUTED, anchor="ma")

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

    left = [
        (2.7, "{"),
        (3.35, '  "urgent": true,'),
        (4.1, '  "department": "infrastructure",'),
        (4.85, '  "sentiment": "concerned"'),
        (5.45, "}"),
    ]
    right = [
        (2.65, "{"),
        (2.72, '  "urgent": true,              0.98'),
        (2.79, '  "department": "infrastructure", 0.73'),
        (2.86, '  "sentiment": "concerned"    0.66'),
        (2.93, "}"),
    ]
    if 2.1 <= t < 2.7:
        put(draw, (95, 310), "generating…", 21, WARM)
        put(draw, (690, 310), "scoring allowed choices…", 21, ACCENT)
    for i, (start, line) in enumerate(left):
        if t >= start:
            put(draw, (95, 315 + i * 43), line, 19, mono=True)
    for i, (start, line) in enumerate(right):
        if t >= start:
            put(draw, (690, 315 + i * 43), line, 17, mono=True)
    if t >= 3.02:
        put(draw, (690, 520), "45.9 ms mean", 22, ACCENT)
    if t >= 5.6:
        put(draw, (95, 520), "369.0 ms mean", 22, WARM)
    if t >= 6.4:
        bbox = draw.textbbox((W // 2, 590), "8.0x faster", font=font(36), anchor="ma")
        draw.rounded_rectangle((bbox[0] - 26, bbox[1] - 12, bbox[2] + 26, bbox[3] + 14), 24,
                               fill=TINT, outline=ACCENT, width=1)
        put(draw, (W // 2, 590), "8.0x faster", 36, ACCENT, anchor="ma")
    if t >= 7.2:
        put(draw, (W // 2, 650), "No JSON tokens generated · output assembled in code", 20, MUTED, anchor="ma")
    put(draw, (W // 2, 696), "Local smoke test · Qwen3-1.7B BF16 · animation slowed for visibility", 14, MUTED, anchor="ms")
    return image


output = Path(__file__).with_name("comparison.mp4")
command = [
    "ffmpeg", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W}x{H}",
    "-r", str(FPS), "-i", "-", "-an", "-c:v", "libx264", "-preset", "medium",
    "-crf", "18", "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(output),
]
with subprocess.Popen(command, stdin=subprocess.PIPE) as process:
    for index in range(FPS * SECONDS):
        process.stdin.write(frame(index / FPS).tobytes())
    process.stdin.close()
    if process.wait():
        raise SystemExit("ffmpeg failed")

print(output)
