"""Shared Pillow and FFmpeg helpers for the demo videos."""

from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Iterable

from PIL import Image, ImageDraw, ImageFont

W, H, FPS = 1280, 720, 30
FONT = "/System/Library/Fonts/SFNS.ttf"
MONO = "/System/Library/Fonts/SFNSMono.ttf"

BG, PANEL, TEXT, MUTED = "#0a0a0b", "#17171a", "#f5f5f7", "#8f8f98"
BORDER, ACCENT, WARM, TINT = "#26262b", "#ff7a18", "#a4836a", "#2a1a0e"


def font(size: int, mono: bool = False):
    return ImageFont.truetype(MONO if mono else FONT, size)


def new_frame() -> tuple[Image.Image, ImageDraw.ImageDraw]:
    image = Image.new("RGB", (W, H), BG)
    return image, ImageDraw.Draw(image)


def put(draw, xy, text, size, color=TEXT, mono=False, anchor="la") -> None:
    draw.text(xy, text, font=font(size, mono), fill=color, anchor=anchor)


def write_video(frames: Iterable[Image.Image], output: Path) -> Path:
    """Pipe raw RGB frames into ffmpeg and write an H.264 MP4."""
    command = [
        "ffmpeg", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W}x{H}",
        "-r", str(FPS), "-i", "-", "-an", "-c:v", "libx264", "-preset", "medium",
        "-crf", "18", "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(output),
    ]
    with subprocess.Popen(command, stdin=subprocess.PIPE) as process:
        for image in frames:
            process.stdin.write(image.tobytes())
        process.stdin.close()
        if process.wait():
            raise SystemExit("ffmpeg failed")
    return output
