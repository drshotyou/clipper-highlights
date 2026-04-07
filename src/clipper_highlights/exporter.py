from __future__ import annotations

import subprocess
from pathlib import Path

from clipper_highlights.config import ExportConfig
from clipper_highlights.models import RankedClip


def export_clips(
    video_path: Path,
    clips: list[RankedClip],
    output_dir: Path,
    config: ExportConfig,
) -> list[Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    exported: list[Path] = []
    for index, clip in enumerate(clips, start=1):
        output_path = output_dir / f"{index:02d}_{_slugify(clip.title)}.{config.output_format}"
        duration = clip.end - clip.start
        command = [
            "ffmpeg",
            "-y",
            "-ss",
            f"{clip.start:.3f}",
            "-i",
            str(video_path),
            "-t",
            f"{duration:.3f}",
            "-c:v",
            config.video_codec,
            "-preset",
            config.preset,
            "-crf",
            str(config.crf),
            "-c:a",
            config.audio_codec,
            "-movflags",
            "+faststart",
            str(output_path),
        ]
        result = subprocess.run(command, capture_output=True, text=True)
        if result.returncode != 0:
            raise RuntimeError(f"ffmpeg clip export failed for {output_path.name}:\n{result.stderr}")
        exported.append(output_path)
    return exported


def _slugify(text: str) -> str:
    cleaned = "".join(char.lower() if char.isalnum() else "-" for char in text)
    compact = "-".join(part for part in cleaned.split("-") if part)
    return compact[:80] or "clip"
