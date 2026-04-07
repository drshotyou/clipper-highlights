import subprocess
from pathlib import Path

import pytest

from clipper_highlights.config import ExportConfig
from clipper_highlights.exporter import _slugify, export_clips
from clipper_highlights.models import RankedClip


def test_export_clips_builds_expected_ffmpeg_command(monkeypatch, tmp_path):
    calls = []

    def fake_run(command, capture_output, text):
        calls.append(command)
        return subprocess.CompletedProcess(command, 0, "", "")

    monkeypatch.setattr("clipper_highlights.exporter.subprocess.run", fake_run)

    clip = RankedClip(
        candidate_id="cand_001",
        start=12.0,
        end=20.5,
        score=5.0,
        title="Big Win!",
        reason="test",
    )
    video_path = tmp_path / "session.mp4"
    video_path.write_text("fake")

    exported = export_clips(video_path, [clip], tmp_path / "clips", ExportConfig())

    assert len(exported) == 1
    assert exported[0].name == "01_big-win.mp4"
    assert calls[0][:8] == ["ffmpeg", "-y", "-ss", "12.000", "-i", str(video_path), "-t", "8.500"]
    assert calls[0][-1] == str(exported[0])


def test_export_clips_raises_when_ffmpeg_fails(monkeypatch, tmp_path):
    def fake_run(command, capture_output, text):
        return subprocess.CompletedProcess(command, 1, "", "boom")

    monkeypatch.setattr("clipper_highlights.exporter.subprocess.run", fake_run)

    clip = RankedClip(
        candidate_id="cand_001",
        start=1.0,
        end=2.0,
        score=1.0,
        title="bad",
        reason="test",
    )

    with pytest.raises(RuntimeError, match="ffmpeg clip export failed"):
        export_clips(tmp_path / "session.mp4", [clip], tmp_path / "clips", ExportConfig())


def test_slugify_normalizes_titles():
    assert _slugify("Clutch!!! Final Fight") == "clutch-final-fight"
