from pathlib import Path

from typer.testing import CliRunner

from clipper_highlights.cli import app
from clipper_highlights.models import CandidateWindow, RankedClip, TimelineBundle
from clipper_highlights.pipeline import PipelineResult


runner = CliRunner()


def test_init_config_writes_yaml_file(tmp_path):
    destination = tmp_path / "config.yaml"

    result = runner.invoke(app, ["init-config", str(destination)])

    assert result.exit_code == 0
    assert destination.exists()
    assert "transcription:" in destination.read_text()


def test_run_command_uses_default_output_dir_and_keyword_overrides(monkeypatch, tmp_path):
    input_video = tmp_path / "session.mp4"
    input_video.write_text("video")
    captured = {}

    def fake_run_pipeline(video_path, output_dir, config, *, game, export_enabled, force, progress_callback=None):
        captured["video_path"] = video_path
        captured["output_dir"] = output_dir
        captured["config"] = config
        captured["game"] = game
        captured["export_enabled"] = export_enabled
        captured["force"] = force
        captured["progress_callback"] = progress_callback
        if progress_callback is not None:
            progress_callback("Mock progress message")
        return PipelineResult(
            run_dir=output_dir,
            artifact_dir=output_dir / "artifacts",
            clips_dir=output_dir / "clips",
            bundle=TimelineBundle(
                source_video=str(video_path),
                audio_path="audio.wav",
                candidate_windows=[CandidateWindow(candidate_id="cand_001", start=1.0, end=4.0, score=3.0)],
            ),
            ranked_clips=[RankedClip(candidate_id="cand_001", start=1.0, end=4.0, score=3.0, title="fight", reason="ok")],
            exported_paths=[],
        )

    monkeypatch.setattr("clipper_highlights.cli.run_pipeline", fake_run_pipeline)

    result = runner.invoke(
        app,
        [
            "run",
            str(input_video),
            "--game",
            "tarkov",
            "--keyword",
            "pmc=4.5",
            "--no-llm",
            "--no-export",
            "--force",
        ],
    )

    assert result.exit_code == 0
    assert captured["video_path"] == input_video
    assert captured["output_dir"] == Path("runs") / "session"
    assert captured["game"] == "tarkov"
    assert captured["export_enabled"] is False
    assert captured["force"] is True
    assert callable(captured["progress_callback"])
    assert captured["config"].llm.provider == "none"
    assert captured["config"].candidate.keywords["pmc"] == 4.5
    assert "Ranked clips" in result.stdout
    assert "Starting" in result.stdout
    assert "Mock progress message" in result.stdout
