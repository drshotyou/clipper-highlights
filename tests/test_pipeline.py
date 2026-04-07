import json
from pathlib import Path

from clipper_highlights.config import ProjectConfig
from clipper_highlights.models import AudioSpike, CandidateWindow, RankedClip, TranscriptSegment
from clipper_highlights.pipeline import run_pipeline


def test_run_pipeline_writes_artifacts_and_exports(monkeypatch, tmp_path):
    video_path = tmp_path / "session.mp4"
    video_path.write_text("video")

    transcript = [TranscriptSegment(start=10.0, end=12.0, text="pmc on me")]
    spikes = [AudioSpike(start=11.0, end=11.5, peak_dbfs=-5.0, combined_score=4.0, onset_peak=3.0)]
    candidates = [CandidateWindow(candidate_id="cand_001", start=5.0, end=15.0, score=7.0)]
    ranked = [RankedClip(candidate_id="cand_001", start=5.0, end=15.0, score=7.0, title="fight", reason="good")]

    def fake_extract(video, output, config):
        output.write_text("audio")
        return output

    monkeypatch.setattr("clipper_highlights.pipeline.extract_audio", fake_extract)
    monkeypatch.setattr("clipper_highlights.pipeline.transcribe_audio", lambda audio, config: transcript)
    monkeypatch.setattr("clipper_highlights.pipeline.analyze_audio", lambda audio, config: spikes)
    monkeypatch.setattr("clipper_highlights.pipeline.generate_candidate_windows", lambda *args: candidates)
    monkeypatch.setattr("clipper_highlights.pipeline.rank_candidates", lambda bundle, config: ranked)
    monkeypatch.setattr(
        "clipper_highlights.pipeline.export_clips",
        lambda video, clips, output_dir, config: [output_dir / "01_fight.mp4"],
    )

    result = run_pipeline(video_path, tmp_path / "run", ProjectConfig(), game="tarkov")

    assert result.exported_paths == [tmp_path / "run" / "clips" / "01_fight.mp4"]
    assert (tmp_path / "run" / "artifacts" / "audio.wav").exists()
    assert json.loads((tmp_path / "run" / "artifacts" / "transcript.json").read_text())[0]["text"] == "pmc on me"
    assert json.loads((tmp_path / "run" / "artifacts" / "ranked_clips.json").read_text())[0]["title"] == "fight"


def test_run_pipeline_uses_cached_artifacts_without_force(monkeypatch, tmp_path):
    video_path = tmp_path / "session.mp4"
    video_path.write_text("video")
    output_dir = tmp_path / "run"
    artifact_dir = output_dir / "artifacts"
    artifact_dir.mkdir(parents=True)
    (artifact_dir / "audio.wav").write_text("audio")
    (artifact_dir / "transcript.json").write_text(json.dumps([{"start": 1.0, "end": 2.0, "text": "pmc"}]))
    (
        artifact_dir / "audio_spikes.json"
    ).write_text(json.dumps([{"start": 1.1, "end": 1.5, "peak_dbfs": -4.0, "combined_score": 5.0, "onset_peak": 2.0}]))

    calls = {"extract": 0, "transcribe": 0, "analyze": 0}

    def fail_extract(*args, **kwargs):
        calls["extract"] += 1
        raise AssertionError("extract_audio should not be called when cache exists")

    def fail_transcribe(*args, **kwargs):
        calls["transcribe"] += 1
        raise AssertionError("transcribe_audio should not be called when cache exists")

    def fail_analyze(*args, **kwargs):
        calls["analyze"] += 1
        raise AssertionError("analyze_audio should not be called when cache exists")

    monkeypatch.setattr("clipper_highlights.pipeline.extract_audio", fail_extract)
    monkeypatch.setattr("clipper_highlights.pipeline.transcribe_audio", fail_transcribe)
    monkeypatch.setattr("clipper_highlights.pipeline.analyze_audio", fail_analyze)
    monkeypatch.setattr(
        "clipper_highlights.pipeline.generate_candidate_windows",
        lambda transcript, spikes, config: [CandidateWindow(candidate_id="cand_001", start=0.0, end=8.0, score=4.0)],
    )
    monkeypatch.setattr(
        "clipper_highlights.pipeline.rank_candidates",
        lambda bundle, config: [RankedClip(candidate_id="cand_001", start=0.0, end=8.0, score=4.0, title="cached", reason="cached")],
    )

    result = run_pipeline(video_path, output_dir, ProjectConfig(), export_enabled=False)

    assert result.ranked_clips[0].title == "cached"
    assert calls == {"extract": 0, "transcribe": 0, "analyze": 0}
