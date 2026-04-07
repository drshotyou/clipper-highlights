from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from clipper_highlights.audio import analyze_audio, extract_audio
from clipper_highlights.candidates import generate_candidate_windows
from clipper_highlights.config import ProjectConfig
from clipper_highlights.exporter import export_clips
from clipper_highlights.llm import rank_candidates
from clipper_highlights.models import RankedClip, TimelineBundle
from clipper_highlights.transcription import transcribe_audio


@dataclass(slots=True)
class PipelineResult:
    run_dir: Path
    artifact_dir: Path
    clips_dir: Path
    bundle: TimelineBundle
    ranked_clips: list[RankedClip]
    exported_paths: list[Path]


def run_pipeline(
    video_path: Path,
    output_dir: Path,
    config: ProjectConfig,
    *,
    game: str | None = None,
    export_enabled: bool = True,
    force: bool = False,
) -> PipelineResult:
    artifact_dir = output_dir / config.artifact_dir_name
    clips_dir = output_dir / config.export_dir_name
    artifact_dir.mkdir(parents=True, exist_ok=True)
    clips_dir.mkdir(parents=True, exist_ok=True)

    audio_path = artifact_dir / "audio.wav"
    transcript_path = artifact_dir / "transcript.json"
    spikes_path = artifact_dir / "audio_spikes.json"
    bundle_path = artifact_dir / "timeline_bundle.json"
    ranked_path = artifact_dir / "ranked_clips.json"

    if force or not audio_path.exists():
        extract_audio(video_path, audio_path, config.audio)

    if force or not transcript_path.exists():
        transcript_segments = transcribe_audio(audio_path, config.transcription)
        _write_json(transcript_path, [item.model_dump(mode="json") for item in transcript_segments])
    else:
        transcript_segments = TimelineBundle.model_validate(
            {
                "source_video": str(video_path),
                "audio_path": str(audio_path),
                "transcript_segments": json.loads(transcript_path.read_text()),
            }
        ).transcript_segments

    if force or not spikes_path.exists():
        audio_spikes = analyze_audio(audio_path, config.audio)
        _write_json(spikes_path, [item.model_dump(mode="json") for item in audio_spikes])
    else:
        audio_spikes = TimelineBundle.model_validate(
            {
                "source_video": str(video_path),
                "audio_path": str(audio_path),
                "audio_spikes": json.loads(spikes_path.read_text()),
            }
        ).audio_spikes

    candidate_windows = generate_candidate_windows(
        transcript_segments,
        audio_spikes,
        config.candidate,
    )
    bundle = TimelineBundle(
        source_video=str(video_path),
        audio_path=str(audio_path),
        game=game,
        transcript_segments=transcript_segments,
        audio_spikes=audio_spikes,
        candidate_windows=candidate_windows,
    )
    _write_json(bundle_path, bundle.model_dump(mode="json"))

    ranked_clips = rank_candidates(bundle, config.llm)
    _write_json(ranked_path, [item.model_dump(mode="json") for item in ranked_clips])

    exported_paths: list[Path] = []
    if export_enabled and ranked_clips:
        exported_paths = export_clips(video_path, ranked_clips, clips_dir, config.export)

    return PipelineResult(
        run_dir=output_dir,
        artifact_dir=artifact_dir,
        clips_dir=clips_dir,
        bundle=bundle,
        ranked_clips=ranked_clips,
        exported_paths=exported_paths,
    )


def _write_json(path: Path, payload) -> None:
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
