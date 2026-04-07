from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

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
    progress_callback: Callable[[str], None] | None = None,
) -> PipelineResult:
    artifact_dir = output_dir / config.artifact_dir_name
    clips_dir = output_dir / config.export_dir_name
    artifact_dir.mkdir(parents=True, exist_ok=True)
    clips_dir.mkdir(parents=True, exist_ok=True)
    _emit(progress_callback, f"Run directory: {output_dir}")
    _emit(progress_callback, f"Artifacts directory: {artifact_dir}")

    audio_path = artifact_dir / "audio.wav"
    transcript_path = artifact_dir / "transcript.json"
    spikes_path = artifact_dir / "audio_spikes.json"
    bundle_path = artifact_dir / "timeline_bundle.json"
    ranked_path = artifact_dir / "ranked_clips.json"

    if force or not audio_path.exists():
        _emit(progress_callback, f"Extracting audio from {video_path.name}")
        extract_audio(video_path, audio_path, config.audio)
        _emit(progress_callback, f"Wrote extracted audio to {audio_path}")
    else:
        _emit(progress_callback, f"Reusing cached audio: {audio_path}")

    if force or not transcript_path.exists():
        _emit(progress_callback, "Running speech-to-text")
        transcript_segments = transcribe_audio(
            audio_path,
            config.transcription,
            progress_callback=progress_callback,
        )
        _write_json(transcript_path, [item.model_dump(mode="json") for item in transcript_segments])
        _emit(progress_callback, f"Wrote transcript artifact: {transcript_path}")
    else:
        _emit(progress_callback, f"Reusing cached transcript: {transcript_path}")
        transcript_segments = TimelineBundle.model_validate(
            {
                "source_video": str(video_path),
                "audio_path": str(audio_path),
                "transcript_segments": json.loads(transcript_path.read_text()),
            }
        ).transcript_segments

    if force or not spikes_path.exists():
        _emit(progress_callback, "Analyzing audio for adaptive spikes")
        audio_spikes = analyze_audio(audio_path, config.audio)
        _write_json(spikes_path, [item.model_dump(mode="json") for item in audio_spikes])
        _emit(progress_callback, f"Detected {len(audio_spikes)} audio spikes")
    else:
        _emit(progress_callback, f"Reusing cached audio spike artifact: {spikes_path}")
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
    _emit(progress_callback, f"Generated {len(candidate_windows)} candidate windows")
    bundle = TimelineBundle(
        source_video=str(video_path),
        audio_path=str(audio_path),
        game=game,
        transcript_segments=transcript_segments,
        audio_spikes=audio_spikes,
        candidate_windows=candidate_windows,
    )
    _write_json(bundle_path, bundle.model_dump(mode="json"))
    _emit(progress_callback, f"Wrote timeline bundle: {bundle_path}")

    _emit(progress_callback, f"Ranking candidates with provider '{config.llm.provider}'")
    ranked_clips = rank_candidates(bundle, config.llm)
    _write_json(ranked_path, [item.model_dump(mode="json") for item in ranked_clips])
    _emit(progress_callback, f"Wrote ranked clips: {ranked_path}")
    _emit(progress_callback, f"Selected {len(ranked_clips)} ranked clips")

    exported_paths: list[Path] = []
    if export_enabled and ranked_clips:
        _emit(progress_callback, f"Exporting {len(ranked_clips)} clips")
        exported_paths = export_clips(video_path, ranked_clips, clips_dir, config.export)
        _emit(progress_callback, f"Export complete: {len(exported_paths)} files in {clips_dir}")
    elif not export_enabled:
        _emit(progress_callback, "Skipping clip export because --no-export was requested")
    else:
        _emit(progress_callback, "No ranked clips to export")

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


def _emit(progress_callback: Callable[[str], None] | None, message: str) -> None:
    if progress_callback is not None:
        progress_callback(message)
