from __future__ import annotations

from pathlib import Path

import yaml
from pydantic import BaseModel, Field


DEFAULT_KEYWORDS = {
    "shoot": 1.5,
    "shooting": 1.5,
    "fight": 1.2,
    "he's dead": 1.8,
    "i'm down": 2.5,
    "one shot": 2.0,
    "knocked": 1.3,
    "clutch": 2.0,
    "wipe": 1.8,
    "pmc": 2.0,
    "scav": 1.0,
}


class AudioConfig(BaseModel):
    sample_rate: int = 16000
    frame_length: int = 2048
    hop_length: int = 512
    min_dbfs: float = -45.0
    energy_weight: float = 0.6
    onset_weight: float = 0.7
    delta_weight: float = 0.4
    combined_threshold: float = 2.4


class TranscriptionConfig(BaseModel):
    model: str = "small.en"
    device: str = "auto"
    compute_type: str = "int8"
    language: str | None = "en"
    beam_size: int = 5
    vad_filter: bool = True
    word_timestamps: bool = True
    initial_prompt: str | None = None
    hotwords: list[str] = Field(default_factory=list)


class CandidateConfig(BaseModel):
    keywords: dict[str, float] = Field(default_factory=lambda: DEFAULT_KEYWORDS.copy())
    audio_event_weight: float = 1.0
    keyword_event_weight: float = 1.25
    crossreference_window_seconds: float = 10.0
    crossreference_boost: float = 2.0
    pre_roll_seconds: float = 6.0
    post_roll_seconds: float = 5.0
    merge_gap_seconds: float = 8.0
    min_candidate_score: float = 3.0
    max_candidates: int = 25
    min_clip_seconds: float = 8.0
    max_clip_seconds: float = 75.0


class LLMConfig(BaseModel):
    provider: str = "none"
    model: str | None = None
    api_key_env: str | None = None
    base_url: str | None = None
    organization: str | None = None
    project: str | None = None
    temperature: float = 0.1
    top_candidate_windows: int = 15
    max_output_clips: int = 8
    timeout_seconds: int = 60


class ExportConfig(BaseModel):
    output_format: str = "mp4"
    video_codec: str = "libx264"
    audio_codec: str = "aac"
    crf: int = 20
    preset: str = "medium"


class ProjectConfig(BaseModel):
    artifact_dir_name: str = "artifacts"
    export_dir_name: str = "clips"
    audio: AudioConfig = Field(default_factory=AudioConfig)
    transcription: TranscriptionConfig = Field(default_factory=TranscriptionConfig)
    candidate: CandidateConfig = Field(default_factory=CandidateConfig)
    llm: LLMConfig = Field(default_factory=LLMConfig)
    export: ExportConfig = Field(default_factory=ExportConfig)

    @classmethod
    def load(cls, path: str | Path | None = None) -> "ProjectConfig":
        if path is None:
            return cls()

        raw = yaml.safe_load(Path(path).read_text()) or {}
        return cls.model_validate(raw)

    def dump_yaml(self) -> str:
        data = self.model_dump(mode="python")
        return yaml.safe_dump(data, sort_keys=False)
