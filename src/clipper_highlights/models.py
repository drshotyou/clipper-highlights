from __future__ import annotations

from pydantic import BaseModel, Field


class WordSegment(BaseModel):
    start: float
    end: float
    text: str
    probability: float | None = None


class TranscriptSegment(BaseModel):
    start: float
    end: float
    text: str
    avg_logprob: float | None = None
    no_speech_prob: float | None = None
    words: list[WordSegment] = Field(default_factory=list)


class AudioSpike(BaseModel):
    start: float
    end: float
    peak_dbfs: float
    combined_score: float
    onset_peak: float


class Evidence(BaseModel):
    kind: str
    start: float
    end: float
    score: float
    label: str
    detail: str | None = None


class CandidateWindow(BaseModel):
    candidate_id: str
    start: float
    end: float
    score: float
    reasons: list[str] = Field(default_factory=list)
    transcript_excerpt: str = ""
    evidences: list[Evidence] = Field(default_factory=list)


class RankedClip(BaseModel):
    candidate_id: str | None = None
    start: float
    end: float
    score: float
    title: str
    reason: str


class TimelineBundle(BaseModel):
    source_video: str
    audio_path: str
    game: str | None = None
    transcript_segments: list[TranscriptSegment] = Field(default_factory=list)
    audio_spikes: list[AudioSpike] = Field(default_factory=list)
    candidate_windows: list[CandidateWindow] = Field(default_factory=list)
